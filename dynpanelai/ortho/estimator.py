"""Orthogonal and debiased Lasso for heterogeneous treatment effects.

Implements

    Semenova, V., Goldman, M., Chernozhukov, V. and Taddy, M. (2023).
    Inference on heterogeneous treatment effects in high-dimensional dynamic
    panels under weak dependence.  *Quantitative Economics*.

Model
-----
.. math::
    Y_{it} = D_{it}'\\beta_0 + e_0(X_{it}) + \\xi_i^E + U_{it},
    \\qquad D_{it} = P_{it}K(X_{it}),

so the conditional average treatment effect is a *high-dimensional* object:
a low-dimensional base treatment :math:`P_{it}` interacted with a rich
dictionary :math:`K(X_{it})` of heterogeneity-relevant controls.

Three stages
------------
1. **Residualise** :math:`Y` and :math:`P` on the controls using
   neighbours-left-out cross-fitting, so nuisance error is orthogonal to the
   second-stage moment.
2. **Orthogonal Lasso**: regress the outcome residual on the treatment
   residual interacted with the dictionary.  When the CATE function is
   simpler than the control function, this attains the near-oracle rate
   :math:`\\sqrt{s\\log d/NT}` -- faster than a single-stage regression.
3. **Debias** with a CLIME approximate inverse, then form pointwise and
   simultaneous confidence bands.

The debiasing step matters: :math:`\\ell_1` shrinkage biases every coefficient,
so the raw Lasso cannot support inference.  The correction

.. math::
    \\widehat\\beta_{DL} = \\widehat\\beta_L
        + \\widehat\\Omega\\,\\frac{1}{NT}\\sum_{i,t}
          \\widehat V_{it}\\bigl(\\widehat{\\tilde Y}_{it}
          - \\widehat V_{it}'\\widehat\\beta_L\\bigr)

restores asymptotic normality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.panel import PanelData
from ..core.results import PanelResults
from ..dml.folds import nlo_folds
from ..penalized.clime import clime
from ..penalized.rlasso import rlasso

__all__ = ["OrthogonalLasso", "orthogonal_lasso", "simultaneous_ci"]


def simultaneous_ci(
    beta: np.ndarray,
    cov: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int | None = 0,
) -> tuple[np.ndarray, float]:
    """Simultaneous confidence bands via the Gaussian multiplier bootstrap.

    Draws :math:`Z\\sim N(0,\\widehat C)` with :math:`\\widehat C` the
    correlation matrix implied by ``cov``, and takes the
    :math:`(1-\\alpha)` quantile of :math:`\\|Z\\|_\\infty` as the critical
    value.  Justified by the high-dimensional CLT of Chernozhukov,
    Chetverikov and Kato (2013), which holds even when ``d`` far exceeds the
    sample size.

    Parameters
    ----------
    beta : ndarray of shape (d,)
    cov : ndarray of shape (d, d)
    alpha : float, default 0.05
    n_boot : int, default 2000
    seed : int, optional

    Returns
    -------
    bands : ndarray of shape (d, 2)
        Lower and upper simultaneous bounds.
    crit : float
        The critical value.

    Notes
    -----
    Simultaneous bands are wider than pointwise ones by construction.  Use
    them whenever you intend to make a claim about *which* coefficients are
    non-zero, rather than about one pre-specified coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.array([1.0, -0.5]); C = np.eye(2) * 0.04
    >>> bands, crit = simultaneous_ci(b, C, n_boot=500, seed=0)
    >>> bool(crit > 1.96)
    True
    """
    d = len(beta)
    se = np.sqrt(np.clip(np.diag(cov), 1e-300, np.inf))
    corr = cov / np.outer(se, se)
    corr = np.nan_to_num(corr, nan=0.0)
    corr[np.diag_indices(d)] = 1.0

    # nearest PSD via eigenvalue clipping
    w, V = np.linalg.eigh(corr)
    w = np.clip(w, 1e-10, None)
    L = V @ np.diag(np.sqrt(w))

    rng = np.random.default_rng(seed)
    draws = np.abs(L @ rng.standard_normal((d, n_boot))).max(axis=0)
    crit = float(np.quantile(draws, 1 - alpha))
    bands = np.column_stack([beta - crit * se, beta + crit * se])
    return bands, crit


@dataclass
class OrthoResult:
    """Container for orthogonal-Lasso output."""

    beta_lasso: pd.Series
    beta_debiased: pd.Series
    cov: np.ndarray
    residual_y: np.ndarray
    residual_d: np.ndarray
    names: list[str]
    extra: dict = field(default_factory=dict)


class OrthogonalLasso:
    """Orthogonal / debiased Lasso for high-dimensional CATE in panels.

    Parameters
    ----------
    y : str
        Outcome column.
    p : str
        Base treatment column (low-dimensional, e.g. log price).
    controls : sequence of str
        Control variables entering the first stage.
    heterogeneity : sequence of str, optional
        Columns whose interaction with the treatment residual defines the
        CATE dictionary.  Categorical columns are one-hot encoded, so a single
        product-category column can generate hundreds of coefficients.
    k_blocks : int, default 10
        NLO cross-fitting blocks.  The paper recommends at least 10.
    mundlak : bool, default True
        Add within-unit means of the controls, giving a correlated-random-
        effects representation of the unit heterogeneity instead of
        differencing it away.
    second_stage : {'lasso', 'ols'}, default 'lasso'
        Use ``'ols'`` when the dictionary is low-dimensional.
    debias : {'clime', 'ridge', 'none'}, default 'clime'
        Approximate-inverse construction.  ``'ridge'`` is the cheap fallback
        used in the authors' own replication code and scales to large
        dictionaries; ``'clime'`` matches the paper's theory.
    clime_lambda : float, optional
    seed : int, default 0

    Examples
    --------
    >>> from dynpanelai.ortho import OrthogonalLasso
    >>> est = OrthogonalLasso(y="LogSales", p="LogPrice",
    ...                       controls=["LogPrice_lag", "LogSales_lag"],
    ...                       heterogeneity=["Level2"])
    >>> res = est.fit(panel)                        # doctest: +SKIP
    >>> est.elasticities().head()                   # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        p: str,
        controls: Sequence[str],
        heterogeneity: Sequence[str] | None = None,
        *,
        k_blocks: int = 10,
        mundlak: bool = True,
        second_stage: str = "lasso",
        debias: str = "clime",
        clime_lambda: float | None = None,
        seed: int = 0,
    ) -> None:
        self.y = y
        self.p = p
        self.controls = list(controls)
        self.heterogeneity = list(heterogeneity) if heterogeneity else []
        self.k_blocks = k_blocks
        self.mundlak = mundlak
        self.second_stage = second_stage
        self.debias = debias
        self.clime_lambda = clime_lambda
        self.seed = seed
        self.result_: OrthoResult | None = None

    # ------------------------------------------------------------------
    def _dictionary(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """One-hot encode the heterogeneity columns, with an intercept."""
        if not self.heterogeneity:
            return np.ones((len(frame), 1)), ["(average)"]
        parts = [pd.Series(1.0, index=frame.index, name="(average)")]
        for col in self.heterogeneity:
            if pd.api.types.is_numeric_dtype(frame[col]) and frame[col].nunique() > 20:
                parts.append(frame[col].rename(col))
            else:
                dummies = pd.get_dummies(frame[col], prefix=col, drop_first=True)
                parts.append(dummies.astype(float))
        K = pd.concat(parts, axis=1)
        return K.to_numpy(dtype=float), list(K.columns)

    def fit(self, panel: PanelData) -> PanelResults:
        """Run the three-stage procedure.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        PanelResults
            ``params`` holds the debiased CATE coefficients; ``extra`` carries
            the raw Lasso estimates, the residuals, and simultaneous bands.
        """
        frame = panel.df.copy()
        ctrl = list(self.controls)

        if self.mundlak:
            for c in self.controls + [self.p]:
                if pd.api.types.is_numeric_dtype(frame[c]):
                    name = f"mean_{c}"
                    frame[name] = frame.groupby("_i")[c].transform("mean")
                    ctrl.append(name)

        need = [self.y, self.p] + [c for c in ctrl if c in frame.columns]
        frame = frame.dropna(subset=need)
        if len(frame) == 0:
            raise ValueError("no complete observations after dropping missing values")

        X = frame[[c for c in ctrl if pd.api.types.is_numeric_dtype(frame[c])]].to_numpy(float)
        y = frame[self.y].to_numpy(float)
        p_var = frame[self.p].to_numpy(float)
        tpos = frame["_t"].to_numpy()
        units = frame["_i"].to_numpy()

        # ---- stage 1: NLO cross-fitted residuals -----------------------
        usable = np.sort(np.unique(tpos))
        folds = nlo_folds(usable, k=min(self.k_blocks, max(3, len(usable) // 2)))
        res_y = np.full(len(frame), np.nan)
        res_p = np.full(len(frame), np.nan)

        for fold in folds:
            test_mask = np.isin(tpos, usable[fold.test])
            train_mask = np.isin(tpos, usable[fold.train])
            if train_mask.sum() < 5 or test_mask.sum() == 0:
                continue
            fit_y = rlasso(X[train_mask], y[train_mask], post=True)
            fit_p = rlasso(X[train_mask], p_var[train_mask], post=True)
            res_y[test_mask] = y[test_mask] - fit_y.predict(X[test_mask])
            res_p[test_mask] = p_var[test_mask] - fit_p.predict(X[test_mask])

        ok = np.isfinite(res_y) & np.isfinite(res_p)
        frame = frame[ok]
        res_y, res_p = res_y[ok], res_p[ok]
        units = units[ok]

        # ---- stage 2: orthogonal (group) Lasso -------------------------
        K, names = self._dictionary(frame)
        V = K * res_p[:, None]
        n, d = V.shape

        if self.second_stage == "ols":
            beta_l, *_ = np.linalg.lstsq(V, res_y, rcond=None)
        else:
            fit = rlasso(V, res_y, post=False, intercept=False)
            beta_l = fit.coef

        # ---- stage 3: debias -------------------------------------------
        Q = V.T @ V / n
        if self.debias == "clime":
            lam = self.clime_lambda or np.sqrt(np.log(max(d, 2)) / n)
            Omega = clime(Q, lam=lam)
        elif self.debias == "ridge":
            lam = self.clime_lambda or np.sqrt(np.log(max(d, 2)) / n)
            Omega = np.linalg.pinv(Q + lam * np.eye(d))
        elif self.debias == "none":
            Omega = np.zeros((d, d))
        else:
            raise ValueError("debias must be 'clime', 'ridge' or 'none'")

        score = V.T @ (res_y - V @ beta_l) / n
        beta_db = beta_l + Omega @ score

        # ---- variance: cluster on unit ---------------------------------
        u = res_y - V @ beta_db
        contrib = V * u[:, None]
        labels, codes = np.unique(units, return_inverse=True)
        sums = np.zeros((len(labels), d))
        np.add.at(sums, codes, contrib)
        Gamma = sums.T @ sums / n**2
        cov = Omega @ Gamma @ Omega.T if self.debias != "none" else np.linalg.pinv(Q) @ Gamma @ np.linalg.pinv(Q)

        bands, crit = simultaneous_ci(beta_db, cov, seed=self.seed)

        res = PanelResults(
            params=pd.Series(beta_db, index=names),
            cov=cov,
            method=(
                f"Orthogonal {'Lasso' if self.second_stage == 'lasso' else 'OLS'} "
                f"+ debiasing ({self.debias}), NLO cross-fitting"
            ),
            n_obs=n,
            n_units=len(labels),
            n_periods=len(usable),
            dependent=self.y,
            diagnostics={
                "dictionary size": d,
                "NLO blocks": len(folds),
                "simultaneous crit. value": float(crit),
                "Mundlak means": self.mundlak,
            },
            extra={
                "beta_lasso": pd.Series(beta_l, index=names),
                "residual_y": res_y,
                "residual_p": res_p,
                "simultaneous_bands": pd.DataFrame(
                    bands, index=names, columns=["lower", "upper"]
                ),
                "frame": frame,
                "K": K,
            },
        )
        self.results_ = res
        return res


def orthogonal_lasso(panel: PanelData, y: str, p: str, controls, **kwargs) -> PanelResults:
    """Functional wrapper around :class:`OrthogonalLasso`."""
    return OrthogonalLasso(y=y, p=p, controls=controls, **kwargs).fit(panel)
