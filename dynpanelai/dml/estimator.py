"""Double machine learning for dynamic panels.

Implements the blocked-time cross-fitted partialling-out estimator of Sneller
(2026) for the partially linear dynamic panel

.. math::
    Y_{it} = \\rho_0 Y_{i,t-1} + \\theta_0 D_{it} + g_0(X_{it})
             + \\alpha_i + U_{it}.

After within-demeaning, the target :math:`\\theta_0` is identified by the
Neyman-orthogonal partialling-out moment

.. math::
    \\mathbb{E}\\bigl[\\tilde D_{it}(\\tilde Y_{it}
        - \\theta_0\\tilde D_{it})\\bigr] = 0,
    \\qquad
    \\tilde Y = \\ddot Y - \\ell_0(W),\\quad
    \\tilde D = \\ddot D - m_0(W),

with nuisances :math:`\\ell_0(w)=\\mathbb{E}[\\ddot Y_{it}\\mid W_{it}=w]` and
:math:`m_0(w)=\\mathbb{E}[\\ddot D_{it}\\mid W_{it}=w]` learned by any ML
method, and the information set

.. math::
    W_{it} = \\bigl(\\ddot Y_{i,t-1:t-L_y},\\;
                    \\ddot D_{i,t-1:t-L_d},\\;
                    \\ddot X_{i,t:t-L_x}\\bigr).

Orthogonality means nuisance error enters only at second order, so the
product-rate condition :math:`\\delta_{Y}\\delta_{D}=o(n^{-1/2})` suffices even
when each nuisance converges slowly.

Caveat
------
Because :math:`W_{it}` contains within-demeaned lagged outcomes, Nickell-type
contamination enters the moment at order :math:`O(1/T)`.  Valid
:math:`\\sqrt N` inference therefore requires :math:`\\sqrt N / T\\to 0`.  In
genuinely short panels use :mod:`dynpanelai.gmm` or :mod:`dynpanelai.ablasso`
instead; :meth:`DMLDynamicPanel.fit` warns when ``sqrt(N)/T`` is large.

References
----------
Chernozhukov, V. et al. (2018). Double/debiased machine learning.
*Econometrics Journal* 21(1), C1-C68.

Sneller, L. (2026). Double machine learning for dynamic panel data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.panel import PanelData
from ..core.results import PanelResults
from ..core.variance import (
    cluster_variance,
    driscoll_kraay_variance,
    twoway_cluster_variance,
)
from .folds import blocked_time_folds, buffer_rules, nlo_folds, suggest_buffer_acf

__all__ = ["DMLDynamicPanel", "dml_dynamic_panel"]


def _make_learner(name: str, seed: int, **kwargs):
    """Instantiate a nuisance learner by name."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = name.lower()
    if name == "lasso":
        model = LassoCV(
            alphas=np.logspace(-4, 1, kwargs.get("n_alphas", 50)),
            cv=kwargs.get("cv", 5),
            max_iter=kwargs.get("max_iter", 5000),
            random_state=seed,
        )
    elif name in {"enet", "elasticnet"}:
        model = ElasticNetCV(
            l1_ratio=kwargs.get("l1_ratio", [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]),
            cv=kwargs.get("cv", 5),
            max_iter=kwargs.get("max_iter", 5000),
            random_state=seed,
        )
    elif name in {"rf", "forest", "random_forest"}:
        model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 500),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            max_depth=kwargs.get("max_depth", None),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=seed,
        )
    elif name in {"gbm", "hgb", "boosting"}:
        model = HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 200),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=seed,
        )
    elif name == "ols":
        model = LinearRegression()
    else:
        raise ValueError(
            f"unknown learner {name!r}; expected one of "
            "'lasso', 'enet', 'rf', 'gbm', 'ols', or pass a fitted "
            "scikit-learn estimator directly"
        )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


class DMLDynamicPanel:
    """Blocked-time cross-fitted DML estimator for a dynamic panel.

    Parameters
    ----------
    y : str
        Outcome column.
    d : str
        Treatment / policy column whose coefficient is the target.
    x : sequence of str, optional
        Additional controls entering the information set.
    y_lags, d_lags, x_lags : int, default 1
        Lag depths :math:`L_y, L_d, L_x` in :math:`W_{it}`.  ``x_lags`` also
        admits the contemporaneous :math:`X_{it}`.
    k_folds : int, default 4
        Number of blocked-time folds.
    buffer : int or {'log', 'sqrt', 'acf'}, default 'log'
        Extra separation ``B`` between train and test blocks.  ``'log'`` gives
        :math:`\\lceil\\log T\\rceil` (the paper's default), ``'sqrt'`` gives
        :math:`\\lfloor\\sqrt T\\rfloor`, ``'acf'`` uses the autocorrelation
        heuristic.  ``0`` disables buffering -- a stress test, not a default.
    fold_scheme : {'blocked', 'nlo'}, default 'blocked'
        ``'nlo'`` uses neighbours-left-out blocks instead.
    learner : str or estimator, default 'lasso'
        Nuisance learner: ``'lasso'``, ``'enet'``, ``'rf'``, ``'gbm'``,
        ``'ols'``, or any object with scikit-learn ``fit``/``predict``.
    demeaning : {'global', 'fold'}, default 'global'
        ``'fold'`` recomputes unit means from training periods only, for
        strict fold purity at the cost of some efficiency.
    vcov : {'cluster', 'twoway', 'driscoll-kraay'}, default 'cluster'
        Variance estimator; ``'cluster'`` clusters on the unit.
    trim : float, optional
        Quantile of :math:`|\\tilde D|/\\mathrm{sd}(\\tilde D)` above which
        observations are dropped, e.g. ``0.995``.  ``None`` disables trimming.
    seed : int, default 42

    Attributes
    ----------
    results_ : PanelResults
        Populated by :meth:`fit`.

    Examples
    --------
    >>> from dynpanelai import PanelData
    >>> from dynpanelai.dml import DMLDynamicPanel
    >>> est = DMLDynamicPanel(y="logdc", d="school", x=["pmask", "pshelter"],
    ...                       k_folds=4, buffer="log", learner="lasso")
    >>> res = est.fit(panel)                        # doctest: +SKIP
    >>> print(res.summary())                        # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        d: str,
        x: Sequence[str] | None = None,
        *,
        y_lags: int = 1,
        d_lags: int = 1,
        x_lags: int = 1,
        k_folds: int = 4,
        buffer: int | str = "log",
        fold_scheme: str = "blocked",
        learner: Any = "lasso",
        demeaning: str = "global",
        vcov: str = "cluster",
        trim: float | None = None,
        seed: int = 42,
        learner_kwargs: dict | None = None,
    ) -> None:
        self.y = y
        self.d = d
        self.x = list(x) if x else []
        self.y_lags = y_lags
        self.d_lags = d_lags
        self.x_lags = x_lags
        self.k_folds = k_folds
        self.buffer = buffer
        self.fold_scheme = fold_scheme
        self.learner = learner
        self.demeaning = demeaning
        self.vcov = vcov
        self.trim = trim
        self.seed = seed
        self.learner_kwargs = learner_kwargs or {}
        self.results_: PanelResults | None = None

    # ------------------------------------------------------------------
    def _build_features(self, panel: PanelData) -> tuple[pd.DataFrame, list[str]]:
        """Assemble ``(Y, D, W)`` after within-demeaning, and name the W columns."""
        base = [self.y, self.d] + self.x
        missing = [c for c in base if c not in panel.df.columns]
        if missing:
            raise KeyError(f"columns not found in the panel: {missing}")

        work = panel.df.copy()
        lag_specs: list[tuple[str, int]] = []
        if self.y_lags > 0:
            lag_specs.append((self.y, self.y_lags))
        if self.d_lags > 0:
            lag_specs.append((self.d, self.d_lags))
        for c in self.x:
            if self.x_lags > 0:
                lag_specs.append((c, self.x_lags))

        w_cols: list[str] = []
        for col, L in lag_specs:
            lagged = panel.lag(col, L)
            for name in lagged.columns:
                work[name] = lagged[name]
                w_cols.append(name)
        # contemporaneous controls also belong in W
        w_cols = self.x + w_cols

        keep = [self.y, self.d] + w_cols
        work = work.dropna(subset=keep)
        return work, w_cols

    def _demean(
        self, frame: pd.DataFrame, cols: Sequence[str], group: np.ndarray
    ) -> pd.DataFrame:
        g = pd.Series(group, index=frame.index)
        out = frame[list(cols)].copy()
        for c in cols:
            out[c] = out[c] - out[c].groupby(g).transform("mean")
        return out

    # ------------------------------------------------------------------
    def fit(self, panel: PanelData) -> PanelResults:
        """Estimate :math:`\\theta_0`.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        PanelResults
            ``params`` holds the single coefficient on ``d``; ``diagnostics``
            reports the fold design, the residualised-treatment variance, and
            the Nickell-regime warning statistic :math:`\\sqrt N / T`.

        Raises
        ------
        ValueError
            If no observations survive lag construction, or the fold design is
            infeasible for the available number of periods.
        """
        work, w_cols = self._build_features(panel)
        if len(work) == 0:
            raise ValueError(
                "no observations remain after lag construction; reduce the "
                "lag depths or check for gaps in the time index"
            )

        units = work["_i"].to_numpy()
        tpos = work["_t"].to_numpy()
        usable = np.sort(np.unique(tpos))
        T_use = len(usable)
        max_lag = max(self.y_lags, self.d_lags, self.x_lags)

        # ---- buffer choice --------------------------------------------
        if isinstance(self.buffer, str):
            rules = buffer_rules(T_use)
            if self.buffer == "log":
                B = rules["log"]
            elif self.buffer == "sqrt":
                B = rules["sqrt"]
            elif self.buffer == "acf":
                ddot_y = self._demean(work, [self.y], units)[self.y]
                by_t = ddot_y.groupby(pd.Series(tpos, index=work.index)).mean()
                B = suggest_buffer_acf(by_t.to_numpy())
            else:
                raise ValueError(
                    f"unknown buffer rule {self.buffer!r}; use an int or one "
                    "of 'log', 'sqrt', 'acf'"
                )
        else:
            B = int(self.buffer)

        # ---- folds -----------------------------------------------------
        if self.fold_scheme == "nlo":
            folds = nlo_folds(usable, k=self.k_folds)
        else:
            folds = blocked_time_folds(
                usable, k=self.k_folds, buffer=B, max_lag=max_lag
            )

        # ---- global demeaning (default) --------------------------------
        all_cols = [self.y, self.d] + w_cols
        if self.demeaning == "global":
            dd = self._demean(work, all_cols, units)
        elif self.demeaning != "fold":
            raise ValueError("demeaning must be 'global' or 'fold'")

        y_res, d_res, keep_units, keep_t = [], [], [], []
        fold_diag = []

        for f_idx, fold in enumerate(folds):
            test_periods = usable[fold.test]
            train_periods = usable[fold.train]
            test_mask = np.isin(tpos, test_periods)
            train_mask = np.isin(tpos, train_periods)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            if self.demeaning == "fold":
                means = (
                    work.loc[train_mask, all_cols]
                    .groupby(units[train_mask])
                    .mean()
                )
                idx_tr = means.index.get_indexer(units[train_mask])
                idx_te = means.index.get_indexer(units[test_mask])
                if (idx_tr < 0).any() or (idx_te < 0).any():
                    raise ValueError(
                        "fold-specific demeaning failed: some units have no "
                        "training observations in a fold. Use "
                        "demeaning='global' for highly unbalanced panels."
                    )
                mv = means.to_numpy()
                tr = work.loc[train_mask, all_cols].to_numpy() - mv[idx_tr]
                te = work.loc[test_mask, all_cols].to_numpy() - mv[idx_te]
            else:
                tr = dd.loc[train_mask, all_cols].to_numpy()
                te = dd.loc[test_mask, all_cols].to_numpy()

            y_tr, d_tr, W_tr = tr[:, 0], tr[:, 1], tr[:, 2:]
            y_te, d_te, W_te = te[:, 0], te[:, 1], te[:, 2:]

            if isinstance(self.learner, str):
                model_y = _make_learner(self.learner, self.seed, **self.learner_kwargs)
                model_d = _make_learner(self.learner, self.seed, **self.learner_kwargs)
            else:
                from sklearn.base import clone

                model_y = clone(self.learner)
                model_d = clone(self.learner)

            model_y.fit(W_tr, y_tr)
            model_d.fit(W_tr, d_tr)

            y_res.append(y_te - model_y.predict(W_te))
            d_res.append(d_te - model_d.predict(W_te))
            keep_units.append(units[test_mask])
            keep_t.append(tpos[test_mask])
            fold_diag.append(
                {"fold": f_idx + 1, "n_train": int(train_mask.sum()),
                 "n_test": int(test_mask.sum())}
            )

        y_tilde = np.concatenate(y_res)
        d_tilde = np.concatenate(d_res)
        u = np.concatenate(keep_units)
        t = np.concatenate(keep_t)

        # ---- optional trimming ----------------------------------------
        trim_report: dict[str, Any] = {"enabled": False}
        if self.trim is not None:
            sd = float(np.std(d_tilde))
            z = np.abs(d_tilde / sd) if sd > 0 else np.zeros_like(d_tilde)
            cut = float(np.quantile(z, self.trim))
            keep = z <= cut
            trim_report = {
                "enabled": True,
                "rule": f"quantile {self.trim}",
                "cutoff": cut,
                "trimmed": int((~keep).sum()),
                "fraction": float((~keep).mean()),
            }
            y_tilde, d_tilde, u, t = y_tilde[keep], d_tilde[keep], u[keep], t[keep]

        # ---- moment solution -------------------------------------------
        denom = float(d_tilde @ d_tilde)
        if denom <= 0:
            raise ValueError(
                "the residualised treatment has no variation; the treatment is "
                "fully explained by the controls (no overlap)"
            )
        theta = float(d_tilde @ y_tilde / denom)

        psi = d_tilde * (y_tilde - theta * d_tilde)
        n = len(y_tilde)
        J = -denom / n

        if self.vcov == "cluster":
            omega = cluster_variance(psi, u)
        elif self.vcov == "twoway":
            omega = twoway_cluster_variance(psi, u, t)
        elif self.vcov in {"driscoll-kraay", "dk"}:
            omega = driscoll_kraay_variance(psi, t)
        else:
            raise ValueError(
                f"unknown vcov {self.vcov!r}; use 'cluster', 'twoway' or "
                "'driscoll-kraay'"
            )
        var = float(omega[0, 0]) / (n**2 * J**2)

        n_units = len(np.unique(u))
        nickell = np.sqrt(n_units) / max(T_use, 1)
        if nickell > 1.0:
            warnings.warn(
                f"sqrt(N)/T = {nickell:.2f} is large. The DML partialling-out "
                "estimator needs sqrt(N)/T -> 0 for valid inference when the "
                "information set contains within-demeaned lagged outcomes. "
                "Consider dynpanelai.ablasso or dynpanelai.gmm for short panels.",
                UserWarning,
                stacklevel=2,
            )

        res = PanelResults(
            params=pd.Series({self.d: theta}),
            cov=np.array([[var]]),
            method=(
                f"DML dynamic panel (partialling-out, {self.fold_scheme} folds, "
                f"learner={self.learner if isinstance(self.learner, str) else 'custom'})"
            ),
            n_obs=n,
            n_units=n_units,
            n_periods=T_use,
            dependent=self.y,
            diagnostics={
                "folds": self.k_folds,
                "buffer B": B,
                "effective buffer B*": B + max_lag,
                "Var(D residualised)": float(np.var(d_tilde)),
                "sqrt(N)/T": float(nickell),
                "vcov": self.vcov,
                "trimming": trim_report,
            },
            extra={
                "y_tilde": y_tilde,
                "d_tilde": d_tilde,
                "units": u,
                "fold_diagnostics": fold_diag,
                "w_cols": w_cols,
            },
        )
        self.results_ = res
        return res


def dml_dynamic_panel(panel: PanelData, y: str, d: str, **kwargs) -> PanelResults:
    """Functional wrapper around :class:`DMLDynamicPanel`.

    Parameters
    ----------
    panel : PanelData
    y, d : str
    **kwargs
        Forwarded to :class:`DMLDynamicPanel`.

    Returns
    -------
    PanelResults

    Examples
    --------
    >>> res = dml_dynamic_panel(panel, y="logdc", d="school",
    ...                         x=["pmask"], k_folds=4)   # doctest: +SKIP
    """
    return DMLDynamicPanel(y=y, d=d, **kwargs).fit(panel)
