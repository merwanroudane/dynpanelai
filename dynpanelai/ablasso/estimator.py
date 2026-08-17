"""Arellano-Bond LASSO estimator for dynamic linear panel models.

Implements

    Chernozhukov, V., Fernandez-Val, I., Huang, C. and Wang, W. (2024).
    Arellano-Bond LASSO estimator for dynamic linear panel models.

The problem
-----------
The Arellano-Bond estimator uses every lag as an instrument, so the number of
moment conditions grows like :math:`T^2`.  Overfitting in the projection of
the endogenous regressors onto the instruments then produces a bias of order
:math:`m/n = T/N`, and valid inference needs the small-bias condition
:math:`m^2/n = T^3/N \\to 0`.  In the paper's COVID application
(:math:`m = 3375`, :math:`NT \\approx 68{,}000`) that quantity is about 168 --
the standard estimator is badly biased.

The fix
-------
Two steps.

1. **Select moments.**  For each period ``t`` separately, LASSO-regress each
   transformed regressor on the full instrument history
   :math:`X_{i1},\\dots,X_{it}`.  The moment conditions are approximately
   sparse, with effective dimension :math:`\\min(\\log N, t)`.

2. **Estimate by IV** using the fitted values as instruments.

   .. math::
       \\widehat\\theta = \\Bigl(\\sum_{i,t}
           \\widehat{\\Delta X}_{it}\\Delta X_{it}'\\Bigr)^{-1}
           \\sum_{i,t}\\widehat{\\Delta X}_{it}\\Delta Y_{it}.

   Note this is the **IV** form, not 2SLS: only the IV version has a
   Neyman-orthogonal moment function with respect to the first-stage
   coefficients (Remark 2.3 of the paper).

Fixed effects are removed by **forward orthogonal deviations** by default.
FOD leaves the transformed error serially uncorrelated, which both improves
efficiency over first differences and means cross-fitting does not change the
large-sample properties.

``AB-LASSO-SS`` adds cross-sectional sample splitting with cross-fitting and
aggregates over many random splits by the median, which removes the remaining
overfitting bias and makes the estimator invariant to the ordering of units.

Warning
-------
The CRAN ``ablasso`` package implements the *first-difference* variant from an
earlier draft.  This module defaults to ``transform='fod'`` to match the
published paper; pass ``transform='fd'`` to reproduce the R package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.panel import PanelData
from ..core.results import PanelResults
from ..core.transforms import fod_matrix
from ..penalized.rlasso import rlasso

__all__ = ["ABLasso", "ab_lasso"]


def _transform_wide(Y: np.ndarray, method: str, time_demean: bool) -> np.ndarray:
    """Apply FOD or FD to a ``(T, N)`` matrix, returning ``(T-1, N)``."""
    T = Y.shape[0]
    if method == "fod":
        out = fod_matrix(T) @ Y
    elif method == "fd":
        out = np.diff(Y, axis=0)
    else:
        raise ValueError("transform must be 'fod' or 'fd'")
    if time_demean:
        out = out - np.nanmean(out, axis=1, keepdims=True)
    return out


class ABLasso:
    """Arellano-Bond LASSO estimator.

    Parameters
    ----------
    y : str
        Outcome column.
    d : str, optional
        Treatment column, entering contemporaneously.
    c : sequence of str, optional
        Other predetermined covariates, entering at ``t-1``.
    lags : int, default 1
        Number of lags of the outcome to include as regressors.
    transform : {'fod', 'fd'}, default 'fod'
        Fixed-effect removal.  ``'fod'`` matches the published paper.
    split : bool, default True
        Use cross-sectional sample splitting with cross-fitting
        (``AB-LASSO-SS``).  Strongly recommended; the unsplit version retains
        an overfitting bias of order :math:`\\sqrt{s^*/N}`.
    k_folds : int, default 2
        Folds for cross-fitting.  The paper studies 2 and 5.
    n_splits : int, default 100
        Random re-splits, aggregated by the median.
    post : bool, default True
        Post-LASSO refit in the first stage (Remark 2.2).
    lambda_c : float, default 1.1
        Slack constant in the penalty.
    lambda_rule : {'plugin', 'paper'}, default 'paper'
        ``'paper'`` uses the replication code's
        :math:`\\lambda = 1.1\\sqrt N\\,\\Phi^{-1}(1-0.1/2p)`;
        ``'plugin'`` uses the full iterative ``rlasso`` penalty.
    time_demean : bool, default True
        Remove additive time effects by cross-sectional demeaning.
    seed : int, default 202304

    Examples
    --------
    >>> from dynpanelai.ablasso import ABLasso
    >>> est = ABLasso(y="logdc", d="dlogtests",
    ...               c=["school", "college", "pmask"], lags=4)
    >>> res = est.fit(panel)                       # doctest: +SKIP
    >>> print(res.summary())                       # doctest: +SKIP
    >>> res.long_run("dlogtests", [f"L{j}.logdc" for j in range(1, 5)])  # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        d: str | None = None,
        c: Sequence[str] | None = None,
        *,
        lags: int = 1,
        transform: str = "fod",
        split: bool = True,
        k_folds: int = 2,
        n_splits: int = 100,
        post: bool = True,
        lambda_c: float = 1.1,
        lambda_rule: str = "paper",
        time_demean: bool = True,
        seed: int = 202304,
    ) -> None:
        self.y = y
        self.d = d
        self.c = list(c) if c else []
        self.lags = lags
        self.transform = transform.lower()
        self.split = split
        self.k_folds = k_folds
        self.n_splits = n_splits
        self.post = post
        self.lambda_c = lambda_c
        self.lambda_rule = lambda_rule
        self.time_demean = time_demean
        self.seed = seed
        self.results_: PanelResults | None = None

    # ------------------------------------------------------------------
    def _names(self) -> list[str]:
        names = [f"L{j}.{self.y}" for j in range(1, self.lags + 1)]
        if self.d is not None:
            names.append(self.d)
        names += [f"L1.{v}" for v in self.c]
        return names

    def _lambda_for(self, n: int, p: int) -> float | None:
        if self.lambda_rule == "paper":
            return self.lambda_c * np.sqrt(n) * stats.norm.ppf(1 - 0.1 / (2 * p))
        return None

    def _first_stage(
        self, X_hist: np.ndarray, target: np.ndarray, fit_idx: np.ndarray, pred_idx: np.ndarray
    ) -> np.ndarray:
        """LASSO-project ``target`` on the instrument history, predict out of sample."""
        p = X_hist.shape[1]
        lam = self._lambda_for(len(fit_idx), max(p, 2))
        fit = rlasso(
            X_hist[fit_idx],
            target[fit_idx],
            post=self.post,
            intercept=True,
            lambda_start=lam,
            c=self.lambda_c,
        )
        return fit.predict(X_hist[pred_idx])

    # ------------------------------------------------------------------
    def fit(self, panel: PanelData) -> PanelResults:
        """Estimate the model.

        Parameters
        ----------
        panel : PanelData
            Must be balanced.  Call ``panel.balance()`` first if needed.

        Returns
        -------
        PanelResults

        Raises
        ------
        ValueError
            If the panel is unbalanced or too short for the requested lags.
        """
        if not panel.balanced:
            raise ValueError(
                "AB-LASSO requires a balanced panel; call panel.balance() "
                "first, or drop units with missing periods"
            )
        T, N = panel.T, panel.N
        if T < self.lags + 3:
            raise ValueError(
                f"T={T} is too short for lags={self.lags}; need T >= {self.lags + 3}"
            )

        Yw = panel.wide(self.y)
        Dw = panel.wide(self.d) if self.d is not None else None
        Cw = [panel.wide(v) for v in self.c]

        rng = np.random.default_rng(self.seed)
        names = self._names()
        k = len(names)

        n_rep = self.n_splits if self.split else 1
        thetas = np.zeros((n_rep, k))
        ses = np.zeros((n_rep, k))

        for rep in range(n_rep):
            if self.split:
                order = rng.permutation(N)
                folds = np.array_split(order, self.k_folds)
            else:
                folds = [np.arange(N)]

            theta_folds = []
            # accumulate for variance using the full sample
            Z_store: dict[int, np.ndarray] = {}
            R_store: dict[int, np.ndarray] = {}
            y_store: dict[int, np.ndarray] = {}

            for main in folds:
                aux = np.setdiff1d(np.arange(N), main) if self.split else main

                # transform separately within each subsample, as the paper does
                def tr(mat, idx):
                    return _transform_wide(mat[:, idx], self.transform, self.time_demean)

                Y_t_main = tr(Yw, main)
                Y_t_aux = tr(Yw, aux)
                D_t_main = tr(Dw, main) if Dw is not None else None
                D_t_aux = tr(Dw, aux) if Dw is not None else None
                C_t_main = [tr(m, main) for m in Cw]
                C_t_aux = [tr(m, aux) for m in Cw]

                Tt = Y_t_main.shape[0]
                t_start = self.lags
                t_end = Tt - 1  # need Y_t at t+? ; keep one period of slack

                Zs, Rs, ys = [], [], []
                for t in range(t_start, t_end + 1):
                    # instrument history available at t: levels up to t
                    hist_main = [Yw[: t, main].T]
                    hist_aux = [Yw[: t, aux].T]
                    if Dw is not None:
                        hist_main.append(Dw[: t + 1, main].T)
                        hist_aux.append(Dw[: t + 1, aux].T)
                    for m in Cw:
                        hist_main.append(m[: t, main].T)
                        hist_aux.append(m[: t, aux].T)
                    H_main = np.hstack(hist_main)
                    H_aux = np.hstack(hist_aux)

                    # endogenous regressors at t
                    regs_main, regs_aux = [], []
                    for j in range(1, self.lags + 1):
                        regs_main.append(Y_t_main[t - j])
                        regs_aux.append(Y_t_aux[t - j])
                    if D_t_main is not None:
                        regs_main.append(D_t_main[t])
                        regs_aux.append(D_t_aux[t])
                    for m_main, m_aux in zip(C_t_main, C_t_aux):
                        regs_main.append(m_main[t - 1])
                        regs_aux.append(m_aux[t - 1])
                    R_t = np.column_stack(regs_main)
                    R_t_aux = np.column_stack(regs_aux)

                    # LASSO on the auxiliary sample, predict on the main sample
                    Z_t = np.zeros_like(R_t)
                    for col in range(k):
                        if self.split:
                            X_all = np.vstack([H_aux, H_main])
                            tgt = np.concatenate([R_t_aux[:, col], R_t[:, col]])
                            fit_idx = np.arange(len(H_aux))
                            pred_idx = np.arange(len(H_aux), len(X_all))
                            Z_t[:, col] = self._first_stage(X_all, tgt, fit_idx, pred_idx)
                        else:
                            idx = np.arange(len(H_main))
                            Z_t[:, col] = self._first_stage(
                                H_main, R_t[:, col], idx, idx
                            )
                    Zs.append(Z_t)
                    Rs.append(R_t)
                    ys.append(Y_t_main[t])

                Z = np.vstack(Zs)
                R = np.vstack(Rs)
                yv = np.concatenate(ys)

                A = Z.T @ R
                b = Z.T @ yv
                try:
                    theta_folds.append(np.linalg.solve(A, b))
                except np.linalg.LinAlgError:
                    theta_folds.append(np.linalg.pinv(A) @ b)

                fid = len(Z_store)
                Z_store[fid] = np.stack(Zs)
                R_store[fid] = np.stack(Rs)
                y_store[fid] = np.stack(ys)

            theta = np.mean(theta_folds, axis=0)
            thetas[rep] = theta
            ses[rep] = self._variance(Z_store, R_store, y_store, theta)

        theta_hat = np.median(thetas, axis=0)
        se_hat = np.median(ses, axis=0)

        cov = np.diag(se_hat**2)
        res = PanelResults(
            params=pd.Series(theta_hat, index=names),
            cov=cov,
            method=(
                f"AB-LASSO{'-SS' if self.split else ''} "
                f"({self.transform.upper()}"
                f"{f', {self.k_folds} folds x {self.n_splits} splits' if self.split else ''})"
            ),
            n_obs=int(N * (T - self.lags - 1)),
            n_units=N,
            n_periods=T,
            dependent=self.y,
            diagnostics={
                "transform": self.transform.upper(),
                "lags": self.lags,
                "cross-fitting": f"{self.k_folds} folds" if self.split else "none",
                "random splits": self.n_splits if self.split else 0,
                "first stage": "post-LASSO" if self.post else "LASSO",
                "penalty rule": self.lambda_rule,
            },
            extra={"theta_draws": thetas, "se_draws": ses},
        )
        self.results_ = res
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _variance(Z_store, R_store, y_store, theta) -> np.ndarray:
        """Clustered variance with a one-lag (MA(1)) correction.

        Mirrors the replication code: demean the per-unit score, form the
        contemporaneous outer product plus the first autocovariance, and
        sandwich with the Jacobian.
        """
        A_total = None
        Sigma = None
        for fid in Z_store:
            Z = Z_store[fid]          # (T_eff, n_i, k)
            R = R_store[fid]
            yv = y_store[fid]
            Teff, n_i, k = Z.shape

            resid = yv - np.einsum("tik,k->ti", R, theta)
            score = Z * resid[:, :, None]           # (T, n, k)
            mu = score.mean(axis=0)                  # (n, k)
            dev = score - mu[None, :, :]

            S0 = np.einsum("tik,til->kl", dev, dev)
            S1 = np.einsum("tik,til->kl", dev[:-1], dev[1:])
            w = (Teff - 1.0) / Teff if Teff > 1 else 0.0
            Sig = S0 + w * (S1 + S1.T)

            A = np.einsum("tik,til->kl", Z, R)
            A_total = A if A_total is None else A_total + A
            Sigma = Sig if Sigma is None else Sigma + Sig

        try:
            A_inv = np.linalg.inv(A_total)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A_total)
        V = A_inv @ Sigma @ A_inv.T
        return np.sqrt(np.clip(np.diag(V), 0, np.inf))


def ab_lasso(panel: PanelData, y: str, **kwargs) -> PanelResults:
    """Functional wrapper around :class:`ABLasso`.

    Parameters
    ----------
    panel : PanelData
    y : str
    **kwargs
        Forwarded to :class:`ABLasso`.

    Returns
    -------
    PanelResults

    Examples
    --------
    >>> res = ab_lasso(panel, y="y", d="d", lags=1,
    ...                n_splits=10)               # doctest: +SKIP
    """
    return ABLasso(y=y, **kwargs).fit(panel)
