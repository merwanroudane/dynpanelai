"""Uniform inference in high-dimensional dynamic panels.

Implements

    Kock, A. B. and Tang, H. (2019). Uniform inference in high-dimensional
    dynamic panel data models.  *Econometric Theory*.

Model
-----
.. math::
    y_{it} = \\sum_{\\ell=1}^{L}\\alpha_\\ell y_{i,t-\\ell}
             + x_{it}'\\beta + \\eta_i + \\varepsilon_{it},

where ``p`` and ``N`` may both exceed ``NT``.

Two ideas make this work.

**Differential penalisation.**  There are :math:`NT` observations available to
estimate each :math:`\\alpha_j` but only :math:`T` for each :math:`\\eta_i`, so
the objective penalises them at different rates:

.. math::
    \\mathcal L(\\gamma) = \\|y - \\Pi\\gamma\\|^2
        + 2\\lambda_N\\|\\alpha\\|_1
        + \\frac{2\\lambda_N}{\\sqrt N}\\|\\eta\\|_1,
    \\qquad
    \\lambda_N = \\sqrt{4M\\,NT\\,(\\log(p\\vee N))^3}.

The fixed effects are assumed only *weakly* sparse,
:math:`\\sum_i|\\eta_i|^\\nu\\le E` for some :math:`0<\\nu<1` -- no individual
effect need be exactly zero.  This is a genuine middle ground between random
effects (which forbid correlation with the regressors) and unrestricted fixed
effects (which cannot be estimated in high dimensions).

**Desparsification.**  The Lasso is biased, so inference uses

.. math::
    \\tilde\\gamma = \\hat\\gamma
        + \\widehat\\Theta\\,S^{-1}\\Pi'(y - \\Pi\\hat\\gamma)/\\ldots,

with :math:`\\widehat\\Theta = \\mathrm{diag}(\\widehat\\Theta_Z, I_N)` and
:math:`\\widehat\\Theta_Z` built from nodewise regressions.  The resulting
confidence bands are *honest*: valid uniformly over the parameter space, and
usable for a growing number of coefficients simultaneously.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.panel import PanelData
from ..core.results import PanelResults
from ..penalized.clime import nodewise_inverse
from ..penalized.rlasso import _shooting

__all__ = ["PanelLasso", "panel_lasso"]


class PanelLasso:
    """Weakly-sparse panel Lasso with desparsified inference.

    Parameters
    ----------
    y : str
        Outcome column.
    lags : int, default 1
        Number of lags :math:`L` of the outcome.
    x : sequence of str, optional
        Exogenous / predetermined covariates.
    lambda_M : float, default 0.5
        Constant :math:`M` in :math:`\\lambda_N`.
    penalize_fe : bool, default True
        Apply the :math:`\\lambda_N/\\sqrt N` penalty to the unit effects.
        Turning this off recovers an unpenalised dummy-variable Lasso.
    desparsify : bool, default True
        Compute the debiased estimator and its heteroskedasticity-robust
        variance.  This is the expensive step: ``p`` nodewise Lasso fits.
    seed : int, default 0

    Examples
    --------
    >>> from dynpanelai.hdpanel import PanelLasso
    >>> est = PanelLasso(y="y", lags=2, x=[f"x{j}" for j in range(50)])
    >>> res = est.fit(panel)                        # doctest: +SKIP
    >>> print(res.summary())                        # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        *,
        lags: int = 1,
        x: Sequence[str] | None = None,
        lambda_M: float = 0.5,
        penalize_fe: bool = True,
        desparsify: bool = True,
        seed: int = 0,
    ) -> None:
        self.y = y
        self.lags = lags
        self.x = list(x) if x else []
        self.lambda_M = lambda_M
        self.penalize_fe = penalize_fe
        self.desparsify = desparsify
        self.seed = seed
        self.results_: PanelResults | None = None

    # ------------------------------------------------------------------
    def fit(self, panel: PanelData) -> PanelResults:
        """Fit the panel Lasso and (optionally) desparsify.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        PanelResults
            ``params`` holds the :math:`\\alpha` and :math:`\\beta`
            coefficients.  The estimated fixed effects are in
            ``extra['fixed_effects']``.
        """
        lagged = panel.lag(self.y, self.lags)
        frame = panel.df.copy()
        for c in lagged.columns:
            frame[c] = lagged[c]
        z_cols = [f"{self.y}_lag{j}" for j in range(1, self.lags + 1)] + self.x
        frame = frame.dropna(subset=[self.y] + z_cols)

        y = frame[self.y].to_numpy(float)
        Z = frame[z_cols].to_numpy(float)
        units = frame["_i"].to_numpy()
        uniq = np.unique(units)
        N, n = len(uniq), len(y)
        p = Z.shape[1]
        T_bar = n / N

        D = np.zeros((n, N))
        D[np.arange(n), np.searchsorted(uniq, units)] = 1.0
        Pi = np.hstack([Z, D])

        lam_N = np.sqrt(
            4.0 * self.lambda_M * n * (np.log(max(p, N, 3)) ** 3)
        )
        pen = np.concatenate(
            [
                np.full(p, lam_N),
                np.full(N, lam_N / np.sqrt(N)) if self.penalize_fe else np.zeros(N),
            ]
        )

        gamma = _shooting(Pi, y, 2.0 * pen)
        alpha = gamma[:p]
        eta = gamma[p:]
        resid = y - Pi @ gamma

        names = [f"L{j}.{self.y}" for j in range(1, self.lags + 1)] + self.x

        if not self.desparsify:
            cov = np.full((p, p), np.nan)
            method = "Panel Lasso (weakly sparse fixed effects)"
        else:
            Theta_Z, tau2 = nodewise_inverse(Z)
            # desparsified coefficients for the alpha block
            correction = Theta_Z @ (Z.T @ resid) / n
            alpha = alpha + correction

            # heteroskedasticity-robust variance, clustered on the unit
            contrib = Z * resid[:, None]
            labels, codes = np.unique(units, return_inverse=True)
            sums = np.zeros((len(labels), p))
            np.add.at(sums, codes, contrib)
            Omega = sums.T @ sums / n**2
            cov = Theta_Z @ Omega @ Theta_Z.T
            method = "Desparsified panel Lasso (uniform inference)"

        # Weak sparsity is an assumption, not a conclusion: if every unit
        # effect is shrunk to zero while the within variation is large, the
        # data are dense and the estimator degenerates toward pooled OLS,
        # which is upward-biased on the lag in a dynamic panel.
        n_nonzero = int(np.sum(np.abs(eta) > 1e-8))
        within_sd = float(
            pd.Series(y).groupby(units).transform("mean").std(ddof=0)
        )
        if n_nonzero == 0 and within_sd > 0.5 * float(np.std(y)):
            warnings.warn(
                "all unit effects were shrunk to zero although between-unit "
                "variation is substantial: the weak-sparsity assumption "
                "(sum |eta_i|^nu <= E, with E small) looks violated. The "
                "estimator then behaves like pooled OLS and the lag "
                "coefficient will be biased upward. Lower `lambda_M`, or use "
                "dynpanelai.gmm / dynpanelai.ablasso instead.",
                UserWarning,
                stacklevel=2,
            )

        res = PanelResults(
            params=pd.Series(alpha, index=names),
            cov=cov,
            method=method,
            n_obs=n,
            n_units=N,
            n_periods=panel.T,
            dependent=self.y,
            diagnostics={
                "lambda_N": float(lam_N),
                "penalised fixed effects": self.penalize_fe,
                "non-zero fixed effects": int(np.sum(np.abs(eta) > 1e-8)),
                "sum |eta|^0.5 (weak sparsity)": float(np.sum(np.abs(eta) ** 0.5)),
                "avg T per unit": float(T_bar),
            },
            extra={
                "fixed_effects": pd.Series(eta, index=panel.units[uniq]),
                "residuals": resid,
                "lasso_alpha": gamma[:p],
            },
        )
        self.results_ = res
        return res


def panel_lasso(panel: PanelData, y: str, **kwargs) -> PanelResults:
    """Functional wrapper around :class:`PanelLasso`."""
    return PanelLasso(y=y, **kwargs).fit(panel)
