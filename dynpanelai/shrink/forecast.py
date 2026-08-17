"""Shrinkage estimators for dynamic panel forecasting.

Implements the estimator suite compared in

    Cornejo, M. and Sosa-Escudero, W. (2026). Machine learning and shrinkage
    in dynamic panel forecasting.

The organising idea is the bias-variance trade-off.  Pooled OLS imposes
:math:`\\eta_i = 0` for every unit (maximum bias, minimum variance); the
within/LSDV estimator gives every unit its own :math:`\\eta_i` (minimum bias,
maximum variance).  Shrinkage sits in between, and when the fixed effects are
sparse or weakly dispersed it beats both on out-of-sample MSE -- even though
IV estimators such as Anderson-Hsiao and Arellano-Bond dominate on *bias*.

The key implementation detail, easy to get wrong: the penalty is applied
**only to the fixed effects**.  The autoregressive coefficient and the slopes
on the observed regressors are left unpenalised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.panel import PanelData

__all__ = [
    "PenalizedFE",
    "rolling_origin_blocks",
    "select_lambda_rolling",
    "forecast_metrics",
]


def rolling_origin_blocks(periods: Sequence, k: int = 5) -> list[np.ndarray]:
    """Split periods into ``k`` sequential blocks for rolling-origin CV.

    Parameters
    ----------
    periods : sequence
        Sorted distinct periods.
    k : int, default 5

    Returns
    -------
    list of ndarray
        Contiguous blocks of period labels, earliest first.

    Examples
    --------
    >>> [len(b) for b in rolling_origin_blocks(range(20), k=5)]
    [4, 4, 4, 4, 4]
    """
    arr = np.asarray(sorted(set(periods)))
    if k < 2:
        raise ValueError("rolling-origin CV needs at least 2 blocks")
    if len(arr) < k:
        raise ValueError(f"cannot build {k} blocks from {len(arr)} periods")
    return list(np.array_split(arr, k))


def select_lambda_rolling(
    X: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    lambdas: np.ndarray,
    *,
    penalty_factor: np.ndarray,
    l1_ratio: float = 1.0,
    k_blocks: int = 5,
    one_se: bool = True,
) -> float:
    """Choose a penalty by rolling-origin blocked CV with the 1-SE rule.

    Block ``k`` is validated using a model fit on blocks ``1..k-1``, so the
    training window never contains information from the future -- the panel
    analogue of a proper time-series backtest.

    Parameters
    ----------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    periods : ndarray of shape (n,)
        Period label for each row.
    lambdas : ndarray
        Candidate penalties, descending.
    penalty_factor : ndarray of shape (p,)
        Per-coefficient multiplier; zero leaves a coefficient unpenalised.
    l1_ratio : float, default 1.0
        1 = LASSO, 0 = ridge, in between = elastic net.
    k_blocks : int, default 5
    one_se : bool, default True
        Return the largest penalty within one standard error of the minimum
        validation loss -- the more parsimonious choice.

    Returns
    -------
    float
        The selected penalty.
    """
    from sklearn.linear_model import ElasticNet

    blocks = rolling_origin_blocks(np.unique(periods), k=k_blocks)
    losses = np.full((len(lambdas), len(blocks) - 1), np.nan)

    for col, kk in enumerate(range(1, len(blocks))):
        train_p = np.concatenate(blocks[:kk])
        val_p = blocks[kk]
        tr = np.isin(periods, train_p)
        va = np.isin(periods, val_p)
        if tr.sum() == 0 or va.sum() == 0:
            continue
        Xw = X / np.where(penalty_factor > 0, penalty_factor, 1.0)
        for row, lam in enumerate(lambdas):
            if lam <= 0:
                model = ElasticNet(alpha=1e-12, l1_ratio=l1_ratio, fit_intercept=False)
            else:
                model = ElasticNet(alpha=lam, l1_ratio=l1_ratio, fit_intercept=False,
                                   max_iter=5000)
            free = penalty_factor == 0
            model.fit(Xw[tr], y[tr])
            coef = model.coef_ / np.where(penalty_factor > 0, penalty_factor, 1.0)
            if free.any():
                # refit unpenalised block conditional on the penalised part
                resid = y[tr] - X[tr][:, ~free] @ coef[~free]
                b_free, *_ = np.linalg.lstsq(X[tr][:, free], resid, rcond=None)
                coef[free] = b_free
            losses[row, col] = float(np.mean((y[va] - X[va] @ coef) ** 2))

    mean_loss = np.nanmean(losses, axis=1)
    if np.all(np.isnan(mean_loss)):
        return float(lambdas[len(lambdas) // 2])
    j_min = int(np.nanargmin(mean_loss))
    if not one_se:
        return float(lambdas[j_min])
    n_val = np.sum(~np.isnan(losses), axis=1)
    se = np.nanstd(losses, axis=1) / np.sqrt(np.maximum(n_val, 1))
    thresh = mean_loss[j_min] + se[j_min]
    eligible = lambdas[mean_loss <= thresh]
    return float(np.max(eligible)) if len(eligible) else float(lambdas[j_min])


@dataclass
class ForecastResult:
    """Output of a shrinkage forecasting fit.

    Attributes
    ----------
    gamma : float
        Estimated autoregressive coefficient.
    beta : ndarray
        Slopes on the observed regressors.
    fixed_effects : pandas.Series
        Estimated (and possibly shrunk) unit effects.
    method : str
    lam : float
        Selected penalty, where applicable.
    """

    gamma: float
    beta: np.ndarray
    fixed_effects: pd.Series
    method: str
    lam: float = 0.0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ForecastResult(method={self.method!r}, gamma={self.gamma:.4f})"


class PenalizedFE:
    """Dynamic panel forecasting with shrinkage on the fixed effects.

    Parameters
    ----------
    y : str
        Outcome column.
    x : sequence of str, optional
        Observed regressors (excluding the lagged dependent variable, which is
        added automatically).
    method : str, default 'lasso'
        One of

        ``'pols'``
            Pooled OLS -- no fixed effects at all.
        ``'fe'`` / ``'lsdv'``
            Unpenalised dummy-variable estimator.
        ``'lasso'``, ``'ridge'``, ``'enet'``
            Penalise the unit dummies only, tuning by rolling-origin CV.
        ``'ebmle'``, ``'ure'``
            Fit LSDV, then shrink the estimated effects optimally
            (:func:`~dynpanelai.shrink.fe_shrink.fe_shrink`).

    l1_ratio : float, default 0.5
        Elastic-net mixing when ``method='enet'``.
    k_blocks : int, default 5
        Rolling-origin CV blocks.
    n_lambda : int, default 60
    one_se : bool, default True

    Examples
    --------
    >>> from dynpanelai.shrink import PenalizedFE
    >>> est = PenalizedFE(y="y", x=["x"], method="ure")
    >>> res = est.fit(panel)                    # doctest: +SKIP
    >>> yhat = est.predict(panel)               # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        x: Sequence[str] | None = None,
        *,
        method: str = "lasso",
        l1_ratio: float = 0.5,
        k_blocks: int = 5,
        n_lambda: int = 60,
        one_se: bool = True,
    ) -> None:
        self.y = y
        self.x = list(x) if x else []
        self.method = method.lower()
        self.l1_ratio = l1_ratio
        self.k_blocks = k_blocks
        self.n_lambda = n_lambda
        self.one_se = one_se
        self.result_: ForecastResult | None = None

    # ------------------------------------------------------------------
    def _design(self, panel: PanelData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        lagged = panel.lag(self.y, 1)[f"{self.y}_lag1"]
        frame = panel.df.copy()
        frame["_lagy"] = lagged
        cols = ["_lagy"] + self.x
        frame = frame.dropna(subset=[self.y] + cols)

        units = frame["_i"].to_numpy()
        periods = frame["_t"].to_numpy()
        y = frame[self.y].to_numpy(dtype=float)
        core = frame[cols].to_numpy(dtype=float)

        uniq = np.unique(units)
        dummies = np.zeros((len(frame), len(uniq)))
        dummies[np.arange(len(frame)), np.searchsorted(uniq, units)] = 1.0
        X = np.hstack([core, dummies])
        return X, y, units, periods, list(uniq)

    def fit(self, panel: PanelData) -> ForecastResult:
        """Estimate the model.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        ForecastResult
        """
        X, y, units, periods, uniq = self._design(panel)
        k_core = 1 + len(self.x)
        n_units = len(uniq)

        if self.method == "pols":
            Xc = np.hstack([np.ones((len(y), 1)), X[:, :k_core]])
            coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            fe = pd.Series(coef[0], index=uniq)
            res = ForecastResult(float(coef[1]), coef[2:], fe, "Pooled OLS")

        elif self.method in {"fe", "lsdv", "within"}:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            fe = pd.Series(coef[k_core:], index=uniq)
            res = ForecastResult(float(coef[0]), coef[1:k_core], fe, "LSDV")

        elif self.method in {"lasso", "ridge", "enet"}:
            l1 = {"lasso": 1.0, "ridge": 0.0, "enet": self.l1_ratio}[self.method]
            pf = np.concatenate([np.zeros(k_core), np.ones(n_units)])
            lam_max = float(np.max(np.abs(X[:, k_core:].T @ y)) / len(y))
            lambdas = np.logspace(np.log10(max(lam_max, 1e-6)), np.log10(max(lam_max, 1e-6)) - 4,
                                  self.n_lambda)
            lam = select_lambda_rolling(
                X, y, periods, lambdas,
                penalty_factor=pf, l1_ratio=l1 if l1 > 0 else 1e-6,
                k_blocks=self.k_blocks, one_se=self.one_se,
            )
            coef = self._fit_penalized(X, y, k_core, lam, l1)
            fe = pd.Series(coef[k_core:], index=uniq)
            res = ForecastResult(
                float(coef[0]), coef[1:k_core], fe,
                {"lasso": "LASSO-FE", "ridge": "Ridge-FE", "enet": "Elastic-Net-FE"}[self.method],
                lam=lam,
            )

        elif self.method in {"ebmle", "ure"}:
            from .fe_shrink import fe_shrink

            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            dof = max(len(y) - X.shape[1], 1)
            sigma2 = float(resid @ resid / dof)
            XtX_inv = np.linalg.pinv(X.T @ X)
            var_fe = sigma2 * np.diag(XtX_inv)[k_core:]
            alpha_hat = coef[k_core:]

            out = fe_shrink(
                alpha_hat, np.maximum(var_fe, 1e-12),
                method="URE" if self.method == "ure" else "EBMLE",
                centering="gen",
            )
            fe = pd.Series(out.theta.ravel(), index=uniq)
            res = ForecastResult(
                float(coef[0]), coef[1:k_core], fe,
                "URE" if self.method == "ure" else "EBMLE",
                extra={"shrinkage": out.shrinkage, "Lambda": out.Lambda,
                       "mu": float(out.mu[0, 0])},
            )
        else:
            raise ValueError(
                f"unknown method {self.method!r}; expected 'pols', 'fe', "
                "'lasso', 'ridge', 'enet', 'ebmle' or 'ure'"
            )

        self.result_ = res
        self._uniq = uniq
        return res

    @staticmethod
    def _fit_penalized(X, y, k_core, lam, l1_ratio):
        """Penalise only the dummy block, profiling out the free coefficients."""
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(
            alpha=max(lam, 1e-12),
            l1_ratio=max(l1_ratio, 1e-6),
            fit_intercept=False,
            max_iter=10000,
        )
        model.fit(X[:, k_core:], y - X[:, :k_core] @ np.zeros(k_core))
        # alternate between free and penalised blocks until stable
        coef = np.zeros(X.shape[1])
        for _ in range(50):
            resid = y - X[:, k_core:] @ coef[k_core:]
            b_free, *_ = np.linalg.lstsq(X[:, :k_core], resid, rcond=None)
            model.fit(X[:, k_core:], y - X[:, :k_core] @ b_free)
            new = np.concatenate([b_free, model.coef_])
            if np.max(np.abs(new - coef)) < 1e-8:
                coef = new
                break
            coef = new
        return coef

    def predict(self, panel: PanelData) -> pd.Series:
        """One-step-ahead in-sample/out-of-sample predictions.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        pandas.Series
            Predictions aligned to the rows of ``panel.df`` that have a
            defined lag; other rows are ``NaN``.
        """
        if self.result_ is None:
            raise RuntimeError("call fit() before predict()")
        r = self.result_
        lagged = panel.lag(self.y, 1)[f"{self.y}_lag1"]
        frame = panel.df.copy()
        frame["_lagy"] = lagged
        cols = ["_lagy"] + self.x
        ok = frame[cols].notna().all(axis=1)

        fe_vals = frame["_i"].map(r.fixed_effects).to_numpy(dtype=float)
        pred = np.full(len(frame), np.nan)
        Xc = frame.loc[ok, cols].to_numpy(dtype=float)
        pred[ok.to_numpy()] = (
            r.gamma * Xc[:, 0]
            + (Xc[:, 1:] @ r.beta if len(self.x) else 0.0)
            + fe_vals[ok.to_numpy()]
        )
        return pd.Series(pred, index=frame.index, name=f"{self.y}_hat")


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Bias, prediction variance, MSE, RMSE and MAE.

    Parameters
    ----------
    y_true, y_pred : ndarray

    Returns
    -------
    dict
        Keys ``bias``, ``variance``, ``mse``, ``rmse``, ``mae``, ``n``.

    Notes
    -----
    ``variance`` is the variance of the *predictions*, matching the definition
    used in the Cornejo and Sosa-Escudero tables, not the error variance.

    Examples
    --------
    >>> m = forecast_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.8]))
    >>> round(m["rmse"], 4)
    0.1581
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() == 0:
        return {k: float("nan") for k in ("bias", "variance", "mse", "rmse", "mae", "n")}
    err = y_true[ok] - y_pred[ok]
    mse = float(np.mean(err**2))
    return {
        "bias": float(np.mean(err)),
        "variance": float(np.var(y_pred[ok])),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "n": int(ok.sum()),
    }
