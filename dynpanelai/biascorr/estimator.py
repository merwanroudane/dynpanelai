"""Bias corrections for the fixed-effects estimator in dynamic panels.

The within estimator of a dynamic panel is inconsistent for fixed ``T``: the
demeaned lag is mechanically correlated with the demeaned error, giving the
Nickell (1981) bias of order :math:`1/T`.  Three families of correction are
implemented, all of which remove the leading :math:`O(1/T)` term.

Analytical (:func:`debiased_fe`)
    Estimate the score's bias directly and subtract it.  This is the ``DFE-A``
    estimator in the Arellano-Bond LASSO replication, following Hahn and
    Kuersteiner (2002) and Chen, Chernozhukov and Fernandez-Val (2019).

Split-panel jackknife (:func:`split_panel_jackknife`)
    Dhaene and Jochmans (2015).  If the bias is :math:`B/T`, then estimating on
    half-panels doubles it, so :math:`2\\widehat\\theta_{full} -
    \\overline{\\widehat\\theta}_{half}` cancels the leading term.  Splitting on
    *time* is the original; splitting on the *cross-section* is the variant
    Chen et al. apply to Arellano-Bond, and is what ``DAB`` does.

Bias-corrected LSDV (:func:`bias_corrected_lsdv`)
    Kiviet (1995), extended to unbalanced panels by Bruno (2005).  Uses a
    consistent initial estimator to evaluate a closed-form bias approximation.

References
----------
Chen, S., Chernozhukov, V. and Fernandez-Val, I. (2019). Mastering panel
metrics. *AEA Papers and Proceedings* 109, 77-82.

Dhaene, G. and Jochmans, K. (2015). Split-panel jackknife estimation of
fixed-effect models. *Review of Economic Studies* 82(3), 991-1030.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..core.panel import PanelData
from ..core.results import PanelResults
from ..core.variance import cluster_variance, sandwich

__all__ = [
    "fixed_effects",
    "debiased_fe",
    "split_panel_jackknife",
    "bias_corrected_lsdv",
    "half_panel_jackknife",
]


def _design(panel: PanelData, y: str, lags: int, x: Sequence[str], twoway: bool):
    """Within-transformed design for a dynamic FE regression."""
    lagged = panel.lag(y, lags)
    frame = panel.df.copy()
    for c in lagged.columns:
        frame[c] = lagged[c]
    cols = [f"{y}_lag{j}" for j in range(1, lags + 1)] + list(x)
    frame = frame.dropna(subset=[y] + cols)

    g = frame["_i"]
    yv = frame[y] - frame[y].groupby(g).transform("mean")
    X = frame[cols].apply(lambda s: s - s.groupby(g).transform("mean"))
    if twoway:
        h = frame["_t"]
        yv = yv - yv.groupby(h).transform("mean")
        X = X.apply(lambda s: s - s.groupby(h).transform("mean"))

    names = [f"L{j}.{y}" for j in range(1, lags + 1)] + list(x)
    return (
        yv.to_numpy(),
        X.to_numpy(),
        frame["_i"].to_numpy(),
        frame["_t"].to_numpy(),
        names,
    )


def fixed_effects(
    panel: PanelData,
    y: str,
    *,
    lags: int = 1,
    x: Sequence[str] | None = None,
    twoway: bool = False,
) -> PanelResults:
    """Within (LSDV) estimator with unit-clustered standard errors.

    Parameters
    ----------
    panel : PanelData
    y : str
    lags : int, default 1
    x : sequence of str, optional
    twoway : bool, default False
        Also remove period means.

    Returns
    -------
    PanelResults

    Warnings
    --------
    Inconsistent for fixed ``T`` when ``lags >= 1``.  Reported here as the
    benchmark that the corrections in this module repair.

    Examples
    --------
    >>> res = fixed_effects(panel, "y", lags=1, x=["d"])   # doctest: +SKIP
    """
    x = list(x) if x else []
    yv, X, units, times, names = _design(panel, y, lags, x, twoway)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta

    n = len(yv)
    J = -(X.T @ X) / n
    omega = cluster_variance(X * resid[:, None], units)
    cov = sandwich(J, omega) / n**2

    return PanelResults(
        params=pd.Series(beta, index=names),
        cov=cov,
        method=f"Fixed effects ({'two-way' if twoway else 'one-way'})",
        n_obs=n,
        n_units=len(np.unique(units)),
        n_periods=panel.T,
        dependent=y,
        diagnostics={"note": "Nickell-biased for small T when lags >= 1"},
        extra={"resid": resid, "X": X, "units": units, "times": times},
    )


def debiased_fe(
    panel: PanelData,
    y: str,
    *,
    lags: int = 1,
    x: Sequence[str] | None = None,
    twoway: bool = False,
) -> PanelResults:
    """Analytically debiased fixed-effects estimator (``DFE-A``).

    The within score has a non-zero mean of order :math:`1/T` because the
    demeaned regressor is correlated with the demeaned error at adjacent
    periods.  Estimating that correlation with the lagged residual and
    subtracting gives

    .. math::
        \\widehat\\theta_{DFE} = \\widehat\\theta_{FE}
            + \\Bigl(\\tfrac1n X'X\\Bigr)^{-1}
              \\Bigl(\\tfrac1n\\sum_{i,t} X_{it}\\widehat u_{i,t-1}\\Bigr).

    Parameters
    ----------
    panel : PanelData
    y : str
    lags : int, default 1
    x : sequence of str, optional
    twoway : bool, default False

    Returns
    -------
    PanelResults
        Standard errors are those of the uncorrected within estimator; the
        correction is :math:`o_p(1)` and does not change the asymptotic
        variance.

    Examples
    --------
    >>> res = debiased_fe(panel, "y", lags=1, x=["d"])     # doctest: +SKIP
    """
    base = fixed_effects(panel, y, lags=lags, x=x, twoway=twoway)
    X = base.extra["X"]
    resid = base.extra["resid"]
    units = base.extra["units"]
    times = base.extra["times"]
    n = len(resid)

    key = pd.MultiIndex.from_arrays([units, times])
    s = pd.Series(resid, index=key)
    lagged = s.reindex(pd.MultiIndex.from_arrays([units, times - 1])).to_numpy()
    ok = np.isfinite(lagged)

    # Mean score evaluated at the once-lagged residual.  The N/n = 1/T factor
    # is what makes the correction O(1/T), matching the order of the Nickell
    # bias itself; without it the estimator is over-corrected by a factor T.
    n_units = len(np.unique(units))
    score = (X[ok] * lagged[ok, None]).sum(axis=0) / ok.sum()
    jac = np.linalg.pinv(X.T @ X / n)
    bias = jac @ score * (n_units / n)

    res = PanelResults(
        params=base.params + bias,
        cov=base.cov,
        method="Debiased fixed effects (analytical, DFE-A)",
        n_obs=base.n_obs,
        n_units=base.n_units,
        n_periods=base.n_periods,
        dependent=y,
        diagnostics={"bias correction": dict(zip(base.params.index, np.round(bias, 5)))},
    )
    return res


def split_panel_jackknife(
    panel: PanelData,
    y: str,
    estimator: Callable[[PanelData], PanelResults] | None = None,
    *,
    dimension: str = "time",
    lags: int = 1,
    x: Sequence[str] | None = None,
    **kwargs,
) -> PanelResults:
    """Split-panel jackknife bias correction.

    Computes :math:`2\\widehat\\theta_{full} -
    \\tfrac12(\\widehat\\theta_A + \\widehat\\theta_B)`, which cancels a bias of
    order :math:`1/T` (time split) or :math:`m/N` (cross-section split).

    Parameters
    ----------
    panel : PanelData
    y : str
    estimator : callable, optional
        Any function mapping a :class:`PanelData` to a :class:`PanelResults`.
        Defaults to :func:`fixed_effects`.  Pass a GMM estimator to reproduce
        the ``DAB`` (debiased Arellano-Bond) estimator.
    dimension : {'time', 'unit'}, default 'time'
        Split the panel over periods (Dhaene-Jochmans) or over units
        (Chen, Chernozhukov and Fernandez-Val).
    lags, x
        Forwarded to the default estimator.
    **kwargs
        Forwarded to the estimator.

    Returns
    -------
    PanelResults

    Examples
    --------
    Debiased Arellano-Bond, splitting on the cross-section:

    >>> from dynpanelai.gmm import diff_gmm
    >>> res = split_panel_jackknife(
    ...     panel, "y", estimator=lambda p: diff_gmm(p, "y", lags=1),
    ...     dimension="unit")                                # doctest: +SKIP
    """
    if estimator is None:
        def estimator(p: PanelData) -> PanelResults:  # noqa: D401
            return fixed_effects(p, y, lags=lags, x=x, **kwargs)

    full = estimator(panel)

    if dimension == "time":
        cut = panel.times[len(panel.times) // 2]
        halves = [
            panel.df[panel.df[panel.time] <= cut],
            panel.df[panel.df[panel.time] > cut],
        ]
    elif dimension == "unit":
        rng = np.random.default_rng(kwargs.pop("seed", 0))
        order = rng.permutation(panel.units)
        first = set(order[: len(order) // 2])
        halves = [
            panel.df[panel.df[panel.unit].isin(first)],
            panel.df[~panel.df[panel.unit].isin(first)],
        ]
    else:
        raise ValueError("dimension must be 'time' or 'unit'")

    sub_params = []
    for h in halves:
        sub = PanelData(h.drop(columns=["_i", "_t", "_tkey"]), panel.unit, panel.time)
        sub_params.append(estimator(sub).params)

    avg = sum(sub_params) / len(sub_params)
    corrected = 2 * full.params - avg.reindex(full.params.index)

    return PanelResults(
        params=corrected,
        cov=full.cov,
        method=f"Split-panel jackknife ({dimension} split) on {full.method}",
        n_obs=full.n_obs,
        n_units=full.n_units,
        n_periods=full.n_periods,
        dependent=y,
        diagnostics={
            "base estimator": full.method,
            "split": dimension,
            "raw estimate": dict(zip(full.params.index, np.round(full.params, 5))),
        },
    )


def half_panel_jackknife(
    panel: PanelData,
    y: str,
    *,
    lags: int = 1,
    x: Sequence[str] | None = None,
) -> PanelResults:
    """Half-panel jackknife of Chudik, Pesaran and Yang (2018).

    A time split into two halves, as in :func:`split_panel_jackknife` with
    ``dimension='time'``, but applied to the within estimator specifically.

    Warnings
    --------
    Requires unconditional stationarity of all variables.  This fails for
    staggered policy indicators, which is why the Arellano-Bond LASSO paper
    declines to use it in the COVID application.  Prefer
    :func:`split_panel_jackknife` or :func:`debiased_fe` when regressors are
    non-stationary.
    """
    return split_panel_jackknife(
        panel, y, dimension="time", lags=lags, x=x
    )


def bias_corrected_lsdv(
    panel: PanelData,
    y: str,
    *,
    lags: int = 1,
    x: Sequence[str] | None = None,
    initial: str = "ah",
    iterations: int = 3,
) -> PanelResults:
    """Kiviet (1995) / Bruno (2005) bias-corrected LSDV.

    Uses a consistent initial estimator to evaluate the leading bias term of
    the within estimator, then subtracts it, optionally iterating.

    Parameters
    ----------
    panel : PanelData
    y : str
    lags : int, default 1
    x : sequence of str, optional
    initial : {'ah', 'fe'}, default 'ah'
        Consistent initialiser: Anderson-Hsiao or (inconsistent) within.
    iterations : int, default 3

    Returns
    -------
    PanelResults

    Notes
    -----
    This implements the :math:`O(1/T)` term of the Kiviet expansion, which is
    what dominates in practice.  Bruno's higher-order terms are not included;
    for very small ``T`` prefer :func:`split_panel_jackknife`.
    """
    x = list(x) if x else []
    base = fixed_effects(panel, y, lags=lags, x=x)

    if initial == "ah":
        from ..gmm.estimator import anderson_hsiao

        try:
            init = anderson_hsiao(panel, y, lags=lags, exogenous=x)
            gamma = float(init.params.iloc[0])
        except Exception:
            gamma = float(base.params.iloc[0])
    else:
        gamma = float(base.params.iloc[0])

    T = panel.T
    beta = base.params.copy()
    for _ in range(iterations):
        g = np.clip(gamma, -0.999, 0.999)
        # leading Nickell term for the AR coefficient
        bias = -(1 + g) / (T - 1) * (
            1 - (1 - g**(T - 1)) / ((T - 1) * (1 - g))
        ) / (1 - (2 * g) / ((T - 1) * (1 - g)) * (1 - (1 - g**(T - 1)) / ((T - 1) * (1 - g))))
        beta.iloc[0] = base.params.iloc[0] - bias
        gamma = float(beta.iloc[0])

    return PanelResults(
        params=beta,
        cov=base.cov,
        method="Bias-corrected LSDV (Kiviet/Bruno)",
        n_obs=base.n_obs,
        n_units=base.n_units,
        n_periods=base.n_periods,
        dependent=y,
        diagnostics={"initialiser": initial, "iterations": iterations},
    )
