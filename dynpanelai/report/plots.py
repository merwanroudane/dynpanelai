"""Publication-quality figures.

A small, deliberately restrained set of plots: coefficient (forest) plots,
Monte Carlo bias-coverage diagnostics, lag-weight heatmaps, and forecast-error
comparisons.  The house style is greyscale-safe, with no chartjunk, sized for
a single journal column by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.results import PanelResults

__all__ = [
    "set_style",
    "coefficient_plot",
    "comparison_plot",
    "monte_carlo_plot",
    "lag_weight_plot",
    "forecast_error_plot",
]

# Colour-blind-safe, prints legibly in greyscale.
PALETTE = ["#1b1b1b", "#4477AA", "#CC6677", "#117733", "#DDCC77", "#882255", "#88CCEE"]


def set_style(context: str = "paper") -> None:
    """Apply the package's matplotlib style.

    Parameters
    ----------
    context : {'paper', 'talk'}, default 'paper'
        ``'talk'`` scales fonts and line widths up for slides.

    Examples
    --------
    >>> set_style("paper")
    """
    import matplotlib as mpl

    scale = 1.0 if context == "paper" else 1.35
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9 * scale,
            "axes.titlesize": 10 * scale,
            "axes.labelsize": 9 * scale,
            "legend.fontsize": 8 * scale,
            "xtick.labelsize": 8 * scale,
            "ytick.labelsize": 8 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.3 * scale,
            "legend.frameon": False,
        }
    )


def coefficient_plot(
    res: PanelResults,
    *,
    params: Sequence[str] | None = None,
    alpha: float = 0.05,
    ax=None,
    title: str | None = None,
):
    """Forest plot of coefficients with confidence intervals.

    Parameters
    ----------
    res : PanelResults
    params : sequence of str, optional
        Subset of coefficients to display.
    alpha : float, default 0.05
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = coefficient_plot(res)                  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    set_style()
    tab = res.table(alpha)
    if params is not None:
        tab = tab.loc[[p for p in params if p in tab.index]]

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 0.42 * len(tab) + 1.1))

    ypos = np.arange(len(tab))[::-1]
    ax.errorbar(
        tab["coef"], ypos,
        xerr=[tab["coef"] - tab["ci_lower"], tab["ci_upper"] - tab["coef"]],
        fmt="o", color=PALETTE[0], ecolor=PALETTE[1],
        capsize=2.5, markersize=4, elinewidth=1.2,
    )
    ax.axvline(0.0, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(tab.index)
    ax.set_xlabel("Coefficient")
    ax.set_title(title or res.method, loc="left")
    return ax


def comparison_plot(
    results: Mapping[str, PanelResults],
    param: str,
    *,
    truth: float | None = None,
    alpha: float = 0.05,
    ax=None,
):
    """Compare one coefficient across estimators.

    Parameters
    ----------
    results : mapping of str to PanelResults
    param : str
        The coefficient to compare.
    truth : float, optional
        Draws a reference line -- useful in simulations.
    alpha : float, default 0.05
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = comparison_plot({"FE": r1, "GMM": r2}, "L1.y", truth=0.75)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    set_style()
    labels, coefs, los, his = [], [], [], []
    for label, r in results.items():
        idx = {str(i): i for i in r.params.index}
        if param not in idx:
            continue
        tab = r.table(alpha)
        row = tab.loc[idx[param]]
        labels.append(label)
        coefs.append(row["coef"])
        los.append(row["ci_lower"])
        his.append(row["ci_upper"])

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 0.42 * len(labels) + 1.1))

    ypos = np.arange(len(labels))[::-1]
    coefs = np.asarray(coefs)
    ax.errorbar(
        coefs, ypos,
        xerr=[coefs - np.asarray(los), np.asarray(his) - coefs],
        fmt="o", color=PALETTE[0], ecolor=PALETTE[1],
        capsize=2.5, markersize=4.5, elinewidth=1.2,
    )
    if truth is not None:
        ax.axvline(truth, color=PALETTE[2], linestyle="-", linewidth=1.1,
                   label="true value", zorder=0)
        ax.legend(loc="lower right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(param)
    ax.set_title(f"Estimates of {param} across methods", loc="left")
    return ax


def monte_carlo_plot(
    estimates: Mapping[str, np.ndarray],
    truth: float,
    *,
    ses: Mapping[str, np.ndarray] | None = None,
    axes=None,
):
    """Sampling distributions and, if available, coverage.

    Parameters
    ----------
    estimates : mapping of str to ndarray
    truth : float
    ses : mapping of str to ndarray, optional
        Adds a coverage panel; nominal 95% is drawn as a reference line.
    axes : sequence of matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Coverage is the diagnostic that matters.  An estimator can be nearly
    unbiased and still have 60% coverage if its standard errors are wrong,
    which is exactly the failure mode that orthogonalisation and cross-fitting
    are designed to prevent.
    """
    import matplotlib.pyplot as plt

    set_style()
    n_panels = 2 if ses else 1
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 3.2))
        axes = np.atleast_1d(axes)
    else:
        fig = axes[0].figure

    ax = axes[0]
    names = list(estimates)
    data = [np.asarray(estimates[k], float)[np.isfinite(estimates[k])] for k in names]
    bp = ax.boxplot(data, vert=True, labels=names, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#E8EDF3")
        patch.set_edgecolor(PALETTE[0])
    for med in bp["medians"]:
        med.set_color(PALETTE[2])
    ax.axhline(truth, color=PALETTE[2], linestyle="--", linewidth=1.1, label="truth")
    ax.set_ylabel("Estimate")
    ax.set_title("Sampling distribution", loc="left")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=30)

    if ses:
        ax2 = axes[1]
        cov = []
        for k in names:
            v = np.asarray(estimates[k], float)
            s = np.asarray(ses[k], float)[: len(v)]
            ok = np.isfinite(v) & np.isfinite(s)
            cov.append(
                float(np.mean((v[ok] - 1.96 * s[ok] <= truth)
                              & (truth <= v[ok] + 1.96 * s[ok])))
                if ok.any() else np.nan
            )
        ax2.bar(names, cov, color=PALETTE[1], edgecolor=PALETTE[0], linewidth=0.7)
        ax2.axhline(0.95, color=PALETTE[2], linestyle="--", linewidth=1.1,
                    label="nominal 95%")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Coverage")
        ax2.set_title("95% CI coverage", loc="left")
        ax2.legend(loc="lower right")
        ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig


def lag_weight_plot(
    lag_weights: pd.DataFrame,
    *,
    effective_lag: pd.Series | None = None,
    max_entities: int = 40,
    ax=None,
):
    """Heatmap of entity-specific lag-weight distributions.

    Parameters
    ----------
    lag_weights : pandas.DataFrame
        Entities by lags, rows summing to one.
    effective_lag : pandas.Series, optional
        If given, entities are sorted by it and it is overlaid as a line.
    max_entities : int, default 40
        Subsample for legibility when there are many entities.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = lag_weight_plot(res.lag_weights,
    ...                      effective_lag=res.effective_lag)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    set_style()
    W = lag_weights.copy()
    if effective_lag is not None:
        W = W.loc[effective_lag.sort_values().index]
        k = effective_lag.sort_values()
    else:
        k = None
    if len(W) > max_entities:
        step = max(1, len(W) // max_entities)
        W = W.iloc[::step]
        if k is not None:
            k = k.iloc[::step]

    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 3.4))
    im = ax.imshow(W.to_numpy(), aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([c.replace("lag", "") for c in W.columns])
    ax.set_xlabel("Lag $k$")
    ax.set_ylabel("Entity (sorted by $k^\\star$)")
    ax.set_title("Entity-conditioned lag weights $\\omega_{i,k}$", loc="left")
    ax.grid(False)
    if k is not None:
        ax.plot(k.to_numpy() - 1, np.arange(len(k)), color="white",
                linewidth=1.4, label="$k^\\star_i$")
        ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="weight", fraction=0.046, pad=0.04)
    return ax


def forecast_error_plot(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "rmse",
    baseline: str | None = None,
    ax=None,
):
    """Bar chart of forecast accuracy across estimators.

    Parameters
    ----------
    metrics : mapping of str to mapping
        Estimator name to the dict returned by
        :func:`~dynpanelai.shrink.forecast.forecast_metrics`.
    metric : str, default 'rmse'
    baseline : str, optional
        If given, plot percentage differences relative to this estimator --
        the presentation used in the Cornejo and Sosa-Escudero figures.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    set_style()
    names = list(metrics)
    vals = np.array([metrics[n][metric] for n in names], dtype=float)

    if baseline is not None:
        if baseline not in metrics:
            raise KeyError(f"baseline {baseline!r} is not among the estimators")
        base = float(metrics[baseline][metric])
        vals = 100.0 * (vals - base) / base
        ylabel = f"% difference in {metric.upper()} vs {baseline}"
    else:
        ylabel = metric.upper()

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.2))
    colors = [PALETTE[2] if v > 0 else PALETTE[3] for v in vals] if baseline else PALETTE[1]
    ax.bar(names, vals, color=colors, edgecolor=PALETTE[0], linewidth=0.7)
    if baseline is not None:
        ax.axhline(0.0, color=PALETTE[0], linewidth=0.9)
    ax.set_ylabel(ylabel)
    ax.set_title("Out-of-sample forecast accuracy", loc="left")
    ax.tick_params(axis="x", rotation=30)
    return ax
