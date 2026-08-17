"""Fixed-effect removal transforms.

Four transforms cover every estimator in this package:

``within``
    Subtract the unit mean.  Removes :math:`\\alpha_i` but induces the Nickell
    (1981) correlation between the transformed lag and the transformed error.

``fd`` (first differences)
    :math:`\\Delta Z_{it} = Z_{it} - Z_{i,t-1}`.  Removes :math:`\\alpha_i`;
    the transformed error is MA(1), so lags 2 and deeper remain valid
    instruments (Arellano and Bond, 1991).

``fod`` (forward orthogonal deviations)
    .. math::

        \\Delta Z_{it} = c_t\\Bigl(Z_{it}
            - \\frac{1}{T-t}\\sum_{s=t+1}^{T} Z_{is}\\Bigr),
        \\qquad c_t = \\sqrt{\\frac{T-t}{T-t+1}}.

    Arellano and Bover (1995).  Removes :math:`\\alpha_i` while leaving the
    transformed error serially *uncorrelated*, which is why Chernozhukov,
    Fernandez-Val, Huang and Wang (2024) make it the default for AB-LASSO: it
    both improves efficiency relative to first differences and means
    cross-fitting does not alter the large-sample properties.

``mundlak``
    Append unit means of the time-varying covariates (Mundlak, 1978;
    Chamberlain, 1982) instead of differencing them away.  Used by Semenova,
    Goldman, Chernozhukov and Taddy for the correlated-random-effects
    representation of unit heterogeneity.

References
----------
Arellano, M. and Bond, S. (1991). *Review of Economic Studies* 58(2), 277-297.
Arellano, M. and Bover, O. (1995). *Journal of Econometrics* 68(1), 29-51.
Mundlak, Y. (1978). *Econometrica* 46(1), 69-85.
Nickell, S. (1981). *Econometrica* 49(6), 1417-1426.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .panel import PanelData

__all__ = [
    "within_transform",
    "first_difference",
    "forward_orthogonal_deviation",
    "fod_matrix",
    "mundlak_means",
    "apply_transform",
]


def within_transform(
    panel: PanelData,
    cols: Sequence[str],
    *,
    time_demean: bool = False,
) -> pd.DataFrame:
    """Subtract unit means (and optionally period means).

    Parameters
    ----------
    panel : PanelData
    cols : sequence of str
        Columns to transform.
    time_demean : bool, default False
        Also subtract the period mean, giving the two-way within transform.
        Note this is the "one-way then one-way" operator, which coincides with
        the exact two-way projection only for balanced panels; the grand mean
        is added back so that the result is properly centred.

    Returns
    -------
    pandas.DataFrame
        Transformed columns, suffixed ``"_w"``, aligned to ``panel.df``.

    Examples
    --------
    >>> within_transform(panel, ["y", "d"]).columns.tolist()
    ['y_w', 'd_w']
    """
    df = panel.df
    out = {}
    for col in cols:
        v = df[col]
        centred = v - v.groupby(df["_i"]).transform("mean")
        if time_demean:
            centred = centred - centred.groupby(df["_t"]).transform("mean")
        out[f"{col}_w"] = centred.to_numpy()
    return pd.DataFrame(out, index=df.index)


def first_difference(
    panel: PanelData,
    cols: Sequence[str],
    *,
    time_demean: bool = False,
) -> pd.DataFrame:
    """First-difference within units, respecting gaps in time.

    Parameters
    ----------
    panel : PanelData
    cols : sequence of str
    time_demean : bool, default False
        Subtract the cross-sectional mean of each differenced series, which
        removes an additive time effect :math:`\\gamma_t`.  This is what the
        CRAN ``ablasso`` package does via ``rowMeans``.

    Returns
    -------
    pandas.DataFrame
        Differenced columns suffixed ``"_fd"``.  The first observed period of
        each unit is ``NaN`` by construction.
    """
    lagged = panel.lag(list(cols), 1)
    df = panel.df
    out = {}
    for col in cols:
        d = df[col].to_numpy() - lagged[f"{col}_lag1"].to_numpy()
        s = pd.Series(d, index=df.index)
        if time_demean:
            s = s - s.groupby(df["_t"]).transform("mean")
        out[f"{col}_fd"] = s.to_numpy()
    return pd.DataFrame(out, index=df.index)


def fod_matrix(T: int) -> np.ndarray:
    """Return the ``(T-1, T)`` forward orthogonal deviations operator.

    Row ``t`` (0-indexed) implements

    .. math::
        c_t\\Bigl(Z_t - \\frac{1}{T-t-1}\\sum_{s>t} Z_s\\Bigr),
        \\qquad c_t = \\sqrt{\\frac{T-t-1}{T-t}}.

    Parameters
    ----------
    T : int
        Number of time periods.  Must be at least 2.

    Returns
    -------
    ndarray of shape (T - 1, T)

    Notes
    -----
    The operator ``A`` satisfies ``A @ ones(T) == 0`` (so unit effects are
    annihilated) and ``A @ A.T == I`` (so a homoskedastic error stays
    homoskedastic and serially uncorrelated).  Both properties are checked in
    the test suite.

    Examples
    --------
    >>> A = fod_matrix(3)
    >>> A.shape
    (2, 3)
    >>> bool(np.allclose(A @ np.ones(3), 0))
    True
    """
    if T < 2:
        raise ValueError("forward orthogonal deviations require T >= 2")
    A = np.zeros((T - 1, T))
    for t in range(T - 1):
        n_future = T - t - 1
        c_t = np.sqrt(n_future / (n_future + 1.0))
        A[t, t] = c_t
        A[t, t + 1 :] = -c_t / n_future
    return A


def forward_orthogonal_deviation(
    panel: PanelData,
    cols: Sequence[str],
    *,
    time_demean: bool = True,
) -> pd.DataFrame:
    """Apply forward orthogonal deviations (Arellano and Bover, 1995).

    Parameters
    ----------
    panel : PanelData
        Should be balanced; unbalanced panels are handled unit by unit using
        each unit's own observed periods, which is the standard convention but
        makes the resulting :math:`c_t` unit-specific.
    cols : sequence of str
    time_demean : bool, default True
        Subtract the cross-sectional mean of each transformed series, removing
        an additive time effect.  This matches the AB-LASSO replication code,
        where :math:`\\tilde\\Delta Z_{it} = \\Delta Z_{it} - N^{-1}\\sum_j
        \\Delta Z_{jt}`.

    Returns
    -------
    pandas.DataFrame
        Transformed columns suffixed ``"_fod"``.  The final observed period of
        each unit is ``NaN``, since the forward mean is undefined there.

    Examples
    --------
    >>> fod = forward_orthogonal_deviation(panel, ["y", "d"])
    >>> fod.columns.tolist()
    ['y_fod', 'd_fod']
    """
    df = panel.df
    out = {col: np.full(len(df), np.nan) for col in cols}

    for _, idx in df.groupby("_i").groups.items():
        rows = df.loc[idx]
        order = np.argsort(rows["_t"].to_numpy())
        pos = np.asarray(idx)[order]
        T_i = len(pos)
        if T_i < 2:
            continue
        A = fod_matrix(T_i)
        for col in cols:
            z = df.loc[pos, col].to_numpy(dtype=float)
            if np.isnan(z).any():
                continue
            out[col][df.index.get_indexer(pos[:-1])] = A @ z

    frame = pd.DataFrame(
        {f"{c}_fod": out[c] for c in cols}, index=df.index
    )
    if time_demean:
        for c in cols:
            s = frame[f"{c}_fod"]
            frame[f"{c}_fod"] = (s - s.groupby(df["_t"]).transform("mean")).to_numpy()
    return frame


def mundlak_means(
    panel: PanelData,
    cols: Sequence[str],
    *,
    prefix: str = "mean_",
) -> pd.DataFrame:
    """Append within-unit means of ``cols`` (correlated random effects).

    Rather than differencing the unit effect away, Mundlak (1978) models it as
    a linear function of the unit means of the time-varying covariates.
    Semenova et al. extend this to high dimensions by treating the residual
    unit effect as a *weakly sparse* deviation from the Mundlak projection.

    Parameters
    ----------
    panel : PanelData
    cols : sequence of str
    prefix : str, default ``"mean_"``

    Returns
    -------
    pandas.DataFrame
        One column per input, named ``f"{prefix}{col}"``, constant within unit.
    """
    df = panel.df
    out = {
        f"{prefix}{col}": df.groupby("_i")[col].transform("mean").to_numpy()
        for col in cols
    }
    return pd.DataFrame(out, index=df.index)


def apply_transform(
    panel: PanelData,
    cols: Sequence[str],
    method: str = "within",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to one of the fixed-effect transforms by name.

    Parameters
    ----------
    panel : PanelData
    cols : sequence of str
    method : {'within', 'fd', 'fod', 'mundlak', 'none'}, default 'within'
    **kwargs
        Forwarded to the underlying transform.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    method = method.lower()
    if method == "within":
        return within_transform(panel, cols, **kwargs)
    if method in {"fd", "diff", "first_difference"}:
        return first_difference(panel, cols, **kwargs)
    if method in {"fod", "forward_orthogonal"}:
        return forward_orthogonal_deviation(panel, cols, **kwargs)
    if method == "mundlak":
        return mundlak_means(panel, cols, **kwargs)
    if method == "none":
        return panel.df[list(cols)].copy()
    raise ValueError(
        f"unknown transform {method!r}; expected one of "
        "'within', 'fd', 'fod', 'mundlak', 'none'"
    )
