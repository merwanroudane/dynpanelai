"""Instrument-matrix construction for difference and system GMM.

Separated from the estimator so the instrument logic can be tested directly --
the absence of such tests is what allowed ``collapse`` and ``level`` to be
silently inert in 0.1.0.

Layout
------
Everything is built in *long* form: rows are ``(unit, period)`` observations,
columns are instruments.  In that layout the two options have clean meanings.

**Uncollapsed** (``collapse=False``).  One instrument column per
``(period, lag)`` pair, non-zero only on rows belonging to that period.  The
count therefore grows like :math:`T^2`.

**Collapsed** (``collapse=True``).  One instrument column per *lag depth*,
non-zero on every row for which that lag exists.  The count grows like
:math:`T`.  This is the Roodman (2009) / ``xtabond2`` convention.

**System GMM** (``level=True``).  The transformed equation is stacked on top of
the untransformed (levels) equation, and the instrument matrix is block
diagonal:

.. math::
    Z = \\begin{pmatrix} Z_{\\Delta} & 0 \\\\ 0 & Z_{L}\\end{pmatrix},

with :math:`Z_{\\Delta}` the lagged *levels* instrumenting the differenced
equation and :math:`Z_{L}` the lagged *differences* instrumenting the levels
equation, plus a constant.  Blundell and Bond (1998).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GMMDesign", "build_design"]


@dataclass
class GMMDesign:
    """Assembled GMM design.

    Attributes
    ----------
    y : ndarray of shape (n,)
    X : ndarray of shape (n, k)
    Z : ndarray of shape (n, m)
    units : ndarray of shape (n,)
    periods : ndarray of shape (n,)
    is_level : ndarray of bool, shape (n,)
        ``True`` for rows belonging to the levels equation.
    names : list of str
    n_diff_instruments, n_level_instruments : int
    """

    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    units: np.ndarray
    periods: np.ndarray
    is_level: np.ndarray
    names: list[str]
    n_diff_instruments: int
    n_level_instruments: int

    @property
    def n_instruments(self) -> int:
        return self.Z.shape[1]


def _transform(M: np.ndarray, how: str) -> np.ndarray:
    """Apply FD or FOD to a ``(T, N)`` matrix, returning ``(T-1, N)``."""
    if how == "fd":
        return np.diff(M, axis=0)
    if how == "fod":
        from ..core.transforms import fod_matrix

        return fod_matrix(M.shape[0]) @ M
    raise ValueError("transformation must be 'fd' or 'fod'")


def build_design(
    panel,
    *,
    y: str,
    lags: int = 1,
    predetermined: list[str] | None = None,
    exogenous: list[str] | None = None,
    transformation: str = "fd",
    level: bool = False,
    gmm_lags: tuple[int, int | None] = (2, None),
    collapse: bool = False,
    time_dummies: bool = False,
) -> GMMDesign:
    """Build the stacked outcome, regressors and instruments.

    Parameters
    ----------
    panel : PanelData
    y : str
        Outcome column.
    lags : int, default 1
        Lags of the outcome entering as regressors.
    predetermined, exogenous : list of str, optional
    transformation : {'fd', 'fod'}, default 'fd'
    level : bool, default False
        Add the levels equation, giving system GMM.
    gmm_lags : (int, int or None), default (2, None)
        Lag range used to build GMM-style instruments for the transformed
        equation.  ``None`` means "all available".
    collapse : bool, default False
        Collapse instruments to one column per lag depth.
    time_dummies : bool, default False
        Add period dummies as regressors (and as their own instruments).

    Returns
    -------
    GMMDesign

    Raises
    ------
    ValueError
        If the panel is too short for the requested lag structure.

    Examples
    --------
    >>> d = build_design(panel, y="n", lags=2, collapse=True)   # doctest: +SKIP
    >>> d.n_instruments < d.Z.shape[0]                          # doctest: +SKIP
    True
    """
    predetermined = list(predetermined or [])
    exogenous = list(exogenous or [])
    T, N = panel.T, panel.N

    if T < lags + 2:
        raise ValueError(
            f"T={T} is too short for lags={lags}: need at least {lags + 2} periods"
        )

    Y = panel.wide(y)
    P = {v: panel.wide(v) for v in predetermined}
    E = {v: panel.wide(v) for v in exogenous}

    Yt = _transform(Y, transformation)
    Pt = {v: _transform(M, transformation) for v, M in P.items()}
    Et = {v: _transform(M, transformation) for v, M in E.items()}

    lo, hi = gmm_lags
    start = lags + 1
    diff_periods = list(range(start, T))
    if not diff_periods:
        raise ValueError("no usable periods after lag construction")

    names = [f"L{j}.{y}" for j in range(1, lags + 1)] + predetermined + exogenous

    # ---------------- regressors and outcome -----------------------------
    def _rows(t: int, use_level: bool):
        if use_level:
            dep = Y[t]
            regs = [Y[t - j] for j in range(1, lags + 1)]
            regs += [P[v][t] for v in predetermined]
            regs += [E[v][t] for v in exogenous]
        else:
            dep = Yt[t - 1]
            regs = [Yt[t - 1 - j] for j in range(1, lags + 1)]
            regs += [Pt[v][t - 1] for v in predetermined]
            regs += [Et[v][t - 1] for v in exogenous]
        return dep, np.column_stack(regs)

    blocks = [(t, False) for t in diff_periods]
    if level:
        blocks += [(t, True) for t in diff_periods]

    y_parts, X_parts, unit_parts, per_parts, lev_parts = [], [], [], [], []
    for t, is_lev in blocks:
        dep, Xt = _rows(t, is_lev)
        y_parts.append(dep)
        X_parts.append(Xt)
        unit_parts.append(np.arange(N))
        per_parts.append(np.full(N, t))
        lev_parts.append(np.full(N, is_lev))

    y_vec = np.concatenate(y_parts)
    X_mat = np.vstack(X_parts)
    units = np.concatenate(unit_parts)
    periods = np.concatenate(per_parts)
    is_level = np.concatenate(lev_parts).astype(bool)
    n = len(y_vec)

    # ---------------- time dummies ---------------------------------------
    if time_dummies:
        uniq_t = sorted(set(periods.tolist()))[1:]  # drop one for collinearity
        D = np.zeros((n, len(uniq_t)))
        for j, tt in enumerate(uniq_t):
            D[periods == tt, j] = 1.0
        if not level:
            # differenced equation: the dummies must be differenced too, which
            # for period indicators is exactly the indicator on the diff rows
            pass
        X_mat = np.hstack([X_mat, D])
        names = names + [f"T{int(tt)}" for tt in uniq_t]
        time_dummy_block = D
    else:
        time_dummy_block = None

    # ---------------- GMM instruments for the transformed equation -------
    gmm_sources: list[tuple[str, np.ndarray]] = [(y, Y)]
    gmm_sources += [(v, P[v]) for v in predetermined]

    diff_cols: list[np.ndarray] = []
    for _, source in gmm_sources:
        max_lag = T if hi is None else hi
        if collapse:
            # one column per lag depth, active on every diff row
            for lag in range(lo, max_lag + 1):
                col = np.zeros(n)
                any_used = False
                for t in diff_periods:
                    idx = t - lag
                    if idx < 0:
                        continue
                    sel = (periods == t) & (~is_level)
                    col[sel] = np.nan_to_num(source[idx])
                    any_used = True
                if any_used:
                    diff_cols.append(col)
        else:
            # one column per (period, lag), active only on that period's rows
            for t in diff_periods:
                hi_t = min(max_lag, t)
                for lag in range(lo, hi_t + 1):
                    idx = t - lag
                    if idx < 0:
                        continue
                    col = np.zeros(n)
                    sel = (periods == t) & (~is_level)
                    col[sel] = np.nan_to_num(source[idx])
                    diff_cols.append(col)

    # strictly exogenous variables instrument themselves (transformed)
    for v in exogenous:
        col = np.zeros(n)
        for t in diff_periods:
            sel = (periods == t) & (~is_level)
            col[sel] = np.nan_to_num(Et[v][t - 1])
        diff_cols.append(col)

    if time_dummies and time_dummy_block is not None:
        for j in range(time_dummy_block.shape[1]):
            col = time_dummy_block[:, j].copy()
            col[is_level] = 0.0
            diff_cols.append(col)

    n_diff = len(diff_cols)

    # ---------------- instruments for the levels equation ----------------
    level_cols: list[np.ndarray] = []
    if level:
        for _, source in ([(y, Y)] + [(v, P[v]) for v in predetermined]):
            dsrc = np.diff(source, axis=0)
            if collapse:
                col = np.zeros(n)
                for t in diff_periods:
                    if t - 1 < 0 or t - 1 >= dsrc.shape[0]:
                        continue
                    sel = (periods == t) & is_level
                    col[sel] = np.nan_to_num(dsrc[t - 1])
                level_cols.append(col)
            else:
                for t in diff_periods:
                    if t - 1 < 0 or t - 1 >= dsrc.shape[0]:
                        continue
                    col = np.zeros(n)
                    sel = (periods == t) & is_level
                    col[sel] = np.nan_to_num(dsrc[t - 1])
                    level_cols.append(col)

        for v in exogenous:
            col = np.zeros(n)
            for t in diff_periods:
                sel = (periods == t) & is_level
                col[sel] = np.nan_to_num(E[v][t])
            level_cols.append(col)

        # constant, active on the levels block only
        const = np.zeros(n)
        const[is_level] = 1.0
        level_cols.append(const)

        if time_dummies and time_dummy_block is not None:
            for j in range(time_dummy_block.shape[1]):
                col = time_dummy_block[:, j].copy()
                col[~is_level] = 0.0
                level_cols.append(col)

    n_level = len(level_cols)
    Z_mat = np.column_stack(diff_cols + level_cols) if (diff_cols or level_cols) else np.zeros((n, 0))

    # drop instrument columns that carry no information
    keep = Z_mat.std(axis=0) > 0
    if keep.sum() < Z_mat.shape[1]:
        dropped_diff = int((~keep[:n_diff]).sum())
        n_diff -= dropped_diff
        n_level -= int((~keep[n_diff + dropped_diff :]).sum()) if n_level else 0
        Z_mat = Z_mat[:, keep]

    # drop rows with missing outcome or regressors
    ok = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=1)
    return GMMDesign(
        y=y_vec[ok],
        X=X_mat[ok],
        Z=np.nan_to_num(Z_mat[ok]),
        units=units[ok],
        periods=periods[ok],
        is_level=is_level[ok],
        names=names,
        n_diff_instruments=max(n_diff, 0),
        n_level_instruments=max(n_level, 0),
    )
