"""Dependence-aware cross-fitting folds.

Standard i.i.d. cross-fitting draws folds at random, which leaks information
across time in a panel: an observation held out at period ``t`` is strongly
correlated with its neighbours at ``t-1`` and ``t+1``, which sit in the
training set.  Two fold designs in the literature fix this.

Blocked-time folds with a buffer (Sneller, 2026)
    Hold out contiguous blocks of periods and additionally purge a buffer of
    ``B`` periods on each side.  The *effective* buffer is
    :math:`B_* = B + L_*` where :math:`L_*` is the deepest lag entering the
    feature vector, so no training observation's lag window can overlap the
    held-out block.

Neighbours-left-out (NLO) folds (Semenova, Goldman, Chernozhukov and Taddy)
    Partition the periods into ``K`` adjacent blocks and, when scoring block
    ``k``, fit nuisances on every block *except* ``k`` and its two immediate
    neighbours.  Powered theoretically by Strassen's coupling: under
    beta-mixing the block and its quasi-complement can be replaced by
    independent copies with vanishing error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Fold",
    "blocked_time_folds",
    "nlo_folds",
    "clustered_unit_folds",
    "buffer_rules",
    "suggest_buffer_acf",
]


@dataclass
class Fold:
    """One cross-fitting split, expressed in period positions.

    Attributes
    ----------
    test : ndarray
        Periods scored by this fold.
    train : ndarray
        Periods used to fit the nuisance functions.
    buffer : ndarray
        Periods discarded to separate train from test.
    """

    test: np.ndarray
    train: np.ndarray
    buffer: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Fold(test={len(self.test)} periods, train={len(self.train)}, "
            f"buffer={len(self.buffer)})"
        )


def buffer_rules(T: int) -> dict[str, int]:
    """Theory-aligned default buffer lengths.

    Both rules satisfy :math:`B_T\\to\\infty` and :math:`B_T/T\\to 0`, the
    condition under which train-test dependence vanishes asymptotically.

    Parameters
    ----------
    T : int
        Number of usable periods.

    Returns
    -------
    dict
        ``{'log': ceil(log T), 'sqrt': floor(sqrt T)}``.

    Examples
    --------
    >>> buffer_rules(100)
    {'log': 5, 'sqrt': 10}
    """
    safe = max(int(T), 3)
    return {
        "log": int(max(1, np.ceil(np.log(safe)))),
        "sqrt": int(max(1, np.floor(np.sqrt(safe)))),
    }


def suggest_buffer_acf(
    series: np.ndarray,
    *,
    threshold: float = 0.05,
    max_lag: int | None = None,
) -> int:
    """Data-driven buffer suggestion from the autocorrelation function.

    Returns the smallest ``B`` such that :math:`|\\hat\\rho(\\ell)| <`
    ``threshold`` for every :math:`\\ell \\ge B`.

    Parameters
    ----------
    series : ndarray
        A time series, typically the period-averaged within-demeaned outcome.
    threshold : float, default 0.05
    max_lag : int, optional
        Defaults to ``min(20, len(series) - 1)``.

    Returns
    -------
    int

    Warnings
    --------
    This is a rule of thumb, not part of the asymptotic theory.  Report it as
    a robustness check rather than as the primary specification.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0
    cap = min(int(max_lag) if max_lag is not None else 20, x.size - 1)
    if cap <= 0:
        return 0
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return 0
    acf = np.array([float(x[:-l] @ x[l:]) / denom for l in range(1, cap + 1)])
    for b in range(1, cap + 1):
        if np.all(np.abs(acf[b - 1 :]) < threshold):
            return b
    return cap


def blocked_time_folds(
    periods: Sequence,
    k: int = 4,
    buffer: int = 0,
    *,
    max_lag: int = 0,
) -> list[Fold]:
    """Contiguous time blocks with a separation buffer.

    Parameters
    ----------
    periods : sequence
        Usable periods, in ascending order.  Positions, not labels, are
        returned in the folds.
    k : int, default 4
        Number of folds.
    buffer : int, default 0
        Additional separation ``B``.  Use ``buffer_rules(T)['log']`` for the
        theory-aligned default; ``0`` is a short-panel stress test only.
    max_lag : int, default 0
        Deepest lag :math:`L_*` in the feature vector.  Added to ``buffer`` to
        form the effective buffer :math:`B_* = B + L_*`, guaranteeing no
        training row's lag window reaches into the test block.

    Returns
    -------
    list of Fold

    Raises
    ------
    ValueError
        If ``k`` exceeds the number of periods, or the buffer leaves some fold
        with no training data.

    Examples
    --------
    >>> folds = blocked_time_folds(range(20), k=4, buffer=2, max_lag=1)
    >>> len(folds)
    4
    >>> folds[0]
    Fold(test=5 periods, train=12, buffer=3)
    """
    n = len(list(periods))
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"cannot build {k} folds from {n} periods")
    if buffer < 0 or max_lag < 0:
        raise ValueError("buffer and max_lag must be non-negative")

    b_star = buffer + max_lag
    idx = np.arange(n)
    base, rem = divmod(n, k)
    sizes = [base + 1 if i < rem else base for i in range(k)]

    folds: list[Fold] = []
    start = 0
    for size in sizes:
        stop = start + size
        test = idx[start:stop]
        if b_star > 0:
            lo = max(0, start - b_star)
            hi = min(n, stop + b_star)
            buf = np.concatenate([idx[lo:start], idx[stop:hi]])
        else:
            buf = np.array([], dtype=int)
        train = np.setdiff1d(idx, np.concatenate([test, buf]))
        folds.append(Fold(test=test, train=train, buffer=buf))
        start = stop

    empty = [i for i, f in enumerate(folds) if len(f.train) == 0]
    if empty:
        raise ValueError(
            f"folds {empty} have no training periods: T={n}, k={k}, "
            f"effective buffer={b_star}. Reduce k or the buffer, or use a "
            "longer panel."
        )
    return folds


def nlo_folds(periods: Sequence, k: int = 10) -> list[Fold]:
    """Neighbours-left-out folds (Semenova et al.).

    Block ``k`` is scored using nuisances fit on every block except ``k`` and
    its immediate neighbours ``k-1`` and ``k+1``.

    Parameters
    ----------
    periods : sequence
        Usable periods in ascending order.
    k : int, default 10
        Number of blocks.  The paper recommends ``k >= 10`` so that at least
        ~70% of the data is available for fitting each nuisance.

    Returns
    -------
    list of Fold

    Raises
    ------
    ValueError
        If ``k < 3``, since the construction needs a block plus two neighbours.

    Examples
    --------
    >>> folds = nlo_folds(range(100), k=10)
    >>> len(folds[0].train) < 100
    True
    """
    n = len(list(periods))
    if k < 3:
        raise ValueError("NLO cross-fitting requires k >= 3")
    if n < k:
        raise ValueError(f"cannot build {k} blocks from {n} periods")

    idx = np.arange(n)
    base, rem = divmod(n, k)
    sizes = [base + 1 if i < rem else base for i in range(k)]
    bounds, start = [], 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size

    folds: list[Fold] = []
    for j, (lo, hi) in enumerate(bounds):
        test = idx[lo:hi]
        neigh = [m for m in (j - 1, j + 1) if 0 <= m < k]
        buf = np.concatenate(
            [idx[bounds[m][0] : bounds[m][1]] for m in neigh]
        ) if neigh else np.array([], dtype=int)
        train = np.setdiff1d(idx, np.concatenate([test, buf]))
        folds.append(Fold(test=test, train=train, buffer=buf))
    return folds


def clustered_unit_folds(
    n_units: int, k: int = 5, *, seed: int | None = None
) -> list[Fold]:
    """Hold out sets of *units* rather than periods.

    Conservative choice when cross-sectional dependence (common shocks beyond
    the unit effect) is the binding concern rather than serial dependence.

    Parameters
    ----------
    n_units : int
    k : int, default 5
    seed : int, optional

    Returns
    -------
    list of Fold
        ``test`` and ``train`` hold unit indices, and ``buffer`` is empty.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_units)
    splits = np.array_split(order, k)
    return [
        Fold(
            test=np.sort(s),
            train=np.sort(np.setdiff1d(np.arange(n_units), s)),
            buffer=np.array([], dtype=int),
        )
        for s in splits
    ]
