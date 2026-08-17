"""Panel data container and lag construction.

The :class:`PanelData` object is the single entry point for every estimator in
``dynpanelai``.  It normalises a long-format :class:`pandas.DataFrame` into the
(unit, time) indexed representation used throughout the package, records
whether the panel is balanced, and provides the lag / transform machinery that
the dynamic-panel estimators rely on.

Notes
-----
Every estimator in this package assumes independence across units ``i`` and
allows arbitrary (weak) dependence within a unit over ``t``.  That is the
sampling scheme of Arellano and Bond (1991), Kock and Tang (2019),
Chernozhukov, Fernandez-Val, Huang and Wang (2024) and Sneller (2026) alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["PanelData", "PanelSpec"]


def _calendar_key(values: pd.Series, times: pd.Index) -> np.ndarray:
    """Map time labels to a spacing-aware integer key.

    Integer-valued and datetime time columns are mapped so that consecutive
    calendar periods differ by exactly 1, which makes gaps visible to the lag
    operator.  For labels with no natural spacing (strings, categories) the
    key falls back to position within the observed periods, in which case a
    lag means "previous observed period".

    Parameters
    ----------
    values : pandas.Series
        The time column.
    times : pandas.Index
        Sorted distinct time labels.

    Returns
    -------
    ndarray of int
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        if len(times) > 1:
            deltas = np.diff(times.view("int64"))
            step = int(np.min(deltas[deltas > 0])) if (deltas > 0).any() else 1
        else:
            step = 1
        base = int(times.view("int64")[0])
        return ((values.view("int64") - base) // step).to_numpy()

    if pd.api.types.is_numeric_dtype(values):
        arr = values.to_numpy(dtype=float)
        uniq = np.asarray(times, dtype=float)
        if np.allclose(uniq, np.round(uniq)):
            return (arr - uniq.min()).round().astype(np.int64)
        if len(uniq) > 1:
            deltas = np.diff(np.sort(uniq))
            step = float(np.min(deltas[deltas > 0])) if (deltas > 0).any() else 1.0
            return np.round((arr - uniq.min()) / step).astype(np.int64)
        return np.zeros(len(arr), dtype=np.int64)

    return times.get_indexer(values).astype(np.int64)


@dataclass
class PanelSpec:
    """Column roles within a panel.

    Parameters
    ----------
    unit : str
        Column holding the cross-sectional identifier ``i``.
    time : str
        Column holding the time identifier ``t``.
    y : str
        Outcome variable.
    d : str, optional
        Treatment / policy variable of primary interest.
    x : sequence of str, optional
        Additional covariates (controls).
    """

    unit: str
    time: str
    y: str
    d: str | None = None
    x: list[str] = field(default_factory=list)

    def columns(self) -> list[str]:
        cols = [self.unit, self.time, self.y]
        if self.d is not None:
            cols.append(self.d)
        cols.extend(self.x)
        # preserve order, drop duplicates
        seen: dict[str, None] = {}
        for c in cols:
            seen.setdefault(c, None)
        return list(seen)


class PanelData:
    """A (unit, time) indexed panel.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format data: one row per (unit, time).
    unit : str
        Name of the unit-identifier column.
    time : str
        Name of the time-identifier column.
    copy : bool, default True
        Copy the input frame before mutating.  Set ``False`` only if you are
        certain the caller does not need the original.

    Attributes
    ----------
    df : pandas.DataFrame
        The sorted frame, with the original columns plus internal integer
        codes ``_i`` (unit position) and ``_t`` (time position).
    N : int
        Number of units.
    T : int
        Number of distinct time periods observed anywhere in the panel.
    balanced : bool
        ``True`` when every unit is observed in every period.

    Examples
    --------
    >>> import pandas as pd
    >>> from dynpanelai import PanelData
    >>> df = pd.DataFrame({
    ...     "firm": [1, 1, 1, 2, 2, 2],
    ...     "year": [2000, 2001, 2002, 2000, 2001, 2002],
    ...     "y": [1.0, 1.2, 1.5, 2.0, 2.1, 2.4],
    ... })
    >>> pd_ = PanelData(df, unit="firm", time="year")
    >>> pd_.N, pd_.T, pd_.balanced
    (2, 3, True)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        unit: str,
        time: str,
        *,
        copy: bool = True,
    ) -> None:
        if unit not in df.columns:
            raise KeyError(f"unit column {unit!r} not found in the DataFrame")
        if time not in df.columns:
            raise KeyError(f"time column {time!r} not found in the DataFrame")

        work = df.copy() if copy else df
        work = work.sort_values([unit, time]).reset_index(drop=True)

        if work.duplicated([unit, time]).any():
            dup = work[work.duplicated([unit, time], keep=False)]
            raise ValueError(
                "the panel contains duplicate (unit, time) pairs; "
                f"first offending rows:\n{dup.head()}"
            )

        self.unit = unit
        self.time = time
        self.df = work

        self._units = pd.Index(work[unit].unique())
        self._times = pd.Index(np.sort(work[time].unique()))
        self.df["_i"] = self._units.get_indexer(work[unit])
        # positional index into the observed periods: used for reshaping and
        # for grouping by period
        self.df["_t"] = self._times.get_indexer(work[time])
        # calendar key: spacing-aware, so that a unit missing period t-1 gets a
        # missing lag rather than silently borrowing period t-2
        self.df["_tkey"] = _calendar_key(work[time], self._times)

        self.N = len(self._units)
        self.T = len(self._times)
        self.balanced = len(work) == self.N * self.T

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------
    @property
    def units(self) -> pd.Index:
        """Distinct unit identifiers, in first-appearance order."""
        return self._units

    @property
    def times(self) -> pd.Index:
        """Distinct time identifiers, sorted ascending."""
        return self._times

    @property
    def n_obs(self) -> int:
        """Number of (unit, time) observations actually present."""
        return len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        kind = "balanced" if self.balanced else "unbalanced"
        return (
            f"PanelData(N={self.N}, T={self.T}, obs={self.n_obs}, {kind}, "
            f"unit={self.unit!r}, time={self.time!r})"
        )

    def summary(self) -> pd.Series:
        """Return a compact description of the panel's shape.

        Returns
        -------
        pandas.Series
            Units, periods, observations, balance, and per-unit observation
            counts (min / mean / max).
        """
        per_unit = self.df.groupby("_i").size()
        return pd.Series(
            {
                "units (N)": self.N,
                "periods (T)": self.T,
                "observations": self.n_obs,
                "balanced": self.balanced,
                "obs per unit (min)": int(per_unit.min()),
                "obs per unit (mean)": float(per_unit.mean()),
                "obs per unit (max)": int(per_unit.max()),
            }
        )

    # ------------------------------------------------------------------
    # lags and leads
    # ------------------------------------------------------------------
    def lag(self, cols: str | Sequence[str], lags: int | Iterable[int] = 1) -> pd.DataFrame:
        """Construct within-unit lags, respecting gaps in time.

        Unlike a plain ``groupby().shift()``, this aligns on the *time index*,
        so a unit missing period ``t-1`` yields ``NaN`` rather than silently
        borrowing period ``t-2``.  This matters for unbalanced panels and for
        any dynamic model where the lag must be the true previous period.

        Parameters
        ----------
        cols : str or sequence of str
            Column name(s) to lag.
        lags : int or iterable of int, default 1
            Lag order(s).  ``lags=3`` produces lags 1, 2, 3; an explicit
            iterable such as ``[1, 4]`` produces exactly those lags.

        Returns
        -------
        pandas.DataFrame
            One column per (variable, lag) named ``"{col}_lag{k}"``, aligned to
            ``self.df``'s row order.

        Examples
        --------
        >>> panel.lag("y", 2).columns.tolist()
        ['y_lag1', 'y_lag2']
        """
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(lags, int):
            lag_list = list(range(1, lags + 1))
        else:
            lag_list = list(lags)
        if any(k < 1 for k in lag_list):
            raise ValueError("lag orders must be >= 1")

        key = pd.MultiIndex.from_arrays([self.df["_i"], self.df["_tkey"]])
        out = {}
        for col in cols:
            if col not in self.df.columns:
                raise KeyError(f"column {col!r} not found in the panel")
            source = pd.Series(self.df[col].to_numpy(), index=key)
            for k in lag_list:
                shifted_key = pd.MultiIndex.from_arrays(
                    [self.df["_i"], self.df["_tkey"] - k]
                )
                out[f"{col}_lag{k}"] = source.reindex(shifted_key).to_numpy()
        return pd.DataFrame(out, index=self.df.index)

    def lead(self, cols: str | Sequence[str], leads: int | Iterable[int] = 1) -> pd.DataFrame:
        """Construct within-unit leads.  See :meth:`lag` for semantics."""
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(leads, int):
            lead_list = list(range(1, leads + 1))
        else:
            lead_list = list(leads)
        if any(k < 1 for k in lead_list):
            raise ValueError("lead orders must be >= 1")
        key = pd.MultiIndex.from_arrays([self.df["_i"], self.df["_tkey"]])
        out = {}
        for col in cols:
            if col not in self.df.columns:
                raise KeyError(f"column {col!r} not found in the panel")
            source = pd.Series(self.df[col].to_numpy(), index=key)
            for k in lead_list:
                shifted_key = pd.MultiIndex.from_arrays(
                    [self.df["_i"], self.df["_tkey"] + k]
                )
                out[f"{col}_lead{k}"] = source.reindex(shifted_key).to_numpy()
        return pd.DataFrame(out, index=self.df.index)

    def with_lags(
        self,
        cols: str | Sequence[str],
        lags: int | Iterable[int] = 1,
        *,
        dropna: bool = False,
    ) -> "PanelData":
        """Return a new panel with lag columns appended.

        Parameters
        ----------
        cols, lags
            As in :meth:`lag`.
        dropna : bool, default False
            Drop rows where any newly created lag is missing.  This is the
            usual "burn-in" step for dynamic models.

        Returns
        -------
        PanelData
            A new panel; the original is untouched.
        """
        lagged = self.lag(cols, lags)
        new_df = pd.concat([self.df.drop(columns=["_i", "_t", "_tkey"]), lagged], axis=1)
        if dropna:
            new_df = new_df.dropna(subset=lagged.columns.tolist())
        return PanelData(new_df, self.unit, self.time, copy=False)

    # ------------------------------------------------------------------
    # array extraction
    # ------------------------------------------------------------------
    def matrix(self, cols: Sequence[str], *, dropna: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Extract a design matrix plus the corresponding unit codes.

        Parameters
        ----------
        cols : sequence of str
            Columns to stack, in order.
        dropna : bool, default True
            Drop rows containing any missing value in ``cols``.

        Returns
        -------
        X : ndarray of shape (n, len(cols))
        units : ndarray of shape (n,)
            Integer unit codes, for clustering.
        """
        sub = self.df[list(cols) + ["_i"]]
        if dropna:
            sub = sub.dropna()
        return sub[list(cols)].to_numpy(dtype=float), sub["_i"].to_numpy()

    def wide(self, col: str) -> np.ndarray:
        """Return a ``(T, N)`` matrix for one variable, ``NaN`` where missing.

        Several estimators in the literature (notably the Arellano-Bond LASSO
        code of Chernozhukov et al.) are written against a ``T x N`` layout
        rather than long format.  This is the bridge.

        Parameters
        ----------
        col : str
            Column to reshape.

        Returns
        -------
        ndarray of shape (T, N)
        """
        if col not in self.df.columns:
            raise KeyError(f"column {col!r} not found in the panel")
        out = np.full((self.T, self.N), np.nan)
        out[self.df["_t"].to_numpy(), self.df["_i"].to_numpy()] = self.df[col].to_numpy()
        return out

    def from_wide(self, mat: np.ndarray, name: str) -> pd.Series:
        """Inverse of :meth:`wide`: map a ``(T, N)`` matrix back to long form."""
        mat = np.asarray(mat, dtype=float)
        if mat.shape != (self.T, self.N):
            raise ValueError(
                f"expected shape {(self.T, self.N)}, received {mat.shape}"
            )
        values = mat[self.df["_t"].to_numpy(), self.df["_i"].to_numpy()]
        return pd.Series(values, index=self.df.index, name=name)

    def balance(self) -> "PanelData":
        """Drop units not observed in every period, returning a balanced panel."""
        counts = self.df.groupby("_i")["_t"].nunique()
        keep = counts[counts == self.T].index
        sub = self.df[self.df["_i"].isin(keep)].drop(columns=["_i", "_t", "_tkey"])
        return PanelData(sub, self.unit, self.time, copy=False)
