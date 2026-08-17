"""Real economics datasets bundled with ``dynpanelai``.

Two panels ship with the package, both used as the running examples
throughout the documentation:

:func:`load_covid_counties`
    2,510 US counties x 32 weeks of COVID-19 case growth and mitigation
    policies.  The application panel of Chernozhukov, Fernandez-Val, Huang
    and Wang (2024), originally assembled by Chernozhukov, Kasahara and
    Schrimpf.  Large ``N``, moderate ``T``, staggered non-stationary policy
    indicators -- the setting AB-LASSO was designed for.

:func:`load_abond_employment`
    140 UK firms x 9 years of employment, wages and capital: the canonical
    Arellano and Bond (1991) panel.  Small ``N``, very short ``T`` -- the
    setting where GMM is appropriate and ML methods are not.

Both are returned as tidy long-format frames, ready for :class:`PanelData`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_covid_counties",
    "load_abond_employment",
    "available_datasets",
    "DATA_DIR",
]

DATA_DIR = Path(__file__).parent / "data"


def available_datasets() -> pd.DataFrame:
    """List the bundled datasets and their dimensions.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> available_datasets()["name"].tolist()
    ['covid_counties', 'abond_employment']
    """
    return pd.DataFrame(
        [
            {
                "name": "covid_counties",
                "loader": "load_covid_counties",
                "units": 2510,
                "periods": 32,
                "unit": "fips (county)",
                "time": "week",
                "outcome": "logdc",
                "source": "Chernozhukov, Fernandez-Val, Huang & Wang (2024)",
            },
            {
                "name": "abond_employment",
                "loader": "load_abond_employment",
                "units": 140,
                "periods": 9,
                "unit": "id (firm)",
                "time": "year",
                "outcome": "n (log employment)",
                "source": "Arellano & Bond (1991)",
            },
        ]
    )


def load_covid_counties(*, balanced: bool = True, add_growth: bool = True) -> pd.DataFrame:
    """COVID-19 case growth and mitigation policies in US counties.

    Weekly county-level panel, 1 April to 2 December 2020.  Aggregating to
    weeks avoids the spurious serial correlation that the underlying
    seven-day moving averages would otherwise induce.

    Parameters
    ----------
    balanced : bool, default True
        Keep only counties observed in all 32 weeks.
    add_growth : bool, default True
        Append ``dlogdc``, the weekly change in log cases.

    Returns
    -------
    pandas.DataFrame
        Columns:

        ``fips``
            County identifier.
        ``week``
            Week number.
        ``logdc``
            Log reported COVID-19 cases (the outcome).
        ``dlogtests``
            Weekly growth rate of tests -- a contemporaneous control for
            detection intensity.
        ``school``, ``college``
            SafeGraph foot-traffic measures for K-12 schools and colleges.
        ``pmask``, ``pshelter``, ``pgather50``
            Policy indicators: mask mandates, stay-at-home orders, and bans on
            gatherings over 50 people.

    Notes
    -----
    The policy variables are *staggered* and non-stationary, which is why the
    half-panel jackknife of Chudik, Pesaran and Yang cannot be used here --
    it assumes unconditional stationarity.

    Examples
    --------
    >>> df = load_covid_counties()
    >>> df.shape[1] >= 9
    True
    >>> sorted(df["week"].unique())[:3]
    [17, 18, 19]
    """
    path = DATA_DIR / "covid_counties.parquet"
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"bundled dataset missing at {path}; reinstall dynpanelai"
        )
    df = pd.read_parquet(path)
    df["fips"] = df["fips"].astype(str)
    df["week"] = df["week"].astype(int)
    df = df.sort_values(["fips", "week"]).reset_index(drop=True)

    if balanced:
        counts = df.groupby("fips")["week"].nunique()
        keep = counts[counts == counts.max()].index
        df = df[df["fips"].isin(keep)].reset_index(drop=True)

    if add_growth:
        df["dlogdc"] = df.groupby("fips")["logdc"].diff()

    cols = [
        "fips", "week", "logdc", "dlogtests",
        "school", "college", "pmask", "pshelter", "pgather50",
    ]
    if add_growth:
        cols.append("dlogdc")
    return df[cols]


def load_abond_employment(*, add_logs: bool = True) -> pd.DataFrame:
    """UK firm employment panel of Arellano and Bond (1991).

    140 firms observed 1976-1984, the standard test bed for dynamic panel
    GMM.  Reproduces the specification in the ``xtabond2`` documentation.

    Parameters
    ----------
    add_logs : bool, default True
        Keep the pre-computed logs ``n``, ``w``, ``k``, ``ys``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``year``, and

        ``n``
            Log employment (the outcome).
        ``w``
            Log real wage.
        ``k``
            Log gross capital.
        ``ys``
            Log industry output.
        ``emp``, ``wage``, ``cap``, ``indoutpt``
            The corresponding levels.

    Notes
    -----
    With ``T = 9`` this panel is far too short for the machine-learning
    estimators in this package, which need ``sqrt(N)/T -> 0``.  It is included
    precisely to make that contrast concrete: use
    :mod:`dynpanelai.gmm` here, not :mod:`dynpanelai.dml`.

    Examples
    --------
    >>> df = load_abond_employment()
    >>> df["id"].nunique()
    140
    """
    path = DATA_DIR / "abond_employment.parquet"
    if not path.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"bundled dataset missing at {path}; reinstall dynpanelai"
        )
    df = pd.read_parquet(path)
    keep = ["id", "year", "n", "w", "k", "ys", "emp", "wage", "cap", "indoutpt"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["id"] = df["id"].astype(int)
    df["year"] = df["year"].astype(int)
    if not add_logs:
        df = df.drop(columns=[c for c in ("n", "w", "k", "ys") if c in df])
    return df.sort_values(["id", "year"]).reset_index(drop=True)
