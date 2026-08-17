"""Example 2 — School openings and COVID-19 spread (AB-LASSO).

Reproduces the empirical application of Chernozhukov, Fernandez-Val, Huang and
Wang (2024): weekly county-level case growth as a function of mitigation
policies, with four lags of the outcome.

Why AB-LASSO here?  With T = 32 and four lags, plain Arellano-Bond uses
thousands of moment conditions.  The small-bias condition m^2/(NT) -> 0 fails
badly, so the standard estimator is biased; LASSO moment selection fixes it.

Run
---
    python examples/02_ablasso_covid.py            # subsample, fast
    python examples/02_ablasso_covid.py --full     # full panel, slow
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

import dynpanelai as dp

warnings.filterwarnings("ignore", category=UserWarning)

POLICIES = ["school", "college", "pmask", "pshelter", "pgather50"]


def main(full: bool = False, n_splits: int = 5) -> None:
    df = dp.datasets.load_covid_counties()

    if not full:
        rng = np.random.default_rng(0)
        keep = rng.choice(df["fips"].unique(), 300, replace=False)
        df = df[df["fips"].isin(keep)].copy()
        print(f"Using a 300-county subsample. Pass --full for all counties.\n")

    panel = dp.PanelData(df, unit="fips", time="week").balance()
    print(panel, "\n")

    lags = 4
    m = panel.T * (panel.T - 1) // 2
    print(f"Approximate moment count m ~ {m}")
    print(f"m^2/(NT) = {m**2 / (panel.N * panel.T):.1f}")
    print("  -> well above 1: plain Arellano-Bond is biased here.\n")

    est = dp.ABLasso(
        y="logdc",
        d="dlogtests",
        c=POLICIES,
        lags=lags,
        transform="fod",       # the published paper's default
        split=True,
        k_folds=2,
        n_splits=n_splits,
        seed=202302,
    )
    res = est.fit(panel)
    print(res.summary())

    print("\n" + "=" * 78)
    print("Long-run policy effects")
    print("=" * 78)
    lag_names = [f"L{j}.logdc" for j in range(1, lags + 1)]
    print(f"{'policy':<16}{'long-run':>12}{'std.err.':>12}{'z':>10}")
    for pol in POLICIES:
        name = f"L1.{pol}"
        if name not in res.params.index:
            continue
        lr, se = res.long_run(name, lag_names)
        print(f"{pol:<16}{lr:>12.4f}{se:>12.4f}{lr / se:>10.2f}")

    print("\nNote: FOD is the paper's transform. The CRAN `ablasso` package")
    print("implements the earlier first-difference variant (transform='fd').")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="use all 2,510 counties")
    p.add_argument("--splits", type=int, default=5, help="random sample splits")
    a = p.parse_args()
    main(full=a.full, n_splits=a.splits)
