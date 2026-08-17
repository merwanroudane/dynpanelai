"""Example 3 -- DML, orthogonal Lasso, shrinkage and AC-GATE on simulated data.

Every design here has known truth, so the estimates can be checked rather than
merely reported.  This is the script to run first when validating an install.

Run
---
    python examples/03_dml_and_shrinkage.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import dynpanelai as dp
from dynpanelai.sim import (
    make_partially_linear_panel,
    make_shrinkage_panel,
)

warnings.filterwarnings("ignore")


def dml_demo() -> None:
    print("=" * 78)
    print("A. Double machine learning with high-dimensional controls")
    print("=" * 78)
    df, truth = make_partially_linear_panel(N=150, T=40, p=15, theta=1.0, seed=7)
    panel = dp.PanelData(df, "unit", "time")

    out = {}
    for learner in ("lasso", "rf"):
        res = dp.DMLDynamicPanel(
            y="y", d="d", x=[f"x{j}" for j in range(15)],
            k_folds=4, buffer="log", learner=learner, seed=7,
        ).fit(panel)
        out[learner] = res
        print(f"{learner:>6}: theta = {res.params['d']:.4f} "
              f"(se {res.bse['d']:.4f})   bias = {res.params['d'] - truth['theta']:+.4f}")
    print(f"true theta = {truth['theta']}\n")


def ortho_demo() -> None:
    print("=" * 78)
    print("B. Orthogonal Lasso: is the effect heterogeneous?")
    print("=" * 78)
    df, truth = make_partially_linear_panel(N=120, T=30, p=12, theta=1.0, seed=3)
    df["group"] = (df["unit"] % 4).astype(str)
    panel = dp.PanelData(df, "unit", "time")

    res = dp.OrthogonalLasso(
        y="y", p="d", controls=[f"x{j}" for j in range(12)],
        heterogeneity=["group"], k_blocks=5,
        debias="ridge", second_stage="ols",
    ).fit(panel)
    print(res.table()[["coef", "std_err", "p_value", "sig"]].round(4).to_string())
    print("\nSimultaneous 95% bands:")
    print(res.extra["simultaneous_bands"].round(4).to_string())
    print("\nThe DGP has a HOMOGENEOUS effect of 1.0, so every group")
    print("interaction is truly zero.  Note what happens: at least one")
    print("interaction is 'significant' at 5% POINTWISE -- a false positive")
    print("from testing three coefficients at once.  Every simultaneous band")
    print("still covers zero.  This is why you report simultaneous bands when")
    print("the claim is about WHICH groups differ.\n")


def shrinkage_demo() -> None:
    print("=" * 78)
    print("C. Forecasting: shrinkage versus IV")
    print("=" * 78)
    df, truth = make_shrinkage_panel(N=100, T=20, gamma=0.2, sparsity=0.5, seed=3)
    train = dp.PanelData(df[df.time < 16], "unit", "time")
    test = dp.PanelData(df[df.time >= 15], "unit", "time")

    rows = {}
    for method in ("pols", "fe", "lasso", "ridge", "enet", "ebmle", "ure"):
        est = dp.PenalizedFE(y="y", x=["x"], method=method)
        r = est.fit(train)
        pred = est.predict(test)
        m = dp.forecast_metrics(test.df["y"].to_numpy(), pred.to_numpy())
        rows[r.method] = {
            "gamma": r.gamma, "RMSE": m["rmse"], "MAE": m["mae"],
        }
    tab = pd.DataFrame(rows).T.round(4)
    print(tab.to_string())
    print(f"\ntrue gamma = {truth['gamma']};  "
          f"{int(100 * truth['sparsity'])}% of fixed effects are exactly zero")
    print("Shrinkage trades bias for variance and wins on out-of-sample error.\n")


def acgate_demo() -> None:
    print("=" * 78)
    print("D. AC-GATE: entity-conditioned lag discovery")
    print("=" * 78)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch not installed; skipping. pip install dynpanelai[neural]\n")
        return

    from dynpanelai.neural import ACGate, run_audit
    from dynpanelai.sim import make_heterogeneous_lag_panel

    df, truth = make_heterogeneous_lag_panel(N=40, T=70, K=8, n_features=3,
                                             noise=0.2, seed=2)
    panel = dp.PanelData(df, "unit", "time")

    ks, mets = [], []
    for s in range(3):
        r = ACGate(y="y", features=["f0", "f1", "f2"],
                   proxies=["proxy0", "proxy1"], K=8, epochs=40, seed=s).fit(panel)
        ks.append(r.effective_lag.to_numpy())
        mets.append(r.metrics)

    strat = {
        "proxy0": df.groupby("unit")["proxy0"].first().to_numpy(),
        "proxy1": df.groupby("unit")["proxy1"].first().to_numpy(),
    }
    report = run_audit(ks, strat, metrics_by_seed=mets,
                       k_true=truth["k_star"], n_perm=300)
    print(report.summary())


if __name__ == "__main__":
    dml_demo()
    ortho_demo()
    shrinkage_demo()
    acgate_demo()
