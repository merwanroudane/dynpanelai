"""Test suite for dynpanelai.

Three kinds of test:

1. **Mathematical identities** -- the FOD operator annihilates constants and is
   orthonormal; gap-aware lags return NaN across gaps.  These must hold exactly.
2. **Recovery** -- each estimator recovers the truth of the DGP it was designed
   for, within a tolerance calibrated to the sample size.
3. **Guardrails** -- misuse raises, and estimators warn when their assumptions
   look violated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import dynpanelai as dp
from dynpanelai.core.transforms import fod_matrix
from dynpanelai.penalized import clime, nodewise_inverse, rlasso
from dynpanelai.sim import (
    make_ab_lasso_panel,
    make_partially_linear_panel,
    make_shrinkage_panel,
)


# ---------------------------------------------------------------- identities
@pytest.mark.parametrize("T", [2, 3, 5, 9, 20])
def test_fod_annihilates_constants_and_is_orthonormal(T):
    A = fod_matrix(T)
    assert A.shape == (T - 1, T)
    assert np.allclose(A @ np.ones(T), 0.0, atol=1e-12)
    assert np.allclose(A @ A.T, np.eye(T - 1), atol=1e-12)


def test_lag_is_gap_aware():
    df = pd.DataFrame(
        {"id": [1, 1, 1, 2, 2], "yr": [2000, 2001, 2003, 2000, 2001],
         "y": [1.0, 2.0, 3.0, 10.0, 20.0]}
    )
    panel = dp.PanelData(df, "id", "yr")
    lag = panel.lag("y", 1)["y_lag1"].tolist()
    assert np.isnan(lag[0])
    assert lag[1] == 1.0
    assert np.isnan(lag[2]), "a missing 2002 must not silently borrow 2001"
    assert np.isnan(lag[3])
    assert lag[4] == 10.0


def test_duplicate_pairs_raise():
    df = pd.DataFrame({"id": [1, 1], "yr": [2000, 2000], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        dp.PanelData(df, "id", "yr")


def test_wide_roundtrip():
    df, _ = make_ab_lasso_panel(N=20, T=8, seed=0)
    panel = dp.PanelData(df, "unit", "time")
    wide = panel.wide("y")
    assert wide.shape == (panel.T, panel.N)
    back = panel.from_wide(wide, "y2")
    assert np.allclose(back.to_numpy(), panel.df["y"].to_numpy())


# ------------------------------------------------------------------ recovery
def test_rlasso_recovers_sparse_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 60))
    beta = np.zeros(60)
    beta[[0, 5]] = [2.0, -1.5]
    y = X @ beta + rng.standard_normal(300) * 0.5
    fit = rlasso(X, y)
    assert fit.selected[0] and fit.selected[5]
    assert fit.n_selected <= 10
    assert abs(fit.coef[0] - 2.0) < 0.15
    assert abs(fit.coef[5] + 1.5) < 0.15


def test_clime_inverts_a_well_conditioned_matrix():
    Q = np.array([[2.0, 0.3], [0.3, 1.0]])
    Omega = clime(Q, lam=1e-8)
    assert np.allclose(Q @ Omega, np.eye(2), atol=1e-4)


def test_nodewise_inverse_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 6))
    Theta, tau2 = nodewise_inverse(X)
    assert Theta.shape == (6, 6)
    assert np.all(tau2 > 0)


def test_dml_recovers_treatment_effect():
    df, truth = make_partially_linear_panel(N=150, T=40, p=15, theta=1.0, seed=7)
    panel = dp.PanelData(df, "unit", "time")
    res = dp.DMLDynamicPanel(
        y="y", d="d", x=[f"x{j}" for j in range(15)],
        k_folds=4, buffer="log", learner="lasso", seed=7,
    ).fit(panel)
    assert abs(res.params["d"] - truth["theta"]) < 0.08
    assert res.bse["d"] > 0
    lo, hi = res.conf_int().loc["d"]
    assert lo < truth["theta"] < hi


def test_ablasso_recovers_ar_and_treatment():
    df, truth = make_ab_lasso_panel(N=200, T=20, seed=11)
    panel = dp.PanelData(df, "unit", "time")
    res = dp.ABLasso(
        y="y", d="d", lags=1, transform="fod",
        split=True, k_folds=2, n_splits=3, seed=1,
    ).fit(panel)
    assert abs(res.params["L1.y"] - truth["theta_1"]) < 0.10
    assert abs(res.params["d"] - truth["theta_2"]) < 0.06


def test_ablasso_fd_and_fod_both_run():
    df, _ = make_ab_lasso_panel(N=60, T=12, seed=2)
    panel = dp.PanelData(df, "unit", "time")
    for transform in ("fod", "fd"):
        res = dp.ABLasso(y="y", d="d", lags=1, transform=transform,
                         split=False, n_splits=1).fit(panel)
        assert np.isfinite(res.params).all()


def test_gmm_handles_unbalanced_panel():
    df = dp.datasets.load_abond_employment()
    panel = dp.PanelData(df, "id", "year")
    assert not panel.balanced
    res = dp.diff_gmm(panel, "n", lags=2, predetermined=["w"],
                      exogenous=["k"], gmm_lags=(2, 4), steps=2)
    # must keep all 140 firms rather than forcing balance
    assert res.n_units == 140
    assert res.n_obs > 500
    assert "Hansen J" in res.diagnostics


def test_bias_correction_moves_toward_truth():
    df, truth = make_ab_lasso_panel(N=300, T=12, seed=5)
    panel = dp.PanelData(df, "unit", "time")
    fe = dp.fixed_effects(panel, "y", lags=1, x=["d"])
    dfe = dp.debiased_fe(panel, "y", lags=1, x=["d"])
    truth_rho = truth["theta_1"]
    assert abs(dfe.params["L1.y"] - truth_rho) < abs(fe.params["L1.y"] - truth_rho)


def test_fe_shrink_reduces_mse():
    rng = np.random.default_rng(0)
    eta = np.where(rng.random(200) < 0.5, 0.0, rng.normal(0, 1, 200))
    var = np.full(200, 0.25)
    y = eta + rng.normal(0, np.sqrt(var))
    for method in ("URE", "EBMLE"):
        out = dp.fe_shrink(y, var, method=method)
        mse_raw = float(np.mean((y - eta) ** 2))
        mse_shrunk = float(np.mean((out.theta.ravel() - eta) ** 2))
        assert mse_shrunk < mse_raw
        assert np.all((out.shrinkage >= 0) & (out.shrinkage <= 1))


def test_penalized_fe_forecast_runs_for_all_methods():
    df, _ = make_shrinkage_panel(N=60, T=16, sparsity=0.5, seed=3)
    train = dp.PanelData(df[df.time < 13], "unit", "time")
    test = dp.PanelData(df[df.time >= 12], "unit", "time")
    for method in ("pols", "fe", "lasso", "ridge", "enet", "ebmle", "ure"):
        est = dp.PenalizedFE(y="y", x=["x"], method=method)
        est.fit(train)
        pred = est.predict(test)
        m = dp.forecast_metrics(test.df["y"].to_numpy(), pred.to_numpy())
        assert np.isfinite(m["rmse"])


def test_panel_lasso_recovers_under_weak_sparsity():
    rng = np.random.default_rng(4)
    N, T, p = 100, 30, 40
    eta = np.zeros(N)
    eta[rng.choice(N, 5, replace=False)] = rng.normal(0, 1, 5)
    beta = np.zeros(p)
    beta[:3] = [1.5, -1.0, 0.8]
    X = rng.standard_normal((T, N, p))
    y = np.zeros((T, N))
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + X[t] @ beta + eta + rng.standard_normal(N)
    df = pd.DataFrame({
        "unit": np.tile(np.arange(N), T),
        "time": np.repeat(np.arange(T), N),
        "y": y.ravel(),
    })
    for j in range(p):
        df[f"x{j}"] = X[:, :, j].ravel()
    panel = dp.PanelData(df, "unit", "time")
    res = dp.PanelLasso(y="y", lags=1, x=[f"x{j}" for j in range(p)],
                        lambda_M=0.05).fit(panel)
    assert abs(res.params["L1.y"] - 0.4) < 0.05
    assert abs(res.params["x0"] - 1.5) < 0.10
    assert abs(res.params["x3"]) < 0.10


def test_orthogonal_lasso_finds_no_spurious_heterogeneity():
    df, truth = make_partially_linear_panel(N=100, T=30, p=10, theta=1.0, seed=3)
    df["grp"] = (df["unit"] % 3).astype(str)
    panel = dp.PanelData(df, "unit", "time")
    res = dp.OrthogonalLasso(
        y="y", p="d", controls=[f"x{j}" for j in range(10)],
        heterogeneity=["grp"], k_blocks=5, debias="ridge", second_stage="ols",
    ).fit(panel)
    assert abs(res.params["(average)"] - truth["theta"]) < 0.15
    bands = res.extra["simultaneous_bands"]
    for name in bands.index:
        if name == "(average)":
            continue
        assert bands.loc[name, "lower"] <= 0 <= bands.loc[name, "upper"], (
            "simultaneous bands must cover zero when the effect is homogeneous"
        )


# ---------------------------------------------------------------- guardrails
def test_long_run_rejects_unit_root():
    res = dp.PanelResults(
        params=pd.Series({"d": 1.0, "L1.y": 1.0}),
        cov=np.eye(2) * 0.01,
    )
    with pytest.raises(ValueError, match="unit root"):
        res.long_run("d", ["L1.y"])


def test_long_run_matches_hand_calculation():
    res = dp.PanelResults(
        params=pd.Series({"d": 0.25, "L1.y": 0.75}),
        cov=np.eye(2) * 0.0,
    )
    lr, se = res.long_run("d", ["L1.y"])
    assert abs(lr - 1.0) < 1e-12
    assert se == pytest.approx(0.0)


def test_ablasso_requires_balanced_panel():
    df = dp.datasets.load_abond_employment()
    panel = dp.PanelData(df, "id", "year")
    with pytest.raises(ValueError, match="balanced"):
        dp.ABLasso(y="n", d="w", lags=1).fit(panel)


def test_blocked_folds_reject_infeasible_design():
    from dynpanelai.dml import blocked_time_folds

    with pytest.raises(ValueError):
        blocked_time_folds(range(8), k=4, buffer=10, max_lag=1)


def test_dml_warns_when_panel_too_short():
    df, _ = make_partially_linear_panel(N=400, T=10, p=5, seed=1)
    panel = dp.PanelData(df, "unit", "time")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dp.DMLDynamicPanel(y="y", d="d", x=["x0"], k_folds=2,
                           buffer=0, learner="ols").fit(panel)
    assert any("sqrt(N)/T" in str(w.message) for w in caught)


def test_panel_lasso_warns_when_effects_are_dense():
    df, _ = make_partially_linear_panel(N=40, T=20, p=8, seed=1)
    panel = dp.PanelData(df, "unit", "time")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dp.PanelLasso(y="y", lags=1, x=["d"], desparsify=False).fit(panel)
    assert any("weak-sparsity" in str(w.message) for w in caught)


def test_unknown_options_raise_clear_errors():
    df, _ = make_ab_lasso_panel(N=30, T=10, seed=0)
    panel = dp.PanelData(df, "unit", "time")
    with pytest.raises(ValueError, match="vcov"):
        dp.DMLDynamicPanel(y="y", d="d", vcov="nonsense", buffer=0).fit(panel)
    with pytest.raises(ValueError, match="transform"):
        dp.ABLasso(y="y", d="d", transform="nope", split=False).fit(panel)


# ---------------------------------------------------------------- reporting
def test_comparison_table_and_latex():
    df, _ = make_ab_lasso_panel(N=60, T=12, seed=2)
    panel = dp.PanelData(df, "unit", "time")
    results = {
        "FE": dp.fixed_effects(panel, "y", lags=1, x=["d"]),
        "DFE": dp.debiased_fe(panel, "y", lags=1, x=["d"]),
    }
    tab = dp.comparison_table(results, params=["L1.y", "d"])
    assert "FE" in tab.columns and "DFE" in tab.columns
    assert "Observations" in tab.index

    tex = dp.comparison_to_latex(results, params=["L1.y"], caption="T")
    assert "\\begin{table}" in tex and "\\toprule" in tex


def test_monte_carlo_table_reports_coverage():
    rng = np.random.default_rng(0)
    est = {"A": rng.normal(1.0, 0.1, 400)}
    ses = {"A": np.full(400, 0.1)}
    tab = dp.monte_carlo_table(est, truth=1.0, ses=ses)
    assert "coverage" in tab.columns
    assert 0.90 < float(tab.loc["A", "coverage"]) < 1.0


def test_results_summary_renders():
    df, _ = make_ab_lasso_panel(N=40, T=10, seed=0)
    panel = dp.PanelData(df, "unit", "time")
    res = dp.fixed_effects(panel, "y", lags=1, x=["d"])
    text = res.summary()
    assert "coef" in text and "L1.y" in text


def test_datasets_load():
    covid = dp.datasets.load_covid_counties()
    assert covid["fips"].nunique() == 2510
    assert covid["week"].nunique() == 32
    abond = dp.datasets.load_abond_employment()
    assert abond["id"].nunique() == 140
