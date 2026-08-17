"""Tests that every GMM option actually changes the estimator.

These exist because version 0.1.0 shipped ``level``, ``collapse`` and
``time_dummies`` as silent no-ops: the parameters were accepted, stored and
documented, but never reached the instrument matrix.  ``system_gmm()`` returned
difference-GMM numbers under a "System GMM" label.

The lesson is that "it runs and returns plausible numbers" is not a test.  Each
option below is verified by its *structural* consequence -- how many
instruments it creates, how many rows it stacks, which columns are non-zero --
not merely that the call succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

import dynpanelai as dp
from dynpanelai.gmm.build import build_design
from dynpanelai.sim import make_ab_lasso_panel


@pytest.fixture(scope="module")
def employment():
    df = dp.datasets.load_abond_employment()
    return dp.PanelData(df, "id", "year")


BASE = dict(y="n", lags=2, predetermined=["w"], exogenous=["k"])


# ------------------------------------------------------------------ collapse
def test_collapse_reduces_instrument_count(employment):
    full = build_design(employment, collapse=False, **BASE)
    coll = build_design(employment, collapse=True, **BASE)
    assert coll.n_instruments < full.n_instruments, (
        "collapse=True must shrink the instrument matrix"
    )
    # uncollapsed grows like T^2, collapsed like T
    assert full.n_instruments > 2 * coll.n_instruments


def test_collapse_columns_span_all_periods(employment):
    """A collapsed column is active in many periods; an uncollapsed one is not."""
    full = build_design(employment, collapse=False, **BASE)
    coll = build_design(employment, collapse=True, **BASE)

    def share_spanning(d):
        multi = 0
        for j in range(d.Z.shape[1]):
            nz = d.Z[:, j] != 0
            if nz.any() and len(np.unique(d.periods[nz])) > 1:
                multi += 1
        return multi / d.Z.shape[1]

    # in the collapsed design most columns are active across many periods;
    # uncollapsed columns are period-specific by construction
    assert share_spanning(coll) > share_spanning(full) + 0.3


def test_collapse_changes_the_estimate(employment):
    a = dp.diff_gmm(employment, collapse=False, gmm_lags=(2, 4), steps=2, **BASE)
    b = dp.diff_gmm(employment, collapse=True, gmm_lags=(2, 4), steps=2, **BASE)
    assert a.diagnostics["instruments"] != b.diagnostics["instruments"]
    assert not np.isclose(a.params["L1.n"], b.params["L1.n"], atol=1e-10)


# -------------------------------------------------------------- system GMM
def test_system_gmm_stacks_the_level_equation(employment):
    diff = build_design(employment, level=False, **BASE)
    syst = build_design(employment, level=True, **BASE)
    assert syst.y.shape[0] > diff.y.shape[0], "system GMM must stack extra rows"
    assert syst.is_level.any(), "no level-equation rows were created"
    assert (~syst.is_level).any(), "the differenced rows disappeared"


def test_system_gmm_adds_level_instruments(employment):
    syst = build_design(employment, level=True, **BASE)
    assert syst.n_level_instruments > 0
    assert syst.n_instruments > syst.n_diff_instruments


def test_system_instrument_matrix_is_block_diagonal(employment):
    """Difference instruments must not load on level rows, and vice versa."""
    d = build_design(employment, level=True, **BASE)
    n_d = d.n_diff_instruments
    diff_block = d.Z[:, :n_d]
    level_block = d.Z[:, n_d:]
    assert np.allclose(diff_block[d.is_level], 0.0), (
        "difference instruments leak into the level equation"
    )
    assert np.allclose(level_block[~d.is_level], 0.0), (
        "level instruments leak into the differenced equation"
    )


def test_system_gmm_differs_from_difference_gmm(employment):
    """The regression that 0.1.0 failed: same numbers, different label.

    Exercised against the internal experimental path, since the public
    entry point is disabled.
    """
    from dynpanelai.gmm.estimator import _system_gmm_experimental

    kw = dict(lags=2, predetermined=["w"], exogenous=["k"],
              gmm_lags=(2, 4), steps=2)
    d = dp.diff_gmm(employment, "n", **kw)
    s = _system_gmm_experimental(employment, "n", **kw)
    assert not np.isclose(d.params["L1.n"], s.params["L1.n"], atol=1e-8)
    assert s.n_obs > d.n_obs
    assert "System" in s.method and "Difference" in d.method


# ------------------------------------------------------------- time dummies
def test_time_dummies_add_regressors_and_instruments(employment):
    a = build_design(employment, time_dummies=False, **BASE)
    b = build_design(employment, time_dummies=True, **BASE)
    assert b.X.shape[1] > a.X.shape[1], "time dummies never reached the regressors"
    assert b.n_instruments > a.n_instruments, "time dummies are not instrumenting"
    assert any(nm.startswith("T") for nm in b.names)


def test_time_dummies_change_the_estimate(employment):
    kw = dict(lags=2, predetermined=["w"], exogenous=["k"],
              gmm_lags=(2, 4), steps=2)
    a = dp.diff_gmm(employment, "n", time_dummies=False, **kw)
    b = dp.diff_gmm(employment, "n", time_dummies=True, **kw)
    assert not np.isclose(a.params["L1.n"], b.params["L1.n"], atol=1e-10)


# ------------------------------------------------------------------- recovery
def test_system_gmm_refuses_rather_than_misleads():
    """Disabled until the level instruments validate; must raise, not guess."""
    df, _ = make_ab_lasso_panel(N=80, T=8, seed=17)
    panel = dp.PanelData(df, "unit", "time")
    with pytest.raises(NotImplementedError, match="does not validate"):
        dp.system_gmm(panel, "y", lags=1, predetermined=["d"])


def test_diff_gmm_collapsed_recovers_a_known_parameter():
    """The validated path: collapsed difference GMM on a known DGP."""
    df, truth = make_ab_lasso_panel(N=300, T=10, seed=17)
    panel = dp.PanelData(df, "unit", "time")
    res = dp.diff_gmm(panel, "y", lags=1, predetermined=["d"],
                      gmm_lags=(2, 4), collapse=True, steps=2)
    assert abs(res.params["L1.y"] - truth["theta_1"]) < 0.10


def test_ar_tests_use_only_differenced_rows(employment):
    """Under system GMM (unit, period) is duplicated; the AR test must not choke."""
    from dynpanelai.gmm.estimator import _system_gmm_experimental

    res = _system_gmm_experimental(employment, "n", lags=2,
                                   predetermined=["w"], exogenous=["k"],
                                   gmm_lags=(2, 4), steps=2)
    for key in ("AR(1) [approximate]", "AR(2) [approximate]"):
        assert key in res.diagnostics
        assert "nan" not in str(res.diagnostics[key]).lower()


def test_ar_diagnostics_are_labelled_approximate(employment):
    """The simplified m-test must not be presented as the full Arellano-Bond test."""
    res = dp.diff_gmm(employment, "n", lags=2, predetermined=["w"],
                      exogenous=["k"], gmm_lags=(2, 4))
    assert any("approximate" in k for k in res.diagnostics), (
        "AR diagnostics must be labelled approximate until the full m-test lands"
    )


# ---------------------------------------------------------------- AC-GATE
def test_acgate_respects_the_layers_argument():
    pytest.importorskip("torch")
    from dynpanelai.neural import ACGate

    for depth in (1, 3):
        est = ACGate(y="y", features=["f0"], proxies=["p0"], K=4, layers=depth)
        net = est._build(n_features=1, n_proxies=1)
        assert net.lstm.num_layers == depth, (
            f"layers={depth} was ignored; LSTM depth is {net.lstm.num_layers}"
        )


def test_acgate_hidden_size_is_respected():
    pytest.importorskip("torch")
    from dynpanelai.neural import ACGate

    est = ACGate(y="y", features=["f0"], proxies=["p0"], K=4, hidden=17)
    net = est._build(n_features=1, n_proxies=1)
    assert net.lstm.hidden_size == 17


def test_acgate_lag_weights_are_a_distribution():
    torch = pytest.importorskip("torch")
    from dynpanelai.neural import ACGate

    est = ACGate(y="y", features=["f0", "f1"], proxies=["p0", "p1"], K=6)
    net = est._build(n_features=2, n_proxies=2)
    p = torch.randn(9, 2)
    w, _ = net.lag_weights(p)
    assert w.shape == (9, 6)
    assert torch.allclose(w.sum(dim=1), torch.ones(9), atol=1e-5)
    assert bool((w >= 0).all())


# ----------------------------------------------------------------- warnings
def test_instrument_proliferation_warning_is_not_silenced(employment):
    """The guardrail must survive pytest's warning filters."""
    small = dp.PanelData(
        employment.df[employment.df["id"] <= 12].drop(
            columns=["_i", "_t", "_tkey"]),
        "id", "year")
    with pytest.warns(UserWarning, match="instrument"):
        dp.diff_gmm(small, "n", lags=1, predetermined=["w"],
                    gmm_lags=(2, None), collapse=False, steps=1)
