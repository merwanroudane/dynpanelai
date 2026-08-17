"""Data-generating processes from the papers this package implements.

Each DGP reproduces the design of a specific paper's Monte Carlo, so the
estimators can be validated against known truth.

Functions
---------
:func:`make_partially_linear_panel`
    Sneller (2026), Section 6: partially linear dynamic panel with
    high-dimensional controls and an optional nonlinear ``g0``.
:func:`make_ab_lasso_panel`
    Chernozhukov, Fernandez-Val, Huang and Wang (2024), Section 4, following
    the Bun and Kiviet (2006) design used by Moral-Benito.
:func:`make_shrinkage_panel`
    Cornejo and Sosa-Escudero, Section 3: dynamic panel with a controllable
    fraction of exactly-zero fixed effects.
:func:`make_heterogeneous_lag_panel`
    Xu (2026): entity-conditioned heterogeneous lags with known ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_partially_linear_panel",
    "make_ab_lasso_panel",
    "make_shrinkage_panel",
    "make_heterogeneous_lag_panel",
]


def make_partially_linear_panel(
    N: int = 200,
    T: int = 40,
    p: int = 20,
    *,
    theta: float = 1.0,
    rho: float = 0.4,
    ar_x: float = 0.5,
    nonlinear: bool = False,
    burn_in: int = 20,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Partially linear dynamic panel with high-dimensional controls.

    .. math::
        Y_{it} = \\rho Y_{i,t-1} + \\theta D_{it} + g_0(X_{it})
                 + \\alpha_i + U_{it},
        \\qquad
        D_{it} = m_0(X_{it}) + \\alpha_i + V_{it},

    with :math:`X_{it}` a ``p``-dimensional AR(1) vector.  With
    ``nonlinear=False`` both :math:`g_0` and :math:`m_0` are sparse linear
    functions; with ``nonlinear=True`` they include squared and interaction
    terms, so a linear learner is misspecified and boosting or forests are
    needed to satisfy the product-rate condition.

    Parameters
    ----------
    N, T, p : int
        Units, periods (after burn-in), and number of controls.
    theta : float, default 1.0
        True treatment effect -- the estimand.
    rho : float, default 0.4
        Autoregressive coefficient.
    ar_x : float, default 0.5
        Persistence of each control.
    nonlinear : bool, default False
    burn_in : int, default 20
        Periods generated then discarded, to remove initial-condition effects.
    seed : int, optional

    Returns
    -------
    df : pandas.DataFrame
        Columns ``unit``, ``time``, ``y``, ``d``, ``x0``...``x{p-1}``.
    truth : dict
        ``{'theta': ..., 'rho': ..., 'beta_y': ..., 'beta_d': ...}``.

    Examples
    --------
    >>> df, truth = make_partially_linear_panel(N=50, T=20, p=10, seed=0)
    >>> truth["theta"]
    1.0
    >>> sorted(df.columns)[:4]
    ['d', 'time', 'unit', 'x0']
    """
    rng = np.random.default_rng(seed)
    T_tot = T + burn_in

    s = min(5, p)
    beta_y = np.zeros(p)
    beta_d = np.zeros(p)
    beta_y[:s] = rng.uniform(0.5, 1.5, s) * rng.choice([-1, 1], s)
    beta_d[:s] = rng.uniform(0.3, 1.0, s) * rng.choice([-1, 1], s)

    alpha = rng.normal(0, 1, N)
    X = np.zeros((T_tot, N, p))
    X[0] = rng.normal(0, 1, (N, p))
    for t in range(1, T_tot):
        X[t] = ar_x * X[t - 1] + rng.normal(0, 1, (N, p)) * np.sqrt(1 - ar_x**2)

    def g0(Xt: np.ndarray) -> np.ndarray:
        lin = Xt @ beta_y
        if nonlinear:
            return lin + 0.5 * Xt[:, 0] ** 2 + 0.5 * Xt[:, 0] * Xt[:, 1]
        return lin

    def m0(Xt: np.ndarray) -> np.ndarray:
        lin = Xt @ beta_d
        if nonlinear:
            return lin + 0.4 * np.tanh(Xt[:, 1]) + 0.3 * Xt[:, 2] ** 2
        return lin

    Y = np.zeros((T_tot, N))
    D = np.zeros((T_tot, N))
    for t in range(1, T_tot):
        D[t] = m0(X[t]) + alpha + rng.normal(0, 1, N)
        Y[t] = rho * Y[t - 1] + theta * D[t] + g0(X[t]) + alpha + rng.normal(0, 1, N)

    sl = slice(burn_in, T_tot)
    unit = np.tile(np.arange(N), T)
    time = np.repeat(np.arange(T), N)
    data = {
        "unit": unit,
        "time": time,
        "y": Y[sl].ravel(),
        "d": D[sl].ravel(),
    }
    for j in range(p):
        data[f"x{j}"] = X[sl, :, j].ravel()
    return pd.DataFrame(data), {
        "theta": theta,
        "rho": rho,
        "beta_y": beta_y,
        "beta_d": beta_d,
    }


def make_ab_lasso_panel(
    N: int = 200,
    T: int = 40,
    *,
    theta: tuple[float, float] = (0.75, 0.25),
    rho: float = 0.5,
    phi: float = -0.17,
    pi: float = 0.67,
    sigma_alpha: float = np.sqrt(2.96),
    df_t: int = 4,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The Bun-Kiviet design used in the AB-LASSO Monte Carlo.

    .. math::
        Y_{it} = \\alpha_i + \\theta_1 Y_{i,t-1} + \\theta_2 D_{it}
                 + \\varepsilon_{it},
        \\qquad
        D_{it} = \\rho D_{i,t-1} + \\phi Y_{i,t-1} + \\pi\\alpha_i + v_{it},

    with :math:`t`-distributed innovations.  The treatment is *predetermined*
    but not strictly exogenous: :math:`\\phi\\neq 0` means past outcomes feed
    back into current treatment, which is precisely the case where the within
    estimator fails and moment-based methods are needed.

    Initial conditions are set to the mean-stationary solution, as in
    Moral-Benito (2013).

    Parameters
    ----------
    N, T : int
    theta : (float, float), default (0.75, 0.25)
        ``(theta_1, theta_2)``: the AR coefficient and the treatment effect.
    rho, phi, pi : float
        Treatment persistence, outcome feedback, and unit-effect loading.
    sigma_alpha : float
        Standard deviation of the unit effects.
    df_t : int, default 4
        Degrees of freedom of the ``t`` innovations.
    seed : int, optional

    Returns
    -------
    df : pandas.DataFrame
        Columns ``unit``, ``time``, ``y``, ``d``.
    truth : dict
        Includes ``theta_lr = theta_2 / (1 - theta_1)``, the long-run effect.

    Examples
    --------
    >>> df, truth = make_ab_lasso_panel(N=100, T=20, seed=1)
    >>> round(truth["theta_lr"], 3)
    1.0
    """
    rng = np.random.default_rng(seed)
    th1, th2 = theta

    alpha = rng.normal(0, sigma_alpha, N)
    eps_y = rng.standard_t(df_t, size=(T + 1, N))
    eps_d = rng.standard_t(df_t, size=(T + 1, N))

    Y = np.zeros((T + 1, N))
    D = np.zeros((T + 1, N))
    denom = (1 - rho) * (1 - th1) - th2 * phi
    D[0] = ((phi + pi * (1 - th1)) / denom) * alpha + eps_d[0]
    Y[0] = ((th2 * pi + (1 - rho)) / denom) * alpha + eps_y[0]
    for t in range(1, T + 1):
        D[t] = rho * D[t - 1] + phi * Y[t - 1] + pi * alpha + eps_d[t]
        Y[t] = th1 * Y[t - 1] + th2 * D[t] + alpha + eps_y[t]

    Y, D = Y[1:], D[1:]
    df = pd.DataFrame(
        {
            "unit": np.tile(np.arange(N), T),
            "time": np.repeat(np.arange(T), N),
            "y": Y.ravel(),
            "d": D.ravel(),
        }
    )
    return df, {
        "theta_1": th1,
        "theta_2": th2,
        "theta_lr": th2 / (1 - th1),
        "rho": rho,
        "phi": phi,
    }


def make_shrinkage_panel(
    N: int = 100,
    T: int = 20,
    *,
    gamma: float = 0.2,
    beta: float | None = None,
    sparsity: float = 0.0,
    rho_x: float = 0.5,
    sigma_eta: float = 1.0,
    sigma_eps: float = 1.0,
    sigma_xi: float = 2.0,
    burn_in: int = 10,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Cornejo and Sosa-Escudero's forecasting design with sparse fixed effects.

    .. math::
        y_{it} = \\gamma y_{i,t-1} + x_{it}'\\beta + \\eta_i + \\varepsilon_{it},

    where a fraction ``sparsity`` of the :math:`\\eta_i` are set to exactly
    zero.  This is the structure that shrinkage estimators exploit and that
    IV-based estimators cannot.

    Parameters
    ----------
    N, T : int
    gamma : float, default 0.2
        Autoregressive coefficient; the paper uses 0.2 (low persistence) and
        0.8 (high).
    beta : float, optional
        Slope on ``x``.  Defaults to ``1 - gamma``, as in the paper.
    sparsity : float, default 0.0
        Fraction of units with :math:`\\eta_i = 0`, in ``[0, 1]``.
    rho_x : float, default 0.5
        AR(1) coefficient of the regressor.
    sigma_eta, sigma_eps, sigma_xi : float
        Standard deviations.  The paper fixes ``sigma_xi = 2`` (variance 4).
    burn_in : int, default 10
    seed : int, optional

    Returns
    -------
    df : pandas.DataFrame
        Columns ``unit``, ``time``, ``y``, ``x``.
    truth : dict
        Includes the realised ``eta`` vector, for evaluating shrinkage.

    Examples
    --------
    >>> df, truth = make_shrinkage_panel(N=50, T=10, sparsity=0.5, seed=0)
    >>> float((truth["eta"] == 0).mean())
    0.5
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = 1.0 - gamma
    T_tot = T + burn_in

    eta = rng.normal(0, sigma_eta, N)
    n_zero = int(round(sparsity * N))
    if n_zero:
        eta[rng.choice(N, n_zero, replace=False)] = 0.0

    x = np.zeros((T_tot, N))
    for t in range(1, T_tot):
        x[t] = rho_x * x[t - 1] + rng.normal(0, sigma_xi, N)

    y = np.zeros((T_tot, N))
    for t in range(1, T_tot):
        y[t] = gamma * y[t - 1] + beta * x[t] + eta + rng.normal(0, sigma_eps, N)

    sl = slice(burn_in, T_tot)
    df = pd.DataFrame(
        {
            "unit": np.tile(np.arange(N), T),
            "time": np.repeat(np.arange(T), N),
            "y": y[sl].ravel(),
            "x": x[sl].ravel(),
        }
    )
    return df, {"gamma": gamma, "beta": beta, "eta": eta, "sparsity": sparsity}


def make_heterogeneous_lag_panel(
    N: int = 60,
    T: int = 80,
    *,
    K: int = 8,
    n_features: int = 4,
    nonlinear: bool = False,
    noise: float = 0.3,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Entity-conditioned heterogeneous lags with known ground truth.

    Each unit ``i`` has a latent proxy score that determines its own effective
    lag :math:`k^\\star_i`; the outcome responds to the covariates at that lag.
    Used to test whether AC-GATE recovers the true lag ordering.

    Parameters
    ----------
    N, T : int
    K : int, default 8
        Maximum lag horizon.
    n_features : int, default 4
        Number of dynamic covariates.
    nonlinear : bool, default False
        Make the proxy-to-lag map nonlinear.
    noise : float, default 0.3
    seed : int, optional

    Returns
    -------
    df : pandas.DataFrame
        Columns ``unit``, ``time``, ``y``, ``f0``...``f{n-1}``, ``proxy0``,
        ``proxy1``.
    truth : dict
        ``{'k_star': ndarray of shape (N,)}`` -- the true effective lag.

    Examples
    --------
    >>> df, truth = make_heterogeneous_lag_panel(N=20, T=40, seed=0)
    >>> truth["k_star"].shape
    (20,)
    """
    rng = np.random.default_rng(seed)

    proxy = rng.uniform(0, 1, (N, 2))
    score = proxy[:, 0] - 0.5 * proxy[:, 1]
    if nonlinear:
        score = np.tanh(3 * score) + 0.3 * proxy[:, 0] ** 2
    lo, hi = score.min(), score.max()
    unit_scale = (score - lo) / (hi - lo + 1e-12)
    # stronger proxies respond faster (shorter effective lag)
    k_star = 1 + np.round((K - 1) * (1 - unit_scale)).astype(int)

    F = rng.normal(0, 1, (T, N, n_features))
    for t in range(1, T):
        F[t] = 0.6 * F[t - 1] + np.sqrt(1 - 0.36) * F[t]

    w = rng.normal(0, 1, n_features)
    y = np.full((T, N), np.nan)
    for i in range(N):
        k = k_star[i]
        for t in range(k, T):
            y[t, i] = F[t - k, i] @ w + noise * rng.normal()

    mask = ~np.isnan(y)
    unit = np.tile(np.arange(N), (T, 1))
    time = np.repeat(np.arange(T)[:, None], N, axis=1)
    data = {
        "unit": unit[mask],
        "time": time[mask],
        "y": y[mask],
    }
    for j in range(n_features):
        data[f"f{j}"] = F[:, :, j][mask]
    for j in range(2):
        data[f"proxy{j}"] = np.tile(proxy[:, j], (T, 1))[mask]
    df = pd.DataFrame(data).sort_values(["unit", "time"]).reset_index(drop=True)
    return df, {"k_star": k_star, "weights": w, "K": K}
