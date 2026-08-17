"""Optimal shrinkage of fixed effects: URE and Empirical Bayes MLE.

A Python port of Kwon's ``FEShR``, implementing the estimators of

    Kwon, S. (2026). Optimal shrinkage estimation of fixed effects in linear
    panel data models. *Econometrica* 94(2), 663-677.

Setup
-----
Let :math:`y_j\\in\\mathbb R^{T}` be the least-squares estimate of unit ``j``'s
fixed effect (``T = 1`` for a time-invariant effect) with variance matrix
:math:`M_j`.  Shrink toward a location :math:`\\mu` with

.. math::
    \\widehat\\theta_j = (I - S_j)\\mu + S_j y_j,
    \\qquad S_j = \\Lambda(\\Lambda + M_j)^{-1},

and choose the hyper-parameters :math:`(\\mu,\\Lambda)` by one of

EBMLE
    Minimise the negative log marginal likelihood

    .. math::
        \\mathrm{nll} = \\frac1J\\sum_j\\Bigl[
            \\log\\det(\\Lambda+M_j)
            + (y_j-\\mu)'(\\Lambda+M_j)^{-1}(y_j-\\mu)\\Bigr].

URE
    Minimise Stein's unbiased risk estimate

    .. math::
        \\mathrm{URE} = \\frac1J\\sum_j\\Bigl[
            -2\\,\\mathrm{tr}\\bigl((\\Lambda+M_j)^{-1}M_j W M_j\\bigr)
            + \\bigl\\|M_j(\\Lambda+M_j)^{-1}(y_j-\\mu)\\bigr\\|_W^2\\Bigr].

URE requires no distributional assumption on the fixed effects, which is why
Cornejo and Sosa-Escudero prefer it to EBMLE for forecasting.

:math:`\\Lambda` is parameterised as :math:`LL'` with ``L`` lower triangular,
so the optimisation over ``L`` is unconstrained while :math:`\\Lambda` stays
positive semi-definite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["FEShrinkResult", "fe_shrink", "shrink_estimator"]


def _lower_to_lambda(L_flat: np.ndarray, T: int) -> np.ndarray:
    """Rebuild :math:`\\Lambda = LL'` from the packed lower triangle."""
    L = np.zeros((T, T))
    L[np.tril_indices(T)] = L_flat
    return L @ L.T


def _safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _nll(mu: np.ndarray, Lam: np.ndarray, y: np.ndarray, M: list[np.ndarray]) -> float:
    """Negative log marginal likelihood, averaged over units."""
    total = 0.0
    for j, M_j in enumerate(M):
        A = Lam + M_j
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf
        r = y[:, j] - mu[:, j]
        total += logdet + float(r @ _safe_inv(A) @ r)
    return total / len(M)


def _ure(
    mu: np.ndarray,
    Lam: np.ndarray,
    y: np.ndarray,
    M: list[np.ndarray],
    W: np.ndarray,
) -> float:
    """Unbiased risk estimate, averaged over units."""
    total = 0.0
    for j, M_j in enumerate(M):
        inv = _safe_inv(Lam + M_j)
        r = y[:, j] - mu[:, j]
        v = M_j @ inv @ r
        total += -2.0 * np.trace(inv @ M_j @ W @ M_j) + float(v @ W @ v)
    return total / len(M)


def _opt_mu_nll(Lam: np.ndarray, y: np.ndarray, M: list[np.ndarray]) -> np.ndarray:
    """GLS-style closed form for :math:`\\mu` under EBMLE."""
    T = y.shape[0]
    denom = np.zeros((T, T))
    numer = np.zeros(T)
    for j, M_j in enumerate(M):
        inv = _safe_inv(Lam + M_j)
        denom += inv
        numer += inv @ y[:, j]
    mu = _safe_inv(denom) @ numer
    return np.tile(mu[:, None], (1, y.shape[1]))


def _opt_mu_ure(
    Lam: np.ndarray,
    y: np.ndarray,
    M: list[np.ndarray],
    W: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Box-constrained quadratic program for :math:`\\mu` under URE.

    Minimises :math:`\\tfrac12\\mu'D\\mu - d'\\mu` subject to
    :math:`|\\mu_t|\\le` ``bounds[t]``.
    """
    T, J = y.shape
    D = np.zeros((T, T))
    d = np.zeros(T)
    for j, M_j in enumerate(M):
        inv_M = _safe_inv(Lam + M_j) @ M_j
        C = inv_M @ W @ inv_M.T
        D += C
        d += C @ y[:, j]
    D /= J
    d /= J

    if T == 1:
        mu_free = d[0] / D[0, 0] if D[0, 0] > 0 else 0.0
        mu = np.array([np.clip(mu_free, -bounds[0], bounds[0])])
    else:
        res = minimize(
            lambda m: 0.5 * m @ D @ m - d @ m,
            x0=np.zeros(T),
            jac=lambda m: D @ m - d,
            bounds=[(-b, b) for b in bounds],
            method="L-BFGS-B",
        )
        mu = res.x
    return np.tile(mu[:, None], (1, J))


@dataclass
class FEShrinkResult:
    """Output of :func:`fe_shrink`.

    Attributes
    ----------
    theta : ndarray of shape (T, J)
        Shrunken fixed-effect estimates.
    mu : ndarray of shape (T, J)
        Optimal shrinkage location (columns identical unless covariates used).
    Lambda : ndarray of shape (T, T)
        Optimal prior variance.
    obj : float
        Minimised objective (URE or negative log likelihood).
    shrinkage : ndarray of shape (J,)
        Per-unit shrinkage weight ``tr(S_j) / T``: 1 means no shrinkage
        (keep the raw estimate), 0 means full shrinkage to ``mu``.
    method : str
    """

    theta: np.ndarray
    mu: np.ndarray
    Lambda: np.ndarray
    obj: float
    shrinkage: np.ndarray
    method: str

    def __repr__(self) -> str:
        return (
            f"FEShrinkResult(method={self.method!r}, obj={self.obj:.4f}, "
            f"mean shrinkage={self.shrinkage.mean():.3f})"
        )


def shrink_estimator(
    mu: np.ndarray, Lam: np.ndarray, y: np.ndarray, M: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :math:`\\widehat\\theta_j = (I-S_j)\\mu + S_j y_j`.

    Parameters
    ----------
    mu : ndarray of shape (T, J)
    Lam : ndarray of shape (T, T)
    y : ndarray of shape (T, J)
    M : list of ndarray
        Per-unit variance matrices.

    Returns
    -------
    theta : ndarray of shape (T, J)
    shrinkage : ndarray of shape (J,)
        ``tr(S_j) / T`` for each unit.
    """
    T, J = y.shape
    theta = np.zeros((T, J))
    weights = np.zeros(J)
    for j, M_j in enumerate(M):
        S = Lam @ _safe_inv(Lam + M_j)
        theta[:, j] = (np.eye(T) - S) @ mu[:, j] + S @ y[:, j]
        weights[j] = float(np.trace(S)) / T
    return theta, weights


def fe_shrink(
    y: np.ndarray,
    M: list[np.ndarray] | np.ndarray,
    *,
    method: str = "URE",
    centering: str = "gen",
    W: np.ndarray | None = None,
    tau: float = 0.95,
    n_init: int = 1,
    diag_lambda: bool = False,
    seed: int | None = 0,
    maxiter: int = 500,
) -> FEShrinkResult:
    """Optimally shrink a vector of estimated fixed effects.

    Parameters
    ----------
    y : ndarray of shape (T, J) or (J,)
        Least-squares fixed-effect estimates.  A 1-D input is treated as
        ``T = 1``, the usual time-invariant case.
    M : list of ndarray, or ndarray of shape (J,)
        Variance of each unit's estimate.  For ``T = 1`` a plain array of
        variances is accepted.
    method : {'URE', 'EBMLE'}, default 'URE'
    centering : {'gen', '0'}, default 'gen'
        ``'gen'`` optimises a data-driven shrinkage location; ``'0'`` shrinks
        toward zero (equivalent to the grand mean when the effects are
        demeaned).
    W : ndarray of shape (T, T), optional
        Weight matrix in the risk criterion.  Defaults to the identity.
    tau : float, default 0.95
        Quantile of the data bounding the search for ``mu`` under URE.
    n_init : int, default 1
        Random restarts for the outer optimisation.
    diag_lambda : bool, default False
        Restrict :math:`\\Lambda` to be diagonal.  Irrelevant when ``T = 1``.
    seed : int, optional
    maxiter : int, default 500

    Returns
    -------
    FEShrinkResult

    Raises
    ------
    ValueError
        If ``method`` or ``centering`` is unrecognised, or shapes disagree.

    Examples
    --------
    Shrink 200 noisy unit effects, half of which are truly zero:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> eta = np.where(rng.random(200) < 0.5, 0.0, rng.normal(0, 1, 200))
    >>> se2 = np.full(200, 0.25)
    >>> y = eta + rng.normal(0, np.sqrt(se2))
    >>> out = fe_shrink(y, se2, method="URE")
    >>> float(np.mean((out.theta.ravel() - eta) ** 2)) < float(np.mean((y - eta) ** 2))
    True

    Notes
    -----
    Shrinkage reduces mean squared error whenever the fixed effects have
    limited dispersion relative to their estimation noise -- exactly the
    short-panel case.  It introduces bias, so use it for *forecasting*, not
    for inference on individual unit effects.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    T, J = y.shape

    if isinstance(M, np.ndarray) and M.ndim == 1:
        if len(M) != J:
            raise ValueError(f"M has length {len(M)} but y has {J} units")
        M = [np.array([[float(m)]]) for m in M]
    M = list(M)
    if len(M) != J:
        raise ValueError(f"len(M) = {len(M)} does not match y's {J} columns")

    method = method.upper()
    if method not in {"URE", "EBMLE"}:
        raise ValueError("method must be 'URE' or 'EBMLE'")
    if centering not in {"gen", "0"}:
        raise ValueError("centering must be 'gen' or '0'")

    if W is None:
        W = np.eye(T)
    bounds = np.array([np.quantile(y[t, :], tau) for t in range(T)])
    bounds = np.abs(bounds) + 1e-8

    rng = np.random.default_rng(seed)
    n_par = T if diag_lambda else T * (T + 1) // 2

    def build_lambda(par: np.ndarray) -> np.ndarray:
        if diag_lambda:
            return np.diag(par**2)
        return _lower_to_lambda(par, T)

    def objective(par: np.ndarray) -> float:
        Lam = build_lambda(par)
        if centering == "0":
            mu = np.zeros((T, J))
        elif method == "URE":
            mu = _opt_mu_ure(Lam, y, M, W, bounds)
        else:
            mu = _opt_mu_nll(Lam, y, M)
        val = _ure(mu, Lam, y, M, W) if method == "URE" else _nll(mu, Lam, y, M)
        return val if np.isfinite(val) else 1e10

    best_val, best_par = np.inf, None
    for _ in range(max(1, n_init)):
        x0 = rng.standard_normal(n_par)
        res = minimize(objective, x0, method="BFGS", options={"maxiter": maxiter})
        if res.fun < best_val:
            best_val, best_par = float(res.fun), res.x

    Lam = build_lambda(best_par)
    if centering == "0":
        mu = np.zeros((T, J))
    elif method == "URE":
        mu = _opt_mu_ure(Lam, y, M, W, bounds)
    else:
        mu = _opt_mu_nll(Lam, y, M)

    theta, weights = shrink_estimator(mu, Lam, y, M)
    return FEShrinkResult(
        theta=theta,
        mu=mu,
        Lambda=Lam,
        obj=best_val,
        shrinkage=weights,
        method=method,
    )
