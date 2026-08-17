"""Variance estimators for panel moment conditions.

Every estimator in ``dynpanelai`` ultimately solves a moment condition
:math:`\\sum_{i,t}\\psi_{it}(\\theta)=0` and reports

.. math::
    \\widehat V = \\widehat J^{-1}\\,\\widehat\\Omega\\,\\widehat J^{-\\top},

where :math:`\\widehat J` is the sample Jacobian and :math:`\\widehat\\Omega`
the variance of the score.  The functions here supply
:math:`\\widehat\\Omega` under the dependence structures the papers assume:

- :func:`cluster_variance` -- independence across units, arbitrary dependence
  within a unit.  The default everywhere in this package.
- :func:`twoway_cluster_variance` -- adds clustering on time, for panels with
  aggregate shocks (Cameron, Gelbach and Miller, 2011).
- :func:`driscoll_kraay_variance` -- HAC in the time dimension, for common
  shocks with persistence.
- :func:`newey_west_panel_variance` -- the one-lag correction used by the
  AB-LASSO code, where forward orthogonal deviations leave an MA(1) remainder.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cluster_variance",
    "twoway_cluster_variance",
    "driscoll_kraay_variance",
    "newey_west_panel_variance",
    "sandwich",
    "windmeijer_correction",
]


def _as_2d(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return scores[:, None] if scores.ndim == 1 else scores


def cluster_variance(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    dof_correct: bool = False,
) -> np.ndarray:
    """One-way cluster-robust score variance.

    Aggregates score contributions within each cluster before squaring:

    .. math::
        \\widehat\\Omega = \\sum_{i}
            \\Bigl(\\sum_{t\\in i}\\psi_{it}\\Bigr)
            \\Bigl(\\sum_{t\\in i}\\psi_{it}\\Bigr)^{\\top}.

    Parameters
    ----------
    scores : ndarray of shape (n,) or (n, k)
        Observation-level score contributions.
    clusters : ndarray of shape (n,)
        Cluster labels (typically the unit identifier).
    dof_correct : bool, default False
        Apply the finite-sample factor ``G / (G - 1)`` with ``G`` the number
        of clusters.

    Returns
    -------
    ndarray of shape (k, k)

    Examples
    --------
    >>> omega = cluster_variance(psi, units)
    >>> omega.shape
    (1, 1)
    """
    S = _as_2d(scores)
    clusters = np.asarray(clusters)
    if len(clusters) != len(S):
        raise ValueError("scores and clusters must have the same length")

    labels, codes = np.unique(clusters, return_inverse=True)
    k = S.shape[1]
    sums = np.zeros((len(labels), k))
    np.add.at(sums, codes, S)
    omega = sums.T @ sums
    if dof_correct and len(labels) > 1:
        omega *= len(labels) / (len(labels) - 1.0)
    return omega


def twoway_cluster_variance(
    scores: np.ndarray,
    clusters_a: np.ndarray,
    clusters_b: np.ndarray,
) -> np.ndarray:
    """Two-way cluster-robust score variance.

    Uses the inclusion-exclusion formula of Cameron, Gelbach and Miller (2011):

    .. math::
        \\widehat\\Omega = \\widehat\\Omega_A + \\widehat\\Omega_B
                          - \\widehat\\Omega_{A\\cap B}.

    Parameters
    ----------
    scores : ndarray of shape (n,) or (n, k)
    clusters_a, clusters_b : ndarray of shape (n,)
        The two clustering dimensions, e.g. unit and time.

    Returns
    -------
    ndarray of shape (k, k)

    Warnings
    --------
    The two-way estimator is not guaranteed positive semi-definite in finite
    samples.  If the returned matrix has negative eigenvalues, fall back to
    one-way clustering on the dimension with fewer clusters.
    """
    a = np.asarray(clusters_a)
    b = np.asarray(clusters_b)
    both = np.char.add(a.astype(str), np.char.add("|", b.astype(str)))
    return (
        cluster_variance(scores, a)
        + cluster_variance(scores, b)
        - cluster_variance(scores, both)
    )


def driscoll_kraay_variance(
    scores: np.ndarray,
    times: np.ndarray,
    *,
    lags: int | None = None,
) -> np.ndarray:
    """Driscoll-Kraay HAC variance, robust to cross-sectional dependence.

    Cross-sectionally averages the scores by period, then applies a Bartlett
    kernel across periods.  Appropriate when common shocks make units
    dependent on one another.

    Parameters
    ----------
    scores : ndarray of shape (n,) or (n, k)
    times : ndarray of shape (n,)
        Period labels.
    lags : int, optional
        Bartlett bandwidth.  Defaults to ``floor(4 (T/100)^(2/9))``.

    Returns
    -------
    ndarray of shape (k, k)
    """
    S = _as_2d(scores)
    times = np.asarray(times)
    labels, codes = np.unique(times, return_inverse=True)
    T = len(labels)
    k = S.shape[1]

    h = np.zeros((T, k))
    np.add.at(h, codes, S)

    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, T - 1))

    omega = h.T @ h
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        gamma = h[ell:].T @ h[:-ell]
        omega += w * (gamma + gamma.T)
    return omega


def newey_west_panel_variance(
    scores: np.ndarray,
    clusters: np.ndarray,
    order: np.ndarray,
    *,
    lags: int = 1,
    demean_within_cluster: bool = True,
) -> np.ndarray:
    """Within-cluster Newey-West variance with a fixed lag truncation.

    This reproduces the estimator used in the Arellano-Bond LASSO replication
    code, where the forward-orthogonal-deviation transform leaves at most an
    MA(1) remainder, so a one-lag correction suffices:

    .. math::
        \\widehat\\Sigma = \\Sigma_0
            + \\tfrac{T-2}{T-1}\\bigl(\\Sigma_1 + \\Sigma_1^{\\top}\\bigr),

    with :math:`\\Sigma_0` the contemporaneous outer product of demeaned
    scores and :math:`\\Sigma_1` the first within-unit autocovariance.

    Parameters
    ----------
    scores : ndarray of shape (n,) or (n, k)
    clusters : ndarray of shape (n,)
        Unit labels.
    order : ndarray of shape (n,)
        Time ordering within each unit.
    lags : int, default 1
        Number of autocovariance terms to include.
    demean_within_cluster : bool, default True
        Subtract each unit's mean score before forming the products, as the
        reference implementation does.

    Returns
    -------
    ndarray of shape (k, k)
    """
    S = _as_2d(scores).copy()
    clusters = np.asarray(clusters)
    order = np.asarray(order)

    labels, codes = np.unique(clusters, return_inverse=True)
    k = S.shape[1]

    if demean_within_cluster:
        counts = np.bincount(codes, minlength=len(labels)).astype(float)
        means = np.zeros((len(labels), k))
        np.add.at(means, codes, S)
        means /= counts[:, None]
        S = S - means[codes]

    omega = S.T @ S
    for ell in range(1, lags + 1):
        gamma = np.zeros((k, k))
        n_pairs = 0
        for g in range(len(labels)):
            sel = np.flatnonzero(codes == g)
            sel = sel[np.argsort(order[sel])]
            if len(sel) <= ell:
                continue
            a = S[sel[:-ell]]
            b = S[sel[ell:]]
            gamma += a.T @ b
            n_pairs += len(sel) - ell
        if n_pairs == 0:
            continue
        # Bartlett-style shrinkage matching the AB-LASSO code: (T-1-l)/(T-1)
        per_unit = np.bincount(codes, minlength=len(labels))
        T_eff = per_unit.max()
        w = max(0.0, (T_eff - ell - 1.0) / max(T_eff - 1.0, 1.0))
        omega += w * (gamma + gamma.T)
    return omega


def sandwich(jacobian: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Form :math:`J^{-1}\\Omega J^{-\\top}`, using a pseudo-inverse if needed.

    Parameters
    ----------
    jacobian : ndarray of shape (k, k)
    omega : ndarray of shape (k, k)

    Returns
    -------
    ndarray of shape (k, k)
    """
    J = np.atleast_2d(np.asarray(jacobian, dtype=float))
    Omega = np.atleast_2d(np.asarray(omega, dtype=float))
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        J_inv = np.linalg.pinv(J)
    return J_inv @ Omega @ J_inv.T


def windmeijer_correction(
    M: np.ndarray,
    M_XZ_W: np.ndarray,
    W_inv: np.ndarray,
    zs: np.ndarray,
    vcov_prev: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    resid_prev: np.ndarray,
    N: int,
) -> np.ndarray:
    """Windmeijer (2005) finite-sample correction for two-step GMM.

    Two-step GMM standard errors are severely downward-biased in short panels
    because the optimal weight matrix is estimated from first-step residuals.
    Windmeijer's correction adds the first-order effect of that estimation.

    Parameters
    ----------
    M : ndarray
        ``(X'ZW^{-1}Z'X)^{-1}`` at the second step.
    M_XZ_W : ndarray
        ``M @ X'Z @ W^{-1}``.
    W_inv : ndarray
        Inverse of the second-step weight matrix.
    zs : ndarray
        ``Z'u`` summed over units, evaluated at the **second-step**
        estimate.  Using first-step residuals here is a common and easy
        mistake; it changes the correction materially.
    vcov_prev : ndarray
        First-step variance matrix.
    X, Z : ndarray
        Stacked regressor and instrument matrices.
    resid_prev : ndarray
        First-step residuals.
    N : int
        Number of units.

    Returns
    -------
    ndarray
        The corrected variance matrix.

    References
    ----------
    Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators.  *Journal of Econometrics*
    126(1), 25-51.
    """
    k = X.shape[1]
    z_height = Z.shape[0] // N
    x_height = X.shape[0] // N

    D = np.zeros((M.shape[0], k))
    for j in range(k):
        zxz = np.zeros((Z.shape[1], Z.shape[1]))
        for i in range(N):
            x_i = X[i * x_height : (i + 1) * x_height, :]
            u_i = resid_prev[i * x_height : (i + 1) * x_height, 0:1]
            z_i = Z[i * z_height : (i + 1) * z_height, :]
            xu = x_i[:, j : j + 1] @ u_i.T
            zxz += z_i @ (xu + xu.T) @ z_i.T
        partial_dir = (-1.0 / N) * zxz
        D[:, j : j + 1] = -(M_XZ_W @ partial_dir @ W_inv @ zs)

    D_M = D @ M
    return N * M + N * D_M + N * D_M.T + D @ vcov_prev @ D.T
