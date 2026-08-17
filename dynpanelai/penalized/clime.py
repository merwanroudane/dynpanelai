"""CLIME and nodewise estimators of a precision matrix.

Debiasing a LASSO requires an approximate inverse :math:`\\widehat\\Omega` of
the sample Gram matrix :math:`\\widehat Q`.  Two constructions appear in the
literature this package implements:

CLIME (Cai, Liu and Luo, 2011)
    .. math::
        \\widehat\\Omega = \\arg\\min \\|\\Omega\\|_1
        \\quad\\text{s.t.}\\quad
        \\|\\widehat Q\\Omega - I\\|_\\infty \\le \\lambda_Q,

    which decomposes column by column into ``d`` independent linear programs.
    Semenova, Goldman, Chernozhukov and Taddy adopt CLIME because it needs
    only *approximate* sparsity of :math:`Q^{-1}`, whereas nodewise regression
    needs exact sparsity.

Nodewise regression (van de Geer, Buhlmann, Ritov and Dezeure, 2014)
    Regress each column of the design on the others by LASSO and assemble
    :math:`\\widehat\\Theta = \\widehat T^{-2}\\widehat C`.  This is the
    construction Kock and Tang (2019) use for the dynamic panel.

References
----------
Cai, T. T., Liu, W. and Luo, X. (2011). A constrained l1 minimization approach
to sparse precision matrix estimation. *JASA* 106(494), 594-607.

van de Geer, S., Buhlmann, P., Ritov, Y. and Dezeure, R. (2014). On
asymptotically optimal confidence regions and tests for high-dimensional
models. *Annals of Statistics* 42(3), 1166-1202.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

__all__ = ["clime", "clime_column", "nodewise_inverse", "symmetrize_clime"]


def clime_column(
    Q: np.ndarray,
    j: int,
    lam: float,
    *,
    method: str = "highs",
) -> np.ndarray:
    """Solve one CLIME column.

    Minimises :math:`\\|\\omega\\|_1` subject to
    :math:`\\|Q\\omega - e_j\\|_\\infty\\le\\lambda`.

    The problem is linearised by the split :math:`\\omega = u - v`,
    :math:`u,v\\ge 0`, giving

    .. math::
        \\min_{u,v\\ge 0} \\mathbf{1}'(u+v)
        \\quad\\text{s.t.}\\quad
        \\begin{pmatrix} Q & -Q \\\\ -Q & Q\\end{pmatrix}
        \\begin{pmatrix} u \\\\ v\\end{pmatrix}
        \\le
        \\begin{pmatrix} \\lambda + e_j \\\\ \\lambda - e_j\\end{pmatrix}.

    Parameters
    ----------
    Q : ndarray of shape (d, d)
        Sample second-moment matrix.
    j : int
        Column index to solve for.
    lam : float
        Constraint tolerance :math:`\\lambda_Q`.
    method : str, default 'highs'
        SciPy linear-programming solver.

    Returns
    -------
    ndarray of shape (d,)
        The ``j``-th column of the CLIME estimate.  Falls back to the ridge
        solution if the LP is infeasible.
    """
    d = Q.shape[0]
    e_j = np.zeros(d)
    e_j[j] = 1.0

    c = np.ones(2 * d)
    A_ub = np.vstack(
        [np.hstack([Q, -Q]), np.hstack([-Q, Q])]
    )
    b_ub = np.concatenate([lam + e_j, lam - e_j])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method=method)
    if not res.success:
        ridge = np.linalg.solve(Q + lam * np.eye(d), e_j)
        return ridge
    return res.x[:d] - res.x[d:]


def symmetrize_clime(Omega: np.ndarray) -> np.ndarray:
    """Symmetrise by taking the smaller-magnitude entry of each pair.

    Cai, Liu and Luo's rule: :math:`\\omega_{ij}` if
    :math:`|\\omega_{ij}| < |\\omega_{ji}|`, else :math:`\\omega_{ji}`.

    Parameters
    ----------
    Omega : ndarray of shape (d, d)

    Returns
    -------
    ndarray of shape (d, d)
        Symmetric matrix.
    """
    pick_ij = np.abs(Omega) <= np.abs(Omega.T)
    return np.where(pick_ij, Omega, Omega.T)


def clime(
    Q: np.ndarray,
    lam: float | None = None,
    *,
    n: int | None = None,
    symmetrize: bool = True,
    c_clime: float = 1.0,
    method: str = "highs",
) -> np.ndarray:
    """CLIME estimate of :math:`Q^{-1}`.

    Parameters
    ----------
    Q : ndarray of shape (d, d)
        Sample second-moment matrix, e.g. ``V.T @ V / n``.
    lam : float, optional
        Constraint tolerance.  If omitted and ``n`` is given, uses
        :math:`\\lambda_Q = c\\sqrt{\\log d / n}`.
    n : int, optional
        Sample size, used only for the default ``lam``.
    symmetrize : bool, default True
        Apply :func:`symmetrize_clime`.
    c_clime : float, default 1.0
        Constant in the default ``lam``.
    method : str, default 'highs'

    Returns
    -------
    ndarray of shape (d, d)

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([[2.0, 0.3], [0.3, 1.0]])
    >>> Om = clime(Q, lam=1e-6)
    >>> bool(np.allclose(Q @ Om, np.eye(2), atol=1e-3))
    True

    Notes
    -----
    Cost is ``d`` linear programs of size ``2d``, so this is practical for
    ``d`` in the low hundreds.  For larger problems prefer
    :func:`nodewise_inverse`, or pass a diagonal approximation.
    """
    Q = np.asarray(Q, dtype=float)
    d = Q.shape[0]
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    if lam is None:
        if n is None:
            raise ValueError("supply either `lam` or `n` to set the tolerance")
        lam = c_clime * np.sqrt(np.log(max(d, 2)) / n)

    Omega = np.column_stack([clime_column(Q, j, lam, method=method) for j in range(d)])
    return symmetrize_clime(Omega) if symmetrize else Omega


def nodewise_inverse(
    X: np.ndarray,
    *,
    lam: float | None = None,
    c: float = 1.1,
    gamma: float | None = None,
    post: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Nodewise-regression approximate inverse of ``X'X / n``.

    For each column ``j``, fit :math:`z_j = Z_{-j}\\varphi_j + \\zeta_j` by
    LASSO, then set

    .. math::
        \\widehat\\tau_j^2 = n^{-1}\\|z_j - Z_{-j}\\widehat\\varphi_j\\|^2
                             + \\lambda\\|\\widehat\\varphi_j\\|_1,
        \\qquad
        \\widehat\\Theta_j = \\widehat C_j / \\widehat\\tau_j^2,

    with :math:`\\widehat C` the matrix of ones on the diagonal and
    :math:`-\\widehat\\varphi_{j,l}` off it.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (already demeaned / transformed as appropriate).
    lam : float, optional
        Common penalty level for the nodewise regressions.  Defaults to the
        plug-in level from :func:`~dynpanelai.penalized.rlasso.lambda_plugin`.
    c, gamma : float
        Penalty constants passed through to the plug-in rule.
    post : bool, default False
        Use post-LASSO in each nodewise regression.

    Returns
    -------
    Theta : ndarray of shape (p, p)
        Approximate inverse.
    tau2 : ndarray of shape (p,)
        The :math:`\\widehat\\tau_j^2` normalisers.

    Notes
    -----
    Kock and Tang show that the KKT conditions of the nodewise LASSO imply
    :math:`\\|n^{-1}Z'Z\\widehat\\Theta_j - e_j\\|_\\infty \\le
    \\lambda_{node}/\\widehat\\tau_j^2`, which is exactly the bound needed to
    make the desparsification remainder negligible.
    """
    from .rlasso import lambda_plugin, rlasso as _rlasso

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    C = np.eye(p)
    tau2 = np.ones(p)

    for j in range(p):
        others = np.delete(np.arange(p), j)
        Z_j = X[:, others]
        z_j = X[:, j]
        fit = _rlasso(
            Z_j,
            z_j,
            post=post,
            intercept=False,
            lambda_start=lam,
            c=c,
            gamma=gamma,
        )
        resid = z_j - Z_j @ fit.coef
        lam_j = fit.lambda0 / (2.0 * n) if lam is None else lam / (2.0 * n)
        tau2[j] = float(resid @ resid / n + lam_j * np.abs(fit.coef).sum())
        tau2[j] = max(tau2[j], 1e-12)
        C[j, others] = -fit.coef

    Theta = C / tau2[:, None]
    return Theta, tau2
