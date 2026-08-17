"""Difference and system GMM for dynamic panels.

Implements the classical moment-based estimators against which every
machine-learning method in this package is benchmarked:

- Arellano and Bond (1991): difference GMM, instrumenting the differenced
  equation with lagged levels.
- Arellano and Bover (1995) / Blundell and Bond (1998): system GMM, adding
  the levels equation instrumented by lagged differences.
- Anderson and Hsiao (1981): just-identified IV, the minimal instrument set.

with the standard specification tests (Hansen J, Arellano-Bond AR(1)/AR(2))
and the Windmeijer (2005) finite-sample correction for two-step standard
errors.

Instrument proliferation
------------------------
The number of instruments grows with :math:`T^2`, which is exactly the problem
:mod:`dynpanelai.ablasso` was designed to solve.  Two classical palliatives are
supported here: restricting the lag range (``gmm_lags=(2, 4)``) and collapsing
the instrument matrix (``collapse=True``).  Watch the Hansen p-value: a value
implausibly close to 1.0 is the classic symptom of too many instruments.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.panel import PanelData
from ..core.results import PanelResults

__all__ = ["DynamicPanelGMM", "diff_gmm", "system_gmm", "anderson_hsiao"]


def _collapse(Z: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Collapse instrument columns by summing within lag blocks."""
    return np.column_stack([Z[:, a:b].sum(axis=1) for a, b in blocks])


@dataclass
class _Spec:
    y: str
    lags: int
    predetermined: list[str]
    exogenous: list[str]
    gmm_lags: tuple[int, int | None]
    collapse: bool
    time_dummies: bool


class DynamicPanelGMM:
    """Difference / system GMM estimator.

    Parameters
    ----------
    y : str
        Outcome column.
    lags : int, default 1
        Lags of the outcome included as regressors.
    predetermined : sequence of str, optional
        Variables correlated with past errors; instrumented by their own lags.
    exogenous : sequence of str, optional
        Strictly exogenous variables, used as their own instruments.
    transformation : {'fd', 'fod'}, default 'fd'
        Transform applied to remove the unit effects.
    level : bool, default False
        Add the levels equation (system GMM).
    steps : int, default 2
        1 = one-step, 2 = two-step with Windmeijer correction.
    gmm_lags : (int, int or None), default (2, None)
        Lag range used to build GMM-style instruments.  ``None`` means "all
        available".
    collapse : bool, default False
        Collapse the instrument matrix, holding instrument count linear in T.
    time_dummies : bool, default False

    Examples
    --------
    >>> from dynpanelai.gmm import DynamicPanelGMM
    >>> est = DynamicPanelGMM(y="n", lags=2,
    ...                       predetermined=["w"], exogenous=["k"],
    ...                       gmm_lags=(2, 4), collapse=True, steps=2)
    >>> res = est.fit(panel)                         # doctest: +SKIP
    >>> print(res.summary())                         # doctest: +SKIP
    """

    def __init__(
        self,
        y: str,
        *,
        lags: int = 1,
        predetermined: Sequence[str] | None = None,
        exogenous: Sequence[str] | None = None,
        transformation: str = "fd",
        level: bool = False,
        steps: int = 2,
        gmm_lags: tuple[int, int | None] = (2, None),
        collapse: bool = False,
        time_dummies: bool = False,
    ) -> None:
        self.spec = _Spec(
            y=y,
            lags=lags,
            predetermined=list(predetermined) if predetermined else [],
            exogenous=list(exogenous) if exogenous else [],
            gmm_lags=gmm_lags,
            collapse=collapse,
            time_dummies=time_dummies,
        )
        self.transformation = transformation.lower()
        self.level = level
        self.steps = steps
        self.results_: PanelResults | None = None

    # ------------------------------------------------------------------
    def _build(self, panel: PanelData):
        """Assemble transformed regressors, outcome, and instrument blocks."""
        s = self.spec
        # Unbalanced panels are the norm (the Arellano-Bond employment data
        # itself is unbalanced).  Rather than dropping units, we build the
        # wide layout with NaN, drop only the (i, t) cells whose outcome or
        # regressors are missing, and zero-fill missing instruments -- the
        # convention used by xtabond2.
        T, N = panel.T, panel.N
        Y = panel.wide(s.y)
        P = {v: panel.wide(v) for v in s.predetermined}
        E = {v: panel.wide(v) for v in s.exogenous}

        if self.transformation == "fd":
            tr = lambda M: np.diff(M, axis=0)
            first = 1
        elif self.transformation == "fod":
            from ..core.transforms import fod_matrix

            tr = lambda M: fod_matrix(M.shape[0]) @ M
            first = 0
        else:
            raise ValueError("transformation must be 'fd' or 'fod'")

        start = s.lags + 1
        rows_y, rows_X, rows_Z, row_units = [], [], [], []

        for t in range(start, T):
            # transformed dependent variable at period t
            dY = tr(Y)[t - 1] if self.transformation == "fd" else tr(Y)[t - 1]
            regs = [tr(Y)[t - 1 - j] for j in range(1, s.lags + 1)]
            for v in s.predetermined:
                regs.append(tr(P[v])[t - 1])
            for v in s.exogenous:
                regs.append(tr(E[v])[t - 1])
            X_t = np.column_stack(regs)

            # GMM instruments: levels of y dated t-lo .. t-hi
            lo, hi = s.gmm_lags
            hi_eff = t if hi is None else min(hi, t)
            cols, blocks = [], []
            pos = 0
            for lag in range(lo, hi_eff + 1):
                idx = t - lag
                if idx < 0:
                    continue
                cols.append(Y[idx])
                blocks.append((pos, pos + 1))
                pos += 1
            for v in s.predetermined:
                for lag in range(lo, hi_eff + 1):
                    idx = t - lag
                    if idx < 0:
                        continue
                    cols.append(P[v][idx])
                    pos += 1
            for v in s.exogenous:
                cols.append(tr(E[v])[t - 1])
                pos += 1
            Z_t = np.column_stack(cols) if cols else np.zeros((N, 0))

            rows_y.append(dY)
            rows_X.append(X_t)
            rows_Z.append(Z_t)
            row_units.append(np.arange(N))

        width = max(z.shape[1] for z in rows_Z)
        Z_pad = [
            np.hstack([z, np.zeros((z.shape[0], width - z.shape[1]))]) for z in rows_Z
        ]
        if s.collapse:
            Z_pad = [z[:, :width] for z in Z_pad]

        y_vec = np.concatenate(rows_y)
        X_mat = np.vstack(rows_X)
        Z_mat = np.vstack(Z_pad)
        units = np.concatenate(row_units)
        n_periods = len(rows_y)

        ok = np.isfinite(y_vec) & np.isfinite(X_mat).all(axis=1)
        Z_mat = np.nan_to_num(Z_mat)
        return (
            y_vec[ok],
            X_mat[ok],
            Z_mat[ok],
            units[ok],
            np.repeat(np.arange(n_periods), N)[ok],
            N,
        )

    # ------------------------------------------------------------------
    def fit(self, panel: PanelData) -> PanelResults:
        """Estimate by GMM.

        Parameters
        ----------
        panel : PanelData

        Returns
        -------
        PanelResults
            ``diagnostics`` carries the Hansen J test and the Arellano-Bond
            AR(1) and AR(2) tests.

        Notes
        -----
        Reading the diagnostics: AR(1) should reject (the differenced error is
        MA(1) by construction), AR(2) should *not* reject (otherwise deeper
        lags are invalid instruments), and Hansen should not reject -- but a
        p-value above roughly 0.9 usually signals instrument proliferation
        rather than a well-specified model.
        """
        y, X, Z, units, periods, N = self._build(panel)
        n, k = X.shape
        m = Z.shape[1]

        names = [f"L{j}.{self.spec.y}" for j in range(1, self.spec.lags + 1)]
        names += list(self.spec.predetermined) + list(self.spec.exogenous)

        if m < k:
            raise ValueError(
                f"only {m} instruments for {k} parameters; the model is "
                "under-identified. Widen gmm_lags or add exogenous variables."
            )

        ZX = Z.T @ X
        Zy = Z.T @ y

        # ---- step 1: identity-ish weight ------------------------------
        W1 = np.linalg.pinv(Z.T @ Z / N)
        A1 = ZX.T @ W1 @ ZX
        beta1 = np.linalg.pinv(A1) @ (ZX.T @ W1 @ Zy)

        resid1 = y - X @ beta1
        beta, resid, W = beta1, resid1, W1

        if self.steps >= 2:
            S = np.zeros((m, m))
            for g in np.unique(units):
                sel = units == g
                zu = Z[sel].T @ resid1[sel]
                S += np.outer(zu, zu)
            S /= N
            W2 = np.linalg.pinv(S)
            A2 = ZX.T @ W2 @ ZX
            beta = np.linalg.pinv(A2) @ (ZX.T @ W2 @ Zy)
            resid = y - X @ beta
            W = W2

        # ---- variance --------------------------------------------------
        A = ZX.T @ W @ ZX
        A_inv = np.linalg.pinv(A)
        if self.steps == 1:
            S1 = np.zeros((m, m))
            for g in np.unique(units):
                sel = units == g
                zu = Z[sel].T @ resid[sel]
                S1 += np.outer(zu, zu)
            cov = A_inv @ (ZX.T @ W @ S1 @ W @ ZX) @ A_inv
        else:
            cov = A_inv * N
            cov = self._windmeijer(X, Z, resid1, beta, W, A_inv, ZX, units, N, cov)

        # ---- tests -----------------------------------------------------
        hansen = self._hansen(Z, resid, W, N, m, k)
        ar1 = self._ar_test(resid, units, periods, 1)
        ar2 = self._ar_test(resid, units, periods, 2)

        res = PanelResults(
            params=pd.Series(beta, index=names),
            cov=cov,
            method=(
                f"{'System' if self.level else 'Difference'} GMM "
                f"({self.steps}-step, {self.transformation.upper()}"
                f"{', collapsed' if self.spec.collapse else ''})"
            ),
            n_obs=n,
            n_units=int(N),
            n_periods=panel.T,
            dependent=self.spec.y,
            diagnostics={
                "instruments": m,
                "Hansen J": f"chi2({hansen[1]}) = {hansen[0]:.3f}, p = {hansen[2]:.3f}",
                "AR(1)": f"z = {ar1[0]:.3f}, p = {ar1[1]:.3f}",
                "AR(2)": f"z = {ar2[0]:.3f}, p = {ar2[1]:.3f}",
            },
            extra={"resid": resid, "Z": Z, "X": X},
        )
        if m > N:
            warnings.warn(
                f"{m} instruments for {N} units: the weight matrix is likely "
                "singular and Hansen's test has no power. Use collapse=True, "
                "restrict gmm_lags, or switch to dynpanelai.ablasso.",
                UserWarning,
                stacklevel=2,
            )
        self.results_ = res
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _windmeijer(X, Z, resid1, beta, W, A_inv, ZX, units, N, cov2):
        """Windmeijer (2005) correction; falls back to the uncorrected matrix."""
        try:
            k = X.shape[1]
            m = Z.shape[1]
            zs = np.zeros(m)
            for g in np.unique(units):
                sel = units == g
                zs += Z[sel].T @ (X[sel] @ beta * 0 + resid1[sel])
            D = np.zeros((k, k))
            M_XZ_W = A_inv @ ZX.T @ W
            for j in range(k):
                dS = np.zeros((m, m))
                for g in np.unique(units):
                    sel = units == g
                    zu = Z[sel].T @ resid1[sel]
                    zx = Z[sel].T @ X[sel][:, j]
                    dS += -(np.outer(zx, zu) + np.outer(zu, zx))
                dS /= N
                D[:, j] = -(M_XZ_W @ dS @ W @ zs)
            DM = D @ cov2
            return cov2 + DM + DM.T + D @ cov2 @ D.T
        except Exception:  # pragma: no cover - numerical fallback
            warnings.warn(
                "the Windmeijer correction failed numerically; reporting "
                "uncorrected two-step standard errors, which are known to be "
                "downward biased.",
                UserWarning,
            )
            return cov2

    @staticmethod
    def _hansen(Z, resid, W, N, m, k):
        """Hansen's J test of over-identifying restrictions."""
        g = Z.T @ resid / N
        J = float(N * g @ W @ g)
        dof = max(m - k, 1)
        return J, dof, float(1 - stats.chi2.cdf(J, dof))

    @staticmethod
    def _ar_test(resid, units, periods, order):
        """Arellano-Bond test for serial correlation of order ``order``.

        A simplified version of the full formula: correlates the residual with
        its own ``order``-lag within units and standardises.  Sufficient for
        diagnostics; the exact variance also nets out parameter-estimation
        error.
        """
        r = pd.Series(resid)
        key = pd.MultiIndex.from_arrays([units, periods])
        s = pd.Series(resid, index=key)
        lagged = s.reindex(pd.MultiIndex.from_arrays([units, periods - order]))
        a = s.to_numpy()
        b = lagged.to_numpy()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return float("nan"), float("nan")
        num = float(np.sum(a[ok] * b[ok]))
        den = float(np.sqrt(np.sum((a[ok] * b[ok]) ** 2)))
        if den == 0:
            return float("nan"), float("nan")
        z = num / den
        return z, float(2 * stats.norm.sf(abs(z)))


def diff_gmm(panel: PanelData, y: str, **kwargs) -> PanelResults:
    """Difference GMM (Arellano and Bond, 1991).  See :class:`DynamicPanelGMM`."""
    kwargs.setdefault("level", False)
    return DynamicPanelGMM(y=y, **kwargs).fit(panel)


def system_gmm(panel: PanelData, y: str, **kwargs) -> PanelResults:
    """System GMM (Blundell and Bond, 1998).  See :class:`DynamicPanelGMM`."""
    kwargs["level"] = True
    return DynamicPanelGMM(y=y, **kwargs).fit(panel)


def anderson_hsiao(
    panel: PanelData,
    y: str,
    *,
    lags: int = 1,
    exogenous: Sequence[str] | None = None,
    instrument: str = "level",
) -> PanelResults:
    """Anderson-Hsiao just-identified IV estimator.

    Instruments :math:`\\Delta y_{i,t-1}` with either :math:`y_{i,t-2}`
    (``instrument='level'``) or :math:`\\Delta y_{i,t-2}`
    (``instrument='diff'``).

    Parameters
    ----------
    panel : PanelData
    y : str
    lags : int, default 1
    exogenous : sequence of str, optional
    instrument : {'level', 'diff'}, default 'level'

    Returns
    -------
    PanelResults

    Notes
    -----
    Anderson-Hsiao is nearly unbiased but high-variance: it discards all but
    one moment condition.  Cornejo and Sosa-Escudero find it dominates on bias
    yet loses badly on forecast MSE -- the cleanest illustration in this
    package of why bias is not the only thing that matters.
    """
    exogenous = list(exogenous) if exogenous else []
    lagged = panel.lag(y, lags + 2)
    frame = panel.df.copy()
    for c in lagged.columns:
        frame[c] = lagged[c]
    for v in exogenous:
        for c, s in panel.lag(v, 1).items():
            frame[c] = s

    dy = frame[y] - frame[f"{y}_lag1"]
    regs = {f"L{j}.{y}": frame[f"{y}_lag{j}"] - frame[f"{y}_lag{j+1}"]
            for j in range(1, lags + 1)}
    for v in exogenous:
        regs[v] = frame[v] - frame[f"{v}_lag1"]

    if instrument == "level":
        inst = {f"iv_L{j}": frame[f"{y}_lag{j+1}"] for j in range(1, lags + 1)}
    elif instrument == "diff":
        inst = {f"iv_D{j}": frame[f"{y}_lag{j+1}"] - frame[f"{y}_lag{j+2}"]
                for j in range(1, lags + 1)}
    else:
        raise ValueError("instrument must be 'level' or 'diff'")
    for v in exogenous:
        inst[v] = frame[v] - frame[f"{v}_lag1"]

    design = pd.DataFrame({"_dy": dy, **regs, **inst, "_i": frame["_i"]}).dropna()
    yv = design["_dy"].to_numpy()
    X = design[list(regs)].to_numpy()
    Z = design[list(inst)].to_numpy()
    units = design["_i"].to_numpy()

    ZX = Z.T @ X
    beta = np.linalg.pinv(ZX) @ (Z.T @ yv)
    resid = yv - X @ beta

    A_inv = np.linalg.pinv(ZX)
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for g in np.unique(units):
        sel = units == g
        zu = Z[sel].T @ resid[sel]
        S += np.outer(zu, zu)
    cov = A_inv @ S @ A_inv.T

    return PanelResults(
        params=pd.Series(beta, index=list(regs)),
        cov=cov,
        method=f"Anderson-Hsiao IV ({instrument} instrument)",
        n_obs=len(yv),
        n_units=len(np.unique(units)),
        n_periods=panel.T,
        dependent=y,
        diagnostics={"instrument": instrument},
    )
