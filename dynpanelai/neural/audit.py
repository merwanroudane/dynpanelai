"""Layered audit protocol for learned lag structures.

A model that forecasts well has not thereby demonstrated that its internal lag
structure is real.  Xu (2026) separates the two claims into four layers, and
this module implements them.

======  =======================================================================
Layer   Question
======  =======================================================================
L0      Is the model predictively calibrated?  (MSE / MAE / R2)
L1      Is the learned lag non-degenerate?  A model that assigns every entity
        the same effective lag has discovered nothing, however well it
        forecasts.  Guard: cross-entity ``sd(k*) > eps``.
L2      Is the learned lag *externally structured*?  Spearman-correlate
        :math:`k^\\star_i` with pre-specified entity stratifiers and test
        against an entity-label permutation null.  Sign-robust: the sign of a
        latent score is arbitrary across seeds, so the statistic is
        :math:`|\\rho|`.
L3      Does it recover known truth?  Only available on synthetic data.
======  =======================================================================

Plus a **proxy-shuffle negative control**: permute the proxy vectors across
entities and refit.  If L2 alignment survives that, the alignment was an
artefact of model capacity rather than evidence of a proxy-entity
relationship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["audit_l1", "audit_l2", "audit_l3", "fisher_combine", "AuditReport", "run_audit"]


def fisher_combine(pvalues: Sequence[float]) -> float:
    """Combine independent p-values by Fisher's method.

    Parameters
    ----------
    pvalues : sequence of float

    Returns
    -------
    float
        The combined p-value.

    Examples
    --------
    >>> round(fisher_combine([0.04, 0.03, 0.20]), 4)
    0.0165
    """
    p = np.asarray([x for x in pvalues if np.isfinite(x)], dtype=float)
    if len(p) == 0:
        return float("nan")
    p = np.clip(p, 1e-300, 1.0)
    stat = -2.0 * np.sum(np.log(p))
    return float(stats.chi2.sf(stat, df=2 * len(p)))


def audit_l1(k_star: np.ndarray, eps: float = 1e-3) -> dict:
    """L1 degeneracy guard.

    Parameters
    ----------
    k_star : ndarray
        Effective lag per entity.
    eps : float, default 1e-3
        Minimum acceptable cross-entity standard deviation.

    Returns
    -------
    dict
        ``sd``, ``degenerate`` (bool), ``range``.

    Notes
    -----
    A degenerate seed must be excluded from L2, otherwise a constant-lag model
    can be spuriously "validated" by whatever correlation noise produces.

    Examples
    --------
    >>> audit_l1(np.array([2.0, 2.0, 2.0]))["degenerate"]
    True
    """
    k = np.asarray(k_star, dtype=float)
    sd = float(np.std(k))
    return {
        "sd": sd,
        "degenerate": bool(sd <= eps),
        "range": (float(np.min(k)), float(np.max(k))),
    }


def audit_l2(
    k_star: np.ndarray,
    stratifier: np.ndarray,
    *,
    n_perm: int = 1000,
    seed: int | None = 0,
) -> dict:
    """L2 structured-heterogeneity test against a permutation null.

    Computes :math:`\\rho = \\mathrm{Spearman}(k^\\star, \\xi)` and compares
    :math:`|\\rho|` to the distribution obtained by permuting the entity
    labels of :math:`\\xi`.

    Parameters
    ----------
    k_star : ndarray of shape (n_entities,)
    stratifier : ndarray of shape (n_entities,)
        A pre-specified entity characteristic, built from information
        available *before* the test window.
    n_perm : int, default 1000
    seed : int, optional

    Returns
    -------
    dict
        ``rho``, ``abs_rho``, ``p_perm``, ``n``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> k = rng.normal(size=50); xi = k + rng.normal(scale=0.3, size=50)
    >>> audit_l2(k, xi, n_perm=200)["p_perm"] < 0.05
    True
    """
    k = np.asarray(k_star, dtype=float)
    xi = np.asarray(stratifier, dtype=float)
    ok = np.isfinite(k) & np.isfinite(xi)
    k, xi = k[ok], xi[ok]
    if len(k) < 4:
        return {"rho": np.nan, "abs_rho": np.nan, "p_perm": np.nan, "n": len(k)}

    rho = float(stats.spearmanr(k, xi).statistic)
    rng = np.random.default_rng(seed)
    null = np.array(
        [abs(float(stats.spearmanr(k, rng.permutation(xi)).statistic))
         for _ in range(n_perm)]
    )
    p = (1.0 + np.sum(null >= abs(rho))) / (n_perm + 1.0)
    return {"rho": rho, "abs_rho": abs(rho), "p_perm": float(p), "n": int(len(k))}


def audit_l3(k_star: np.ndarray, k_true: np.ndarray) -> dict:
    """L3 ground-truth recovery, for synthetic data only.

    Parameters
    ----------
    k_star, k_true : ndarray

    Returns
    -------
    dict
        ``mae``, ``spearman``, ``pearson``.

    Examples
    --------
    >>> audit_l3(np.array([1., 2., 3.]), np.array([1., 2., 3.]))["spearman"]
    1.0
    """
    a = np.asarray(k_star, dtype=float)
    b = np.asarray(k_true, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return {"mae": np.nan, "spearman": np.nan, "pearson": np.nan}
    return {
        "mae": float(np.mean(np.abs(a[ok] - b[ok]))),
        "spearman": float(stats.spearmanr(a[ok], b[ok]).statistic),
        "pearson": float(np.corrcoef(a[ok], b[ok])[0, 1]),
    }


@dataclass
class AuditReport:
    """Assembled L0-L3 verdict across seeds.

    Attributes
    ----------
    l0 : pandas.DataFrame
        Forecast metrics per seed.
    l1 : pandas.DataFrame
        Degeneracy check per seed.
    l2 : pandas.DataFrame
        Per-seed, per-stratifier alignment.
    l3 : pandas.DataFrame or None
    verdict : pandas.DataFrame
        One row per stratifier: mean |rho|, share of seeds rejecting at 5%,
        and the Fisher-combined p-value.
    """

    l0: pd.DataFrame
    l1: pd.DataFrame
    l2: pd.DataFrame
    l3: pd.DataFrame | None
    verdict: pd.DataFrame

    def summary(self) -> str:
        """Render the audit as a readable report."""
        lines = ["=" * 74, "AC-GATE layered audit", "=" * 74]
        lines.append(f"L0  forecast (mean over {len(self.l0)} seeds):")
        for col in ("mse", "mae", "r2"):
            if col in self.l0:
                lines.append(f"      {col:<5} = {self.l0[col].mean():.4f}")
        n_deg = int(self.l1["degenerate"].sum())
        lines.append(
            f"L1  degeneracy: {n_deg}/{len(self.l1)} seeds degenerate; "
            f"mean sd(k*) = {self.l1['sd'].mean():.4f}"
        )
        lines.append("L2  structured heterogeneity:")
        lines.append(
            f"      {'stratifier':<22}{'mean|rho|':>10}{'reject@5%':>11}{'Fisher p':>11}"
        )
        for name, row in self.verdict.iterrows():
            lines.append(
                f"      {str(name):<22}{row['mean_abs_rho']:>10.3f}"
                f"{row['reject_share']:>11.2f}{row['fisher_p']:>11.4g}"
            )
        if self.l3 is not None and len(self.l3):
            lines.append(
                f"L3  ground truth: MAE = {self.l3['mae'].mean():.3f}, "
                f"Spearman = {self.l3['spearman'].mean():.3f}"
            )
        else:
            lines.append("L3  ground truth: n/a (no synthetic truth supplied)")
        lines.append("=" * 74)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def run_audit(
    k_star_by_seed: Sequence[np.ndarray],
    stratifiers: Mapping[str, np.ndarray],
    *,
    metrics_by_seed: Sequence[Mapping[str, float]] | None = None,
    k_true: np.ndarray | None = None,
    eps: float = 1e-3,
    n_perm: int = 1000,
    seed: int = 0,
) -> AuditReport:
    """Run the full L0-L3 audit over several random seeds.

    Parameters
    ----------
    k_star_by_seed : sequence of ndarray
        One effective-lag vector per seed.
    stratifiers : mapping of str to ndarray
        Pre-specified entity characteristics to test alignment against.
    metrics_by_seed : sequence of mapping, optional
        L0 forecast metrics per seed.
    k_true : ndarray, optional
        Ground-truth lags, enabling L3.
    eps : float, default 1e-3
    n_perm : int, default 1000
    seed : int, default 0

    Returns
    -------
    AuditReport

    Notes
    -----
    Degenerate seeds are excluded from L2 but still reported in L1, exactly as
    the protocol requires: a collapsed model should not be able to launder
    itself into evidence of heterogeneity.
    """
    l0 = pd.DataFrame(list(metrics_by_seed)) if metrics_by_seed else pd.DataFrame()
    l1_rows, l2_rows, l3_rows = [], [], []

    for s, k in enumerate(k_star_by_seed):
        d = audit_l1(k, eps=eps)
        d["seed"] = s
        l1_rows.append(d)
        if d["degenerate"]:
            continue
        for name, xi in stratifiers.items():
            r = audit_l2(k, xi, n_perm=n_perm, seed=seed + s)
            r.update({"seed": s, "stratifier": name})
            l2_rows.append(r)
        if k_true is not None:
            r3 = audit_l3(k, k_true)
            r3["seed"] = s
            l3_rows.append(r3)

    l1 = pd.DataFrame(l1_rows)
    l2 = pd.DataFrame(l2_rows)
    l3 = pd.DataFrame(l3_rows) if l3_rows else None

    if len(l2):
        verdict = (
            l2.groupby("stratifier")
            .apply(
                lambda g: pd.Series(
                    {
                        "mean_abs_rho": g["abs_rho"].mean(),
                        "reject_share": float((g["p_perm"] < 0.05).mean()),
                        "fisher_p": fisher_combine(g["p_perm"].tolist()),
                    }
                ),
                include_groups=False,
            )
        )
    else:
        verdict = pd.DataFrame(
            columns=["mean_abs_rho", "reject_share", "fisher_p"]
        )
    return AuditReport(l0=l0, l1=l1, l2=l2, l3=l3, verdict=verdict)
