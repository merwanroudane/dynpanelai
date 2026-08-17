"""Shared results container.

Every estimator in ``dynpanelai`` returns a :class:`PanelResults`.  A single
container means one ``summary()`` you learn once, one ``to_latex()`` that
produces the same house style everywhere, and the ability to line several
estimators up side by side in a comparison table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["PanelResults", "stars"]


def stars(p: float, levels: Sequence[float] = (0.01, 0.05, 0.10)) -> str:
    """Return significance stars for a p-value.

    Parameters
    ----------
    p : float
    levels : sequence of float, default ``(0.01, 0.05, 0.10)``
        Thresholds for ``***``, ``**``, ``*``.

    Returns
    -------
    str

    Examples
    --------
    >>> stars(0.004)
    '***'
    >>> stars(0.3)
    ''
    """
    if not np.isfinite(p):
        return ""
    for n, level in enumerate(levels):
        if p < level:
            return "*" * (len(levels) - n)
    return ""


@dataclass
class PanelResults:
    """Estimation output for a dynamic panel model.

    Parameters
    ----------
    params : pandas.Series
        Point estimates, indexed by coefficient name.
    cov : ndarray, optional
        Estimated variance matrix, ordered as ``params``.
    method : str
        Human-readable estimator name, used as the table title.
    n_obs, n_units, n_periods : int, optional
        Sample dimensions reported in the header.
    diagnostics : mapping, optional
        Specification tests and other scalars, printed beneath the table.
    extra : mapping, optional
        Anything an individual estimator wants to carry along (fold-level
        estimates, selected instruments, trimming reports, ...).

    Attributes
    ----------
    bse : pandas.Series
        Standard errors, derived from the diagonal of ``cov``.
    tvalues, pvalues : pandas.Series
    """

    params: pd.Series
    cov: np.ndarray | None = None
    method: str = "Panel estimator"
    n_obs: int | None = None
    n_units: int | None = None
    n_periods: int | None = None
    dependent: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    use_t: bool = False
    df_resid: int | None = None

    # ------------------------------------------------------------------
    @property
    def bse(self) -> pd.Series:
        """Standard errors."""
        if self.cov is None:
            return pd.Series(np.nan, index=self.params.index, name="std_err")
        se = np.sqrt(np.clip(np.diag(np.atleast_2d(self.cov)), 0.0, np.inf))
        return pd.Series(se, index=self.params.index, name="std_err")

    @property
    def tvalues(self) -> pd.Series:
        """Coefficient divided by its standard error."""
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.params.to_numpy() / self.bse.to_numpy()
        return pd.Series(t, index=self.params.index, name="statistic")

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided p-values (normal, or t when ``use_t``)."""
        t = self.tvalues.to_numpy()
        if self.use_t and self.df_resid:
            p = 2 * stats.t.sf(np.abs(t), df=self.df_resid)
        else:
            p = 2 * stats.norm.sf(np.abs(t))
        return pd.Series(p, index=self.params.index, name="p_value")

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Two-sided confidence intervals.

        Parameters
        ----------
        alpha : float, default 0.05

        Returns
        -------
        pandas.DataFrame
            Columns ``lower`` and ``upper``.
        """
        if self.use_t and self.df_resid:
            crit = stats.t.ppf(1 - alpha / 2, df=self.df_resid)
        else:
            crit = stats.norm.ppf(1 - alpha / 2)
        se = self.bse.to_numpy()
        return pd.DataFrame(
            {
                "lower": self.params.to_numpy() - crit * se,
                "upper": self.params.to_numpy() + crit * se,
            },
            index=self.params.index,
        )

    # ------------------------------------------------------------------
    def table(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return the coefficient table as a tidy DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns: ``coef``, ``std_err``, ``statistic``, ``p_value``,
            ``ci_lower``, ``ci_upper``, ``sig``.
        """
        ci = self.conf_int(alpha)
        out = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "statistic": self.tvalues,
                "p_value": self.pvalues,
                "ci_lower": ci["lower"],
                "ci_upper": ci["upper"],
            }
        )
        out["sig"] = [stars(p) for p in out["p_value"]]
        return out

    def summary(self, alpha: float = 0.05, digits: int = 4) -> str:
        """Render a Stata-style text summary.

        Parameters
        ----------
        alpha : float, default 0.05
        digits : int, default 4

        Returns
        -------
        str

        Examples
        --------
        >>> print(res.summary())
        =============================================================
        Arellano-Bond LASSO (FOD, cross-fitted)
        ...
        """
        tab = self.table(alpha)
        width = 78
        lines = ["=" * width, self.method]
        if self.dependent:
            lines.append(f"Dependent variable: {self.dependent}")

        header_bits = []
        if self.n_obs is not None:
            header_bits.append(f"Observations = {self.n_obs:,}")
        if self.n_units is not None:
            header_bits.append(f"Units = {self.n_units:,}")
        if self.n_periods is not None:
            header_bits.append(f"Periods = {self.n_periods:,}")
        if header_bits:
            lines.append("   ".join(header_bits))
        lines.append("-" * width)

        name_w = max(12, min(28, max((len(str(i)) for i in tab.index), default=12) + 2))
        fmt = f"{{:<{name_w}}}{{:>12}}{{:>12}}{{:>10}}{{:>10}}  {{:<3}}"
        lines.append(fmt.format("", "coef", "std.err.", "z", "P>|z|", ""))
        lines.append("-" * width)
        for name, row in tab.iterrows():
            lines.append(
                fmt.format(
                    str(name)[: name_w - 1],
                    f"{row['coef']:.{digits}f}",
                    f"{row['std_err']:.{digits}f}",
                    f"{row['statistic']:.3f}",
                    f"{row['p_value']:.3f}",
                    row["sig"],
                )
            )
        lines.append("-" * width)
        lines.append(f"Signif.: *** p<0.01, ** p<0.05, * p<0.10   ({100*(1-alpha):.0f}% CI)")

        if self.diagnostics:
            lines.append("-" * width)
            for key, value in self.diagnostics.items():
                lines.append(f"{key}: {_fmt_diag(value, digits)}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()

    def to_latex(
        self,
        alpha: float = 0.05,
        digits: int = 3,
        caption: str | None = None,
        label: str | None = None,
        se_below: bool = True,
    ) -> str:
        """Export a publication-ready LaTeX table.

        Produces a ``booktabs`` table with standard errors in parentheses
        beneath each coefficient, significance stars, and diagnostics in the
        table notes -- the layout used by most economics journals.

        Parameters
        ----------
        alpha : float, default 0.05
        digits : int, default 3
        caption, label : str, optional
        se_below : bool, default True
            Put standard errors under the coefficients rather than beside them.

        Returns
        -------
        str
            LaTeX source.  Requires ``\\usepackage{booktabs}``.
        """
        from ..report.tables import results_to_latex

        return results_to_latex(
            self,
            alpha=alpha,
            digits=digits,
            caption=caption,
            label=label,
            se_below=se_below,
        )

    def long_run(self, coef: str, lag_coefs: Sequence[str]) -> tuple[float, float]:
        """Long-run effect and its delta-method standard error.

        For a dynamic model with lags :math:`\\rho_1,\\dots,\\rho_L` of the
        outcome, the long-run effect of a covariate with short-run coefficient
        :math:`\\theta` is

        .. math::
            \\theta^{LR} = \\frac{\\theta}{1 - \\sum_{j} \\rho_j}.

        The standard error follows from the Jacobian
        :math:`(1-\\sum\\rho)^{-1}\\,(1, \\theta^{LR},\\dots,\\theta^{LR})`.

        Parameters
        ----------
        coef : str
            Name of the short-run coefficient.
        lag_coefs : sequence of str
            Names of the lagged-dependent-variable coefficients.

        Returns
        -------
        (float, float)
            The long-run effect and its standard error.

        Raises
        ------
        ValueError
            If the process is not stable (``sum(rho) >= 1``) or ``cov`` is None.

        Examples
        --------
        >>> lr, se = res.long_run("school", ["L1.logdc", "L2.logdc"])
        """
        if self.cov is None:
            raise ValueError("long-run standard errors require a covariance matrix")
        names = list(self.params.index)
        if coef not in names:
            raise KeyError(f"{coef!r} is not an estimated coefficient")
        missing = [c for c in lag_coefs if c not in names]
        if missing:
            raise KeyError(f"lag coefficients not found: {missing}")

        rho_sum = float(self.params[list(lag_coefs)].sum())
        denom = 1.0 - rho_sum
        if abs(denom) < 1e-10:
            raise ValueError(
                "the autoregressive process is at or beyond the unit root "
                f"(sum of lag coefficients = {rho_sum:.4f}); the long-run "
                "effect is not defined"
            )

        theta = float(self.params[coef])
        lr = theta / denom

        idx = [names.index(coef)] + [names.index(c) for c in lag_coefs]
        jac = np.concatenate(([1.0], np.full(len(lag_coefs), lr))) / denom
        sub = np.atleast_2d(self.cov)[np.ix_(idx, idx)]
        se = float(np.sqrt(max(jac @ sub @ jac, 0.0)))
        return lr, se


def _fmt_diag(value: Any, digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt_diag(v, digits)}" for k, v in value.items())
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_fmt_diag(v, digits) for v in value)
    return str(value)
