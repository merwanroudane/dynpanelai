"""Publication-quality tables.

Produces the layout economics journals expect: coefficients with standard
errors in parentheses beneath, significance stars, sample dimensions in the
panel footer, and diagnostics in the table notes.  Output targets are LaTeX
(``booktabs``), Markdown, and plain text.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.results import PanelResults, stars

__all__ = [
    "results_to_latex",
    "comparison_table",
    "comparison_to_latex",
    "monte_carlo_table",
]


def _fmt(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def results_to_latex(
    res: PanelResults,
    *,
    alpha: float = 0.05,
    digits: int = 3,
    caption: str | None = None,
    label: str | None = None,
    se_below: bool = True,
) -> str:
    """Render a single :class:`PanelResults` as a LaTeX table.

    Parameters
    ----------
    res : PanelResults
    alpha : float, default 0.05
    digits : int, default 3
    caption, label : str, optional
    se_below : bool, default True
        Standard errors under the coefficients rather than in a second column.

    Returns
    -------
    str
        LaTeX source.  Requires ``\\usepackage{booktabs}``.

    Examples
    --------
    >>> print(results_to_latex(res, caption="Main results"))  # doctest: +SKIP
    """
    tab = res.table(alpha)
    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    if se_below:
        lines += ["\\begin{tabular}{lc}", "\\toprule",
                  f" & {res.dependent or 'Estimate'} \\\\", "\\midrule"]
        for name, row in tab.iterrows():
            lines.append(
                f"{name} & {_fmt(row['coef'], digits)}$^{{{row['sig']}}}$ \\\\"
            )
            lines.append(f" & ({_fmt(row['std_err'], digits)}) \\\\")
    else:
        lines += ["\\begin{tabular}{lccc}", "\\toprule",
                  " & Coef. & Std.\\ err. & $p$ \\\\", "\\midrule"]
        for name, row in tab.iterrows():
            lines.append(
                f"{name} & {_fmt(row['coef'], digits)}$^{{{row['sig']}}}$ & "
                f"{_fmt(row['std_err'], digits)} & {_fmt(row['p_value'], digits)} \\\\"
            )

    lines.append("\\midrule")
    if res.n_obs is not None:
        lines.append(f"Observations & {res.n_obs:,} \\\\")
    if res.n_units is not None:
        lines.append(f"Units & {res.n_units:,} \\\\")
    if res.n_periods is not None:
        lines.append(f"Periods & {res.n_periods:,} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    notes = [f"\\emph{{Notes:}} {res.method}."]
    if res.diagnostics:
        diag = "; ".join(f"{k}: {v}" for k, v in res.diagnostics.items()
                         if not isinstance(v, dict))
        if diag:
            notes.append(diag + ".")
    notes.append(
        "Standard errors in parentheses. "
        "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
    )
    lines.append(
        "\\begin{minipage}{\\linewidth}\\footnotesize " + " ".join(notes) + "\\end{minipage}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def comparison_table(
    results: Mapping[str, PanelResults],
    *,
    params: Sequence[str] | None = None,
    digits: int = 4,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Line several estimators up side by side.

    This is the table that carries the argument in every one of the papers
    implemented here: the same coefficient, estimated many ways, so the reader
    can see how much the method matters.

    Parameters
    ----------
    results : mapping of str to PanelResults
        Column label to fitted result.
    params : sequence of str, optional
        Coefficients to show.  Defaults to the union across estimators, in
        first-appearance order.
    digits : int, default 4
    alpha : float, default 0.05

    Returns
    -------
    pandas.DataFrame
        Rows alternate coefficient and ``(standard error)``, with a footer
        block carrying the sample dimensions.

    Examples
    --------
    >>> comparison_table({"FE": fe_res, "GMM": gmm_res})   # doctest: +SKIP
    """
    if params is None:
        seen: dict[str, None] = {}
        for r in results.values():
            for name in r.params.index:
                seen.setdefault(str(name), None)
        params = list(seen)

    rows: dict[str, dict[str, str]] = {}
    for name in params:
        rows[name] = {}
        rows[f"  ({name})"] = {}
    footer = {"Observations": {}, "Units": {}, "Method": {}}

    for label, r in results.items():
        tab = r.table(alpha)
        idx = {str(i): i for i in tab.index}
        for name in params:
            if name in idx:
                row = tab.loc[idx[name]]
                rows[name][label] = f"{row['coef']:.{digits}f}{row['sig']}"
                rows[f"  ({name})"][label] = f"({row['std_err']:.{digits}f})"
            else:
                rows[name][label] = ""
                rows[f"  ({name})"][label] = ""
        footer["Observations"][label] = f"{r.n_obs:,}" if r.n_obs else ""
        footer["Units"][label] = f"{r.n_units:,}" if r.n_units else ""
        footer["Method"][label] = r.method

    body = pd.DataFrame(rows).T
    foot = pd.DataFrame(footer).T
    out = pd.concat([body, foot])
    out.index.name = ""
    return out.fillna("")


def comparison_to_latex(
    results: Mapping[str, PanelResults],
    *,
    params: Sequence[str] | None = None,
    digits: int = 4,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """LaTeX version of :func:`comparison_table`.

    Parameters
    ----------
    results : mapping of str to PanelResults
    params : sequence of str, optional
    digits : int, default 4
    caption, label : str, optional

    Returns
    -------
    str
    """
    tab = comparison_table(results, params=params, digits=digits)
    tab = tab.drop(index="Method", errors="ignore")
    cols = list(tab.columns)

    lines = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for name, row in tab.iterrows():
        if name in ("Observations", "Units"):
            continue
        cells = " & ".join(str(v) for v in row.tolist())
        lines.append(f"{name} & {cells} \\\\")
    lines.append("\\midrule")
    for name in ("Observations", "Units"):
        if name in tab.index:
            cells = " & ".join(str(v) for v in tab.loc[name].tolist())
            lines.append(f"{name} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    lines.append(
        "\\begin{minipage}{\\linewidth}\\footnotesize \\emph{Notes:} "
        "Standard errors in parentheses, clustered by unit. "
        "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
        "\\end{minipage}"
    )
    lines.append("\\end{table}")
    return "\n".join(lines)


def monte_carlo_table(
    estimates: Mapping[str, np.ndarray],
    truth: float,
    *,
    ses: Mapping[str, np.ndarray] | None = None,
    digits: int = 4,
) -> pd.DataFrame:
    """Summarise a Monte Carlo experiment.

    Parameters
    ----------
    estimates : mapping of str to ndarray
        Estimator name to the vector of replication estimates.
    truth : float
        The true parameter value.
    ses : mapping of str to ndarray, optional
        Replication standard errors.  Supplying them adds mean SE and 95%
        coverage, which is what actually distinguishes a well-behaved
        estimator from a merely unbiased one.
    digits : int, default 4

    Returns
    -------
    pandas.DataFrame
        Columns ``bias``, ``sd``, ``rmse``, ``mae``, and where available
        ``mean_se``, ``coverage``, ``mcse_bias``.

    Examples
    --------
    >>> import numpy as np
    >>> est = {"A": np.random.default_rng(0).normal(1.0, 0.1, 500)}
    >>> monte_carlo_table(est, truth=1.0).columns.tolist()[:3]
    ['reps', 'bias', 'sd']
    """
    rows = {}
    for name, vals in estimates.items():
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        bias = float(np.mean(v) - truth)
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        rmse = float(np.sqrt(np.mean((v - truth) ** 2)))
        entry = {
            "reps": len(v),
            "bias": bias,
            "sd": sd,
            "rmse": rmse,
            "mae": float(np.mean(np.abs(v - truth))),
            "mcse_bias": sd / np.sqrt(len(v)) if np.isfinite(sd) else np.nan,
        }
        if ses is not None and name in ses:
            s = np.asarray(ses[name], dtype=float)[: len(v)]
            ok = np.isfinite(s)
            if ok.any():
                lo = v[ok] - 1.96 * s[ok]
                hi = v[ok] + 1.96 * s[ok]
                entry["mean_se"] = float(np.mean(s[ok]))
                entry["coverage"] = float(np.mean((lo <= truth) & (truth <= hi)))
        rows[name] = entry
    return pd.DataFrame(rows).T.round(digits)
