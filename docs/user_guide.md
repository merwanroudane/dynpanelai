# dynpanelai — User Guide

**How to write the code, step by step.**

This guide takes you from a CSV file to a finished journal table. Every code
block is runnable as written. Read it in order the first time; afterwards, use
the [syntax reference](syntax_reference.md) to look things up.

---

## Contents

1. [Install and check](#1-install-and-check)
2. [Step 1 — Load your data](#step-1--load-your-data)
3. [Step 2 — Build the panel](#step-2--build-the-panel)
4. [Step 3 — Inspect before you estimate](#step-3--inspect-before-you-estimate)
5. [Step 4 — Choose your method](#step-4--choose-your-method)
6. [Step 5 — Estimate](#step-5--estimate)
7. [Step 6 — Read the output](#step-6--read-the-output)
8. [Step 7 — Compare estimators](#step-7--compare-estimators)
9. [Step 8 — Long-run effects](#step-8--long-run-effects)
10. [Step 9 — Tables and figures](#step-9--tables-and-figures)
11. [Complete worked examples](#complete-worked-examples)
12. [Common mistakes](#common-mistakes)

---

## 1. Install and check

```bash
pip install dynpanelai[all]
```

```python
import dynpanelai as dp
print(dp.__version__)
```

---

## Step 1 — Load your data

`dynpanelai` wants **long format**: one row per unit-period.

```python
import pandas as pd

df = pd.read_csv("my_panel.csv")
print(df.head())
```

```
   firm  year      y      d     x1     x2
0     1  2000  1.243  0.512  0.883 -0.221
1     1  2001  1.401  0.489  0.912 -0.198
2     1  2002  1.588  0.631  0.874 -0.245
3     2  2000  2.014  0.204  1.203  0.412
4     2  2001  2.190  0.298  1.188  0.388
```

If your data is wide (one column per period), reshape first:

```python
df = wide.melt(id_vars="firm", var_name="year", value_name="y")
df["year"] = df["year"].astype(int)
```

Or start from one of the bundled panels:

```python
df = dp.datasets.load_abond_employment()   # 140 UK firms x 9 years
df = dp.datasets.load_covid_counties()     # 2,510 US counties x 32 weeks
```

---

## Step 2 — Build the panel

Everything starts with `PanelData`. It sorts, validates, and computes the
internal indices the estimators need.

```python
panel = dp.PanelData(df, unit="firm", time="year")
print(panel)
```

```
PanelData(N=140, T=9, obs=1031, unbalanced, unit='firm', time='year')
```

**What it checks for you:**

- duplicate `(unit, time)` pairs raise immediately, rather than silently
  corrupting your lags;
- gaps in time are detected, so a firm missing 2003 gets `NaN` for its 2004
  lag rather than silently borrowing 2002.

That second point is worth dwelling on. A plain `groupby().shift(1)` would
hand you the 2002 value and call it the 2003 lag. `PanelData.lag()` will not.

### Creating lags manually

Usually the estimators handle this. When you need them explicitly:

```python
lags = panel.lag("y", 2)              # y_lag1, y_lag2
lags = panel.lag(["y", "d"], [1, 4])  # specific lags only
panel2 = panel.with_lags("y", 2, dropna=True)   # append and burn in
```

---

## Step 3 — Inspect before you estimate

Two numbers decide which methods are even admissible.

```python
print(panel.summary())
```

```
units (N)                140
periods (T)                9
observations            1031
balanced               False
obs per unit (min)         7
obs per unit (mean)      7.4
obs per unit (max)         9
```

**Check 1 — is `T` long enough for machine learning?**

```python
import numpy as np
print("sqrt(N)/T =", np.sqrt(panel.N) / panel.T)
```

The DML and orthogonal-Lasso estimators need `sqrt(N)/T → 0`. At `T = 9` and
`N = 140` this is `1.31` — far too short. Use GMM here.

**Check 2 — will you have too many instruments?**

```python
T, N = panel.T, panel.N
m = T * (T - 1) // 2                 # rough count for difference GMM
print("m^2/(NT) =", m**2 / (N * T))
```

If this is much above 1, plain Arellano–Bond is biased and you want
`ablasso`. In the COVID application it is about 168.

---

## Step 4 — Choose your method

```
Is T short (under ~15)?
├── Yes → gmm.diff_gmm (collapse=True), or biascorr.*
└── No  → Many controls, or nonlinear ones?
          ├── No  → m²/(NT) large?
          │         ├── Yes → ablasso.ABLasso
          │         └── No  → gmm.diff_gmm
          └── Yes → What do you want?
                    ├── One treatment effect        → dml.DMLDynamicPanel
                    ├── Effects across many groups  → ortho.OrthogonalLasso
                    ├── All coefficients, uniformly → hdpanel.PanelLasso
                    ├── A forecast                  → shrink.PenalizedFE
                    └── Who responds when           → neural.ACGate
```

---

## Step 5 — Estimate

Every estimator follows the same shape. Two equivalent styles:

```python
# Functional
res = dp.diff_gmm(panel, y="n", lags=2, predetermined=["w"],
                  exogenous=["k"], collapse=True)

# Object-oriented — reusable, inspectable
est = dp.DynamicPanelGMM(y="n", lags=2, predetermined=["w"], exogenous=["k"])
res = est.fit(panel)
```

### The variable roles

Getting these right matters more than any tuning parameter.

| Role | Meaning | Argument |
|---|---|---|
| **Strictly exogenous** | uncorrelated with the error at *all* leads and lags | `exogenous=` |
| **Predetermined** | uncorrelated with current and future errors, but correlated with past ones | `predetermined=` |
| **Endogenous** | correlated with the current error | instrument it |

The lagged dependent variable is always predetermined — that is the whole
problem.

---

## Step 6 — Read the output

```python
print(res.summary())
```

```
==============================================================================
Difference GMM (2-step, FD, collapsed)
Dependent variable: n
Observations = 611   Units = 140   Periods = 9
------------------------------------------------------------------------------
                    coef    std.err.         z     P>|z|
------------------------------------------------------------------------------
L1.n              0.0769      0.1461     0.526     0.599
w                -1.6588      0.6870    -2.415     0.016  **
k                 0.4051      0.1173     3.454     0.001  ***
------------------------------------------------------------------------------
instruments: 7
Hansen J: chi2(3) = 2.089, p = 0.554
AR(1) [approximate]: z = -1.977, p = 0.048
AR(2) [approximate]: z = -0.529, p = 0.597
==============================================================================
```

### How to read the diagnostics

| Test | What you want | Why |
|---|---|---|
| **AR(1)** `[approximate]` | **rejects** (p < 0.05) | The differenced error is MA(1) by construction. Not rejecting suggests something is wrong. |
| **AR(2)** `[approximate]` | **does not reject** | If it rejects, lag-2 instruments are invalid — go deeper with `gmm_lags=(3, ...)`. |
| **Hansen J** | does not reject, but **p < 0.9** | A p-value near 1.00 is the classic symptom of instrument proliferation, not of a good model. |
| **instruments** | fewer than the number of units | Otherwise the weight matrix is singular and Hansen has no power. |

The example above passes all four.

### Getting at the numbers

```python
res.params           # Series of coefficients
res.bse              # standard errors
res.pvalues
res.conf_int(0.05)   # DataFrame: lower, upper
res.table()          # everything, tidy
res.diagnostics      # dict of tests
res.extra            # estimator-specific extras
```

---

## Step 7 — Compare estimators

This is the point of a unified package. Run several, table them together.

```python
results = {
    "FE":       dp.fixed_effects(panel, "n", lags=2, x=["w", "k"]),
    "DFE-A":    dp.debiased_fe(panel, "n", lags=2, x=["w", "k"]),
    "AH":       dp.anderson_hsiao(panel, "n", lags=1, exogenous=["w", "k"]),
    "Diff GMM": dp.diff_gmm(panel, "n", lags=2, predetermined=["w"],
                            exogenous=["k"], collapse=True),
}
print(dp.comparison_table(results, params=["L1.n", "L2.n", "w", "k"]))
```

```
                       FE       DFE-A          AH    Diff GMM
L1.n            0.6279***   0.7628***   1.0936***      0.0769
  (L1.n)         (0.0972)    (0.0972)    (0.2424)    (0.1461)
w              -0.4344***  -0.4100***   -0.5566**   -1.6588**
  (w)            (0.1235)    (0.1235)    (0.2571)    (0.6870)
Observations          751         751         751         611
Units                 140         140         140         140
```

Read across the row: FE understates persistence (Nickell bias), the analytical
correction moves it up, and the IV estimators move it further — at a large
cost in variance. That trade-off is the substance of this literature.

---

## Step 8 — Long-run effects

In a dynamic model the coefficient on `d` is only the *impact* effect. The
long-run effect is `θ / (1 − Σρ)`, with a delta-method standard error:

```python
lr, se = res.long_run("w", ["L1.n", "L2.n"])
print(f"long-run effect: {lr:.4f} (se {se:.4f})")
```

If the process is at or beyond a unit root, this raises `ValueError` rather
than returning a meaningless number.

---

## Step 9 — Tables and figures

### LaTeX

```python
open("table1.tex", "w").write(
    dp.comparison_to_latex(results,
                           params=["L1.n", "w", "k"],
                           caption="Employment dynamics in UK firms",
                           label="tab:employment")
)
```

Needs `\usepackage{booktabs}`. Standard errors sit under the coefficients,
stars follow journal convention, and the diagnostics land in the table notes.

### Figures

```python
import matplotlib.pyplot as plt
from dynpanelai.report import coefficient_plot, comparison_plot

comparison_plot(results, "L1.n", truth=None)
plt.savefig("fig1.pdf")          # 300 dpi, greyscale-safe palette
```

---

## Complete worked examples

### A. COVID-19 policy effects with AB-LASSO

`T = 32` with four lags generates thousands of moment conditions — exactly the
case where plain Arellano–Bond breaks down.

```python
import dynpanelai as dp

df = dp.datasets.load_covid_counties()
panel = dp.PanelData(df, unit="fips", time="week")

est = dp.ABLasso(
    y="logdc",                # log COVID-19 cases
    d="dlogtests",            # contemporaneous: test growth
    c=["school", "college", "pmask", "pshelter", "pgather50"],
    lags=4,
    transform="fod",          # forward orthogonal deviations (the paper's default)
    split=True, k_folds=2, n_splits=20,
)
res = est.fit(panel)
print(res.summary())

lr, se = res.long_run("L1.school", [f"L{j}.logdc" for j in range(1, 5)])
print(f"long-run school effect: {lr:.3f} ({se:.3f})")
```

> **Note.** The CRAN `ablasso` package implements the *first-difference*
> variant from an earlier draft. `transform="fod"` matches the published
> paper; pass `transform="fd"` to reproduce the R package.

### B. A treatment effect with many controls (DML)

```python
from dynpanelai.sim import make_partially_linear_panel

df, truth = make_partially_linear_panel(N=150, T=40, p=15, theta=1.0, seed=7)
panel = dp.PanelData(df, "unit", "time")

est = dp.DMLDynamicPanel(
    y="y", d="d", x=[f"x{j}" for j in range(15)],
    k_folds=4,
    buffer="log",       # ceil(log T); use "sqrt", "acf", or an int
    learner="lasso",    # or "rf", "gbm", "enet", or any sklearn estimator
)
res = est.fit(panel)
print(res.summary())
print("true theta:", truth["theta"])
```

Why the buffer? Blocked-time folds hold out contiguous periods, but the
observation just before a held-out block is still correlated with it. The
buffer purges those periods so the nuisance functions are fit on data that is
genuinely (weakly) independent of what they are scoring.

### C. Heterogeneous effects across groups

```python
est = dp.OrthogonalLasso(
    y="LogSales", p="LogPrice",
    controls=["LogPrice_lag", "LogSales_lag"],
    heterogeneity=["Level2"],     # categorical: one-hot expanded automatically
    k_blocks=10,                  # neighbours-left-out cross-fitting
    debias="clime",               # or "ridge" for large dictionaries
)
res = est.fit(panel)
print(res.extra["simultaneous_bands"])
```

Use the **simultaneous** bands, not the pointwise ones, when you intend to say
*which* groups differ — otherwise you are multiple-testing without correction.

### D. Forecasting

```python
train = dp.PanelData(df[df.time < 16], "unit", "time")
test  = dp.PanelData(df[df.time >= 15], "unit", "time")

out = {}
for method in ["pols", "fe", "lasso", "ridge", "ebmle", "ure"]:
    est = dp.PenalizedFE(y="y", x=["x"], method=method)
    est.fit(train)
    pred = est.predict(test)
    out[method] = dp.forecast_metrics(test.df["y"].to_numpy(), pred.to_numpy())

import pandas as pd
print(pd.DataFrame(out).T.round(4))
```

Shrinkage buys forecast accuracy by accepting bias. Do not then report the
shrunken coefficients as if they were unbiased estimates — these are different
jobs.

### E. Who responds over what horizon (AC-GATE)

```python
from dynpanelai.neural import ACGate, run_audit

est = ACGate(y="y", features=["f0", "f1", "f2"],
             proxies=["proxy0", "proxy1"], K=8, epochs=40)
res = est.fit(panel)
print(res.effective_lag.head())

report = run_audit([res.effective_lag.to_numpy()],
                   {"proxy0": proxy_values},
                   metrics_by_seed=[res.metrics])
print(report.summary())
```

Always run the audit. A model can forecast well and still have learned nothing
about lags; L1 catches the degenerate case where every entity gets the same
effective lag, and L2 tests alignment against a permutation null.

---

## Common mistakes

**Calling `system_gmm()`.** It raises in this release; the level-equation
instruments do not validate. Use `diff_gmm(collapse=True)`.

**Using DML on a short panel.** `sqrt(N)/T` must go to zero. At `T = 9` it
does not. The estimator warns you; heed it.

**Reading a Hansen p-value of 0.99 as success.** It means you have too many
instruments for the test to have power. Set `collapse=True` or restrict
`gmm_lags`.

**Forgetting that FOD ≠ FD.** They give different estimates and different
efficiency. The AB-LASSO paper uses FOD; the CRAN package uses FD.

**Interpreting the impact coefficient as the total effect.** In a dynamic
model, use `long_run()`.

**Reporting pointwise intervals for many coefficients.** Use
`simultaneous_ci` or the bands in `extra["simultaneous_bands"]`.

**Assuming weak sparsity holds.** `PanelLasso` assumes the fixed effects are
weakly sparse. If they are dense, it degenerates toward pooled OLS and biases
the lag coefficient upward. It warns you when it detects this.

---

## Where to next

- [Syntax reference](syntax_reference.md) — every argument of every estimator
- [Methods](methods.md) — the econometrics, with the equations
- [`examples/`](../examples) — runnable scripts
