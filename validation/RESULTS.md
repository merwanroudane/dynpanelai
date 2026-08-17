# Cross-validation against `xtabond2`

Reference: StataNow 19.5 MP · `xtabond2` 3.7.2 (Roodman) · `webuse abdata`
(140 firms, 1976–1984, unbalanced, 1,031 rows), against `dynpanelai` 0.1.2.

> **Two earlier corrections.** A first version of this file reported standard
> errors "off by factors of 0.4× to 3×" — that comparison used figures written
> into the documentation by hand rather than read from a run, and was wrong. A
> second version reported a genuine 7–9% gap on `se(w)`. That gap was real and
> has now been fixed. Everything below is from actual output on both sides.

---

## SPEC A — difference GMM, collapsed, two-step

`xtabond2 n L(1/2).n w k, gmm(n w, lag(2 4) collapse) iv(k) noleveleq twostep robust`

| | `xtabond2` | `dynpanelai` | ratio |
|---|---|---|---|
| Observations | 611 | 611 | ✅ |
| Groups | 140 | 140 | ✅ |
| Instruments | 7 | 7 | ✅ |
| L1.n | 0.0572805 | 0.0572805 | **1.0000** |
| L2.n | 0.0222885 | 0.0222885 | **1.0000** |
| w | −1.680533 | −1.680533 | **1.0000** |
| k | 0.4103221 | 0.4103221 | **1.0000** |
| se(L1.n) | 0.4420636 | 0.4414 | 0.9986 |
| se(L2.n) | 0.1467757 | 0.1466 | 0.9989 |
| se(w) | 0.7757580 | 0.7752 | 0.9993 |
| se(k) | 0.0824243 | 0.0823 | 0.9981 |
| Hansen | 1.67, p=0.645 | 1.665, p=0.645 | ✅ |
| AR(1) | z=−1.47, p=0.141 | z=−1.93, p=0.054 | ⚠️ approximate |
| AR(2) | z=−0.58, p=0.563 | z=−0.51, p=0.610 | ⚠️ approximate |

## SPEC E — SPEC A, one-step

| | `xtabond2` | `dynpanelai` | ratio |
|---|---|---|---|
| L1.n | −0.0287032 | −0.0287032 | **1.0000** |
| L2.n | 0.0455574 | 0.0455574 | **1.0000** |
| w | −1.851055 | −1.851055 | **1.0000** |
| k | 0.4064098 | 0.4064098 | **1.0000** |
| se(L1.n) | 0.4021562 | 0.4022 | **1.0000** |
| se(L2.n) | 0.1303349 | 0.1303 | **1.0000** |
| se(w) | 0.7427967 | 0.7428 | **1.0000** |
| se(k) | 0.0759036 | 0.0759 | **1.0000** |

---

## The two bugs this exposed

### 1. The one-step weight matrix

The efficient one-step weight is **not** `(Z'Z)^{-1}`. First-differencing
induces an MA(1) error, so the kernel must be

$$H = egin{pmatrix} 2 & -1 & & \ -1 & 2 & -1 & \ & \ddots & \ddots & \ddots \end{pmatrix},
\qquad W_1 = \Bigl(\sum_i Z_i' H Z_i\Bigr)^{-1}.$$

Using `H = I` is not merely inefficient — it changes the point estimate,
the standard errors *and* the Hansen statistic, and it does so unevenly across
coefficients, because each regressor's own serial correlation interacts with
the omitted off-diagonal. That is why the damage looked concentrated in `w`.

Forward orthogonal deviations leave the transformed error uncorrelated, so
there `H = I` is correct and the code keeps `Z'Z`.

Gaps in a unit's time series break the MA(1) link, so the `-1` entries are
written only where two rows are genuinely adjacent in time — which matters on
this unbalanced panel.

### 2. Windmeijer's score was evaluated at the wrong residuals

```
D[:, j] = -M X'Z W  (dS/dbeta_j)  W  sum_i Z_i' u_i(beta_2)
```

The score `sum_i Z_i' u_i` is evaluated at the **second-step**
estimate; only \(\partial\widehat S/\partialeta_j\) uses first-step
residuals, since it is the first step that produced the weight matrix being
differentiated. The implementation used first-step residuals in both places.

Both fixes are locked in by regression tests
(`tests/test_gmm_options.py::test_matches_xtabond2_standard_errors`).

---

## Still open

| Priority | Issue |
|---|---|
| P1 | AR(1)/AR(2) remain the simplified m-test, labelled `[approximate]`. On SPEC A: p=0.054 against 0.141. Same direction, but do not read them near a decision boundary — take those from Stata. |
| P1 | System GMM disabled. Target: 751 observations, 10 instruments (7 difference + 3 level: `DL.(n w)` collapsed, `k`, `_cons`). The disabled path stacks 1,282 rows, so the equation stacking is wrong before instruments are considered. |
| P2 | Uncollapsed two-step coefficients still drift from the reference (L1.n 0.056 vs 0.014). `xtabond2` reports a singular moment covariance there and falls back to a generalised inverse; the two implementations pick different ones. |

**Usable today:** `diff_gmm(..., collapse=True)` at one or two steps, for both
point estimates and inference.
