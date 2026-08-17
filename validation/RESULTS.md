# Cross-validation against `xtabond2`

Reference: StataNow 19.5 MP · `xtabond2` 3.7.2 (Roodman) · `webuse abdata`
(140 firms, 1976–1984, unbalanced, 1,031 rows), compared against
`dynpanelai` 0.1.1.

> **Correction.** An earlier version of this file reported standard errors
> "off by factors of 0.4× to 3×". That conclusion was wrong: it compared
> `xtabond2` against numbers that had been written into the documentation by
> hand rather than read from a run. The figures below are all taken from
> actual output on both sides.

---

## SPEC A — difference GMM, collapsed, two-step

`xtabond2 n L(1/2).n w k, gmm(n w, lag(2 4) collapse) iv(k) noleveleq twostep robust`

| | `xtabond2` | `dynpanelai` | ratio |
|---|---|---|---|
| **Observations** | 611 | 611 | ✅ |
| **Groups** | 140 | 140 | ✅ |
| **Instruments** | **7** | **7** | ✅ |
| **Hansen dof** | chi2(3) | chi2(3) | ✅ |
| L1.n | 0.0573 | 0.0769 | +0.020 |
| L2.n | 0.0223 | 0.0171 | −0.005 |
| w | −1.6805 | −1.6588 | +0.022 |
| k | 0.4103 | 0.4051 | −0.005 |
| se(L1.n) | 0.4421 | 0.4326 | 0.98 |
| se(L2.n) | 0.1468 | 0.1493 | 1.02 |
| se(w) | 0.7758 | 0.7026 | **0.91** |
| se(k) | 0.0824 | 0.0823 | 1.00 |
| Hansen | 1.67, p=0.645 | 2.09, p=0.554 | close |
| AR(1) | z=−1.47, p=0.141 | z=−1.97, p=0.048 | ⚠️ opposite verdict at 5% |
| AR(2) | z=−0.58, p=0.563 | z=−0.53, p=0.597 | close |

## SPEC E — SPEC A, one-step

Included to isolate the Windmeijer correction: at one step no correction
applies, so any gap here belongs to the base robust variance formula.

| | `xtabond2` | `dynpanelai` | ratio |
|---|---|---|---|
| se(L1.n) | 0.4022 | 0.3965 | 0.99 |
| se(L2.n) | 0.1303 | 0.1360 | 1.04 |
| se(w) | 0.7428 | 0.6943 | **0.93** |
| se(k) | 0.0759 | 0.0751 | 0.99 |

**Diagnosis.** The `w` gap of 7–9% is present at *one step*, so it is not the
Windmeijer correction. The correction itself behaves correctly: it inflates
every standard error by roughly the same factor as in `xtabond2`
(e.g. L1.n 0.3965→0.4326, a 9.1% inflation, against 0.4022→0.4421, 9.9%).

The residual gap is therefore in the base variance or in the treatment of the
predetermined variable `w`, and is concentrated in that one coefficient. It is
small enough not to change any inferential verdict in this specification, but
it is not yet explained.

## SPEC B — same, uncollapsed

| | `xtabond2` | `dynpanelai` |
|---|---|---|
| **Instruments** | **35** | **35** ✅ |
| Observations | 611 | 611 ✅ |
| L1.n | 0.0145 | 0.0559 |
| w | −1.2087 | −1.0441 |
| k | 0.4961 | 0.4867 |
| Hansen | chi2(31)=39.53, p=0.140 | chi2(31)=46.49, p=0.037 |

`xtabond2` warns that the two-step moment covariance is singular and falls back
to a generalised inverse — the same situation `dynpanelai` handles with `pinv`.
Coefficients drift further apart here than in the collapsed case, consistent
with two different generalised inverses of a near-singular matrix.

## SPEC C — system GMM (target for the disabled path)

`xtabond2 n L(1/2).n w k, gmm(n w, lag(2 4) collapse) iv(k) twostep robust`

| | `xtabond2` |
|---|---|
| Observations | **751** |
| Instruments | **10** (7 difference + 3 level) |
| Level instruments | `DL.(n w)` collapsed, `k`, `_cons` |
| L1.n | 1.6228 (0.4417) |
| L2.n | −0.3893 (0.1810) |
| w | −0.2723 (0.2293) |
| k | −0.1734 (0.2156) |
| _cons | 0.4880 (1.0518) |
| Hansen | chi2(5)=6.85, p=0.232 |

`dynpanelai`'s disabled path produced **1,282** observations — it stacked the
two equations additively (611 + 671) where `xtabond2` reports **751**. The
equation stacking is wrong before the instruments are even considered; that is
the first thing to fix.

Note that even the reference returns L1.n = 1.62, an explosive root. The
collapsed `lag(2 4)` specification is weakly identified for the level equation
on this panel — a property of the specification, not of either implementation.

## SPEC D — native `xtabond`, two-step

20 instruments, L1.n = 0.3913 (0.2046), w = −0.4986, k = 0.4358. A different
instrument set (`L(2/4).n` only, `D.w D.k` standard), so not directly
comparable; included as an independent check on magnitudes.

---

## Verdict

**Validated**

- Instrument construction: 7 collapsed and 35 uncollapsed match exactly, as do
  observations, groups and Hansen degrees of freedom. The 0.1.1 `collapse` fix
  is confirmed against the reference.
- Coefficients agree to ±0.02 on every parameter in SPEC A.
- Standard errors agree to within 2% on three of four coefficients, and the
  Windmeijer inflation factor matches (≈9% against ≈10%).

**Open**

| Priority | Issue |
|---|---|
| P1 | `se(w)` runs ~7–9% below `xtabond2` at both one and two steps. Present in the base variance, not the correction. Concentrated in the predetermined variable. |
| P1 | AR(1) is the simplified m-test and reaches the opposite verdict at the 5% line here (0.048 vs 0.141). Replace with the exact Arellano–Bond statistic. |
| P1 | System GMM disabled. Target: 751 observations, 10 instruments. |
| P2 | Uncollapsed coefficients drift further than collapsed, likely generalised-inverse choice under a singular weight matrix. |

**Usable today:** `diff_gmm(..., collapse=True)` for point estimates and
inference, with the caveat that `w`'s standard error is mildly optimistic and
that AR(1) should be read from Stata when it falls near 5%.
