# Cross-validation against `xtabond2`

Reference run: StataNow 19.5 MP, `xtabond2`, `webuse abdata`
(140 firms, 1976–1984, unbalanced, 1,031 rows).
Compared against `dynpanelai` 0.1.1.

---

## SPEC A — difference GMM, collapsed, two-step

`xtabond2 n L(1/2).n w k, gmm(n w, lag(2 4) collapse) iv(k) noleveleq twostep robust`

| | `xtabond2` | `dynpanelai` | verdict |
|---|---|---|---|
| **Observations** | 611 | 611 | ✅ exact |
| **Groups** | 140 | 140 | ✅ exact |
| **Instruments** | **7** | **7** | ✅ exact |
| **Hansen dof** | chi2(**3**) | chi2(**3**) | ✅ exact |
| L1.n | 0.0573 | 0.0769 | ≈ (+0.020) |
| L2.n | 0.0223 | 0.0171 | ≈ (−0.005) |
| w | −1.6805 | −1.6588 | ≈ (+0.022) |
| k | 0.4103 | 0.4051 | ≈ (−0.005) |
| se(L1.n) | **0.4421** | **0.1461** | ❌ 3.0× too small |
| se(L2.n) | 0.1468 | 0.0576 | ❌ 2.5× too small |
| se(w) | 0.7758 | 0.6870 | ❌ 11% too small |
| se(k) | 0.0824 | 0.1173 | ❌ 42% too large |
| Hansen | 1.67, p=0.645 | 2.089, p=0.554 | ≈ |
| AR(1) | z=−1.47, p=0.141 | z=−1.98, p=0.048 | ❌ opposite verdict |
| AR(2) | z=−0.58, p=0.563 | z=−0.53, p=0.597 | ≈ |

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

## SPEC C — system GMM (the target for the disabled path)

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
| AR(1) / AR(2) | −2.79 (0.005) / −0.83 (0.406) |

`dynpanelai`'s disabled implementation produced **1,282** observations, i.e. it
stacked the difference and level equations additively (611 + 671). `xtabond2`
reports **751**. That single discrepancy is the most likely root cause of the
level-equation failure and is the first thing to fix.

Note also that even the reference system GMM returns L1.n = 1.62 here, i.e.
an explosive root. The collapsed `lag(2 4)` specification is weak for the level
equation on this panel; that is a property of the specification, not of any
implementation.

## SPEC D — native `xtabond`, two-step

20 instruments, L1.n = 0.3913 (0.2046), w = −0.4986, k = 0.4358.
Different instrument set (`L(2/4).n` only, with `D.w D.k` standard), so not
directly comparable — included as an independent sanity check on magnitudes.

## SPEC E — SPEC A, one-step

| | `xtabond2` | `dynpanelai` |
|---|---|---|
| Instruments | 7 | 7 ✅ |
| L1.n | −0.0287 (0.4022) | — |
| w | −1.8511 (0.7428) | — |
| k | 0.4064 (0.0759) | — |

The one-step and two-step coefficients differ substantially in `xtabond2` too
(−0.029 vs 0.057 on L1.n), confirming this specification is weakly identified.

---

## Conclusions

**What is validated**

- Instrument *counting and structure* are correct. 7 collapsed and 35
  uncollapsed both match exactly, as do observations, groups and Hansen degrees
  of freedom. The 0.1.1 collapse fix is confirmed against the reference.
- Coefficients agree to roughly ±0.02 on every parameter in SPEC A.

**What is not**

1. **Standard errors** (P0). Off by factors of 0.4× to 3×, in both directions.
   Since the coefficients and the instrument matrix agree, the fault is almost
   certainly in the Windmeijer correction rather than in `Z`. Testable by
   comparing one-step robust SEs first, where no correction applies.
2. **AR(1)** (P1). Already labelled `[approximate]`; SPEC A shows it can reach
   the opposite verdict (p=0.048 vs 0.141 at the 5% line). The exact
   Arellano–Bond m-test should replace it.
3. **System GMM** (P1). Remains disabled. Target: 751 observations and 10
   instruments; the current attempt produces 1,282 observations, so the
   equation stacking is wrong before the instruments are even considered.
4. **Uncollapsed coefficients** (P2) drift further than collapsed ones
   (L1.n 0.056 vs 0.014), consistent with a near-singular weight matrix being
   inverted slightly differently.

**Priority for 0.1.2:** fix the variance estimator first. Coefficients are
close enough to be usable; the standard errors are not, and they are what
inference rests on.
