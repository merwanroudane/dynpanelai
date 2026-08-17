# Changelog

All notable changes to `dynpanelai`. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows
[Semantic Versioning](https://semver.org/).

The package is pre-1.0: the public API may still change. It will reach 1.0.0
when system GMM is validated, the exact Arellano–Bond m-test replaces the
approximate one, and the uncollapsed two-step path matches the reference.

---

## [0.1.2] — 2026-08-17

Cross-validated against `xtabond2` 3.7.2 (Roodman) on `webuse abdata`. Two
genuine bugs in the variance path were found and fixed; difference GMM now
reproduces the reference to within 0.2%.

### Fixed

- **One-step weight matrix used the wrong kernel.** The efficient one-step
  weight is not `(Z'Z)⁻¹`. First-differencing induces an MA(1) error, so the
  kernel must be the tridiagonal `H` with 2 on the diagonal and −1 adjacent,
  giving `W₁ = (Σᵢ Zᵢ' H Zᵢ)⁻¹`.

  Using `H = I` is not merely inefficient: it changes the point estimate, the
  standard errors *and* the Hansen statistic, and unevenly across
  coefficients, because each regressor's own serial correlation interacts with
  the omitted off-diagonal. Forward orthogonal deviations leave the error
  uncorrelated, so `H = I` remains correct there. Gaps in a unit's series
  break the MA(1) link, so −1 entries are written only between genuinely
  adjacent periods.

- **Windmeijer correction evaluated its score at the wrong residuals.** The
  score `Σᵢ Zᵢ'ûᵢ` must be evaluated at the **second-step** estimate; only the
  derivative of the weight matrix uses first-step residuals, since it is the
  first step that produced the matrix being differentiated.

### Added

- Six regression tests pinning the `xtabond2` reference values for
  coefficients and standard errors at one and two steps.
- [`validation/RESULTS.md`](validation/RESULTS.md) — full comparison and
  diagnosis.
- [`validation/stata_crosscheck.do`](validation/stata_crosscheck.do) — the
  reference do-file, five matched specifications.

### Verified

| | `xtabond2` | `dynpanelai` |
|---|---|---|
| L1.n | 0.0572805 | 0.0572805 |
| w | −1.680533 | −1.680533 |
| se(L1.n) | 0.4420636 | 0.4414 (0.999) |
| Hansen | 1.67, p=0.645 | 1.665, p=0.645 |

One-step agreement is exact to four decimals on all eight quantities.

---

## [0.1.1] — 2026-08-17

**Yank recommended.** Contains the weight-matrix bug fixed in 0.1.2.

### Fixed

- `collapse=True` now genuinely collapses. Instrument construction moved to
  `gmm/build.py`, emitting one column per lag depth (O(T)) rather than one per
  (period, lag) pair (O(T²)). On the employment panel: 55 → 15 instruments.
  Previously `_collapse()` was defined but never called.
- `time_dummies=True` now reaches both the regressor matrix and the instrument
  set. Previously stored and ignored.
- AR tests computed on differenced rows only, fixing a crash once the level
  equation is stacked.
- `ACGate.layers` was ignored (`num_layers` hardcoded to 2).
- `pytest` no longer silences `UserWarning` globally — the package uses
  warnings as its guardrail mechanism.

### Changed

- `system_gmm()` raises `NotImplementedError`. The level-equation block is
  implemented but does not validate: on a mean-stationary simulated panel
  where the extra moments hold by construction, Hansen rejects at p<0.001 and
  the AR coefficient is recovered as 0.34 against a true 0.75. Returning
  plausible but wrong numbers under a "System GMM" label is worse than
  refusing.
- AR(1)/AR(2) relabelled `[approximate]`.

### Added

- 17 tests asserting each GMM option's *structural* consequence.
- GitHub Actions CI: ruff, pytest on 3.10/3.11/3.12 plus Windows, a separate
  job for the neural extra, and a build job verifying bundled data ships.

---

## [0.1.0] — 2026-08-17

**Yank recommended.** `collapse`, `time_dummies` and `level` were accepted,
documented and completely inert; `system_gmm()` returned difference-GMM
numbers under a "System GMM" label.

### Added

Initial release: eight methodologies for dynamic panel data behind one
container and one results object — difference GMM, bias corrections,
Arellano–Bond LASSO, double machine learning, orthogonal/debiased Lasso,
weakly sparse panel Lasso, URE/EB shrinkage, and AC-GATE lag discovery. Two
bundled real datasets, publication-quality tables and figures, and the Monte
Carlo designs from each source paper.

---

## Known limitations

| Priority | Issue |
|---|---|
| P1 | AR(1)/AR(2) are a simplified m-test, labelled `[approximate]`. On the reference specification: p=0.054 against `xtabond2`'s 0.141 — same direction, different verdict at 5%. |
| P1 | System GMM disabled. Target: 751 observations, 10 instruments (7 difference + 3 level: `DL.(n w)` collapsed, `k`, `_cons`). |
| P2 | Uncollapsed two-step coefficients drift from the reference; `xtabond2` reports a singular moment covariance there and the two implementations pick different generalised inverses. |
