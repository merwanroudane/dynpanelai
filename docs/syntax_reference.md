# dynpanelai — Syntax Reference

Every public estimator, every argument. For the narrative walkthrough see the
[user guide](user_guide.md).

Conventions: **bold** = required, `default` shown otherwise.

---

## Contents

- [Core](#core)
- [GMM](#gmm)
- [Bias corrections](#bias-corrections)
- [AB-LASSO](#ab-lasso)
- [Double machine learning](#double-machine-learning)
- [Orthogonal / debiased Lasso](#orthogonal--debiased-lasso)
- [High-dimensional panel Lasso](#high-dimensional-panel-lasso)
- [Shrinkage and forecasting](#shrinkage-and-forecasting)
- [Neural lag discovery](#neural-lag-discovery)
- [Penalised building blocks](#penalised-building-blocks)
- [Reporting](#reporting)
- [Datasets and simulation](#datasets-and-simulation)

---

## Core

### `PanelData(df, unit, time, copy=True)`

| Argument | Type | Description |
|---|---|---|
| **`df`** | `DataFrame` | Long format, one row per unit-period |
| **`unit`** | `str` | Unit identifier column |
| **`time`** | `str` | Time identifier column |
| `copy` | `bool = True` | Copy before mutating |

**Attributes:** `N`, `T`, `n_obs`, `balanced`, `units`, `times`, `df`

**Methods**

| Method | Returns | Notes |
|---|---|---|
| `.summary()` | `Series` | Shape and balance diagnostics |
| `.lag(cols, lags=1)` | `DataFrame` | Gap-aware; missing period → `NaN` |
| `.lead(cols, leads=1)` | `DataFrame` | |
| `.with_lags(cols, lags, dropna=False)` | `PanelData` | Appends lag columns |
| `.matrix(cols, dropna=True)` | `(ndarray, ndarray)` | Design matrix + unit codes |
| `.wide(col)` | `ndarray (T, N)` | For matrix-oriented estimators |
| `.from_wide(mat, name)` | `Series` | Inverse of `.wide()` |
| `.balance()` | `PanelData` | Drop incomplete units |

### `PanelResults`

| Property / method | Returns |
|---|---|
| `.params`, `.bse`, `.tvalues`, `.pvalues` | `Series` |
| `.conf_int(alpha=0.05)` | `DataFrame` |
| `.table(alpha=0.05)` | `DataFrame` |
| `.summary(alpha=0.05, digits=4)` | `str` |
| `.to_latex(alpha, digits, caption, label, se_below)` | `str` |
| `.long_run(coef, lag_coefs)` | `(float, float)` |
| `.diagnostics`, `.extra` | `dict` |

### Transforms

```python
within_transform(panel, cols, time_demean=False)
first_difference(panel, cols, time_demean=False)
forward_orthogonal_deviation(panel, cols, time_demean=True)
mundlak_means(panel, cols, prefix="mean_")
fod_matrix(T)                       # (T-1, T) operator
apply_transform(panel, cols, method="within", **kw)
```

---

## GMM

### `DynamicPanelGMM(y, ...)` · `diff_gmm(panel, y, ...)`

> **`system_gmm()` raises `NotImplementedError` in this release** — the
> level-equation instruments do not validate. Use `diff_gmm(collapse=True)`.

| Argument | Default | Description |
|---|---|---|
| **`y`** | — | Outcome column |
| `lags` | `1` | Lags of the outcome as regressors |
| `predetermined` | `None` | Correlated with past errors; instrumented by own lags |
| `exogenous` | `None` | Strictly exogenous; own instruments |
| `transformation` | `'fd'` | `'fd'` or `'fod'` |
| `level` | `False` | Add the levels equation (system GMM) |
| `steps` | `2` | 1 = one-step, 2 = two-step + Windmeijer |
| `gmm_lags` | `(2, None)` | Lag range for instruments; `None` = all |
| `collapse` | `False` | Collapse instruments, keeping count linear in `T` |
| `time_dummies` | `False` | |

**Diagnostics returned:** `instruments` (split by equation), `Hansen J`,
`AR(1) [approximate]`, `AR(2) [approximate]`

### `anderson_hsiao(panel, y, lags=1, exogenous=None, instrument='level')`

`instrument`: `'level'` uses `y[t-2]`, `'diff'` uses `Δy[t-2]`.

---

## Bias corrections

```python
fixed_effects(panel, y, lags=1, x=None, twoway=False)
debiased_fe(panel, y, lags=1, x=None, twoway=False)
bias_corrected_lsdv(panel, y, lags=1, x=None, initial='ah', iterations=3)
half_panel_jackknife(panel, y, lags=1, x=None)

split_panel_jackknife(panel, y, estimator=None, dimension='time',
                      lags=1, x=None, **kwargs)
```

`split_panel_jackknife` accepts **any** estimator, so it composes:

```python
# Debiased Arellano-Bond (DAB): jackknife over the cross-section
dp.split_panel_jackknife(
    panel, "y",
    estimator=lambda p: dp.diff_gmm(p, "y", lags=1),
    dimension="unit",
)
```

---

## AB-LASSO

### `ABLasso(y, d=None, c=None, ...)` · `ab_lasso(panel, y, ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`** | — | Outcome |
| `d` | `None` | Treatment, entering contemporaneously |
| `c` | `None` | Other predetermined covariates, entering at `t-1` |
| `lags` | `1` | Lags of the outcome |
| `transform` | `'fod'` | `'fod'` (the paper) or `'fd'` (the CRAN package) |
| `split` | `True` | Cross-sectional splitting + cross-fitting (AB-LASSO-SS) |
| `k_folds` | `2` | Folds; the paper studies 2 and 5 |
| `n_splits` | `100` | Random re-splits, aggregated by median |
| `post` | `True` | Post-LASSO refit in the first stage |
| `lambda_c` | `1.1` | Penalty slack constant |
| `lambda_rule` | `'paper'` | `'paper'` = replication code's λ; `'plugin'` = full `rlasso` |
| `time_demean` | `True` | Remove additive time effects |
| `seed` | `202304` | |

**Requires a balanced panel.** Call `panel.balance()` first.

---

## Double machine learning

### `DMLDynamicPanel(y, d, x=None, ...)` · `dml_dynamic_panel(panel, y, d, ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`**, **`d`** | — | Outcome and treatment |
| `x` | `None` | Additional controls |
| `y_lags`, `d_lags`, `x_lags` | `1` | Lag depths in the information set `W` |
| `k_folds` | `4` | Blocked-time folds |
| `buffer` | `'log'` | `'log'`, `'sqrt'`, `'acf'`, or an `int` |
| `fold_scheme` | `'blocked'` | `'blocked'` or `'nlo'` |
| `learner` | `'lasso'` | `'lasso'`, `'enet'`, `'rf'`, `'gbm'`, `'ols'`, or a sklearn estimator |
| `demeaning` | `'global'` | `'fold'` for strict fold purity |
| `vcov` | `'cluster'` | `'cluster'`, `'twoway'`, `'driscoll-kraay'` |
| `trim` | `None` | e.g. `0.995` to trim extreme residualised treatment |
| `seed` | `42` | |

**Diagnostics:** `buffer B`, `effective buffer B*`, `Var(D residualised)`,
`sqrt(N)/T`, `trimming`

### Fold constructors

```python
blocked_time_folds(periods, k=4, buffer=0, max_lag=0)
nlo_folds(periods, k=10)
clustered_unit_folds(n_units, k=5, seed=None)
buffer_rules(T)                 # {'log': ..., 'sqrt': ...}
suggest_buffer_acf(series, threshold=0.05, max_lag=None)
```

---

## Orthogonal / debiased Lasso

### `OrthogonalLasso(y, p, controls, heterogeneity=None, ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`**, **`p`** | — | Outcome and base treatment |
| **`controls`** | — | First-stage controls |
| `heterogeneity` | `None` | Columns defining the CATE dictionary; categoricals one-hot expanded |
| `k_blocks` | `10` | NLO cross-fitting blocks |
| `mundlak` | `True` | Add unit means (correlated random effects) |
| `second_stage` | `'lasso'` | `'lasso'` or `'ols'` |
| `debias` | `'clime'` | `'clime'`, `'ridge'`, `'none'` |
| `clime_lambda` | `None` | |
| `seed` | `0` | |

**`extra` contains:** `beta_lasso`, `residual_y`, `residual_p`,
`simultaneous_bands`

```python
simultaneous_ci(beta, cov, alpha=0.05, n_boot=2000, seed=0)  # -> (bands, crit)
```

---

## High-dimensional panel Lasso

### `PanelLasso(y, lags=1, x=None, ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`** | — | Outcome |
| `lags` | `1` | Lags of the outcome |
| `x` | `None` | Covariates |
| `lambda_M` | `0.5` | Constant `M` in `λ_N = sqrt(4 M n log(p∨N)³)` |
| `penalize_fe` | `True` | Apply the `λ_N/√N` penalty to unit effects |
| `desparsify` | `True` | Debias + robust variance (expensive: `p` nodewise fits) |
| `seed` | `0` | |

Lower `lambda_M` if all fixed effects collapse to zero — the estimator warns
when that happens alongside large between-unit variation.

---

## Shrinkage and forecasting

### `fe_shrink(y, M, method='URE', centering='gen', ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`** | — | `(T, J)` or `(J,)` fixed-effect estimates |
| **`M`** | — | Per-unit variance matrices, or a `(J,)` array of variances |
| `method` | `'URE'` | `'URE'` or `'EBMLE'` |
| `centering` | `'gen'` | `'gen'` (data-driven location) or `'0'` |
| `W` | `None` | Weight matrix in the risk criterion |
| `tau` | `0.95` | Quantile bounding the search for `mu` |
| `n_init` | `1` | Random restarts |
| `diag_lambda` | `False` | Restrict `Λ` to diagonal |

Returns `FEShrinkResult(theta, mu, Lambda, obj, shrinkage, method)`.

### `PenalizedFE(y, x=None, method='lasso', ...)`

| `method` | Meaning |
|---|---|
| `'pols'` | Pooled OLS — no fixed effects |
| `'fe'` / `'lsdv'` | Unpenalised dummies |
| `'lasso'`, `'ridge'`, `'enet'` | Penalty on the **fixed effects only** |
| `'ebmle'`, `'ure'` | LSDV then optimal shrinkage |

Other arguments: `l1_ratio=0.5`, `k_blocks=5`, `n_lambda=60`, `one_se=True`.

Methods: `.fit(panel)`, `.predict(panel)`.

```python
forecast_metrics(y_true, y_pred)   # bias, variance, mse, rmse, mae, n
rolling_origin_blocks(periods, k=5)
```

---

## Neural lag discovery

### `ACGate(y, features, proxies, ...)`

| Argument | Default | Description |
|---|---|---|
| **`y`** | — | Target |
| **`features`** | — | Time-varying covariates (enter only through lags) |
| **`proxies`** | — | Time-invariant entity characteristics conditioning the gate |
| `K` | `8` | Maximum lag horizon |
| `hidden`, `layers` | `32`, `2` | LSTM backbone |
| `tau` | `1.0` | Softmax temperature |
| `lam_pos` | `0.5` | Normalised position bias `λ k/K` |
| `lam_recon` | `0.1` | Proxy-reconstruction loss weight |
| `epochs`, `lr`, `batch_size` | `60`, `1e-3`, `256` | |
| `val_fraction`, `test_fraction` | `0.15`, `0.15` | Chronological splits |

Returns `ACGateResult(effective_lag, lag_weights, metrics, history)`.
Requires `pip install dynpanelai[neural]`.

### Audit protocol

```python
audit_l1(k_star, eps=1e-3)                       # degeneracy guard
audit_l2(k_star, stratifier, n_perm=1000, seed=0)  # permutation test
audit_l3(k_star, k_true)                         # ground-truth recovery
fisher_combine(pvalues)
run_audit(k_star_by_seed, stratifiers, metrics_by_seed=None,
          k_true=None, eps=1e-3, n_perm=1000, seed=0)
```

---

## Penalised building blocks

### `rlasso(X, y, ...)`

Plug-in penalty LASSO (Belloni, Chen, Chernozhukov & Hansen 2012; cluster-robust
loadings from Belloni, Chernozhukov, Hansen & Kozbur 2016).

| Argument | Default | Description |
|---|---|---|
| `post` | `True` | Post-LASSO OLS refit on the selected support |
| `intercept` | `True` | |
| `homoskedastic` | `False` | |
| `x_dependent` | `False` | Simulation-based penalty level |
| `lambda_start` | `None` | Override `λ₀` |
| `c`, `gamma` | `1.1`, `0.1/log n` | Penalty constants |
| `clusters` | `None` | **Cluster-robust loadings — use for panel data** |
| `max_iter`, `tol` | `15`, `1e-5` | Loading refinement |

Returns `RLasso(coef, intercept, selected, lambda0, loadings, lambdas, n_iter)`.

### CLIME and nodewise

```python
clime(Q, lam=None, n=None, symmetrize=True, c_clime=1.0)
clime_column(Q, j, lam)
nodewise_inverse(X, lam=None, c=1.1, gamma=None, post=False)
lambda_plugin(n, p, c=1.1, gamma=None, n_endog=1)
```

---

## Reporting

```python
comparison_table(results, params=None, digits=4, alpha=0.05)
comparison_to_latex(results, params=None, digits=4, caption=None, label=None)
results_to_latex(res, alpha, digits, caption, label, se_below=True)
monte_carlo_table(estimates, truth, ses=None, digits=4)

set_style(context="paper")            # or "talk"
coefficient_plot(res, params=None, alpha=0.05, ax=None, title=None)
comparison_plot(results, param, truth=None, alpha=0.05, ax=None)
monte_carlo_plot(estimates, truth, ses=None, axes=None)
lag_weight_plot(lag_weights, effective_lag=None, max_entities=40, ax=None)
forecast_error_plot(metrics, metric="rmse", baseline=None, ax=None)
```

---

## Datasets and simulation

```python
load_covid_counties(balanced=True, add_growth=True)
load_abond_employment(add_logs=True)
available_datasets()
```

```python
make_partially_linear_panel(N=200, T=40, p=20, theta=1.0, rho=0.4,
                            nonlinear=False, seed=None)
make_ab_lasso_panel(N=200, T=40, theta=(0.75, 0.25), rho=0.5,
                    phi=-0.17, pi=0.67, seed=None)
make_shrinkage_panel(N=100, T=20, gamma=0.2, sparsity=0.0, seed=None)
make_heterogeneous_lag_panel(N=60, T=80, K=8, nonlinear=False, seed=None)
```

Each returns `(DataFrame, truth_dict)`.
