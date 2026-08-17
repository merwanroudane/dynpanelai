# dynpanelai

[![Docs](https://img.shields.io/badge/docs-website-2A9D8F?style=flat-square)](https://merwanroudane.github.io/dynpanelai/)
[![PyPI version](https://img.shields.io/pypi/v/dynpanelai.svg)](https://pypi.org/project/dynpanelai/)
[![Python versions](https://img.shields.io/pypi/pyversions/dynpanelai.svg)](https://pypi.org/project/dynpanelai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Machine learning and modern inference for dynamic panel data models.**

### 🔗 Links

| | |
|---|---|
| 📘 **Documentation site** | **https://merwanroudane.github.io/dynpanelai/** |
| 📦 **PyPI package** | **https://pypi.org/project/dynpanelai/** · v0.1.1 |
| 💻 Source | https://github.com/merwanroudane/dynpanelai |
| 📖 User guide | [docs/user_guide.md](docs/user_guide.md) |
| 📑 Syntax reference | [docs/syntax_reference.md](docs/syntax_reference.md) |
| 🧮 Methods & theory | [docs/methods.md](docs/methods.md) |
| ▶️ Examples | [examples/](examples/) |

---

A unified Python implementation of eight methodologies for dynamic panels with
high-dimensional controls — classical GMM, bias corrections, LASSO-based
moment selection, double machine learning, orthogonal/debiased Lasso, optimal
shrinkage, and neural lag discovery — behind one data container, one results
object, and one publication-quality reporting layer.

```bash
pip install dynpanelai
```

---

## Why this package

Dynamic panel models face a specific tension. Fixed effects are essential for
persistent heterogeneity, but they interact with the lagged dependent variable
to produce the Nickell bias. The classical fix — instrument with lagged
levels — generates a number of moment conditions that grows like `T²`, which
reintroduces bias through overfitting. And modern applications add hundreds of
controls, where naive post-LASSO inference is simply invalid.

Each module here solves one part of that problem, and they share enough
infrastructure that you can run all of them on the same panel and compare.

| Module | Method | Source |
|---|---|---|
| `gmm` | Difference GMM, Anderson–Hsiao (system GMM disabled, see below) | Arellano & Bond (1991); Blundell & Bond (1998) |
| `biascorr` | Analytical, split-panel, Kiviet bias corrections | Hahn & Kuersteiner (2002); Dhaene & Jochmans (2015) |
| `ablasso` | Arellano–Bond LASSO moment selection | Chernozhukov, Fernández-Val, Huang & Wang (2024) |
| `dml` | Double ML, blocked-time cross-fitting | Sneller (2026) |
| `ortho` | Orthogonal + debiased Lasso for high-dimensional CATE | Semenova, Goldman, Chernozhukov & Taddy (2023) |
| `hdpanel` | Uniform inference, weakly sparse fixed effects | Kock & Tang (2019) |
| `shrink` | URE / Empirical Bayes shrinkage, penalised-FE forecasting | Kwon (2026); Cornejo & Sosa-Escudero (2026) |
| `neural` | AC-GATE entity-conditioned lag discovery + audit | Xu (2026) |

---

## Quick start

```python
import dynpanelai as dp

df = dp.datasets.load_abond_employment()          # 140 UK firms x 9 years
panel = dp.PanelData(df, unit="id", time="year")

res = dp.diff_gmm(panel, y="n", lags=2,
                  predetermined=["w"], exogenous=["k"],
                  gmm_lags=(2, 4), collapse=True, steps=2)
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
L1.n              0.0573      0.4414     0.130     0.897
L2.n              0.0223      0.1466     0.152     0.879
w                -1.6805      0.7752    -2.168     0.030  **
k                 0.4103      0.0823     4.988     0.000  ***
------------------------------------------------------------------------------
instruments: 7
Hansen J: chi2(3) = 1.665, p = 0.645
AR(1) [approximate]: z = -1.927, p = 0.054
AR(2) [approximate]: z = -0.510, p = 0.610
==============================================================================
```

---

## Known limitations

- **`system_gmm()` raises `NotImplementedError`.** The level-equation
  instrument block does not validate: on a mean-stationary simulated panel
  where the extra moments hold by construction, Hansen rejects at p<0.001 and
  the AR coefficient comes back 0.34 against a true 0.75. Use
  `diff_gmm(..., collapse=True)`, which recovers 0.778 on the same design.
- **Difference GMM reproduces `xtabond2` 3.7.2 to within 0.2%**
  ([full comparison](validation/RESULTS.md)) on coefficients, standard errors,
  instrument counts, Hansen and sample dimensions, at both one and two steps.
- **AR(1)/AR(2) are labelled `[approximate]`.** They are a simplified m-test
  that does not net out parameter-estimation error. Directionally reliable,
  not a substitute for the exact Arellano-Bond statistic.

---

## Which method should I use?

Start from the shape of your panel, not from the method you have heard of.

```
Is T short (under ~15)?
├── Yes → gmm.diff_gmm (collapse=True), or biascorr.*
│         The ML estimators need sqrt(N)/T → 0 and will mislead you here.
└── No  → Do you have many controls, or nonlinear ones?
          ├── No  → Is m²/(NT) large?  (many moment conditions)
          │         ├── Yes → ablasso.ABLasso
          │         └── No  → gmm.diff_gmm
          └── Yes → What do you want to learn?
                    ├── One treatment effect       → dml.DMLDynamicPanel
                    ├── Effects across many groups → ortho.OrthogonalLasso
                    ├── All coefficients, uniformly → hdpanel.PanelLasso
                    ├── A forecast                 → shrink.PenalizedFE
                    └── Who responds over what horizon → neural.ACGate
```

Every estimator warns you when you are outside its assumptions — for example
`DMLDynamicPanel` reports `sqrt(N)/T` and warns when it exceeds 1, and
`PanelLasso` warns when the weak-sparsity assumption looks violated.

---

## Comparing estimators

The comparison table is the point of a unified package:

```python
res = {
    "FE":      dp.fixed_effects(panel, "n", lags=2, x=["w", "k"]),
    "DFE-A":   dp.debiased_fe(panel, "n", lags=2, x=["w", "k"]),
    "AH":      dp.anderson_hsiao(panel, "n", lags=1, exogenous=["w", "k"]),
    "Diff GMM": dp.diff_gmm(panel, "n", lags=2,
                            predetermined=["w"], exogenous=["k"]),
}
print(dp.comparison_table(res, params=["L1.n", "L2.n", "w", "k"]))
print(dp.comparison_to_latex(res, caption="Employment dynamics"))
```

Long-run effects with delta-method standard errors come for free:

```python
lr, se = res["Diff GMM"].long_run("w", ["L1.n", "L2.n"])
```

---

## Bundled real data

| Dataset | Shape | Use |
|---|---|---|
| `load_covid_counties()` | 2,510 US counties × 32 weeks | AB-LASSO application: school openings and COVID-19 spread |
| `load_abond_employment()` | 140 UK firms × 9 years | The canonical Arellano–Bond GMM panel |

Both load as tidy long-format frames, ready for `PanelData`.

---

## Documentation

- [`docs/user_guide.md`](docs/user_guide.md) — step-by-step, from a CSV to a
  finished table
- [`docs/syntax_reference.md`](docs/syntax_reference.md) — every estimator,
  every argument
- [`docs/methods.md`](docs/methods.md) — the econometrics behind each module
- [`examples/`](examples/) — runnable scripts reproducing each paper's headline
  exercise

---

## Installation

```bash
pip install dynpanelai              # core
pip install dynpanelai[plots]       # + matplotlib figures
pip install dynpanelai[neural]      # + PyTorch, for AC-GATE
pip install dynpanelai[all]         # everything
```

From source:

```bash
git clone https://github.com/merwanroudane/dynpanelai.git
cd dynpanelai
pip install -e ".[dev]"
pytest
```

---

## Citation

```bibtex
@software{roudane_dynpanelai_2026,
  author  = {Roudane, Merwan},
  title   = {dynpanelai: Machine Learning and Modern Inference for
             Dynamic Panel Data Models},
  year    = {2026},
  url     = {https://github.com/merwanroudane/dynpanelai},
  version = {0.1.0}
}
```

Please also cite the paper behind whichever estimator you use; each module
docstring gives the full reference.

---

## License

MIT — see [LICENSE](LICENSE).
