"""Example 1 — Dynamic labour demand in UK firms.

Reproduces the classic Arellano and Bond (1991) exercise and compares the
within estimator, its bias corrections, and GMM on the same panel.

The point of the comparison: the within estimator understates persistence
(Nickell bias), the corrections move it back up, and the IV estimators move it
further still -- at a large cost in variance.

Run
---
    python examples/01_gmm_employment.py
"""

from __future__ import annotations

import warnings

import dynpanelai as dp

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    df = dp.datasets.load_abond_employment()
    panel = dp.PanelData(df, unit="id", time="year")

    print(panel)
    print(panel.summary().to_string(), "\n")

    print(f"sqrt(N)/T = {panel.N ** 0.5 / panel.T:.2f}")
    print("  -> far above 0: this panel is too short for the ML estimators.\n")

    results = {
        "FE": dp.fixed_effects(panel, "n", lags=2, x=["w", "k"]),
        "DFE-A": dp.debiased_fe(panel, "n", lags=2, x=["w", "k"]),
        "Kiviet": dp.bias_corrected_lsdv(panel, "n", lags=2, x=["w", "k"]),
        "AH": dp.anderson_hsiao(panel, "n", lags=1, exogenous=["w", "k"]),
        "Diff GMM (1-step)": dp.diff_gmm(
            panel, "n", lags=2, predetermined=["w"], exogenous=["k"],
            gmm_lags=(2, 4), steps=1,
        ),
        "Diff GMM (2-step)": dp.diff_gmm(
            panel, "n", lags=2, predetermined=["w"], exogenous=["k"],
            gmm_lags=(2, 4), steps=2,
        ),
    }

    print("=" * 78)
    print("Comparison of estimators")
    print("=" * 78)
    print(dp.comparison_table(results, params=["L1.n", "L2.n", "w", "k"]).to_string())

    print("\n" + "=" * 78)
    print("Preferred specification with diagnostics")
    print("=" * 78)
    gmm = results["Diff GMM (2-step)"]
    print(gmm.summary())

    lr, se = gmm.long_run("w", ["L1.n", "L2.n"])
    print(f"\nLong-run wage elasticity: {lr:.4f} (se {se:.4f})")
    print("  Impact effect understates the total adjustment because employment")
    print("  is persistent; divide by (1 - sum of lag coefficients).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from dynpanelai.report import comparison_plot

        comparison_plot(results, "L1.n")
        plt.savefig("example01_persistence.pdf")
        plt.close()
        print("\nFigure written to example01_persistence.pdf")
    except ImportError:
        print("\n(install matplotlib for figures)")

    with open("example01_table.tex", "w", encoding="utf-8") as fh:
        fh.write(
            dp.comparison_to_latex(
                results,
                params=["L1.n", "L2.n", "w", "k"],
                caption="Dynamic labour demand in UK firms",
                label="tab:employment",
            )
        )
    print("LaTeX table written to example01_table.tex")


if __name__ == "__main__":
    main()
