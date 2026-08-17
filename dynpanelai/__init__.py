"""dynpanelai: machine learning and modern inference for dynamic panel data.

A unified implementation of seven methodologies for dynamic panels with
high-dimensional controls, all sharing one data container, one results object,
and one reporting layer.

=========================  ====================================================
Module                     Method and source
=========================  ====================================================
:mod:`~dynpanelai.gmm`     Difference / system GMM, Anderson-Hsiao.
                           Arellano and Bond (1991); Blundell and Bond (1998).
:mod:`~dynpanelai.biascorr`  Analytical, split-panel and Kiviet bias
                           corrections for the within estimator.
:mod:`~dynpanelai.ablasso` Arellano-Bond LASSO.  Chernozhukov, Fernandez-Val,
                           Huang and Wang (2024).
:mod:`~dynpanelai.dml`     Double machine learning with blocked-time
                           cross-fitting.  Sneller (2026).
:mod:`~dynpanelai.ortho`   Orthogonal and debiased Lasso for high-dimensional
                           CATE.  Semenova, Goldman, Chernozhukov and Taddy.
:mod:`~dynpanelai.hdpanel` Uniform inference with weakly sparse fixed effects.
                           Kock and Tang (2019).
:mod:`~dynpanelai.shrink`  URE / Empirical Bayes shrinkage and penalised-FE
                           forecasting.  Kwon (2026); Cornejo and
                           Sosa-Escudero (2026).
:mod:`~dynpanelai.neural`  AC-GATE entity-conditioned lag discovery with the
                           L0-L3 audit protocol.  Xu (2026).
=========================  ====================================================

Quick start
-----------
>>> import dynpanelai as dp
>>> df = dp.datasets.load_abond_employment()
>>> panel = dp.PanelData(df, unit="id", time="year")
>>> res = dp.diff_gmm(panel, y="n", lags=2,
...                   predetermined=["w"], exogenous=["k"])   # doctest: +SKIP
>>> print(res.summary())                                      # doctest: +SKIP

Choosing a method
-----------------
Start from the shape of your panel and what you want to learn.

- **Short T (under ~15), low-dimensional controls** -- use
  :func:`~dynpanelai.gmm.diff_gmm` or
  :func:`~dynpanelai.gmm.system_gmm`.  The ML estimators here need
  ``sqrt(N)/T -> 0`` and will mislead you in short panels.
- **Long T, many moment conditions** -- use
  :class:`~dynpanelai.ablasso.ABLasso`.  Check whether ``m^2/(NT)`` is large;
  if it is, plain Arellano-Bond is biased.
- **High-dimensional or nonlinear controls, one treatment effect** -- use
  :class:`~dynpanelai.dml.DMLDynamicPanel`.
- **Heterogeneous effects across many groups** -- use
  :class:`~dynpanelai.ortho.OrthogonalLasso`.
- **Forecasting rather than inference** -- use
  :class:`~dynpanelai.shrink.PenalizedFE`.
- **Which units respond over what horizon** -- use
  :class:`~dynpanelai.neural.ACGate`.

See ``docs/user_guide.md`` for the step-by-step walkthrough.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Merwan Roudane"

from . import (
    ablasso,
    biascorr,
    datasets,
    dml,
    gmm,
    hdpanel,
    neural,
    ortho,
    penalized,
    report,
    shrink,
    sim,
)
from .ablasso import ABLasso, ab_lasso
from .biascorr import (
    bias_corrected_lsdv,
    debiased_fe,
    fixed_effects,
    split_panel_jackknife,
)
from .core import (
    PanelData,
    PanelResults,
    PanelSpec,
    first_difference,
    forward_orthogonal_deviation,
    within_transform,
)
from .dml import DMLDynamicPanel, dml_dynamic_panel
from .gmm import DynamicPanelGMM, anderson_hsiao, diff_gmm, system_gmm
from .hdpanel import PanelLasso, panel_lasso
from .ortho import OrthogonalLasso, orthogonal_lasso
from .penalized import clime, rlasso
from .report import comparison_table, comparison_to_latex, monte_carlo_table
from .shrink import PenalizedFE, fe_shrink, forecast_metrics

__all__ = [
    "__version__",
    # containers
    "PanelData",
    "PanelSpec",
    "PanelResults",
    # transforms
    "within_transform",
    "first_difference",
    "forward_orthogonal_deviation",
    # estimators
    "DynamicPanelGMM",
    "diff_gmm",
    "system_gmm",
    "anderson_hsiao",
    "fixed_effects",
    "debiased_fe",
    "split_panel_jackknife",
    "bias_corrected_lsdv",
    "ABLasso",
    "ab_lasso",
    "DMLDynamicPanel",
    "dml_dynamic_panel",
    "OrthogonalLasso",
    "orthogonal_lasso",
    "PanelLasso",
    "panel_lasso",
    "PenalizedFE",
    "fe_shrink",
    "forecast_metrics",
    # building blocks
    "rlasso",
    "clime",
    # reporting
    "comparison_table",
    "comparison_to_latex",
    "monte_carlo_table",
    # subpackages
    "ablasso",
    "biascorr",
    "datasets",
    "dml",
    "gmm",
    "hdpanel",
    "neural",
    "ortho",
    "penalized",
    "report",
    "shrink",
    "sim",
]
