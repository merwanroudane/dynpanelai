"""Monte Carlo data-generating processes from the implemented papers."""

from .dgp import (
    make_ab_lasso_panel,
    make_heterogeneous_lag_panel,
    make_partially_linear_panel,
    make_shrinkage_panel,
)

__all__ = [
    "make_partially_linear_panel",
    "make_ab_lasso_panel",
    "make_shrinkage_panel",
    "make_heterogeneous_lag_panel",
]
