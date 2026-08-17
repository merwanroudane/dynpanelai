"""Double machine learning for dynamic panels (Sneller, 2026)."""

from .estimator import DMLDynamicPanel, dml_dynamic_panel
from .folds import (
    Fold,
    blocked_time_folds,
    buffer_rules,
    clustered_unit_folds,
    nlo_folds,
    suggest_buffer_acf,
)

__all__ = [
    "DMLDynamicPanel",
    "dml_dynamic_panel",
    "Fold",
    "blocked_time_folds",
    "nlo_folds",
    "clustered_unit_folds",
    "buffer_rules",
    "suggest_buffer_acf",
]
