"""Core panel infrastructure: containers, transforms, variance, results."""

from .panel import PanelData, PanelSpec
from .results import PanelResults, stars
from .transforms import (
    apply_transform,
    first_difference,
    fod_matrix,
    forward_orthogonal_deviation,
    mundlak_means,
    within_transform,
)
from .variance import (
    cluster_variance,
    driscoll_kraay_variance,
    newey_west_panel_variance,
    sandwich,
    twoway_cluster_variance,
    windmeijer_correction,
)

__all__ = [
    "PanelData",
    "PanelSpec",
    "PanelResults",
    "stars",
    "within_transform",
    "first_difference",
    "forward_orthogonal_deviation",
    "fod_matrix",
    "mundlak_means",
    "apply_transform",
    "cluster_variance",
    "twoway_cluster_variance",
    "driscoll_kraay_variance",
    "newey_west_panel_variance",
    "sandwich",
    "windmeijer_correction",
]
