"""Real economics datasets bundled with dynpanelai."""

from .loaders import (
    DATA_DIR,
    available_datasets,
    load_abond_employment,
    load_covid_counties,
)

__all__ = [
    "load_covid_counties",
    "load_abond_employment",
    "available_datasets",
    "DATA_DIR",
]
