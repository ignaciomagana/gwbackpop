"""Experimental emulator-based BackPop selection prototype.

This package is intentionally outside the production BackPop import path.
"""

from .api import (
    CosmicInjectionCatalog,
    GaussianConditionalEmulator,
    estimate_selection_alpha,
    featurize_hyperparameters,
    generate_cosmic_injection_catalog,
    load_cosmic_injection_catalog,
    sample_predicted_mergers,
)

__all__ = [
    "CosmicInjectionCatalog",
    "GaussianConditionalEmulator",
    "estimate_selection_alpha",
    "featurize_hyperparameters",
    "generate_cosmic_injection_catalog",
    "load_cosmic_injection_catalog",
    "sample_predicted_mergers",
]
