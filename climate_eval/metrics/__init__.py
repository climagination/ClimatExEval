"""
Metrics module.
Import all metric functions for easy access.
"""

# Marginal metrics
from climate_eval.metrics.marginal import (
    bias,
    quantile_comparison,
)

# Spatial metrics
from climate_eval.metrics.spatial import (
    spatial_correlation,
)

# Temporal metrics
from climate_eval.metrics.temporal import (
    temporal_autocorrelation,
)

__all__ = [
    # Marginal
    'bias',
    'quantile_comparison',
    # Spatial
    'spatial_correlation',
    # Temporal
    'temporal_autocorrelation',
]