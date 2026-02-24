"""
ClimatExEval: Evaluation framework for climate downscaling models.

Part of the Climagination climate downscaling toolkit.
"""

__version__ = "0.1.0"
__author__ = "Seamus Beairsto"

# Import main modules for easy access
from climate_eval import core
from climate_eval import io
from climate_eval import metrics
from climate_eval import plotting
from climate_eval import utils

# Import key classes for convenience
from climate_eval.core import (
    EvaluationConfig,
    LoadedDataset,
    EvaluationResults,
)

__all__ = [
    "core",
    "io",
    "metrics",
    "plotting",
    "utils",
    "EvaluationConfig",
    "LoadedDataset",
    "EvaluationResults",
    "__version__",
]