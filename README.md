# ClimatExEval

Evaluation framework for WGAN-downscaled climate data from ClimatExML.

## Overview

ClimatExEval provides tools to evaluate downscaled climate variables across multiple metrics:
- **Marginal**: Distribution-based metrics (RMSE, bias, quantiles)
- **Spatial**: Spatial structure and patterns
- **Temporal**: Time series characteristics
- **Multivariate**: Cross-variable relationships

## Installation

```bash
git clone https://github.com/Climagination/ClimatExEval.git
cd ClimatExEval
pip install -e .
