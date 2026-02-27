# ClimatExEval

Evaluation framework for WGAN-downscaled climate data from ClimatExML.

## Overview

ClimatExEval provides tools to evaluate downscaled climate variables across multiple metrics:
- **Marginal**: Distribution-based metrics (RMSE, bias, quantiles)
- **Spatial**: Spatial structure and patterns
- **Temporal**: Time series characteristics
- **Multivariate**: Cross-variable relationships

## 💽 Installation

Clone this repository:

```bash
git clone https://github.com/climagination/ClimatExEval.git
cd ClimatExEval
```

(Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install -e .
```

## List of metrics of interest
https://docs.google.com/document/d/1x3kdlWl1QjaxwLWncEikmhnUHU4Lw1ot1yXJ2EzYN78/edit?usp=sharing
