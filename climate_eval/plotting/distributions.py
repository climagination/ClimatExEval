"""
Distribution visualization functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Tuple


def plot_histogram_comparison(
    pred: xr.DataArray,
    ref: xr.DataArray,
    bins: int = 50,
    title: str = "Distribution Comparison",
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot overlaid histograms of predicted and reference data.
    
    Args:
        pred: Predicted values
        ref: Reference values
        bins: Number of bins
        title: Plot title
        figsize: Figure size
        
    Returns:
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pred_flat = pred.values.flatten()
    ref_flat = ref.values.flatten()
    
    # Remove NaNs
    pred_flat = pred_flat[~np.isnan(pred_flat)]
    ref_flat = ref_flat[~np.isnan(ref_flat)]
    
    ax.hist(ref_flat, bins=bins, alpha=0.5, label='Reference', density=True, **kwargs)
    ax.hist(pred_flat, bins=bins, alpha=0.5, label='Predicted', density=True, **kwargs)
    
    ax.set_xlabel(pred.name or 'Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax