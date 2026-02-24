"""
Time series visualization functions.
"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from typing import Tuple


def plot_acf(
    acf: xr.DataArray,
    title: str = "Autocorrelation Function",
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot autocorrelation function.
    
    Args:
        acf: Autocorrelation values with 'lag' dimension
        title: Plot title
        figsize: Figure size
        
    Returns:
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Average over space if spatial dimensions present
    spatial_dims = [d for d in ['lat', 'lon'] if d in acf.dims]
    if spatial_dims:
        acf_mean = acf.mean(dim=spatial_dims)
    else:
        acf_mean = acf
    
    lag = acf_mean.lag.values
    
    ax.plot(lag, acf_mean.values, 'o-', **kwargs)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax