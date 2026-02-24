"""
Spatial visualization functions.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
from typing import Optional, Tuple


def plot_spatial_field(
    data: xr.DataArray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 6),
    add_coastlines: bool = True,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a spatial field on a map.
    
    Args:
        data: 2D spatial field (lat, lon)
        title: Plot title
        cmap: Colormap
        vmin, vmax: Color scale limits
        figsize: Figure size
        add_coastlines: Whether to add coastline features
        
    Returns:
        (figure, axes)
    """
    # Load data if it's dask-backed
    if hasattr(data.data, 'chunks'):
        data = data.load()
    
    # Compute vmin/vmax if they're DataArrays (from operations like abs().max())
    if vmin is not None and hasattr(vmin, 'compute'):
        vmin = float(vmin.compute())
    if vmax is not None and hasattr(vmax, 'compute'):
        vmax = float(vmax.compute())
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Plot data
    im = data.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        add_colorbar=True,
        cbar_kwargs={'label': data.name or ''},
        **kwargs
    )
    
    if add_coastlines:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    ax.set_title(title, fontsize=14)
    ax.gridlines(draw_labels=True)
    
    plt.tight_layout()
    return fig, ax