"""
Spatial structure metrics.
These evaluate spatial patterns and coherence.
"""
import xarray as xr


def spatial_correlation(
    pred: xr.DataArray,
    ref: xr.DataArray,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Correlation at each spatial point over time.
    
    Args:
        pred: Predicted values
        ref: Reference values
        dim: Dimension to correlate over (typically 'time')
        
    Returns:
        Spatial map of correlation coefficients
    """
    return xr.corr(pred, ref, dim=dim)