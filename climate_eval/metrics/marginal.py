"""
Marginal (distribution-based) metrics.
These operate on the statistical distribution of values.
"""
import xarray as xr
import numpy as np
from typing import Optional, List


def _get_spatial_dims(data: xr.DataArray) -> List[str]:
    """Helper to detect spatial dimension names."""
    spatial_dims = []
    for dim in data.dims:
        if dim in ['lat', 'latitude', 'y', 'rlat']:
            spatial_dims.append(dim)
        elif dim in ['lon', 'longitude', 'x', 'rlon']:
            spatial_dims.append(dim)
    return spatial_dims


def bias(pred: xr.DataArray, ref: xr.DataArray, dim: Optional[List[str]] = None) -> xr.DataArray:
    """
    Mean bias.
    
    Args:
        pred: Predicted values
        ref: Reference values
        dim: Dimensions to compute over (None = all)
        
    Returns:
        Bias value(s)
    """
    return (pred - ref).mean(dim=dim)


def quantile_comparison(
    pred: xr.DataArray,
    ref: xr.DataArray,
    quantiles: List[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
) -> xr.Dataset:
    """
    Compare quantiles between predicted and reference.
    
    Args:
        pred: Predicted values
        ref: Reference values
        quantiles: List of quantiles to compute
        
    Returns:
        Dataset with predicted and reference quantiles
    """
    # Get spatial dimensions dynamically
    spatial_dims = _get_spatial_dims(pred)
    
    # Get time dimension
    time_dim = 'time' if 'time' in pred.dims else None
    
    # Build dimension list
    dims_to_reduce = []
    if time_dim:
        dims_to_reduce.append(time_dim)
    dims_to_reduce.extend(spatial_dims)
    
    pred_q = pred.quantile(quantiles, dim=dims_to_reduce)
    ref_q = ref.quantile(quantiles, dim=dims_to_reduce)
    
    return xr.Dataset({
        'predicted': pred_q,
        'reference': ref_q,
        'difference': pred_q - ref_q
    })