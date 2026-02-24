"""
Temporal metrics.
These evaluate time series behavior.
"""
import xarray as xr
import numpy as np


def temporal_autocorrelation(
    data: xr.DataArray,
    max_lag: int = 30,
    dim: str = 'time'
) -> xr.DataArray:
    """
    Compute autocorrelation function.
    
    Args:
        data: Input data
        max_lag: Maximum lag to compute
        dim: Time dimension
        
    Returns:
        Autocorrelation at each lag
    """
    def autocorr_1d(x, max_lag):
        """Compute autocorrelation for 1D array."""
        x = x - np.nanmean(x)
        c0 = np.nansum(x * x)
        acf = np.array([
            np.nansum(x[:-lag] * x[lag:]) / c0
            if lag > 0 else 1.0
            for lag in range(max_lag + 1)
        ])
        return acf
    
    # Load data into memory if it's dask-backed
    # (ACF computation is not easily parallelizable)
    if hasattr(data.data, 'chunks'):
        import logging
        logging.info("   Loading data into memory for ACF computation...")
        data = data.load()
    
    # Apply along time dimension
    result = xr.apply_ufunc(
        autocorr_1d,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[['lag']],
        vectorize=True,
        kwargs={'max_lag': max_lag},
        output_dtypes=[float]
    )
    
    result = result.assign_coords(lag=np.arange(max_lag + 1))
    return result