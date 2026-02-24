"""
Helper utilities for ClimatExEval.
"""
import numpy as np
import xarray as xr
import logging
from typing import Optional


def setup_logging(level: str = "INFO"):
    """
    Configure logging for ClimatExEval.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(message)s'
    )


def detect_dimension_names(ds: xr.Dataset) -> dict:
    """
    Detect common dimension names in dataset.
    
    Returns dictionary with standardized names mapping to actual names.
    """
    dim_map = {}
    
    # Latitude
    for lat_name in ['lat', 'latitude', 'y', 'rlat']:
        if lat_name in ds.dims:
            dim_map['lat'] = lat_name
            break
    
    # Longitude
    for lon_name in ['lon', 'longitude', 'x', 'rlon']:
        if lon_name in ds.dims:
            dim_map['lon'] = lon_name
            break
    
    # Time
    for time_name in ['time', 't', 'date']:
        if time_name in ds.dims:
            dim_map['time'] = time_name
            break
    
    return dim_map


def standardize_dimension_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename dimensions to standard names (lat, lon, time).
    
    Args:
        ds: Input dataset
        
    Returns:
        Dataset with standardized dimension names
    """
    dim_map = detect_dimension_names(ds)
    
    # Invert the map for renaming
    rename_dict = {v: k for k, v in dim_map.items() if v != k}
    
    if rename_dict:
        logging.info(f"Renaming dimensions: {rename_dict}")
        ds = ds.rename(rename_dict)
    
    return ds


def handle_ensemble_dimension(
    ds: xr.Dataset,
    method: str = 'mean',
    realization_dim: str = 'realization'
) -> xr.Dataset:
    """
    Handle ensemble/realization dimension in stochastic model output.
    
    Args:
        ds: Dataset potentially with realization dimension
        method: How to handle ensemble ('mean', 'median', 'select', or 'keep')
        realization_dim: Name of realization dimension
        
    Returns:
        Dataset with realization dimension handled
    """
    if realization_dim not in ds.dims:
        logging.info(f"   No '{realization_dim}' dimension found")
        return ds
    
    n_realizations = ds.sizes[realization_dim]
    logging.info(f"   Found {n_realizations} realizations")
    
    if method == 'mean':
        logging.info(f"   Taking ensemble mean")
        return ds.mean(dim=realization_dim)
    elif method == 'median':
        logging.info(f"   Taking ensemble median")
        return ds.median(dim=realization_dim)
    elif method == 'select':
        logging.info(f"   Selecting first realization")
        return ds.isel({realization_dim: 0})
    elif method == 'keep':
        logging.info(f"   Keeping all realizations")
        return ds
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', 'select', or 'keep'")