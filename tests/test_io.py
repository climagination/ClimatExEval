"""
Tests for I/O functions.
"""
import pytest
import xarray as xr
import numpy as np

from climate_eval.core import DomainConfig
from climate_eval.io import subset_domain


def test_subset_domain():
    """Test domain subsetting."""
    # Create test dataset
    ds = xr.Dataset(
        {
            'temperature': (['time', 'lat', 'lon'], 
                          np.random.rand(10, 20, 30))
        },
        coords={
            'time': range(10),
            'lat': np.linspace(0, 90, 20),
            'lon': np.linspace(-180, 180, 30)
        }
    )
    
    # Test subsetting
    domain = DomainConfig(lat_range=(20, 60))
    subset = subset_domain(ds, domain)
    
    assert subset.dims['lat'] < ds.dims['lat']
    assert subset.dims['lon'] == ds.dims['lon']