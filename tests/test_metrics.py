"""
Tests for metrics functions.
"""
import pytest
import xarray as xr
import numpy as np

from climate_eval.metrics import bias, spatial_correlation, temporal_autocorrelation


def create_test_data():
    """Create simple test data."""
    time = range(100)
    lat = np.linspace(30, 50, 10)
    lon = np.linspace(-120, -100, 10)
    
    pred = xr.DataArray(
        np.random.rand(100, 10, 10),
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon']
    )
    
    # Reference is similar but with some bias
    ref = pred + 0.5 + np.random.rand(100, 10, 10) * 0.1
    
    return pred, ref


def test_bias():
    """Test bias calculation."""
    pred, ref = create_test_data()
    
    result = bias(pred, ref, dim='time')
    
    assert isinstance(result, xr.DataArray)
    assert 'time' not in result.dims


def test_spatial_correlation():
    """Test spatial correlation calculation."""
    pred, ref = create_test_data()
    
    result = spatial_correlation(pred, ref, dim='time')
    
    assert isinstance(result, xr.DataArray)
    assert 'time' not in result.dims
    # Correlation should be between -1 and 1
    assert result.min() >= -1
    assert result.max() <= 1


def test_temporal_autocorrelation():
    """Test temporal autocorrelation calculation."""
    pred, _ = create_test_data()
    
    result = temporal_autocorrelation(pred, max_lag=10)
    
    assert isinstance(result, xr.DataArray)
    assert 'lag' in result.dims
    assert result.sel(lag=0).mean() == pytest.approx(1.0, abs=0.01)