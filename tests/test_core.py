"""
Tests for core dataclasses.
"""
import pytest
from pathlib import Path
import tempfile
import yaml

from climate_eval.core import (
    DatasetConfig,
    DomainConfig,
    MetricsConfig,
    EvaluationConfig
)


def test_dataset_config():
    """Test DatasetConfig initialization."""
    config = DatasetConfig(
        path="/path/to/data.zarr",
        format="zarr",
        variables=["temperature", "precipitation"]
    )
    
    assert config.format == "zarr"
    assert len(config.variables) == 2
    assert isinstance(config.path, Path)


def test_domain_config():
    """Test DomainConfig."""
    # No subsetting
    config = DomainConfig()
    assert not config.is_subset()
    
    # With subsetting
    config = DomainConfig(lat_range=(30, 50))
    assert config.is_subset()


def test_metrics_config():
    """Test MetricsConfig."""
    config = MetricsConfig(
        marginal=['bias'],
        spatial=['spatial_correlation']
    )
    
    assert len(config.all_metrics()) == 2
    assert 'bias' in config.marginal