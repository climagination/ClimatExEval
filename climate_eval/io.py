"""
Input/output utilities for loading and saving evaluation data.
"""
import xarray as xr
import zarr
import torch
from pathlib import Path
from typing import Optional, Dict
import yaml
import logging

from climate_eval.core import (
    EvaluationConfig, 
    DatasetConfig, 
    LoadedDataset,
    EvaluationResults
)


def load_config(config_path: str = "config.yaml") -> EvaluationConfig:
    """
    Load configuration file and return EvaluationConfig object.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        EvaluationConfig object
    """
    return EvaluationConfig.from_yaml(config_path)


def load_dataset(dataset_config: DatasetConfig, dataset_type: str = "unknown") -> LoadedDataset:
    """
    Load climate data from configuration.
    
    Args:
        dataset_config: DatasetConfig object
        dataset_type: 'predicted' or 'reference'
        
    Returns:
        LoadedDataset object
    """
    from climate_eval.utils import handle_ensemble_dimension  # Import here to avoid circular import
    
    path = dataset_config.path
    format = dataset_config.format
    
    if format == "zarr":
        ds = xr.open_zarr(path)
    elif format == "netcdf":
        ds = xr.open_dataset(path)
    elif format == "pt":
        ds = _load_pytorch_files(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Select only requested variables
    if dataset_config.variables:
        ds = ds[dataset_config.variables]
    
    # Handle ensemble dimension if method is specified
    if dataset_config.ensemble_method is not None:
        ds = handle_ensemble_dimension(ds, method=dataset_config.ensemble_method)
    
    logging.info(f"📥 Loaded {dataset_type} data: {dict(ds.sizes)}")
    
    return LoadedDataset(
        data=ds,
        config=dataset_config,
        type=dataset_type
    )


def _load_pytorch_files(directory: Path) -> xr.Dataset:
    """
    Load PyTorch files from preprocessing repo.
    
    This is a placeholder - adapt to your preprocessing output structure.
    """
    raise NotImplementedError(
        "PyTorch file loading needs to be adapted to your preprocessing format. "
        "See nc2pt output structure for implementation details."
    )


def subset_domain(ds: xr.Dataset, domain_config) -> xr.Dataset:
    """
    Subset dataset using DomainConfig.
    
    Args:
        ds: Input dataset
        domain_config: DomainConfig object
        
    Returns:
        Subsetted dataset
    """
    if not domain_config.is_subset():
        return ds
    
    logging.info("✂️  Subsetting domain...")
    
    if domain_config.lat_range:
        ds = ds.sel(lat=slice(*domain_config.lat_range))
    if domain_config.lon_range:
        ds = ds.sel(lon=slice(*domain_config.lon_range))
    if domain_config.time_range:
        ds = ds.sel(time=slice(*domain_config.time_range))
    
    logging.info(f"   New shape: {dict(ds.dims)}")
    return ds


def align_datasets(
    pred: LoadedDataset,
    ref: LoadedDataset
) -> tuple[LoadedDataset, LoadedDataset]:
    """
    Align predicted and reference datasets.
    
    - Renames variables according to mapping
    - Ensures same spatial/temporal coordinates
    
    Args:
        pred: Predicted LoadedDataset
        ref: Reference LoadedDataset
        
    Returns:
        (aligned_pred, aligned_ref)
    """
    logging.info("🔄 Aligning datasets...")
    
    # Rename reference variables if mapping provided
    if ref.config.variable_mapping:
        ref.data = ref.data.rename(ref.config.variable_mapping)
        logging.info(f"   Renamed variables: {ref.config.variable_mapping}")
    
    # Get common variables
    common_vars = list(set(pred.data.data_vars) & set(ref.data.data_vars))
    pred.data = pred.data[common_vars]
    ref.data = ref.data[common_vars]
    logging.info(f"   Common variables: {common_vars}")
    
    # Check if grids match
    pred_dims = set(pred.data.dims)
    ref_dims = set(ref.data.dims)
    
    if pred_dims == ref_dims:
        # Check if coordinate values match
        coords_match = True
        for dim in pred_dims:
            if dim in ref.data.dims:
                # Check if coordinates are close enough
                if not pred.data[dim].equals(ref.data[dim]):
                    # Check if they're at least the same length
                    if len(pred.data[dim]) != len(ref.data[dim]):
                        coords_match = False
                        break
        
        if coords_match:
            logging.info(f"   Grids already aligned")
            return pred, ref
    
    # Try to align coordinates
    try:
        # Only interpolate if we have 1D coordinates
        # Check if lat/lon are 1D
        has_1d_coords = True
        for coord in ['lat', 'lon', 'time']:
            if coord in ref.data.coords and ref.data[coord].ndim > 1:
                has_1d_coords = False
                break
        
        if has_1d_coords:
            ref.data = ref.data.interp_like(pred.data, method='linear')
            logging.info(f"   Aligned coordinates via interpolation")
        else:
            logging.warning(f"   Cannot auto-align: coordinates are not 1D")
            logging.warning(f"   Predicted dims: {dict(pred.data.sizes)}")
            logging.warning(f"   Reference dims: {dict(ref.data.sizes)}")
            logging.warning(f"   Proceeding without alignment - metrics may fail if grids don't match")
    except Exception as e:
        logging.warning(f"   Could not align grids: {e}")
        logging.warning(f"   Proceeding without alignment - metrics may fail if grids don't match")
    
    return pred, ref


def save_results(
    results: EvaluationResults,
    name: str = "evaluation"
):
    """
    Save evaluation results to configured output directory.
    
    Args:
        results: EvaluationResults object
        name: Base name for output files
    """
    output_dir = results.config.output.dir / results.config.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"💾 Saving results to {output_dir}")
    
    # Save each result
    for metric_name, result in results.results.items():
        if isinstance(result.value, (xr.Dataset, xr.DataArray)):
            output_file = output_dir / f"{name}_{metric_name}.nc"
            result.value.to_netcdf(output_file)
            logging.info(f"   Saved {metric_name} to {output_file.name}")
    
    # Save scalar summary
    summary = results.summary()
    if summary:
        summary_file = output_dir / f"{name}_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f)
        logging.info(f"   Saved summary to {summary_file.name}")