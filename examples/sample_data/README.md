# Sample Data

This directory is for small sample datasets to test ClimatExEval.

Due to file size limitations, sample data is not included in the repository.

To test the package, you can:
1. Generate synthetic data using the notebook
2. Add your own small test datasets here
3. Update paths in `example_config.yaml` to point to your data

## Expected Structure
sample_data/
├── predicted.zarr/     # Zarr format predicted data
├── reference.nc        # NetCDF format reference data
└── README.md          # This file