from setuptools import setup

setup(
    name="climatexeval",
    version="0.1.0",
    author="Seamus Beairsto",
    author_email="seamus.beairsto@gmail.com",
    packages=["climate_eval"],
    license="LICENSE",
    description="Python module to evaluate climate downscaling models.",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "xarray",
        "dask",
        "zarr",
        "netCDF4",
        "torch",
        "seaborn",
        "cartopy",
        "pyyaml",
    ],
)