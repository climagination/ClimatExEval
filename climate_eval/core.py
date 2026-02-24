"""
Core dataclasses for ClimatExEval.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import xarray as xr
import yaml
import logging


@dataclass
class DatasetConfig:
    """Configuration for a single dataset (predicted or reference)."""
    path: str
    format: str  # 'zarr', 'netcdf', 'pt'
    variables: List[str]
    variable_mapping: Optional[Dict[str, str]] = None
    ensemble_method: Optional[str] = 'mean'
    
    def __post_init__(self):
        self.path = Path(self.path)
        logging.info(f"📊 Dataset configured: {self.path} ({self.format})")


@dataclass
class DomainConfig:
    """Spatial and temporal domain for evaluation."""
    lat_range: Optional[tuple] = None
    lon_range: Optional[tuple] = None
    time_range: Optional[tuple] = None
    
    def is_subset(self) -> bool:
        """Check if any subsetting is specified."""
        return any([self.lat_range, self.lon_range, self.time_range])


@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute."""
    marginal: List[str] = field(default_factory=list)
    spatial: List[str] = field(default_factory=list)
    temporal: List[str] = field(default_factory=list)
    multivariate: List[str] = field(default_factory=list)
    
    def all_metrics(self) -> List[str]:
        """Get flat list of all metrics."""
        return self.marginal + self.spatial + self.temporal + self.multivariate
    
    def __post_init__(self):
        logging.info(f"📈 Metrics configured: {len(self.all_metrics())} total")


@dataclass
class ComputeConfig:
    """Configuration for computation settings."""
    use_dask: bool = True
    n_workers: int = 4
    chunk_size: str = "auto"
    
    def __post_init__(self):
        if self.use_dask:
            logging.info(f"⚡ Dask enabled with {self.n_workers} workers")


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    dir: str = "./results"
    save_intermediate: bool = True
    formats: List[str] = field(default_factory=lambda: ["png"])
    dpi: int = 300
    
    def __post_init__(self):
        self.dir = Path(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"💾 Output directory: {self.dir}")


@dataclass
class EvaluationConfig:
    """Main configuration class for evaluation."""
    project_name: str
    description: str
    data_predicted: DatasetConfig
    data_reference: DatasetConfig
    domain: DomainConfig
    metrics: MetricsConfig
    compute: ComputeConfig
    output: OutputConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logging.info(f"🔧 Loading configuration from {path}")
        
        # Extract and clean data configs (remove None values)
        pred_config = {k: v for k, v in config_dict['data']['predicted'].items() if v is not None}
        ref_config = {k: v for k, v in config_dict['data']['reference'].items() if v is not None}
        
        return cls(
            project_name=config_dict['project_name'],
            description=config_dict.get('description', ''),
            data_predicted=DatasetConfig(**pred_config),
            data_reference=DatasetConfig(**ref_config),
            domain=DomainConfig(**config_dict.get('domain', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            compute=ComputeConfig(**config_dict.get('compute', {})),
            output=OutputConfig(**config_dict.get('output', {})),
        )
    
    def __post_init__(self):
        logging.info(f"🌍 Evaluation project: {self.project_name}")
        logging.info(f"   {self.description}")


@dataclass
class LoadedDataset:
    """Container for loaded climate data with metadata."""
    data: xr.Dataset
    config: DatasetConfig
    type: str  # 'predicted' or 'reference'
    
    def __getitem__(self, key):
        """Allow direct access to variables like dataset['temperature']."""
        return self.data[key]
    
    def __repr__(self):
        return f"LoadedDataset(type={self.type}, variables={list(self.data.data_vars)}, shape={dict(self.data.sizes)})"


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_spatial(self) -> bool:
        """Check if result has spatial dimensions."""
        if isinstance(self.value, (xr.DataArray, xr.Dataset)):
            return any(dim in self.value.dims for dim in ['lat', 'lon', 'latitude', 'longitude'])
        return False
    
    def is_scalar(self) -> bool:
        """Check if result is a single number."""
        return isinstance(self.value, (int, float))


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""
    config: EvaluationConfig
    results: Dict[str, MetricResult] = field(default_factory=dict)
    
    def add_result(self, name: str, value: Any, metadata: Optional[Dict] = None):
        """Add a metric result."""
        self.results[name] = MetricResult(
            name=name,
            value=value,
            metadata=metadata or {}
        )
        logging.info(f"✅ Computed metric: {name}")
    
    def get_result(self, name: str) -> MetricResult:
        """Get a specific metric result."""
        return self.results[name]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of scalar results."""
        return {
            name: float(result.value) if hasattr(result.value, 'item') else result.value
            for name, result in self.results.items() 
            if result.is_scalar()
        }
    
    def spatial_results(self) -> Dict[str, MetricResult]:
        """Get all spatial results."""
        return {
            name: result 
            for name, result in self.results.items() 
            if result.is_spatial()
        }
    
    def __repr__(self):
        return f"EvaluationResults(project={self.config.project_name}, metrics={len(self.results)})"