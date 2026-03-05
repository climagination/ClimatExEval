"""
Distribution visualization functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Tuple


def plot_histogram_comparison(
    pred: xr.DataArray,
    ref: xr.DataArray,
    bins: int = 50,
    title: str = "Distribution Comparison",
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot overlaid histograms of predicted and reference data.
    
    Args:
        pred: Predicted values
        ref: Reference values
        bins: Number of bins
        title: Plot title
        figsize: Figure size
        
    Returns:
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pred_flat = pred.values.flatten()
    ref_flat = ref.values.flatten()
    
    # Remove NaNs
    pred_flat = pred_flat[~np.isnan(pred_flat)]
    ref_flat = ref_flat[~np.isnan(ref_flat)]
    
    ax.hist(ref_flat, bins=bins, label='Reference', density=True, histtype=u'step',  **kwargs)
    ax.hist(pred_flat, bins=bins, label='Predicted', density=True, histtype=u'step',  **kwargs)
    
    ax.set_xlabel(pred.name or 'Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_qq(
    qq_dataset: xr.Dataset,
    title: str = "Q-Q Plot",
    figsize: Tuple[int, int] = (10, 10),
    add_stats: bool = True,
    point_size: int = 20,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quantile-quantile plot from pre-computed Q-Q data.
    
    Args:
        qq_dataset: Dataset from metrics.qq_data() with 'predicted' and 'reference'
        title: Plot title
        figsize: Figure size
        add_stats: Whether to add statistical annotations
        point_size: Size of scatter points
        
    Returns:
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pred_q = qq_dataset['predicted'].values
    ref_q = qq_dataset['reference'].values
    
    # Scatter plot with larger points
    ax.scatter(ref_q, pred_q, alpha=0.6, s=point_size, edgecolors='none', **kwargs)
    
    # 1:1 line
    min_val = min(ref_q.min(), pred_q.min())
    max_val = max(ref_q.max(), pred_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5, label='1:1 line', zorder=5)
    
    ax.set_xlabel('Reference Quantiles', fontsize=12)
    ax.set_ylabel('Predicted Quantiles', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics text box
    if add_stats:
        rmse = np.sqrt(np.mean((pred_q - ref_q)**2))
        bias = np.mean(pred_q - ref_q)
        stats_text = (
            f'RMSE = {rmse:.4f}\n'
            f'Bias = {bias:.4f}\n'
            f'N points = {len(pred_q):,}'
        )
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    return fig, ax


def plot_qq_with_residuals(
    qq_dataset: xr.Dataset,
    title: str = "Q-Q Analysis",
    figsize: Tuple[int, int] = (14, 6),
    point_size: int = 20,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Q-Q plot with residual analysis.
    
    The residual plot shows (Predicted - Reference) vs Reference quantile.
    This helps identify:
    - Systematic over/under-prediction at different parts of the distribution
    - Whether model struggles with extremes (tails) or central values
    - Patterns in model bias (e.g., consistent over-prediction at high values)
    
    Args:
        qq_dataset: Dataset from metrics.qq_data() with 'predicted' and 'reference'
        title: Plot title
        figsize: Figure size
        point_size: Size of scatter points
        
    Returns:
        (figure, axes array)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    pred_q = qq_dataset['predicted'].values
    ref_q = qq_dataset['reference'].values
    quantile_levels = qq_dataset['quantile'].values
    
    # Left: Q-Q plot
    ax1.scatter(ref_q, pred_q, alpha=0.6, s=point_size, edgecolors='none')
    min_val = min(ref_q.min(), pred_q.min())
    max_val = max(ref_q.max(), pred_q.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5, label='1:1 line')
    ax1.set_xlabel('Reference Quantiles', fontsize=12)
    ax1.set_ylabel('Predicted Quantiles', fontsize=12)
    ax1.set_title('Q-Q Plot', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right: Residuals (Predicted - Reference)
    residuals = pred_q - ref_q
    scatter = ax2.scatter(ref_q, residuals, alpha=0.6, s=point_size, 
                         c=quantile_levels, cmap='viridis', edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='-', linewidth=1.5, label='Zero residual')
    ax2.set_xlabel('Reference Quantiles', fontsize=12)
    ax2.set_ylabel('Residuals (Pred - Ref)', fontsize=12)
    ax2.set_title('Q-Q Residuals', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for quantile levels
    cbar = plt.colorbar(scatter, ax=ax2, label='Quantile Level')
    
    fig.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    return fig, (ax1, ax2)