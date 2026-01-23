"""
Visualization utilities for ML experiment results.
Creates heatmaps to visualize model predictions across the feature space.
Style reference: Purple → Cyan → Green → Yellow colormap with data point markers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def create_viridis_like_colormap():
    """
    Creates a colormap similar to the reference image.
    Purple → Blue → Cyan → Green → Yellow
    """
    colors = [
        '#440154',  # Dark purple
        '#3b528b',  # Blue-purple
        '#21918c',  # Cyan/Teal
        '#5ec962',  # Green
        '#fde725'   # Yellow
    ]
    cmap = LinearSegmentedColormap.from_list('viridis_like', colors, N=256)
    return cmap


def train_best_model(df, target_col='y1'):
    """
    Trains the best model (XGBoost Regressor) on the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing x1, x2, and target columns
    target_col : str
        Target column name (default: 'y1' for Rg)
    
    Returns:
    --------
    model : fitted XGBoost model
    data : DataFrame used for training (for plotting data points)
    """
    if XGBRegressor is None:
        raise ImportError("XGBoost is required. Install with: pip install xgboost")
    
    # Drop rows where target is missing
    data = df.dropna(subset=[target_col]).copy()
    X = data[['x1', 'x2']]
    y = data[target_col]
    
    # Train XGBoost (best performing model based on results)
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X, y)
    
    print(f"Model trained on {len(X)} samples")
    print(f"Target: {target_col} (Rg - Radius of Gyration)")
    
    return model, data


def generate_prediction_grid(model, x1_range=(1, 100), x2_range=(1, 100), resolution=100):
    """
    Generates a grid of predictions for the heatmap.
    
    Parameters:
    -----------
    model : fitted model
        The trained model
    x1_range : tuple
        (min, max) range for x1 (IL/EAN concentration)
    x2_range : tuple
        (min, max) range for x2 (Protein concentration)
    resolution : int
        Number of points along each axis
    
    Returns:
    --------
    X1_grid, X2_grid, Y_pred : np.ndarray
        Meshgrid arrays and predictions
    """
    x1_values = np.linspace(x1_range[0], x1_range[1], resolution)
    x2_values = np.linspace(x2_range[0], x2_range[1], resolution)
    
    X1_grid, X2_grid = np.meshgrid(x1_values, x2_values)
    
    # Flatten for prediction
    X_flat = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    
    # Predict
    Y_pred_flat = model.predict(X_flat)
    
    # Reshape back to grid
    Y_pred = Y_pred_flat.reshape(X1_grid.shape)
    
    return X1_grid, X2_grid, Y_pred


def plot_heatmap_styled(X1_grid, X2_grid, Y_pred, 
                        data_df=None,
                        y2_col='y2',
                        save_path=None,
                        figsize=(10, 8),
                        dpi=150):
    """
    Creates a styled heatmap matching the reference image.
    
    Features:
    - Purple → Cyan → Green → Yellow colormap
    - X2 (Protein) on horizontal axis
    - X1 (Concentration of IL) on vertical axis  
    - Rg (Å) on colorbar
    - Data point markers (gray squares, red squares for crystalline)
    - White grid lines
    
    Parameters:
    -----------
    X1_grid, X2_grid : np.ndarray
        Meshgrid arrays for X1 and X2
    Y_pred : np.ndarray
        Predicted Y1 (Rg) values
    data_df : pd.DataFrame, optional
        Original data for plotting data points
    y2_col : str
        Column name for crystalline classification (for coloring markers)
    save_path : str
        Path to save the figure (optional)
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create colormap (viridis-like: purple → cyan → green → yellow)
    cmap = create_viridis_like_colormap()
    
    # Create figure with specific styling
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot heatmap (X2 on horizontal, X1 on vertical)
    im = ax.pcolormesh(X2_grid, X1_grid, Y_pred, 
                       cmap=cmap, 
                       shading='auto')
    
    # Add white grid lines (styled like reference)
    x2_unique = np.unique(X2_grid)
    x1_unique = np.unique(X1_grid)
    
    # Add grid lines at reasonable intervals
    x2_ticks = [1, 5, 20, 50, 70, 100]
    x1_ticks = [1, 5, 10, 20, 30, 40, 55, 60, 65, 70, 75, 80, 100]
    
    for x in x2_ticks:
        if x >= X2_grid.min() and x <= X2_grid.max():
            ax.axvline(x=x, color='white', linewidth=0.5, alpha=0.5)
    
    for y in x1_ticks:
        if y >= X1_grid.min() and y <= X1_grid.max():
            ax.axhline(y=y, color='white', linewidth=0.5, alpha=0.5)
    
    # Plot data points if provided
    if data_df is not None:
        # Check if y2 column exists for differentiating crystalline/non-crystalline
        if y2_col in data_df.columns:
            # Non-crystalline (y2=0): gray squares
            non_crystal = data_df[data_df[y2_col] == 0]
            ax.scatter(non_crystal['x2'], non_crystal['x1'], 
                      marker='s', s=30, c='#404040', 
                      edgecolors='none', alpha=0.8, zorder=5,
                      label='Non-crystalline')
            
            # Crystalline (y2=1): red squares
            crystal = data_df[data_df[y2_col] == 1]
            ax.scatter(crystal['x2'], crystal['x1'], 
                      marker='s', s=30, c='#FF0000', 
                      edgecolors='none', alpha=0.8, zorder=5,
                      label='Crystalline')
        else:
            # All points: gray squares
            ax.scatter(data_df['x2'], data_df['x1'], 
                      marker='s', s=30, c='#404040', 
                      edgecolors='none', alpha=0.8, zorder=5)
    
    # Colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Rg (Å)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Get y range for colorbar ticks
    y_min, y_max = Y_pred.min(), Y_pred.max()
    cbar_ticks = np.linspace(int(y_min), int(y_max) + 1, 
                             min(7, int(y_max - y_min) + 2))
    cbar.set_ticks(cbar_ticks)
    
    # Axis labels (matching reference style)
    ax.set_xlabel('Protein (mg/mL)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration of IL (wt%)', fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(X2_grid.min(), X2_grid.max())
    ax.set_ylim(X1_grid.min(), X1_grid.max())
    
    # Custom tick positions (matching reference)
    ax.set_xticks([1, 5, 20, 50, 70, 100])
    ax.set_yticks([1, 5, 10, 20, 30, 40, 55, 60, 65, 70, 75, 80, 100])
    
    # Tick label styling
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#333333')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Heatmap saved to: {save_path}")
    
    return fig, ax


def create_styled_heatmap_pipeline(df, target_col='y1',
                                    x1_range=(1, 100),
                                    x2_range=(1, 100),
                                    resolution=100,
                                    save_path=None,
                                    show_data_points=True):
    """
    Complete pipeline: train model → generate predictions → plot styled heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with x1, x2, y1, y2 columns
    target_col : str
        Target column (default 'y1' for Rg)
    x1_range, x2_range : tuple
        Ranges for each axis
    resolution : int
        Grid resolution
    save_path : str
        Path to save figure (optional)
    show_data_points : bool
        Whether to show original data points
    
    Returns:
    --------
    model, fig, ax : trained model and plot objects
    """
    print("=" * 50)
    print("Creating Styled Rg (Y1) Heatmap Visualization")
    print("=" * 50)
    
    # Step 1: Train best model
    print("\n[Step 1] Training XGBoost model...")
    model, data = train_best_model(df, target_col)
    
    # Step 2: Generate prediction grid
    print(f"\n[Step 2] Generating {resolution}x{resolution} prediction grid...")
    print(f"  X1 range: {x1_range}")
    print(f"  X2 range: {x2_range}")
    X1_grid, X2_grid, Y_pred = generate_prediction_grid(
        model, x1_range, x2_range, resolution
    )
    
    # Step 3: Plot styled heatmap
    print("\n[Step 3] Creating styled heatmap...")
    fig, ax = plot_heatmap_styled(
        X1_grid, X2_grid, Y_pred,
        data_df=data if show_data_points else None,
        save_path=save_path
    )
    
    print("\n[Complete] Heatmap created successfully!")
    print(f"  Y1 (Rg) range: {Y_pred.min():.4f} to {Y_pred.max():.4f}")
    
    return model, fig, ax


# Keep old functions for backward compatibility
def create_rg_colormap():
    """Legacy: Red-Green colormap"""
    colors = ['#FF0000', '#FFFF00', '#00FF00']
    return LinearSegmentedColormap.from_list('RG', colors, N=256)


def create_rg_colormap_direct():
    """Legacy: Direct Red-Green colormap"""
    colors = ['#FF0000', '#00FF00']
    return LinearSegmentedColormap.from_list('RG_direct', colors, N=256)


if __name__ == "__main__":
    # Example usage / test
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'cleaned_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded data: {df.shape}")
        
        # Create styled heatmap
        output_path = os.path.join(base_dir, 'heatmap_y1_rg.png')
        model, fig, ax = create_styled_heatmap_pipeline(
            df, 
            target_col='y1',
            x1_range=(1, 100),
            x2_range=(1, 100),
            resolution=100,
            save_path=output_path,
            show_data_points=True
        )
        
        plt.show()
    else:
        print(f"Data file not found: {data_path}")
        print("Please run run_analysis.py first to generate cleaned_data.csv")
