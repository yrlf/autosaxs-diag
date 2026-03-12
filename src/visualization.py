"""
Visualization utilities for ML experiment results.
Creates heatmaps to visualize model predictions across the feature space.
Style reference: Purple → Cyan → Green → Yellow colormap with data point markers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
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


def create_zone_pastel_colormap():
    """
    Pastel colormap inspired by the "Soluble/Crystallization/Nucleation" zone figure:
    light blue → light yellow → light green.

    This is intentionally distinct from the Rg heatmap colormap.
    """
    colors = [
        "#cfe8f6",  # light blue (soluble-like)
        "#f3efb0",  # light yellow (crystallization-like)
        "#b7dcae",  # light green (nucleation-like)
    ]
    return LinearSegmentedColormap.from_list("zone_pastel", colors, N=256)


def create_threshold_colormap_for_range(
    vmin: float,
    vmax: float,
    threshold: float,
    *,
    dark_color: str = "#1f1f1f",
    bright_colors=None,
    name: str = "threshold_cmap",
):
    """
    Create a colormap with an intentional contrast at a threshold.

    Requirement:
    - values < threshold: dark colors
    - values >= threshold: bright colors

    Notes:
    - This colormap is constructed *for a specific vmin/vmax* so that the
      threshold maps to the correct position on the colorbar.
    """
    if bright_colors is None:
        # Bright palette (different from Rg heatmap): cyan → green → yellow
        bright_colors = ["#00c6ff", "#5ec962", "#fde725"]

    if vmax <= vmin:
        # Degenerate case; fall back to a simple two-color map
        return LinearSegmentedColormap.from_list(
            name,
            [dark_color, bright_colors[-1]],
            N=256,
        )

    t = (threshold - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))

    # Keep the low region visually "dark" and force a sharp transition at t
    eps = min(1e-6, t / 2.0) if t > 0 else 0.0

    stops = [
        (0.0, dark_color),
        (max(0.0, t - eps), dark_color),
        (t, bright_colors[0]),
        (min(1.0, t + 0.25 * (1.0 - t)), bright_colors[1]),
        (1.0, bright_colors[2]),
    ]
    return LinearSegmentedColormap.from_list(name, stops, N=256)


def train_best_model(df, target_col='y1'):
    """
    Trains a strong default model on the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing x1, x2, and target columns
    target_col : str
        Target column name (default: 'y1' for Rg)
    
    Returns:
    --------
    model : fitted model (XGBoost if available, otherwise RandomForest)
    data : DataFrame used for training (for plotting data points)
    """
    # Drop rows where target is missing
    data = df.dropna(subset=[target_col]).copy()
    X = data[['x1', 'x2']]
    y = data[target_col]
    
    # Prefer XGBoost when available; otherwise fall back to RandomForest
    model = None
    if XGBRegressor is not None:
        model = XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    else:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as e:
            raise ImportError(
                "Need either xgboost or scikit-learn for heatmap model training. "
                "Install with: pip install -r requirements.txt"
            ) from e
        model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)

    model.fit(X, y)
    
    print(f"Model trained on {len(X)} samples")
    print(f"Target: {target_col}")
    
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
                        cmap=None,
                        colorbar_label='Rg (Å)',
                        vmin=None,
                        vmax=None,
                        threshold=None,
                        binarize_threshold=None,
                        contour_threshold=None,
                        contour_color="#00c6ff",
                        contour_linewidth=2.5,
                        contour_alpha=0.95,
                        font_scale: float = 1.0,
                        bold: bool = False,
                        n_xticks: int = 6,
                        n_yticks: int = 6,
                        tick_step: float | None = None,
                        colorbar_integer_ticks: bool = False,
                        colorbar_tick_format: str | None = None,
                        threshold_dark_color="#1f1f1f",
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
    # Optional: binarize predictions into two classes (no gradient)
    # Requirement for CI: >0.1 vs <=0.1 (<= threshold = dark, > threshold = bright)
    is_binary = binarize_threshold is not None
    if is_binary:
        thr = float(binarize_threshold)
        Y_plot = (np.asarray(Y_pred) > thr).astype(int)
        # Default (if cmap not provided): discrete Blue/Red (no gradient)
        # 0 (<=thr) -> Blue, 1 (>thr) -> Red
        if cmap is None:
            cmap = ListedColormap(["#2b6cb0", "#c53030"], name="ci_blue_red_binary")
        norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2, clip=True)
    else:
        Y_plot = Y_pred

        # Color scale bounds (optional)
        if vmin is None:
            vmin = float(np.nanmin(Y_pred))
        if vmax is None:
            vmax = float(np.nanmax(Y_pred))
        if threshold is not None:
            # Ensure threshold sits within the color scale
            vmin = float(min(vmin, threshold))
            vmax = float(max(vmax, threshold))

        # Colormap
        if cmap is None:
            if threshold is not None:
                cmap = create_threshold_colormap_for_range(
                    vmin=vmin,
                    vmax=vmax,
                    threshold=float(threshold),
                    dark_color=threshold_dark_color,
                    name="ci_threshold_cmap",
                )
            else:
                # Default: viridis-like (purple → cyan → green → yellow)
                cmap = create_viridis_like_colormap()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure with specific styling
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot heatmap (X2 on horizontal, X1 on vertical)
    im = ax.pcolormesh(X2_grid, X1_grid, Y_plot, 
                       cmap=cmap,
                       norm=norm,
                       shading='auto')

    # Optional: draw a contour line at a specific CI threshold (e.g., 0.1)
    # Use the *continuous* predictions for contours (not binarized Y_plot).
    if (contour_threshold is not None) and (not is_binary):
        try:
            ax.contour(
                X2_grid,
                X1_grid,
                np.asarray(Y_pred),
                levels=[float(contour_threshold)],
                colors=contour_color,
                linewidths=float(contour_linewidth),
                alpha=float(contour_alpha),
                zorder=6,
            )
        except Exception:
            # Contour can fail for degenerate/constant arrays; ignore gracefully
            pass
    
    # Ticks (prefer "nice" step like 20 -> 0,20,40...)
    def _nice_step_ticks(vmin_, vmax_, step):
        step = float(step)
        start = np.floor(float(vmin_) / step) * step
        end = np.ceil(float(vmax_) / step) * step
        ticks = np.arange(start, end + 0.5 * step, step)
        # Clean up tiny float artifacts
        ticks = np.round(ticks, 6)
        return ticks

    def _uniform_ticks(vmin_, vmax_, n):
        n = int(n)
        if n <= 1:
            return np.array([(vmin_ + vmax_) / 2.0], dtype=float)
        ticks = np.linspace(float(vmin_), float(vmax_), n)
        ticks_rounded = np.unique(np.round(ticks).astype(int)).astype(float)
        if len(ticks_rounded) < 2:
            return ticks
        return ticks_rounded

    if tick_step is not None:
        x2_ticks = _nice_step_ticks(X2_grid.min(), X2_grid.max(), tick_step)
        x1_ticks = _nice_step_ticks(X1_grid.min(), X1_grid.max(), tick_step)
    else:
        x2_ticks = _uniform_ticks(X2_grid.min(), X2_grid.max(), n_xticks)
        x1_ticks = _uniform_ticks(X1_grid.min(), X1_grid.max(), n_yticks)
    
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
    
    # Colorbar styling (slightly larger + bold, consistent with other figures)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    axis_label_fs = int(round(16 * float(font_scale)))
    tick_fs = int(round(13 * float(font_scale)))
    font_weight = "bold" if bool(bold) else "normal"
    cbar.set_label(colorbar_label, fontsize=axis_label_fs, fontweight=font_weight)
    cbar.ax.tick_params(labelsize=tick_fs)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight(font_weight)
    
    # Colorbar ticks
    if is_binary:
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([f"≤ {binarize_threshold}", f"> {binarize_threshold}"])
    elif threshold is not None:
        # CI-like: keep a few ticks and ensure threshold is shown explicitly
        ticks = np.linspace(vmin, vmax, 6)
        ticks = np.unique(np.concatenate([ticks, np.array([float(threshold)])]))
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cbar.set_ticks(ticks)
    else:
        # Keep ticks strictly within the plotted normalization range to avoid
        # any "blank caps" at the ends when ticks exceed vmin/vmax.
        cbar_ticks = np.linspace(float(vmin), float(vmax), 6)
        cbar.set_ticks(cbar_ticks)

    if (not is_binary) and (colorbar_tick_format is not None):
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(str(colorbar_tick_format)))
    elif colorbar_integer_ticks and (not is_binary):
        # Display as integers (no decimals) for cleaner look
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # Axis labels (slightly larger + bold, consistent with other figures)
    ax.set_xlabel('Protein (mg/mL)', fontsize=axis_label_fs, fontweight=font_weight)
    ax.set_ylabel('Concentration of IL (wt%)', fontsize=axis_label_fs, fontweight=font_weight)
    
    # Set axis limits
    ax.set_xlim(X2_grid.min(), X2_grid.max())
    ax.set_ylim(X1_grid.min(), X1_grid.max())
    
    # Use the same uniform tick positions for axes
    ax.set_xticks(x2_ticks)
    ax.set_yticks(x1_ticks)
    
    # Tick label styling
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight(font_weight)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight(font_weight)
    
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
                                    show_data_points=True,
                                    cmap=None,
                                    colorbar_label=None,
                                    vmin=None,
                                    vmax=None,
                                    threshold=None,
                                    binarize_threshold=None,
                                    contour_threshold=None,
                                    contour_color="#00c6ff",
                                    contour_linewidth=2.5,
                                    contour_alpha=0.95,
                                    font_scale: float = 1.0,
                                    bold: bool = False,
                                    tick_step: float | None = None,
                                    colorbar_integer_ticks: bool = False,
                                    colorbar_tick_format: str | None = None):
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
    print("Creating Styled Heatmap Visualization")
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
        save_path=save_path,
        cmap=cmap,
        colorbar_label=(colorbar_label if colorbar_label is not None else target_col),
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        binarize_threshold=binarize_threshold,
        contour_threshold=contour_threshold,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        contour_alpha=contour_alpha,
        font_scale=font_scale,
        bold=bold,
        tick_step=tick_step,
        colorbar_integer_ticks=colorbar_integer_ticks,
        colorbar_tick_format=colorbar_tick_format,
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
