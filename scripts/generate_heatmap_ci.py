"""
Generate CI Regression Heatmap Visualization (threshold-styled)

Reference: Rg heatmap generation in scripts/generate_heatmap.py

Requirements:
- Use a different colorbar/colormap from Rg
- Values < 0.1 shown as dark colors
- Values >= 0.1 shown as bright colors
"""

import os
import sys

# Ensure local imports + matplotlib cache are writable (important in sandbox/WSL)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(project_root, ".mplconfig"))

import pandas as pd
import matplotlib.pyplot as plt

from src.visualization import create_styled_heatmap_pipeline, create_zone_pastel_colormap


def main():
    data_path = os.path.join(project_root, "data", "cleaned_data.csv")
    output_path = os.path.join(project_root, "outputs", "heatmap_ci_regression.png")

    print("=" * 60)
    print("   CI REGRESSION HEATMAP GENERATION   ")
    print("   Threshold styling: <0.1 dark, >=0.1 bright")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please run scripts/run_analysis.py first to generate cleaned_data.csv")
        return

    df = pd.read_csv(data_path)
    if "CI" not in df.columns:
        raise ValueError("Column 'CI' not found in cleaned data. Check src/data_cleaning.py mapping.")

    print(f"\nLoaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"X1 (IL/EAN Concentration) range: {df['x1'].min():.2f} - {df['x1'].max():.2f}")
    print(f"X2 (Protein) range: {df['x2'].min():.2f} - {df['x2'].max():.2f}")
    print(f"CI range: {df['CI'].min():.4f} - {df['CI'].max():.4f}")

    print("\nGenerating CI heatmap...")
    _model, _fig, _ax = create_styled_heatmap_pipeline(
        df,
        target_col="CI",
        x1_range=(0, 100),
        x2_range=(0, 100),
        resolution=100,
        save_path=output_path,
        show_data_points=False,
        colorbar_label="CI",
        # Common sequential colormap (distinct from Rg) + explicit CI=0.1 contour line
        # Pastel scale inspired by zone diagram (light blue/yellow/green)
        cmap=create_zone_pastel_colormap(),
        vmin=0.0,
        vmax=1.0,
        threshold=0.1,
        contour_threshold=0.1,
        contour_color="black",
        contour_linewidth=2.0,
        font_scale=2.0,
        bold=False,
        tick_step=20,
        # CI is in [0,1], so show one decimal to avoid duplicate integer labels
        colorbar_integer_ticks=False,
        colorbar_tick_format="%.1f",
    )

    print("\n" + "=" * 60)
    print("   GENERATION COMPLETE   ")
    print("=" * 60)
    print(f"Output file: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

