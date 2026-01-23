"""
Generate Rg (Y1) Heatmap Visualization - Styled Version
Based on the best model: XGBoost Regressor (CV R² = 0.796)

Style features:
- Purple → Cyan → Green → Yellow colormap
- Horizontal axis: X2 - Protein (mg/mL)
- Vertical axis: X1 - Concentration of IL (wt%)
- Color: Y1 - Rg (Å)
- Data point markers (gray = non-crystalline, red = crystalline)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import create_styled_heatmap_pipeline

def main():
    # Setup paths
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    
    data_path = os.path.join(project_root, 'data', 'cleaned_data.csv')
    output_path = os.path.join(project_root, 'outputs', 'heatmap_y1_rg.png')
    
    print("=" * 60)
    print("   Rg (Y1) STYLED HEATMAP GENERATION   ")
    print("   Based on XGBoost Model (Best Performer)")
    print("=" * 60)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please run run_analysis.py first to generate cleaned_data.csv")
        return
    
    df = pd.read_csv(data_path)
    print(f"\nLoaded data: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"X1 (IL Concentration) range: {df['x1'].min():.2f} - {df['x1'].max():.2f} wt%")
    print(f"X2 (Protein) range: {df['x2'].min():.2f} - {df['x2'].max():.2f} mg/mL")
    print(f"Y1 (Rg) range: {df['y1'].min():.2f} - {df['y1'].max():.2f} Å")
    
    if 'y2' in df.columns:
        crystal_count = df['y2'].sum()
        total_count = len(df.dropna(subset=['y2']))
        print(f"Y2 (Crystalline): {int(crystal_count)}/{total_count} samples")
    
    # === Generate Styled Heatmap ===
    print("\n" + "-" * 60)
    print("Generating Styled Heatmap...")
    
    model, fig, ax = create_styled_heatmap_pipeline(
        df,
        target_col='y1',
        x1_range=(1, 100),
        x2_range=(1, 100),
        resolution=100,
        save_path=output_path,
        show_data_points=False  # 不显示原始数据点，只显示纯净热力图
    )
    
    # Display results summary
    print("\n" + "=" * 60)
    print("   GENERATION COMPLETE   ")
    print("=" * 60)
    print(f"\nOutput file: {output_path}")
    print(f"\nStyle features:")
    print("  - Colormap: Purple → Cyan → Green → Yellow")
    print("  - Data markers: Gray (non-crystalline), Red (crystalline)")
    print("  - Axis: X2 (Protein) horizontal, X1 (IL) vertical")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
