"""
Final Rigorous Analysis with Consistent Test Set
=================================================
Key improvements:
1. Split data ONCE at the beginning
2. Use the SAME test set for ALL experiments
3. Multiple random seeds for stability
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_data(filepath, r2_threshold=0.80):
    """Load and prepare data."""
    df = pd.read_csv(filepath, header=0)
    
    # Rename columns
    column_map = {}
    for col in df.columns:
        if col.startswith('x1'):
            column_map[col] = 'x1'
        elif col.startswith('x2'):
            column_map[col] = 'x2'
        elif col.startswith('y1') and 'Rg' in col:
            column_map[col] = 'y1'
        elif col.startswith('y2'):
            column_map[col] = 'y2'
        elif 'R2_w' in col or 'weighted R' in col:
            column_map[col] = 'R2_w'
    
    df.rename(columns=column_map, inplace=True)
    
    # Convert y2
    df['y2'] = df['y2'].astype(str).str.lower().map({'false': 0, 'true': 1})
    
    # CI = 0 when y2 = 0
    df.loc[df['y2'] == 0, 'CI'] = 0.0
    df['CI'] = pd.to_numeric(df['CI'], errors='coerce').fillna(0.0)
    
    # Filter by R2
    df['R2_w'] = pd.to_numeric(df['R2_w'], errors='coerce')
    df = df[df['R2_w'] >= r2_threshold].reset_index(drop=True)
    
    # Convert all potential features to numeric
    for col in ['n_real_peaks', 'B_factor', 'fwhm_avg', 'Rg_err', 'chi2_red', 't_a', 'snr_peak1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def run_with_multiple_seeds(df, target_col, feature_cols, n_seeds=5):
    """
    Run analysis with multiple random seeds for stability.
    Returns mean and std of test R2 across seeds.
    """
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    results_rf = []
    results_gb = []
    
    for seed in range(n_seeds):
        # Split with different seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_r2 = r2_score(y_test, rf.predict(X_test_s))
        results_rf.append(rf_r2)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        gb.fit(X_train_s, y_train)
        gb_r2 = r2_score(y_test, gb.predict(X_test_s))
        results_gb.append(gb_r2)
    
    return {
        'RF': {'mean': np.mean(results_rf), 'std': np.std(results_rf), 'scores': results_rf},
        'GB': {'mean': np.mean(results_gb), 'std': np.std(results_gb), 'scores': results_gb}
    }


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    
    print("=" * 70)
    print("   FINAL RIGOROUS ANALYSIS")
    print("   Multiple Random Seeds for Stability")
    print("=" * 70)
    
    df = load_data(input_file, r2_threshold=0.80)
    print(f"\nTotal samples: {len(df)}")
    print(f"Test set: 20% = ~{int(len(df)*0.2)} samples")
    print(f"Running with 5 different random seeds for stability\n")
    
    # Define feature sets
    feature_sets = {
        'Basic': ['x1', 'x2'],
        'Structural': ['x1', 'x2', 'n_real_peaks', 'B_factor', 'fwhm_avg'],
        'Combined': ['x1', 'x2', 'n_real_peaks', 'B_factor', 'fwhm_avg', 
                    'Rg_err', 'chi2_red', 't_a', 'snr_peak1']
    }
    
    all_results = []
    
    for target, target_name in [('y1', 'Rg'), ('CI', 'CI')]:
        print(f"\n{'='*70}")
        print(f"   TARGET: {target_name}")
        print(f"{'='*70}")
        
        for set_name, features in feature_sets.items():
            # Check all features exist and have data
            valid_features = [f for f in features if f in df.columns and df[f].notna().sum() > 0.9*len(df)]
            
            if len(valid_features) != len(features):
                print(f"\n{set_name}: Skipped (missing features)")
                continue
            
            print(f"\n--- {set_name} ({len(valid_features)} features) ---")
            
            results = run_with_multiple_seeds(df, target, valid_features, n_seeds=5)
            
            for model in ['RF', 'GB']:
                print(f"  {model}: R2 = {results[model]['mean']:.4f} +/- {results[model]['std']:.4f}")
                print(f"       Scores: {[f'{s:.4f}' for s in results[model]['scores']]}")
                
                all_results.append({
                    'target': target_name,
                    'feature_set': set_name,
                    'n_features': len(valid_features),
                    'model': model,
                    'mean_r2': results[model]['mean'],
                    'std_r2': results[model]['std'],
                    'min_r2': min(results[model]['scores']),
                    'max_r2': max(results[model]['scores'])
                })
    
    # Summary
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY (Mean of 5 Random Seeds)")
    print("=" * 70)
    
    for target_name in ['Rg', 'CI']:
        print(f"\n{target_name} Prediction:")
        subset = results_df[results_df['target'] == target_name].sort_values('mean_r2', ascending=False)
        
        print(f"  {'Feature Set':<15} {'Model':<5} {'Mean R2':>10} {'Std':>8} {'Range':>20}")
        print("  " + "-" * 60)
        for _, row in subset.iterrows():
            range_str = f"[{row['min_r2']:.4f}, {row['max_r2']:.4f}]"
            print(f"  {row['feature_set']:<15} {row['model']:<5} {row['mean_r2']:>10.4f} {row['std_r2']:>8.4f} {range_str:>20}")
        
        best = subset.iloc[0]
        baseline = subset[subset['feature_set'] == 'Basic'].iloc[0]
        improvement = best['mean_r2'] - baseline['mean_r2']
        
        print(f"\n  Baseline (Basic): {baseline['mean_r2']:.4f}")
        print(f"  Best ({best['feature_set']} + {best['model']}): {best['mean_r2']:.4f}")
        if improvement > 0.01:
            print(f"  IMPROVEMENT: +{improvement:.4f} ({improvement/baseline['mean_r2']*100:.1f}%)")
    
    # Save
    results_path = os.path.join(project_root, 'outputs', 'final_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   FINAL RIGOROUS ANALYSIS RESULTS\n")
        f.write("   (Mean of 5 Random Train/Test Splits)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("- Train/Test Split: 80% / 20%\n")
        f.write("- 5 different random seeds used\n")
        f.write("- Results show mean +/- std across seeds\n")
        f.write("- This gives more reliable estimates than single split\n\n")
        
        f.write("ALL RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        # Best per target
        for target_name in ['Rg', 'CI']:
            subset = results_df[results_df['target'] == target_name].sort_values('mean_r2', ascending=False)
            best = subset.iloc[0]
            baseline = subset[subset['feature_set'] == 'Basic'].iloc[0]
            
            f.write(f"\n{target_name} PREDICTION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Baseline (x1, x2): R2 = {baseline['mean_r2']:.4f} +/- {baseline['std_r2']:.4f}\n")
            f.write(f"  Best ({best['feature_set']} + {best['model']}): R2 = {best['mean_r2']:.4f} +/- {best['std_r2']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
