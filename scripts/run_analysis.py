"""
ML Analysis Script - Dual Regression Tasks
===========================================
Predict Rg and CI using ONLY controllable experimental inputs:
- x1: EAN concentration (wt%)
- x2: Protein concentration (mg/ml)

Methodology:
- 80/20 Train/Test split
- 5 random seeds for stability
- Only use features available BEFORE experiment
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_cleaning import DataCleaner


def run_analysis(df, target_col, target_name, n_seeds=5):
    """
    Run regression analysis with multiple random seeds.
    
    Features: ONLY x1 and x2 (experimental inputs)
    """
    print(f"\n{'='*70}")
    print(f"   {target_name} PREDICTION")
    print(f"   Features: x1 (EAN conc.), x2 (Protein conc.)")
    print(f"{'='*70}")
    
    # Prepare data - ONLY use x1, x2 as features
    X = df[['x1', 'x2']].values
    y = df[target_col].values
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Features: x1, x2 (controllable experimental inputs only)")
    print(f"Running {n_seeds} random train/test splits for stability")
    
    results = {
        'RF': {'test_r2': [], 'cv_r2': []},
        'GB': {'test_r2': [], 'cv_r2': []},
    }
    if HAS_XGBOOST:
        results['XGB'] = {'test_r2': [], 'cv_r2': []}
    
    for seed in range(n_seeds):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Models
        models = {
            'RF': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'GB': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        }
        if HAS_XGBOOST:
            models['XGB'] = XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        
        for name, model in models.items():
            # CV on training set
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='r2')
            
            # Train and evaluate on test set
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            test_r2 = r2_score(y_test, y_pred)
            
            results[name]['test_r2'].append(test_r2)
            results[name]['cv_r2'].append(np.mean(cv_scores))
    
    # Summary
    print(f"\n{'Model':<15} {'Test R2':>12} {'Std':>10} {'CV R2':>12}")
    print("-" * 50)
    
    summary = []
    for name in results:
        test_mean = np.mean(results[name]['test_r2'])
        test_std = np.std(results[name]['test_r2'])
        cv_mean = np.mean(results[name]['cv_r2'])
        
        print(f"{name:<15} {test_mean:>12.4f} {test_std:>10.4f} {cv_mean:>12.4f}")
        
        summary.append({
            'Model': name,
            'Test R2 Mean': test_mean,
            'Test R2 Std': test_std,
            'CV R2 Mean': cv_mean
        })
    
    summary_df = pd.DataFrame(summary).sort_values('Test R2 Mean', ascending=False)
    best = summary_df.iloc[0]
    
    print(f"\nBest: {best['Model']} | Test R2 = {best['Test R2 Mean']:.4f} +/- {best['Test R2 Std']:.4f}")
    
    return summary_df, best


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    output_file = os.path.join(project_root, 'data', 'cleaned_data.csv')

    print("=" * 70)
    print("   ML ANALYSIS - DUAL REGRESSION")
    print("   Using ONLY controllable inputs: x1, x2")
    print("=" * 70)
    
    # Load and clean data
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    cleaner.save_clean_data(output_file)
    cleaner.get_summary()
    
    # Run analysis for both targets
    rg_results, rg_best = run_analysis(df, 'y1', 'Rg (Radius of Gyration)')
    ci_results, ci_best = run_analysis(df, 'CI', 'CI (Crystallinity Index)')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("   FINAL RESULTS")
    print("   (Mean of 5 random train/test splits)")
    print("=" * 70)
    
    print(f"\nRg Prediction:")
    print(f"  Best Model: {rg_best['Model']}")
    print(f"  Test R2: {rg_best['Test R2 Mean']:.4f} +/- {rg_best['Test R2 Std']:.4f}")
    
    print(f"\nCI Prediction:")
    print(f"  Best Model: {ci_best['Model']}")
    print(f"  Test R2: {ci_best['Test R2 Mean']:.4f} +/- {ci_best['Test R2 Std']:.4f}")
    
    # Save results
    results_path = os.path.join(project_root, 'outputs', 'results_summary.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   ML EXPERIMENT RESULTS\n")
        f.write("   Predicting Rg and CI from Experimental Inputs\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("Features (Controllable Inputs):\n")
        f.write("  - x1: EAN concentration (wt%)\n")
        f.write("  - x2: Protein concentration (mg/ml)\n\n")
        f.write("Targets (Experimental Outputs):\n")
        f.write("  - Rg: Radius of Gyration\n")
        f.write("  - CI: Crystallinity Index (CI=0 when no crystals)\n\n")
        f.write("Validation:\n")
        f.write("  - 80/20 Train/Test split\n")
        f.write("  - 5 different random seeds\n")
        f.write("  - Results show mean +/- std across seeds\n\n")
        
        f.write("DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Quality filter: R2_w >= 0.80\n")
        f.write(f"Samples with crystals: {int(df['y2'].sum())}/{len(df)}\n\n")
        
        f.write("Rg PREDICTION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(rg_results.to_string(index=False))
        f.write(f"\n\nBest: {rg_best['Model']} | R2 = {rg_best['Test R2 Mean']:.4f} +/- {rg_best['Test R2 Std']:.4f}\n\n")
        
        f.write("CI PREDICTION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(ci_results.to_string(index=False))
        f.write(f"\n\nBest: {ci_best['Model']} | R2 = {ci_best['Test R2 Mean']:.4f} +/- {ci_best['Test R2 Std']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


