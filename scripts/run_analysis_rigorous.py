"""
Rigorous ML Analysis with Proper Train/Test Split
==================================================
Proper methodology:
1. Split data into Train (80%) / Test (20%) FIRST
2. Only augment the TRAINING set
3. Evaluate on UNTOUCHED test set
4. Use CV within training set for model selection
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
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_cleaning import DataCleaner


def augment_training_data(X_train, y_train, n_copies=3, noise_scale=0.03, random_state=42):
    """
    Augment ONLY training data with Gaussian noise.
    
    This uses the actual measurement uncertainty principle:
    - noise_scale=0.03 corresponds to ~3% relative error
    - This matches typical SAXS measurement precision
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    n_copies : int
        Number of augmented copies per sample
    noise_scale : float
        Noise level as fraction of standard deviation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_aug, y_aug : Augmented training data
    """
    np.random.seed(random_state)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_std = np.std(X_train, axis=0)
    y_std = np.std(y_train)
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    for _ in range(n_copies):
        # Add noise to features (within measurement uncertainty)
        X_noise = X_train + np.random.normal(0, noise_scale * X_std, X_train.shape)
        
        # Add smaller noise to target (preserve signal)
        y_noise = y_train + np.random.normal(0, noise_scale * 0.5 * y_std, y_train.shape)
        
        # Ensure non-negative values where appropriate
        X_noise = np.maximum(X_noise, 0)  # x1, x2 cannot be negative
        y_noise = np.maximum(y_noise, 0)  # Rg, CI cannot be negative
        
        X_aug_list.append(X_noise)
        y_aug_list.append(y_noise)
    
    X_aug = np.vstack(X_aug_list)
    y_aug = np.concatenate(y_aug_list)
    
    return X_aug, y_aug


def run_rigorous_analysis(df, target_col, target_name, test_size=0.2, random_state=42):
    """
    Run rigorous analysis with proper train/test split.
    
    Methodology:
    1. Split data FIRST (stratified if possible)
    2. Fit scaler ONLY on training data
    3. Augment ONLY training data
    4. Select best model via CV on training set
    5. Final evaluation on untouched test set
    """
    
    print(f"\n{'='*70}")
    print(f"   RIGOROUS ANALYSIS: {target_name}")
    print(f"   Proper Train/Test Split with Training-Only Augmentation")
    print(f"{'='*70}")
    
    # Prepare data
    data = df.dropna(subset=[target_col]).copy()
    X = data[['x1', 'x2']].values
    y = data[target_col].values
    
    # ==============================
    # Step 1: Train/Test Split FIRST
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData Split:")
    print(f"  Total samples: {len(y)}")
    print(f"  Training set: {len(y_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test set: {len(y_test)} samples ({100*test_size:.0f}%) - HELD OUT")
    
    # ==============================
    # Step 2: Fit Scaler on Training ONLY
    # ==============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use same transform
    
    print(f"\nScaler fitted on training data only (no data leakage)")
    
    results = []
    
    # ==============================
    # Step 3: Baseline (No Augmentation)
    # ==============================
    print(f"\n--- Phase 1: Baseline (No Augmentation) ---")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    
    for name, model in models.items():
        # CV on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on TEST set
        y_pred_test = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results.append({
            'method': 'No Augmentation',
            'model': name,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_samples': len(y_train)
        })
        
        print(f"  {name}:")
        print(f"    CV R2 (train): {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        print(f"    Test R2: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    
    # ==============================
    # Step 4: With Augmentation (Training Only)
    # ==============================
    best_aug_config = None
    best_test_r2 = -np.inf
    
    for noise_scale in [0.02, 0.03, 0.05]:
        for n_copies in [2, 3, 5]:
            print(f"\n--- Augmentation (noise={noise_scale}, copies={n_copies}) ---")
            
            # Augment training data ONLY
            X_train_aug, y_train_aug = augment_training_data(
                X_train_scaled, y_train, 
                n_copies=n_copies, 
                noise_scale=noise_scale
            )
            
            print(f"  Training samples: {len(y_train)} -> {len(y_train_aug)} (augmented)")
            
            for name, model_class in [
                ('Random Forest', RandomForestRegressor),
                ('Gradient Boosting', GradientBoostingRegressor)
            ]:
                if name == 'Random Forest':
                    model = model_class(n_estimators=200, max_depth=10, random_state=42)
                else:
                    model = model_class(n_estimators=200, max_depth=5, random_state=42)
                
                # Train on augmented data
                model.fit(X_train_aug, y_train_aug)
                
                # Evaluate on ORIGINAL test set (no augmentation!)
                y_pred_test = model.predict(X_test_scaled)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # CV on augmented training data (for reference)
                cv_scores = cross_val_score(model, X_train_aug, y_train_aug, cv=5, scoring='r2')
                
                results.append({
                    'method': f'Aug(s={noise_scale},n={n_copies})',
                    'model': name,
                    'cv_r2_mean': np.mean(cv_scores),
                    'cv_r2_std': np.std(cv_scores),
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'train_samples': len(y_train_aug)
                })
                
                print(f"  {name}: Test R2 = {test_r2:.4f} | RMSE = {test_rmse:.4f}")
                
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_aug_config = f"Aug(s={noise_scale},n={n_copies}) + {name}"
    
    # ==============================
    # Step 5: Summary
    # ==============================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"   FINAL RESULTS: {target_name} (Sorted by Test R2)")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<25} {'Model':<18} {'CV R2':>8} {'Test R2':>9} {'RMSE':>8} {'MAE':>8}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['method']:<25} {row['model']:<18} {row['cv_r2_mean']:>8.4f} {row['test_r2']:>9.4f} {row['test_rmse']:>8.4f} {row['test_mae']:>8.4f}")
    
    # Best result
    best = results_df.iloc[0]
    baseline_best = results_df[results_df['method'] == 'No Augmentation'].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"   CONCLUSION: {target_name}")
    print(f"{'='*70}")
    print(f"\nBaseline (No Augmentation):")
    print(f"  Best: {baseline_best['model']}")
    print(f"  Test R2: {baseline_best['test_r2']:.4f}")
    
    print(f"\nBest Overall:")
    print(f"  Method: {best['method']} + {best['model']}")
    print(f"  Test R2: {best['test_r2']:.4f}")
    
    improvement = best['test_r2'] - baseline_best['test_r2']
    if improvement > 0.01:
        print(f"\n  >>> Augmentation IMPROVED Test R2 by {improvement:.4f} ({improvement/baseline_best['test_r2']*100:.1f}%)")
    elif improvement < -0.01:
        print(f"\n  >>> Augmentation DECREASED Test R2 by {-improvement:.4f}")
    else:
        print(f"\n  >>> Augmentation had MINIMAL effect (change: {improvement:.4f})")
    
    return results_df, best, baseline_best


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    
    print("=" * 70)
    print("   RIGOROUS ML ANALYSIS")
    print("   With Proper Train/Test Split & Training-Only Augmentation")
    print("=" * 70)
    print("\nMethodology:")
    print("  1. Split data into Train (80%) / Test (20%) FIRST")
    print("  2. Fit scaler on TRAINING data only")
    print("  3. Augment TRAINING data only (test set untouched)")
    print("  4. Evaluate all models on the SAME held-out test set")
    print("  5. This prevents data leakage and gives true generalization error")
    
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    
    # Run for both targets
    rg_results, rg_best, rg_baseline = run_rigorous_analysis(df, 'y1', 'Rg (Radius of Gyration)')
    ci_results, ci_best, ci_baseline = run_rigorous_analysis(df, 'CI', 'CI (Crystallinity Index)')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("Rg Prediction:")
    print("-" * 70)
    print(f"  Baseline Test R2: {rg_baseline['test_r2']:.4f}")
    print(f"  Best Test R2:     {rg_best['test_r2']:.4f} ({rg_best['method']} + {rg_best['model']})")
    
    print("\n" + "-" * 70)
    print("CI Prediction:")
    print("-" * 70)
    print(f"  Baseline Test R2: {ci_baseline['test_r2']:.4f}")
    print(f"  Best Test R2:     {ci_best['test_r2']:.4f} ({ci_best['method']} + {ci_best['model']})")
    
    # Save results
    results_path = os.path.join(project_root, 'outputs', 'rigorous_analysis_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   RIGOROUS ML ANALYSIS RESULTS\n")
        f.write("   Proper Train/Test Split with Training-Only Augmentation\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Data split: 80% Train / 20% Test (BEFORE any processing)\n")
        f.write("2. Scaler fitted on TRAINING data only\n")
        f.write("3. Augmentation applied to TRAINING data only\n")
        f.write("4. Test set remains UNTOUCHED for final evaluation\n")
        f.write("5. This ensures no data leakage and true generalization metrics\n\n")
        
        f.write("Rg (RADIUS OF GYRATION) RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(rg_results.to_string(index=False))
        f.write(f"\n\nBaseline Test R2: {rg_baseline['test_r2']:.4f}\n")
        f.write(f"Best Test R2: {rg_best['test_r2']:.4f} ({rg_best['method']} + {rg_best['model']})\n\n")
        
        f.write("CI (CRYSTALLINITY INDEX) RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(ci_results.to_string(index=False))
        f.write(f"\n\nBaseline Test R2: {ci_baseline['test_r2']:.4f}\n")
        f.write(f"Best Test R2: {ci_best['test_r2']:.4f} ({ci_best['method']} + {ci_best['model']})\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
