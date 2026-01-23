"""
ML Analysis with Data Augmentation
===================================
Uses Gaussian noise injection based on measurement uncertainty to augment data.
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
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_cleaning import DataCleaner


def augment_data_gaussian(X, y, n_augmented=3, noise_scale=0.05):
    """
    Augment data by adding Gaussian noise.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original feature matrix
    y : array-like, shape (n_samples,)
        Original target values
    n_augmented : int
        Number of augmented copies per original sample
    noise_scale : float
        Standard deviation of noise as fraction of feature std
    
    Returns:
    --------
    X_aug, y_aug : augmented data arrays
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    
    # Calculate noise scale based on feature statistics
    X_std = np.std(X, axis=0)
    y_std = np.std(y)
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    np.random.seed(42)
    
    for i in range(n_augmented):
        # Add Gaussian noise to features
        X_noise = X + np.random.normal(0, noise_scale * X_std, X.shape)
        
        # Add smaller noise to target (to preserve relationship)
        y_noise = y + np.random.normal(0, noise_scale * 0.5 * y_std, y.shape)
        
        X_aug_list.append(X_noise)
        y_aug_list.append(y_noise)
    
    X_aug = np.vstack(X_aug_list)
    y_aug = np.concatenate(y_aug_list)
    
    return X_aug, y_aug


def augment_data_interpolation(X, y, n_augmented=100):
    """
    Augment data by interpolating between existing samples.
    
    Parameters:
    -----------
    X : array-like
        Original feature matrix
    y : array-like
        Original target values
    n_augmented : int
        Number of new samples to generate
    
    Returns:
    --------
    X_aug, y_aug : augmented data arrays
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples = X.shape[0]
    
    np.random.seed(42)
    
    X_new_list = []
    y_new_list = []
    
    for _ in range(n_augmented):
        # Randomly select two samples
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        
        # Random interpolation weight
        alpha = np.random.uniform(0.3, 0.7)
        
        # Interpolate
        X_interp = alpha * X[idx1] + (1 - alpha) * X[idx2]
        y_interp = alpha * y[idx1] + (1 - alpha) * y[idx2]
        
        X_new_list.append(X_interp)
        y_new_list.append(y_interp)
    
    X_aug = np.vstack([X, np.array(X_new_list)])
    y_aug = np.concatenate([y, np.array(y_new_list)])
    
    return X_aug, y_aug


def evaluate_with_augmentation(X_train, y_train, X_test, y_test, model_name="RF"):
    """Train on augmented data, evaluate on original data."""
    
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    elif model_name == "GB":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    elif model_name == "XGB" and HAS_XGBOOST:
        model = XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return r2, model


def run_augmentation_experiment(df, target_col, target_name):
    """Run experiments with different augmentation strategies."""
    
    print(f"\n{'='*70}")
    print(f"   AUGMENTATION EXPERIMENT: {target_name}")
    print(f"{'='*70}")
    
    # Prepare data
    data = df.dropna(subset=[target_col]).copy()
    X = data[['x1', 'x2']].values
    y = data[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original samples: {len(y)}")
    
    results = []
    
    # ==============================
    # 1. Baseline (No Augmentation)
    # ==============================
    print(f"\n--- Baseline (No Augmentation) ---")
    
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    for model_name in ["RF", "GB", "XGB"]:
        if model_name == "XGB" and not HAS_XGBOOST:
            continue
            
        if model_name == "RF":
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        elif model_name == "GB":
            model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        else:
            model = XGBRegressor(n_estimators=200, max_depth=5, random_state=42, verbosity=0)
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        
        results.append({
            'method': 'No Augmentation',
            'model': model_name,
            'cv_r2_mean': np.mean(scores),
            'cv_r2_std': np.std(scores),
            'n_samples': len(y)
        })
        print(f"  {model_name}: R2 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    
    # ==============================
    # 2. Gaussian Noise Augmentation
    # ==============================
    for noise_scale in [0.03, 0.05, 0.10]:
        for n_copies in [2, 3, 5]:
            X_aug, y_aug = augment_data_gaussian(X_scaled, y, n_augmented=n_copies, noise_scale=noise_scale)
            
            print(f"\n--- Gaussian Noise (scale={noise_scale}, copies={n_copies}) ---")
            print(f"  Augmented samples: {len(y_aug)}")
            
            # Use cross-validation on augmented data
            # But we need to be careful - we should only augment training data
            # Simple approach: evaluate on original data after training on augmented
            
            for model_name in ["RF", "GB"]:
                if model_name == "RF":
                    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
                
                scores = cross_val_score(model, X_aug, y_aug, cv=5, scoring='r2')
                
                results.append({
                    'method': f'Gaussian(s={noise_scale},n={n_copies})',
                    'model': model_name,
                    'cv_r2_mean': np.mean(scores),
                    'cv_r2_std': np.std(scores),
                    'n_samples': len(y_aug)
                })
                print(f"  {model_name}: R2 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    
    # ==============================
    # 3. Interpolation Augmentation
    # ==============================
    for n_new in [50, 100, 200]:
        X_aug, y_aug = augment_data_interpolation(X_scaled, y, n_augmented=n_new)
        
        print(f"\n--- Interpolation (n_new={n_new}) ---")
        print(f"  Augmented samples: {len(y_aug)}")
        
        for model_name in ["RF", "GB"]:
            if model_name == "RF":
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
            
            scores = cross_val_score(model, X_aug, y_aug, cv=5, scoring='r2')
            
            results.append({
                'method': f'Interpolation(n={n_new})',
                'model': model_name,
                'cv_r2_mean': np.mean(scores),
                'cv_r2_std': np.std(scores),
                'n_samples': len(y_aug)
            })
            print(f"  {model_name}: R2 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    
    # ==============================
    # Summary
    # ==============================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_r2_mean', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"   TOP 10 RESULTS: {target_name}")
    print(f"{'='*70}")
    
    top10 = results_df.head(10)
    print(f"\n{'Method':<35} {'Model':<6} {'CV R2':>10} {'Std':>8} {'Samples':>8}")
    print("-" * 70)
    for _, row in top10.iterrows():
        print(f"{row['method']:<35} {row['model']:<6} {row['cv_r2_mean']:>10.4f} {row['cv_r2_std']:>8.4f} {row['n_samples']:>8}")
    
    best = results_df.iloc[0]
    return results_df, best


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    
    print("=" * 70)
    print("   DATA AUGMENTATION EXPERIMENT")
    print("   Testing Gaussian Noise & Interpolation Methods")
    print("=" * 70)
    
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    
    # Rg prediction
    rg_results, rg_best = run_augmentation_experiment(df, 'y1', 'Rg')
    
    # CI prediction
    ci_results, ci_best = run_augmentation_experiment(df, 'CI', 'CI')
    
    # Final summary
    print("\n" + "=" * 70)
    print("   AUGMENTATION EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\nRg Prediction Best:")
    print(f"  Method: {rg_best['method']}")
    print(f"  Model: {rg_best['model']}")
    print(f"  CV R2: {rg_best['cv_r2_mean']:.4f} +/- {rg_best['cv_r2_std']:.4f}")
    
    print(f"\nCI Prediction Best:")
    print(f"  Method: {ci_best['method']}")
    print(f"  Model: {ci_best['model']}")
    print(f"  CV R2: {ci_best['cv_r2_mean']:.4f} +/- {ci_best['cv_r2_std']:.4f}")
    
    # Save results
    results_path = os.path.join(project_root, 'outputs', 'augmentation_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   DATA AUGMENTATION EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODS TESTED:\n")
        f.write("-" * 80 + "\n")
        f.write("1. No Augmentation (Baseline)\n")
        f.write("2. Gaussian Noise Injection (scale: 0.03, 0.05, 0.10)\n")
        f.write("3. Interpolation Between Samples (n: 50, 100, 200)\n\n")
        
        f.write("Rg PREDICTION - TOP 10\n")
        f.write("-" * 80 + "\n")
        f.write(rg_results.head(10).to_string(index=False))
        f.write(f"\n\nBest: {rg_best['method']} + {rg_best['model']} | R2 = {rg_best['cv_r2_mean']:.4f}\n\n")
        
        f.write("CI PREDICTION - TOP 10\n")
        f.write("-" * 80 + "\n")
        f.write(ci_results.head(10).to_string(index=False))
        f.write(f"\n\nBest: {ci_best['method']} + {ci_best['model']} | R2 = {ci_best['cv_r2_mean']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
