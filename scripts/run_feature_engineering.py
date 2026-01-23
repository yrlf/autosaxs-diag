"""
Enhanced Feature Engineering Analysis
======================================
Use additional physical/chemical features from the dataset:
- Rg_err, chi2_red, t_a (quality indicators)
- n_real_peaks, B_factor, fwhm_avg/min/max (structural)
- snr_peak1 (signal quality)
- Polynomial & interaction terms
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_and_prepare_data(filepath, r2_threshold=0.80):
    """Load data with extended features."""
    df = pd.read_csv(filepath, header=0)
    
    # Rename key columns
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
    
    # Convert y2 to numeric
    df['y2'] = df['y2'].astype(str).str.lower().map({'false': 0, 'true': 1})
    
    # Set CI=0 when y2=0
    df.loc[df['y2'] == 0, 'CI'] = 0.0
    df['CI'] = pd.to_numeric(df['CI'], errors='coerce').fillna(0.0)
    
    # Filter by R2
    df['R2_w'] = pd.to_numeric(df['R2_w'], errors='coerce')
    df = df[df['R2_w'] >= r2_threshold]
    
    print(f"Loaded {len(df)} samples after R2 filter")
    return df


def create_feature_sets(df):
    """Create different feature sets for comparison."""
    
    feature_sets = {}
    
    # Set 1: Basic (original)
    feature_sets['Basic (x1, x2)'] = ['x1', 'x2']
    
    # Set 2: Add polynomial features
    # These will be created separately
    
    # Set 3: Add structural features
    structural_features = ['x1', 'x2', 'n_real_peaks', 'B_factor', 'fwhm_avg']
    # Check which exist
    available = [f for f in structural_features if f in df.columns]
    if len(available) > 2:
        feature_sets['Structural'] = available
    
    # Set 4: Add quality indicators
    quality_features = ['x1', 'x2', 'Rg_err', 'chi2_red', 't_a']
    available = [f for f in quality_features if f in df.columns]
    if len(available) > 2:
        feature_sets['Quality'] = available
    
    # Set 5: Combined
    all_candidates = ['x1', 'x2', 'n_real_peaks', 'B_factor', 'fwhm_avg', 
                     'Rg_err', 'chi2_red', 't_a', 'snr_peak1']
    available = [f for f in all_candidates if f in df.columns]
    # Convert to numeric and check non-null
    for f in available:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    
    # Filter features with enough non-null values
    good_features = [f for f in available if df[f].notna().sum() > 0.9 * len(df)]
    if len(good_features) > 2:
        feature_sets['Combined'] = good_features
    
    return feature_sets


def create_engineered_features(X, feature_names):
    """Create additional engineered features."""
    X = np.array(X)
    new_features = X.copy()
    new_names = list(feature_names)
    
    if X.shape[1] >= 2:
        # x1/x2 ratio (protein to EAN ratio)
        ratio = X[:, 1] / (X[:, 0] + 0.1)  # Add small value to avoid div by 0
        new_features = np.column_stack([new_features, ratio])
        new_names.append('x2/x1')
        
        # x1 * x2 interaction
        interaction = X[:, 0] * X[:, 1]
        new_features = np.column_stack([new_features, interaction])
        new_names.append('x1*x2')
        
        # Log transforms (for features > 0)
        if np.all(X[:, 0] > 0) and np.all(X[:, 1] > 0):
            log_x1 = np.log1p(X[:, 0])
            log_x2 = np.log1p(X[:, 1])
            new_features = np.column_stack([new_features, log_x1, log_x2])
            new_names.extend(['log(x1)', 'log(x2)'])
    
    return new_features, new_names


def run_feature_comparison(df, target_col, target_name, test_size=0.2, random_state=42):
    """Compare different feature sets."""
    
    print(f"\n{'='*70}")
    print(f"   FEATURE ENGINEERING ANALYSIS: {target_name}")
    print(f"{'='*70}")
    
    # Get feature sets
    feature_sets = create_feature_sets(df)
    
    # Prepare target
    data = df.dropna(subset=[target_col]).copy()
    y = data[target_col].values
    
    results = []
    
    for set_name, features in feature_sets.items():
        print(f"\n--- Feature Set: {set_name} ---")
        print(f"  Features: {features}")
        
        # Check for missing values
        subset = data[features].copy()
        for f in features:
            subset[f] = pd.to_numeric(subset[f], errors='coerce')
        
        # Drop rows with NaN in any feature
        valid_mask = subset.notna().all(axis=1)
        X = subset[valid_mask].values
        y_valid = y[valid_mask]
        
        print(f"  Samples with complete data: {len(y_valid)}")
        
        if len(y_valid) < 50:
            print(f"  Skipping (not enough samples)")
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_valid, test_size=test_size, random_state=random_state
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test models
        for model_name, model in [
            ('RF', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
            ('GB', GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)),
        ]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # CV on training
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results.append({
                'feature_set': set_name,
                'n_features': len(features),
                'n_samples': len(y_valid),
                'model': model_name,
                'cv_r2': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'test_r2': test_r2,
                'test_rmse': test_rmse
            })
            
            print(f"  {model_name}: Test R2 = {test_r2:.4f}")
    
    # Now test with engineered features
    print(f"\n--- Feature Set: Engineered (x1, x2 + derived) ---")
    
    X_basic = data[['x1', 'x2']].values
    X_eng, eng_names = create_engineered_features(X_basic, ['x1', 'x2'])
    
    print(f"  Features: {eng_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, model in [
        ('RF', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        ('GB', GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results.append({
            'feature_set': 'Engineered',
            'n_features': len(eng_names),
            'n_samples': len(y),
            'model': model_name,
            'cv_r2': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'test_r2': test_r2,
            'test_rmse': test_rmse
        })
        
        print(f"  {model_name}: Test R2 = {test_r2:.4f}")
        
        # Feature importance
        if model_name == 'RF':
            importances = model.feature_importances_
            print(f"\n  Feature Importance ({model_name}):")
            for name, imp in sorted(zip(eng_names, importances), key=lambda x: -x[1]):
                print(f"    {name}: {imp:.4f}")
    
    # Summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"   RESULTS SUMMARY: {target_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Feature Set':<20} {'Model':<6} {'#Feat':<6} {'CV R2':>8} {'Test R2':>9}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['feature_set']:<20} {row['model']:<6} {row['n_features']:<6} {row['cv_r2']:>8.4f} {row['test_r2']:>9.4f}")
    
    best = results_df.iloc[0]
    baseline = results_df[results_df['feature_set'] == 'Basic (x1, x2)'].iloc[0]
    
    improvement = best['test_r2'] - baseline['test_r2']
    
    print(f"\nBaseline (x1, x2): Test R2 = {baseline['test_r2']:.4f}")
    print(f"Best: {best['feature_set']} + {best['model']}: Test R2 = {best['test_r2']:.4f}")
    
    if improvement > 0.01:
        print(f"\n>>> IMPROVEMENT: +{improvement:.4f} ({improvement/baseline['test_r2']*100:.1f}%)")
    
    return results_df, best


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    
    print("=" * 70)
    print("   FEATURE ENGINEERING ANALYSIS")
    print("   Testing Additional Physical/Chemical Features")
    print("=" * 70)
    
    df = load_and_prepare_data(input_file, r2_threshold=0.80)
    
    # Run for both targets
    rg_results, rg_best = run_feature_comparison(df, 'y1', 'Rg')
    ci_results, ci_best = run_feature_comparison(df, 'CI', 'CI')
    
    # Final summary
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nRg Prediction:")
    print(f"  Best: {rg_best['feature_set']} + {rg_best['model']}")
    print(f"  Test R2: {rg_best['test_r2']:.4f}")
    
    print(f"\nCI Prediction:")
    print(f"  Best: {ci_best['feature_set']} + {ci_best['model']}")
    print(f"  Test R2: {ci_best['test_r2']:.4f}")
    
    # Save
    results_path = os.path.join(project_root, 'outputs', 'feature_engineering_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   FEATURE ENGINEERING ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("FEATURE SETS TESTED:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Basic: x1, x2 (original)\n")
        f.write("2. Structural: x1, x2, n_real_peaks, B_factor, fwhm_avg\n")
        f.write("3. Quality: x1, x2, Rg_err, chi2_red, t_a\n")
        f.write("4. Combined: All above features\n")
        f.write("5. Engineered: x1, x2, x2/x1, x1*x2, log(x1), log(x2)\n\n")
        
        f.write("Rg RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(rg_results.to_string(index=False))
        f.write(f"\n\nBest: {rg_best['feature_set']} + {rg_best['model']} | R2 = {rg_best['test_r2']:.4f}\n\n")
        
        f.write("CI RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(ci_results.to_string(index=False))
        f.write(f"\n\nBest: {ci_best['feature_set']} + {ci_best['model']} | R2 = {ci_best['test_r2']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
