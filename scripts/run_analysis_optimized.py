"""
Optimized ML Analysis Script - Enhanced Performance
====================================================
Improvements:
1. Polynomial feature engineering (degree 2-3)
2. Hyperparameter tuning with GridSearchCV
3. Stacking ensemble models
4. Repeated cross-validation for stable results
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Define paths relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, KFold, GridSearchCV, RepeatedKFold
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    StackingRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_cleaning import DataCleaner


def create_polynomial_features(X, degree=2):
    """Create polynomial and interaction features."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(['x1', 'x2'])
    return X_poly, feature_names


def tune_random_forest(X, y, cv=5):
    """Tune Random Forest with GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_gradient_boosting(X, y, cv=5):
    """Tune Gradient Boosting with GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_split': [2, 5, 10]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_xgboost(X, y, cv=5):
    """Tune XGBoost with GridSearchCV."""
    if not HAS_XGBOOST:
        return None, None, None
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBRegressor(random_state=42, verbosity=0)
    grid_search = GridSearchCV(
        xgb, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def tune_svr(X, y, cv=5):
    """Tune SVR with GridSearchCV."""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(
        svr, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def create_stacking_ensemble(base_models, meta_model=None):
    """Create a stacking ensemble."""
    if meta_model is None:
        meta_model = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    return stacking


def create_voting_ensemble(models):
    """Create a voting ensemble."""
    voting = VotingRegressor(
        estimators=models,
        n_jobs=-1
    )
    return voting


def evaluate_model(model, X, y, cv=None, name="Model"):
    """Evaluate model with repeated cross-validation."""
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    # Fit on full data for training metrics
    model.fit(X, y)
    y_pred = model.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mse = mean_squared_error(y, y_pred)
    
    return {
        'name': name,
        'cv_r2_mean': np.mean(scores),
        'cv_r2_std': np.std(scores),
        'train_r2': train_r2,
        'train_mse': train_mse
    }


def run_optimized_analysis(df, target_col, target_name):
    """Run optimized analysis for a target variable."""
    
    print(f"\n{'='*70}")
    print(f"   OPTIMIZED ANALYSIS: {target_name}")
    print(f"{'='*70}")
    
    # Prepare data
    data = df.dropna(subset=[target_col]).copy()
    X_raw = data[['x1', 'x2']].values
    y = data[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Create polynomial features (degree 2)
    X_poly2, poly2_names = create_polynomial_features(X_scaled, degree=2)
    # Create polynomial features (degree 3)
    X_poly3, poly3_names = create_polynomial_features(X_scaled, degree=3)
    
    print(f"\nData: {len(y)} samples")
    print(f"Original features: 2 (x1, x2)")
    print(f"Poly degree 2 features: {X_poly2.shape[1]} ({', '.join(poly2_names)})")
    print(f"Poly degree 3 features: {X_poly3.shape[1]}")
    
    results = []
    
    # ==============================
    # 1. Baseline (Original Features)
    # ==============================
    print(f"\n--- Phase 1: Baseline Models (Original Features) ---")
    
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42)
    result = evaluate_model(rf_base, X_scaled, y, name="RF (Baseline)")
    results.append(result)
    print(f"Random Forest (Baseline): R2 = {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    # ==============================
    # 2. Polynomial Features
    # ==============================
    print(f"\n--- Phase 2: Polynomial Feature Engineering ---")
    
    rf_poly2 = RandomForestRegressor(n_estimators=100, random_state=42)
    result = evaluate_model(rf_poly2, X_poly2, y, name="RF (Poly2)")
    results.append(result)
    print(f"Random Forest (Poly2): R2 = {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    rf_poly3 = RandomForestRegressor(n_estimators=100, random_state=42)
    result = evaluate_model(rf_poly3, X_poly3, y, name="RF (Poly3)")
    results.append(result)
    print(f"Random Forest (Poly3): R2 = {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    # ==============================
    # 3. Hyperparameter Tuning
    # ==============================
    print(f"\n--- Phase 3: Hyperparameter Tuning (GridSearchCV) ---")
    
    # Choose best feature set based on preliminary results
    best_X = X_poly2 if results[-2]['cv_r2_mean'] > results[-1]['cv_r2_mean'] else X_poly3
    best_poly = "Poly2" if results[-2]['cv_r2_mean'] > results[-1]['cv_r2_mean'] else "Poly3"
    print(f"Using {best_poly} features for tuning...")
    
    print("Tuning Random Forest...")
    rf_tuned, rf_params, rf_score = tune_random_forest(best_X, y)
    result = evaluate_model(rf_tuned, best_X, y, name=f"RF Tuned ({best_poly})")
    results.append(result)
    print(f"  Best R2: {result['cv_r2_mean']:.4f} | Params: {rf_params}")
    
    print("Tuning Gradient Boosting...")
    gb_tuned, gb_params, gb_score = tune_gradient_boosting(best_X, y)
    result = evaluate_model(gb_tuned, best_X, y, name=f"GB Tuned ({best_poly})")
    results.append(result)
    print(f"  Best R2: {result['cv_r2_mean']:.4f} | Params: {gb_params}")
    
    print("Tuning SVR...")
    svr_tuned, svr_params, svr_score = tune_svr(best_X, y)
    result = evaluate_model(svr_tuned, best_X, y, name=f"SVR Tuned ({best_poly})")
    results.append(result)
    print(f"  Best R2: {result['cv_r2_mean']:.4f} | Params: {svr_params}")
    
    if HAS_XGBOOST:
        print("Tuning XGBoost...")
        xgb_tuned, xgb_params, xgb_score = tune_xgboost(best_X, y)
        result = evaluate_model(xgb_tuned, best_X, y, name=f"XGB Tuned ({best_poly})")
        results.append(result)
        print(f"  Best R2: {result['cv_r2_mean']:.4f} | Params: {xgb_params}")
    
    # ==============================
    # 4. Ensemble Methods
    # ==============================
    print(f"\n--- Phase 4: Ensemble Methods ---")
    
    # Stacking Ensemble
    base_models = [
        ('rf', rf_tuned),
        ('gb', gb_tuned),
        ('svr', svr_tuned)
    ]
    if HAS_XGBOOST:
        base_models.append(('xgb', xgb_tuned))
    
    stacking = create_stacking_ensemble(base_models)
    result = evaluate_model(stacking, best_X, y, name=f"Stacking ({best_poly})")
    results.append(result)
    print(f"Stacking Ensemble: R2 = {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    # Voting Ensemble
    voting = create_voting_ensemble(base_models)
    result = evaluate_model(voting, best_X, y, name=f"Voting ({best_poly})")
    results.append(result)
    print(f"Voting Ensemble: R2 = {result['cv_r2_mean']:.4f} +/- {result['cv_r2_std']:.4f}")
    
    # ==============================
    # 5. Summary
    # ==============================
    print(f"\n{'='*70}")
    print(f"   RESULTS SUMMARY: {target_name}")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cv_r2_mean', ascending=False)
    
    print(f"\n{'Model':<30} {'CV R2 Mean':>12} {'CV R2 Std':>12} {'Train R2':>12}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['name']:<30} {row['cv_r2_mean']:>12.4f} {row['cv_r2_std']:>12.4f} {row['train_r2']:>12.4f}")
    
    best = results_df.iloc[0]
    print(f"\nBEST MODEL: {best['name']}")
    print(f"  CV R2 = {best['cv_r2_mean']:.4f} +/- {best['cv_r2_std']:.4f}")
    
    return results_df, best


def main():
    # Load and clean data
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    
    print("=" * 70)
    print("   OPTIMIZED ML ANALYSIS")
    print("   With Feature Engineering, Hyperparameter Tuning & Ensembles")
    print("=" * 70)
    
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    
    # Task 1: Predict Rg
    rg_results, rg_best = run_optimized_analysis(df, 'y1', 'Rg (Radius of Gyration)')
    
    # Task 2: Predict CI
    ci_results, ci_best = run_optimized_analysis(df, 'CI', 'CI (Crystallinity Index)')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("   FINAL OPTIMIZATION RESULTS")
    print("=" * 70)
    
    print(f"\nRg Prediction:")
    print(f"  Best Model: {rg_best['name']}")
    print(f"  CV R2: {rg_best['cv_r2_mean']:.4f} +/- {rg_best['cv_r2_std']:.4f}")
    
    print(f"\nCI Prediction:")
    print(f"  Best Model: {ci_best['name']}")
    print(f"  CV R2: {ci_best['cv_r2_mean']:.4f} +/- {ci_best['cv_r2_std']:.4f}")
    
    # Save results
    results_path = os.path.join(project_root, 'outputs', 'optimized_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("   OPTIMIZED ML ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OPTIMIZATION TECHNIQUES APPLIED:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Polynomial Feature Engineering (degree 2-3)\n")
        f.write("2. Hyperparameter Tuning via GridSearchCV\n")
        f.write("3. Stacking & Voting Ensemble Methods\n")
        f.write("4. Repeated K-Fold Cross-Validation (5-fold x 3 repeats)\n\n")
        
        f.write("Rg (RADIUS OF GYRATION) PREDICTION\n")
        f.write("-" * 80 + "\n")
        f.write(rg_results.to_string(index=False))
        f.write(f"\n\nBest: {rg_best['name']} | R2 = {rg_best['cv_r2_mean']:.4f}\n\n")
        
        f.write("CI (CRYSTALLINITY INDEX) PREDICTION\n")
        f.write("-" * 80 + "\n")
        f.write(ci_results.to_string(index=False))
        f.write(f"\n\nBest: {ci_best['name']} | R2 = {ci_best['cv_r2_mean']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
