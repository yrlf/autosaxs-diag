"""
Comprehensive Multi-Model Evaluation
=====================================
Models: LR, RF, NN, SVM, XGBoost, GBM, AdaBoost, DT
Metrics: Youden index, F1, Calibration slope, Brier score, AUC, ACC
Plots: SHAP, Feature importance, DCA, Calibration

Results saved to: outputs/evaluation/
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, roc_curve, f1_score, accuracy_score, 
    precision_score, recall_score, confusion_matrix,
    brier_score_loss, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy import stats

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.data_cleaning import DataCleaner


def get_classification_models():
    """Return dictionary of classification models."""
    models = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'DT': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'NN': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                                          use_label_encoder=False, eval_metric='logloss')
    return models


def get_regression_models():
    """Return dictionary of regression models."""
    models = {
        'LR': Ridge(alpha=1.0, random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'DT': DecisionTreeRegressor(max_depth=10, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'NN': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
    return models


def calculate_classification_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all classification metrics."""
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity & Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Youden Index
    youden = sensitivity + specificity - 1
    
    # AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        auc = np.nan
    
    # Brier Score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Calibration slope
    try:
        slope, intercept, r_value, _, _ = stats.linregress(y_pred_proba, y_true)
        calibration_slope = slope
    except:
        calibration_slope = np.nan
    
    return {
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Youden_Index': youden,
        'AUC': auc,
        'Brier_Score': brier,
        'Calibration_Slope': calibration_slope
    }


def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calibration slope
    try:
        slope, intercept, r_value, _, _ = stats.linregress(y_pred, y_true)
        calibration_slope = slope
        calibration_r = r_value
    except:
        calibration_slope = np.nan
        calibration_r = np.nan
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Calibration_Slope': calibration_slope,
        'Calibration_R': calibration_r
    }


def plot_roc_curves(results, output_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        if 'fpr' in data and 'tpr' in data:
            plt.plot(data['fpr'], data['tpr'], 
                    label=f"{name} (AUC = {data['metrics']['AUC']:.3f})", lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(results, output_path):
    """Plot calibration curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        if 'y_test' in data and 'y_pred_proba' in data:
            prob_true, prob_pred = calibration_curve(data['y_test'], data['y_pred_proba'], n_bins=5)
            plt.plot(prob_pred, prob_true, 's-', label=name, lw=2, markersize=8)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curves - All Models', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dca(results, output_path):
    """Plot Decision Curve Analysis."""
    plt.figure(figsize=(10, 8))
    
    # Threshold range
    thresholds = np.linspace(0.01, 0.99, 50)
    
    for name, data in results.items():
        if 'y_test' in data and 'y_pred_proba' in data:
            y_test = data['y_test']
            y_proba = data['y_pred_proba']
            
            net_benefits = []
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                tp = np.sum((y_pred == 1) & (y_test == 1))
                fp = np.sum((y_pred == 1) & (y_test == 0))
                n = len(y_test)
                
                net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
                net_benefits.append(net_benefit)
            
            plt.plot(thresholds, net_benefits, label=name, lw=2)
    
    # Treat all
    prevalence = np.mean(list(results.values())[0]['y_test'])
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_all, 'k--', label='Treat All', lw=2)
    
    # Treat none
    plt.axhline(y=0, color='gray', linestyle=':', label='Treat None', lw=2)
    
    plt.xlim([0, 1])
    plt.ylim([-0.1, max(0.5, prevalence + 0.1)])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(results, feature_names, output_path):
    """Plot feature importance for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    importance_data = []
    for name, data in results.items():
        if data.get('feature_importance') is not None:
            for i, fname in enumerate(feature_names):
                importance_data.append({
                    'Model': name,
                    'Feature': fname,
                    'Importance': data['feature_importance'][i]
                })
    
    if importance_data:
        df = pd.DataFrame(importance_data)
        pivot = df.pivot(index='Model', columns='Feature', values='Importance')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Feature Importance by Model', fontsize=14)
        ax.legend(title='Feature')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_summary(model, X_train, feature_names, model_name, output_path):
    """Generate SHAP summary plot."""
    if not HAS_SHAP:
        return
    
    try:
        if hasattr(model, 'predict_proba') or hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
        
        shap_values = explainer.shap_values(X_train[:50])
        
        plt.figure(figsize=(8, 4))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap.summary_plot(shap_values, X_train[:50], feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")


def run_classification_evaluation(df, target_col, target_name, output_dir):
    """Run classification evaluation for all models."""
    
    print(f"\n{'='*70}")
    print(f"   CLASSIFICATION: {target_name}")
    print(f"{'='*70}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data - binary classification (CI > 0.1 = crystalline)
    X = df[['x1', 'x2']].values
    y = (df[target_col] > 0.1).astype(int).values
    feature_names = ['x1 (EAN)', 'x2 (Protein)']
    
    print(f"Samples: {len(y)} | Positive: {y.sum()} | Negative: {len(y) - y.sum()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Get models
    models = get_classification_models()
    results = {}
    
    print(f"\n{'Model':<12} {'AUC':>8} {'ACC':>8} {'F1':>8} {'Youden':>8} {'Brier':>8}")
    print("-" * 60)
    
    for name, model in models.items():
        try:
            # Train
            model.fit(X_train_s, y_train)
            
            # Predict
            y_pred = model.predict(X_test_s)
            y_pred_proba = model.predict_proba(X_test_s)[:, 1]
            
            # Calculate metrics
            metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feat_imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feat_imp = np.abs(model.coef_[0])
            else:
                feat_imp = None
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'fpr': fpr,
                'tpr': tpr,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'feature_importance': feat_imp
            }
            
            print(f"{name:<12} {metrics['AUC']:>8.4f} {metrics['Accuracy']:>8.4f} "
                  f"{metrics['F1']:>8.4f} {metrics['Youden_Index']:>8.4f} {metrics['Brier_Score']:>8.4f}")
            
        except Exception as e:
            print(f"{name:<12} Error: {e}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_roc_curves(results, os.path.join(output_dir, 'roc_curves.png'))
    print("  Saved: roc_curves.png")
    
    plot_calibration_curves(results, os.path.join(output_dir, 'calibration_curves.png'))
    print("  Saved: calibration_curves.png")
    
    plot_dca(results, os.path.join(output_dir, 'dca.png'))
    print("  Saved: dca.png")
    
    plot_feature_importance(results, feature_names, os.path.join(output_dir, 'feature_importance.png'))
    print("  Saved: feature_importance.png")
    
    # SHAP for best model
    best_model_name = max(results, key=lambda x: results[x]['metrics']['AUC'])
    shap_path = os.path.join(output_dir, f'shap_{best_model_name}.png')
    plot_shap_summary(results[best_model_name]['model'], X_train_s, feature_names, 
                     best_model_name, shap_path)
    print(f"  Saved: shap_{best_model_name}.png")
    
    # Save metrics table
    metrics_df = pd.DataFrame([{**{'Model': name}, **data['metrics']} 
                               for name, data in results.items()])
    metrics_df = metrics_df.sort_values('AUC', ascending=False)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print("  Saved: metrics.csv")
    
    return results, metrics_df


def run_regression_evaluation(df, target_col, target_name, output_dir):
    """Run regression evaluation for all models."""
    
    print(f"\n{'='*70}")
    print(f"   REGRESSION: {target_name}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X = df[['x1', 'x2']].values
    y = df[target_col].values
    feature_names = ['x1 (EAN)', 'x2 (Protein)']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = get_regression_models()
    results = {}
    
    print(f"\n{'Model':<12} {'R2':>8} {'RMSE':>10} {'MAE':>10} {'Cal_Slope':>10}")
    print("-" * 55)
    
    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            metrics = calculate_regression_metrics(y_test, y_pred)
            
            if hasattr(model, 'feature_importances_'):
                feat_imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feat_imp = np.abs(model.coef_) if model.coef_.ndim == 1 else np.abs(model.coef_[0])
            else:
                feat_imp = None
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_importance': feat_imp
            }
            
            print(f"{name:<12} {metrics['R2']:>8.4f} {metrics['RMSE']:>10.4f} "
                  f"{metrics['MAE']:>10.4f} {metrics['Calibration_Slope']:>10.4f}")
            
        except Exception as e:
            print(f"{name:<12} Error: {e}")
    
    # Calibration plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(results.items()):
        if i < len(axes):
            ax = axes[i]
            ax.scatter(data['y_pred'], data['y_test'], alpha=0.6)
            min_val = min(data['y_test'].min(), data['y_pred'].min())
            max_val = max(data['y_test'].max(), data['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f"{name} (R²={data['metrics']['R2']:.3f})")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_all_models.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: calibration_all_models.png")
    
    # Feature importance
    plot_feature_importance(results, feature_names, os.path.join(output_dir, 'feature_importance.png'))
    print("Saved: feature_importance.png")
    
    # Save metrics
    metrics_df = pd.DataFrame([{**{'Model': name}, **data['metrics']} 
                               for name, data in results.items()])
    metrics_df = metrics_df.sort_values('R2', ascending=False)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print("Saved: metrics.csv")
    
    return results, metrics_df


def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    output_base = os.path.join(project_root, 'outputs', 'evaluation')
    
    print("=" * 70)
    print("   COMPREHENSIVE MULTI-MODEL EVALUATION")
    print("   Models: LR, RF, DT, SVM, NN, GBM, AdaBoost, XGBoost")
    print("=" * 70)
    
    # Load data
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    
    # Rg Regression
    rg_dir = os.path.join(output_base, 'Rg_regression')
    rg_results, rg_metrics = run_regression_evaluation(df, 'y1', 'Rg', rg_dir)
    
    # CI Regression
    ci_reg_dir = os.path.join(output_base, 'CI_regression')
    ci_reg_results, ci_reg_metrics = run_regression_evaluation(df, 'CI', 'CI', ci_reg_dir)
    
    # CI Classification (CI > 0.1 = crystalline)
    ci_clf_dir = os.path.join(output_base, 'CI_classification')
    ci_clf_results, ci_clf_metrics = run_classification_evaluation(df, 'CI', 'CI (Crystalline)', ci_clf_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("   FINAL SUMMARY")
    print("=" * 70)
    
    print("\nRg Regression - Best Models:")
    print(rg_metrics.head(3).to_string(index=False))
    
    print("\nCI Regression - Best Models:")
    print(ci_reg_metrics.head(3).to_string(index=False))
    
    print("\nCI Classification - Best Models:")
    print(ci_clf_metrics[['Model', 'AUC', 'F1', 'Youden_Index', 'Brier_Score']].head(3).to_string(index=False))
    
    print(f"\n\nResults saved to: {output_base}/")
    print("  - Rg_regression/")
    print("  - CI_regression/")
    print("  - CI_classification/")


if __name__ == "__main__":
    main()
