"""
Generate Publication-Quality Plots
==================================
Generates scientific plots for both Regression (Rg, CI) and Classification (CI > 0.1).
Includes:
- Model Comparison Bar Charts (R2, AUC, F1, etc.)
- ROC Curves
- Decision Curve Analysis (DCA)
- Calibration Curves
- SHAP Summary Plots

Style: High-impact journal quality (Arial, No Grid, High DPI).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, roc_curve, auc, brier_score_loss, accuracy_score, f1_score, confusion_matrix
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.data_cleaning import DataCleaner

# ==========================================
# 1. SETUP & STYLING
# ==========================================

def set_scientific_style():
    """Set matplotlib style for scientific publication."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.5)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'lines.linewidth': 2,
        'axes.grid': False
    })

# ==========================================
# 2. MODELS
# ==========================================

def get_regression_models():
    models = {
        'Linear Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
    return models

def get_classification_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss')
    return models

# ==========================================
# 3. EVALUATION FUNCTIONS
# ==========================================

def evaluate_classification(model, X, y, model_name):
    """Run CV to get probas and calculate all metrics."""
    # Use cross_val_predict to get out-of-fold predictions
    # This might fail for some models without predict_proba, but all selected models support it
    y_probas = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
    y_preds = (y_probas >= 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    fpr, tpr, _ = roc_curve(y, y_probas)
    auc_val = auc(fpr, tpr)
    brier = brier_score_loss(y, y_probas)
    
    # Youden Index
    tn, fp, fn, tp = confusion_matrix(y, y_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    
    # Calibration Slope (linear regression of probas vs actual 0/1 doesn't make sense directly, 
    # but usually done on binned data or just regressing y ~ proba? 
    # Standard way: Logistic calibration or just linear fit of points in reliability diagram.
    # Here we perform linear regression of the reliability curve points themselves or just P(observed) vs P(predicted)
    # Simple method: slope of standard errors? No.
    # We will use slope of linear regression of y on log-odds (logit) of probability?
    # Let's use simple linear regression of observations vs probabilities for now, though logit is more correct.
    slope, intercept, _, _, _ = stats.linregress(y_probas, y)
    
    # DCA Data
    dca_points = []
    thresholds = np.linspace(0.01, 0.99, 50)
    for thresh in thresholds:
        tp_thresh = np.sum((y_probas >= thresh) & (y == 1))
        fp_thresh = np.sum((y_probas >= thresh) & (y == 0))
        n = len(y)
        net_benefit = (tp_thresh / n) - (fp_thresh / n) * (thresh / (1 - thresh))
        dca_points.append(net_benefit)
        
    return {
        'Model': model_name,
        'AUC': auc_val,
        'ACC': acc,
        'F1': f1,
        'Youden': youden,
        'Brier': brier,
        'Calibration_Slope': slope,
        'y_true': y,
        'y_proba': y_probas,
        'dca_thresholds': thresholds,
        'dca_benefits': dca_points
    }

def evaluate_regression(model, X, y):
    """Run CV to get R2 scores."""
    scores = []
    for i in range(5):
        cv_preds = cross_val_predict(model, X, y, cv=5)
        scores.append(r2_score(y, cv_preds))
    return np.mean(scores), np.std(scores)

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================

def plot_regression_bar(results_df, output_path):
    """Function to plot R2 comparison (Vertical Labels)."""
    set_scientific_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort
    df_sorted = results_df.sort_values('R2_Mean', ascending=False)
    
    # Palette
    palette = sns.color_palette("viridis", n_colors=len(df_sorted))
    
    sns.barplot(data=df_sorted, x='Model', y='R2_Mean', ax=ax, palette=palette, edgecolor='black', linewidth=1.5)
    ax.errorbar(x=range(len(df_sorted)), y=df_sorted['R2_Mean'], yerr=df_sorted['R2_Std'], 
                fmt='none', c='black', capsize=5, elinewidth=2)
    
    ax.set_title('Rg Prediction Performance', fontweight='bold', pad=20)
    ax.set_ylabel('$R^2$ Score', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=90) # Vertical
    
    # Thick spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_classification_metrics_bar(results_list, output_path):
    """Grouped bar chart for Classification Metrics."""
    set_scientific_style()
    
    # Prepare DataFrame
    metrics = ['AUC', 'F1', 'ACC', 'Youden']
    data = []
    for r in results_list:
        for m in metrics:
            data.append({'Model': r['Model'], 'Metric': m, 'Value': r[m]})
    df = pd.DataFrame(data)
    
    # Sort models by AUC
    rank = df[df['Metric']=='AUC'].sort_values('Value', ascending=False)['Model'].tolist()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.barplot(data=df, x='Model', y='Value', hue='Metric', order=rank,
                palette='Set2', edgecolor='black', linewidth=1.2, ax=ax)
    
    ax.set_title('Classification Model Performance (CI)', fontweight='bold', pad=20)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.tick_params(axis='x', rotation=90) # Vertical
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curves_panel(results_list, output_path):
    """Panel with ROC, DCA, Calibration."""
    set_scientific_style()
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Sort by AUC
    results_list.sort(key=lambda x: x['AUC'], reverse=True)
    top_models = results_list[:5] # Plot only top 5 to avoid clutter
    colors = sns.color_palette("deep", n_colors=len(top_models))
    
    # 1. ROC Curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    for i, res in enumerate(top_models):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
        ax.plot(fpr, tpr, lw=2.5, color=colors[i], label=f"{res['Model']} ({res['AUC']:.2f})")
    ax.set_title('ROC Curve', fontweight='bold')
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.legend(frameon=False, fontsize=10)
    
    # 2. Decision Curve Analysis (DCA)
    ax = axes[1]
    ax.axhline(0, color='gray', lw=1.5, linestyle=':')
    prev = np.mean(results_list[0]['y_true'])
    thresholds = results_list[0]['dca_thresholds']
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all, 'k--', label='Treat All', lw=1.5)
    
    for i, res in enumerate(top_models):
        ax.plot(res['dca_thresholds'], res['dca_benefits'], lw=2.5, color=colors[i], label=res['Model'])
    
    ax.set_title('Decision Curve Analysis', fontweight='bold')
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_ylim(-0.05, max(0.4, np.max(treat_all)))
    
    # 3. Calibration Curve
    ax = axes[2]
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    for i, res in enumerate(top_models):
        prob_true, prob_pred = calibration_curve(res['y_true'], res['y_proba'], n_bins=5)
        ax.plot(prob_pred, prob_true, 's-', lw=2.5, markersize=6, color=colors[i], label=res['Model'])
    
    ax.set_title('Calibration Curve', fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    
    # 4. Metrics Table
    ax = axes[3]
    ax.axis('off')
    table_data = [['Model', 'Slope', 'Brier']]
    for res in top_models:
        table_data.append([res['Model'], f"{res['Calibration_Slope']:.2f}", f"{res['Brier']:.3f}"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax.set_title('Calibration Metrics', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_pub(model, X, feature_names, output_path):
    """High quality SHAP summary."""
    if not HAS_SHAP: return
    
    # Must use TreeExplainer for tree models, Linear for linear, etc
    # For simplicity, generic or Tree
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
    except:
        # Fallback for some models
        try:
             explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))
             shap_values = explainer.shap_values(X)
        except:
             return

    plt.figure(figsize=(10, 6))
    # Check shape of shap_values for classifier (it outputs list for classes)
    if isinstance(shap_values, list):
         # Class 1
         shap.summary_plot(shap_values[1], X, feature_names=feature_names, show=False)
    elif len(shap_values.shape) == 3:
         shap.summary_plot(shap_values[:,:,1], X, feature_names=feature_names, show=False)
    else:
         shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    ax = plt.gca()
    ax.set_title(f'SHAP Feature Importance', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v3.csv')
    output_dir = os.path.join(project_root, 'outputs', 'publication_figures')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    cleaner = DataCleaner(input_file)
    cleaner.load_data(header_row=0)
    df = cleaner.clean_data(r2_threshold=0.80)
    
    X = df[['x1', 'x2']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = ['EAN Conc.', 'Protein Conc.']
    
    # --- REGRESSION (Rg) ---
    print("Evaluating Regression (Rg)...")
    y_reg = df['y1'].values
    reg_models = get_regression_models()
    reg_results = []
    
    for name, model in reg_models.items():
        try:
            mean, std = evaluate_regression(model, X_scaled, y_reg)
            reg_results.append({'Model': name, 'R2_Mean': mean, 'R2_Std': std})
        except Exception as e:
            print(f"Skipping {name}: {e}")
    
    pd.DataFrame(reg_results).to_csv(os.path.join(output_dir, 'rg_metrics.csv'), index=False)
    plot_regression_bar(pd.DataFrame(reg_results), os.path.join(output_dir, 'Rg_Performance_Bar.png'))
    
    # --- CLASSIFICATION (CI) ---
    print("Evaluating Classification (CI > 0.1)...")
    y_clf = (df['CI'] > 0.1).astype(int).values
    clf_models = get_classification_models()
    clf_results = []
    
    for name, model in clf_models.items():
        try:
            model.fit(X_scaled, y_clf) # create a fitted instance for shap later, but evaluation function uses CV
            # Actually evaluate_classification does CV internally, but we need a fitted model for SHAP?
            # Let's pass the fresh model to evaluate_classification which does cross_val_predict
            res = evaluate_classification(model, X_scaled, y_clf, name)
            clf_results.append(res)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            
    # Save Metrics
    metrics_cols = ['Model', 'AUC', 'F1', 'ACC', 'Youden', 'Brier', 'Calibration_Slope']
    pd.DataFrame([{k: v for k, v in r.items() if k in metrics_cols} for r in clf_results]) \
      .to_csv(os.path.join(output_dir, 'ci_metrics.csv'), index=False)
    
    # Plots
    print("Generating Classification Plots...")
    plot_classification_metrics_bar(clf_results, os.path.join(output_dir, 'CI_Metrics_Bar.png'))
    plot_curves_panel(clf_results, os.path.join(output_dir, 'CI_Curves_Panel.png'))
    
    # SHAP for Best Classifier
    if clf_results:
        best_clf_idx = np.argmax([r['AUC'] for r in clf_results])
        best_model_name = clf_results[best_clf_idx]['Model']
        print(f"Generating SHAP for best classifier: {best_model_name}")
        
        # Retrain best model on full data for SHAP
        best_model = clf_models[best_model_name]
        best_model.fit(X_scaled, y_clf)
        plot_shap_pub(best_model, X_scaled, feature_names, os.path.join(output_dir, f'SHAP_{best_model_name}.png'))
    
    print("Done. All plots updated.")

if __name__ == "__main__":
    main()
