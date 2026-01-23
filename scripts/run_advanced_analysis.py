"""
Advanced ML Analysis with Hyperparameter Tuning and SHAP Interpretability
高级机器学习分析：10折交叉验证、超参数调优、全面评估指标、SHAP可解释性

评估指标:
- Brier Score (布里尔评分)
- Youden Index (约登指数 = Sensitivity + Specificity - 1)
- Calibration Slope (校准斜率)
- F1 Score
- AUC (ROC曲线下面积)
- Accuracy (准确率)

可解释性:
- SHAP (SHapley Additive exPlanations)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    cross_val_predict, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'DejaVu Sans'


class AdvancedClassificationAnalyzer:
    """高级分类分析器"""
    
    def __init__(self, df, target_col='y2', n_folds=10, random_state=42):
        self.df = df.dropna(subset=[target_col]).copy()
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.X = self.df[['x1', 'x2']].values
        self.y = self.df[target_col].astype(int).values
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        self.results = {}
        self.best_models = {}
        
        print(f"Dataset: {len(self.y)} samples")
        print(f"Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")
        print(f"Cross-validation: {n_folds}-Fold Stratified")
    
    def get_models_with_params(self):
        """返回模型及其超参数搜索空间"""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=self.random_state),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'SVC': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = {
                'model': XGBClassifier(random_state=self.random_state, verbosity=0, 
                                       use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        
        return models
    
    def calculate_youden_index(self, y_true, y_pred_proba, threshold=0.5):
        """计算约登指数 (Youden's J = Sensitivity + Specificity - 1)"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        youden_j = sensitivity + specificity - 1
        return youden_j, sensitivity, specificity
    
    def calculate_calibration_slope(self, y_true, y_pred_proba):
        """计算校准斜率"""
        from sklearn.linear_model import LogisticRegression
        
        # 使用logit变换的预测概率拟合
        epsilon = 1e-10
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        logit_pred = np.log(y_pred_proba / (1 - y_pred_proba))
        
        # 拟合校准斜率
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(logit_pred.reshape(-1, 1), y_true)
        
        calibration_slope = lr.coef_[0][0]
        return calibration_slope
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """使用约登指数找最优阈值"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, youden_index[optimal_idx]
    
    def tune_and_evaluate(self):
        """超参数调优并评估所有模型"""
        print("\n" + "=" * 70)
        print("   HYPERPARAMETER TUNING & EVALUATION (10-Fold CV)")
        print("=" * 70)
        
        models = self.get_models_with_params()
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        results = []
        
        for name, config in models.items():
            print(f"\n{'─' * 70}")
            print(f"Tuning: {name}")
            print(f"{'─' * 70}")
            
            # GridSearchCV
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(self.X_scaled, self.y)
            
            best_model = grid_search.best_estimator_
            self.best_models[name] = best_model
            
            print(f"Best params: {grid_search.best_params_}")
            
            # Cross-validation predictions
            y_pred_proba = cross_val_predict(
                best_model, self.X_scaled, self.y, 
                cv=cv, method='predict_proba'
            )[:, 1]
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate all metrics
            accuracy = accuracy_score(self.y, y_pred)
            f1 = f1_score(self.y, y_pred, average='weighted')
            auc = roc_auc_score(self.y, y_pred_proba)
            brier = brier_score_loss(self.y, y_pred_proba)
            
            # Youden Index with optimal threshold
            optimal_threshold, optimal_youden = self.find_optimal_threshold(self.y, y_pred_proba)
            youden, sensitivity, specificity = self.calculate_youden_index(self.y, y_pred_proba, 0.5)
            
            # Calibration slope
            calibration_slope = self.calculate_calibration_slope(self.y, y_pred_proba)
            
            result = {
                'Model': name,
                'Best Params': str(grid_search.best_params_),
                'Accuracy': accuracy,
                'F1 Score': f1,
                'AUC': auc,
                'Brier Score': brier,
                'Youden Index': youden,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Calibration Slope': calibration_slope,
                'Optimal Threshold': optimal_threshold
            }
            results.append(result)
            
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"F1 Score:    {f1:.4f}")
            print(f"AUC:         {auc:.4f}")
            print(f"Brier Score: {brier:.4f} (lower is better)")
            print(f"Youden Index: {youden:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"Calibration Slope: {calibration_slope:.4f} (ideal=1)")
        
        self.results['evaluation'] = pd.DataFrame(results)
        return self.results['evaluation']
    
    def plot_comprehensive_evaluation(self, save_dir):
        """绘制综合评估图"""
        if 'evaluation' not in self.results:
            print("Please run tune_and_evaluate() first!")
            return
        
        df = self.results['evaluation']
        
        # === Figure 1: Metrics Comparison Bar Chart ===
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['Accuracy', 'F1 Score', 'AUC', 'Brier Score', 'Youden Index', 'Calibration Slope']
        colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#7ad151', '#fde725']
        
        for ax, metric, color in zip(axes.flatten(), metrics, colors):
            df_sorted = df.sort_values(metric, ascending=(metric == 'Brier Score'))
            
            bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=color, edgecolor='white')
            
            # 数值标签
            for bar, val in zip(bars, df_sorted[metric]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(metric, fontsize=14, fontweight='bold')
            
            if metric == 'Brier Score':
                ax.invert_xaxis()  # Lower is better
        
        fig1.suptitle('Classification Model Evaluation Metrics\n(10-Fold CV with Hyperparameter Tuning)',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        path1 = os.path.join(save_dir, 'advanced_metrics_comparison.png')
        fig1.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path1}")
        
        # === Figure 2: ROC Curves ===
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for i, (name, model) in enumerate(self.best_models.items()):
            y_pred_proba = cross_val_predict(model, self.X_scaled, self.y, cv=cv, method='predict_proba')[:, 1]
            fpr, tpr, _ = roc_curve(self.y, y_pred_proba)
            auc = roc_auc_score(self.y, y_pred_proba)
            
            ax2.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
        ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
        ax2.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        path2 = os.path.join(save_dir, 'advanced_roc_curves.png')
        fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path2}")
        
        # === Figure 3: Calibration Plots ===
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        for name, model in self.best_models.items():
            y_pred_proba = cross_val_predict(model, self.X_scaled, self.y, cv=cv, method='predict_proba')[:, 1]
            
            try:
                prob_true, prob_pred = calibration_curve(self.y, y_pred_proba, n_bins=10)
                ax3.plot(prob_pred, prob_true, 'o-', label=name, linewidth=2, markersize=5)
            except:
                pass
        
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        ax3.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
        ax3.set_title('Calibration Curves', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        path3 = os.path.join(save_dir, 'advanced_calibration_curves.png')
        fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path3}")
        
        return fig1, fig2, fig3
    
    def run_shap_analysis(self, model_name='Random Forest', save_dir='.'):
        """SHAP可解释性分析"""
        if not HAS_SHAP:
            print("SHAP not installed! Install with: pip install shap")
            return None
        
        if model_name not in self.best_models:
            print(f"Model '{model_name}' not found. Available: {list(self.best_models.keys())}")
            return None
        
        print(f"\n" + "=" * 70)
        print(f"   SHAP INTERPRETABILITY ANALYSIS - {model_name}")
        print("=" * 70)
        
        model = self.best_models[model_name]
        model.fit(self.X_scaled, self.y)
        
        feature_names = ['X1 (EAN Concentration)', 'X2 (Protein Concentration)']
        
        try:
            # Create SHAP explainer based on model type
            if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_scaled)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, get class 1
                elif len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]  # New format (n_samples, n_features, n_classes)
            else:
                # Use background data for kernel explainer
                background = shap.kmeans(self.X_scaled, 10)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(self.X_scaled[:50])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            # Ensure shap_values is 2D
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, 1)
            
            print(f"SHAP values shape: {shap_values.shape}")
            
            # === Figure 1: SHAP Summary Plot (Bar) ===
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_scaled, feature_names=feature_names, 
                             plot_type='bar', show=False)
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path1 = os.path.join(save_dir, 'shap_feature_importance.png')
            plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path1}")
            plt.close()
            
            # === Figure 2: SHAP Beeswarm Plot ===
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_scaled, feature_names=feature_names, show=False)
            plt.title(f'SHAP Values Distribution - {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path2 = os.path.join(save_dir, 'shap_beeswarm.png')
            plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path2}")
            plt.close()
            
            # === Figure 3: SHAP Dependence Plots ===
            fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for i, (ax, feat_name) in enumerate(zip(axes, feature_names)):
                shap.dependence_plot(i, shap_values, self.X_scaled, 
                                    feature_names=feature_names, ax=ax, show=False)
                ax.set_title(f'SHAP Dependence: {feat_name}', fontsize=12, fontweight='bold')
            
            fig3.suptitle(f'SHAP Dependence Plots - {model_name}', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            path3 = os.path.join(save_dir, 'shap_dependence.png')
            fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path3}")
            plt.close()
            
            print("\nSHAP analysis complete!")
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            print(f"\nMean |SHAP| values:")
            for name, val in zip(feature_names, mean_shap):
                print(f"  {name}: {val:.4f}")
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analysis error: {str(e)}")
            print("Skipping SHAP analysis due to compatibility issues.")
            return None
    
    def save_comprehensive_report(self, save_path):
        """保存综合报告"""
        if 'evaluation' not in self.results:
            print("Please run tune_and_evaluate() first!")
            return
        
        df = self.results['evaluation']
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("   ADVANCED ML CLASSIFICATION ANALYSIS REPORT\n")
            f.write("   10-Fold Cross-Validation with Hyperparameter Tuning\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Samples: {len(self.y)}\n")
            f.write(f"Features: X1 (EAN Concentration), X2 (Protein Concentration)\n")
            f.write(f"Target: {self.target_col} (Crystalline Detection)\n")
            f.write(f"Class Distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}\n")
            f.write(f"Cross-Validation: {self.n_folds}-Fold Stratified\n\n")
            
            f.write("2. EVALUATION METRICS SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Brier':>8} {'Youden':>8} {'Cal.Slope':>10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['Model']:<20} {row['Accuracy']:>10.4f} {row['F1 Score']:>8.4f} "
                       f"{row['AUC']:>8.4f} {row['Brier Score']:>8.4f} {row['Youden Index']:>8.4f} "
                       f"{row['Calibration Slope']:>10.4f}\n")
            
            f.write("\n3. BEST MODEL PER METRIC\n")
            f.write("-" * 80 + "\n")
            
            # Best by each metric
            best_acc = df.loc[df['Accuracy'].idxmax()]
            best_f1 = df.loc[df['F1 Score'].idxmax()]
            best_auc = df.loc[df['AUC'].idxmax()]
            best_brier = df.loc[df['Brier Score'].idxmin()]  # Lower is better
            best_youden = df.loc[df['Youden Index'].idxmax()]
            
            f.write(f"Best Accuracy:    {best_acc['Model']} ({best_acc['Accuracy']:.4f})\n")
            f.write(f"Best F1 Score:    {best_f1['Model']} ({best_f1['F1 Score']:.4f})\n")
            f.write(f"Best AUC:         {best_auc['Model']} ({best_auc['AUC']:.4f})\n")
            f.write(f"Best Brier Score: {best_brier['Model']} ({best_brier['Brier Score']:.4f})\n")
            f.write(f"Best Youden Index: {best_youden['Model']} ({best_youden['Youden Index']:.4f})\n")
            
            f.write("\n4. DETAILED MODEL PARAMETERS\n")
            f.write("-" * 80 + "\n")
            for _, row in df.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  Params: {row['Best Params']}\n")
                f.write(f"  Sensitivity: {row['Sensitivity']:.4f}\n")
                f.write(f"  Specificity: {row['Specificity']:.4f}\n")
                f.write(f"  Optimal Threshold: {row['Optimal Threshold']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("   METRIC DEFINITIONS\n")
            f.write("=" * 80 + "\n")
            f.write("- Brier Score: Mean squared error of probability predictions (0=perfect, 0.25=random)\n")
            f.write("- Youden Index: Sensitivity + Specificity - 1 (range: -1 to 1, higher is better)\n")
            f.write("- Calibration Slope: Slope of calibration curve (ideal=1, <1=overconfident, >1=underconfident)\n")
            f.write("- AUC: Area Under ROC Curve (0.5=random, 1=perfect)\n")
            f.write("- F1 Score: Harmonic mean of precision and recall\n\n")
        
        print(f"Saved: {save_path}")


def main():
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    
    data_path = os.path.join(project_root, 'data', 'cleaned_data.csv')
    output_dir = os.path.join(project_root, 'outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 80)
    print("   ADVANCED ML ANALYSIS")
    print("   10-Fold CV | Hyperparameter Tuning | SHAP Interpretability")
    print("=" * 80)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    
    # Initialize analyzer
    analyzer = AdvancedClassificationAnalyzer(df, target_col='y2', n_folds=10)
    
    # Run hyperparameter tuning and evaluation
    results = analyzer.tune_and_evaluate()
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("   GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    analyzer.plot_comprehensive_evaluation(output_dir)
    
    # SHAP Analysis
    if HAS_SHAP:
        # Find best model for SHAP
        best_model_name = results.loc[results['AUC'].idxmax(), 'Model']
        analyzer.run_shap_analysis(model_name=best_model_name, save_dir=output_dir)
    else:
        print("\nSHAP not installed. Skipping interpretability analysis.")
        print("Install with: pip install shap")
    
    # Save report
    analyzer.save_comprehensive_report(os.path.join(output_dir, 'advanced_analysis_report.txt'))
    
    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - advanced_metrics_comparison.png")
    print("  - advanced_roc_curves.png")
    print("  - advanced_calibration_curves.png")
    print("  - advanced_analysis_report.txt")
    if HAS_SHAP:
        print("  - shap_feature_importance.png")
        print("  - shap_beeswarm.png")
        print("  - shap_dependence.png")
        print("  - shap_force_plot.png")
    
    plt.show()


if __name__ == "__main__":
    main()
