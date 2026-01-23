"""
Model Performance Analysis Visualization
生成模型性能结果的可视化分析图

包括：
1. 回归模型 R² 对比图
2. 分类模型准确率对比图
3. 综合性能仪表盘
4. 模型对比雷达图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_viridis_colors(n=5):
    """返回viridis风格的颜色列表"""
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    return colors[:n]


def plot_regression_comparison(save_path=None):
    """
    绘制回归模型 R² 分数对比条形图
    """
    # 数据来自 results_summary.txt
    models = ['Linear Regression', 'Random Forest', 'XGBoost']
    r2_mean = [0.061886, 0.778672, 0.795858]
    r2_std = [0.400306, 0.053140, 0.051559]
    training_mse = [5.889582, 0.171415, 0.000128]
    
    colors = ['#7f7f7f', '#21918c', '#5ec962']  # Gray, Teal, Green
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === 左图：R² 对比 ===
    ax1 = axes[0]
    bars = ax1.bar(models, r2_mean, color=colors, edgecolor='white', linewidth=2)
    ax1.errorbar(models, r2_mean, yerr=r2_std, fmt='none', color='black', 
                 capsize=8, capthick=2, elinewidth=2)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, r2_mean, r2_std):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_ylabel('Cross-Validation R² Score', fontsize=14, fontweight='bold')
    ax1.set_title('Regression Models - R² Comparison\n(Target: Y1 - Rg)', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim(-0.5, 1.1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Good R² (0.8)')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='both', labelsize=12)
    
    # 添加网格
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # === 右图：Training MSE 对比（对数尺度）===
    ax2 = axes[1]
    bars2 = ax2.bar(models, training_mse, color=colors, edgecolor='white', linewidth=2)
    ax2.set_yscale('log')
    
    # 添加数值标签
    for bar, mse in zip(bars2, training_mse):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Training MSE (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_title('Regression Models - Training MSE\n(Lower is Better)', 
                  fontsize=16, fontweight='bold', pad=15)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_classification_comparison(save_path=None):
    """
    绘制分类模型准确率对比条形图
    """
    models = ['Random Forest', 'SVC (RBF Kernel)']
    accuracy_mean = [0.888947, 0.767895]
    accuracy_std = [0.037194, 0.080569]
    
    colors = ['#21918c', '#440154']  # Teal, Purple
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(models))
    width = 0.6
    
    bars = ax.bar(x, accuracy_mean, width, color=colors, edgecolor='white', linewidth=2)
    ax.errorbar(x, accuracy_mean, yerr=accuracy_std, fmt='none', color='black',
                capsize=10, capthick=2, elinewidth=2)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, accuracy_mean, accuracy_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
               f'{mean:.1%}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Classification Models - Accuracy Comparison\n(Target: Y2 - Crystalline Detection)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylim(0, 1.1)
    
    # 参考线
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='80% Accuracy')
    ax.axhline(y=0.9, color='gold', linestyle='--', linewidth=1.5, alpha=0.5, label='90% Accuracy')
    ax.legend(loc='lower right', fontsize=11)
    
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_model_performance_dashboard(save_path=None):
    """
    综合性能仪表盘 - 所有模型一览
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # === 1. 回归 R² 水平条形图 ===
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    reg_models = ['Linear Regression', 'Random Forest Regressor', 'XGBoost Regressor']
    r2_scores = [0.062, 0.779, 0.796]
    colors_reg = ['#7f7f7f', '#21918c', '#5ec962']
    
    y_pos = np.arange(len(reg_models))
    bars = ax1.barh(y_pos, r2_scores, color=colors_reg, edgecolor='white', height=0.6)
    
    # 数值标签
    for bar, score in zip(bars, r2_scores):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', va='center', fontsize=13, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(reg_models, fontsize=12)
    ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Regression Task (Y1: Rg)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # === 2. 分类准确率水平条形图 ===
    ax2 = fig.add_subplot(gs[1, 0:2])
    
    clf_models = ['SVC (RBF)', 'Random Forest Classifier']
    accuracy = [0.768, 0.889]
    colors_clf = ['#440154', '#21918c']
    
    y_pos2 = np.arange(len(clf_models))
    bars2 = ax2.barh(y_pos2, accuracy, color=colors_clf, edgecolor='white', height=0.6)
    
    for bar, score in zip(bars2, accuracy):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{score:.1%}', va='center', fontsize=13, fontweight='bold')
    
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(clf_models, fontsize=12)
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Task (Y2: Crystalline)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.9, color='gold', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # === 3. 最佳模型指标卡片 ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # 最佳回归模型信息
    ax3.text(0.5, 0.95, '🏆 Best Regression Model', fontsize=14, fontweight='bold',
             ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.75, 'XGBoost Regressor', fontsize=16, fontweight='bold',
             ha='center', transform=ax3.transAxes, color='#5ec962')
    ax3.text(0.5, 0.55, 'R² = 0.796 ± 0.052', fontsize=14,
             ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.40, 'MSE = 0.0001', fontsize=12, color='gray',
             ha='center', transform=ax3.transAxes)
    
    # 添加边框
    rect = mpatches.FancyBboxPatch((0.05, 0.25), 0.9, 0.7, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor='#f0f8f0', edgecolor='#5ec962',
                                    linewidth=2, transform=ax3.transAxes)
    ax3.add_patch(rect)
    
    # === 4. 最佳分类模型指标卡片 ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    ax4.text(0.5, 0.95, '🏆 Best Classification Model', fontsize=14, fontweight='bold',
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.75, 'Random Forest', fontsize=16, fontweight='bold',
             ha='center', transform=ax4.transAxes, color='#21918c')
    ax4.text(0.5, 0.55, 'Accuracy = 88.9% ± 3.7%', fontsize=14,
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.40, 'Stratified 5-Fold CV', fontsize=12, color='gray',
             ha='center', transform=ax4.transAxes)
    
    rect2 = mpatches.FancyBboxPatch((0.05, 0.25), 0.9, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#f0f8f8', edgecolor='#21918c',
                                     linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(rect2)
    
    # 总标题
    fig.suptitle('ML Experiment Performance Dashboard\nDataset: 99 samples | Features: X1 (EAN), X2 (Protein)',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig


def plot_model_comparison_radar(save_path=None):
    """
    模型综合能力雷达图
    """
    # 标准化的性能指标 (0-1)
    categories = ['R² Score', 'Accuracy\n(if applicable)', 'Training Speed', 
                  'Interpretability', 'Stability\n(1-CV Std)']
    
    # 各模型得分
    linear_reg = [0.06, 0, 1.0, 1.0, 0.6]  # Linear Regression
    rf = [0.78, 0.89, 0.7, 0.5, 0.96]      # Random Forest
    xgb = [0.80, 0, 0.6, 0.4, 0.95]        # XGBoost
    svc = [0, 0.77, 0.8, 0.3, 0.92]        # SVC
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 数据闭合
    linear_reg += linear_reg[:1]
    rf += rf[:1]
    xgb += xgb[:1]
    svc += svc[:1]
    
    # 绘制
    ax.plot(angles, rf, 'o-', linewidth=2, color='#21918c', label='Random Forest')
    ax.fill(angles, rf, alpha=0.25, color='#21918c')
    
    ax.plot(angles, xgb, 's-', linewidth=2, color='#5ec962', label='XGBoost')
    ax.fill(angles, xgb, alpha=0.25, color='#5ec962')
    
    ax.plot(angles, linear_reg, '^-', linewidth=2, color='#7f7f7f', label='Linear Regression')
    ax.fill(angles, linear_reg, alpha=0.15, color='#7f7f7f')
    
    ax.plot(angles, svc, 'd-', linewidth=2, color='#440154', label='SVC')
    ax.fill(angles, svc, alpha=0.15, color='#440154')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.set_title('Model Capability Comparison\n(Normalized Scores)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig, ax


def generate_all_analysis_figures(output_dir):
    """
    生成所有分析可视化图
    """
    print("=" * 60)
    print("   GENERATING MODEL ANALYSIS VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 回归模型对比
    print("\n[1/4] Generating Regression Comparison...")
    plot_regression_comparison(
        save_path=os.path.join(output_dir, 'analysis_regression_comparison.png')
    )
    
    # 2. 分类模型对比  
    print("[2/4] Generating Classification Comparison...")
    plot_classification_comparison(
        save_path=os.path.join(output_dir, 'analysis_classification_comparison.png')
    )
    
    # 3. 综合仪表盘
    print("[3/4] Generating Performance Dashboard...")
    plot_model_performance_dashboard(
        save_path=os.path.join(output_dir, 'analysis_performance_dashboard.png')
    )
    
    # 4. 雷达图
    print("[4/4] Generating Radar Chart...")
    plot_model_comparison_radar(
        save_path=os.path.join(output_dir, 'analysis_model_radar.png')
    )
    
    print("\n" + "=" * 60)
    print("   ALL VISUALIZATIONS GENERATED!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    print("  1. analysis_regression_comparison.png")
    print("  2. analysis_classification_comparison.png")
    print("  3. analysis_performance_dashboard.png")
    print("  4. analysis_model_radar.png")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.join(project_root, 'outputs')
    generate_all_analysis_figures(output_dir)
    plt.show()
