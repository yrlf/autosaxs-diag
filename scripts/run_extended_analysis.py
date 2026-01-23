"""
Extended ML Analysis with Multiple Models
运行扩展的机器学习分析，包含多种模型对比和精美可视化

Models included:
- Regression: 12 models
- Classification: 9 models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'


def get_color_palette(n):
    """获取专业配色方案"""
    # 紫色到黄色的渐变
    if n <= 6:
        return ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'][:n]
    else:
        # 更多颜色
        colors = [
            '#440154', '#481567', '#453781', '#3f4788', '#39568c',
            '#2d708e', '#238a8d', '#1f998a', '#29af7f', '#55c667',
            '#7ad151', '#bddf26', '#fde725'
        ]
        return colors[:n]


def plot_regression_comparison_extended(reg_df, save_path=None):
    """
    扩展的回归模型对比图 - 横向条形图
    """
    # 按 R² 排序
    reg_df_sorted = reg_df.sort_values('CV R2 Mean', ascending=True)
    
    n_models = len(reg_df_sorted)
    colors = get_color_palette(n_models)
    
    fig, ax = plt.subplots(figsize=(14, max(8, n_models * 0.6)))
    
    y_pos = np.arange(n_models)
    r2_values = reg_df_sorted['CV R2 Mean'].values
    r2_std = reg_df_sorted['CV R2 Std'].values
    models = reg_df_sorted['Model'].values
    
    # 横向条形图
    bars = ax.barh(y_pos, r2_values, xerr=r2_std, 
                   color=colors, edgecolor='white', linewidth=1.5,
                   height=0.7, capsize=4, error_kw={'linewidth': 1.5})
    
    # 数值标签
    for i, (bar, r2, std) in enumerate(zip(bars, r2_values, r2_std)):
        width = bar.get_width()
        label_x = max(width + std + 0.02, 0.05)
        ax.text(label_x, bar.get_y() + bar.get_height()/2.,
               f'{r2:.3f} ± {std:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 参考线
    ax.axvline(x=0.8, color='#22a884', linestyle='--', linewidth=2, alpha=0.7, label='Good (R²=0.8)')
    ax.axvline(x=0.9, color='#fde725', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (R²=0.9)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Cross-Validation R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Regression Models Comparison\n(Target: Y1 - Rg, 5-Fold Cross-Validation)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(-0.1, 1.1)
    ax.legend(loc='lower right', fontsize=10)
    
    # 高亮最佳模型
    best_idx = n_models - 1  # 最后一个是最好的（已排序）
    bars[best_idx].set_edgecolor('#fde725')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_classification_comparison_extended(clf_df, save_path=None):
    """
    扩展的分类模型对比图 - 横向条形图
    """
    clf_df_sorted = clf_df.sort_values('CV Accuracy Mean', ascending=True)
    
    n_models = len(clf_df_sorted)
    colors = get_color_palette(n_models)
    
    fig, ax = plt.subplots(figsize=(14, max(8, n_models * 0.6)))
    
    y_pos = np.arange(n_models)
    acc_values = clf_df_sorted['CV Accuracy Mean'].values
    acc_std = clf_df_sorted['CV Accuracy Std'].values
    models = clf_df_sorted['Model'].values
    
    bars = ax.barh(y_pos, acc_values, xerr=acc_std,
                   color=colors, edgecolor='white', linewidth=1.5,
                   height=0.7, capsize=4, error_kw={'linewidth': 1.5})
    
    # 数值标签 (百分比)
    for bar, acc, std in zip(bars, acc_values, acc_std):
        width = bar.get_width()
        label_x = max(width + std + 0.02, 0.05)
        ax.text(label_x, bar.get_y() + bar.get_height()/2.,
               f'{acc:.1%} ± {std:.1%}', va='center', fontsize=10, fontweight='bold')
    
    # 参考线
    ax.axvline(x=0.8, color='#22a884', linestyle='--', linewidth=2, alpha=0.7, label='80% Accuracy')
    ax.axvline(x=0.9, color='#fde725', linestyle='--', linewidth=2, alpha=0.7, label='90% Accuracy')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Classification Models Comparison\n(Target: Y2 - Crystalline, Stratified 5-Fold CV)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.1)
    ax.legend(loc='lower right', fontsize=10)
    
    # 高亮最佳模型
    best_idx = n_models - 1
    bars[best_idx].set_edgecolor('#fde725')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_combined_dashboard(reg_df, clf_df, save_path=None):
    """
    综合性能仪表盘 - 顶级模型对比
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 获取 top 5
    reg_top5 = reg_df.nlargest(5, 'CV R2 Mean').sort_values('CV R2 Mean', ascending=True)
    clf_top5 = clf_df.nlargest(5, 'CV Accuracy Mean').sort_values('CV Accuracy Mean', ascending=True)
    
    colors_reg = ['#2d708e', '#238a8d', '#29af7f', '#7ad151', '#fde725']
    colors_clf = ['#440154', '#453781', '#2a788e', '#22a884', '#7ad151']
    
    # === 左图：Top 5 回归模型 ===
    ax1 = fig.add_subplot(1, 2, 1)
    
    y_pos = np.arange(5)
    bars1 = ax1.barh(y_pos, reg_top5['CV R2 Mean'].values, 
                     xerr=reg_top5['CV R2 Std'].values,
                     color=colors_reg, edgecolor='white', linewidth=2,
                     height=0.65, capsize=5)
    
    for bar, r2, std in zip(bars1, reg_top5['CV R2 Mean'].values, reg_top5['CV R2 Std'].values):
        ax1.text(bar.get_width() + std + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{r2:.3f}', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(reg_top5['Model'].values, fontsize=12)
    ax1.set_xlabel('R² Score', fontsize=13, fontweight='bold')
    ax1.set_title('Top 5 Regression Models\n(Target: Y1 - Rg)', fontsize=15, fontweight='bold')
    ax1.set_xlim(0, 1.05)
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    
    # 最佳模型标注
    best_reg = reg_top5.iloc[-1]
    ax1.annotate(f"Best: {best_reg['Model']}", 
                xy=(best_reg['CV R2 Mean'], 4), xytext=(0.5, 4.5),
                fontsize=11, fontweight='bold', color='#fde725',
                arrowprops=dict(arrowstyle='->', color='#fde725'))
    
    # === 右图：Top 5 分类模型 ===
    ax2 = fig.add_subplot(1, 2, 2)
    
    bars2 = ax2.barh(y_pos, clf_top5['CV Accuracy Mean'].values,
                     xerr=clf_top5['CV Accuracy Std'].values,
                     color=colors_clf, edgecolor='white', linewidth=2,
                     height=0.65, capsize=5)
    
    for bar, acc, std in zip(bars2, clf_top5['CV Accuracy Mean'].values, clf_top5['CV Accuracy Std'].values):
        ax2.text(bar.get_width() + std + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1%}', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clf_top5['Model'].values, fontsize=12)
    ax2.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Top 5 Classification Models\n(Target: Y2 - Crystalline)', fontsize=15, fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0.9, color='gold', linestyle='--', alpha=0.5)
    
    # 最佳模型标注
    best_clf = clf_top5.iloc[-1]
    ax2.annotate(f"Best: {best_clf['Model']}", 
                xy=(best_clf['CV Accuracy Mean'], 4), xytext=(0.5, 4.5),
                fontsize=11, fontweight='bold', color='#7ad151',
                arrowprops=dict(arrowstyle='->', color='#7ad151'))
    
    # 总标题
    fig.suptitle('ML Model Performance Dashboard\nDataset: 99 samples | Features: X1 (EAN), X2 (Protein) | 5-Fold CV',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_model_ranking_heatmap(reg_df, clf_df, save_path=None):
    """
    模型排名热力图
    """
    # 合并所有模型
    all_models = set(reg_df['Model'].tolist() + clf_df['Model'].tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(reg_df) * 0.5)))
    
    # === 回归热力图 ===
    ax1 = axes[0]
    reg_sorted = reg_df.sort_values('CV R2 Mean', ascending=False)
    
    # 创建热力图数据
    r2_values = reg_sorted['CV R2 Mean'].values.reshape(-1, 1)
    
    im1 = ax1.imshow(r2_values, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    
    ax1.set_yticks(range(len(reg_sorted)))
    ax1.set_yticklabels(reg_sorted['Model'].values, fontsize=10)
    ax1.set_xticks([])
    ax1.set_title('Regression R² Ranking', fontsize=14, fontweight='bold')
    
    # 添加数值
    for i, r2 in enumerate(reg_sorted['CV R2 Mean'].values):
        color = 'white' if r2 < 0.5 else 'black'
        ax1.text(0, i, f'{r2:.3f}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color=color)
    
    # colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('R² Score', fontsize=12)
    
    # === 分类热力图 ===
    ax2 = axes[1]
    clf_sorted = clf_df.sort_values('CV Accuracy Mean', ascending=False)
    
    acc_values = clf_sorted['CV Accuracy Mean'].values.reshape(-1, 1)
    
    im2 = ax2.imshow(acc_values, cmap='viridis', aspect='auto', vmin=0.5, vmax=1)
    
    ax2.set_yticks(range(len(clf_sorted)))
    ax2.set_yticklabels(clf_sorted['Model'].values, fontsize=10)
    ax2.set_xticks([])
    ax2.set_title('Classification Accuracy Ranking', fontsize=14, fontweight='bold')
    
    for i, acc in enumerate(clf_sorted['CV Accuracy Mean'].values):
        color = 'white' if acc < 0.75 else 'black'
        ax2.text(0, i, f'{acc:.1%}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
    
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Accuracy', fontsize=12)
    
    fig.suptitle('Model Performance Ranking Heatmap', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def save_extended_results(reg_df, clf_df, best_models, output_path):
    """保存扩展结果到文本文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("     EXTENDED ML EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. EXPERIMENT SETUP\n")
        f.write("-" * 70 + "\n")
        f.write("Dataset: 99 samples, 2 features (X1: EAN, X2: Protein)\n")
        f.write("Validation: 5-Fold Cross-Validation\n")
        f.write(f"Regression Models: {len(reg_df)}\n")
        f.write(f"Classification Models: {len(clf_df)}\n\n")
        
        f.write("2. REGRESSION RESULTS (Target: Y1 - Rg)\n")
        f.write("-" * 70 + "\n")
        reg_sorted = reg_df.sort_values('CV R2 Mean', ascending=False)
        f.write(f"{'Rank':<5} {'Model':<25} {'R² Mean':>10} {'R² Std':>10} {'MSE':>12}\n")
        f.write("-" * 70 + "\n")
        for i, (_, row) in enumerate(reg_sorted.iterrows(), 1):
            f.write(f"{i:<5} {row['Model']:<25} {row['CV R2 Mean']:>10.4f} {row['CV R2 Std']:>10.4f} {row['Training MSE']:>12.6f}\n")
        f.write("\n")
        
        f.write("3. CLASSIFICATION RESULTS (Target: Y2 - Crystalline)\n")
        f.write("-" * 70 + "\n")
        clf_sorted = clf_df.sort_values('CV Accuracy Mean', ascending=False)
        f.write(f"{'Rank':<5} {'Model':<25} {'Acc Mean':>10} {'Acc Std':>10}\n")
        f.write("-" * 70 + "\n")
        for i, (_, row) in enumerate(clf_sorted.iterrows(), 1):
            f.write(f"{i:<5} {row['Model']:<25} {row['CV Accuracy Mean']:>10.4f} {row['CV Accuracy Std']:>10.4f}\n")
        f.write("\n")
        
        f.write("4. BEST MODELS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best Regression: {best_models['regression']['model']}\n")
        f.write(f"  R² = {best_models['regression']['r2']:.4f} ± {best_models['regression']['r2_std']:.4f}\n\n")
        f.write(f"Best Classification: {best_models['classification']['model']}\n")
        f.write(f"  Accuracy = {best_models['classification']['accuracy']:.4f} ± {best_models['classification']['accuracy_std']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    # Setup
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)
    
    data_path = os.path.join(project_root, 'data', 'cleaned_data.csv')
    output_dir = os.path.join(project_root, 'outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("   EXTENDED ML ANALYSIS WITH MULTIPLE MODELS")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        print("Please run run_analysis.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"\nLoaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Initialize trainer
    from src.modeling import ExtendedModelTrainer
    trainer = ExtendedModelTrainer(df)
    
    # Train all models
    reg_results = trainer.train_regression(target_col='y1', n_folds=5)
    clf_results = trainer.train_classification(target_col='y2', n_folds=5)
    
    # Get best models
    best_models = trainer.get_best_models()
    
    print("\n" + "=" * 70)
    print("   BEST MODELS")
    print("=" * 70)
    print(f"\nRegression:     {best_models['regression']['model']}")
    print(f"                R² = {best_models['regression']['r2']:.4f} ± {best_models['regression']['r2_std']:.4f}")
    print(f"\nClassification: {best_models['classification']['model']}")
    print(f"                Accuracy = {best_models['classification']['accuracy']:.4f} ± {best_models['classification']['accuracy_std']:.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("   GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print("\n[1/4] Regression comparison chart...")
    plot_regression_comparison_extended(
        reg_results,
        save_path=os.path.join(output_dir, 'extended_regression_comparison.png')
    )
    
    print("[2/4] Classification comparison chart...")
    plot_classification_comparison_extended(
        clf_results,
        save_path=os.path.join(output_dir, 'extended_classification_comparison.png')
    )
    
    print("[3/4] Combined dashboard...")
    plot_combined_dashboard(
        reg_results, clf_results,
        save_path=os.path.join(output_dir, 'extended_dashboard.png')
    )
    
    print("[4/4] Ranking heatmap...")
    plot_model_ranking_heatmap(
        reg_results, clf_results,
        save_path=os.path.join(output_dir, 'extended_ranking_heatmap.png')
    )
    
    # Save results
    save_extended_results(
        reg_results, clf_results, best_models,
        os.path.join(output_dir, 'extended_results_summary.txt')
    )
    
    print("\n" + "=" * 70)
    print("   COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - extended_regression_comparison.png")
    print("  - extended_classification_comparison.png")
    print("  - extended_dashboard.png")
    print("  - extended_ranking_heatmap.png")
    print("  - extended_results_summary.txt")
    
    plt.show()


if __name__ == "__main__":
    main()
