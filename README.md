# Protein Crystallization ML Analysis

A machine learning project to predict protein crystallization behavior using EAN (Ethylammonium Nitrate) ionic liquid concentration and protein concentration as features.

## 📊 Project Overview

This project uses various machine learning models to:
1. **Regression Task (Y1)**: Predict Rg (Radius of Gyration) values
2. **Classification Task (Y2)**: Predict crystalline formation (0/1)

## 🔬 Dataset

- **Samples**: 99
- **Features**:
  - X1: EAN Concentration (wt%)
  - X2: Protein Concentration (mg/mL)
- **Targets**:
  - Y1: Rg (Radius of Gyration) - Continuous
  - Y2: Crystalline Present (0/1) - Binary

## 🤖 Models Evaluated

### Regression Models (12)
- Linear Regression, Ridge, Lasso, ElasticNet
- SVR (RBF, Linear)
- Decision Tree, Random Forest, Gradient Boosting
- XGBoost, KNN

### Classification Models (9)
- Logistic Regression
- SVC (RBF, Linear)
- Decision Tree, Random Forest, Gradient Boosting
- XGBoost, KNN

## 📈 Best Results

### Regression (Target: Y1 - Rg)
| Model | CV R² | CV Std |
|-------|-------|--------|
| **XGBoost** | **0.796** | ±0.052 |
| Random Forest | 0.778 | ±0.054 |
| Gradient Boosting | 0.771 | ±0.102 |

### Classification (Target: Y2 - Crystalline)
| Model | CV Accuracy | AUC | F1 Score | Youden Index |
|-------|-------------|-----|----------|--------------|
| **Gradient Boosting** | **91.9%** | 0.942 | 0.919 | 0.809 |
| Random Forest | 90.9% | **0.957** | 0.909 | 0.775 |

## 📁 Project Structure

```
HQ/
├── src/
│   ├── data_cleaning.py      # Data preprocessing
│   ├── modeling.py           # ML models (extended)
│   └── visualization.py      # Heatmap visualization
├── run_analysis.py           # Basic analysis script
├── run_extended_analysis.py  # Extended model comparison
├── run_advanced_analysis.py  # Advanced analysis with SHAP
├── generate_heatmap.py       # Heatmap generation
├── generate_analysis_charts.py
├── cleaned_data.csv          # Preprocessed data
├── results_summary.txt       # Basic results
├── extended_results_summary.txt
├── advanced_analysis_report.txt
└── *.png                     # Visualization outputs
```

## 🚀 Quick Start

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost matplotlib shap
```

### Run Analysis
```bash
# Basic analysis
python run_analysis.py

# Extended model comparison (12 regression, 9 classification models)
python run_extended_analysis.py

# Advanced analysis with 10-fold CV, hyperparameter tuning, and SHAP
python run_advanced_analysis.py

# Generate heatmap visualization
python generate_heatmap.py
```

## 📊 Visualizations

### Rg Prediction Heatmap
![Heatmap](heatmap_y1_rg.png)

### Model Performance Comparison
![Dashboard](extended_dashboard.png)

### ROC Curves
![ROC](advanced_roc_curves.png)

### SHAP Feature Importance
![SHAP](shap_feature_importance.png)

## 🔍 Key Findings

1. **XGBoost** performs best for Rg prediction (R² = 0.796)
2. **Gradient Boosting** achieves highest accuracy for crystalline classification (91.9%)
3. **X1 (EAN Concentration)** is the dominant feature (SHAP importance 3.3x higher than X2)
4. Higher EAN concentration increases crystallization probability

## 📚 Evaluation Metrics

- **Brier Score**: Probability prediction accuracy (lower is better)
- **Youden Index**: Sensitivity + Specificity - 1
- **Calibration Slope**: Prediction confidence calibration
- **AUC**: Area Under ROC Curve
- **F1 Score**: Harmonic mean of precision and recall

## 📝 License

MIT License

## 👤 Author

Generated with ML analysis pipeline
