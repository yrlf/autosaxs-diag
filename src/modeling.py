"""
Extended ML Modeling Module
包含多种常见机器学习模型进行全面对比

回归模型 (Regression):
- Linear Regression
- Ridge Regression  
- Lasso Regression
- ElasticNet
- SVR (Support Vector Regression)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- KNN Regressor

分类模型 (Classification):
- Logistic Regression
- SVC
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- KNN Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost models.")


class ExtendedModelTrainer:
    """扩展的模型训练器，包含多种常见ML模型"""
    
    def __init__(self, df, random_state=42):
        self.df = df
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
    
    def get_regression_models(self):
        """返回所有回归模型字典"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=10000),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0),
            'SVR (Linear)': SVR(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
            'KNN (k=10)': KNeighborsRegressor(n_neighbors=10),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=self.random_state, verbosity=0)
        
        return models
    
    def get_classification_models(self):
        """返回所有分类模型字典"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'SVC (RBF)': SVC(kernel='rbf', random_state=self.random_state),
            'SVC (Linear)': SVC(kernel='linear', random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
            'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        }
        
        if HAS_XGBOOST:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=self.random_state, verbosity=0, use_label_encoder=False, eval_metric='logloss')
        
        return models
    
    def train_regression(self, target_col='y1', n_folds=5, scale_features=True):
        """
        训练所有回归模型
        
        Parameters:
        -----------
        target_col : str
            目标变量列名
        n_folds : int
            交叉验证折数
        scale_features : bool
            是否标准化特征
        """
        print(f"\n{'='*60}")
        print(f"Training Regression Models for {target_col}")
        print(f"{'='*60}")
        
        # 准备数据
        data = self.df.dropna(subset=[target_col]).copy()
        X = data[['x1', 'x2']].values
        y = data[target_col].values
        
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        print(f"Samples: {len(data)} | Features: 2 | Folds: {n_folds}")
        print("-" * 60)
        
        models = self.get_regression_models()
        results = []
        
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                # Train on full data for MSE
                model.fit(X, y)
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                
                result = {
                    'Task': 'Regression',
                    'Target': target_col,
                    'Model': name,
                    'CV R2 Mean': np.mean(cv_scores),
                    'CV R2 Std': np.std(cv_scores),
                    'Training MSE': mse,
                    'Training R2': r2_score(y, y_pred)
                }
                results.append(result)
                
                print(f"{name:25s} | R²: {np.mean(cv_scores):7.4f} ± {np.std(cv_scores):.4f} | MSE: {mse:.4f}")
                
            except Exception as e:
                print(f"{name:25s} | Error: {str(e)[:40]}")
        
        print("-" * 60)
        
        self.results['regression'] = pd.DataFrame(results)
        return self.results['regression']
    
    def train_classification(self, target_col='y2', n_folds=5, scale_features=True):
        """
        训练所有分类模型
        
        Parameters:
        -----------
        target_col : str
            目标变量列名
        n_folds : int
            交叉验证折数
        scale_features : bool
            是否标准化特征
        """
        print(f"\n{'='*60}")
        print(f"Training Classification Models for {target_col}")
        print(f"{'='*60}")
        
        # 准备数据
        data = self.df.dropna(subset=[target_col]).copy()
        X = data[['x1', 'x2']].values
        y = data[target_col].astype(int).values
        
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        print(f"Samples: {len(data)} | Features: 2 | Folds: {n_folds}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print("-" * 60)
        
        models = self.get_classification_models()
        results = []
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                
                # Train on full data
                model.fit(X, y)
                y_pred = model.predict(X)
                train_acc = accuracy_score(y, y_pred)
                
                result = {
                    'Task': 'Classification',
                    'Target': target_col,
                    'Model': name,
                    'CV Accuracy Mean': np.mean(cv_scores),
                    'CV Accuracy Std': np.std(cv_scores),
                    'Training Accuracy': train_acc
                }
                results.append(result)
                
                print(f"{name:25s} | Acc: {np.mean(cv_scores):7.4f} ± {np.std(cv_scores):.4f} | Train: {train_acc:.4f}")
                
            except Exception as e:
                print(f"{name:25s} | Error: {str(e)[:40]}")
        
        print("-" * 60)
        
        self.results['classification'] = pd.DataFrame(results)
        return self.results['classification']
    
    def get_best_models(self):
        """返回最佳模型信息"""
        best = {}
        
        if 'regression' in self.results:
            reg_df = self.results['regression']
            best_reg_idx = reg_df['CV R2 Mean'].idxmax()
            best['regression'] = {
                'model': reg_df.loc[best_reg_idx, 'Model'],
                'r2': reg_df.loc[best_reg_idx, 'CV R2 Mean'],
                'r2_std': reg_df.loc[best_reg_idx, 'CV R2 Std']
            }
        
        if 'classification' in self.results:
            clf_df = self.results['classification']
            best_clf_idx = clf_df['CV Accuracy Mean'].idxmax()
            best['classification'] = {
                'model': clf_df.loc[best_clf_idx, 'Model'],
                'accuracy': clf_df.loc[best_clf_idx, 'CV Accuracy Mean'],
                'accuracy_std': clf_df.loc[best_clf_idx, 'CV Accuracy Std']
            }
        
        return best


# 保留原有的 ModelTrainer 类以保持向后兼容
class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.results = {}

    def train_regression(self, target_col='y1'):
        """Trains regression models for the given target."""
        print(f"\n--- Training Regression Models for {target_col} ---")
        
        data = self.df.dropna(subset=[target_col]).copy()
        X = data[['x1', 'x2']]
        y = data[target_col]
        
        print(f"Samples for regression: {len(data)}")

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        }
        if HAS_XGBOOST:
            models['XGBoost Regressor'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

        results = []
        for name, model in models.items():
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            
            res = {
                'Task': 'Regression',
                'Target': target_col,
                'Model': name,
                'CV R2 Mean': np.mean(cv_scores),
                'CV R2 Std': np.std(cv_scores),
                'Training MSE': mse
            }
            results.append(res)
            print(f"{name}: CV R2 = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
        return pd.DataFrame(results)

    def train_classification(self, target_col='y2'):
        """Trains classification models for the given target."""
        print(f"\n--- Training Classification Models for {target_col} ---")
        
        data = self.df.dropna(subset=[target_col]).copy()
        X = data[['x1', 'x2']]
        y = data[target_col].astype(int)
        
        print(f"Samples for classification: {len(data)}")
        print(f"Class balance: {y.value_counts().to_dict()}")

        models = {
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVC': SVC(kernel='rbf', random_state=42)
        }

        results = []
        for name, model in models.items():
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            model.fit(X, y)
            
            res = {
                'Task': 'Classification',
                'Target': target_col,
                'Model': name,
                'CV Accuracy Mean': np.mean(cv_scores),
                'CV Accuracy Std': np.std(cv_scores)
            }
            results.append(res)
            print(f"{name}: CV Accuracy = {np.mean(cv_scores):.4f}")
            
        return pd.DataFrame(results)
