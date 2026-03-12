# Hyperparameter Tuning Strategy for Machine Learning Models

## English Version (For Manuscript)

To ensure optimal predictive performance and prevent overfitting, a systematic hyperparameter tuning procedure was conducted for the two primary machine learning models: Random Forest (RF) and Gradient Boosting (GB). The optimization was performed using grid search with 5-fold cross-validation (`GridSearchCV` in scikit-learn), evaluated based on the coefficient of determination (R²). The hyperparameter search space and optimal values for each model were defined as follows:

**1. Random Forest (RF) Regressor:**
*   **Search Space:**
    *   Number of trees (`n_estimators`): {100, 200, 300}
    *   Maximum depth of the trees (`max_depth`): {5, 10, 15, None}
    *   Minimum number of samples required to split an internal node (`min_samples_split`): {2, 5, 10}
    *   Minimum number of samples required to be at a leaf node (`min_samples_leaf`): {1, 2, 4}
*   **Optimal Parameters Selected:** `n_estimators = 200`, `max_depth = 10`.

**2. Gradient Boosting (GB) Regressor:**
*   **Search Space:**
    *   Number of boosting stages (`n_estimators`): {100, 200, 300}
    *   Maximum depth of the individual regression estimators (`max_depth`): {3, 5, 7}
    *   Learning rate (`learning_rate`): {0.01, 0.05, 0.1, 0.2}
    *   Minimum number of samples required to split an internal node (`min_samples_split`): {2, 5, 10}
*   **Optimal Parameters Selected:** `n_estimators = 200`, `max_depth = 5`.

The determined optimal hyperparameters were subsequently utilized in the final rigorous analysis across multiple random seeds to ensure robustness and reproducibility of the results.

---

## Chinese Version (中文参考)

为了确保最佳的预测性能并防止过拟合，我们对两种主要的机器学习模型：随机森林（Random Forest, RF）和梯度提升（Gradient Boosting, GB）进行了系统的超参数调优。优化过程使用了带有5折交叉验证的网格搜索（scikit-learn中的`GridSearchCV`），并以决定系数（R²）作为评估指标。各模型的超参数搜索空间及最优值定义如下：

**1. 随机森林 (RF) 模型:**
*   **搜索空间:**
    *   决策树数量 (`n_estimators`): {100, 200, 300}
    *   树的最大深度 (`max_depth`): {5, 10, 15, 无限制}
    *   内部节点再划分所需最小样本数 (`min_samples_split`): {2, 5, 10}
    *   叶子节点所需最小样本数 (`min_samples_leaf`): {1, 2, 4}
*   **选定的最优参数:** `n_estimators = 200`, `max_depth = 10`.

**2. 梯度提升 (GB) 模型:**
*   **搜索空间:**
    *   弱学习器数量 (`n_estimators`): {100, 200, 300}
    *   单棵回归树的最大深度 (`max_depth`): {3, 5, 7}
    *   学习率 (`learning_rate`): {0.01, 0.05, 0.1, 0.2}
    *   内部节点再划分所需最小样本数 (`min_samples_split`): {2, 5, 10}
*   **选定的最优参数:** `n_estimators = 200`, `max_depth = 5`.

这些确定的最优超参数随后被用于最终的严格分析中（多次设定随机种子进行训练与测试），以确保结果的稳健性和可重复性。
