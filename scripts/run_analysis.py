import os
import sys

# Define paths relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from src.data_cleaning import DataCleaner

def main():
    # Define paths
    input_file = os.path.join(project_root, 'data', 'ML_targets_crystal_oligo v2.csv')
    output_file = os.path.join(project_root, 'data', 'cleaned_data.csv')

    print("--- Starting Data Cleaning Process ---")
    
    # Initialize and run cleaner
    cleaner = DataCleaner(input_file)
    cleaner.load_data()
    df = cleaner.clean_data()
    
    # Preview cleaned data
    print("\n--- Cleaned Data Preview ---")
    print(df.head())
    print("\n--- Statistics ---")
    print(df.describe())
    
    # Save cleaned data
    cleaner.save_clean_data(output_file)

    # --- Modeling Phase ---
    print("\n--- Starting Modeling Phase ---")
    from src.modeling import ModelTrainer
    
    trainer = ModelTrainer(df)

    # 1. Regression for y1-Rg
    reg_results = trainer.train_regression(target_col='y1')

    # 2. Classification for y2-crystalline
    clf_results = trainer.train_classification(target_col='y2')

    # --- Summary ---
    print("\n=== FINAL RESULTS SUMMARY ===")
    print("\n[Regression (y1-Rg)]")
    print(reg_results.to_string(index=False))
    print("\n[Classification (y2-crystalline)]")
    print(clf_results.to_string(index=False))

    # Save results to file
    results_path = os.path.join(project_root, 'outputs', 'results_summary.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("==========================================\n")
        f.write("       ML EXPERIMENT RESULTS SUMMARY      \n")
        f.write("==========================================\n\n")
        
        f.write("1. EXPERIMENT SETUP (实验设置)\n")
        f.write("-----------------------------\n")
        f.write(f"Dataset Shape: {df.shape} (Rows, Columns)\n")
        f.write("Independent Variables (X - 自变量):\n")
        f.write("  - x1: EAN concentration (wt%)\n")
        f.write("  - x2: Protein concentration (mg/ml)\n")
        f.write("\n")
        f.write("Dependent Variables (Y - 因变量):\n")
        f.write("  - y1: Rg (Radius of Gyration) [Regression]\n")
        f.write("  - y2: Crystalline Present (0/1) [Classification]\n")
        f.write("\n")
        f.write("Validation Strategy:\n")
        f.write("  - Regression: K-Fold Cross Validation (k=5, shuffle=True)\n")
        f.write("  - Classification: Stratified K-Fold CV (k=5, shuffle=True)\n\n")

        f.write("2. MODEL CONFIGURATIONS (模型参数)\n")
        f.write("--------------------------------\n")
        f.write("Random Forest (Regressor/Classifier):\n")
        f.write("  - n_estimators: 100\n")
        f.write("  - random_state: 42\n")
        f.write("XGBoost Regressor:\n")
        f.write("  - n_estimators: 100\n")
        f.write("  - random_state: 42\n")
        f.write("Linear Regression / SVC:\n")
        f.write("  - Default settings (SVC kernel='rbf')\n\n")
        
        f.write("3. PERFORMANCE RESULTS (性能结果)\n")
        f.write("--------------------------------\n")
        f.write("[Regression Task - Target: y1-Rg]\n")
        f.write(reg_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("[Classification Task - Target: y2-crystalline]\n")
        f.write(clf_results.to_string(index=False))
        f.write("\n\n")
        f.write("==========================================\n")
    
    print(f"\nDetailed results saved to {results_path}")
    print("\nProcess finished successfully.")

if __name__ == "__main__":
    main()
