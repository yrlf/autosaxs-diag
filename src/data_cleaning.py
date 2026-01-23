"""
Data Cleaning Module for ML Analysis
=====================================
Features:
- Load raw CSV data
- Rename columns (x1, x2, y1, y2, CI, R2_w)
- Handle missing values
- When y2=FALSE (no crystal), CI is set to 0
- Optional: Filter out rows with R2 < threshold
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """
    Data cleaning module for SAXS ML analysis.
    
    Main functions:
    - Load raw CSV data
    - Rename columns to standardized names
    - Handle missing values
    - Process CI (Crystallinity Index): set to 0 when y2=FALSE
    - Optional: Filter by R2 quality threshold
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self, header_row=0):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        header_row : int
            Row index for header (default=0 for v3.csv format)
        """
        try:
            self.df = pd.read_csv(self.filepath, header=header_row)
            print(f"Successfully loaded data. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

    def clean_data(self, r2_threshold=None):
        """
        Perform data cleaning steps.
        
        Parameters:
        -----------
        r2_threshold : float, optional
            If provided, rows with R2_w < r2_threshold will be removed.
            Example: r2_threshold=0.80 removes rows with R2 < 0.80
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_shape = self.df.shape
        
        # 1. Clean column names - Remove whitespace
        self.df.columns = self.df.columns.str.strip()
        
        # 2. Identify and rename key columns
        column_map = {}
        for col in self.df.columns:
            if col.startswith('x1') and 'EAN' in col:
                column_map[col] = 'x1'
            elif col.startswith('x2') and 'Protein' in col:
                column_map[col] = 'x2'
            elif col.startswith('y1') and 'Rg' in col:
                column_map[col] = 'y1'
            elif col.startswith('y2') and 'crystalline' in col:
                column_map[col] = 'y2'
            elif col == 'CI':
                column_map[col] = 'CI'
            elif 'R2_w' in col or 'weighted R' in col:
                column_map[col] = 'R2_w'
                
        # Rename columns
        self.df.rename(columns=column_map, inplace=True)
        renamed_cols = [c for c in self.df.columns if c in ['x1', 'x2', 'y1', 'y2', 'CI', 'R2_w']]
        print(f"Renamed columns: {renamed_cols}")

        # 3. Drop Unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        # 4. Handle Missing Values for core features
        if 'y1' in self.df.columns:
            self.df.dropna(subset=['y1'], inplace=True)
        
        if 'x1' in self.df.columns and 'x2' in self.df.columns:
            self.df.dropna(subset=['x1', 'x2'], inplace=True)

        # 5. Clean y2 (Crystalline Present) - Convert to 0/1
        if 'y2' in self.df.columns:
            print(f"Unique values in y2 before cleaning: {self.df['y2'].unique()}")
            
            self.df['y2'] = self.df['y2'].astype(str).str.lower().map({
                'false': 0, 
                'true': 1, 
                '0': 0, 
                '1': 1,
                'no': 0,
                'yes': 1
            })
            self.df.dropna(subset=['y2'], inplace=True)
            self.df['y2'] = self.df['y2'].astype(int)
        
        # 6. Process CI (Crystallinity Index)
        # When y2=0 (no crystal), CI should be 0
        if 'CI' in self.df.columns and 'y2' in self.df.columns:
            self.df['CI'] = pd.to_numeric(self.df['CI'], errors='coerce')
            
            # Set CI=0 when y2=0 (no crystal present)
            self.df.loc[self.df['y2'] == 0, 'CI'] = 0.0
            
            # Fill remaining NaN with 0
            self.df['CI'] = self.df['CI'].fillna(0.0)
            
            print(f"CI processed: y2=0 -> CI=0")
            print(f"CI range: {self.df['CI'].min():.4f} to {self.df['CI'].max():.4f}")
        
        # 7. Process R2_w (Weighted R2) quality indicator
        if 'R2_w' in self.df.columns:
            self.df['R2_w'] = pd.to_numeric(self.df['R2_w'], errors='coerce')
            print(f"R2_w range before filtering: {self.df['R2_w'].min():.4f} to {self.df['R2_w'].max():.4f}")
            
            # Apply threshold filter if provided
            if r2_threshold is not None:
                before_filter = len(self.df)
                self.df = self.df[self.df['R2_w'] >= r2_threshold]
                after_filter = len(self.df)
                print(f"R2 Filter Applied: Removed {before_filter - after_filter} rows with R2_w < {r2_threshold}")
                print(f"Remaining samples: {after_filter}")
        
        print(f"\nData cleaning complete. Shape: {initial_shape} -> {self.df.shape}")
        return self.df

    def save_clean_data(self, output_path):
        """Save cleaned data to CSV."""
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
    
    def get_summary(self):
        """Print summary statistics of the cleaned data."""
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(self.df)}")
        print(f"\nFeatures:")
        print(f"  x1 (EAN concentration): {self.df['x1'].min():.2f} - {self.df['x1'].max():.2f} wt%")
        print(f"  x2 (Protein concentration): {self.df['x2'].min():.2f} - {self.df['x2'].max():.2f} mg/ml")
        print(f"\nTargets:")
        if 'y1' in self.df.columns:
            print(f"  y1 (Rg): {self.df['y1'].min():.2f} - {self.df['y1'].max():.2f} A")
        if 'y2' in self.df.columns:
            print(f"  y2 (Crystalline): {int(self.df['y2'].sum())}/{len(self.df)} samples have crystals")
        if 'CI' in self.df.columns:
            ci_nonzero = self.df[self.df['CI'] > 0]['CI']
            print(f"  CI (Crystallinity): {len(ci_nonzero)} samples with CI > 0")
            if len(ci_nonzero) > 0:
                print(f"     Range (non-zero): {ci_nonzero.min():.4f} - {ci_nonzero.max():.4f}")
        if 'R2_w' in self.df.columns:
            print(f"  R2_w (Quality): {self.df['R2_w'].min():.4f} - {self.df['R2_w'].max():.4f}")
        print("=" * 60)


if __name__ == "__main__":
    # Test run
    cleaner = DataCleaner('../data/ML_targets_crystal_oligo v3.csv')
    cleaner.load_data(header_row=0)
    cleaner.clean_data(r2_threshold=0.80)
    cleaner.get_summary()

