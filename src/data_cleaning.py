import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Loads data from CSV, skipping the first row which is metadata."""
        try:
            # Header is on the second row (index 1)
            self.df = pd.read_csv(self.filepath, header=1)
            print(f"Successfully loaded data. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

    def clean_data(self):
        """Performs data cleaning steps."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_shape = self.df.shape
        
        # 1. Clean column names
        # Remove whitespace
        self.df.columns = self.df.columns.str.strip()
        
        # Identify key columns mapping
        # We need to map the long descriptions to short codes for easier handling
        # Note: Adjust these string matches based on exact CSV headers if needed
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
                
        # Rename columns
        self.df.rename(columns=column_map, inplace=True)
        print(f"Renamed columns to: {[c for c in self.df.columns if c in ['x1', 'x2', 'y1', 'y2']]}")

        # 2. Select only relevant columns + any others we might want to keep? 
        # For now, let's keep all but focus cleaning on the main ones.
        # Actually, let's look for Unnamed columns and drop them.
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        # 3. Handle Missing Values
        # Target y1 (Rg): Drop rows where y1 is missing
        if 'y1' in self.df.columns:
            self.df.dropna(subset=['y1'], inplace=True)
        
        # Input features: Drop rows where x1 or x2 is missing
        if 'x1' in self.df.columns and 'x2' in self.df.columns:
            self.df.dropna(subset=['x1', 'x2'], inplace=True)

        # 4. specific cleaning for y2 (Crystalline)
        # It might be boolean or string. Convert to 0/1.
        if 'y2' in self.df.columns:
            # Check unique values
            print(f"Unique values in y2 before cleaning: {self.df['y2'].unique()}")
            
            # Map False/True to 0/1, and handle strings if any
            self.df['y2'] = self.df['y2'].astype(str).str.lower().map({
                'false': 0, 
                'true': 1, 
                '0': 0, 
                '1': 1,
                'no': 0,
                'yes': 1
            })
            # Fill remaining NaNs in y2 with 0 (Assuming NaN means no crystal observed, OR drop? 
            # Plan said: Ensure boolean/string is converted. 
            # If y2 is the target, we shouldn't guess. Let's drop NaNs in y2 for now to be safe, or check count.
            self.df.dropna(subset=['y2'], inplace=True)
        
        print(f"Data cleaning complete. Shape: {initial_shape} -> {self.df.shape}")
        return self.df

    def save_clean_data(self, output_path):
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    # Test run
    cleaner = DataCleaner('../ML_targets_crystal_oligo v2.csv')
    cleaner.load_data()
    cleaner.clean_data()
