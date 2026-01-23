
import sys
import argparse
import pandas as pd
from pathlib import Path

# Local imports (assuming running from skill root or scripts dir)
# Add parent dir to path if needed for imports? 
# Actually simpler to keep imports relative if running as module, or simple path append.
sys.path.append(str(Path(__file__).parent))

from utils import read_dat, get_dat_files
from saxs_math import calculate_guinier, calculate_crystallinity

def analyze_folder(data_dir: Path, output_csv: Path):
    files = get_dat_files(data_dir)
    print(f"Found {len(files)} .dat files in {data_dir}")
    
    results = []
    
    for f in files:
        q, I, err = read_dat(f)
        
        # 1. Guinier
        guinier = calculate_guinier(q, I)
        
        # 2. Crystallinity
        crystal = calculate_crystallinity(q, I)
        
        # Merge results
        row = {"filename": f.name}
        row.update(guinier)
        row.update(crystal)
        results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SAXS Analysis")
    parser.add_argument("data_dir", help="Directory with .dat files")
    parser.add_argument("--out", default="saxs_results.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    analyze_folder(Path(args.data_dir), Path(args.out))
