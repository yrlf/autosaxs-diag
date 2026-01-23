
import numpy as np
from pathlib import Path
from typing import Tuple, List

def read_dat(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a standard SAXS .dat file (q, I, err).
    Handles headers (lines starting with #, !, etc).
    
    Returns:
        (q, I, err) as numpy arrays.
    """
    q_list, I_list, err_list = [], [], []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#!%;":
                continue
            
            parts = line.split()
            # We need at least q and I
            if len(parts) < 2:
                continue
                
            try:
                qv = float(parts[0])
                iv = float(parts[1])
                # Err is optional, default to 0.0 or estimate? Let's use 0.0 if missing
                ev = float(parts[2]) if len(parts) > 2 else 0.0
                
                q_list.append(qv)
                I_list.append(iv)
                err_list.append(ev)
            except ValueError:
                continue
                
    return (
        np.array(q_list), 
        np.array(I_list), 
        np.array(err_list)
    )

def get_dat_files(directory: Path) -> List[Path]:
    """Finds all .dat files in directory."""
    return sorted(list(directory.glob("*.dat")))
