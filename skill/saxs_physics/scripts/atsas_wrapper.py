
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict

class AtsasWrapper:
    """
    Minimal wrapper for ATSAS tools (FFMAKER, OLIGOMER).
    """
    
    def __init__(self, bin_path: Optional[str] = None):
        if bin_path:
            self.bin = Path(bin_path)
        else:
            # Try env var or standard defaults
            env = os.environ.get("ATSAS_BIN")
            if env:
                self.bin = Path(env)
            else:
                # Common Windows default
                self.bin = Path(r"C:\Program Files\ATSAS-4.1.3\bin")
        
        self.ffmaker = self.bin / "ffmaker.exe"
        self.oligomer = self.bin / "oligomer.exe"
        
        # Verify
        if not self.ffmaker.exists():
            print(f"[WARN] FFMAKER not found at {self.ffmaker}. ATSAS functionality may fail.")

    def make_form_factors(self, pdb_files: List[Path], output_ff: Path, 
                         s_min: float = 0.0, s_max: float = 0.3, ns: int = 101):
        """
        Runs FFMAKER to generate a theoretical form factor file (.dat) from PDB models.
        """
        if not pdb_files:
            raise ValueError("No PDB files provided.")

        # Ensure absolute paths
        pdb_args = [str(p.resolve()) for p in pdb_files]
        out_str = str(output_ff.resolve())

        # Build command: ffmaker [options] -o out.dat pdb1 pdb2 ...
        cmd = [
            str(self.ffmaker),
            f"--smin={s_min}",
            f"--smax={s_max}",
            f"--ns={ns}",
            "-o", out_str
        ] + pdb_args

        print(f"Running FFMAKER: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        if not output_ff.exists():
            raise RuntimeError("FFMAKER ran but output file was not created.")

    def run_oligomer(self, dat_file: Path, ff_file: Path, output_file: Path = None) -> float:
        """
        Runs OLIGOMER to fit a mixture model.
        Returns the Chi-squared value.
        """
        if not output_file:
            # Default to .fit in same dir
            output_file = dat_file.with_suffix(".fit")
            
        cmd = [
            str(self.oligomer),
            "-ff", str(ff_file.resolve()),
            str(dat_file.resolve())
            # Note: OLIGOMER automatically writes to .fit or stdout depending on version,
            # this is a simplified call. Real wrapping might need stdout parsing.
        ]
        
        # Capture output to parse Chi^2
        print(f"Running OLIGOMER on {dat_file.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=dat_file.parent)
        
        if result.returncode != 0:
            print(f"[ERROR] OLIGOMER failed: {result.stderr}")
            return float('nan')

        # Simple parsing logic (ATSAS versions vary, but often print "Chi^2 : X.XXX")
        import re
        match = re.search(r"Chi\^?2[^:]*:\s*([0-9.]+)", result.stdout + result.stderr) 
        if match:
            return float(match.group(1))
        
        return float('nan')
