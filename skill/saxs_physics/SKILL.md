---
name: saxs-physics
description: A lightweight, physics-based toolkit for analyzing Small-Angle X-ray Scattering (SAXS) data. Calculates Rg, I(0), and Crystallinity Index without bloat.
---

# SAXS Physics Toolkit

This skill provides a minimalist, pure-Python library for extracting physical parameters from SAXS data. Unlike complex pipelines, these scripts focus solely on the mathematical analysis.

## Core Capabilities

1.  **Guinier Analysis**: Extract Radius of Gyration ($R_g$) and forward scattering $I(0)$.
2.  **Crystallinity Analysis**: Detect Bragg peaks and calculate Crystallinity Index (CI).
3.  **Oligomer Wrapper**: Simple interface to ATSAS tools (requires local installation).

## Usage

### 1. Batch Analysis (CLI)
Run the batch analyzer on a directory of `.dat` files:

```bash
python skill/saxs_physics/scripts/analyze_batch.py /path/to/data --out results.csv
```

### 2. Library Usage (Python)
You can import the math functions directly in your own scripts:

```python
from skill.saxs_physics.scripts.utils import read_dat
from skill.saxs_physics.scripts.saxs_math import calculate_guinier

q, I, err = read_dat("sample.dat")
res = calculate_guinier(q, I)
print(f"Rg = {res['Rg']} A")
```

## Structure
*   `scripts/saxs_math.py`: Pure numpy implementation of physics equations.
*   `scripts/atsas_wrapper.py`: Minimal wrapper for `oligomer.exe`.
*   `scripts/analyze_batch.py`: CLI tool for bulk processing.
