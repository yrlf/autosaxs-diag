---
name: SAXS Physics Pipeline
description: An automated 5-stage pipeline for analyzing SAXS data, from raw curves to physical parameters (Rg, CI) and oligomer decomposition.
version: 2.0.0
author: Antigravity
---

# SAXS Physics Analysis Pipeline

This skill encapsulates a comprehensive, automated workflow for extracting physical and crystallographic insights from Small-Angle X-ray Scattering (SAXS) data. It is designed to handle large batches of experimental files (`.dat`) and produce publication-ready visualizations.

## Pipeline Stages

The workflow consists of 5 sequential stages:

1.  **Guinier Analysis (Step 1)**
    *   **Script**: `pipeline/step1_guinier.py`
    *   **Function**: Performs automated pseudo-Rg analysis using a sliding window or forced window approach.
    *   **Outputs**: $R_g$, $I(0)$, data quality diagnostics (curvature checks), and normalized logs (`log-saxs.csv`).

2.  **Phase Diagram Construction (Step 1.5)**
    *   **Script**: `pipeline/step1_5_phase_diagram.py`
    *   **Function**: Visualizes the phase space (e.g., Protein Concentration vs. IL Concentration).
    *   **Outputs**: Heatmaps and phase diagrams showing regions of Solubility, Aggregation, and Crystallization.

3.  **Crystallinity Analysis (Step 2)**
    *   **Script**: `pipeline/step2_crystallinity.py`
    *   **Function**: Peak finding algorithm to detect Bragg peaks.
    *   **Outputs**:
        *   **Crystallinity Index (CI)**: Quantification of crystal content.
        *   **Domain Size**: Scherrer equation analysis on the first peak.
        *   **Diagnostics**: Mosaic plots of all crystalline samples.

4.  **Oligomer Decomposition (Step 3)**
    *   **Script**: `pipeline/step3_oligomer.py`
    *   **Function**: Interfaces with **ATSAS (OLIGOMER)** to fit experimental curves as linear combinations of theoretical form factors (Monomer, Dimer, Hexamer, etc.).
    *   **Requirements**: ATSAS installed locally (`FFMAKER.exe`, `OLIGOMER.exe`) and PDB models.
    *   **Outputs**: Composition percentages (e.g., 80% Monomer, 20% Hexamer) and fit quality ($\chi^2$).

5.  **Final Visualization (Step 4)**
    *   **Script**: `pipeline/step4_plotting.py`
    *   **Function**: Generates polished, publication-style figures (Bar graphs of composition, Stacked fit plots).

## Directory Structure

```text
skill/
├── SKILL.md                 # This file
├── run_pipeline.py          # Unified entry point (TODO)
└── pipeline/                # Core logic scripts
    ├── step1_guinier.py
    ├── step1_5_phase_diagram.py
    ├── step2_crystallinity.py
    ├── step3_oligomer.py
    └── step4_plotting.py
```

## Usage

To run a specific stage of the pipeline:

```bash
# Example: Run Step 1 (Guinier)
python skill/pipeline/step1_guinier.py

# Example: Run Step 2 (Crystallinity)
python skill/pipeline/step2_crystallinity.py
```

*Note: Paths to input data are currently hardcoded in the `CONFIG` sections of each script. Refer to the top of each `.py` file to adjust `DATA_DIR` or `ANALYSIS_DIR`.*

## Dependencies

*   Python 3.8+
*   `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
*   `openpyxl` (for Excel I/O)
*   **ATSAS** (External software required for Step 3)
