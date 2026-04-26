# autosaxs-diag

Automated diagnostics for protein samples in **ethylammonium nitrate (EAN)** ionic-liquid buffers
from Small-Angle X-ray Scattering (SAXS) curves. The pipeline pairs a lightweight
physics extractor with a comparative ML benchmark over experimental controls
(EAN concentration, protein concentration).

## What the project does

1. **Physics extraction** (`skill/saxs_physics/`) — pure-Python Guinier fit (Rg, I(0)),
   Bragg-peak detection, and Crystallinity Index (CI) from raw `.dat` files.
2. **ML benchmark** (`scripts/evaluate.py`) — eight regressors and eight classifiers
   trained on `(x1, x2) = (EAN wt%, protein mg/mL)` to predict
   - **Rg** (continuous, Å),
   - **CI** (continuous),
   - **crystalline / amorphous** (binary, CI > 0.1).

   Reports R², RMSE, AUC, F1, Youden, Brier, calibration slope, plus ROC,
   calibration, decision-curve, feature-importance, and SHAP plots.
3. **Publication plots** (`scripts/generate_publication_plots.py`,
   `scripts/generate_heatmap.py`, `scripts/generate_heatmap_ci.py`) —
   journal-ready figures including 2-D prediction heatmaps over the
   `(x1, x2)` space.

## Dataset

99 samples spanning 11 EAN concentrations × 9 protein concentrations,
extracted from the raw curves under `raw_saxs_data/` and consolidated in
`data/cleaned_data.csv` (also `data/ML_targets_crystal_oligo v3.csv`,
the source spreadsheet exported as CSV).

| Column | Meaning |
| ------ | ------- |
| `x1`   | EAN concentration (wt%) |
| `x2`   | Protein concentration (mg/mL) |
| `y1`   | Rg (Å, Guinier fit) |
| `y2`   | Crystalline present (0/1) |
| `CI`   | Crystallinity Index |
| `R2_w` | Weighted R² of Guinier fit (quality flag) |

## Installation

```bash
pip install -e .
```

Targets **Python ≥ 3.9**. Optional ATSAS tools (`autorg`, `oligomer`) are required
only if you re-run the oligomer fits inside `skill/saxs_physics/`.

## Usage

Run the canonical multi-model evaluation:

```bash
python scripts/evaluate.py
```

This loads `data/ML_targets_crystal_oligo v3.csv`, applies the cleaning pipeline
in `src/data_cleaning.py` (R² ≥ 0.80 quality filter, CI = 0 when
`y2 = FALSE`), trains all models on an 80/20 split and writes metrics + plots
under `outputs/evaluation/{Rg_regression, CI_regression, CI_classification}/`.

Generate publication figures:

```bash
python scripts/generate_publication_plots.py
python scripts/generate_heatmap.py        # Rg(x1, x2) heatmap
python scripts/generate_heatmap_ci.py     # CI(x1, x2) heatmap
```

Re-extract physics features from raw curves:

```bash
python skill/saxs_physics/scripts/analyze_batch.py raw_saxs_data --out features.csv
```

## Layout

```
.
├── src/                              # Cleaning + modelling library
│   ├── data_cleaning.py              # DataCleaner (column rename, R² filter, CI logic)
│   ├── modeling.py                   # Model zoo wrappers
│   └── visualization.py              # Heatmap pipeline + colormaps
├── scripts/
│   ├── evaluate.py                   # canonical: train + eval all models
│   ├── generate_publication_plots.py # journal-style ROC / DCA / SHAP
│   ├── generate_heatmap.py           # Rg heatmap
│   ├── generate_heatmap_ci.py        # CI heatmap
│   ├── generate_analysis_charts.py   # dashboard / radar plots
│   ├── run_augmentation_test.py      # noise-augmentation experiment
│   └── run_feature_engineering.py    # polynomial / engineered features
├── skill/saxs_physics/               # physics extractor (Guinier, CI, oligomer)
├── data/
│   ├── cleaned_data.csv              # consolidated training table (n=99)
│   └── ML_targets_crystal_oligo v3.csv  # raw spreadsheet export
├── raw_saxs_data/                    # 100+ .dat curves (EAN/protein grid + buffer)
├── manuscript.md                     # paper draft
├── hyperparameter_tuning_description.md
├── pyproject.toml
├── LICENSE
└── README.md
```

## Headline results

Best models per task on the held-out 20% test set (see `evaluate.py` output):

| Task                    | Best model              | Metric                       |
| ----------------------- | ----------------------- | ---------------------------- |
| Rg regression           | XGBoost / Random Forest | R² ≈ 0.78 – 0.80             |
| CI regression           | Gradient Boosting       | R² ≈ 0.7                     |
| Crystallinity (CI>0.1)  | Gradient Boosting / RF  | AUC ≈ 0.94, Youden ≈ 0.81    |

`x1` (EAN concentration) dominates feature importance — SHAP magnitudes are
roughly 3× larger than `x2` for every tree-based model.

## License

Released under the MIT License (see `LICENSE`).
