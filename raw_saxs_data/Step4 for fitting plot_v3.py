# -*- coding: utf-8 -*-
r"""
plot_v3_only.py

Minimal script to generate only two figures from the ML workflow:
  - analysis3/plot/composition_bargraph_v3.png
  - analysis3/fit/plot/fits_composition_v3.png

Defaults match the parameters currently used in Step3_FFmaker_Oligo_v9.py.
Optionally pass custom paths:
  python plot_v3_only.py [ANALYSIS_DIR] [FIT_DIR]

If not provided, defaults to:
  ANALYSIS_DIR = C:\Users\E100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML\3sol\analysis3
  FIT_DIR      = ANALYSIS_DIR / "fit"
"""
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- Global style (centralized settings) ----------
# Tick labels: 14; all other text (titles, labels, legends, annotations): 20
FS_TICK = 14
FS_TEXT = 20

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['xtick.labelsize'] = FS_TICK
matplotlib.rcParams['ytick.labelsize'] = FS_TICK
matplotlib.rcParams['axes.titlesize'] = FS_TEXT
matplotlib.rcParams['axes.labelsize'] = FS_TEXT
matplotlib.rcParams['legend.fontsize'] = FS_TEXT

# ---------- Defaults (match current project) ----------
BASE_DIR = Path(r"C:\Users\E100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML")
DATA_DIR = BASE_DIR / "3sol"
ANALYSIS_DIR = DATA_DIR / "analysis3"
FIT_DIR = ANALYSIS_DIR / "fit"

# Plot constants (copied to keep behavior unchanged)
PLOT_DPI = 300
PLOT_EXP_MARKER = "."
PLOT_EXP_MARKERSIZE = 3.0
PLOT_LINEWIDTH = 3
PLOT_FIT_LINEWIDTH_MULT = 2.0

# Log-x ticks/labels for SAXS fits
LOGX_TICKS = [0.01, 0.1, 0.2]
LOGX_TICKLABELS = ["0.01", "0.1", "0.2"]

# Composition text settings (used by fits v3)
COMP_TEXT_FONTSIZE = FS_TEXT
COMP_TEXT_LINEHEIGHT = 1.3

# v3 bar-graph settings (keep current values)
COMP_BAR_V3_ENABLED = True
COMP_BAR_V3_TARGET_SAMPLES = [
    "Buffer", "5min", "7min", "9min", "11min",
    "13min", "15min", "17min", "20min", "22min"
]
COMP_BAR_V3_BAR_WIDTH = 0.9
COMP_BAR_V3_FIG_WIDTH_PER_SAMPLE = 1.0
COMP_BAR_V3_FIG_WIDTH_MIN = 8
COMP_BAR_V3_YLIM = (56, 110)  # use 56-110 as in current function body
COMP_BAR_V3_LEGEND_FONTSIZE = FS_TEXT

# Color mapping (v2 palette)
COMP_COLORS_V2 = {
    "monomer": "#749D65",
    "dimer": "#9366BC",
    "tetramer": "#5989B5",
    "hexamer": "#E37F48",
    "octamer": "#D4B833",
}

# ---------- Helpers ----------
FLOAT_RE = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?"

def _read_oligomer_fit_file(fit_path: Path):
    """Read ATSAS/OLIGOMER .fit file.
    Returns: q, Iexp, Ifit (np.ndarray or None)
    """
    chi2 = np.nan
    q_list, ie_list, sig_list, if_list = [], [], [], []

    with fit_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if first:
            m = re.search(r"Chi\^2\s*=\s*([0-9\.Ee\+\-]+)", first)
            if m:
                try:
                    chi2 = float(m.group(1))
                except Exception:
                    chi2 = np.nan
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                q = float(parts[0]); ie = float(parts[1])
            except Exception:
                continue
            q_list.append(q); ie_list.append(ie)
            if len(parts) >= 4:
                try:
                    if_list.append(float(parts[3]))
                except Exception:
                    if_list.append(np.nan)

    q = np.asarray(q_list, dtype=float)
    ie = np.asarray(ie_list, dtype=float)
    ifit = np.asarray(if_list, dtype=float) if len(if_list) == len(q_list) else None
    return q, ie, ifit, chi2

# ---------- Plot: fits_composition_v3 ----------

def plot_oligomer_fits_with_composition_v3(analysis_dir: Path, fit_dir: Path):
    """
    V3: 2 rows x 5 cols layout with specific samples.
    Selected samples: Buffer, 5min, 7min, 9min, 11min, 13min, 15min, 17min, 20min, 22min
    """
    xlsx_path = analysis_dir / "ML_targets_crystal_oligo.xlsx"
    if not xlsx_path.exists():
        print(f"INFO: fits_composition_v3 skipped - {xlsx_path} not found")
        return None

    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"WARNING: failed to read ML_targets_crystal_oligo.xlsx: {e}")
        return None

    # Standardize file column
    if "file" not in df.columns:
        for c in ["File", "filename", "Filename"]:
            if c in df.columns:
                df["file"] = df[c]
                break

    target_samples = [
        "Buffer", "5min", "7min", "9min", "11min",
        "13min", "15min", "17min", "20min", "22min"
    ]
    sample_mapping = {}

    # Build mapping: display_name -> (row, fit_path)
    for target in target_samples:
        if target == "Buffer":
            candidates = df[df["file"].str.contains("buf", case=False, na=False)]
        else:
            candidates = df[df["file"].str.contains(target, case=False, na=False)]
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        fname = str(row.get("file", ""))
        stem = Path(fname).stem
        fit_path = fit_dir / f"{stem}.fit"
        if not fit_path.exists():
            cand = list(fit_dir.glob(f"{stem}*.fit"))
            if cand:
                fit_path = cand[0]
        if fit_path.exists():
            sample_mapping[target] = (row, fit_path)

    if not sample_mapping:
        print("INFO: no matching samples found for v3 fits")
        return None

    # Read fits and axis limits
    fit_data = {}
    qmins, qmaxs, imins, imaxs = [], [], [], []
    for disp, (row, fit_path) in sample_mapping.items():
        q, ie, ifit, chi2 = _read_oligomer_fit_file(fit_path)
        if len(q) == 0:
            continue
        fit_data[disp] = (q, ie, ifit)
        vals = []
        if ie is not None and len(ie):
            vals.append(ie[ie > 0])
        if ifit is not None and len(ifit):
            vals.append(ifit[ifit > 0])
        vals = [v for v in vals if v is not None and len(v)]
        if len(vals):
            all_pos = np.concatenate(vals)
            qmins.append(float(np.min(q)))
            qmaxs.append(float(np.max(q)))
            imins.append(float(np.min(all_pos)))
            imaxs.append(float(np.max(all_pos)))

    if not fit_data:
        print("INFO: no fit files readable for v3 fits")
        return None

    qmin = float(np.min(qmins)) if qmins else 0.01
    qmax = float(np.max(qmaxs)) if qmaxs else 0.30
    qmax = max(qmax, 0.35)
    ymin = float(np.min(imins)) if imins else 1e-6
    ymax = float(np.max(imaxs)) if imaxs else 1.0

    out_dir = fit_dir.parent / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(16, 8), sharex=False, sharey=True)
    axes = axes.flatten()

    # Global font handled via rcParams above

    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]

    for subplot_idx, display_name in enumerate(target_samples):
        ax = axes[subplot_idx]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(mticker.FixedLocator(LOGX_TICKS))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(LOGX_TICKLABELS))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlim(qmin, qmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', which='major', length=4, width=0.8)
        ax.tick_params(axis='both', which='minor', length=2, width=0.6)
        ax.set_yticklabels([])
        if subplot_idx % 5 == 0:
            ax.set_ylabel(r"Log $I(q)$, a.u.", fontsize=FS_TEXT, family='Arial')
        if display_name not in fit_data:
            ax.set_axis_off()
            continue
        row = sample_mapping[display_name][0]
        q, ie, ifit = fit_data[display_name]
        if ie is not None and len(ie):
            m = (ie > 0) & np.isfinite(ie) & np.isfinite(q)
            if np.any(m):
                ax.plot(q[m], ie[m], linestyle="None", marker=PLOT_EXP_MARKER,
                        markersize=PLOT_EXP_MARKERSIZE, color="0.5")
        if ifit is not None and len(ifit):
            m = (ifit > 0) & np.isfinite(ifit) & np.isfinite(q)
            if np.any(m):
                ax.plot(q[m], ifit[m], lw=PLOT_LINEWIDTH * PLOT_FIT_LINEWIDTH_MULT,
                        color="#EC4706", alpha=0.6)
        title_text = display_name
        ax.text(0.5, 0.99, title_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=FS_TEXT, weight='normal', family='Arial')
        # Composition text
        comp_lines = []
        for comp in comp_names:
            try:
                val = float(row.get(comp, 0)) * 100 if comp in row else 0
                comp_lines.append(f"{comp.capitalize()}  {val:.1f}%")
            except Exception:
                comp_lines.append(f"{comp.capitalize()}  N/A")
        comp_text = "\n".join(comp_lines)
        ax.text(0.02, 0.05, comp_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=COMP_TEXT_FONTSIZE,
                linespacing=COMP_TEXT_LINEHEIGHT, family='Arial')

    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    out_path = out_dir / "fits_composition_v3.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")
    return out_path

# ---------- Plot: composition_bargraph_v3 ----------

def plot_oligomer_composition_bargraph_v3(analysis_dir: Path):
    """
    V3 bar graph with fixed sample order (same as fits v3).
    Y-axis 56-110, Arial fonts.
    """
    if not COMP_BAR_V3_ENABLED:
        print("INFO: composition_bargraph_v3 disabled by config")
        return None

    xlsx_path = analysis_dir / "ML_targets_crystal_oligo.xlsx"
    if not xlsx_path.exists():
        print(f"INFO: composition_bargraph_v3 skipped - {xlsx_path} not found")
        return None

    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"WARNING: failed to read ML_targets_crystal_oligo.xlsx: {e}")
        return None

    if "file" not in df.columns:
        for c in ["File", "filename", "Filename"]:
            if c in df.columns:
                df["file"] = df[c]
                break

    target_samples = list(COMP_BAR_V3_TARGET_SAMPLES)
    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
    comp_colors = COMP_COLORS_V2

    samples, compositions, rgs = [], [], []
    for target in target_samples:
        if target == "Buffer":
            candidates = df[df["file"].str.contains("buf", case=False, na=False)]
        else:
            candidates = df[df["file"].str.contains(target, case=False, na=False)]
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        comp_row = []
        for comp in comp_names:
            try:
                val = float(row.get(comp, 0)) * 100 if comp in row else 0
                comp_row.append(val)
            except Exception:
                comp_row.append(0.0)
        rg = None
        for rg_col in ["apparent Rg_fit", "Rg"]:
            if rg_col in row:
                try:
                    rg = float(row[rg_col])
                    break
                except Exception:
                    pass
        samples.append(target)
        compositions.append(comp_row)
        rgs.append(rg)

    if not samples:
        print("INFO: no valid samples for v3 bar graph")
        return None

    compositions = np.array(compositions)
    n_samples = len(samples)

    fig_width = max(COMP_BAR_V3_FIG_WIDTH_MIN, n_samples * COMP_BAR_V3_FIG_WIDTH_PER_SAMPLE)
    fig, ax = plt.subplots(figsize=(fig_width, 8), dpi=300)

    # Global font handled via rcParams above

    x_pos = np.arange(n_samples)
    bottom = np.zeros(n_samples)
    bar_width = float(COMP_BAR_V3_BAR_WIDTH)

    for i, comp in enumerate(comp_names):
        color = comp_colors[comp]
        ax.bar(x_pos, compositions[:, i], bottom=bottom, label=comp.capitalize(), color=color, width=bar_width)
        for j, val in enumerate(compositions[:, i]):
            if val > 1:
                if comp == "monomer":
                    y_pos = 80.0  # place monomer percentage around y=80
                else:
                    y_pos = bottom[j] + val / 2
                ax.text(j, y_pos, f"{val:.0f}%", ha="center", va="center",
                        fontsize=FS_TEXT, color="white", weight="bold", family='Arial')
        bottom += compositions[:, i]

    # Rg annotations near y ≈ 100 (slightly above)
    for i, rg in enumerate(rgs):
        if rg is not None and not np.isnan(rg):
            ax.text(i, 101, f"{rg:.1f}", ha="center", va="bottom", fontsize=FS_TEXT, weight="bold", family='Arial')

    ax.set_xlabel("", fontsize=FS_TEXT, weight="bold", family='Arial')
    ax.set_ylabel("Composition (%)", fontsize=FS_TEXT, weight="bold", family='Arial')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=FS_TICK, family='Arial')
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(*COMP_BAR_V3_YLIM)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5,
              frameon=False, fontsize=COMP_BAR_V3_LEGEND_FONTSIZE, handlelength=1.2)

    ax.set_axisbelow(True)
    ax.grid(False)

    fig.tight_layout(pad=0.5)

    out_dir = analysis_dir / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_bargraph_v3.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote composition bargraph v3: {out_path}")
    return out_path

# ---------- Main ----------

def main():
    analysis_dir = ANALYSIS_DIR
    fit_dir = FIT_DIR
    if len(sys.argv) >= 2:
        analysis_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        fit_dir = Path(sys.argv[2])
    plot_oligomer_composition_bargraph_v3(analysis_dir)
    plot_oligomer_fits_with_composition_v3(analysis_dir, fit_dir)

if __name__ == "__main__":
    main()
