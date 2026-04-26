# -*- coding: utf-8 -*-
"""
Heatmaps from existing Excel grids (VS Code friendly)

Reads ONLY (from diagram/saxs_phase_diagram_with_grids.xlsx):
- Grid_Rg
- Grid_Phase
- Grid_Crystal

Outputs (diagram/, overwritten):
- phase_diagram_heatmap_points.png
- phase_diagram_heatmap_v2.png
- phase_diagram_heatmap_v2_block_boundaries.png  (NEW: block heatmap with visible cell boundaries)

Why your *_block.png likely had "missing boundaries":
- pcolormesh defaults to edgecolors='none' (or edgecolors='face') and linewidth=0, so adjacent cells blend.
- If antialiasing is on and linewidth is tiny, edges can disappear when rasterized/saved.

Fix in this script:
- Uses pcolormesh with explicit edgecolors + linewidth + antialiased=False to force crisp cell borders.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LinearLocator, StrMethodFormatter

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =============================================================================
# CONFIG (edit these)
# =============================================================================

ANALYSIS_DIR = r"C:\Users\e100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML\6\analysis\selected\analysis"
DIAGRAM_SUBDIR = "diagram"
XLSX_NAME = "saxs_phase_diagram_with_grids.xlsx"

# Global font settings
FONT_FAMILY = "Arial"
# Base tick-label font size; most other text is scaled by FONT_SCALE_FROM_TICKS
BASE_TICK_FONTSIZE = 14
FONT_SCALE_FROM_TICKS = 1.5   # axis labels / legend / colorbar label = tick fontsize * this

# Apply globally (safe fallback if Arial not available)
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = BASE_TICK_FONTSIZE

# Common figure size for key outputs (inches)
# Keep phase_diagram_heatmap_v2_block_boundaries and phase_diagram_phase2_block_boundaries identical:
FIGSIZE_COMMON = (10.0, 6.0)   # (width, height)

# v2 plot labels (edit here)
V2_XLABEL = "Molar percent of ILs (mol%)"
V2_YLABEL = "Protein (mg/mL)"
V2_CBAR_LABEL = "Rg (Å)"
V2_LEGEND_BBOX_Y = 0.98
V2_LEGEND_NCOL = 4


# Outputs to generate
MAKE_HEATMAP1_CONTINUOUS = True
MAKE_HEATMAP_V2_CONTINUOUS = True
MAKE_HEATMAP_V2_BLOCK = True

OUT_HEATMAP1 = "phase_diagram_heatmap_points.png"
OUT_HEATMAP_V2 = "phase_diagram_heatmap_v2.png"
OUT_HEATMAP_V2_BLOCK = "phase_diagram_heatmap_v2_block_boundaries.png"

# =============================================================================
# NEW: Grid_Phase2 + Grid_Rg2 (protein vs IL wt%) plots
# =============================================================================
# Requested combined plot:
#   - block heatmap from Grid_Rg2 (Rg values) with visible cell boundaries
#   - overlaid phase "points" (squares) from Grid_Phase2
# Both in ONE figure, saved as OUT_PHASE2_COMBINED.
MAKE_PHASE2_COMBINED = True

# Optional separate Phase2-only diagnostics (keep False unless you need them)
MAKE_PHASE2_POINTS = False
MAKE_PHASE2_BLOCK = False

OUT_PHASE2_COMBINED = "phase_diagram_phase2_block_boundaries.png"
OUT_PHASE2_POINTS = "phase_diagram_phase2_points.png"
OUT_PHASE2_BLOCK = "phase_diagram_phase2_phase_block_boundaries.png"

# Sheet name for the Phase2 heatmap layer (Rg). If not present, the script falls back to Grid_Rg and prints a warning.
PHASE2_RG_SHEET = "Grid_Rg2"
PHASE2_RG_SHEET_FALLBACK = "Grid_Rg"

# Sheet name for Phase labels (points overlay)
PHASE2_PHASE_SHEET = "Grid_Phase2"

# Rg color scale for Phase2 combined plot (set to numbers to lock; keep None for auto)
PHASE2_RG_VMIN = 11
PHASE2_RG_VMAX = 23
PHASE2_RG_CBAR_TICKS = None
PHASE2_RG_ALPHA = 0.70   # match HEATMAP_ALPHA if you want consistency

# Figure size (inches) for Phase2 figures
# Set width/height explicitly (NOT forced square)
PHASE2_FIGSIZE = (8.0, 6.0)   # (width, height)
# Backward-compatible square size (used only if PHASE2_FIGSIZE is None)
PHASE2_FIGSIZE_INCH = 7.5

# Axis scaling for Phase2 plot:
#   "numeric" -> true coordinate scaling (linear axes; cell sizes follow actual x/y spacing)
#   "categorical" -> equal-size cells (NOT recommended for Phase2)
PHASE2_AXIS_MODE = "numeric"   # "categorical" or "numeric"

# Swap axes for the Phase2 combined plot:
#   True: x uses IL (wt%) and y uses Protein (mg/mL)
#   False: x uses Protein and y uses IL
PHASE2_SWAP_XY = False

# Tick density control:
#   "linear" -> a fixed number of evenly spaced major ticks on each axis
#   "all"    -> show every grid value (dense)
PHASE2_TICK_MODE = "linear"   # "linear" or "all"
PHASE2_NTICKS_X = 6
PHASE2_NTICKS_Y = 6

# Explicit tick positions (numeric axes). If not None, override PHASE2_TICK_MODE/NTICKS.
# NOTE: If you set PHASE2_SWAP_XY = True, you probably want to swap these lists too.
PHASE2_XTICKS_EXPLICIT = [1, 5, 20, 50, 70, 100]
PHASE2_YTICKS_EXPLICIT = [0, 5, 10, 40, 60, 80, 100]
PHASE2_FILTER_EXPLICIT_TICKS_TO_RANGE = True  # drop ticks outside axis limits

# Axis limits (numeric axes). Use None to follow the grid extents.
# Example: PHASE2_XLIM = (1, 112.5)
PHASE2_XLIM = (1, None)    # Protein axis (mg/mL)
PHASE2_YLIM = (0, None)    # IL axis (wt%)

# Legend placement (above the plot) and spacing
PHASE2_LEGEND_BBOX_Y = 0.98       # >1 puts legend above axes
PHASE2_LEGEND_NCOL = None         # None -> auto up to 4
PHASE2_TIGHTLAYOUT_TOP = 0.92     # reserve space for top legend (0-1)

# Colorbar formatting (Phase2)
PHASE2_CBAR_LABEL = "Rg (A)"
PHASE2_CBAR_PAD = 0.015           # gap between axes and colorbar (smaller -> tighter)
PHASE2_CBAR_SHRINK = 0.95
PHASE2_CBAR_ASPECT = 30
PHASE2_CBAR_TICKS = [11, 13, 15, 17, 19, 21, 23]          # e.g. [12, 14, 16, 18, 20, 22]

# Phase overlay square styling (applies to the Phase2 combined plot)
PHASE2_PHASE_POINT_SIZE_SCALE = 1   # halve the square size
PHASE2_PHASE_POINT_ALPHA = 1.0       # more transparent (lower opacity)
PHASE2_PHASE_POINT_EDGE = "none"

# Block boundary settings (Phase2)
PHASE2_SHOW_CELL_BORDERS = True
PHASE2_CELL_BORDER_COLOR = "white"
PHASE2_CELL_BORDER_LINEWIDTH = 0.6
PHASE2_CELL_BORDER_ANTIALIASED = False

# Marker size for the "points" plot (squares)
PHASE2_POINT_SIZE = 50

# Export resolution for Phase2 figures
# (kept separate from other outputs so you can tune Phase2 without changing v2/v1 plots)
PHASE2_DPI = 300

# Tick label formatting
PHASE2_XTICK_ROT = 0
PHASE2_YTICK_ROT = 0
PHASE2_TICK_FONTSIZE = BASE_TICK_FONTSIZE

# Axes labels for Phase2 plots
PHASE2_XLABEL = "Protein (mg/mL)"
PHASE2_YLABEL = "Concentration of IL (wt%)"

# Continuous heatmap settings
HEATMAP_ALPHA = 0.70
HEATMAP_INTERPOLATION = "bilinear"   # "nearest"/"bilinear"/"bicubic"

# Block heatmap boundary settings (NEW)
SHOW_CELL_BORDERS = True
CELL_BORDER_COLOR = "white"          # good contrast for viridis; change to "k" if preferred
CELL_BORDER_LINEWIDTH = 0.0         # 0.25–0.6 is typically readable
CELL_BORDER_ANTIALIASED = False      # IMPORTANT: keeps borders crisp in saved PNGs

# Shared color scale (manual)
RG_VMIN = 11
RG_VMAX = 23
RG_CBAR_TICKS = [11, 13, 15, 17, 19, 21, 23]

# =============================================================================
# Legacy 2-panel broken-axis layout (points)
# =============================================================================
BROKEN_WIDTH_LEFT = 9
BROKEN_WIDTH_RIGHT = 1
WSPACE = 0.01

X_LEFT_MIN, X_LEFT_MAX = -2, 35
X_LEFT_TICKS = [0, 10, 20, 30]

# Right panel display range
X_RIGHT_MIN, X_RIGHT_MAX = 90, 102
X_RIGHT_TICKS = [100]

# Mapping real-x to right-panel x
X_RIGHT_DATA_MIN, X_RIGHT_DATA_MAX = 90, 100  # auto-adjusted if grid differs

# Y display range
Y_MIN, Y_MAX = 2.5, 102
Y_TICKS = [5, 20, 40, 60, 80, 100]

# Scatter styling
POINT_SIZE = 50

# Gap fill color mode (between broken panels)
GAP_FILL_MODE = "match_left_edge"  # "match_left_edge" / "match_right_edge" / "white"

# Fill entire right panel using the LAST COLUMN (e.g., 93 mol%) colors
RIGHT_FILL_WITH_LAST_COLUMN = True

# =============================================================================
# v2: 3-panel broken axis (left zoom + mid + right)
# =============================================================================
X_V2_LEFT_MIN, X_V2_LEFT_MAX = 0, 11
X_V2_MID_MIN, X_V2_MID_MAX = 14, 35

BROKEN_V2_WIDTH_LEFT = 6
BROKEN_V2_WIDTH_MID = 3
BROKEN_V2_WIDTH_RIGHT = 1
WSPACE_V2 = 0.01

X_V2_MID_TICKS = [17, 25, 33]  # adjust if needed

# =============================================================================
# Phase labels & colors
# =============================================================================
PHASE_ORDER = ["Soluble", "Crystalline/Ordered", "Aggregated/Oligomer", "Uncertain"]
PHASE_COLORS = {
    "Soluble": (0.35, 0.35, 0.35, 1.0),
    "Crystalline/Ordered": (0.84, 0.15, 0.16, 1.0),
    "Aggregated/Oligomer": (0.85, 0.40, 0.05, 1.0),
    "Uncertain": (0.20, 0.45, 0.75, 1.0),
}


def normalize_phase_label(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""
    sl = s.lower()
    if "solub" in sl:
        return "Soluble"
    if "cryst" in sl:
        return "Crystalline/Ordered"
    if "agg" in sl or "olig" in sl or "precip" in sl:
        return "Aggregated/Oligomer"
    if "uncertain" in sl or "insol" in sl:
        return "Uncertain"
    for p in PHASE_ORDER:
        if sl == p.lower():
            return p
    return s


def make_top_legend(fig, present: List[str], fontsize: int = 12, markersize: int = 8, bbox_y: float = 0.97, ncol: int | None = None) -> None:
    handles = []
    for p in present:
        if p not in PHASE_COLORS:
            continue
        handles.append(
            Line2D(
                [0], [0], marker="s", linestyle="None",
                markersize=int(markersize), markerfacecolor=PHASE_COLORS[p],
                markeredgecolor="none", label=p
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, float(bbox_y)),
            ncol=(min(4, len(handles)) if ncol is None else int(ncol)),
            frameon=False, fontsize=int(fontsize)
        )


# =============================================================================
# Excel readers
# =============================================================================
def _read_sheet_indexed(xlsx: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet, index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    return df


def read_grid_rg(xlsx: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_sheet_indexed(xlsx, "Grid_Rg")
    x = pd.to_numeric(df.columns, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.index, errors="coerce").to_numpy(float)[::-1]  # flip y order
    z = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[::-1, :]  # flip rows
    return x, y, z


def read_grid_phase(xlsx: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_sheet_indexed(xlsx, "Grid_Phase")
    x = pd.to_numeric(df.columns, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.index, errors="coerce").to_numpy(float)[::-1]
    raw = df.to_numpy(dtype=object)[::-1, :]
    norm = np.vectorize(normalize_phase_label, otypes=[object])(raw)
    return x, y, norm


def read_grid_crystal(xlsx: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_sheet_indexed(xlsx, "Grid_Crystal")
    x = pd.to_numeric(df.columns, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.index, errors="coerce").to_numpy(float)
    z = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return x, y, z


def read_grid_phase2(xlsx: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read Grid_Phase2.

    Expected structure (as in your current Excel):
      - index: protein (mg/mL)
      - columns: IL concentration (wt%)
      - values: phase labels

    Returns:
      x_protein (nx,)
      y_il (ny,)
      phase (ny, nx)  # y-major for plotting (rows = y, cols = x)
    """
    df = _read_sheet_indexed(xlsx, "Grid_Phase2")

    # axes
    x_protein = pd.to_numeric(df.index, errors="coerce").to_numpy(float)
    y_il = pd.to_numeric(df.columns, errors="coerce").to_numpy(float)

    # df is (protein rows, IL cols) -> transpose for (IL rows, protein cols)
    raw_t = df.to_numpy(dtype=object).T
    phase = np.vectorize(normalize_phase_label, otypes=[object])(raw_t)
    return x_protein, y_il, phase





def read_phase2_rg2_and_phase2(
    xlsx: Path,
    rg_sheet: str = "Grid_Rg2",
    phase_sheet: str = "Grid_Phase2",
    rg_fallback: str = "Grid_Rg",
    master_fallback_sheet: str = "Master",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Read and ALIGN Phase2 Rg heatmap grid with Phase2 phase-label grid.

    In the *current Excel you provided*, Grid_Phase2 is arranged as:
      - rows (index): IL concentration (wt%)
      - columns: Protein (mg/mL)
      - values: phase labels (strings)

    This function supports BOTH layouts for the Rg grid (Grid_Rg2):
      A) index=IL(wt%), columns=protein  (preferred)
      B) index=protein, columns=IL(wt%)  (will be auto-transposed)

    If rg_sheet is missing, it will try:
      1) build an Rg2 grid from the 'Master' sheet by parsing 'tag' (EANxx_yyymg, Buf_yyymg)
      2) else fall back to rg_fallback (Grid_Rg) with best-effort alignment (may not match Phase2 axes)

    Returns:
      x_protein (nx,)
      y_il (ny,)
      rg   (ny, nx)      # rows=y (IL), cols=x (protein)
      phase(ny, nx)      # rows=y (IL), cols=x (protein)
      rg_sheet_used
    """
    # ---- helpers ----
    def _read_indexed(sheet: str) -> pd.DataFrame:
        df = pd.read_excel(xlsx, sheet_name=sheet, index_col=0)
        df.index = pd.to_numeric(df.index, errors="coerce")
        df.columns = pd.to_numeric(df.columns, errors="coerce")
        df = df.loc[df.index.notna(), df.columns.notna()]
        # sort numeric axes
        df = df.sort_index(axis=0).sort_index(axis=1)
        return df

    def _build_rg2_from_master() -> pd.DataFrame:
        dfm = pd.read_excel(xlsx, sheet_name=master_fallback_sheet)
        if "tag" not in dfm.columns or "Rg" not in dfm.columns:
            raise ValueError(f"'{master_fallback_sheet}' must contain columns 'tag' and 'Rg'.")
        tags = dfm["tag"].astype(str).tolist()
        il_vals = []
        pr_vals = []

        for t in tags:
            if t.startswith("Buf"):
                m = re.search(r"Buf_([0-9.]+)mg", t)
                if m:
                    il_vals.append(0.0)
                    pr_vals.append(float(m.group(1)))
                    continue
                m = re.search(r"Buf_([0-9.]+)mgb", t)
                if m:
                    il_vals.append(0.0)
                    pr_vals.append(float(m.group(1)))
                    continue

            m = re.search(r"EAN([0-9.]+)_([0-9.]+)mg", t)
            if m:
                il_vals.append(float(m.group(1)))
                pr_vals.append(float(m.group(2)))
                continue
            m = re.search(r"EAN([0-9.]+)_([0-9.]+)mgb", t)
            if m:
                il_vals.append(float(m.group(1)))
                pr_vals.append(float(m.group(2)))
                continue

            il_vals.append(np.nan)
            pr_vals.append(np.nan)

        dfm2 = dfm.copy()
        dfm2["IL_wt"] = pd.to_numeric(il_vals, errors="coerce")
        dfm2["protein_mg"] = pd.to_numeric(pr_vals, errors="coerce")
        dfm2["Rg"] = pd.to_numeric(dfm2["Rg"], errors="coerce")
        dfm2 = dfm2.dropna(subset=["IL_wt", "protein_mg", "Rg"])

        # pivot to index=IL (wt%), columns=protein (mg/mL)
        out = dfm2.pivot_table(index="IL_wt", columns="protein_mg", values="Rg", aggfunc="mean")
        out = out.sort_index(axis=0).sort_index(axis=1)
        return out

    def _phase_agg(values: pd.Series) -> object:
        """Aggregate multiple phase labels for the same (IL, protein) condition."""
        vals = [normalize_phase_label(v) for v in values.tolist()]
        vals = [v for v in vals if v not in {"", None} and str(v).strip() != ""]
        if not vals:
            return np.nan
        vc = pd.Series(vals, dtype=object).value_counts()
        top = int(vc.max())
        candidates = [c for c, n in vc.items() if int(n) == top]
        # tie-break by PHASE_ORDER, else first candidate
        for p in PHASE_ORDER:
            if p in candidates:
                return p
        return candidates[0]

    def _build_phase2_from_master() -> pd.DataFrame:
        """
        Build a Phase2 grid (index=IL wt%, columns=protein mg/mL) from the Master sheet.

        Requirements in 'Master':
          - 'tag' column with patterns like EANxx_yyymg or Buf_yyymg
          - a phase label column: 'phase' (preferred) or 'Phase'
        """
        dfm = pd.read_excel(xlsx, sheet_name=master_fallback_sheet)

        if "tag" not in dfm.columns:
            raise ValueError(f"'{master_fallback_sheet}' must contain a 'tag' column.")

        phase_col = None
        for cand in ["phase", "Phase", "phase_label", "Phase_label", "phase2", "Phase2"]:
            if cand in dfm.columns:
                phase_col = cand
                break
        if phase_col is None:
            raise ValueError(
                f"'{master_fallback_sheet}' must contain a phase column (e.g., 'phase')."
            )

        tags = dfm["tag"].astype(str).tolist()
        il_vals: List[float] = []
        pr_vals: List[float] = []

        for t in tags:
            if t.startswith("Buf"):
                m2 = re.search(r"Buf_([0-9.]+)mg", t)
                if m2:
                    il_vals.append(0.0)
                    pr_vals.append(float(m2.group(1)))
                    continue
                m2 = re.search(r"Buf_([0-9.]+)mgb", t)
                if m2:
                    il_vals.append(0.0)
                    pr_vals.append(float(m2.group(1)))
                    continue

            m2 = re.search(r"EAN([0-9.]+)_([0-9.]+)mg", t)
            if m2:
                il_vals.append(float(m2.group(1)))
                pr_vals.append(float(m2.group(2)))
                continue
            m2 = re.search(r"EAN([0-9.]+)_([0-9.]+)mgb", t)
            if m2:
                il_vals.append(float(m2.group(1)))
                pr_vals.append(float(m2.group(2)))
                continue

            il_vals.append(np.nan)
            pr_vals.append(np.nan)

        dfm2 = dfm.copy()
        dfm2["IL_wt"] = pd.to_numeric(il_vals, errors="coerce")
        dfm2["protein_mg"] = pd.to_numeric(pr_vals, errors="coerce")
        dfm2["phase_norm"] = dfm2[phase_col].apply(normalize_phase_label)

        dfm2 = dfm2.dropna(subset=["IL_wt", "protein_mg"])
        dfm2 = dfm2[dfm2["phase_norm"].astype(str).str.strip() != ""]
        if dfm2.empty:
            raise ValueError(f"No usable phase entries found in '{master_fallback_sheet}'.")

        out = dfm2.pivot_table(
            index="IL_wt",
            columns="protein_mg",
            values="phase_norm",
            aggfunc=_phase_agg,
        )
        out = out.sort_index(axis=0).sort_index(axis=1)
        return out

    # ---- discover sheets ----
    try:
        sheets = set(pd.ExcelFile(xlsx).sheet_names)
    except Exception:
        sheets = set()

    # ---- read/build Phase2 phase labels (authoritative axes) ----
    if phase_sheet in sheets:
        try:
            df_ph = _read_indexed(phase_sheet).astype(object)
        except Exception as e:
            raise ValueError(f"Failed to read '{phase_sheet}' from {xlsx}: {e}")
    else:
        if master_fallback_sheet in sheets:
            try:
                df_ph = _build_phase2_from_master().astype(object)
                print(f"[INFO] '{phase_sheet}' not found. Built Phase2 phase grid from '{master_fallback_sheet}'.")
            except Exception as e:
                raise ValueError(f"Failed to build Phase2 phase grid from '{master_fallback_sheet}': {e}")
        else:
            raise ValueError(f"Neither '{phase_sheet}' nor '{master_fallback_sheet}' is available in: {xlsx}")

    # normalize labels
    df_ph = df_ph.applymap(normalize_phase_label)

    # Phase2 axes convention for plotting: y=index (IL), x=columns (protein)
    y_il_ph = df_ph.index.to_numpy(dtype=float)
    x_pr_ph = df_ph.columns.to_numpy(dtype=float)
    # ---- read / build Rg grid ----
    # (sheet list already discovered above)

    rg_sheet_used = None
    df_rg = None

    if rg_sheet in sheets:
        rg_sheet_used = rg_sheet
        df_rg = _read_indexed(rg_sheet_used).apply(pd.to_numeric, errors="coerce")
    else:
        # try Master-derived rg2
        if master_fallback_sheet in sheets:
            try:
                df_rg = _build_rg2_from_master()
                rg_sheet_used = f"{master_fallback_sheet}→Rg2(pivot)"
                print(f"[INFO] '{rg_sheet}' not found. Built Phase2 Rg grid from '{master_fallback_sheet}'.")
            except Exception as e:
                print(f"[WARN] Failed to build Rg2 from '{master_fallback_sheet}': {e}")
                df_rg = None

        # final fallback: rg_fallback
        if df_rg is None:
            if rg_fallback in sheets:
                rg_sheet_used = rg_fallback
                df_rg = _read_indexed(rg_sheet_used).apply(pd.to_numeric, errors="coerce")
                print(f"[WARN] Using '{rg_fallback}' as Phase2 Rg background. Axes may not match Phase2.")
            else:
                raise ValueError(f"Neither '{rg_sheet}' nor '{master_fallback_sheet}' nor '{rg_fallback}' is usable in: {xlsx}")

    # ---- auto-orient Rg grid to match Phase2 axes ----
    # Score both orientations by overlap counts
    idx_rg = df_rg.index.to_numpy(dtype=float)
    col_rg = df_rg.columns.to_numpy(dtype=float)
    score_same = len(set(idx_rg).intersection(set(y_il_ph))) + len(set(col_rg).intersection(set(x_pr_ph)))
    score_swap = len(set(idx_rg).intersection(set(x_pr_ph))) + len(set(col_rg).intersection(set(y_il_ph)))
    if score_swap > score_same:
        df_rg = df_rg.T
        idx_rg = df_rg.index.to_numpy(dtype=float)
        col_rg = df_rg.columns.to_numpy(dtype=float)
        print(f"[INFO] Transposed Rg grid '{rg_sheet_used}' to match Phase2 axes (y=IL, x=protein).")

    # ---- align by OUTER union so axes are stable; overlay only where phase is present ----
    y_all = np.array(sorted(set(df_ph.index.to_list()).union(set(df_rg.index.to_list()))), dtype=float)
    x_all = np.array(sorted(set(df_ph.columns.to_list()).union(set(df_rg.columns.to_list()))), dtype=float)

    df_ph2 = df_ph.reindex(index=y_all, columns=x_all)
    df_rg2 = df_rg.reindex(index=y_all, columns=x_all)

    # arrays for plotting
    x_protein = x_all
    y_il = y_all
    rg = df_rg2.to_numpy(dtype=float)
    phase = df_ph2.to_numpy(dtype=object)

    return x_protein, y_il, rg, phase, rg_sheet_used


# =============================================================================
# Helpers
# =============================================================================
def safe_right_mapping_bounds(x_vals: np.ndarray) -> Tuple[float, float]:
    finite = x_vals[np.isfinite(x_vals)]
    if finite.size == 0:
        return float(X_RIGHT_DATA_MIN), float(X_RIGHT_DATA_MAX)

    cand = finite[finite >= float(X_RIGHT_DATA_MIN)]
    if cand.size >= 2:
        return float(np.nanmin(cand)), float(np.nanmax(cand))
    if cand.size == 1:
        v = float(cand[0])
        return v, v
    v = float(np.nanmax(finite))
    return v, v


def map_to_span(x: np.ndarray, dmin: float, dmax: float, out_min: float, out_max: float) -> np.ndarray:
    x = np.asarray(x, float)
    denom = float(dmax - dmin)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.full_like(x, 0.5 * (float(out_min) + float(out_max)), dtype=float)
    return float(out_min) + (x - dmin) / denom * (float(out_max) - float(out_min))


def map_to_right(x: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    return map_to_span(x, dmin, dmax, float(X_RIGHT_MIN), float(X_RIGHT_MAX))


def add_broken_axis_slash(ax, where: str = "right") -> None:
    d = 0.012
    kw = dict(color="k", clip_on=False)
    if where == "right":
        ax.plot((1 - d, 1 + d), (-d, +d), transform=ax.transAxes, **kw)
    elif where == "left":
        ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, **kw)


def fill_gap(fig, ax_left, ax_right, color_rgba=(1, 1, 1, 1)) -> None:
    pos1 = ax_left.get_position()
    pos2 = ax_right.get_position()
    gx0, gx1 = pos1.x1, pos2.x0
    gy0, gy1 = pos1.y0, pos1.y1
    fig.add_artist(
        Rectangle(
            (gx0, gy0), gx1 - gx0, gy1 - gy0,
            transform=fig.transFigure,
            facecolor=color_rgba, edgecolor="none", zorder=0
        )
    )


def _apply_crystal_override(phase_grid: np.ndarray, crystal_grid: np.ndarray | None) -> np.ndarray:
    phase = np.array(phase_grid, dtype=object)
    if crystal_grid is not None and np.isfinite(crystal_grid).any():
        cm = (np.nan_to_num(crystal_grid, nan=0.0) >= 1.0)
        if cm.shape == phase.shape:
            phase = phase.copy()
            phase[cm] = "Crystalline/Ordered"
    return phase


def _ensure_ascending_axis(centers: np.ndarray, z: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure centers are strictly increasing. If decreasing, flip centers and z along the same axis.
    """
    c = np.asarray(centers, float)
    if c.size >= 2 and np.nanmedian(np.diff(c)) < 0:
        c = c[::-1]
        z = np.flip(z, axis=axis)
    return c, z


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    Convert cell centers to cell edges, supporting non-uniform spacing.
    centers: shape (n,)
    returns edges: shape (n+1,)
    """
    c = np.asarray(centers, float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if c.size == 1:
        # caller should override if they want a specific span
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)

    mid = 0.5 * (c[1:] + c[:-1])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(float)

# =============================================================================
# Phase2 tick helpers
# =============================================================================
def _phase2_tick_positions_indices(n: int, nticks: int) -> np.ndarray:
    """Evenly spaced tick positions for categorical axes (index units)."""
    n = int(max(n, 0))
    nticks = int(max(nticks, 2))
    if n <= 1:
        return np.array([0], dtype=int)
    idx = np.unique(np.round(np.linspace(0, n - 1, nticks)).astype(int))
    return idx

def _phase2_tick_positions_numeric(vals: np.ndarray, nticks: int) -> np.ndarray:
    """Evenly spaced tick values for numeric axes."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    nticks = int(max(nticks, 2))
    if v.size == 0:
        return np.array([], dtype=float)
    if v.size == 1:
        return np.array([float(v[0])], dtype=float)
    return np.linspace(float(np.nanmin(v)), float(np.nanmax(v)), nticks)




# =============================================================================
# NEW: Phase2 helpers & plots (Grid_Phase2)
# =============================================================================
def _encode_phases_for_mesh(phase: np.ndarray) -> Tuple[np.ma.MaskedArray, List[str]]:
    """
    Encode phase labels to integer codes for pcolormesh.
    Unknown/blank -> masked.
    """
    present: List[str] = []
    for p in PHASE_ORDER:
        if np.any(phase == p):
            present.append(p)

    code_map = {p: i for i, p in enumerate(present)}
    codes = np.full(phase.shape, fill_value=-1, dtype=int)
    for p, i in code_map.items():
        codes[phase == p] = i

    masked = np.ma.masked_where(codes < 0, codes)
    return masked, present


def plot_phase2_points(outpng: Path, x_vals: np.ndarray, y_vals: np.ndarray, phase: np.ndarray) -> None:
    """
    Categorical phase points (squares) on a square canvas.
    """
    # Ensure ascending axes (and keep phase aligned)
    x_vals, phase = _ensure_ascending_axis(x_vals, phase, axis=1)
    y_vals, phase = _ensure_ascending_axis(y_vals, phase, axis=0)

    nx = int(np.sum(np.isfinite(x_vals)))
    ny = int(np.sum(np.isfinite(y_vals)))
    if nx == 0 or ny == 0:
        raise ValueError("Grid_Phase2: empty x/y axes after numeric parsing.")

    x_vals = x_vals[:nx]
    y_vals = y_vals[:ny]
    phase = phase[:ny, :nx]

    if str(PHASE2_AXIS_MODE).lower() == "categorical":
        x_plot = np.arange(nx, dtype=float)
        y_plot = np.arange(ny, dtype=float)
        x_ticks = x_plot
        y_ticks = y_plot
        x_labels = [f"{v:g}" for v in x_vals]
        y_labels = [f"{v:g}" for v in y_vals]
    else:
        x_plot = x_vals
        y_plot = y_vals
        x_ticks = x_vals
        y_ticks = y_vals
        x_labels = [f"{v:g}" for v in x_vals]
        y_labels = [f"{v:g}" for v in y_vals]

    figsize = PHASE2_FIGSIZE if PHASE2_FIGSIZE is not None else (float(PHASE2_FIGSIZE_INCH), float(PHASE2_FIGSIZE_INCH))
    fig, ax = plt.subplots(figsize=figsize)

    present = set()
    for iy, y in enumerate(y_plot):
        for ix, x in enumerate(x_plot):
            p = normalize_phase_label(phase[iy, ix])
            if p not in PHASE_COLORS:
                continue
            present.add(p)
            ax.scatter([x], [y],
                       marker="s",
                       s=float(PHASE2_POINT_SIZE),
                       facecolors=[PHASE_COLORS[p]],
                       edgecolors="none",
                       zorder=5)

    # Optional gridlines to make cells visually clear (categorical mode)
    if str(PHASE2_AXIS_MODE).lower() == "categorical":
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_xticks(np.arange(-0.5, nx, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ny, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel(PHASE2_XLABEL)
    ax.set_ylabel(PHASE2_YLABEL)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=float(PHASE2_XTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, rotation=float(PHASE2_YTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))

    present_ordered = [p for p in PHASE_ORDER if p in present]
    make_top_legend(fig, present_ordered, fontsize=21, bbox_y=float(V2_LEGEND_BBOX_Y), ncol=V2_LEGEND_NCOL)

    fig.tight_layout(rect=[0, 0, 1, float(PHASE2_TIGHTLAYOUT_TOP)])
    fig.savefig(outpng, dpi=int(PHASE2_DPI), bbox_inches="tight")
    plt.close(fig)


def plot_phase2_block(outpng: Path, x_vals: np.ndarray, y_vals: np.ndarray, phase: np.ndarray) -> None:
    """
    Block phase heatmap with visible cell boundaries.
    """
    # Ensure ascending axes (and keep phase aligned)
    x_vals, phase = _ensure_ascending_axis(x_vals, phase, axis=1)
    y_vals, phase = _ensure_ascending_axis(y_vals, phase, axis=0)

    nx = int(np.sum(np.isfinite(x_vals)))
    ny = int(np.sum(np.isfinite(y_vals)))
    if nx == 0 or ny == 0:
        raise ValueError("Grid_Phase2: empty x/y axes after numeric parsing.")

    x_vals = x_vals[:nx]
    y_vals = y_vals[:ny]
    phase = phase[:ny, :nx]

    codes, present = _encode_phases_for_mesh(phase)

    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [PHASE_COLORS[p] for p in present]
    cmap = ListedColormap(colors)
    cmap.set_bad((1, 1, 1, 0))  # transparent for blanks
    norm = BoundaryNorm(np.arange(-0.5, len(present) + 0.5, 1), cmap.N)

    figsize = PHASE2_FIGSIZE if PHASE2_FIGSIZE is not None else (float(PHASE2_FIGSIZE_INCH), float(PHASE2_FIGSIZE_INCH))
    fig, ax = plt.subplots(figsize=figsize)

    if str(PHASE2_AXIS_MODE).lower() == "categorical":
        # Equal-size cells (no visual gaps)
        x_edges = np.arange(nx + 1) - 0.5
        y_edges = np.arange(ny + 1) - 0.5
        ax.pcolormesh(
            x_edges, y_edges, codes,
            cmap=cmap, norm=norm, shading="flat",
            edgecolors=(PHASE2_CELL_BORDER_COLOR if PHASE2_SHOW_CELL_BORDERS else "face"),
            linewidth=float(PHASE2_CELL_BORDER_LINEWIDTH if PHASE2_SHOW_CELL_BORDERS else 0.0),
            antialiased=bool(PHASE2_CELL_BORDER_ANTIALIASED if PHASE2_SHOW_CELL_BORDERS else True),
        )
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        ax.set_xticks(np.arange(nx))
        ax.set_xticklabels([f"{v:g}" for v in x_vals],
                           rotation=float(PHASE2_XTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))
        ax.set_yticks(np.arange(ny))
        ax.set_yticklabels([f"{v:g}" for v in y_vals],
                           rotation=float(PHASE2_YTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))
    else:
        # True numeric scaling
        x_edges = centers_to_edges(x_vals)
        y_edges = centers_to_edges(y_vals)
        ax.pcolormesh(
            x_edges, y_edges, codes,
            cmap=cmap, norm=norm, shading="flat",
            edgecolors=(PHASE2_CELL_BORDER_COLOR if PHASE2_SHOW_CELL_BORDERS else "face"),
            linewidth=float(PHASE2_CELL_BORDER_LINEWIDTH if PHASE2_SHOW_CELL_BORDERS else 0.0),
            antialiased=bool(PHASE2_CELL_BORDER_ANTIALIASED if PHASE2_SHOW_CELL_BORDERS else True),
        )

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{v:g}" for v in x_vals],
                           rotation=float(PHASE2_XTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))
        ax.set_yticks(y_vals)
        ax.set_yticklabels([f"{v:g}" for v in y_vals],
                           rotation=float(PHASE2_YTICK_ROT), fontsize=float(PHASE2_TICK_FONTSIZE))

    ax.set_xlabel(PHASE2_XLABEL)
    ax.set_ylabel(PHASE2_YLABEL)

    make_top_legend(fig, present, fontsize=14, bbox_y=0.98, ncol=4)

    fig.tight_layout()
    fig.savefig(outpng, dpi=int(PHASE2_DPI), bbox_inches="tight")
    plt.close(fig)




def _auto_vmin_vmax(z: np.ndarray) -> Tuple[float, float]:
    """Robust auto scale for Rg heatmap."""
    zz = np.asarray(z, float)
    finite = zz[np.isfinite(zz)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 2))
    vmax = float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
    return vmin, vmax


def plot_phase2_rg_block_plus_phase_points(
    outpng: Path,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    rg: np.ndarray,
    phase: np.ndarray,
    rg_sheet_used: str = "Grid_Rg2",
) -> None:
    """
    ONE-FIGURE Phase2 plot:
      - block heatmap from Grid_Rg2 (continuous colormap) with visible cell borders
      - phase squares (categorical colors) from Grid_Phase2 overlaid

    Requested adjustments in this version:
      - swap x/y axes (x uses original y; y uses original x) via PHASE2_SWAP_XY
      - linear (evenly spaced) major tick density via PHASE2_TICK_MODE / PHASE2_NTICKS_*
      - phase squares: half size + more transparent via PHASE2_PHASE_POINT_* configs
      - Arial font by default; label/legend/colorbar font size = 2× tick label size
    """
    # --- ensure x/y ascending; keep rg/phase aligned
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)

    if x_vals.size >= 2 and np.nanmedian(np.diff(x_vals)) < 0:
        x_vals = x_vals[::-1]
        rg = rg[:, ::-1]
        phase = phase[:, ::-1]
    if y_vals.size >= 2 and np.nanmedian(np.diff(y_vals)) < 0:
        y_vals = y_vals[::-1]
        rg = rg[::-1, :]
        phase = phase[::-1, :]

    nx = int(min(rg.shape[1], phase.shape[1], np.sum(np.isfinite(x_vals))))
    ny = int(min(rg.shape[0], phase.shape[0], np.sum(np.isfinite(y_vals))))
    if nx == 0 or ny == 0:
        raise ValueError("Phase2 combined plot: empty grid after parsing.")

    x_vals = x_vals[:nx]
    y_vals = y_vals[:ny]
    rg = rg[:ny, :nx]
    phase = phase[:ny, :nx]

    # -------------------------------------------------------------------------
    # Axis swap (requested): x <-> y for the Phase2 combined plot
    # -------------------------------------------------------------------------
    swap_xy = bool(PHASE2_SWAP_XY)
    if swap_xy:
        # Display axes: x = original y (IL wt%), y = original x (Protein mg/mL)
        x_disp = y_vals
        y_disp = x_vals
        rg_disp = rg.T
        phase_disp = phase.T
    else:
        x_disp = x_vals
        y_disp = y_vals
        rg_disp = rg
        phase_disp = phase

    nx_d = int(np.sum(np.isfinite(x_disp)))
    ny_d = int(np.sum(np.isfinite(y_disp)))
    x_disp = x_disp[:nx_d]
    y_disp = y_disp[:ny_d]
    rg_disp = rg_disp[:ny_d, :nx_d]
    phase_disp = phase_disp[:ny_d, :nx_d]

    # Font sizes
    tick_fs = int(PHASE2_TICK_FONTSIZE)
    label_fs = int(round(tick_fs * float(FONT_SCALE_FROM_TICKS)))

    # Phase overlay styling
    phase_point_s = float(PHASE2_POINT_SIZE) * float(PHASE2_PHASE_POINT_SIZE_SCALE)
    phase_point_alpha = float(PHASE2_PHASE_POINT_ALPHA)

    # Heatmap scale
    if PHASE2_RG_VMIN is None or PHASE2_RG_VMAX is None:
        vmin_auto, vmax_auto = _auto_vmin_vmax(rg_disp)
        vmin = float(vmin_auto if PHASE2_RG_VMIN is None else PHASE2_RG_VMIN)
        vmax = float(vmax_auto if PHASE2_RG_VMAX is None else PHASE2_RG_VMAX)
    else:
        vmin = float(PHASE2_RG_VMIN)
        vmax = float(PHASE2_RG_VMAX)

    # Border settings
    if PHASE2_SHOW_CELL_BORDERS:
        edgecolors = PHASE2_CELL_BORDER_COLOR
        linewidth = float(PHASE2_CELL_BORDER_LINEWIDTH)
        antialiased = bool(PHASE2_CELL_BORDER_ANTIALIASED)
    else:
        edgecolors = "face"
        linewidth = 0.0
        antialiased = True

    figsize = PHASE2_FIGSIZE if PHASE2_FIGSIZE is not None else (float(PHASE2_FIGSIZE_INCH), float(PHASE2_FIGSIZE_INCH))
    fig, ax = plt.subplots(figsize=figsize)

    axis_mode = str(PHASE2_AXIS_MODE).lower().strip()
    tick_mode = str(PHASE2_TICK_MODE).lower().strip()

    present = set()

    if axis_mode == "categorical":
        # Equal-size cells (no visual gaps)
        x_edges = np.arange(nx_d + 1) - 0.5
        y_edges = np.arange(ny_d + 1) - 0.5

        m = ax.pcolormesh(
            x_edges, y_edges, rg_disp,
            vmin=vmin, vmax=vmax,
            shading="flat", alpha=float(PHASE2_RG_ALPHA),
            edgecolors=edgecolors, linewidth=linewidth, antialiased=antialiased,
        )

        # Explicit numeric limits (important for linear tick locators)
        xlim_lo = float(x_edges[0]) if (PHASE2_XLIM[0] is None) else float(PHASE2_XLIM[0])
        xlim_hi = float(x_edges[-1]) if (PHASE2_XLIM[1] is None) else float(PHASE2_XLIM[1])
        ax.set_xlim(xlim_lo, xlim_hi)
        ylim_lo = float(y_edges[0]) if (PHASE2_YLIM[0] is None) else float(PHASE2_YLIM[0])
        ylim_hi = float(y_edges[-1]) if (PHASE2_YLIM[1] is None) else float(PHASE2_YLIM[1])
        ax.set_ylim(ylim_lo, ylim_hi)

        # Overlay phase squares at cell centers (index units)
        for iy in range(ny_d):
            for ix in range(nx_d):
                p = normalize_phase_label(phase_disp[iy, ix])
                if p not in PHASE_COLORS:
                    continue
                present.add(p)
                ax.scatter(
                    [ix], [iy],
                    marker="s",
                    s=phase_point_s,
                    facecolors=[PHASE_COLORS[p]],
                    edgecolors=PHASE2_PHASE_POINT_EDGE,
                    alpha=phase_point_alpha,
                    zorder=5,
                )

        ax.set_xlim(-0.5, nx_d - 0.5)
        ax.set_ylim(-0.5, ny_d - 0.5)

        # Linear (evenly spaced) tick density
        if tick_mode == "linear":
            xt_idx = _phase2_tick_positions_indices(nx_d, PHASE2_NTICKS_X)
            yt_idx = _phase2_tick_positions_indices(ny_d, PHASE2_NTICKS_Y)
        else:
            xt_idx = np.arange(nx_d, dtype=int)
            yt_idx = np.arange(ny_d, dtype=int)

        ax.set_xticks(xt_idx)
        ax.set_xticklabels([f"{x_disp[i]:g}" for i in xt_idx],
                           rotation=float(PHASE2_XTICK_ROT), fontsize=tick_fs)
        ax.set_yticks(yt_idx)
        ax.set_yticklabels([f"{y_disp[i]:g}" for i in yt_idx],
                           rotation=float(PHASE2_YTICK_ROT), fontsize=tick_fs)

        # Keep cells visually square
        ax.set_aspect("equal", adjustable="box")

    else:
        # True numeric scaling
        x_edges = centers_to_edges(x_disp)
        y_edges = centers_to_edges(y_disp)

        m = ax.pcolormesh(
            x_edges, y_edges, rg_disp,
            vmin=vmin, vmax=vmax,
            shading="flat", alpha=float(PHASE2_RG_ALPHA),
            edgecolors=edgecolors, linewidth=linewidth, antialiased=antialiased,
        )

        # Explicit numeric limits (important for linear tick locators)
        xlim_lo = float(x_edges[0]) if (PHASE2_XLIM[0] is None) else float(PHASE2_XLIM[0])
        xlim_hi = float(x_edges[-1]) if (PHASE2_XLIM[1] is None) else float(PHASE2_XLIM[1])
        ax.set_xlim(xlim_lo, xlim_hi)
        ylim_lo = float(y_edges[0]) if (PHASE2_YLIM[0] is None) else float(PHASE2_YLIM[0])
        ylim_hi = float(y_edges[-1]) if (PHASE2_YLIM[1] is None) else float(PHASE2_YLIM[1])
        ax.set_ylim(ylim_lo, ylim_hi)

        for iy, yy in enumerate(y_disp):
            for ix, xx in enumerate(x_disp):
                p = normalize_phase_label(phase_disp[iy, ix])
                if p not in PHASE_COLORS:
                    continue
                present.add(p)
                ax.scatter(
                    [xx], [yy],
                    marker="s",
                    s=phase_point_s,
                    facecolors=[PHASE_COLORS[p]],
                    edgecolors=PHASE2_PHASE_POINT_EDGE,
                    alpha=phase_point_alpha,
                    zorder=5,
                )

        # Linear ticks on numeric axes
        ax.set_xscale("linear")
        ax.set_yscale("linear")

        # --- ticks (numeric axes) ---
        # Priority (numeric axis_mode):
        #   1) explicit ticks (PHASE2_*TICKS_EXPLICIT)
        #   2) linear locator with NTICKS
        #   3) all grid values (dense)
        if (axis_mode != "categorical") and ((PHASE2_XTICKS_EXPLICIT is not None) or (PHASE2_YTICKS_EXPLICIT is not None)):
            if PHASE2_XTICKS_EXPLICIT is not None:
                xt = np.asarray(PHASE2_XTICKS_EXPLICIT, float)
                if bool(PHASE2_FILTER_EXPLICIT_TICKS_TO_RANGE):
                    xt = xt[(xt >= float(xlim_lo)) & (xt <= float(xlim_hi))]
                ax.set_xticks(xt)
                ax.set_xticklabels([f"{v:g}" for v in xt],
                                   rotation=float(PHASE2_XTICK_ROT), fontsize=tick_fs)
            if PHASE2_YTICKS_EXPLICIT is not None:
                yt = np.asarray(PHASE2_YTICKS_EXPLICIT, float)
                if bool(PHASE2_FILTER_EXPLICIT_TICKS_TO_RANGE):
                    yt = yt[(yt >= float(ylim_lo)) & (yt <= float(ylim_hi))]
                ax.set_yticks(yt)
                ax.set_yticklabels([f"{v:g}" for v in yt],
                                   rotation=float(PHASE2_YTICK_ROT), fontsize=tick_fs)
        elif tick_mode == "linear":
            # Evenly spaced tick positions across the numeric span (true linear scale)
            ax.xaxis.set_major_locator(LinearLocator(int(PHASE2_NTICKS_X)))
            ax.yaxis.set_major_locator(LinearLocator(int(PHASE2_NTICKS_Y)))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.tick_params(axis="x", labelrotation=float(PHASE2_XTICK_ROT), labelsize=tick_fs)
            ax.tick_params(axis="y", labelrotation=float(PHASE2_YTICK_ROT), labelsize=tick_fs)
        else:
            # Show all grid values (dense)
            ax.set_xticks(x_disp)
            ax.set_xticklabels([f"{v:g}" for v in x_disp],
                               rotation=float(PHASE2_XTICK_ROT), fontsize=tick_fs)
            ax.set_yticks(y_disp)
            ax.set_yticklabels([f"{v:g}" for v in y_disp],
                               rotation=float(PHASE2_YTICK_ROT), fontsize=tick_fs)
        ax.set_aspect("auto")

    # Labels (swap-aware) + linear scales
    xlabel = PHASE2_YLABEL if swap_xy else PHASE2_XLABEL
    ylabel = PHASE2_XLABEL if swap_xy else PHASE2_YLABEL
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    # Colorbar
    cb = fig.colorbar(m, ax=ax, pad=float(PHASE2_CBAR_PAD), shrink=float(PHASE2_CBAR_SHRINK), aspect=float(PHASE2_CBAR_ASPECT))
    cb.set_label(str(PHASE2_CBAR_LABEL), fontsize=label_fs)
    cb.ax.tick_params(labelsize=tick_fs)
    if PHASE2_CBAR_TICKS is not None:
        cb.set_ticks(PHASE2_CBAR_TICKS)
    elif PHASE2_RG_CBAR_TICKS is not None:
        cb.set_ticks(PHASE2_RG_CBAR_TICKS)

    # Tight layout with reserved top margin for the legend
    fig.tight_layout(rect=[0, 0, 1, float(PHASE2_TIGHTLAYOUT_TOP)])

    present_ordered = [p for p in PHASE_ORDER if p in present]
    make_top_legend(
        fig, present_ordered,
        fontsize=label_fs, markersize=8,
        bbox_y=float(PHASE2_LEGEND_BBOX_Y),
        ncol=PHASE2_LEGEND_NCOL,
    )

    fig.savefig(outpng, dpi=int(PHASE2_DPI), bbox_inches="tight")
    plt.close(fig)
# =============================================================================
# Heatmap 1 (points; 2-panel; continuous imshow)
# =============================================================================
def plot_heatmap1_points(outpng: Path,
                        x_vals: np.ndarray, y_vals: np.ndarray, rg: np.ndarray,
                        phase_grid: np.ndarray, crystal_grid: np.ndarray) -> None:

    phase = _apply_crystal_override(phase_grid, crystal_grid)
    rmin, rmax = safe_right_mapping_bounds(x_vals)

    xi = np.asarray(x_vals, float)
    idx_left = np.where(np.isfinite(xi) & (xi >= float(X_LEFT_MIN)) & (xi <= float(X_LEFT_MAX)))[0]
    if idx_left.size == 0:
        idx_left = np.where(np.isfinite(xi))[0]

    if RIGHT_FILL_WITH_LAST_COLUMN:
        finite_cols = np.where(np.isfinite(xi))[0]
        last = int(finite_cols[-1]) if finite_cols.size else 0
        idx_right = np.array([last], dtype=int)
    else:
        idx_right = np.where(np.isfinite(xi) & (xi >= float(rmin)) & (xi <= float(rmax)))[0]
        if idx_right.size == 0:
            finite_cols = np.where(np.isfinite(xi))[0]
            last = int(finite_cols[-1]) if finite_cols.size else 0
            idx_right = np.array([last], dtype=int)

    rg_left = rg[:, idx_left]
    rg_right = rg[:, idx_right]

    fig = plt.figure(figsize=FIGSIZE_COMMON)
    gs = fig.add_gridspec(1, 2, width_ratios=[BROKEN_WIDTH_LEFT, BROKEN_WIDTH_RIGHT], wspace=float(WSPACE))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(labelleft=False, left=False)
    ax2.yaxis.set_visible(False)

    im1 = ax1.imshow(
        rg_left, origin="lower",
        extent=(float(X_LEFT_MIN), float(X_LEFT_MAX), float(Y_MIN), float(Y_MAX)),
        aspect="auto", alpha=float(HEATMAP_ALPHA),
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        interpolation=str(HEATMAP_INTERPOLATION),
    )
    ax2.imshow(
        rg_right, origin="lower",
        extent=(float(X_RIGHT_MIN), float(X_RIGHT_MAX), float(Y_MIN), float(Y_MAX)),
        aspect="auto", alpha=float(HEATMAP_ALPHA),
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        interpolation=str(HEATMAP_INTERPOLATION),
    )

    cb = fig.colorbar(im1, ax=[ax1, ax2], shrink=1.0, aspect=30)
    cb.set_label(str(V2_CBAR_LABEL), fontsize=21)
    cb.set_ticks(RG_CBAR_TICKS)

    present = set()
    for iy, y in enumerate(y_vals):
        if not np.isfinite(y):
            continue
        for ix, x in enumerate(x_vals):
            if not np.isfinite(x):
                continue
            if iy >= phase.shape[0] or ix >= phase.shape[1]:
                continue
            p = normalize_phase_label(phase[iy, ix])
            if p not in PHASE_COLORS:
                continue
            present.add(p)
            if x <= float(X_LEFT_MAX):
                ax1.scatter([x], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")
            if x >= float(rmin):
                xr = map_to_right(np.array([x], float), rmin, rmax)[0]
                ax2.scatter([xr], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")

    ax1.set_xlim(float(X_LEFT_MIN), float(X_LEFT_MAX))
    ax2.set_xlim(float(X_RIGHT_MIN), float(X_RIGHT_MAX))
    ax1.set_ylim(float(Y_MIN), float(Y_MAX))

    ax1.set_xticks(X_LEFT_TICKS)
    ax2.set_xticks(X_RIGHT_TICKS)
    ax2.set_xticklabels([str(int(X_RIGHT_TICKS[0]))] if X_RIGHT_TICKS else [])
    ax1.set_yticks(Y_TICKS)

    ax1.set_xlabel("Molar percent of ILs (mol%)", fontsize=21)
    ax1.set_ylabel("Protein (mg/mL)", fontsize=21)

    add_broken_axis_slash(ax1, "right")

    # fill gap to match edge color
    cmap = im1.get_cmap()
    norm = im1.norm
    if GAP_FILL_MODE == "match_left_edge":
        edge_val = np.nanmedian(rg_left[:, -1]) if rg_left.size else np.nanmedian(rg)
    elif GAP_FILL_MODE == "match_right_edge":
        edge_val = np.nanmedian(rg_right[:, 0]) if rg_right.size else np.nanmedian(rg)
    else:
        edge_val = np.nan
    gap_color = cmap(norm(edge_val)) if np.isfinite(edge_val) else (1, 1, 1, 1)
    fill_gap(fig, ax1, ax2, gap_color)

    present_ordered = [p for p in PHASE_ORDER if p in present]
    make_top_legend(fig, present_ordered, fontsize=21, bbox_y=float(V2_LEGEND_BBOX_Y), ncol=V2_LEGEND_NCOL)

    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# v2 (3-panel broken axis) — continuous (imshow)
# =============================================================================
def plot_heatmap_v2_continuous(outpng: Path,
                               x_vals: np.ndarray, y_vals: np.ndarray, rg: np.ndarray,
                               phase_grid: np.ndarray, crystal_grid: np.ndarray) -> None:
    phase = _apply_crystal_override(phase_grid, crystal_grid)

    xi = np.asarray(x_vals, float)
    yi = np.asarray(y_vals, float)
    rmin, rmax = safe_right_mapping_bounds(x_vals)

    idx_L = np.where(np.isfinite(xi) & (xi >= float(X_V2_LEFT_MIN)) & (xi <= float(X_V2_LEFT_MAX)))[0]
    idx_M = np.where(np.isfinite(xi) & (xi >= float(X_V2_MID_MIN)) & (xi <= float(X_V2_MID_MAX)))[0]
    if idx_L.size == 0:
        idx_L = np.where(np.isfinite(xi))[0][:1]
    if idx_M.size == 0:
        idx_M = np.where(np.isfinite(xi) & (xi >= float(X_LEFT_MIN)) & (xi <= float(X_LEFT_MAX)))[0]
        if idx_M.size == 0:
            idx_M = np.where(np.isfinite(xi))[0]

    if RIGHT_FILL_WITH_LAST_COLUMN:
        finite_cols = np.where(np.isfinite(xi))[0]
        last = int(finite_cols[-1]) if finite_cols.size else 0
        idx_R = np.array([last], dtype=int)
    else:
        idx_R = np.where(np.isfinite(xi) & (xi >= float(rmin)) & (xi <= float(rmax)))[0]
        if idx_R.size == 0:
            finite_cols = np.where(np.isfinite(xi))[0]
            last = int(finite_cols[-1]) if finite_cols.size else 0
            idx_R = np.array([last], dtype=int)

    rgL = rg[:, idx_L]
    rgM = rg[:, idx_M]
    rgR = rg[:, idx_R]

    fig = plt.figure(figsize=FIGSIZE_COMMON)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[BROKEN_V2_WIDTH_LEFT, BROKEN_V2_WIDTH_MID, BROKEN_V2_WIDTH_RIGHT],
        wspace=float(WSPACE_V2),
    )
    axL = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[0, 1], sharey=axL)
    axR = fig.add_subplot(gs[0, 2], sharey=axL)

    axL.spines["right"].set_visible(False)
    axM.spines["left"].set_visible(False)
    axM.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)

    axL.tick_params(labelleft=True, left=True)
    axL.yaxis.set_visible(True)
    axL.set_ylabel("Protein (mg/mL)", fontsize=16)
    axL.set_yticks(Y_TICKS)

    axM.tick_params(labelleft=False, left=False)
    axM.yaxis.set_visible(False)
    axR.tick_params(labelleft=False, left=False)
    axR.yaxis.set_visible(False)

    imL = axL.imshow(
        rgL, origin="lower",
        extent=(float(X_V2_LEFT_MIN), float(X_V2_LEFT_MAX), float(Y_MIN), float(Y_MAX)),
        aspect="auto", alpha=float(HEATMAP_ALPHA),
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        interpolation=str(HEATMAP_INTERPOLATION),
    )
    imM = axM.imshow(
        rgM, origin="lower",
        extent=(float(X_V2_MID_MIN), float(X_V2_MID_MAX), float(Y_MIN), float(Y_MAX)),
        aspect="auto", alpha=float(HEATMAP_ALPHA),
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        interpolation=str(HEATMAP_INTERPOLATION),
    )
    axR.imshow(
        rgR, origin="lower",
        extent=(float(X_RIGHT_MIN), float(X_RIGHT_MAX), float(Y_MIN), float(Y_MAX)),
        aspect="auto", alpha=float(HEATMAP_ALPHA),
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        interpolation=str(HEATMAP_INTERPOLATION),
    )

    cb = fig.colorbar(imM, ax=[axL, axM, axR], shrink=1.0, aspect=30)
    cb.set_label(str(V2_CBAR_LABEL), fontsize=21)
    cb.set_ticks(RG_CBAR_TICKS)

    present = set()
    for iy_full, y in enumerate(yi):
        if not np.isfinite(y):
            continue
        for ix_full, x in enumerate(xi):
            if not np.isfinite(x):
                continue
            if iy_full >= phase.shape[0] or ix_full >= phase.shape[1]:
                continue
            p = normalize_phase_label(phase[iy_full, ix_full])
            if p not in PHASE_COLORS:
                continue
            present.add(p)

            if float(X_V2_LEFT_MIN) <= x <= float(X_V2_LEFT_MAX):
                axL.scatter([x], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")
            if float(X_V2_MID_MIN) <= x <= float(X_V2_MID_MAX):
                axM.scatter([x], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")
            if x >= float(rmin):
                xr = map_to_right(np.array([x], float), rmin, rmax)[0]
                axR.scatter([xr], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")

    axL.set_xlim(float(X_V2_LEFT_MIN), float(X_V2_LEFT_MAX))
    axM.set_xlim(float(X_V2_MID_MIN), float(X_V2_MID_MAX))
    axR.set_xlim(float(X_RIGHT_MIN), float(X_RIGHT_MAX))
    axM.set_ylim(float(Y_MIN), float(Y_MAX))

    xL_tick_vals = np.linspace(float(X_V2_LEFT_MIN), float(X_V2_LEFT_MAX), 5)
    axL.set_xticks(xL_tick_vals)
    axL.set_xticklabels([f"{v:.0f}" for v in xL_tick_vals])

    axM.set_xticks(X_V2_MID_TICKS)
    axR.set_xticks(X_RIGHT_TICKS)
    axR.set_xticklabels([str(int(X_RIGHT_TICKS[0]))] if X_RIGHT_TICKS else [])

    fig.text(0.5, 0.04, "Molar percent of ILs (mol%)",
             ha="center", va="center",
             fontsize=plt.rcParams.get("axes.labelsize", 21))

    add_broken_axis_slash(axL, "right")
    add_broken_axis_slash(axM, "right")

    cmap = imM.get_cmap()
    norm = imM.norm

    if GAP_FILL_MODE == "match_left_edge":
        edge_val_LM = np.nanmedian(rgL[:, -1]) if rgL.size else np.nanmedian(rg)
    elif GAP_FILL_MODE == "match_right_edge":
        edge_val_LM = np.nanmedian(rgM[:, 0]) if rgM.size else np.nanmedian(rg)
    else:
        edge_val_LM = np.nan
    gap_color_LM = cmap(norm(edge_val_LM)) if np.isfinite(edge_val_LM) else (1, 1, 1, 1)
    fill_gap(fig, axL, axM, gap_color_LM)

    if GAP_FILL_MODE == "match_left_edge":
        edge_val_MR = np.nanmedian(rgM[:, -1]) if rgM.size else np.nanmedian(rg)
    elif GAP_FILL_MODE == "match_right_edge":
        edge_val_MR = np.nanmedian(rgR[:, 0]) if rgR.size else np.nanmedian(rg)
    else:
        edge_val_MR = np.nan
    gap_color_MR = cmap(norm(edge_val_MR)) if np.isfinite(edge_val_MR) else (1, 1, 1, 1)
    fill_gap(fig, axM, axR, gap_color_MR)

    present_ordered = [p for p in PHASE_ORDER if p in present]
    make_top_legend(fig, present_ordered)

    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# v2 (3-panel broken axis) — block rendering with visible boundaries (pcolormesh)
# =============================================================================
def plot_heatmap_v2_block(outpng: Path,
                          x_vals: np.ndarray, y_vals: np.ndarray, rg: np.ndarray,
                          phase_grid: np.ndarray, crystal_grid: np.ndarray) -> None:
    """
    Block heatmap that preserves your grid cells and draws explicit borders between cells.

    Key implementation details:
    - pcolormesh with x/y EDGES (not centers), so each cell maps to its true size.
    - edgecolors + linewidth + antialiased=False to force boundaries to show in PNG output.
    """
    phase = _apply_crystal_override(phase_grid, crystal_grid)

    xi_full = np.asarray(x_vals, float)
    yi_full = np.asarray(y_vals, float)
    rmin, rmax = safe_right_mapping_bounds(x_vals)

    # Panel column indices (data-space)
    idx_L = np.where(np.isfinite(xi_full) & (xi_full >= float(X_V2_LEFT_MIN)) & (xi_full <= float(X_V2_LEFT_MAX)))[0]
    idx_M = np.where(np.isfinite(xi_full) & (xi_full >= float(X_V2_MID_MIN)) & (xi_full <= float(X_V2_MID_MAX)))[0]
    if idx_L.size == 0:
        idx_L = np.where(np.isfinite(xi_full))[0][:1]
    if idx_M.size == 0:
        idx_M = np.where(np.isfinite(xi_full) & (xi_full >= float(X_LEFT_MIN)) & (xi_full <= float(X_LEFT_MAX)))[0]
        if idx_M.size == 0:
            idx_M = np.where(np.isfinite(xi_full))[0]

    if RIGHT_FILL_WITH_LAST_COLUMN:
        finite_cols = np.where(np.isfinite(xi_full))[0]
        last = int(finite_cols[-1]) if finite_cols.size else 0
        idx_R = np.array([last], dtype=int)
    else:
        idx_R = np.where(np.isfinite(xi_full) & (xi_full >= float(rmin)) & (xi_full <= float(rmax)))[0]
        if idx_R.size == 0:
            finite_cols = np.where(np.isfinite(xi_full))[0]
            last = int(finite_cols[-1]) if finite_cols.size else 0
            idx_R = np.array([last], dtype=int)

    # Data for each panel
    xL = xi_full[idx_L]
    xM = xi_full[idx_M]
    yC = yi_full.copy()
    zL = rg[:, idx_L]
    zM = rg[:, idx_M]
    zR = rg[:, idx_R]

    # Ensure y is ascending to match pcolormesh expectation
    yC, zL = _ensure_ascending_axis(yC, zL, axis=0)
    _,  zM = _ensure_ascending_axis(yi_full.copy(), zM, axis=0)  # reuse check but don't reassign y twice
    _,  zR = _ensure_ascending_axis(yi_full.copy(), zR, axis=0)

    # Ensure x is ascending per panel
    xL, zL = _ensure_ascending_axis(xL, zL, axis=1)
    xM, zM = _ensure_ascending_axis(xM, zM, axis=1)

    # Convert centers to edges
    y_edges = centers_to_edges(yC)

    xL_edges = centers_to_edges(xL)
    xM_edges = centers_to_edges(xM)

    # Right panel: x is DISPLAY-space (compressed). Create edges accordingly.
    if RIGHT_FILL_WITH_LAST_COLUMN:
        xR_edges = np.array([float(X_RIGHT_MIN), float(X_RIGHT_MAX)], dtype=float)
    else:
        xR_cent = map_to_right(xi_full[idx_R], rmin, rmax)
        xR_cent, zR = _ensure_ascending_axis(xR_cent, zR, axis=1)
        xR_edges = centers_to_edges(xR_cent)
        # Ensure the right panel fully covers the set display span
        xR_edges[0] = min(float(X_RIGHT_MIN), float(xR_edges[0]))
        xR_edges[-1] = max(float(X_RIGHT_MAX), float(xR_edges[-1]))

    # pcolormesh border settings
    if SHOW_CELL_BORDERS:
        edgecolors = CELL_BORDER_COLOR
        linewidth = float(CELL_BORDER_LINEWIDTH)
        antialiased = bool(CELL_BORDER_ANTIALIASED)
    else:
        edgecolors = "face"
        linewidth = 0.0
        antialiased = True

    fig = plt.figure(figsize=FIGSIZE_COMMON)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[BROKEN_V2_WIDTH_LEFT, BROKEN_V2_WIDTH_MID, BROKEN_V2_WIDTH_RIGHT],
        wspace=float(WSPACE_V2),
    )
    axL = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[0, 1], sharey=axL)
    axR = fig.add_subplot(gs[0, 2], sharey=axL)

    # spines
    axL.spines["right"].set_visible(False)
    axM.spines["left"].set_visible(False)
    axM.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)

    axL.tick_params(labelleft=True, left=True)
    axL.yaxis.set_visible(True)
    axL.set_ylabel(V2_YLABEL, fontsize=21)
    axL.set_yticks(Y_TICKS)

    axM.tick_params(labelleft=False, left=False)
    axM.yaxis.set_visible(False)
    axR.tick_params(labelleft=False, left=False)
    axR.yaxis.set_visible(False)

    # block heatmaps
    mL = axL.pcolormesh(
        xL_edges, y_edges, zL,
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        shading="flat", alpha=float(HEATMAP_ALPHA),
        edgecolors=edgecolors, linewidth=linewidth, antialiased=antialiased,
    )
    mM = axM.pcolormesh(
        xM_edges, y_edges, zM,
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        shading="flat", alpha=float(HEATMAP_ALPHA),
        edgecolors=edgecolors, linewidth=linewidth, antialiased=antialiased,
    )
    mR = axR.pcolormesh(
        xR_edges, y_edges, zR,
        vmin=float(RG_VMIN), vmax=float(RG_VMAX),
        shading="flat", alpha=float(HEATMAP_ALPHA),
        edgecolors=edgecolors, linewidth=linewidth, antialiased=antialiased,
    )

    # Colorbar (tie to mid)
    cb = fig.colorbar(mM, ax=[axL, axM, axR], shrink=1.0, aspect=30)
    cb.set_label(str(V2_CBAR_LABEL), fontsize=21)
    cb.set_ticks(RG_CBAR_TICKS)

    # Phase scatter overlays
    present = set()
    for iy_full, y in enumerate(yi_full):
        if not np.isfinite(y):
            continue
        for ix_full, x in enumerate(xi_full):
            if not np.isfinite(x):
                continue
            if iy_full >= phase.shape[0] or ix_full >= phase.shape[1]:
                continue
            p = normalize_phase_label(phase[iy_full, ix_full])
            if p not in PHASE_COLORS:
                continue
            present.add(p)

            if float(X_V2_LEFT_MIN) <= x <= float(X_V2_LEFT_MAX):
                axL.scatter([x], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")
            if float(X_V2_MID_MIN) <= x <= float(X_V2_MID_MAX):
                axM.scatter([x], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")
            if x >= float(rmin):
                xr = map_to_right(np.array([x], float), rmin, rmax)[0]
                axR.scatter([xr], [y], marker="s", s=float(POINT_SIZE),
                            facecolors=[PHASE_COLORS[p]], edgecolors="none")

    # limits/ticks
    axL.set_xlim(float(X_V2_LEFT_MIN), float(X_V2_LEFT_MAX))
    axM.set_xlim(float(X_V2_MID_MIN), float(X_V2_MID_MAX))
    axR.set_xlim(float(X_RIGHT_MIN), float(X_RIGHT_MAX))
    axM.set_ylim(float(Y_MIN), float(Y_MAX))

    xL_tick_vals = np.linspace(float(X_V2_LEFT_MIN), float(X_V2_LEFT_MAX), 5)
    axL.set_xticks(xL_tick_vals)
    axL.set_xticklabels([f"{v:.0f}" for v in xL_tick_vals])

    axM.set_xticks(X_V2_MID_TICKS)
    axR.set_xticks(X_RIGHT_TICKS)
    axR.set_xticklabels([str(int(X_RIGHT_TICKS[0]))] if X_RIGHT_TICKS else [])

    fig.text(0.5, 0.04, str(V2_XLABEL),
             ha="center", va="center",
             fontsize=21)

    # broken slashes + gap fill
    add_broken_axis_slash(axL, "right")
    add_broken_axis_slash(axM, "right")

    # For block mode, a neutral gap is usually clearest.
    if GAP_FILL_MODE == "white":
        fill_gap(fig, axL, axM, (1, 1, 1, 1))
        fill_gap(fig, axM, axR, (1, 1, 1, 1))
    else:
        # match edges (use median of bordering columns)
        def _safe_median(a):
            return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan

        cmap = mM.cmap
        norm = mM.norm

        if GAP_FILL_MODE == "match_left_edge":
            vLM = _safe_median(zL[:, -1]) if zL.size else np.nan
            vMR = _safe_median(zM[:, -1]) if zM.size else np.nan
        else:  # match_right_edge
            vLM = _safe_median(zM[:, 0]) if zM.size else np.nan
            vMR = _safe_median(zR[:, 0]) if zR.size else np.nan

        fill_gap(fig, axL, axM, cmap(norm(vLM)) if np.isfinite(vLM) else (1, 1, 1, 1))
        fill_gap(fig, axM, axR, cmap(norm(vMR)) if np.isfinite(vMR) else (1, 1, 1, 1))

    present_ordered = [p for p in PHASE_ORDER if p in present]
    make_top_legend(fig, present_ordered, fontsize=21, bbox_y=float(V2_LEGEND_BBOX_Y), ncol=V2_LEGEND_NCOL)

    fig.savefig(outpng, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    analysis_dir = Path(ANALYSIS_DIR)
    diagram_dir = analysis_dir / DIAGRAM_SUBDIR
    diagram_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = diagram_dir / XLSX_NAME
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    x_rg, y_rg, rg = read_grid_rg(xlsx_path)
    _, _, ph = read_grid_phase(xlsx_path)
    _, _, cr = read_grid_crystal(xlsx_path)

    if MAKE_HEATMAP1_CONTINUOUS:
        plot_heatmap1_points(diagram_dir / OUT_HEATMAP1, x_rg, y_rg, rg, ph, cr)
        print(f"[OK] Wrote: {diagram_dir / OUT_HEATMAP1}")

    if MAKE_HEATMAP_V2_CONTINUOUS:
        plot_heatmap_v2_continuous(diagram_dir / OUT_HEATMAP_V2, x_rg, y_rg, rg, ph, cr)
        print(f"[OK] Wrote: {diagram_dir / OUT_HEATMAP_V2}")

    if MAKE_HEATMAP_V2_BLOCK:
        plot_heatmap_v2_block(diagram_dir / OUT_HEATMAP_V2_BLOCK, x_rg, y_rg, rg, ph, cr)
        print(f"[OK] Wrote: {diagram_dir / OUT_HEATMAP_V2_BLOCK}")

    # --- NEW: Phase2 combined plot (Grid_Rg2 block + Grid_Phase2 points) ---
    x_p2, y_p2, rg2, ph2, rg2_sheet_used = read_phase2_rg2_and_phase2(
        xlsx_path,
        rg_sheet=PHASE2_RG_SHEET,
        phase_sheet=PHASE2_PHASE_SHEET,
        rg_fallback=PHASE2_RG_SHEET_FALLBACK,
    )

    if MAKE_PHASE2_COMBINED:
        plot_phase2_rg_block_plus_phase_points(
            diagram_dir / OUT_PHASE2_COMBINED,
            x_p2, y_p2, rg2, ph2,
            rg_sheet_used=rg2_sheet_used,
        )
        print(f"[OK] Wrote: {diagram_dir / OUT_PHASE2_COMBINED}")

    # Optional Phase2-only plots (diagnostics)
    if MAKE_PHASE2_POINTS:
        plot_phase2_points(diagram_dir / OUT_PHASE2_POINTS, x_p2, y_p2, ph2)
        print(f"[OK] Wrote: {diagram_dir / OUT_PHASE2_POINTS}")

    if MAKE_PHASE2_BLOCK:
        plot_phase2_block(diagram_dir / OUT_PHASE2_BLOCK, x_p2, y_p2, ph2)
        print(f"[OK] Wrote: {diagram_dir / OUT_PHASE2_BLOCK}")



if __name__ == "__main__":
    main()
