# -*- coding: utf-8 -*-
"""
Step3_FFmaker_Oligo_v3.py
- FFMAKER: 依据 PDB 列表生成 ff2.dat（若已存在且非空则跳过）
- 规范化 .dat 实验数据到 analysis/normalized（统一三列: s I err；容错两列）
- OLIGOMER: 针对规范化数据拟合；解析日志，写出 analysis/oligomer_summary.xlsx（all/selected[OK+Borderline]）
- ML 目标表生成与合并：
    * 自动读取 analysis/Guinier_stats.txt 与 analysis/useful_parameters.csv
    * 若 ML_targets_crystal.xlsx 不存在，则据此生成
    * 再把 OLIGOMER 的 filtered 表合并进去，输出 ML_targets_crystal_oligo.xlsx
兼容 Windows + Spyder，路径参数集中于前部。

参考与兼容实现：
- 现有 Step3 的 ff2.dat 生成/规范化/OLIGOMER/日志解析与合并流程（保持列名与结构）  [CITE]  # see filecite below
- 旧 crystal-aware 管线中的 tag/文件名规范化、Guinier 列容错解析策略            [CITE]
"""

import subprocess, shlex, re, sys, shutil, math
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, List

# ========== 用户可配置 ==========
#ATSAS_BIN = Path(r"D:\Program Files\ATSAS-3.2.1\bin")
ATSAS_BIN = Path(r"C:\Program Files\ATSAS-4.1.3\bin")
FFMAKER_EXE = ATSAS_BIN / "FFMAKER.exe"
OLIGOMER_EXE = ATSAS_BIN / "oligomer.exe"

BASE_DIR = Path(r"C:\Users\E100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML")
PDB_DIR  = Path(r"C:\Users\E100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML\0PDB")
DATA_DIR = BASE_DIR / "3sol"

ANALYSIS_DIR = DATA_DIR / "analysis3"
FIT_DIR      = ANALYSIS_DIR / "fit"

S_MIN = 0.015
S_MAX = 0.30
NS    = 201
LM    = 20
FF_OUT_NAME = "ff2.dat"

# PDB 顺序（列顺序与标题参考）
PDB_ORDER = ["dimer","hexamer","monomer","octamer","tetramer"]

# OLIGOMER 参数
UNITS = 1
USE_CONST = True
OLIGO_SMIN = 0.01
OLIGO_SMAX = 0.30
SINGLE_DAT = None      # e.g. "EAN1.5_7.5mg.dat"；留空=批量
LOG_NAME   = "oligomer_lysozyme.log"

# ---- Fit-quality thresholds (requested) ----
# Keep: OK (<200) and Borderline (200–1000)
# Drop: Bad (>1000) and Fail (constant-only)
CHI2_OK_MAX = 200.0
CHI2_BORDERLINE_MAX = 1000.0

# Detect "constant-only" failures from OLIGOMER volume fractions
# (constant fraction ~1.0 and all other components ~0)
CONST_ONLY_CONST_MIN = 0.99
CONST_ONLY_OTHER_MAX = 1e-6

# ---- Plot settings ----
PLOT_NCOLS = 6
PLOT_NROWS = 9               # 9x9 = 81 panels per page (matches your examples)
PLOT_DPI   = 300

# Experimental curve style
PLOT_EXP_AS_POINTS_LOGX = True          # Requirement (logx): grey curve as points (not a line)
PLOT_EXP_MARKER = "."
PLOT_EXP_MARKERSIZE = 3.0

# Fit curve style
PLOT_LINEWIDTH = 3                   # base width
PLOT_FIT_LINEWIDTH_MULT = 2.0          # Requirement: fit line thicker (x2)

PLOT_ANNOT_FONTSIZE = 12
PLOT_TIGHT = True

# Log-x ticks/labels: make 10^-1 display as 0.1, and mark 0.01 and 0.2
LOGX_TICKS = [0.01, 0.1, 0.2]
LOGX_TICKLABELS = ["0.01", "0.1", "0.2"]

# Composition annotation from ML_targets_crystal_oligo.xlsx (values in 0-1, shown as %)
ML_TARGETS_OLIGO_XLSX = None           # None -> try ANALYSIS_DIR/ML_targets_crystal_oligo.xlsx
SHOW_OLIGO_COMPOSITION_LOGX = True
SHOW_OLIGO_COMPOSITION_V2   = True  # Also show composition on *_noannot_v2 (linear-x)
OLIGO_COMP_DISPLAY_ORDER = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
OLIGO_COMP_TEXT_POS = (0.02, 0.04)     # axes fraction (left-bottom)
OLIGO_COMP_FONTSIZE = 9

# ---- Composition table settings (fits_composition figures) ----
COMP_TABLE_FONTSIZE = 10           # 组分表格字体大小（相应增大）
COMP_TABLE_ROW_HEIGHT = 1.5 * 1.5  # 组分表格行间距倍数（增加1.5倍：1.5 * 1.5 = 2.25）
COMP_TABLE_BBOX = [0.02, 0.02, 0.6, 0.5]  # 左下角位置和大小（宽高都增加1.5倍）

# ---- Composition text settings (fits_composition_v2) ----
COMP_TEXT_FONTSIZE = 9             # 组分文字显示字体大小
COMP_TEXT_LINEHEIGHT = 1.3         # 行间距倍数

# ---- Bar graph color mapping (v2) ----
COMP_COLORS_V2 = {
    "monomer": "#749D65",
    "dimer": "#9366BC",
    "tetramer": "#5989B5",
    "hexamer": "#E37F48",
    "octamer": "#D4B833",
}

# ---- composition_bargraph_v3 全局参数（可在此调整） ----
# 是否启用 composition_bargraph_v3 出图
COMP_BAR_V3_ENABLED = False
# 指定要展示的样品顺序（按 'file' 列子串匹配；"Buffer" 会匹配包含 "buf" 的样品）
COMP_BAR_V3_TARGET_SAMPLES = [
    "Buffer", "5min", "7min", "9min", "11min",
    "13min", "15min", "17min", "20min", "22min"
]
# 单柱宽度（0-1，越接近1，柱间距越小，更接近期刊风格）
COMP_BAR_V3_BAR_WIDTH = 0.9
# 图宽度控制：按样品数量线性放大，并设下限
COMP_BAR_V3_FIG_WIDTH_PER_SAMPLE = 1.0
COMP_BAR_V3_FIG_WIDTH_MIN = 8

# 其他可调设置
COMP_BAR_V3_YLIM = (71, 110)                # y轴范围
COMP_BAR_V3_YLABEL = "Composition (%)"      # y轴标题文本
COMP_BAR_V3_YLABEL_FONTSIZE = 24            # y轴标题字号
COMP_BAR_V3_YLABEL_WEIGHT = "normal"        # y轴标题是否加粗：'normal' / 'bold'
COMP_BAR_V3_XTICK_LABELSIZE = 20            # x轴刻度文字大小
COMP_BAR_V3_YTICK_LABELSIZE = 20            # y轴刻度文字大小
COMP_BAR_V3_XTICK_ROTATION = 30             # x轴刻度文字旋转角度
COMP_BAR_V3_LEGEND_FONTSIZE = 24            # 图例字体大小
COMP_BAR_V3_TICK_LENGTH = 4                 # 刻度线长度
COMP_BAR_V3_TICK_WIDTH = 1.0                # 刻度线线宽

# ---------- 工具 ----------
def run(cmd: List[str], cwd: Optional[Path] = None):
    print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                         capture_output=True, text=True, shell=False)
    if res.stdout:
        print(res.stdout.rstrip())
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr.rstrip())
        raise RuntimeError(f"Command failed (rc={res.returncode}).")
    return res

def _basename_without_ext(x):
    return Path(x).name.rsplit(".", 1)[0]

def _normkey_filename(v: str) -> str:
    """用于不同表之间的文件名匹配：小写、去扩展名。
    
    Only removes known data file extensions (.dat, .fit, .txt), 
    not intermediate dots in names like "5.5min".
    """
    stem = Path(str(v)).name.strip()
    # Only remove known data file extensions, not intermediate dots
    for ext in ['.dat', '.fit', '.txt']:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem.lower()


def _natural_sort_key(filename_str: str):
    """
    智能排序函数，能处理混合数字和字母的文件名。
    规则：
      1. Buffer 放在最前（单独处理）
      2. 其余按数字顺序排序（支持小数点，如 1, 2.5, 10.5, 102 等）
      3. 同一数字后的字母也会排序
    
    返回一个元组，用于 sorted() 或 sort_values()
    """
    s = str(filename_str).strip().lower()
    
    # Buffer 优先
    if "buf" in s:
        return (0, "")
    
    # 提取所有数字段和字母段
    parts = []
    import re as regex_module
    tokens = regex_module.findall(r'(\d+\.?\d*|[a-z]+|[^a-z0-9]+)', s)
    
    for token in tokens:
        if regex_module.match(r'^\d+\.?\d*$', token):
            # 数字部分：转为浮点数
            parts.append((0, float(token)))
        else:
            # 字母或其他部分：保持字符串
            parts.append((1, token))
    
    return (1, tuple(parts))


def _load_oligo_composition_map(analysis_dir: Path) -> dict:
    """Load ML_targets_crystal_oligo.xlsx and build a composition lookup.

    Excel values are expected in 0-1 fractions (e.g., monomer=0.80). This
    function returns a mapping for fast lookup during plotting.

    Returns:
      dict[key] -> {"monomer": float, "dimer": float, ...}
    """
    # Resolve the workbook path
    xlsx = None
    if ML_TARGETS_OLIGO_XLSX is not None:
        try:
            xlsx = Path(ML_TARGETS_OLIGO_XLSX)
        except Exception:
            xlsx = None
    if xlsx is None:
        xlsx = analysis_dir / "ML_targets_crystal_oligo.xlsx"

    # Fallbacks: script directory and current working directory
    if not xlsx.exists():
        try:
            here = Path(__file__).resolve().parent
            cand = here / "ML_targets_crystal_oligo.xlsx"
            if cand.exists():
                xlsx = cand
        except Exception:
            pass
    if not xlsx.exists():
        cand = Path.cwd() / "ML_targets_crystal_oligo.xlsx"
        if cand.exists():
            xlsx = cand

    if not xlsx.exists():
        return {}

    print(f"INFO: composition table: {xlsx}")

    try:
        # Load first sheet by default
        df = pd.read_excel(xlsx, sheet_name=0)
    except Exception as e:
        print(f"WARNING: failed to read ML_targets_crystal_oligo.xlsx: {e}")
        return {}

    # Identify key column
    key_col = None
    for c in ["file", "File", "filename", "Filename", "name", "Name", "tag", "Tag"]:
        if c in df.columns:
            key_col = c
            break
    if key_col is None:
        return {}

    # Ensure component columns exist
    comp_cols = [c for c in OLIGO_COMP_DISPLAY_ORDER if c in df.columns]
    if not comp_cols:
        return {}

    def _rec_score(_rec: dict) -> tuple:
        # Prefer rows that actually have composition numbers (avoid NaN overwriting).
        nn = 0
        nz = 0
        for _c in comp_cols:
            _v = _rec.get(_c, np.nan)
            try:
                _v = float(_v)
            except Exception:
                _v = np.nan
            if np.isfinite(_v):
                nn += 1
                if _v > 0:
                    nz += 1
        return (nn, nz)

    out = {}
    for _, r in df.iterrows():
        k = _normkey_filename(r.get(key_col, ""))
        if not k:
            continue
        rec = {}
        for c in comp_cols:
            try:
                v = float(r.get(c))
            except Exception:
                v = np.nan
            rec[c] = v
        prev = out.get(k)
        if (prev is None) or (_rec_score(rec) > _rec_score(prev)):
            out[k] = rec
    return out


def _format_oligo_composition_lines(comp: dict) -> str:
    """Format composition dict (0-1 fractions) into multiline 'nameXX%' lines."""
    lines = []
    for name in OLIGO_COMP_DISPLAY_ORDER:
        v = comp.get(name, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        if not np.isfinite(v) or v <= 0:
            continue
        pct = v * 100.0
        # Prefer integer percentages when close
        if abs(pct - round(pct)) < 0.05:
            pct_s = f"{int(round(pct))}"
        else:
            pct_s = f"{pct:.1f}"
        lines.append(f"{name}{pct_s}%")
    return "\n".join(lines)


def _write_subplot_order_txt(out_path: Path, page_files: List[Path], nrows: int, ncols: int):
    """Write subplot order mapping as requested (row-wise, left-to-right)."""
    with out_path.open("w", encoding="utf-8") as w:
        for r in range(nrows):
            start = r * ncols
            end = min((r + 1) * ncols, nrows * ncols)
            names = []
            for i in range(start, end):
                if i < len(page_files):
                    names.append(page_files[i].stem)
                else:
                    names.append("")
            w.write(f"row{r+1}: " + "\t".join(names) + "\n")


# ---------- PDB & FFMAKER ----------
def _list_pdbs(folder: Path):
    return sorted([p for p in folder.glob("*.pdb") if p.is_file()])

def _resolve_pdb_order(pdb_dir: Path, order_hints):
    files = _list_pdbs(pdb_dir)
    if not files:
        raise FileNotFoundError(f"No PDB files found in: {pdb_dir}")
    by_lower = {p.name.lower(): p for p in files}
    resolved = []
    for hint in order_hints:
        if hint.lower().endswith(".pdb") and hint.lower() in by_lower:
            resolved.append(by_lower[hint.lower()]); continue
        exact = by_lower.get(f"{hint.lower()}.pdb")
        if exact:
            resolved.append(exact); continue
        # 模糊包含
        hit = next((p for p in files if hint.lower() in p.name.lower()), None)
        if not hit:
            raise FileNotFoundError(
                f"Cannot find PDB for hint '{hint}' in {pdb_dir}. "
                f"Available: {', '.join(p.name for p in files)}"
            )
        resolved.append(hit)
    return resolved

def make_ffdat(pdb_dir: Path):
    print("\n=== FFMAKER: build form-factor in PDB_DIR ===")
    pdb_files = _resolve_pdb_order(pdb_dir, PDB_ORDER)
    for p in pdb_files: print("PDB:", p.name)

    ffdat = pdb_dir / FF_OUT_NAME
    if ffdat.exists(): ffdat.unlink()

    cmd = [str(FFMAKER_EXE),
           f"--smin={S_MIN}", f"--smax={S_MAX}",
           f"--ns={NS}", f"--lm={LM}",
           "-o", str(ffdat)]
    cmd += [str(p.resolve()) for p in pdb_files]
    run(cmd)

    if not ffdat.exists() or ffdat.stat().st_size == 0:
        raise RuntimeError("FFMAKER finished but ff2.dat not created.")

    # 保存列映射供核对
    map_txt = pdb_dir / "ff_columns_map.txt"
    with map_txt.open("w", encoding="utf-8") as w:
        w.write("FF columns map (col1 = s)\n")
        for i, nm in enumerate(pdb_files, start=2):
            w.write(f"col {i} -> {nm.name}\n")
    print(f"wrote: {map_txt}")
    return ffdat, [p.name for p in pdb_files]

def ensure_ffdat(pdb_dir: Path):
    ffdat = pdb_dir / FF_OUT_NAME
    if ffdat.exists() and ffdat.stat().st_size > 0:
        print(f"ff2.dat found, skipping FFMAKER: {ffdat}")
        pdb_files = _resolve_pdb_order(pdb_dir, PDB_ORDER)
        return ffdat, [p.name for p in pdb_files]
    return make_ffdat(pdb_dir)


# ---------- 规范化 .dat ----------
import math, re

def _coerce_float_fields(parts):
    out = []
    for tok in parts:
        try:
            out.append(float(tok))
        except Exception:
            return None
    return out

def normalize_dat_files(data_dir: Path, out_dir: Path, smin: float, smax: float):
    """
    容错读取 data_dir/*.dat，统一三列 s I err；裁剪到 [smin,smax]，按 s 去重排序。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_paths = []
    src_files = sorted([p for p in data_dir.glob("*.dat") if p.is_file()])
    if not src_files:
        return norm_paths

    for src in src_files:
        rows = []
        with src.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith(("#",";","//")):
                    continue
                parts = re.split(r"[,\s;]+", ln)
                nums = _coerce_float_fields(parts)
                if not nums:
                    continue
                if len(nums) >= 3:
                    s, I, err = nums[0], nums[1], nums[2]
                elif len(nums) >= 2:
                    s, I = nums[0], nums[1]
                    err = max(abs(0.01 * I), 1e-6)  # 合成一个小相对误差
                else:
                    continue
                if not (math.isfinite(s) and math.isfinite(I) and math.isfinite(err)):
                    continue
                if not (smin <= s <= smax):
                    continue
                rows.append((s, I, err))

        if not rows:
            print(f"WARNING: no usable points in {src.name} within [{smin}, {smax}]")
            continue

        rows.sort(key=lambda t: t[0])
        uniq = []
        last_s = None
        for t in rows:
            if last_s is None or abs(t[0] - last_s) > 1e-12:
                uniq.append(t); last_s = t[0]

        dst = out_dir / src.name
        with dst.open("w", encoding="utf-8") as w:
            for s, I, err in uniq:
                w.write(f"{s:.8g} {I:.8g} {err:.8g}\n")
        norm_paths.append(dst)
        print(f"normalized: {src.name} -> {dst.name}  (n={len(uniq)}, s=[{uniq[0][0]:.4g},{uniq[-1][0]:.4g}])")

    return norm_paths


# ---------- 运行 OLIGOMER ----------
def run_oligomer(ffdat_path: Path, data_dir: Path):
    """
    规范化 -> 对每个样品调用 OLIGOMER -> 收集 .fit -> 输出统一日志。
    （此版本写单一日志；如需逐样品日志，可扩展为 per-file log 变种）
    """
    print("\n=== OLIGOMER: normalize & fit experimental curves ===")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    norm_dir = ANALYSIS_DIR / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    norm_files = normalize_dat_files(data_dir, norm_dir, OLIGO_SMIN, OLIGO_SMAX)
    if SINGLE_DAT:
        wanted = norm_dir / SINGLE_DAT
        if wanted.exists():
            norm_files = [wanted]
        else:
            print(f"WARNING: normalized SINGLE_DAT not found ({wanted}); continue with batch.")

    if not norm_files:
        raise RuntimeError(f"No normalized .dat files to fit in: {norm_dir}")

    log_path_for_oligomer = ANALYSIS_DIR / LOG_NAME
    fits = []

    for nf in norm_files:
        cmd = [str(OLIGOMER_EXE),
               "-ff", str(ffdat_path.resolve()),
               "-un", str(UNITS),
               "-smin", str(OLIGO_SMIN),
               "-smax", str(OLIGO_SMAX),
               "-out", str(log_path_for_oligomer.resolve()),
               str(nf.name)]
        if USE_CONST:
            cmd.insert(-1, "-cst")
        run(cmd, cwd=norm_dir)

        fit1 = norm_dir / f"{nf.name}.fit"
        fit2 = norm_dir / f"{nf.stem}.fit"
        if fit1.exists():
            dest = FIT_DIR / fit1.name
            if dest.exists(): dest.unlink()
            shutil.move(str(fit1), str(dest))
            fits.append(dest)
        elif fit2.exists():
            dest = FIT_DIR / fit2.name
            if dest.exists(): dest.unlink()
            shutil.move(str(fit2), str(dest))
            fits.append(dest)
        else:
            print(f"WARNING: no .fit found for {nf.name}")

    print(f"done. log: {log_path_for_oligomer}")
    return fits, log_path_for_oligomer


# ---------- 解析 OLIGOMER 日志 & 写 Summary ----------
FLOAT = r'[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?'
RE_EXP  = re.compile(r'^\s*Experimental data file\s+(.+?)\s+Range of Scattering', re.I)
RE_LINE = re.compile(r'^\s*(\S+)\s+(' + FLOAT + r')\s+(' + FLOAT + r')\s+(' + FLOAT + r')\s+(.*)$')
RE_VE   = re.compile(r'(' + FLOAT + r')\s*\+\-\s*(' + FLOAT + r')')

def parse_oligomer_log(log_path: Path, component_names, has_constant):
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_exp = None
    rows = []
    for ln in text:
        m_exp = RE_EXP.search(ln)
        if m_exp:
            current_exp = m_exp.group(1).strip()
            continue
        m = RE_LINE.match(ln)
        if not m:
            continue
        token, chi, mw, rg, tail = m.groups()
        chi, mw, rg = float(chi), float(mw), float(rg)
        pairs = RE_VE.findall(tail)
        if not pairs:
            continue
        vols = [float(v) for v,_ in pairs]
        errs = [float(e) for _,e in pairs]
        expected = len(component_names) + (1 if has_constant else 0)
        vols = vols[:expected]; errs = errs[:expected]

        expname = current_exp or token
        rec = {"file": expname, "Chi2": chi, "MW": mw, "Rg": rg}
        for name, v, e in zip(component_names, vols[:len(component_names)], errs[:len(component_names)]):
            rec[name] = v; rec[name+"_err"] = e
        if has_constant and len(vols) > len(component_names):
            rec["constant"] = vols[-1]; rec["constant_err"] = errs[-1]
        rows.append(rec)
    return rows

def write_summary_xlsx(log_path: Path, component_files, has_constant, out_dir: Path):
    """Parse OLIGOMER log and write summary workbook.

    Sheets written:
      - all:        all parsed runs with fit-class labels
      - selected:   OK + Borderline only (per thresholds requested)
      - OK / Borderline / Bad / Fail: convenience subsets

    Returns: (df_all, df_selected, xlsx_path)
    """
    comp_names = [_basename_without_ext(p) for p in component_files]
    rows = parse_oligomer_log(log_path, comp_names, has_constant)
    if not rows:
        print("WARNING: no results parsed from log.")
        return None, None, None

    df_all = pd.DataFrame(rows)

    # Harmonize key columns
    if "Rg" in df_all.columns:
        df_all = df_all.rename(columns={"Rg": "apparent Rg_fit"})
    if "Chi2" in df_all.columns:
        df_all["Chi2"] = pd.to_numeric(df_all["Chi2"], errors="coerce")

    # Normalize file -> stem for matching .fit names
    if "file" not in df_all.columns:
        df_all["file"] = pd.NA
    df_all["stem"] = df_all["file"].astype(str).map(lambda s: Path(s).stem)

    # Detect constant-only failures (requires constant column)
    if has_constant and "constant" in df_all.columns:
        # Sum of non-constant components (treat missing as 0)
        nonconst_sum = 0.0
        for cn in comp_names:
            if cn in df_all.columns:
                nonconst_sum = nonconst_sum + pd.to_numeric(df_all[cn], errors="coerce").fillna(0.0)
        const_frac = pd.to_numeric(df_all["constant"], errors="coerce").fillna(0.0)
        df_all["is_fail_constant_only"] = (const_frac >= CONST_ONLY_CONST_MIN) & (nonconst_sum <= CONST_ONLY_OTHER_MAX)
    else:
        df_all["is_fail_constant_only"] = False

    # Fit quality class
    def _class_row(r):
        chi = r.get("Chi2")
        if pd.isna(chi):
            return "Unknown"
        if bool(r.get("is_fail_constant_only", False)):
            return "Fail"
        if chi > CHI2_BORDERLINE_MAX:
            return "Bad"
        if chi >= CHI2_OK_MAX:
            return "Borderline"
        return "OK"

    df_all["Oligo_fit_class"] = df_all.apply(_class_row, axis=1)

    # Selection: keep OK + Borderline only
    df_selected = df_all[df_all["Oligo_fit_class"].isin(["OK", "Borderline"])].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / "oligomer_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl", mode="w") as writer:
        df_all.to_excel(writer, sheet_name="all", index=False)
        df_selected.to_excel(writer, sheet_name="selected", index=False)
        # Borderline sorted by Chi2 (descending) for quick spot-check
        sub_bl = df_all[df_all["Oligo_fit_class"] == "Borderline"].copy()
        if not sub_bl.empty and "Chi2" in sub_bl.columns:
            sub_bl = sub_bl.sort_values("Chi2", ascending=False)
        if not sub_bl.empty:
            sub_bl.to_excel(writer, sheet_name="Borderline_sorted", index=False)
        for cls in ["OK", "Borderline", "Bad", "Fail", "Unknown"]:
            sub = df_all[df_all["Oligo_fit_class"] == cls].copy()
            if not sub.empty:
                sheet = cls[:31]  # Excel sheet name limit
                sub.to_excel(writer, sheet_name=sheet, index=False)

    # Console summary
    counts = df_all["Oligo_fit_class"].value_counts(dropna=False).to_dict()
    print(f"wrote: {xlsx} (counts: {counts})")
    return df_all, df_selected, xlsx


# ---------- 读取 Guinier_stats / useful_parameters，构建 ML 基表 ----------
# ---------- Plot selected .fit curves ----------
def _parse_stem_for_sort(stem: str):
    """Return (sort_key_tuple, display_label) from a sample stem.

    Expected stems like:
      - Buf_7.5mg
      - Buf_25mgb
      - EAN54.6_25mg
      - EAN54.6_25mgb
    """
    label = stem
    solvent = None
    ean = None
    mg = None
    suffix_b = 0

    try:
        parts = stem.split("_")
        if len(parts) >= 2:
            head, tail = parts[0], parts[1]
        else:
            head, tail = stem, ""

        if head.startswith("EAN"):
            solvent = "EAN"
            try:
                ean = float(head[3:])
            except Exception:
                ean = 9999.0
        elif head.startswith("Buf") or head == "Buf":
            solvent = "Buf"
            ean = -1.0
        else:
            solvent = "Other"
            ean = 9999.0

        if tail:
            t = tail
            # keep only the last token if more underscores exist
            if len(parts) > 2:
                t = parts[-1]
            if t.endswith("mgb"):
                suffix_b = 1
                tnum = t[:-3]  # remove 'mgb'
            elif t.endswith("mg"):
                tnum = t[:-2]
            else:
                tnum = t
            try:
                mg = float(tnum)
            except Exception:
                mg = 9999.0
        else:
            mg = 9999.0
    except Exception:
        solvent = "Other"; ean = 9999.0; mg = 9999.0; suffix_b = 0

    # Order: Buf first, then EAN by ean%, then mg, then 'b' suffix
    solvent_rank = 0 if solvent == "Buf" else (1 if solvent == "EAN" else 2)
    key = (solvent_rank, float(ean if ean is not None else 9999.0), float(mg if mg is not None else 9999.0), int(suffix_b), stem)
    return key, label


def _read_oligomer_fit_file(fit_path: Path):
    """Read OLIGOMER .fit.

    Expected format (as produced by ATSAS/OLIGOMER):
      First line: contains 'Chi^2 = <value>'
      Data lines: 4 numeric columns -> s, Iexp, sigma, Ifit

    Returns:
      s (np.ndarray), Iexp (np.ndarray), sigma (np.ndarray), Ifit (np.ndarray|None), chi2 (float|np.nan)
    """
    chi2 = np.nan
    s_list, ie_list, sig_list, if_list = [], [], [], []

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
                s = float(parts[0]); ie = float(parts[1]); sig = float(parts[2])
            except Exception:
                continue
            s_list.append(s); ie_list.append(ie); sig_list.append(sig)
            if len(parts) >= 4:
                try:
                    if_list.append(float(parts[3]))
                except Exception:
                    if_list.append(np.nan)

    s = np.asarray(s_list, dtype=float)
    ie = np.asarray(ie_list, dtype=float)
    sig = np.asarray(sig_list, dtype=float)
    ifit = np.asarray(if_list, dtype=float) if len(if_list) == len(s_list) else None
    return s, ie, sig, ifit, chi2




def plot_selected_fits_from_df(df_selected: pd.DataFrame, fit_dir: Path):
    """Plot selected (OK+Borderline) samples from df_selected using .fit files in fit_dir.

    Requirements implemented:
      1) PLOT_NCOLS at the top controls how many panels per row.
      2) Read .fit: col1=x, col2=Iexp (grey), col4=Ifit (red).
      3) Parse Chi^2 from the first line in each .fit and annotate each panel.

    Outputs into: fit_dir / 'plot'
      - log-saxs_page01.png (annotated with sample name + chi2)
      - log-saxs_page01_noannot.png (no sample name; chi2 still shown)
      - log-saxs_all_noannot.png (alias when only one page)
    If more than (PLOT_NCOLS * PLOT_NROWS) curves, additional pages are written.
    """
    if df_selected is None or df_selected.empty:
        print("No selected rows (OK/Borderline); skip plotting.")
        return []

    # Collect available .fit files matching stems
    stems = df_selected["stem"].astype(str).tolist() if "stem" in df_selected.columns else df_selected["file"].astype(str).map(lambda s: Path(s).stem).tolist()
    stems = [s for s in stems if s and s.lower() != 'nan']
    uniq_stems = sorted(set(stems), key=lambda s: _parse_stem_for_sort(s)[0])

    fit_files: List[Path] = []
    for st in uniq_stems:
        fp = fit_dir / f"{st}.fit"
        if fp.exists():
            fit_files.append(fp)
            continue
        # OLIGOMER may vary case/suffix; try glob
        cand = list(fit_dir.glob(f"{st}*.fit"))
        if cand:
            fit_files.append(cand[0])
        else:
            print(f"WARNING: .fit not found for selected stem: {st}")

    if not fit_files:
        print(f"No .fit files found under: {fit_dir}")
        return []

    # Sort by filename-derived key (Buf first, then EAN% asc, then mg asc, then mgb last)
    fit_files = sorted(fit_files, key=lambda p: _parse_stem_for_sort(p.stem)[0])

    # Pre-read to define common axis limits (shared axes)
    qmins, qmaxs, imins, imaxs = [], [], [], []
    cache = {}
    for fp in fit_files:
        q, ie, sig, ifit, chi2 = _read_oligomer_fit_file(fp)
        # keep only positive intensities for log scale
        vals = []
        if ie is not None and len(ie):
            vals.append(ie[ie > 0])
        if ifit is not None and len(ifit):
            vals.append(ifit[ifit > 0])
        vals = [v for v in vals if v is not None and len(v)]
        if len(q) == 0 or not vals:
            continue
        all_pos = np.concatenate(vals) if len(vals) > 1 else vals[0]
        cache[fp] = (q, ie, ifit, chi2)
        qmins.append(float(np.min(q))); qmaxs.append(float(np.max(q)))
        imins.append(float(np.min(all_pos))); imaxs.append(float(np.max(all_pos)))

    qmin = float(np.min(qmins)) if qmins else OLIGO_SMIN
    qmax = float(np.max(qmaxs)) if qmaxs else OLIGO_SMAX
    qmax = max(qmax, 0.35)  # Ensure x-axis extends to at least 0.35
    ymin = float(np.min(imins)) if imins else 1e-6
    ymax = float(np.max(imaxs)) if imaxs else 1.0

    out_dir = fit_dir / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load composition table for per-panel annotation (used in log-x figure)
    comp_map = _load_oligo_composition_map(ANALYSIS_DIR) if (SHOW_OLIGO_COMPOSITION_LOGX or SHOW_OLIGO_COMPOSITION_V2) else {}

    per_page = int(PLOT_NCOLS * PLOT_NROWS)
    pages = (len(fit_files) + per_page - 1) // per_page
    written: List[Path] = []

    def _make_page(page_files: List[Path], annotated: bool, log_x: bool = False, chi2_style: str = "left"):
        """
        chi2_style: 'left' = 原始左上角版本, 'right' = v2右上角版本
        """
        figsize = (PLOT_NCOLS * 2.1, PLOT_NROWS * 2.0)
        # Don't use sharex when log_x is True to avoid matplotlib conflicts
        fig, axes = plt.subplots(PLOT_NROWS, PLOT_NCOLS, figsize=figsize, 
                                 sharex=(not log_x), sharey=True)
        axes = np.asarray(axes).reshape(PLOT_NROWS, PLOT_NCOLS)

        for i_ax in range(per_page):
            r = i_ax // PLOT_NCOLS
            c = i_ax % PLOT_NCOLS
            ax = axes[r, c]
            ax.set_yscale('log')
            if log_x:
                ax.set_xscale('log')
                # Requirement: show 0.1 (not 10^-1) and explicitly label 0.01 and 0.2
                ax.xaxis.set_major_locator(mticker.FixedLocator(LOGX_TICKS))
                ax.xaxis.set_major_formatter(mticker.FixedFormatter(LOGX_TICKLABELS))
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_xlim(qmin, qmax)
            ax.set_ylim(ymin, ymax)
            # Show ticks on all subplots
            ax.tick_params(axis='both', which='major', length=4, width=0.8)
            ax.tick_params(axis='both', which='minor', length=2, width=0.6)

            if c == 0:
                ax.set_ylabel("log I(q), a.u.")

            if i_ax >= len(page_files):
                ax.set_axis_off()
                continue

            fp = page_files[i_ax]
            q, ie, ifit, chi2 = cache.get(fp, (None, None, None, np.nan))
            if q is None:
                q, ie, sig, ifit, chi2 = _read_oligomer_fit_file(fp)

            # Plot experimental (grey) and fit (red)
            if ie is not None and len(ie):
                m = (ie > 0) & np.isfinite(ie) & np.isfinite(q)
                if np.any(m):
                    if log_x and PLOT_EXP_AS_POINTS_LOGX:
                        ax.plot(q[m], ie[m], linestyle="None", marker=PLOT_EXP_MARKER,
                                markersize=PLOT_EXP_MARKERSIZE, color="0.5")
                    else:
                        ax.plot(q[m], ie[m], lw=PLOT_LINEWIDTH, color="0.5")  # grey
            if ifit is not None and len(ifit):
                m = (ifit > 0) & np.isfinite(ifit) & np.isfinite(q)
                if np.any(m):
                    ax.plot(q[m], ifit[m], lw=PLOT_LINEWIDTH * PLOT_FIT_LINEWIDTH_MULT,
                            color="#EC4706", alpha=0.6)

            # Oligomer composition annotation (multiline, left-bottom)
            # - log-x figure: controlled by SHOW_OLIGO_COMPOSITION_LOGX
            # - *_noannot_v2 (linear-x, chi2 on right): controlled by SHOW_OLIGO_COMPOSITION_V2
            if comp_map and (
                (log_x and SHOW_OLIGO_COMPOSITION_LOGX) or
                ((not log_x) and (chi2_style == "right") and SHOW_OLIGO_COMPOSITION_V2)
            ):
                key = _normkey_filename(fp.stem)
                comp = comp_map.get(key, None)
                # Fallback: strip common suffixes
                if comp is None and key.endswith("_fit"):
                    comp = comp_map.get(key[:-4], None)
                if isinstance(comp, dict):
                    txt = _format_oligo_composition_lines(comp)
                    if txt:
                        ax.text(OLIGO_COMP_TEXT_POS[0], OLIGO_COMP_TEXT_POS[1], txt,
                                transform=ax.transAxes, ha="left", va="bottom",
                                fontsize=OLIGO_COMP_FONTSIZE)

            # Chi2 annotation with style selection
            if not (chi2 is None or (isinstance(chi2, float) and np.isnan(chi2))):
                if chi2_style == "right":
                    # v2 version: chi2 in top-right with chi symbol
                    ax.text(0.98, 0.96, f"χ²={chi2:.2f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=max(PLOT_ANNOT_FONTSIZE - 2, 8))
                else:
                    # Original: chi2 in top-left
                    ax.text(0.02, 0.96, f"χ²={chi2:.2f}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=max(PLOT_ANNOT_FONTSIZE - 2, 8))

            # Sample label (only in annotated figure)
            if annotated:
                ax.text(0.5, 0.13, fp.stem, transform=ax.transAxes,
                        ha='center', va='center', fontsize=PLOT_ANNOT_FONTSIZE)

        if PLOT_TIGHT:
            fig.tight_layout(pad=0.2)
        else:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        return fig

    for pi in range(pages):
        page_no = pi + 1

        start = pi * per_page
        end = min(len(fit_files), (pi + 1) * per_page)
        page_files = fit_files[start:end]

        # Original version with chi2 on left
        figA = _make_page(page_files, annotated=True, log_x=False, chi2_style="left")
        outA = out_dir / f"log-saxs_page{page_no:02d}.png"
        figA.savefig(outA, dpi=PLOT_DPI)
        plt.close(figA)
        written.append(outA)

        figN = _make_page(page_files, annotated=False, log_x=False, chi2_style="left")
        outN = out_dir / f"log-saxs_page{page_no:02d}_noannot.png"
        figN.savefig(outN, dpi=PLOT_DPI)
        plt.close(figN)
        written.append(outN)

        # v2 version with chi2 on right (χ² symbol)
        figN_v2 = _make_page(page_files, annotated=False, log_x=False, chi2_style="right")
        outN_v2 = out_dir / f"log-saxs_page{page_no:02d}_noannot_v2.png"
        figN_v2.savefig(outN_v2, dpi=PLOT_DPI)
        plt.close(figN_v2)
        written.append(outN_v2)

        # log-x version with ticks and chi2 on right
        figN_logx = _make_page(page_files, annotated=False, log_x=True, chi2_style="right")
        outN_logx = out_dir / f"log-saxs_page{page_no:02d}_noannot_logx.png"
        figN_logx.savefig(outN_logx, dpi=PLOT_DPI)
        plt.close(figN_logx)
        written.append(outN_logx)

        # Requirement: write the subplot->file mapping in row-wise order
        order_txt = out_dir / f"log-saxs_page{page_no:02d}_noannot_logx_order.txt"
        _write_subplot_order_txt(order_txt, page_files, PLOT_NROWS, PLOT_NCOLS)
        written.append(order_txt)

    if pages == 1:
        try:
            (out_dir / "log-saxs_all_noannot.png").write_bytes((out_dir / "log-saxs_page01_noannot.png").read_bytes())
            (out_dir / "log-saxs_all_noannot_v2.png").write_bytes((out_dir / "log-saxs_page01_noannot_v2.png").read_bytes())
            (out_dir / "log-saxs_all_noannot_logx.png").write_bytes((out_dir / "log-saxs_page01_noannot_logx.png").read_bytes())
            (out_dir / "log-saxs_all_noannot_logx_order.txt").write_bytes((out_dir / "log-saxs_page01_noannot_logx_order.txt").read_bytes())
        except Exception:
            pass

    print(f"Plotted {len(fit_files)} selected .fit curves into: {out_dir}")
    return written


def plot_oligomer_composition_bargraph(analysis_dir: Path):
    """
    从 ML_targets_crystal_oligo.xlsx 读取 Oligo_fit_class==OK 或有有效数据的样品，
    绘制每个样品的组分堆积柱状图（bar graph），并标注 Rg 值。
    
    组分顺序: monomer, dimer, tetramer, hexamer, octamer
    颜色: 固定色系，buffer 样品放在第一位置
    输出: analysis3/plot/composition_bargraph.png
    """
    xlsx_path = analysis_dir / "ML_targets_crystal_oligo.xlsx"
    if not xlsx_path.exists():
        print(f"INFO: composition_bargraph skipped - {xlsx_path} not found")
        return None
    
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"WARNING: failed to read ML_targets_crystal_oligo.xlsx: {e}")
        return None
    
    # 检查 Oligo_fit_class 列
    if "Oligo_fit_class" not in df.columns:
        print("WARNING: Oligo_fit_class column not found")
        return None
    
    # 筛选：OK 类别或有 monomer/dimer/tetramer/hexamer/octamer 数据的样品
    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
    has_composition = df[comp_names].notna().any(axis=1)
    df_ok = df[(df["Oligo_fit_class"] == "OK") | has_composition].copy()
    
    if df_ok.empty:
        print("INFO: no samples with valid composition data found; skip composition bargraph")
        return None
    
    # 规范化文件名列
    if "file" not in df_ok.columns:
        for c in ["File", "filename", "Filename"]:
            if c in df_ok.columns:
                df_ok["file"] = df_ok[c]
                break
    
    # 组分列与颜色映射 - 使用 v2 新颜色
    comp_colors = COMP_COLORS_V2
    
    # 排序：buffer 优先，然后按数字排序
    df_ok = df_ok.sort_values(by="file", key=lambda x: x.map(_natural_sort_key))
    
    # 提取组分数据与 Rg
    samples = []
    compositions = []
    rgs = []
    
    for _, row in df_ok.iterrows():
        fname = str(row.get("file", ""))
        if not fname or fname.lower() == "nan":
            continue
        
        # Bufcontrol 改名为 Buffer
        display_name = "Buffer" if "buf" in fname.lower() else fname
        
        comp_row = []
        for comp in comp_names:
            try:
                val = float(row.get(comp, 0)) * 100 if comp in row else 0
                comp_row.append(val)
            except Exception:
                comp_row.append(0.0)
        
        # 提取 Rg（优先取 apparent Rg_fit，否则取 Rg）
        rg = None
        for rg_col in ["apparent Rg_fit", "Rg"]:
            if rg_col in row:
                try:
                    rg = float(row[rg_col])
                    break
                except Exception:
                    pass
        
        samples.append(display_name)
        compositions.append(comp_row)
        rgs.append(rg)
    
    if not samples:
        print("INFO: no valid samples after filtering")
        return None
    
    compositions = np.array(compositions)  # shape: (n_samples, 5)
    
    # 绘图 - 根据样品数自动调整宽度
    n_samples = len(samples)
    fig_width = max(14, n_samples * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    
    x_pos = np.arange(n_samples)
    bottom = np.zeros(n_samples)
    
    # 调整 bar 宽度：左边第一个和右边最后一个离外部边框太窄，缩小宽度以增加边距
    bar_width = 0.8  # 减小从 0.7 到 0.5
    
    for i, comp in enumerate(comp_names):
        color = comp_colors[comp]
        ax.bar(x_pos, compositions[:, i], bottom=bottom, label=comp, color=color, width=bar_width)
        
        # 在堆积部分顶部标注数值（非零时）
        for j, val in enumerate(compositions[:, i]):
            if val > 1:  # 只标注大于1%的
                y_pos = bottom[j] + val / 2
                ax.text(j, y_pos, f"{val:.0f}%", ha="center", va="center",
                       fontsize=8, color="white", weight="bold")
        
        bottom += compositions[:, i]
    
    # 在柱顶标注 Rg
    for i, rg in enumerate(rgs):
        if rg is not None and not np.isnan(rg):
            ax.text(i, 102, f"{rg:.1f}", ha="center", va="bottom", fontsize=11, weight="bold")
    
    ax.set_xlabel("", fontsize=12, weight="bold")
    ax.set_ylabel("Composition (%)", fontsize=12, weight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=10)
    # ensure bars are centered with margin on both sides
    ax.set_xlim(-0.5, n_samples - 0.5)
    # set lower y bound to 39 as requested
    ax.set_ylim(56, 110)
    
    # Legend：水平排列，无背景框，放在图框稍微上方
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, 
             frameon=False, fontsize=11, handlelength=1.5)
    
    # 无 grid
    ax.set_axisbelow(True)
    ax.grid(False)
    
    fig.tight_layout()
    
    out_dir = analysis_dir / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "composition_bargraph.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Wrote composition bargraph: {out_path}")
    return out_path


def plot_oligomer_fits_with_composition_v2(analysis_dir: Path, fit_dir: Path):
    """
    从 ML_targets_crystal_oligo.xlsx 读取 Oligo_fit_class==OK 或有有效数据的样品，
    绘制对应的 .fit 曲线，排列方式参考 log-saxs_all_noannot_logx.png 风格，
    改进v2:
      1. 按 Excel 顺序排列（buffer 优先）
      2. 每个 subplot 标题为文件名（加上.5min如果存在），放在框的上方
      3. 在每个 subplot 左下角用纯文字显示组分百分比（无表格背景）
      4. Y轴标题为 "Log I(q), a.u."（I和q斜体）
    
    输出: analysis3/fit/plot/fits_composition_v2_page01.png (一张或多张分页)
    """
    xlsx_path = analysis_dir / "ML_targets_crystal_oligo.xlsx"
    if not xlsx_path.exists():
        print(f"INFO: fits_with_composition_table skipped - {xlsx_path} not found")
        return None
    
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception as e:
        print(f"WARNING: failed to read ML_targets_crystal_oligo.xlsx: {e}")
        return None
    
    # 检查 Oligo_fit_class 列
    if "Oligo_fit_class" not in df.columns:
        print("WARNING: Oligo_fit_class column not found")
        return None
    
    # 组分列与颜色映射
    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
    comp_colors = COMP_COLORS_V2  # 使用配置中的新颜色
    
    # 筛选：OK 类别或有 monomer/dimer/tetramer/hexamer/octamer 数据的样品
    has_composition = df[comp_names].notna().any(axis=1)
    df_ok = df[(df["Oligo_fit_class"] == "OK") | has_composition].copy()
    
    if df_ok.empty:
        print("INFO: no samples with valid composition data found; skip fits_with_composition_table")
        return None
    
    # 规范化文件名列
    if "file" not in df_ok.columns:
        for c in ["File", "filename", "Filename"]:
            if c in df_ok.columns:
                df_ok["file"] = df_ok[c]
                break
    
    # 排序：buffer 优先，然后按数字排序
    df_ok = df_ok.sort_values(by="file", key=lambda x: x.map(_natural_sort_key))
    df_ok = df_ok.reset_index(drop=True)
    
    # 预读所有 .fit 文件以确定共享轴范围
    fit_data = {}  # stem -> (q, ie, ifit, chi2)
    qmins, qmaxs, imins, imaxs = [], [], [], []
    
    for _, row in df_ok.iterrows():
        fname = str(row.get("file", ""))
        if not fname or fname.lower() == "nan":
            continue
        
        stem = Path(fname).stem
        fit_path = fit_dir / f"{stem}.fit"
        
        if not fit_path.exists():
            # 尝试 glob 查找
            cand = list(fit_dir.glob(f"{stem}*.fit"))
            if cand:
                fit_path = cand[0]
            else:
                print(f"WARNING: .fit not found for {stem}")
                continue
        
        q, ie, sig, ifit, chi2 = _read_oligomer_fit_file(fit_path)
        if len(q) == 0:
            continue
        
        fit_data[stem] = (q, ie, ifit, chi2)
        
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
        print("INFO: no .fit files found for OK samples")
        return None
    
    qmin = float(np.min(qmins)) if qmins else OLIGO_SMIN
    qmax = float(np.max(qmaxs)) if qmaxs else OLIGO_SMAX
    qmax = max(qmax, 0.35)
    ymin = float(np.min(imins)) if imins else 1e-6
    ymax = float(np.max(imaxs)) if imaxs else 1.0
    
    out_dir = fit_dir.parent / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 分页处理
    per_page = int(PLOT_NCOLS * PLOT_NROWS)
    page_samples = []  # 每页对应的行号列表
    
    for idx in range(len(df_ok)):
        page_idx = idx // per_page
        if len(page_samples) <= page_idx:
            page_samples.append([])
        page_samples[page_idx].append(idx)
    
    written = []
    
    for page_no, sample_indices in enumerate(page_samples, start=1):
        figsize = (PLOT_NCOLS * 2.1, PLOT_NROWS * 1.85)  # 稍微减少行距
        fig, axes = plt.subplots(PLOT_NROWS, PLOT_NCOLS, figsize=figsize, sharex=False, sharey=True)
        axes = np.asarray(axes).reshape(PLOT_NROWS, PLOT_NCOLS)
        
        for subplot_idx in range(per_page):
            r = subplot_idx // PLOT_NCOLS
            c = subplot_idx % PLOT_NCOLS
            ax = axes[r, c]
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # log-x 刻度
            ax.xaxis.set_major_locator(mticker.FixedLocator(LOGX_TICKS))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(LOGX_TICKLABELS))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            
            ax.set_xlim(qmin, qmax)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(axis='both', which='major', length=4, width=0.8)
            ax.tick_params(axis='both', which='minor', length=2, width=0.6)
            ax.set_yticklabels([])  # 移除Y轴tick label
            
            if c == 0:
                ax.set_ylabel("log I(q), a.u.")
            
            if subplot_idx >= len(sample_indices):
                ax.set_axis_off()
                continue
            
            sample_idx = sample_indices[subplot_idx]
            row = df_ok.iloc[sample_idx]
            fname = str(row.get("file", ""))
            
            if not fname or fname.lower() == "nan":
                ax.set_axis_off()
                continue
            
            stem = Path(fname).stem
            if stem not in fit_data:
                ax.set_axis_off()
                continue
            
            q, ie, ifit, chi2 = fit_data[stem]
            
            # 绘制实验曲线（灰色点）与拟合曲线（红色线）
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
            
            # 标题：显示.dat文件名（去掉扩展名），Bufcontrol改名为Buffer
            title_text = stem
            if "buf" in title_text.lower():
                title_text = "Buffer"
            ax.text(0.5, 0.99, title_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10, weight='bold')
            
            # 组分表格（左下角，约占1/3空间）
            comp_data = []
            for comp in comp_names:
                try:
                    val = float(row.get(comp, 0)) * 100 if comp in row else 0
                    comp_data.append([comp.capitalize(), f"{val:.1f}%"])
                except Exception:
                    comp_data.append([comp.capitalize(), "N/A"])

            # Add header row so data rows index from 1..N
            comp_table_data = [["Component", "%"]] + comp_data

            table = ax.table(
                cellText=comp_table_data,
                cellLoc="center",
                loc="lower left",
                bbox=COMP_TABLE_BBOX,  # 使用配置参数
                colWidths=[0.7, 0.3]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(COMP_TABLE_FONTSIZE)
            table.scale(1, COMP_TABLE_ROW_HEIGHT)  # 使用配置参数

            # Header styling
            for i in range(2):
                table[(0, i)].set_facecolor("#f0f0f0")
                table[(0, i)].set_text_props(weight="bold")

            # Data-row first column colored by component palette
            n_rows = len(comp_table_data)  # includes header
            for i, comp in enumerate(comp_names):
                row_idx = i + 1  # data rows start at 1
                if row_idx < n_rows:
                    table[(row_idx, 0)].set_facecolor(comp_colors[comp])
                    table[(row_idx, 0)].set_text_props(color="white", weight="bold")
        
        # 减少行间距
        fig.subplots_adjust(wspace=0.05, hspace=0.08)
        
        out_path = out_dir / f"fits_composition_page{page_no:02d}.png"
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        print(f"Wrote: {out_path}")
    
    # 若只有一页，创建别名
    if len(page_samples) == 1:
        try:
            src = out_dir / "fits_composition_page01.png"
            dst = out_dir / "fits_composition_all.png"
            dst.write_bytes(src.read_bytes())
        except Exception:
            pass
    
    print(f"Plotted oligomer fits with composition table: {len(written)} file(s)")
    return written


def plot_oligomer_fits_with_composition(analysis_dir: Path, fit_dir: Path):
    """
    Deprecated: Use plot_oligomer_fits_with_composition_v2 instead.
    This version is kept for backward compatibility.
    """
    return plot_oligomer_fits_with_composition_v2(analysis_dir, fit_dir)


def plot_oligomer_fits_with_composition_v3(analysis_dir: Path, fit_dir: Path):
    """
    V3: 2 rows x 5 cols layout with specific samples.
    Selected samples: Buffer, 5min, 7min, 9min, 11min, 13min, 15min, 17min, 20min, 22min
    
    Features:
      - 2x5 grid (10 subplots)
      - Non-bold titles with Arial font
      - Composition text positioned lower
      - All text in Arial
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
    
    # Fixed sample order
    target_samples = ["Buffer", "5min", "7min", "9min", "11min", "13min", "15min", "17min", "20min", "22min"]
    sample_mapping = {}
    
    # Build mapping: display_name -> (file_from_excel, composition_data, fit_path)
    for target in target_samples:
        if target == "Buffer":
            candidates = df[df["file"].str.contains("buf", case=False, na=False)]
        else:
            candidates = df[df["file"].str.contains(target, case=False, na=False)]
        
        if not candidates.empty:
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
    
    if len(sample_mapping) == 0:
        print("INFO: no matching samples found for v3")
        return None
    
    # Read fit files and determine axis ranges
    fit_data = {}
    qmins, qmaxs, imins, imaxs = [], [], [], []
    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
    
    for display_name, (row, fit_path) in sample_mapping.items():
        q, ie, sig, ifit, chi2 = _read_oligomer_fit_file(fit_path)
        if len(q) == 0:
            continue
        fit_data[display_name] = (q, ie, ifit)
        
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
        print("INFO: no fit files found for v3")
        return None
    
    qmin = float(np.min(qmins)) if qmins else OLIGO_SMIN
    qmax = float(np.max(qmaxs)) if qmaxs else OLIGO_SMAX
    qmax = max(qmax, 0.35)
    ymin = float(np.min(imins)) if imins else 1e-6
    ymax = float(np.max(imaxs)) if imaxs else 1.0
    
    out_dir = fit_dir.parent / "plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2 rows, 5 cols
    fig, axes = plt.subplots(2, 5, figsize=(16, 8), sharex=False, sharey=True)
    axes = axes.flatten()
    
    # Set global font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    for subplot_idx, display_name in enumerate(target_samples):
        ax = axes[subplot_idx]
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        # Log-x ticks
        ax.xaxis.set_major_locator(mticker.FixedLocator(LOGX_TICKS))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(LOGX_TICKLABELS))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        
        ax.set_xlim(qmin, qmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', which='major', length=4, width=0.8)
        ax.tick_params(axis='both', which='minor', length=2, width=0.6)
        ax.set_yticklabels([])
        
        # Only left column gets y-axis label
        if subplot_idx % 5 == 0:
            ax.set_ylabel(r"Log $I(q)$, a.u.", fontsize=11, family='Arial')
        
        if display_name not in fit_data:
            ax.set_axis_off()
            continue
        
        row = sample_mapping[display_name][0]
        q, ie, ifit = fit_data[display_name]
        
        # Plot experimental (grey points) and fit (red line)
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
        
        # Title: non-bold, Arial
        ax.text(0.5, 0.99, display_name, transform=ax.transAxes, 
               ha='center', va='top', fontsize=10, weight='normal', family='Arial')
        
        # Composition text: lower position (changed from 0.15 to 0.05)
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


def plot_oligomer_composition_bargraph_v3(analysis_dir: Path):
    """
    V3 bar graph with same samples as fits_composition_v3.
    Y-axis starts at 56, larger bar labels, Arial font.
    """
    # 允许通过全局开关关闭该出图
    if not COMP_BAR_V3_ENABLED:
        print("INFO: composition_bargraph_v3 disabled by COMP_BAR_V3_ENABLED=False")
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
    
    # Standardize file column
    if "file" not in df.columns:
        for c in ["File", "filename", "Filename"]:
            if c in df.columns:
                df["file"] = df[c]
                break
    
    # Fixed sample order from global config (same as v3 fits by default)
    target_samples = list(COMP_BAR_V3_TARGET_SAMPLES)
    comp_names = ["monomer", "dimer", "tetramer", "hexamer", "octamer"]
    # Define color mapping
    comp_colors = {
        "monomer": "#749D65",
        "dimer": "#9366BC",
        "tetramer": "#5989B5",
        "hexamer": "#E37F48",
        "octamer": "#D4B833",
    }
    
    samples = []
    compositions = []
    rgs = []
    
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
                val = float(row.get(comp, 0)) * 100 if comp in row else  0
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
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    
    # Set global font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    x_pos = np.arange(n_samples)
    bottom = np.zeros(n_samples)
    # 使用配置的柱宽，柱间距尽量缩小以符合高质量文章配图风格
    bar_width = float(COMP_BAR_V3_BAR_WIDTH)
    
    for i, comp in enumerate(comp_names):
        color = comp_colors[comp]
        ax.bar(x_pos, compositions[:, i], bottom=bottom, label=comp.capitalize(), color=color, width=bar_width)
        
        # Larger bar labels (increased fontsize from 8 to 11)
        for j, val in enumerate(compositions[:, i]):
            if val > 1:
                y_pos = bottom[j] + val / 2
                ax.text(j, y_pos, f"{val:.0f}%", ha="center", va="center",
                       fontsize=11, color="white", weight="bold", family='Arial')
        
        bottom += compositions[:, i]
    
    # Rg annotations at bar tops
    for i, rg in enumerate(rgs):
        if rg is not None and not np.isnan(rg):
            ax.text(i, 58, f"{rg:.1f}", ha="center", va="bottom", fontsize=11, weight="bold", family='Arial')
    
    ax.set_xlabel("", fontsize=12, weight="bold", family='Arial')
    ax.set_ylabel("Composition (%)", fontsize=12, weight="bold", family='Arial')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=10, family='Arial')
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(56, 110)  # Y-axis starts at 56
    
    # Legend with reduced spacing
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, 
             frameon=False, fontsize=14, handlelength=1.2)
    
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


def _load_guinier_stats(analysis_dir: Path) -> Optional[pd.DataFrame]:
    """
    容错读取 analysis/Guinier_stats.txt，标准化列名：tag, Rg, Rg_err, chi2_red, t_a, verdict, R2_w
    未出现的列会补 NA；tag 会做文件名规范化（无扩展名小写）。
    """
    path = analysis_dir / "Guinier_stats.txt"
    if not path.exists():
        return None
    try:
        # 自动识别逗号分隔；第一行为表头（见示例文件）
        df = pd.read_csv(path, sep=",", skipinitialspace=True, engine="python", encoding="utf-8", comment="#")
        df.columns = [c.strip() for c in df.columns]
        # 找样品名列（旧数据常叫 dat set / dataset / file）
        col_lut = {c.lower().replace(" ",""): c for c in df.columns}
        name_col = None
        for key in ("dataset","datset","dat_set","datset:","datset", "datset:", "datset", "datset", "datset", "datset", "file","sample","datset", "datset"):
            if key in col_lut:
                name_col = col_lut[key]; break
        if name_col is None:
            # 备选：原样第一列
            name_col = df.columns[0]
        df = df.rename(columns={name_col: "file"})
        # 标准列映射
        mapping = {}
        if "rg"        in col_lut: mapping[col_lut["rg"]] = "Rg"
        if "rg_err"    in col_lut: mapping[col_lut["rg_err"]] = "Rg_err"
        if "rgerr"     in col_lut: mapping[col_lut["rgerr"]] = "Rg_err"
        if "chi2_red"  in col_lut: mapping[col_lut["chi2_red"]] = "chi2_red"
        if "chi2red"   in col_lut: mapping[col_lut["chi2red"]] = "chi2_red"
        if "t_a"       in col_lut: mapping[col_lut["t_a"]] = "t_a"
        if "ta"        in col_lut: mapping[col_lut["ta"]] = "t_a"
        if "r2_w"      in col_lut: mapping[col_lut["r2_w"]] = "R2_w"
        if "r2w"       in col_lut: mapping[col_lut["r2w"]] = "R2_w"
        if "verdict"   in col_lut: mapping[col_lut["verdict"]] = "verdict"
        df = df.rename(columns=mapping)
        for c in ["Rg","Rg_err","chi2_red","t_a","R2_w"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = pd.NA
        if "verdict" not in df.columns:
            df["verdict"] = pd.NA
        # 归一化 file -> __key__
        df["file"] = df["file"].astype(str)
        df["__key__"] = df["file"].map(_normkey_filename)
        return df[["file","__key__","Rg","Rg_err","chi2_red","t_a","verdict","R2_w"]].copy()
    except Exception as e:
        print("WARNING: failed to parse Guinier_stats.txt:", e)
        return None

def _load_useful_parameters(analysis_dir: Path) -> Optional[pd.DataFrame]:
    path = analysis_dir / "useful_parameters.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if "file" not in df.columns:
            # 尝试候选列
            for c in ["File","filename","Filename","name","Name","tag","Tag"]:
                if c in df.columns:
                    df["file"] = df[c].astype(str); break
            else:
                df["file"] = df[df.columns[0]].astype(str)
        df["__key__"] = df["file"].map(_normkey_filename)
        return df
    except Exception as e:
        print("WARNING: failed to read useful_parameters.csv:", e)
        return None

def build_ml_targets_crystal_if_missing(analysis_dir: Path, out_name: str = "ML_targets_crystal.xlsx") -> Optional[Path]:
    """
    若 analysis/ML_targets_crystal.xlsx 不存在，则
    基于 Guinier_stats.txt + useful_parameters.csv 构造一个基础 ML 目标表。
    """
    xlsx = analysis_dir / out_name
    if xlsx.exists():
        print(f"{out_name} already exists; skip building.")
        return xlsx

    df_g = _load_guinier_stats(analysis_dir)     # tag-like统计
    df_u = _load_useful_parameters(analysis_dir) # 晶体峰/CI/B因子等

    if df_g is None and df_u is None:
        print(f"NOTE: neither Guinier_stats.txt nor useful_parameters.csv present; cannot build {out_name}.")
        return None

    # 基于 Guinier 为主键（若无则基于 useful）
    if df_g is not None and not df_g.empty:
        base = df_g.copy()
    else:
        base = pd.DataFrame({"file": df_u["file"].astype(str)})
        base["__key__"] = base["file"].map(_normkey_filename)
        for c in ["Rg","Rg_err","chi2_red","t_a","verdict","R2_w"]:
            base[c] = pd.NA

    # 合并 useful 中的晶体相关列（若存在）
    if df_u is not None and not df_u.empty:
        cols_want = [
            "crystalline_present","n_real_peaks","CI","B_factor","B_fit_chi2_red",
            "fwhm_avg","fwhm_min","fwhm_max","snr_peak1",
            # 若你的 useful 里还有晶体域尺寸、峰位、比率等可在此追加
        ]
        for c in cols_want:
            if c not in df_u.columns:
                df_u[c] = pd.NA
        keep = ["__key__"] + cols_want
        base = base.merge(df_u[keep], on="__key__", how="left")

    # 整理列顺序
    cols_out = [
        "file","Rg","Rg_err","chi2_red","t_a","verdict","R2_w",
        "crystalline_present","n_real_peaks","CI","B_factor","B_fit_chi2_red",
        "fwhm_avg","fwhm_min","fwhm_max","snr_peak1"
    ]
    for c in cols_out:
        if c not in base.columns:
            base[c] = pd.NA
    out_df = base[cols_out].copy()

    try:
        out_df.to_excel(xlsx, index=False, engine="openpyxl")
        print(f"Wrote ML base workbook: {xlsx}")
        return xlsx
    except Exception as e:
        print("ERROR: failed to write ML_targets_crystal.xlsx:", e)
        return None


# ---------- 合并 OLIGOMER filtered -> ML_targets_crystal.xlsx ----------
def update_ml_targets_with_oligomer(
    analysis_dir: Path,
    filtered_summary_df: Optional[pd.DataFrame],
    targets_excel_name: str = "ML_targets_crystal.xlsx",
    out_excel_name: str = "ML_targets_crystal_oligo.xlsx",
    targets_sheet_name: Optional[str] = None,
):
    if filtered_summary_df is None or filtered_summary_df.empty:
        print("No filtered OLIGOMER rows; skip creating ML_targets_crystal_oligo.xlsx.")
        return None

    # 需要合并的 OLIGOMER 列
    oligomer_cols = [
        "Chi2","Oligo_fit_class","apparent Rg_fit",
        "dimer","dimer_err",
        "hexamer","hexamer_err",
        "monomer","monomer_err",
        "octamer","octamer_err",
        "tetramer","tetramer_err",
        "constant","constant_err"
    ]
    s = filtered_summary_df.copy()
    # 标准化 file -> __key__
    if "file" not in s.columns:
        s["file"] = pd.NA
    s["__key__"] = s["file"].astype(str).map(_normkey_filename)
    for c in oligomer_cols:
        if c not in s.columns:
            s[c] = pd.NA
    merge_block = s[["__key__"] + oligomer_cols].copy()

    # Deduplicate per sample key ("__key__") to avoid one sample producing multiple rows
    # in the merged workbook. This is critical when sample names are repeated or when
    # multiple OLIGOMER fits exist for the same file.
    comp_base = [c for c in ["monomer","dimer","tetramer","hexamer","octamer","constant"] if c in merge_block.columns]
    rank_map = {"OK":0, "Borderline":1, "Bad":2, "Fail":3}
    merge_block["__rank__"] = merge_block["Oligo_fit_class"].map(rank_map).fillna(9).astype(int)
    if comp_base:
        merge_block["__nonnull_comp__"] = merge_block[comp_base].notna().sum(axis=1)
        merge_block["__nonzero_comp__"] = (merge_block[comp_base].fillna(0) > 0).sum(axis=1)
    else:
        merge_block["__nonnull_comp__"] = 0
        merge_block["__nonzero_comp__"] = 0
    # Sort preference: more composition numbers > more non-zero comps > better class > lower Chi2
    merge_block = (
        merge_block.sort_values(
            ["__key__", "__nonnull_comp__", "__nonzero_comp__", "__rank__", "Chi2"],
            ascending=[True, False, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["__key__"], keep="first")
        .drop(columns=["__rank__", "__nonnull_comp__", "__nonzero_comp__"])
    )

    # 找到 ML 基表
    base_path = analysis_dir / targets_excel_name
    if not base_path.exists():
        print(f"WARNING: {targets_excel_name} not found; will try to build it first.")
        built = build_ml_targets_crystal_if_missing(analysis_dir, targets_excel_name)
        if not built:
            print("Cannot build ML base; abort merge.")
            return None
        base_path = built

    xfile = pd.ExcelFile(base_path)
    tgt_sheet = targets_sheet_name or (xfile.sheet_names[0] if xfile.sheet_names else None)
    if tgt_sheet is None:
        print(f"WARNING: No sheets in {base_path}; skip.")
        return None

    sheets_out = {}
    for sheet in xfile.sheet_names:
        df = pd.read_excel(base_path, sheet_name=sheet)
        if sheet != tgt_sheet:
            sheets_out[sheet] = df
            continue

        # 找用于构造 __key__ 的列
        key_col = None
        for c in ["__key__","file","File","filename","Filename","name","Name","tag","Tag"]:
            if c in df.columns:
                key_col = c; break
        if key_col != "__key__":
            df["__key__"] = df[key_col].astype(str).map(_normkey_filename)

        # 删除可能已存在的 OLIGOMER 列，避免重复
        drop_existing = [c for c in oligomer_cols if c in df.columns]
        if drop_existing:
            df = df.drop(columns=drop_existing)

        merged = df.merge(merge_block, on="__key__", how="left")
        sheets_out[sheet] = merged.drop(columns=["__key__"])

    out_path = analysis_dir / out_excel_name
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in sheets_out.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Wrote merged workbook with OLIGOMER fields: {out_path}")
    return out_path


# ---------- 主流程 ----------
def main():
    # 0) 路径/可执行程序检查
    for exe, name in [(FFMAKER_EXE,"FFMAKER.exe"), (OLIGOMER_EXE,"oligomer.exe")]:
        if not exe.exists():
            raise FileNotFoundError(f"{name} not found: {exe}")
    if not PDB_DIR.exists():
        raise FileNotFoundError(f"PDB_DIR not found: {PDB_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    # 1) ff2.dat
    ffdat_path, pdb_files_in_order = ensure_ffdat(PDB_DIR)

    # 2) OLIGOMER
    fits, log_path = run_oligomer(ffdat_path, DATA_DIR)
    if fits:
        print("Generated .fit (moved to analysis/fit):")
        for p in fits:
            print("  -", p.name)

    # 3) 解析日志 -> oligomer_summary.xlsx（all/filtered）
    df_all, df_selected, summary_xlsx = write_summary_xlsx(
        log_path, pdb_files_in_order, USE_CONST, ANALYSIS_DIR
    )

    # 4) 若 ML 基表不存在，先用 Guinier_stats + useful_parameters 生成
    build_ml_targets_crystal_if_missing(ANALYSIS_DIR, "ML_targets_crystal.xlsx")

    # 5) 先合并 OLIGOMER filtered -> ML_targets_crystal_oligo.xlsx
    #    (确保后续绘图首次运行就能读到 monomer/dimer/... 信息)
    update_ml_targets_with_oligomer(
        ANALYSIS_DIR, df_selected,
        targets_excel_name="ML_targets_crystal.xlsx",
        out_excel_name="ML_targets_crystal_oligo.xlsx",
    )

    # 6) Plot selected .fit curves (OK + Borderline only) -> fit/plot
    try:
        plot_selected_fits_from_df(df_selected, FIT_DIR)
    except Exception as e:
        print(f"WARNING: plotting skipped due to error: {e}")

    # 7) 新增：绘制组分堆积柱状图
    try:
        plot_oligomer_composition_bargraph(ANALYSIS_DIR)
    except Exception as e:
        print(f"WARNING: composition bargraph skipped due to error: {e}")

    # 8) 新增：绘制带组分表的拟合曲线
    try:
        plot_oligomer_fits_with_composition(ANALYSIS_DIR, FIT_DIR)
    except Exception as e:
        print(f"WARNING: fits with composition table skipped due to error: {e}")
        traceback.print_exc()

    # 9) 新增：绘制改进版 v2 拟合曲线（纯文本组分显示）
    try:
        plot_oligomer_fits_with_composition_v2(ANALYSIS_DIR, FIT_DIR)
    except Exception as e:
        print(f"WARNING: fits composition v2 skipped due to error: {e}")
        traceback.print_exc()

    # 10) 新增：绘制 v3 版本（2x5 grid，固定样品，Arial字体）
    try:
        plot_oligomer_fits_with_composition_v3(ANALYSIS_DIR, FIT_DIR)
    except Exception as e:
        print(f"WARNING: fits composition v3 skipped due to error: {e}")
        traceback.print_exc()

    # 11) 新增：绘制 v3 bar 图（同样品，y轴从56开始）
    if COMP_BAR_V3_ENABLED:
        try:
            plot_oligomer_composition_bargraph_v3(ANALYSIS_DIR)
        except Exception as e:
            print(f"WARNING: composition bargraph v3 skipped due to error: {e}")
            traceback.print_exc()
    else:
        print("INFO: Skipped composition_bargraph_v3 (disabled by config)")

if __name__ == "__main__":
    main()