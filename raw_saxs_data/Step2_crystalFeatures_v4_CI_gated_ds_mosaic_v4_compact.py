# -*- coding: utf-8 -*-
"""
Step2_crystalFeatures_v3
========================
- Harmonize all *.dat onto a common q-grid (no extrapolation).
- Pick reference peaks from the "peakiest" pattern within a q band.
- Measure per-peak height/area/FWHM/SNR with robust background.
- Gate real peaks by absolute height (fraction of baseline) and SNR.
- Compute crystallinity index (CI) using gated peak areas (CI_gated; default) and report CI_all for diagnostics.
- Estimate Debye–Waller B by WLS on ln(peak heights) vs q^2 (fallback to ln I).
- NEW crystal size analyses based on the FIRST Bragg peak:
    * peak1_domain_nm_meas       ≈ 2π / FWHM
    * peak1_domain_nm_deconv     : instrument deconvolved (if possible)
    * scherrer_peak1_nm_K09      : Scherrer-like with K=0.9 → 0.9*(2π/FWHM)
    * d_spacing_peak1_A          : interplanar spacing from q1 = 2π / q1
    * peak1_L_from_integral_nm   : L ≈ 2π / β_area using integral breadth (area/height)
- Writes analysis/useful_parameters.csv

This file is designed to be a drop-in replacement for Step2_crystalFeatures_v2.py.
"""


from __future__ import annotations

import math, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================== Config ==================
ANALYSIS_DIR        = 'analysis4'          # subdirectory name for outputs
Q_BAND              = (0.02, 0.35)         # Å^-1, peak search band
SMOOTH_WIN_FRAC     = 0.01                 # fraction-of-points for smoothing
WINDOW_FRAC         = 0.02                 # ± fraction of band width per-peak window
MAX_PEAKS           = 8
PEAK_MERGE_DQ       = 0.003                # Å^-1, merge centers closer than this
MIN_PTS_IN_BAND     = 20
SNR_THRESHOLD       = 2.5
MIN_ABS_PEAK_FRAC   = 0.05                 # abs height >= 10% of baseline
MIN_REAL_PEAKS      = 2
MIN_CI              = 0.02
B_FACTOR_Q_BAND     = (0.15, 0.30)         # default fallback region for ln I vs q^2

MOSAIC_NCOLS = 4          # 每行几个 panel（你要改就改这里）
MOSAIC_DPI = 350          # 输出 DPI（建议 300–600）
MOSAIC_Q_HALF_WIDTH = 0.02  # 共享 x 轴范围：以 peak1_q 为中心 ±0.02 Å^-1


ANNOT_FONTSIZE = 12

# Add the desired target data folder here (change to your actual folder containing .dat files)
# Example: Path(r"c:\path\to\your\data")
TARGET_DATA_DIR = Path(r"C:\Users\E100104\OneDrive - RMIT University\DATA\2025\5Early aggregation\ML\6\analysis\selected")  # <-- update this to your data folder if you want a non-default

# ================== Small utils ==================
def _clip_small(x: float, eps: float = 1e-20) -> float:
    return x if (x is not None and math.isfinite(x) and x > eps) else eps

def odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1

def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    w = odd(w)
    if w <= 1: return y.copy()
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(w, dtype=float)/w
    return np.convolve(ypad, kernel, mode='valid')

def local_rms(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size < 5: return float('nan')
    mu = np.median(y)
    return float(np.sqrt(np.mean((y - mu)**2)))

# ================== I/O ==================
def read_dat_flexible(path: Path):
    """Accept 2- or 3-column .dat: q I [Err]. If Err missing, assume 5% rel. error."""
    q,I,E = [],[],[]
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if not ln.strip(): continue
            parts = re.split(r'[,\s]+', ln.strip())
            try:
                qv = float(parts[0]); Iv = float(parts[1])
                Ev = float(parts[2]) if len(parts) > 2 else float('nan')
            except Exception:
                continue
            if (not math.isfinite(qv)) or (not math.isfinite(Iv)): continue
            q.append(qv); I.append(Iv); E.append(Ev)
    q = np.asarray(q, float); I = np.asarray(I, float); E = np.asarray(E, float)
    if np.isnan(E).all():
        E = 0.05*np.maximum(I, 1e-12)
    else:
        E = np.where(np.isfinite(E) & (E>0), E, 0.05*np.maximum(I, 1e-12))
    m = (I>0) & np.isfinite(q) & np.isfinite(I) & np.isfinite(E)
    q,I,E = q[m], I[m], E[m]
    order = np.argsort(q)
    return q[order], I[order], E[order]

# ================== Grid harmonization ==================
def build_common_grid(q_arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    qs = [q for q in q_arrays if isinstance(q, np.ndarray) and q.size >= 2]
    if not qs: return None
    qmin = max(float(q.min()) for q in qs)
    qmax = min(float(q.max()) for q in qs)
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin: return None
    meds = [np.median(np.diff(q)) for q in qs]
    dq = float(max(meds)) if meds else None
    if dq is None or dq <= 0: return None
    n = int(np.floor((qmax - qmin)/dq)) + 1
    qg = qmin + dq*np.arange(n, dtype=float)
    qg[-1] = min(qg[-1], qmax)
    return qg

def interp_to_grid(q, y, qg):
    yg = np.full_like(qg, np.nan, dtype=float)
    if q.size >= 2:
        m = (qg >= q.min()) & (qg <= q.max())
        if m.any():
            yg[m] = np.interp(qg[m], q, y)
    return yg

# ================== Peak picking / measurement ==================
def pick_reference_peaks(q_list, I_list, q_band=Q_BAND, smooth_frac=SMOOTH_WIN_FRAC,
                         max_peaks=MAX_PEAKS, merge_dq=PEAK_MERGE_DQ):
    # choose the "peakiest" curve inside band
    best_idx, best_score = -1, -np.inf
    for i, (q,I) in enumerate(zip(q_list, I_list)):
        m = (q>=q_band[0]) & (q<=q_band[1])
        if m.sum() < MIN_PTS_IN_BAND: continue
        qq, yy = q[m], I[m]
        w = odd(max(3, int(round(smooth_frac*qq.size))))
        ys = moving_average(yy, w)
        yc = moving_average(yy, odd(max(7, 3*w)))
        res = ys - yc[:ys.size]
        score = float(np.nanstd(res))
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx < 0:
        return []

    q_ref, I_ref = q_list[best_idx], I_list[best_idx]
    m = (q_ref>=q_band[0]) & (q_ref<=q_band[1])
    q_ref = q_ref[m]; y = I_ref[m]
    w = odd(max(3, int(round(smooth_frac*q_ref.size))))
    ys = moving_average(y, w)

    # local maxima on smoothed trace
    peaks = [i for i in range(1, ys.size-1) if ys[i-1] < ys[i] > ys[i+1]]
    if not peaks: return []
    rms = local_rms(ys - moving_average(ys, odd(max(7, 3*w))))
    prom = sorted([(ys[i]/max(rms,1e-12), i) for i in peaks], reverse=True)
    centers = [q_ref[i] for _,i in prom[:max_peaks*2]]
    centers.sort()
    merged = []
    for c in centers:
        if not merged or abs(c - merged[-1]) > merge_dq:
            merged.append(c)
    return merged[:max_peaks]

def _measure_peak_window(qq: np.ndarray, yy: np.ndarray, q0: float, dq: float) -> Dict[str,float]:
    m = (qq>=q0-dq)&(qq<=q0+dq)
    if m.sum() < 5:
        return dict(height=np.nan, area=np.nan, fwhm=np.nan, snr=np.nan)
    qwin, ywin = qq[m], yy[m]
    # robust smooth & baseline
    w_final = odd(min(max(3, int(round(0.3*m.sum()))), max(3, int(m.sum()/5))))
    ys = moving_average(ywin, w_final)
    k = max(1, int(0.1*ys.size))
    base = np.linspace(np.median(ys[:k]), np.median(ys[-k:]), ys.size)
    resid = np.maximum(ys - base, 0.0)

    height = float(np.max(resid)); half = height/2.0
    idx_max = int(np.argmax(resid))

    # FWHM via linear interpolation at half max
    iL = idx_max
    while iL>0 and resid[iL] > half: iL -= 1
    if iL < idx_max and resid[iL] != resid[iL+1]:
        frac = (half - resid[iL])/(resid[iL+1]-resid[iL])
        qL = qwin[iL] + frac*(qwin[iL+1]-qwin[iL])
    else: qL = float('nan')
    iR = idx_max
    while iR<resid.size-1 and resid[iR] > half: iR += 1
    if iR > idx_max and resid[iR-1] != resid[iR]:
        frac = (half - resid[iR-1])/(resid[iR]-resid[iR-1])
        qR = qwin[iR-1] + frac*(qwin[iR]-qwin[iR-1])
    else: qR = float('nan')
    fwhm = float(qR - qL) if (np.isfinite(qL) and np.isfinite(qR)) else np.nan

    nA = min(resid.size, qwin.size)
    # trapezoid (safe)
    area = float(np.trapezoid(resid[:nA], qwin[:nA]))

    # SNR from noise window excluding ±30% span around peak
    span = int(0.3*resid.size)
    mask_noise = np.ones(resid.size, dtype=bool)
    lo = max(0, idx_max - span); hi = min(resid.size, idx_max + span)
    mask_noise[lo:hi] = False
    rms = local_rms(resid[mask_noise])
    snr = float(height/max(rms,1e-12)) if np.isfinite(rms) else np.nan
    return dict(height=height, area=area, fwhm=fwhm, snr=snr)

# ================== Core features ==================
def crystal_features(q: np.ndarray, I: np.ndarray, q_centers: List[float],
                     q_band: Tuple[float,float] = Q_BAND,
                     abs_gate_frac: float = MIN_ABS_PEAK_FRAC,
                     snr_gate: float = SNR_THRESHOLD,
                     bin_dq: Optional[float] = None,
                     fwhm_instr_from_quant: Optional[float] = None) -> Dict[str,Any]:
    feats: Dict[str,Any] = {}
    if not q_centers or q.size < 10:
        feats['crystalline_present'] = False
        return feats

    band = (max(q_band[0], float(q.min())), min(q_band[1], float(q.max())))
    if band[1] <= band[0] or (np.sum((q>=band[0])&(q<=band[1])) < MIN_PTS_IN_BAND):
        feats['crystalline_present'] = False
        return feats

    mband = (q>=band[0]) & (q<=band[1])
    qq, yy = q[mband], I[mband]
    dq_global = WINDOW_FRAC*(band[1]-band[0])

    # band-level background & residual for CI
    ys_band = moving_average(yy, odd(max(5, int(round(0.02*yy.size)))))
    base_band = moving_average(ys_band, odd(max(7, int(round(0.2*ys_band.size)))))
    resid_band = np.maximum(ys_band - base_band[:ys_band.size], 0.0)
    n_int = min(resid_band.size, qq.size)
    total_pos_resid = float(np.trapezoid(resid_band[:n_int], qq[:n_int]))

    # per-peak measurements
    per = []
    for q0 in q_centers[:MAX_PEAKS]:
        if band[0] <= q0 <= band[1]:
            per.append(_measure_peak_window(qq, yy, q0, dq_global))
        else:
            per.append(dict(height=np.nan, area=np.nan, fwhm=np.nan, snr=np.nan))

    heights = np.array([p['height'] for p in per], float)
    areas   = np.array([p['area']   for p in per], float)
    fwhms   = np.array([p['fwhm']   for p in per], float)
    snrs    = np.array([p['snr']    for p in per], float)

    sumA_all = float(np.nansum(areas))
    baseline = float(np.nanmedian(ys_band)) if np.isfinite(ys_band).any() else np.nan
    abs_gate = np.isfinite(heights) & (heights >= abs_gate_frac*max(baseline,1e-9))
    snr_gate_mask = np.isfinite(snrs) & (snrs >= snr_gate)
    mask = abs_gate & snr_gate_mask
    n_real = int(np.sum(mask))

    # --- CI (crystallinity index) ---
    # CI_all   : uses ALL candidate peak areas (more sensitive to peak list drift)
    # CI_gated : uses ONLY peaks that pass abs+SNR gates (recommended; more stable across batches)
    sumA_gated = float(np.nansum(areas[mask])) if mask.any() else 0.0

    denom = max(total_pos_resid, 1e-20)
    CI_all = float(sumA_all/denom) if (np.isfinite(sumA_all) and sumA_all>0 and np.isfinite(total_pos_resid) and total_pos_resid>0) else float('nan')
    CI_gated = float(sumA_gated/denom) if (np.isfinite(sumA_gated) and sumA_gated>0 and np.isfinite(total_pos_resid) and total_pos_resid>0) else float('nan')

    # By default, use gated CI for classification and downstream merging
    crystalline = bool((n_real >= MIN_REAL_PEAKS) and (np.isfinite(CI_gated) and CI_gated >= MIN_CI))

    feats.update({
        'crystalline_present': crystalline,
        'n_real_peaks': n_real,
        'CI': CI_gated,
        'CI_all': CI_all,
        'CI_gated': CI_gated,
        'CI_sumA_all': sumA_all,
        'CI_sumA_gated': sumA_gated,
        'CI_pos_resid_area': total_pos_resid,
        'fwhm_avg': float(np.nanmean(fwhms)) if np.isfinite(fwhms).any() else np.nan,
        'fwhm_min': float(np.nanmin(fwhms)) if np.isfinite(fwhms).any() else np.nan,
        'fwhm_max': float(np.nanmax(fwhms)) if np.isfinite(fwhms).any() else np.nan,
        'snr_peak1': float(snrs[0]) if snrs.size>0 else np.nan,
    })

    for i, q0 in enumerate(q_centers[:MAX_PEAKS]):
        feats[f'peak{i+1}_q'] = float(q0)
        feats[f'peak{i+1}_fwhm'] = float(fwhms[i]) if i < fwhms.size else np.nan

    # peak ratios to q1
    if len(q_centers) >= 2 and np.isfinite(q_centers[0]) and q_centers[0] > 0:
        q1 = float(q_centers[0])
        for i in range(1, min(len(q_centers), MAX_PEAKS)):
            feats[f'peak_ratio_q{i+1}_over_q1'] = float(q_centers[i]/q1)

    # ---------- B-factor via WLS on ln(height) vs q^2 (peaks that pass gates) ----------
    B, chi2r = float('nan'), float('nan')
    if mask.sum() >= 3:
        x = (np.array(q_centers, float)[mask]**2).astype(float)
        y = np.log(np.maximum(heights[mask], 1e-30)).astype(float)
        sigma = 1.0/np.maximum(snrs[mask], 1e-6)
        # weighted linear fit
        w = 1.0/(np.maximum(sigma, 1e-12)**2)
        Sw = np.sum(w); Sx = np.sum(w*x); Sy = np.sum(w*y)
        Sxx = np.sum(w*x*x); Sxy = np.sum(w*x*y)
        D = Sw*Sxx - Sx*Sx
        if np.isfinite(D) and D != 0.0:
            m = (Sw*Sxy - Sx*Sy)/D
            c = (Sy*Sxx - Sx*Sxy)/D
            yhat = m*x + c
            chi2 = float(np.sum(((y - yhat)*np.sqrt(w))**2))
            dof = max(1, x.size-2)
            chi2r = chi2/dof
            B = -m  # m ~ -B
    # fallback: slope of ln I(q) in B_FACTOR_Q_BAND
    if not np.isfinite(B):
        m = (q>=B_FACTOR_Q_BAND[0]) & (q<=B_FACTOR_Q_BAND[1]) & np.isfinite(I) & (I>0)
        if np.sum(m) >= 6:
            xx = (q[m]**2).astype(float); yy2 = np.log(I[m]).astype(float)
            A = np.vstack([xx, np.ones_like(xx)]).T
            try:
                sol, *_ = np.linalg.lstsq(A, yy2, rcond=None)
                slope = sol[0]; B = float(-4.0*slope)  # conventional scaling
            except Exception:
                B = float('nan')
    feats['B_factor'] = float(B); feats['B_fit_chi2_red'] = float(chi2r)

    # ---------- NEW crystal size metrics from the FIRST peak ----------
    q1 = feats.get('peak1_q', np.nan)
    fwhm1 = feats.get('peak1_fwhm', np.nan)

    # d-spacing (Å) from peak position
    if np.isfinite(q1) and q1 > 0:
        feats['d_spacing_peak1_A'] = float(2.0*np.pi / q1)
    else:
        feats['d_spacing_peak1_A'] = np.nan

    # Measured domain (nm) ≈ 2π/FWHM
    if np.isfinite(fwhm1) and fwhm1 > 0:
        L_A = 2.0*np.pi / fwhm1
        feats['peak1_domain_nm_meas'] = float(L_A/10.0)
        # Scherrer-like (K=0.9)
        feats['scherrer_peak1_nm_K09'] = float(0.9 * L_A / 10.0)

        # Integral breadth β_area = area/height → L≈2π/β_area
        A1 = areas[0] if areas.size>0 else np.nan
        h1 = heights[0] if heights.size>0 else np.nan
        if np.isfinite(A1) and np.isfinite(h1) and h1>0:
            beta_area = float(A1 / h1)
            feats['peak1_L_from_integral_nm'] = float((2.0*np.pi/max(beta_area,1e-20))/10.0)
        else:
            feats['peak1_L_from_integral_nm'] = np.nan

        # Instrument deconvolution: prefer explicit fwhm_instr_from_quant; else use bin_dq
        fwhm_instr = None
        if fwhm_instr_from_quant and math.isfinite(fwhm_instr_from_quant) and fwhm_instr_from_quant > 0:
            fwhm_instr = float(fwhm_instr_from_quant)
        elif bin_dq and math.isfinite(bin_dq) and bin_dq > 0:
            # convert sampling bin Δq to Gaussian FWHM (σ = Δq/sqrt(12); FWHM = 2√(2 ln2) σ)
            sigma_instr = bin_dq / math.sqrt(12.0)
            fwhm_instr = 2.0*math.sqrt(2.0*math.log(2.0))*sigma_instr

        if fwhm_instr and fwhm1 > fwhm_instr:
            f2 = fwhm1**2 - fwhm_instr**2
            if f2 > 0:
                L_A_dec = 2.0*np.pi / math.sqrt(f2)
                feats['peak1_domain_nm_deconv'] = float(L_A_dec/10.0)
            else:
                feats['peak1_domain_nm_deconv'] = np.nan
        else:
            feats['peak1_domain_nm_deconv'] = np.nan
    else:
        feats['peak1_domain_nm_meas'] = np.nan
        feats['scherrer_peak1_nm_K09'] = np.nan
        feats['peak1_L_from_integral_nm'] = np.nan
        feats['peak1_domain_nm_deconv'] = np.nan

    return feats


# ================== Filename parsing & mosaic helpers (v3) ==================
# Goal:
#   - Build two mosaic figures (annot & simple) directly from .dat data (NOT by stitching PNGs),
#     so that all panels share the same x-axis range.
#   - Layout uses MOSAIC_NCOLS panels per row (default 4; user-adjustable at top).

_RE_EAN_NUM = re.compile(r"(?:^|[_\-])EAN\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RE_SOLVENT_BUF = re.compile(r"\bBuf\b", re.IGNORECASE)
_RE_SOLVENT_EAN = re.compile(r"\bEAN\b", re.IGNORECASE)
_RE_MG = re.compile(r"(?:^|[_\-])([0-9]+(?:\.[0-9]+)?)\s*mg\b", re.IGNORECASE)
_RE_TEMP_ANY = re.compile(r"([0-9]+(?:\.[0-9]+)?)C", re.IGNORECASE)         # supports 90C20C etc
_RE_TIME = re.compile(r"(?:^|[_\-])([0-9]+(?:\.[0-9]+)?)\s*(min|mins|m|h|hr|hrs|hour|hours|d|day|days)\b", re.IGNORECASE)
_RE_BATCH_B = re.compile(r"(?:^|[_\-])([0-9]+(?:\.[0-9]+)?)mgb\b", re.IGNORECASE)  # 25mgb -> batch b

def _parse_name_params(stem: str) -> Dict[str, Any]:
    """Parse solvent/EAN/protein/temp/time/batch from a file stem (no extension)."""
    s = str(stem)

    # solvent (prefer Buf if both appear)
    if _RE_SOLVENT_BUF.search(s):
        solvent = "Buf"
    elif _RE_SOLVENT_EAN.search(s):
        solvent = "EAN"
    else:
        solvent = "UNK"

    # EAN numeric e.g. EAN1.5; if only 'EAN_' without number -> NaN
    m = _RE_EAN_NUM.search(s)
    EAN = float(m.group(1)) if m else np.nan

    # protein mg/mL: take LAST match
    mg_all = _RE_MG.findall(s)
    protein_mgml = float(mg_all[-1]) if mg_all else np.nan

    # batch tag: 25mgb => batch b
    batch_tag = "b" if _RE_BATCH_B.search(s) else ""
    batch_id = batch_tag if batch_tag else "NA"

    # temperature: may contain multiple e.g. 90C20C
    temps = _RE_TEMP_ANY.findall(s)
    if temps:
        temp_C = float(temps[0])
        # keep full profile if multiple
        temp_profile = "".join([f"{t}C" for t in temps]) if len(temps) > 1 else f"{temps[0]}C"
    else:
        temp_C = np.nan
        temp_profile = ""

    # time: convert to minutes (min)
    m = _RE_TIME.search(s)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in {"min", "mins", "m"}:
            time_min = val
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            time_min = val * 60.0
        elif unit in {"d", "day", "days"}:
            time_min = val * 24.0 * 60.0
        else:
            time_min = np.nan
    else:
        time_min = np.nan

    return dict(
        solvent=solvent,
        EAN=EAN,
        protein_mgml=protein_mgml,
        batch_tag=batch_tag,
        batch_id=batch_id,
        temp_C=temp_C,
        temp_profile=temp_profile,
        time_min=time_min,
    )

def _group_key_from_params(p: Dict[str, Any]) -> str:
    """Build a row-group label used for mosaic sorting."""
    solvent = p.get("solvent", "UNK")
    ean = p.get("EAN", np.nan)
    temp_prof = p.get("temp_profile", "")
    tmin = p.get("time_min", np.nan)

    # base group name
    if solvent == "EAN" and np.isfinite(ean):
        base = f"EAN{ean:g}"
    else:
        base = solvent

    # add condition suffix only if present (keeps concentration datasets clean)
    cond = []
    if isinstance(temp_prof, str) and temp_prof:
        cond.append(f"T{temp_prof}")
    if np.isfinite(tmin):
        cond.append(f"t{int(round(tmin))}min")
    if cond:
        return base + "_" + "_".join(cond)
    return base

def _protein_label(p: Dict[str, Any]) -> str:
    """Column label/title for a tile: 5mg / 7.5mg / NAb for batch-b."""
    mg = p.get("protein_mgml", np.nan)
    bt = str(p.get("batch_tag", "")).lower()
    if bt == "b":
        return "NAb"
    if np.isfinite(mg):
        # keep 7.5 formatting
        if abs(mg - round(mg)) < 1e-6:
            return f"{int(round(mg))}mg"
        return f"{mg:g}mg"
    return "NA"

def _sort_key_for_item(item: Dict[str, Any]) -> Tuple:
    """Sort within a group by protein mg/mL then batch."""
    p = item["params"]
    mg = p.get("protein_mgml", np.nan)
    mg_key = mg if np.isfinite(mg) else 1e9
    bt = str(p.get("batch_tag","")).lower()
    bt_key = 1 if bt == "b" else 0
    return (mg_key, bt_key, str(item["stem"]))

def _read_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 2-column .dat file (q, I). Ignores comment/header lines."""
    arr = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!") or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                q = float(parts[0]); I = float(parts[1])
            except ValueError:
                continue
            arr.append((q, I))
    if not arr:
        return np.array([]), np.array([])
    a = np.array(arr, float)
    return a[:,0], a[:,1]

def _build_mosaic_from_data(
    items: List[Dict[str, Any]],
    out_path: Path,
    annotated: bool,
    ncols: int,
    dpi: int,
    q_half_width: float,
    compact: bool = False,
):
    """Create a grouped mosaic figure from .dat data.

    Layout rules:
      - Items are assumed sorted by 'group' then protein.
      - Each group starts on a new row (col=0).
      - Within a group, items are wrapped every ncols panels (so "4 per row" works).
      - A group label is drawn on the left of the first row for that group.
      - All panels share the same x-range (shared x-axis).
    """
    if not items:
        return None

    ncols = max(1, int(ncols))

    # Determine a shared x-range based on peak1 q (preferred) or global q-range fallback
    qmins = []
    qmaxs = []
    for it in items:
        q1 = it.get("peak1_q", np.nan)
        if np.isfinite(q1):
            qmins.append(q1 - q_half_width)
            qmaxs.append(q1 + q_half_width)
    if qmins:
        xmin = float(min(qmins)); xmax = float(max(qmaxs))
    else:
        # fallback: compute from data
        qs = []
        for it in items:
            q, _ = _read_dat(it["dat_path"])
            if q.size:
                qs.append(q)
        if qs:
            allq = np.concatenate(qs)
            xmin = float(np.nanmin(allq)); xmax = float(np.nanmax(allq))
        else:
            xmin, xmax = 0.0, 1.0

    # ---- Build grouped rows (each group starts at col 0) ----
    rows: List[List[Optional[Dict[str, Any]]]] = []
    row_group_labels: List[str] = []
    cur_group = None
    cur_buf: List[Dict[str, Any]] = []

    def flush_group(gname: str, buf: List[Dict[str, Any]]):
        if not buf:
            return
        # chunk into rows of ncols
        first = True
        for k in range(0, len(buf), ncols):
            chunk = buf[k:k+ncols]
            pad = [None] * (ncols - len(chunk))
            rows.append(chunk + pad)
            row_group_labels.append(gname if first else "")
            first = False

    for it in items:
        g = it.get("group", "UNK")
        if cur_group is None:
            cur_group = g
            cur_buf = [it]
        elif g == cur_group:
            cur_buf.append(it)
        else:
            flush_group(cur_group, cur_buf)
            cur_group = g
            cur_buf = [it]
    flush_group(cur_group, cur_buf)

    nrows = len(rows)
    fig_w = 4.4 * ncols
    fig_h = 3.3 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            it = rows[r][c]
            if it is None:
                ax.axis("off")
                continue

            q, I = _read_dat(it["dat_path"])
            if q.size == 0:
                ax.text(0.5, 0.5, "NO DATA", ha="center", va="center")
                ax.set_xlim(xmin, xmax)
                continue

            m = (q >= xmin) & (q <= xmax)
            qv = q[m]; Iv = I[m]
            if qv.size == 0:
                qv, Iv = q, I

            ax.plot(qv, Iv, lw=1.2)

            # peak markers
            q1 = it.get("peak1_q", np.nan)
            fwhm = it.get("peak1_fwhm", np.nan)
            if np.isfinite(q1):
                ax.axvline(q1, lw=1.0, alpha=0.9)
            if np.isfinite(q1) and np.isfinite(fwhm) and fwhm > 0:
                ax.axvspan(q1 - 0.5*fwhm, q1 + 0.5*fwhm, alpha=0.18)

            # titles/annotations
            p = it["params"]
            col_title = _protein_label(p)
            if not compact:
                ax.set_title(col_title, fontsize=13, pad=6)

            if annotated:
                binA = it.get("peak1_binA", np.nan)
                dom = it.get("peak1_domain_nm_meas", np.nan)
                txt = []
                txt.append(str(it["stem"]))
                if np.isfinite(q1):
                    txt.append(f"peak1 q={q1:.4f}")
                if np.isfinite(binA):
                    txt.append(f"bin Δq={binA:.6f} Å⁻¹")
                if np.isfinite(fwhm):
                    txt.append(f"FWHM={fwhm:.6f} Å⁻¹")
                if np.isfinite(dom):
                    txt.append(f"domain≈{dom:.1f} nm")
                ax.text(0.98, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", ha="right", fontsize=ANNOT_FONTSIZE,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
            else:
                txt = str(it["stem"])
                if np.isfinite(q1):
                    txt += f"  q1={q1:.4f}"
                ax.text(0.98, 0.98, txt, transform=ax.transAxes, va="top", ha="right", fontsize=ANNOT_FONTSIZE,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

            ax.set_xlim(xmin, xmax)

            # group label (first column of that group's first row)
            if row_group_labels[r] and c == 0:
                ax.text(-0.35, 0.5, row_group_labels[r], transform=ax.transAxes,
                        va="center", ha="right", fontsize=16)

            # axis labels / ticks
            if not compact:
                # keep clean but readable
                ax.set_xlabel("q (Å$^{-1}$)")
                ax.set_ylabel("I(q), a.u.")
            else:
                # compact mosaic: remove axis titles and tick labels for a tight grid
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", which="both", labelsize=8, length=3, width=0.8)
                ax.set_yticks([])
                ax.tick_params(axis="y", which="both", length=0)

    # Layout
    if compact:
        # tighter spacing between panels (including vertical gap)
        fig.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.06, wspace=0.01, hspace=0.01)
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path
def build_crystalline_mosaics(cryst_dir: Path, df: pd.DataFrame, data_dir: Path):
    """Create two mosaics in cryst_dir: annotated and simplified, using crystalline_present_true rows in df."""
    cryst_dir = Path(cryst_dir)
    data_dir = Path(data_dir)

    if df is None or df.empty:
        return

    # filter crystalline present
    if "crystalline_present" not in df.columns:
        return

    dfc = df[df["crystalline_present"] == True].copy()
    if dfc.empty:
        return

    items: List[Dict[str, Any]] = []
    for _, r in dfc.iterrows():
        stem = str(r.get("file", ""))
        dat_path = data_dir / f"{stem}.dat"
        if not dat_path.exists():
            # try case-insensitive match
            cand = next((p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower()==".dat" and p.stem==stem), None)
            if cand is None:
                continue
            dat_path = cand

        params = _parse_name_params(stem)
        items.append(dict(
            stem=stem,
            dat_path=dat_path,
            params=params,
            group=_group_key_from_params(params),
            peak1_q=pd.to_numeric(r.get("peak1_q", np.nan), errors="coerce"),
            peak1_fwhm=pd.to_numeric(r.get("peak1_fwhm_Ainv", r.get("peak1_fwhm", np.nan)), errors="coerce"),
            peak1_binA=pd.to_numeric(r.get("peak1_binA", np.nan), errors="coerce"),
            peak1_domain_nm_meas=pd.to_numeric(r.get("peak1_domain_nm_meas", np.nan), errors="coerce"),
        ))

    if not items:
        return

    # sort by group, then by protein
    items.sort(key=lambda it: (_group_key_from_params(it["params"]), _sort_key_for_item(it)))

    # Build a single mosaic per group (wrapped to MOSAIC_NCOLS columns), and also a combined mosaic.
    # Here we output a combined mosaic across all crystalline samples, keeping grouping order.
    # The user can filter by group by running on a subset folder if desired.
    out_annot = cryst_dir / "crystalline_present_true_mosaic_annot.png"
    out_annot_compact = cryst_dir / "crystalline_present_true_mosaic_annot_compact.png"
    out_simple = cryst_dir / "crystalline_present_true_mosaic_simple.png"

    _build_mosaic_from_data(
        items,
        out_path=out_annot,
        annotated=True,
        ncols=MOSAIC_NCOLS,
        dpi=MOSAIC_DPI,
        q_half_width=MOSAIC_Q_HALF_WIDTH,
    )

    # Extra compact version: no axis labels/tick labels/titles, tighter spacing
    _build_mosaic_from_data(
        items,
        out_path=out_annot_compact,
        annotated=True,
        ncols=MOSAIC_NCOLS,
        dpi=MOSAIC_DPI,
        q_half_width=MOSAIC_Q_HALF_WIDTH,
        compact=True,
    )
    _build_mosaic_from_data(
        items,
        out_path=out_simple,
        annotated=False,
        ncols=MOSAIC_NCOLS,
        dpi=MOSAIC_DPI,
        q_half_width=MOSAIC_Q_HALF_WIDTH,
    )
    return out_annot, out_simple
# ================== Runner ==================
def run_step2(data_dir: Path):
    data_dir = Path(data_dir)
    out_dir = data_dir / ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # folder for plots of crystalline samples (first peak)
    cryst_dir = out_dir / 'crystalline_present_true'
    cryst_dir.mkdir(parents=True, exist_ok=True)

    # read all .dat (case-insensitive)
    dats = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == '.dat'])

    if not dats:
        print("No .dat files found."); return None

    q_list, I_list, E_list, tags = [], [], [], []
    for p in dats:
        q,I,E = read_dat_flexible(p)
        if q.size == 0: 
            print(f"[WARN] empty/unreadable: {p.name}")
            continue
        q_list.append(q); I_list.append(I); E_list.append(E); tags.append(p.stem)

    if not q_list:
        print("All files empty/unreadable."); return None

    qg = build_common_grid(q_list)
    if qg is None:
        qh_list, Ih_list, Eh_list = q_list, I_list, E_list
        dq_bin = float('nan')
    else:
        Ih_list = [interp_to_grid(q,I,qg) for q,I in zip(q_list,I_list)]
        Eh_list = [interp_to_grid(q,E,qg) for q,E in zip(q_list,E_list)]
        qh_list = [qg]*len(Ih_list)
        dq_bin = float(np.median(np.diff(qg)))

    # Reference peaks (using harmonized intensities)
    q_centers = pick_reference_peaks(qh_list, Ih_list, q_band=Q_BAND, smooth_frac=SMOOTH_WIN_FRAC,
                                     max_peaks=MAX_PEAKS, merge_dq=PEAK_MERGE_DQ)

    rows = []
    for q, I, tag in zip(qh_list, Ih_list, tags):
        feats = dict(file=tag, dataset_id=str(data_dir.name), data_dir=str(data_dir))
        feats.update(crystal_features(q, I, q_centers, q_band=Q_BAND,
                                      abs_gate_frac=MIN_ABS_PEAK_FRAC,
                                      snr_gate=SNR_THRESHOLD,
                                      bin_dq=dq_bin))
        rows.append(feats)

        # If sample is crystalline, make a figure for the FIRST peak
        try:
            if feats.get('crystalline_present', False):
                q1 = feats.get('peak1_q', np.nan)
                fwhm1 = feats.get('peak1_fwhm', np.nan)
                domain_nm = feats.get('peak1_domain_nm_meas', np.nan)
                # choose window around first peak: ± max(3*FWHM, 0.02 Å^-1)
                if np.isfinite(q1):
                    delta = 0.02
                    if np.isfinite(fwhm1) and fwhm1 > 0:
                        delta = max(delta, 3.0 * fwhm1)
                    qmin = max(float(np.nanmin(q)), q1 - delta)
                    qmax = min(float(np.nanmax(q)), q1 + delta)
                    mask = (q >= qmin) & (q <= qmax) & np.isfinite(I)
                else:
                    # fallback: show central region
                    mask = np.isfinite(I)

                q_plot = q[mask]; I_plot = I[mask]

                fig, ax = plt.subplots(figsize=(5,3.5), dpi=140)
                ax.plot(q_plot, I_plot, 'o', ms=4, mec='none', alpha=0.8)
                ax.set_xlabel('q (Å$^{-1}$)')
                ax.set_ylabel('I (arb)')
                ax.set_title(f'{tag} — peak1 q={q1:.4g}')
                # annotate values
                text_lines = []
                text_lines.append(f'bin Δq = {dq_bin:.4g}' if np.isfinite(dq_bin) else 'bin Δq = N/A')
                text_lines.append(f'FWHM = {fwhm1:.4g} Å⁻¹' if np.isfinite(fwhm1) else 'FWHM = N/A')
                text_lines.append(f'domain ≈ {domain_nm:.4g} nm' if np.isfinite(domain_nm) else 'domain = N/A')
                ax.text(0.02, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                        va='top', ha='left', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                ax.set_xlim(q_plot.min() - 0.002 if q_plot.size else qmin, q_plot.max() + 0.002 if q_plot.size else qmax)
                # mark the peak center and FWHM interval if available
                if np.isfinite(q1):
                    ax.axvline(q1, color='red', alpha=0.7, lw=1)
                if np.isfinite(fwhm1) and np.isfinite(q1):
                    ax.axvspan(q1 - 0.5*fwhm1, q1 + 0.5*fwhm1, color='red', alpha=0.15)
                fig.tight_layout()
                fname_annot = cryst_dir / f'{tag}_peak1_annot.png'
                fig.savefig(fname_annot)
                plt.close(fig)

                # simplified version (minimal annotation)
                fig2, ax2 = plt.subplots(figsize=(5,3.5), dpi=140)
                ax2.plot(q_plot, I_plot, 'o', ms=4, mec='none', alpha=0.8)
                ax2.set_xlabel('q (Å$^{-1}$)')
                ax2.set_ylabel('I (arb)')
                # minimal label: sample + peak1 position
                label = tag
                if np.isfinite(q1):
                    label = f"{tag} (q1={q1:.4g})"
                ax2.text(0.02, 0.95, label, transform=ax2.transAxes,
                         va='top', ha='left', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                ax2.set_xlim(q_plot.min() - 0.002 if q_plot.size else qmin,
                             q_plot.max() + 0.002 if q_plot.size else qmax)
                if np.isfinite(q1):
                    ax2.axvline(q1, color='red', alpha=0.6, lw=1)
                if np.isfinite(fwhm1) and np.isfinite(q1) and fwhm1 > 0:
                    ax2.axvspan(q1 - 0.5*fwhm1, q1 + 0.5*fwhm1, color='red', alpha=0.10)

                fig2.tight_layout()
                fname_simple = cryst_dir / f'{tag}_peak1_simple.png'
                fig2.savefig(fname_simple)
                plt.close(fig2)
        except Exception as e:
            print(f"[plot] {tag}: {e}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / 'useful_parameters.csv'
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # build mosaic figures for crystalline samples
    try:
        build_crystalline_mosaics(cryst_dir, df, data_dir)
    except Exception as e:
        print(f"[mosaic] {e}")

    return df

from pathlib import Path

if __name__ == "__main__":
    # use configured target directory from the top of the file
    DATA_DIR = Path(TARGET_DATA_DIR).expanduser().resolve()
    print(f"使用的数据目录 / data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        print("目录不存在。请将 .dat 文件放到该路径，或修改 TARGET_DATA_DIR。")
    else:
        dats = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == '.dat'])
        if dats:
            print(f"Found {len(dats)} .dat file(s):")
            for p in dats[:50]:
                print(" -", p.name)
            if len(dats) > 50:
                print(f" ... and {len(dats)-50} more")
        else:
            print("No .dat files found in that folder. 把你的 .dat 文件放在该目录，例如：")
            print(f"  {DATA_DIR}\\your_file.dat")
    run_step2(DATA_DIR)

