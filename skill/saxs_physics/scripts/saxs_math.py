
import numpy as np
from typing import Tuple, Dict, Optional, List

# =============================================================================
# 1. Guinier Analysis (Rg, I0)
# =============================================================================

def calculate_guinier(q: np.ndarray, I: np.ndarray, q_range: Tuple[float, float] = (0.02, 0.05)) -> Dict[str, float]:
    """
    Performs a simple Guinier fit: ln(I) = ln(I0) - (Rg^2 / 3) * q^2
    
    Args:
        q: Scattering vector array (A^-1)
        I: Intensity array
        q_range: (min_q, max_q) to use for the linear fit.
    
    Returns:
        Dict with 'Rg', 'I0', 'r_squared', 'quality'
    """
    # Filter range
    mask = (q >= q_range[0]) & (q <= q_range[1]) & (I > 0)
    q_sel = q[mask]
    I_sel = I[mask]
    
    if len(q_sel) < 5:
        return {"Rg": np.nan, "I0": np.nan, "r_squared": 0.0, "quality": 0.0}

    # Linear Regression: y = mx + c
    # y = ln(I), x = q^2
    # slope m = -Rg^2 / 3  => Rg = sqrt(-3m)
    # intercept c = ln(I0) => I0 = exp(c)
    
    x = q_sel ** 2
    y = np.log(I_sel)
    
    # Simple unweighted least squares for speed/simplicity
    # (The original script used weighted, but for a general skill, simple LS is often sufficient for initial screening)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Calculate R-squared
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Extract Physics
    if m >= 0:
        # Positive slope = aggregation or structure factor effect (non-particulate)
        Rg = np.nan
    else:
        Rg = np.sqrt(-3 * m)
        
    I0 = np.exp(c)

    return {
        "Rg": Rg,
        "I0": I0,
        "r_squared": r2,
        "fit_slope": m,
        "points_used": len(q_sel)
    }

# =============================================================================
# 2. Crystallinity Analysis (CI)
# =============================================================================

def calculate_crystallinity(q: np.ndarray, I: np.ndarray, 
                           peak_search_range: Tuple[float, float] = (0.02, 0.35)) -> Dict[str, float]:
    """
    Estimates Crystallinity Index (CI) by separating sharp peaks from broad baseline.
    
    Simplified Logic:
    1. Estimate baseline (amorphous) using a rolling median/percentile filter.
    2. Subtract baseline -> peaks.
    3. CI = Area(Peaks) / Total Area.
    """
    
    # Work only in valid range
    mask = (q >= peak_search_range[0]) & (q <= peak_search_range[1]) & (np.isfinite(I))
    q_sub = q[mask]
    I_sub = I[mask]
    
    if len(q_sub) < 10:
        return {"CI": 0.0, "num_peaks": 0}
    
    # 1. Baseline Estimation (Running minimum/percentile)
    # Window size ~ 0.05 A^-1 (arbitrary heuristic for SAXS peaks)
    # Convert window from q-units to indices
    dq_avg = np.mean(np.diff(q_sub))
    window_pts = int(0.05 / dq_avg) 
    window_pts = max(5, window_pts)
    
    # Simple baseline: minimum filter (erosion) followed by smoothing
    from scipy.ndimage import minimum_filter, gaussian_filter1d
    
    # Erosion to remove spikes
    baseline_est = minimum_filter(I_sub, size=window_pts)
    # Smooth the result to get a clean amorphous curve
    baseline_smooth = gaussian_filter1d(baseline_est, sigma=window_pts/2)
    
    # Ensure baseline !> signal
    baseline_final = np.minimum(I_sub, baseline_smooth)
    
    # 2. Integration
    # Area Total = Integral(I)
    # Area Amorphous = Integral(Baseline)
    # Area Crystalline = Total - Amorphous
    
    total_area = np.trapz(I_sub, q_sub)
    amorphous_area = np.trapz(baseline_final, q_sub)
    crystalline_area = total_area - amorphous_area
    
    if total_area <= 0:
        return {"CI": 0.0}
        
    CI = crystalline_area / total_area
    
    # 3. Peak Finding (Bonus: Find biggest peakq)
    residuals = I_sub - baseline_final
    try:
        from scipy.signal import find_peaks
        # Height threshold: at least 5% of signal max
        thresh = 0.05 * np.max(I_sub)
        peaks, _ = find_peaks(residuals, height=thresh, distance=window_pts)
        num_peaks = len(peaks)
        
        first_peak_q = q_sub[peaks[0]] if len(peaks) > 0 else np.nan
    except ImportError:
        num_peaks = -1
        first_peak_q = np.nan

    return {
        "CI": float(CI),
        "total_area": float(total_area),
        "crystalline_area": float(crystalline_area),
        "num_peaks": int(num_peaks),
        "first_peak_q": float(first_peak_q)
    }
