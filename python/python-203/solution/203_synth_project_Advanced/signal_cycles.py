"""
signal_cycles.py -- Signal Processing and Statistical Cycle Analysis
=====================================================================
Implements bandpass filtering, cycle extraction, AUC-based discriminability
metrics, and mixed-effects statistical testing for analyzing stochastic
signal characteristics in the UQ context.

Pipeline:
1. Extract rhythmic cycles from stochastic time series via bandpass filtering
2. Compute signal power for each frequency band
3. Calculate discriminability metric (AUC) between different UQ scenarios
4. Fit mixed-effects models to test significance of stochastic parameters

Seed references:
  - 1053_cnnp-lab_DiminishedRhythmsPathology: cycle extraction, AUC, mixed-effects
  - 883_polygon_average: iterative averaging/smoothing
"""
import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Butterworth bandpass filter (zero-phase, SOS form)
# ---------------------------------------------------------------------------
def butterworth_bandpass(low_freq: float, high_freq: float,
                         sampling_rate: float, order: int = 2
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth bandpass filter in SOS (second-order sections) form.

    The transfer function of an N-th order Butterworth filter:
        |H(f)|^2 = 1 / (1 + (f/f_c)^{2N})
    where f_c is the cutoff frequency.

    For bandpass: cascade lowpass and highpass Butterworth prototypes.

    Parameters
    ----------
    low_freq : float
        Lower cutoff frequency (Hz).
    high_freq : float
        Upper cutoff frequency (Hz).
    sampling_rate : float
        Sampling rate (Hz).
    order : int
        Filter order (per section).

    Returns
    -------
    sos : ndarray, shape (n_sections, 6)
        Second-order sections coefficients [b0, b1, b2, a0, a1, a2].
    nyquist : float
        Nyquist frequency.
    """
    nyquist = sampling_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clamp to valid range
    low = max(low, 1e-6)
    high = min(high, 1.0 - 1e-6)

    if low >= high:
        raise ValueError(f"low_freq ({low_freq}) must be < high_freq ({high_freq})")

    # Design Butterworth SOS using bilinear transform
    # For simplicity, implement a direct 2nd-order bandpass section
    # Pre-warp frequencies
    wl = np.tan(np.pi * low / 2.0)
    wh = np.tan(np.pi * high / 2.0)
    bw = wh - wl
    w0 = np.sqrt(wl * wh)

    # Analog prototype: H(s) = bw*s / (s^2 + bw*s + w0^2)
    # Bilinear transform: s = 2*(z-1)/(z+1)
    # Resulting digital SOS coefficients
    a0 = 4.0 + 4.0 * bw + w0 ** 2
    sos = np.array([
        [4.0 * bw / a0, 0.0, -4.0 * bw / a0,   # numerator
         1.0, (2.0 * w0 ** 2 - 8.0) / a0,
         (4.0 - 4.0 * bw + w0 ** 2) / a0]        # denominator
    ])

    return sos, nyquist


def apply_sos_filter(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Apply SOS filter with zero-phase forward-backward filtering.

    Implements the second-order section difference equation:
        y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0

    Forward-backward (filtfilt) doubles the effective order and eliminates
    phase distortion.
    """
    def sos_section(x, section):
        b0, b1, b2, a0, a1, a2 = section
        n = len(x)
        y = np.zeros(n)
        # Forward pass
        for i in range(n):
            y[i] = (b0 * x[i] + b1 * (x[i - 1] if i > 0 else 0) +
                    b2 * (x[i - 2] if i > 1 else 0) -
                    a1 * (y[i - 1] if i > 0 else 0) -
                    a2 * (y[i - 2] if i > 1 else 0)) / a0
        return y

    # Forward-backward (zero-phase)
    y_fwd = sos_section(signal, sos[0])
    y_rev = sos_section(y_fwd[::-1], sos[0])
    return y_rev[::-1]


# ---------------------------------------------------------------------------
# Cycle extraction and power computation
# ---------------------------------------------------------------------------
def extract_cycles(signal: np.ndarray, sampling_rate: float,
                   period_ranges: List[Tuple[float, float]]
                   ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Extract rhythmic cycles from a time series using narrow bandpass filtering.

    For each period range [T_low, T_high]:
        f_low = 1/T_high, f_high = 1/T_low
        cycle_i = bandpass(signal, f_low, f_high)

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input time series.
    sampling_rate : float
        Sampling rate (Hz).
    period_ranges : list of tuples
        Each (T_low, T_high) in seconds.

    Returns
    -------
    filtered_signals : list of ndarrays
        Bandpass-filtered signals for each period range.
    power : list of floats
        RMS power of each filtered signal.
    """
    cycles = []
    powers = []

    for T_low, T_high in period_ranges:
        f_low = 1.0 / T_high
        f_high = 1.0 / T_low

        try:
            sos, _ = butterworth_bandpass(f_low, f_high, sampling_rate)
            filtered = apply_sos_filter(signal, sos)
            cycles.append(filtered)
            rms = np.sqrt(np.mean(filtered ** 2))
            powers.append(float(rms))
        except (ValueError, FloatingPointError):
            cycles.append(np.zeros_like(signal))
            powers.append(0.0)

    return cycles, powers


def compute_impute_missing(signal: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """
    Impute missing values (NaN) in a time series using linear interpolation
    with small noise injection for non-zero gaps.

    For isolated NaN: replace with mean of neighbors
    For short gaps (<= max_gap): linear interpolation + small noise
    For long gaps (> max_gap): replace with global mean

    Parameters
    ----------
    signal : ndarray
        Input signal with possible NaN values.
    max_gap : int
        Maximum gap length for interpolation.

    Returns
    -------
    imputed : ndarray
        Signal with NaN values filled.
    """
    imputed = signal.copy()
    nan_mask = np.isnan(imputed)

    if not np.any(nan_mask):
        return imputed

    n = len(imputed)
    global_mean = np.nanmean(imputed)

    # Find contiguous NaN gaps
    gaps = []
    start = None
    for i in range(n):
        if nan_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                gaps.append((start, i))
                start = None
    if start is not None:
        gaps.append((start, n))

    rng = np.random.RandomState(42)

    for gap_start, gap_end in gaps:
        gap_len = gap_end - gap_start
        if gap_len == 1:
            # Single NaN: average of neighbors
            left = imputed[gap_start - 1] if gap_start > 0 else global_mean
            right = imputed[gap_end] if gap_end < n else global_mean
            imputed[gap_start] = 0.5 * (left + right)
        elif gap_len <= max_gap:
            # Short gap: linear interpolation
            left = imputed[gap_start - 1] if gap_start > 0 else global_mean
            right = imputed[gap_end] if gap_end < n else global_mean
            interp = np.linspace(left, right, gap_len + 2)[1:-1]
            noise = 0.01 * np.std(imputed[~nan_mask]) * rng.randn(gap_len)
            imputed[gap_start:gap_end] = interp + noise
        else:
            # Long gap: global mean
            imputed[gap_start:gap_end] = global_mean

    return imputed


# ---------------------------------------------------------------------------
# AUC-based discriminability metric
# ---------------------------------------------------------------------------
def compute_auc(scores_group1: np.ndarray, scores_group2: np.ndarray) -> float:
    """
    Compute the Area Under the ROC Curve (AUC) between two groups.

    The AUC (equivalent to the Mann-Whitney U statistic normalized):
        AUC = (1/(n1*n2)) sum_{i,j} I(x_i > y_j)
    where x_i in group1, y_j in group2, I is the indicator function.

    AUC = 0.5: no discriminability (random)
    AUC = 1.0: perfect separation
    AUC = 0.0: perfect anti-separation

    Parameters
    ----------
    scores_group1 : ndarray
        Scores for group 1 (e.g., SOZ region power).
    scores_group2 : ndarray
        Scores for group 2 (e.g., non-SOZ region power).

    Returns
    -------
    auc : float
        AUC value in [0, 1].
    """
    n1 = len(scores_group1)
    n2 = len(scores_group2)
    if n1 == 0 or n2 == 0:
        return 0.5

    # Mann-Whitney U statistic
    u_sum = 0.0
    for x in scores_group1:
        for y in scores_group2:
            if x > y:
                u_sum += 1.0
            elif x == y:
                u_sum += 0.5

    return u_sum / (n1 * n2)


# ---------------------------------------------------------------------------
# Mixed-effects model (simplified)
# ---------------------------------------------------------------------------
def fit_mixed_effects(y: np.ndarray, X: np.ndarray, groups: np.ndarray
                      ) -> dict:
    """
    Fit a simplified linear mixed-effects model:
        y = X * beta + Z * u + epsilon
    where:
        beta: fixed effects (estimated by GLS)
        u ~ N(0, sigma_u^2 * I): random intercepts per group
        epsilon ~ N(0, sigma^2 * I): residual

    Uses a profile likelihood approach:
    1. Estimate variance components by REML
    2. Estimate fixed effects by GLS given variance components

    Parameters
    ----------
    y : ndarray, shape (n,)
        Response variable.
    X : ndarray, shape (n, p)
        Fixed effects design matrix.
    groups : ndarray, shape (n,)
        Group labels for random intercepts.

    Returns
    -------
    result : dict
        'beta': fixed effect estimates
        'sigma_u': random effect std
        'sigma_e': residual std
        'log_likelihood': log-likelihood value
        'aic': Akaike Information Criterion
    """
    n, p = X.shape
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Build random effects design matrix Z
    Z = np.zeros((n, n_groups))
    for i, g in enumerate(unique_groups):
        Z[groups == g, i] = 1.0

    # Initialize variance components
    sigma_u = 1.0
    sigma_e = 1.0

    # EM iteration for variance components (simplified)
    for _ in range(20):
        # V = sigma_u^2 * Z Z^T + sigma_e^2 * I
        V = sigma_u ** 2 * Z @ Z.T + sigma_e ** 2 * np.eye(n)
        V_inv = np.linalg.inv(V + 1e-10 * np.eye(n))

        # GLS estimate of beta
        XtV = X.T @ V_inv
        beta = np.linalg.solve(XtV @ X + 1e-10 * np.eye(p), XtV @ y)

        # Residuals
        r = y - X @ beta

        # Update variance components (simplified REML-like)
        sigma_e = np.sqrt(max(np.mean(r ** 2) * 0.8, 1e-10))
        Zu = Z.T @ r
        sigma_u = np.sqrt(max(np.mean(Zu ** 2) / n_groups - sigma_e ** 2, 1e-10))

    # Final V and log-likelihood
    V = sigma_u ** 2 * Z @ Z.T + sigma_e ** 2 * np.eye(n)
    try:
        sign, logdet = np.linalg.slogdet(V + 1e-10 * np.eye(n))
    except np.linalg.LinAlgError:
        logdet = 0.0
    r = y - X @ beta
    V_inv = np.linalg.inv(V + 1e-10 * np.eye(n))
    log_lik = -0.5 * (n * np.log(2 * np.pi) + logdet + r @ V_inv @ r)

    # AIC
    n_params = p + 2  # beta + 2 variance components
    aic = -2 * log_lik + 2 * n_params

    return {
        'beta': beta,
        'sigma_u': float(sigma_u),
        'sigma_e': float(sigma_e),
        'log_likelihood': float(log_lik),
        'aic': float(aic)
    }


# ---------------------------------------------------------------------------
# Polygon averaging iteration (for signal smoothing)
# ---------------------------------------------------------------------------
def polygon_average_iteration(signal_2d: np.ndarray, n_iter: int = 10
                              ) -> np.ndarray:
    """
    Apply the polygon averaging iteration to a 2D signal trace.

    Each iteration:
    1. Replace each point by the average of itself and its neighbor
    2. Subtract the centroid
    3. Scale to max-norm = 1

    This converges to an ellipse-like shape (Elmachtoub & Van Loan, 2010),
    providing a smooth representation of the dominant oscillation mode.

    Parameters
    ----------
    signal_2d : ndarray, shape (n, 2)
        2D polygon vertices (e.g., [cos(signal), sin(signal)]).
    n_iter : int
        Number of averaging iterations.

    Returns
    -------
    result : ndarray, shape (n, 2)
        Smoothed polygon vertices.
    """
    p = signal_2d.copy()
    n = len(p)

    for _ in range(n_iter):
        # Step 1: neighbor averaging
        p2 = np.zeros_like(p)
        for i in range(n):
            p2[i] = 0.5 * (p[i] + p[(i + 1) % n])

        # Step 2: subtract centroid
        centroid = np.mean(p2, axis=0)
        p2 -= centroid

        # Step 3: max-norm scaling (prevent degenerate normalization)
        for d in range(2):
            max_val = np.max(np.abs(p2[:, d]))
            if max_val > 1e-14:
                p2[:, d] /= max_val

        p = p2

    return p
