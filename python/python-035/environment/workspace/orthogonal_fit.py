import numpy as np
from constants import TINY




def orthopoly_construct(n, weight_func, a, b, n_quad=64):

    from quadrature_engine import legendre_gauss_rule
    nodes, weights = legendre_gauss_rule(n_quad)
    

    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    phys_nodes = nodes * scale + shift
    phys_weights = weights * scale
    
    alphas = np.zeros(n)
    betas = np.zeros(n)
    

    p0_sq = np.sum(phys_weights * weight_func(phys_nodes))
    if p0_sq < TINY:
        p0_sq = TINY
    norm0 = 1.0 / np.sqrt(p0_sq)
    


    p_prev = np.ones(n_quad) * norm0
    p_curr = np.zeros(n_quad)
    
    for k in range(n):

        num = np.sum(phys_weights * weight_func(phys_nodes) * phys_nodes * p_prev ** 2)
        den = np.sum(phys_weights * weight_func(phys_nodes) * p_prev ** 2)
        alphas[k] = num / max(den, TINY)
        

        if k == 0:
            p_curr = (phys_nodes - alphas[k]) * p_prev
        else:
            p_curr = (phys_nodes - alphas[k]) * p_prev - betas[k - 1] * p_prev_prev
        

        num_beta = np.sum(phys_weights * weight_func(phys_nodes) * p_curr ** 2)
        den_beta = np.sum(phys_weights * weight_func(phys_nodes) * p_prev ** 2)
        if k < n - 1:
            betas[k + 1] = num_beta / max(den_beta, TINY)
        
        p_prev_prev = p_prev.copy()
        p_prev = p_curr.copy()
    
    return alphas, betas, norm0


def orthopoly_eval(alphas, betas, norm0, x, k_max):
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)
    n = len(x)
    
    vals = np.zeros((n, k_max + 1))
    vals[:, 0] = norm0
    
    if k_max >= 1:
        vals[:, 1] = (x - alphas[0]) * vals[:, 0]
        for k in range(1, k_max):
            if k < len(alphas) and k < len(betas):
                vals[:, k + 1] = (x - alphas[k]) * vals[:, k] - betas[k] * vals[:, k - 1]
    
    if scalar_input:
        return vals[0, :]
    return vals





def clenshaw_chebyshev(coeffs, x):
    x = float(x)
    x = np.clip(x, -1.0, 1.0)
    n = len(coeffs) - 1
    
    b_prev2 = 0.0
    b_prev1 = 0.0
    
    for k in range(n, 0, -1):
        b = 2.0 * x * b_prev1 - b_prev2 + coeffs[k]
        b_prev2 = b_prev1
        b_prev1 = b
    
    return coeffs[0] + x * b_prev1 - b_prev2





def orthogonal_background_fit(mass_bins, counts, degree=4, weight_func=None, a=80.0, b=170.0):
    mass_bins = np.asarray(mass_bins, dtype=float)
    counts = np.asarray(counts, dtype=float)
    
    if weight_func is None:
        weight_func = lambda x: np.ones_like(np.atleast_1d(x))
    
    alphas, betas, norm0 = orthopoly_construct(degree, weight_func, a, b)
    

    P_vals = orthopoly_eval(alphas, betas, norm0, mass_bins, degree)
    
    coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        num = np.sum(counts * P_vals[:, k])
        den = np.sum(P_vals[:, k] ** 2)
        if den > TINY:
            coeffs[k] = num / den
    
    fitted = P_vals @ coeffs
    
    return coeffs, (alphas, betas, norm0), fitted


def predict_background(mass, coeffs, ortho_params):
    alphas, betas, norm0 = ortho_params
    P_vals = orthopoly_eval(alphas, betas, norm0, mass, len(coeffs) - 1)
    if P_vals.ndim == 1:
        return float(np.dot(P_vals, coeffs))
    return P_vals @ coeffs





def extract_signal(mass_bins, counts, fitted_background):
    signal = np.maximum(counts - fitted_background, 0.0)
    significance = np.zeros_like(signal)
    for i in range(len(signal)):
        if fitted_background[i] > 1.0:
            significance[i] = signal[i] / np.sqrt(fitted_background[i])
    return signal, significance





def analyze_mass_spectrum(mass_bins, counts, background_degree=4):
    a = np.min(mass_bins)
    b = np.max(mass_bins)
    
    coeffs, ortho_params, fitted = orthogonal_background_fit(
        mass_bins, counts, degree=background_degree, a=a, b=b
    )
    
    signal, significance = extract_signal(mass_bins, counts, fitted)
    

    peak_idx = np.argmax(significance)
    peak_mass = mass_bins[peak_idx]
    peak_significance = significance[peak_idx]
    

    total_signal = np.sum(signal)
    total_background = np.sum(fitted)
    
    return {
        "mass_bins": mass_bins,
        "observed": counts,
        "background_fit": fitted,
        "signal": signal,
        "significance": significance,
        "peak_mass": peak_mass,
        "peak_significance": peak_significance,
        "total_signal": total_signal,
        "total_background": total_background,
        "s_over_sqrt_b": total_signal / np.sqrt(total_background) if total_background > 1.0 else 0.0,
        "polynomial_coeffs": coeffs,
        "ortho_params": ortho_params,
    }
