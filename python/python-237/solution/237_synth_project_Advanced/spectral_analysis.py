"""
Spectral analysis of the Wilson-Dirac operator.

Adapted from:
  - 543_histogramize (histogram construction)
  - 800_newton_interp_1d (Newton polynomial interpolation)

Key formulas:
    Banks-Casher relation: <psi-bar psi> = -pi * rho(0) / V
    Newton interpolation: p(x) = c_0 + c_1 (x - x_0) + c_2 (x - x_0)(x - x_1) + ...
"""

import numpy as np
from constants import LatticeParams
from gauge_field import GaugeField
from wilson_dirac import WilsonDiracOperator, SpinorField, _gamma5


def arnoldi_eigenvalues(D: WilsonDiracOperator, n_eig: int = 10,
                         seed: int = 42, which: str = 'SM') -> dict:
    if n_eig < 1:
        raise ValueError(f"n_eig must be >= 1, got {n_eig}")
    rng = np.random.default_rng(seed)
    p = D.p
    max_iter = n_eig * 20
    if max_iter < n_eig:
        max_iter = n_eig * 5
    Q = []
    H = np.zeros((max_iter + 1, max_iter), dtype=complex)
    v = SpinorField(p, zero=False)
    v.data = rng.standard_normal(v.data.shape) + 1j * rng.standard_normal(v.data.shape)
    nrm = np.sqrt(v.norm_sq() + 1e-30)
    v.data /= nrm
    Q.append(v)
    breakdown = False
    j_final = 0
    for j in range(max_iter):
        w = D.apply(Q[j])
        for i in range(j + 1):
            H[i, j] = Q[i].dot(w)
            w.data -= H[i, j] * Q[i].data
        H[j + 1, j] = np.sqrt(w.norm_sq() + 1e-30)
        if H[j + 1, j] < 1e-12:
            breakdown = True
            j_final = j + 1
            break
        v_new = SpinorField(p, zero=False)
        v_new.data = w.data / H[j + 1, j]
        Q.append(v_new)
        j_final = j + 1
    n = j_final
    H_sub = H[:n, :n]
    eigs = np.linalg.eigvals(H_sub)
    if which == 'SM':
        eigs = eigs[np.argsort(np.abs(eigs))]
    elif which == 'LM':
        eigs = eigs[np.argsort(-np.abs(eigs))]
    elif which == 'SR':
        eigs = eigs[np.argsort(eigs.real)]
    elif which == 'LR':
        eigs = eigs[np.argsort(-eigs.real)]
    else:
        raise ValueError(f"Unknown which='{which}'")
    return {
        'eigenvalues': eigs[:n_eig],
        'converged': not breakdown,
        'n_iterations': j_final,
        'breakdown': breakdown,
    }


def spectral_density_histogram(eigenvalues: np.ndarray, n_bins: int = 30,
                                 mode: str = 'real') -> dict:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(eigenvalues) == 0:
        raise ValueError("eigenvalues array is empty")
    if mode == 'real':
        data = eigenvalues.real
    elif mode == 'imag':
        data = eigenvalues.imag
    elif mode == 'abs':
        data = np.abs(eigenvalues)
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    data_min, data_max = float(data.min()), float(data.max())
    if data_max - data_min < 1e-12:
        return {
            'bin_centers': np.array([data_min]),
            'counts': np.array([len(data)]),
            'density': np.array([1.0]),
            'bin_edges': np.array([data_min - 0.5, data_min + 0.5]),
            'mean': data_min,
            'std': 0.0,
        }
    bin_edges = np.linspace(data_min, data_max, n_bins + 1)
    counts, _ = np.histogram(data, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    total = counts.sum()
    density = counts / (total * bin_widths + 1e-30)
    return {
        'bin_centers': bin_centers,
        'counts': counts,
        'density': density,
        'bin_edges': bin_edges,
        'mean': float(data.mean()),
        'std': float(data.std()),
    }


def banks_casher_estimate(eigenvalues: np.ndarray, V: int,
                            lambda_window: float = 0.1) -> dict:
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    if lambda_window <= 0.0:
        raise ValueError(f"lambda_window must be > 0, got {lambda_window}")
    abs_eigs = np.abs(eigenvalues)
    n_in_window = np.sum(abs_eigs < lambda_window)
    rho_0 = n_in_window / (2.0 * lambda_window * V + 1e-30)
    condensate = -np.pi * rho_0 / V
    return {
        'rho_0': float(rho_0),
        'condensate': float(condensate),
        'n_in_window': int(n_in_window),
        'lambda_window': float(lambda_window),
    }


def newton_interp_spectral_function(eigenvalues: np.ndarray,
                                      x_eval: np.ndarray,
                                      func: str = 'resolvent') -> np.ndarray:
    if len(eigenvalues) == 0:
        raise ValueError("eigenvalues array is empty")
    n_eig = len(eigenvalues)
    n_sample = min(n_eig, 20)
    if n_sample < 2:
        return _direct_spectral_eval(eigenvalues, x_eval, func)
    sorted_idx = np.argsort(np.abs(eigenvalues))
    sample_idx = sorted_idx[np.linspace(0, n_eig - 1, n_sample).astype(int)]
    x_nodes = eigenvalues[sample_idx].real
    f_nodes = np.array([_spectral_func_at(eigenvalues, xn, func) for xn in x_nodes])
    n = len(x_nodes)
    c = f_nodes.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denom = x_nodes[i] - x_nodes[i - j]
            if abs(denom) < 1e-14:
                c[i] = 0.0
            else:
                c[i] = (c[i] - c[i - 1]) / denom
    result = np.zeros(len(x_eval), dtype=complex)
    for k, x in enumerate(x_eval):
        val = c[0]
        prod = 1.0
        for j in range(1, n):
            prod *= (x - x_nodes[j - 1])
            val += c[j] * prod
        result[k] = val
    return result


def _spectral_func_at(eigenvalues: np.ndarray, x: complex, func: str) -> complex:
    if func == 'resolvent':
        return complex(np.mean(1.0 / (x - eigenvalues + 1e-14j)))
    elif func == 'mode_sum':
        return complex(np.mean(np.exp(-x * np.abs(eigenvalues) ** 2)))
    elif func == 'level_density_smooth':
        sigma = 0.1 * (eigenvalues.real.max() - eigenvalues.real.min() + 1e-10)
        return complex(np.sum(np.exp(-0.5 * ((x - eigenvalues.real) / sigma) ** 2))
                       / (sigma * np.sqrt(2 * np.pi) * len(eigenvalues)))
    else:
        raise ValueError(f"Unknown func '{func}'")


def _direct_spectral_eval(eigenvalues: np.ndarray, x_eval: np.ndarray,
                           func: str) -> np.ndarray:
    result = np.zeros(len(x_eval), dtype=complex)
    for k, x in enumerate(x_eval):
        result[k] = _spectral_func_at(eigenvalues, x, func)
    return result
