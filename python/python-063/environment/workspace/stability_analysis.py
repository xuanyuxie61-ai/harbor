
import numpy as np

SIGMA = 5.670374419e-8
Q_SOLAR = 1361.0 / 4.0


def poly_eval(coeffs, z):
    result = np.zeros_like(z, dtype=np.complex128)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-12, max_iter=50):
    n = len(coeffs) - 1
    if n <= 0:
        return np.array([], dtype=np.complex128)

    denom = max(np.abs(coeffs[-1]), 1e-300)
    R = 1.0 + np.max(np.abs(coeffs[:-1])) / denom
    roots = R * np.exp(2j * np.pi * np.arange(n) / n)

    for _ in range(max_iter):
        p_vals = poly_eval(coeffs, roots)
        corrections = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            denom_prod = 1.0 + 0j
            for j in range(n):
                if i != j:
                    diff = roots[i] - roots[j]
                    if abs(diff) < 1e-15:
                        diff = 1e-15 * (1.0 + 1.0j)
                    denom_prod *= diff
            corrections[i] = p_vals[i] / denom_prod
        roots -= corrections
        if np.max(np.abs(corrections)) < tol:
            break
    return roots


def faddeev_leverrier(A, max_dim=20):
    n_full = A.shape[0]
    if n_full > max_dim:
        n = max_dim
        A = A[:n, :n]
    else:
        n = n_full

    coeffs = np.zeros(n + 1, dtype=np.float64)
    coeffs[0] = 1.0
    B = np.eye(n)
    for k in range(1, n + 1):
        C = A @ B
        coeffs[k] = -np.trace(C) / k
        B = C + coeffs[k] * np.eye(n)
    return coeffs


def build_ebm_jacobian(n_nodes, diffusion_coeff=0.55, epsilon=0.6, T_eq=288.0):









    pass


def analyze_climate_stability(n_nodes, vertices, T_eq=288.0, **kwargs):
    A = build_ebm_jacobian(n_nodes, **kwargs)


    if n_nodes <= 50:
        eigenvalues = np.linalg.eigvals(A)
    else:
        coeffs = faddeev_leverrier(A)
        eigenvalues = wdk_roots(coeffs)

    max_real = float(np.max(np.real(eigenvalues)))
    if max_real > 0.01:
        stability = 'unstable'
    elif max_real > -0.01:
        stability = 'marginally_stable'
    else:
        stability = 'stable'

    oscillatory = bool(np.any(np.abs(np.imag(eigenvalues)) > 0.01))
    if oscillatory:
        stability += '_oscillatory'

    dom_idx = int(np.argmax(np.real(eigenvalues)))
    return {
        'eigenvalues': eigenvalues,
        'stability_type': stability,
        'dominant_mode': eigenvalues[dom_idx],
        'max_real_part': max_real
    }
