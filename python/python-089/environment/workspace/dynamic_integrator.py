
import numpy as np
from numpy.polynomial.legendre import leggauss


def cauchy_principal_value(f, a, b, x_sing, n=20):
    if n % 2 != 0:
        n += 1
    
    xi, wi = leggauss(n)
    

    cpv = 0.0
    for i in range(n):
        t = 0.5 * ((1.0 - xi[i]) * a + (1.0 + xi[i]) * b)




        denom = t - x_sing
        if abs(denom) < 1e-14:

            dt = (b - a) * 0.5
            t_plus = t + 1e-6 * dt
            t_minus = t - 1e-6 * dt
            val = (f(t_plus) - f(t_minus)) / (2e-6 * dt)
        else:
            val = f(t) / denom
        cpv += wi[i] * val
    
    cpv *= 0.5 * (b - a)
    return cpv


def frequency_response_function(omega, omega_n, zeta, mode_shape=None):
    omega = np.asarray(omega, dtype=float)
    omega_n = np.asarray(omega_n, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    
    if omega_n.ndim == 0:

        denom = omega_n ** 2 - omega ** 2 + 2.0j * zeta * omega_n * omega
        return 1.0 / denom
    else:

        H = np.zeros_like(omega, dtype=complex)
        for m in range(len(omega_n)):
            denom = omega_n[m] ** 2 - omega ** 2 + 2.0j * zeta[m] * omega_n[m] * omega
            if mode_shape is not None:
                modal_amp = mode_shape[m] ** 2
            else:
                modal_amp = 1.0
            H += modal_amp / denom
        return H


def psd_response(omega, omega_n, zeta, psd_input, mode_shape=None):
    H = frequency_response_function(omega, omega_n, zeta, mode_shape)
    if callable(psd_input):
        Sx = psd_input(omega)
    else:
        Sx = np.asarray(psd_input)
    return np.abs(H) ** 2 * Sx


def integrate_psd_cpv(omega_n, zeta, psd_input, omega_max=None, n_quad=200):
    if omega_max is None:
        omega_max = 3.0 * omega_n
    
    delta = omega_n * max(0.01, 2.0 * zeta)
    

    def integrand(omega):
        Sy = psd_response(omega, omega_n, zeta, psd_input)
        return Sy
    

    omega_reg_1 = np.linspace(0, max(0, omega_n - delta), n_quad // 3)
    omega_reg_2 = np.linspace(omega_n + delta, omega_max, n_quad // 3)
    

    omega_sing = np.linspace(omega_n - delta, omega_n + delta, n_quad // 3)
    

    omega_all = np.sort(np.unique(np.concatenate([omega_reg_1, omega_sing, omega_reg_2])))
    omega_all = omega_all[omega_all >= 0]
    
    Sy_all = integrand(omega_all)
    

    sigma_y_sq = np.trapezoid(Sy_all, omega_all) / (2.0 * np.pi)
    

    moment2 = np.trapezoid(omega_all ** 2 * Sy_all, omega_all) / (2.0 * np.pi)
    if sigma_y_sq > 1e-20:
        omega_0 = np.sqrt(moment2 / sigma_y_sq)
    else:
        omega_0 = omega_n
    
    return sigma_y_sq, omega_0


def first_passage_probability(sigma_y, threshold, omega_0, T_duration,
                               method='poisson'):
    if sigma_y <= 1e-14:
        return 0.0 if threshold > 0 else 1.0, 0.0
    
    b = threshold / sigma_y
    
    nu_up = omega_0 / (2.0 * np.pi) * np.exp(-0.5 * b ** 2)
    
    if method == 'poisson':
        P_f = 1.0 - np.exp(-nu_up * T_duration)
    elif method == 'vanmarcke':

        delta_eff = 0.3
        factor = 1.0 - np.exp(-np.sqrt(np.pi / 2.0) * delta_eff * b)
        P_f = 1.0 - np.exp(-nu_up * T_duration * factor)
    else:
        P_f = 1.0 - np.exp(-nu_up * T_duration)
    

    P_f = np.clip(P_f, 0.0, 1.0)
    
    return P_f, nu_up


def modal_superposition_response(omega, K, M, zeta_vec, load_dof,
                                  response_dof, omega_n_modes=None):

    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(K, M)
    except ImportError:

        M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        K_tilde = M_inv_sqrt @ K @ M_inv_sqrt
        eigvals, eigvecs_tilde = np.linalg.eigh(K_tilde)
        eigvecs = M_inv_sqrt @ eigvecs_tilde
    

    omega_m = np.sqrt(np.maximum(eigvals, 0.0))
    
    if omega_n_modes is not None:
        n_modes = min(omega_n_modes, len(omega_m))
        omega_m = omega_m[:n_modes]
        eigvecs = eigvecs[:, :n_modes]
        zeta_vec = zeta_vec[:n_modes]
    

    for m in range(eigvecs.shape[1]):
        mass_norm = np.sqrt(eigvecs[:, m].T @ M @ eigvecs[:, m])
        if mass_norm > 1e-12:
            eigvecs[:, m] = eigvecs[:, m] / mass_norm
    
    omega = np.asarray(omega, dtype=float)
    H = np.zeros_like(omega, dtype=complex)
    
    n_modes = len(omega_m)
    for m in range(n_modes):
        if omega_m[m] < 1e-10:
            continue
        modal_participation = eigvecs[response_dof, m] * eigvecs[load_dof, m]
        denom = omega_m[m] ** 2 - omega ** 2 + 2.0j * zeta_vec[m] * omega_m[m] * omega
        H += modal_participation / denom
    
    return H, omega_m, eigvecs


def compute_stress_psd_from_displacement_psd(S_u, B_mat, E, nu, h=1.0):
    D_mat = E / (1.0 - nu ** 2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    

    DB = D_mat @ B_mat
    amp_factor = np.max(np.abs(DB))
    
    return (amp_factor ** 2) * S_u
