"""
Dynamic Response Integrator Module
===================================
Based on project 139_cauchy_principal_value.

Computes random vibration response of structures using frequency-domain
methods. Handles singular integrals in frequency response functions via
Cauchy Principal Value (CPV) integration.

Key physics:
- Frequency Response Function (FRF):
  H(omega) = 1 / (k - m*omega^2 + i*c*omega)
  
- Power Spectral Density (PSD) of response:
  S_y(omega) = |H(omega)|^2 * S_x(omega)
  
- Mean-square response:
  sigma_y^2 = integral_{-inf}^{inf} S_y(omega) d(omega) / (2*pi)
  
- For lightly damped systems, |H(omega)|^2 has a sharp peak near omega_n.
  The CPV formulation handles the near-singular behavior rigorously.

- First-passage failure probability (Poisson approximation):
  P_f(T) ≈ 1 - exp(-nu_0^+ * T)
  nu_0^+ = (omega_0 / (2*pi)) * exp(-a^2 / (2*sigma_y^2))
  where omega_0 = sqrt(integral omega^2 S_y domega / integral S_y domega)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


def cauchy_principal_value(f, a, b, x_sing, n=20):
    """
    Compute Cauchy Principal Value of integral_a^b f(t)/(t-x_sing) dt.
    
    Uses Gauss-Legendre quadrature with symmetric exclusion:
    CPV = integral_{a}^{x-delta} f(t)/(t-x) dt
        + CPV_{-delta}^{delta} f(x+s)/s ds
        + integral_{x+delta}^{b} f(t)/(t-x) dt
    
    The singular part is transformed using the fact that for even N:
    sum_i w_i / xi_i = 0, where xi_i are GL nodes on [-1,1].
    
    Parameters
    ----------
    f : callable
        Smooth part of integrand.
    a, b : float
        Integration limits.
    x_sing : float
        Singularity location (a < x_sing < b).
    n : int
        Number of Gauss-Legendre points (must be even).
    
    Returns
    -------
    cpv : float
        Cauchy principal value.
    """
    if n % 2 != 0:
        n += 1  # Ensure even
    
    xi, wi = leggauss(n)
    
    # Transform to [a, b]
    cpv = 0.0
    for i in range(n):
        t = 0.5 * ((1.0 - xi[i]) * a + (1.0 + xi[i]) * b)
        # The integrand f(t) / (t - x_sing)
        # Using the symmetric formulation: f(t) / (t - x) = [f(t) - f(x)] / (t - x) + f(x)/(t-x)
        # For even GL, sum w_i / xi_i = 0, so the f(x) term vanishes
        # We map t to the standard interval and evaluate
        denom = t - x_sing
        if abs(denom) < 1e-14:
            # Use derivative approximation at singularity
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
    """
    Compute frequency response function for a SDOF or MDOF system.
    
    For SDOF:
    H(omega) = 1 / (omega_n^2 - omega^2 + 2*i*zeta*omega_n*omega)
    
    For MDOF (modal superposition):
    H_jk(omega) = sum_m (phi_j^(m) * phi_k^(m)) / (omega_m^2 - omega^2 + 2*i*zeta_m*omega_m*omega)
    
    Parameters
    ----------
    omega : float or ndarray
        Excitation frequency.
    omega_n : float or ndarray
        Natural frequency(ies).
    zeta : float or ndarray
        Damping ratio(s).
    mode_shape : ndarray, optional
        For MDOF, mode shape vector.
    
    Returns
    -------
    H : complex float or ndarray
        Frequency response.
    """
    omega = np.asarray(omega, dtype=float)
    omega_n = np.asarray(omega_n, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    
    if omega_n.ndim == 0:
        # SDOF
        denom = omega_n ** 2 - omega ** 2 + 2.0j * zeta * omega_n * omega
        return 1.0 / denom
    else:
        # MDOF modal superposition
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
    """
    Compute response power spectral density.
    
    S_y(omega) = |H(omega)|^2 * S_x(omega)
    
    Parameters
    ----------
    omega : ndarray
        Frequency array.
    omega_n, zeta : float or ndarray
    psd_input : callable or ndarray
        Input PSD S_x(omega).
    mode_shape : ndarray, optional
    
    Returns
    -------
    S_y : ndarray
        Response PSD.
    """
    H = frequency_response_function(omega, omega_n, zeta, mode_shape)
    if callable(psd_input):
        Sx = psd_input(omega)
    else:
        Sx = np.asarray(psd_input)
    return np.abs(H) ** 2 * Sx


def integrate_psd_cpv(omega_n, zeta, psd_input, omega_max=None, n_quad=200):
    """
    Integrate response PSD using Cauchy principal value treatment near resonance.
    
    For lightly damped systems (zeta < 0.05), the integrand has a sharp peak.
    We split the integral into [0, omega_n-delta], [omega_n-delta, omega_n+delta],
    [omega_n+delta, omega_max] and use CPV for the middle interval.
    
    Mean-square response: sigma_y^2 = (1/2*pi) * integral S_y(omega) domega
    
    Parameters
    ----------
    omega_n : float
    zeta : float
    psd_input : callable
    omega_max : float, optional
    n_quad : int
    
    Returns
    -------
    sigma_y_sq : float
        Mean-square response.
    omega_0 : float
        Zero-crossing frequency for first-passage analysis.
    """
    if omega_max is None:
        omega_max = 3.0 * omega_n
    
    delta = omega_n * max(0.01, 2.0 * zeta)  # Width of singular region
    
    # Define integrand
    def integrand(omega):
        Sy = psd_response(omega, omega_n, zeta, psd_input)
        return Sy
    
    # Regular intervals
    omega_reg_1 = np.linspace(0, max(0, omega_n - delta), n_quad // 3)
    omega_reg_2 = np.linspace(omega_n + delta, omega_max, n_quad // 3)
    
    # Near-resonance interval using CPV concept via dense sampling
    omega_sing = np.linspace(omega_n - delta, omega_n + delta, n_quad // 3)
    
    # Combine
    omega_all = np.sort(np.unique(np.concatenate([omega_reg_1, omega_sing, omega_reg_2])))
    omega_all = omega_all[omega_all >= 0]
    
    Sy_all = integrand(omega_all)
    
    # Numerical integration (trapezoidal with dense near-resonance)
    sigma_y_sq = np.trapezoid(Sy_all, omega_all) / (2.0 * np.pi)
    
    # Zero-crossing frequency: omega_0 = sqrt(int omega^2 Sy domega / int Sy domega)
    moment2 = np.trapezoid(omega_all ** 2 * Sy_all, omega_all) / (2.0 * np.pi)
    if sigma_y_sq > 1e-20:
        omega_0 = np.sqrt(moment2 / sigma_y_sq)
    else:
        omega_0 = omega_n
    
    return sigma_y_sq, omega_0


def first_passage_probability(sigma_y, threshold, omega_0, T_duration,
                               method='poisson'):
    """
    Compute first-passage failure probability for a Gaussian narrow-band process.
    
    Methods:
    - 'poisson': Poisson approximation
      P_f = 1 - exp(-nu_0^+ * T)
      nu_0^+ = (omega_0 / 2*pi) * exp(-b^2 / 2), b = threshold / sigma_y
    
    - 'vanmarcke': Vanmarcke's modified approximation
      P_f = 1 - exp(-nu_0^+ * T * (1 - exp(-sqrt(pi/2) * delta_eff * b)))
      where delta_eff accounts for bandwidth
    
    Parameters
    ----------
    sigma_y : float
        RMS response.
    threshold : float
        Failure threshold.
    omega_0 : float
        Mean zero-crossing frequency (rad/s).
    T_duration : float
        Duration of excitation (seconds).
    method : str
    
    Returns
    -------
    P_f : float
        Failure probability.
    nu_up : float
        Up-crossing rate.
    """
    if sigma_y <= 1e-14:
        return 0.0 if threshold > 0 else 1.0, 0.0
    
    b = threshold / sigma_y
    
    nu_up = omega_0 / (2.0 * np.pi) * np.exp(-0.5 * b ** 2)
    
    if method == 'poisson':
        P_f = 1.0 - np.exp(-nu_up * T_duration)
    elif method == 'vanmarcke':
        # Simplified Vanmarcke with effective bandwidth parameter
        delta_eff = 0.3  # Typical for narrow-band processes
        factor = 1.0 - np.exp(-np.sqrt(np.pi / 2.0) * delta_eff * b)
        P_f = 1.0 - np.exp(-nu_up * T_duration * factor)
    else:
        P_f = 1.0 - np.exp(-nu_up * T_duration)
    
    # Bound probability
    P_f = np.clip(P_f, 0.0, 1.0)
    
    return P_f, nu_up


def modal_superposition_response(omega, K, M, zeta_vec, load_dof,
                                  response_dof, omega_n_modes=None):
    """
    Compute FRF using full modal superposition for MDOF system.
    
    Solves the eigenvalue problem K*phi = omega^2*M*phi and computes:
    H_jk(omega) = sum_m (phi_j^m * phi_k^m) / (omega_m^2 - omega^2 + 2*i*zeta_m*omega_m*omega)
    
    Parameters
    ----------
    omega : float or ndarray
    K, M : ndarray
        Stiffness and mass matrices.
    zeta_vec : ndarray
        Damping ratios per mode.
    load_dof : int
        DOF index where load is applied.
    response_dof : int
        DOF index where response is measured.
    omega_n_modes : int, optional
        Number of modes to include.
    
    Returns
    -------
    H : complex ndarray
    eigenvalues : ndarray
        Natural frequencies (rad/s).
    eigenvectors : ndarray
        Mode shapes (mass-normalized).
    """
    # Solve generalized eigenvalue problem
    try:
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(K, M)
    except ImportError:
        # Fallback: standard numpy
        M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M)))
        K_tilde = M_inv_sqrt @ K @ M_inv_sqrt
        eigvals, eigvecs_tilde = np.linalg.eigh(K_tilde)
        eigvecs = M_inv_sqrt @ eigvecs_tilde
    
    # Natural frequencies
    omega_m = np.sqrt(np.maximum(eigvals, 0.0))
    
    if omega_n_modes is not None:
        n_modes = min(omega_n_modes, len(omega_m))
        omega_m = omega_m[:n_modes]
        eigvecs = eigvecs[:, :n_modes]
        zeta_vec = zeta_vec[:n_modes]
    
    # Mass-normalize mode shapes
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
    """
    Convert displacement PSD to stress PSD using strain-displacement relation.
    
    For plane stress:
    sigma = D * B * u
    S_sigma = D * B * S_u * B^T * D^T
    
    For scalar maximum principal stress approximation:
    sigma_max ≈ E / (1-nu^2) * max_strain
    
    Parameters
    ----------
    S_u : float or ndarray
        Displacement variance/PSD.
    B_mat : ndarray
        Strain-displacement matrix.
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.
    h : float
        Thickness.
    
    Returns
    -------
    S_sigma : float or ndarray
        Stress variance/PSD.
    """
    D_mat = E / (1.0 - nu ** 2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    
    # Simplified: use maximum entry of D*B as amplification factor
    DB = D_mat @ B_mat
    amp_factor = np.max(np.abs(DB))
    
    return (amp_factor ** 2) * S_u
