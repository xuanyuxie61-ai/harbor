"""
Rayleigh-Taylor Instability Modeling for ICF Ablator-Fuel Interface

Based on:
- leaf_chaos (Project 655): Iterated function system for chaotic interface tracking

Models:
- Classical RT growth rate: gamma = sqrt(A_t * k * g)
- Nonlinear mode coupling via chaotic iterated function system
- Bubble-spike amplitude evolution
- Mix width growth
"""

import numpy as np


def atwood_number(rho_heavy, rho_light):
    """
    Atwood number: A = (rho_h - rho_l) / (rho_h + rho_l)
    """
    if rho_heavy + rho_light <= 0.0:
        return 0.0
    return (rho_heavy - rho_light) / (rho_heavy + rho_light)


def classical_rt_growth_rate(k, g, A_t):
    """
    Classical incompressible RT growth rate:
    gamma = sqrt(A_t * g * k)  for k << k_cutoff
    """
    if A_t <= 0.0 or g <= 0.0 or k <= 0.0:
        return 0.0
    return np.sqrt(A_t * g * k)


def ablative_rt_growth_rate(k, g, A_t, L_b, v_b):
    """
    Ablative RT growth rate with cutoff:
    gamma = sqrt(A_t * g * k / (1 + k * L_b)) - beta * k * v_b
    where L_b is density scale length, v_b is ablation velocity.
    """
    if A_t <= 0.0 or g <= 0.0 or k <= 0.0:
        return 0.0
    beta = 1.5  # empirical coefficient
    gamma_cl = np.sqrt(A_t * g * k / (1.0 + k * L_b))
    gamma_ab = gamma_cl - beta * k * v_b
    return max(gamma_ab, 0.0)


def rt_mode_amplitude_linear(a0, gamma, t):
    """
    Linear RT amplitude growth: a(t) = a0 * exp(gamma * t)
    """
    if gamma <= 0.0 or t <= 0.0:
        return a0
    return a0 * np.exp(gamma * t)


def rt_chaos_ifs(n_modes, n_iterations, seed=None):
    """
    Chaotic mode coupling model for RT instability.
    Based on leaf_chaos iterated function system from Project 655.

    We map the IFS concept to mode amplitude space, where each mode
    evolves under competing linear transformations representing
    different nonlinear coupling regimes.
    """
    if seed is not None:
        np.random.seed(seed)

    # IFS matrices for mode amplitude transformations
    # Each represents different nonlinear coupling regimes
    A0 = np.array([[0.80, 0.00], [0.00, 0.80]])
    A1 = np.array([[0.50, 0.00], [0.00, 0.50]])
    A2 = np.array([[0.355, -0.355], [0.355, 0.355]])
    A3 = np.array([[0.355, 0.355], [-0.355, 0.355]])

    b0 = np.array([0.10, 0.04])
    b1 = np.array([0.25, 0.40])
    b2 = np.array([0.266, 0.078])
    b3 = np.array([0.378, 0.434])

    As = [A0, A1, A2, A3]
    bs = [b0, b1, b2, b3]

    # Initialize random amplitudes
    amplitudes = np.zeros((n_iterations, 2))
    x = np.random.rand(2)

    for i in range(n_iterations):
        j = np.random.randint(0, 4)
        x = As[j] @ x + bs[j]
        amplitudes[i] = x.copy()

    return amplitudes


def compute_mode_spectrum(radius, rho_profile, g_eff, L_b, v_b,
                           n_modes=64, k_min=None, k_max=None):
    """
    Compute RT mode spectrum for ablator-fuel interface.
    """
    if k_min is None:
        k_min = 2.0 * np.pi / (2.0 * radius)
    if k_max is None:
        k_max = 2.0 * np.pi / max(L_b, 1e-10)

    ks = np.linspace(k_min, k_max, n_modes)
    gammas = np.zeros(n_modes)

    for i, k in enumerate(ks):
        # Approximate densities at interface
        rho_h = rho_profile if np.isscalar(rho_profile) else np.mean(rho_profile)
        rho_l = 0.1 * rho_h  # corona density
        A_t = atwood_number(rho_h, rho_l)
        gammas[i] = ablative_rt_growth_rate(k, g_eff, A_t, L_b, v_b)

    return ks, gammas


def mix_width_estimate(a0, gamma_eff, t, nonlinear_saturation=True):
    """
    Estimate turbulent mix width.
    W(t) = 2 * a0 * exp(gamma_eff * t)   (linear)
    W(t) = alpha * A_t * g * t^2          (nonlinear, alpha ~ 0.05-0.07)
    """
    if t <= 0.0:
        return 0.0

    if nonlinear_saturation:
        alpha_mix = 0.06
        # Use minimum of estimates for saturation
        W_linear = 2.0 * a0 * np.exp(gamma_eff * t)
        W_nonlinear = alpha_mix * 0.5 * 9.8 * t**2  # simplified
        W = min(W_linear, W_nonlinear)
    else:
        W = 2.0 * a0 * np.exp(gamma_eff * t)

    return max(W, 0.0)


def apply_rt_perturbation(r_boundaries, mode_ks, mode_amplitudes, phase_shifts):
    """
    Apply RT perturbation to spherical shell boundaries.
    delta_r(theta, phi) = sum_m a_m * sin(k_m * theta + phi_m)
    For 1D: apply as radial perturbation envelope.
    """
    n = len(r_boundaries)
    r_perturbed = r_boundaries.copy()

    # Apply as radial modulation at each boundary
    for i in range(n):
        dr = 0.0
        for k, amp, phase in zip(mode_ks, mode_amplitudes, phase_shifts):
            dr += amp * np.sin(k * float(i) / max(n - 1, 1) * np.pi + phase)
        r_perturbed[i] += dr
        r_perturbed[i] = max(r_perturbed[i], 1e-15)

    return r_perturbed
