"""
Utility functions for CO2 amine absorption dynamics simulator.
Provides numerical robustness helpers, physical constants, and validation.
"""

import numpy as np

# Physical constants
R_GAS = 8.314462618  # J/(mol·K)
STANDARD_TEMP = 298.15  # K
STANDARD_PRESSURE = 101325.0  # Pa


def validate_positive(value, name, allow_zero=False):
    """Validate that a value is positive (or non-negative if allow_zero)."""
    if value is None:
        raise ValueError(f"{name} cannot be None")
    if np.isscalar(value):
        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        else:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
    else:
        arr = np.asarray(value)
        if allow_zero:
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative")
        else:
            if np.any(arr <= 0):
                raise ValueError(f"{name} must be positive")


def safe_log(x, eps=1e-300):
    """Compute log with robustness near zero."""
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, eps))


def safe_divide(a, b, eps=1e-300):
    """Safe division avoiding division by zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > eps, a / b, 0.0)


def arrhenius_rate(A, Ea, T):
    """
    Arrhenius rate constant: k = A * exp(-Ea / (R * T))
    Parameters:
        A: pre-exponential factor (s^-1 or m^3/(mol·s) etc.)
        Ea: activation energy (J/mol)
        T: temperature (K), must be > 0
    Returns:
        k: rate constant
    """
    validate_positive(T, "Temperature")
    validate_positive(A, "Pre-exponential factor", allow_zero=True)
    return A * np.exp(-Ea / (R_GAS * T))


def van_t_hoff(K0, dH, T, T0=STANDARD_TEMP):
    """
    Van't Hoff equation for temperature-dependent equilibrium constant:
        K = K0 * exp(-dH/R * (1/T - 1/T0))
    """
    validate_positive(T, "Temperature")
    validate_positive(K0, "Reference equilibrium constant")
    return K0 * np.exp(-dH / R_GAS * (1.0 / T - 1.0 / T0))


def wilke_chang_diffusion(T, mu, Vb, alpha_assoc=2.6):
    """
    Wilke-Chang correlation for liquid-phase diffusion coefficient:
        D_AB = 7.4e-8 * (alpha_assoc * M_B)^0.5 * T / (mu * Vb_A^0.6)
    Parameters:
        T: temperature (K)
        mu: solvent viscosity (cP)
        Vb: solute molar volume at normal boiling point (cm^3/mol)
        alpha_assoc: solvent association parameter (2.6 for water, 1.9 for MEA)
    Returns:
        D_AB: diffusion coefficient (cm^2/s)
    """
    validate_positive(T, "Temperature")
    validate_positive(mu, "Viscosity")
    validate_positive(Vb, "Molar volume")
    return 7.4e-8 * (alpha_assoc * T) ** 0.5 * T / (mu * Vb ** 0.6)


def hatta_number(k2, D_A, c_B, k_L):
    """
    Hatta number for second-order reactions in gas-liquid absorption:
        Ha = sqrt(k2 * D_A * c_B) / k_L
    """
    validate_positive(k2, "Rate constant")
    validate_positive(D_A, "Diffusivity")
    validate_positive(c_B, "Concentration")
    validate_positive(k_L, "Mass transfer coefficient")
    return np.sqrt(k2 * D_A * c_B) / k_L


def enhancement_factor_hatta(Ha, E_infinite=None):
    """
    Enhancement factor for gas-liquid absorption with chemical reaction.
    For fast reactions (Ha >> 1): E ≈ Ha
    For instantaneous reactions: E ≈ E_infinite
    """
    if E_infinite is not None:
        E = np.minimum(Ha, E_infinite)
    else:
        E = Ha
    # Smooth transition for numerical stability
    return np.where(Ha < 0.01, 1.0 + Ha * Ha / 3.0, E)


def clip_concentration(c, eps=1e-12):
    """Clip concentration to avoid negative values."""
    return np.maximum(c, eps)


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """Generate Chebyshev nodes of the second kind on [a, b]."""
    if n < 2:
        raise ValueError("n must be at least 2")
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    # Scale to [a, b]
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def chebyshev_differentiation_matrix(n, a=-1.0, b=1.0):
    """
    Construct the Chebyshev differentiation matrix on [a, b].
    Based on Trefethen's spectral methods.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    x = chebyshev_nodes(n, a, b)
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n))
    X = np.tile(x, (n, 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
