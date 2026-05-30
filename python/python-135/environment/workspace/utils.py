
import numpy as np


R_GAS = 8.314462618
STANDARD_TEMP = 298.15
STANDARD_PRESSURE = 101325.0


def validate_positive(value, name, allow_zero=False):
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
    x = np.asarray(x, dtype=float)
    return np.log(np.maximum(x, eps))


def safe_divide(a, b, eps=1e-300):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(np.abs(b) > eps, a / b, 0.0)


def arrhenius_rate(A, Ea, T):
    validate_positive(T, "Temperature")
    validate_positive(A, "Pre-exponential factor", allow_zero=True)
    return A * np.exp(-Ea / (R_GAS * T))


def van_t_hoff(K0, dH, T, T0=STANDARD_TEMP):
    validate_positive(T, "Temperature")
    validate_positive(K0, "Reference equilibrium constant")
    return K0 * np.exp(-dH / R_GAS * (1.0 / T - 1.0 / T0))


def wilke_chang_diffusion(T, mu, Vb, alpha_assoc=2.6):
    validate_positive(T, "Temperature")
    validate_positive(mu, "Viscosity")
    validate_positive(Vb, "Molar volume")
    return 7.4e-8 * (alpha_assoc * T) ** 0.5 * T / (mu * Vb ** 0.6)


def hatta_number(k2, D_A, c_B, k_L):
    validate_positive(k2, "Rate constant")
    validate_positive(D_A, "Diffusivity")
    validate_positive(c_B, "Concentration")
    validate_positive(k_L, "Mass transfer coefficient")
    return np.sqrt(k2 * D_A * c_B) / k_L


def enhancement_factor_hatta(Ha, E_infinite=None):
    if E_infinite is not None:
        E = np.minimum(Ha, E_infinite)
    else:
        E = Ha

    return np.where(Ha < 0.01, 1.0 + Ha * Ha / 3.0, E)


def clip_concentration(c, eps=1e-12):
    return np.maximum(c, eps)


def chebyshev_nodes(n, a=-1.0, b=1.0):
    if n < 2:
        raise ValueError("n must be at least 2")
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))

    return 0.5 * (a + b) + 0.5 * (b - a) * x


def chebyshev_differentiation_matrix(n, a=-1.0, b=1.0):
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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
