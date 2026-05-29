"""
parameter_inference.py
======================
SVD-Based Linear Least Squares for Turbulent Burning Velocity Correlation.

Based on seed project 1189 (svd_lls.m):
- SVD decomposition for robust least-squares fitting
- Parameter estimation from DNS data

Scientific Context:
-------------------
The turbulent burning velocity S_T is a key quantity in combustion modeling.
For premixed flames, Damköhler's correlation relates S_T to the laminar
flame speed S_L and turbulence intensity u':

  S_T / S_L = 1 + C * (u' / S_L)^n

Taking logarithms for linear fitting:
  ln(S_T/S_L - 1) = ln(C) + n * ln(u'/S_L)

This can be written as a linear system Ax = b where:
  A = [1, ln(u'/S_L)]
  x = [ln(C), n]^T
  b = ln(S_T/S_L - 1)

The SVD approach provides robustness against ill-conditioning:
  A = U Σ V^T
  x = V Σ^{-1} U^T b

For overdetermined systems, the SVD solution minimizes ||Ax - b||₂.

Generalized Correlation:
------------------------
We also consider a multi-parameter correlation including Reynolds and
Damköhler numbers:

  S_T / S_L = a_0 + a_1 * Re_t^{a_2} * Da^{a_3}

where:
  Re_t = u' l_t / ν   (turbulent Reynolds number)
  Da   = l_t S_L / (u' δ_L)   (Damköhler number)
  δ_L  = ν / S_L      (laminar flame thickness)

This is log-linearized as:
  ln(S_T/S_L) = ln(a_0) + a_2 * ln(Re_t) + a_3 * ln(Da)
"""

import numpy as np


def svd_linear_least_squares(A, b):
    """
    Solve Ax ≈ b using SVD decomposition.
    Based on seed 1189 (svd_lls.m).

    x = V Σ⁺ U^T b
    where Σ⁺ is the pseudo-inverse of Σ.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Pseudo-inverse via SVD
    tol = 1e-12 * s[0] if len(s) > 0 else 1e-12
    s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s])
    x = np.dot(Vt.T, s_inv * np.dot(U.T, b))
    residual = np.linalg.norm(np.dot(A, x) - b)
    return x, residual, s


def fit_turbulent_burning_velocity(u_prime_over_sl, st_over_sl):
    """
    Fit Damköhler correlation: S_T/S_L = 1 + C * (u'/S_L)^n
    Returns C, n, and R².
    """
    u = np.asarray(u_prime_over_sl)
    s = np.asarray(st_over_sl)

    # Filter valid points
    valid = (u > 0) & (s > 1.0)
    u = u[valid]
    s = s[valid]

    if len(u) < 2:
        return 1.0, 1.0, 0.0

    # Log transform
    X = np.log(u)
    Y = np.log(s - 1.0)

    A = np.vstack([np.ones_like(X), X]).T
    coeffs, residual, _ = svd_linear_least_squares(A, Y)
    ln_C, n = coeffs
    C = np.exp(ln_C)

    # R²
    ss_res = residual**2
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return C, n, r2


def fit_multi_parameter_correlation(Re_t, Da, st_over_sl):
    """
    Fit generalized correlation:
    ln(S_T/S_L) = a_0 + a_1 * ln(Re_t) + a_2 * ln(Da)
    """
    Re = np.asarray(Re_t)
    Da_arr = np.asarray(Da)
    s = np.asarray(st_over_sl)

    valid = (Re > 0) & (Da_arr > 0) & (s > 0)
    Re = Re[valid]
    Da_arr = Da_arr[valid]
    s = s[valid]

    if len(Re) < 3:
        return np.zeros(3), 0.0

    X1 = np.log(Re)
    X2 = np.log(Da_arr)
    Y = np.log(s)

    A = np.vstack([np.ones_like(X1), X1, X2]).T
    coeffs, residual, _ = svd_linear_least_squares(A, Y)
    a0, a1, a2 = coeffs

    ss_res = residual**2
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return np.array([a0, a1, a2]), r2


def predict_turbulent_flame_speed(u_prime, S_L, l_t, nu, C=1.5, n=0.7):
    """
    Predict turbulent flame speed using fitted correlation.

    S_T = S_L * (1 + C * (u'/S_L)^n)

    Also compute:
      Re_t = u' * l_t / ν
      Ka = (u'/S_L) * (δ_L/l_t)^{0.5}   (Karlovitz number)
    """
    ratio = u_prime / max(S_L, 1e-12)
    st = S_L * (1.0 + C * ratio**n)
    Re_t = u_prime * l_t / max(nu, 1e-12)
    delta_L = nu / max(S_L, 1e-12)
    Ka = ratio * np.sqrt(delta_L / max(l_t, 1e-12))
    return st, Re_t, Ka


def turbulent_flame_regime_diagram(u_prime, S_L, l_t, delta_L):
    """
    Classify combustion regime based on Borghi diagram:
      - Laminar:        u'/S_L < 1
      - Wrinkled:       1 < u'/S_L < Re_L^{0.5}
      - Corrugated:     Re_L^{0.5} < u'/S_L < Da
      - Thin reaction:  Da < u'/S_L < Ka
      - Broken:         u'/S_L > Ka
    """
    ratio = u_prime / max(S_L, 1e-12)
    Re_L = S_L * l_t / max(delta_L * u_prime, 1e-12)  # Lam. flame Reynolds
    Da = l_t * S_L / max(u_prime * delta_L, 1e-12)
    Ka = ratio * np.sqrt(delta_L / max(l_t, 1e-12))

    if ratio < 1.0:
        regime = "laminar"
    elif ratio < np.sqrt(Re_L):
        regime = "wrinkled_flamelets"
    elif ratio < Da:
        regime = "corrugated_flamelets"
    elif ratio < Ka:
        regime = "thin_reaction_zones"
    else:
        regime = "broken_reaction_zones"

    return regime, Re_L, Da, Ka


def compute_dns_turbulent_burning_velocity(c_field, u_field, v_field, dx, dy, dt, S_L):
    """
    Estimate turbulent burning velocity from DNS data using the global
    consumption rate method:

      S_T = (1 / A_f) * ∫ ω̇_c dV

    where A_f is the flame surface area.
    Approximated by temporal evolution of burned volume:
      S_T ≈ (dV_burned/dt) / A_f + S_L
    """
    burned_volume = np.sum(c_field > 0.5) * dx * dy
    # Flame surface area approximated by perimeter * grid spacing
    # Use simple front counting
    front_mask = ((c_field > 0.4) & (c_field < 0.6))
    front_length = np.sum(front_mask) * np.sqrt(dx * dy)

    if front_length < 1e-12:
        return S_L

    # Consumption speed from reaction rate (approximate)
    # For 1-step chemistry: ω̇_c ≈ S_L / δ_L * c(1-c)
    # We estimate δ_L ≈ 3*dx (resolved)
    delta_L = 3.0 * dx
    omega = S_L / delta_L * c_field * (1.0 - c_field)
    consumption = np.sum(omega) * dx * dy

    st = consumption / front_length
    return st
