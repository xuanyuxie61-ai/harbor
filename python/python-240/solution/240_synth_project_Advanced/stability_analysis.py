"""
稳定性分析模块
Stability analysis: von Neumann analysis, CFL conditions,
spectral radius, and dispersion relations for high-order FD schemes.

Algorithms sourced from:
  - 1073_balloon_dynamics (adaptive CFL for expanding systems)
  - 846_paraheat_functional (parabolic stability bounds)
"""
import numpy as np
from physics_constants import compute_sound_speed_squared


def cfl_condition(max_wave_speed, dx, dy, safety_factor=0.8):
    """
    CFL (Courant-Friedrichs-Lewy) stability condition:
        Δt ≤ safety_factor * min(Δx, Δy) / max(|v| + c_s)

    For relativistic hydro: max wave speed = max(|v| + c_s) ≤ 1 + c_s

    Parameters
    ----------
    max_wave_speed : float
        Maximum wave speed (c_s or |v| + c_s)
    dx, dy : float
        Grid spacing
    safety_factor : float
        Safety factor (< 1)

    Returns
    -------
    dt_max : float
        Maximum stable time step
    """
    h_min = min(dx, dy)
    if max_wave_speed <= 0:
        return np.inf
    return safety_factor * h_min / max_wave_speed


def von_neumann_stability_fd1d(order, c_s, dx):
    """
    Von Neumann stability analysis for 1D advection equation
    with finite difference of given order.

    For ∂u/∂t + c ∂u/∂x = 0 with FTCS:
        Amplification factor g(k) = 1 - i c Δt Σ_m a_m sin(m k Δx) / Δx

    Stability requires |g(k)| ≤ 1 for all k.

    Parameters
    ----------
    order : int
        Finite difference order (2, 4, 6, 8)
    c_s : float
        Advection speed (sound speed)
    dx : float
        Grid spacing

    Returns
    -------
    dict with keys:
        'stable_explicit' : bool
            Whether forward Euler is stable
        'max_cfl' : float
            Maximum stable CFL number
        'k_range' : ndarray
            Wavenumber range tested
        'g_mag' : ndarray
            |g(k)| values
    """
    # Wavenumber range
    k_max = np.pi / dx
    k_range = np.linspace(0, k_max, 200)

    # FD stencil coefficients (scaled by dx)
    if order == 2:
        coeffs = np.array([-0.5, 0.5])
        offsets = np.array([-1, 1])
    elif order == 4:
        coeffs = np.array([1.0/12.0, -8.0/12.0, 8.0/12.0, -1.0/12.0])
        offsets = np.array([-2, -1, 1, 2])
    elif order == 6:
        coeffs = np.array([-1.0/60.0, 9.0/60.0, -45.0/60.0,
                          45.0/60.0, -9.0/60.0, 1.0/60.0])
        offsets = np.array([-3, -2, -1, 1, 2, 3])
    elif order == 8:
        coeffs = np.array([3.0/840.0, -32.0/840.0, 168.0/840.0, -672.0/840.0,
                          672.0/840.0, -168.0/840.0, 32.0/840.0, -3.0/840.0])
        offsets = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    else:
        raise ValueError(f"Unsupported order {order}")

    # Compute amplification factor for CFL = 1
    cfl = 1.0
    g_mag = np.zeros_like(k_range)
    for idx, k in enumerate(k_range):
        symbol = 0j
        for c, m in zip(coeffs, offsets):
            symbol += c * np.exp(1j * m * k * dx)
        # For FTCS: g = 1 - i * CFL * symbol
        g = 1.0 - 1j * cfl * c_s * symbol / dx * dx / c_s
        g_mag[idx] = abs(g)

    # For FTCS, explicit scheme is unconditionally unstable for pure advection
    # But with Lax-Friedrichs or upwinding, stable for CFL ≤ 1
    stable = np.all(g_mag <= 1.0 + 1e-10)

    # Find max stable CFL by bisection
    cfl_lo, cfl_hi = 0.0, 2.0
    for _ in range(30):
        cfl_test = 0.5 * (cfl_lo + cfl_hi)
        g_max = 0.0
        for k in k_range:
            symbol = 0j
            for c, m in zip(coeffs, offsets):
                symbol += c * np.exp(1j * m * k * dx)
            g = 1.0 - 1j * cfl_test * symbol
            g_max = max(g_max, abs(g))
        if g_max <= 1.0 + 1e-10:
            cfl_lo = cfl_test
        else:
            cfl_hi = cfl_test

    return {
        'stable_explicit': bool(stable),
        'max_cfl': cfl_lo,
        'k_range': k_range,
        'g_mag': g_mag,
    }


def spectral_radius_laplacian(nx, ny, dx, dy, order=2):
    """
    Compute spectral radius of discrete Laplacian operator.
    For stability: Δt ≤ 2 / ρ(∇²)

    For 2nd order: ρ = 4(1/Δx² + 1/Δy²)
    For higher order: larger spectral radius → more restrictive Δt

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    order : int
        FD order (2, 4, 6)

    Returns
    -------
    float
        Spectral radius ρ
    """
    # Exact spectral radius for periodic BC
    if order == 2:
        rho = 4.0 * (1.0 / dx**2 + 1.0 / dy**2)
    elif order == 4:
        # 4th order Laplacian has larger spectral radius
        rho = (30.0 / 12.0) * (1.0 / dx**2 + 1.0 / dy**2) * 2.0
    elif order == 6:
        rho = (490.0 / 180.0) * (1.0 / dx**2 + 1.0 / dy**2) * 2.0
    else:
        # Estimate for higher order
        rho = 4.0 * (order / 2)**2 * (1.0 / dx**2 + 1.0 / dy**2)

    return rho


def diffusion_stability_limit(diffusivity, dx, dy, order=2):
    """
    Stability limit for diffusion equation ∂u/∂t = D ∇²u:
        Δt ≤ 1 / (D * ρ(∇²))

    From 846_paraheat_functional: parabolic stability bound.

    Parameters
    ----------
    diffusivity : float
        Diffusion coefficient D
    dx, dy : float
        Grid spacing
    order : int
        FD order

    Returns
    -------
    float
        Maximum stable Δt
    """
    nx, ny = 64, 64  # Representative size
    rho = spectral_radius_laplacian(nx, ny, dx, dy, order)
    if diffusivity <= 0 or rho <= 0:
        return np.inf
    return 1.0 / (diffusivity * rho)


def adaptive_cfl_timestep(eps, mx, my, dx, dy, safety=0.5):
    """
    Compute adaptive time step based on local wave speeds.
    From 1073_balloon_dynamics: adaptive CFL for expanding systems.

    For relativistic hydro:
        Δt = safety * min(Δx, Δy) / max(|v| + c_s)

    Parameters
    ----------
    eps : ndarray
        Energy density
    mx, my : ndarray
        Momentum density
    dx, dy : float
        Grid spacing
    safety : float
        Safety factor

    Returns
    -------
    dt : float
        Recommended time step
    max_speed : float
        Maximum wave speed
    """
    from equation_of_state import (
        pressure_lattice, temperature_from_energy_density, sound_speed_squared
    )

    T = temperature_from_energy_density(eps)
    p = pressure_lattice(T)
    w = eps + p
    w = np.maximum(w, 1e-10)

    # Local velocity magnitude
    v_mag = np.sqrt(mx**2 + my**2) / w
    v_mag = np.minimum(v_mag, 0.99)  # Bound

    # Sound speed
    cs = np.sqrt(sound_speed_squared(T))

    # Maximum wave speed
    max_speed = np.max(v_mag + cs)

    if max_speed <= 0:
        return np.inf, 0.0

    dt = safety * min(dx, dy) / max_speed
    return dt, max_speed


def dispersion_relation(order, k_range, dx):
    """
    Compute numerical dispersion relation for FD scheme.
    Compare numerical phase speed to exact.

    Parameters
    ----------
    order : int
        FD order
    k_range : ndarray
        Wavenumber range
    dx : float
        Grid spacing

    Returns
    -------
    omega_numerical : ndarray
        Numerical frequency
    omega_exact : ndarray
        Exact frequency (ω = k for c=1)
    """
    if order == 2:
        coeffs = np.array([-0.5, 0.5])
        offsets = np.array([-1, 1])
    elif order == 4:
        coeffs = np.array([1.0/12.0, -8.0/12.0, 8.0/12.0, -1.0/12.0])
        offsets = np.array([-2, -1, 1, 2])
    elif order == 6:
        coeffs = np.array([-1.0/60.0, 9.0/60.0, -45.0/60.0,
                          45.0/60.0, -9.0/60.0, 1.0/60.0])
        offsets = np.array([-3, -2, -1, 1, 2, 3])
    else:
        coeffs = np.array([3.0/840.0, -32.0/840.0, 168.0/840.0, -672.0/840.0,
                          672.0/840.0, -168.0/840.0, 32.0/840.0, -3.0/840.0])
        offsets = np.array([-4, -3, -2, -1, 1, 2, 3, 4])

    omega_num = np.zeros_like(k_range)
    for idx, k in enumerate(k_range):
        symbol = 0j
        for c, m in zip(coeffs, offsets):
            symbol += c * np.exp(1j * m * k * dx)
        # ω_numerical = Im(symbol) / Δx
        omega_num[idx] = np.imag(symbol) / dx

    omega_exact = k_range  # c = 1

    return omega_num, omega_exact


def stability_report(fd_order, dx, dy, temperature_field, diffusivity=0.1):
    """
    Generate comprehensive stability report.

    Parameters
    ----------
    fd_order : int
        FD order
    dx, dy : float
        Grid spacing
    temperature_field : ndarray
        Temperature values
    diffusivity : float
        Thermal diffusivity

    Returns
    -------
    dict
        Stability metrics
    """
    from equation_of_state import sound_speed_squared

    cs_max = np.sqrt(np.max(sound_speed_squared(temperature_field)))

    report = {
        'fd_order': fd_order,
        'dx': dx,
        'dy': dy,
        'cs_max': cs_max,
    }

    # CFL limit
    report['cfl_dt'] = cfl_condition(cs_max, dx, dy)

    # Diffusion limit
    report['diffusion_dt'] = diffusion_stability_limit(diffusivity, dx, dy, fd_order)

    # Spectral radius
    report['spectral_radius'] = spectral_radius_laplacian(
        64, 64, dx, dy, fd_order)

    # Von Neumann analysis
    vn = von_neumann_stability_fd1d(fd_order, cs_max, dx)
    report['vn_max_cfl'] = vn['max_cfl']
    report['vn_stable'] = vn['stable_explicit']

    # Recommended dt
    report['recommended_dt'] = min(report['cfl_dt'], report['diffusion_dt'])

    return report
