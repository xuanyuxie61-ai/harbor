"""
Semiclassical Electron Dynamics in Twisted Bilayer Graphene
============================================================
Integrates the equations of motion for Bloch electrons under electric
and magnetic fields using high-order Runge-Kutta methods.

Scientific Background
---------------------
In the semiclassical approximation, a wave packet centered at band n
and crystal momentum k obeys

    ħ dk/dt = −e [ E(r, t) + v_n(k) × B(r, t) ]
    dr/dt = v_n(k) = (1/ħ) ∇_k E_n(k)

where e > 0 is the elementary charge, E is the electric field, B is the
magnetic field, and v_n(k) is the group velocity.  These are the
analogues of the Lorentz force in crystal momentum space.

For a uniform time-independent E field in the plane of the 2D material,
the equations reduce to

    dk/dt = −(e/ħ) E
    dr/dt = (1/ħ) ∇_k E_n(k) .

In a perpendicular magnetic field B = B ẑ, the crystal momentum evolves
as

    ħ dk/dt = −e v_n(k) × B = −(eB/ħ) (∂E_n/∂k_y, −∂E_n/∂k_x) .

This gives cyclotron motion in k-space with frequency

    ω_c = eB / m*

where m* is the effective mass.

We use the classical Runge-Kutta 4/5 (RK45) method with embedded error
estimation for adaptive step-size control.  The Butcher table for the
Fehlberg coefficients is:

    k1 = h f(t_n, y_n)
    k2 = h f(t_n + c2 h, y_n + a21 k1)
    ...
    y_{n+1} = y_n + Σ_i b_i k_i       (5th order)
    z_{n+1} = y_n + Σ_i d_i k_i       (4th order)
    e = |y_{n+1} − z_{n+1}|           (error estimate)

The Lax-Wendroff approach for hyperbolic PDEs (from the original
ball_and_stick project) is adapted here as a predictor-corrector
framework for the semiclassical transport equations.
"""

import numpy as np
from typing import Callable, Tuple, Optional


# Fehlberg RK45 Butcher tableau coefficients
RK45_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
    [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0],
    [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0],
    [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0],
])

RK45_C = np.array([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0])

RK45_B5 = np.array([16.0 / 135.0, 0.0, 6656.0 / 12825.0,
                     28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0])

RK45_B4 = np.array([25.0 / 216.0, 0.0, 1408.0 / 2565.0,
                     2197.0 / 4104.0, -1.0 / 5.0, 0.0])


def semiclassical_rhs(
    state: np.ndarray,
    band_energies_func: Callable,
    E_field: np.ndarray,
    B_field: float,
    band_index: int,
) -> np.ndarray:
    """
    Right-hand side of the semiclassical equations of motion.

    State vector: y = [k_x, k_y, r_x, r_y]

    dy/dt = [dk_x/dt, dk_y/dt, dr_x/dt, dr_y/dt]

    Parameters
    ----------
    state : np.ndarray of shape (4,)
        [k_x, k_y, r_x, r_y].
    band_energies_func : callable
        Function k → (energies, velocities) or energies alone.
    E_field : np.ndarray of shape (2,)
        In-plane electric field (V/nm).
    B_field : float
        Out-of-plane magnetic field (Tesla).  Converted to appropriate
        units.
    band_index : int
        Which band to track.

    Returns
    -------
    np.ndarray of shape (4,)
        Time derivative.
    """
    k = state[0:2]
    hbar = 0.6582119  # eV·fs
    e_charge = 1.0  # elementary charge in natural units
    # Magnetic field conversion: 1 T ≈ 1.519e-4 eV²·fs/(nm·e·ħ) in these units
    # Simplified: B_eff = e B / ħ in (nm·fs)^{-1}
    B_eff = e_charge * B_field * 1.519e-4 / hbar

    # TODO: Hole 3 - implement semiclassical equations of motion
    # Scientific background:
    #   Group velocity:  v = (1/ħ) ∇_k E  (finite differences)
    #   dk/dt = -(e/ħ) (E + v × B)
    #   dr/dt = v
    raise NotImplementedError("Hole 3: implement semiclassical RHS (group velocity + Lorentz force)")


def rk45_step(
    f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    h: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single adaptive RK45 step.

    Parameters
    ----------
    f : callable
        RHS function f(y) → dy/dt.
    y : np.ndarray
        Current state.
    t : float
        Current time (unused but kept for API consistency).
    h : float
        Step size.

    Returns
    -------
    y_next : np.ndarray
        5th-order accurate next state.
    error : np.ndarray
        Estimated local truncation error.
    h_new : float
        Suggested next step size.
    """
    s = 6
    k = np.zeros((s, y.size))

    for i in range(s):
        yi = y.copy()
        for j in range(i):
            yi += h * RK45_A[i, j] * k[j]
        k[i] = f(yi)

    y5 = y + h * np.dot(RK45_B5, k)
    y4 = y + h * np.dot(RK45_B4, k)
    error = np.abs(y5 - y4)

    # Step size adaptation
    tol = 1e-6
    scale = tol + tol * np.abs(y5)
    err_norm = np.linalg.norm(error / scale)
    if err_norm == 0.0:
        h_new = 2.0 * h
    else:
        h_new = h * min(5.0, max(0.1, 0.9 * (1.0 / err_norm) ** 0.2))

    return y5, error, h_new


def integrate_trajectory(
    band_energies_func: Callable,
    k0: np.ndarray,
    r0: np.ndarray,
    E_field: np.ndarray,
    B_field: float,
    band_index: int,
    t_max: float = 1000.0,
    h_init: float = 1.0,
    h_min: float = 0.01,
    h_max: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the semiclassical equations of motion from t=0 to t=t_max
    using adaptive RK45.

    Parameters
    ----------
    band_energies_func : callable
        k → array of band energies.
    k0, r0 : np.ndarray of shape (2,)
        Initial crystal momentum and real-space position.
    E_field : np.ndarray of shape (2,)
    B_field : float
    band_index : int
    t_max : float
        Maximum time in fs.
    h_init : float
        Initial step size in fs.
    h_min, h_max : float
        Step size bounds.

    Returns
    -------
    t_array : np.ndarray
        Time points.
    k_array : np.ndarray of shape (M, 2)
    r_array : np.ndarray of shape (M, 2)
    """
    y = np.concatenate([np.asarray(k0, dtype=float),
                        np.asarray(r0, dtype=float)])
    t = 0.0
    h = h_init

    t_list = [t]
    y_list = [y.copy()]

    def rhs(state):
        return semiclassical_rhs(
            state, band_energies_func, E_field, B_field, band_index
        )

    while t < t_max:
        y_next, error, h_suggest = rk45_step(rhs, y, t, h)
        err_max = np.max(error)
        if err_max > 1e-3 and h > h_min:
            # Reject step and retry with smaller h
            h = max(h_suggest, h_min)
            continue

        y = y_next
        t += h
        h = max(h_min, min(h_max, h_suggest))

        t_list.append(t)
        y_list.append(y.copy())

        if len(t_list) > 50000:
            break

    t_array = np.array(t_list)
    y_array = np.array(y_list)
    k_array = y_array[:, 0:2]
    r_array = y_array[:, 2:4]
    return t_array, k_array, r_array


def cyclotron_frequency(
    effective_mass: float,
    B_field: float,
) -> float:
    """
    Compute the cyclotron frequency

        ω_c = e B / m*

    Parameters
    ----------
    effective_mass : float
        In units of electron mass m_e.
    B_field : float
        Magnetic field in Tesla.

    Returns
    -------
    float
        Cyclotron frequency in rad/fs.
    """
    m_e = 5.685e-5  # eV·fs²/nm², not needed directly
    # Simplified: ω_c [rad/fs] ≈ 0.1759 * B[T] / m* (with m* in m_e units)
    return 0.1759 * B_field / effective_mass


def lax_wendroff_predictor_corrector(
    f: Callable,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    Lax-Wendroff predictor-corrector step for hyperbolic systems,
    adapted here as a fast second-order integrator for the
    semiclassical transport equations.

    Predictor (half step):
        y* = y + (h/2) f(y)

    Corrector (full step):
        y_new = y + h f(y*)

    Parameters
    ----------
    f : callable
        RHS function.
    y : np.ndarray
        Current state.
    h : float
        Step size.

    Returns
    -------
    np.ndarray
        Updated state.
    """
    y_star = y + 0.5 * h * f(y)
    y_new = y + h * f(y_star)
    return y_new
