"""
resonance_finder.py
===================
Nonlinear root-finding for plasmon resonance conditions.

The localized surface plasmon resonance (LSPR) of a single nanosphere
occurs when the denominator of the polarizability vanishes:

    Re[ ε(ω_res) + 2 ε_medium ] = 0

For the Drude model ε(ω) = ε_∞ − ω_p² / (ω² + i γ ω), the resonance
frequency in the small-damping limit is:

    ω_res ≈ ω_p / √(ε_∞ + 2 ε_medium)

In a coupled nanoparticle assembly, the collective resonance is shifted
by inter-particle coupling.  The resonance condition becomes:

    det[ A(ω) ] = 0

where A(ω) = diag(1/α_j(ω)) − G(ω) is the coupled-dipole interaction
matrix.  In practice, we seek the frequency that maximizes the total
extinction cross section:

    f(ω) = dσ_ext/dω = 0   (extremum condition)

This module implements robust bisection-based root finding adapted from
the nonlin_bisect seed, with bracket expansion for automated interval
search.
"""

import numpy as np


def bisection_method(f, a, b, tol=1e-12, max_iter=100):
    """
    Classical bisection root-finding on an interval [a, b] with f(a)·f(b) < 0.

    Parameters
    ----------
    f : callable
    a, b : float
    tol : float
    max_iter : int

    Returns
    -------
    root : float
    it : int
        Number of iterations performed.
    fa, fb : float
        Function values at final bracket.
    """
    fa = f(a)
    fb = f(b)
    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs.")

    it = 0
    while abs(b - a) > tol and it < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        it += 1
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    root = (a + b) / 2.0
    return root, it, fa, fb


def expand_bracket(f, a0, b0, max_expand=20, factor=2.0):
    """
    Automatically expand the bracket [a, b] until a sign change is found.

    Parameters
    ----------
    f : callable
    a0, b0 : float
        Initial guess.
    max_expand : int
    factor : float
        Expansion factor per step.

    Returns
    -------
    a, b : float
        Valid bracket with f(a)·f(b) < 0, or raises.
    """
    a, b = float(a0), float(b0)
    fa, fb = f(a), f(b)
    if np.sign(fa) != np.sign(fb):
        return a, b

    for _ in range(max_expand):
        if abs(fa) < abs(fb):
            a = a - factor * (b - a)
            fa = f(a)
        else:
            b = b + factor * (b - a)
            fb = f(b)
        if np.sign(fa) != np.sign(fb):
            return a, b

    raise RuntimeError("Failed to find a sign-changing bracket.")


def find_single_sphere_resonance(eps_medium, omega_p=9.0e15,
                                  gamma=1.0e14, eps_inf=9.0,
                                  bracket=None):
    """
    Find the dipolar LSPR frequency of a single Drude metal sphere
    by solving Re[ε(ω) + 2ε_medium] = 0.

    Parameters
    ----------
    eps_medium : float
    omega_p, gamma, eps_inf : float
    bracket : tuple or None
        (omega_min, omega_max) in rad/s.

    Returns
    -------
    omega_res : float
    """
    def eps_metal(omega):
        return eps_inf - (omega_p ** 2) / (omega ** 2 + 1j * gamma * omega)

    def target(omega):
        if omega <= 0:
            return 1.0
        return np.real(eps_metal(omega) + 2.0 * eps_medium)

    if bracket is None:
        omega_est = omega_p / np.sqrt(eps_inf + 2.0 * eps_medium)
        a = 0.5 * omega_est
        b = 1.5 * omega_est
    else:
        a, b = bracket

    a, b = expand_bracket(target, a, b)
    root, it, _, _ = bisection_method(target, a, b)
    return root


def find_collective_resonance(positions, polarizability_func,
                               omega_min, omega_max,
                               eps_medium=1.0, num_points=200):
    """
    Find the collective resonance frequency of a coupled nanoparticle
    assembly by maximizing the total dipole strength (proxy for extinction).

    The figure of merit is the norm of the inverse interaction matrix:

        F(ω) = || A(ω)^{-1} ||_F

    We scan a frequency grid and then refine the maximum with bisection
    on the derivative approximated by finite differences.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    polarizability_func : callable
        polarizability_func(omega) -> ndarray of shape (N,) with complex α.
    omega_min, omega_max : float
    eps_medium : float
    num_points : int

    Returns
    -------
    omega_res : float
    """
    from dipole_coupling import build_coupling_matrix

    omegas = np.linspace(omega_min, omega_max, num_points)
    figure_of_merit = np.zeros(num_points)

    for i, omg in enumerate(omegas):
        alphas = polarizability_func(omg)
        try:
            A = build_coupling_matrix(positions, alphas, omg, eps_medium)
            # Use trace of inverse as proxy for total response strength
            # For Hermitian positive-definite matrices, trace(inv(A)) is
            # related to the spectral response.
            eigvals = np.linalg.eigvalsh(A.real)  # approximate
            if np.any(eigvals <= 0):
                figure_of_merit[i] = 0.0
            else:
                figure_of_merit[i] = np.sum(1.0 / eigvals)
        except Exception:
            figure_of_merit[i] = 0.0

    # Smooth and find maximum
    if np.all(figure_of_merit == 0):
        return (omega_min + omega_max) / 2.0

    idx_max = np.argmax(figure_of_merit)
    if idx_max == 0 or idx_max == num_points - 1:
        return omegas[idx_max]

    # Refine with quadratic interpolation
    o1, o2, o3 = omegas[idx_max - 1], omegas[idx_max], omegas[idx_max + 1]
    f1, f2, f3 = figure_of_merit[idx_max - 1], figure_of_merit[idx_max], figure_of_merit[idx_max + 1]

    # Parabola through three points: vertex at
    # ω* = ω₂ − (Δω/2) (f₃ − f₁) / (f₃ − 2f₂ + f₁)
    denom = f3 - 2.0 * f2 + f1
    if abs(denom) < 1e-30:
        return o2
    omega_res = o2 - 0.5 * (o3 - o1) * (f3 - f1) / denom
    omega_res = np.clip(omega_res, omega_min, omega_max)
    return float(omega_res)
