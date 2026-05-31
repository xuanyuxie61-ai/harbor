"""
Magnetic Spiral Module
======================
Based on seed project 1371_ulam_spiral:
- spiral_array.m  →  2D spiral integer array

Physics:
--------
In relativistic GRB jets, the magnetic field is believed to carry
a helical (spiral) topology inherited from the central engine
(Blandford-Znajek process or magnetohydrodynamic disk winds).
The spiral array provides a discrete model for the toroidal
field winding number as a function of radius:

    B_φ(r) / B_z(r) ≈ tan(ψ(r))

where the pitch angle ψ varies with radius.  For a spiral with
constant angular velocity Ω and outflow velocity v_z:

    ψ(r) = arctan( (Ω r) / v_z )

The Ulam spiral construction maps integers to positions on a
square lattice in outward-spiralling order, which we use to
discretize the winding structure of the jet magnetic field.

The critical magnetization parameter σ is:

    σ = B² / (4π Γ ρ c²)

and the ratio of Poynting flux to kinetic flux is:

    σ_M = (B² v_z) / (Γ ρ c² v_z) = σ · 4π

For a jet to remain magnetically dominated at the photosphere,
σ must exceed unity, implying B > √(4π Γ ρ c²).
"""

import numpy as np


def spiral_array(thick, base=1):
    """
    Produces a (2·thick+1) × (2·thick+1) spiral array of integers
    spiralling outward from the center value `base`.

    Parameters
    ----------
    thick : int
        Spiral radius (≥ 0).
    base : int
        Central value.

    Returns
    -------
    S : ndarray, shape (2*thick+1, 2*thick+1)
        Spiral array.
    """
    n = 2 * thick + 1
    S = np.zeros((n, n), dtype=int)

    row = thick
    col = thick - 1
    k = base - 1

    for t in range(thick + 1):
        col += 1
        k += 1
        S[row, col] = k
        done = False

        while not done:
            if row == t + thick and col == t + thick:
                done = True
                break
            elif col == t + thick and -t + thick < row:
                row -= 1
            elif row == -t + thick and -t + thick < col:
                col -= 1
            elif col == -t + thick and row < t + thick:
                row += 1
            elif row == t + thick and col < t + thick:
                col += 1
            k += 1
            S[row, col] = k

    return S


def prime_spiral_mask(thick):
    """
    Returns a boolean mask indicating prime positions in the spiral.
    """
    S = spiral_array(thick)

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(np.sqrt(n))
        for d in range(3, r + 1, 2):
            if n % d == 0:
                return False
        return True

    mask = np.vectorize(is_prime)(S)
    return mask


def magnetic_pitch_angle_grid(n_r=32, r_max=1e13, v_z=2.99e10, Omega=1e-3):
    """
    Compute the magnetic pitch angle ψ(r) on a radial grid using
    the spiral winding model.

        tan ψ = Ω r / v_z

    Parameters
    ----------
    n_r : int
        Number of radial points.
    r_max : float
        Outer radius in cm.
    v_z : float
        Axial outflow velocity in cm/s.
    Omega : float
        Angular velocity in rad/s.

    Returns
    -------
    r : ndarray
        Radial coordinates.
    psi : ndarray
        Pitch angles in radians.
    B_ratio : ndarray
        B_φ / B_z.
    """
    r = np.linspace(0.0, r_max, n_r)
    r_safe = np.where(r > 1e-6, r, 1e-6)

    tan_psi = Omega * r_safe / v_z
    tan_psi = np.clip(tan_psi, 0.0, 10.0)
    psi = np.arctan(tan_psi)

    B_ratio = tan_psi
    return r, psi, B_ratio


def magnetization_parameter(rho, B, Gamma):
    """
    Compute the magnetization parameter:

        σ = B² / (4π Γ ρ c²)
    """
    c = 2.99792458e10
    sigma = B ** 2 / (4.0 * np.pi * Gamma * rho * c ** 2)
    sigma = np.clip(sigma, 0.0, 1e6)
    return sigma
