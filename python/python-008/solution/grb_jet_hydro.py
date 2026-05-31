"""
GRB Jet Hydrodynamics Module
==============================
Based on seed project 211_continuity_exact:
- continuity_exact.m  →  relativistic continuity equation for GRB jet
- grid_2d.m           →  2D polar grid generation
- uv_spiral.m         →  zero-divergence spiral velocity field

Physics:
--------
The relativistic continuity equation for the GRB jet ejecta:

    ∂(Γρ)/∂t + ∇ · (Γρ v) = 0

where Γ = (1 - v²/c²)^(-1/2) is the Lorentz factor,
ρ is the comoving mass density, and v is the three-velocity.

For a steady, axisymmetric jet in cylindrical coordinates (r, φ, z):

    (1/r) ∂(r Γρ v_r)/∂r + ∂(Γρ v_z)/∂z = 0

We construct a stream function Ψ(r,z) such that:

    Γρ v_r =  -(1/r) ∂Ψ/∂z
    Γρ v_z =   (1/r) ∂Ψ/∂r

which guarantees mass conservation identically.

The magnetic field advection is governed by the induction equation
in the ideal MHD limit:

    ∂B/∂t = ∇ × (v × B)

For a poloidal field B_p = B_r e_r + B_z e_z with vanishing toroidal
component, the flux function A(r,z) satisfies:

    v_r ∂A/∂r + v_z ∂A/∂z = 0
"""

import numpy as np


def grid_2d(x_num, x_lo, x_hi, y_num, y_lo, y_hi):
    """
    Returns a regular 2D grid.

    Parameters
    ----------
    x_num, y_num : int
        Number of points in x and y directions.
    x_lo, x_hi, y_lo, y_hi : float
        Domain bounds.

    Returns
    -------
    x, y : ndarray, shape (x_num, y_num)
        Grid coordinates.
    """
    x = np.zeros((x_num, y_num))
    y = np.zeros((x_num, y_num))

    if x_num == 1:
        x[:, :] = (x_lo + x_hi) / 2.0
    else:
        for i in range(x_num):
            xi = ((x_num - 1 - i) * x_lo + i * x_hi) / (x_num - 1)
            x[i, :] = xi

    if y_num == 1:
        y[:, :] = (y_lo + y_hi) / 2.0
    else:
        for j in range(y_num):
            yi = ((y_num - 1 - j) * y_lo + j * y_hi) / (y_num - 1)
            y[:, j] = yi

    return x, y


def phi_stream(z, c):
    """
    Stream function profile for GRB jet.

    Φ(Z) = (1 - cos(C π Z)) (1 - Z)²

    This profile yields vanishing velocity at the jet boundary Z=1
    and maximum flow at the jet axis Z=0.
    """
    return (1.0 - np.cos(c * np.pi * z)) * (1.0 - z) ** 2


def dphi_stream(z, c):
    """First derivative of the stream function profile."""
    term1 = c * np.pi * np.sin(c * np.pi * z) * (1.0 - z) ** 2
    term2 = (1.0 - np.cos(c * np.pi * z)) * 2.0 * (1.0 - z)
    return term1 - term2


def uv_spiral(n, x, y, c):
    """
    Computes a zero-divergence spiral velocity vector field.

    The velocity field is derived from a stream function:

        U(X,Y) =  +10 · ∂Φ(Y)/∂Y · Φ(X)
        V(X,Y) =  -10 · ∂Φ(X)/∂X · Φ(Y)

    which satisfies ∇·v = ∂U/∂X + ∂V/∂Y = 0 exactly.

    In the GRB context, this models the helical motion of
    shocked shells in the relativistic jet, where the spiral
    structure arises from the rotational drag of the magnetic
    field anchored in the central engine.

    Parameters
    ----------
    n : int
        Number of evaluation points.
    x, y : ndarray, shape (n,)
        Normalized coordinates (0 ≤ x,y ≤ 1).
    c : float
        Winding parameter, typically 0.75–2.0.

    Returns
    -------
    u, v : ndarray, shape (n,)
        Velocity components.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    phi_x = phi_stream(x, c)
    phi_y = phi_stream(y, c)
    dphi_x = dphi_stream(x, c)
    dphi_y = dphi_stream(y, c)

    u = 10.0 * phi_x * dphi_y
    v = -10.0 * dphi_x * phi_y
    return u, v


def lorentz_factor(vx, vy, vz, c_light=2.99792458e10):
    """
    Computes the Lorentz factor:

        Γ = 1 / √(1 - v²/c²)

    with v² = vx² + vy² + vz².
    """
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    # Numerical robustness: clamp v2/c² to [0, 1-ε]
    beta2 = np.clip(v2 / c_light ** 2, 0.0, 1.0 - 1e-12)
    return 1.0 / np.sqrt(1.0 - beta2)


def relativistic_continuity_residual(rho, vr, vz, r, z, Gamma):
    """
    Computes the residual of the relativistic continuity equation
    in cylindrical coordinates on a structured grid:

        R = (1/r) ∂(r Γρ v_r)/∂r + ∂(Γρ v_z)/∂z

    Parameters
    ----------
    rho : ndarray
        Comoving mass density.
    vr, vz : ndarray
        Radial and axial velocities.
    r, z : ndarray
        Grid coordinates.
    Gamma : ndarray
        Lorentz factor.

    Returns
    -------
    residual : ndarray
        Continuity residual.
    """
    flux_r = r * Gamma * rho * vr
    flux_z = Gamma * rho * vz

    dr = r[1, 0] - r[0, 0] if r.shape[0] > 1 else 1.0
    dz = z[0, 1] - z[0, 0] if z.shape[1] > 1 else 1.0

    dfr_dr = np.zeros_like(flux_r)
    dfz_dz = np.zeros_like(flux_z)

    # Central differences for interior
    if flux_r.shape[0] > 2:
        dfr_dr[1:-1, :] = (flux_r[2:, :] - flux_r[:-2, :]) / (2.0 * dr)
        dfr_dr[0, :] = (flux_r[1, :] - flux_r[0, :]) / dr
        dfr_dr[-1, :] = (flux_r[-1, :] - flux_r[-2, :]) / dr

    if flux_z.shape[1] > 2:
        dfz_dz[:, 1:-1] = (flux_z[:, 2:] - flux_z[:, :-2]) / (2.0 * dz)
        dfz_dz[:, 0] = (flux_z[:, 1] - flux_z[:, 0]) / dz
        dfz_dz[:, -1] = (flux_z[:, -1] - flux_z[:, -2]) / dz

    # Avoid division by zero at r=0
    r_safe = np.where(r > 1e-12, r, 1e-12)
    residual = dfr_dr / r_safe + dfz_dz
    return residual


def compute_jet_profiles(n_r=32, n_z=64, r_max=1e13, z_max=1e15,
                         c_param=1.0, rho_0=1e-24, Gamma_0=300.0):
    """
    Compute axisymmetric GRB jet hydrodynamic profiles on a 2D grid.

    Returns
    -------
    dict with keys: r, z, rho, vr, vz, Gamma, residual
    """
    r, z = grid_2d(n_r, 0.0, r_max, n_z, 0.0, z_max)

    # Normalized coordinates for stream function
    x_norm = r / r_max
    y_norm = z / z_max

    # Flatten for vectorized uv_spiral, then reshape
    n_pts = n_r * n_z
    x_flat = x_norm.reshape(n_pts)
    y_flat = y_norm.reshape(n_pts)
    u_flat, v_flat = uv_spiral(n_pts, x_flat, y_flat, c_param)

    u = u_flat.reshape(n_r, n_z)
    v = v_flat.reshape(n_r, n_z)

    # Map (u,v) to physical (vr, vz) with Lorentz-boosted scaling
    # The spiral field is scaled to represent transverse and axial motions
    vr = u * 1e9  # cm/s
    vz = 0.9 * 2.99792458e10 + v * 1e7  # ultra-relativistic axial + perturbation

    # Ensure vz does not exceed c
    c_light = 2.99792458e10
    vz = np.clip(vz, 0.0, 0.99 * c_light)

    Gamma = lorentz_factor(vr, np.zeros_like(vr), vz)

    # Density profile: concentrated on axis, decreasing with z
    rho = rho_0 * (1.0 + 10.0 * np.exp(-(r / (0.1 * r_max)) ** 2)) * (z_max / (z + 1e10))
    rho = np.clip(rho, 1e-30, 1e-18)

    residual = relativistic_continuity_residual(rho, vr, vz, r, z, Gamma)

    return {
        "r": r,
        "z": z,
        "rho": rho,
        "vr": vr,
        "vz": vz,
        "Gamma": Gamma,
        "residual": residual,
    }
