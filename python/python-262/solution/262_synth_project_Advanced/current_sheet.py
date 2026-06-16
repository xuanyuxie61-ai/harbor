"""
current_sheet.py
================
Initialization of Harris-type current sheet equilibria for magnetic
reconnection simulations, with perturbations to seed the tearing mode.

Physical background:
  - The Harris equilibrium (Harris 1962) is an exact 1D Vlasov equilibrium:
        B_x(z) = B_0 tanh(z/L)
        rho(z) = rho_0 / cosh^2(z/L) + rho_bg
        p(z) = B_0^2 / (2 mu0 cosh^2(z/L)) + p_bg

  - Force balance: grad p + grad(B^2/(2 mu0)) - (B.grad)B/mu0 = 0

  - Tearing mode perturbation: a flux function perturbation
        delta_psi = psi_0 cos(k_x x) sech^2(z/L)
    seeds the magnetic island (plasmoid) instability.

  - GEM challenge: the standard benchmark for reconnection codes
    (Birn et al. 2001, JGR)

Maps seed projects:
  - 329_ellipse_distance: elliptical perturbation geometry
  - 785_naca: current sheet profile shape parameterization
  - 990_r8poly: polynomial representation of equilibrium profiles
  - 1151: polymer topology → magnetic field line topology
"""

import numpy as np


def harris_equilibrium(grid, plasma, perturbation_amplitude=0.0,
                        k_pert=None, rho_bg_frac=0.2, p_bg_frac=0.2):
    """
    Initialize the Harris current sheet equilibrium on the grid.

    In normalized units (B0=1, rho0=1, L_cs=1):
        Bx(x, z) = tanh(z) + delta_B * perturbation
        Bz(x, z) = 0 (initially) + delta_Bz * perturbation
        rho(x, z) = sech^2(z) + rho_bg_frac
        p(x, z) = 0.5 * sech^2(z) + p_bg_frac  (from pressure balance)

    With tearing mode perturbation:
        delta_psi = A_pert * cos(k_x * x) * sech^2(z / L)
        delta_Bx = -d(delta_psi)/dz
        delta_Bz = d(delta_psi)/dx

    Input:
        grid: ReconnectionGrid object (coordinates in units of L_cs)
        plasma: SolarCoronaPlasma object
        perturbation_amplitude: amplitude of the initial perturbation
        k_pert: wavenumber of the perturbation (default: 2pi / Lx)

    Output:
        Bx, Bz, rho, p: 2D arrays of shape (nx, nz)
    """
    nx, nz = grid.nx, grid.nz
    x = grid.X  # (nx, nz)
    z = grid.Z

    # In normalized units, L_cs = 1.0. The current sheet is at z = Lz/2.
    L_cs_norm = 1.0
    z_norm = (z - 0.5 * grid.Lz) / L_cs_norm

    # Harris equilibrium profiles
    sech_z = 1.0 / np.cosh(z_norm)
    Bx = np.tanh(z_norm)
    Bz = np.zeros_like(Bx)
    rho = sech_z ** 2 + rho_bg_frac
    p = 0.5 * sech_z ** 2 + p_bg_frac

    # Add perturbation if requested
    if perturbation_amplitude > 0:
        if k_pert is None:
            k_pert = 2.0 * np.pi / max(grid.Lx, 1e-30)

        # Flux function perturbation (single X-point)
        delta_psi = (perturbation_amplitude
                     * np.cos(k_pert * (x - 0.5 * grid.Lx))
                     * sech_z ** 2)

        # delta_Bx = -d(delta_psi)/dz
        # d/dz[sech^2(z/L)] = -2 sech^2(z/L) tanh(z/L) / L
        dsech2_dz = -2.0 * sech_z ** 2 * np.tanh(z_norm) / L_cs_norm
        delta_Bx = -perturbation_amplitude * np.cos(
            k_pert * (x - 0.5 * grid.Lx)) * dsech2_dz

        # delta_Bz = d(delta_psi)/dx
        delta_Bz = -perturbation_amplitude * k_pert * np.sin(
            k_pert * (x - 0.5 * grid.Lx)) * sech_z ** 2

        Bx = Bx + delta_Bx
        Bz = delta_Bz

    return Bx, Bz, rho, p


def double_current_sheet(grid, plasma, separation=2.0, amplitude=1.0):
    """
    Initialize a double current sheet configuration, relevant for
    studying plasmoid-mediated reconnection in long current sheets.

    Two Harris sheets at z = ±separation/2:
        Bx = tanh(z + sep/2) - tanh(z - sep/2) - 1

    This creates a system with B_x → +1 for z >> sep/2,
    B_x → -1 for z << -sep/2, and current sheets at both boundaries.
    """
    nx, nz = grid.nx, grid.nz
    z_norm = (grid.Z - 0.5 * grid.Lz) / 1.0  # L_cs_norm = 1

    Bx = (amplitude * (np.tanh(z_norm + 0.5 * separation)
                        - np.tanh(z_norm - 0.5 * separation)
                        - 1.0))
    Bz = np.zeros_like(Bx)
    sech_p = 1.0 / np.cosh(z_norm + 0.5 * separation)
    sech_m = 1.0 / np.cosh(z_norm - 0.5 * separation)
    rho = 0.2 + sech_p ** 2 + sech_m ** 2
    p = 0.1 + 0.5 * (sech_p ** 2 + sech_m ** 2)

    return Bx, Bz, rho, p


def force_free_sheet(grid, plasma, alpha=1.0):
    """
    Force-free current sheet (J x B = 0):
        Bx(z) = B0 tanh(z/L)
        By(z) = B0 sech(z/L) / sqrt(alpha)

    This has J parallel to B, relevant for solar corona where
    the guide field is strong.
    """
    z_norm = (grid.Z - 0.5 * grid.Lz) / 1.0
    sech = 1.0 / np.cosh(z_norm)

    Bx = np.tanh(z_norm)
    By = sech / np.sqrt(max(alpha, 0.01))
    Bz = np.zeros_like(Bx)
    rho = sech ** 2 + 0.2
    p = 0.1 * np.ones_like(Bx)  # Uniform pressure for force-free

    return Bx, Bz, rho, p


def elliptical_perturbation(grid, plasma, a_axis, b_axis, center_x,
                              center_z, amplitude=0.1):
    """
    Add an elliptical Gaussian perturbation to the current sheet,
    modeling a localized flux emergence or magnetic island seed.

    Maps to the ellipse sampling geometry (329_ellipse_distance).

    The perturbation is:
        delta_psi = A * exp(-((x-cx)^2/a^2 + (z-cz)^2/b^2))
    """
    x = grid.X
    z = grid.Z

    r2 = ((x - center_x) / max(a_axis, 1e-30)) ** 2 + \
         ((z - center_z) / max(b_axis, 1e-30)) ** 2
    delta_psi = amplitude * np.exp(-r2)

    # Convert to magnetic field perturbation
    # delta_Bx = -d(delta_psi)/dz
    delta_Bx = delta_psi * 2.0 * (z - center_z) / max(b_axis, 1e-30) ** 2
    # delta_Bz = d(delta_psi)/dx
    delta_Bz = -delta_psi * 2.0 * (x - center_x) / max(a_axis, 1e-30) ** 2

    return delta_Bx, delta_Bz


def naca_profiled_sheet(grid, plasma, t_frac=0.12):
    """
    Create a current sheet with a thickness profile shaped like a
    NACA airfoil thickness distribution.

    This provides a more realistic profile for the reconnection layer
    boundary, with a rounded leading edge and sharp trailing edge.

    Maps to the NACA 4-digit symmetric airfoil (785_naca).
    """
    from polynomial_basis import naca_thickness_profile

    x_along = grid.X[0, :]  # x coordinate along the sheet
    L_cs = grid.Lx
    thickness = naca_thickness_profile(t_frac, x_along, L_cs)

    # Modulate the Harris profile by the local thickness
    z_norm_base = (grid.Z - 0.5 * grid.Lz) / 1.0

    # Local half-width modulation
    h_local = thickness / max(np.max(thickness), 1e-30) + 0.5
    z_modulated = z_norm_base / h_local[np.newaxis, :]

    Bx = np.tanh(z_modulated)
    Bz = np.zeros_like(Bx)
    sech = 1.0 / np.cosh(z_modulated)
    rho = sech ** 2 + 0.2
    p = 0.5 * sech ** 2 + 0.1

    return Bx, Bz, rho, p


def compute_current_density(Bx, Bz, grid):
    """
    Compute the current density J_y = (curl B)_y = dBx/dz - dBz/dx.
    In the Harris sheet, J_y peaks at z=0 (the neutral line).
    """
    from mhd_operators import curl_2d
    return curl_2d(Bx, Bz, grid)


def equilibrium_residual(Bx, Bz, rho, p, grid, gamma=5.0 / 3.0):
    """
    Compute the force balance residual for the equilibrium:
        R = |grad p + grad(B^2/2) - (B.grad)B| / |grad p|

    For a true equilibrium, R should be ~ machine epsilon.
    """
    from mhd_operators import grad_2d, d_dx_4th, d_dz_4th

    B_sq = Bx ** 2 + Bz ** 2

    # grad p
    dp_dx, dp_dz = grad_2d(p, grid)

    # grad(B^2/2)
    dBsq2_dx, dBsq2_dz = grad_2d(0.5 * B_sq, grid)

    # (B.grad)B_x = Bx dBx/dx + Bz dBx/dz
    Bdotgrad_Bx = Bx * d_dx_4th(Bx, grid.dx) + Bz * d_dz_4th(Bx, grid.dz)
    Bdotgrad_Bz = Bx * d_dx_4th(Bz, grid.dx) + Bz * d_dz_4th(Bz, grid.dz)

    # Residual
    Rx = dp_dx + dBsq2_dx - Bdotgrad_Bx
    Rz = dp_dz + dBsq2_dz - Bdotgrad_Bz

    R_mag = np.sqrt(Rx ** 2 + Rz ** 2)
    grad_p_mag = np.sqrt(dp_dx ** 2 + dp_dz ** 2) + 1e-30

    residual = np.max(R_mag) / np.max(grad_p_mag)
    return residual
