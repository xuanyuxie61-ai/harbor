"""
diagnostics.py
==============
Physics diagnostics for the magnetic reconnection simulation:
  - Magnetic energy and kinetic energy budgets
  - Magnetic helicity
  - Current sheet thickness evolution
  - Reconnection rate (inflow/outflow Alfvén Mach numbers)
  - Energy conversion rate (magnetic → kinetic → thermal)
  - Sweet-Parker layer aspect ratio
  - Plasmoid statistics

Physical equations:
  - Magnetic energy: E_B = integral B^2 / (2 mu0) dV
  - Kinetic energy: E_K = integral rho v^2 / 2 dV
  - Thermal energy: E_th = integral p / (gamma - 1) dV
  - Magnetic helicity: H_m = integral A . B dV
  - Cross helicity: H_c = integral v . B dV
  - Current sheet half-thickness: delta_cs from Bx(z) profile fit
  - Reconnection rate: M_A = v_in / V_A_in = E_y / (B_in V_A_in)

Maps seed projects:
  - 1151: radius of gyration → magnetic energy "radius" (spatial extent)
  - 907_praxis: optimization of energy dissipation rate
  - 1051_boxinz17_smart: structured matrix decomposition →
    energy mode decomposition (POD-like)
"""

import numpy as np


# ============================================================
# Energy Diagnostics
# ============================================================

def magnetic_energy(Bx, Bz, grid, By=None):
    """
    Total magnetic energy:
        E_B = (1 / 2 mu0) integral B^2 dV

    In normalized units (mu0 = 1):
        E_B = 0.5 * integral (Bx^2 + Bz^2) dx dz

    If guide field By is present, include it:
        E_B += 0.5 * integral By^2 dx dz
    """
    B_sq = Bx ** 2 + Bz ** 2
    if By is not None:
        B_sq += By ** 2
    E_B = 0.5 * np.sum(B_sq) * grid.dA
    return E_B


def kinetic_energy(vx, vz, rho, grid, vy=None):
    """
    Total kinetic energy:
        E_K = integral 0.5 rho v^2 dV
    """
    v_sq = vx ** 2 + vz ** 2
    if vy is not None:
        v_sq += vy ** 2
    E_K = 0.5 * np.sum(rho * v_sq) * grid.dA
    return E_K


def thermal_energy(p, grid, gamma=5.0 / 3.0):
    """
    Total thermal energy:
        E_th = integral p / (gamma - 1) dV
    """
    E_th = np.sum(p) * grid.dA / (gamma - 1.0)
    return E_th


def total_energy(Bx, Bz, vx, vz, rho, p, grid, gamma=5.0 / 3.0):
    """Total MHD energy: E = E_B + E_K + E_th."""
    return (magnetic_energy(Bx, Bz, grid)
            + kinetic_energy(vx, vz, rho, grid)
            + thermal_energy(p, grid, gamma))


def energy_budget(history, grid, gamma=5.0 / 3.0):
    """
    Compute the energy budget over time from simulation history.

    Returns:
        times: array of times
        E_B, E_K, E_th, E_total: arrays of energies
        dE_B_dt, dE_K_dt: energy conversion rates
    """
    times = []
    E_B_list = []
    E_K_list = []
    E_th_list = []
    E_tot_list = []

    for t, state in history:
        Bx = state['Bx']
        Bz = state['Bz']
        vx = state.get('vx', np.zeros_like(Bx))
        vz = state.get('vz', np.zeros_like(Bx))
        rho = state.get('rho', np.ones_like(Bx))
        p = state.get('p', np.zeros_like(Bx))

        times.append(t)
        E_B_list.append(magnetic_energy(Bx, Bz, grid))
        E_K_list.append(kinetic_energy(vx, vz, rho, grid))
        E_th_list.append(thermal_energy(p, grid, gamma))
        E_tot_list.append(E_B_list[-1] + E_K_list[-1] + E_th_list[-1])

    times = np.array(times)
    E_B = np.array(E_B_list)
    E_K = np.array(E_K_list)
    E_th = np.array(E_th_list)
    E_total = np.array(E_tot_list)

    # Energy conversion rates (finite differences)
    dE_B_dt = np.gradient(E_B, times) if len(times) > 1 else np.zeros_like(E_B)
    dE_K_dt = np.gradient(E_K, times) if len(times) > 1 else np.zeros_like(E_K)

    return {
        'times': times,
        'E_B': E_B,
        'E_K': E_K,
        'E_th': E_th,
        'E_total': E_total,
        'dE_B_dt': dE_B_dt,
        'dE_K_dt': dE_K_dt,
    }


# ============================================================
# Magnetic Helicity
# ============================================================

def magnetic_helicity_2d(psi, Bz, grid):
    """
    Magnetic helicity in 2D (approximate):
        H_m ~ integral psi * Bz dx dz

    For 2D, A = (0, psi, 0) and B = (dpsi/dz, 0, -dpsi/dx),
    so A . B = psi * (-dpsi/dx) ... but in the reduced form:
        H_m = integral psi Bz dx dz

    Helicity is conserved even under resistive evolution (Taylor 1974),
    making it a key invariant for reconnection dynamics.
    """
    H_m = np.sum(psi * Bz) * grid.dA
    return H_m


def cross_helicity(vx, vz, Bx, Bz, grid):
    """
    Cross helicity:
        H_c = integral v . B dV = integral (vx Bx + vz Bz) dx dz

    This measures the alignment between velocity and magnetic field.
    In ideal MHD, H_c is also a conserved quantity.
    """
    H_c = np.sum(vx * Bx + vz * Bz) * grid.dA
    return H_c


# ============================================================
# Current Sheet Diagnostics
# ============================================================

def current_sheet_thickness(Bx, grid):
    """
    Estimate the current sheet half-thickness delta_cs from the
    Bx(z) profile at the X-point (center of domain in x).

    Method: fit Bx(z) at x = Lx/2 to a tanh profile:
        Bx(z) ~ tanh(z / delta_cs)
    and extract delta_cs from the maximum gradient:
        delta_cs = 1 / max|dBx/dz|

    This is analogous to the "radius of gyration" for polymers (1151):
    it measures the spatial extent of the current layer.
    """
    nx_mid = grid.nx // 2
    Bx_slice = Bx[nx_mid, :]

    # Compute gradient
    dBx_dz = np.gradient(Bx_slice, grid.dz)

    # Maximum gradient gives 1/delta_cs
    max_grad = np.max(np.abs(dBx_dz))
    if max_grad > 1e-10:
        delta_cs = 1.0 / max_grad
    else:
        delta_cs = grid.Lz

    return delta_cs


def current_sheet_aspect_ratio(Bx, Bz, grid):
    """
    Compute the Sweet-Parker layer aspect ratio: L / delta

    where L is the half-length of the current sheet and
    delta is the half-thickness.

    Sweet-Parker theory predicts:
        L / delta ~ S^{1/2}

    For the plasmoid instability, the critical aspect ratio is:
        (L / delta)_crit ~ S^{1/2} * (some function of S)
    """
    delta_cs = current_sheet_thickness(Bx, grid)

    # Find the length of the current sheet (where |J_y| is significant)
    from mhd_operators import curl_2d
    Jy = curl_2d(Bx, Bz, grid)
    Jy_max = np.max(np.abs(Jy))
    threshold = 0.1 * Jy_max if Jy_max > 0 else 0.0

    # Find extent along x where |Jy| > threshold at z=0 (center)
    nz_mid = grid.nz // 2
    active = np.abs(Jy[:, nz_mid]) > threshold
    if np.any(active):
        L_cs = np.sum(active) * grid.dx * 0.5
    else:
        L_cs = grid.Lx * 0.5

    aspect = L_cs / max(delta_cs, 1e-30)
    return aspect, L_cs, delta_cs


# ============================================================
# Reconnection Rate Diagnostics
# ============================================================

def reconnection_rate(Bx, Bz, vx, vz, eta, grid):
    """
    Measure the dimensionless reconnection rate.

    Method 1: Electric field at the X-point
        E_rec = |E_y(x_null, z_null)| = |eta J_y + (vz Bx - vx Bz)|

    Method 2: Inflow Alfvén Mach number
        M_A = v_in / V_A_in
    where v_in is the inflow velocity at the sheet edge and
    V_A_in = B_in / sqrt(mu0 rho) is the upstream Alfvén speed.

    Method 3: Outflow velocity
        v_out ~ V_A (for fast reconnection)

    Returns:
        dict with E_rec, M_A_in, v_out, and related quantities
    """
    # Find X-point (minimum |B|)
    B_sq = Bx ** 2 + Bz ** 2
    flat_idx = np.unravel_index(np.argmin(B_sq), B_sq.shape)
    ix_null, iz_null = flat_idx

    # Local values at the null
    Bx_null = Bx[ix_null, iz_null]
    Bz_null = Bz[ix_null, iz_null]
    vx_null = vx[ix_null, iz_null]
    vz_null = vz[ix_null, iz_null]

    # Current density at null
    from mhd_operators import curl_2d
    Jy = curl_2d(Bx, Bz, grid)
    Jy_null = Jy[ix_null, iz_null]

    # Electric field at null
    E_conv_null = vz_null * Bx_null - vx_null * Bz_null
    E_res_null = eta * Jy_null
    E_total_null = E_conv_null + E_res_null

    # Inflow: measure at some distance from the sheet
    nz_up = min(iz_null + grid.nz // 6, grid.nz - 1)
    B_in = np.sqrt(Bx[ix_null, nz_up] ** 2 + Bz[ix_null, nz_up] ** 2)
    v_in = abs(vz[ix_null, nz_up])
    rho_in = 1.0  # normalized

    V_A_in = B_in / np.sqrt(rho_in)
    M_A_in = v_in / max(V_A_in, 1e-30)

    # Outflow: measure along x at the sheet center
    nx_out = min(ix_null + grid.nx // 4, grid.nx - 1)
    v_out = abs(vx[nx_out, iz_null])

    return {
        'E_rec': abs(E_total_null),
        'E_conv': abs(E_conv_null),
        'E_res': abs(E_res_null),
        'M_A_in': M_A_in,
        'v_out': v_out,
        'B_in': B_in,
        'v_in': v_in,
        'x_null': grid.x[ix_null],
        'z_null': grid.z[iz_null],
    }


# ============================================================
# Plasmoid Statistics
# ============================================================

def count_plasmoids(Bx, Bz, grid, threshold=0.15):
    """
    Count the number of magnetic islands (plasmoids) formed during
    reconnection.

    Method: count O-type null points (local maxima of |psi|).

    In the plasmoid-unstable regime (S > S_crit ~ 10^4):
        Number of plasmoids ~ S^{3/8} (Loureiro et al. 2007)
        or N ~ (L/delta_SP)^{1/2} (Uzdensky et al. 2010)
    """
    from topology_analyzer import find_null_points
    nulls = find_null_points(Bx, Bz, grid, threshold)

    n_x = sum(1 for n in nulls if n['type'] == 'X')
    n_o = sum(1 for n in nulls if n['type'] == 'O')

    return {
        'n_plasmoids': n_o,
        'n_x_points': n_x,
        'nulls': nulls,
    }


# ============================================================
# Energy Mode Decomposition (maps to 1051 SMART)
# ============================================================

def energy_mode_decomposition(Bx, Bz, grid, n_modes=5):
    """
    Decompose the magnetic energy into spatial modes using a
    simplified Proper Orthogonal Decomposition (POD)-like approach.

    Maps to the SMART/structured matrix decomposition (1051_boxinz17_smart):
    the magnetic field tensor is decomposed into ranked modes by energy content.

    For 2D, we decompose the z-profiles of Bx(x, z) at each x into
    vertical modes:
        Bx(x, z) = sum_k a_k(x) phi_k(z)

    The mode energies E_k = 0.5 * integral |a_k(x)|^2 dx give the
    energy in each vertical mode.
    """
    nx, nz = Bx.shape

    # SVD of the Bx matrix
    U, s, Vt = np.linalg.svd(Bx, full_matrices=False)

    # Mode energies (proportional to singular values squared)
    mode_energies = 0.5 * s ** 2 * grid.dA
    total_energy = np.sum(mode_energies)

    # Fraction of energy in each mode
    energy_fraction = mode_energies / max(total_energy, 1e-30)
    cumulative_fraction = np.cumsum(energy_fraction)

    # Keep only n_modes
    n_modes = min(n_modes, len(s))

    return {
        'singular_values': s[:n_modes],
        'mode_energies': mode_energies[:n_modes],
        'energy_fractions': energy_fraction[:n_modes],
        'cumulative_fractions': cumulative_fraction[:n_modes],
        'total_energy': total_energy,
        'V_modes': Vt[:n_modes, :],  # vertical mode shapes
    }


# ============================================================
# Summary Report
# ============================================================

def generate_diagnostics_report(state, grid, plasma, history=None):
    """
    Generate a comprehensive diagnostics report.
    """
    Bx = state['Bx']
    Bz = state['Bz']
    vx = state.get('vx', np.zeros_like(Bx))
    vz = state.get('vz', np.zeros_like(Bx))
    rho = state.get('rho', np.ones_like(Bx))
    p = state.get('p', np.zeros_like(Bx))
    eta = plasma.eta_normalized

    E_B = magnetic_energy(Bx, Bz, grid)
    E_K = kinetic_energy(vx, vz, rho, grid)
    E_th = thermal_energy(p, grid, plasma.gamma_ad)
    E_tot = E_B + E_K + E_th

    delta_cs = current_sheet_thickness(Bx, grid)
    aspect, L_cs, _ = current_sheet_aspect_ratio(Bx, Bz, grid)

    rec = reconnection_rate(Bx, Bz, vx, vz, eta, grid)
    plas = count_plasmoids(Bx, Bz, grid)

    lines = [
        "=" * 60,
        "RECONNECTION DIAGNOSTICS REPORT",
        "=" * 60,
        f"  Energy Budget:",
        f"    E_magnetic  = {E_B:.6e}",
        f"    E_kinetic   = {E_K:.6e}",
        f"    E_thermal   = {E_th:.6e}",
        f"    E_total     = {E_tot:.6e}",
        f"    E_K/E_B     = {E_K / max(E_B, 1e-30):.4e}",
        f"  Current Sheet:",
        f"    delta_cs    = {delta_cs:.4e}",
        f"    L_cs        = {L_cs:.4e}",
        f"    Aspect L/d  = {aspect:.2f}",
        f"  Reconnection Rate:",
        f"    E_rec       = {rec['E_rec']:.4e}",
        f"    M_A (inflow)= {rec['M_A_in']:.4e}",
        f"    v_out       = {rec['v_out']:.4e}",
        f"  Plasmoids:",
        f"    N_islands   = {plas['n_plasmoids']}",
        f"    N_X-points  = {plas['n_x_points']}",
        "=" * 60,
    ]

    return "\n".join(lines)
