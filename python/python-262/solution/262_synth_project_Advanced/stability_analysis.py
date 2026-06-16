"""
stability_analysis.py
=====================
Von Neumann stability analysis and CFL condition computation for the
high-order finite-difference MHD reconnection solver.

Physical/mathematical background:
  - Von Neumann analysis: substitute Fourier mode exp(i k x - omega t)
    into the discrete equations and solve for the amplification factor g(k)
  - Stability requires |g(k)| <= 1 for all resolvable wavenumbers k
  - CFL condition: dt <= C * dx / max(|v| + V_A) where C depends on
    the order and type of the scheme

Key equations:
  - Ideal MHD wave speeds: fast magnetosonic v_f, Alfvén v_A, slow v_s
  - CFL for explicit schemes: dt_CFL = CFL_num * dx_min / (|v| + v_f)
  - Lundquist-based constraint: dt_eta <= dx^2 / (2 eta) for diffusion
  - Modified wavenumber for FD schemes: k' dx = sum w_j sin(k j dx)

Maps seed projects:
  - 127_burgers_time_viscous: viscous stability dt ~ dx^2/nu
  - 907_praxis: optimization to find critical stability boundary
  - 990_r8poly: dispersion relation polynomial in exp(ik dx)
"""

import numpy as np


# ============================================================
# Modified Wavenumber Analysis
# ============================================================

def modified_wavenumber_fd4(k, dx):
    """
    Modified (effective) wavenumber for 4th-order central FD 1st derivative.

    For the scheme:
        f'_j ~ (-f_{j+2} + 8 f_{j+1} - 8 f_{j-1} + f_{j-2}) / (12 dx)

    Substituting f_j = exp(i k j dx):
        ik' = [-exp(2ikdx) + 8 exp(ikdx) - 8 exp(-ikdx) + exp(-2ikdx)]
              / (12 dx)
            = i [-2 sin(2kdx) + 16 sin(kdx)] / (12 dx)

    So: k' dx = [-2 sin(2 k dx) + 16 sin(k dx)] / 12
              = [16 sin(kdx) - 2 sin(2 kdx)] / 12
              = [16 sin(theta) - 4 sin(theta) cos(theta)] / 12
    where theta = k dx.
    """
    theta = k * dx
    k_prime_dx = (16.0 * np.sin(theta) - 2.0 * np.sin(2.0 * theta)) / 12.0
    return k_prime_dx


def modified_wavenumber_fd6(k, dx):
    """
    Modified wavenumber for 6th-order central FD 1st derivative.

    7-point stencil:
        f'_j = (f_{j+3} - 9 f_{j+2} + 45 f_{j+1} - 45 f_{j-1}
                + 9 f_{j-2} - f_{j-3}) / (60 dx)

    k' dx = [2 sin(3 theta) - 18 sin(2 theta) + 90 sin(theta)] / 60
    """
    theta = k * dx
    k_prime_dx = (2.0 * np.sin(3.0 * theta)
                  - 18.0 * np.sin(2.0 * theta)
                  + 90.0 * np.sin(theta)) / 60.0
    return k_prime_dx


def modified_wavenumber_fd2(k, dx):
    """
    Modified wavenumber for 2nd-order central FD (reference):
        k' dx = sin(k dx)
    """
    return np.sin(k * dx)


def modified_wavenumber_2nd_deriv_fd4(k, dx):
    """
    Modified wavenumber for 4th-order 2nd derivative.

    f''_j ~ (-f_{j+2} + 16 f_{j+1} - 30 f_j + 16 f_{j-1} - f_{j-2})
            / (12 dx^2)

    -(k')^2 dx^2 = [-exp(2i theta) + 16 exp(i theta) - 30
                     + 16 exp(-i theta) - exp(-2i theta)] / 12
                  = [-2 cos(2 theta) + 32 cos(theta) - 30] / 12

    So: (k' dx)^2 = [30 - 32 cos(theta) + 2 cos(2 theta)] / 12
    """
    theta = k * dx
    k_prime_sq_dx2 = (30.0 - 32.0 * np.cos(theta)
                      + 2.0 * np.cos(2.0 * theta)) / 12.0
    return np.sqrt(np.maximum(k_prime_sq_dx2, 0.0))


# ============================================================
# CFL Condition
# ============================================================

def cfl_condition(vx_max, vz_max, V_A, cs, dx, dz, cfl_num=0.4):
    """
    Compute the maximum stable time step from the CFL condition.

    For ideal MHD, the maximum wave speed is the fast magnetosonic speed:
        v_f = sqrt(0.5 * (v_A^2 + c_s^2 + sqrt((v_A^2 + c_s^2)^2
                - 4 v_A^2 c_s^2 cos^2(theta_B))))

    For anti-parallel reconnection (cos(theta_B) ~ 1):
        v_f ~ max(v_A, c_s)

    CFL: dt <= CFL_num * min(dx, dz) / v_max

    The factor 0.4 is typical for 4th-order RK + 4th-order FD.
    For 2nd-order schemes, use ~0.5; for 6th-order, use ~0.3.
    """
    v_f = np.sqrt(0.5 * (V_A ** 2 + cs ** 2
                + np.sqrt((V_A ** 2 + cs ** 2) ** 2
                          - 4.0 * V_A ** 2 * cs ** 2)))
    v_max = max(abs(vx_max), abs(vz_max)) + v_f

    dx_min = min(dx, dz)
    dt_cfl = cfl_num * dx_min / max(v_max, 1e-30)
    return dt_cfl, v_f, v_max


def diffusion_cfl(eta, dx, dz, safety=0.4):
    """
    CFL condition for the resistive diffusion term.

    For explicit treatment of eta nabla^2 B:
        dt_eta <= safety * dx^2 / (2 * dim * eta)

    where dim = 2 for 2D.
    """
    dx_min = min(dx, dz)
    dt_eta = safety * dx_min ** 2 / (4.0 * max(eta, 1e-30))
    return dt_eta


def viscous_cfl(nu, dx, dz, safety=0.4):
    """
    CFL condition for viscous diffusion (maps to Burgers equation).

    For explicit nu nabla^2 v:
        dt_nu <= safety * dx^2 / (2 * dim * nu)
    """
    dx_min = min(dx, dz)
    dt_nu = safety * dx_min ** 2 / (4.0 * max(nu, 1e-30))
    return dt_nu


# ============================================================
# Von Neumann Stability Analysis
# ============================================================

def von_neumann_advection(k_array, v_adv, dt, dx, scheme='fd4'):
    """
    Von Neumann analysis for the advection equation:
        du/dt + v du/dx = 0

    Amplification factor g(k) for various FD schemes.

    For RK4 + FD4:
        g = 1 + z + z^2/2 + z^3/6 + z^4/24
        where z = -i v dt k' (modified wavenumber)

    Stability requires |g(k)| <= 1 for all k in [0, pi/dx].
    """
    if scheme == 'fd4':
        k_prime_dx = modified_wavenumber_fd4(k_array, dx)
    elif scheme == 'fd6':
        k_prime_dx = modified_wavenumber_fd6(k_array, dx)
    else:
        k_prime_dx = modified_wavenumber_fd2(k_array, dx)

    z = -1j * v_adv * dt * k_prime_dx / dx

    # RK4 amplification polynomial: g = 1 + z + z^2/2 + z^3/6 + z^4/24
    g = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

    return g, np.abs(g)


def von_neumann_diffusion(k_array, eta, dt, dx, scheme='fd4'):
    """
    Von Neumann analysis for the diffusion equation:
        du/dt = eta d^2u/dx^2

    Amplification factor:
        g = 1 + z + z^2/2 + z^3/6 + z^4/24
        where z = -eta dt (k')^2 (real, negative)

    For stability, z must satisfy |g(z)| <= 1.
    For RK4, the stability region on the negative real axis extends
    to approximately z = -2.785.

    So: eta dt (k')^2_max <= 2.785
    """
    if scheme == 'fd4':
        k_prime_dx = modified_wavenumber_2nd_deriv_fd4(k_array, dx)
    else:
        k_prime_dx = np.sin(k_array * dx)

    z = -eta * dt * k_prime_dx ** 2 / dx ** 2
    g = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

    return g, np.abs(g)


def von_neumann_mhd_full(kx_array, kz_array, Bx0, Bz0, rho0, eta, nu,
                          dt, dx, dz):
    """
    Full von Neumann analysis for the linearized resistive MHD equations.

    Linearize around a uniform state (Bx0, 0, Bz0), rho0:
        dB1/dt = ik x (v1 x B0) - eta k^2 B1
        dv1/dt = ik (B0 . B1) / (mu0 rho0) - nu k^2 v1

    The dispersion relation is a polynomial in omega = omega_r + i omega_i.
    Instability when omega_i > 0.

    Returns the amplification factor for each (kx, kz) mode.
    """
    nkx = len(kx_array)
    nkz = len(kz_array)
    g_max = np.zeros((nkx, nkz))

    mu0 = 1.0  # normalized

    for ix, kx in enumerate(kx_array):
        for iz, kz in enumerate(kz_array):
            k2 = kx ** 2 + kz ** 2
            if k2 < 1e-30:
                g_max[ix, iz] = 1.0
                continue

            # Alfvén frequency: omega_A = k . B0 / sqrt(mu0 rho0)
            omega_A = (kx * Bx0 + kz * Bz0) / np.sqrt(mu0 * rho0)

            # Diffusion rate
            gamma_diff = eta * k2 + nu * k2

            # Eigenvalues: omega = ±omega_A - i gamma_diff
            # Amplification (Euler): g = 1 - i omega dt
            omega_plus = omega_A - 1j * gamma_diff
            omega_minus = -omega_A - 1j * gamma_diff

            g_plus = np.exp(-1j * omega_plus * dt)
            g_minus = np.exp(-1j * omega_minus * dt)

            g_max[ix, iz] = max(abs(g_plus), abs(g_minus))

    return g_max


# ============================================================
# Stability Boundary Search (maps to 907_praxis)
# ============================================================

def find_critical_cfl(scheme='fd4', n_k=1000):
    """
    Find the critical CFL number for the given FD scheme + RK time integrator.

    For RK4 + FD4 advection:
        The stability region boundary intersects the imaginary axis at
        approximately |z| = 2*sqrt(2) ~ 2.828.

    The critical CFL is then:
        CFL_crit = |z_crit| / max|k' dx| over k in [0, pi/dx]

    For FD4: max|k' dx| ~ 1.26 (at k dx ~ 1.7)
    CFL_crit ~ 2.828 / 1.26 ~ 2.24 (for advection, but MHD has
    additional wave modes).
    """
    k_array = np.linspace(0, np.pi, n_k)

    if scheme == 'fd4':
        k_prime_dx = modified_wavenumber_fd4(k_array, 1.0)
    elif scheme == 'fd6':
        k_prime_dx = modified_wavenumber_fd6(k_array, 1.0)
    else:
        k_prime_dx = np.sin(k_array)

    max_k_prime = np.max(np.abs(k_prime_dx))

    # RK4 stability boundary on imaginary axis: |z| <= 2*sqrt(2)
    z_crit = 2.0 * np.sqrt(2.0)

    cfl_crit = z_crit / max(max_k_prime, 1e-30)
    return cfl_crit, max_k_prime, z_crit


def dispersion_relation_ideal_mhd(kx, kz, Bx0, Bz0, rho0, p0, gamma):
    """
    Dispersion relation for ideal MHD waves:

    omega^4 - omega^2 k^2 (v_A^2 + c_s^2) + k^2 v_A^2 k_par^2 c_s^2 = 0

    where k_par = k . B0 / |B0| is the parallel wavenumber.

    Solutions:
        omega^2 = 0.5 k^2 [(v_A^2 + c_s^2)
                    ± sqrt((v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 cos^2(theta))]

    Three MHD wave families:
        - Fast magnetosonic: + branch (highest speed)
        - Alfvén: omega = k_par v_A (incompressible)
        - Slow magnetosonic: - branch (lowest speed)

    Returns: (omega_fast, omega_alfven, omega_slow)
    """
    k2 = kx ** 2 + kz ** 2
    k = np.sqrt(k2)
    B0_sq = Bx0 ** 2 + Bz0 ** 2
    v_A2 = B0_sq / rho0  # mu0 = 1
    c_s2 = gamma * p0 / rho0

    cos2_theta = ((kx * Bx0 + kz * Bz0) ** 2) / (k2 * B0_sq + 1e-30)

    discriminant = (v_A2 + c_s2) ** 2 - 4.0 * v_A2 * c_s2 * cos2_theta
    discriminant = max(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    omega_fast_sq = 0.5 * k2 * (v_A2 + c_s2 + sqrt_disc)
    omega_slow_sq = 0.5 * k2 * (v_A2 + c_s2 - sqrt_disc)
    omega_alfven_sq = k2 * (kx * Bx0 + kz * Bz0) ** 2 / (k2 * B0_sq + 1e-30)

    return (np.sqrt(max(omega_fast_sq, 0.0)),
            np.sqrt(max(omega_alfven_sq, 0.0)),
            np.sqrt(max(omega_slow_sq, 0.0)))


# ============================================================
# Tearing Mode Instability Growth Rate
# ============================================================

def tearing_growth_rate(k_tearing, S, L_cs=1.0, gamma_ad=5.0 / 3.0):
    """
    Linear tearing mode instability growth rate in the resistive MHD regime.

    Furth-Killeen-Rosenbluth (FKR) theory:
        gamma_FKR ~ (k L_cs)^{1/2} * S^{-3/5} * (V_A / L_cs)

    Coppi regime (large k):
        gamma_Coppi ~ S^{-1/3} * (V_A / L_cs)

    The maximum growth rate occurs at k_max L_cs ~ S^{-1/4}:
        gamma_max ~ S^{-1/2} * (V_A / L_cs)  (Plasmoid unstable regime)

    For the plasmoid instability (S > S_crit ~ 10^4):
        gamma_plasmoid ~ S^{1/4} * (V_A / L_cs)  (Loureiro et al. 2007)

    Input:
        k_tearing: wavenumber of the tearing mode
        S: Lundquist number
        L_cs: current sheet half-thickness (normalized = 1)
    """
    if S <= 0:
        return 0.0

    # FKR rate
    kL = k_tearing * L_cs
    if kL > 0:
        gamma_fkr = kL ** 0.5 * S ** (-0.6)
    else:
        gamma_fkr = 0.0

    # Coppi rate
    gamma_coppi = S ** (-1.0 / 3.0)

    # Plasmoid regime (for S > 10^4)
    if S > 1e4:
        gamma_plasmoid = S ** 0.25 * (1.0 / (1.0 + (kL * S ** 0.25) ** 2))
    else:
        gamma_plasmoid = 0.0

    # Return the dominant rate
    return max(gamma_fkr, gamma_coppi, gamma_plasmoid)
