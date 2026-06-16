"""
粘性流体力学演化模块
2+1D relativistic viscous hydrodynamics for QGP evolution.

Evolution equations:
  ∂_μ T^{μν} = 0
  T^{μν} = (ε + P + Π) u^μ u^ν - (P + Π) g^{μν} + π^{μν}

where Π = bulk viscous pressure, π^{μν} = shear stress tensor.

Algorithms sourced from:
  - 410_fem2d_predator_prey_fast (fast stencil operations)
  - 846_paraheat_functional (parabolic evolution)
  - 1073_balloon_dynamics (radial expansion, adaptive CFL)
"""
import numpy as np
from high_order_fd import (
    laplacian_2d, gradient_2d, apply_1d_derivative, apply_2nd_derivative
)
from equation_of_state import (
    pressure_lattice, energy_density_lattice, entropy_density_lattice,
    sound_speed_squared, temperature_from_energy_density,
    shear_viscosity_eta_over_s, bulk_viscosity_zeta_over_s,
    relaxation_time_tau_pi, enthalpy_density
)


def ideal_stress_tensor(eps, p, ux, uy, uz):
    """
    Ideal (perfect fluid) energy-momentum tensor:
        T^{μν}_ideal = (ε + P) u^μ u^ν - P g^{μν}

    For 2+1D (τ, x, y) with Bjorken scaling:
        u^μ = (cosh η, v_x, v_y, sinh η) ≈ (1, v_x, v_y, 0) at mid-rapidity

    Parameters
    ----------
    eps : ndarray (nx, ny)
        Energy density
    p : ndarray (nx, ny)
        Pressure
    ux, uy, uz : ndarray (nx, ny)
        Four-velocity components (spatial)

    Returns
    -------
    t00, t0x, t0y, txx, txy, tyy : ndarray
        Components of T^{μν}
    """
    w = eps + p  # enthalpy density
    u0 = np.sqrt(1.0 + ux**2 + uy**2)  # time component

    t00 = w * u0**2 - p
    t0x = w * u0 * ux
    t0y = w * u0 * uy
    txx = w * ux**2 + p
    txy = w * ux * uy
    tyy = w * uy**2 + p

    return t00, t0x, t0y, txx, txy, tyy


def shear_tensor_pi_munu(ux, uy, dx, dy, eta):
    """
    Shear stress tensor π^{μν} in Navier-Stokes approximation:
        π^{μν} = 2 η σ^{μν}

    where σ^{μν} = (1/2)(∂^μ u^ν + ∂^ν u^μ) - (1/3) Δ^{μν} (∂·u)

    For 2+1D at mid-rapidity, we compute spatial components:
        π^{xx} = 2η (∂_x u_x - (1/3)(∂_x u_x + ∂_y u_y))
        π^{yy} = 2η (∂_y u_y - (1/3)(∂_x u_x + ∂_y u_y))
        π^{xy} = η (∂_x u_y + ∂_y u_x)

    Parameters
    ----------
    ux, uy : ndarray (nx, ny)
        Velocity components
    dx, dy : float
        Grid spacing
    eta : ndarray (nx, ny)
        Shear viscosity η

    Returns
    -------
    pi_xx, pi_yy, pi_xy : ndarray
        Shear stress tensor components
    """
    # Compute velocity gradients (4th order)
    dux_dx = apply_1d_derivative(ux, dx, order=4, axis=0)
    duy_dy = apply_1d_derivative(uy, dy, order=4, axis=1)
    dux_dy = apply_1d_derivative(ux, dy, order=4, axis=1)
    duy_dx = apply_1d_derivative(uy, dx, order=4, axis=0)

    # Expansion rate
    theta = dux_dx + duy_dy

    # Shear tensor components
    pi_xx = 2.0 * eta * (dux_dx - theta / 3.0)
    pi_yy = 2.0 * eta * (duy_dy - theta / 3.0)
    pi_xy = eta * (dux_dy + duy_dx)

    return pi_xx, pi_yy, pi_xy


def bulk_viscous_pressure(theta, zeta):
    """
    Bulk viscous pressure:
        Π = -ζ (∂·u)

    Parameters
    ----------
    theta : ndarray
        Expansion rate ∂·u
    zeta : ndarray
        Bulk viscosity ζ

    Returns
    -------
    ndarray
        Bulk pressure Π
    """
    return -zeta * theta


def compute_velocity_from_momentum(mx, my, eps_plus_p):
    """
    Recover velocity from momentum density:
        v^i = T^{0i} / (ε + P)

    With constraint u^μ u_μ = 1 → v = p / √((ε+P)² + p²)

    Parameters
    ----------
    mx, my : ndarray
        Momentum density T^{0x}, T^{0y}
    eps_plus_p : ndarray
        Enthalpy density w = ε + P

    Returns
    -------
    ux, uy : ndarray
        Spatial four-velocity components
    """
    eps_plus_p = np.maximum(eps_plus_p, 1e-10)
    v_mag_sq = (mx**2 + my**2) / eps_plus_p**2
    lorentz_factor = np.sqrt(1.0 + v_mag_sq)

    ux = mx / eps_plus_p
    uy = my / eps_plus_p

    # Bound velocity to prevent superluminal flow
    v_max = 0.99
    ux = np.clip(ux, -v_max, v_max)
    uy = np.clip(uy, -v_max, v_max)

    return ux, uy


def hydro_step_rk2(eps, mx, my, dx, dy, dt, fd_order=4):
    """
    Single time step of 2+1D ideal hydrodynamics using RK2 (Heun's method).

    Evolution equations (conservative form):
        ∂_τ ε = -∂_x (w v_x) - ∂_y (w v_y)
        ∂_τ (w v_x) = -∂_x (w v_x² + P) - ∂_y (w v_x v_y)
        ∂_τ (w v_y) = -∂_x (w v_x v_y) - ∂_y (w v_y² + P)

    Parameters
    ----------
    eps : ndarray (nx, ny)
        Energy density
    mx, my : ndarray
        Momentum densities
    dx, dy : float
        Grid spacing
    dt : float
        Time step
    fd_order : int
        Finite difference order

    Returns
    -------
    eps_new, mx_new, my_new : ndarray
        Updated conserved variables
    """
    def rhs(eps_cur, mx_cur, my_cur):
        # Recover primitive variables
        p_cur = pressure_lattice(temperature_from_energy_density(eps_cur))
        w_cur = eps_cur + p_cur
        w_cur = np.maximum(w_cur, 1e-10)

        ux_cur, uy_cur = compute_velocity_from_momentum(mx_cur, my_cur, w_cur)

        # Fluxes
        fx_eps = w_cur * ux_cur
        fy_eps = w_cur * uy_cur

        fx_mx = w_cur * ux_cur**2 + p_cur
        fy_mx = w_cur * ux_cur * uy_cur

        fx_my = w_cur * ux_cur * uy_cur
        fy_my = w_cur * uy_cur**2 + p_cur

        # Divergence of fluxes (using high-order FD)
        deps_dtau = -(apply_1d_derivative(fx_eps, dx, order=fd_order, axis=0)
                      + apply_1d_derivative(fy_eps, dy, order=fd_order, axis=1))
        dmx_dtau = -(apply_1d_derivative(fx_mx, dx, order=fd_order, axis=0)
                     + apply_1d_derivative(fy_mx, dy, order=fd_order, axis=1))
        dmy_dtau = -(apply_1d_derivative(fx_my, dx, order=fd_order, axis=0)
                     + apply_1d_derivative(fy_my, dy, order=fd_order, axis=1))

        return deps_dtau, dmx_dtau, dmy_dtau

    # RK2: predictor
    k1_eps, k1_mx, k1_my = rhs(eps, mx, my)

    # RK2: corrector (Heun's method)
    eps_pred = eps + dt * k1_eps
    mx_pred = mx + dt * k1_mx
    my_pred = my + dt * k1_my

    k2_eps, k2_mx, k2_my = rhs(eps_pred, mx_pred, my_pred)

    eps_new = eps + 0.5 * dt * (k1_eps + k2_eps)
    mx_new = mx + 0.5 * dt * (k1_mx + k2_mx)
    my_new = my + 0.5 * dt * (k1_my + k2_my)

    # Enforce positivity of energy density
    eps_new = np.maximum(eps_new, 1e-6)

    return eps_new, mx_new, my_new


def viscous_correction_to_stress(eps, ux, uy, dx, dy, dt, temperature):
    """
    Compute viscous corrections to energy-momentum tensor.

    Israel-Stewart second-order equations:
        τ_π Δ^{μα} Δ^{νβ} D π_{αβ} + π^{μν} = 2η σ^{μν} + ...

    Simplified: relax π^{μν} toward Navier-Stokes value.

    Parameters
    ----------
    eps : ndarray
        Energy density
    ux, uy : ndarray
        Velocity components
    dx, dy : float
        Grid spacing
    dt : float
        Time step
    temperature : ndarray
        Temperature field

    Returns
    -------
    pi_xx, pi_yy, pi_xy : ndarray
        Shear stress tensor
    bulk_pi : ndarray
        Bulk viscous pressure
    """
    T = temperature
    eta_over_s = shear_viscosity_eta_over_s(T)
    zeta_over_s = bulk_viscosity_zeta_over_s(T)
    s = entropy_density_lattice(T)
    eta = eta_over_s * s
    zeta = zeta_over_s * s

    # Velocity gradients
    dux_dx = apply_1d_derivative(ux, dx, order=4, axis=0)
    duy_dy = apply_1d_derivative(uy, dy, order=4, axis=1)
    dux_dy = apply_1d_derivative(ux, dy, order=4, axis=1)
    duy_dx = apply_1d_derivative(uy, dx, order=4, axis=0)
    theta = dux_dx + duy_dy

    # Navier-Stokes target
    pi_xx_ns = 2.0 * eta * (dux_dx - theta / 3.0)
    pi_yy_ns = 2.0 * eta * (duy_dy - theta / 3.0)
    pi_xy_ns = eta * (dux_dy + duy_dx)
    bulk_pi_ns = -zeta * theta

    # Israel-Stewart relaxation (simplified)
    tau_pi = relaxation_time_tau_pi(T, eta_over_s)
    relax_factor = dt / (tau_pi + dt)

    # For simplicity, assume π starts at 0 and relaxes toward NS
    pi_xx = relax_factor * pi_xx_ns
    pi_yy = relax_factor * pi_yy_ns
    pi_xy = relax_factor * pi_xy_ns
    bulk_pi = relax_factor * bulk_pi_ns

    return pi_xx, pi_yy, pi_xy, bulk_pi


def apply_boundary_conditions(eps, mx, my, bc_type='outflow'):
    """
    Apply boundary conditions to conserved variables.

    Parameters
    ----------
    eps, mx, my : ndarray
        Conserved variables
    bc_type : str
        'outflow' (zero gradient) or 'reflecting'

    Returns
    -------
    ndarray
        Updated fields with BCs applied
    """
    nx, ny = eps.shape

    if bc_type == 'outflow':
        # Zero-gradient (copy interior to boundary)
        eps[0, :] = eps[1, :]
        eps[-1, :] = eps[-2, :]
        eps[:, 0] = eps[:, 1]
        eps[:, -1] = eps[:, -2]

        mx[0, :] = mx[1, :]
        mx[-1, :] = mx[-2, :]
        mx[:, 0] = mx[:, 1]
        mx[:, -1] = mx[:, -2]

        my[0, :] = my[1, :]
        my[-1, :] = my[-2, :]
        my[:, 0] = my[:, 1]
        my[:, -1] = my[:, -2]

    elif bc_type == 'reflecting':
        # Reflect normal velocity at boundaries
        eps[0, :] = eps[1, :]
        eps[-1, :] = eps[-2, :]
        eps[:, 0] = eps[:, 1]
        eps[:, -1] = eps[:, -2]

        mx[0, :] = -mx[1, :]
        mx[-1, :] = -mx[-2, :]
        mx[:, 0] = mx[:, 1]
        mx[:, -1] = mx[:, -2]

        my[0, :] = my[1, :]
        my[-1, :] = my[-2, :]
        my[:, 0] = -my[:, 1]
        my[:, -1] = -my[:, -2]

    return eps, mx, my


def freezeout_sampler(eps, temperature, t_freeze=0.120):
    """
    Identify freezeout surface cells where T < T_freeze.

    Parameters
    ----------
    eps : ndarray
        Energy density
    temperature : ndarray
        Temperature field
    t_freeze : float
        Freezeout temperature in GeV

    Returns
    -------
    ndarray (bool)
        Mask of freezeout cells
    """
    return temperature < t_freeze


def cooper_frye_integrand(eps, p, ux, uy, particle_mass, momentum):
    """
    Cooper-Frye freezeout integrand for particle spectrum:
        dN/d³p = ∫_Σ f(x,p) p^μ dσ_μ

    For Boltzmann distribution:
        f = exp(-(p·u - μ)/T) ≈ exp(-p·u/T)

    Parameters
    ----------
    eps : ndarray
        Energy density at freezeout
    p : ndarray
        Pressure
    ux, uy : ndarray
        Velocity components
    particle_mass : float
        Particle mass in GeV
    momentum : tuple (px, py, pz)
        Particle momentum in GeV

    Returns
    -------
    ndarray
        Integrand values
    """
    px, py, pz = momentum
    T = temperature_from_energy_density(eps)
    T = np.maximum(T, 0.05)  # Avoid division by zero

    u0 = np.sqrt(1.0 + ux**2 + uy**2)
    p_dot_u = px * ux + py * uy  # Simplified for mid-rapidity

    E_particle = np.sqrt(particle_mass**2 + px**2 + py**2 + pz**2)
    exponent = -(E_particle * u0 - p_dot_u) / T

    # Boltzmann factor
    f_boltz = np.exp(np.clip(exponent, -50, 0))

    return f_boltz
