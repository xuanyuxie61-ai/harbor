"""
transport.py

Mass and heat transport equations at the solid-liquid interface.

Synthesizes concepts from:
    - 1069_shallow_water_1d_display: 1D hyperbolic conservation law solver
    - 908_predator_prey_ode: Coupled reaction-diffusion ODEs
    - 1025_ripple_ode: Nonlinear wave dynamics
    
Physical Model:
    At the solid-liquid interface, species transport is governed by the
    advection-diffusion-reaction equation:
    
        dc/dt + v * dc/dz = D * d^2c/dz^2 + R(c)
    
    where:
        - c(z,t) is the concentration of species B
        - v is the interface velocity
        - D is the diffusion coefficient
        - R(c) is the reaction term (dissolution/precipitation)
    
    Discretization using finite differences:
        dc/dt = -v * (c_{i+1} - c_{i-1}) / (2*dz)
                + D * (c_{i+1} - 2*c_i + c_{i-1}) / dz^2
                + R(c_i)
    
    Interface tracking:
        dz_interface/dt = v = M * (mu_liquid - mu_solid)
    
    where M is the interface mobility and mu is the chemical potential.
    
    Chemical potential (ideal solution model):
        mu = mu^0 + k_B * T * ln(c)
    
    Shock-capturing for hyperbolic terms:
        We use a minmod flux limiter to prevent oscillations.
    
    The shallow water equations analogy:
        The interface dynamics can be mapped to a 1D conservation law:
            d(phi)/dt + d(flux)/dz = source
        where phi is the phase fraction (0=liquid, 1=solid).
"""

import numpy as np
from config import (
    DIFFUSION_COEFFICIENT_SOLID, DIFFUSION_COEFFICIENT_LIQUID,
    INTERFACE_MOBILITY, DISSOLUTION_RATE, PRECIPITATION_RATE,
    TIME_STEP
)


# =============================================================================
# 1D Finite Difference Solver for Advection-Diffusion-Reaction
# =============================================================================

def minmod(a, b):
    """
    Minmod flux limiter for shock capturing.
    
    minmod(a, b) = sign(a) * min(|a|, |b|)  if a*b > 0
                   = 0                        otherwise
    
    This is a total variation diminishing (TVD) limiter that prevents
    spurious oscillations near discontinuities.
    
    Args:
        a, b: slope estimates
        
    Returns:
        limited slope
    """
    if a * b <= 0:
        return 0.0
    return np.sign(a) * min(abs(a), abs(b))


def advection_flux(c, v, dz):
    """
    Compute advective flux using upwind scheme with minmod limiter.
    
    F_{i+1/2} = v * c_i + 0.5 * v * (1 - v*dt/dz) * delta_i
    
    where delta_i is the limited slope.
    
    Args:
        c: concentration array
        v: velocity (scalar)
        dz: grid spacing
        
    Returns:
        flux array at cell interfaces
    """
    n = len(c)
    flux = np.zeros(n + 1)
    
    for i in range(1, n):
        # Left and right slopes
        if i > 1:
            slope_left = (c[i - 1] - c[i - 2]) if i >= 2 else 0.0
        else:
            slope_left = 0.0
        
        if i < n - 1:
            slope_right = (c[i + 1] - c[i]) if i < n - 1 else 0.0
        else:
            slope_right = 0.0
        
        slope_center = c[i] - c[i - 1]
        
        # Minmod limiter
        if v >= 0:
            delta = minmod(slope_center, slope_left)
            flux[i] = v * c[i - 1] + 0.5 * v * delta
        else:
            delta = minmod(slope_center, slope_right)
            flux[i] = v * c[i] - 0.5 * abs(v) * delta
    
    # Boundary fluxes
    flux[0] = v * c[0]
    flux[n] = v * c[n - 1]
    
    return flux


def reaction_term(c_solid, c_liquid, rate_dissolve=DISSOLUTION_RATE,
                  rate_precipitate=PRECIPITATION_RATE):
    """
    Compute reaction terms for species exchange at interface.
    
    R_s = -k_d * c_s + k_p * c_l
    R_l = +k_d * c_s - k_p * c_l
    
    Args:
        c_solid: solid concentration
        c_liquid: liquid concentration
        rate_dissolve: dissolution rate constant
        rate_precipitate: precipitation rate constant
        
    Returns:
        R_s, R_l
    """
    R_s = -rate_dissolve * c_solid + rate_precipitate * c_liquid
    R_l = rate_dissolve * c_solid - rate_precipitate * c_liquid
    return R_s, R_l


def solve_transport_1d(c_initial, z_grid, v_interface, D_solid, D_liquid,
                       z_interface, dt, n_steps, interface_width=3.0):
    """
    Solve 1D advection-diffusion-reaction equation for species concentration.
    
    The diffusion coefficient varies across the interface:
        D(z) = D_solid + (D_liquid - D_solid) * phi(z)
    
    where phi(z) is the interface profile function.
    
    Args:
        c_initial: initial concentration profile
        z_grid: 1D spatial grid
        v_interface: interface velocity
        D_solid: solid diffusion coefficient
        D_liquid: liquid diffusion coefficient
        z_interface: interface position
        dt: time step
        n_steps: number of time steps
        interface_width: interface width parameter
        
    Returns:
        c_profile: final concentration profile
        c_history: concentration history (n_steps, n_z)
    """
    n_z = len(z_grid)
    dz = z_grid[1] - z_grid[0]
    
    c = c_initial.copy()
    c_history = np.zeros((n_steps, n_z))
    
    for step in range(n_steps):
        c_history[step] = c.copy()
        
        # Compute diffusion coefficient profile
        phi = 0.5 * (1.0 + np.tanh((z_grid - z_interface) / interface_width))
        D_profile = D_solid + (D_liquid - D_solid) * phi
        
        # Advection flux
        flux_adv = advection_flux(c, v_interface, dz)
        
        # Diffusion term (central differences with variable D)
        dcdz = np.zeros(n_z)
        for i in range(1, n_z - 1):
            dcdz[i] = (c[i + 1] - c[i - 1]) / (2.0 * dz)
        
        d2cdz2 = np.zeros(n_z)
        for i in range(1, n_z - 1):
            D_plus = 0.5 * (D_profile[i + 1] + D_profile[i])
            D_minus = 0.5 * (D_profile[i] + D_profile[i - 1])
            d2cdz2[i] = (D_plus * (c[i + 1] - c[i]) - D_minus * (c[i] - c[i - 1])) / (dz ** 2)
        
        # Reaction term (only near interface)
        R = np.zeros(n_z)
        for i in range(n_z):
            if abs(z_grid[i] - z_interface) < 2.0 * interface_width:
                c_s = c[i]
                c_l = c[i]  # simplified: same concentration
                R[i] = reaction_term(c_s, c_l)[1]
        
        # Time update (explicit Euler)
        # dc/dt = -d(flux)/dz + D * d2c/dz2 + R
        dc_dt = np.zeros(n_z)
        for i in range(n_z):
            if 0 < i < n_z - 1:
                dc_dt[i] = -(flux_adv[i + 1] - flux_adv[i]) / dz + d2cdz2[i] + R[i]
        
        # CFL condition check
        max_D = max(D_solid, D_liquid)
        cfl_diff = max_D * dt / (dz ** 2)
        cfl_adv = abs(v_interface) * dt / dz
        
        # Adaptive time step if CFL violated
        if cfl_diff > 0.5 or cfl_adv > 1.0:
            dt_safe = min(0.25 * dz ** 2 / max_D, 0.5 * dz / max(abs(v_interface), 1e-10))
            # Sub-stepping
            n_sub = int(np.ceil(dt / dt_safe))
            dt_sub = dt / n_sub
            
            for _ in range(n_sub):
                # Recompute at sub-step
                flux_adv_sub = advection_flux(c, v_interface, dz)
                d2cdz2_sub = np.zeros(n_z)
                for j in range(1, n_z - 1):
                    D_plus = 0.5 * (D_profile[j + 1] + D_profile[j])
                    D_minus = 0.5 * (D_profile[j] + D_profile[j - 1])
                    d2cdz2_sub[j] = (D_plus * (c[j + 1] - c[j]) - D_minus * (c[j] - c[j - 1])) / (dz ** 2)
                
                dc_dt_sub = np.zeros(n_z)
                for j in range(n_z):
                    if 0 < j < n_z - 1:
                        dc_dt_sub[j] = -(flux_adv_sub[j + 1] - flux_adv_sub[j]) / dz + d2cdz2_sub[j] + R[j]
                
                c += dt_sub * dc_dt_sub
                c = np.clip(c, 0.0, 1.0)
        else:
            c += dt * dc_dt
        
        # Boundary conditions
        c[0] = c[1]    # Neumann at left
        c[-1] = c[-2]  # Neumann at right
        
        # Physical constraints
        c = np.clip(c, 0.0, 1.0)
    
    return c, c_history


# =============================================================================
# Interface Velocity from Thermodynamic Driving Force
# =============================================================================

def interface_velocity(chemical_potential_solid, chemical_potential_liquid,
                       mobility=INTERFACE_MOBILITY):
    """
    Compute interface velocity from chemical potential difference.
    
    v = M * (mu_liquid - mu_solid)
    
    where M is the interface mobility.
    
    For an ideal solution:
        mu = mu^0 + k_B * T * ln(c)
    
    Args:
        chemical_potential_solid: solid chemical potential
        chemical_potential_liquid: liquid chemical potential
        mobility: interface mobility
        
    Returns:
        velocity
    """
    delta_mu = chemical_potential_liquid - chemical_potential_solid
    v = mobility * delta_mu
    
    # Boundary handling
    v_max = 1.0  # maximum velocity in m/s
    v = np.clip(v, -v_max, v_max)
    
    return v


# =============================================================================
# Thermal Transport (Heat Equation)
# =============================================================================

def solve_heat_equation_1d(T_initial, z_grid, alpha_thermal, dt, n_steps,
                           T_left=None, T_right=None):
    """
    Solve 1D heat equation using implicit trapezoidal method.
    
        dT/dt = alpha * d^2T/dz^2
    
    Discretized using Crank-Nicolson (trapezoidal in time, central in space):
        (T^{n+1}_i - T^n_i) / dt = alpha/2 * [ (T^{n+1}_{i+1} - 2*T^{n+1}_i + T^{n+1}_{i-1})/dz^2
                                               + (T^n_{i+1} - 2*T^n_i + T^n_{i-1})/dz^2 ]
    
    This leads to a tridiagonal system that can be solved efficiently.
    
    Args:
        T_initial: initial temperature profile
        z_grid: spatial grid
        alpha_thermal: thermal diffusivity
        dt: time step
        n_steps: number of steps
        T_left, T_right: boundary temperatures (None = insulating)
        
    Returns:
        T_profile: final temperature
        T_history: temperature history
    """
    n_z = len(z_grid)
    dz = z_grid[1] - z_grid[0]
    
    T = T_initial.copy()
    T_history = np.zeros((n_steps, n_z))
    
    # Coefficients for tridiagonal system
    r = alpha_thermal * dt / (2.0 * dz ** 2)
    
    # Build tridiagonal matrix
    main_diag = np.ones(n_z) * (1.0 + 2.0 * r)
    off_diag = np.ones(n_z - 1) * (-r)
    
    # Boundary conditions
    if T_left is not None:
        main_diag[0] = 1.0
        off_diag[0] = 0.0
    else:
        main_diag[0] = 1.0 + r
    
    if T_right is not None:
        main_diag[-1] = 1.0
    else:
        main_diag[-1] = 1.0 + r
    
    # Thomas algorithm for tridiagonal solve
    for step in range(n_steps):
        T_history[step] = T.copy()
        
        # Right-hand side
        rhs = np.zeros(n_z)
        for i in range(1, n_z - 1):
            rhs[i] = r * T[i - 1] + (1.0 - 2.0 * r) * T[i] + r * T[i + 1]
        
        if T_left is not None:
            rhs[0] = T_left
        else:
            rhs[0] = T[0] + r * (T[1] - T[0])
        
        if T_right is not None:
            rhs[-1] = T_right
        else:
            rhs[-1] = T[-1] + r * (T[-2] - T[-1])
        
        # Solve tridiagonal system using Thomas algorithm
        T = thomas_algorithm(main_diag, off_diag, off_diag, rhs)
    
    return T, T_history


def thomas_algorithm(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.
    
    System: a_i * x_i + b_i * x_{i+1} + c_{i-1} * x_{i-1} = d_i
    
    Args:
        a: main diagonal (n,)
        b: upper diagonal (n-1,)
        c: lower diagonal (n-1,)
        d: right-hand side (n,)
        
    Returns:
        x: solution
    """
    n = len(a)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    # Forward sweep
    cp[0] = b[0] / a[0]
    dp[0] = d[0] / a[0]
    
    for i in range(1, n - 1):
        denom = a[i] - c[i - 1] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = b[i] / denom
        dp[i] = (d[i] - c[i - 1] * dp[i - 1]) / denom
    
    denom = a[n - 1] - c[n - 2] * cp[n - 2]
    if abs(denom) < 1e-14:
        denom = 1e-14
    dp[n - 1] = (d[n - 1] - c[n - 2] * dp[n - 2]) / denom
    
    # Back substitution
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x
