"""
reactor_fem1d.py
================
One-dimensional finite element solver for pulverized coal combustion reactor.

Incorporates fem1d_function_10_display (389): 1D FEM with piecewise linear
(P1) Lagrange basis functions.

Scientific problem: solve the 1D steady convection-diffusion-reaction
system for temperature T(x) and species mass fractions Y_i(x) in a
pulverized coal burner.

Governing equations:
    Continuity:     d(rho*u)/dx = 0  => rho*u = constant
    
    Energy:         rho*u*c_p * dT/dx = d/dx(k * dT/dx) + Q_reaction
    
    Species i:      rho*u * dY_i/dx = d/dx(rho*D_i * dY_i/dx) + omega_i/rho

Weak form (Galerkin FEM with P1 elements):
    For test function v in H^1_0(Omega):
        integral( rho*u*c_p * dT/dx * v ) dx =
            -integral( k * dT/dx * dv/dx ) dx + integral( Q * v ) dx

Discretized on n nodes {x_j}, j=0,...,n-1 with mesh size h_j = x_{j+1} - x_j.
P1 basis functions:
    phi_j(x) = (x - x_{j-1})/h_{j-1}   for x in [x_{j-1}, x_j]
             = (x_{j+1} - x)/h_j       for x in [x_j, x_{j+1}]
             = 0                       otherwise

Element matrices for convection-diffusion (upwind stabilization):
    Peclet number:  Pe = rho*u*c_p*h / (2*k)
    If Pe > 1, add artificial diffusion: k_art = 0.5*rho*u*c_p*h * (coth(Pe) - 1/Pe)
"""

import numpy as np
from utils import tridiagonal_solve
from thermophysical_props import (
    thermal_conductivity, mixture_density, cp_mixture,
    mass_diffusivity_NO, dynamic_viscosity
)


# ======================================================================
# 1. Mesh generation
# ======================================================================

def generate_nonuniform_mesh(
    x_min: float, x_max: float, n_nodes: int, clustering: str = "center"
) -> np.ndarray:
    """
    Generate a non-uniform 1D mesh with Chebyshev-like clustering.
    For 'center': cluster nodes around the center (high flame-temperature zone).
    """
    if n_nodes < 2:
        return np.array([x_min, x_max])
    
    if clustering == "uniform":
        return np.linspace(x_min, x_max, n_nodes)
    
    # Chebyshev nodes mapped to [x_min, x_max]
    # theta_i = (n-1-i)*pi/(n-1), i=0,...,n-1
    # x_i = (1-cos(theta_i))/2 maps [0,pi] to [0,1]
    i = np.arange(n_nodes)
    theta = (n_nodes - 1 - i) * np.pi / (n_nodes - 1)
    s = 0.5 * (1.0 - np.cos(theta))
    
    if clustering == "center":
        # Cluster around center: map s->s' with s' = 0.5 + 0.5*sin(pi*(s-0.5))
        s = 0.5 + 0.5 * np.sin(np.pi * (s - 0.5))
    elif clustering == "left":
        # Cluster near inlet
        s = s ** 2
    elif clustering == "right":
        # Cluster near outlet
        s = 1.0 - (1.0 - s) ** 2
    
    x = x_min + s * (x_max - x_min)
    x[0] = x_min
    x[-1] = x_max
    return x


# ======================================================================
# 2. Element matrix assembly (convection-diffusion-reaction)
# ======================================================================

def assemble_fem_system(
    x: np.ndarray, a_func, k_func, s_func, bc_left: tuple, bc_right: tuple
) -> tuple:
    """
    Assemble FEM system for:
        -d/dx(k(x) * du/dx) + a(x) * du/dx + s(x) * u = f(x)
    with boundary conditions:
        bc = ('dirichlet', value) or ('neumann', flux)
    
    Uses streamline-upwind Petrov-Galerkin (SUPG) stabilization for
    high Peclet numbers.
    
    Returns:
        A: (n,n) sparse system matrix (returned as dense for simplicity)
        F: (n,) right-hand side
    """
    n = len(x)
    A = np.zeros((n, n))
    F = np.zeros(n)
    
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        if h <= 1e-30:
            continue
        xc = 0.5 * (x[e] + x[e + 1])
        
        a_val = a_func(xc)
        k_val = k_func(xc)
        s_val = s_func(xc)
        f_val = s_func(xc)  # source term (reusing signature for simplicity)
        
        # Peclet number
        Pe = abs(a_val) * h / (2.0 * max(k_val, 1e-30))
        
        # SUPG stabilization parameter
        tau = 0.0
        if Pe > 1e-6:
            tau = h / (2.0 * abs(a_val)) * (1.0 / np.tanh(Pe) - 1.0 / Pe)
        
        # Element matrices (linear basis)
        # Diffusion: k/h * [1, -1; -1, 1]
        K_diff = (k_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        # Convection: a/2 * [-1, 1; -1, 1]
        K_conv = 0.5 * a_val * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        
        # Reaction: s*h/6 * [2, 1; 1, 2]
        K_reac = (s_val * h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        
        # SUPG stabilization: tau*a^2/h * [1, -1; -1, 1]
        K_supg = (tau * a_val * a_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        # Source: f*h/2 * [1, 1]
        F_elem = 0.5 * f_val * h * np.array([1.0, 1.0])
        
        # SUPG source correction: tau*a * f/2 * [-1, 1]
        F_supg = 0.5 * tau * a_val * f_val * np.array([-1.0, 1.0])
        
        # Assembly
        for i in range(2):
            gi = e + i
            F[gi] += F_elem[i] + F_supg[i]
            for j in range(2):
                gj = e + j
                A[gi, gj] += K_diff[i, j] + K_conv[i, j] + K_reac[i, j] + K_supg[i, j]
    
    # Apply boundary conditions
    # Left BC
    bctype, bcval = bc_left
    if bctype == "dirichlet":
        A[0, :] = 0.0
        A[0, 0] = 1.0
        F[0] = bcval
    elif bctype == "neumann":
        F[0] += bcval
    elif bctype == "robin":
        alpha, beta = bcval
        A[0, 0] += alpha
        F[0] += beta
    
    # Right BC
    bctype, bcval = bc_right
    if bctype == "dirichlet":
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        F[-1] = bcval
    elif bctype == "neumann":
        F[-1] += bcval
    elif bctype == "robin":
        alpha, beta = bcval
        A[-1, -1] += alpha
        F[-1] += beta
    
    return A, F


# ======================================================================
# 3. Temperature field solver
# ======================================================================

def solve_temperature_field(
    x: np.ndarray, u_inlet: float, T_inlet: float, T_wall: float,
    Q_heat_release: np.ndarray, Y_mix: dict, P: float = 101325.0
) -> np.ndarray:
    """
    Solve 1D energy equation for temperature profile T(x).
    
    rho*u*c_p * dT/dx = d/dx(k * dT/dx) + Q(x)
    
    with:
        T(0) = T_inlet (Dirichlet)
        -k * dT/dx(L) = h_conv * (T(L) - T_wall) (Robin)
    """
    n = len(x)
    rho = mixture_density(T_inlet, P)
    cp = cp_mixture(T_inlet, Y_mix)
    k_therm = thermal_conductivity(T_inlet)
    
    # Convection coefficient for Robin BC (approximate)
    h_conv = 50.0  # W/(m^2·K)
    
    def a_func(xc):
        return rho * u_inlet * cp
    
    def k_func(xc):
        # Average temperature estimate for property evaluation
        return k_therm
    
    def s_func(xc):
        return 0.0  # No linear reaction term in energy eq
    
    def f_func(xc):
        # Find nearest index for Q
        idx = np.argmin(np.abs(x - xc))
        return Q_heat_release[idx]
    
    # Override: assemble manually for source term
    A = np.zeros((n, n))
    F = np.zeros(n)
    
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        if h <= 1e-30:
            continue
        xc = 0.5 * (x[e] + x[e + 1])
        
        a_val = rho * u_inlet * cp
        k_val = k_func(xc)
        
        Pe = abs(a_val) * h / (2.0 * max(k_val, 1e-30))
        tau = 0.0
        if Pe > 1e-6:
            tau = h / (2.0 * abs(a_val)) * (1.0 / np.tanh(Pe) - 1.0 / Pe)
        
        K_diff = (k_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K_conv = 0.5 * a_val * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        K_supg = (tau * a_val * a_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        f_val = f_func(xc)
        F_elem = 0.5 * f_val * h * np.array([1.0, 1.0])
        F_supg = 0.5 * tau * a_val * f_val * np.array([-1.0, 1.0])
        
        for i in range(2):
            gi = e + i
            F[gi] += F_elem[i] + F_supg[i]
            for j in range(2):
                gj = e + j
                A[gi, gj] += K_diff[i, j] + K_conv[i, j] + K_supg[i, j]
    
    # Left: Dirichlet T = T_inlet
    A[0, :] = 0.0
    A[0, 0] = 1.0
    F[0] = T_inlet
    
    # Right: Robin -k*dT/dx = h_conv*(T - T_wall)
    # => k/h * (T_n - T_{n-1}) + h_conv*(T_n - T_wall) = 0
    # Add to last equation
    h_last = x[-1] - x[-2]
    if h_last > 0:
        A[-1, -1] += h_conv
        F[-1] += h_conv * T_wall
    
    # Solve
    try:
        T = np.linalg.solve(A, F)
    except np.linalg.LinAlgError:
        T = np.linalg.lstsq(A, F, rcond=None)[0]
    
    return np.clip(T, 200.0, 5000.0)


# ======================================================================
# 4. NO mass fraction field solver
# ======================================================================

def solve_species_field(
    x: np.ndarray, u_inlet: float, Y_inlet: float,
    source_terms: np.ndarray, D_species: float, P: float = 101325.0,
    T_field: np.ndarray = None
) -> np.ndarray:
    """
    Solve 1D convection-diffusion equation for species mass fraction:
        rho*u * dY/dx = d/dx(rho*D * dY/dx) + S(x)
    
    with:
        Y(0) = Y_inlet
        dY/dx(L) = 0  (fully developed)
    """
    n = len(x)
    rho = mixture_density(1500.0, P) if T_field is None else mixture_density(np.mean(T_field), P)
    
    A = np.zeros((n, n))
    F = np.zeros(n)
    
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        if h <= 1e-30:
            continue
        xc = 0.5 * (x[e] + x[e + 1])
        
        a_val = rho * u_inlet
        k_val = rho * D_species
        
        Pe = abs(a_val) * h / (2.0 * max(k_val, 1e-30))
        tau = 0.0
        if Pe > 1e-6:
            tau = h / (2.0 * abs(a_val)) * (1.0 / np.tanh(Pe) - 1.0 / Pe)
        
        K_diff = (k_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K_conv = 0.5 * a_val * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        K_supg = (tau * a_val * a_val / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        idx = np.argmin(np.abs(x - xc))
        f_val = source_terms[idx]
        F_elem = 0.5 * f_val * h * np.array([1.0, 1.0])
        F_supg = 0.5 * tau * a_val * f_val * np.array([-1.0, 1.0])
        
        for i in range(2):
            gi = e + i
            F[gi] += F_elem[i] + F_supg[i]
            for j in range(2):
                gj = e + j
                A[gi, gj] += K_diff[i, j] + K_conv[i, j] + K_supg[i, j]
    
    # Left: Dirichlet
    A[0, :] = 0.0
    A[0, 0] = 1.0
    F[0] = Y_inlet
    
    # Right: weak outflow Robin condition to stabilize high-Pe convection
    A[-1, -1] += rho * u_inlet
    
    try:
        Y = np.linalg.solve(A, F)
    except np.linalg.LinAlgError:
        Y = np.linalg.lstsq(A, F, rcond=None)[0]
    
    return np.clip(Y, 0.0, 1.0)


# ======================================================================
# 5. Full reactor simulation coupling
# ======================================================================

def simulate_1d_burner(
    L: float = 5.0, n_nodes: int = 101,
    u_inlet: float = 5.0, T_inlet: float = 400.0,
    T_wall: float = 800.0, P: float = 101325.0
) -> dict:
    """
    Simulate a 1D pulverized coal burner.
    Returns temperature and NO mass fraction profiles.
    """
    x = generate_nonuniform_mesh(0.0, L, n_nodes, clustering="uniform")
    
    # Heat release profile: Gaussian around x = L/3 (flame stabilization)
    x_peak = L / 3.0
    sigma = L / 10.0
    Q_max = 2.5e6  # W/m^3 (realistic for pulverized coal)
    Q_comb = Q_max * np.exp(-((x - x_peak) ** 2) / (2.0 * sigma * sigma))
    
    Y_mix = {"N2": 0.79, "O2": 0.21, "CO2": 0.0, "H2O": 0.0}
    
    # Iterative solve with radiation loss (only active above 1200 K)
    T = np.full_like(x, T_inlet)
    sigma_sb = 5.670374419e-8
    eps_rad = 0.15
    for _ in range(10):
        T_active = np.where(T > 1200.0, T, 1200.0)
        Q_rad = 4.0 * sigma_sb * eps_rad * (T_active ** 4 - T_wall ** 4)
        Q = Q_comb - Q_rad
        T_new = solve_temperature_field(x, u_inlet, T_inlet, T_wall, Q, Y_mix, P)
        if np.max(np.abs(T_new - T)) < 1.0:
            break
        T = T_new
    
    # NO source from thermal NOx (simplified Zeldovich)
    # S_NO ~ rho * A * exp(-Ea/(R*T)) * [N2]*[O2]^0.5
    R_gas = 8.314462618
    Ea_zeldovich = 319.0e3
    A_zeldovich = 1.8e11
    MW_N2 = 28.0134e-3
    MW_O2 = 31.9988e-3
    rho = mixture_density(np.mean(T), P)
    
    S_NO = np.zeros(n_nodes)
    for i in range(n_nodes):
        Ti = max(T[i], 300.0)
        rho_i = mixture_density(Ti, P)
        X_N2 = 0.79
        X_O2 = 0.21 * np.exp(-x[i] / (L * 0.3))  # O2 depletion
        # Simplified thermal NOx source [kg_NO/(m^3*s)]
        # Correlation: dY_NO/dt ~ A * exp(-Ea/RT) * Y_N2 * Y_O2^0.5
        A_corr = 3.0e5
        Ea_corr = 271.0e3
        k_corr = A_corr * np.exp(-Ea_corr / (R_gas * Ti))
        S_NO[i] = rho_i * k_corr * X_N2 * (X_O2 ** 0.5)
    
    D_NO = mass_diffusivity_NO(np.mean(T), P)
    Y_NO = solve_species_field(x, u_inlet, 0.0, S_NO, D_NO, P, T)
    
    return {
        "x": x,
        "T": T,
        "Y_NO": Y_NO,
        "Q_heat": Q,
        "S_NO": S_NO,
        "max_T": np.max(T),
        "max_NO_ppm": np.max(Y_NO) * 1e6,
        "outlet_NO_ppm": Y_NO[-1] * 1e6,
    }
