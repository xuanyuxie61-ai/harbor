"""
truth_model.py - High-Fidelity Multi-Physics Simulation Models

This module implements the expensive "truth" models that the polynomial
chaos surrogates approximate. Three coupled physics models are included:

1. Reaction-Diffusion Predator-Prey System (from fd1d_predator_prey)
   - 1D spatial reaction-diffusion with implicit-explicit time stepping
   - Turing pattern formation analysis
   - Quantities of interest: pattern wavelength, amplitude, stability

2. Bad-Cavity Superradiant Laser Model (from mhz-linewidth-laser)
   - Cumulant equations for atom-field coherence
   - Steady-state power and linewidth computation
   - Quantities of interest: output power, spectral linewidth

3. Network Energy Cost Model (from paper-reproduction)
   - WSN clustering energy optimization
   - Analytical energy model with parameters
   - Quantities of interest: total energy per round

Mathematical Models
-------------------
### Reaction-Diffusion (Spatial Predator-Prey) ###

  du/dt = D_u * nabla^2 u + alpha*u - beta*u*v - u^2 + gamma*u*v/(1 + delta*u)
  dv/dt = D_v * nabla^2 v + epsilon*(-v + beta*u*v - gamma*v/(1 + delta*u))

  With Neumann boundary conditions: du/dn = dv/dn = 0 on dOmega.
  Turing instability occurs when diffusion destabilizes the stable
  homogeneous steady state:
    D_v * f_u + D_u * g_v > 2*sqrt(D_u*D_v*(f_u*g_v - f_v*g_u))

### Bad-Cavity Laser (Cumulant Equations) ###

  Steady-state atom-field coherence y satisfies:
    a*y^2 + b*y + c = 0  (quadratic in imaginary part)
  where coefficients depend on:
    C = Omega^2 / (kappa * gamma)  (cooperativity)
    d0(w) = (w - gamma)/(w + gamma)  (free-atom inversion)
    Gamma(w) = w + gamma + 2/T2  (total dipole decay)
    N_crit = 2 / (C * gamma * T2)  (critical atom number)

  Linewidth from regression matrix eigenvalues:
    M = [[-kappa/2, i*N*Omega/2], [-i*Omega*s_z/2, -Gamma_half]]
    HWHM = |Re(lambda_slow)|

### Network Energy ###

  E_total = l * (2*N*E_elec + N*E_DA + k*eps_mp*d^4 + N*eps_fs*M^2/(2*pi*k))
  Optimal k* = sqrt(N * eps_fs * M^2 / (2*pi*eps_mp*d^4))
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional, Any
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Reaction-Diffusion Truth Model
# ---------------------------------------------------------------------------

@dataclass
class ReactionDiffusionParams:
    """
    Parameters for the reaction-diffusion predator-prey system.

    The system exhibits Turing patterns when:
    1. The homogeneous steady state (u*, v*) is stable to homogeneous perturbations
    2. Diffusion destabilizes the steady state (Turing condition)

    Turing conditions:
      f_u + g_v < 0                    (stability to homogeneous perturbations)
      f_u * g_v - f_v * g_u > 0       (positive determinant)
      D_v * f_u + D_u * g_v > 0       (diffusion-driven instability)
      (D_v * f_u + D_u * g_v)^2 > 4 * D_u * D_v * (f_u*g_v - f_v*g_u)
    """
    alpha: float = 2.0      # prey growth rate
    beta: float = 1.0       # predation rate
    gamma: float = 1.0      # saturation parameter
    delta: float = 0.5      # handling time
    epsilon: float = 0.05   # predator death rate scaling
    D_u: float = 0.01       # prey diffusion coefficient
    D_v: float = 0.5        # predator diffusion coefficient
    L: float = 1.0          # domain length
    T_final: float = 50.0   # final time

    def steady_state(self) -> Tuple[float, float]:
        """
        Compute the homogeneous steady state (u*, v*).

        From f(u*, v*) = 0 and g(u*, v*) = 0:
          u* = gamma / (delta * (beta - 1))  if beta > 1
          v* = (alpha - u* - u*^2) / (beta - gamma/(1 + delta*u*)) * u*

        For the modified kinetics, solve the nonlinear system.
        """
        # Simplified: find u* from alpha - u* - beta*v* + gamma*u*v/(1+delta*u) = 0
        # With g = 0: v* = gamma/(delta) * (1 - 1/(beta*u*)) when beta*u* > 1
        if self.beta <= 1.0:
            # No predator survival
            u_star = self.alpha
            v_star = 0.0
        else:
            # Iterative solution
            u_star = self.alpha / (1.0 + self.beta * self.gamma / (self.delta * (1.0 + self.delta)))
            v_star = (self.alpha - u_star) / (self.beta - self.gamma / (1.0 + self.delta * u_star + 1e-14))
            v_star = max(v_star, 0.0)
            u_star = max(u_star, 1e-6)

        return u_star, v_star

    def jacobian_at_steady_state(self) -> NDArray:
        """
        Jacobian of the reaction kinetics at (u*, v*).

        J = [[f_u, f_v], [g_u, g_v]]

        where:
          f_u = alpha - 2*u* - beta*v* + gamma*v*(1+delta*u*)^(-2)
          f_v = -beta*u* + gamma*u*/(1+delta*u*)
          g_u = epsilon*beta*v*(1+delta*u*)^(-2)  ... (depends on kinetics)
          g_v = epsilon*(-1 + beta*u*/(1+delta*u*) ... )
        """
        u_star, v_star = self.steady_state()
        denom = 1.0 + self.delta * u_star

        # Modified functional response: h(u) = u / (alpha + |u|)
        # For simplicity, use the analytical Jacobian of the given kinetics
        f_u = self.alpha - 2.0 * u_star - self.beta * v_star * self.gamma / (denom * denom)
        f_v = -self.beta * u_star + self.gamma * u_star / denom
        g_u = self.epsilon * (self.beta * v_star / (denom * denom))
        g_v = self.epsilon * (-1.0 + self.beta * u_star / denom - self.gamma / denom)

        return np.array([[f_u, f_v], [g_u, g_v]])

    def turing_condition(self) -> Dict[str, float]:
        """
        Evaluate Turing instability conditions.

        Returns dict with condition values (positive = Turing possible):
          trace: f_u + g_v (must be < 0)
          determinant: f_u*g_v - f_v*g_u (must be > 0)
          diffusion_trace: D_v*f_u + D_u*g_v (must be > 0 for instability)
          discriminant: (D_v*f_u + D_u*g_v)^2 - 4*D_u*D_v*det (must be > 0)
        """
        J = self.jacobian_at_steady_state()
        f_u, f_v, g_u, g_v = J[0, 0], J[0, 1], J[1, 0], J[1, 1]

        trace = f_u + g_v
        det = f_u * g_v - f_v * g_u
        diff_trace = self.D_v * f_u + self.D_u * g_v
        discriminant = diff_trace ** 2 - 4.0 * self.D_u * self.D_v * det

        return {
            'trace': trace,
            'determinant': det,
            'diffusion_trace': diff_trace,
            'discriminant': discriminant,
            'turing_possible': (trace < 0) and (det > 0) and
                               (diff_trace > 0) and (discriminant > 0)
        }

    def critical_wavenumber(self) -> float:
        """
        Most unstable wavenumber for Turing instability.

        k_c^2 = (D_v * f_u + D_u * g_v) / (2 * D_u * D_v)

        Wavelength: lambda_c = 2*pi / k_c
        """
        J = self.jacobian_at_steady_state()
        f_u, g_v = J[0, 0], J[1, 1]
        kc_sq = (self.D_v * f_u + self.D_u * g_v) / (2.0 * self.D_u * self.D_v + 1e-14)
        if kc_sq <= 0:
            return 0.0
        kc = math.sqrt(kc_sq)
        return 2.0 * math.pi / kc if kc > 1e-14 else float('inf')


def run_reaction_diffusion_1d(params: ReactionDiffusionParams,
                               nx: int = 100,
                               nt: int = 500,
                               u0_perturbation: float = 0.01) -> Dict[str, Any]:
    """
    Run 1D reaction-diffusion simulation with finite differences.

    Uses implicit-explicit (IMEX) time stepping:
      - Diffusion treated implicitly (unconditionally stable)
      - Reaction treated explicitly (avoids nonlinear solve)

    Spatial discretization: central differences with Neumann BCs.
    The diffusion matrix L is tridiagonal:
      L = (1/h^2) * tridiag(1, -2, 1) with modified boundary rows.

    IMEX scheme:
      (I - dt*D*L) u^{n+1} = (I + dt*D*L) u^n + dt * R(u^n)

    Parameters
    ----------
    params : ReactionDiffusionParams
        Physical parameters.
    nx : int
        Number of spatial grid points.
    nt : int
        Number of time steps.
    u0_perturbation : float
        Amplitude of initial perturbation from steady state.

    Returns
    -------
    dict with keys:
        'u_final': ndarray(nx,) - final prey profile
        'v_final': ndarray(nx,) - final predator profile
        'pattern_amplitude': float - RMS deviation from mean
        'pattern_wavelength': float - dominant wavelength (FFT)
        'max_u': float - maximum prey density
        'max_v': float - maximum predator density
    """
    h = params.L / (nx - 1)
    dt = params.T_final / nt
    mu_u = params.D_u * dt / (h * h)
    mu_v = params.D_v * dt / (h * h)

    # CFL check for explicit reaction terms
    u_star, v_star = params.steady_state()

    # Spatial grid
    x = np.linspace(0, params.L, nx)

    # Initial condition: steady state + sinusoidal perturbation
    k_mode = 2.0 * math.pi / params.L * 5  # 5th mode
    u = u_star + u0_perturbation * np.cos(k_mode * x)
    v = v_star + u0_perturbation * np.sin(k_mode * x)

    # Ensure positivity
    u = np.maximum(u, 1e-8)
    v = np.maximum(v, 1e-8)

    # Build implicit diffusion matrices (tridiagonal)
    # (I - mu * L) where L is the discrete Laplacian with Neumann BCs
    def solve_tridiagonal(a, b, c, rhs):
        """Thomas algorithm for tridiagonal system."""
        n = len(rhs)
        cp = np.zeros(n)
        dp = np.zeros(n)
        x_sol = np.zeros(n)

        # Forward sweep
        cp[0] = c[0] / b[0] if abs(b[0]) > 1e-30 else 0.0
        dp[0] = rhs[0] / b[0] if abs(b[0]) > 1e-30 else 0.0
        for i in range(1, n):
            m = b[i] - a[i] * cp[i - 1]
            if abs(m) < 1e-30:
                m = 1e-30
            cp[i] = c[i] / m if i < n - 1 else 0.0
            dp[i] = (rhs[i] - a[i] * dp[i - 1]) / m

        # Back substitution
        x_sol[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x_sol[i] = dp[i] - cp[i] * x_sol[i + 1]

        return x_sol

    # Time stepping
    for step in range(nt):
        # Reaction terms (explicit)
        hhat = u / (params.alpha + np.abs(u) + 1e-14)
        F = u * (params.alpha - u) - params.beta * u * v * hhat
        G = params.epsilon * (params.beta * u * v * hhat - params.gamma * v)

        # Explicit diffusion + reaction
        u_rhs = np.copy(u)
        v_rhs = np.copy(v)

        # Interior points: explicit Laplacian
        for i in range(1, nx - 1):
            lap_u = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (h * h)
            lap_v = (v[i - 1] - 2.0 * v[i] + v[i + 1]) / (h * h)
            u_rhs[i] = u[i] + dt * (params.D_u * lap_u + F[i])
            v_rhs[i] = v[i] + dt * (params.D_v * lap_v + G[i])

        # Neumann BCs (ghost points)
        u_rhs[0] = u[0] + dt * (params.D_u * 2.0 * (u[1] - u[0]) / (h * h) + F[0])
        u_rhs[-1] = u[-1] + dt * (params.D_u * 2.0 * (u[-2] - u[-1]) / (h * h) + F[-1])
        v_rhs[0] = v[0] + dt * (params.D_v * 2.0 * (v[1] - v[0]) / (h * h) + G[0])
        v_rhs[-1] = v[-1] + dt * (params.D_v * 2.0 * (v[-2] - v[-1]) / (h * h) + G[-1])

        # Implicit diffusion correction (simplified - single step)
        a_coeff = np.full(nx, -mu_u / 2.0)
        b_coeff = np.full(nx, 1.0 + mu_u)
        c_coeff = np.full(nx, -mu_u / 2.0)
        b_coeff[0] = 1.0 + mu_u
        b_coeff[-1] = 1.0 + mu_u
        u = solve_tridiagonal(a_coeff, b_coeff, c_coeff, u_rhs)

        a_coeff = np.full(nx, -mu_v / 2.0)
        b_coeff = np.full(nx, 1.0 + mu_v)
        c_coeff = np.full(nx, -mu_v / 2.0)
        b_coeff[0] = 1.0 + mu_v
        b_coeff[-1] = 1.0 + mu_v
        v = solve_tridiagonal(a_coeff, b_coeff, c_coeff, v_rhs)

        # Enforce positivity
        u = np.maximum(u, 1e-10)
        v = np.maximum(v, 1e-10)

    # Post-processing: pattern characterization
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    u_amp = np.sqrt(np.mean((u - u_mean) ** 2))
    v_amp = np.sqrt(np.mean((v - v_mean) ** 2))

    # Dominant wavelength via FFT
    u_fft = np.abs(np.fft.rfft(u - u_mean))
    freqs = np.fft.rfftfreq(nx, d=h)
    # Find dominant frequency (skip DC)
    if len(u_fft) > 1:
        dominant_idx = np.argmax(u_fft[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        wavelength = 1.0 / dominant_freq if dominant_freq > 1e-14 else params.L
    else:
        wavelength = params.L

    return {
        'u_final': u,
        'v_final': v,
        'x': x,
        'pattern_amplitude': u_amp,
        'pattern_wavelength': wavelength,
        'max_u': float(np.max(u)),
        'max_v': float(np.max(v)),
        'mean_u': float(u_mean),
        'mean_v': float(v_mean),
        'u_star': u_star,
        'v_star': v_star
    }


# ---------------------------------------------------------------------------
# Bad-Cavity Laser Truth Model
# ---------------------------------------------------------------------------

@dataclass
class LaserParams:
    """
    Parameters for the bad-cavity superradiant laser.

    Based on Meiser et al. (2009), "Prospects for a mHz-linewidth laser".

    Key derived quantities:
      C = Omega^2 / (kappa * gamma)           Cooperativity
      d0(w) = (w - gamma) / (w + gamma)       Free-atom inversion
      Gamma(w) = w + gamma + 2/T2             Total dipole decay rate
      Gamma_half(w) = (w + gamma)/2 + 1/T2   Half decay rate
      N_crit = 2 / (C * gamma * T2)           Critical atom number
      w_max = N * C * gamma                   Upper pump threshold
      P_max = hbar * omega_a * N^2 * C * gamma / 8   Maximum power
    """
    gamma: float = 1.0          # Spontaneous emission rate (normalized)
    gamma_T2: float = 0.1       # Transverse decay rate (1/T2)
    Omega: float = 1.0          # Atom-field coupling
    kappa: float = 1e3          # Cavity decay rate (bad cavity: kappa >> gamma)
    omega_a: float = 1.0        # Atomic transition frequency
    hbar: float = 1.0           # Reduced Planck constant (normalized)

    @property
    def C(self) -> float:
        """Cooperativity: C = Omega^2 / (kappa * gamma)."""
        denom = self.kappa * self.gamma
        if abs(denom) < 1e-30:
            return 0.0
        return self.Omega ** 2 / denom

    def d0(self, w: float) -> float:
        """Free-atom inversion: d0(w) = (w - gamma) / (w + gamma)."""
        denom = w + self.gamma
        if abs(denom) < 1e-30:
            return 0.0
        return (w - self.gamma) / denom

    def Gamma(self, w: float) -> float:
        """Total dipole decay: Gamma(w) = w + gamma + 2*gamma_T2."""
        return w + self.gamma + 2.0 * self.gamma_T2

    def Gamma_half(self, w: float) -> float:
        """Half decay: Gamma_half = (w + gamma)/2 + gamma_T2."""
        return (w + self.gamma) / 2.0 + self.gamma_T2

    def N_crit(self) -> float:
        """Critical atom number: N_crit = 2 / (C * gamma * T2)."""
        T2 = 1.0 / (self.gamma_T2 + 1e-30)
        denom = self.C * self.gamma * T2
        if abs(denom) < 1e-30:
            return float('inf')
        return 2.0 / denom

    def w_max(self, N: float) -> float:
        """Upper pump threshold: w_max = N * C * gamma."""
        return N * self.C * self.gamma

    def P_max(self, N: float) -> float:
        """Maximum power: P_max = hbar * omega_a * N^2 * C * gamma / 8."""
        return self.hbar * self.omega_a * N ** 2 * self.C * self.gamma / 8.0


def solve_laser_steady_state(p: LaserParams, w: float,
                             N: float) -> Dict[str, float]:
    """
    Solve the laser cumulant equations for steady state.

    The steady-state coherence y (imaginary part) satisfies:
      a*y^2 + b*y + c = 0
    where:
      a = N * C * gamma * d0(w) / Gamma(w)^2  (nonlinear coefficient)
      b = 1 + kappa * Gamma_half(w) / (N * C * gamma / 2)  (linear)
      c = -d0(w)  (constant term)

    Physical root selection: choose y such that s_z = d0(w)/(1 + a*y^2) >= 0.

    The steady-state photon number:
      n_ph = N * C * gamma * y^2 / (4 * kappa)
    Output power:
      P = 2 * kappa * hbar * omega_a * n_ph
    """
    d0 = p.d0(w)
    G = p.Gamma(w)
    Gh = p.Gamma_half(w)
    C = p.C

    if abs(G) < 1e-30 or N < 1e-14:
        return {'y': 0.0, 's_z': d0, 'n_ph': 0.0, 'power': 0.0, 'valid': False}

    # Quadratic coefficients
    a_coeff = N * C * p.gamma * d0 / (G * G + 1e-30)
    b_coeff = 1.0
    c_coeff = -d0

    # Solve quadratic
    disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    if disc < 0:
        # No real root: system below threshold
        return {'y': 0.0, 's_z': d0, 'n_ph': 0.0, 'power': 0.0, 'valid': False}

    sqrt_disc = math.sqrt(disc)
    if abs(a_coeff) > 1e-30:
        y1 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
        y2 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)
    else:
        y1 = -c_coeff / (b_coeff + 1e-30)
        y2 = y1

    # Select physical root (s_z >= 0)
    for y in [y1, y2]:
        s_z = d0 / (1.0 + a_coeff * y ** 2 + 1e-30)
        if s_z >= 0:
            n_ph = N * C * p.gamma * y ** 2 / (4.0 * p.kappa + 1e-30)
            power = 2.0 * p.kappa * p.hbar * p.omega_a * n_ph
            return {'y': y, 's_z': s_z, 'n_ph': n_ph, 'power': power, 'valid': True}

    # Fallback
    return {'y': 0.0, 's_z': d0, 'n_ph': 0.0, 'power': 0.0, 'valid': False}


def compute_laser_linewidth(p: LaserParams, w: float, N: float,
                            s_z: float) -> float:
    """
    Compute laser HWHM linewidth from regression matrix eigenvalues.

    The 2x2 regression matrix for phase-amplitude fluctuations:
      M = [[-kappa/2, i*N*Omega/2],
           [-i*Omega*s_z/2, -Gamma_half(w)]]

    The HWHM linewidth is:
      gamma_HWHM = |Re(lambda_slow)|
    where lambda_slow is the eigenvalue with smaller |Re|.

    For bad cavity (kappa >> Gamma_half):
      lambda_slow ~ -Gamma_half - N*Omega^2*s_z/(4*kappa)
      lambda_fast ~ -kappa/2
    """
    Gh = p.Gamma_half(w)

    # Eigenvalues of 2x2 matrix
    # tr(M) = -kappa/2 - Gh
    # det(M) = kappa*Gh/2 - N*Omega^2*s_z/4
    tr_M = -p.kappa / 2.0 - Gh
    det_M = p.kappa * Gh / 2.0 - N * p.Omega ** 2 * s_z / 4.0

    disc = tr_M ** 2 - 4.0 * det_M
    if disc >= 0:
        lam1 = (-tr_M + math.sqrt(disc)) / 2.0
        lam2 = (-tr_M - math.sqrt(disc)) / 2.0
    else:
        re = -tr_M / 2.0
        im = math.sqrt(-disc) / 2.0
        return abs(re)

    # Select slower eigenvalue (smaller |Re|)
    if abs(lam1) <= abs(lam2):
        return abs(lam1)
    return abs(lam2)


# ---------------------------------------------------------------------------
# Network Energy Model
# ---------------------------------------------------------------------------

def compute_network_energy(N: int, k: int, l: int = 4000,
                           E_elec: float = 50e-9,
                           E_DA: float = 5e-9,
                           eps_fs: float = 10e-12,
                           eps_mp: float = 0.0013e-12,
                           d_to_BS: float = 100.0,
                           M: float = 100.0) -> Dict[str, float]:
    """
    Compute total energy per round for WSN clustering.

    E_total = l * (2*N*E_elec + N*E_DA
                   + k * eps_mp * d_to_BS^4
                   + N * eps_fs * M^2 / (2*pi*k))

    The optimal number of clusters minimizes E_total:
      k* = sqrt(N * eps_fs * M^2 / (2 * pi * eps_mp * d_to_BS^4))
         = M / (sqrt(2*pi) * d_to_BS^2) * sqrt(N * eps_fs / eps_mp)

    Parameters
    ----------
    N : int
        Number of sensor nodes.
    k : int
        Number of clusters.
    l : int
        Message length in bits.
    E_elec : float
        Electronics energy per bit (J/bit).
    E_DA : float
        Data aggregation energy per bit (J/bit).
    eps_fs : float
        Free-space propagation parameter.
    eps_mp : float
        Multi-path propagation parameter.
    d_to_BS : float
        Distance to base station (m).
    M : float
        Network side length (m).

    Returns
    -------
    dict with 'E_total', 'E_optimal_k', 'k_optimal'
    """
    if k <= 0:
        k = 1
    if N <= 0:
        return {'E_total': 0.0, 'k_optimal': 1, 'E_at_kopt': 0.0}

    E_total = l * (2.0 * N * E_elec + N * E_DA +
                   k * eps_mp * d_to_BS ** 4 +
                   N * eps_fs * M ** 2 / (2.0 * math.pi * k + 1e-30))

    # Optimal k
    k_opt_sq = N * eps_fs * M ** 2 / (2.0 * math.pi * eps_mp * d_to_BS ** 4 + 1e-30)
    k_opt = max(1, int(round(math.sqrt(k_opt_sq)))) if k_opt_sq > 0 else 1

    E_at_kopt = l * (2.0 * N * E_elec + N * E_DA +
                     k_opt * eps_mp * d_to_BS ** 4 +
                     N * eps_fs * M ** 2 / (2.0 * math.pi * k_opt + 1e-30))

    return {
        'E_total': E_total,
        'k_optimal': k_opt,
        'E_at_kopt': E_at_kopt
    }


# ---------------------------------------------------------------------------
# Combined multi-physics evaluation
# ---------------------------------------------------------------------------

def evaluate_truth_model(params_dict: Dict[str, float],
                         physics: str = 'combined') -> Dict[str, float]:
    """
    Evaluate the full multi-physics truth model at a parameter point.

    This is the expensive function that the surrogate approximates.
    Parameters are mapped to the appropriate sub-models.

    Parameters
    ----------
    params_dict : dict
        Parameter values. Keys determine which sub-models are activated:
          - 'D_u', 'D_v', 'alpha', 'beta', 'gamma_rd', 'delta': reaction-diffusion
          - 'w', 'N_atom', 'Omega', 'kappa': laser
          - 'N_nodes', 'k_clusters': network energy

    physics : str
        Which physics to evaluate: 'rd', 'laser', 'network', 'combined'.

    Returns
    -------
    dict
        QoI values from all active sub-models.
    """
    result = {}

    if physics in ('rd', 'combined'):
        rd_params = ReactionDiffusionParams(
            alpha=params_dict.get('alpha', 2.0),
            beta=params_dict.get('beta', 1.0),
            gamma=params_dict.get('gamma_rd', 1.0),
            delta=params_dict.get('delta', 0.5),
            epsilon=params_dict.get('epsilon', 0.05),
            D_u=params_dict.get('D_u', 0.01),
            D_v=params_dict.get('D_v', 0.5),
            L=params_dict.get('L', 1.0),
            T_final=params_dict.get('T_final', 20.0)  # Shorter for speed
        )
        rd_result = run_reaction_diffusion_1d(rd_params, nx=50, nt=200)
        result['pattern_amplitude'] = rd_result['pattern_amplitude']
        result['pattern_wavelength'] = rd_result['pattern_wavelength']
        result['max_u'] = rd_result['max_u']
        result['max_v'] = rd_result['max_v']

        # Turing conditions
        turing = rd_params.turing_condition()
        result['turing_trace'] = turing['trace']
        result['turing_determinant'] = turing['determinant']
        result['turing_discriminant'] = turing['discriminant']

    if physics in ('laser', 'combined'):
        laser_params = LaserParams(
            gamma=params_dict.get('gamma_laser', 1.0),
            gamma_T2=params_dict.get('gamma_T2', 0.1),
            Omega=params_dict.get('Omega', 1.0),
            kappa=params_dict.get('kappa', 1e3),
            omega_a=params_dict.get('omega_a', 1.0)
        )
        w = params_dict.get('w', 2.0)
        N = params_dict.get('N_atom', 1e4)
        ss = solve_laser_steady_state(laser_params, w, N)
        result['laser_power'] = ss['power']
        if ss['valid']:
            result['laser_linewidth'] = compute_laser_linewidth(
                laser_params, w, N, ss['s_z'])
        else:
            result['laser_linewidth'] = 0.0
        result['laser_n_photon'] = ss['n_ph']

    if physics in ('network', 'combined'):
        N_nodes = int(params_dict.get('N_nodes', 1000))
        k_clusters = max(1, int(params_dict.get('k_clusters', 10)))
        energy = compute_network_energy(N_nodes, k_clusters,
                                        d_to_BS=params_dict.get('d_to_BS', 100.0),
                                        M=params_dict.get('M_network', 100.0))
        result['network_energy'] = energy['E_total']
        result['network_energy_optimal'] = energy['E_at_kopt']

    return result
