"""
ftcs_fission_pde.py
====================
FTCS (Forward Time Centered Space) finite-difference solution of the
fission dynamics PDE in collective coordinate space.

Maps from: 434_fisher_pde_ftcs (KPP-Fisher PDE with FTCS scheme)

Physical context
----------------
The probability distribution P(c, t) in elongation space evolves by:

  dP/dt = D * d2P/dc2 - d/dc[F(c)*P] + S(c, t) - L(c, t)

where:
- D is the collective diffusion coefficient (related to temperature and friction)
- F(c) = -dV/dc is the collective force from the PES
- S is a source term (compound nucleus formation)
- L is a loss term (absorption at scission)

This is analogous to the Fisher-KPP equation:
  du/dt = D * u_xx + r*u*(1 - u/K)

where the "reaction" term r*u*(1-u/K) represents the probability flux.

Boundary conditions:
  Left: P(c_min, t) = P_0(t) (compound nucleus injection)
  Right: dP/dc(c_max, t) = 0 (absorbing at scission)
"""

import math
from typing import List, Dict, Tuple, Optional

from potential_energy_surface import FissionPES


class FTCSFissionPDE:
    """
    FTCS solver for the fission probability-diffusion equation.

    PDE: dP/dt = D * d2P/dc2 - d/dc[F(c)*P]

    Discretised (FTCS):
    P_i^{n+1} = P_i^n + dt * {D*(P_{i+1}^n - 2*P_i^n + P_{i-1}^n)/dc^2
                                - [F_{i+1}*P_{i+1}^n - F_{i-1}*P_{i-1}^n]/(2*dc)}

    CFL stability: dt < dc^2 / (2*D)
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236,
                 n_grid: int = 101, c_min: float = 0.9,
                 c_max: float = 2.5, temperature: float = 1.5):
        self.pes = FissionPES(z_cn, a_cn)
        self.Z = z_cn
        self.A = a_cn
        self.T = temperature

        self.n = n_grid
        self.c_min = c_min
        self.c_max = c_max
        self.dc = (c_max - c_min) / (n_grid - 1)
        self.c_grid = [c_min + i * self.dc for i in range(n_grid)]

        # Diffusion coefficient (from Einstein relation):
        # D = T / gamma (temperature / friction)
        self.gamma_friction = 5.0e21  # s^{-1}
        self.D = temperature * 1.0e-3 / self.gamma_friction  # simplified

        # Time step (CFL condition)
        self.dt_max = self.dc ** 2 / (2.0 * max(self.D, 1e-30))
        self.dt = 0.8 * self.dt_max  # safety factor

        # Solution arrays
        self.P = [0.0] * n_grid
        self.F = [0.0] * n_grid  # collective force at grid points

        # Precompute force field
        self._compute_force_field()

        # Solution history
        self.history: List[List[float]] = []

    def _compute_force_field(self) -> None:
        """Precompute collective force F(c) = -dV/dc at each grid point."""
        h_fixed = 1.0
        alpha_fixed = 0.0
        for i, c in enumerate(self.c_grid):
            grad_c, _, _ = self.pes.gradient(c, h_fixed, alpha_fixed)
            self.F[i] = -grad_c

    def initialize_gaussian(self, c0: float = 1.0,
                            sigma: float = 0.05) -> None:
        """
        Initialise probability as a Gaussian centred at c0:
          P(c, 0) = exp(-(c - c0)^2 / (2*sigma^2)) / (sigma*sqrt(2*pi))
        """
        norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        for i, c in enumerate(self.c_grid):
            self.P[i] = norm * math.exp(-(c - c0) ** 2 / (2.0 * sigma * sigma))

        # Normalise to 1
        total = sum(self.P) * self.dc
        if total > 0:
            for i in range(self.n):
                self.P[i] /= total

    def step_ftcs(self) -> None:
        """
        Single FTCS time step.

        P_i^{n+1} = P_i^n + dt * [D*(P_{i+1} - 2*P_i + P_{i-1})/dc^2
                        - (F_{i+1}*P_{i+1} - F_{i-1}*P_{i-1})/(2*dc)]

        Analogous to Fisher PDE step from 434_fisher_pde_ftcs:
          u_new = u + dt * (dudxx + u.*(1-u))
        but with fission-specific force and diffusion terms.
        """
        P_new = [0.0] * self.n
        dt = self.dt
        dc = self.dc
        D = self.D

        for i in range(1, self.n - 1):
            # Diffusion term (central difference)
            d2P = (self.P[i + 1] - 2.0 * self.P[i] + self.P[i - 1]) / (dc * dc)

            # Advection term (central difference)
            dFP = (self.F[i + 1] * self.P[i + 1]
                   - self.F[i - 1] * self.P[i - 1]) / (2.0 * dc)

            # Fisher-like reaction term: probability growth in barrier region
            # (analogous to u*(1-u) in Fisher equation)
            # This models the compound nucleus -> fission transition
            c = self.c_grid[i]
            v_pes = self.pes.total_potential(c, 1.0, 0.0)
            if self.T > 0 and v_pes > 0:
                boltz = math.exp(-v_pes / self.T)
            else:
                boltz = 0.0
            reaction = boltz * self.P[i] * (1.0 - self.P[i]) if self.P[i] < 1.0 else 0.0

            P_new[i] = self.P[i] + dt * (D * d2P - dFP + reaction)

            # Prevent negative probabilities
            if P_new[i] < 0:
                P_new[i] = 0.0

        # Boundary conditions
        # Left: Dirichlet (compound nucleus source)
        P_new[0] = self.P[0] * 0.99  # slow decay
        # Right: Neumann (zero gradient at scission)
        P_new[-1] = P_new[-2]

        self.P = P_new
        self.history.append(self.P[:])

    def run(self, n_steps: int = 500) -> Dict[str, any]:
        """
        Run the FTCS solver for n_steps time steps.

        Returns dict with solution statistics.
        """
        for step in range(n_steps):
            self.step_ftcs()

        # Compute statistics
        total_prob = sum(self.P) * self.dc
        mean_c = sum(c * p for c, p in zip(self.c_grid, self.P)) * self.dc
        if total_prob > 0:
            mean_c /= total_prob

        # Flux at scission (right boundary)
        flux_scission = self.D * (self.P[-1] - self.P[-2]) / self.dc

        # Fission probability (probability that has crossed scission)
        c_scission = 2.0
        prob_fissioned = 0.0
        for i, c in enumerate(self.c_grid):
            if c > c_scission:
                prob_fissioned += self.P[i] * self.dc

        return {
            'n_steps': n_steps,
            'total_time': n_steps * self.dt,
            'total_probability': total_prob,
            'mean_elongation': mean_c,
            'flux_at_scission': flux_scission,
            'prob_fissioned': prob_fissioned,
            'dt': self.dt,
            'dc': self.dc,
            'cfl_number': self.D * self.dt / (self.dc * self.dc),
        }

    def compute_energy_dissipation(self) -> float:
        """
        Compute energy dissipated by friction during the evolution:
          E_diss = integral gamma * v^2 dt
        where v ~ d<c>/dt is the collective velocity.

        Approximated from the probability flux.
        """
        if len(self.history) < 2:
            return 0.0

        e_diss = 0.0
        for t_step in range(1, len(self.history)):
            P_prev = self.history[t_step - 1]
            P_curr = self.history[t_step]

            # Mean velocity
            mean_c_prev = sum(c * p for c, p in zip(self.c_grid, P_prev)) * self.dc
            mean_c_curr = sum(c * p for c, p in zip(self.c_grid, P_curr)) * self.dc
            v_coll = (mean_c_curr - mean_c_prev) / self.dt

            e_diss += self.gamma_friction * v_coll * v_coll * self.dt

        return e_diss


def kramers_escape_rate(barrier_height: float, temperature: float,
                        omega_gs: float, omega_sp: float,
                        gamma: float) -> float:
    """
    Kramers escape rate for fission:

    Gamma_K = (omega_gs / (2*pi)) *
              (sqrt(1 + (gamma/(2*omega_sp))^2) - gamma/(2*omega_sp)) *
              exp(-B_f / T)

    where omega_gs is the ground-state frequency, omega_sp the saddle-point
    frequency, gamma the friction coefficient, B_f the barrier height,
    T the temperature.
    """
    if temperature <= 0:
        return 0.0

    gamma_ratio = gamma / (2.0 * omega_sp) if omega_sp > 0 else 0.0
    kramers_factor = (math.sqrt(1.0 + gamma_ratio * gamma_ratio) - gamma_ratio)

    rate = (omega_gs / (2.0 * math.pi)) * kramers_factor * math.exp(-barrier_height / temperature)
    return rate
