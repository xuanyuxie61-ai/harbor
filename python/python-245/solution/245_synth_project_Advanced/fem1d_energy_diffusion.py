"""
fem1d_energy_diffusion.py
==========================
1D finite element method for the diffusion of excitation energy
in a fissioning nucleus (along the elongation coordinate).

Maps from: 390_fem1d_heat_explicit (explicit FEM heat equation solver)

Physical context
----------------
The excitation energy distribution E*(c, t) along the elongation coordinate
evolves according to:

  C(c) * dE*/dt = d/dc[k(c) * dE*/dc] - h(c) * (E* - E_eq)

where:
- C(c) is the heat capacity at deformation c
- k(c) is the thermal conductivity (deformation-dependent)
- h(c) is a heat transfer coefficient (energy dissipation to collective motion)
- E_eq is the equilibrium excitation energy

This is solved using 1D FEM with linear basis functions and explicit
forward Euler time stepping, exactly as in the fem1d_heat_explicit project.

FEM discretisation:
  M * dU/dt = -K * U + F

where M is the mass matrix, K the stiffness matrix, F the load vector.

CFL condition:
  dt < min(dx^2 * C / (2*k))
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from stability_analysis import (
    compute_fem_mass_matrix_1d, compute_fem_stiffness_matrix_1d,
    quadrature_on_element
)


class FEM1DEnergyDiffusion:
    """
    1D FEM solver for excitation energy diffusion in fission.

    Uses linear (hat) basis functions on a uniform mesh,
    with explicit forward Euler time integration.
    """

    def __init__(self, n_elements: int = 50, length: float = 1.6,
                 c_min: float = 0.9, diffusivity: float = 0.1):
        """
        Parameters
        ----------
        n_elements : int
            Number of finite elements.
        length : float
            Domain length in elongation space.
        c_min : float
            Minimum elongation coordinate.
        diffusivity : float
            Thermal diffusivity k/C (fm^2 / zeptosecond).
        """
        self.ne = n_elements
        self.nn = n_elements + 1  # number of nodes
        self.length = length
        self.c_min = c_min
        self.c_max = c_min + length
        self.dx = length / n_elements
        self.D = diffusivity

        # Node coordinates
        self.x = np.array([c_min + i * self.dx for i in range(self.nn)])

        # Assemble FEM matrices
        self.M = compute_fem_mass_matrix_1d(n_elements, length)
        self.K = compute_fem_stiffness_matrix_1d(n_elements, length, diffusivity)

        # CFL time step
        self.dt_max = self.dx ** 2 / (2.0 * max(diffusivity, 1e-30))
        self.dt = 0.8 * self.dt_max

        # Solution vector
        self.U = np.zeros(self.nn)

        # Deformation-dependent heat capacity and conductivity
        self.heat_capacity = self._compute_heat_capacity()
        self.conductivity = self._compute_conductivity()

        # History
        self.history: List[np.ndarray] = []

    def _compute_heat_capacity(self) -> np.ndarray:
        """
        Heat capacity C(c) at each node.

        C = a_param * T  where a_param is the level density parameter
        and T is the local temperature.

        For the fissioning nucleus:
          C(c) ~ A/8 * T(c)  [MeV / MeV = dimensionless * T]
        """
        A = 236
        a_param = A / 8.0  # MeV^{-1}
        T_typical = 1.5  # MeV

        # Deformation dependence: increases near scission
        c_norm = (self.x - self.c_min) / self.length
        c_factor = 1.0 + 0.5 * c_norm  # increases with elongation

        return a_param * T_typical * c_factor

    def _compute_conductivity(self) -> np.ndarray:
        """
        Thermal conductivity k(c) at each node.

        k = D * C(c) where D is the diffusivity.

        In the nucleus, thermal conductivity depends on the mean free path
        of nucleons: k ~ (1/3) * C_v * v_F * l_mfp
        where v_F ~ 0.3c (Fermi velocity) and l_mfp ~ 3-5 fm.
        """
        return self.D * self.heat_capacity

    def initialize_gaussian(self, x0: float = 1.1,
                            sigma: float = 0.1,
                            amplitude: float = 5.0) -> None:
        """
        Initialise excitation energy as a Gaussian:
          E*(c, 0) = A * exp(-(c - x0)^2 / (2*sigma^2))
        """
        for i in range(self.nn):
            self.U[i] = amplitude * math.exp(
                -(self.x[i] - x0) ** 2 / (2.0 * sigma * sigma)
            )

    def apply_boundary_conditions(self) -> None:
        """
        Apply boundary conditions:
        - Left: Dirichlet E*(c_min) = E0 (compound nucleus excitation)
        - Right: Neumann dE*/dc = 0 (insulated at scission)
        """
        # Dirichlet at left (compound nucleus energy)
        E_compound = 6.5  # MeV (typical for U-236*)
        self.U[0] = E_compound

        # Neumann at right: U[-1] = U[-2]
        self.U[-1] = self.U[-2]

    def assemble_load_vector(self) -> np.ndarray:
        """
        Assemble load vector F with source terms.

        F_i = integral f(x) * phi_i(x) dx

        where f(x) is the energy source (e.g., from dissipation of
        collective kinetic energy).
        """
        F = np.zeros(self.nn)
        # Source: dissipation of collective energy (Gaussian in elongation)
        for i in range(self.nn):
            c = self.x[i]
            # Source near the barrier region
            source = 0.5 * math.exp(-(c - 1.3) ** 2 / 0.05)
            F[i] = source * self.dx

        return F

    def step_explicit(self) -> None:
        """
        Explicit forward Euler time step:

        M * dU/dt = -K * U + F
        => U^{n+1} = U^n + dt * M^{-1} * (-K * U^n + F)

        For lumped mass matrix M_lump = diag(m_i):
          U_i^{n+1} = U_i^n + dt/m_i * (-sum_j K_ij * U_j^n + F_i)

        From 390_fem1d_heat_explicit:
          u_new = u + dt * M^{-1} * (-K*u + b)
        """
        # Lumped mass matrix (diagonal)
        m_lump = np.zeros(self.nn)
        for i in range(self.nn):
            m_lump[i] = np.sum(self.M[i, :])
            if m_lump[i] < 1e-14:
                m_lump[i] = 1e-14

        # Load vector
        F = self.assemble_load_vector()

        # The stiffness matrix K_fem is assembled with positive diagonal
        # For linear elements: (K*u)_i = (D/h)*(u_{i-1} - 2u_i + u_{i+1})
        #                      = D * h * d^2 u/dx^2 (approximately)
        # With lumped mass m_lump = h per node:
        #   M^{-1} * K * u = D * d^2 u/dx^2  (the heat equation)
        # So dU/dt = + K*u / m_lump gives forward heat equation.
        KU = self.K @ self.U

        # Update
        dU_dt = (-KU + F) / m_lump
        self.U = self.U + self.dt * dU_dt

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Store history
        self.history.append(self.U.copy())

    def run(self, n_steps: int = 200) -> Dict[str, any]:
        """
        Run the FEM solver for n_steps time steps.
        """
        for step in range(n_steps):
            self.step_explicit()

        # Statistics
        total_energy = np.sum(self.U) * self.dx
        mean_energy = np.sum(self.x * self.U) * self.dx / max(total_energy, 1e-30)

        return {
            'n_steps': n_steps,
            'total_time': n_steps * self.dt,
            'total_energy': total_energy,
            'mean_position': mean_energy,
            'max_energy': float(np.max(self.U)),
            'min_energy': float(np.min(self.U)),
            'dt': self.dt,
            'cfl_number': self.D * self.dt / (self.dx ** 2),
        }

    def basis_function(self, element: int, x_val: float) -> Tuple[float, float]:
        """
        Evaluate linear basis functions for given element at position x.

        phi_1(x) = (x_{i+1} - x) / h  (left node)
        phi_2(x) = (x - x_i) / h      (right node)

        From 390_fem1d_heat_explicit basis_function concept.
        """
        if element < 0 or element >= self.ne:
            return (0.0, 0.0)

        x_left = self.x[element]
        x_right = self.x[element + 1]
        h = x_right - x_left

        if h < 1e-14:
            return (0.0, 0.0)

        xi = (x_val - x_left) / h  # local coordinate [0, 1]

        phi1 = 1.0 - xi
        phi2 = xi

        # Clamp
        phi1 = max(0.0, min(1.0, phi1))
        phi2 = max(0.0, min(1.0, phi2))

        return (phi1, phi2)

    def interpolate_solution(self, x_val: float) -> float:
        """
        Interpolate the FEM solution at an arbitrary point x.
        """
        if x_val <= self.x[0]:
            return self.U[0]
        if x_val >= self.x[-1]:
            return self.U[-1]

        # Find element
        element = int((x_val - self.x[0]) / self.dx)
        element = max(0, min(self.ne - 1, element))

        phi1, phi2 = self.basis_function(element, x_val)
        return phi1 * self.U[element] + phi2 * self.U[element + 1]
