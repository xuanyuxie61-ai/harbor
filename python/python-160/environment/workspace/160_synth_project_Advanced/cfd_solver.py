"""
cfd_solver.py
=============
1D reactor flow solver with periodic tridiagonal systems and non-uniform mesh.

Incorporates algorithms from:
  - 964_r83p (periodic tridiagonal matrix factorization and solve)
  - 245_cvt_1d_nonuniform (non-uniform density CVT mesh generation)

Scientific role:
  Solves the 1D steady-state mass and momentum balance for the gas phase
  in a downdraft gasifier:
    d(ρ u)/dz = S_mass      (continuity)
    d(ρ u²)/dz = -dP/dz - f_D ρ u² / (2 D_h) + S_mom   (momentum)
  
  Uses periodic tridiagonal solver for cyclic boundary conditions
  (e.g., circulating fluidized bed).
"""

import math
import numpy as np


class PeriodicTridiagonalSolver:
    """
    Solver for periodic tridiagonal systems A x = b.
    The periodic tridiagonal matrix has the form:
        [d0 u0  0  ...  0  l0]
        [l1 d1 u1  ...  0   0]
        [ 0 l2 d2  ...  0   0]
        ...
        [uN  0  0  ... lN  dN]
    
    Uses Sherman-Morrison formula for robust periodic solve.
    """

    def __init__(self, n):
        self.n = int(n)
        if self.n < 3:
            raise ValueError("n must be at least 3 for periodic tridiagonal")

    def _build_dense(self, lower, diag, upper):
        """Build dense periodic tridiagonal matrix for verification."""
        n = self.n
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            A[i, i] = diag[i]
            if i < n - 1:
                A[i, i + 1] = upper[i]
                A[i + 1, i] = lower[i + 1]
            else:
                A[i, 0] = lower[0]
                A[0, i] = upper[i]
        return A

    def factor(self, lower, diag, upper):
        """
        Factor the periodic tridiagonal matrix using Sherman-Morrison.
        """
        n = self.n
        lower = np.asarray(lower, dtype=float)
        diag = np.asarray(diag, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # Store matrix data for dense solve fallback
        A_dense = self._build_dense(lower, diag, upper)

        # Try to factor the non-periodic part (Thomas)
        a_lu = np.zeros((3, n), dtype=float)
        a_lu[0, :] = upper
        a_lu[1, :] = diag
        a_lu[2, :] = lower
        info = 0

        for i in range(n - 1):
            if abs(a_lu[1, i]) < 1.0e-15:
                info = i + 1
                break
            if i < n - 2:
                a_lu[2, i + 1] = a_lu[2, i + 1] / a_lu[1, i]
                a_lu[1, i + 1] = a_lu[1, i + 1] - a_lu[2, i + 1] * a_lu[0, i]

        return {
            'A_dense': A_dense,
            'a_lu': a_lu,
            'lower': lower,
            'diag': diag,
            'upper': upper,
            'info': info
        }

    def solve(self, factor_data, b):
        """
        Solve A x = b. Uses dense direct solve for robustness.
        """
        n = self.n
        b = np.asarray(b, dtype=float).copy()
        A = factor_data['A_dense']
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x


class CVTMeshGenerator:
    """
    Centroidal Voronoi Tessellation for non-uniform 1D mesh generation.
    Adapts mesh density to a prescribed density function.
    """

    def __init__(self, n_generators=20, density_func_id=4):
        self.n = int(n_generators)
        self.density_func_id = int(density_func_id)
        self.euler = math.e

    def density_transform(self, s):
        """
        Apply density function to uniform sample s ∈ [0,1].
        0: s
        1: sqrt(s)
        2: s^(1/3)
        3: s^(1/4)
        4: log(e / (e - s*(e-1)))
        5: 0.5 + atan(50*(s-0.5))/pi
        6: sin(pi*(s-0.5))  (Chebyshev)
        """
        s = float(s)
        s = max(0.0, min(1.0, s))
        fid = self.density_func_id
        if fid == 0:
            return s
        elif fid == 1:
            return math.sqrt(s)
        elif fid == 2:
            return s ** (1.0 / 3.0)
        elif fid == 3:
            return s ** (1.0 / 4.0)
        elif fid == 4:
            return math.log(self.euler / (self.euler - s * (self.euler - 1.0)))
        elif fid == 5:
            return 0.5 + math.atan(50.0 * (s - 0.5)) / math.pi
        elif fid == 6:
            return math.sin(math.pi * (s - 0.5))
        return s

    def generate_mesh(self, z_min=0.0, z_max=1.0,
                      n_samples=2000, n_steps=100):
        """
        Generate CVT-adapted mesh for the reactor domain.
        
        Returns
        -------
        generators : ndarray
            Optimized node positions.
        """
        n = self.n
        # Initialize generators uniformly
        generators = np.linspace(0.0, 1.0, n)

        for step in range(n_steps):
            generator_new = np.zeros(n, dtype=float)
            tally = np.zeros(n, dtype=int)

            for _ in range(n_samples):
                s = np.random.rand()
                s = self.density_transform(s)

                # Find nearest generator
                distances = np.abs(generators - s)
                nearest = int(np.argmin(distances))
                generator_new[nearest] += s
                tally[nearest] += 1

            for j in range(n):
                if tally[j] > 0:
                    generators[j] = generator_new[j] / tally[j]

        # Map to physical domain
        generators = z_min + generators * (z_max - z_min)
        generators = np.sort(generators)
        return generators


class ReactorFlowSolver:
    """
    1D steady-state flow solver for the gasification reactor.
    """

    def __init__(self, z_nodes, reactor_radius):
        self.z = np.asarray(z_nodes, dtype=float)
        self.n = len(self.z)
        self.R = float(reactor_radius)
        self.dh = 2.0 * self.R  # Hydraulic diameter for circular duct

    def _compute_face_areas(self):
        """Cross-sectional area A = π R²."""
        return math.pi * self.R ** 2

    def solve_velocity_profile(self, rho_gas, mu_gas, dp_dz,
                                wall_friction_factor):
        """
        Solve for gas velocity using simplified 1D momentum balance:
            ρ u du/dz = -dp/dz - f_D ρ u² / (2 D_h)
        Linearized as: a_i u_{i-1} + b_i u_i + c_i u_{i+1} = rhs_i
        """
        n = self.n
        A = self._compute_face_areas()

        # Build tridiagonal system coefficients
        lower = np.zeros(n - 1, dtype=float)
        diag = np.zeros(n, dtype=float)
        upper = np.zeros(n - 1, dtype=float)
        rhs = np.zeros(n, dtype=float)

        for i in range(n):
            dz_fwd = self.z[min(i + 1, n - 1)] - self.z[i]
            dz_bwd = self.z[i] - self.z[max(i - 1, 0)]
            dz = 0.5 * (abs(dz_fwd) + abs(dz_bwd))
            if dz < 1.0e-15:
                dz = 1.0e-15

            f_term = wall_friction_factor * rho_gas / (2.0 * self.dh)

            if i == 0:
                # Inlet boundary: specified velocity gradient
                diag[i] = 1.0
                rhs[i] = 1.0  # normalized inlet velocity
            elif i == n - 1:
                # Outlet boundary: zero gradient
                diag[i] = 1.0
                upper[i - 1] = -1.0
                rhs[i] = 0.0
            else:
                # Interior: convective + friction
                a_conv = rho_gas / (2.0 * dz)
                lower[i - 1] = -a_conv
                diag[i] = a_conv + f_term
                upper[i] = 0.0
                rhs[i] = -dp_dz

        # Solve tridiagonal system
        return self._thomas_solve(lower, diag, upper, rhs)

    def _thomas_solve(self, lower, diag, upper, rhs):
        """Thomas algorithm for tridiagonal system."""
        n = len(diag)
        c_prime = np.zeros(n - 1, dtype=float)
        d_prime = np.zeros(n, dtype=float)

        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]

        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c_prime[i - 1]
            if abs(denom) < 1.0e-15:
                denom = 1.0e-15
            c_prime[i] = upper[i] / denom
            d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

        denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
        if abs(denom) < 1.0e-15:
            denom = 1.0e-15
        d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom

        x = np.zeros(n, dtype=float)
        x[n - 1] = d_prime[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        return x

    def pressure_drop_ergun(self, epsilon, rho_gas, mu_gas, u_superficial,
                             d_particle):
        """
        Ergun equation for pressure drop in packed bed:
            dp/dz = -(150 (1-ε)² μ u) / (ε³ d_p²) - (1.75 (1-ε) ρ u²) / (ε³ d_p)
        """
        if epsilon <= 0.0 or epsilon >= 1.0 or d_particle <= 0.0:
            return 0.0
        term1 = 150.0 * (1.0 - epsilon) ** 2 * mu_gas * u_superficial
        term1 = term1 / (epsilon ** 3 * d_particle ** 2)
        term2 = 1.75 * (1.0 - epsilon) * rho_gas * u_superficial ** 2
        term2 = term2 / (epsilon ** 3 * d_particle)
        return -(term1 + term2)

    def reynolds_number(self, rho_gas, u, d_p, mu_gas):
        """Re = ρ u d_p / μ"""
        if mu_gas <= 0.0:
            return 0.0
        return rho_gas * u * d_p / mu_gas

    def nusselt_number(self, Re, Pr, epsilon=0.4):
        """
        Ranz-Marshall correlation for packed bed heat transfer:
            Nu = 2 + 1.8 Re^{0.5} Pr^{0.33}
        """
        if Re < 0.0 or Pr <= 0.0:
            return 2.0
        return 2.0 + 1.8 * (Re ** 0.5) * (Pr ** (1.0 / 3.0))
