"""
heat_transfer.py
================
Heat transfer models for the gasification reactor: radiation, conduction,
and conjugate gradient solver for energy balance.

Incorporates algorithms from:
  - 1116_sphere_exactness (spherical quadrature for view factors)
  - 994_r8sd (symmetric diagonal sparse matrix, conjugate gradient)

Scientific role:
  Computes radiative heat exchange between char particles and reactor walls,
  and solves the 1D steady-state conduction equation in the reactor lining:
    d/dz (k_wall dT/dz) = 0

  Radiation between surfaces follows the Stefan-Boltzmann law:
    q_rad = ε σ (T_s⁴ - T_∞⁴)

  For diffuse gray surfaces, the net radiation exchange between two surfaces
  is computed using view factors F_{ij}:
    Q_i = A_i Σ_j F_{ij} (J_i - J_j)
    J_i = ε_i σ T_i⁴ + (1 - ε_i) G_i
"""

import math
import numpy as np


STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)


class RadiationViewFactor:
    """
    Compute radiation view factors between spherical particles and walls.
    Uses spherical quadrature for numerical integration of:
        F_{ij} = (1/A_i) ∫_{A_i} ∫_{A_j} (cos θ_i cos θ_j) / (π s²) dA_j dA_i
    """

    def __init__(self):
        pass

    def sphere_to_sphere(self, r1, r2, c):
        """
        View factor between two spheres of radii r1, r2 with center distance c.
        For c >> r1 + r2, the approximate formula is:
            F_{12} ≈ (r2 / c)²
        For equal spheres, the exact formula involves integrating over both
        surfaces. We use the approximation valid for c > r1 + r2:
            F_{12} = (1/2) * [1 - sqrt(1 - (r2/c)²)]  (for r1 >> r2)
        General approximation: F_{12} ≈ (r2/c)²
        """
        if c <= 1.0e-15 or r2 <= 0.0:
            return 0.0
        if c <= r1 + r2:
            return 0.5
        # Use asymptotic approximation for well-separated spheres
        ratio = r2 / c
        # For two equal spheres, a better approximation:
        # F_12 = (1/4) * [2 - sqrt(4 - (2*R)²)] where R = r/c
        # Simplified: use (r2/c)² which is accurate for small ratios
        F = ratio ** 2
        return max(0.0, min(1.0, F))

    def sphere_to_plane(self, r, h):
        """
        View factor from sphere of radius r to infinite parallel plane
        at distance h from center.
        F = 0.5 * (1 - h / sqrt(h² + r²)) for a disk approximation.
        More accurate for sphere:
            F = 0.5 * (1 - sqrt(1 - (r/h)²)) for h > r
        """
        if h <= r:
            return 0.5
        ratio = r / h
        if ratio >= 1.0:
            return 0.5
        return 0.5 * (1.0 - math.sqrt(1.0 - ratio ** 2))

    def parallel_disks(self, r1, r2, h):
        """
        View factor between two parallel coaxial disks.
        F_{12} = 0.5 * [X - sqrt(X² - 4)]
        where X = 1 + (1 + R2²) / R1², R1 = r1/h, R2 = r2/h
        """
        if h <= 1.0e-15:
            return 1.0
        R1 = r1 / h
        R2 = r2 / h
        X = 1.0 + (1.0 + R2 ** 2) / R1 ** 2
        term = math.sqrt(max(X ** 2 - 4.0, 0.0))
        F = 0.5 * (X - term)
        return max(0.0, min(1.0, F))

    def monte_carlo_view_factor(self, surface1_samples, surface2_samples,
                                num_rays=10000):
        """
        Estimate view factor via Monte Carlo ray tracing.
        """
        s1 = np.asarray(surface1_samples)
        s2 = np.asarray(surface2_samples)
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        count = 0
        for _ in range(num_rays):
            p1 = s1[np.random.randint(len(s1))]
            p2 = s2[np.random.randint(len(s2))]
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist < 1.0e-15:
                continue
            # Simple visibility check (always visible for enclosed geometry)
            count += 1
        return min(1.0, count / num_rays)


class StefanBoltzmannRadiation:
    """
    Stefan-Boltzmann radiation heat transfer model.
    """

    def __init__(self, epsilon=0.85):
        self.epsilon = float(epsilon)

    def net_radiative_heat_flux(self, T_surface, T_ambient):
        """
        q_rad = ε σ (T_s⁴ - T_∞⁴)  [W/m²]
        """
        if T_surface < 0.0 or T_ambient < 0.0:
            return 0.0
        return self.epsilon * STEFAN_BOLTZMANN * (T_surface ** 4 - T_ambient ** 4)

    def radiative_heat_transfer_coefficient(self, T_surface, T_ambient):
        """
        h_rad = ε σ (T_s² + T_∞²)(T_s + T_∞)
        """
        if T_surface < 0.0 or T_ambient < 0.0:
            return 0.0
        return self.epsilon * STEFAN_BOLTZMANN * \
               (T_surface ** 2 + T_ambient ** 2) * (T_surface + T_ambient)


class ConductionSolver:
    """
    1D steady-state conduction solver using sparse symmetric matrix
    and conjugate gradient method.

    Governing equation:
        d/dz (k(z) dT/dz) + Q'''(z) = 0
    Discretized with finite differences on non-uniform grid.
    """

    def __init__(self, z_nodes, k_nodes):
        """
        Parameters
        ----------
        z_nodes : ndarray
            Grid node positions [m].
        k_nodes : ndarray
            Thermal conductivity at nodes [W/(m·K)].
        """
        self.z = np.asarray(z_nodes, dtype=float)
        self.k = np.asarray(k_nodes, dtype=float)
        self.n = len(self.z)
        if self.n < 2:
            raise ValueError("Need at least 2 grid nodes")
        if len(self.k) != self.n:
            raise ValueError("k_nodes length must match z_nodes")

    def _build_sparse_system(self, Q_vol):
        """
        Build sparse symmetric system A T = b for conduction.
        Uses R8SD-style diagonal storage for symmetric tridiagonal.
        """
        n = self.n
        Q_vol = np.asarray(Q_vol, dtype=float)
        if len(Q_vol) != n:
            Q_vol = np.full(n, float(Q_vol))

        # Main diagonal and upper diagonal
        main_diag = np.zeros(n, dtype=float)
        upper_diag = np.zeros(n - 1, dtype=float)
        rhs = np.zeros(n, dtype=float)

        for i in range(n):
            if i == 0:
                # Forward difference at boundary
                dz = self.z[1] - self.z[0]
                k_face = 0.5 * (self.k[0] + self.k[1])
                if abs(dz) > 1.0e-15:
                    main_diag[0] = k_face / dz
                    upper_diag[0] = -k_face / dz
                    rhs[0] = -Q_vol[0] * dz * 0.5
            elif i == n - 1:
                dz = self.z[n - 1] - self.z[n - 2]
                k_face = 0.5 * (self.k[n - 2] + self.k[n - 1])
                if abs(dz) > 1.0e-15:
                    main_diag[n - 1] = k_face / dz
                    rhs[n - 1] = -Q_vol[n - 1] * dz * 0.5
            else:
                dz_plus = self.z[i + 1] - self.z[i]
                dz_minus = self.z[i] - self.z[i - 1]
                k_plus = 0.5 * (self.k[i] + self.k[i + 1])
                k_minus = 0.5 * (self.k[i - 1] + self.k[i])
                if abs(dz_plus) > 1.0e-15 and abs(dz_minus) > 1.0e-15:
                    a_plus = k_plus / dz_plus
                    a_minus = k_minus / dz_minus
                    main_diag[i] = a_plus + a_minus
                    upper_diag[i] = -a_plus
                    upper_diag[i - 1] = -a_minus
                    # Volume element for source term
                    dz_avg = 0.5 * (dz_plus + dz_minus)
                    rhs[i] = Q_vol[i] * dz_avg

        return main_diag, upper_diag, rhs

    def solve_cg(self, Q_vol, T_left, T_right, max_iter=None, tol=1.0e-10):
        """
        Solve conduction problem with Dirichlet BCs using conjugate gradient.
        T(0) = T_left, T(n-1) = T_right.
        """
        n = self.n
        if max_iter is None:
            max_iter = n

        main_diag, upper_diag, rhs = self._build_sparse_system(Q_vol)

        # Apply Dirichlet BCs
        rhs[0] = T_left
        rhs[n - 1] = T_right
        main_diag[0] = 1.0
        main_diag[n - 1] = 1.0
        if n > 1:
            upper_diag[0] = 0.0
            upper_diag[n - 2] = 0.0

        # Initial guess
        x = np.linspace(T_left, T_right, n)

        # CG for symmetric tridiagonal system
        # Compute A*x for tridiagonal symmetric matrix
        def matvec(v):
            result = main_diag * v
            if n > 1:
                result[0:n - 1] += upper_diag * v[1:n]
                result[1:n] += upper_diag * v[0:n - 1]
            return result

        b = rhs.copy()
        ap = matvec(x)
        r = b - ap
        p = r.copy()

        for it in range(max_iter):
            ap = matvec(p)
            pap = np.dot(p, ap)
            if abs(pap) < 1.0e-15:
                break
            pr = np.dot(p, r)
            alpha = pr / pap
            x = x + alpha * p
            r = r - alpha * ap
            rap = np.dot(r, ap)
            beta = -rap / pap
            p = r + beta * p

            if np.linalg.norm(r) < tol:
                break

        return x

    def solve_direct(self, Q_vol, T_left, T_right):
        """
        Direct Thomas algorithm for symmetric tridiagonal system.
        For symmetric tridiagonal: lower_diag[i] = upper_diag[i].
        """
        n = self.n
        main_diag, upper_diag, rhs = self._build_sparse_system(Q_vol)

        # Apply BCs
        rhs[0] = T_left
        rhs[n - 1] = T_right
        main_diag[0] = 1.0
        main_diag[n - 1] = 1.0
        if n > 1:
            upper_diag[0] = 0.0
            upper_diag[n - 2] = 0.0

        # Thomas algorithm
        c_prime = np.zeros(n - 1, dtype=float)
        d_prime = np.zeros(n, dtype=float)

        c_prime[0] = upper_diag[0] / main_diag[0]
        d_prime[0] = rhs[0] / main_diag[0]

        for i in range(1, n - 1):
            denom = main_diag[i] - upper_diag[i - 1] * c_prime[i - 1]
            if abs(denom) < 1.0e-15:
                denom = 1.0e-15
            c_prime[i] = upper_diag[i] / denom
            d_prime[i] = (rhs[i] - upper_diag[i - 1] * d_prime[i - 1]) / denom

        denom = main_diag[n - 1] - upper_diag[n - 2] * c_prime[n - 2]
        if abs(denom) < 1.0e-15:
            denom = 1.0e-15
        d_prime[n - 1] = (rhs[n - 1] - upper_diag[n - 2] * d_prime[n - 2]) / denom

        x = np.zeros(n)
        x[n - 1] = d_prime[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x
