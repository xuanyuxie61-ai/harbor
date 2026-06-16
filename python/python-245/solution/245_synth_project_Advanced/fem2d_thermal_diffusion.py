"""
fem2d_thermal_diffusion.py
============================
2D finite element solver for thermal diffusion in the fissioning
nucleus with piecewise-constant diffusivity.

Maps from: 846_paraheat_functional (2D heat equation FEM with
           piecewise diffusivity on triangular mesh)

Physical context
----------------
After scission, the two fragments are hot and cool by:
1. Neutron evaporation
2. Gamma emission
3. Thermal conduction between fragments (before full separation)

The 2D heat equation with spatially-varying thermal conductivity:
  C(r,z) * dT/dt = div[k(r,z) * grad(T)] + Q(r,z,t)

where k(r,z) is piecewise constant over sub-regions corresponding
to the light and heavy fragments.

We use a 2D triangular mesh FEM (from paraheat_functional):
- Rectangular domain [r_min, r_max] x [z_min, z_max]
- Each cell split into 2 triangles (T3 elements)
- Piecewise constant diffusivity on a coarser sub-grid
- 7-point quadrature on reference triangle for stiffness matrix assembly
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np


class FEM2DThermalDiffusion:
    """
    2D FEM solver for thermal diffusion in a fissioning nucleus.

    Domain: rectangular [0, L_r] x [0, L_z]
    Mesh: uniform triangulation (each rect split into 2 triangles)
    Diffusivity: piecewise constant over sub-grid
    """

    def __init__(self, nx: int = 11, nz: int = 11,
                 lr: float = 10.0, lz: float = 20.0,
                 k_light: float = 0.5, k_heavy: float = 0.3):
        """
        Parameters
        ----------
        nx : int
            Number of nodes in r-direction.
        nz : int
            Number of nodes in z-direction.
        lr : float
            Domain length in r (fm).
        lz : float
            Domain length in z (fm).
        k_light : float
            Thermal conductivity in light fragment region.
        k_heavy : float
            Thermal conductivity in heavy fragment region.
        """
        self.nx = nx
        self.nz = nz
        self.nn = nx * nz  # total nodes
        self.lr = lr
        self.lz = lz
        self.dx = lr / (nx - 1)
        self.dz = lz / (nz - 1)
        self.k_light = k_light
        self.k_heavy = k_heavy

        # Node coordinates
        self.x_nodes = np.zeros(self.nn)
        self.y_nodes = np.zeros(self.nn)
        for i in range(nx):
            for j in range(nz):
                idx = i * nz + j
                self.x_nodes[idx] = i * self.dx
                self.y_nodes[idx] = j * self.dz

        # Build triangulation (each rectangle -> 2 triangles)
        self.triangles = []
        for i in range(nx - 1):
            for j in range(nz - 1):
                n0 = i * nz + j
                n1 = (i + 1) * nz + j
                n2 = (i + 1) * nz + (j + 1)
                n3 = i * nz + (j + 1)
                # Two triangles per rectangle
                self.triangles.append((n0, n1, n2))
                self.triangles.append((n0, n2, n3))

        self.ne = len(self.triangles)

        # Diffusivity sub-grid (2x2 regions)
        self.nxc = 2
        self.nyc = 2
        self.diffusivity_map = self._build_diffusivity_map()

        # Temperature field
        self.T = np.zeros(self.nn)

        # FEM matrices (assembled once if diffusivity is constant)
        self.K_global: Optional[np.ndarray] = None
        self.M_global: Optional[np.ndarray] = None

    def _build_diffusivity_map(self) -> np.ndarray:
        """
        Build piecewise-constant diffusivity on sub-grid.

        Light fragment occupies left half (z < lz/2)
        Heavy fragment occupies right half (z >= lz/2)
        """
        kappa = np.zeros((self.nxc, self.nyc))
        for i in range(self.nxc):
            for j in range(self.nyc):
                z_mid = (j + 0.5) * self.lz / self.nyc
                if z_mid < self.lz / 2.0:
                    kappa[i, j] = self.k_light
                else:
                    kappa[i, j] = self.k_heavy
        return kappa

    def get_diffusivity(self, x: float, y: float) -> float:
        """Get diffusivity at point (x, y) from sub-grid."""
        ix = int(x / (self.lr / self.nxc))
        iy = int(y / (self.lz / self.nyc))
        ix = max(0, min(self.nxc - 1, ix))
        iy = max(0, min(self.nyc - 1, iy))
        return self.diffusivity_map[ix, iy]

    def triangle_area(self, tri: Tuple[int, int, int]) -> float:
        """
        Compute area of triangle with given vertex indices.

        Area = 0.5 * |x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1)|
        """
        i0, i1, i2 = tri
        x0, y0 = self.x_nodes[i0], self.y_nodes[i0]
        x1, y1 = self.x_nodes[i1], self.y_nodes[i1]
        x2, y2 = self.x_nodes[i2], self.y_nodes[i2]

        area = 0.5 * abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        return area

    def reference_quadrature_7pt(self) -> Tuple[List[Tuple[float, float]],
                                                  List[float]]:
        """
        7-point quadrature rule on reference triangle (0,0)-(1,0)-(0,1).

        From 846_paraheat_functional: 7-point rule for T3 elements.
        """
        # Points and weights for 7-point rule (degree 4 exactness)
        pts = [
            (1.0 / 3.0, 1.0 / 3.0),
            (0.059715871789770, 0.470142064105115),
            (0.470142064105115, 0.059715871789770),
            (0.470142064105115, 0.470142064105115),
            (0.797426985353087, 0.101286507323456),
            (0.101286507323456, 0.797426985353087),
            (0.101286507323456, 0.101286507323456),
        ]
        wts = [
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
        return pts, wts

    def assemble_stiffness_matrix(self) -> np.ndarray:
        """
        Assemble global stiffness matrix K.

        K_ij = integral k(x,y) * (dphi_i/dx * dphi_j/dx +
                                    dphi_i/dy * dphi_j/dy) dA

        For T3 (linear triangle):
          dphi/dx and dphi/dy are constant within each element.

        Uses 7-point quadrature for piecewise-constant k.
        """
        K = np.zeros((self.nn, self.nn))
        pts, wts = self.reference_quadrature_7pt()

        for tri in self.triangles:
            i0, i1, i2 = tri
            x0, y0 = self.x_nodes[i0], self.y_nodes[i0]
            x1, y1 = self.x_nodes[i1], self.y_nodes[i1]
            x2, y2 = self.x_nodes[i2], self.y_nodes[i2]

            # Jacobian
            det_J = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(det_J) < 1e-14:
                continue

            # Shape function gradients (constant for T3)
            # dphi/dx, dphi/dy for each node
            inv_det = 1.0 / det_J
            dN_dx = [
                (y1 - y2) * inv_det,
                (y2 - y0) * inv_det,
                (y0 - y1) * inv_det,
            ]
            dN_dy = [
                (x2 - x1) * inv_det,
                (x0 - x2) * inv_det,
                (x1 - x0) * inv_det,
            ]

            # Quadrature for piecewise-constant k
            for qp, qw in zip(pts, wts):
                # Physical coordinates of quadrature point
                xi, eta = qp
                N0 = 1.0 - xi - eta
                N1 = xi
                N2 = eta
                x_qp = N0 * x0 + N1 * x1 + N2 * x2
                y_qp = N0 * y0 + N1 * y1 + N2 * y2

                kappa = self.get_diffusivity(x_qp, y_qp)

                # Element stiffness contribution
                for a in range(3):
                    for b in range(3):
                        k_ab = kappa * (dN_dx[a] * dN_dx[b] +
                                       dN_dy[a] * dN_dy[b])
                        K_global_idx_a = tri[a]
                        K_global_idx_b = tri[b]
                        K[K_global_idx_a, K_global_idx_b] += k_ab * qw * abs(det_J)

        self.K_global = K
        return K

    def assemble_mass_matrix(self) -> np.ndarray:
        """
        Assemble consistent mass matrix M.

        M_ij = integral rho*c * phi_i * phi_j dA

        For T3 element with area A:
          M_e = (A/12) * [2 1 1; 1 2 1; 1 1 2]
        """
        M = np.zeros((self.nn, self.nn))

        for tri in self.triangles:
            area = self.triangle_area(tri)
            i0, i1, i2 = tri

            # Element mass matrix
            m_e = (area / 12.0) * np.array([
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0]
            ])

            for a in range(3):
                for b in range(3):
                    M[tri[a], tri[b]] += m_e[a, b]

        self.M_global = M
        return M

    def initialize_temperature(self, t_light: float = 3.0,
                               t_heavy: float = 2.0) -> None:
        """
        Initialise temperature: hotter light fragment, cooler heavy fragment.
        """
        for i in range(self.nn):
            z = self.y_nodes[i]
            if z < self.lz / 2.0:
                self.T[i] = t_light
            else:
                self.T[i] = t_heavy

    def apply_boundary_conditions(self, T_boundary: float = 0.5) -> None:
        """
        Apply zero-flux (insulating) boundary conditions on all sides.
        Equivalent to: T at boundary nodes = T_boundary (cooling).
        """
        # Boundary nodes: i=0, i=nx-1, j=0, j=nz-1
        for i in range(self.nx):
            idx_bottom = i * self.nz
            idx_top = i * self.nz + (self.nz - 1)
            self.T[idx_bottom] = T_boundary
            self.T[idx_top] = T_boundary

        for j in range(self.nz):
            idx_left = j
            idx_right = (self.nx - 1) * self.nz + j
            self.T[idx_left] = T_boundary
            self.T[idx_right] = T_boundary

    def step_explicit(self, dt: float) -> None:
        """
        Explicit Euler time step: T^{n+1} = T^n + dt * M^{-1} * (-K * T^n)

        For lumped mass: M_lump = diag(sum of row of M)
        """
        if self.K_global is None:
            self.assemble_stiffness_matrix()
        if self.M_global is None:
            self.assemble_mass_matrix()

        # Lumped mass
        m_lump = np.sum(self.M_global, axis=1)
        m_lump = np.maximum(m_lump, 1e-14)

        # Heat equation: dT/dt = -M^{-1} K T
        dT = self.K_global @ self.T
        self.T = self.T - dt * dT / m_lump

        self.apply_boundary_conditions()

    def run(self, n_steps: int = 100, dt: float = 0.01) -> Dict[str, any]:
        """
        Run the 2D FEM thermal diffusion solver.
        """
        self.assemble_stiffness_matrix()
        self.assemble_mass_matrix()

        # CFL limit
        dx_min = min(self.dx, self.dz)
        k_max = max(self.k_light, self.k_heavy)
        dt_max = 0.25 * dx_min * dx_min / k_max
        dt_use = min(dt, 0.9 * dt_max)

        for step in range(n_steps):
            self.step_explicit(dt_use)

        return {
            'n_steps': n_steps,
            'dt_used': dt_use,
            'dt_max_cfl': dt_max,
            'T_min': float(np.min(self.T)),
            'T_max': float(np.max(self.T)),
            'T_mean': float(np.mean(self.T)),
            'T_light_region': float(np.mean([
                self.T[i * self.nz + j]
                for i in range(self.nx)
                for j in range(self.nz // 2)
            ])),
            'T_heavy_region': float(np.mean([
                self.T[i * self.nz + j]
                for i in range(self.nx)
                for j in range(self.nz // 2, self.nz)
            ])),
        }

    def sample_at_point(self, x: float, y: float) -> float:
        """
        Sample temperature at arbitrary point using element search
        + barycentric interpolation (from 846_paraheat_functional concept).
        """
        # Find containing triangle
        for tri in self.triangles:
            i0, i1, i2 = tri
            x0, y0 = self.x_nodes[i0], self.y_nodes[i0]
            x1, y1 = self.x_nodes[i1], self.y_nodes[i1]
            x2, y2 = self.x_nodes[i2], self.y_nodes[i2]

            # Barycentric coordinates
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-14:
                continue

            lam0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
            lam1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
            lam2 = 1.0 - lam0 - lam1

            if lam0 >= -0.01 and lam1 >= -0.01 and lam2 >= -0.01:
                # Point is in this triangle
                return (lam0 * self.T[i0] + lam1 * self.T[i1] + lam2 * self.T[i2])

        # If not found, return nearest node value
        min_dist = float('inf')
        nearest_val = 0.0
        for i in range(self.nn):
            dist = (self.x_nodes[i] - x) ** 2 + (self.y_nodes[i] - y) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_val = self.T[i]

        return nearest_val
