"""
hexagonal_lattice_geometry.py
==============================
Hexagonal Wigner-Seitz cell geometry and lattice operations for
the 2D dusty plasma crystal.

Physical motivation:
    Dusty plasma crystals in laboratory RF discharges form 2D
    hexagonal (triangular) lattices. The Wigner-Seitz cell is a
    regular hexagon. Key geometric operations include:

    1. Generating the hexagonal lattice coordinates
    2. Computing Voronoi cells (Wigner-Seitz cells)
    3. Circle packing (Dust grains occupy circular cross-sections)
    4. Computing nearest-neighbor shells and coordination numbers
    5. Arc length along grain trajectories (from 016_arclength)

    The lattice constant a is related to the dust density by:
        n_d = 2/(sqrt(3)*a^2) for hexagonal lattice
        or equivalently: pi*a_ws^2*n_d = 1

    Grain positions: R_{n1,n2} = n1*a1 + n2*a2
    where a1 = a*(1,0), a2 = a*(1/2, sqrt(3)/2)

References:
    - From 185_circles: circular cross-section representation
    - From 016_arclength: arc length integration for trajectories
"""

import numpy as np
from typing import Tuple, List, Optional


class HexagonalLattice:
    """
    2D hexagonal lattice for dusty plasma crystal.

    Attributes
    ----------
    a : float
        Lattice constant (nearest-neighbor distance).
    a_ws : float
        Wigner-Seitz radius (pi*a_ws^2*n_d = 1).
    n_d : float
        Dust number density [m^-2] (2D).
    lattice_vectors : np.ndarray, shape (2, 2)
        Primitive lattice vectors [a1; a2] as rows.
    n_cells : int
        Number of unit cells.
    """

    def __init__(
        self,
        a: float = 1.0e-3,
        n_cells_x: int = 5,
        n_cells_y: int = 5,
    ):
        """
        Parameters
        ----------
        a : float
            Lattice constant [m].
        n_cells_x, n_cells_y : int
            Number of unit cells in each direction.
        """
        self.a = a
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y

        # Lattice vectors
        self.a1 = np.array([a, 0.0])
        self.a2 = np.array([a * 0.5, a * np.sqrt(3.0) / 2.0])
        self.lattice_vectors = np.array([self.a1, self.a2])

        # Derived quantities
        self.area_cell = abs(np.cross(self.a1, self.a2))  # = a^2*sqrt(3)/2
        self.n_d = 1.0 / self.area_cell  # 2D number density
        self.a_ws = 1.0 / np.sqrt(np.pi * self.n_d)  # Wigner-Seitz radius

        # Generate lattice sites
        self.positions = self._generate_positions()
        self.n_grains = len(self.positions)

        # Neighbor lists
        self.neighbor_shells = self._compute_neighbor_shells(max_shells=3)

    def _generate_positions(self) -> np.ndarray:
        """
        Generate dust grain positions on the hexagonal lattice.

        Returns
        -------
        positions : np.ndarray, shape (N, 2)
            Grain coordinates [m].
        """
        positions = []
        for n1 in range(self.n_cells_x):
            for n2 in range(self.n_cells_y):
                pos = n1 * self.a1 + n2 * self.a2
                positions.append(pos)
        return np.array(positions)

    def _compute_neighbor_shells(self, max_shells: int = 3) -> dict:
        """
        Compute neighbor shells for each grain.

        For hexagonal lattice:
            Shell 1: 6 neighbors at distance a
            Shell 2: 6 neighbors at distance a*sqrt(3)
            Shell 3: 6 neighbors at distance 2a
            Shell 4: 12 neighbors at distance a*sqrt(7)
            ...

        Parameters
        ----------
        max_shells : int
            Maximum number of shells to compute.

        Returns
        -------
        shells : dict
            shells[shell_num] = list of (grain_idx, neighbor_idx, distance)
        """
        shells = {i: [] for i in range(1, max_shells + 1)}

        for i in range(self.n_grains):
            dists = []
            for j in range(self.n_grains):
                if i == j:
                    continue
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                dists.append((j, d))
            dists.sort(key=lambda x: x[1])

            # Group into shells by distance
            shell_boundaries = [0.0]
            expected_distances = []
            for s in range(1, max_shells + 2):
                # Hexagonal lattice shell distances
                d_s = self.a * np.sqrt(
                    sum(1 for _ in range(s))  # Placeholder
                )
                expected_distances.append(d_s)

            # Simple grouping: group by distance tolerance
            current_shell = 0
            prev_dist = 0.0
            for j, d in dists:
                if current_shell >= max_shells:
                    break
                if abs(d - prev_dist) > 0.1 * self.a and prev_dist > 0:
                    current_shell += 1
                if current_shell < max_shells:
                    shells[current_shell + 1].append((i, j, d))
                    prev_dist = d

        return shells

    def wigner_seitz_cell(self, grain_idx: int) -> np.ndarray:
        """
        Compute the Wigner-Seitz cell (Voronoi cell) for a grain.

        For a perfect hexagonal lattice, the WS cell is a regular
        hexagon with vertices at distance a/sqrt(3) from the grain.

        Parameters
        ----------
        grain_idx : int
            Index of the grain.

        Returns
        -------
        vertices : np.ndarray, shape (6, 2)
            Vertices of the hexagonal WS cell.
        """
        center = self.positions[grain_idx]
        r_ws = self.a / np.sqrt(3.0)  # Inradius of hexagon

        angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
        vertices = np.array([
            center + r_ws * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ])
        return vertices

    def grain_cross_section_area(self, r_d: float) -> float:
        """
        Cross-sectional area of a dust grain (circle, from 185_circles).

        A_d = pi * r_d^2

        Parameters
        ----------
        r_d : float
            Grain radius [m].

        Returns
        -------
        area : float
            Cross-sectional area [m^2].
        """
        return np.pi * r_d**2

    def packing_fraction(self, r_d: float) -> float:
        """
        2D packing fraction of dust grains.

        eta = n_d * pi * r_d^2

        Parameters
        ----------
        r_d : float
            Grain radius [m].

        Returns
        -------
        eta : float
            Packing fraction (dimensionless).
        """
        return self.n_d * self.grain_cross_section_area(r_d)

    def fill_factor_check(self, r_d: float) -> bool:
        """
        Check that grains don't overlap.

        For hexagonal lattice, max packing fraction is pi/(2*sqrt(3)) ~ 0.9069.
        Grains don't overlap if 2*r_d < a (nearest-neighbor distance).

        Parameters
        ----------
        r_d : float
            Grain radius [m].

        Returns
        -------
        valid : bool
            True if grains don't overlap.
        """
        return 2.0 * r_d < self.a

    def get_reciprocal_lattice(self) -> np.ndarray:
        """
        Compute reciprocal lattice vectors.

        b1, b2 such that a_i . b_j = 2*pi*delta_{ij}

        For 2D hexagonal:
            b1 = (2*pi/a) * (1, -1/sqrt(3))
            b2 = (2*pi/a) * (0, 2/sqrt(3))

        Returns
        -------
        reciprocal_vectors : np.ndarray, shape (2, 2)
            [b1; b2] as rows.
        """
        det = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
        b1 = 2 * np.pi * np.array([self.a2[1], -self.a2[0]]) / det
        b2 = 2 * np.pi * np.array([-self.a1[1], self.a1[0]]) / det
        return np.array([b1, b2])

    def debye_waller_factor(
        self,
        kappa: float,
        T_d: float,
        m_d: float,
        G: np.ndarray,
    ) -> float:
        """
        Compute the Debye-Waller factor for Bragg scattering.

        The Debye-Waller factor accounts for thermal fluctuations
        of grain positions:
            f_DW(G) = exp(-<u^2>*|G|^2/2)
        where <u^2> = k_B*T_d/(m_d*omega_E^2) is the mean-square
        displacement and G is a reciprocal lattice vector.

        Parameters
        ----------
        kappa : float
            Screening parameter.
        T_d : float
            Dust temperature [J].
        m_d : float
            Dust grain mass [kg].
        G : np.ndarray, shape (2,)
            Reciprocal lattice vector.

        Returns
        -------
        f_dw : float
            Debye-Waller factor.
        """
        from dusty_plasma_physics_constants import BOLTZMANN_CONSTANT
        G_mag = np.linalg.norm(G)

        # Einstein frequency (approximate)
        # omega_E^2 ~ (Q_d^2/(m_d*a^3)) * (1+kappa)*exp(-kappa)
        # For now, use a simplified estimate
        omega_E_sq = 1.0  # normalized
        u_sq = T_d / (m_d * omega_E_sq) if m_d * omega_E_sq > 0 else 0.0

        return np.exp(-0.5 * u_sq * G_mag**2)


def compute_arc_length(
    trajectory: np.ndarray,
    n_quad: int = 100,
) -> float:
    """
    Compute arc length along a grain trajectory (from 016_arclength).

    For a parametric curve r(t) = (x(t), y(t)):
        s = integral |dr/dt| dt = integral sqrt((dx/dt)^2 + (dy/dt)^2) dt

    We use the composite trapezoidal rule.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 2)
        Sequence of positions [x, y] at equal time intervals.
    n_quad : int
        Number of quadrature subintervals per segment.

    Returns
    -------
    total_length : float
        Total arc length.
    """
    N = len(trajectory)
    if N < 2:
        return 0.0

    total = 0.0
    for k in range(N - 1):
        p1 = trajectory[k]
        p2 = trajectory[k + 1]
        # Simple chord length as approximation
        total += np.linalg.norm(p2 - p1)

    return total


def compute_arc_length_spectral(
    x_func,
    y_func,
    dx_func,
    dy_func,
    t_start: float,
    t_end: float,
    n_points: int = 200,
) -> float:
    """
    Compute arc length using spectral derivatives (from 016_arclength).

    s = integral_{t_start}^{t_end} sqrt((dx/dt)^2 + (dy/dt)^2) dt

    Parameters
    ----------
    x_func, y_func : callable
        Parametric curve x(t), y(t).
    dx_func, dy_func : callable
        Derivatives dx/dt, dy/dt.
    t_start, t_end : float
        Integration bounds.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    s : float
        Arc length.
    """
    t = np.linspace(t_start, t_end, n_points)
    dt = t[1] - t[0]

    integrand = np.zeros(n_points)
    for i in range(n_points):
        dxdt = dx_func(t[i])
        dydt = dy_func(t[i])
        integrand[i] = np.sqrt(dxdt**2 + dydt**2)

    # Composite trapezoidal rule
    s = dt * (np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1]))
    return s
