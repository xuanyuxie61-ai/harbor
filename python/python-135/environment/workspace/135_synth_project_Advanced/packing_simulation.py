"""
Packing Geometry, Mesh Generation, and Spatial Sampling
Integrates concepts from:
- fem_to_xml (finite element mesh conversion)
- line_grid (1D grid generation)
- hilbert_curve_3d (3D space-filling curve for spatial traversal)

Applications:
- Structured packing geometry representation
- Pore space traversal for effective diffusivity calculation
- Finite element preprocessing for flow simulations
"""

import numpy as np
from utils import validate_positive


class StructuredPackingGeometry:
    """
    Model structured packing geometry (e.g., Mellapak) as a series of
    corrugated sheets with triangular cross-section channels.
    """

    def __init__(self, corrugation_angle=45.0, crimp_height=0.01, channel_width=0.008):
        self.theta = np.radians(corrugation_angle)
        self.h = crimp_height
        self.b = channel_width
        self.S = self.b / np.cos(self.theta)  # Corrugation side length

    def specific_area(self):
        """Specific surface area [m^2/m^3]."""
        return 4.0 * np.sin(self.theta) / (self.b * self.h)

    def hydraulic_diameter(self):
        """Hydraulic diameter of triangular channel [m]."""
        # Cross-sectional area / wetted perimeter
        A_cs = 0.5 * self.b * self.h
        P_wet = self.b + 2.0 * self.S
        return 4.0 * A_cs / P_wet

    def void_fraction(self):
        """Void fraction of packing."""
        return 1.0 - 0.5 * self.h / self.S

    def effective_diffusivity(self, D_bulk, tortuosity=1.5):
        """
        Effective diffusivity in packing using tortuosity model:
            D_eff = D_bulk * eps / tau
        """
        eps = self.void_fraction()
        return D_bulk * eps / tortuosity


class MeshGenerator:
    """
    Simple mesh generation for finite element preprocessing.
    Based on fem_to_xml.m concept but simplified for internal use.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.nodes = []
        self.elements = []

    def generate_rectangular_mesh(self, nx, ny, xlim=(0, 1), ylim=(0, 1)):
        """Generate structured triangular mesh for a rectangle."""
        validate_positive(nx, "nx")
        validate_positive(ny, "ny")

        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        dx = (xlim[1] - xlim[0]) / (nx - 1)
        dy = (ylim[1] - ylim[0]) / (ny - 1)

        nodes = []
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j]])
        self.nodes = np.array(nodes)

        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n0 + nx
                n3 = n2 + 1
                # Two triangles per rectangle
                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])
        self.elements = np.array(elements, dtype=int)

        return self.nodes, self.elements

    def generate_1d_line_mesh(self, n, xlim=(0, 1)):
        """Generate 1D line mesh with linear elements."""
        x = np.linspace(xlim[0], xlim[1], n)
        nodes = np.column_stack([x, np.zeros(n)])
        elements = np.array([[i, i + 1] for i in range(n - 1)], dtype=int)
        return nodes, elements

    def mesh_quality_metrics(self):
        """Compute mesh quality metrics (aspect ratio, skewness)."""
        if len(self.elements) == 0 or self.dim != 2:
            return {}

        aspect_ratios = []
        for elem in self.elements:
            p0, p1, p2 = self.nodes[elem]
            # Edge lengths
            e0 = np.linalg.norm(p1 - p0)
            e1 = np.linalg.norm(p2 - p1)
            e2 = np.linalg.norm(p0 - p2)
            edges = sorted([e0, e1, e2])
            if edges[0] > 1e-12:
                aspect_ratios.append(edges[-1] / edges[0])

        return {
            "max_aspect_ratio": max(aspect_ratios) if aspect_ratios else 0.0,
            "mean_aspect_ratio": np.mean(aspect_ratios) if aspect_ratios else 0.0,
            "num_elements": len(self.elements),
            "num_nodes": len(self.nodes)
        }

    def export_to_simple_format(self):
        """Export mesh to simple dictionary format."""
        return {
            "dimension": self.dim,
            "nodes": self.nodes.tolist(),
            "elements": self.elements.tolist()
        }


def hilbert_curve_3d(h, r):
    """
    Convert a linear Hilbert coordinate to (x, y, z) in 3D.
    Based on h_to_xyz.m (Burkardt).
    Space-filling curve for ordered traversal of 3D packing pore space.
    """
    h = int(h)
    r = int(r)

    o = h % 8
    coords_map = {
        0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 0, 1), 3: (0, 0, 1),
        4: (0, 1, 1), 5: (1, 1, 1), 6: (1, 1, 0), 7: (0, 1, 0)
    }
    x, y, z = coords_map[o]

    w = 2
    h = h // 8

    while h > 0:
        o = h % 8
        xold, yold, zold = x, y, z

        transforms = {
            0: (yold, zold, xold),
            1: (zold + w, xold, yold),
            2: (zold + w, xold, yold + w),
            3: (w - xold - 1, yold, 2 * w - zold - 1),
            4: (w - xold - 1, yold + w, 2 * w - zold - 1),
            5: (zold + w, 2 * w - xold - 1, 2 * w - yold - 1),
            6: (zold + w, 2 * w - xold - 1, w - yold - 1),
            7: (w - yold - 1, 2 * w - zold - 1, xold)
        }
        x, y, z = transforms[o]
        h = h // 8
        w = w * w

    # Rotation correction
    def rmin_val(x, y, z):
        if x == 0 and y == 0 and z == 0:
            return 0
        rm = 0
        while ((x >> rm) & 1) == 0 and ((y >> rm) & 1) == 0 and ((z >> rm) & 1) == 0:
            rm += 1
        return rm

    rm = rmin_val(x, y, z)
    t = (r - rm) % 3
    if t == 1:
        x, y, z = y, z, x
    elif t == 2:
        x, y, z = z, x, y

    return x, y, z


def generate_hilbert_pore_sample(n_order, domain_size=1.0):
    """
    Generate ordered sample points through 3D pore space using Hilbert curve.
    Returns array of (x, y, z) coordinates.
    """
    N = 2 ** (3 * n_order)
    points = np.zeros((N, 3))
    max_coord = 2 ** n_order - 1
    scale = domain_size / max_coord if max_coord > 0 else 1.0

    for h in range(N):
        x, y, z = hilbert_curve_3d(h, n_order)
        points[h, :] = [x * scale, y * scale, z * scale]

    return points


def effective_property_hilbert_sampling(property_func, n_order=3):
    """
    Compute effective property by averaging along Hilbert curve traversal.
    Ensures spatial locality in sampling for correlated porous media.
    """
    points = generate_hilbert_pore_sample(n_order)
    values = np.array([property_func(p[0], p[1], p[2]) for p in points])
    return np.mean(values), np.std(values)


class PoreNetworkModel:
    """
    Simplified pore network model for effective transport properties
    in structured packing.
    """

    def __init__(self, num_pores, porosity, throat_radius_mean, throat_radius_std):
        self.N = num_pores
        self.phi = porosity
        self.r_mean = throat_radius_mean
        self.r_std = throat_radius_std
        self.throat_radii = np.maximum(
            np.random.normal(throat_radius_mean, throat_radius_std, num_pores),
            throat_radius_mean * 0.1
        )

    def permeability_kozeny_carman(self, particle_diameter):
        """
        Kozeny-Carman permeability:
            k = d_p^2 * phi^3 / (180 * (1-phi)^2)
        """
        validate_positive(particle_diameter, "particle_diameter")
        return particle_diameter ** 2 * self.phi ** 3 / (180.0 * (1.0 - self.phi) ** 2)

    def effective_diffusivity_from_pores(self, D_bulk):
        """
        Effective diffusivity using pore network theory:
            D_eff = D_bulk * phi * (1 - lambda)^2
        where lambda = molecular diameter / pore diameter.
        """
        lambda_ratio = 0.1  # simplified
        return D_bulk * self.phi * (1.0 - lambda_ratio) ** 2

    def throat_conductance(self, viscosity):
        """
        Hydraulic conductance of cylindrical throat:
            g = pi * r^4 / (8 * mu * L)
        """
        L = self.r_mean * 10.0  # throat length
        return np.pi * self.throat_radii ** 4 / (8.0 * viscosity * L)
