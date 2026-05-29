"""
reactor_geometry.py
===================
Cylindrical reactor geometry, coordinate transformations, and 3D mesh handling.

Incorporates algorithms from:
  - 180_circle_map (matrix mapping of unit circle, norm computations)
  - 873_ply_io (3D mesh data structure parsing)

Scientific role:
  Defines the spatial domain of a downdraft biomass gasification reactor.
  Provides coordinate transforms for cylindrical geometry and handles
  vertex/face connectivity for packed-bed particle representation.
"""

import math
import numpy as np


class CylindricalReactor:
    """
    Downdraft biomass gasification reactor geometry.

    Dimensions:
        H : total reactor height [m]
        R : internal radius [m]
        H_bed : packed bed height [m]
        H_freeboard : freeboard height [m]
        H_reduction : reduction zone height [m]
        H_combustion : combustion zone height [m]

    The reactor is partitioned into three zones along the axial (z) direction:
        1. Drying/Pyrolysis (z ∈ [0, H_bed])
        2. Combustion      (z ∈ [H_bed, H_bed + H_combustion])
        3. Reduction       (z ∈ [H_bed + H_combustion, H])
    """

    def __init__(self, H=2.5, R=0.3, H_bed=1.0,
                 H_combustion=0.5, H_reduction=1.0):
        self.H = float(H)
        self.R = float(R)
        self.H_bed = float(H_bed)
        self.H_combustion = float(H_combustion)
        self.H_reduction = float(H_reduction)
        # Validate dimensions
        total = self.H_bed + self.H_combustion + self.H_reduction
        if abs(total - self.H) > 1.0e-6:
            # Auto-adjust to maintain consistency
            scale = self.H / total if total > 0 else 1.0
            self.H_bed *= scale
            self.H_combustion *= scale
            self.H_reduction *= scale

    def volume(self):
        """Total reactor volume V = π R² H."""
        return math.pi * self.R ** 2 * self.H

    def cross_section_area(self):
        """Cross-sectional area A = π R²."""
        return math.pi * self.R ** 2

    def zone_volume(self, zone):
        """Volume of a specific zone."""
        a = self.cross_section_area()
        if zone == 'bed':
            return a * self.H_bed
        elif zone == 'combustion':
            return a * self.H_combustion
        elif zone == 'reduction':
            return a * self.H_reduction
        return 0.0

    def zone_for_z(self, z):
        """Return the zone name for a given axial coordinate."""
        if z < 0.0 or z > self.H:
            return 'outside'
        if z <= self.H_bed:
            return 'bed'
        elif z <= self.H_bed + self.H_combustion:
            return 'combustion'
        else:
            return 'reduction'

    def cylindrical_to_cartesian(self, r, theta, z):
        """
        Convert cylindrical coordinates to Cartesian.
        x = r cos θ, y = r sin θ, z = z
        """
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return x, y, z

    def cartesian_to_cylindrical(self, x, y, z):
        """
        Convert Cartesian to cylindrical coordinates.
        r = √(x² + y²), θ = atan2(y, x)
        """
        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        return r, theta, z

    def map_circle_transform(self, A, norm_type=2, num_points=75):
        """
        Map unit circle points through matrix A and return transformed points.

        For a 2x2 matrix A, the image of the unit circle under A is an ellipse
        with semi-axes equal to the singular values of A. The aspect ratio
        σ_max / σ_min is the condition number cond(A).

        This is applied to map velocity perturbation ellipses in the r-θ plane
        at a fixed axial location.

        Parameters
        ----------
        A : ndarray, shape (2, 2)
            Transformation matrix.
        norm_type : int
            Norm used to define the unit circle: 1, 2, or np.inf.
        num_points : int
            Number of sample points.

        Returns
        -------
        points : ndarray, shape (num_points, 2)
            Transformed points.
        condition_number : float
            Aspect ratio of the mapped ellipse.
        """
        A = np.asarray(A, dtype=float)
        if A.shape != (2, 2):
            raise ValueError("Matrix A must be 2x2")

        points = np.zeros((num_points, 2), dtype=float)
        # SVD for condition number
        try:
            u, s, vh = np.linalg.svd(A)
            cond_num = s[0] / s[1] if s[1] > 1.0e-15 else 1.0e15
        except np.linalg.LinAlgError:
            cond_num = 1.0e15

        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points
            if norm_type == 1:
                # L1 unit circle: diamond
                if abs(math.cos(angle)) >= abs(math.sin(angle)):
                    x_unit = math.copysign(1.0, math.cos(angle)) * (1.0 - abs(math.tan(angle)))
                    y_unit = math.sin(angle) / max(abs(math.cos(angle)), 1.0e-15)
                else:
                    y_unit = math.copysign(1.0, math.sin(angle)) * (1.0 - abs(1.0 / math.tan(angle)))
                    x_unit = math.cos(angle) / max(abs(math.sin(angle)), 1.0e-15)
                norm_val = abs(x_unit) + abs(y_unit)
                if norm_val > 1.0e-15:
                    x_unit /= norm_val
                    y_unit /= norm_val
            elif norm_type == np.inf:
                # L∞ unit circle: square
                x_unit = math.cos(angle)
                y_unit = math.sin(angle)
                maxc = max(abs(x_unit), abs(y_unit))
                if maxc > 1.0e-15:
                    x_unit /= maxc
                    y_unit /= maxc
            else:
                # L2 unit circle
                x_unit = math.cos(angle)
                y_unit = math.sin(angle)

            vec = np.array([x_unit, y_unit], dtype=float)
            mapped = A.dot(vec)
            points[i, :] = mapped

        return points, cond_num


class Mesh3D:
    """
    Simple 3D mesh data structure for packed-bed particle representation.
    Analogous to PLY vertex/face structure but simplified for reactor bed.
    """

    def __init__(self):
        self.vertices = []   # list of (x, y, z)
        self.faces = []      # list of vertex index triples

    def add_vertex(self, x, y, z):
        """Add a vertex and return its index."""
        self.vertices.append((float(x), float(y), float(z)))
        return len(self.vertices) - 1

    def add_face(self, i, j, k):
        """Add a triangular face."""
        n = len(self.vertices)
        if 0 <= i < n and 0 <= j < n and 0 <= k < n:
            self.faces.append((int(i), int(j), int(k)))
            return True
        return False

    def face_area(self, face_idx):
        """Compute area of a triangular face via cross product."""
        if face_idx < 0 or face_idx >= len(self.faces):
            return 0.0
        i, j, k = self.faces[face_idx]
        v1 = np.array(self.vertices[j]) - np.array(self.vertices[i])
        v2 = np.array(self.vertices[k]) - np.array(self.vertices[i])
        cp = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cp)

    def total_surface_area(self):
        """Sum of all face areas."""
        return sum(self.face_area(idx) for idx in range(len(self.faces)))

    def bounding_box(self):
        """Return axis-aligned bounding box (min_coords, max_coords)."""
        if not self.vertices:
            return np.zeros(3), np.zeros(3)
        verts = np.array(self.vertices)
        return verts.min(axis=0), verts.max(axis=0)

    def sample_on_surface(self, num_samples):
        """
        Sample points uniformly on the mesh surface using face-area weighting.
        Used for radiation view factor Monte Carlo sampling.
        """
        if not self.faces:
            return np.zeros((num_samples, 3))
        areas = np.array([self.face_area(i) for i in range(len(self.faces))])
        total_area = areas.sum()
        if total_area <= 1.0e-15:
            return np.zeros((num_samples, 3))
        probs = areas / total_area
        samples = np.zeros((num_samples, 3), dtype=float)
        for s in range(num_samples):
            fidx = np.random.choice(len(self.faces), p=probs)
            i, j, k = self.faces[fidx]
            vi = np.array(self.vertices[i])
            vj = np.array(self.vertices[j])
            vk = np.array(self.vertices[k])
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            samples[s, :] = vi + r1 * (vj - vi) + r2 * (vk - vi)
        return samples
