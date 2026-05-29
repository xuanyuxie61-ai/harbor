"""
particle_geometry.py
====================
Geometric modeling of pulverized coal particles.

Incorporates four seed-project ideas:
1. tortoise_grid_word (1282): boundary-word encoding for discrete direction
   sequences describing particle contour.
2. bird_egg (093): universal egg-shape formula for parametric particle shape.
3. bezier_surface (083): bicubic Bezier patches for smooth surface parameterization.
4. stla_to_tri_surface (1167): surface triangulation and area computation.

Scientific goal: model realistic non-spherical coal particles to compute
surface area, volume, and shape factors affecting char burnout and NOx
precursor (HCN, NH3) release rates.

Key formulas:
  - Universal egg shape (Narushin et al.):
        y(x) = (B/2) * sqrt( (L^2 - 4x^2) / (L^2 + 8wx + 4w^2) )
    where B=maximum width, L=length, w=offset of max width from center.

  - Bicubic Bezier surface patch:
        X(u,v) = sum_{i=0}^3 sum_{j=0}^3 B_{i,3}(u) B_{j,3}(v) P_{ij}
    with Bernstein polynomials:
        B_{k,3}(t) = C(3,k) t^k (1-t)^{3-k}

  - Surface area via triangulation:
        A = sum_{triangles} (1/2) || (v2-v1) x (v3-v1) ||

  - Sphericity:
        Psi = A_sphere(V) / A_particle  (<= 1)
"""

import numpy as np
from typing import List, Tuple


# ======================================================================
# 1. Universal egg-shape parametric profile (from bird_egg)
# ======================================================================

def egg_profile_chicken(x: np.ndarray, L: float, B: float, w: float) -> np.ndarray:
    """
    Chicken egg profile: y(x) for x in [-L/2, L/2].
    Formula:
        y(x) = (B/2) * sqrt( (L^2 - 4x^2) / (L^2 + 8wx + 4w^2) )
    When L=B and w=0, the shape is an ellipse.
    When w=0, the shape is symmetric.
    """
    if L <= 0 or B <= 0:
        return np.zeros_like(x)
    x = np.clip(x, -L / 2.0, L / 2.0)
    denom = L * L + 8.0 * w * x + 4.0 * w * w
    denom = np.where(denom > 0, denom, 1e-300)
    numer = L * L - 4.0 * x * x
    numer = np.where(numer > 0, numer, 0.0)
    y = 0.5 * B * np.sqrt(numer / denom)
    return y


def egg_profile_pyriform(x: np.ndarray, L: float, B: float, w: float, D: float = None) -> np.ndarray:
    """
    Pear-shaped (pyriform) egg profile with optional fourth parameter D
    (diameter at 1/4 length from tip).
    """
    if L <= 0 or B <= 0:
        return np.zeros_like(x)
    x = np.clip(x, -L / 2.0, L / 2.0)
    # Modified denominator for pear shape
    denom = L * L + 8.0 * w * x + 4.0 * w * w + 2.0 * (x + L / 2.0) ** 2
    denom = np.where(denom > 0, denom, 1e-300)
    numer = L * L - 4.0 * x * x
    numer = np.where(numer > 0, numer, 0.0)
    y = 0.5 * B * np.sqrt(numer / denom)
    if D is not None and D > 0:
        # Correction factor at x = -L/4
        xq = -L / 4.0
        denom_q = L * L + 8.0 * w * xq + 4.0 * w * w + 2.0 * (xq + L / 2.0) ** 2
        y_q = 0.5 * B * np.sqrt(max(L * L - 4.0 * xq * xq, 0.0) / max(denom_q, 1e-300))
        if y_q > 1e-12:
            factor = (D / 2.0) / y_q
            y = y * factor
    return y


# ======================================================================
# 2. Bicubic Bezier surface patch (from bezier_surface)
# ======================================================================

def bernstein_basis_3(t: float) -> np.ndarray:
    """
    Cubic Bernstein basis vector:
        B(t) = [(1-t)^3, 3t(1-t)^2, 3t^2(1-t), t^3]
    """
    t = np.clip(t, 0.0, 1.0)
    u = 1.0 - t
    return np.array([u * u * u,
                     3.0 * t * u * u,
                     3.0 * t * t * u,
                     t * t * t])


def bezier_patch_evaluate(P: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    Evaluate a bicubic Bezier patch at parameters (u,v).
    P is a 4x4x3 array of control points.
    Returns point [x,y,z] in R^3.
    """
    Bu = bernstein_basis_3(u)
    Bv = bernstein_basis_3(v)
    # X = Bu^T * P_x * Bv, etc.
    xyz = np.zeros(3)
    for dim in range(3):
        xyz[dim] = Bu @ P[:, :, dim] @ Bv
    return xyz


def generate_ellipsoid_bezier_control_points(a: float, b: float, c: float) -> np.ndarray:
    """
    Generate a 4x4x3 Bezier control point net approximating an ellipsoid
    x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 in the first octant.
    """
    P = np.zeros((4, 4, 3))
    for i in range(4):
        u = i / 3.0
        theta_u = 0.5 * np.pi * u
        for j in range(4):
            v = j / 3.0
            phi_v = 0.5 * np.pi * v
            x = a * np.sin(theta_u) * np.cos(phi_v)
            y = b * np.sin(theta_u) * np.sin(phi_v)
            z = c * np.cos(theta_u)
            P[i, j, 0] = max(x, 0.0)
            P[i, j, 1] = max(y, 0.0)
            P[i, j, 2] = max(z, 0.0)
    return P


# ======================================================================
# 3. Surface triangulation and area (from stla_to_tri_surface)
# ======================================================================

def triangulate_revolution_profile(
    x_profile: np.ndarray, y_profile: np.ndarray, n_theta: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Revolve a 2D profile (x, y>=0) around the x-axis to create a triangulated
    surface. Returns (vertices, faces).
    
    Vertices: N x 3 array
    Faces:    M x 3 array (0-based vertex indices)
    """
    if len(x_profile) < 2 or len(y_profile) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    
    nx = len(x_profile)
    n_theta = max(n_theta, 3)
    
    # Generate vertices
    vertices = []
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    for i in range(nx):
        for j in range(n_theta):
            xv = x_profile[i]
            yv = y_profile[i] * np.cos(theta[j])
            zv = y_profile[i] * np.sin(theta[j])
            vertices.append([xv, yv, zv])
    vertices = np.array(vertices)
    
    # Generate triangular faces
    faces = []
    for i in range(nx - 1):
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            v0 = i * n_theta + j
            v1 = i * n_theta + j1
            v2 = (i + 1) * n_theta + j
            v3 = (i + 1) * n_theta + j1
            # Two triangles per quad
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    faces = np.array(faces, dtype=int)
    return vertices, faces


def compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute total surface area from triangulated mesh:
        A = sum_k (0.5 * || (v2-v1) x (v3-v1) || )
    """
    if len(faces) == 0:
        return 0.0
    area = 0.0
    for f in faces:
        v1 = vertices[f[0]]
        v2 = vertices[f[1]]
        v3 = vertices[f[2]]
        cross = np.cross(v2 - v1, v3 - v1)
        area += 0.5 * np.linalg.norm(cross)
    return area


def compute_volume_revolution(x: np.ndarray, y: np.ndarray) -> float:
    """
    Volume of solid of revolution using Simpson's rule:
        V = pi * integral_{x_min}^{x_max} y(x)^2 dx
    """
    if len(x) < 2:
        return 0.0
    y_sq = y * y
    # Simpson integration
    n = len(x)
    if n % 2 == 0:
        # Use trapezoidal for last interval if even number of points
        h = x[1] - x[0]
        integral = (h / 3.0) * (y_sq[0] + 4.0 * np.sum(y_sq[1:n - 1:2]) +
                                2.0 * np.sum(y_sq[2:n - 2:2]) + y_sq[n - 2])
        integral += 0.5 * (x[-1] - x[-2]) * (y_sq[-1] + y_sq[-2])
    else:
        h = x[1] - x[0]
        integral = (h / 3.0) * (y_sq[0] + 4.0 * np.sum(y_sq[1:n:2]) +
                                2.0 * np.sum(y_sq[2:n - 1:2]) + y_sq[n - 1])
    return np.pi * max(integral, 0.0)


# ======================================================================
# 4. Boundary-word encoding (from tortoise_grid_word)
# ======================================================================

def boundary_word_encode(directions: List[int], steps: List[float]) -> str:
    """
    Encode a 2D boundary walk as a word over an alphabet of directions.
    Directions 0-7 correspond to angles 0, 45, 90, ..., 315 degrees.
    This discrete encoding is used to compare and classify particle shapes.
    """
    alphabet = "ABCDEFGH"
    word = ""
    for d, s in zip(directions, steps):
        d = int(d) % 8
        # Encode step magnitude into letter case / repetition
        repeats = max(1, int(round(s)))
        word += alphabet[d] * repeats
    return word


def boundary_word_discrete_profile(x: np.ndarray, y: np.ndarray, n_dirs: int = 8) -> str:
    """
    Convert a continuous profile to a discrete boundary word by sampling
    edge directions between consecutive points.
    """
    if len(x) < 2:
        return ""
    directions = []
    steps = []
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        angle = np.arctan2(dy, dx)
        # Map to n_dirs sectors
        sector = int(np.round((angle % (2 * np.pi)) / (2 * np.pi / n_dirs))) % n_dirs
        dist = np.sqrt(dx * dx + dy * dy)
        directions.append(sector)
        steps.append(dist)
    return boundary_word_encode(directions, steps)


# ======================================================================
# 5. Coal particle class integrating all geometric models
# ======================================================================

class CoalParticle:
    """
    Represents a single pulverized coal particle with realistic geometry.
    """
    
    def __init__(self, L: float = 100e-6, B: float = 70e-6,
                 w: float = 5e-6, shape_type: str = "egg"):
        """
        Args:
            L: major axis length [m]
            B: maximum width [m]
            w: offset of maximum width from center [m]
            shape_type: 'egg', 'pyriform', 'bezier'
        """
        self.L = max(L, 1e-9)
        self.B = max(B, 1e-9)
        self.w = w
        self.shape_type = shape_type
        self._profile_x = None
        self._profile_y = None
        self._vertices = None
        self._faces = None
        self._surface_area = None
        self._volume = None
        self._build_geometry()
    
    def _build_geometry(self):
        nx = 128
        x = np.linspace(-self.L / 2.0, self.L / 2.0, nx)
        if self.shape_type == "egg":
            y = egg_profile_chicken(x, self.L, self.B, self.w)
        elif self.shape_type == "pyriform":
            y = egg_profile_pyriform(x, self.L, self.B, self.w)
        else:
            y = egg_profile_chicken(x, self.L, self.B, self.w)
        
        self._profile_x = x
        self._profile_y = np.maximum(y, 0.0)
        self._vertices, self._faces = triangulate_revolution_profile(
            self._profile_x, self._profile_y, n_theta=64
        )
        self._surface_area = compute_surface_area(self._vertices, self._faces)
        self._volume = compute_volume_revolution(self._profile_x, self._profile_y)
    
    @property
    def surface_area(self) -> float:
        return self._surface_area
    
    @property
    def volume(self) -> float:
        return self._volume
    
    @property
    def equivalent_diameter(self) -> float:
        """Diameter of sphere with same volume."""
        return (6.0 * self._volume / np.pi) ** (1.0 / 3.0)
    
    @property
    def sphericity(self) -> float:
        """
        Sphericity Psi = A_sphere(V) / A_particle.
        Perfect sphere -> Psi = 1.0.
        """
        d_eq = self.equivalent_diameter
        A_sphere = np.pi * d_eq * d_eq
        if self._surface_area < 1e-30:
            return 1.0
        return min(A_sphere / self._surface_area, 1.0)
    
    @property
    def surface_to_volume_ratio(self) -> float:
        if self._volume < 1e-30:
            return 0.0
        return self._surface_area / self._volume
    
    def shape_descriptor(self) -> dict:
        """Return geometric shape factors relevant to combustion."""
        return {
            "L": self.L,
            "B": self.B,
            "aspect_ratio": self.L / max(self.B, 1e-30),
            "sphericity": self.sphericity,
            "surface_area": self.surface_area,
            "volume": self.volume,
            "eq_diameter": self.equivalent_diameter,
            "Sv_ratio": self.surface_to_volume_ratio,
            "boundary_word": boundary_word_discrete_profile(
                self._profile_x, self._profile_y
            ),
        }
