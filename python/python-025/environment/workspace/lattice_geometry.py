"""
lattice_geometry.py
===================
Geometric utilities for crystal lattice analysis synthesized from seed projects:
  - 1309_triangle_interpolate (triangle area, barycentric interpolation, uniform sampling)
  - 150_cg_lab_triangles (signed point-to-line distance)

Core algorithms:
  - Signed triangle area via shoelace formula
  - Linear barycentric interpolation on triangular elements
  - Uniform random sampling inside a triangle (Turk's Rule #1)
  - Signed perpendicular distance from point to line
  - Hexagonal and square lattice builders for 2D/3D dusty plasma crystals
"""

import numpy as np


def triangle_area(p1, p2, p3):
    """
    Signed area of a triangle via the shoelace formula.
    
    Based on seed 1309_triangle_interpolate.
    
    Area = 0.5 * [ x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2) ]
    
    Positive for counterclockwise vertex ordering, negative for clockwise.
    """
    p1, p2, p3 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float), np.asarray(p3, dtype=float)
    return 0.5 * (p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))


def barycentric_interpolate(query_points, p1, p2, p3, v1, v2, v3):
    """
    Linear barycentric interpolation of vertex data to query points inside a triangle.
    
    Based on seed 1309_triangle_interpolate.
    
    For a point p inside triangle (p1, p2, p3), the interpolated value is:
      v(p) = (Area(p,p2,p3) * v1 + Area(p1,p,p3) * v2 + Area(p1,p2,p) * v3) / Area(p1,p2,p3)
    
    where Area() is the signed triangle area. The weights w1, w2, w3 are the
    barycentric coordinates of p.
    """
    query_points = np.asarray(query_points, dtype=float)
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    
    A = triangle_area(p1, p2, p3)
    if abs(A) < 1e-14:
        raise ValueError("Degenerate triangle: area is near zero")
    
    v1, v2, v3 = np.asarray(v1), np.asarray(v2), np.asarray(v3)
    results = []
    for p in query_points:
        A1 = triangle_area(p, p2, p3)
        A2 = triangle_area(p1, p, p3)
        A3 = triangle_area(p1, p2, p)
        w1, w2, w3 = A1 / A, A2 / A, A3 / A
        results.append(w1 * v1 + w2 * v2 + w3 * v3)
    return np.array(results)


def uniform_in_triangle(v1, v2, v3, n):
    """
    Generate n uniformly distributed random points inside a triangle.
    
    Based on seed 1309_triangle_interpolate (Turk's Rule #1 from Graphics Gems).
    
    Algorithm:
      Draw r1, r2 ~ Uniform(0,1).
      a = 1 - sqrt(r2)
      b = (1 - r1) * sqrt(r2)
      c = r1 * sqrt(r2)
      p = a*v1 + b*v2 + c*v3
    
    The mapping (r1, r2) -> (a, b, c) produces uniform barycentric coordinates.
    """
    v1, v2, v3 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float), np.asarray(v3, dtype=float)
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    sqrt_r2 = np.sqrt(r2)
    a = 1.0 - sqrt_r2
    b = (1.0 - r1) * sqrt_r2
    c = r1 * sqrt_r2
    return (a[:, None] * v1 + b[:, None] * v2 + c[:, None] * v3)


def signed_point_line_distance(p1, p2, p):
    """
    Signed perpendicular distance from point p to the line through p1 and p2.
    
    Based on seed 150_cg_lab_triangles.
    
    Direction vector: d = p2 - p1
    Unit normal:      n = (-d_y, d_x) / ||d||
    Signed distance:  dist = n^T * (p - p1)
    
    Positive indicates the point lies on one side of the directed line;
    negative indicates the opposite side; zero means on the line.
    Used for detecting lattice planes and stacking faults in crystal structures.
    """
    p1, p2, p = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float), np.asarray(p, dtype=float)
    d = p2 - p1
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-14:
        return 0.0
    n = np.array([-d[1], d[0]]) / norm_d
    return float(np.dot(n, p - p1))


def build_hexagonal_lattice(n_rows, n_cols, a):
    """
    Build a 2D hexagonal (triangular) lattice with lattice constant a.
    
    In 2D, each particle has 6 nearest neighbors at distance a.
    The basis vectors are:
      a1 = a * (1, 0)
      a2 = a * (1/2, sqrt(3)/2)
    """
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * a
            y = i * a * np.sqrt(3.0) / 2.0
            if i % 2 == 1:
                x += 0.5 * a
            positions.append([x, y, 0.0])
    return np.array(positions, dtype=float)


def build_square_lattice(n_rows, n_cols, a):
    """Build a 2D square lattice with lattice constant a."""
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            positions.append([j * a, i * a, 0.0])
    return np.array(positions, dtype=float)


def build_simple_cubic_lattice(nx, ny, nz, a):
    """Build a 3D simple cubic lattice with lattice constant a."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions.append([i * a, j * a, k * a])
    return np.array(positions, dtype=float)
