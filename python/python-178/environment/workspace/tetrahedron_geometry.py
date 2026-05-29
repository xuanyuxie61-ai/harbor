"""
tetrahedron_geometry.py
=======================
3D finite element geometry infrastructure synthesized from fem3d_pack.
Provides reference tetrahedron mappings, shape functions, barycentric
coordinates, volume computations, and Gauss-Jordan linear solvers
required by the DG discretization on tetrahedra.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Reference tetrahedron geometry
# ---------------------------------------------------------------------------

REFERENCE_TET4_VERTICES = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def tetrahedron_volume(vertices: np.ndarray) -> float:
    """
    Compute tetrahedron volume from 4x3 vertex array.
    V = |det([v1-v0, v2-v0, v3-v0])| / 6
    """
    M = np.vstack([
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0]
    ])
    return abs(np.linalg.det(M)) / 6.0


def barycentric_coordinates_3d(vertices: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of point p with respect to tetrahedron vertices.
    Solves 3x3 linear system via Cramer's rule for robustness.
    """
    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]
    v3 = vertices[3]
    # Compute signed volumes
    def signed_vol(a, b, c, d):
        M = np.vstack([b - a, c - a, d - a])
        return np.linalg.det(M) / 6.0

    vol = signed_vol(v0, v1, v2, v3)
    if abs(vol) < 1e-30:
        raise ValueError("Degenerate tetrahedron.")
    lam0 = signed_vol(p, v1, v2, v3) / vol
    lam1 = signed_vol(v0, p, v2, v3) / vol
    lam2 = signed_vol(v0, v1, p, v3) / vol
    lam3 = signed_vol(v0, v1, v2, p) / vol
    return np.array([lam0, lam1, lam2, lam3], dtype=np.float64)


def reference_to_physical_tet4(physical_vertices: np.ndarray,
                               xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Map from reference tetrahedron (xi,eta,zeta) with xi,eta,zeta >= 0, xi+eta+zeta <= 1
    to physical tetrahedron via affine transformation.
    x_phy = v0 + (v1-v0)*xi + (v2-v0)*eta + (v3-v0)*zeta
    """
    v0 = physical_vertices[0]
    v1 = physical_vertices[1]
    v2 = physical_vertices[2]
    v3 = physical_vertices[3]
    return v0 + (v1 - v0) * xi + (v2 - v0) * eta + (v3 - v0) * zeta


def physical_to_reference_tet4(physical_vertices: np.ndarray,
                               p: np.ndarray) -> np.ndarray:
    """
    Inverse map: physical point -> reference coordinates (xi, eta, zeta).
    Solves linear system J * [xi,eta,zeta]^T = p - v0.
    """
    J = np.column_stack([
        physical_vertices[1] - physical_vertices[0],
        physical_vertices[2] - physical_vertices[0],
        physical_vertices[3] - physical_vertices[0],
    ])
    rhs = p - physical_vertices[0]
    # Use numpy solve with fallback to pseudo-inverse for near-degenerate cases
    try:
        xi_eta_zeta = np.linalg.solve(J, rhs)
    except np.linalg.LinAlgError:
        xi_eta_zeta = np.linalg.lstsq(J, rhs, rcond=None)[0]
    return xi_eta_zeta


def jacobian_tet4(physical_vertices: np.ndarray) -> np.ndarray:
    """Return 3x3 Jacobian matrix of the reference->physical map."""
    return np.column_stack([
        physical_vertices[1] - physical_vertices[0],
        physical_vertices[2] - physical_vertices[0],
        physical_vertices[3] - physical_vertices[0],
    ])


def shape_function_linear_tet4(xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Linear (P1) shape functions on reference tetrahedron.
    phi = [1-xi-eta-zeta, xi, eta, zeta]
    """
    phi0 = 1.0 - xi - eta - zeta
    return np.array([phi0, xi, eta, zeta], dtype=np.float64)


def shape_function_quadratic_tet10(xi: float, eta: float, zeta: float) -> np.ndarray:
    """
    Quadratic (P2) shape functions on reference tetrahedron.
    10 nodes: 4 vertices + 6 edge midpoints.
    """
    l0 = 1.0 - xi - eta - zeta
    l1 = xi
    l2 = eta
    l3 = zeta
    phi = np.zeros(10, dtype=np.float64)
    # Vertices
    phi[0] = (2.0 * l0 - 1.0) * l0
    phi[1] = (2.0 * l1 - 1.0) * l1
    phi[2] = (2.0 * l2 - 1.0) * l2
    phi[3] = (2.0 * l3 - 1.0) * l3
    # Edges
    phi[4] = 4.0 * l0 * l1
    phi[5] = 4.0 * l0 * l2
    phi[6] = 4.0 * l0 * l3
    phi[7] = 4.0 * l1 * l2
    phi[8] = 4.0 * l1 * l3
    phi[9] = 4.0 * l2 * l3
    return phi


# ---------------------------------------------------------------------------
# Gauss-Jordan elimination with partial pivoting
# ---------------------------------------------------------------------------

def r8mat_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A*X = B via Gauss-Jordan elimination with partial pivoting.
    a : ndarray of shape (n, n)
    b : ndarray of shape (n, nrhs)
    Returns x of shape (n, nrhs).
    """
    n = a.shape[0]
    a = a.astype(np.float64).copy()
    b = b.astype(np.float64).copy()
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    nrhs = b.shape[1]
    for col in range(n):
        # Partial pivoting
        pivot = col
        max_val = abs(a[col, col])
        for row in range(col + 1, n):
            if abs(a[row, col]) > max_val:
                max_val = abs(a[row, col])
                pivot = row
        if max_val < 1e-30:
            raise ValueError("Singular matrix encountered in Gauss-Jordan elimination.")
        if pivot != col:
            a[[col, pivot], :] = a[[pivot, col], :]
            b[[col, pivot], :] = b[[pivot, col], :]
        # Normalize pivot row
        piv = a[col, col]
        a[col, :] /= piv
        b[col, :] /= piv
        # Eliminate other rows
        for row in range(n):
            if row != col and abs(a[row, col]) > 0.0:
                factor = a[row, col]
                a[row, :] -= factor * a[col, :]
                b[row, :] -= factor * b[col, :]
    return b.reshape(n, nrhs)


def r8mat_det_3d(a: np.ndarray) -> float:
    """Determinant of a 3x3 matrix."""
    if a.shape != (3, 3):
        raise ValueError("Matrix must be 3x3.")
    return (
        a[0,0]*(a[1,1]*a[2,2]-a[1,2]*a[2,1])
        - a[0,1]*(a[1,0]*a[2,2]-a[1,2]*a[2,0])
        + a[0,2]*(a[1,0]*a[2,1]-a[1,1]*a[2,0])
    )


# ---------------------------------------------------------------------------
# Brick hexahedron shape functions (for mesh coupling)
# ---------------------------------------------------------------------------

def shape_function_brick8(r: float, s: float, t: float) -> np.ndarray:
    """Trilinear brick8 shape functions on [-1,1]^3."""
    phi = np.zeros(8, dtype=np.float64)
    coeffs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
              (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    for i, (cr, cs, ct) in enumerate(coeffs):
        phi[i] = 0.125 * (1 + cr * r) * (1 + cs * s) * (1 + ct * t)
    return phi


def shape_function_brick27(r: float, s: float, t: float) -> np.ndarray:
    """Triquadratic brick27 shape functions on [-1,1]^3."""
    phi = np.zeros(27, dtype=np.float64)
    # Corner nodes
    corner_idx = 0
    for kr in [-1, 1]:
        for ks in [-1, 1]:
            for kt in [-1, 1]:
                nr = 0.5 * r * (r + kr) if kr == 1 else 0.5 * r * (r - 1)
                if kr == -1:
                    nr = 0.5 * r * (r - 1)
                elif kr == 1:
                    nr = 0.5 * r * (r + 1)
                else:
                    nr = (1 - r * r)
                # Actually simpler: use 1D Lagrange polynomials
                def lagrange_1d(xi, node):
                    if node == -1:
                        return 0.5 * xi * (xi - 1.0)
                    elif node == 0:
                        return 1.0 - xi * xi
                    elif node == 1:
                        return 0.5 * xi * (xi + 1.0)
                    else:
                        raise ValueError
                # Override above
                pass
    # Simplified implementation
    def N1(x): return 0.5 * x * (x - 1.0)
    def N2(x): return 1.0 - x * x
    def N3(x): return 0.5 * x * (x + 1.0)
    coords = [(-1, N1), (0, N2), (1, N3)]
    idx = 0
    for (kr, fr) in coords:
        for (ks, fs) in coords:
            for (kt, ft) in coords:
                phi[idx] = fr(r) * fs(s) * ft(t)
                idx += 1
    return phi
