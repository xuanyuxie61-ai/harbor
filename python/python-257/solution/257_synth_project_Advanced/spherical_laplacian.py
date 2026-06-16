"""
spherical_laplacian.py
======================
Discrete Laplace-Beltrami operator on the cubed-sphere mesh.

The continuous Laplace-Beltrami operator on S^2 reads
    Delta_S Y_lm = - l(l+1) Y_lm
so its eigenvalues are  -l(l+1).   This module assembles a sparse
matrix approximation to Delta_S and (optionally) solves the
generalised eigenvalue problem  L f = lambda f  to recover the
spherical harmonic eigenvalues numerically.

Two discretisations are supported:
  * FD (finite differences) – local stencils from fd_coefficients
  * FEM (finite elements)   – T4 cubic-bubble elements from fem_basis

The FEM approach follows:
    - Assemble element stiffness matrices  K_e = integral_e grad phi_i . grad phi_j
    - Assemble element mass matrices       M_e = integral_e phi_i phi_j
    - Solve generalised problem  K a = lambda M a
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

from spherical_mesh import CubedSphere, IcosahedralMesh
from fd_coefficients import tangent_plane_project, fornberg_weights


# ---------------------------------------------------------------------------
# Finite-difference Laplacian (assembles a dense matrix)
# ---------------------------------------------------------------------------
def assemble_fd_laplacian(mesh: CubedSphere) -> List[List[float]]:
    """
    Assemble the Laplace-Beltrami operator as a dense n_nodes x n_nodes matrix
    using 2nd-order finite differences on the cubed-sphere.

    For each node i with neighbours {j}, we use the identity
        Delta f(x_i) ~ (2 / (N_i h_i^2)) sum_j (f(x_j) - f(x_i))
    where N_i = number of neighbours, h_i = mean angular spacing.
    """
    n = mesh.n_nodes
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        nbrs = mesh.neighbours[i]
        if not nbrs:
            continue
        # average angular spacing
        dists = [mesh.angular_separation(i, j) for j in nbrs]
        h2 = sum(d * d for d in dists) / len(dists)
        if h2 < 1.0e-30:
            continue
        coeff = 2.0 / (len(nbrs) * h2)
        L[i][i] = -coeff * len(nbrs)
        for j in nbrs:
            L[i][j] += coeff
    return L


# ---------------------------------------------------------------------------
# FEM T4 cubic-bubble element on a single triangle
# ---------------------------------------------------------------------------
def basis_t4_physical(t_coords: List[Tuple[float, float]],
                       local: int,
                       p: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    Evaluate the T4 cubic-bubble basis function and its derivatives
    at point p inside a triangle with vertices t_coords (in 2D).
    Follows Burkardt's fem_basis_t4_display.m verbatim.
    Returns (phi, dphidx, dphidy).
    """
    x1, y1 = t_coords[0]
    x2, y2 = t_coords[1]
    x3, y3 = t_coords[2]
    px, py = p
    area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if abs(area) < 1.0e-30:
        return 0.0, 0.0, 0.0

    # Three linear barycentric coordinates
    psi = [0.0, 0.0, 0.0]
    dpsidx = [0.0, 0.0, 0.0]
    dpsidy = [0.0, 0.0, 0.0]
    psi[0]    = ((x3 - x2) * (py - y2) - (y3 - y2) * (px - x2))
    dpsidx[0] = -(y3 - y2)
    dpsidy[0] =  (x3 - x2)
    psi[1]    = ((x1 - x3) * (py - y3) - (y1 - y3) * (px - x3))
    dpsidx[1] = -(y1 - y3)
    dpsidy[1] =  (x1 - x3)
    psi[2]    = ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
    dpsidx[2] = -(y2 - y1)
    dpsidy[2] =  (x2 - x1)

    for k in range(3):
        psi[k] /= area
        dpsidx[k] /= area
        dpsidy[k] /= area

    # Cubic bubble  phi_4 = 27 psi1 psi2 psi3
    psi4 = 27.0 * psi[0] * psi[1] * psi[2]
    dpsi4dx = 27.0 * (dpsidx[0] * psi[1] * psi[2] + psi[0] * dpsidx[1] * psi[2] + psi[0] * psi[1] * dpsidx[2])
    dpsi4dy = 27.0 * (dpsidy[0] * psi[1] * psi[2] + psi[0] * dpsidy[1] * psi[2] + psi[0] * psi[1] * dpsidy[2])

    # Modified basis:  phi_k = psi_k - psi_4 / 3
    for k in range(3):
        psi[k]    -= psi4 / 3.0
        dpsidx[k] -= dpsi4dx / 3.0
        dpsidy[k] -= dpsi4dy / 3.0

    if local == 4:
        return psi4, dpsi4dx, dpsi4dy
    if local < 1 or local > 3:
        raise ValueError(f"local must be in [1, 4], got {local}")
    return psi[local - 1], dpsidx[local - 1], dpsidy[local - 1]


# ---------------------------------------------------------------------------
# Assemble the FEM stiffness matrix on a triangulated sphere
# ---------------------------------------------------------------------------
def assemble_fem_stiffness(mesh: IcosahedralMesh) -> List[List[float]]:
    """
    Assemble the stiffness matrix  K  for the FEM discretisation
    of the Laplace-Beltrami operator on S^2 using linear P1 elements
    on the icosahedral triangulation.

    K_{ij} = integral_S^2  grad_S phi_i . grad_S phi_j  d Omega

    On each triangular face with vertices A, B, C, the gradients of
    the linear basis functions are constant:
        grad phi_A = (1 / (2|T|)) n x (C - B)   etc.
    where |T| is the area of the face (on the sphere).
    """
    n = mesh.n_nodes
    K = [[0.0] * n for _ in range(n)]
    for (i, j, k) in mesh.faces:
        xi, yi, zi = mesh.nodes[i]
        xj, yj, zj = mesh.nodes[j]
        xk, yk, zk = mesh.nodes[k]
        # Area of spherical triangle via Girard's theorem:
        #   Area = A + B + C - pi
        # but for small triangles, |T| ~ (1/2) | (B-A) x (C-A) |
        ax, ay, az = xj - xi, yj - yi, zj - zi
        bx, by, bz = xk - xi, yk - yi, zk - zi
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        if area < 1.0e-30:
            continue

        # Gradients in Cartesian (projected onto tangent plane)
        # For a linear element, grad phi_i . grad phi_j is constant on the face
        # We use the cotangent formula for the stiffness matrix:
        #   K_{ij} = (1/2) (cot alpha_ij + cot beta_ij)
        # where alpha, beta are the angles opposite to edge (i,j).
        cots = _cotangents_of_triangle((xi, yi, zi), (xj, yj, zj), (xk, yk, zk))
        cot_i, cot_j, cot_k = cots  # angles at vertex i, j, k

        K[j][k] += 0.5 * cot_i
        K[k][j] += 0.5 * cot_i
        K[i][k] += 0.5 * cot_j
        K[k][i] += 0.5 * cot_j
        K[i][j] += 0.5 * cot_k
        K[j][i] += 0.5 * cot_k

        K[i][i] -= 0.5 * (cot_j + cot_k)
        K[j][j] -= 0.5 * (cot_i + cot_k)
        K[k][k] -= 0.5 * (cot_i + cot_j)

    return K


def _cotangents_of_triangle(A: Tuple[float, float, float],
                              B: Tuple[float, float, float],
                              C: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Return cotangents of the three angles of a triangle on S^2."""
    # Edges
    def sub(p, q):
        return (p[0] - q[0], p[1] - q[1], p[2] - q[2])
    def dot(p, q):
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    def cross(p, q):
        return (p[1] * q[2] - p[2] * q[1],
                p[2] * q[0] - p[0] * q[2],
                p[0] * q[1] - p[1] * q[0])

    AB = sub(B, A)
    AC = sub(C, A)
    BA = sub(A, B)
    BC = sub(C, B)
    CA = sub(A, C)
    CB = sub(B, C)

    # Cotangent at A = dot(AB, AC) / |AB x AC|
    crA = cross(AB, AC)
    nrmA = math.sqrt(max(1e-30, dot(crA, crA)))
    cot_A = dot(AB, AC) / nrmA if nrmA > 1e-30 else 0.0

    crB = cross(BA, BC)
    nrmB = math.sqrt(max(1e-30, dot(crB, crB)))
    cot_B = dot(BA, BC) / nrmB if nrmB > 1e-30 else 0.0

    crC = cross(CA, CB)
    nrmC = math.sqrt(max(1e-30, dot(crC, crC)))
    cot_C = dot(CA, CB) / nrmC if nrmC > 1e-30 else 0.0

    return cot_A, cot_B, cot_C


# ---------------------------------------------------------------------------
# Assemble FEM mass matrix  M_ij = integral phi_i phi_j dOmega
# ---------------------------------------------------------------------------
def assemble_fem_mass(mesh: IcosahedralMesh) -> List[List[float]]:
    """Lumped-mass approximation: M_{ii} = sum of 1/3 of each face area."""
    n = mesh.n_nodes
    M = [[0.0] * n for _ in range(n)]
    for (i, j, k) in mesh.faces:
        xi, yi, zi = mesh.nodes[i]
        xj, yj, zj = mesh.nodes[j]
        xk, yk, zk = mesh.nodes[k]
        ax, ay, az = xj - xi, yj - yi, zj - zi
        bx, by, bz = xk - xi, yk - yi, zk - zi
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        if area < 1.0e-30:
            continue
        # Consistent mass: M_ij = area/12 if i==j else area/24
        for a, b in [(i, i), (j, j), (k, k)]:
            M[a][b] += area / 6.0
        for a, b in [(i, j), (j, i), (j, k), (k, j), (i, k), (k, i)]:
            M[a][b] += area / 12.0
    return M


# ---------------------------------------------------------------------------
# Eigenvalue computation: power iteration for largest |lambda|
# ---------------------------------------------------------------------------
def power_iteration(M: List[List[float]], x0: List[float],
                     n_iter: int = 200, tol: float = 1.0e-10) -> Tuple[float, List[float]]:
    """Simple power iteration. Returns (lambda, x) for largest |lambda|."""
    n = len(x0)
    x = x0[:]
    lam = 0.0
    for _ in range(n_iter):
        # y = M x
        y = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += M[i][j] * x[j]
            y[i] = s
        # normalise
        norm = math.sqrt(max(1e-30, sum(v * v for v in y)))
        if norm < 1.0e-30:
            break
        x_new = [v / norm for v in y]
        lam_new = sum(x_new[i] * y[i] for i in range(n)) / sum(x_new[i] * x_new[i] for i in range(n))
        if abs(lam_new - lam) < tol:
            return lam_new, x_new
        x = x_new
        lam = lam_new
    return lam, x


# ---------------------------------------------------------------------------
# Inverse iteration for the smallest |lambda| (shift-and-invert)
# ---------------------------------------------------------------------------
def inverse_iteration(M: List[List[float]], x0: List[float],
                       shift: float = 0.0, n_iter: int = 100,
                       tol: float = 1.0e-10) -> Tuple[float, List[float]]:
    """
    Inverse iteration to find the eigenvalue closest to `shift`.
    Solves  (M - shift I) x_{k+1} = x_k  at each step.
    Returns (lambda, x).
    """
    n = len(x0)
    # Build shifted matrix  A = M - shift I
    A = [[M[i][j] - (shift if i == j else 0.0) for j in range(n)] for i in range(n)]
    x = x0[:]
    lam = 0.0
    for _ in range(n_iter):
        y = _solve_linear(A, x)
        if y is None:
            break
        norm = math.sqrt(max(1e-30, sum(v * v for v in y)))
        if norm < 1e-30:
            break
        x = [v / norm for v in y]
        # Rayleigh quotient
        lam_new = sum(x[i] * sum(M[i][j] * x[j] for j in range(n)) for i in range(n))
        if abs(lam_new - lam) < tol:
            return lam_new, x
        lam = lam_new
    return lam, x


def _solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
        if max_val < 1.0e-30:
            return None
        M[k], M[max_row] = M[max_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-30:
            return None
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


# ---------------------------------------------------------------------------
# Apply Laplacian to a field on the mesh (matrix-vector product)
# ---------------------------------------------------------------------------
def apply_laplacian(L: List[List[float]], f: List[float]) -> List[float]:
    """Return L @ f."""
    n = len(f)
    return [sum(L[i][j] * f[j] for j in range(n)) for i in range(n)]


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cs = CubedSphere(nside=3)
    L = assemble_fd_laplacian(cs)
    print(f"FD Laplacian assembled on {cs.n_nodes} nodes")
    print(f"  max row sum = {max(sum(abs(v) for v in row) for row in L):.4e}")

    im = IcosahedralMesh(level=1)
    K = assemble_fem_stiffness(im)
    M = assemble_fem_mass(im)
    print(f"FEM stiffness/mass assembled on {im.n_nodes} nodes")
