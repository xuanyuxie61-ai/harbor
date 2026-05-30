
import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK






def hat_basis(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    n = len(xi)
    phi = np.zeros((len(x), n), dtype=float)
    for i in range(n - 1):
        h = xi[i + 1] - xi[i]
        if abs(h) < EPS:
            continue
        mask = (x >= xi[i]) & (x <= xi[i + 1])
        phi[mask, i] = (xi[i + 1] - x[mask]) / h
        phi[mask, i + 1] = (x[mask] - xi[i]) / h
    return phi


def assemble_fem_matrices_1d(nodes: np.ndarray,
                              diffusion_coeff: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    if n < 2:
        raise ValueError("Need at least 2 nodes.")
    M = np.zeros((n, n), dtype=float)
    K = np.zeros((n, n), dtype=float)











    raise NotImplementedError("Hole 3: FEM 矩阵组装待实现")


def extract_tridiagonal(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    r83 = np.zeros((3, n), dtype=float)
    r83[1, :] = np.diag(A)
    if n > 1:
        r83[0, 1:] = np.diag(A, -1)
        r83[2, :-1] = np.diag(A, 1)
    return r83






def triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    cross = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
    return 0.5 * abs(cross)


def integrate_on_2d_section(r_nodes: np.ndarray, z_nodes: np.ndarray,
                            f_values: np.ndarray) -> float:
    r_nodes = np.asarray(r_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    f_values = np.asarray(f_values, dtype=float)
    n = len(r_nodes)
    if n < 3:
        return 0.0
    integral = 0.0

    center_r = np.mean(r_nodes)
    center_z = np.mean(z_nodes)
    for i in range(n):
        i1 = i
        i2 = (i + 1) % n
        v1 = np.array([center_r, center_z])
        v2 = np.array([r_nodes[i1], z_nodes[i1]])
        v3 = np.array([r_nodes[i2], z_nodes[i2]])
        area = triangle_area(v1, v2, v3)

        g1 = (2.0 * v1 + v2 + v3) / 4.0
        g2 = (v1 + 2.0 * v2 + v3) / 4.0
        g3 = (v1 + v2 + 2.0 * v3) / 4.0

        f_g = (f_values[i1] + f_values[i2]) / 2.0
        integral += area * f_g
    return integral






def project_function_to_fem(nodes: np.ndarray, func) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    return func(nodes)


def fem_l2_norm(nodes: np.ndarray, u: np.ndarray, M: np.ndarray = None) -> float:
    u = np.asarray(u, dtype=float)
    if M is None:
        M, _ = assemble_fem_matrices_1d(nodes)
    norm_sq = float(u @ (M @ u))
    return np.sqrt(max(norm_sq, 0.0))
