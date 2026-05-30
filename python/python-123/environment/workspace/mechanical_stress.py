
import numpy as np
from typing import Tuple


def build_5point_laplacian_2d(nx: int, ny: int, hx: float, hy: float) -> np.ndarray:
    N = nx * ny
    L = np.zeros((N, N))
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    c0 = -2.0 * (cx + cy)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            L[k, k] = c0
            if i > 0:
                L[k, idx(i - 1, j)] = cx
            if i < nx - 1:
                L[k, idx(i + 1, j)] = cx
            if j > 0:
                L[k, idx(i, j - 1)] = cy
            if j < ny - 1:
                L[k, idx(i, j + 1)] = cy

    return L


def biharmonic_stress_operator(
    nx: int, ny: int, hx: float, hy: float, mu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < mu < 1.0):
        raise ValueError("biharmonic_stress_operator: mu 必须在 (0,1)")
    if nx < 3 or ny < 3:
        raise ValueError("biharmonic_stress_operator: 网格尺寸至少为 3x3")

    N = nx * ny
    L = build_5point_laplacian_2d(nx, ny, hx, hy)



    A = L @ L


    B = np.eye(N)


    def idx(i, j):
        return i * ny + j

    boundary_indices = set()
    for i in range(nx):
        boundary_indices.add(idx(i, 0))
        boundary_indices.add(idx(i, ny - 1))
    for j in range(ny):
        boundary_indices.add(idx(0, j))
        boundary_indices.add(idx(nx - 1, j))

    boundary_indices = sorted(list(boundary_indices))
    interior_indices = [k for k in range(N) if k not in boundary_indices]


    A_ii = A[np.ix_(interior_indices, interior_indices)]
    B_ii = B[np.ix_(interior_indices, interior_indices)]


    k = min(6, len(interior_indices))
    eigenvalues, eigenvectors = np.linalg.eigh(A_ii)


    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx][:k]
    eigenvectors = eigenvectors[:, sort_idx][:, :k]


    full_vectors = np.zeros((N, k))
    full_vectors[interior_indices, :] = eigenvectors


    phi = full_vectors[:, 0]
    phi_grid = phi.reshape((nx, ny))


    w_xx = np.zeros_like(phi_grid)
    w_yy = np.zeros_like(phi_grid)
    w_xy = np.zeros_like(phi_grid)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            w_xx[i, j] = (phi_grid[i + 1, j] - 2.0 * phi_grid[i, j] +
                          phi_grid[i - 1, j]) / (hx ** 2)
            w_yy[i, j] = (phi_grid[i, j + 1] - 2.0 * phi_grid[i, j] +
                          phi_grid[i, j - 1]) / (hy ** 2)
            w_xy[i, j] = (phi_grid[i + 1, j + 1] - phi_grid[i + 1, j - 1] -
                          phi_grid[i - 1, j + 1] + phi_grid[i - 1, j - 1]) / (4.0 * hx * hy)

    E = 1.0
    denom = 1.0 - mu ** 2
    denom = max(denom, 1e-15)

    sigma_xx = E / denom * (w_xx + mu * w_yy)
    sigma_yy = E / denom * (w_yy + mu * w_xx)
    sigma_xy = E / (2.0 * (1.0 + mu)) * w_xy

    stress_vm = np.sqrt(np.maximum(
        sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3.0 * sigma_xy ** 2,
        0.0
    ))

    return eigenvalues, full_vectors, stress_vm.ravel()


def compute_stress_induced_apoptosis(
    stress_vm: np.ndarray, threshold: float = 0.5, steepness: float = 10.0
) -> np.ndarray:
    stress_vm = np.asarray(stress_vm, dtype=float)
    z = steepness * (stress_vm - threshold)

    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_tumor_stress_metrics(stress_vm: np.ndarray) -> dict:
    if stress_vm.size == 0:
        return {
            "max_stress": 0.0,
            "mean_stress": 0.0,
            "std_stress": 0.0,
            "high_stress_fraction": 0.0,
        }

    max_s = float(np.max(stress_vm))
    mean_s = float(np.mean(stress_vm))
    std_s = float(np.std(stress_vm))
    frac = float(np.mean(stress_vm > 0.5 * max_s))

    return {
        "max_stress": max_s,
        "mean_stress": mean_s,
        "std_stress": std_s,
        "high_stress_fraction": frac,
    }
