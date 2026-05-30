
import numpy as np
from typing import Tuple





def laplacian_1d_dd(n: int, L: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / (n + 1.0)
    diag = 2.0 * np.ones(n)
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_dn(n: int, L: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / n
    diag = 2.0 * np.ones(n)
    diag[-1] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_nd(n: int, L: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / n
    diag = 2.0 * np.ones(n)
    diag[0] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_nn(n: int, L: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / (n - 1.0) if n > 1 else L
    diag = 2.0 * np.ones(n)
    diag[0] = 1.0
    diag[-1] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_pp(n: int, L: float = 1.0) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2 for periodic")
    h = L / n
    diag = 2.0 * np.ones(n)
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    A[0, -1] = -1.0
    A[-1, 0] = -1.0
    return A / (h * h)






def laplacian_dd_eigenvalues(n: int, L: float = 1.0) -> np.ndarray:
    h = L / (n + 1.0)
    j = np.arange(1, n + 1)
    lam = (4.0 / (h * h)) * np.sin(j * np.pi / (2.0 * (n + 1.0))) ** 2
    return lam


def laplacian_dd_eigenvectors(n: int) -> np.ndarray:
    V = np.empty((n, n), dtype=float)
    scale = np.sqrt(2.0 / (n + 1.0))
    for j in range(1, n + 1):
        i = np.arange(1, n + 1)
        V[:, j - 1] = scale * np.sin(i * j * np.pi / (n + 1.0))
    return V


def laplacian_pp_eigenvalues(n: int, L: float = 1.0) -> np.ndarray:
    h = L / n
    j = np.arange(n)
    lam = (4.0 / (h * h)) * np.sin(j * np.pi / n) ** 2
    return lam






def laplacian_apply(u: np.ndarray, bc_type: str = "DD", L: float = 1.0) -> np.ndarray:
    n = len(u)
    if n < 1:
        raise ValueError("u must be non-empty")
    h = L / (n + 1.0) if bc_type == "DD" else L / n
    if bc_type == "DD":
        h = L / (n + 1.0)
        Au = np.zeros(n)
        Au[0] = (2.0 * u[0] - u[1]) / (h * h)
        Au[-1] = (2.0 * u[-1] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au
    elif bc_type == "PP":
        h = L / n
        Au = np.zeros(n)
        Au[0] = (2.0 * u[0] - u[1] - u[-1]) / (h * h)
        Au[-1] = (2.0 * u[-1] - u[0] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au
    else:

        h = L / n if n > 1 else L
        Au = np.zeros(n)
        if bc_type in ("DN", "NN"):
            Au[0] = (u[0] - u[1]) / (h * h)
        else:
            Au[0] = (2.0 * u[0] - u[1]) / (h * h)
        if bc_type in ("ND", "NN"):
            Au[-1] = (u[-1] - u[-2]) / (h * h)
        else:
            Au[-1] = (2.0 * u[-1] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au






def laplacian_3d_tensor(nx: int, ny: int, nz: int,
                         Lx: float, Ly: float, Lz: float) -> np.ndarray:
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Grid dimensions must be positive")
    Ax = laplacian_1d_dd(nx, Lx)
    Ay = laplacian_1d_dd(ny, Ly)
    Az = laplacian_1d_dd(nz, Lz)
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    Iz = np.eye(nz)

    A = (np.kron(Iz, np.kron(Iy, Ax))
         + np.kron(Iz, np.kron(Ay, Ix))
         + np.kron(Az, np.kron(Iy, Ix)))
    return A


def solve_steady_state_diffusion(n: int, L: float, source: np.ndarray,
                                  D: float = 1.0, bc_type: str = "DD") -> np.ndarray:
    if len(source) != n:
        raise ValueError("source length must match n")
    if bc_type != "DD":
        raise NotImplementedError("Only DD supported for direct solve currently")
    A = laplacian_1d_dd(n, L)
    rhs = source / D

    lam = laplacian_dd_eigenvalues(n, L)
    V = laplacian_dd_eigenvectors(n)

    C = V @ ((V.T @ rhs) / lam)
    return C






def solve_tissue_concentration_profile(n_points: int, tissue_length: float,
                                        D_eff: float, clearance_rate: float,
                                        influx: float) -> np.ndarray:
    if D_eff <= 0.0 or clearance_rate < 0.0 or tissue_length <= 0.0:
        raise ValueError("Invalid diffusion parameters")
    h = tissue_length / (n_points + 1.0)
    x = np.linspace(h, tissue_length - h, n_points)

    alpha = np.sqrt(clearance_rate / D_eff)
    if alpha * tissue_length > 700:

        C_exact = influx * np.exp(-alpha * x)
    else:
        C_exact = influx * np.sinh(alpha * (tissue_length - x)) / np.sinh(alpha * tissue_length)

    A = laplacian_1d_dd(n_points, tissue_length)

    rhs = np.zeros(n_points)
    rhs[0] += D_eff * influx / (h * h)
    C_fd = np.linalg.solve(D_eff * A + clearance_rate * np.eye(n_points), rhs)
    return x, C_exact, C_fd






if __name__ == "__main__":
    n = 20
    L = 1.0
    A = laplacian_1d_dd(n, L)
    lam = laplacian_dd_eigenvalues(n, L)
    lam_num = np.linalg.eigvalsh(A)
    print(f"Eigenvalue max error: {np.max(np.abs(np.sort(lam) - np.sort(lam_num))):.2e}")
    x, C_ex, C_fd = solve_tissue_concentration_profile(50, 0.01, 1e-9, 0.01, 1.0)
    print(f"Max FD error vs exact: {np.max(np.abs(C_ex - C_fd)):.2e}")
    A3 = laplacian_3d_tensor(4, 4, 4, 1.0, 1.0, 1.0)
    print(f"3D Laplacian shape: {A3.shape}")
