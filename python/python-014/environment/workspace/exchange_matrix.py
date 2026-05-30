
import numpy as np
from typing import Tuple, Optional
from utils import build_skyline_from_tridiagonal, skyline_mv, EPS_MACHINE


def exchange_laplacian_1d(n: int, h: float, bc: str = "dirichlet") -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3")
    if h <= 0.0:
        raise ValueError("h must be positive")

    L = np.zeros((n, n), dtype=float)
    inv_h2 = 1.0 / (h * h)

    if bc == "dirichlet":

        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "neumann":

        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "periodic":

        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        L[0, n - 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, 0] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

    return L


def exchange_laplacian_2d(nx: int, ny: int, hx: float, hy: float, bc: str = "dirichlet") -> np.ndarray:
    if nx < 2 or ny < 2:
        raise ValueError("nx, ny must be >= 2")
    Lx = exchange_laplacian_1d(nx, hx, bc)
    Ly = exchange_laplacian_1d(ny, hy, bc)
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    L = np.kron(Iy, Lx) + np.kron(Ly, Ix)
    return L


def exchange_skyline_1d(n: int, h: float, bc: str = "dirichlet") -> Tuple[int, np.ndarray, np.ndarray]:
    L = exchange_laplacian_1d(n, h, bc)

    lower = np.diag(L, -1).copy()
    diag = np.diag(L).copy()
    return build_skyline_from_tridiagonal(lower, diag, lower)


def apply_exchange_operator(J: np.ndarray, spins: np.ndarray) -> np.ndarray:
    if spins.ndim == 1:

        return J @ spins
    elif spins.ndim == 2 and spins.shape[1] == 3:

        hx = J @ spins[:, 0]
        hy = J @ spins[:, 1]
        hz = J @ spins[:, 2]
        return np.column_stack([hx, hy, hz])
    else:
        raise ValueError("spins must be 1D (Ising) or 2D with shape (N, 3) (Heisenberg)")


def exchange_energy(J: np.ndarray, spins: np.ndarray) -> float:
    H = apply_exchange_operator(J, spins)
    if spins.ndim == 1:
        e = 0.5 * np.dot(spins, H)
    else:
        e = 0.5 * np.sum(spins * H)
    return float(e)


def add_disorder(J: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    N = J.shape[0]
    disorder = np.random.normal(0.0, std, size=(N, N))
    disorder = (disorder + disorder.T) * 0.5
    np.fill_diagonal(disorder, 0.0)
    return J + disorder
