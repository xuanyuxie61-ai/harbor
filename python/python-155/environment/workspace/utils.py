import numpy as np
from typing import Tuple, Optional


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if np.isclose(b, 0.0):
        return default
    return a / b


def normalize_vector(v: np.ndarray, ord: int = 2) -> np.ndarray:
    norm = np.linalg.norm(v, ord=ord)
    if np.isclose(norm, 0.0):
        return np.zeros_like(v)
    return v / norm


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def gcd_vec(a: np.ndarray) -> int:
    from math import gcd
    g = int(a[0])
    for val in a[1:]:
        g = gcd(g, int(val))
        if g == 1:
            break
    return g


def ensure_hermitian(H: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    Hh = 0.5 * (H + H.conj().T)
    if np.max(np.abs(H - Hh)) > tol:
        pass
    return Hh


def ensure_unitary(U: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    eye = np.eye(U.shape[0])
    diff = np.max(np.abs(U @ U.conj().T - eye))
    if diff > tol:

        Q, R = np.linalg.qr(U)

        d = np.diag(R)
        ph = d / np.abs(d)
        Q = Q * ph
        return Q
    return U


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    if n < 1:
        return np.array([])
    k = np.arange(n)
    x = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))

    return 0.5 * (a + b) + 0.5 * (b - a) * x


def hadamard_matrix(d: int) -> np.ndarray:
    if d == 1:
        return np.array([[1.0]], dtype=float)
    if not is_power_of_two(d):

        d2 = next_power_of_two(d)
        H = hadamard_matrix(d2)
        return H[:d, :d]
    H = np.array([[1.0]], dtype=float)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(2.0)**int(np.log2(d))


def grover_coin(d: int) -> np.ndarray:
    if d < 1:
        raise ValueError("Grover coin dimension must be >= 1")
    s = np.ones(d, dtype=float) / np.sqrt(d)
    return 2.0 * np.outer(s, s) - np.eye(d)


def discrete_laplacian_1d(n: int, periodic: bool = False) -> np.ndarray:
    if n < 2:
        return np.zeros((n, n))
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    L = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    if periodic and n > 2:
        L[0, -1] = -1.0
        L[-1, 0] = -1.0
    return L


def grid_neighbors_2d(i: int, j: int, nx: int, ny: int, periodic: bool = False) -> list:
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if periodic:
            ni = ni % nx
            nj = nj % ny
        if 0 <= ni < nx and 0 <= nj < ny:
            neighbors.append((ni, nj))
    return neighbors


def tensor_index_to_flat(idx: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    flat = 0
    stride = 1
    for d, dim in zip(reversed(idx), reversed(dims)):
        if not (0 <= d < dim):
            raise IndexError("Index out of bounds")
        flat += d * stride
        stride *= dim
    return flat


def flat_to_tensor_index(flat: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    idx = []
    for dim in reversed(dims):
        idx.append(flat % dim)
        flat //= dim
    return tuple(reversed(idx))
