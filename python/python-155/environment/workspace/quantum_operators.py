import numpy as np
from typing import List, Optional, Tuple
from utils import normalize_vector, ensure_unitary, ensure_hermitian, clamp





def piecewise_constant_2d(xc: np.ndarray, yc: np.ndarray,
                          values: np.ndarray) -> callable:
    nx = len(xc) - 1
    ny = len(yc) - 1
    if values.shape != (nx, ny):
        raise ValueError(f"values shape {values.shape} does not match grid ({nx}, {ny})")

    def f(x: float, y: float) -> float:

        x = clamp(x, xc[0], xc[-1])
        y = clamp(y, yc[0], yc[-1])

        ix = min(nx - 1, max(0, int(np.searchsorted(xc, x, side='right') - 1)))
        iy = min(ny - 1, max(0, int(np.searchsorted(yc, y, side='right') - 1)))
        return float(values[ix, iy])

    return f


def piecewise_constant_1d(xc: np.ndarray, values: np.ndarray) -> callable:
    n = len(xc) - 1
    if len(values) != n:
        raise ValueError(f"values length {len(values)} does not match grid {n}")

    def f(x: float) -> float:
        x = clamp(x, xc[0], xc[-1])
        ix = min(n - 1, max(0, int(np.searchsorted(xc, x, side='right') - 1)))
        return float(values[ix])

    return f





def hadamard_coin(d: int) -> np.ndarray:
    from utils import hadamard_matrix
    H = hadamard_matrix(d)
    return H


def grover_coin(d: int) -> np.ndarray:
    from utils import grover_coin as gc
    return gc(d)


def fourier_coin(d: int) -> np.ndarray:
    F = np.zeros((d, d), dtype=complex)
    omega = np.exp(2.0j * np.pi / d)
    for j in range(d):
        for k in range(d):
            F[j, k] = omega ** (j * k) / np.sqrt(d)
    return F


def custom_phase_coin(d: int, phases: np.ndarray) -> np.ndarray:
    if len(phases) != d:
        raise ValueError("phases length must equal coin dimension")
    D = np.diag(np.exp(1.0j * phases))
    return D





def shift_operator_1d(n: int, coin_dim: int = 2,
                      periodic: bool = True) -> np.ndarray:
    N = n * coin_dim
    S = np.zeros((N, N), dtype=complex)
    for x in range(n):
        for c in range(coin_dim):
            if coin_dim == 2:

                dx = 1 if c == 0 else -1
            else:
                dx = c - coin_dim // 2
            x_next = (x + dx) % n if periodic else clamp(x + dx, 0, n - 1)

            if not periodic and x_next != x + dx:
                x_next = clamp(x - dx, 0, n - 1)
            idx = c * n + x
            idx_next = c * n + x_next
            S[idx_next, idx] = 1.0
    return S


def shift_operator_graph(adj: List[List[int]], coin_dim: Optional[int] = None) -> np.ndarray:
    n = len(adj)
    degrees = [len(adj[v]) for v in range(n)]
    if coin_dim is None:
        coin_dim = max(degrees) if degrees else 1
    N = n * coin_dim
    S = np.zeros((N, N), dtype=complex)
    for v in range(n):
        dv = degrees[v]
        if dv == 0:
            continue
        for j, u in enumerate(adj[v]):
            if j >= coin_dim:
                break

            du = degrees[u]
            j_prime = None
            for jj, vv in enumerate(adj[u]):
                if vv == v:
                    j_prime = jj
                    break
            if j_prime is None or j_prime >= coin_dim:
                j_prime = j
            idx_from = j * n + v
            idx_to = j_prime * n + u
            S[idx_to, idx_from] = 1.0
    return S





def oracle_operator(n: int, coin_dim: int, marked_vertices: List[int],
                    phase: float = np.pi) -> np.ndarray:
    N = n * coin_dim
    O = np.eye(N, dtype=complex)
    for v in marked_vertices:
        if 0 <= v < n:
            for c in range(coin_dim):
                idx = c * n + v
                O[idx, idx] = np.exp(1.0j * phase)
    return O


def marked_state_projection(n: int, coin_dim: int, marked_vertices: List[int]) -> np.ndarray:
    N = n * coin_dim
    P = np.zeros((N, N), dtype=complex)
    for v in marked_vertices:
        if 0 <= v < n:
            for c in range(coin_dim):
                idx = c * n + v
                P[idx, idx] = 1.0
    return P





def graph_laplacian(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    L = np.zeros((n, n), dtype=float)
    for v in range(n):
        dv = len(adj[v])
        L[v, v] = dv
        for u in adj[v]:
            L[v, u] = -1.0
    return L


def graph_adjacency_matrix(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    A = np.zeros((n, n), dtype=float)
    for v in range(n):
        for u in adj[v]:
            A[v, u] = 1.0
    return A


def ctqw_hamiltonian(adj: List[List[int]],
                     potential: Optional[callable] = None,
                     positions: Optional[np.ndarray] = None,
                     gamma: Optional[float] = None) -> np.ndarray:
    n = len(adj)
    if gamma is None:
        gamma = 1.0 / np.mean([len(adj[v]) for v in range(n)]) if n > 0 else 1.0
    H = gamma * graph_laplacian(adj)
    if potential is not None and positions is not None:
        for v in range(n):
            if positions.ndim == 1:
                H[v, v] += potential(positions[v])
            elif positions.ndim == 2:
                H[v, v] += potential(positions[v, 0], positions[v, 1])
    return ensure_hermitian(H)


def ctqw_hamiltonian_with_marked(adj: List[List[int]],
                                 marked_vertices: List[int],
                                 gamma: float = 1.0) -> np.ndarray:




    raise NotImplementedError("Hole 1: ctqw_hamiltonian_with_marked not implemented")





def unitary_evolution(H: np.ndarray, t: float) -> np.ndarray:
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    eigs, V = np.linalg.eigh(H)

    D = np.exp(-1.0j * eigs * t)
    U = V @ np.diag(D) @ V.conj().T
    return U


def chebyshev_propagator(H: np.ndarray, t: float, order: int = 20) -> np.ndarray:
    n = H.shape[0]
    if n == 0:
        return H.copy()

    eigs = np.linalg.eigvalsh(H)
    E_max = max(np.abs(eigs[0]), np.abs(eigs[-1]))
    if np.isclose(E_max, 0.0):
        return np.eye(n, dtype=complex)
    H_tilde = H / E_max


    from scipy.special import jv
    coeffs = [(-1.0j) ** k * jv(k, E_max * t) for k in range(order + 1)]
    coeffs[0] *= 0.5


    T0 = np.eye(n, dtype=complex)
    T1 = H_tilde.astype(complex)
    U = coeffs[0] * T0 + coeffs[1] * T1
    for k in range(2, order + 1):
        T2 = 2.0 * H_tilde @ T1 - T0
        U += coeffs[k] * T2
        T0, T1 = T1, T2
    return U





def spectral_gap(H: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(H)
    unique = np.unique(np.round(eigs, 12))
    if len(unique) < 2:
        return 0.0
    return float(unique[1] - unique[0])


def eigenstate_localization(eigvec: np.ndarray) -> float:
    probs = np.abs(eigvec) ** 2
    return float(np.sum(probs ** 2))
