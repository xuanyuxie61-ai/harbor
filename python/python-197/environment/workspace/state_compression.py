
import numpy as np





def svd_compress(state: np.ndarray, rank: int) -> tuple:
    state = np.asarray(state, dtype=float)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    n, m = state.shape
    rank = min(rank, n, m)
    if rank < 1:
        return None, None, None, np.zeros_like(state)
    U, s, Vt = np.linalg.svd(state, full_matrices=False)
    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]
    compressed = U_r @ np.diag(s_r) @ Vt_r
    return U_r, s_r, Vt_r, compressed


def svd_reconstruct(U_r: np.ndarray, s_r: np.ndarray, Vt_r: np.ndarray) -> np.ndarray:
    return U_r @ np.diag(s_r) @ Vt_r


def optimal_rank(state: np.ndarray, energy_threshold: float = 0.99) -> int:
    state = np.asarray(state, dtype=float)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    _, s, _ = np.linalg.svd(state, full_matrices=False)
    total = np.sum(s * s)
    if total == 0.0:
        return 1
    cumsum = np.cumsum(s * s)
    rank = np.searchsorted(cumsum, energy_threshold * total) + 1
    return int(min(rank, len(s)))





def trigcardinal(xi: float, xj: float, n: int, h: float) -> float:
    if abs(xi - xj) < 1.0e-14:
        return 1.0
    arg1 = np.pi * (xi - xj) / h
    arg2 = np.pi * (xi - xj) / (n * h)
    if n % 2 == 1:
        return np.sin(arg1) / (n * np.sin(arg2) + 1.0e-30)
    else:
        return np.sin(arg1) / (n * np.tan(arg2) + 1.0e-30)


def trig_interpolant(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    xi = np.asarray(xi, dtype=float)
    n = len(xd)
    if n < 2:
        return np.zeros_like(xi)
    h = xd[1] - xd[0]
    yi = np.zeros_like(xi)
    for j in range(n):
        for k in range(len(xi)):
            yi[k] += yd[j] * trigcardinal(xi[k], xd[j], n, h)
    return yi


def compress_state_trig(state_1d: np.ndarray, n_coarse: int) -> tuple:
    state_1d = np.asarray(state_1d, dtype=float)
    N = len(state_1d)
    if n_coarse >= N:
        n_coarse = max(1, N // 2)
    indices = np.linspace(0, N - 1, n_coarse, dtype=int)
    xd_coarse = np.linspace(0.0, 1.0, n_coarse)
    yd_coarse = state_1d[indices].copy()
    return xd_coarse, yd_coarse, N


def reconstruct_state_trig(xd_coarse: np.ndarray, yd_coarse: np.ndarray, N: int) -> np.ndarray:
    xi_fine = np.linspace(0.0, 1.0, N)
    return trig_interpolant(xd_coarse, yd_coarse, xi_fine)
