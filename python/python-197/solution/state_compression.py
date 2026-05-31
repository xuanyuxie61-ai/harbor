"""
state_compression.py
================================================================================
高性能计算检查点容错：状态向量压缩与谱插值恢复模块

融合原项目：
  - 1189_svd_lls (SVD 最小二乘与低秩分解)
  - 596_interp_trig (三角插值)

科学角色：
  1) 对高维 PDE 状态向量进行 SVD 低秩压缩，显著减少检查点存储；
  2) 利用三角插值在粗网格检查点之间进行谱恢复，降低 I/O 开销；
  3) 计算压缩误差 ||u - \tilde{u}||_2 与截断误差估计。
================================================================================
"""

import numpy as np


# =============================================================================
# SVD 低秩压缩
# =============================================================================
def svd_compress(state: np.ndarray, rank: int) -> tuple:
    """
    对状态矩阵 state (n x m) 进行截断 SVD 压缩，保留 rank 个奇异值。
    返回 (U_r, s_r, Vt_r, compressed_state)。
    压缩比 = (n*m) / (rank*(n+m+1))。
    """
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
    """从截断 SVD 分量重建状态。"""
    return U_r @ np.diag(s_r) @ Vt_r


def optimal_rank(state: np.ndarray, energy_threshold: float = 0.99) -> int:
    """
    根据能量阈值自动选择最优秩:
        sum_{i=1}^{r} sigma_i^2 / sum_{i=1}^{min(n,m)} sigma_i^2 >= energy_threshold
    """
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


# =============================================================================
# 三角插值压缩（一维谱插值）
# =============================================================================
def trigcardinal(xi: float, xj: float, n: int, h: float) -> float:
    """
    三角基函数 tau_j(xi)。
    当 n 为奇数: sin(pi*(xi-xj)/h) / (n * sin(pi*(xi-xj)/(n*h)))
    当 n 为偶数: sin(pi*(xi-xj)/h) / (n * tan(pi*(xi-xj)/(n*h)))
    """
    if abs(xi - xj) < 1.0e-14:
        return 1.0
    arg1 = np.pi * (xi - xj) / h
    arg2 = np.pi * (xi - xj) / (n * h)
    if n % 2 == 1:
        return np.sin(arg1) / (n * np.sin(arg2) + 1.0e-30)
    else:
        return np.sin(arg1) / (n * np.tan(arg2) + 1.0e-30)


def trig_interpolant(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    对均匀分布节点 xd 上的数据 yd 构造三角插值，并在 xi 上求值。
    xd 必须等距，包含 n 个点。
    """
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
    """
    将一维状态向量通过三角插值降采样到 n_coarse 个节点，
    保留粗网格值 (xd_coarse, yd_coarse) 用于后续恢复。
    返回 (xd_coarse, yd_coarse, original_length)。
    """
    state_1d = np.asarray(state_1d, dtype=float)
    N = len(state_1d)
    if n_coarse >= N:
        n_coarse = max(1, N // 2)
    indices = np.linspace(0, N - 1, n_coarse, dtype=int)
    xd_coarse = np.linspace(0.0, 1.0, n_coarse)
    yd_coarse = state_1d[indices].copy()
    return xd_coarse, yd_coarse, N


def reconstruct_state_trig(xd_coarse: np.ndarray, yd_coarse: np.ndarray, N: int) -> np.ndarray:
    """从粗网格三角插值恢复细网格状态。"""
    xi_fine = np.linspace(0.0, 1.0, N)
    return trig_interpolant(xd_coarse, yd_coarse, xi_fine)
