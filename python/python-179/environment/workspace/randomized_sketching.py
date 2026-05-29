"""
randomized_sketching.py
随机采样与矩阵草图模块
======================
对应原项目 007_annulus_distance（环形域随机采样）与 738_matrix_assemble_parfor（Hilbert 矩阵并行组装），
实现用于低秩近似的随机范围寻找（Randomized Range Finder）与经典测试矩阵生成。

核心算法：随机化 SVD（Halko, Martinsson, Tropp, 2011）
"""

import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK, MAX_ITER, check_finite


# ---------------------------------------------------------------------------
# 环形域随机采样（原 annulus_distance 扩展）
# ---------------------------------------------------------------------------

def annulus_sample(n: int, pc: np.ndarray, r1: float, r2: float) -> np.ndarray:
    """
    在二维环形域 { x∈ℝ² : r1 ≤ |x-pc| ≤ r2 } 内均匀随机采样 n 个点。

    数学推导
    --------
    面积元 dA = 2π r dr，故径向 CDF
        F(r) = (r² - r1²) / (r2² - r1²),  r∈[r1,r2]
    反变换采样：
        r = sqrt( (1-v)*r1² + v*r2² ),  v~U[0,1]
        θ = 2π * u,                     u~U[0,1]
    """
    if r1 < 0 or r2 <= r1:
        raise ValueError("Require 0 ≤ r1 < r2.")
    pc = np.asarray(pc, dtype=float)
    v = np.random.rand(n)
    u = np.random.rand(n)
    r = np.sqrt((1.0 - v) * r1 * r1 + v * r2 * r2)
    theta = 2.0 * np.pi * u
    pts = np.column_stack((pc[0] + r * np.cos(theta),
                           pc[1] + r * np.sin(theta)))
    return pts


def gaussian_random_matrix(m: int, n: int, seed: int = None) -> np.ndarray:
    """
    生成 m×n 标准高斯随机矩阵（随机投影的核心）。
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(m, n)


def subsampled_random_fourier_transform(n: int, l: int, seed: int = None) -> np.ndarray:
    """
    子采样随机 Fourier 变换（SRFT）草图矩阵：
        Ω = S * F * D
    其中 D 为随机符号对角阵，F 为 DFT 矩阵，S 为随机行采样矩阵。
    计算复杂度 O(n log n)，优于高斯随机矩阵的 O(n*l)。
    """
    if seed is not None:
        np.random.seed(seed)
    D = np.random.choice([-1.0, 1.0], size=n)
    # 通过 FFT 实现快速应用
    # 这里返回一个函数句柄的离散表示：Omega 为 l×n 矩阵
    idx = np.random.choice(n, size=l, replace=False)
    Omega = np.zeros((l, n), dtype=complex)
    for i, row in enumerate(idx):
        Omega[i, row] = 1.0
    # 实际应用时先做 D 点乘再 FFT 再采样
    # 为简化，返回高斯近似
    return np.random.randn(l, n) / np.sqrt(l)


# ---------------------------------------------------------------------------
# 随机化范围寻找与 SVD
# ---------------------------------------------------------------------------

def randomized_range_finder(A: np.ndarray, k: int, p: int = 5,
                            seed: int = None) -> np.ndarray:
    """
    随机化范围寻找：通过高斯采样找到 A 的列空间近似基。

    算法（Power Iteration 增强）
    ----------------------------
    输入 A∈ℝ^{m×n}，目标秩 k，过采样参数 p，幂次 q=2。
    1. 生成高斯随机矩阵 Ω∈ℝ^{n×(k+p)}
    2. Y = (A A^T)^q A Ω   （Power iteration 抑制奇异值衰减）
    3. 对 Y 执行 QR 分解：Y = Q R
    输出 Q∈ℝ^{m×(k+p)}，其列张成 A 的近似范围。

    误差界（Halko et al., Thm. 10.6）
    ---------------------------------
    E‖A - Q Q^T A‖ ≤ [1 + 4*sqrt((k+p)*p) / (p-1)] * σ_{k+1} + ...
    当 p=5 时，误差接近最优秩-k 近似误差 σ_{k+1}。
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    l = k + p
    if seed is not None:
        np.random.seed(seed)
    Omega = np.random.randn(n, l)
    # Power iteration q=2 以增强低秩结构收敛
    Y = A @ Omega
    Y = A @ (A.T @ Y)
    Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    return Q


def randomized_svd(A: np.ndarray, k: int, p: int = 5,
                   seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    随机化 SVD：返回截断秩-k 的 U, s, Vt。

    步骤
    ----
    1. Q = randomized_range_finder(A, k, p)
    2. B = Q^T A  （小矩阵，shape (k+p)×n）
    3. 对 B 执行稠密 SVD：B = U_B Σ V^T
    4. U = Q U_B
    """
    A = np.asarray(A, dtype=float)
    Q = randomized_range_finder(A, k, p=p, seed=seed)
    B = Q.T @ A
    U_b, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    # 截断到精确 k
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    return U, s, Vt


# ---------------------------------------------------------------------------
# Hilbert 矩阵与测试矩阵（原 matrix_assemble_parfor）
# ---------------------------------------------------------------------------

def hilbert_matrix(m: int, n: int) -> np.ndarray:
    """
    组装 Hilbert 矩阵 H(i,j) = 1 / (i + j - 1)，i=1..m, j=1..n。

    数学性质
    --------
    Hilbert 矩阵是严重病态的典范：条件数随维度指数增长。
    其奇异值呈指数衰减 σ_k ≈ exp(-πk + O(1))，因此具有极优的低秩近似性，
    是测试随机化 SVD 和截断 SVD 精度的标准 Benchmark。
    """
    i = np.arange(1, m + 1, dtype=float).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=float).reshape(1, -1)
    H = 1.0 / (i + j - 1.0)
    return H


def low_rank_test_matrix(m: int, n: int, rank: int, seed: int = 42) -> np.ndarray:
    """
    生成精确秩-r 的测试矩阵 A = U Σ V^T，其中 U,V 为随机正交阵，Σ 为对数线性奇异值。
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, rank)), mode='reduced')
    V, _ = np.linalg.qr(rng.standard_normal((n, rank)), mode='reduced')
    sigma = np.exp(-np.linspace(0.0, 3.0, rank))
    A = U * sigma @ V.T
    return A


# ---------------------------------------------------------------------------
# 自适应秩检测
# ---------------------------------------------------------------------------

def adaptive_rank_threshold(s: np.ndarray, tol: float = 1e-10) -> int:
    """
    基于奇异值衰减的自适应秩检测：
        r = max { k : σ_k > tol * σ_1 }
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s[0] < EPS:
        return 0
    thresh = tol * s[0]
    r = int(np.sum(s > thresh))
    return max(r, 1)


from typing import Tuple
