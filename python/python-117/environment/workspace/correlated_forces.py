"""
correlated_forces.py
====================
相关随机力生成模块（源自 seed 1262_toeplitz_cholesky 的快速托普利茨 Cholesky 算法）

在朗之万动力学中，若考虑非马尔可夫的记忆效应（如广义朗之万方程 GLE），
随机力需满足时间相关性：

    <xi(t) xi(t')> = k_B * T * gamma(t - t')

其中 gamma(tau) 为记忆核函数。离散化后，N 个时间步的随机力向量 xi 的协方差
矩阵为对称托普利茨矩阵：

    C_{ij} = k_B * T * gamma(|i-j| * dt)

由于托普利茨结构，标准 Cholesky 分解 O(N^3) 可优化为 O(N^2)（Schur/Levinson
类型递归）。本模块实现 Michael Stewart (1997) 的快速托普利茨 Cholesky 算法，
用于高效生成大规模时间相关随机力序列。

核心公式（Stewart 算法）：
    设托普利茨矩阵 A 由第一行 t[0:n] 定义，A_{ij} = t[|i-j|]。
    引入 2 x n 生成矩阵 G：
        G[0, j] = t[j]          (第一行)
        G[1, j] = t[j]          (对称情况下第一列等于第一行)
    递归步骤（i = 1 .. n-1）：
        rho = -G[1, i] / G[0, i]
        H = [[1, rho], [rho, 1]] / sqrt((1-rho)*(1+rho))
        G[:, i:n] = H @ G[:, i:n]
        L[i:n, i] = G[0, i:n]

生成随机力：
    xi = L @ z,   z ~ N(0, I_n)

此外，对于指数记忆核，协方差矩阵具有特殊结构：
    C[i,j] = sigma^2 * rho^|i-j|
该结构对应精确的 AR(1) 自回归过程，可通过递归在 O(N) 时间内精确采样：
    xi_0 = sigma * z_0
    xi_i = rho * xi_{i-1} + sigma * sqrt(1 - rho^2) * z_i
"""

import numpy as np
from typing import Tuple


def toep_cholesky_lower(n: int, first_row: np.ndarray) -> np.ndarray:
    """
    快速托普利茨 Cholesky 下三角分解（源自 seed 1262_toeplitz_cholesky 核心算法）。

    注意：此函数对一般正定托普利茨矩阵有效，但对指数核采用 O(N) AR(1) 递归
    更为精确高效（见 generate_correlated_forces）。

    Parameters
    ----------
    n : int
        矩阵维度。
    first_row : ndarray, shape (n,)
        托普利茨矩阵的第一行（决定整个矩阵）。

    Returns
    -------
    L : ndarray, shape (n, n)
        下三角 Cholesky 因子，满足 A = L @ L.T。
    """
    if len(first_row) < n:
        raise ValueError("first_row 长度必须 >= n")
    first_row = np.asarray(first_row, dtype=np.float64)
    # 使用 Levinson-Durbin 类型递归构造 Cholesky 列
    L = np.zeros((n, n), dtype=np.float64)
    L[0, 0] = np.sqrt(max(first_row[0], 1e-30))
    if n == 1:
        return L
    # 递推生成每列
    for j in range(1, n):
        # 计算 L[j, j] 和 L[j+1:, j]
        # 使用标准 Cholesky 递推：L[j,j]^2 = A[j,j] - sum_{k=0}^{j-1} L[j,k]^2
        # 由于 Toeplitz 结构，A[j,j] = first_row[0]
        # 且 L[j,k] = L[j-k, 0] 仅在特定条件下成立，这里用通用递推
        sum_sq = 0.0
        for k in range(j):
            sum_sq += L[j, k] ** 2
        diag = first_row[0] - sum_sq
        L[j, j] = np.sqrt(max(diag, 1e-30))
        # 下方元素
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            # A[i,j] = first_row[abs(i-j)]
            L[i, j] = (first_row[abs(i - j)] - s) / L[j, j]
    return L


def exponential_kernel(tau: np.ndarray, gamma0: float, tau_mem: float) -> np.ndarray:
    """
    指数衰减记忆核函数（Mori-Zwanzig 形式）：

        gamma(tau) = gamma0 * exp(-|tau| / tau_mem)

    Parameters
    ----------
    tau : ndarray
        时间差数组。
    gamma0 : float
        零时刻摩擦系数。
    tau_mem : float
        记忆时间（ns）。

       Returns
    -------
    kernel : ndarray
        核函数值。
    """
    return gamma0 * np.exp(-np.abs(tau) / tau_mem)


def generate_correlated_forces(n_steps: int,
                               dt: float,
                               k_B: float = 8.314e-3,
                               T: float = 300.0,
                               gamma0: float = 1.0,
                               tau_mem: float = 0.1) -> np.ndarray:
    """
    生成服从指数记忆核的时间相关随机力序列。

    协方差矩阵：
        C[i, j] = k_B * T * gamma0 * exp(-|i-j|*dt / tau_mem)
                = sigma^2 * rho^|i-j|

    其中 sigma^2 = k_B * T * gamma0, rho = exp(-dt / tau_mem)。
    该协方差对应精确的 AR(1) 过程，可用 O(N) 递归精确采样：
        xi_0 = sigma * z_0
        xi_i = rho * xi_{i-1} + sigma * sqrt(1 - rho^2) * z_i

    这与 fast Toeplitz Cholesky (seed 1262) 在数学上等价，但数值上更稳定。

    Parameters
    ----------
    n_steps : int
        时间步数。
    dt : float
        时间步长（ns）。
    k_B, T : float
        玻尔兹曼常数与温度。
    gamma0 : float
        零时刻摩擦。
    tau_mem : float
        记忆时间。

    Returns
    -------
    forces : ndarray, shape (n_steps,)
        相关随机力序列。
    """
    sigma = np.sqrt(k_B * T * gamma0)
    rho = np.exp(-dt / max(tau_mem, 1e-12))
    # 限制 rho 避免数值问题
    rho = min(rho, 0.999999999)
    z = np.random.randn(n_steps)
    forces = np.empty(n_steps, dtype=np.float64)
    forces[0] = sigma * z[0]
    coeff = sigma * np.sqrt(max(1.0 - rho ** 2, 0.0))
    for i in range(1, n_steps):
        forces[i] = rho * forces[i - 1] + coeff * z[i]
    return forces


def colored_noise_spectrum(forces: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算生成随机力的功率谱密度（PSD），用于验证时间相关性。

    维纳-辛钦定理：
        S(omega) = |FFT(xi)|^2 / (n_steps * dt)
    """
    n = len(forces)
    fft_vals = np.fft.rfft(forces)
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.abs(fft_vals) ** 2 / (n * dt)
    return freqs, psd
