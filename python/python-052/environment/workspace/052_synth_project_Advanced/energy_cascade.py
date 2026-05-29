"""
energy_cascade.py
能量串级 (Energy Cascade) 与谱分析模块

科学背景:
在海洋湍流中, 能量通过不同尺度之间的非线性相互作用传递.
准地转湍流的能量串级遵循以下物理定律:

  1. 逆向串级 (Inverse cascade, k < k_f):
     能量从大尺度向更大尺度传递, 对应二维/准地转湍流的特征.
     能谱 E(k) ~ C * \epsilon^{2/3} * k^{-5/3}
     其中 \epsilon 为能量注入率.

  2. 正向串级 (Forward cascade, k > k_f):
     涡度从大尺度向小尺度传递.
     涡度能谱 Z(k) ~ C_z * \eta^{2/3} * k^{-1}
     其中 \eta 为涡度注入率.

能量通量 (Energy flux):
  \Pi(k) = - dE/dt |_{<k} = - \int_0^k T(k') dk'
  其中 T(k) 为非线性转移项.

特征模态分解 (Vertical Mode Decomposition):
  通过 Jacobi 特征值迭代求解垂直结构函数:
    d^2\phi_n/dz^2 + (N^2(z)/c_n^2) \phi_n = 0
  其中 N(z) 为浮力频率, c_n 为 Rossby 变形波速.

本模块实现:
  - 能量谱与涡度谱计算
  - 能量通量 \Pi(k) 诊断
  - 动态规划能量转移路径计数 (from football_dynamic)
  - Jacobi 特征值分解用于垂直模态 (from jacobi_eigenvalue)
  - 谱能量预算分析

融合来源:
- 604_jacobi_eigenvalue: Jacobi 特征值迭代
- 444_football_dynamic: 动态规划用于能量转移路径计数
"""

import numpy as np
from numerics_core import givens_rotation, safe_divide
from typing import Tuple, Optional, List


# ============================================================
# 1. 能量谱与通量诊断
# ============================================================

def compute_radial_energy_spectrum(psih: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                                   Ld: float = 1.0, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从流函数谱计算径向能量谱 E(k).

    能量密度:
      E(k) = (1/2) * (k^2 + 1/L_d^2) * |\hat{ψ}(k)|^2

    对径向波数分箱求和/平均.
    """
    Ny, Nx = psih.shape
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K = np.sqrt(K2)

    energy_density = 0.5 * (K2 + 1.0 / (Ld ** 2)) * np.abs(psih) ** 2

    if n_bins is None:
        n_bins = max(Nx, Ny) // 2

    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk
    E = np.zeros(n_bins)
    count = np.zeros(n_bins)

    k_flat = K.flatten()
    e_flat = energy_density.flatten()
    for i in range(n_bins):
        k_low = i * dk
        k_high = (i + 1) * dk
        mask = (k_flat >= k_low) & (k_flat < k_high)
        if np.any(mask):
            E[i] = np.sum(e_flat[mask])
            count[i] = np.sum(mask)

    E = safe_divide(E, count)
    return k_bins, E


def compute_energy_flux(psih: np.ndarray, q_h: np.ndarray,
                        kx: np.ndarray, ky: np.ndarray,
                        Ld: float = 1.0, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算能量通量 \Pi(k).

    能量方程:
      dE/dt = T + D + F
    其中 T 为非线性转移项, D 为耗散, F 为强迫.

    谱空间能量方程:
      d/dt E(k) = Re[ \hat{ψ}^*(k) * J(ψ,q)_hat(k) ] + dissipation + forcing

    能量通量:
      \Pi(k) = -\int_0^k T(k') dk'
             = -\int_0^k Re[ \hat{ψ}^*(k') * J(ψ,q)_hat(k') ] dk'

    正值表示能量从 <k 传递到 >k (正向串级).
    负值表示能量从 >k 传递到 <k (逆向串级).
    """
    Ny_r, Nx_r = psih.shape
    # 物理空间完整维度
    Nx_phys = 2 * (Nx_r - 1)
    Ny_phys = Ny_r
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K = np.sqrt(K2)

    # 计算 Jacobian 的谱
    dpsi_dx_h = 1j * KX * psih
    dpsi_dy_h = 1j * KY * psih
    dq_dx_h = 1j * KX * q_h
    dq_dy_h = 1j * KY * q_h

    dpsi_dx = np.fft.irfft2(dpsi_dx_h, s=(Ny_phys, Nx_phys))
    dpsi_dy = np.fft.irfft2(dpsi_dy_h, s=(Ny_phys, Nx_phys))
    dq_dx = np.fft.irfft2(dq_dx_h, s=(Ny_phys, Nx_phys))
    dq_dy = np.fft.irfft2(dq_dy_h, s=(Ny_phys, Nx_phys))

    jac_phys = dpsi_dx * dq_dy - dpsi_dy * dq_dx
    jac_h = np.fft.rfft2(jac_phys)

    # 转移谱
    transfer = np.real(np.conj(psih) * jac_h)

    if n_bins is None:
        n_bins = max(Nx_phys, Ny_phys) // 2

    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk

    k_flat = K.flatten()
    t_flat = transfer.flatten()

    # 累积通量
    Pi = np.zeros(n_bins)
    for i in range(n_bins):
        k_cut = k_bins[i]
        mask = k_flat <= k_cut
        if np.any(mask):
            Pi[i] = -np.sum(t_flat[mask])

    return k_bins, Pi


def compute_enstrophy_spectrum(q_h: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                               n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    涡度谱: Z(k) = (1/2) |\hat{q}(k)|^2.
    """
    Ny_r, Nx_r = q_h.shape
    Nx_phys = 2 * (Nx_r - 1)
    Ny_phys = Ny_r
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    z_density = 0.5 * np.abs(q_h) ** 2

    if n_bins is None:
        n_bins = max(Nx_phys, Ny_phys) // 2
    k_max = max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = k_max / n_bins
    k_bins = np.arange(n_bins) * dk + 0.5 * dk
    Z = np.zeros(n_bins)
    count = np.zeros(n_bins)

    k_flat = K.flatten()
    z_flat = z_density.flatten()
    for i in range(n_bins):
        k_low = i * dk
        k_high = (i + 1) * dk
        mask = (k_flat >= k_low) & (k_flat < k_high)
        if np.any(mask):
            Z[i] = np.sum(z_flat[mask])
            count[i] = np.sum(mask)

    Z = safe_divide(Z, count)
    return k_bins, Z


# ============================================================
# 2. 动态规划能量转移路径计数 (from 444_football_dynamic)
# ============================================================

def energy_transfer_path_count(max_scale: int, allowed_steps: List[int] = None) -> np.ndarray:
    """
    使用动态规划计数能量通过不同尺度转移的路径数.

    物理意义:
    在波数空间, 能量从尺度 0 (最大尺度) 向尺度 max_scale (最小尺度)
    传递. 每一步能量可以"跳跃"若干尺度 (对应三波/四波共振条件).

    设 dp[s] 为到达尺度 s 的路径总数:
      dp[0] = 1
      dp[s] = sum_{step in allowed_steps} dp[s - step]

    这与 football_dynamic 的整数分拆计数同构, 但解释为:
    能量级联中从注入尺度到耗散尺度的所有可能非线性相互作用路径数.

    Parameters
    ----------
    max_scale : int
        最大尺度数
    allowed_steps : list of int
        允许的步长 (对应三波共振允许的波数差)

    Returns
    -------
    np.ndarray
        dp[0..max_scale] 路径数
    """
    if max_scale < 0:
        raise ValueError("max_scale must be non-negative")
    if allowed_steps is None:
        # 典型的准地转三波共振: 允许步长 1,2,3 (对应近邻/次近邻相互作用)
        allowed_steps = [1, 2, 3]
    allowed_steps = [s for s in allowed_steps if s > 0]
    if not allowed_steps:
        raise ValueError("allowed_steps must contain positive integers")

    dp = np.zeros(max_scale + 1, dtype=np.int64)
    dp[0] = 1
    for s in range(1, max_scale + 1):
        total = 0
        for step in allowed_steps:
            if s - step >= 0:
                total += dp[s - step]
        dp[s] = total
    return dp


def cascade_path_entropy(dp: np.ndarray) -> float:
    """
    计算能量级联路径的熵 (复杂度度量).

    H = -sum_s p_s log(p_s),  p_s = dp[s] / sum(dp)
    """
    p = dp.astype(float) / np.sum(dp)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# ============================================================
# 3. Jacobi 特征值分解 (from 604_jacobi_eigenvalue)
# ============================================================

def jacobi_eigenvalue(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobi 特征值迭代法: 计算实对称矩阵的全部特征值和特征向量.

    算法步骤:
      1. 初始化 V = I
      2. 迭代寻找最大非对角元 A[p,q]
      3. 计算 Givens 旋转角 θ, 使得旋转后 A'[p,q] = 0
         tau = (A[q,q] - A[p,p]) / (2*A[p,q])
         t = sign(tau) / (|tau| + sqrt(1+tau^2))
         c = 1/sqrt(1+t^2), s = t*c
      4. 更新 A 和 V
      5. 当所有 |A[p,q]| < tol 时收敛

    应用于垂直模态分解:
      将离散化的垂直结构算子 L_{ij} = N^2(z_i) * delta_{ij}
      通过特征值分解得到正交模态.

    Parameters
    ----------
    A : np.ndarray
        NxN 实对称矩阵
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差

    Returns
    -------
    eigvals : np.ndarray
        排序后的特征值 (降序)
    eigvecs : np.ndarray
        对应的特征向量 (列向量)
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.array([])
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    V = np.eye(n)

    for it in range(max_iter):
        # 寻找严格上三角最大元素
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        if max_val < tol:
            break

        # 计算旋转角
        if A[p, p] == A[q, q]:
            theta = np.pi / 4.0
        else:
            tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
            if tau >= 0:
                t = 1.0 / (tau + np.sqrt(1.0 + tau ** 2))
            else:
                t = -1.0 / (-tau + np.sqrt(1.0 + tau ** 2))
            c = 1.0 / np.sqrt(1.0 + t ** 2)
            s = t * c

        # 应用 Givens 旋转
        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0

        for i in range(n):
            if i != p and i != q:
                aip = A[i, p]
                aiq = A[i, q]
                A[i, p] = c * aip - s * aiq
                A[p, i] = A[i, p]
                A[i, q] = s * aip + c * aiq
                A[q, i] = A[i, q]

            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq

    eigvals = np.diag(A)
    # 降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = V[:, idx]
    return eigvals, eigvecs


def vertical_mode_decomposition(N2_profile: np.ndarray, dz: float,
                                f0: float = 1e-4, n_modes: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    垂直模态分解: 求解海洋分层中的垂直结构函数.

    控制方程 (线性化, 刚性顶/底边界):
      d^2\phi/dz^2 + (N^2(z) / c^2) \phi = 0
      d\phi/dz = 0  at z=0, -H

    离散化为矩阵特征值问题:
      L \phi = \lambda \phi,  \lambda = 1/c^2

    其中 L 是二阶差分算子结合 N^2 权重.

    Parameters
    ----------
    N2_profile : np.ndarray
        浮力频率平方 N^2(z), 长度 nz
    dz : float
        垂直网格间距
    f0 : float
        参考 Coriolis 参数
    n_modes : int
        返回的模态数

    Returns
    -------
    c_n : np.ndarray
        Rossby 变形波速 c_n = f0 / sqrt(lambda_n)
    phi_n : np.ndarray
        垂直结构函数 (nz x n_modes)
    Ld_n : np.ndarray
        Rossby 变形半径 Ld_n = c_n / |f0|
    """
    nz = len(N2_profile)
    if nz < 3:
        raise ValueError("N2_profile must have at least 3 points")
    N2 = np.asarray(N2_profile, dtype=float)
    if np.any(N2 < 0):
        raise ValueError("N^2 must be non-negative")

    # 构造二阶差分矩阵 (Neumann 边界)
    # d^2\phi/dz^2 ≈ (\phi_{i+1} - 2\phi_i + \phi_{i-1}) / dz^2
    # 边界: (\phi_1 - \phi_0)/dz = 0, (\phi_{nz-1} - \phi_{nz-2})/dz = 0
    # 简化为内部点方程
    L = np.zeros((nz, nz))
    for i in range(1, nz - 1):
        L[i, i - 1] = 1.0 / dz ** 2
        L[i, i] = -2.0 / dz ** 2
        L[i, i + 1] = 1.0 / dz ** 2
    # 边界行 (反射边界)
    L[0, 0] = -1.0 / dz ** 2
    L[0, 1] = 1.0 / dz ** 2
    L[-1, -2] = 1.0 / dz ** 2
    L[-1, -1] = -1.0 / dz ** 2

    # 广义特征值问题: L \phi + (N^2/c^2) \phi = 0
    # → -L \phi = (N^2/c^2) \phi
    # 乘以 N^{-2}: -N^{-2} L \phi = (1/c^2) \phi
    # 构造对称化矩阵 A = -N^{-1/2} L N^{-1/2}
    N2_safe = np.where(N2 < 1e-15, 1e-15, N2)
    N_sqrt = np.sqrt(N2_safe)
    N_inv_sqrt = 1.0 / N_sqrt

    A = -np.diag(N_inv_sqrt) @ L @ np.diag(N_inv_sqrt)
    # 强制对称化 (消除数值误差)
    A = 0.5 * (A + A.T)

    eigvals, eigvecs = jacobi_eigenvalue(A, max_iter=5000, tol=1e-14)

    # 转换回原始变量
    phi_raw = np.diag(N_inv_sqrt) @ eigvecs

    # 特征值 \lambda = 1/c^2 > 0
    # 过滤负特征值 (数值噪声)
    positive_mask = eigvals > 1e-15
    eigvals = eigvals[positive_mask]
    phi_raw = phi_raw[:, positive_mask]

    c_n = np.sqrt(1.0 / eigvals)
    Ld_n = c_n / abs(f0)

    # 归一化模态
    n_available = phi_raw.shape[1]
    if n_modes is None or n_modes > n_available:
        n_modes = n_available

    phi_n = np.zeros((nz, n_modes))
    for m in range(n_modes):
        norm = np.sqrt(np.trapezoid(phi_raw[:, m] ** 2, dx=dz))
        if norm > 1e-15:
            phi_n[:, m] = phi_raw[:, m] / norm
        else:
            phi_n[:, m] = phi_raw[:, m]

    return c_n[:n_modes], phi_n, Ld_n[:n_modes]


# ============================================================
# 4. 谱预算分析器
# ============================================================

class SpectralBudgetAnalyzer:
    """
    谱能量预算分析器: 追踪能量在不同尺度间的转移、耗散和注入.
    """

    def __init__(self, k_bins: np.ndarray):
        self.k_bins = np.asarray(k_bins)
        self.n_bins = len(k_bins)
        self.E_history = []
        self.Pi_history = []
        self.Z_history = []
        self.t_history = []

    def record(self, t: float, E: np.ndarray, Pi: np.ndarray, Z: np.ndarray):
        """记录当前时刻的谱数据."""
        self.t_history.append(t)
        self.E_history.append(E.copy())
        self.Pi_history.append(Pi.copy())
        self.Z_history.append(Z.copy())

    def compute_time_averaged_spectrum(self, t_start: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算时间平均谱."""
        t_arr = np.array(self.t_history)
        idx = t_arr >= t_start
        if not np.any(idx):
            return self.k_bins, np.zeros(self.n_bins), np.zeros(self.n_bins)
        E_avg = np.mean(np.array(self.E_history)[idx], axis=0)
        Pi_avg = np.mean(np.array(self.Pi_history)[idx], axis=0)
        Z_avg = np.mean(np.array(self.Z_history)[idx], axis=0)
        return self.k_bins, E_avg, Pi_avg

    def compute_cascade_direction(self, E: np.ndarray, Pi: np.ndarray) -> dict:
        """
        判断串级方向.

        逆串级特征: E(k) ~ k^{-5/3}, Pi < 0 (能量向小波数传递)
        正串级特征: Z(k) ~ k^{-1}, Pi > 0 (涡度向大波数传递)
        """
        # 寻找能量通量过零点
        zero_crossings = []
        for i in range(len(Pi) - 1):
            if Pi[i] * Pi[i + 1] < 0:
                zero_crossings.append(self.k_bins[i])

        # 拟合幂律 E ~ k^alpha
        k_safe = np.where(self.k_bins < 1e-10, 1e-10, self.k_bins)
        mask = (E > 1e-15) & (k_safe > 1e-10)
        if np.sum(mask) > 3:
            logk = np.log(k_safe[mask])
            logE = np.log(E[mask])
            alpha = np.polyfit(logk, logE, 1)[0]
        else:
            alpha = np.nan

        return {
            "zero_crossings": zero_crossings,
            "power_law_slope": float(alpha),
            "inverse_cascade_indicator": float(np.mean(Pi[:len(Pi)//2])),
            "forward_cascade_indicator": float(np.mean(Pi[len(Pi)//2:]))
        }


if __name__ == "__main__":
    # 测试 Jacobi 特征值
    A = np.array([[4.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    vals, vecs = jacobi_eigenvalue(A)
    print("Eigenvalues:", vals)
    print("Residual:", np.max(np.abs(A @ vecs - vecs @ np.diag(vals))))

    # 测试垂直模态
    nz = 50
    z = np.linspace(0, -1000, nz)
    N2 = 1e-5 * np.exp(z / 200)  # 指数分层
    c, phi, Ld = vertical_mode_decomposition(N2, abs(z[1]-z[0]), f0=1e-4, n_modes=5)
    print("Vertical modes c_n:", c)
    print("Rossby radii Ld_n (km):", Ld / 1e3)

    # 测试 DP 路径计数
    dp = energy_transfer_path_count(20)
    print("Path counts (first 10):", dp[:10])
    print("Cascade entropy:", cascade_path_entropy(dp))
