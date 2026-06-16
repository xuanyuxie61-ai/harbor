# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================

有限差分格式的稳定性分析模块

本模块对暗物质相空间输运方程的离散化格式进行严格的稳定性分析,
包括:
  1. Von Neumann 稳定性分析 (Fourier 空间)
  2. CFL (Courant-Friedrichs-Lewy) 条件计算
  3. 矩阵谱半径分析
  4. 刚性系统特征值分解 (对应种子项目 1222 VAR 稳定性)

科学公式
--------
Von Neumann 放大因子:
  对于线性扩散方程 ∂u/∂t = D ∂²u/∂x²:
    2阶: g(k) = 1 - 4r sin²(kh/2),  稳定条件: r = D dt/h² ≤ 1/2
    4阶: g(k) = 1 - (r/3)(4 sin(kh/2) - sin(kh))²

  对于对流方程 ∂u/∂t + v ∂u/∂x = 0:
    CFL 条件: |v| dt / h ≤ 1  (2阶)
    RK4 + 4阶空间: CFL ≈ 2.83 / (1 + D dt/h²)

矩阵稳定性:
  离散算子 A 的特征值谱 {λ_i}:
    稳定 ⟺ Re(λ_i) ≤ 0 对所有 i
    强稳定 ⟺ Re(λ_i) < -δ < 0
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# 第一部分: Von Neumann 稳定性分析
# ---------------------------------------------------------------------------

def von_neumann_diffusion_2nd(r: np.ndarray, k: np.ndarray, h: float) -> np.ndarray:
    """
    2阶扩散方程 Von Neumann 放大因子:
      g(k) = 1 - 4r sin²(kh/2)
    其中 r = D dt / h²

    稳定性: |g(k)| ≤ 1 ⟺ r ≤ 1/2
    """
    return 1.0 - 4.0 * r * np.sin(k * h / 2.0)**2


def von_neumann_diffusion_4th(r: np.ndarray, k: np.ndarray, h: float) -> np.ndarray:
    """
    4阶扩散方程 Von Neumann 放大因子:
      g(k) = 1 - (r/3) * [4 sin(kh) - sin(2kh)]²
             实际上等价于使用 4 阶 Laplacian 符号
    """
    # 4阶 Laplacian 的 Fourier 符号:
    # L_4(k) = (-1 + 16 cos(kh) - cos(2kh) - 14) / (6h²)
    #        = (16 cos(kh) - cos(2kh) - 15) / (6h²)
    L4_symbol = (16.0 * np.cos(k * h) - np.cos(2.0 * k * h) - 15.0) / (6.0 * h**2)
    # g = 1 + r * h² * L4_symbol (注意 L4 ≤ 0)
    g = 1.0 + r * h**2 * L4_symbol
    return g


def von_neumann_advection_4th(cfl: float, k: np.ndarray, h: float) -> np.ndarray:
    """
    4阶迎风 + RK4 组合的 Von Neumann 放大因子。

    空间算子 Fourier 符号 (4阶中心差分一阶导数):
      D_4(k) = i * (8 sin(kh/2) - sin(kh)) / (6h)

    RK4 时间积分:
      g = 1 + z + z²/2 + z³/6 + z⁴/24
    其中 z = -v dt * D_4(k) = -i cfl * (8 sin(kh/2) - sin(kh)) / 6
    """
    z_real = 0.0
    z_imag = -cfl * (8.0 * np.sin(k * h / 2.0) - np.sin(k * h)) / 6.0
    # RK4 放大因子: |g|² = |1 + z + z²/2 + z³/6 + z⁴/24|²
    # 对于纯虚数 z = iβ:
    # g = 1 - β²/2 + β⁴/24 + i(β - β³/6)
    beta = z_imag
    g_real = 1.0 - beta**2 / 2.0 + beta**4 / 24.0
    g_imag = beta - beta**3 / 6.0
    return np.sqrt(g_real**2 + g_imag**2)


def cfl_condition(
    v_max: float,
    D: float,
    h: float,
    scheme: str = 'rk4_fd4',
) -> Dict[str, float]:
    """
    计算 CFL 稳定性条件, 返回最大允许时间步长 dt_max。

    对于组合对流-扩散方程:
      ∂u/∂t + v ∂u/∂x = D ∂²u/∂x²

    各格式的 CFL 条件:
      显式 Euler + 2阶空间: dt ≤ h² / (2D) 且 dt ≤ h / |v|
      RK4 + 4阶空间: dt ≤ 2.83 h / |v| 且 dt ≤ 0.6 h² / D
      RK4 + 6阶空间: dt ≈ 2.37 h / |v| (对流部分更严格)
    """
    if h <= 0:
        raise ValueError(f"步长 h 必须 > 0, 得到 h={h}")

    results = {'v_max': v_max, 'D': D, 'h': h}

    # 对流 CFL
    if v_max > 0:
        if scheme == 'euler_fd2':
            cfl_adv = 1.0
        elif scheme == 'rk4_fd4':
            cfl_adv = 2.83  # RK4 稳定性域与虚轴交点
        elif scheme == 'rk4_fd6':
            cfl_adv = 2.37
        else:
            cfl_adv = 1.0
        dt_adv = cfl_adv * h / v_max
        results['cfl_advective'] = cfl_adv
        results['dt_advective'] = dt_adv
    else:
        results['dt_advective'] = np.inf

    # 扩散 CFL
    if D > 0:
        if scheme == 'euler_fd2':
            cfl_diff = 0.5
        elif scheme == 'rk4_fd4':
            cfl_diff = 0.6  # 近似
        elif scheme == 'rk4_fd6':
            cfl_diff = 0.55
        else:
            cfl_diff = 0.5
        dt_diff = cfl_diff * h**2 / D
        results['cfl_diffusive'] = cfl_diff
        results['dt_diffusive'] = dt_diff
    else:
        results['dt_diffusive'] = np.inf

    results['dt_max'] = min(
        results.get('dt_advective', np.inf),
        results.get('dt_diffusive', np.inf)
    )

    return results


# ---------------------------------------------------------------------------
# 第二部分: 矩阵谱稳定性
# ---------------------------------------------------------------------------

def build_diffusion_matrix_1d(N: int, D: float, h: float, order: int = 4) -> np.ndarray:
    """
    构建 1D 扩散算子的离散矩阵 (Dirichlet 边界)。

    2阶:
      A = (D/h²) * tridiag(1, -2, 1)
    4阶:
      A = (D/(12h²)) * pentadiag(-1, 16, -30, 16, -1)
    """
    A = np.zeros((N, N))
    if order == 2:
        coeff = D / (h * h)
        for i in range(N):
            A[i, i] = -2.0 * coeff
            if i > 0:
                A[i, i-1] = coeff
            if i < N - 1:
                A[i, i+1] = coeff
    elif order == 4:
        coeff = D / (12.0 * h * h)
        for i in range(N):
            A[i, i] = -30.0 * coeff
            if i >= 1:
                A[i, i-1] = 16.0 * coeff
            if i >= 2:
                A[i, i-2] = -1.0 * coeff
            if i < N - 1:
                A[i, i+1] = 16.0 * coeff
            if i < N - 2:
                A[i, i+2] = -1.0 * coeff
    else:
        raise ValueError(f"order 必须为 2 或 4, 得到 {order}")
    return A


def spectral_stability(A: np.ndarray, dt: float) -> Dict[str, float]:
    """
    分析矩阵 A 的谱稳定性。

    对于时间推进 du/dt = A u + b:
      - 特征值 {λ_i} = eig(A)
      - 稳定 ⟺ Re(λ_i) ≤ 0
      - 时间离散稳定 ⟺ |1 + dt λ_i| ≤ 1 (Euler)
                        或 |R(dt λ_i)| ≤ 1 (RK4)

    返回: {max_real_part, max_imag_part, spectral_radius, dt_max_euler, dt_max_rk4}
    """
    eigenvalues = np.linalg.eigvals(A)
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)
    abs_vals = np.abs(eigenvalues)

    max_real = float(np.max(real_parts))
    min_real = float(np.min(real_parts))
    max_abs = float(np.max(abs_vals))
    max_imag = float(np.max(np.abs(imag_parts)))

    # Euler 稳定: dt ≤ 2 / |λ_max_real| (如果全负)
    dt_euler = 2.0 / max_abs if max_abs > 0 else np.inf
    # RK4 稳定: 在实轴上 |R(z)| ≤ 1 for z ∈ [-2.785, 0]
    dt_rk4 = 2.785 / max_abs if max_abs > 0 else np.inf

    return {
        'max_real_part': max_real,
        'min_real_part': min_real,
        'max_abs_eigenvalue': max_abs,
        'max_imag_part': max_imag,
        'is_stable_continuous': max_real <= 1e-10,
        'dt_max_euler': dt_euler,
        'dt_max_rk4': dt_rk4,
        'n_eigenvalues': len(eigenvalues),
    }


# ---------------------------------------------------------------------------
# 第三部分: 耦合 ODE 系统稳定性 (源自 VAR 稳定性分析)
# ---------------------------------------------------------------------------

def var_stability_matrix(
    coupling_matrices: list,
    N: int,
    max_lag: int,
) -> np.ndarray:
    """
    构建 VAR(p) 模型的稳定性伴随矩阵。

    对于 VAR(p): X_t = Σ_{j=1}^{p} Φ_j X_{t-j} + ε_t
    稳定性矩阵:
      | Φ_1  Φ_2  ...  Φ_{p-1}  Φ_p |
      | I    0    ...  0        0    |
      | 0    I    ...  0        0    |
      | ...                          |
      | 0    0    ...  I        0    |

    稳定 ⟺ 所有特征值模 < 1
    """
    stab = np.zeros((N * max_lag, N * max_lag))
    for j in range(max_lag):
        stab[:N, j*N:(j+1)*N] = coupling_matrices[j]
        if j < max_lag - 1:
            stab[(j+1)*N:(j+2)*N, j*N:(j+1)*N] = np.eye(N)

    return stab


def check_var_stationarity(
    coupling_matrices: list,
    N: int,
    max_lag: int,
) -> Dict:
    """
    检查 VAR(p) 模型的平稳性 (源自 1222 Bagged Time Series Causality)。

    返回: {stationary, max_eigenvalue_modulus, eigenvalues}
    """
    stab = var_stability_matrix(coupling_matrices, N, max_lag)
    eigvals = np.linalg.eigvals(stab)
    mod_eigvals = np.abs(eigvals)
    max_mod = float(np.max(mod_eigvals))
    return {
        'stationary': max_mod < 1.0,
        'max_eigenvalue_modulus': max_mod,
        'eigenvalues': eigvals,
        'spectral_radius': max_mod,
    }


# ---------------------------------------------------------------------------
# 第四部分: DM 输运方程稳定性
# ---------------------------------------------------------------------------

def dm_transport_stability(
    v_flow: float,
    D_phase: float,
    sigma_scatter: float,
    rho_dm: float,
    h: float,
    N: int,
    order: int = 4,
) -> Dict:
    """
    暗物质相空间输运方程的线性稳定性分析。

    方程形式:
      ∂f/∂t + v ∂f/∂x = D ∂²f/∂x² - σ n f + S

    其中:
      v         = DM 流速
      D         = 相空间扩散系数
      σ n       = 散射率
      S         = 源项

    离散矩阵 A = A_adv + A_diff + A_scat
    其中 A_scat = -σ n I (对角)
    """
    # 对流矩阵 (4阶中心差分, 反对称)
    A_adv = np.zeros((N, N))
    for i in range(N):
        for k in [-2, -1, 1, 2]:
            j = i + k
            if 0 <= j < N:
                if k == -2:
                    A_adv[i, j] = v_flow / (12.0 * h)
                elif k == -1:
                    A_adv[i, j] = -8.0 * v_flow / (12.0 * h)
                elif k == 1:
                    A_adv[i, j] = 8.0 * v_flow / (12.0 * h)
                elif k == 2:
                    A_adv[i, j] = -v_flow / (12.0 * h)

    # 扩散矩阵
    A_diff = build_diffusion_matrix_1d(N, D_phase, h, order)

    # 散射矩阵 (对角)
    A_scat = -sigma_scatter * rho_dm * np.eye(N)

    # 总矩阵
    A_total = A_adv + A_diff + A_scat

    # 谱分析
    stab = spectral_stability(A_total, dt=0.0)

    # CFL 条件
    cfl = cfl_condition(abs(v_flow), D_phase, h, scheme='rk4_fd4')

    return {
        'matrix_A': A_total,
        'spectral_analysis': stab,
        'cfl_condition': cfl,
        'dt_recommended': 0.5 * min(
            stab.get('dt_max_rk4', np.inf),
            cfl.get('dt_max', np.inf)
        ),
    }
