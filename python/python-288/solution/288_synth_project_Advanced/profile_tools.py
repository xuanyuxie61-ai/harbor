"""
profile_tools.py - 等离子体剖面插值与重构工具

本模块融合以下种子项目的核心算法：
  - 926_pwl_interp_1d → 分段线性插值
  - 077_bernstein_approximation → Bernstein多项式逼近（已在legendre_fd中实现，此处提供接口）

功能：
  1. 分段线性插值（hat函数基）
  2. 三次样条插值
  3. 等离子体密度/温度剖面生成
  4. 靶板热负荷剖面映射
  5. 剖面光滑重建
"""

import numpy as np
from typing import Tuple, Optional, Callable


# =============================================================================
# 分段线性插值（来自926_pwl_interp_1d）
# =============================================================================
def pwl_interp_1d(xd: np.ndarray, yd: np.ndarray,
                    xi: np.ndarray) -> np.ndarray:
    """
    一维分段线性插值

    对于每个求值点xi，找到所在区间[k, k+1]，然后：
      yi = (1-t)*yd[k] + t*yd[k+1]
    其中 t = (xi - xd[k]) / (xd[k+1] - xd[k])

    对于超出范围的值，使用最近端点外推

    参数:
        xd: 数据点x坐标（必须升序）
        yd: 数据点y值
        xi: 求值点
    返回:
        yi: 插值结果
    """
    n = len(xd)
    m = len(xi)
    yi = np.zeros(m)

    for j in range(m):
        x = xi[j]

        # 边界处理
        if x <= xd[0]:
            yi[j] = yd[0]
            continue
        if x >= xd[n - 1]:
            yi[j] = yd[n - 1]
            continue

        # 找到包含x的区间
        k = 0
        for i in range(n - 1):
            if xd[i] <= x <= xd[i + 1]:
                k = i
                break

        # 线性插值
        dx = xd[k + 1] - xd[k]
        if abs(dx) < 1e-30:
            yi[j] = yd[k]
        else:
            t = (x - xd[k]) / dx
            yi[j] = (1.0 - t) * yd[k] + t * yd[k + 1]

    return yi


def pwl_basis_1d(xd: np.ndarray, xi: np.ndarray, node_index: int) -> np.ndarray:
    """
    分段线性基函数（hat函数）求值

    phi_k(x) = (x - x_{k-1})/(x_k - x_{k-1})  if x_{k-1} <= x <= x_k
             = (x_{k+1} - x)/(x_{k+1} - x_k)  if x_k <= x <= x_{k+1}
             = 0                                otherwise

    参数:
        xd: 节点坐标
        xi: 求值点
        node_index: 基函数对应的节点编号
    返回:
        phi: 基函数值
    """
    k = node_index
    n = len(xd)
    m = len(xi)
    phi = np.zeros(m)

    for j in range(m):
        x = xi[j]

        # 左支撑
        if k > 0 and xd[k - 1] <= x <= xd[k]:
            dx = xd[k] - xd[k - 1]
            if abs(dx) > 1e-30:
                phi[j] = (x - xd[k - 1]) / dx

        # 右支撑
        elif k < n - 1 and xd[k] <= x <= xd[k + 1]:
            dx = xd[k + 1] - xd[k]
            if abs(dx) > 1e-30:
                phi[j] = (xd[k + 1] - x) / dx

        # 节点处
        elif abs(x - xd[k]) < 1e-14:
            phi[j] = 1.0

    return phi


# =============================================================================
# 三次样条插值
# =============================================================================
def cubic_spline_coefficients(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    """
    计算自然三次样条的系数

    在每个区间[x_k, x_{k+1}]上:
      S_k(x) = a_k + b_k*(x-x_k) + c_k*(x-x_k)^2 + d_k*(x-x_k)^3

    边界条件: S''(x_0) = S''(x_n) = 0 (自然样条)

    参数:
        xd: 节点坐标
        yd: 节点值
    返回:
        coeffs: shape (n-1, 4) 系数矩阵 [a, b, c, d]
    """
    n = len(xd)
    h = np.diff(xd)

    # 构造三对角系统求c_k
    # h_{k-1}*c_{k-1} + 2*(h_{k-1}+h_k)*c_k + h_k*c_{k+1} = 3*((y_{k+1}-y_k)/h_k - (y_k-y_{k-1})/h_{k-1})
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    A[0, 0] = 1.0  # 自然边界
    A[n - 1, n - 1] = 1.0

    for k in range(1, n - 1):
        A[k, k - 1] = h[k - 1]
        A[k, k] = 2.0 * (h[k - 1] + h[k])
        A[k, k + 1] = h[k]
        rhs[k] = 3.0 * ((yd[k + 1] - yd[k]) / h[k] - (yd[k] - yd[k - 1]) / h[k - 1])

    # 求解
    c = np.linalg.solve(A, rhs)

    # 计算其他系数
    coeffs = np.zeros((n - 1, 4))
    for k in range(n - 1):
        coeffs[k, 0] = yd[k]  # a_k
        coeffs[k, 1] = (yd[k + 1] - yd[k]) / h[k] - h[k] * (2 * c[k] + c[k + 1]) / 3.0  # b_k
        coeffs[k, 2] = c[k]  # c_k
        coeffs[k, 3] = (c[k + 1] - c[k]) / (3.0 * h[k])  # d_k

    return coeffs


def cubic_spline_eval(xd: np.ndarray, coeffs: np.ndarray,
                        xi: np.ndarray) -> np.ndarray:
    """
    三次样条求值

    参数:
        xd: 节点坐标
        coeffs: 样条系数
        xi: 求值点
    返回:
        yi: 插值结果
    """
    n = len(xd)
    m = len(xi)
    yi = np.zeros(m)

    for j in range(m):
        x = xi[j]

        # 边界处理
        if x <= xd[0]:
            x = xd[0]
        if x >= xd[n - 1]:
            x = xd[n - 1] - 1e-14

        # 找区间
        k = 0
        for i in range(n - 1):
            if xd[i] <= x < xd[i + 1]:
                k = i
                break

        dx = x - xd[k]
        a, b, c, d = coeffs[k]
        yi[j] = a + b * dx + c * dx**2 + d * dx**3

    return yi


# =============================================================================
# 等离子体剖面生成
# =============================================================================
def generate_density_profile(r_norm: np.ndarray,
                               n_pedestal: float = 3.0e19,
                               n_core: float = 8.0e19,
                               n_sol: float = 1.0e18,
                               rho_ped: float = 0.9) -> np.ndarray:
    """
    生成典型的托卡马克密度剖面

    剖面形状:
      rho < rho_ped: n = n_core - (n_core - n_pedestal) * (rho/rho_ped)^2
      rho > rho_ped: n = n_pedestal * exp(-(rho - rho_ped) / lambda_SOL)

    参数:
        r_norm: 归一化半径 [0, 1.3]
        n_pedestal: 脚部密度 [m^-3]
        n_core: 芯部密度 [m^-3]
        n_sol: SOL密度 [m^-3]
        rho_ped: 脚部位置
    返回:
        n_profile: 密度剖面 [m^-3]
    """
    n = np.zeros_like(r_norm)
    lambda_sol = 0.05  # SOL衰减长度（归一化）

    for i, rho in enumerate(r_norm):
        if rho <= rho_ped:
            # 芯部：抛物线+常数
            n[i] = n_core - (n_core - n_pedestal) * (rho / rho_ped)**2
        else:
            # SOL：指数衰减
            n[i] = n_pedestal * np.exp(-(rho - rho_ped) / lambda_sol)

        # 物理约束
        n[i] = max(n[i], n_sol * 0.01)

    return n


def generate_temperature_profile(r_norm: np.ndarray,
                                   T_pedestal: float = 2.0,
                                   T_core: float = 10.0,
                                   T_edge: float = 0.05,
                                   rho_ped: float = 0.9) -> np.ndarray:
    """
    生成典型的托卡马克温度剖面 (eV)

    参数:
        r_norm: 归一化半径
        T_pedestal: 脚部温度 [eV]
        T_core: 芯部温度 [eV]
        T_edge: 边缘温度 [eV]
        rho_ped: 脚部位置
    返回:
        T_profile: 温度剖面 [eV]
    """
    T = np.zeros_like(r_norm)
    alpha_T = 2.0  # 芯部剖面指数
    lambda_sol = 0.04

    for i, rho in enumerate(r_norm):
        if rho <= rho_ped:
            # 芯部: T = T_core * (1 - (rho/rho_ped)^alpha)^alpha_T
            T[i] = T_core * (1.0 - (rho / rho_ped)**2)**alpha_T
            T[i] = max(T[i], T_pedestal)
        else:
            # SOL
            T[i] = T_pedestal * np.exp(-(rho - rho_ped) / lambda_sol)

        T[i] = max(T[i], T_edge)

    return T


# =============================================================================
# 靶板热负荷分布映射
# =============================================================================
def map_heat_flux_to_target(R_target: np.ndarray,
                              q_upstream: float,
                              lambda_q: float = 0.003,
                              R_strike: float = 1.5,
                              B_p_B_t: float = 0.1) -> np.ndarray:
    """
    将上游热流映射到偏滤器靶板

    使用指数衰减模型:
      q_target(R) = q_upstream * exp(-(R - R_strike) / lambda_q) / (B_p/B_t)

    其中 lambda_q 是SOL热宽度，B_p/B_t 是极向/环向磁场比

    参数:
        R_target: 靶板径向位置 [m]
        q_upstream: 上游热流 [W/m²]
        lambda_q: SOL热宽度 [m]
        R_strike: 打击点位置 [m]
        B_p_B_t: 磁场比
    返回:
        q_target: 靶板热负荷分布 [W/m²]
    """
    dR = R_target - R_strike

    # 指数衰减（考虑磁场扩张因子）
    q_target = q_upstream * np.exp(-np.abs(dR) / (lambda_q + 1e-30)) / (B_p_B_t + 1e-30)

    # 不对称性（内外偏滤器不同）
    inner_mask = dR < 0
    q_target[inner_mask] *= 0.7  # 内偏滤器通常热负荷较低

    return q_target


# =============================================================================
# 剖面光滑重建（使用Bernstein多项式）
# =============================================================================
def smooth_profile_bernstein(profile: np.ndarray,
                               r_norm: np.ndarray,
                               degree: int = 8) -> np.ndarray:
    """
    使用Bernstein多项式光滑重建等离子体剖面

    Bernstein逼近的优点：
    1. 一致收敛（Weierstrass定理）
    2. 保形性（保持单调性、凸性）
    3. 数值稳定性高（无Runge现象）

    参数:
        profile: 原始剖面
        r_norm: 归一化半径
        degree: Bernstein多项式阶数
    返回:
        smoothed: 光滑后的剖面
    """
    from legendre_fd import bernstein_approximation_vectorized

    # 映射到 [0, 1] 区间
    r_min, r_max = r_norm[0], r_norm[-1]

    smoothed = bernstein_approximation_vectorized(
        profile, r_norm, r_min, r_max
    )

    return smoothed


# =============================================================================
# 剖面诊断量
# =============================================================================
def compute_profile_gradients(profile: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
    """
    计算剖面梯度 dn/dr

    使用中心差分（内部）和单侧差分（边界）

    参数:
        profile: 剖面数据
        r_norm: 坐标
    返回:
        gradient: 梯度数组
    """
    n = len(profile)
    grad = np.zeros(n)

    # 内部点：中心差分
    for i in range(1, n - 1):
        dr = r_norm[i + 1] - r_norm[i - 1]
        if abs(dr) > 1e-30:
            grad[i] = (profile[i + 1] - profile[i - 1]) / dr

    # 边界点
    if n >= 2:
        dr0 = r_norm[1] - r_norm[0]
        if abs(dr0) > 1e-30:
            grad[0] = (profile[1] - profile[0]) / dr0
        drN = r_norm[-1] - r_norm[-2]
        if abs(drN) > 1e-30:
            grad[n - 1] = (profile[-1] - profile[-2]) / drN

    return grad


def compute_profile_curvature(profile: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
    """
    计算剖面曲率 d²n/dr²

    参数:
        profile: 剖面数据
        r_norm: 坐标
    返回:
        curvature: 曲率数组
    """
    n = len(profile)
    curv = np.zeros(n)

    for i in range(1, n - 1):
        dr1 = r_norm[i] - r_norm[i - 1]
        dr2 = r_norm[i + 1] - r_norm[i]

        if abs(dr1) > 1e-30 and abs(dr2) > 1e-30:
            # 非均匀网格的二阶导数
            curv[i] = 2.0 * ((profile[i + 1] - profile[i]) / dr2 -
                               (profile[i] - profile[i - 1]) / dr1) / (dr1 + dr2)

    return curv
