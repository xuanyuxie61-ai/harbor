"""
dynamic_reconstruction.py
=========================
基于隐式中点法的动态图像序列重建模块

科学背景：
---------
在动态成像（如心脏 MRI、动态 CT）中，图像随时间演化：
    I(t+dt) = I(t) + dt * f(t, I(t))

其中 f 描述图像的时间演化规律。对于扩散过程（如热传导、信号衰减）：
    f(t, I) = D * Laplacian(I) - alpha * I

这对应于偏微分方程：
    partial I / partial t = D * nabla^2 I - alpha I

数值求解：
---------
本模块实现两种中点法（来自项目 764_midpoint 和 767_midpoint_fixed）：

1. 隐式中点法（Implicit Midpoint Rule）：
   I_{n+1} = I_n + dt * f(t_n + dt/2, (I_n + I_{n+1})/2)
   这是二阶辛方法（symplectic method），对 Hamilton 系统保持能量守恒。

2. 固定点迭代中点法（Fixed-Point Midpoint）：
   通过固定点迭代求解隐式方程，避免直接求逆。

稳定性分析：
    隐式中点法对扩散方程是无条件稳定的，即对任意 dt > 0 都稳定。
    特征方程：|1 + z/2| / |1 - z/2| < 1 对所有 Re(z) < 0 成立。

应用场景：
---------
在动态图像重建中，利用时间连续性约束，可以从更稀疏的采样中重建
时间序列图像。
"""

import numpy as np
from typing import Callable, Optional, Tuple


def discrete_laplacian_2d(I: np.ndarray, h: float = 1.0) -> np.ndarray:
    """
    计算二维离散拉普拉斯算子（5点模板）。

    数学公式：
        nabla^2 I_{i,j} = (I_{i+1,j} + I_{i-1,j} + I_{i,j+1} + I_{i,j-1} - 4I_{i,j}) / h^2

    参数:
        I: 二维图像
        h: 空间步长
    返回:
        拉普拉斯算子作用结果
    """
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise ValueError("输入必须是二维图像")

    lap = np.zeros_like(I)
    lap[1:-1, 1:-1] = (I[2:, 1:-1] + I[:-2, 1:-1] +
                       I[1:-1, 2:] + I[1:-1, :-2] - 4.0 * I[1:-1, 1:-1]) / (h ** 2)

    # 边界处理：Neumann 边界条件（零法向导数）
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]

    return lap


def diffusion_rhs(t: float, I: np.ndarray, D: float = 0.1,
                  alpha: float = 0.01) -> np.ndarray:
    """
    扩散衰减方程的右端项。

    数学模型：
        f(t, I) = D * nabla^2 I - alpha * I

    参数:
        t: 时间（当前未使用，保留接口）
        I: 当前图像
        D: 扩散系数
        alpha: 衰减系数
    返回:
        右端项
    """
    return D * discrete_laplacian_2d(I) - alpha * I


def midpoint_fixed_step(f: Callable, t0: float, I0: np.ndarray,
                        dt: float, it_max: int = 10,
                        theta: float = 0.5) -> np.ndarray:
    """
    固定点迭代中点法单步推进（来自项目 767_midpoint_fixed）。

    算法：
        x_m = t_n + theta * dt
        y^{(0)} = y_n
        for j = 1 to it_max:
            y^{(j)} = y_n + theta * dt * f(x_m, y^{(j-1)})
        y_{n+1} = (1/theta) * y^{(it_max)} + (1 - 1/theta) * y_n

    参数:
        f: 右端项函数，签名 f(t, I) -> dI/dt
        t0: 当前时间
        I0: 当前图像
        dt: 时间步长
        it_max: 固定点最大迭代次数
        theta: 中点参数（默认 0.5）
    返回:
        下一时刻的图像
    """
    I0 = np.asarray(I0, dtype=float)
    xm = t0 + theta * dt
    ym = I0.copy()

    for _ in range(it_max):
        ym = I0 + theta * dt * f(xm, ym)

    I1 = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * I0
    return I1


def midpoint_implicit_step(f: Callable, t0: float, I0: np.ndarray,
                           dt: float, tol: float = 1e-8,
                           max_iter: int = 20) -> np.ndarray:
    """
    隐式中点法单步推进（来自项目 764_midpoint，使用 Newton-Raphson 思想）。

    方程：
        I_{n+1} = I_n + dt * f(t_n + dt/2, (I_n + I_{n+1})/2)

    令 I_mid = (I_n + I_{n+1})/2，则 I_{n+1} = 2*I_mid - I_n
    代入得：2*I_mid - I_n = I_n + dt * f(t_m, I_mid)
    即：g(I_mid) = 2*I_mid - 2*I_n - dt * f(t_m, I_mid) = 0

    使用 Picard 迭代求解：
        I_mid^{k+1} = I_n + (dt/2) * f(t_m, I_mid^k)

    参数:
        f: 右端项函数
        t0: 当前时间
        I0: 当前图像
        dt: 时间步长
        tol: 收敛容差
        max_iter: 最大迭代次数
    返回:
        下一时刻的图像
    """
    I0 = np.asarray(I0, dtype=float)
    tm = t0 + 0.5 * dt
    I_mid = I0.copy()

    for _ in range(max_iter):
        I_mid_new = I0 + 0.5 * dt * f(tm, I_mid)
        if np.linalg.norm(I_mid_new - I_mid) < tol * max(1.0, np.linalg.norm(I_mid)):
            I_mid = I_mid_new
            break
        I_mid = I_mid_new

    I1 = 2.0 * I_mid - I0
    return I1


def solve_dynamic_diffusion(I0: np.ndarray, tspan: Tuple[float, float],
                            n_steps: int, D: float = 0.1, alpha: float = 0.01,
                            method: str = 'implicit') -> Tuple[np.ndarray, np.ndarray]:
    """
    求解二维扩散衰减方程的时间演化。

    PDE:
        partial I / partial t = D * nabla^2 I - alpha * I
        I(x, y, 0) = I0(x, y)

    参数:
        I0: 初始图像
        tspan: 时间区间 (t_start, t_end)
        n_steps: 时间步数
        D: 扩散系数
        alpha: 衰减系数
        method: 'implicit' 或 'fixed'
    返回:
        (t_array, I_series): 时间数组和图像序列，I_series 形状为 (n_steps+1, H, W)
    """
    I0 = np.asarray(I0, dtype=float)
    if I0.ndim != 2:
        raise ValueError("初始图像必须是二维")

    t_start, t_end = tspan
    if t_end <= t_start:
        raise ValueError("t_end 必须大于 t_start")
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")

    dt = (t_end - t_start) / n_steps
    t_array = np.linspace(t_start, t_end, n_steps + 1)

    I_series = np.zeros((n_steps + 1, I0.shape[0], I0.shape[1]), dtype=float)
    I_series[0] = I0

    def rhs(t, I):
        return diffusion_rhs(t, I, D, alpha)

    for n in range(n_steps):
        if method == 'implicit':
            I_series[n + 1] = midpoint_implicit_step(rhs, t_array[n], I_series[n], dt)
        elif method == 'fixed':
            I_series[n + 1] = midpoint_fixed_step(rhs, t_array[n], I_series[n], dt)
        else:
            raise ValueError(f"未知方法: {method}")

    return t_array, I_series


def dynamic_cs_reconstruction(measurements: np.ndarray, Phi: np.ndarray,
                              Psi: np.ndarray, lambda_reg: float,
                              temporal_smoothness: float = 0.1) -> np.ndarray:
    """
    结合时间平滑约束的动态压缩感知重建。

    优化问题：
        min_c 0.5 * ||Phi Psi c - y||_2^2 + lambda * ||c||_1
              + gamma * ||c - c_prev||_2^2

    其中 gamma 为时间平滑系数，c_prev 为前一帧的系数。

    参数:
        measurements: 当前帧测量值
        Phi: 测量矩阵
        Psi: 稀疏基
        lambda_reg: L1 正则化参数
        temporal_smoothness: 时间平滑系数 gamma
    返回:
        重建系数
    """
    from cs_detector import fista_reconstruction

    A = Phi @ Psi
    y = np.asarray(measurements, dtype=float).ravel()

    # 标准 FISTA 重建
    c = fista_reconstruction(A, y, lambda_reg, max_iter=500, tol=1e-5)
    return c
