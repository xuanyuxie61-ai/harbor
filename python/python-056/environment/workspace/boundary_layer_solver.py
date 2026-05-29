"""
boundary_layer_solver.py
================================================================================
病态边值问题求解模块 (来源于 572_ill_bvp 项目)
================================================================================
本模块实现病态边值问题 (BVP) 的数值求解，特别针对潮汐能提取系统中
涡轮叶片和支撑结构周围的薄边界层问题。当雷诺数极高时，边界层
方程呈现病态特性（小参数乘以最高阶导数），需要特殊的数值处理。

核心公式:
    模型方程 (奇异摄动边值问题):
        ε y'' - x y' + y = 0,   x ∈ [-1, 1]
        y(-1) = 2,   y(1) = 1

    其中 ε << 1 为小的摄动参数，代表边界层厚度尺度。

    打靶法 (Shooting Method):
        将 BVP 转化为初值问题 (IVP):
            y' = z
            z' = (x z - y) / ε
        通过 Newton-Raphson 迭代调整 y'(-1) 使得 y(1) = 1。

    有限差分法 (紧凑格式):
        在内部点 i 使用中心差分:
            ε (y_{i-1} - 2y_i + y_{i+1})/h² - x_i (y_{i+1} - y_{i-1})/(2h) + y_i = 0

    边界层厚度估计:
        δ ∝ √ε
"""

import numpy as np
from typing import Callable, Tuple


def solve_ivp_rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    四阶 Runge-Kutta 方法求解常微分方程组。

    参数:
        f: 右端项函数 f(t, y)
        y0: 初始条件
        t_span: (t0, tf)
        n_steps: 步数

    返回:
        (t, y): 时间网格和解矩阵
    """
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    m = len(y0)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for i in range(n_steps):
        k1 = f(t[i], y[i, :])
        k2 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, y[i, :] + 0.5 * h * k2)
        k4 = f(t[i] + h, y[i, :] + h * k3)
        y[i + 1, :] = y[i, :] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, y


def shooting_method(
    epsilon: float,
    ya: float,
    yb: float,
    a: float = -1.0,
    b: float = 1.0,
    n_shoot: int = 1000,
    tol: float = 1e-8,
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    使用打靶法求解病态边值问题 ε y'' - x y' + y = 0。

    参数:
        epsilon: 摄动参数
        ya: 左边界条件 y(a)
        yb: 右边界条件 y(b)
        a, b: 区间端点
        n_shoot: 射击步数
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        (x, y, converged)
    """
    def ode(t: float, Y: np.ndarray) -> np.ndarray:
        dY = np.zeros(2)
        dY[0] = Y[1]
        if abs(epsilon) < 1e-14:
            dY[1] = 0.0
        else:
            dY[1] = (t * Y[1] - Y[0]) / epsilon
        return dY

    # 初始猜测导数
    s = (yb - ya) / (b - a)

    for it in range(max_iter):
        Y0 = np.array([ya, s])
        t, Y = solve_ivp_rk4(ode, Y0, (a, b), n_shoot)
        phi = Y[-1, 0] - yb
        if abs(phi) < tol:
            return t, Y[:, 0], True

        # 数值微分求 Jacobian
        ds = 1e-6 * max(abs(s), 1.0)
        Y0p = np.array([ya, s + ds])
        _, Yp = solve_ivp_rk4(ode, Y0p, (a, b), n_shoot)
        dphi_ds = (Yp[-1, 0] - Y[-1, 0]) / ds

        if abs(dphi_ds) < 1e-14:
            break
        s = s - phi / dphi_ds

    Y0 = np.array([ya, s])
    t, Y = solve_ivp_rk4(ode, Y0, (a, b), n_shoot)
    return t, Y[:, 0], False


def finite_difference_bvp(
    epsilon: float,
    ya: float,
    yb: float,
    a: float = -1.0,
    b: float = 1.0,
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用中心差分求解病态 BVP。

    离散方程:
        (ε/h² - x_i/(2h)) y_{i-1} + (-2ε/h² + 1) y_i
        + (ε/h² + x_i/(2h)) y_{i+1} = 0

    参数:
        epsilon: 摄动参数
        ya, yb: 边界值
        a, b: 区间
        n: 内部网格点数

    返回:
        (x, y)
    """
    h = (b - a) / (n + 1)
    x = np.linspace(a + h, b - h, n)

    # 构造三对角矩阵 (中心差分)
    # ε(y_{i-1} - 2y_i + y_{i+1})/h² - x_i(y_{i+1} - y_{i-1})/(2h) + y_i = 0
    # => (ε/h² + x_i/(2h)) y_{i-1} + (-2ε/h² + 1) y_i + (ε/h² - x_i/(2h)) y_{i+1} = 0
    main = np.full(n, -2.0 * epsilon / (h * h) + 1.0)
    lower = epsilon / (h * h) + x / (2.0 * h)   # 系数 for y_{i-1}
    upper = epsilon / (h * h) - x / (2.0 * h)   # 系数 for y_{i+1}

    # 边界条件修正
    rhs = np.zeros(n)
    rhs[0] -= lower[0] * ya
    rhs[-1] -= upper[-1] * yb

    # Thomas 算法求解三对角系统
    y = _thomas_algorithm(lower, main, upper, rhs)

    x_full = np.concatenate(([a], x, [b]))
    y_full = np.concatenate(([ya], y, [yb]))
    return x_full, y_full


def _thomas_algorithm(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Thomas 算法求解三对角系统。

    参数:
        lower: 下次对角线 (长度 n，lower[0] 不使用)
        main: 主对角线 (长度 n)
        upper: 上次对角线 (长度 n，upper[-1] 不使用)
        rhs: 右端项

    返回:
        解向量
    """
    n = len(main)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]

    for i in range(1, n):
        denom = main[i] - lower[i] * c_prime[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def boundary_layer_thickness(epsilon: float, U_ref: float = 1.0, L_ref: float = 1.0) -> float:
    """
    估计边界层厚度。

    公式:
        δ/L ≈ √(ν / (U·L)) = √ε
        其中 ε = ν/(U·L) 为逆雷诺数

    参数:
        epsilon: 逆雷诺数
        U_ref: 参考流速
        L_ref: 参考长度

    返回:
        边界层厚度 (m)
    """
    return L_ref * np.sqrt(epsilon)


def compute_blade_boundary_layer(
    Re: float,
    chord_length: float = 2.0,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算潮汐涡轮叶片表面的边界层速度分布。

    物理模型:
        将叶片表面坐标映射到 [-1, 1]，求解奇异摄动 BVP。

    参数:
        Re: 雷诺数
        chord_length: 弦长 (m)
        n_points: 离散点数

    返回:
        (x, u): 表面坐标和无量纲速度
    """
    epsilon = 1.0 / max(Re, 1.0)
    x, y, _ = shooting_method(epsilon, ya=0.0, yb=1.0, a=-1.0, b=1.0, n_shoot=n_points)
    # 将无量纲解映射到物理坐标
    x_phys = chord_length * 0.5 * (x + 1.0)
    u_norm = np.clip(y, 0.0, 1.0)
    return x_phys, u_norm
