"""
serial_quadrature.py
================================================================================
数值积分模块 (来源于 944_quad_serial 项目)
================================================================================
本模块提供一维数值积分算法，用于计算潮汐能提取中的能量通量、
功率输出以及结构响应的功。在海洋流体力学中，数值积分用于
离散化后的体积力、面积力和能量积分。

核心公式:
    - 复合梯形法则:
        ∫_a^b f(x) dx ≈ (b-a)/N · Σ_{i=1}^{N} f(x_i)

    - Simpson 法则:
        ∫_a^b f(x) dx ≈ h/3 · [f(x_0) + 4Σ f(x_{odd}) + 2Σ f(x_{even}) + f(x_N)]

    - 能量通量积分:
        P = ∫_A ½ ρ |u|³ dA
"""

import numpy as np
from typing import Callable


def quad_serial(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """
    使用复合梯形法则估计定积分。

    参数:
        f: 被积函数，接受数组返回数组
        a: 积分下限
        b: 积分上限
        n: 采样点数

    返回:
        积分估计值
    """
    if n < 2:
        raise ValueError("quad_serial: n 必须至少为 2")
    x = np.linspace(a, b, n)
    fx = f(x)
    h = (b - a) / (n - 1)
    return h * (0.5 * fx[0] + np.sum(fx[1:-1]) + 0.5 * fx[-1])


def quad_simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """
    使用复合 Simpson 法则估计定积分。

    公式:
        ∫_a^b f(x) dx ≈ h/3 [f_0 + 4(f_1+f_3+...) + 2(f_2+f_4+...) + f_n]

    参数:
        f: 被积函数
        a, b: 积分限
        n: 采样点数 (必须为 2k+1，即偶数个子区间)

    返回:
        积分估计值
    """
    if n < 3 or n % 2 == 0:
        n = n + 1 if n % 2 == 0 else n
        if n < 3:
            n = 3
    x = np.linspace(a, b, n)
    fx = f(x)
    h = (b - a) / (n - 1)
    return h / 3.0 * (fx[0] + 4.0 * np.sum(fx[1:-1:2]) + 2.0 * np.sum(fx[2:-1:2]) + fx[-1])


def quad_adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 20,
) -> float:
    """
    自适应 Simpson 积分。

    参数:
        f: 被积函数
        a, b: 积分限
        tol: 容差
        max_depth: 最大递归深度

    返回:
        积分估计值
    """
    def _recurse(left: float, right: float, fl: float, fm: float, fr: float, depth: int) -> float:
        mid = 0.5 * (left + right)
        m1 = 0.5 * (left + mid)
        m2 = 0.5 * (mid + right)
        f1 = f(m1)
        f2 = f(m2)

        h = right - left
        whole = h / 6.0 * (fl + 4.0 * fm + fr)
        left_part = h / 12.0 * (fl + 4.0 * f1 + fm)
        right_part = h / 12.0 * (fm + 4.0 * f2 + fr)
        total = left_part + right_part

        if depth >= max_depth or abs(total - whole) <= 15.0 * tol:
            return total + (total - whole) / 15.0

        return (_recurse(left, mid, fl, f1, fm, depth + 1)
                + _recurse(mid, right, fm, f2, fr, depth + 1))

    mid = 0.5 * (a + b)
    return _recurse(a, b, f(a), f(mid), f(b), 0)


def integrate_power_density(velocity: np.ndarray, rho: float = 1025.0) -> float:
    """
    计算水动能功率密度的时间平均值。

    公式:
        P_avg = 1/T ∫_0^T ½ ρ |u(t)|³ dt

    参数:
        velocity: 流速时间序列 (m/s)
        rho: 水密度 (kg/m³)，默认 1025

    返回:
        平均功率密度 (W/m²)
    """
    u = np.asarray(velocity, dtype=float)
    if u.size == 0:
        return 0.0
    if u.size == 1:
        return 0.5 * rho * abs(u[0]) ** 3
    power = 0.5 * rho * np.abs(u) ** 3
    # 梯形积分
    return np.trapezoid(power) / (power.size - 1) if power.size > 1 else power[0]


def integrate_structural_work(force: np.ndarray, displacement: np.ndarray) -> float:
    """
    计算结构响应的总功 (力-位移曲线下的面积)。

    公式:
        W = ∫ F · dx

    参数:
        force: 力数组
        displacement: 位移数组

    返回:
        功 (J)
    """
    return np.trapezoid(force, displacement)
