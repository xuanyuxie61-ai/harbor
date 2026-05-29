"""
================================================================================
数值工具模块 (utils_numerical.py)
================================================================================
融合项目:
  - 034_asa082 (detq): 正交矩阵行列式计算
  - 094_bisection: 二分法求根

本模块提供可压缩CFD求解器所需的底层数值工具，包括：
  1. 正交矩阵行列式验证（几何守恒律）
  2. 二分法非线性方程求根（状态方程反解）
  3. 数值稳定性判断与边界处理
================================================================================
"""

import numpy as np


def detq_orthogonal(a: np.ndarray, n: int) -> tuple:
    """
    计算正交矩阵的行列式 (基于 Algorithm AS 82, Gower 1975)

    在CFD中用于验证坐标变换Jacobian矩阵的正交性，确保几何守恒律满足：

        det(J) = 1  ⇒  网格映射保持体积守恒

    数学推导:
        对于正交矩阵 Q，有 Q^T Q = I。
        算法通过 Householder 反射将 Q 化为对角形式，
        行列式符号由反射次数的奇偶性决定。

    参数:
        a: 正交矩阵 (N x N)
        n: 矩阵阶数

    返回:
        d: 行列式值 (+1 或 -1)
        ifault: 错误码 (0=无错误)
    """
    ifault = 0
    tol = 1e-10

    if n <= 0:
        return 0.0, 1

    a2 = a.flatten().copy()
    d = 1.0
    r = 0

    for k in range(1, n + 1):
        q = r
        x = a2[r]
        y = np.sign(x)
        d *= y
        y = -1.0 / (x + y)
        x = abs(x) - 1.0

        if tol < abs(x):
            if x > 0:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault

            for i in range(k, n):
                q += n
                x = a2[q] * y
                p = r
                s = q
                for j in range(k, n):
                    p += 1
                    s += 1
                    a2[s] += x * a2[p]

        r += n + 1

    # 数值稳定性：强制归一化到 ±1
    if abs(abs(d) - 1.0) > tol:
        d = np.sign(d)

    return d, ifault


def bisection_root_find(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 100) -> tuple:
    """
    二分法求非线性方程 f(x)=0 的根

    在可压缩CFD中用于求解状态方程的隐式反问题，例如由总能 E 反解温度 T：

        E = ρ c_v T + (1/2) ρ (u² + v²)
        ⇒  T = f^{-1}(E)  需用二分法迭代求解

    收敛性:
        区间长度按几何级数衰减: |b_k - a_k| = |b_0 - a_0| / 2^k
        满足 k ≥ log₂(|b_0 - a_0| / ε) 时达到精度 ε

    参数:
        f: 目标函数
        a, b: 有根区间端点 (需满足 f(a)·f(b) < 0)
        tol: 容差
        max_iter: 最大迭代次数

    返回:
        root: 近似根
        it: 实际迭代次数
        converged: 是否收敛
    """
    fa = f(a)
    fb = f(b)

    # 边界检查
    if fa == 0.0:
        return float(a), 0, True
    if fb == 0.0:
        return float(b), 0, True
    if fa * fb > 0.0:
        # 尝试扩展区间
        for scale in [2.0, 5.0, 10.0, 50.0, 100.0]:
            b_new = b + scale * abs(b - a)
            fb_new = f(b_new)
            if fa * fb_new <= 0.0:
                b = b_new
                fb = fb_new
                break
        else:
            # 若仍无符号变化，返回中点并标记未收敛
            return (a + b) / 2.0, 0, False

    it = 0
    while abs(b - a) > tol and it < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        it += 1

        if fc == 0.0:
            return float(c), it, True
        elif np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    root = (a + b) / 2.0
    converged = it < max_iter or abs(b - a) <= tol
    return float(root), it, converged


def safe_sqrt(x: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """带保护的平方根，防止负值导致NaN"""
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """带保护除法，防止除零"""
    return a / np.where(np.abs(b) < eps, np.sign(b + eps) * eps, b)


def check_cfl(dx: float, dy: float, u: float, v: float, c: float, nu: float, CFL_max: float = 0.8) -> float:
    """
    计算满足CFL条件的最大时间步长

    对于可压缩NS方程，CFL条件为：

        Δt ≤ CFL · min( Δx / (|u| + c), Δy / (|v| + c), Δx² / (4ν), Δy² / (4ν) )

    其中 c = √(γ p / ρ) 为声速，ν = μ / ρ 为运动粘性系数。
    """
    dt_conv_x = dx / (abs(u) + c + 1e-14)
    dt_conv_y = dy / (abs(v) + c + 1e-14)
    dt_visc_x = dx * dx / (4.0 * nu + 1e-14)
    dt_visc_y = dy * dy / (4.0 * nu + 1e-14)

    dt = CFL_max * min(dt_conv_x, dt_conv_y, dt_visc_x, dt_visc_y)
    return float(dt)


def limiter_minmod(r: np.ndarray, theta: float = 2.0) -> np.ndarray:
    """
    Minmod限制器，用于MUSCL重构抑制激波附近的数值振荡

        φ(r) = max(0, min(θr, (1+r)/2, θ))

    当 θ=1 时为标准minmod，θ=2 时为MC限制器。
    """
    return np.maximum(0.0, np.minimum(np.minimum(theta * r, (1.0 + r) / 2.0), theta))
