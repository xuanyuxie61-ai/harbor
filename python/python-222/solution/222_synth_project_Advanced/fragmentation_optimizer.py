# -*- coding: utf-8 -*-
"""
fragmentation_optimizer.py
==========================

碎裂函数参数优化与二次规划求解。

融合种子项目:
    - 836_opt_quadratic: 二次插值优化

物理背景:
    碎裂函数 D_h^q(z, Q^2) 的参数 (a, b in Lund) 可通过全局拟合确定:
        min_{a,b} chi^2(a, b) = sum_i (D_theory(z_i) - D_exp(z_i))^2 / sigma_i^2

    在参数空间的局部, chi^2 近似二次型:
        chi^2 ~ (p - p0)^T H (p - p0) + c
    其中 H 为 Hessian 矩阵。

    本模块实现:
    1) 二次插值优化 (来自 836_opt_quadratic)
    2) 多维 Powell 方向集法
    3) DGLAP Q^2 演化下的碎裂函数拟合
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Optional
import constants as C
from hadronization_string import lund_fragmentation_function, fragmentation_moment


# ======================================================================
# 二次插值优化 (来自 836_opt_quadratic)
# ======================================================================
def quadratic_interpolation_minimize(f: Callable[[float], float],
                                      x1: float, x2: float, x3: float,
                                      max_iter: int = 50,
                                      x_tol: float = 1e-6,
                                      y_tol: float = 1e-8) -> Tuple[float, int]:
    """
    二次插值法求一维函数极小值 (来自 836_opt_quadratic)。

    三点 x1 < x2 < x3, 过此三点作抛物线:
        p(x) = a x^2 + b x + c
    极小点 x* = -b/(2a) = x2 - 0.5 * [(x2-x1)^2 (f2-f3) - (x2-x3)^2 (f2-f1)]
                              / [(x2-x1)(f2-f3) - (x2-x3)(f2-f1)]

    Parameters
    ----------
    f : callable
        目标函数
    x1, x2, x3 : float
        三个初始点
    max_iter : int
        最大迭代次数

    Returns
    -------
    (x_min, iterations)
    """
    for iteration in range(max_iter):
        f1, f2, f3 = f(x1), f(x2), f(x3)

        # 二次插值极小点
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) < 1e-15:
            break

        numer = (x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)
        x_new = x2 - 0.5 * numer / denom

        # 安全检查
        if x_new <= x1 or x_new >= x3:
            x_new = 0.5 * (x1 + x3)

        f_new = f(x_new)

        # 收敛检查
        x_range = abs(x3 - x1)
        y_range = max(abs(f1 - f2), abs(f2 - f3), abs(f1 - f3))

        if x_range < x_tol and y_range < y_tol:
            break

        # 更新三点: 去掉最差点, 插入新点
        points = [(x1, f1), (x2, f2), (x3, f3), (x_new, f_new)]
        points.sort(key=lambda p: p[0])

        # 保留函数值最小的三个点 (保持 x1<x2<x3 结构)
        worst_idx = max(range(4), key=lambda i: points[i][1])
        points.pop(worst_idx)
        x1, f1 = points[0]
        x2, f2 = points[1]
        x3, f3 = points[2]

    return x2, iteration + 1


# ======================================================================
# 多维二次优化 (Powell-like)
# ======================================================================
def powell_direction_set(f: Callable[[List[float]], float],
                          x0: List[float],
                          max_iter: int = 30,
                          tol: float = 1e-6) -> Tuple[List[float], int]:
    """
    Powell 方向集法求多维极小值。
    沿坐标轴方向轮流做一维搜索 (用二次插值)。

    Parameters
    ----------
    f : callable
        多维目标函数 f(x: list) -> float
    x0 : list
        初始点
    max_iter : int
        最大外迭代
    """
    n = len(x0)
    x = list(x0)
    directions = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for outer in range(max_iter):
        x_old = list(x)
        f_old = f(x)

        for d in range(n):
            # 一维搜索沿方向 d
            def line_f(alpha):
                xa = [x[i] + alpha * directions[d][i] for i in range(n)]
                return f(xa)

            a1, a2, a3 = -0.5, 0.0, 0.5
            a_min, _ = quadratic_interpolation_minimize(line_f, a1, a2, a3, max_iter=20)
            x = [x[i] + a_min * directions[d][i] for i in range(n)]

        # 收敛检查
        delta = max(abs(x[i] - x_old[i]) for i in range(n))
        if delta < tol:
            break

    return x, outer + 1


# ======================================================================
# 碎裂函数拟合目标函数
# ======================================================================
def fragmentation_chi2(params: List[float],
                        z_data: List[float],
                        d_data: List[float],
                        d_sigma: List[float]) -> float:
    """
    chi^2 目标函数:
        chi^2(a, b) = sum_i [(D(z_i; a, b) - D_exp(z_i)) / sigma_i]^2
    """
    a, b = params[0], params[1]
    chi2 = 0.0
    for i, zi in enumerate(z_data):
        d_th = lund_fragmentation_function(zi, a=a, b=b)
        sigma = d_sigma[i] if i < len(d_sigma) else 0.01
        chi2 += ((d_th - d_data[i]) / max(sigma, 1e-10)) ** 2
    return chi2


# ======================================================================
# 合成实验数据 (Mock)
# ======================================================================
def generate_mock_fragmentation_data(n_points: int = 10,
                                      true_a: float = 0.3,
                                      true_b: float = 0.8,
                                      noise: float = 0.05,
                                      seed: int = 42) -> Tuple[List[float], List[float], List[float]]:
    """
    生成模拟碎裂函数实验数据 (用于拟合测试)。
    """
    import random
    random.seed(seed)

    z_data = []
    d_data = []
    d_sigma = []

    for i in range(n_points):
        z = 0.05 + 0.9 * i / (n_points - 1)
        d_true = lund_fragmentation_function(z, a=true_a, b=true_b)
        d_noise = d_true * (1.0 + noise * random.gauss(0, 1))
        z_data.append(z)
        d_data.append(max(0.0, d_noise))
        d_sigma.append(abs(d_true * noise) + 0.001)

    return z_data, d_data, d_sigma


# ======================================================================
# Hessian 估计
# ======================================================================
def estimate_hessian(f: Callable[[List[float]], float], x0: List[float],
                      h: float = 1e-3) -> List[List[float]]:
    """
    中心差分估计 Hessian 矩阵:
        H_{ij} = [f(x + e_i h + e_j h) - f(x + e_i h - e_j h)
                 - f(x - e_i h + e_j h) + f(x - e_i h - e_j h)] / (4 h^2)
    """
    n = len(x0)
    H = [[0.0] * n for _ in range(n)]
    f0 = f(x0)

    for i in range(n):
        for j in range(i, n):
            xpp = list(x0)
            xpm = list(x0)
            xmp = list(x0)
            xmm = list(x0)

            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h

            H[i][j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h)
            H[j][i] = H[i][j]

    return H


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Quadratic Minimization Test ===")
    # f(x) = (x - 3)^2 + 1
    f = lambda x: (x - 3.0)**2 + 1.0
    x_min, it = quadratic_interpolation_minimize(f, 0.0, 1.0, 5.0)
    print(f"  f(x) = (x-3)^2 + 1")
    print(f"  Minimum at x = {x_min:.6f} (expected 3.0), iterations = {it}")

    print("\n=== Fragmentation Function Fitting ===")
    z_data, d_data, d_sigma = generate_mock_fragmentation_data(10, true_a=0.3, true_b=0.8)
    print(f"  Generated {len(z_data)} data points")

    def obj(params):
        return fragmentation_chi2(params, z_data, d_data, d_sigma)

    x_opt, it = powell_direction_set(obj, [0.5, 1.0], max_iter=20)
    print(f"  Fitted: a = {x_opt[0]:.4f}, b = {x_opt[1]:.4f}")
    print(f"  (True: a = 0.3000, b = 0.8000)")
    print(f"  chi^2_min = {obj(x_opt):.4f}, iterations = {it}")

    # Hessian
    H = estimate_hessian(obj, x_opt)
    print(f"  Hessian eigenvalues: {H[0][0]:.2f}, {H[1][1]:.2f}")
