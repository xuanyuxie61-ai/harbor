"""
quadrature_integrator.py
数值积分与误差估计模块

包含：
- Vandermonde矩阵求积权重计算（源自 quadrature_weights_vandermonde）
- 金字塔形区域单积分（源自 pyramid01_integral）
- 高维数值积分误差分析

科学背景：
在热-电耦合模拟中，需要计算热源项的空间积分：
    Q_total = integral_Omega q(x,y) dOmega

对于规则区域，使用Vandermonde矩阵构造高精度Newton-Cotes求积公式；
对于金字塔形散热区域（芯片封装中的典型几何），使用精确的单积分公式。
"""

import numpy as np
from utils import binomial_coeff


def vandermonde_quadrature_weights(n, a, b, x):
    """
    通过Vandermonde矩阵求解求积权重。
    源自 quadrature_weights_vandermonde。

    给定 n 个求积点 x_i ∈ [a, b]，要求权重 w_i 使得
        sum_i w_i * x_i^{k-1} = integral_a^b x^{k-1} dx = (b^k - a^k)/k
    对 k = 1,...,n 精确成立。

    这导出了线性系统 V * w = rhs，其中
        V_{k,i} = x_i^{k-1}
        rhs_k = (b^k - a^k) / k

    参数:
        n: int, 求积点数
        a, b: float, 积分区间
        x: ndarray, shape (n,), 求积点

    返回:
        w: ndarray, shape (n,), 求积权重
    """
    x = np.array(x, dtype=float).flatten()
    if x.size != n:
        raise ValueError("x must have length n")
    v = np.zeros((n, n), dtype=float)
    v[0, :] = 1.0
    for i in range(1, n):
        v[i, :] = v[i - 1, :] * x
    rhs = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        rhs[i - 1] = (b ** i - a ** i) / i
    # 使用numpy的线性求解器
    w = np.linalg.solve(v, rhs)
    return w


def pyramid_monomial_integral(expon):
    """
    单位金字塔区域上的单积分。
    源自 pyramid01_integral。

    单位金字塔定义：
        -(1-z) <= x <= 1-z
        -(1-z) <= y <= 1-z
                 0 <= z <= 1

    积分:
        I = integral_{pyramid} x^{expon[0]} y^{expon[1]} z^{expon[2]} dV

    解析解：
        若 expon[0] 或 expon[1] 为奇数，则 I = 0。
        否则:
            i_hi = 2 + expon[0] + expon[1]
            S = sum_{i=0}^{i_hi} (-1)^i * C(i_hi, i) / (i + expon[2] + 1)
            I = S * 2/(expon[0]+1) * 2/(expon[1]+1)

    参数:
        expon: iterable of 3 ints, 各方向指数

    返回:
        float, 积分值
    """
    e0, e1, e2 = int(expon[0]), int(expon[1]), int(expon[2])
    value = 0.0
    if (e0 % 2 == 0) and (e1 % 2 == 0):
        i_hi = 2 + e0 + e1
        s = 0.0
        for i in range(i_hi + 1):
            s += ((-1) ** i) * binomial_coeff(i_hi, i) / (i + e2 + 1)
        value = s * (2.0 / (e0 + 1)) * (2.0 / (e1 + 1))
    return float(value)


def pyramid_volume():
    """
    单位金字塔体积 = 4/3.
    """
    return 4.0 / 3.0


def composite_quadrature_2d(func, xl, xr, yb, yt, nx, ny):
    """
    2D复合Newton-Cotes求积（用于FEM后处理中的热功率积分）。

    将 [xl,xr]×[yb,yt] 划分为 nx×ny 个子矩形，
    每个子矩形上使用2×2点高斯求积。

    参数:
        func: callable, func(x,y) -> float
        xl, xr, yb, yt: float, 区域边界
        nx, ny: int, 子区域划分数

    返回:
        float, 积分近似值
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    hx = (xr - xl) / nx
    hy = (yt - yb) / ny
    # 1D Gauss-Legendre 2-point weights and nodes on [-1,1]
    gl_nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gl_weights = np.array([1.0, 1.0])
    total = 0.0
    for i in range(nx):
        x0 = xl + i * hx
        for j in range(ny):
            y0 = yb + j * hy
            # 映射到 [x0, x0+hx] 和 [y0, y0+hy]
            for xi, wi in zip(gl_nodes, gl_weights):
                x = x0 + hx * 0.5 * (xi + 1.0)
                for eta, wj in zip(gl_nodes, gl_weights):
                    y = y0 + hy * 0.5 * (eta + 1.0)
                    total += wi * wj * func(x, y) * hx * 0.5 * hy * 0.5
    return float(total)


def estimate_quadrature_error(func, xl, xr, yb, yt, n1, n2):
    """
    通过Richardson外推估计数值积分误差。

    计算 I_{n1} 和 I_{n2}，估计误差:
        err ≈ |I_{n2} - I_{n1}| / (1 - (n1/n2)^p)
    对于2点Gauss求积，p = 4。
    """
    i1 = composite_quadrature_2d(func, xl, xr, yb, yt, n1, n1)
    i2 = composite_quadrature_2d(func, xl, xr, yb, yt, n2, n2)
    p = 4.0
    if n2 == n1:
        return 0.0
    factor = 1.0 - (n1 / n2) ** p
    err_est = abs(i2 - i1) / abs(factor)
    return float(err_est)
