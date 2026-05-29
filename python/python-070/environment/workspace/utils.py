"""
utils.py
通用科学计算工具模块
包含数值稳定性处理、边界条件判定、常数定义等
"""

import numpy as np


class NumericalConfig:
    """
    全局数值计算配置类
    控制收敛阈值、迭代次数上限、机器精度等
    """
    EPS = np.finfo(float).eps
    EPS_SQRT = np.sqrt(EPS)
    TOL = 1e-12
    MAX_ITER = 10000
    R8_BIG = 1.0e+30


def safe_divide(a, b, default=0.0):
    """
    安全除法，避免除以零
    当 |b| < EPS 时返回 default
    """
    if np.isscalar(a) and np.isscalar(b):
        if abs(b) < NumericalConfig.EPS:
            return default
        return a / b
    b = np.asarray(b)
    result = np.zeros_like(a, dtype=float)
    mask = np.abs(b) >= NumericalConfig.EPS
    result[mask] = np.asarray(a)[mask] / b[mask]
    result[~mask] = default
    return result


def bound_value(x, x_min, x_max):
    """
    将数值限制在 [x_min, x_max] 区间内
    """
    return np.clip(x, x_min, x_max)


def is_nearly_equal(a, b, tol=None):
    """
    判断两个浮点数是否在数值精度范围内相等
    """
    if tol is None:
        tol = NumericalConfig.TOL
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))


def golden_ratio():
    """
    返回黄金分割率 φ = (1 + sqrt(5)) / 2
    """
    return 0.5 * (1.0 + np.sqrt(5.0))


def inverse_golden_ratio():
    """
    返回黄金分割率的倒数 1/φ = (sqrt(5) - 1) / 2
    也等于 φ - 1
    """
    return 0.5 * (np.sqrt(5.0) - 1.0)


def legendre_polynomial(n, x):
    """
    计算归一化 Legendre 多项式 P_n(x) 在 x 处的值
    使用三项递推关系：
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    初值：P_0(x) = 1, P_1(x) = x
    """
    x = np.asarray(x)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return x.astype(float)
    p0 = np.ones_like(x, dtype=float)
    p1 = x.astype(float)
    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2
    return p1


def solve_tridiagonal(a, b, c, d):
    """
    Thomas 算法求解三对角线性系统
    矩阵形式：
        [b0  c0   0   ...   0   ]
        [a1  b1  c1   ...   0   ]
        [ 0  a2  b2   ...   0   ]
        ...
        [ 0   0   0  ... a_{n-1} b_{n-1}]
    输入：
        a: 下对角线，长度为 n-1，a[0] 无效
        b: 主对角线，长度为 n
        c: 上对角线，长度为 n-1，c[n-1] 无效
        d: 右端项，长度为 n
    返回：
        x: 解向量，长度为 n
    """
    n = len(b)
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("Tridiagonal arrays must have consistent lengths")
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < NumericalConfig.EPS:
            denom = NumericalConfig.EPS * np.sign(denom) if denom != 0 else NumericalConfig.EPS
        cp[i] = safe_divide(c[i], denom, 0.0)
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def gauss_legendre_3point():
    """
    返回 3 点 Gauss-Legendre  quadrature 的节点和权重
    在参考区间 [-1, 1] 上精确积分 5 次多项式
    """
    abscissa = np.array([
        -0.774596669241483377035853079956,
         0.000000000000000000000000000000,
         0.774596669241483377035853079956
    ], dtype=float)
    weight = np.array([
        0.555555555555555555555555555556,
        0.888888888888888888888888888889,
        0.555555555555555555555555555556
    ], dtype=float)
    return abscissa, weight
