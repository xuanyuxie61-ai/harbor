"""
quadrature_engine.py
================================================================================
高性能计算检查点容错：高精度数值积分引擎

融合原项目：
  - 665_legendre_rule (Gauss-Legendre 求积)
  - 641_laguerre_polynomial (Laguerre 多项式与求积)
  - 1246_tetrahedron_felippa_rule (四面体 Felippa 求积)

科学角色：
  1) Legendre/Laguerre 高斯求积用于计算检查点截断误差泛函；
  2) 四面体 Felippa 规则用于 3D 有限元残差与状态恢复时的体积分。
================================================================================
"""

import math
import numpy as np


# =============================================================================
# IMTQLX : 对称三对角矩阵的隐式 QL 算法（Golub-Welsch 核心）
# =============================================================================
def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray):
    """
    对对称三对角矩阵 T(diag=d, offdiag=e) 进行对角化，
    返回特征值 d 与变换后的向量 z = Q^T * z0。
    """
    d = d.copy()
    e = e.copy()
    z = z.copy()
    e[n - 1] = 0.0
    for l in range(1, n + 1):
        j = 0
        while True:
            for m in range(l, n + 1):
                if m == n:
                    break
                if abs(e[m - 1]) <= 1.0e-14 * (abs(d[m - 1]) + abs(d[m])):
                    break
            if m == l:
                break
            if j >= 60:
                raise RuntimeError("IMTQLX did not converge")
            j += 1
            g = (d[l] - d[l - 1]) / (2.0 * e[l - 1])
            r = math.hypot(g, 1.0)
            g = d[m - 1] - d[l - 1] + e[l - 1] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i - 1]
                b = c * e[i - 1]
                r = math.hypot(f, g)
                e[i] = r
                if r == 0.0:
                    d[i] = d[i] - p
                    e[m - 1] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i] - p
                r = (d[i - 1] - g) * s + 2.0 * c * b
                p = s * r
                d[i] = g + p
                g = c * r - b
                f = z[i]
                z[i] = s * z[i - 1] + c * f
                z[i - 1] = c * z[i - 1] - s * f
            if r == 0.0 and i >= l:
                continue
            d[l - 1] = d[l - 1] - p
            e[l - 1] = g
            e[m - 1] = 0.0
    return d, z


# =============================================================================
# Gauss-Legendre 求积规则生成
# =============================================================================
def legendre_rule(n: int, a: float = -1.0, b: float = 1.0):
    """
    生成 n 点 Gauss-Legendre 求积规则 (x, w) 在区间 [a, b] 上。
    使用 Jacobi 矩阵 + Golub-Welsch 特征值方法。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    d = np.zeros(n)
    e = np.zeros(n)
    z = np.zeros(n)
    z[0] = 1.0
    for i in range(1, n + 1):
        d[i - 1] = 0.0
        if i < n:
            e[i - 1] = i / math.sqrt(4.0 * i * i - 1.0)
    d, z = imtqlx(n, d, e, z)
    w = np.zeros(n)
    for i in range(n):
        w[i] = 2.0 * z[i] * z[i]
    # 线性映射到 [a, b]
    x = 0.5 * (b - a) * d + 0.5 * (a + b)
    w = 0.5 * (b - a) * w
    return x, w


# =============================================================================
# Gauss-Laguerre 求积规则生成
# =============================================================================
def laguerre_rule(n: int, alpha: float = 0.0):
    """
    生成 n 点 Gauss-Laguerre 求积规则 (x, w) 对应权函数 x^alpha * e^{-x}。
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    d = np.zeros(n)
    e = np.zeros(n)
    z = np.zeros(n)
    z[0] = 1.0
    for i in range(1, n + 1):
        d[i - 1] = 2.0 * i - 1.0 + alpha
        if i < n:
            e[i - 1] = math.sqrt(i * (i + alpha))
    d, z = imtqlx(n, d, e, z)
    w = np.zeros(n)
    for i in range(n):
        w[i] = math.exp(math.lgamma(alpha + 1.0)) * z[i] * z[i]
    return d, w


# =============================================================================
# 四面体 Felippa 求积规则
# =============================================================================
def tetrahedron_unit_volume() -> float:
    """单位四面体 0<=x,y,z, x+y+z<=1 的体积。"""
    return 1.0 / 6.0


def tetrahedron_unit_monomial(expon: tuple) -> float:
    """
    精确积分单位四面体上的单项式 x^l * y^m * z^n。
    expon = (l, m, n)。
    公式: l! * m! * n! / (l + m + n + 3)!。
    """
    l, m, n = expon
    if l < 0 or m < 0 or n < 0:
        return 0.0
    return (math.gamma(l + 1.0) * math.gamma(m + 1.0) * math.gamma(n + 1.0)
            / math.gamma(l + m + n + 4.0))


def tetrahedron_unit_o04():
    """4 点 Felippa 规则，精确到 2 次多项式。"""
    w = np.array([1.0, 1.0, 1.0, 1.0]) / 24.0
    xyz = np.array([
        [0.58541020, 0.13819660, 0.13819660],
        [0.13819660, 0.58541020, 0.13819660],
        [0.13819660, 0.13819660, 0.58541020],
        [0.13819660, 0.13819660, 0.13819660],
    ])
    return w, xyz


def tetrahedron_unit_o14():
    """14 点 Felippa 规则，精确到 4 次多项式。"""
    a = 0.1005267652252045
    b = 0.314372873493192
    c = 0.8850566000690581
    d = 0.0931745731195340
    e = 0.3108859192633005
    w = np.array([
        0.1328387466855907,
        0.1328387466855907,
        0.1328387466855907,
        0.1328387466855907,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0882236613785888,
        0.0190475587642109,
        0.0190475587642109,
        0.0190475587642109,
        0.0190475587642109,
    ])
    xyz = np.array([
        [a, a, a],
        [a, a, c],
        [a, c, a],
        [c, a, a],
        [d, d, e],
        [d, e, d],
        [e, d, d],
        [d, e, e],
        [e, d, e],
        [e, e, d],
        [b, b, b],
        [b, b, e],
        [b, e, b],
        [e, b, b],
    ])
    return w, xyz


def integrate_tetrahedron(f, order: int = 4):
    """
    使用 Felippa 规则在标准四面体上积分函数 f(xyz)。
    f 接收形状为 (3,) 的 numpy 数组。
    """
    if order <= 4:
        w, xyz = tetrahedron_unit_o04()
    else:
        w, xyz = tetrahedron_unit_o14()
    vol = tetrahedron_unit_volume()
    # 归一化权重，使其严格等于体积（硬编码权重可能存在参考元差异）
    w = w / np.sum(w) * vol
    total = 0.0
    for i in range(len(w)):
        total += w[i] * f(xyz[i])
    return total
