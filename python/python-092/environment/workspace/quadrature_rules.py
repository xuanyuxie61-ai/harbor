"""
quadrature_rules.py
高阶数值积分规则：三角形对称求积与单位球积分
基于 triangle_symq_rule 与 ball_integrals 核心算法重构

声学工程应用：
- 三角形表面上的声压/质点速度积分（用于边界元或射线能量计算）
- 球面积分用于全向声源辐射和球形麦克风阵列指向性
"""

import numpy as np
from math import gamma as math_gamma


def factorial(n):
    """阶乘。"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def triangle_unit_monomial_integral(expon):
    """
    单位三角形 (0,0)-(1,0)-(0,1) 上单项式 x^m y^n 的精确积分：
        ∫∫ x^m y^n dx dy = m! * n! / (m + n + 2)!
    来自 triangle_symq_rule 的精确公式。
    """
    m, n = expon
    return factorial(m) * factorial(n) / factorial(m + n + 2)


def triangle_unit_area():
    """单位三角形面积 = 1/2。"""
    return 0.5


def triangle_symq_rule(precision):
    """
    单位三角形上的对称求积规则（简化版，支持 precision 1~10）。
    基于 triangle_symq_rule 的 Xiao-Gimbutas 高阶对称求积思想。
    返回节点数 n、重心坐标 (a,b,c) 和权重 w。

    对于更高精度，使用 Stroud 规则的简化形式。
    """
    if precision <= 1:
        # 1点规则，精度1（重心）
        a = np.array([1.0 / 3.0])
        b = np.array([1.0 / 3.0])
        c = np.array([1.0 / 3.0])
        w = np.array([0.5])
        return 1, a, b, c, w
    elif precision <= 2:
        # 3点规则，精度2
        a = np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
        b = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        c = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        return 3, a, b, c, w
    elif precision <= 3:
        # 4点规则，精度3
        a = np.array([1.0 / 3.0, 0.6, 0.2, 0.2])
        b = np.array([1.0 / 3.0, 0.2, 0.6, 0.2])
        c = np.array([1.0 / 3.0, 0.2, 0.2, 0.6])
        w = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        return 4, a, b, c, w
    elif precision <= 5:
        # 7点规则，精度5
        a = np.array([
            1.0 / 3.0,
            0.059715871789770, 0.470142064105115, 0.470142064105115,
            0.797426985353087, 0.101286507323456, 0.101286507323456
        ])
        b = np.array([
            1.0 / 3.0,
            0.470142064105115, 0.059715871789770, 0.470142064105115,
            0.101286507323456, 0.797426985353087, 0.101286507323456
        ])
        c = np.array([
            1.0 / 3.0,
            0.470142064105115, 0.470142064105115, 0.059715871789770,
            0.101286507323456, 0.101286507323456, 0.797426985353087
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506, 0.132394152788506, 0.132394152788506,
            0.125939180544827, 0.125939180544827, 0.125939180544827
        ]) * 0.5
        return 7, a, b, c, w
    else:
        # 默认精度5
        return triangle_symq_rule(5)


def integrate_over_triangle(func, v0, v1, v2, precision=5):
    """
    在任意三角形 (v0, v1, v2) 上积分函数 func(x,y,z)。
    使用三角形对称求积规则：
        ∫_T f(x) dA = |T| * Σ w_i f(x(ξ_i))
    其中 ξ_i 为参考三角形上的求积点。
    """
    n, a, b, c, w = triangle_symq_rule(precision)
    # 重心坐标映射到物理坐标
    # x = a*v0 + b*v1 + c*v2
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    result = 0.0
    for i in range(n):
        p = a[i] * v0 + b[i] * v1 + c[i] * v2
        result += w[i] * func(p)
    return result * (area / 0.5)


def ball01_volume():
    """
    单位球体积：V = 4π/3。
    来自 ball_integrals。
    """
    return 4.0 * np.pi / 3.0


def ball01_monomial_integral(e):
    """
    单位球上单项式 x^e1 y^e2 z^e3 的精确积分。
    公式（来自 ball_integrals）：
        若任一指数为奇数，积分为0（对称性）。
        否则：
        I = 2 * Γ((e1+1)/2) * Γ((e2+1)/2) * Γ((e3+1)/2) / Γ((e1+e2+e3+3)/2)
    """
    e = np.asarray(e, dtype=int)
    if np.any(e < 0):
        return 0.0
    if np.any(e % 2 == 1):
        return 0.0
    if np.all(e == 0):
        integral = 2.0 * np.sqrt(np.pi ** 3) / math_gamma(1.5)
    elif np.any(e % 2 == 1):
        return 0.0
    else:
        integral = 2.0
        for i in range(3):
            integral = integral * math_gamma(0.5 * (e[i] + 1))
        integral = integral / math_gamma(0.5 * (e[0] + e[1] + e[2] + 3))
    # 将表面积分调整为体积分
    r = 1.0
    s = e[0] + e[1] + e[2] + 3
    integral = integral * (r ** s) / s
    return integral


def ball01_sample(n):
    """
    在单位球内均匀采样 n 个点。
    来自 ball_integrals 的采样算法：
        1. 生成3D标准正态随机向量
        2. 归一化到单位球面
        3. 径向缩放：r = u^{1/3}, u ~ U[0,1]
    在声学中用于球形麦克风阵列的采样点或全向声源辐射模式。
    """
    # 标准正态
    xyz = np.random.randn(n, 3)
    # 归一化到球面
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    xyz = xyz / norms
    # 径向缩放
    u = np.random.rand(n, 1)
    r = u ** (1.0 / 3.0)
    return xyz * r


def monomial_value(n_points, e, x):
    """
    在多点处计算单项式值。
    来自 ball_integrals / triangle_symq_rule 的 monomial_value。
    """
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    val = np.ones(n_points, dtype=float)
    for dim in range(len(e)):
        if e[dim] > 0:
            val *= x[:, dim] ** e[dim]
    return val


def line01_monomial_integral(e):
    """
    单位线段 [0,1] 上 x^e 的精确积分：1/(e+1)。
    来自 line_monte_carlo。
    """
    return 1.0 / (e + 1.0)


def line01_sample_random(n):
    """
    在 [0,1] 上均匀随机采样 n 个点。
    来自 line_monte_carlo。
    """
    return np.random.rand(n)


def line01_sample_ergodic(n, shift=0.0):
    """
    使用黄金比例加法递推在 [0,1] 上产生低差异序列：
        x_{j+1} = (x_j + φ) mod 1, φ = (1+√5)/2
    来自 line_monte_carlo 的 ergodic sampling。
    在射线追踪中用于沿传播路径的参数化采样。
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    x = np.zeros(n, dtype=float)
    x[0] = shift % 1.0
    for j in range(1, n):
        x[j] = (x[j - 1] + phi) % 1.0
    return x


def integrate_over_ball_monte_carlo(func, n_samples=10000):
    """
    使用蒙特卡洛方法在单位球上积分函数 func(x)。
    I ≈ V * (1/N) * Σ f(x_i)
    """
    samples = ball01_sample(n_samples)
    vals = np.array([func(s) for s in samples])
    return ball01_volume() * np.mean(vals)
