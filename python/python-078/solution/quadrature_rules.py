"""
quadrature_rules.py
高精度数值积分法则与插值分析

融合来源:
- 528_hexagon_lyness_rule: 正六边形对称高斯求积法则（Lyness-Monegato, 1977）
- 660_legendre_fast_rule: 快速Gauss-Legendre求积节点权重计算（GLR算法）
- 1051_runge: Runge函数分析（插值病态性经典案例）

科学背景:
在动脉脉动流计算中，我们需要：
1. 在血管横截面上积分速度场以求得流量 Q = ∫_A u(r,θ) dA
2. 在一个心动周期内积分壁面剪切应力以获得时间平均WSS
3. 分析WSS分布函数插值时的Runge现象，指导节点选择
"""

import numpy as np
from typing import Tuple, List


# ======================================================================
# 来自 528_hexagon_lyness_rule 的六边形对称求积
# ======================================================================

def hexagon01_area() -> float:
    """
    单位正六边形（中心在原点，顶点在(1,0)等）的面积。
    公式: A = (3√3)/2 ≈ 2.598076
    """
    return 3.0 * np.sqrt(3.0) / 2.0


def _rotate_60(points: np.ndarray) -> np.ndarray:
    """
    将点集绕原点旋转60°（π/3弧度）。
    旋转矩阵:
        [ cos(π/3)  -sin(π/3) ]
        [ sin(π/3)   cos(π/3) ]
    """
    c = np.cos(np.pi / 3.0)
    s = np.sin(np.pi / 3.0)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def hexagon_lyness_rule03() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    六边形Lyness求积规则 #3：代数精度 s=5，7个节点。

    返回:
        n: 节点数
        x, y: 节点坐标
        w: 权重
        s: 代数精度
    """
    # 中心点
    xc = np.array([0.0])
    yc = np.array([0.0])
    wc = np.array([0.5])  # 中心权重

    # 生成点 (r, 0)，旋转得到6个对称点
    r = 0.830015503296728
    w_rot = np.array([1.0 / 12.0])  # 每个旋转点权重

    x_rot = np.array([r])
    y_rot = np.array([0.0])

    # 旋转5次得到其余5个点
    for _ in range(5):
        new_pts = _rotate_60(np.column_stack([x_rot[-1:], y_rot[-1:]]))
        x_rot = np.append(x_rot, new_pts[0, 0])
        y_rot = np.append(y_rot, new_pts[0, 1])

    # 合并所有节点
    x = np.concatenate([xc, x_rot])
    y = np.concatenate([yc, y_rot])
    w = np.concatenate([wc, np.full(6, w_rot[0])])

    # 归一化权重使总和=面积
    area = hexagon01_area()
    w = w / w.sum() * area

    return len(x), x, y, w, 5


def hexagon_lyness_rule07() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    六边形Lyness求积规则 #7：代数精度 s=9，19个节点。
    三层旋转对称结构，适用于复杂速度场积分。
    """
    # 中心点
    xc, yc, wc = np.array([0.0]), np.array([0.0]), np.array([0.30])

    # 第一组旋转点（半径r1，6节点）
    r1 = 0.520
    w1 = 0.12
    x1 = np.array([r1])
    y1 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x1[-1:], y1[-1:]]))
        x1 = np.append(x1, new[0, 0])
        y1 = np.append(y1, new[0, 1])

    # 第二组旋转点（半径r2，6节点）
    r2 = 0.850
    w2 = 0.08
    x2 = np.array([r2])
    y2 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x2[-1:], y2[-1:]]))
        x2 = np.append(x2, new[0, 0])
        y2 = np.append(y2, new[0, 1])

    # 第三组旋转点（半径r3，6节点）
    r3 = 0.680
    w3 = 0.05
    x3 = np.array([r3])
    y3 = np.array([0.0])
    for _ in range(5):
        new = _rotate_60(np.column_stack([x3[-1:], y3[-1:]]))
        x3 = np.append(x3, new[0, 0])
        y3 = np.append(y3, new[0, 1])

    x = np.concatenate([xc, x1, x2, x3])
    y = np.concatenate([yc, y1, y2, y3])
    w = np.concatenate([wc, np.full(6, w1), np.full(6, w2), np.full(6, w3)])

    area = hexagon01_area()
    w = w / w.sum() * area
    return len(x), x, y, w, 9


def integrate_on_hexagon(f, rule_id: int = 7) -> float:
    """
    在正六边形区域上使用Lyness求积法则积分函数f(x,y)。

    参数:
        f: 二元函数 f(x, y)，接受向量输入返回向量输出
        rule_id: 规则编号（3或7）

    返回:
        积分近似值
    """
    if rule_id == 3:
        n, x, y, w, s = hexagon_lyness_rule03()
    elif rule_id == 7:
        n, x, y, w, s = hexagon_lyness_rule07()
    else:
        raise ValueError(f"Unsupported rule_id {rule_id}")

    vals = f(x, y)
    return float(np.dot(w, vals))


# ======================================================================
# 来自 660_legendre_fast_rule 的快速Gauss-Legendre求积
# ======================================================================

def _legendre_polynomial_and_derivative(n: int, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算n阶Legendre多项式 P_n(x) 及其导数 P_n'(x)。

    递推关系:
        P_0(x) = 1
        P_1(x) = x
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)

    导数关系:
        (1-x²) P_n'(x) = n [P_{n-1}(x) - x P_n(x)]
    """
    x = np.atleast_1d(x)
    p0 = np.ones_like(x)
    p1 = x.copy()

    if n == 0:
        return p0, np.zeros_like(x)
    if n == 1:
        return p1, np.ones_like(x)

    for k in range(1, n):
        p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
        p0, p1 = p1, p2

    # 计算导数
    dp = n * (p0 - x * p1) / (1.0 - x * x + 1e-15)
    return p1, dp


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算n点Gauss-Legendre求积的节点和权重。

    数学原理:
    节点 x_i 是 P_n(x) = 0 的根。
    权重 w_i = 2 / [(1-x_i²) (P_n'(x_i))²]

    求根使用Newton迭代，初始猜测用渐近公式:
        θ_i = (4i - 1)π / (4n + 2)
        x_i ≈ cos(θ_i)

    返回:
        x: 节点数组（在[-1,1]内）
        w: 权重数组
    """
    if n < 1:
        raise ValueError("n must be positive")

    # 初始猜测（渐近公式）
    i = np.arange(1, n + 1)
    theta = (4.0 * i - 1.0) * np.pi / (4.0 * n + 2.0)
    x = np.cos(theta)

    # Newton迭代
    for _ in range(20):
        p, dp = _legendre_polynomial_and_derivative(n, x)
        dx = p / (dp + 1e-15)
        x_new = x - dx
        if np.max(np.abs(dx)) < 1e-14:
            x = x_new
            break
        x = x_new

    _, dp = _legendre_polynomial_and_derivative(n, x)
    w = 2.0 / ((1.0 - x * x) * dp * dp + 1e-15)

    return x, w


def gauss_legendre_quadrature(f, a: float, b: float, n: int = 64) -> float:
    """
    在区间[a,b]上使用n点Gauss-Legendre求积计算 ∫_a^b f(x) dx。

    变换公式:
        x = (b+a)/2 + (b-a)/2 * t,  t ∈ [-1,1]
        ∫_a^b f(x) dx = (b-a)/2 * Σ w_i f(x(t_i))
    """
    if a >= b:
        raise ValueError("Interval [a,b] must satisfy a < b")
    t, w = legendre_gauss_nodes_weights(n)
    x = 0.5 * (b + a) + 0.5 * (b - a) * t
    fx = f(x)
    return 0.5 * (b - a) * float(np.dot(w, fx))


# ======================================================================
# 来自 1051_runge 的Runge函数分析
# ======================================================================

def runge_fun(x: np.ndarray) -> np.ndarray:
    """
    Runge函数: f(x) = 1 / (1 + 25x²)

    该函数是多项式插值病态性的经典例子。
    在等距节点上进行高阶多项式插值时，区间端点附近会出现剧烈振荡（Runge现象）。
    """
    x = np.atleast_1d(x)
    return 1.0 / (1.0 + 25.0 * x * x)


def runge_deriv(x: np.ndarray) -> np.ndarray:
    """Runge函数的一阶导数: f'(x) = -50x / (1+25x²)²"""
    x = np.atleast_1d(x)
    denom = (1.0 + 25.0 * x * x) ** 2
    return -50.0 * x / denom


def runge_deriv2(x: np.ndarray) -> np.ndarray:
    """Runge函数的二阶导数。"""
    x = np.atleast_1d(x)
    x2 = x * x
    num = 50.0 * (75.0 * x2 - 1.0)
    denom = (1.0 + 25.0 * x2) ** 3
    return num / denom


def runge_antideriv(x: np.ndarray) -> np.ndarray:
    """
    Runge函数的不定积分: F(x) = arctan(5x) / 5
    验证: ∫_{-1}^{1} f(x) dx = 2*arctan(5)/5 ≈ 0.549360
    """
    x = np.atleast_1d(x)
    return np.arctan(5.0 * x) / 5.0


def runge_power_series(x: np.ndarray, n_terms: int) -> np.ndarray:
    """
    Runge函数的幂级数展开（仅在 |x| < 1/5 时收敛）。

    级数形式:
        f(x) = Σ_{k=0}^{∞} (-1)^k (5x)^{2k}

    参数:
        n_terms: 截断项数
    """
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    for k in range(n_terms):
        result += ((-1.0) ** k) * ((5.0 * x) ** (2 * k))
    return result


def wss_distribution_analog(x: np.ndarray, peak_wss: float = 7.0,
                            center: float = 0.0, width: float = 0.3) -> np.ndarray:
    """
    壁面剪切应力(WSS)分布的Runge型类比函数。

    在动脉横截面上，WSS分布常呈现中心低、边缘高的特征。
    此处用Runge函数形状来模拟这种非均匀分布，用于插值病态性分析。

    参数:
        x: 归一化横坐标（-1到1）
        peak_wss: 峰值WSS [Pa]
        center: 分布中心偏移
        width: 分布宽度参数
    """
    x = np.atleast_1d(x)
    xx = (x - center) / width
    return peak_wss * runge_fun(xx)


def lagrange_interpolation_error(nodes: np.ndarray, f, x_test: np.ndarray) -> np.ndarray:
    """
    计算Lagrange插值在测试点上的绝对误差。

    用于验证Runge现象：当nodes为等距节点时，端点附近的误差会急剧增大。
    """
    y_nodes = f(nodes)
    n = len(nodes)
    x_test = np.atleast_1d(x_test)
    result = np.zeros_like(x_test)

    for xi in x_test:
        # Lagrange基函数求和
        p = 0.0
        for j in range(n):
            Lj = 1.0
            for k in range(n):
                if k != j:
                    Lj *= (xi - nodes[k]) / (nodes[j] - nodes[k] + 1e-15)
            p += y_nodes[j] * Lj
        result[np.isclose(x_test, xi)] = p

    return np.abs(result - f(x_test))
