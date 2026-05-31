# -*- coding: utf-8 -*-
"""
quadrature_rules.py
===================
高维数值积分规则与球面积分。

融合种子项目：
- 641_laguerre_polynomial / 525_hermite_rule : 高斯求积
- 1121_sphere_lebedev_rule_display          : Lebedev 球面积分规则
- 1214_test_interp_nd                       : Genz 多维测试函数包
"""

import numpy as np
import math
from orthogonal_polynomials import gauss_laguerre_rule, gauss_hermite_rule


# ---------------------------------------------------------------------------
# Lebedev 球面积分规则（简化核心：球面坐标转换与节点积分）
# ---------------------------------------------------------------------------

def spherical_to_cartesian(theta, phi):
    """
    球面坐标 (θ, φ) -> 笛卡尔坐标 (x, y, z)。
    θ ∈ [0, π] 为极角，φ ∈ [0, 2π] 为方位角。
    """
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def lebedev_rule_6():
    """
    极简 Lebedev 6 点规则（八面体对称性）。
    积分公式：∫_{S^2} f(x,y,z) dΩ ≈ (4π/6) Σ f(±1,0,0) 及其置换。
    返回节点 (x,y,z) 和权重 w。
    """
    nodes = np.array([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0]
    ], dtype=float)
    w = np.full(6, 4.0 * math.pi / 6.0, dtype=float)
    return nodes, w


def lebedev_rule_14():
    """
    14 点 Lebedev 规则（截角八面体）。
    6 个坐标轴点 + 8 个 (±1,±1,±1)/√3 点。
    """
    nodes_axis = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    w_axis = np.full(6, 0.6666666666666667 * 4.0 * math.pi / 14.0, dtype=float)

    s = 1.0 / math.sqrt(3.0)
    nodes_diag = np.array([
        [ s,  s,  s], [ s,  s, -s], [ s, -s,  s], [ s, -s, -s],
        [-s,  s,  s], [-s,  s, -s], [-s, -s,  s], [-s, -s, -s]
    ], dtype=float)
    w_diag = np.full(8, 0.75 * 4.0 * math.pi / 14.0, dtype=float)

    nodes = np.vstack([nodes_axis, nodes_diag])
    w = np.concatenate([w_axis, w_diag])
    # 归一化使权重和为 4π
    w = w / np.sum(w) * 4.0 * math.pi
    return nodes, w


def integrate_on_sphere(f_eval, rule='14'):
    """
    用 Lebedev 规则在球面上积分函数 f。
    f_eval 接收节点数组 (N,3) 返回函数值 (N,)。
    """
    if rule == '6':
        nodes, w = lebedev_rule_6()
    elif rule == '14':
        nodes, w = lebedev_rule_14()
    else:
        nodes, w = lebedev_rule_14()
    vals = f_eval(nodes)
    return float(np.dot(w, vals))


# ---------------------------------------------------------------------------
# Genz 多维测试函数包（1214_test_interp_nd 核心）
# ---------------------------------------------------------------------------

def genz_cosine(m, c, w, x):
    """
    Genz 振荡型测试函数：
        f(x) = cos(2π w_1 + Σ_{i=1}^m c_i x_i)
    默认：c_i = 1/m, w_1 = 0.3。
    积分精确值在 [0,1]^m 上为解析可求。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = 2.0 * math.pi * w[0] + np.dot(c, x)
    return np.cos(val)


def genz_product_peak(m, c, w, x):
    """
    Genz 积峰型：
        f(x) = ∏_{i=1}^m (c_i^{-2} + (x_i - w_i)^2)^{-1}
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.ones(x.shape[1], dtype=float)
    for i in range(m):
        val *= 1.0 / (c[i] ** (-2) + (x[i, :] - w[i]) ** 2)
    return val


def genz_corner_peak(m, c, w, x):
    """
    Genz 角峰型：
        f(x) = (1 + Σ c_i x_i)^{-(m+1)}
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.ones(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * x[i, :]
    return val ** (-(m + 1))


def genz_gaussian(m, c, w, x):
    """
    Genz 高斯型：
        f(x) = exp( - Σ c_i^2 (x_i - w_i)^2 )
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] ** 2 * (x[i, :] - w[i]) ** 2
    return np.exp(-val)


def genz_c0_function(m, c, w, x):
    """
    Genz C0 连续型：
        f(x) = exp( - Σ c_i |x_i - w_i| )
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * np.abs(x[i, :] - w[i])
    return np.exp(-val)


def genz_discontinuous(m, c, w, x):
    """
    Genz 间断型：
        f(x) = exp( Σ c_i x_i ) if x_1 <= w_1 and x_2 <= w_2 else 0
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    val = np.zeros(x.shape[1], dtype=float)
    for i in range(m):
        val += c[i] * x[i, :]
    mask = (x[0, :] <= w[0])
    if m > 1:
        mask &= (x[1, :] <= w[1])
    return np.exp(val) * mask.astype(float)


# 统一接口调度
def genz_evaluate(prob, m, c, w, x):
    """统一调用 Genz 测试函数，prob ∈ {1,...,6}。"""
    if prob == 1:
        return genz_cosine(m, c, w, x)
    elif prob == 2:
        return genz_product_peak(m, c, w, x)
    elif prob == 3:
        return genz_corner_peak(m, c, w, x)
    elif prob == 4:
        return genz_gaussian(m, c, w, x)
    elif prob == 5:
        return genz_c0_function(m, c, w, x)
    elif prob == 6:
        return genz_discontinuous(m, c, w, x)
    else:
        raise ValueError("prob must be in 1..6")


def genz_integral_exact(prob, m, c, w):
    """
    Genz 测试函数在 [0,1]^m 上的精确积分值（部分有闭式解）。
    """
    if prob == 1:
        # ∫ cos(2π w1 + Σ c_i x_i) dx = Re[exp(i 2π w1) ∏ (exp(i c_j)-1)/(i c_j)]
        # 数值计算
        from scipy.integrate import nquad
        def f(*x):
            return float(genz_cosine(m, c, w, np.array(x)))
        ranges = [(0.0, 1.0)] * m
        val, _ = nquad(f, ranges)
        return val
    elif prob == 4:
        # Gaussian: ∏ sqrt(pi)/(2 c_i) * (erf(c_i (1-w_i)) + erf(c_i w_i))
        val = 1.0
        for i in range(m):
            from math import erf, sqrt
            val *= (sqrt(math.pi) / (2.0 * c[i])) * (erf(c[i] * (1.0 - w[i])) + erf(c[i] * w[i]))
        return val
    else:
        # 对无闭式解的情况，用蒙特卡洛估计（简化）
        rng = np.random.default_rng(42)
        N = 200000
        pts = rng.random((m, N))
        vals = genz_evaluate(prob, m, c, w, pts)
        return float(np.mean(vals))


# ---------------------------------------------------------------------------
# 高维张量积求积规则
# ---------------------------------------------------------------------------

def tensor_product_quadrature_1d(rule_func, n_points, a, b):
    """
    对一维求积规则做线性变换到区间 [a,b]。
    rule_func(n) -> (nodes, weights) on 原始区间。
    返回在 [a,b] 上的节点和权重。
    """
    nodes, weights = rule_func(n_points)
    # 假设 rule_func 给出的是标准区间上的规则
    # 对 Laguerre: [0, ∞)，对 Hermite: (-∞, ∞)
    # 这里做简单仿射变换：若原区间已知，需要额外信息
    # 对一般有限区间 [a,b]，用线性映射
    # 默认节点在 [-1,1] 或 [0,∞)。
    # 为简化，这里假设要做的是到 [a,b] 的映射：
    #   x_mapped = (b-a)/2 * x + (a+b)/2
    #   w_mapped = (b-a)/2 * w
    # 但若 nodes 来自 Laguerre，这是不合适的。
    # 因此本函数仅用于已标准化到 [-1,1] 的规则。
    x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * weights
    return x_mapped, w_mapped


def multidimensional_gauss_legendre_simple(m, n_per_dim, a=0.0, b=1.0):
    """
    简单 m 维张量积 Gauss-Legendre 规则（直接用 numpy polynomial 的节点）。
    返回节点数组 (m, N) 和权重 (N,)，其中 N = n_per_dim^m。
    为避免维度灾难，限制总点数不超过 1e6。
    """
    N_total = n_per_dim ** m
    if N_total > 1_000_000:
        raise ValueError("Total quadrature points exceed 1e6. Reduce m or n_per_dim.")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n_per_dim)
    # 映射到 [a,b]
    nodes_1d = 0.5 * (b - a) * nodes_1d + 0.5 * (a + b)
    weights_1d = 0.5 * (b - a) * weights_1d

    # 张量积
    grids = np.meshgrid(*[nodes_1d] * m, indexing='ij')
    nodes = np.vstack([g.ravel() for g in grids])
    weight_grid = np.meshgrid(*[weights_1d] * m, indexing='ij')
    weights = np.prod(np.stack([g.ravel() for g in weight_grid]), axis=0)
    return nodes, weights


# ---------------------------------------------------------------------------
# 基于 Halton 的准蒙特卡洛积分
# ---------------------------------------------------------------------------

def qmc_integral_halton(f, m, n_samples, a=0.0, b=1.0, seed=0):
    """
    用 Halton 准随机序列计算 m 维积分 ∫_{[a,b]^m} f(x) dx。
    """
    from random_tools import halton_sequence
    pts = halton_sequence(0, n_samples - 1, m)
    # pts 形状 (m, n_samples)，值域 [0,1]
    pts = a + (b - a) * pts
    vals = f(pts)
    volume = (b - a) ** m
    return volume * float(np.mean(vals)), volume * float(np.std(vals) / math.sqrt(n_samples))
