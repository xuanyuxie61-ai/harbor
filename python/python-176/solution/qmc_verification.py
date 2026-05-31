"""
qmc_verification.py
================================================================================
准蒙特卡洛（QMC）采样与积分验证模块

本模块融合以下种子项目的核心算法：
  - 498_hammersley : Hammersley 低差异序列（准随机序列）

科学背景
--------
在最优控制伴随方程方法中，目标泛函 J(q) 及其梯度的数值计算涉及高维积分。
传统伪随机 Monte Carlo 方法的收敛速度为 O(N^{-1/2})，与维数无关但较慢。
准蒙特卡洛（QMC）方法使用低差异序列（low-discrepancy sequences），
其收敛速度可达 O(N^{-1}) 甚至 O(N^{-1} log^{d-1} N)，在高维积分中显著优于
传统 Monte Carlo。

Hammersley 序列是一种经典的低差异序列，其构造方式为：
  - 第 1 维：r_1(i) = mod(i, N) / N，形成均匀网格
  - 第 j 维（j ≥ 2）：将 i 用第 j 个素数 p_j 进制表示，
    然后翻转小数点得到径向逆函数（radical inverse function）。

本模块将 Hammersley 序列用于：
  1. 高维控制参数空间的采样与积分验证
  2. 椭圆域内点的均匀采样
  3. 最优控制问题目标泛函的 QMC 估计，作为 FEM 离散解的独立验证手段

关键公式
--------
1. 径向逆函数（Radical Inverse Function）：
   给定素数基 p，将整数 i 表示为
       i = d_k p^k + d_{k-1} p^{k-1} + ... + d_1 p + d_0
   则 φ_p(i) = d_0/p + d_1/p² + d_2/p³ + ...

2. Hammersley 点（M 维，共 N 个点）：
   x_i^(1) = i / N,      i = 0, 1, ..., N-1
   x_i^(j) = φ_{p_j}(i), j = 2, ..., M

3. Koksma-Hlawka 不等式：
   |∫ f dμ − (1/N) Σ f(x_i)| ≤ V(f) · D_N^*
   其中 V(f) 是 f 的有界变差，D_N^* 是序列的星差异度。
"""

import numpy as np


# 前 1600 个素数的前 50 个（足够用于高维 QMC）
_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
], dtype=int)


def radical_inverse(i, base):
    """
    计算整数 i 在素数基 base 下的径向逆函数 φ_base(i)。
    这是 Hammersley 序列的核心构建块。
    """
    result = 0.0
    f = 1.0 / base
    while i > 0:
        digit = i % base
        result += f * digit
        i //= base
        f /= base
    return result


def hammersley_sequence(dim, n_points, offset=0):
    """
    生成 M 维 Hammersley 低差异序列。

    参数
    ----
    dim      : 维数 M
    n_points : 采样点数 N
    offset   : 起始索引偏移

    返回
    ----
    points : (n_points, dim) 数组，每行是一个 Hammersley 点，坐标在 [0,1]^dim
    """
    if dim < 1:
        raise ValueError("hammersley_sequence: dim 必须 ≥ 1")
    if dim > len(_PRIMES) + 1:
        raise ValueError("hammersley_sequence: 维数过大，超出预计算素数表")

    points = np.zeros((n_points, dim), dtype=float)
    for i in range(n_points):
        idx = i + offset
        # 第 1 维
        points[i, 0] = (idx % n_points) / n_points if n_points > 0 else 0.0
        # 第 2 维及以后
        for j in range(1, dim):
            points[i, j] = radical_inverse(idx, _PRIMES[j - 1])

    return points


def hammersley_ellipse_sample(a, b, n_points, offset=0):
    """
    使用 Hammersley 序列在二维椭圆域 {(x,y) | (x/a)²+(y/b)² ≤ 1} 内均匀采样。
    方法：在单位圆盘内用极坐标映射，再利用椭圆伸缩。
    映射：r = √u, θ = 2πv，则 (x,y) = (a r cosθ, b r sinθ)
    其中 (u,v) 来自 2D Hammersley 序列。
    """
    qmc = hammersley_sequence(2, n_points, offset)
    u = qmc[:, 0]
    v = qmc[:, 1]
    r = np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)
    return np.column_stack((x, y))


def qmc_integrate_ellipse(f, a, b, n_points):
    """
    使用 QMC（Hammersley 序列）在椭圆域上积分函数 f(x,y)。
    积分区域面积 S = π a b。
    估计值 = S · (1/N) Σ_i f(x_i, y_i)
    """
    samples = hammersley_ellipse_sample(a, b, n_points)
    vals = np.array([f(samples[i, 0], samples[i, 1]) for i in range(n_points)])
    area = np.pi * a * b
    estimate = area * np.mean(vals)
    return estimate


def qmc_integrate_box(f, box, n_points, dim=None):
    """
    使用 Hammersley 序列在超矩形 box 上积分函数 f(x)。

    参数
    ----
    f        : 函数 f(x)，x 为长度 dim 的数组
    box      : [(a1,b1), ...] 区间列表
    n_points : 采样点数
    dim      : 维数（自动推断）
    """
    if dim is None:
        dim = len(box)
    points = hammersley_sequence(dim, n_points)
    a = np.array([b[0] for b in box], dtype=float)
    b_arr = np.array([b[1] for b in box], dtype=float)
    scale = b_arr - a
    volume = np.prod(scale)

    # 将 [0,1]^dim 映射到 box
    phys_points = points * scale + a
    vals = np.array([f(phys_points[i]) for i in range(n_points)])
    estimate = volume * np.mean(vals)
    return estimate


def verify_fem_with_qmc(y_fem, nodes, elements, f_integrand, a, b, n_qmc=2000):
    """
    使用 QMC 积分验证 FEM 离散的目标泛函计算。
    计算 ∫_Ω f_integrand(x,y) dx dy 的 QMC 估计，并与 FEM 投影积分比较。

    参数
    ----
    y_fem         : FEM 节点值
    nodes         : 节点坐标
    elements      : 三角形单元
    f_integrand   : 被积函数 f(x,y,y_fem_interp)
    a, b          : 椭圆半轴
    n_qmc         : QMC 采样点数

    返回
    ----
    qmc_estimate : QMC 估计值
    fem_estimate : FEM 估计值（通过单元求积）
    """
    # FEM 估计：遍历单元，用重心坐标插值
    fem_val = 0.0
    for e in elements:
        p = nodes[e]
        area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                         - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        # 使用 3 点重心坐标（单元重心）
        xc = np.mean(p[:, 0])
        yc = np.mean(p[:, 1])
        yc_fem = np.mean(y_fem[e])
        fem_val += area * f_integrand(xc, yc, yc_fem)

    # QMC 估计
    def f_wrapper(x, y):
        # 需要插值 FEM 解到任意点，这里简化为最近节点值
        dists = (nodes[:, 0] - x) ** 2 + (nodes[:, 1] - y) ** 2
        idx = np.argmin(dists)
        return f_integrand(x, y, y_fem[idx])

    qmc_val = qmc_integrate_ellipse(f_wrapper, a, b, n_qmc)
    return qmc_val, fem_val
