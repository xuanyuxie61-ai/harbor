# -*- coding: utf-8 -*-
"""
simplex_sampling.py

博士级单纯形采样与积分库

融合原项目算法：
- 1409_wedge_integrals 的楔形区域精确单形积分与 Dirichlet 采样
- 1248_tetrahedron_integrals 的四面体精确积分与均匀采样

科学应用场景：
1. 多组分结晶体系（如药物共晶、盐溶液）的组成空间是一个概率单纯形。
   通过 Dirichlet 采样可生成均匀分布于组成空间的组成点，用于计算
   多组分溶解度相图上的积分量。
2. 楔形/四面体精确积分公式用于验证高维数值积分规则的精度，以及
   计算有限元方法中标准单元的刚度矩阵元素。
"""

import numpy as np


def dirichlet_sample_uniform_simplex(n_samples, dim, rng=None):
    """
    在 d 维标准单纯形上均匀采样。

    数学原理：
        标准单纯形 S_d = {x ∈ R^d : x_i ≥ 0, Σx_i ≤ 1}
        均匀采样方法：生成 d+1 个独立 Exp(1) 随机变量 E_i = -ln(U_i)，
        然后 x_j = E_j / Σ_{i=1}^{d+1} E_i，j=1,...,d
        这等价于 Dirichlet(1,...,1) 分布。

    参数：
        n_samples : int
            采样点数
        dim : int
            单纯形维数
        rng : numpy.random.Generator, optional

    返回：
        x : ndarray, shape (n_samples, dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_samples <= 0 or dim <= 0:
        return np.empty((max(0, n_samples), max(0, dim)), dtype=float)

    E = -np.log(rng.random((n_samples, dim + 1)))
    E = np.where(E == 0, 1e-300, E)  # 数值鲁棒性
    s = E.sum(axis=1, keepdims=True)
    x = E[:, :-1] / s
    return x


def wedge01_monomial_integral(e):
    """
    计算单位楔形 W = {(x,y,z): x≥0, y≥0, x+y≤1, -1≤z≤1} 上的单项式积分。

    数学公式：
        I = ∫_W x^{e1} y^{e2} z^{e3} dV

    解析解：
        xy-三角形部分：∫_Δ x^{e1} y^{e2} dxdy = e2! / [(e1+1)(e1+2)...(e1+e2+1)(e1+e2+2)]
        通过递推计算：
            value_xy = ∏_{i=1}^{e2} i/(e1+i) · 1/[(e1+e2+1)(e1+e2+2)]
        z-区间部分：
            若 e3 为奇数，∫_{-1}^{1} z^{e3} dz = 0
            若 e3 为偶数，∫_{-1}^{1} z^{e3} dz = 2/(e3+1)

    参数：
        e : array-like, shape (3,)
            指数向量 [e1, e2, e3]

    返回：
        integral : float
    """
    e = np.asarray(e, dtype=int)
    if e.size != 3:
        raise ValueError("e must have exactly 3 elements")
    e1, e2, e3 = e[0], e[1], e[2]
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("Exponents must be non-negative")

    # xy 三角形积分
    value_xy = 1.0
    for i in range(1, e2 + 1):
        value_xy *= float(i) / float(e1 + i)
    value_xy /= float((e1 + e2 + 1) * (e1 + e2 + 2))

    # z 区间积分
    if e3 % 2 == 1:
        value_z = 0.0
    else:
        value_z = 2.0 / float(e3 + 1)

    return value_xy * value_z


def tetrahedron01_monomial_integral(e):
    """
    计算单位四面体 T = {(x,y,z): x≥0, y≥0, z≥0, x+y+z≤1} 上的单项式积分。

    数学公式：
        I = ∫_T x^{e1} y^{e2} z^{e3} dV
          = e1! · e2! · e3! / (e1 + e2 + e3 + 3)!
        等价于 multinomial Beta 函数：B(e1+1, e2+1, e3+1, 1)

    递推算法（避免大数阶乘溢出）：
        I = ∏_{i=1}^3 ∏_{j=1}^{e_i} j / k
        其中 k 从 e1+e2+e3+3 递减。

    参数：
        e : array-like, shape (3,)

    返回：
        integral : float
    """
    e = np.asarray(e, dtype=int)
    if e.size != 3:
        raise ValueError("e must have exactly 3 elements")
    if np.any(e < 0):
        raise ValueError("Exponents must be non-negative")

    e1, e2, e3 = e[0], e[1], e[2]
    k = e1 + e2 + e3 + 3
    value = 1.0
    # e1 部分的递推
    for j in range(1, e1 + 1):
        value *= float(j) / float(k)
        k -= 1
    # e2 部分
    for j in range(1, e2 + 1):
        value *= float(j) / float(k)
        k -= 1
    # e3 部分
    for j in range(1, e3 + 1):
        value *= float(j) / float(k)
        k -= 1
    # 归一化
    for _ in range(k):
        value /= float(k)
        k -= 1
    return value


def tetrahedron01_volume():
    """单位四面体体积 = 1/6。"""
    return 1.0 / 6.0


def monomial_value(m, n, e, x):
    """
    在 n 个点上计算 m 维单项式的值。

    数学公式：
        f(x) = ∏_{i=1}^m x_i^{e_i}
        约定：0^0 = 1

    参数：
        m : int
            空间维数
        n : int
            点数
        e : array-like, shape (m,)
            指数向量
        x : ndarray, shape (m, n)
            点坐标

    返回：
        v : ndarray, shape (n,)
    """
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    if x.shape[0] != m or x.shape[1] != n:
        raise ValueError(f"x shape must be ({m}, {n}), got {x.shape}")

    v = np.ones(n, dtype=float)
    for i in range(m):
        if e[i] == 0:
            continue
        xi = x[i, :]
        # 0^0 = 1 的约定已通过 continue 处理
        # 避免 0 的负幂
        xi = np.where(np.abs(xi) < 1e-300, 1e-300 * np.sign(xi) if np.any(xi < 0) else 1e-300, xi)
        v *= xi ** e[i]
    return v


def composition_space_integral(func, dim, n_samples=10000, rng=None):
    """
    在 d 维概率单纯形上计算函数积分的 Monte Carlo 估计。

    数学公式：
        I = ∫_{S_d} f(x) dx ≈ Vol(S_d) · (1/N) Σ_{i=1}^N f(x_i)
        Vol(S_d) = 1 / d!

    参数：
        func : callable
            接受 shape (n_samples, dim) 的数组，返回 shape (n_samples,)
        dim : int
            维数
        n_samples : int
        rng : numpy.random.Generator

    返回：
        integral : float
        std_err : float
            Monte Carlo 标准误差
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = dirichlet_sample_uniform_simplex(n_samples, dim, rng)
    values = func(samples)
    from math import factorial
    vol = 1.0 / factorial(dim)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    integral = vol * mean_val
    std_err = vol * std_val / np.sqrt(n_samples)
    return integral, std_err
