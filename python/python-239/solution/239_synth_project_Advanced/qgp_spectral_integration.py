"""
qgp_spectral_integration.py — 格点积分与 Fibonacci 准 Monte Carlo
====================================================================

融合种子项目: 654_lattice_rule (格点积分规则)

本模块实现用于相空间积分的格点积分方法,
特别用于 Cooper-Frye 冻结面积分和热力学量的动量空间积分.

标准格点规则 (Standard Lattice Rule):
--------------------------------------
N 点格点:
    x_j = (j * z / N) mod 1, j = 0, ..., N-1

其中 z 为生成向量, 在 s 维情况下 z in Z^s.

积分近似:
    I = integral_{[0,1]^s} f(x) dx ≈ (1/N) * sum_{j=0}^{N-1} f(x_j)

误差界 (Korobov 空间):
    |I - I_N| <= C * N^{-alpha} * (log N)^{s-1}
    其中 alpha 为函数的光滑度参数

Fibonacci 格点 (2D 最优):
    N = F_k (第 k 个 Fibonacci 数)
    z = [1, F_{k-1}]
    x_j = (j * [1, F_{k-1}] / F_k) mod 1

Fibonacci 格点的星偏差:
    D_N* = O(log(N) / N)
    远优于随机 Monte Carlo 的 O(1/sqrt(N))

非周期被积函数变换 (Sidi 变换):
    格点规则要求被积函数为周期函数.
    对于非周期 f(x), 使用变量变换:
        x = phi(t) = t (周期性延拓)
    其中 phi 为 Sidi 多项式变换:
        psi_m(t) = sum_{k=0}^{m-1} C(2m-1, k) * [t^{2m-k} / (2m-k) -
                   ...]
    使得 f(phi(t)) * phi'(t) 为周期函数

Bernoulli 多项式校正:
    对于非周期端点效应, 使用 Euler-Maclaurin 公式:
        I - I_N = sum_{k=1}^{p} B_{2k} / (2k)! * (f^{(2k-1)}(1) - f^{(2k-1)}(0)) * N^{-2k}
                  + O(N^{-2p-2})
"""

import numpy as np
from qgp_config import NumericalParams


def fibonacci_number(k: int) -> int:
    """
    计算第 k 个 Fibonacci 数

    F(0) = 1, F(1) = 1, F(k) = F(k-1) + F(k-2)

    Args:
        k: 阶数

    Returns:
        F_k
    """
    if k <= 0:
        return 1
    a, b = 1, 1
    for _ in range(k):
        a, b = b, a + b
    return a


def standard_lattice_rule(N: int, z: np.ndarray, s: int) -> np.ndarray:
    """
    标准格点规则 (源自 654_lattice_rule)

    x_j = (j * z / N) mod 1, j = 0, ..., N-1

    Args:
        N: 格点数
        z: 生成向量 (s,)
        s: 维度

    Returns:
        格点 (N, s)
    """
    j = np.arange(N, dtype=np.float64)
    points = np.zeros((N, s))
    for d in range(s):
        points[:, d] = np.mod(j * z[d] / N, 1.0)
    return points


def fibonacci_lattice_2d(k: int) -> tuple:
    """
    2D Fibonacci 格点 (源自 654_lattice_rule)

    N = F(k) 个点
    z = [1, F(k-1)]
    x_j = (j * z / N) mod 1

    Args:
        k: Fibonacci 阶数

    Returns:
        (points, N) 其中 points 为 (N, 2)
    """
    N = fibonacci_number(k)
    F_prev = fibonacci_number(k - 1)
    z = np.array([1.0, float(F_prev)])
    points = standard_lattice_rule(N, z, 2)
    return points, N


def sidi_transform(t: np.ndarray, m: int = 3) -> np.ndarray:
    """
    Sidi 多项式变换 (用于非周期被积函数)

    将 [0,1] 映射到 [0,1], 使得端点处所有导数为零,
    从而实现周期化.

    对于 m=3 (三次 Sidi 变换):
        psi(t) = t^3 * (10 - 15*t + 6*t^2)

    满足:
        psi(0) = 0, psi(1) = 1
        psi'(0) = psi'(1) = 0
        psi''(0) = psi''(1) = 0

    导数:
        psi'(t) = 30 * t^2 * (1 - t)^2

    Args:
        t: 输入 (在 [0,1] 中)
        m: 变换阶数

    Returns:
        psi(t)
    """
    if m == 1:
        return t
    elif m == 2:
        return t**2 * (3 - 2*t)
    elif m == 3:
        return t**3 * (10.0 - 15.0*t + 6.0*t**2)
    elif m == 4:
        return t**4 * (35.0 - 84.0*t + 70.0*t**2 - 20.0*t**3)
    else:
        # 一般情况
        result = np.zeros_like(t)
        for k in range(m):
            coeff = (-1)**k * _binomial_coeff(2*m-1, k) / (2*m - k)
            result += coeff * t**(2*m - k)
        return result


def sidi_derivative(t: np.ndarray, m: int = 3) -> np.ndarray:
    """
    Sidi 变换的导数

    psi'(t) = (2m)! / (m!)^2 * t^m * (1-t)^m

    Args:
        t: 输入
        m: 阶数

    Returns:
        psi'(t)
    """
    if m == 1:
        return np.ones_like(t)
    elif m == 2:
        return 6.0 * t * (1.0 - t)
    elif m == 3:
        return 30.0 * t**2 * (1.0 - t)**2
    elif m == 4:
        return 140.0 * t**3 * (1.0 - t)**3
    else:
        from math import factorial
        coeff = factorial(2*m) / factorial(m)**2
        return coeff * t**m * (1.0 - t)**m


def _binomial_coeff(n: int, k: int) -> int:
    """二项式系数 C(n, k)"""
    if k < 0 or k > n:
        return 0
    from math import comb
    return comb(n, k)


def lattice_integrate(func, s: int, N: int = None,
                       use_fibonacci: bool = True,
                       sidi_order: int = 3) -> float:
    """
    格点积分 (主函数)

    使用 Fibonacci 格点或标准格点计算:
        I = integral_{[0,1]^s} f(x) dx

    对于非周期被积函数, 应用 Sidi 变换进行周期化.

    步骤:
    1. 生成格点 (Fibonacci 或标准)
    2. (可选) 应用 Sidi 变换
    3. 计算函数值
    4. 平均得积分近似

    Args:
        func: 被积函数 f(x), x in [0,1]^s -> float
        s: 积分维度
        N: 格点数 (若为 None, 自动选择)
        use_fibonacci: 是否使用 Fibonacci 格点 (仅 2D)
        sidi_order: Sidi 变换阶数 (0 表示不使用)

    Returns:
        积分近似值
    """
    if N is None:
        N = NumericalParams.LATTICE_POINTS

    if s == 2 and use_fibonacci:
        # Fibonacci 格点 (2D)
        # 找到最接近 N 的 Fibonacci 数
        k = 1
        while fibonacci_number(k) < N:
            k += 1
        points, N_actual = fibonacci_lattice_2d(k)
    else:
        # 标准格点 (简单生成向量)
        z = np.array([1] + [fibonacci_number(i+2) for i in range(s-1)],
                     dtype=np.float64)
        z = np.mod(z, N)
        z = np.maximum(z, 1)  # 避免零
        points = standard_lattice_rule(N, z, s)
        N_actual = N

    # Sidi 变换 (非周期被积函数)
    if sidi_order > 0:
        for d in range(s):
            points[:, d] = sidi_transform(points[:, d], sidi_order)

    # 计算函数值
    f_values = np.array([func(points[j]) for j in range(N_actual)])

    return float(np.mean(f_values))


def thermal_integral(mass: float, temperature: float,
                      chemical_potential: float = 0.0,
                      is_fermion: bool = False,
                      n_momentum_dims: int = 3) -> float:
    """
    热力学动量空间积分

    n = g / (2*pi)^3 * integral d^3p / (exp((E - mu)/T) +/- 1)

    对于 Boltzmann 近似:
        n = g * (m*T / (2*pi))^{3/2} * exp(-(m-mu)/T)

    使用格点积分计算:
        I = integral_0^inf 4*pi*p^2 dp * f(p)
        变量替换: p = x/(1-x) * Lambda, x in [0,1]

    Args:
        mass: 粒子质量 (GeV)
        temperature: 温度 (GeV)
        chemical_potential: 化学势 (GeV)
        is_fermion: 是否为费米子
        n_momentum_dims: 动量空间维度

    Returns:
        数密度 n (1/fm^3)
    """
    if temperature < 1.0e-10:
        return 0.0

    Lambda = 5.0 * temperature + mass  # 动量截断

    def integrand_1d(x_arr):
        """一维径向积分"""
        x = x_arr[0] if hasattr(x_arr, '__len__') else x_arr
        if x < 1.0e-10 or x > 1.0 - 1.0e-10:
            return 0.0
        p = x / (1.0 - x) * Lambda
        dp_dx = Lambda / (1.0 - x)**2
        E = np.sqrt(p**2 + mass**2)
        exponent = (E - chemical_potential) / temperature
        if is_fermion:
            f = 1.0 / (np.exp(min(exponent, 500)) + 1.0)
        else:
            f = 1.0 / (np.exp(min(exponent, 500)) - 1.0 + 1.0e-30)
        return 4.0 * np.pi * p**2 * f * dp_dx

    # 一维积分 (使用梯形法则, 因为格点积分在一维效率不高)
    n_quad = 200
    x_arr = np.linspace(1.0e-6, 1.0 - 1.0e-6, n_quad)
    dx = x_arr[1] - x_arr[0]
    values = np.array([integrand_1d(x) for x in x_arr])
    result = np.sum(values) * dx

    # 转换单位: GeV^3 -> 1/fm^3
    result /= HBAR_C**3
    return result


def phase_space_overlap(f1_func, f2_func, s: int = 2,
                         N: int = 200) -> float:
    """
    相空间重叠积分

    I = integral f1(x) * f2(x) dx

    使用格点积分计算两个分布函数的重叠.

    Args:
        f1_func: 第一个分布函数
        f2_func: 第二个分布函数
        s: 维度
        N: 格点数

    Returns:
        重叠积分值
    """
    def product_func(x):
        return f1_func(x) * f2_func(x)

    return lattice_integrate(product_func, s, N)


def compute_stefan_boltzmann_integral(n_dim: int = 3) -> float:
    """
    验证 Stefan-Boltzmann 积分

    integral_0^inf x^n / (exp(x) - 1) dx = Gamma(n+1) * zeta(n+1)

    对于 n=3:
        = 6 * zeta(4) = 6 * pi^4/90 = pi^4/15

    Args:
        n_dim: 动量空间维度

    Returns:
        数值积分结果
    """
    def integrand(x_arr):
        x = x_arr[0] if hasattr(x_arr, '__len__') else x_arr
        if x < 1.0e-10 or x > 1.0 - 1.0e-10:
            return 0.0
        t = x / (1.0 - x)
        dt_dx = 1.0 / (1.0 - x)**2
        return t**n_dim / (np.exp(min(t, 500)) - 1.0 + 1.0e-30) * dt_dx

    n_pts = 500
    x_arr = np.linspace(1.0e-6, 1.0 - 1.0e-6, n_pts)
    dx = x_arr[1] - x_arr[0]
    values = np.array([integrand(x) for x in x_arr])
    return float(np.sum(values) * dx)
