"""
numerical_integration.py
数值积分方法用于中微子振荡概率计算

基于 quad2d 和 circle_rule 的核心算法:
    - 2D 矩形区域上的中点法则积分 (quad2d)
    - 单位圆上的等距角向积分 (circle_rule)
    - 高斯-勒让德积分
    - 辛普森法则

物理应用:
    1. 在 (E, L) 平面上二维积分振荡概率
    2. 在角度空间中积分中微子通量
    3. 在 CP 破坏相位 δ 上进行周期性积分
"""

import numpy as np


def midpoint_quad_2d(nx, ny, a, b, c, d, f):
    """
    使用 2D 中点法则估计矩形 [a,b]×[c,d] 上的积分。
    (源自 quad2d)

    公式:
        x_i = [ (2nx - 2i + 1) a + (2i - 1) b ] / (2nx)
        y_j = [ (2ny - 2j + 1) c + (2j - 1) d ] / (2ny)
        I ≈ (b-a)(d-c)/(nx*ny) * Σ_i Σ_j f(x_i, y_j)

    参数:
        nx, ny: x 和 y 方向的子区间数
        a, b:   x 方向积分限
        c, d:   y 方向积分限
        f:      被积函数 f(x, y)

    返回:
        estimate: 积分估计值
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if b <= a or d <= c:
        raise ValueError("Integration limits must satisfy a < b and c < d")

    estimate = 0.0
    for i in range(1, nx + 1):
        x = ((2 * nx - 2 * i + 1) * a + (2 * i - 1) * b) / (2 * nx)
        for j in range(1, ny + 1):
            y = ((2 * ny - 2 * j + 1) * c + (2 * j - 1) * d) / (2 * ny)
            estimate += f(x, y)

    estimate = (b - a) * (d - c) * estimate / (nx * ny)
    return estimate


def circle_rule(nt):
    """
    计算单位圆上的等距角向积分规则。
    (源自 circle_rule)

    积分公式:
        I(f) ≈ 2π * Σ_{i=1}^{nt} w_i * f(cos θ_i, sin θ_i)

    参数:
        nt: 角度分段数

    返回:
        weights: (nt,) 权重 (均为 1/nt)
        angles:  (nt,) 角度 [rad]
    """
    if nt <= 0:
        raise ValueError("nt must be positive")
    weights = np.ones(nt, dtype=np.float64) / nt
    angles = 2.0 * np.pi * np.arange(nt) / nt
    return weights, angles


def integrate_over_circle(radius, nt, f_polar):
    """
    在半径为 R 的圆盘上积分函数 f(r, θ)。

    使用极坐标:
        I = ∫_0^{2π} ∫_0^R f(r, θ) r dr dθ

    参数:
        radius:   圆盘半径
        nt:       角向分段数
        f_polar:  被积函数 f(r, theta)

    返回:
        integral: 积分值
    """
    weights, angles = circle_rule(nt)
    # 径向使用高斯-勒让德积分 (5 点)
    r_nodes, r_weights = np.polynomial.legendre.leggauss(5)
    # 映射到 [0, radius]
    r_nodes = 0.5 * radius * (r_nodes + 1.0)
    r_weights = 0.5 * radius * r_weights

    integral = 0.0
    for i in range(nt):
        theta = angles[i]
        for j in range(5):
            r = r_nodes[j]
            w = weights[i] * r_weights[j] * r  # r dr dθ 的雅可比
            integral += w * f_polar(r, theta)

    integral *= 2.0 * np.pi
    return integral


def gauss_legendre_integral_1d(f, a, b, n=16):
    """
    一维高斯-勒让德积分。

    参数:
        f: 被积函数
        a, b: 积分限
        n:    积分点数

    返回:
        integral: 积分值
    """
    if n <= 0:
        raise ValueError("n must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # 映射到 [a, b]
    t = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return np.sum(w * f(t))


def oscillation_probability_integral_2d(
        E_min, E_max, L_min, L_max,
        nx=32, ny=32,
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None, delta_m2_31=None,
        hierarchy='normal', initial_flavor=0, final_flavor=0
):
    """
    在 (E, L) 平面上二维积分振荡概率。

    物理场景:
        计算反应堆或加速器中微子实验中,
        在能量谱和基线分布上的平均振荡概率:
            ⟨P⟩ = ∫∫ P(E, L) dE dL / [(E_max-E_min)(L_max-L_min)]

    参数:
        E_min, E_max: 能量范围 [GeV]
        L_min, L_max: 基线范围 [km]
        nx, ny:       积分网格数
        ...           PMNS 参数
        initial_flavor: 0=e, 1=mu, 2=tau
        final_flavor:   0=e, 1=mu, 2=tau

    返回:
        P_avg: 平均概率
    """
    from pmns_matrix import build_pmns_matrix, build_mass_matrix

    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)
    M2 = build_mass_matrix(delta_m2_21, delta_m2_31, hierarchy)

    def prob_func(E, L):
        if E <= 0 or L < 0:
            return 0.0
        H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)
        L_ev_inv = L * 5.067730889e9

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T

        psi0 = np.zeros(3, dtype=np.complex128)
        psi0[initial_flavor] = 1.0
        psi_L = U_prop @ psi0
        return abs(psi_L[final_flavor]) ** 2

    area = (E_max - E_min) * (L_max - L_min)
    integral = midpoint_quad_2d(nx, ny, E_min, E_max, L_min, L_max, prob_func)
    P_avg = integral / area

    return float(P_avg)


def integrate_over_delta_cp(f_delta, n_points=64):
    """
    在 CP 破坏相位 δ_CP ∈ [0, 2π] 上积分函数 f(δ)。

    由于 δ_CP 是周期变量, 使用圆上的积分规则。

    参数:
        f_delta:  被积函数 f(delta) [delta in radians]
        n_points: 积分点数

    返回:
        integral: 积分值
    """
    weights, angles = circle_rule(n_points)
    # 圆规则给出的是 [0, 2π] 上的等距采样
    # f_delta 的积分 = 2π * average
    values = np.array([f_delta(a) for a in angles])
    return 2.0 * np.pi * np.mean(values)


def simpson_integral_1d(y, dx):
    """
    一维 Simpson 积分。

    参数:
        y:  等间距函数值数组
        dx: 间距

    返回:
        integral: 积分值
    """
    n = len(y)
    if n < 3 or n % 2 == 0:
        # 如果点数不够或为偶数, 使用梯形法则
        return np.trapezoid(y, dx=dx)

    integral = y[0] + y[-1]
    integral += 4.0 * np.sum(y[1:-1:2])
    integral += 2.0 * np.sum(y[2:-1:2])
    integral *= dx / 3.0
    return integral


def adaptive_integral_1d(f, a, b, tol=1e-6, max_depth=20):
    """
    自适应 Simpson 积分。

    参数:
        f:         被积函数
        a, b:      积分限
        tol:       容差
        max_depth: 最大递归深度

    返回:
        integral: 积分值
    """
    def simpson(f, a, b):
        c = 0.5 * (a + b)
        h = b - a
        return h / 6.0 * (f(a) + 4.0 * f(c) + f(b))

    def recursive(f, a, b, eps, S, fa, fb, fc, depth):
        c = 0.5 * (a + b)
        d = 0.5 * (a + c)
        e = 0.5 * (c + b)
        fd = f(d)
        fe = f(e)
        Sleft = (c - a) / 6.0 * (fa + 4.0 * fd + fc)
        Sright = (b - c) / 6.0 * (fc + 4.0 * fe + fb)
        S2 = Sleft + Sright
        if depth >= max_depth or abs(S2 - S) <= 15 * eps:
            return S2 + (S2 - S) / 15.0
        return (recursive(f, a, c, eps / 2.0, Sleft, fa, fc, fd, depth + 1) +
                recursive(f, c, b, eps / 2.0, Sright, fc, fb, fe, depth + 1))

    c = 0.5 * (a + b)
    fa, fb, fc = f(a), f(b), f(c)
    S = simpson(f, a, b)
    return recursive(f, a, b, tol, S, fa, fb, fc, 0)
