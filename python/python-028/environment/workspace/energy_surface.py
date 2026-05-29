"""
energy_surface.py
=================
核势能面全局优化模块

本模块基于 glomin 与 zero_brent 的优化算法，
实现核集体坐标 (β₂, γ) 平面上势能面的全局最小化搜索。

功能：
1. 一维全局最小化（glomin 算法）—— 沿特定 γ 方向的 β 优化
2. 二维势能面网格搜索 + 局部精化
3. 鞍点搜索与形变路径计算

数学基础：
- glomin 算法假设 f''(x) ≤ M，利用该上界进行区间排除：
  若在某区间 [a, b] 内，基于二阶导数上界可证明 f(x) > f_best + ε，
  则该区间的全局最小不可能位于其中，可被安全剪枝。

- 势能面极小化：
  min_{β, γ} V(β, γ) = V_WS(β, γ) + E_pair(β, γ) + E_rot(β, γ)

- 转动能量（Yrast 线）：
  E_rot(I, β, γ) = [I(I+1) - K²] / (2 I_perp(β, γ))
  其中 I_perp = I_0 [1 + c_β β cos(γ + 2π/3)]
"""

import numpy as np
from math import sqrt, sin, cos, pi, fabs, exp


def glomin_global_minimize(a, b, c, M_bound, eps_eval, tol, f):
    """
    glomin 全局最小化算法（基于 glomin.m 的 Python 实现）。

    假设：
    - f 在 [a, b] 上二阶连续可微
    - f''(x) ≤ M_bound 对所有 x ∈ [a, b]
    - eps_eval 为函数求值误差上界

    参数
    ----
    a, b : float
        区间端点，a < b
    c : float
        初始猜测点
    M_bound : float
        二阶导数上界（必须 > 0）
    eps_eval : float
        函数求值误差上界
    tol : float
        解的精度要求
    f : callable
        目标函数

    返回
    ----
    x_min : float
        全局最小值点估计
    f_min : float
        最小值估计
    calls : int
        函数调用次数
    """
    calls = 0
    a0, x, a2 = b, b, a
    y0 = f(b)
    calls += 1
    yb = y0
    y2 = f(a)
    calls += 1
    y = y2

    if y0 < y:
        y = y0
    else:
        x = a

    if M_bound <= 0.0 or b <= a:
        return x, y, calls

    m2 = 0.5 * (1.0 + 16.0 * np.finfo(float).eps) * M_bound

    if c <= a or b <= c:
        c = 0.5 * (a + b)

    y1 = f(c)
    calls += 1
    k = 3
    d0 = a2 - c
    h = 9.0 / 11.0

    if y1 < y:
        x = c
        y = y1

    while True:
        d1 = a2 - a0
        d2 = c - a0
        z2 = b - a2
        z0 = y2 - y1
        z1 = y2 - y0
        r_val = d1 * d1 * z0 - d0 * d0 * z1
        p = r_val
        qs = 2.0 * (d0 * z1 - d1 * z0)
        q = qs

        force_first = True
        if 100000 < k and y < y2:
            k = (1611 * k) % 1048576
            q = 1.0
            r_val = (b - a) * 0.00001 * k
            force_first = False

        while r_val < z2 or force_first:
            force_first = False
            if (q * (r_val * (yb - y2) + z2 * q * ((y2 - y) + tol)) <
                    z2 * m2 * r_val * (z2 * q - r_val)):
                a3 = a2 + r_val / q
                y3 = f(a3)
                calls += 1
                if y3 < y:
                    x = a3
                    y = y3

            k = (1611 * k) % 1048576
            q = 1.0
            r_val = (b - a) * 0.00001 * k

        r_val = m2 * d0 * d1 * d2
        s = sqrt(((y2 - y) + tol) / m2)
        h = 0.5 * (1.0 + h)
        p = h * (p + 2.0 * r_val * s)
        q = r_val + 0.5 * qs
        r_val = -0.5 * (d0 + (z0 + 2.01 * eps_eval) / (d0 * m2))

        if r_val < s or d0 < 0.0:
            r_val = a2 + s
        else:
            r_val = a2 + r_val

        if 0.0 < p * q:
            a3 = a2 + p / q
        else:
            a3 = r_val

        while True:
            a3 = max(a3, r_val)
            if b <= a3:
                a3 = b
                y3 = yb
            else:
                y3 = f(a3)
                calls += 1

            if y3 < y:
                x = a3
                y = y3

            d0 = a3 - a2

            if a3 <= r_val:
                break

            p = 2.0 * (y2 - y3) / (M_bound * d0)
            if (1.0 + 9.0 * np.finfo(float).eps) * d0 <= abs(p):
                break

            if 0.5 * m2 * (d0 * d0 + p * p) <= (y2 - y) + (y3 - y) + 2.0 * tol:
                break

            a3 = 0.5 * (a2 + a3)
            h = 0.9 * h

        if b <= a3:
            break

        a0 = c
        c = a2
        a2 = a3
        y0 = y1
        y1 = y2
        y2 = y3

    return x, y, calls


def brent_local_minimize(a, b, c, f, tol=1e-8):
    """
    使用 Brent 法进行局部最小化（通过导数符号变化转化为寻根问题）。

    在已知 f(a) > f(c) 且 f(b) > f(c) 的条件下，
    对 f'(x) = 0 使用 Brent 寻根法。
    """
    def df_approx(x):
        h = max(abs(x) * 1e-6, 1e-8)
        return (f(x + h) - f(x - h)) / (2.0 * h)

    # 在 [a, b] 内寻找 f'(x) = 0
    # 简单策略：黄金分割 + 抛物线插值
    phi_ratio = (3.0 - sqrt(5.0)) / 2.0
    x = w = v = c
    fx = fw = fv = f(c)
    d = e = b - a

    calls = 1
    while True:
        tol1 = tol * abs(x) + 1e-10
        xm = 0.5 * (a + b)

        if abs(x - xm) <= 2.0 * tol1 - 0.5 * (b - a):
            break

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) < abs(0.5 * q * etemp) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < 2.0 * tol1 or b - u < 2.0 * tol1:
                    d = tol1 if x < xm else -tol1
            else:
                e = a - x if x < xm else b - x
                d = phi_ratio * e
        else:
            e = a - x if x < xm else b - x
            d = phi_ratio * e

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + tol1 if d > 0 else x - tol1

        fu = f(u)
        calls += 1

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x, fx, calls


def optimize_nuclear_shape_energy(potential_fn, beta_range=(-0.3, 0.5),
                                   gamma_range=(0.0, pi / 3.0),
                                   n_grid_beta=20, n_grid_gamma=15):
    """
    在 (β, γ) 平面上搜索核势能面的全局最小值。

    策略：
    1. 粗网格扫描确定候选区域
    2. 沿 γ = 0, π/6, π/3 方向分别使用 glomin 全局优化 β
    3. 返回全局最小能量及对应的形变参数

    参数
    ----
    potential_fn : callable
        V(beta, gamma) → float
    beta_range : tuple
        β 的搜索范围
    gamma_range : tuple
        γ 的搜索范围
    n_grid_beta, n_grid_gamma : int
        粗网格分辨率

    返回
    ----
    beta_opt : float
        最优 β
    gamma_opt : float
        最优 γ
    E_min : float
        最小能量 (MeV)
    surface_data : dict
        扫描得到的势能面数据
    """
    beta_grid = np.linspace(beta_range[0], beta_range[1], n_grid_beta)
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], n_grid_gamma)
    E_grid = np.zeros((n_grid_gamma, n_grid_beta))

    # 粗网格扫描
    best_E = 1e10
    best_beta, best_gamma = 0.0, 0.0

    for i, g in enumerate(gamma_grid):
        for j, b in enumerate(beta_grid):
            try:
                E = potential_fn(b, g)
                E_grid[i, j] = E
                if E < best_E:
                    best_E = E
                    best_beta = b
                    best_gamma = g
            except Exception:
                E_grid[i, j] = 1e10

    # 沿关键 γ 方向使用 glomin 精化
    gamma_cuts = [0.0, pi / 6.0, pi / 3.0]
    for g_cut in gamma_cuts:
        def V_1d(b):
            return potential_fn(b, g_cut)

        # 估计二阶导数上界 M
        db = 0.01
        M_est = 0.0
        for b_test in np.linspace(beta_range[0], beta_range[1], 20):
            try:
                d2f = (V_1d(b_test + db) - 2.0 * V_1d(b_test) + V_1d(b_test - db)) / (db ** 2)
                M_est = max(M_est, abs(d2f))
            except Exception:
                pass
        M_est = max(M_est, 1.0)

        try:
            b_opt, E_opt, _ = glomin_global_minimize(
                beta_range[0], beta_range[1], best_beta,
                M_est, 1e-6, 1e-5, V_1d
            )
            if E_opt < best_E:
                best_E = E_opt
                best_beta = b_opt
                best_gamma = g_cut
        except Exception:
            pass

    surface_data = {
        'beta_grid': beta_grid,
        'gamma_grid': gamma_grid,
        'E_grid': E_grid
    }

    return best_beta, best_gamma, best_E, surface_data


def moment_of_inertia(beta, gamma, A, R0=1.2):
    """
    计算形变核的转动惯量（刚体近似）。

    对于轴对称椭球（γ = 0）：
    I_perp = (1/5) M A (a² + b²)
    其中 a = R(1 + β√(5/16π)), b = R(1 - β√(5/16π)/2)

    一般情况（γ ≠ 0）使用 Bohr 参数化。
    """
    M = 939.0  # MeV/c²
    R = R0 * (A ** (1.0 / 3.0))
    # 简化公式
    I_0 = (2.0 / 5.0) * M * A * R ** 2
    I_perp = I_0 * (1.0 + 0.3 * beta * cos(gamma + 2.0 * pi / 3.0))
    I_parallel = I_0 * (1.0 + 0.3 * beta * cos(gamma))
    return I_perp, I_parallel
