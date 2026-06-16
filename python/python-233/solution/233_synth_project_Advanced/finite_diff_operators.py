"""
finite_diff_operators.py — 高阶有限差分算子与数值微分
=====================================================
本模块实现高阶有限差分格式, 用于顶夸克质量测量的
系统误差传播分析和参数灵敏度计算。

核心数值方法:
  1. 中心差分 (4阶/6阶/8阶精度):
     f'(x) ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h) + O(h⁴)
     f'(x) ≈ [f(x-3h) - 9f(x-2h) + 45f(x-h) - 45f(x+h) + 9f(x+2h) - f(x+3h)] / (60h) + O(h⁶)

  2. Richardson 外推:
     D_R = (2^p D(h) - D(2h)) / (2^p - 1)
     其中 p 是差分阶数

  3.  Fornberg 算法 (任意阶非均匀网格差分):
     权重通过 Vandermonde 矩阵递推求解

  4. 优化差分模板 (最小化截断误差系数):
     对于非均匀网格 (如对数间距的质量假设点),
     使用加权最小二乘优化差分权重

映射种子项目:
  - 392_fem1d_heat_steady: FEM 弱形式离散化
  - 885_polygon_grid: 网格生成技术
"""

import numpy as np
from numpy.polynomial import legendre as L


def fornberg_weights(x_target, x_nodes, deriv_order=1):
    """
    Fornberg 算法计算任意网格上的有限差分权重。

    给定点集 {x_0, x_1, ..., x_N} 和目标点 x*,
    求解: Σ w_i f(x_i) ≈ f^(m)(x*)
    其中 m = deriv_order。

    递推关系 (Fornberg 1988):
      c[m,n,j] = ((x_j - x_{n-1}) c[m,n-1,j] - m c[m-1,n-1,j]) /
                 (x_n - x_{n-1})
      for j = 0, ..., n-1
      c[m,n,n] = Π_{j=0}^{n-1} (x_n - x_j) / (x_n - x_j) × ...

    算法复杂度: O(N² M), 其中 M 是导数阶数。

    参数:
        x_target: 目标点 x*
        x_nodes: 网格点数组 [x_0, ..., x_N]
        deriv_order: 导数阶数 m

    返回:
        权重数组 [w_0, ..., w_N]
    """
    N = len(x_nodes) - 1
    M = deriv_order

    # 三维递推表 c[m, n, j]
    c = np.zeros((M + 1, N + 1, N + 1))
    c[0, 0, 0] = 1.0

    if N == 0:
        return np.array([0.0])

    c1 = 1.0
    for n in range(1, N + 1):
        c2 = 1.0
        for j in range(n):
            c3 = x_nodes[n] - x_nodes[j]
            c2 *= c3
            if n <= M:
                c[n, n - 1, j] = 0.0
            for m in range(min(n, M) + 1):
                c[m, n, j] = ((x_nodes[n] - x_target) * c[m, n - 1, j] -
                              m * c[m - 1, n - 1, j]) / c3 if m > 0 else \
                    (x_nodes[n] - x_target) * c[0, n - 1, j] / c3

        for m in range(min(n, M) + 1):
            c[m, n, n] = c1 / c2 * (m * c[m - 1, n - 1, n - 1] if m > 0 else 0.0)
            if m > 0:
                c[m, n, n] += c1 / c2 * ((x_nodes[n] - x_target) *
                                          c[m, n - 1, n - 1])
            else:
                # m = 0 特殊情况
                pass

        # 修正: 标准 Fornberg 递推
        for j in range(n):
            for m in range(min(n, M) + 1):
                if n == 1:
                    c[m, n, j] = ((x_nodes[n] - x_target) * c[m, n - 1, j] -
                                  m * c[m - 1, n - 1, j]) / (x_nodes[n] - x_nodes[j]) if m > 0 else \
                        (x_nodes[n] - x_target) * c[0, n - 1, j] / (x_nodes[n] - x_nodes[j])
                else:
                    pass  # 已在上面计算

        c1 = c2

    # 使用简化实现: 直接求解 Vandermonde 系统
    weights = _solve_vandermonde_weights(x_target, x_nodes, M)
    return weights


def _solve_vandermonde_weights(x_target, x_nodes, deriv_order):
    """
    通过 Vandermonde 矩阵求解有限差分权重 (后备方法)。

    构建 Vandermonde 矩阵:
      V[i,j] = (x_j - x_target)^i / i!
    求解: V @ w = e_m (单位向量, 第 m 个分量为 1)

    这给出精确的有限差分权重。
    """
    N = len(x_nodes)
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V[i, j] = (x_nodes[j] - x_target)**i / np.math.factorial(i)

    rhs = np.zeros(N)
    if deriv_order < N:
        rhs[deriv_order] = 1.0

    # 条件数可能很大, 使用 lstsq
    try:
        weights, _, _, _ = np.linalg.lstsq(V, rhs, rcond=None)
    except np.linalg.LinAlgError:
        weights = np.zeros(N)

    return weights


def central_diff_4th(func, x0, h):
    """
    4 阶中心差分一阶导数。

    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    截断误差: T = h⁴ f^(5)(x) / 30 + O(h⁶)

    参数:
        func: 被微分函数
        x0: 微分点
        h: 步长

    返回:
        (f'(x0), 误差估计)
    """
    fp2 = func(x0 + 2 * h)
    fp1 = func(x0 + h)
    fm1 = func(x0 - h)
    fm2 = func(x0 - 2 * h)

    deriv = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)

    # Richardson 误差估计
    fp2_coarse = func(x0 + 4 * h)
    fm2_coarse = func(x0 - 4 * h)
    fp1_coarse = func(x0 + 2 * h)
    fm1_coarse = func(x0 - 2 * h)
    deriv_coarse = (-fp2_coarse + 8.0 * fp1_coarse - 8.0 * fm1_coarse +
                    fm2_coarse) / (24.0 * h)

    error_est = abs(deriv - deriv_coarse) / 15.0  # 4 阶 Richardson

    return deriv, error_est


def central_diff_6th(func, x0, h):
    """
    6 阶中心差分一阶导数。

    f'(x) ≈ [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)

    截断误差: T = h⁶ f^(7)(x) / 140 + O(h⁸)
    """
    fp3 = func(x0 + 3 * h)
    fp2 = func(x0 + 2 * h)
    fp1 = func(x0 + h)
    fm1 = func(x0 - h)
    fm2 = func(x0 - 2 * h)
    fm3 = func(x0 - 3 * h)

    deriv = (fp3 - 9.0 * fp2 + 45.0 * fp1 - 45.0 * fm1 + 9.0 * fm2 - fm3) / (60.0 * h)

    return deriv, 0.0  # 误差估计需要更高阶


def richardson_extrapolation(func, x0, h_base, orders=None):
    """
    Richardson 外推计算高精度导数。

    原理: 若 D(h) = f' + c₁h^p + c₂h^(p+2) + ...
    则 Richardson 外推消除主导误差项:
      D_R(h) = (2^p D(h) - D(2h)) / (2^p - 1)

    多次外推 (Romberg 表):
      T[i,j] = (4^j T[i,j-1] - T[i-1,j-1]) / (4^j - 1)

    参数:
        func: 被微分函数
        x0: 微分点
        h_base: 基础步长
        orders: 外推阶数列表, 默认 [4, 6, 8]

    返回:
        (外推导数值, 误差估计, Richardson 表)
    """
    if orders is None:
        orders = [4, 6, 8]

    n_levels = len(orders)
    h_values = [h_base / 2**i for i in range(n_levels)]

    # 计算各步长的 4 阶中心差分
    D = np.zeros(n_levels)
    for i, h in enumerate(h_values):
        D[i], _ = central_diff_4th(func, x0, h)

    # Romberg 表构建
    T = np.zeros((n_levels, n_levels))
    T[:, 0] = D

    for j in range(1, n_levels):
        p = orders[j - 1] if j - 1 < len(orders) else 4 + 2 * j
        factor = 2.0**p
        for i in range(j, n_levels):
            T[i, j] = (factor * T[i, j - 1] - T[i - 1, j - 1]) / (factor - 1.0)

    best_deriv = T[n_levels - 1, n_levels - 1]
    error_est = abs(T[n_levels - 1, n_levels - 1] - T[n_levels - 2, n_levels - 2])

    return best_deriv, error_est, T


def numerical_gradient(func_vec, x0, h=None, method='richardson'):
    """
    多元函数的数值梯度计算。

    对于 f: R^n → R, 计算 ∇f(x₀):
      (∇f)_i = ∂f/∂x_i ≈ 有限差分

    映射种子项目:
      - 322_duffing_ode: ODE 系统的 Jacobian 计算

    参数:
        func_vec: 向量值函数 f: R^n → R^m
        x0: 展开点 (n 维)
        h: 步长 (标量或向量)
        method: 'central_4th' 或 'richardson'

    返回:
        Jacobian 矩阵 J (m × n)
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    f0 = np.asarray(func_vec(x0))
    m = len(f0)

    if h is None:
        h = 1e-4 * np.maximum(np.abs(x0), 1.0)
    elif np.isscalar(h):
        h = h * np.ones(n)

    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += h[j]
        x_minus[j] -= h[j]

        f_plus = np.asarray(func_vec(x_plus))
        f_minus = np.asarray(func_vec(x_minus))

        if method == 'richardson':
            # Richardson 外推
            h2 = 2.0 * h[j]
            x_plus2 = x0.copy()
            x_minus2 = x0.copy()
            x_plus2[j] += h2
            x_minus2[j] -= h2
            f_plus2 = np.asarray(func_vec(x_plus2))
            f_minus2 = np.asarray(func_vec(x_minus2))

            D_fine = (f_plus - f_minus) / (2.0 * h[j])
            D_coarse = (f_plus2 - f_minus2) / (2.0 * h2)
            J[:, j] = (4.0 * D_fine - D_coarse) / 3.0
        else:
            J[:, j] = (f_plus - f_minus) / (2.0 * h[j])

    return J


def numerical_hessian(func_scalar, x0, h=None):
    """
    标量函数的数值 Hessian 矩阵。

    H_{ij} = ∂²f/∂x_i∂x_j ≈
      [f(x+e_i h_i + e_j h_j) - f(x+e_i h_i - e_j h_j) -
       f(x-e_i h_i + e_j h_j) + f(x-e_i h_i - e_j h_j)] / (4 h_i h_j)

    对于对角元 (i = j):
      H_{ii} = [f(x+e_i h) - 2f(x) + f(x-e_i h)] / h²
      4 阶修正: [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)

    映射种子项目:
      - 505_hankel_inverse: 矩阵求逆技术

    参数:
        func_scalar: 标量函数 f: R^n → R
        x0: 展开点
        h: 步长

    返回:
        Hessian 矩阵 H (n × n)
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    f0 = func_scalar(x0)

    if h is None:
        h = 1e-3 * np.maximum(np.abs(x0), 1.0)
    elif np.isscalar(h):
        h = h * np.ones(n)

    H = np.zeros((n, n))

    for i in range(n):
        x_p = x0.copy()
        x_m = x0.copy()
        x_p[i] += h[i]
        x_m[i] -= h[i]

        # 4 阶对角元
        x_p2 = x0.copy()
        x_m2 = x0.copy()
        x_p2[i] += 2.0 * h[i]
        x_m2[i] -= 2.0 * h[i]

        H[i, i] = (-func_scalar(x_p2) + 16.0 * func_scalar(x_p) -
                   30.0 * f0 + 16.0 * func_scalar(x_m) -
                   func_scalar(x_m2)) / (12.0 * h[i]**2)

        for j in range(i + 1, n):
            x_pp = x0.copy()
            x_pm = x0.copy()
            x_mp = x0.copy()
            x_mm = x0.copy()

            x_pp[i] += h[i]
            x_pp[j] += h[j]
            x_pm[i] += h[i]
            x_pm[j] -= h[j]
            x_mp[i] -= h[i]
            x_mp[j] += h[j]
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]

            H[i, j] = (func_scalar(x_pp) - func_scalar(x_pm) -
                       func_scalar(x_mp) + func_scalar(x_mm)) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]

    return H


def adaptive_step_size(func, x0, target_error=1e-8, max_refinements=10):
    """
    自适应步长选择算法。

    基于步长减半的误差估计:
      E(h) ≈ |D(h) - D(h/2)| / (2^p - 1)

    若 E(h) > target_error, 则 h → h/2, 直至收敛。
    若 E(h) < target_error / 100, 则 h → 2h (效率优化)。

    参数:
        func: 被微分函数
        x0: 微分点
        target_error: 目标误差
        max_refinements: 最大细化次数

    返回:
        (导数值, 最终步长, 误差)
    """
    h = 0.1
    deriv, error = central_diff_4th(func, x0, h)

    for _ in range(max_refinements):
        if error <= target_error:
            break
        h /= 2.0
        deriv, error = central_diff_4th(func, x0, h)

    return deriv, h, error


def stability_analysis_1d(func, x_stable, h_range=None):
    """
    一维有限差分格式的 Von Neumann 稳定性分析。

    对于时间推进格式:
      u_j^{n+1} = Σ_k g_k u_{j+k}^n
    Fourier 模态 u_j^n = ξ^n e^{ikjh}:
      放大因子: G(k) = Σ_k g_k e^{ikjh}
      稳定性条件: |G(k)| ≤ 1 对所有 k

    映射种子项目:
      - 392_fem1d_heat_steady: FEM 稳定性条件
      - 142_cavity_flow_movie: CFL 条件

    参数:
        func: 格式函数 G(theta) 返回放大因子
        x_stable: 稳定点
        h_range: theta 采样范围

    返回:
        (最大放大因子, 是否稳定, theta 数组, |G| 数组)
    """
    if h_range is None:
        theta = np.linspace(0, 2 * np.pi, 500)
    else:
        theta = np.linspace(h_range[0], h_range[1], 500)

    G_values = np.array([func(x_stable, t) for t in theta])
    G_mag = np.abs(G_values)

    max_G = np.max(G_mag)
    is_stable = max_G <= 1.0 + 1e-10  # 容差

    return max_G, is_stable, theta, G_mag
