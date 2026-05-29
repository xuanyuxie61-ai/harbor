"""
chebyshev_schrodinger.py
========================
径向 Schrödinger 方程高精度求解器

基于种子项目 086_biharmonic_cheby1d 的 Chebyshev 谱方法思想，
本模块提供两种求解模式:
1. Chebyshev 谱微分矩阵用于高阶导数计算与精度校验
2. Numerov 方法 (标准核物理方法) 用于稳健的径向波函数积分

核心公式
--------
径向 Schrödinger 方程:
    d²u/dr² + [k² - V_eff(r)] u(r) = 0

其中 u(r) = r * R(r) 是约化径向波函数。

Chebyshev 谱微分矩阵 (Trefethen):
    x_j = cos(jπ/N)
    D_{ij} = (c_i/c_j)(-1)^{i+j}/(x_i-x_j)  (i≠j)

Numerov 递推公式 (步长 h):
    [1 + h²/12 f_{n+1}] u_{n+1} = 2[1 - 5h²/12 f_n] u_n - [1 + h²/12 f_{n-1}] u_{n-1}

其中 f(r) = k² - V_eff(r)。

相移提取 (对数导数匹配):
    L = u'(r_m) / u(r_m)
    S_l = [L H_l^{-} - (H_l^{-})'] / [(H_l^{+})' - L H_l^{+}]
    δ_l = (1/2i) ln S_l
"""

import numpy as np
from scipy.integrate import solve_ivp


def chebyshev_points(N):
    """生成 N+1 个 Chebyshev–Gauss–Lobatto 点。"""
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    return x


def chebyshev_differentiation_matrix(N):
    """
    构造 Chebyshev 谱一阶微分矩阵 D。
    参照 Trefethen (2000) 标准构造。
    """
    x = chebyshev_points(N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * ((-1.0) ** np.arange(N + 1))

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T

    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D


def chebyshev_second_derivative_matrix(N):
    """通过 D @ D 构造二阶微分矩阵。"""
    D = chebyshev_differentiation_matrix(N)
    return D @ D


def numerov_integration(f_func, r_grid, l, u_start=1e-10):
    """
    使用 Numerov 方法积分径向 Schrödinger 方程。

    Parameters
    ----------
    f_func : callable
        f(r) = k² - V_eff(r)。
    r_grid : ndarray
        等距径向网格。
    l : int
        角动量量子数。
    u_start : float
        起始点 u(r_1) 的值 (近似 r_1^{l+1} 行为)。

    Returns
    -------
    u : ndarray
        波函数值。
    """
    n = len(r_grid)
    h = r_grid[1] - r_grid[0]
    u = np.zeros(n, dtype=complex)

    # 起始条件
    # u(0) = 0, u(r_1) ≈ r_1^{l+1}
    u[0] = 0.0
    u[1] = u_start * (r_grid[1] ** (l + 1))

    f = np.array([f_func(ri) for ri in r_grid])

    # Numerov 递推
    h2_12 = h ** 2 / 12.0
    for i in range(1, n - 1):
        denom = 1.0 + h2_12 * f[i + 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        u[i + 1] = (2.0 * (1.0 - 5.0 * h2_12 * f[i]) * u[i]
                    - (1.0 + h2_12 * f[i - 1]) * u[i - 1]) / denom

    return u


def spherical_bessel_ln(x, l):
    """
    计算球 Bessel j_l(x) 和球 Neumann n_l(x)。
    使用向上递推，边界处理 x→0。
    """
    x = float(x)
    if x < 1e-12:
        return (1.0 if l == 0 else 0.0), -1e12

    j = np.zeros(l + 1)
    j[0] = np.sin(x) / x
    if l >= 1:
        j[1] = np.sin(x) / (x ** 2) - np.cos(x) / x
    for ll in range(2, l + 1):
        j[ll] = (2.0 * ll - 1.0) / x * j[ll - 1] - j[ll - 2]

    n = np.zeros(l + 1)
    n[0] = -np.cos(x) / x
    if l >= 1:
        n[1] = -np.cos(x) / (x ** 2) - np.sin(x) / x
    for ll in range(2, l + 1):
        n[ll] = (2.0 * ll - 1.0) / x * n[ll - 1] - n[ll - 2]

    return j[l], n[l]


def spherical_bessel_derivative(x, l, kind='j'):
    """f_l'(x) = f_{l-1}(x) - (l+1)/x f_l(x)。"""
    x = float(x)
    if x < 1e-12:
        return 0.0
    if kind == 'j':
        fl, _ = spherical_bessel_ln(x, l)
        f_prev = np.sin(x) / x if l == 0 else spherical_bessel_ln(x, l - 1)[0]
    else:
        _, fl = spherical_bessel_ln(x, l)
        f_prev = -np.cos(x) / x if l == 0 else spherical_bessel_ln(x, l - 1)[1]
    return f_prev - (l + 1.0) / x * fl


def solve_radial_schrodinger(params, l, j=None, n_points=2000, r_max=15.0, r_match=10.0):
    """
    求解径向 Schrödinger 方程并提取 S-矩阵与相移。

    使用 Numerov 方法进行稳健积分，同时利用 Chebyshev
    谱微分矩阵进行导数精度校验。

    Parameters
    ----------
    params : OpticalPotentialParameters
        光学势参数。
    l : int
        轨道角动量量子数。
    j : float or None
        总角动量量子数。
    n_points : int
        径向网格点数。
    r_max : float
        最大径向距离 (fm)。
    r_match : float
        匹配半径 (fm)。

    Returns
    -------
    result : dict
        包含 r, u, phase_shift, S_matrix, absorption 等。
    """
    from optical_potential import effective_potential

    # 径向网格 (内区较密)
    t = np.linspace(0.0, 1.0, n_points)
    # 幂律分布使内区更密
    r = r_max * (t ** 1.2)
    r[0] = 1e-8  # 避免零点

    # 有效势函数
    def f_func(ri):
        V_eff = effective_potential(np.array([ri]), params, l, j)[0]
        return params.k ** 2 - V_eff

    # Numerov 积分
    u = numerov_integration(f_func, r, l)

    # 找到匹配点索引
    idx_match = np.argmin(np.abs(r - r_match))
    if idx_match < 5:
        idx_match = 5
    if idx_match >= n_points - 2:
        idx_match = n_points - 3

    # 数值导数 (中心差分)
    h_m = r[idx_match + 1] - r[idx_match - 1]
    u_deriv_match = (u[idx_match + 1] - u[idx_match - 1]) / h_m
    u_match = u[idx_match]

    # Chebyshev 谱导数校验 (在匹配点附近取子区间)
    # 选取匹配点附近的点进行 Chebyshev 谱微分
    idx_cheb_start = max(0, idx_match - 15)
    idx_cheb_end = min(n_points, idx_match + 16)
    r_cheb = r[idx_cheb_start:idx_cheb_end]
    u_cheb = u[idx_cheb_start:idx_cheb_end]
    if len(r_cheb) >= 4:
        N_cheb = len(r_cheb) - 1
        D_cheb = chebyshev_differentiation_matrix(N_cheb)
        # 将 r_cheb 映射到 [-1, 1] 的 Chebyshev 点 (近似)
        # 这里使用直接映射做精度验证
        r_min_c, r_max_c = r_cheb[0], r_cheb[-1]
        x_cheb = 2.0 * (r_cheb - r_min_c) / (r_max_c - r_min_c) - 1.0
        # 链式法则: du/dr = du/dx * dx/dr = du/dx * 2/(r_max-r_min)
        du_dx = D_cheb @ u_cheb
        du_dr_cheb = du_dx * 2.0 / (r_max_c - r_min_c)
        # 找到匹配点在 Chebyshev 子网格中的位置
        idx_local = np.argmin(np.abs(r_cheb - r_match))
        cheb_deriv_check = du_dr_cheb[idx_local]
    else:
        cheb_deriv_check = u_deriv_match

    # 自由粒子 Riccati-Bessel 在匹配半径
    kr = params.k * r_match
    jl, nl = spherical_bessel_ln(kr, l)
    Rl = kr * jl
    Sl = kr * nl
    Rl_deriv = jl + kr * spherical_bessel_derivative(kr, l, 'j')
    Sl_deriv = nl + kr * spherical_bessel_derivative(kr, l, 'n')

    # HOLE 2: 请使用对数导数匹配法计算 S-矩阵与相移
    # H_l^{±} = Rl ± i Sl,  (H_l^{±})' = Rl_deriv ± i Sl_deriv
    # 对数导数 L_int = u'(r_m) / u(r_m)
    # S_l = [L_int * H_l^{-} - (H_l^{-})'] / [(H_l^{+})' - L_int * H_l^{+}]
    # δ_l = (1/2i) ln(S_l)，需归约到主值区间 (-π/2, π/2]
    L_int = u_deriv_match / u_match  # 占位保留，需融入完整匹配逻辑
    S_l = 1.0 + 0j  # 占位，不正确
    delta_l = 0.0   # 占位，不正确

    # 归一化波函数 (使大 r 处振幅合理)
    # 使用 Riccati-Bessel 函数的归一化
    norm_factor = 1.0 / np.max(np.abs(u))
    u_norm = u * norm_factor

    return {
        'r': r,
        'u': u_norm,
        'phase_shift': delta_l,
        'S_matrix': S_l,
        'absorption': abs(S_l),
        'log_derivative': L_int,
        'cheb_deriv_check': cheb_deriv_check,
        'k': params.k,
        'l': l,
        'j': j,
    }


def riccati_bessel_functions(kr, l_max):
    """
    计算 Riccati-Bessel 函数及其导数。
    ê_l(kr) = kr * j_l(kr)
    """
    Rl = np.zeros(l_max + 1)
    Rl_prime = np.zeros(l_max + 1)
    Sl = np.zeros(l_max + 1)
    Sl_prime = np.zeros(l_max + 1)

    for l in range(l_max + 1):
        jl, nl = spherical_bessel_ln(kr, l)
        Rl[l] = kr * jl
        Sl[l] = kr * nl
        Rl_prime[l] = jl + kr * spherical_bessel_derivative(kr, l, 'j')
        Sl_prime[l] = nl + kr * spherical_bessel_derivative(kr, l, 'n')

    return Rl, Rl_prime, Sl, Sl_prime


def chebyshev_spectral_verify(u_func, r_interval, N=32):
    """
    使用 Chebyshev 谱微分验证给定函数的导数精度。

    基于 086_biharmonic_cheby1d 的谱微分思想，
    在区间 r_interval 上构造 Chebyshev 网格并比较
    谱导数与解析/数值导数。

    Parameters
    ----------
    u_func : callable
        波函数 u(r)。
    r_interval : tuple
        (r_min, r_max)。
    N : int
        Chebyshev 阶数。

    Returns
    -------
    max_error : float
        谱导数与中心差分的最大偏差。
    """
    r_min, r_max = r_interval
    x_cheb = chebyshev_points(N)
    # 仿射映射: x ∈ [-1,1] -> r ∈ [r_min, r_max]
    r_cheb = 0.5 * (r_max - r_min) * (x_cheb + 1.0) + r_min
    u_vals = u_func(r_cheb)

    D = chebyshev_differentiation_matrix(N)
    du_dx = D @ u_vals
    du_dr = du_dx * 2.0 / (r_max - r_min)

    # 中心差分参考
    du_dr_fd = np.zeros_like(du_dr)
    for i in range(1, N):
        du_dr_fd[i] = (u_vals[i + 1] - u_vals[i - 1]) / (r_cheb[i + 1] - r_cheb[i - 1])
    du_dr_fd[0] = (u_vals[1] - u_vals[0]) / (r_cheb[1] - r_cheb[0])
    du_dr_fd[N] = (u_vals[N] - u_vals[N - 1]) / (r_cheb[N] - r_cheb[N - 1])

    return np.max(np.abs(du_dr - du_dr_fd))


if __name__ == "__main__":
    from optical_potential import OpticalPotentialParameters
    params = OpticalPotentialParameters('n', 56, 26, 14.0)
    res = solve_radial_schrodinger(params, l=0, n_points=1500)
    print(f"l=0: δ = {res['phase_shift']:.6f} rad, |S| = {res['absorption']:.6f}")
    res2 = solve_radial_schrodinger(params, l=2, j=2.5, n_points=1500)
    print(f"l=2,j=2.5: δ = {res2['phase_shift']:.6f} rad, |S| = {res2['absorption']:.6f}")

    # Chebyshev 精度校验
    u_func = lambda r: np.sin(params.k * r) * np.exp(-r / 5.0)
    err = chebyshev_spectral_verify(u_func, (0.5, 8.0), N=24)
    print(f"Chebyshev 谱导数最大误差: {err:.2e}")
