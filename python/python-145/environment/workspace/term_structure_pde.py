"""
term_structure_pde.py
=====================
博士级期限结构 PDE 求解器：HJM 框架的前向利率随机偏微分方程

本模块实现 Heath-Jarrow-Morton (HJM) 框架下的前向利率曲线演化方程的
确定性 PDE 部分求解，结合有限元空间离散与后向 Euler 时间推进。

数学理论
--------
HJM 框架基本方程:
    df(t, T) = α(t, T) dt + Σ_{i=1}^n σ_i(t, T) dW_i(t)

其中 f(t, T) 为时刻 t 观察到的、在 T 时刻生效的瞬时前向利率。

无套利漂移条件（Musiela 参数化下 s = T - t）:
    α(t, T) = Σ_{i=1}^n σ_i(t, T) ∫_t^T σ_i(t, u) du

在确定性 PDE 极限下（考虑漂移主导情形），前向利率满足输运-扩散方程:
    ∂f/∂t + ∂f/∂T = ν ∂²f/∂T² + μ(T) ∂f/∂T + F(t, T)

其中:
    ν       : 期限结构扩散系数（反映市场微观结构摩擦）
    μ(T)    : 漂移速度函数（反映期限溢价）
    F(t, T) : 外部强迫项（包含政策冲击、流动性冲击等）

定义域: (t, T) ∈ [0, t_max] × [0, T_max]
边界条件:
    f(t, 0) = r(t)        （短期利率边界，Dirichlet）
    f(t, T_max) = r_∞     （长期利率渐近值，Dirichlet）
    f(0, T) = f_0(T)      （初始期限结构，Dirichlet）

债券价格:
    P(t, T) = exp(-∫_t^T f(t, s) ds)

零息收益率:
    y(t, T) = - (1/(T-t)) ln P(t, T) = (1/(T-t)) ∫_t^T f(t, s) ds
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def musiela_drift(sigma_funcs, T, t=0.0):
    """
    计算 HJM 无套利漂移项。

    公式:
        α(t, T) = Σ_{i=1}^n σ_i(t, T) ∫_t^T σ_i(t, u) du

    为数值效率，使用 10 点 Gauss-Legendre 数值积分。

    Parameters
    ----------
    sigma_funcs : list of callable
        各因子的波动率函数 σ_i(t, T)。
    T : float
        到期期限。
    t : float
        当前时间。

    Returns
    -------
    float
        漂移项 α(t, T)。
    """
    if T <= t:
        return 0.0

    # 10-point Gauss-Legendre nodes and weights on [0, 1]
    gl_nodes = np.array([
        0.013046735741414139, 0.06746831665550773, 0.1602952158504878,
        0.2833023029353765, 0.4255628305091844, 0.5744371694908156,
        0.7166976970646235, 0.8397047841495122, 0.9325316833444923,
        0.9869532642585859
    ])
    gl_weights = np.array([
        0.03333567215434411, 0.0747256745752903, 0.109543181257991,
        0.1346333596549982, 0.1477621123573764, 0.1477621123573764,
        0.1346333596549982, 0.109543181257991, 0.0747256745752903,
        0.03333567215434411
    ])

    alpha = 0.0
    scale = T - t
    for sigma in sigma_funcs:
        sigma_T = sigma(t, T)
        u = t + scale * gl_nodes
        integrand = np.array([sigma(t, ui) for ui in u])
        integral = scale * np.dot(gl_weights, integrand)
        alpha += sigma_T * integral
    return alpha


def forward_rate_pde_rhs(t, T_grid, f_current, nu, mu_func, forcing_func,
                         sigma_funcs):
    """
    构造前向利率 PDE 的右端项（空间半离散化）。

    PDE:
        ∂f/∂t = -∂f/∂T + ν ∂²f/∂T² + μ(T) ∂f/∂T + forcing(t, T) + α(t, T)

    采用中心差分近似空间导数:
        ∂f/∂T ≈ (f_{j+1} - f_{j-1}) / (2ΔT)
        ∂²f/∂T² ≈ (f_{j+1} - 2f_j + f_{j-1}) / ΔT²

    Parameters
    ----------
    t : float
        当前时间。
    T_grid : np.ndarray
        期限网格。
    f_current : np.ndarray
        当前前向利率曲线。
    nu : float
        扩散系数，必须 >= 0。
    mu_func : callable
        漂移速度 μ(T)。
    forcing_func : callable
        外部强迫 F(t, T)。
    sigma_funcs : list of callable
        波动率函数列表。

    Returns
    -------
    dfdt : np.ndarray
        时间导数（内部节点）。
    A_fd : scipy.sparse.csr_matrix
        有限差分离散化矩阵（用于 FEM 耦合）。
    rhs_forcing : np.ndarray
        强迫项向量。
    """
    T_grid = np.asarray(T_grid, dtype=float)
    f_current = np.asarray(f_current, dtype=float)
    N = len(T_grid)

    if N < 3:
        raise ValueError("forward_rate_pde_rhs: T_grid 至少要有 3 个点")
    if nu < 0.0:
        raise ValueError("forward_rate_pde_rhs: 扩散系数 nu 必须非负")

    dT = np.diff(T_grid)
    if not np.allclose(dT, dT[0], atol=1e-12):
        # 非均匀网格：使用变步长差分
        dfdt = np.zeros(N, dtype=float)
        for j in range(1, N - 1):
            h_m = T_grid[j] - T_grid[j - 1]
            h_p = T_grid[j + 1] - T_grid[j]
            h_avg = (h_m + h_p) / 2.0

            df_dT = (f_current[j + 1] - f_current[j - 1]) / (h_m + h_p)
            d2f_dT2 = (f_current[j + 1] / (h_p * h_avg) -
                       f_current[j] / (h_m * h_p) +
                       f_current[j - 1] / (h_m * h_avg))

            mu_j = mu_func(T_grid[j])
            forcing_j = forcing_func(t, T_grid[j])

            # HJM 漂移项
            alpha_j = musiela_drift(sigma_funcs, T_grid[j], t)

            dfdt[j] = -df_dT + nu * d2f_dT2 + mu_j * df_dT + forcing_j + alpha_j
    else:
        # 均匀网格
        # TODO HOLE_1: 实现均匀网格下前向利率 PDE 的空间半离散化。
        # 需要计算内部节点 (j=1..N-2) 上的空间导数：
        #   df_dT  = (f_{j+1} - f_{j-1}) / (2*dT)
        #   d2f_dT2 = (f_{j+1} - 2*f_j + f_{j-1}) / dT^2
        # 然后组装 dfdt[j] = -df_dT + nu*d2f_dT2 + mu_j*df_dT + forcing_j + alpha_j
        # 其中 alpha_j = musiela_drift(sigma_funcs, T_grid[j], t)
        dfdt = np.zeros(N, dtype=float)
        raise NotImplementedError("HOLE_1: 均匀网格 PDE 空间离散尚未实现")

    # TODO HOLE_1: 构造有限差分矩阵 A_fd（用于与 FEM 耦合）。
    # A_fd 应满足 dfdt ≈ A_fd @ f_current + rhs_forcing
    # 对内部节点 j，矩阵第 j 行对应：
    #   (-df_dT + nu*d2f_dT2 + mu_j*df_dT) 的线性组合系数
    A_fd = sp.coo_matrix((N, N)).tocsr()

    # TODO HOLE_1: 构造强迫项向量 rhs_forcing。
    # rhs_forcing[j] = forcing_func(t, T_grid[j]) + musiela_drift(sigma_funcs, T_grid[j], t)
    rhs_forcing = np.zeros(N, dtype=float)

    return dfdt, A_fd, rhs_forcing


def bond_price_from_forward(f_curve, T_grid, t, T):
    """
    由前向利率曲线计算零息债券价格。

    公式:
        P(t, T) = exp(-∫_t^T f(t, s) ds)

    Parameters
    ----------
    f_curve : np.ndarray
        前向利率曲线 f(t, T_grid)。
    T_grid : np.ndarray
        期限网格。
    t : float
        当前时间。
    T : float
        债券到期时间，必须 T >= t。

    Returns
    -------
    float
        债券价格 P(t, T)。
    """
    if T < t:
        raise ValueError("bond_price_from_forward: T 必须 >= t")
    if T == t:
        return 1.0

    f_curve = np.asarray(f_curve, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    # 只积分 [t, T] 区间
    mask = (T_grid >= t - 1e-14) & (T_grid <= T + 1e-14)
    T_sub = T_grid[mask]
    f_sub = f_curve[mask]

    if len(T_sub) < 2:
        # 线性外推
        if len(T_sub) == 1:
            integral = f_sub[0] * (T - t)
        else:
            # 使用最近点
            idx = np.argmin(np.abs(T_grid - (t + T) / 2.0))
            integral = f_curve[idx] * (T - t)
    else:
        integral = np.trapezoid(f_sub, T_sub)

    price = np.exp(-integral)
    price = np.clip(price, 0.0, 1.0)
    return price


def zero_yield_from_forward(f_curve, T_grid, t, T):
    """
    由前向利率曲线计算零息收益率。

    公式:
        y(t, T) = (1 / (T - t)) ∫_t^T f(t, s) ds = -ln P(t, T) / (T - t)

    Parameters
    ----------
    f_curve : np.ndarray
        前向利率曲线。
    T_grid : np.ndarray
        期限网格。
    t : float
        当前时间。
    T : float
        到期时间，必须 T > t。

    Returns
    -------
    float
        零息收益率 y(t, T)。
    """
    if T <= t:
        raise ValueError("zero_yield_from_forward: T 必须 > t")

    P = bond_price_from_forward(f_curve, T_grid, t, T)
    if P <= 0.0:
        P = 1e-300
    y = -np.log(P) / (T - t)
    return max(y, 0.0)


def instantaneous_short_rate(f_curve, T_grid):
    """
    由前向利率曲线提取瞬时短期利率 r(t) = f(t, t)。

    使用线性外推:
        r(t) ≈ f(t, T_min) - (T_min - t) * (f(t, T_2) - f(t, T_1)) / (T_2 - T_1)

    Parameters
    ----------
    f_curve : np.ndarray
        前向利率曲线。
    T_grid : np.ndarray
        期限网格。

    Returns
    -------
    float
        短期利率 r(t)。
    """
    f_curve = np.asarray(f_curve, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    if len(T_grid) < 2:
        return float(f_curve[0]) if len(f_curve) > 0 else 0.0

    r = f_curve[0]
    # 保证非负
    return max(r, 0.0)


def solve_term_structure_pde(T_grid, f_init, t_max, dt,
                             nu, mu_func, forcing_func, sigma_funcs,
                             r_short_func, r_long_func):
    """
    求解期限结构 PDE 的完整流程。

    使用隐式后向 Euler 格式:
        (I - dt * A_fd) f^{n+1} = f^n + dt * forcing^n + boundary_correction

    Parameters
    ----------
    T_grid : np.ndarray
        期限空间网格。
    f_init : np.ndarray
        初始前向利率曲线 f(0, T)。
    t_max : float
        模拟终止时间。
    dt : float
        时间步长。
    nu : float
        扩散系数。
    mu_func : callable
        漂移速度 μ(T)。
    forcing_func : callable
        外部强迫 F(t, T)。
    sigma_funcs : list of callable
        波动率函数列表。
    r_short_func : callable
        短期利率边界 r(t) = f(t, 0)。
    r_long_func : callable
        长期利率边界 r_∞(t) = f(t, T_max)。

    Returns
    -------
    t_history : np.ndarray
        时间序列。
    f_history : np.ndarray, shape (n_steps+1, N)
        前向利率曲线历史。
    bond_prices : np.ndarray, shape (n_steps+1, N)
        各期限债券价格历史。
    """
    T_grid = np.asarray(T_grid, dtype=float)
    f_init = np.asarray(f_init, dtype=float)
    N = len(T_grid)
    n_steps = int(np.ceil(t_max / dt))
    dt = t_max / n_steps

    if N < 3:
        raise ValueError("solve_term_structure_pde: T_grid 至少要有 3 个点")

    t_history = np.zeros(n_steps + 1, dtype=float)
    f_history = np.zeros((n_steps + 1, N), dtype=float)
    bond_prices = np.zeros((n_steps + 1, N), dtype=float)

    f = f_init.copy()
    t_history[0] = 0.0
    f_history[0, :] = f

    # 计算初始债券价格
    for j in range(N):
        bond_prices[0, j] = bond_price_from_forward(f, T_grid, 0.0, T_grid[j])

    for step in range(n_steps):
        t = step * dt
        t_new = (step + 1) * dt

        # 构造空间离散矩阵
        _, A_fd, rhs_f = forward_rate_pde_rhs(t, T_grid, f, nu, mu_func,
                                               forcing_func, sigma_funcs)

        # 隐式格式: (I - dt * A_fd) f_new = f + dt * rhs_f
        I = sp.eye(N, format='csr')
        lhs = I - dt * A_fd
        rhs = f + dt * rhs_f

        # Dirichlet 边界条件
        lhs = lhs.tolil()
        lhs[0, :] = 0.0
        lhs[0, 0] = 1.0
        rhs[0] = r_short_func(t_new)

        lhs[N - 1, :] = 0.0
        lhs[N - 1, N - 1] = 1.0
        rhs[N - 1] = r_long_func(t_new)
        lhs = lhs.tocsr()

        f_new = spsolve(lhs, rhs)
        if f_new is None:
            raise RuntimeError("solve_term_structure_pde: 稀疏求解失败")

        f = np.asarray(f_new, dtype=float)
        # 数值鲁棒性：截断负利率
        f = np.clip(f, 0.0, None)

        t_history[step + 1] = t_new
        f_history[step + 1, :] = f

        for j in range(N):
            bond_prices[step + 1, j] = bond_price_from_forward(f, T_grid, t_new, T_grid[j])

    return t_history, f_history, bond_prices
