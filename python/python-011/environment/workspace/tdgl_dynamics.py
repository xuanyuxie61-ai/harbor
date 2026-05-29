# -*- coding: utf-8 -*-
"""
tdgl_dynamics.py
----------------
时间依赖 Ginzburg-Landau (TDGL) 方程与非线性动力学模块。

对应种子项目：
  - 1402_wave_pde：一维波动方程（Method of Lines、有限差分、周期边界）
  - 168_chen_ode：Chen 混沌吸引子（非线性 ODE 系统）

物理背景：
  超导序参量 Δ(r,t) 的动力学由含时 Ginzburg-Landau 方程描述：
      τ ∂Δ/∂t = a(T) Δ + b |Δ|^2 Δ + ξ^2 ∇^2 Δ + η(r,t)
  其中 a(T) = α (T - T_c) 在 T < T_c 时为负（序参量自发形成）。
  当系统受到强驱动（如强磁场、电流）时，序参量振幅和相位
  的耦合可能进入混沌区，可用三维非线性 ODE 系统近似。

核心公式：
  - 波动方程（Goldstone 模式，相位 θ）：
      ∂_t^2 θ = c_s^2 ∇^2 θ
    其中 c_s 为超流速度（Goldstone 声速）。
  - Chen 系统（驱动超导的非线性近似）：
      dx/dt = a (y - x)
      dy/dt = (c - a) x - x z + c y
      dz/dt = x y - b z
"""

import numpy as np
from scipy.integrate import solve_ivp


def solve_wave_equation_mol(nx=64, c=1.0, t_span=(0.0, 2.0 * np.pi), nt_eval=200):
    """
    用 Method of Lines (MOL) 求解一维波动方程：
        u_t = v
        v_t = c * u_xx
    在周期域 [0, 2π] 上，初始条件 u(x,0)=sin(x), v(x,0)=0。

    对应种子项目 1402_wave_pde 的核心算法。

    Parameters
    ----------
    nx : int
        空间格点数。
    c : float
        波速。
    t_span : tuple
        时间区间。
    nt_eval : int
        输出时间点数。

    Returns
    -------
    t : ndarray, shape (nt,)
    u : ndarray, shape (nt, nx)
    v : ndarray, shape (nt, nx)
    energy : ndarray, shape (nt,)
        总能量 H = 0.5 * (c^2 ∫ u_x^2 dx + ∫ v^2 dx)。
    """
    if nx < 3:
        raise ValueError("nx 必须 >= 3。")
    L = 2.0 * np.pi
    dx = L / nx
    x = np.linspace(0, L, nx, endpoint=False)

    # 初始条件
    u0 = np.sin(x)
    v0 = np.zeros_like(x)
    w0 = np.concatenate([u0, v0])

    def deriv(t, w):
        u = w[:nx]
        v = w[nx:]
        # 周期边界离散 Laplacian
        u_xx = np.zeros_like(u)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            u_xx[i] = (u[im] - 2.0 * u[i] + u[ip]) / (dx ** 2)
        du = v
        dv = c * u_xx
        return np.concatenate([du, dv])

    t_eval = np.linspace(t_span[0], t_span[1], nt_eval)
    sol = solve_ivp(deriv, t_span, w0, t_eval=t_eval, method='RK45')

    t = sol.t
    nt = t.size
    u = sol.y[:nx, :].T
    v = sol.y[nx:, :].T

    # 计算能量
    energy = np.zeros(nt)
    for it in range(nt):
        ux = np.zeros(nx)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            ux[i] = (u[it, ip] - u[it, im]) / (2.0 * dx)
        H_u = 0.5 * c ** 2 * np.sum(ux ** 2) * dx
        H_v = 0.5 * np.sum(v[it, :] ** 2) * dx
        energy[it] = H_u + H_v

    return t, u, v, energy


def chen_system_rhs(t, xyz, a=40.0, b=3.0, c=28.0):
    """
    Chen 混沌系统的右端项。

    dx/dt = a (y - x)
    dy/dt = (c - a) x - x z + c y
    dz/dt = x y - b z

    对应种子项目 168_chen_ode。
    """
    x, y, z = xyz
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz])


def solve_chen_system(t_span=(0.0, 15.0), y0=None, params=None, nt_eval=2000):
    """
    数值积分 Chen 系统，用于模拟强驱动下超导序参量的非线性混沌动力学。

    Parameters
    ----------
    t_span : tuple
    y0 : array_like, shape (3,)
        初始条件，默认 [-0.1, 0.5, -0.6]。
    params : dict, optional
        {'a':40.0, 'b':3.0, 'c':28.0}
    nt_eval : int
        输出点数。

    Returns
    -------
    t : ndarray
    sol : ndarray, shape (nt, 3)
    lyapunov_estimate : float
        最大 Lyapunov 指数近似值（通过相邻轨道发散率）。
    """
    if y0 is None:
        y0 = np.array([-0.1, 0.5, -0.6])
    if params is None:
        params = {'a': 40.0, 'b': 3.0, 'c': 28.0}
    a, b, c = params['a'], params['b'], params['c']
    t_eval = np.linspace(t_span[0], t_span[1], nt_eval)

    sol = solve_ivp(
        lambda t, y: chen_system_rhs(t, y, a, b, c),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12
    )
    t = sol.t
    y = sol.y.T

    # 粗略估计最大 Lyapunov 指数
    eps0 = 1e-10
    y0_pert = y0 + np.array([eps0, 0.0, 0.0])
    sol2 = solve_ivp(
        lambda t, y: chen_system_rhs(t, y, a, b, c),
        t_span, y0_pert, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12
    )
    y2 = sol2.y.T
    delta = np.linalg.norm(y2 - y, axis=1)
    # 取后半段线性拟合 log(delta)
    mask = delta > 1e-15
    if np.sum(mask) > 10:
        logd = np.log(delta[mask])
        tt = t[mask]
        # 简单斜率估计
        lam = np.mean(np.diff(logd) / np.diff(tt))
    else:
        lam = 0.0

    return t, y, lam


def tdgl_evolution_1d(Nx=128, L=10.0, T=1.0, Tc=1.5, tau=1.0, xi=1.0,
                      b_coeff=-1.0, t_span=(0.0, 10.0), nt=500):
    """
    一维含时 Ginzburg-Landau 方程：
        τ ∂_t Δ = a(T) Δ + b |Δ|^2 Δ + ξ^2 ∂_x^2 Δ
    其中 a(T) = α (T - T_c)。
    在 T < T_c 时 a < 0，系统自发对称破缺。

    使用有限差分 + 显式 Euler（小时间步）或 solve_ivp。
    这里用 solve_ivp 的 Method of Lines。

    Parameters
    ----------
    Nx : int
        空间格点数。
    L : float
        系统长度。
    T : float
        温度（以 Tc 为单位或绝对值）。
    Tc : float
        临界温度。
    tau : float
        弛豫时间。
    xi : float
        相干长度。
    b_coeff : float
        GL 展开四次项系数（通常为负以稳定解）。
    t_span : tuple
        时间区间。
    nt : int
        输出时间步数。

    Returns
    -------
    t : ndarray
    Delta : ndarray, shape (nt, Nx)
        序参量实部。
    max_amplitude : ndarray, shape (nt,)
    """
    if Nx < 3:
        raise ValueError("Nx 必须 >= 3。")
    dx = L / Nx
    alpha = 1.0  # 比例系数
    a_T = alpha * (T - Tc)

    x = np.linspace(0, L, Nx, endpoint=False)
    # 初始条件：小随机扰动
    Delta0 = 0.1 * np.random.randn(Nx)

    def deriv(t, D):
        D_xx = np.zeros_like(D)
        for i in range(Nx):
            im = (i - 1) % Nx
            ip = (i + 1) % Nx
            D_xx[i] = (D[im] - 2.0 * D[i] + D[ip]) / (dx ** 2)
        # 标准 TDGL: τ ∂Δ/∂t = -a(T) Δ - b |Δ|^2 Δ + ξ^2 ∇^2 Δ
        # 其中 a(T) = α(T-Tc) < 0 for T<Tc, b>0
        # 因此 -a(T) > 0 驱动序参量增长，-b|Δ|^2 Δ 提供饱和
        dD = (-a_T * D - abs(b_coeff) * D ** 3 + xi ** 2 * D_xx) / tau
        return dD

    t_eval = np.linspace(t_span[0], t_span[1], nt)
    sol = solve_ivp(deriv, t_span, Delta0, t_eval=t_eval, method='RK45')
    t = sol.t
    Delta = sol.y.T
    max_amp = np.max(np.abs(Delta), axis=1)
    return t, Delta, max_amp


def compute_superfluid_stiffness(order_parameter_history, dx, dt):
    """
    从 TDGL 演化历史估算超流刚度 ρ_s。

    定义：ρ_s ∝ < |Δ|^2 >，在 GL 理论中
      ρ_s = (ħ^2 / m) |ψ|^2
    这里用序参量振幅平方的平均作为代理量。
    """
    Delta = np.asarray(order_parameter_history, dtype=float)
    if Delta.size == 0:
        return 0.0
    amp2 = np.mean(Delta ** 2, axis=1)
    # 取稳态值（后 20% 时间平均）
    n = amp2.size
    start = int(0.8 * n)
    if start >= n:
        start = n - 1
    return np.mean(amp2[start:])
