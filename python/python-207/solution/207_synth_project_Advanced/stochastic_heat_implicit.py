"""
stochastic_heat_implicit.py — 随机热传导方程的隐式有限差分求解器

科学背景
========
求解一维随机热传导方程:

    ∂u/∂t = ∂/∂x [ κ(x,ω) · ∂u/∂x ] + f(x,t)
    u(0,t) = u_L,   u(L,t) = u_R          (Dirichlet)
    u(x,0) = u₀(x)

其中 κ(x,ω) 为随机扩散系数场 (由 KL 展开生成).

隐式 (后向 Euler) 离散化
========================
在空间节点 x_i = i·Δx (i=0,...,N) 上:

    [u_i^{n+1} - u_i^n] / Δt =
        [κ_{i+1/2}·(u_{i+1}^{n+1} - u_i^{n+1})
       - κ_{i-1/2}·(u_i^{n+1} - u_{i-1}^{n+1})] / Δx²
       + f_i^{n+1}

其中 κ_{i+1/2} = 0.5·(κ_i + κ_{i+1}) 为半节点插值.

整理得三对角系统:
    -r_{i-1/2}·u_{i-1} + (1 + r_{i-1/2} + r_{i+1/2})·u_i
        - r_{i+1/2}·u_{i+1} = u_i^n + Δt·f_i

其中 r_{i+1/2} = Δt·κ_{i+1/2} / Δx²

算法来源 (种子项目 361_fd1d_heat_implicit)
==========================================
直接移植隐式格式的系统矩阵构造和三对角求解.
改进: 支持空间变化的 κ(x), 源项 f(x,t), 多种边界条件.

核心公式
========
1. Fourier 数:  Fo = κ·Δt/Δx²
2. 半节点插值:  κ_{i+1/2} = (κ_i + κ_{i+1})/2
3. 局部 Courant 数:  r_{i+1/2} = Δt·κ_{i+1/2}/Δx²
4. 三对角系统:  A·u^{n+1} = u^n + Δt·f
5. Thomas 算法:  O(N) 求解三对角系统

稳定性
======
隐式格式理论无条件稳定, 但当 Fo >> 1 时:
- 时间截断误差 O(Δt) 可能主导
- 建议 Fo ≤ 5 以保证精度
"""

import numpy as np


def build_implicit_system_matrix(nx, dx, dt, kappa_field):
    """构造隐式格式的系统矩阵 A (三对角).

    A·u^{n+1} = u^n + Δt·f

    参数
    ----
    nx : int
        空间节点数 (含边界)
    dx : float
        空间步长
    dt : float
        时间步长
    kappa_field : ndarray, shape (nx,)
        各节点处的扩散系数 κ(x_i)

    返回
    ----
    A : ndarray, shape (nx, nx)
        三对角系统矩阵
    lower_diag : ndarray, shape (nx-1,)
        下次对角线
    main_diag : ndarray, shape (nx,)
        主对角线
    upper_diag : ndarray, shape (nx-1,)
        上次对角线

    算法 (种子 361)
    ===============
    for i = 1,...,nx-2:
        κ_{i-1/2} = 0.5*(κ[i-1] + κ[i])
        κ_{i+1/2} = 0.5*(κ[i] + κ[i+1])
        r_{i-1/2} = dt * κ_{i-1/2} / dx²
        r_{i+1/2} = dt * κ_{i+1/2} / dx²
        A[i,i-1] = -r_{i-1/2}
        A[i,i]   = 1 + r_{i-1/2} + r_{i+1/2}
        A[i,i+1] = -r_{i+1/2}
    边界行: A[0,0] = 1, A[nx-1,nx-1] = 1 (Dirichlet)
    """
    # 半节点扩散系数
    kappa_half = 0.5 * (kappa_field[:-1] + kappa_field[1:])  # shape (nx-1,)

    # 局部 Courant 数 r_{i+1/2} = dt * κ_{i+1/2} / dx²
    r_half = dt * kappa_half / (dx ** 2)

    # 构造对角线
    main_diag = np.ones(nx)
    lower_diag = np.zeros(nx - 1)
    upper_diag = np.zeros(nx - 1)

    for i in range(1, nx - 1):
        r_left = r_half[i - 1]   # r_{i-1/2}
        r_right = r_half[i]      # r_{i+1/2}
        main_diag[i] = 1.0 + r_left + r_right
        lower_diag[i - 1] = -r_left    # A[i, i-1]
        upper_diag[i] = -r_right        # A[i, i+1] (上对角: upper[i-1] 对应 A[i, i-1])

    # 修正: upper_diag[i] = A[i, i+1], 但数组索引需注意
    # lower_diag[j] = A[j+1, j], upper_diag[j] = A[j, j+1]
    lower_arr = np.zeros(nx - 1)
    upper_arr = np.zeros(nx - 1)
    for i in range(1, nx - 1):
        lower_arr[i - 1] = -r_half[i - 1]   # A[i, i-1]
        upper_arr[i - 1] = -r_half[i]        # A[i, i+1]

    # 构造完整矩阵 (用于验证, 实际求解用 Thomas)
    A = np.diag(main_diag) + np.diag(lower_arr, -1) + np.diag(upper_arr, 1)

    return A, lower_arr, main_diag, upper_arr


def thomas_solver(lower, main, upper, rhs):
    """Thomas 算法 (追赶法) 求解三对角系统.

    求解:
        main[0]·x[0] + upper[0]·x[1] = rhs[0]
        lower[i-1]·x[i-1] + main[i]·x[i] + upper[i]·x[i+1] = rhs[i]
        lower[n-2]·x[n-2] + main[n-1]·x[n-1] = rhs[n-1]

    参数
    ----
    lower : ndarray, shape (n-1,)
    main  : ndarray, shape (n,)
    upper : ndarray, shape (n-1,)
    rhs   : ndarray, shape (n,)

    返回
    ----
    x : ndarray, shape (n,)

    算法
    ====
    前代 (forward sweep):
        c'[0] = upper[0]/main[0],  d'[0] = rhs[0]/main[0]
        c'[i] = upper[i]/(main[i] - lower[i-1]·c'[i-1])
        d'[i] = (rhs[i] - lower[i-1]·d'[i-1])/(main[i] - lower[i-1]·c'[i-1])
    回代 (back substitution):
        x[n-1] = d'[n-1]
        x[i] = d'[i] - c'[i]·x[i+1]
    """
    n = len(main)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([rhs[0] / main[0]]) if abs(main[0]) > 1e-30 else np.array([0.0])

    # 前代
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    if abs(main[0]) < 1.0e-30:
        main[0] = 1.0e-30  # 正则化

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]

    for i in range(1, n):
        denom = main[i] - (lower[i - 1] * c_prime[i - 1] if i < n else 0.0)
        if abs(denom) < 1.0e-30:
            denom = 1.0e-30
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # 回代
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def solve_stochastic_heat_equation(kappa_field, params, source_func=None):
    """求解随机热传导方程的一个实现.

    参数
    ----
    kappa_field : ndarray, shape (nx,)
        随机扩散系数场
    params : UQProblemParameters
        问题参数
    source_func : callable(x, t) or None
        源项 f(x,t)

    返回
    ----
    u_history : ndarray, shape (nt, nx)
        各时刻的温度场
    """
    nx = params.nx
    nt = params.nt
    dx = params.dx
    dt = params.dt

    # 构造系统矩阵
    _, lower, main, upper = build_implicit_system_matrix(nx, dx, dt, kappa_field)

    # 初始条件
    u = np.full(nx, params.u_init)
    u[0] = params.u_left
    u[-1] = params.u_right

    u_history = np.zeros((nt, nx))
    u_history[0] = u.copy()

    # 时间步进
    for n_step in range(1, nt):
        t_new = params.t[n_step]

        # 右端项: rhs = u^n + dt * f
        rhs = u.copy()
        if source_func is not None:
            f_vals = np.array([source_func(params.x[i], t_new) for i in range(nx)])
            rhs += dt * f_vals

        # 边界条件 (Dirichlet)
        rhs[0] = params.u_left
        rhs[-1] = params.u_right

        # Thomas 求解
        u = thomas_solver(lower, main, upper, rhs)

        # 确保边界值精确
        u[0] = params.u_left
        u[-1] = params.u_right

        u_history[n_step] = u.copy()

    return u_history


def solve_stochastic_heat_batch(kappa_fields, params, source_func=None):
    """批量求解随机热传导方程 (多个 MC 实现).

    参数
    ----
    kappa_fields : ndarray, shape (n_mc, nx)
    params : UQProblemParameters
    source_func : callable or None

    返回
    ----
    solutions : ndarray, shape (n_mc, nt, nx)
    """
    n_mc = kappa_fields.shape[0]
    nt = params.nt
    nx = params.nx
    solutions = np.zeros((n_mc, nt, nx))

    for m in range(n_mc):
        solutions[m] = solve_stochastic_heat_equation(
            kappa_fields[m], params, source_func
        )

    return solutions
