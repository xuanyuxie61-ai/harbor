"""
reion_imex_integrator.py
========================
隐式-显式 (IMEX) Runge-Kutta 时间积分器

本模块实现用于再电离演化方程的 IMEX 时间积分格式. 电离分数演化方程具有
刚性结构: 复合项 (alpha * n * xHII) 变化快而刚性, 光电离/扩散项变化缓
而光滑. IMEX 方法将刚性部分隐式处理, 非刚性部分显式处理, 从而避免极小
时间步长的限制.

实现三种 IMEX 方案:
  1. SSP2-ImEx(3,3,2) : 2阶 L-稳定 type I IMEX-RK (Pareschi & Russo 2005)
  2. SSP3-ImEx(3,4,3) : 3阶 L-稳定 type I IMEX-RK
  3. ARK4(3)6L[2]SA   : 4阶 IMEX-RK (Kennedy & Carpenter 2003)

IMEX 通用公式:
    Y_i = x^n + dt * sum_j a^ex_{ij} f_ex(Y_j) + dt * sum_j a^im_{ij} f_im(Y_j)
    x^{n+1} = x^n + dt * sum_i b^ex_i f_ex(Y_i) + dt * sum_i b^im_i f_im(Y_i)

对再电离方程:
    f_ex = 光电离 + 辐射传输 + Hubble 膨胀
    f_im = 复合 (刚性) + 扩散 (可能刚性)

对应种子项目:
  - 1088_ketch_2025_NLSH_reproducibility (IMEX 方案 → 再电离 IMEX)
"""

import numpy as np


def get_imex_scheme(name):
    """获取指定 IMEX 方案的 Butcher 表.

    Returns
    -------
    A_im, A_ex : array [s x s]
        隐式/显式 Runge-Kutta 矩阵
    b_im, b_ex : array [s]
        权重向量
    c_im, c_ex : array [s]
        节点向量
    s : int
        级数
    """
    if name == "SSP2_ImEx":
        # SSP2-IMEX(3,3,2): 2阶 L-稳定 type I
        l = 1.0
        A_im = np.array([
            [l / 4.0, 0.0, 0.0],
            [0.0, l / 4.0, 0.0],
            [l / 3.0, l / 3.0, l / 3.0],
        ])
        b_im = np.array([l / 3.0, l / 3.0, l / 3.0])
        c_im = np.array([l / 4.0, l / 4.0, l])

        A_ex = np.array([
            [0.0, 0.0, 0.0],
            [l / 2.0, 0.0, 0.0],
            [l / 2.0, l / 2.0, 0.0],
        ])
        b_ex = np.array([l / 3.0, l / 3.0, l / 3.0])
        c_ex = np.array([0.0, l / 2.0, l])
        return A_im, A_ex, b_im, b_ex, c_im, c_ex, 3

    elif name == "SSP3_ImEx":
        # SSP3-IMEX(3,4,3): 3阶 L-稳定 type I
        l = 1.0
        a = 0.24169426078821
        b = 0.06042356519705
        eta = 0.12915286960590
        A_im = np.array([
            [a, 0.0, 0.0, 0.0],
            [-a, a, 0.0, 0.0],
            [0.0, l - a, a, 0.0],
            [b, eta, l / 2.0 - b - eta - a, a],
        ])
        b_im = np.array([0.0, l / 6.0, l / 6.0, 2.0 * l / 3.0])
        c_im = np.array([a, 0.0, l, l / 2.0])

        A_ex = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, l, 0.0, 0.0],
            [0.0, l / 4.0, l / 4.0, 0.0],
        ])
        b_ex = np.array([0.0, l / 6.0, l / 6.0, 2.0 * l / 3.0])
        c_ex = np.array([0.0, 0.0, l, l / 2.0])
        return A_im, A_ex, b_im, b_ex, c_im, c_ex, 4

    elif name == "ARK4_3_6":
        # ARK4(3)6L[2]SA: 4阶 IMEX (Kennedy & Carpenter 2003)
        A_ex = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0],
            [-116923316275.0 / 2393684061468.0,
             -2731218467317.0 / 15368042101831.0,
             9408046702089.0 / 11113171139209.0, 0.0, 0.0, 0.0],
            [-451086348788.0 / 2902428689909.0,
             -2682348792572.0 / 7519795681897.0,
             12662868775082.0 / 11960479115383.0,
             3355817975965.0 / 11060851509271.0, 0.0, 0.0],
            [647845179188.0 / 3216320057751.0,
             73281519250.0 / 8382639484533.0,
             552539513391.0 / 3454668386233.0,
             3354512671639.0 / 8306763924573.0,
             4040.0 / 17871.0, 0.0],
        ])
        A_im = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
            [8611.0 / 62500.0, -1743.0 / 31250.0, 1.0 / 4.0, 0.0, 0.0, 0.0],
            [5012029.0 / 34652500.0, -654441.0 / 2922500.0,
             174375.0 / 388108.0, 1.0 / 4.0, 0.0, 0.0],
            [15267082809.0 / 155376265600.0, -71443401.0 / 120774400.0,
             730878875.0 / 902184768.0, 2285395.0 / 8070912.0, 1.0 / 4.0, 0.0],
            [82889.0 / 524892.0, 0.0, 15625.0 / 83664.0,
             69875.0 / 102672.0, -2260.0 / 8211.0, 1.0 / 4.0],
        ])
        c = np.array([0.0, 1.0 / 2.0, 83.0 / 250.0,
                      31.0 / 50.0, 17.0 / 20.0, 1.0])
        b = np.array([82889.0 / 524892.0, 0.0, 15625.0 / 83664.0,
                      69875.0 / 102672.0, -2260.0 / 8211.0, 1.0 / 4.0])
        bhat = np.array([4586570599.0 / 29645900160.0, 0.0,
                         178811875.0 / 945068544.0,
                         814220225.0 / 1159782912.0,
                         -3700637.0 / 11593932.0, 61727.0 / 225920.0])
        return A_im, A_ex, b, bhat, c, c, 6

    else:
        raise ValueError("未知 IMEX 方案: %s" % name)


def imex_step(x, dt, f_explicit, f_implicit_solve, A_im, A_ex,
              b_im, b_ex, c_im, c_ex, s):
    """执行单步 IMEX Runge-Kutta 积分.

    Parameters
    ----------
    x : array [N]
        当前状态 (电离分数分布)
    dt : float
        时间步长 [s]
    f_explicit : callable
        显式右端函数 f_ex(x) → array [N]
    f_implicit_solve : callable
        隐式求解器 f_im_solve(x_guess, rhs) → x_new
        满足 x_new = x_guess + dt * a_ii * f_im(x_new) + rhs
    A_im, A_ex : array [s x s]
    b_im, b_ex : array [s]
    c_im, c_ex : array [s]
    s : int

    Returns
    -------
    x_new : array [N]
    """
    N = len(x)
    # 存储各级值
    Y_ex = [np.zeros(N) for _ in range(s)]
    Y_im = [np.zeros(N) for _ in range(s)]

    for i in range(s):
        # 显式累加
        sum_ex = np.zeros(N)
        for j in range(i):
            f_ex_j = f_explicit(Y_ex[j] if j > 0 else x)
            sum_ex += A_ex[i, j] * f_ex_j

        # 隐式累加 (不含对角项)
        sum_im = np.zeros(N)
        for j in range(i):
            sum_im += A_im[i, j] * Y_im[j]

        # 级值
        Y_ex[i] = x + dt * sum_ex + dt * sum_im

        # 隐式求解对角项
        if A_im[i, i] != 0:
            rhs_i = Y_ex[i]
            Y_im[i] = f_implicit_solve(rhs_i, dt * A_im[i, i])
        else:
            Y_im[i] = Y_ex[i]

    # 最终更新
    sum_b_ex = np.zeros(N)
    sum_b_im = np.zeros(N)
    for i in range(s):
        f_ex_i = f_explicit(Y_ex[i])
        sum_b_ex += b_ex[i] * f_ex_i
        sum_b_im += b_im[i] * Y_im[i]

    x_new = x + dt * sum_b_ex + dt * sum_b_im
    return x_new


def simple_imex_step(x, dt, f_ex_val, D2_matrix, gamma_diff):
    """简化 IMEX Euler 步进 (用于快速测试).

    显式: f_ex (光电离)
    隐式: gamma_diff * D2 @ x_new (扩散)

    方程:
        (I - dt * gamma * D2) x_new = x_old + dt * f_ex

    Parameters
    ----------
    x : array [N]
    dt : float
    f_ex_val : array [N]
    D2_matrix : array [N x N]
    gamma_diff : float

    Returns
    -------
    x_new : array [N]
    """
    N = len(x)
    A = np.eye(N) - dt * gamma_diff * D2_matrix
    rhs = x + dt * f_ex_val
    # Thomas 算法 (三对角)
    x_new = thomas_periodic(A, rhs)
    return x_new


def thomas_periodic(A, rhs):
    """求解周期三对角系统 A x = rhs (Sherman-Morrison).

    A 的形式:
        [a0 b0  0  ... c0]
        [c1 a1 b1 ...  0]
        [ 0 c2 a2 ...  0]
        ...
        [bN-1 0 ... cN-1 aN-1]

    Parameters
    ----------
    A : array [N x N]
        周期三对角矩阵
    rhs : array [N]

    Returns
    -------
    x : array [N]
    """
    N = len(rhs)
    if N < 3:
        return np.linalg.solve(A, rhs)

    # 提取对角线
    a = np.array([A[i, i] for i in range(N)])
    b = np.array([A[i, (i + 1) % N] for i in range(N)])  # 上对角
    c = np.array([A[i, (i - 1) % N] for i in range(N)])  # 下对角

    # Sherman-Morrison 处理周期项
    gamma = -a[0]
    b[0] = 0.0
    c[N - 1] = 0.0
    a[0] = a[0] - gamma
    a[N - 1] = a[N - 1] - c[0] * b[N - 1] / gamma

    # Thomas 前代
    c_prime = np.zeros(N)
    d_prime = np.zeros(N)
    c_prime[0] = b[0] / a[0]
    d_prime[0] = rhs[0] / a[0]
    for i in range(1, N):
        m = a[i] - c[i] * c_prime[i - 1]
        if abs(m) < 1.0e-30:
            m = 1.0e-30
        c_prime[i] = b[i] / m if i < N - 1 else 0.0
        d_prime[i] = (rhs[i] - c[i] * d_prime[i - 1]) / m

    # 回代
    x = np.zeros(N)
    x[N - 1] = d_prime[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    # Sherman-Morrison 修正
    u = np.zeros(N)
    u[0] = gamma
    u[N - 1] = c[0]
    v = np.zeros(N)
    v[0] = 1.0
    v[N - 1] = b[N - 1] / gamma

    # 解 A y = u
    c_y = np.zeros(N)
    d_y = np.zeros(N)
    c_y[0] = b[0] / a[0]
    d_y[0] = u[0] / a[0]
    for i in range(1, N):
        m = a[i] - c[i] * c_y[i - 1]
        if abs(m) < 1.0e-30:
            m = 1.0e-30
        c_y[i] = b[i] / m if i < N - 1 else 0.0
        d_y[i] = (u[i] - c[i] * d_y[i - 1]) / m
    y = np.zeros(N)
    y[N - 1] = d_y[N - 1]
    for i in range(N - 2, -1, -1):
        y[i] = d_y[i] - c_y[i] * y[i + 1]

    factor = np.dot(v, x) / (1.0 + np.dot(v, y))
    return x - factor * y


def evolve_imex_full(x0, t_arr, f_explicit_func, implicit_solver,
                     scheme_name="SSP2_ImEx"):
    """完整 IMEX 时间积分 (从初始条件到终态).

    Parameters
    ----------
    x0 : array [N]
        初始条件
    t_arr : array [N_steps + 1]
        时间数组 (均匀或非均匀)
    f_explicit_func : callable
        f_ex(t, x) → array [N]
    implicit_solver : callable
        solve(x_guess, dt_a_ii, t) → x_new
    scheme_name : str

    Returns
    -------
    x_history : array [N_steps+1, N]
    """
    A_im, A_ex, b_im, b_ex, c_im, c_ex, s = get_imex_scheme(scheme_name)
    N = len(x0)
    N_steps = len(t_arr) - 1
    x_history = np.zeros((N_steps + 1, N))
    x_history[0] = x0.copy()
    x = x0.copy()

    for n in range(N_steps):
        dt = t_arr[n + 1] - t_arr[n]
        t_n = t_arr[n]

        def f_ex_cur(x_test):
            return f_explicit_func(t_n, x_test)

        def f_im_solve(rhs, dt_a):
            return implicit_solver(rhs, dt_a, t_n)

        x = imex_step(x, dt, f_ex_cur, f_im_solve,
                      A_im, A_ex, b_im, b_ex, c_im, c_ex, s)
        x_history[n + 1] = x.copy()

    return x_history
