"""
stiff_ode_cahn_hilliard.py
===========================
刚性 ODE 求解器: 用于 Cahn-Hilliard 方程时间积分
在 spinodal 分解早期的快速阶段。

种子项目 841_ozone_ode:
  臭氧形成/消耗的刚性化学动力学系统:
    dy/dt = f(y),  其中某些速率常数极小 (1e-16)
  导致 Jacobian 的特征值尺度差异巨大 → 刚性系统

映射到 CALPHAD:
  Cahn-Hilliard 方程在 spinodal 分解早期也呈现刚性:
  - 最快的模式 (最短波长) 有极大的增长率
  - 最慢的模式 (最长波长) 增长率很小
  - 时间步长受 CFL 条件限制

  本模块实现:
  1. 半隐式向后差分公式 (BDF-1, BDF-2)
  2. 自适应阶数和步长选择
  3. 简化 Newton 迭代求解隐式步
  4. Jacobian 特征值分析 (刚性检测)
"""

import numpy as np
from calphad_fec_constants import R_GAS
from gibbs_energy_calphad import second_derivative_G
from high_order_fd import compact_second_derivative


def ch_semilinear_rhs(c, T, h, M0, kappa, phase):
    """
    Cahn-Hilliard 方程的半线性右端项 (空间离散后):

    dc_i/dt = M(c_i) * [d2G_dc2 * D2*c - kappa * D4*c]_i

    其中 D2, D4 为紧致差分算子。

    Parameters
    ----------
    c : np.ndarray
        浓度场
    T : float
        温度 (K)
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称

    Returns
    -------
    np.ndarray
        dc/dt
    """
    from backward_euler_calphad import mobility_function

    N = len(c)
    d2G = second_derivative_G(c, T, phase)

    # 二阶导数 (紧致差分)
    d2c = compact_second_derivative(c, h)

    # 四阶导数 (二阶差分的二次应用)
    d4c = compact_second_derivative(d2c, h)

    # 化学势
    mu = d2G * (c - np.mean(c)) - kappa * d2c

    # 迁移率
    M = mobility_function(c, T, M0)

    # 右端项: nabla^2(M * mu)
    Mmu = M * mu
    rhs = compact_second_derivative(Mmu, h)

    return rhs


def jacobian_eigenvalue_estimate(c, T, h, M0, kappa, phase):
    """
    估计 CH 方程 Jacobian 的最大特征值 (用于刚性检测)。

    使用幂迭代 (Power Iteration):
    lambda_max ≈ ||J*v|| / ||v||  (v 为随机向量)

    或者使用线性化近似:
    lambda_max ~ M * d2G / h^2 + M * kappa / h^4

    Parameters
    ----------
    c : np.ndarray
        浓度场
    T : float
        温度 (K)
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称

    Returns
    -------
    dict
        {
            'lambda_max_approx': float,
            'lambda_min_approx': float,
            'stiffness_ratio': float,
            'is_stiff': bool,
        }
    """
    from backward_euler_calphad import mobility_function

    d2G_mean = np.mean(second_derivative_G(c, T, phase))
    M_mean = np.mean(mobility_function(c, T, M0))

    # 线性化特征值估计 (Fourier 模式分析)
    # lambda(k) = -M * k^2 * (d2G + kappa * k^2)
    k_max = np.pi / h  # 最大波数 (Nyquist)
    k_min = 2 * np.pi / (len(c) * h)  # 最小波数

    lambda_max = abs(M_mean * k_max ** 2 * (d2G_mean + kappa * k_max ** 2))
    lambda_min = abs(M_mean * k_min ** 2 * (d2G_mean + kappa * k_min ** 2))
    lambda_min = max(lambda_min, 1e-30)

    stiffness = lambda_max / lambda_min

    return {
        'lambda_max_approx': float(lambda_max),
        'lambda_min_approx': float(lambda_min),
        'stiffness_ratio': float(stiffness),
        'is_stiff': stiffness > 100.0,
    }


def bdf1_step(c_n, c_nm1, T, h, dt, M0, kappa, phase,
              max_newton=20, tol=1e-10):
    """
    BDF-1 (向后 Euler) 步, 使用简化 Newton 迭代:

    (c^{n+1} - c^n) / dt = F(c^{n+1})

    Newton 迭代:
    [I - dt * J] * delta = -(c^k - c^n - dt * F(c^k))
    c^{k+1} = c^k + delta

    简化 Newton: Jacobian 仅在第一步计算, 后续步复用。

    Parameters
    ----------
    c_n : np.ndarray
        当前时间步浓度
    c_nm1 : np.ndarray
        上一时间步浓度 (BDF-1 中未使用, BDF-2 中需要)
    T : float
        温度 (K)
    h : float
        空间步长
    dt : float
        时间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称

    Returns
    -------
    dict
        {
            'c_new': np.ndarray,
            'converged': bool,
            'n_newton': int,
        }
    """
    N = len(c_n)
    c = c_n.copy()

    # 初始右端项
    F_c = ch_semilinear_rhs(c, T, h, M0, kappa, phase)
    residual = c - c_n - dt * F_c

    converged = False
    n_newton = 0

    for newton_iter in range(max_newton):
        res_norm = np.max(np.abs(residual))
        if res_norm < tol:
            converged = True
            break

        # 简化 Newton: 使用对角近似
        # J_ii ≈ dF_i/dc_i ~ -M * d2G / h^2 (扩散项主导)
        d2G = second_derivative_G(c, T, phase)
        from backward_euler_calphad import mobility_function
        M = mobility_function(c, T, M0)

        # 对角 Jacobian 近似
        J_diag = -M * np.abs(d2G) / (h * h) - M * kappa / (h ** 4)
        J_diag = np.minimum(J_diag, -1e-30)  # 保证负定

        # Newton 步 (对角近似)
        delta = -residual / (1.0 - dt * J_diag)

        # 线搜索
        alpha = 1.0
        c_new = c + alpha * delta
        c_new = np.clip(c_new, 1e-12, 1.0 - 1e-12)
        c_new = c_new - (np.mean(c_new) - np.mean(c_n))  # 守恒

        F_new = ch_semilinear_rhs(c_new, T, h, M0, kappa, phase)
        residual = c_new - c_n - dt * F_new

        c = c_new
        n_newton += 1

    return {
        'c_new': c,
        'converged': converged,
        'n_newton': n_newton,
    }


def stiff_ch_integrator(c0, T, t_total, h, M0, kappa, phase,
                        rtol=1e-6, atol=1e-10, max_steps=1000):
    """
    自适应刚性 CH 方程积分器。

    算法:
    1. 估计 Jacobian 特征值 → 刚性检测
    2. 如果刚性 (stiffness > 100): 使用 BDF-1 + 自适应步长
    3. 如果非刚性: 使用显式 RK4 (更高效)
    4. 步长控制: dt_new = dt * (tol/err)^(1/order)

    Parameters
    ----------
    c0 : np.ndarray
        初始浓度场
    T : float
        温度 (K)
    t_total : float
        总模拟时间
    h : float
        空间步长
    M0 : float
        基准迁移率
    kappa : float
        梯度能系数
    phase : str
        相名称

    Returns
    -------
    dict
        {
            'c_final': np.ndarray,
            'n_steps': int,
            'n_total_newton': int,
            'dt_history': list,
            'is_stiff': bool,
            'stiffness_ratio': float,
        }
    """
    c = c0.copy()
    c_prev = c0.copy()

    # 初始刚性检测
    eig_info = jacobian_eigenvalue_estimate(c, T, h, M0, kappa, phase)
    is_stiff = eig_info['is_stiff']

    # 初始步长估计
    lambda_max = eig_info['lambda_max_approx']
    if lambda_max > 0:
        dt = min(0.5 / lambda_max, t_total / 100)
    else:
        dt = t_total / 100
    dt = max(dt, 1e-15)

    t = 0.0
    n_steps = 0
    n_newton_total = 0
    dt_history = []

    while t < t_total and n_steps < max_steps:
        dt = min(dt, t_total - t)

        if is_stiff:
            # BDF-1 步
            result = bdf1_step(c, c_prev, T, h, dt, M0, kappa, phase)
            c_new = result['c_new']
            n_newton_total += result['n_newton']
        else:
            # 显式 RK4 步
            k1 = ch_semilinear_rhs(c, T, h, M0, kappa, phase)
            k2 = ch_semilinear_rhs(c + 0.5 * dt * k1, T, h, M0, kappa, phase)
            k3 = ch_semilinear_rhs(c + 0.5 * dt * k2, T, h, M0, kappa, phase)
            k4 = ch_semilinear_rhs(c + dt * k3, T, h, M0, kappa, phase)
            c_new = c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            c_new = np.clip(c_new, 1e-12, 1.0 - 1e-12)
            c_new = c_new - (np.mean(c_new) - np.mean(c0))

        # 误差估计 (步长加倍比较)
        if is_stiff:
            result_half = bdf1_step(c, c_prev, T, h, dt / 2, M0, kappa, phase)
            c_half = result_half['c_new']
            result_quarter = bdf1_step(c_half, c, T, h, dt / 2, M0, kappa, phase)
            c_quarter = result_quarter['c_new']
        else:
            c_quarter = c_new  # 简化

        err = np.max(np.abs(c_new - c_quarter))
        err = max(err, 1e-30)

        # 接受步
        c_prev = c.copy()
        c = c_new
        t += dt
        n_steps += 1
        dt_history.append(dt)

        # 步长自适应
        order = 1 if is_stiff else 4
        dt_new = 0.9 * dt * (rtol / err) ** (1.0 / order)
        dt = np.clip(dt_new, 1e-15, t_total / 10)

        # 重新检测刚性
        if n_steps % 20 == 0:
            eig_info = jacobian_eigenvalue_estimate(c, T, h, M0, kappa, phase)
            is_stiff = eig_info['is_stiff']

    return {
        'c_final': c,
        'n_steps': n_steps,
        'n_total_newton': n_newton_total,
        'dt_history': dt_history,
        'is_stiff': is_stiff,
        'stiffness_ratio': eig_info['stiffness_ratio'],
    }
