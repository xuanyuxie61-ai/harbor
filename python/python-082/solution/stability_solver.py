"""
stability_solver.py
损伤演化时间积分格式的稳定性分析。
原项目映射：
  - 104_boundary_locus 的绝对稳定性区域分析方法
  - 通过计算 |R(z)| ≤ 1 的区域判断数值方法的稳定性
科学背景：
  损伤演化方程为 stiff ODE，需采用A-稳定或L-稳定的时间积分格式。
  对于线性测试方程 y' = λ y，单步法可写为 y_{n+1} = R(z) y_n，z = hλ。
  绝对稳定区域：S = {z ∈ C : |R(z)| ≤ 1}。
  对于显式Runge-Kutta方法（如我们的5级4阶RK）：
    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 + z^5/120 + ...
  稳定性函数为多项式，显式方法的稳定区域有限。
  对于损伤演化的 stiff 特性，需检查最大特征值是否落在稳定区域内。
"""

import numpy as np


def rk4_stability_function(z):
    """
    经典4阶Runge-Kutta的稳定性函数：
      R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    对于低存储5级4阶RK（如我们的方法），稳定性函数类似。
    """
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def low_storage_rk54_stability_function(z):
    """
    5级4阶低存储RK的近似稳定性函数。
    实际为5次多项式，但4阶精度。
    """
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0 + 0.02 * z ** 5


def evaluate_stability_region(R_func, x_range, y_range, npts=401):
    """
    在复平面上评估稳定性区域 |R(z)| ≤ 1。
    原项目映射：104_boundary_locus 的稳定性区域计算。
    返回：X, Y, |R(z)| 网格。
    """
    x = np.linspace(x_range[0], x_range[1], npts)
    y = np.linspace(y_range[0], y_range[1], npts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    Rval = R_func(Z)
    Rabs = np.abs(Rval)
    return X, Y, Rabs


def check_eigenvalue_in_stability_region(eigenvalues, dt, R_func):
    """
    检查给定特征值在步长dt下是否位于稳定区域内。
    eigenvalues: 系统Jacobian的特征值
    z = dt * λ
    返回：稳定标志数组。
    """
    z = dt * np.asarray(eigenvalues)
    R_vals = R_func(z)
    stable = np.abs(R_vals) <= 1.0 + 1e-10
    return stable


def compute_max_stable_timestep(eigenvalues, R_func):
    """
    计算最大稳定时间步长。
    对实负特征值，求使 |R(dt λ)| = 1 的最大dt。
    """
    ev = np.asarray(eigenvalues)
    real_neg = ev[np.real(ev) < 0]
    if len(real_neg) == 0:
        return np.inf

    # 二分搜索最大稳定dt
    lambda_max = np.max(np.real(real_neg))  # 最接近0的负特征值
    lambda_min = np.min(np.real(real_neg))  # 最负的特征值（stiffness决定）

    def is_stable(dt):
        z = dt * real_neg
        return np.all(np.abs(R_func(z)) <= 1.0 + 1e-10)

    dt_low = 0.0
    dt_high = 10.0 / abs(lambda_min)

    # 扩大搜索范围
    while is_stable(dt_high) and dt_high < 1e6:
        dt_high *= 2.0

    if is_stable(dt_high):
        return dt_high

    for _ in range(50):
        dt_mid = 0.5 * (dt_low + dt_high)
        if is_stable(dt_mid):
            dt_low = dt_mid
        else:
            dt_high = dt_mid

    return dt_low


def analyze_damage_jacobian_eigenvalues(damage_state, stress, params):
    """
    分析损伤演化ODE的Jacobian矩阵特征值。
    系统：y = [d_f, d_m, d_s, d_i]
    计算 ∂(dd/dN)/∂d 的特征值以评估stiffness。
    """
    if hasattr(damage_state, 'to_array'):
        d_arr = damage_state.to_array()
    else:
        d_arr = np.asarray(damage_state)
    d_f, d_m, d_s, d_i = d_arr
    sigma1, sigma2, tau12 = stress

    # 近似Jacobian（对角主导）
    J = np.zeros((4, 4))

    # 纤维损伤对自身导数
    if d_f < 0.99:
        J[0, 0] = params.k_f * ((abs(sigma1) / params.sigma_f0) ** params.m_f) / (
            (1.0 - d_f) ** (params.k_f + 1.0) + 1e-12)

    # 基体损伤对自身导数
    if d_m < 0.99:
        J[1, 1] = params.k_m * ((abs(sigma2) / params.sigma_m0) ** params.m_m) / (
            (1.0 - d_m) ** (params.k_m + 1.0) + 1e-12)

    # 剪切损伤对自身导数
    if d_s < 0.99:
        J[2, 2] = params.k_s * ((abs(tau12) / params.tau_s0) ** params.m_s) / (
            (1.0 - d_s) ** (params.k_s + 1.0) + 1e-12)

    # 界面损伤（heartbeat-like）：∂/∂d_i of -1/ε (d_i^3 - a d_i + ...)
    a_debond = 0.81
    epsilon = params.epsilon_debond
    J[3, 3] = -1.0 / epsilon * (3.0 * d_i ** 2 - a_debond)

    eigvals = np.linalg.eigvals(J)
    return eigvals


def recommend_time_integrator(damage_state, stress, params):
    """
    根据Jacobian特征值推荐时间积分器。
    返回：推荐方法名称、最大稳定步长、稳定性评估。
    """
    eigvals = analyze_damage_jacobian_eigenvalues(damage_state, stress, params)
    max_dt_rk4 = compute_max_stable_timestep(eigvals, rk4_stability_function)
    max_dt_rk54 = compute_max_stable_timestep(eigvals, low_storage_rk54_stability_function)

    ratio = np.max(np.abs(np.real(eigvals))) / (np.min(np.abs(np.real(eigvals))) + 1e-12)

    if ratio > 100.0:
        method = "Implicit BDF2 (stiff ratio > 100)"
    elif ratio > 10.0:
        method = "Implicit Midpoint or RK4 with very small dt"
    else:
        method = "Explicit RK4 or Low-Storage RK54"

    return {
        'method': method,
        'max_dt_rk4': max_dt_rk4,
        'max_dt_rk54': max_dt_rk54,
        'stiffness_ratio': ratio,
        'eigenvalues': eigvals
    }
