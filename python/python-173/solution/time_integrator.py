"""
时间积分模块：处理 PDE 的时空演化与刚性 ODE 系统

融合自:
- 1173_string_pde: 波动方程的有限差分时间步进
- 1374_unstable_ode: 刚性/不稳定 ODE 系统的稳定性分析

时间离散化方法:
1. 向后 Euler (BDF1):
       (u^{n+1} - u^n) / Δt = F(u^{n+1})
   无条件稳定，L-稳定，一阶精度

2. BDF2 (二阶向后差分):
       (3u^{n+1} - 4u^n + u^{n-1}) / (2Δt) = F(u^{n+1})
   无条件稳定，二阶精度

3. Crank-Nicolson (梯形法则):
       (u^{n+1} - u^n) / Δt = 0.5 * (F(u^n) + F(u^{n+1}))
   无条件稳定，二阶精度，但非 L-稳定

稳定性分析:
   对于模型方程 u' = λu，向后 Euler 的放大因子:
       G = 1 / (1 - z),  z = λΔt
   当 Re(λ) < 0 时，|G| < 1 对所有 Δt > 0 成立。

刚性处理:
   当系统存在大特征值分离时 (stiffness ratio S = |λ_max|/|λ_min| >> 1)，
   显式方法受稳定性限制需要极小 Δt，隐式方法可规避此限制。
"""

import numpy as np


def backward_euler_step(u_n, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    """
    向后 Euler 时间步进。
    
    求解非线性方程:
        u^{n+1} - u^n - dt * F(u^{n+1}) = 0
    
    使用不动点迭代（简化 Newton）。
    
    Parameters
    ----------
    u_n : ndarray
        当前解
    dt : float
        时间步长
    rhs_func : callable
        rhs_func(u) -> ndarray，返回 F(u)
    newton_tol : float
    max_iter : int
    
    Returns
    -------
    u_new : ndarray
    converged : bool
    """
    u_new = u_n.copy()

    for _ in range(max_iter):
        F_val = rhs_func(u_new)
        residual = u_new - u_n - dt * F_val
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True

        # 简化的 Picard 迭代
        u_new = u_n + dt * F_val

    return u_new, False


def bdf2_step(u_n, u_nm1, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    """
    BDF2 时间步进。
    
    离散方程:
        (3u^{n+1} - 4u^n + u^{n-1}) / (2dt) = F(u^{n+1})
    
    等价于:
        u^{n+1} = (4u^n - u^{n-1} + 2dt * F(u^{n+1})) / 3
    
    Parameters
    ----------
    u_n : ndarray
        u^n
    u_nm1 : ndarray
        u^{n-1}
    dt : float
    rhs_func : callable
    newton_tol : float
    max_iter : int
    
    Returns
    -------
    u_new : ndarray
    converged : bool
    """
    u_new = u_n.copy()
    coeff = 2.0 * dt / 3.0

    for _ in range(max_iter):
        F_val = rhs_func(u_new)
        residual = u_new - (4.0 * u_n - u_nm1) / 3.0 - coeff * F_val
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True

        u_new = (4.0 * u_n - u_nm1) / 3.0 + coeff * F_val

    return u_new, False


def crank_nicolson_step(u_n, dt, rhs_func, newton_tol=1e-8, max_iter=20):
    """
    Crank-Nicolson 时间步进。
    
    离散方程:
        u^{n+1} = u^n + 0.5*dt*(F(u^n) + F(u^{n+1}))
    
    Parameters
    ----------
    u_n : ndarray
    dt : float
    rhs_func : callable
    newton_tol : float
    max_iter : int
    
    Returns
    -------
    u_new : ndarray
    converged : bool
    """
    F_n = rhs_func(u_n)
    u_new = u_n.copy()

    for _ in range(max_iter):
        F_new = rhs_func(u_new)
        residual = u_new - u_n - 0.5 * dt * (F_n + F_new)
        res_norm = np.linalg.norm(residual)

        if res_norm < newton_tol:
            return u_new, True

        # Picard 迭代
        u_new = u_n + 0.5 * dt * (F_n + F_new)

    return u_new, False


def adaptive_time_stepping(
    u0, t_span, dt_init,
    rhs_func,
    scheme='bdf2',
    rtol=1e-4,
    atol=1e-6,
    dt_min=1e-8,
    dt_max=1.0,
    safety_factor=0.9
):
    """
    自适应时间步进。
    
    通过比较一阶和二阶近似估计局部截断误差，动态调整时间步长:
        err = ||u^{n+1}_{(1)} - u^{n+1}_{(2)}||
        dt_new = dt * (tol / err)^{1/2} * safety_factor
    
    Parameters
    ----------
    u0 : ndarray
        初始条件
    t_span : tuple
        (t_start, t_end)
    dt_init : float
        初始时间步长
    rhs_func : callable
    scheme : str
    rtol, atol : float
        相对和绝对容差
    dt_min, dt_max : float
    safety_factor : float
    
    Returns
    -------
    t_history : list
    u_history : list
    dt_history : list
    """
    t_start, t_end = t_span
    t = t_start
    dt = dt_init
    u = u0.copy()

    t_history = [t]
    u_history = [u.copy()]
    dt_history = [dt]

    u_prev = None  # 用于 BDF2

    while t < t_end:
        dt = min(dt, t_end - t)
        if dt < dt_min:
            dt = dt_min
            if t_end - t < dt_min:
                break

        tol = atol + rtol * np.linalg.norm(u)

        # 使用向前 Euler 作为一阶估计
        u_euler = u + dt * rhs_func(u)

        # 使用向后 Euler 作为稳定估计
        u_impl, _ = backward_euler_step(u, dt, rhs_func)

        # 误差估计
        err = np.linalg.norm(u_euler - u_impl)

        if err < tol or dt <= dt_min:
            # 接受步
            if scheme == 'bdf1' or scheme == 'backward_euler':
                u = u_impl
            elif scheme == 'bdf2' and u_prev is not None:
                u_new, _ = bdf2_step(u, u_prev, dt, rhs_func)
                u_prev = u.copy()
                u = u_new
            elif scheme == 'crank_nicolson':
                u, _ = crank_nicolson_step(u, dt, rhs_func)
            else:
                # 默认或第一步入使用 BDF1
                u = u_impl
                u_prev = u.copy()

            t += dt
            t_history.append(t)
            u_history.append(u.copy())
            dt_history.append(dt)

            # 调整步长
            if err > 1e-14:
                dt_new = dt * safety_factor * np.sqrt(tol / err)
            else:
                dt_new = dt * 2.0
            dt = np.clip(dt_new, dt_min, dt_max)
        else:
            # 拒绝步，减小步长
            dt = max(dt * 0.5, dt_min)

    return t_history, u_history, dt_history


def analyze_stiffness_eigenvalues(A_matrix, M_lumped=None, n_eig=10):
    """
    分析离散系统的刚度比（特征值分布）。
    
    对应原 unstable_ode.m 中对刚性系统的分析思想。
    
    求解广义特征值问题:
        A φ = λ M φ
    
    刚度比:
        S = |Re(λ_max)| / |Re(λ_min)|
    
    Parameters
    ----------
    A_matrix : ndarray
        系统矩阵（如刚度矩阵）
    M_lumped : ndarray, optional
        Lumped 质量矩阵对角元
    n_eig : int
        计算的特征值数量
    
    Returns
    -------
    eigenvalues : ndarray
        特征值（按实部排序）
    stiffness_ratio : float
        刚度比
    spectral_radius : float
        谱半径
    """
    n = A_matrix.shape[0]
    n_eig = min(n_eig, n)

    if M_lumped is not None:
        M_inv_sqrt = 1.0 / np.sqrt(M_lumped)
        A_scaled = M_inv_sqrt[:, None] * A_matrix * M_inv_sqrt[None, :]
    else:
        A_scaled = A_matrix

    # 计算部分特征值
    try:
        eigenvalues = np.linalg.eigvals(A_scaled)
        eigenvalues = eigenvalues[np.argsort(np.real(eigenvalues))]

        real_parts = np.real(eigenvalues)
        abs_real = np.abs(real_parts)
        nonzero = abs_real > 1e-14
        if np.sum(nonzero) > 1:
            stiffness_ratio = np.max(abs_real[nonzero]) / np.min(abs_real[nonzero])
        else:
            stiffness_ratio = 1.0

        spectral_radius = np.max(np.abs(eigenvalues))

        return eigenvalues[:n_eig], stiffness_ratio, spectral_radius
    except Exception:
        # 如果特征值计算失败，返回保守估计
        return np.zeros(n_eig), 1.0, np.linalg.norm(A_scaled, 2)
