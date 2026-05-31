"""
ode_evolver.py
自适应Runge-Kutta演化器与半经典运动方程

凝聚态物理应用：
在拓扑半金属中，电子的半经典运动方程包含Berry曲率修正：

    dr/dt = (1/hbar) * dE/dk - (dr/dt) x Omega(k)
    dk/dt = -(e/hbar) * (E + (dr/dt) x B)

其中Omega(k)为Berry曲率。为简化，本模块主要求解：

1. Berry相位沿路径的演化（参数方程）
2. 能带动力学方程（基于种子项目1030_rk12_adapt）
3. 周期性晶格中的输运方程（基于种子项目703_lorenz96_ode的周期性结构）

数值方法：
使用RK12自适应步长法（一阶Euler + 二阶改进Euler），
通过局部截断误差估计自适应调整步长。

步长控制策略：
    if error > tol * dt: dt = dt / 2
    elif error < tol * dt / 16: dt = dt * 2
    else: 接受当前步

基于种子项目1030_rk12_adapt和703_lorenz96_ode。
"""

import numpy as np
from typing import Callable, Tuple


def rk12_adaptive(yprime: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt: float = 0.01,
                  tol: float = 1e-6,
                  max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    自适应RK12 ODE求解器
    
    基于种子项目1030_rk12_adapt的核心算法。
    
    使用一阶Euler（预测）和二阶改进Euler（校正）：
        k1 = dt * f(t_n, y_n)
        y1 = y_n + k1                (Euler，一阶)
        k2 = dt * f(t_n + dt, y_n + k1)
        y2 = y_n + 0.5*k1 + 0.5*k2   (Heun，二阶)
    
    局部截断误差估计：e = ||y2 - y1||
    
    Parameters
    ----------
    yprime : callable
        导数函数 f(t, y)，返回dy/dt
    tspan : tuple
        (t0, t_final)
    y0 : np.ndarray
        初始条件
    dt : float
        初始步长
    tol : float
        容差，要求 error < tol * dt
    max_steps : int
        最大步数（防止无限循环）
    
    Returns
    -------
    t : np.ndarray, shape (n+1,)
    y : np.ndarray, shape (n+1, m)
    e : np.ndarray, shape (n+1,)
        每步的估计误差
    """
    y0 = np.asarray(y0, dtype=float)
    m = len(y0)
    
    t_list = [tspan[0]]
    y_list = [y0.copy()]
    e_list = [0.0]
    
    t_current = tspan[0]
    y_current = y0.copy()
    step_count = 0
    
    while t_current < tspan[1] and step_count < max_steps:
        step_count += 1
        
        # 确保不超过tspan[1]
        dt_actual = min(dt, tspan[1] - t_current)
        if dt_actual <= 0:
            break
        
        # RK12步
        k1 = dt_actual * yprime(t_current, y_current)
        y1 = y_current + k1
        
        k2 = dt_actual * yprime(t_current + dt_actual, y_current + k1)
        y2 = y_current + 0.5 * k1 + 0.5 * k2
        
        error = np.linalg.norm(y2 - y1)
        
        # 步长控制
        if error > tol * dt_actual and dt_actual > 1e-14:
            # 拒绝当前步，减小步长
            dt = dt_actual / 2.0
            continue
        elif error < tol * dt_actual / 16.0 and dt_actual > 1e-14:
            # 接受并增大步长
            dt = dt_actual * 2.0
        else:
            dt = dt_actual
        
        # 接受步
        t_current += dt_actual
        y_current = y2.copy()
        
        t_list.append(t_current)
        y_list.append(y_current.copy())
        e_list.append(error)
    
    t = np.array(t_list)
    y = np.array(y_list)
    e = np.array(e_list)
    
    return t, y, e


def semiclassical_eom(state: np.ndarray, t: float,
                      hamiltonian_func: Callable,
                      berry_curvature_func: Callable,
                      e_field: np.ndarray = None,
                      b_field: np.ndarray = None,
                      hbar: float = 1.0) -> np.ndarray:
    """
    拓扑半金属中的半经典运动方程
    
    状态矢量 state = [r_x, r_y, r_z, k_x, k_y, k_z]
    
    运动方程（Sundaram-Niu方程）：
        dr/dt = (1/hbar) * dE/dk - (dk/dt) x Omega(k)
        dk/dt = -(e/hbar) * E - (dk/dt) x (e/hbar) * B
    
    对于简化情况（无外磁场B=0）：
        dr/dt = (1/hbar) * dE/dk - (dk/dt) x Omega(k)
        dk/dt = -(e/hbar) * E
    
    Parameters
    ----------
    state : np.ndarray, shape (6,)
    t : float
    hamiltonian_func : callable
        输入k(3,)，输出(E, dE/dk(3,))
    berry_curvature_func : callable
        输入k(3,)，输出Omega(3,3)
    e_field : np.ndarray, shape (3,)
    b_field : np.ndarray, shape (3,)
    hbar : float
    
    Returns
    -------
    dstate_dt : np.ndarray, shape (6,)
    """
    if e_field is None:
        e_field = np.zeros(3)
    if b_field is None:
        b_field = np.zeros(3)
    
    r = state[:3]
    k = state[3:6]
    
    # 计算群速度和Berry曲率
    e_val, de_dk = hamiltonian_func(k)
    omega = berry_curvature_func(k)
    
    # 将Berry曲率张量转换为矢量
    omega_vec = np.array([omega[1, 2], omega[2, 0], omega[0, 1]])
    
    # dk/dt = -(e/hbar) * E （简化，假设无磁场耦合）
    dk_dt = -e_field / hbar
    
    # dr/dt = (1/hbar) * dE/dk - dk/dt x Omega
    # 使用Levi-Civita符号：
    # (a x b)_i = epsilon_{ijk} * a_j * b_k
    cross = np.cross(dk_dt, omega_vec)
    dr_dt = de_dk / hbar - cross
    
    dstate_dt = np.concatenate([dr_dt, dk_dt])
    return dstate_dt


def periodic_lattice_dynamics(y: np.ndarray, force: float = 8.0) -> np.ndarray:
    """
    周期性晶格中的等效动力学模型
    
    基于种子项目703_lorenz96_ode的周期性结构思想，
    模拟周期性边界条件下的等效运动。
    
    方程：
        dy_i/dt = (y_{i+1} - y_{i-2}) * y_{i-1} - y_i + force
    
    在凝聚态物理中，这可以视为周期性晶格中电子密度的
    非线性输运模型（类似Kuramoto-Sivashinsky方程的离散形式）。
    
    Parameters
    ----------
    y : np.ndarray
    force : float
    
    Returns
    -------
    dydt : np.ndarray
    """
    n = len(y)
    if n < 4:
        raise ValueError("至少需要4个变量")
    
    dydt = np.zeros(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dydt[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + force
    
    return dydt


def evolve_berry_phase_along_path(ham: object,
                                   path_func: Callable[[float], np.ndarray],
                                   tspan: Tuple[float, float],
                                   band_index: int = 0,
                                   n_points: int = 200) -> float:
    """
    计算Berry相位沿参数化路径的演化
    
    将路径参数化为 k = path_func(t)，t in [0, 1]。
    
    Berry相位：
        gamma = i \int_0^1 <u(k(t)) | d/dt | u(k(t))> dt
              = i \int_0^1 <u(k) | \nabla_k u(k)> · (dk/dt) dt
    
    Parameters
    ----------
    ham : WeylHamiltonian对象
    path_func : callable
        输入t，输出k(3,)
    tspan : tuple
        参数区间
    band_index : int
    n_points : int
    
    Returns
    -------
    phase : float
    """
    from berry_curvature import berry_connection_numeric
    
    t_vals = np.linspace(tspan[0], tspan[1], n_points)
    dt = (tspan[1] - tspan[0]) / (n_points - 1)
    
    phase = 0.0
    for i in range(n_points - 1):
        t_mid = 0.5 * (t_vals[i] + t_vals[i + 1])
        k_mid = path_func(t_mid)
        
        # Berry联络
        A = berry_connection_numeric(ham, k_mid, band_index)
        
        # dk/dt
        k1 = path_func(t_vals[i])
        k2 = path_func(t_vals[i + 1])
        dk_dt = (k2 - k1) / dt
        
        # 积分项
        phase += np.dot(A, dk_dt) * dt
    
    return phase
