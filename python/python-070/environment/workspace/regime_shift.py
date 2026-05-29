"""
regime_shift.py
渔业生态系统相变与状态转换模块

整合算法：Allen-Cahn 相场方程（基于 allen_cahn_pde）

核心科学问题：
海洋生态系统（如鱼类种群分布）可能发生突然的"状态转换"（regime shift），
例如从"高生物量状态"到"低生物量状态"的突变。
Allen-Cahn 方程作为相场模型，可用于描述这种双稳态系统的时空演化。

数学模型：
1. Allen-Cahn 方程：
   ∂u/∂t = ν ∇²u - (1/ξ²) f(u)
   其中 f(u) = u(u² - 1) 为双稳态势函数的导数
   势函数：F(u) = (1/4)(u² - 1)²

2. 双稳态解释：
   u = +1 对应"高生物量态"（健康渔业）
   u = -1 对应"低生物量态"（衰退渔业）
   u = 0 为不稳定鞍点（临界阈值）

3. 界面动力学：
   相边界以速度 v ≈ (ξ/√2) ∇·n 运动
   界面厚度 ~ ξ

4. 渔业应用扩展（本模块创新）：
   在标准 Allen-Cahn 基础上引入环境强迫项：
   ∂u/∂t = ν ∇²u - (1/ξ²) u(u² - 1) + ε E(t) (1 - u²)
   其中 E(t) 为捕捞压力强迫，ε 为耦合强度
"""

import numpy as np
from utils import NumericalConfig


def allen_cahn_potential(u):
    """
    Allen-Cahn 双稳态势函数
    F(u) = (1/4) (u² - 1)²
    """
    return 0.25 * (u ** 2 - 1.0) ** 2


def allen_cahn_derivative(u, xi):
    """
    势函数的导数（反应项）
    f(u) = u(u² - 1)
    """
    return u * (u ** 2 - 1.0) / (xi ** 2)


def laplacian_1d(u, dx):
    """
    一维拉普拉斯算子的中心差分近似
    u_xx ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²

    Neumann 边界条件：
    左边界：u_xx[0] = u_xx[1]
    右边界：u_xx[-1] = u_xx[-2]
    """
    n = len(u)
    uxx = np.zeros(n, dtype=float)

    for i in range(1, n - 1):
        uxx[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx ** 2)

    # Neumann 边界
    if n > 1:
        uxx[0] = uxx[1]
        uxx[-1] = uxx[-2]

    return uxx


def allen_cahn_rhs(t, u, dx, nu, xi, forcing_func=None):
    """
    计算 Allen-Cahn 方程的右端项

    方程：du/dt = ν u_xx - u(u²-1)/(2ξ²) + forcing

    Parameters
    ----------
    t : float
        当前时间
    u : ndarray
        状态向量
    dx : float
        空间步长
    nu : float
        扩散系数
    xi : float
        界面厚度参数
    forcing_func : callable, optional
        外部强迫函数 forcing_func(t, u)

    Returns
    -------
    dudt : ndarray
        时间导数
    """
    uxx = laplacian_1d(u, dx)
    dudt = nu * uxx - u * (u ** 2 - 1.0) / (2.0 * xi ** 2)

    if forcing_func is not None:
        dudt += forcing_func(t, u)

    # Neumann 边界保持
    if len(dudt) > 1:
        dudt[0] = dudt[1]
        dudt[-1] = dudt[-2]

    return dudt


def rk4_step(y, t, dt, rhs_func):
    """
    四阶 Runge-Kutta 单步推进

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt/2, y + k1/2)
    k3 = dt * f(t + dt/2, y + k2/2)
    k4 = dt * f(t + dt, y + k3)
    y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4) / 6
    """
    k1 = dt * rhs_func(t, y)
    k2 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * rhs_func(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * rhs_func(t + dt, y + k3)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_regime_shift(x_min, x_max, nx, u_initial, nu, xi,
                          T_total, dt, forcing_func=None,
                          save_interval=10):
    """
    模拟渔业生态系统的状态转换过程

    Parameters
    ----------
    x_min, x_max : float
        空间域
    nx : int
        空间网格数
    u_initial : ndarray
        初始状态（已映射到 [-1,1]）
    nu : float
        扩散系数
    xi : float
        界面厚度
    T_total : float
        总模拟时间
    dt : float
        时间步长
    forcing_func : callable, optional
        外部强迫
    save_interval : int
        每隔多少步保存一次

    Returns
    -------
    u_final : ndarray
        最终状态
    t_history : list
        时间历史
    u_history : list
        状态历史
    """
    dx = (x_max - x_min) / (nx - 1)
    x = np.linspace(x_min, x_max, nx)

    if u_initial is None:
        # 默认初始条件：界面位于中间
        u_initial = np.tanh((x - 0.5 * (x_min + x_max)) / (np.sqrt(2.0) * xi))

    u = u_initial.copy()
    n_steps = int(T_total / dt)

    t_history = [0.0]
    u_history = [u.copy()]

    def rhs(t, y):
        return allen_cahn_rhs(t, y, dx, nu, xi, forcing_func)

    for step in range(n_steps):
        t = step * dt
        u = rk4_step(u, t, dt, rhs)

        # 状态截断以保持数值稳定性
        u = np.clip(u, -1.5, 1.5)

        if (step + 1) % save_interval == 0:
            t_history.append((step + 1) * dt)
            u_history.append(u.copy())

    return u, t_history, u_history


def fishery_forcing(t, u, E_t, epsilon, q, K):
    """
    渔业捕捞压力强迫项

    强迫形式：ε E(t) (1 - u²) sign(u)
    当 u > 0（高生物量态）时，捕捞压力促使系统向衰退转变
    当 u < 0（低生物量态）时，禁渔促使恢复

    Parameters
    ----------
    t : float
        时间
    u : ndarray
        状态
    E_t : float
        捕捞努力量
    epsilon : float
        耦合强度
    q : float
        可捕系数
    K : float
        环境承载力

    Returns
    -------
    forcing : ndarray
        强迫向量
    """
    # 将捕捞压力归一化
    E_norm = min(E_t * q / 0.5, 2.0)  # 归一化到合理范围
    forcing = epsilon * E_norm * (1.0 - u ** 2) * np.sign(u)
    return forcing


def compute_regime_shift_time(u_history, threshold=0.0):
    """
    计算系统发生状态转换的临界时间

    当空间平均状态穿越 threshold 时定义为状态转换

    Parameters
    ----------
    u_history : list of ndarray
        状态历史
    threshold : float
        临界阈值

    Returns
    -------
    shift_time : int or None
        状态转换发生的步数索引，None 表示未发生
    """
    for i, u in enumerate(u_history):
        mean_u = np.mean(u)
        if i > 0:
            prev_mean = np.mean(u_history[i - 1])
            if prev_mean > threshold and mean_u <= threshold:
                return i
            if prev_mean < threshold and mean_u >= threshold:
                return i
    return None


def energy_functional(u, dx, nu, xi):
    """
    计算 Allen-Cahn 系统的总自由能泛函

    E[u] = ∫ [ (ν/2) |∇u|² + (1/(4ξ²)) (u² - 1)² ] dx

    在渔业模型中，自由能对应"生态系统稳定性度量"
    自由能越低，系统越稳定

    Parameters
    ----------
    u : ndarray
        状态
    dx : float
        空间步长
    nu : float
        扩散系数
    xi : float
        界面厚度

    Returns
    -------
    energy : float
        总自由能
    """
    # 梯度能
    grad_u = np.gradient(u, dx)
    grad_energy = 0.5 * nu * np.sum(grad_u ** 2) * dx

    # 势能
    pot_energy = np.sum(allen_cahn_potential(u) / (xi ** 2)) * dx

    return grad_energy + pot_energy
