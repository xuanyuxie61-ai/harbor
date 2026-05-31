"""
Feynman-Kac 路径积分求解器
基于 feynman_kac_1d 核心算法：用随机游走（布朗运动）结合路径积分计算 PDE 解。

在蛋白质折叠中的应用：
- 将折叠过程视为粒子在自由能势阱中的扩散-逃逸过程
- 用 Feynman-Kac 公式计算从亚稳态逃逸的概率和平均首通时间 (MFPT)
- 验证自由能景观计算的数值精度

数学基础:
    Feynman-Kac 公式:
        u(x) = E[ exp( -∫_0^τ V(X_s) ds ) * g(X_τ) ]
    
    其中 X_t 为从 x 出发的布朗运动，τ 为首次离开区域 D 的时间，
    V 为势函数，g 为边界条件。
    
    对于蛋白质折叠，可设 V 为自由能景观 F(x)/(k_B T)，
    g 为在边界处的吸收条件（如已折叠/未折叠状态）。
    
    离散随机游走 (Euler-Maruyama):
        X_{t+h} = X_t + sqrt(2*D*h) * Z,  Z ~ N(0,1)
    
    指数积分 (梯形法则):
        Y_{t+h} ≈ Y_t * exp( -0.5*h*(V(X_t)+V(X_{t+h})) )
"""

import numpy as np
from typing import Callable, Tuple


def brownian_walk_1d(x0: float, D: float, dt: float, n_steps: int,
                     boundary_left: float, boundary_right: float) -> Tuple[np.ndarray, float]:
    """
    执行一维布朗运动随机游走，直到离开边界或达到最大步数。
    
    离散格式 (Euler-Maruyama，一阶强收敛):
        X_{k+1} = X_k + sqrt(2*D*dt) * Z_k,  Z_k ~ N(0,1)
    
    Parameters
    ----------
    x0 : float
        初始位置。
    D : float
        扩散系数。
    dt : float
        时间步长。
    n_steps : int
        最大步数。
    boundary_left, boundary_right : float
        吸收边界。
    
    Returns
    -------
    trajectory : np.ndarray
        路径点数组。
    exit_time : float
        首通时间 (若未离开则为 n_steps * dt)。
    """
    if boundary_left >= boundary_right:
        raise ValueError("boundary_left must be less than boundary_right")
    if x0 < boundary_left or x0 > boundary_right:
        raise ValueError("x0 must be within boundaries")
    
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = x0
    sigma = np.sqrt(2.0 * D * dt)
    
    for k in range(n_steps):
        trajectory[k + 1] = trajectory[k] + sigma * np.random.randn()
        if trajectory[k + 1] <= boundary_left or trajectory[k + 1] >= boundary_right:
            return trajectory[:k + 2], (k + 1) * dt
        # 软边界反射 (数值稳定性)
        if trajectory[k + 1] < boundary_left:
            trajectory[k + 1] = boundary_left + (boundary_left - trajectory[k + 1])
        if trajectory[k + 1] > boundary_right:
            trajectory[k + 1] = boundary_right - (trajectory[k + 1] - boundary_right)
    
    return trajectory, n_steps * dt


def feynman_kac_escape_probability(x0: float, potential: Callable[[float], float],
                                   D: float, dt: float, n_steps: int,
                                   boundary_left: float, boundary_right: float,
                                   n_trajectories: int = 10000) -> Tuple[float, float]:
    """
    用 Feynman-Kac 路径积分计算粒子从 x0 出发到达右边界 (folded) 的概率。
    
    物理意义:
        在自由能景观 F(x) 中，设势函数 V(x) = F(x) / (k_B T)。
        左边界为 unfolded 态，右边界为 folded 态。
        计算折叠概率 P_fold(x0)。
    
    Feynman-Kac 离散形式:
        P ≈ (1/N) * sum_{paths} [ exp( -sum_k V(X_k)*dt ) * I{X_τ = right} ]
    
    Parameters
    ----------
    x0 : float
        初始反应坐标。
    potential : callable
        势函数 V(x)。
    D : float
        扩散系数。
    dt : float
        时间步长。
    n_steps : int
        每轨迹最大步数。
    boundary_left, boundary_right : float
        边界。
    n_trajectories : int
        蒙特卡洛轨迹数。
    
    Returns
    -------
    prob : float
        折叠概率估计值。
    std_err : float
        标准误差。
    """
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    
    results = np.zeros(n_trajectories)
    for i in range(n_trajectories):
        traj, _ = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)
        # 判断最终位置更靠近哪边
        final_pos = traj[-1]
        # 若触及边界，则根据最终位置判断
        if final_pos >= boundary_right:
            weight = 1.0
        elif final_pos <= boundary_left:
            weight = 0.0
        else:
            # 未触及边界，按距离加权
            weight = (final_pos - boundary_left) / (boundary_right - boundary_left)
        
        # 路径权重: exp( -∫ V ds )
        path_integral = 0.0
        for k in range(len(traj) - 1):
            vk = potential(traj[k])
            vk1 = potential(traj[k + 1])
            path_integral += 0.5 * dt * (vk + vk1)
        
        results[i] = weight * np.exp(-path_integral)
    
    prob = float(np.mean(results))
    std_err = float(np.std(results) / np.sqrt(n_trajectories))
    return prob, std_err


def mean_first_passage_time_1d(x0: float, potential: Callable[[float], float],
                               D: float, dt: float, n_steps: int,
                               boundary_left: float, boundary_right: float,
                               n_trajectories: int = 5000) -> Tuple[float, float]:
    """
    计算从 x0 出发的平均首通时间 (Mean First Passage Time, MFPT)。
    
    在蛋白质折叠中，MFPT 与折叠速率常数 k_f 的关系:
        k_f ≈ 1 / MFPT
    
    本实现通过大量布朗运动轨迹的模拟估计 MFPT:
        MFPT ≈ (1/N) * sum_{paths} τ_i
    
    其中 τ_i 为第 i 条轨迹首次到达任一吸收边界的时间。
    
    Parameters
    ----------
    x0 : float
        初始位置。
    potential : callable
        势函数。
    D : float
        扩散系数。
    dt : float
        时间步长。
    n_steps : int
        最大步数。
    boundary_left, boundary_right : float
        吸收边界。
    n_trajectories : int
        轨迹数。
    
    Returns
    -------
    mfpt : float
        平均首通时间。
    std_err : float
        标准误差。
    """
    fpt_list = []
    for _ in range(n_trajectories):
        _, tau = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)
        fpt_list.append(tau)
    
    fpt_array = np.array(fpt_list)
    mfpt = float(np.mean(fpt_array))
    std_err = float(np.std(fpt_array) / np.sqrt(n_trajectories))
    return mfpt, std_err


def kramers_rate_approximation(barrier_height: float, kT: float,
                                D: float, curvature_top: float,
                                curvature_bottom: float) -> float:
    """
    Kramers 速率理论近似公式。
    
    对于双势阱系统，折叠速率:
        k ≈ (D / (2π k_B T)) * sqrt( |ω_b| * ω_m ) * exp( -ΔF / (k_B T) )
    
    其中:
        ΔF 为势垒高度
        ω_b 为势垒顶部曲率 (负值，取绝对值)
        ω_m 为势阱底部曲率
    
    简化形式 (Kramers, 1940):
        k_Kramers = (ω_m * |ω_b| / (2π)) * exp( -ΔF / kT )
    
    Parameters
    ----------
    barrier_height : float
        势垒高度 ΔF。
    kT : float
        热能量。
    D : float
        扩散系数。
    curvature_top : float
        势垒顶部曲率 (应为负值)。
    curvature_bottom : float
        势阱底部曲率 (应为正值)。
    
    Returns
    -------
    rate : float
        Kramers 速率估计。
    """
    if barrier_height <= 0:
        raise ValueError("barrier_height must be positive")
    if curvature_bottom <= 0:
        raise ValueError("curvature_bottom must be positive")
    if curvature_top >= 0:
        raise ValueError("curvature_top must be negative")
    
    omega_m = np.sqrt(curvature_bottom)
    omega_b = np.sqrt(abs(curvature_top))
    prefactor = (omega_m * omega_b) / (2.0 * np.pi)
    rate = prefactor * np.exp(-barrier_height / kT)
    return float(rate)


def path_integral_free_energy(x_samples: np.ndarray, potential: Callable[[float], float],
                              D: float, dt: float, n_steps: int,
                              boundary_left: float, boundary_right: float,
                              n_paths_per_x: int = 2000) -> np.ndarray:
    """
    沿反应坐标采样点用路径积分方法估算自由能。
    
    核心思想:
        设 p(x) 为稳态概率密度，则 F(x) = -kT ln p(x)。
        通过大量轨迹在区域上的覆盖时间比例估计 p(x)。
    
    Parameters
    ----------
    x_samples : np.ndarray
        采样点数组。
    potential : callable
        势函数。
    D : float
        扩散系数。
    dt : float
        时间步长。
    n_steps : int
        每轨迹步数。
    boundary_left, boundary_right : float
        边界。
    n_paths_per_x : int
        每采样点的轨迹数。
    
    Returns
    -------
    free_energy : np.ndarray
        估算的自由能值。
    """
    N = len(x_samples)
    histogram = np.zeros(N)
    bin_edges = np.linspace(boundary_left, boundary_right, N + 1)
    
    total_samples = 0
    for x0 in x_samples:
        for _ in range(n_paths_per_x):
            traj, _ = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)
            # 统计轨迹点落在各区间的频次
            counts, _ = np.histogram(traj, bins=bin_edges)
            histogram += counts
            total_samples += len(traj)
    
    # 转换为概率密度
    bin_widths = np.diff(bin_edges)
    prob = histogram / (total_samples * bin_widths + 1e-12)
    prob = np.maximum(prob, 1e-12)
    
    # F = -ln(p) (设 kT=1)
    free_energy = -np.log(prob)
    free_energy -= free_energy.min()
    return free_energy
