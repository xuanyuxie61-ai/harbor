"""
Langevin 随机微分方程数值积分
===============================
融合原始项目:
  - 1041_robertson_ode: 刚性 ODE 数值积分

科学背景:
---------
核裂变的集体运动由多维 Langevin 方程描述:
  dq_i/dt = (1/m_ij) p_j
  dp_j/dt = -∂V/∂q_j - (1/2) Σ_{k,l} (∂m_{kl}/∂q_j) p_k p_l / m²
            - Σ_i γ_{ij} ẋ_i + ξ_j(t)

其中随机力满足:
  ⟨ξ_j(t)⟩ = 0
  ⟨ξ_i(t) ξ_j(t')⟩ = 2 γ_{ij} T δ(t-t')

在过阻尼极限下 (大粘滞)，动量自由度绝热消去:
  γ_{ij} dq_j/dt = -∂V/∂q_i + ξ_i(t)
  =>  dq_i/dt = -μ_{ij} ∂V/∂q_j + g_{ij} Γ_j(t)

其中 μ = γ^{-1} 为迁移率张量，Γ_j 为单位白噪声。

本模块实现基于 Euler-Maruyama 方法的 Langevin 动力学模拟，
用于生成裂变碎片质量分布的系综样本。

对于质量不对称度坐标 β₃，演化方程为:
  dβ₃/dt = -μ₃₃ ∂V/∂β₃ + √(2 D₃₃) · η(t)
其中 η(t) 为标准高斯白噪声，D₃₃ = μ₃₃ T。
"""

import numpy as np
from typing import Tuple, Callable, Optional


def euler_maruyama_step(
    x: np.ndarray,
    drift: np.ndarray,
    diffusion_sqrt: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Euler-Maruyama 单步积分.
    
    x_{n+1} = x_n + drift(x_n) dt + diffusion_sqrt(x_n) · dW
    其中 dW ~ N(0, dt).
    """
    ndim = len(x)
    dW = np.random.normal(0.0, np.sqrt(dt), size=ndim)
    x_new = x + drift * dt + diffusion_sqrt @ dW
    return x_new


def langevin_dynamics_1d(
    V_func: Callable[[float], float],
    gamma: float,
    T: float,
    x0: float,
    t_max: float,
    dt: float,
    x_bounds: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    一维过阻尼 Langevin 方程数值模拟.
    
    dx/dt = - (1/γ) dV/dx + √(2T/γ) η(t)
    
    参数:
        V_func: 势能函数 V(x)
        gamma: 粘滞系数
        T: 温度
        x0: 初始位置
        t_max: 最大模拟时间
        dt: 时间步长
        x_bounds: 位置边界（反射边界）
    返回:
        (时间数组, 轨迹数组)
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if T < 0:
        T = 0.0
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    n_steps = int(t_max / dt) + 1
    t_arr = np.linspace(0.0, t_max, n_steps)
    x_arr = np.zeros(n_steps)
    x_arr[0] = x0
    
    h = 1e-5  # 数值微分步长
    mu = 1.0 / gamma
    diff_coeff = np.sqrt(2.0 * T * mu)
    
    x_min, x_max = x_bounds
    
    for n in range(n_steps - 1):
        xn = x_arr[n]
        # 数值计算 -dV/dx
        dVdx = (V_func(xn + h) - V_func(xn - h)) / (2.0 * h)
        drift = -mu * dVdx
        # Euler-Maruyama
        dW = np.random.normal(0.0, np.sqrt(dt))
        x_new = xn + drift * dt + diff_coeff * dW
        
        # 反射边界条件
        if x_new < x_min:
            x_new = 2.0 * x_min - x_new
        elif x_new > x_max:
            x_new = 2.0 * x_max - x_new
        
        if not np.isfinite(x_new):
            x_new = xn
        
        x_arr[n + 1] = x_new
    
    return t_arr, x_arr


def langevin_ensemble_mass_distribution(
    V_func: Callable[[float], float],
    gamma: float,
    T: float,
    x0: float,
    t_max: float,
    dt: float,
    n_trajectories: int,
    mass_number: int,
    n_bins: int = 80,
    x_bounds: Tuple[float, float] = (-1.5, 1.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过 Langevin 系综模拟计算碎片质量分布.
    
    每个轨迹在 t_max 时刻记录 β₃ 值，转换为碎片质量，
    统计得到质量分布直方图 Y(A).
    
    参数:
        V_func: β₃ 方向的势能函数
        gamma: 粘滞系数
        T: 核温度
        x0: 初始 β₃
        t_max: 演化时间
        dt: 时间步长
        n_trajectories: 轨迹数
        mass_number: 母核质量数
        n_bins: 质量直方图箱数
    返回:
        (mass_centers, yield_distribution)
    """
    from collective_coordinates import mass_asymmetry_to_fragment_mass
    
    final_masses = np.zeros(n_trajectories)
    
    for i in range(n_trajectories):
        t_arr, x_arr = langevin_dynamics_1d(
            V_func, gamma, T, x0, t_max, dt, x_bounds
        )
        # 取最后时刻的 β₃
        beta3_final = x_arr[-1]
        A_L, A_H = mass_asymmetry_to_fragment_mass(beta3_final, mass_number)
        # 记录轻碎片质量（对称性：随机选择轻/重）
        if np.random.rand() < 0.5:
            final_masses[i] = A_L
        else:
            final_masses[i] = A_H
    
    # TODO(Hole_3): 将 Langevin 轨迹终态的碎片质量构建为归一化直方图。
    # 要求：
    #  1. 直方图 bin 范围必须覆盖 collective_coordinates.mass_asymmetry_to_fragment_mass
    #     所返回的有效质量范围（注意该函数对 A_L, A_H 有 clip 到 [1, A-1] 的保护）。
    #  2. 归一化应使得 ∫ Y(A) dA = 1（即 counts / (n_trajectories * bin_width)）。
    #  3. 若 mass_asymmetry_to_fragment_mass 的公式或边界保护发生变化，
    #     此处的 A_min / A_max 和 bin 数可能需要同步调整。
    raise NotImplementedError("Hole_3: Langevin 质量分布直方图构建待修复")
    counts = np.zeros(n_bins)  # 占位
    mass_centers = np.zeros(n_bins)  # 占位
    return mass_centers, counts


def robertson_like_stiff_test(t: float, y: np.ndarray) -> np.ndarray:
    """
    Robertson 型刚性 ODE 系统测试函数 (改编自 robertson_deriv.m).
    
    在核裂变中，中子发射、γ 退激与裂变道之间存在竞争，
    可用类似的刚性 ODE 描述:
    dy1/dt = -λ₁ y1 + λ₂ y2 y3
    dy2/dt =  λ₁ y1 - λ₂ y2 y3 - λ₃ y2²
    dy3/dt =                       λ₃ y2²
    其中 y1, y2, y3 分别代表复合核、过渡态、裂变碎片的比例。
    """
    y1, y2, y3 = y[0], y[1], y[2]
    lam1 = 0.04
    lam2 = 1e4
    lam3 = 3e7
    dydt = np.zeros(3)
    dydt[0] = -lam1 * y1 + lam2 * y2 * y3
    dydt[1] = lam1 * y1 - lam2 * y2 * y3 - lam3 * y2 * y2
    dydt[2] = lam3 * y2 * y2
    return dydt


def rk4_step(f: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    经典四阶 Runge-Kutta 单步.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def fission_decay_dynamics(
    lambda_fission: float,
    lambda_neutron: float,
    lambda_gamma: float,
    t_max: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    裂变竞争过程动力学: 复合核 -> [裂变, 中子蒸发, γ退激].
    
    dN_c/dt = -(λ_f + λ_n + λ_γ) N_c
    dN_f/dt = λ_f N_c
    dN_n/dt = λ_n N_c
    dN_γ/dt = λ_γ N_c
    
    返回各道概率随时间的演化.
    """
    n_steps = int(t_max / dt) + 1
    t_arr = np.linspace(0.0, t_max, n_steps)
    N_c = np.zeros(n_steps)
    N_f = np.zeros(n_steps)
    N_n = np.zeros(n_steps)
    N_g = np.zeros(n_steps)
    
    N_c[0] = 1.0
    lam_total = lambda_fission + lambda_neutron + lambda_gamma
    
    for i in range(n_steps - 1):
        decay = lam_total * N_c[i]
        N_c[i + 1] = N_c[i] - decay * dt
        N_f[i + 1] = N_f[i] + lambda_fission * N_c[i] * dt
        N_n[i + 1] = N_n[i] + lambda_neutron * N_c[i] * dt
        N_g[i + 1] = N_g[i] + lambda_gamma * N_c[i] * dt
    
    return t_arr, N_c, N_f, N_n, N_g
