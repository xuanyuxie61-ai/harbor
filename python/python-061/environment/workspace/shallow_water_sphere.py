"""
球坐标系一维浅水方程数值求解模块
========================================
基于种子项目 1070_shallow_water_1d_movie 的浅水方程思想，
推广至球坐标系，用于描述台风尺度的大尺度环流背景场。

核心物理模型（球坐标一维浅水方程）：

    ∂h/∂t + (1/(R*sin(θ))) * ∂(h*u*sin(θ))/∂θ = 0            (质量守恒)
    
    ∂(h*u)/∂t + (1/(R*sin(θ))) * ∂(h*u²*sin(θ))/∂θ 
        = -(g*h/(R*sin(θ))) * ∂h/∂θ - f*h*v - C_d*|u|*u       (动量方程-经向)
    
    ∂(h*v)/∂t = f*h*u                                         (动量方程-纬向，地转近似)

其中：
    R  = 6.371e6 m       地球半径
    g  = 9.81 m/s²       重力加速度
    f  = 2Ω*sin(θ)       科里奥利参数，Ω = 7.292e-5 rad/s
    θ  = 极距（co-latitude），范围 [0, π]
    h  = 自由表面高度（m）
    u  = 经向速度（m/s，向南为正）
    v  = 纬向速度（m/s，向东为正）
    C_d = 1.2e-3          海面 drag 系数

数值离散采用守恒型有限体积 + 显式中点法（基于 766_midpoint_explicit）
时间推进格式：
    Y_{n+1/2} = Y_n + (Δt/2) * F(t_n, Y_n)
    Y_{n+1}   = Y_n + Δt * F(t_{n+1/2}, Y_{n+1/2})

稳定性条件（CFL，基于 104_boundary_locus 思想）：
    Δt ≤ CFL * min(Δx / (|u| + √(g*h)))
"""

import numpy as np

# 物理常数
EARTH_RADIUS = 6.371e6      # m
GRAVITY = 9.81              # m/s^2
OMEGA = 7.2921159e-5        # rad/s，地球自转角速度
DRAG_COEFF = 1.2e-3         # 海面 drag 系数


def coriolis_parameter(theta):
    """
    科里奥利参数 f = 2*Ω*sin(θ)。
    
    参数:
        theta: 极距（co-latitude），numpy数组，范围 [0, π]
    
    返回:
        f: 科里奥利参数，单位 rad/s
    """
    return 2.0 * OMEGA * np.sin(theta)


def compute_cfl_condition(theta, h, u, dt_max_factor=0.8):
    """
    基于边界轨迹（boundary locus）思想计算CFL稳定性条件。
    
    浅水方程的CFL条件来源于特征速度 λ = u ± √(g*h)。
    在球坐标下，网格间距 Δs = R*Δθ，因此：
    
        Δt ≤ CFL * min( R*Δθ / (|u| + √(g*h)) )
    
    参数:
        theta: 极距网格点，numpy数组
        h: 自由表面高度
        u: 经向速度
        dt_max_factor: CFL数，默认0.8
    
    返回:
        dt_max: 最大允许时间步长
    """
    dtheta = np.diff(theta)
    if len(dtheta) == 0:
        return 1.0
    dx = EARTH_RADIUS * np.min(dtheta)
    wave_speed = np.abs(u) + np.sqrt(GRAVITY * np.maximum(h, 0.1))
    max_speed = np.max(wave_speed)
    max_speed = max(max_speed, 1e-6)
    dt_max = dt_max_factor * dx / max_speed
    return dt_max


def shallow_water_rhs(theta, h, hu, hv):
    """
    计算球坐标浅水方程的右端项 F(t, Y)。
    
    状态向量 Y = [h, hu, hv]，其中 hu = h*u, hv = h*v。
    
    右端项：
        F_h  = -(1/(R*sin(θ))) * d(hu*sin(θ))/dθ
        F_hu = -(1/(R*sin(θ))) * d(hu²/h * sin(θ))/dθ - (g*h/(R*sin(θ))) * dh/dθ - f*hv - C_d*|u|*hu/h
        F_hv = f*hu - C_d*|u|*hv/h
    
    参数:
        theta: 极距网格，numpy数组
        h: 高度
        hu: 经向质量流
        hv: 纬向质量流
    
    返回:
        rhs_h, rhs_hu, rhs_hv: 右端项
    """
    n = len(theta)
    dtheta = theta[1] - theta[0]
    
    # 边界处理：使用对称/反对称边界条件
    # 在极点（theta=0 和 theta=pi）处，sin(theta)=0，需要特殊处理
    sin_theta = np.sin(theta)
    sin_theta = np.where(sin_theta < 1e-10, 1e-10, sin_theta)  # 避免除以零
    
    f = coriolis_parameter(theta)
    
    # 速度
    h_safe = np.where(h > 1e-6, h, 1e-6)
    u = hu / h_safe
    v = hv / h_safe
    
    # === 质量方程右端项 ===
    # -(1/(R*sin(θ))) * d(hu*sin(θ))/dθ
    flux_mass = hu * sin_theta
    dflux_mass = np.zeros(n)
    # 中心差分，内部点
    dflux_mass[1:-1] = (flux_mass[2:] - flux_mass[:-2]) / (2.0 * dtheta)
    # 边界：一阶差分
    dflux_mass[0] = (flux_mass[1] - flux_mass[0]) / dtheta
    dflux_mass[-1] = (flux_mass[-1] - flux_mass[-2]) / dtheta
    rhs_h = -dflux_mass / (EARTH_RADIUS * sin_theta)
    
    # === 经向动量方程右端项 ===
    # 对流项
    flux_u = (hu**2 / h_safe) * sin_theta
    dflux_u = np.zeros(n)
    dflux_u[1:-1] = (flux_u[2:] - flux_u[:-2]) / (2.0 * dtheta)
    dflux_u[0] = (flux_u[1] - flux_u[0]) / dtheta
    dflux_u[-1] = (flux_u[-1] - flux_u[-2]) / dtheta
    conv_u = -dflux_u / (EARTH_RADIUS * sin_theta)
    
    # 气压梯度项: -(g*h/(R*sin(θ))) * dh/dθ
    dh_dtheta = np.zeros(n)
    dh_dtheta[1:-1] = (h[2:] - h[:-2]) / (2.0 * dtheta)
    dh_dtheta[0] = (h[1] - h[0]) / dtheta
    dh_dtheta[-1] = (h[-1] - h[-2]) / dtheta
    pressure_grad = -GRAVITY * h * dh_dtheta / (EARTH_RADIUS * sin_theta)
    
    # 科里奥利力
    coriolis_u = -f * hv
    
    # drag 项
    speed = np.sqrt(u**2 + v**2)
    drag_u = -DRAG_COEFF * speed * u
    
    rhs_hu = conv_u + pressure_grad + coriolis_u + drag_u * h_safe
    
    # === 纬向动量方程右端项 ===
    coriolis_v = f * hu
    drag_v = -DRAG_COEFF * speed * v
    rhs_hv = coriolis_v + drag_v * h_safe
    
    return rhs_h, rhs_hu, rhs_hv


def midpoint_explicit_step(theta, h, hu, hv, dt):
    """
    显式中点法（Modified Euler）单步推进。
    基于种子项目 766_midpoint_explicit 的核心思想。
    
    格式：
        k1 = F(t_n, Y_n)
        Y_mid = Y_n + (dt/2) * k1
        k2 = F(t_n + dt/2, Y_mid)
        Y_{n+1} = Y_n + dt * k2
    
    参数:
        theta: 网格
        h, hu, hv: 当前状态
        dt: 时间步长
    
    返回:
        h_new, hu_new, hv_new: 下一步状态
    """
    # Stage 1
    rhs_h1, rhs_hu1, rhs_hv1 = shallow_water_rhs(theta, h, hu, hv)
    
    # Stage 2 (midpoint predictor)
    h_mid = h + 0.5 * dt * rhs_h1
    hu_mid = hu + 0.5 * dt * rhs_hu1
    hv_mid = hv + 0.5 * dt * rhs_hv1
    
    # 保证高度为正
    h_mid = np.maximum(h_mid, 0.01)
    
    rhs_h2, rhs_hu2, rhs_hv2 = shallow_water_rhs(theta, h_mid, hu_mid, hv_mid)
    
    # Update
    h_new = h + dt * rhs_h2
    hu_new = hu + dt * rhs_hu2
    hv_new = hv + dt * rhs_hv2
    
    h_new = np.maximum(h_new, 0.01)
    
    return h_new, hu_new, hv_new


def initialize_typhoon_background(theta, h0=100.0, amplitude=5.0, theta_center=np.pi/2, width=0.15):
    """
    初始化台风背景场：在球面上生成一个高斯型涡旋。
    
    高度场：
        h(θ) = h0 - amplitude * exp( -((θ - θ_c)/width)² )
    
    参数:
        theta: 极距网格
        h0: 背景高度（m）
        amplitude: 涡旋振幅（m）
        theta_center: 涡旋中心极距
        width: 涡旋宽度
    
    返回:
        h, hu, hv: 初始状态
    """
    h = h0 - amplitude * np.exp(-((theta - theta_center) / width)**2)
    h = np.maximum(h, h0 - 2.0 * amplitude)  # 保证正高度
    
    # 初始静止
    hu = np.zeros_like(theta)
    hv = np.zeros_like(theta)
    
    return h, hu, hv


def solve_shallow_water_sphere(n_theta=180, t_span=(0.0, 86400.0), n_steps=8640):
    """
    求解球坐标浅水方程。
    
    参数:
        n_theta: 空间网格数
        t_span: 时间范围（秒）
        n_steps: 时间步数
    
    返回:
        theta: 空间网格
        t_array: 时间序列
        h_history: 高度历史，形状 (n_theta, n_steps+1)
        hu_history, hv_history: 动量历史
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    h, hu, hv = initialize_typhoon_background(theta)
    
    t0, tf = t_span
    dt_fixed = (tf - t0) / n_steps
    
    t_array = np.zeros(n_steps + 1)
    h_history = np.zeros((n_theta, n_steps + 1))
    hu_history = np.zeros((n_theta, n_steps + 1))
    hv_history = np.zeros((n_theta, n_steps + 1))
    
    h_history[:, 0] = h
    hu_history[:, 0] = hu
    hv_history[:, 0] = hv
    t_array[0] = t0
    
    for i in range(n_steps):
        # 自适应CFL检查
        dt_cfl = compute_cfl_condition(theta, h, hu / np.maximum(h, 1e-6))
        dt = min(dt_fixed, dt_cfl)
        
        h, hu, hv = midpoint_explicit_step(theta, h, hu, hv, dt)
        
        t_array[i + 1] = t_array[i] + dt
        h_history[:, i + 1] = h
        hu_history[:, i + 1] = hu
        hv_history[:, i + 1] = hv
    
    return theta, t_array, h_history, hu_history, hv_history
