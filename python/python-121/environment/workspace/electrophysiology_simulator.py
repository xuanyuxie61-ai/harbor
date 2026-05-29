"""
electrophysiology_simulator.py
心脏电生理集成模拟器

功能:
1. 整合离子通道动力学、组织反应扩散、网格生成、数值积分、随机采样和线性代数模块
2. 提供高层次的电生理模拟接口
3. 心律失常分析工具（触发、传导、折返）
4. 数值稳定性监控与误差分析
"""

import numpy as np
from math import sqrt, exp, pi


def create_stimulus_mask(nx, ny, x_range, y_range, dx, dy):
    """
    创建刺激区域掩码
    
    参数:
        nx, ny: 网格尺寸
        x_range: (xmin, xmax) 刺激x范围（归一化0-1）
        y_range: (ymin, ymax) 刺激y范围（归一化0-1）
        dx, dy: 实际空间步长
    返回:
        mask: (nx, ny) 布尔掩码
    """
    mask = np.zeros((nx, ny), dtype=bool)
    
    ix_min = int(x_range[0] * nx)
    ix_max = int(x_range[1] * nx)
    iy_min = int(y_range[0] * ny)
    iy_max = int(y_range[1] * ny)
    
    mask[ix_min:ix_max, iy_min:iy_max] = True
    return mask


def square_wave_stimulus(t, amplitude=-1.0, duration=2.0, period=300.0):
    """
    方波刺激电流
    
    I_stim(t) = amplitude  if t mod period < duration
                0          otherwise
    """
    if (t % period) < duration:
        return amplitude
    return 0.0


def compute_wavefront_velocity(u_history, dx, dt, threshold=0.5):
    """
    计算波前传导速度
    
    方法:
    1. 检测每个时间步波前位置（u > threshold 的区域质心）
    2. 拟合位置-时间曲线，斜率即为传导速度
    
    参数:
        u_history: (n_time, nx, ny) 膜电位历史
        dx: 空间步长 (cm)
        dt: 时间步长 (ms)
        threshold: 波前检测阈值
    返回:
        velocity: 传导速度估计 (cm/ms)
        positions: 波前位置历史
    """
    n_time, nx, ny = u_history.shape
    positions = []
    times = []
    
    for t_idx in range(n_time):
        # 找到超过阈值的点
        above_threshold = u_history[t_idx] > threshold
        if np.any(above_threshold):
            # 计算质心位置
            indices = np.argwhere(above_threshold)
            if len(indices) > 0:
                cx = np.mean(indices[:, 0]) * dx
                cy = np.mean(indices[:, 1]) * dx
                positions.append((cx, cy))
                times.append(t_idx)
    
    if len(positions) < 2:
        return 0.0, positions
    
    positions = np.array(positions)
    times = np.array(times) * dt
    
    # 线性回归估计速度
    # x(t) = x0 + v*t
    if len(times) > 1:
        vx = np.polyfit(times, positions[:, 0], 1)[0] if len(times) > 1 else 0.0
        vy = np.polyfit(times, positions[:, 1], 1)[0] if len(times) > 1 else 0.0
        velocity = sqrt(vx ** 2 + vy ** 2)
    else:
        velocity = 0.0
    
    return velocity, positions


def detect_reentrant_activity(u_history, threshold=0.5, min_cycle_length=50):
    """
    检测折返活动（reentrant activity）
    
    折返是心律失常的核心机制，特征为:
    1. 波前在组织中持续旋转
    2. 形成螺旋波（spiral wave）
    
    检测方法:
    - 分析波前拓扑结构
    - 检测旋转中心（phase singularity）
    
    参数:
        u_history: 膜电位历史
        threshold: 波前阈值
        min_cycle_length: 最小周期长度（时间步）
    返回:
        reentrant_detected: 是否检测到折返
        phase_singularities: 相位奇点位置列表
    """
    n_time, nx, ny = u_history.shape
    phase_singularities = []
    
    # 计算相位场
    for t_idx in range(0, n_time, max(1, n_time // 20)):
        u = u_history[t_idx]
        
        # 简化：检测波前交叉点
        # 在螺旋波中心，所有相位的等值线汇聚
        grad_x = np.zeros((nx, ny))
        grad_y = np.zeros((nx, ny))
        
        grad_x[1:nx - 1, :] = u[2:nx, :] - u[0:nx - 2, :]
        grad_y[:, 1:ny - 1] = u[:, 2:ny] - u[:, 0:ny - 2]
        
        # 检测梯度方向突变区域
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 寻找局部极小值（相位奇点候选）
        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                local = grad_mag[i - 1:i + 2, j - 1:j + 2]
                if grad_mag[i, j] == np.min(local) and grad_mag[i, j] < 0.1:
                    phase_singularities.append((t_idx, i, j))
    
    reentrant_detected = len(phase_singularities) > 0
    
    return reentrant_detected, phase_singularities


def compute_action_potential_duration(u_history, dt, threshold_up=0.3, threshold_down=0.3):
    """
    计算动作电位时程（APD）
    
    APD: 膜电位从去极化阈值到复极化阈值的时间间隔
    
    参数:
        u_history: 单点膜电位时间序列 或 (n_time, nx, ny) 场
        dt: 时间步长
        threshold_up: 去极化阈值
        threshold_down: 复极化阈值
    返回:
        apd: 动作电位时程 (ms)
    """
    if u_history.ndim == 3:
        # 取中心点
        nx, ny = u_history.shape[1], u_history.shape[2]
        u_series = u_history[:, nx // 2, ny // 2]
    else:
        u_series = u_history
    
    n_time = len(u_series)
    
    # 检测去极化和复极化时刻
    upstrokes = []
    downstrokes = []
    
    for i in range(1, n_time):
        if u_series[i - 1] < threshold_up <= u_series[i]:
            upstrokes.append(i)
        if u_series[i - 1] > threshold_down >= u_series[i]:
            downstrokes.append(i)
    
    # 计算APD
    apds = []
    for up in upstrokes:
        for down in downstrokes:
            if down > up:
                apds.append((down - up) * dt)
                break
    
    if len(apds) > 0:
        return np.mean(apds)
    return 0.0


def compute_refractory_period(u_history, dt, threshold=0.1):
    """
    估算有效不应期（ERP）
    
    ERP: 组织无法被再次激动的最短时间
    """
    apd = compute_action_potential_duration(u_history, dt)
    # 简化估计: ERP ≈ APD + 20ms
    return apd + 20.0


def compute_wavelength(velocity, apd):
    """
    计算波长 λ = v * APD
    
    波长是理解折返稳定性的关键参数:
    - 若组织尺寸 > λ: 折返可能稳定
    - 若组织尺寸 < λ: 折返会自熄灭
    """
    if velocity <= 0 or apd <= 0:
        return 0.0
    return velocity * apd


def arrhythmia_risk_index(u_history, dx, dt, threshold=0.5):
    """
    计算心律失常风险指数
    
    综合多个指标:
    1. 传导速度异常
    2. APD离散度
    3. 波前碎裂程度
    
    返回0-1之间的风险指数
    """
    velocity, _ = compute_wavefront_velocity(u_history, dx, dt, threshold)
    apd = compute_action_potential_duration(u_history, dt)
    
    # 正常传导速度约 0.5-1.0 mm/ms = 0.05-0.1 cm/ms
    # 正常APD约 200-400 ms
    
    # 速度异常度
    v_normal = 0.07  # cm/ms
    v_abnormality = abs(velocity - v_normal) / v_normal if v_normal > 0 else 0.0
    v_abnormality = min(1.0, v_abnormality)
    
    # APD异常度
    apd_normal = 300.0  # ms
    apd_abnormality = abs(apd - apd_normal) / apd_normal if apd_normal > 0 else 0.0
    apd_abnormality = min(1.0, apd_abnormality)
    
    # 综合风险指数
    risk = 0.5 * v_abnormality + 0.5 * apd_abnormality
    
    return risk


def run_full_simulation(nx=64, ny=64, T=500.0, dt=0.05, dx=0.025,
                         D_f=0.001, D_t=0.0002,
                         a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002,
                         solver='adi',
                         n_stimuli=3, stim_period=150.0,
                         fiber_model='parallel',
                         add_noise=False, noise_level=0.01):
    """
    运行完整的心脏电生理模拟
    
    参数:
        nx, ny: 空间网格尺寸
        T: 总模拟时间 (ms)
        dt: 时间步长 (ms)
        dx: 空间步长 (cm)
        D_f, D_t: 纤维/横向扩散系数 (cm²/ms)
        a, k, mu1, mu2, eps: Aliev-Panfilov参数
        solver: 求解器类型
        n_stimuli: 刺激次数
        stim_period: 刺激周期 (ms)
        fiber_model: 纤维模型
        add_noise: 是否添加离子通道噪声
        noise_level: 噪声水平
    返回:
        results: 包含模拟结果的字典
    """
    from ion_channel_dynamics import aliev_panfilov_reaction, generate_ion_channel_noise
    from tissue_reaction_diffusion import (solve_reaction_diffusion_2d,
                                            generate_fiber_angle_field,
                                            build_diffusion_tensor)
    from mesh_generator import generate_cardiac_mesh
    from linear_algebra_core import stability_eigenvalue_analysis
    
    # 生成纤维角度场
    fiber_angle = generate_fiber_angle_field(nx, ny, fiber_model)
    Dxx, Dxy, Dyy = build_diffusion_tensor(D_f, D_t, fiber_angle)
    
    # 初始条件
    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))
    
    # 添加小的随机扰动
    u0 += 0.01 * np.random.randn(nx, ny)
    
    # 定义刺激区域（左下角小块区域）
    stim_mask = create_stimulus_mask(nx, ny, (0.0, 0.15), (0.0, 0.15), dx, dx)
    
    def stimulus(t):
        for i in range(n_stimuli):
            t_stim = i * stim_period
            if t_stim <= t < t_stim + 2.0:
                return -1.0
        return 0.0
    
    # 反应参数
    reaction_params = {'a': a, 'k': k, 'mu1': mu1, 'mu2': mu2, 'eps': eps}
    
    # 扩散系数（简化使用标量平均，或张量）
    D_eff = (Dxx, Dxy, Dyy)
    
    # 运行模拟
    print(f"  Running reaction-diffusion simulation: nx={nx}, ny={ny}, T={T}ms, solver={solver}")
    u_hist, v_hist, t_hist = solve_reaction_diffusion_2d(
        u0, v0, D_eff, dx, dx, dt, T,
        aliev_panfilov_reaction, reaction_params,
        solver=solver,
        stimulus_func=stimulus,
        stimulus_region=stim_mask
    )
    
    # 添加离子通道噪声（后处理）
    if add_noise:
        noise = generate_ion_channel_noise(1, dt, T, D_ion=noise_level, alpha=1.5)
        for t_idx in range(min(len(u_hist), len(noise))):
            u_hist[t_idx] += 0.01 * noise[t_idx, 0]
    
    # 分析结果
    print("  Computing wavefront velocity...")
    velocity, positions = compute_wavefront_velocity(u_hist, dx, dt)
    
    print("  Computing APD...")
    apd = compute_action_potential_duration(u_hist, dt)
    
    print("  Computing wavelength...")
    wavelength = compute_wavelength(velocity, apd)
    
    print("  Computing refractory period...")
    erp = compute_refractory_period(u_hist, dt)
    
    print("  Detecting reentrant activity...")
    reentrant, singularities = detect_reentrant_activity(u_hist)
    
    print("  Computing arrhythmia risk index...")
    risk = arrhythmia_risk_index(u_hist, dx, dt)
    
    # TODO: Hole 3 - 实现稳定性特征值分析
    # 需要构建:
    #   1. 反应项 Jacobian J_reaction (对角矩阵):
    #      df_du = ∂f/∂u, 其中 f(u,v) = -k*u*(u-a)*(u-1) - u*v
    #      即 df_du = -k*(3*u² - 2*(a+1)*u + a) - v
    #   2. 扩散矩阵 D_matrix (简化对角近似):
    #      反映 2D 各向同性拉普拉斯的离散化特征值
    #   3. 调用 stability_eigenvalue_analysis(D_matrix, J_reaction)
    # 注意: Jacobian 必须与 ion_channel_dynamics.py 中 aliev_panfilov_reaction 的 f 一致
    raise NotImplementedError("Hole 3: stability analysis 待实现")
    
    results = {
        'u_history': u_hist,
        'v_history': v_hist,
        't_history': t_hist,
        'nx': nx,
        'ny': ny,
        'dx': dx,
        'dt': dt,
        'T': T,
        'velocity': velocity,
        'apd': apd,
        'wavelength': wavelength,
        'erp': erp,
        'reentrant_detected': reentrant,
        'phase_singularities': singularities,
        'risk_index': risk,
        'lambda_max': lambda_max,
        'is_stable': is_stable,
        'solver': solver,
        'fiber_model': fiber_model
    }
    
    return results
