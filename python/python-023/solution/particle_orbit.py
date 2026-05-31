#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
带电粒子Lorentz力轨道积分器
================================================================================

基于 1064_sensitive_ode 的高敏感性ODE思想，积分带电粒子在电磁场中的
运动方程。在分形湍流磁场中，粒子轨道对初值极其敏感，呈现混沌行为。

核心物理方程：

相对论Lorentz力方程：
    d𝐫/dt = γ⁻¹ 𝐩 / m
    d𝐩/dt = q (𝐄 + 𝐯 × 𝐁)

其中相对论动量 𝐩 = γ m 𝐯，洛伦兹因子 γ = √(1 + p²/(m²c²))。

在均匀磁场 B = B₀ ẑ 下的引导中心漂移：
    𝐯_⟂ = v_⟂ [ cos(Ω t + φ₀) 𝐱̂ - sin(Ω t + φ₀) ŷ ]
    回旋频率 Ω = |q| B₀ / (γ m)

数值方法：
- 显式Boris推进器（标准PIC方法），保持相空间体积
- 对高敏感轨道使用自适应步长Runge-Kutta (Dormand-Prince)
- 李雅普诺夫指数计算以量化混沌程度

混沌指标：
    λ = lim_{t→∞} (1/t) ln( |δ𝐫(t)| / |δ𝐫(0)| )
其中 δ𝐫 为两个邻近轨道的分离。
================================================================================
"""

import numpy as np


def boris_push(x, v, q, m, B, E, dt):
    """
    Boris 推进器：保持相空间体积的显式电磁场粒子推进算法。
    
    数值格式：
        𝐯⁻ = 𝐯^{n-1/2} + (q dt / 2m) 𝐄
        𝐯' = 𝐯⁻ + 𝐯⁻ × 𝐭
        𝐯⁺ = 𝐯⁻ + 𝐯' × 2𝐭 / (1 + |𝐭|²)
        𝐯^{n+1/2} = 𝐯⁺ + (q dt / 2m) 𝐄
        𝐱^{n+1} = 𝐱^n + dt 𝐯^{n+1/2}
    
    其中 𝐭 = (q dt / 2m) 𝐁 / γ
    """
    # 相对论因子
    v_sq = np.sum(v**2)
    c = 2.99792458e8
    gamma = np.sqrt(1.0 + v_sq / c**2)
    
    # 半步电场加速
    v_minus = v + (q * dt / (2.0 * m)) * E
    
    # 磁场旋转
    t_vec = (q * dt / (2.0 * m * gamma)) * B
    t_sq = np.sum(t_vec**2)
    
    v_prime = v_minus + np.cross(v_minus, t_vec)
    
    # 避免除零
    denom = 1.0 + t_sq
    if denom < 1e-30:
        denom = 1e-30
    
    v_plus = v_minus + np.cross(v_prime, 2.0 * t_vec) / denom
    
    # 半步电场加速
    v_new = v_plus + (q * dt / (2.0 * m)) * E
    
    # 位置更新
    x_new = x + dt * v_new / gamma
    
    return x_new, v_new


def rk45_step(f, t, y, dt, args=()):
    """
    Dormand-Prince RK45单步积分，带自适应步长控制。
    
     butcher表 (Dormand-Prince):
    0    |
    1/5  | 1/5
    3/10 | 3/40        9/40
    4/5  | 44/45      -56/15      32/9
    8/9  | 19372/6561 -25360/2187 64448/6561 -212/729
    1    | 9017/3168  -355/33     46732/5247  49/176   -5103/18656
    1    | 35/384      0          500/1113    125/192  -2187/6784   11/84
    -----------------------------------------------------------------
         | 35/384      0          500/1113    125/192  -2187/6784   11/84      0
         | 5179/57600  0          7571/16695  393/640  -92097/339200 187/2100  1/40
    """
    a = [
        [0.0],
        [1.0/5.0],
        [3.0/40.0, 9.0/40.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]
    ]
    
    b4 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    b5 = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]
    
    c = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
    
    k = []
    k.append(np.array(f(t, y, *args)))
    
    for i in range(1, 7):
        ti = t + c[i] * dt
        yi = y.copy()
        for j in range(i):
            yi = yi + dt * a[i][j] * k[j]
        k.append(np.array(f(ti, yi, *args)))
    
    y4 = y + dt * sum(b4[i] * k[i] for i in range(7))
    y5 = y + dt * sum(b5[i] * k[i] for i in range(7))
    
    error = np.linalg.norm(y5 - y4)
    
    return y5, error


def lorentz_force(t, state, q, m, B_field_func, E_field_func):
    """
    Lorentz力方程的右手边：d/dt [x, y, z, vx, vy, vz] = [v, (q/m)(E + v×B)]
    """
    x = state[0:3]
    v = state[3:6]
    
    B = B_field_func(x, t)
    E = E_field_func(x, t)
    
    c = 2.99792458e8
    v_sq = np.sum(v**2)
    gamma = np.sqrt(1.0 + v_sq / c**2)
    
    dxdt = v / gamma
    dvdt = (q / m) * (E + np.cross(v, B))
    
    return np.concatenate([dxdt, dvdt])


def integrate_lorentz_orbits(x0, v0, B0_vec, params, t_span, n_steps):
    """
    积分多个粒子的Lorentz力轨道。
    
    参数
    ----
    x0 : ndarray, shape (N, 3)
        初始位置 [m]。
    v0 : ndarray, shape (N, 3)
        初始速度 [m/s]。
    B0_vec : ndarray, shape (3,)
        平均磁场向量 [T]。
    params : dict
        物理参数。
    t_span : float
        总积分时间 [s]。
    n_steps : int
        时间步数。
        
    返回
    ----
    orbits : ndarray, shape (N, n_steps+1, 6)
        每个粒子在每个时间步的 [x, v]。
    """
    N = x0.shape[0]
    q_e = params['q_e']
    m_e = params['m_e']
    c = params['c']
    
    dt = t_span / n_steps
    
    # 定义场函数
    def B_field(x, t):
        """磁场 = 均匀背景 + 小扰动"""
        if x.ndim > 1:
            x = x.flatten()
        # 均匀背景场
        B = B0_vec.copy()
        # 添加空间依赖的小扰动（模拟分形湍流）
        B[0] += 0.05 * B0_vec[2] * np.sin(2 * np.pi * x[2] / 1e4)
        B[1] += 0.05 * B0_vec[2] * np.cos(2 * np.pi * x[2] / 1e4)
        return B
    
    def E_field(x, t):
        """电场 = 0（简化）"""
        return np.zeros(3)
    
    orbits = np.zeros((N, n_steps + 1, 6))
    orbits[:, 0, 0:3] = x0
    orbits[:, 0, 3:6] = v0
    
    # 步长限制：确保满足回旋分辨率
    Omega_e = params['Omega_e']
    dt_max = 0.1 / Omega_e  # 每回旋周期至少10步
    
    for i in range(N):
        state = np.concatenate([x0[i], v0[i]])
        
        for step in range(n_steps):
            # 使用Boris推进器（主积分）
            x_curr = state[0:3]
            v_curr = state[3:6]
            
            B_curr = B_field(x_curr, step * dt)
            E_curr = E_field(x_curr, step * dt)
            
            # 子步进（确保稳定性）
            n_sub = max(1, int(np.ceil(dt / dt_max)))
            dt_sub = dt / n_sub
            
            for _ in range(n_sub):
                x_curr, v_curr = boris_push(x_curr, v_curr, q_e, m_e, B_curr, E_curr, dt_sub)
            
            state = np.concatenate([x_curr, v_curr])
            orbits[i, step + 1] = state
    
    # 计算最大李雅普诺夫指数（混沌指标）
    if N >= 2:
        separation = np.linalg.norm(orbits[0, :, 0:3] - orbits[1, :, 0:3], axis=1)
        # 避免对零取对数
        separation = np.maximum(separation, 1e-30)
        times = np.linspace(0, t_span, n_steps + 1)
        # 线性拟合 ln(δr) vs t
        valid = separation > 1e-20
        if np.sum(valid) > 10:
            lyap = np.polyfit(times[valid], np.log(separation[valid]), 1)[0]
            print(f"       最大李雅普诺夫指数 λ_max = {lyap:.4e} s⁻¹")
    
    return orbits
