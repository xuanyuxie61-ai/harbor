"""
thermal_solver.py
热场求解与热-密度耦合模块

融合以下种子项目的核心算法：
  - 512_heated_plate：矩形区域稳态热方程的Jacobi迭代
  - 1164_stiff_ode：刚性常微分方程的数值处理
  - 343_euler：Euler法时间推进

物理背景：
  深部地幔密度异常与温度场密切相关，通过热膨胀状态方程耦合：
      rho(T) = rho_0 * [1 - alpha_th * (T - T_0)]
  
  稳态热传导方程（有热源）：
      nabla . (k nabla T) + H = 0
  其中 k 为热导率，H 为放射性生热率。
  
  瞬态热传导方程：
      rho * C_p * dT/dt = nabla . (k nabla T) + H
  
  当考虑地幔对流时，需加入平流项：
      rho * C_p * (dT/dt + v . nabla T) = nabla . (k nabla T) + H + Phi_viscous
"""

import numpy as np


# 典型地球物理参数
RHO_ROCK = 3000.0       # kg/m^3
CP_ROCK = 1000.0        # J/(kg K)
K_THERMAL = 3.0         # W/(m K)
ALPHA_THERMAL = 3e-5    # K^-1
H_RADIOGENIC = 1e-6     # W/m^3


def jacobi_thermal_2d(nx, ny, dx, dy, k_field, H_field, T_boundary,
                       epsilon=1e-8, max_iter=50000):
    """
    使用Jacobi迭代求解二维稳态热传导方程。
    
    融合 512_heated_plate 的核心算法。
    
    离散格式（五点差分）：
        T[i,j] = (k_e T[i+1,j] + k_w T[i-1,j] + k_n T[i,j+1] + k_s T[i,j-1] + H[i,j] dx^2)
                  / (k_e + k_w + k_n + k_s)
    
    其中 k_e = 2*k[i,j]*k[i+1,j] / (k[i,j]+k[i+1,j]) 为调和平均热导率。
    
    参数：
        nx, ny: 网格数
        dx, dy: 网格间距 [m]
        k_field: (nx, ny) 热导率场 [W/(m K)]
        H_field: (nx, ny) 热源项 [W/m^3]
        T_boundary: 边界温度值（字典或固定值）
        epsilon: 收敛容差
        max_iter: 最大迭代次数
    返回：
        T: (nx, ny) 温度场 [K]
        iterations: 实际迭代次数
    """
    if nx < 3 or ny < 3:
        raise ValueError("Grid size must be at least 3x3")
    
    # 初始化
    T = np.ones((nx, ny), dtype=float) * 300.0  # 初始300K
    
    # 设置边界
    if isinstance(T_boundary, dict):
        if 'top' in T_boundary:
            T[0, :] = T_boundary['top']
        if 'bottom' in T_boundary:
            T[-1, :] = T_boundary['bottom']
        if 'left' in T_boundary:
            T[:, 0] = T_boundary['left']
        if 'right' in T_boundary:
            T[:, -1] = T_boundary['right']
    else:
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary
    
    T_new = T.copy()
    iterations = 0
    diff = epsilon + 1.0
    
    while diff >= epsilon and iterations < max_iter:
        T_old = T.copy()
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # 调和平均热导率（处理物性突变）
                ke = 2.0 * k_field[i, j] * k_field[i + 1, j] / (k_field[i, j] + k_field[i + 1, j] + 1e-12)
                kw = 2.0 * k_field[i, j] * k_field[i - 1, j] / (k_field[i, j] + k_field[i - 1, j] + 1e-12)
                kn = 2.0 * k_field[i, j] * k_field[i, j + 1] / (k_field[i, j] + k_field[i, j + 1] + 1e-12)
                ks = 2.0 * k_field[i, j] * k_field[i, j - 1] / (k_field[i, j] + k_field[i, j - 1] + 1e-12)
                
                # 五点差分 Jacobi 更新
                # 热源项 H [W/m^3] 需乘以 dx*dy 并除以等效热导率
                # 等效热导率 k_eq = (ke + kw + kn + ks) / 4
                k_eq = (ke + kw + kn + ks) / 4.0
                source_term = H_field[i, j] * dx * dx / (k_eq + 1e-30)
                
                numerator = ke * T_old[i + 1, j] + kw * T_old[i - 1, j] + \
                            kn * T_old[i, j + 1] + ks * T_old[i, j - 1]
                denominator = ke + kw + kn + ks
                
                if denominator > 1e-15:
                    T_new[i, j] = numerator / denominator + source_term
                else:
                    T_new[i, j] = T_old[i, j]
        
        T = T_new.copy()
        diff = np.max(np.abs(T - T_old))
        iterations += 1
    
    return T, iterations


def adi_thermal_2d(nx, ny, dx, dy, k, H, T_boundary, dt, t_max,
                    rho=3000.0, cp=1000.0):
    """
    使用交替方向隐式法（ADI）求解二维瞬态热传导方程。
    
    ADI格式将二维问题分解为两个一维隐式步：
      半步（x方向隐式）：
          (T^{n+1/2} - T^n) / (dt/2) = k/rho/cp * (d^2 T^{n+1/2}/dx^2 + d^2 T^n/dy^2) + H/rho/cp
      全步（y方向隐式）：
          (T^{n+1} - T^{n+1/2}) / (dt/2) = k/rho/cp * (d^2 T^{n+1/2}/dx^2 + d^2 T^{n+1}/dy^2) + H/rho/cp
    
    无条件稳定，时间精度 O(dt^2)，空间精度 O(dx^2 + dy^2)。
    
    参数：
        nx, ny: 网格数
        dx, dy: 间距 [m]
        k: 热导率 [W/(m K)]（标量或场）
        H: 热源 [W/m^3]
        T_boundary: 边界温度
        dt: 时间步 [s]
        t_max: 总时间 [s]
        rho, cp: 密度和比热
    返回：
        T: 最终温度场
        T_history: 时间序列（每100步保存一次）
    """
    if nx < 3 or ny < 3:
        raise ValueError("Grid too small")
    
    n_steps = int(t_max / dt)
    if n_steps < 1:
        n_steps = 1
        dt = t_max
    
    T = np.ones((nx, ny), dtype=float) * 300.0
    
    # 边界条件
    if isinstance(T_boundary, dict):
        if 'top' in T_boundary: T[0, :] = T_boundary['top']
        if 'bottom' in T_boundary: T[-1, :] = T_boundary['bottom']
        if 'left' in T_boundary: T[:, 0] = T_boundary['left']
        if 'right' in T_boundary: T[:, -1] = T_boundary['right']
    else:
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary
    
    k = float(k)
    kappa = k / (rho * cp)
    rx = kappa * dt / (2.0 * dx**2)
    ry = kappa * dt / (2.0 * dy**2)
    
    T_history = []
    
    for step in range(n_steps):
        T_half = T.copy()
        
        # 第一步：x方向隐式
        for j in range(1, ny - 1):
            # 构造三对角矩阵
            a = np.zeros(nx - 2)
            b = np.zeros(nx - 2)
            c = np.zeros(nx - 2)
            d = np.zeros(nx - 2)
            
            for i in range(1, nx - 1):
                idx = i - 1
                a[idx] = -rx
                b[idx] = 1.0 + 2.0 * rx
                c[idx] = -rx
                d[idx] = T[i, j] + ry * (T[i, j + 1] - 2.0 * T[i, j] + T[i, j - 1]) + (H * dt / 2.0) / (rho * cp)
            
            # Thomas算法求解
            sol = _thomas_algorithm(a, b, c, d)
            T_half[1:nx - 1, j] = sol
        
        # 保持边界
        if isinstance(T_boundary, dict):
            if 'top' in T_boundary: T_half[0, :] = T_boundary['top']
            if 'bottom' in T_boundary: T_half[-1, :] = T_boundary['bottom']
            if 'left' in T_boundary: T_half[:, 0] = T_boundary['left']
            if 'right' in T_boundary: T_half[:, -1] = T_boundary['right']
        
        # 第二步：y方向隐式
        for i in range(1, nx - 1):
            a = np.zeros(ny - 2)
            b = np.zeros(ny - 2)
            c = np.zeros(ny - 2)
            d = np.zeros(ny - 2)
            
            for j in range(1, ny - 1):
                idx = j - 1
                a[idx] = -ry
                b[idx] = 1.0 + 2.0 * ry
                c[idx] = -ry
                d[idx] = T_half[i, j] + rx * (T_half[i + 1, j] - 2.0 * T_half[i, j] + T_half[i - 1, j]) + (H * dt / 2.0) / (rho * cp)
            
            sol = _thomas_algorithm(a, b, c, d)
            T[i, 1:ny - 1] = sol
        
        # 保持边界
        if isinstance(T_boundary, dict):
            if 'top' in T_boundary: T[0, :] = T_boundary['top']
            if 'bottom' in T_boundary: T[-1, :] = T_boundary['bottom']
            if 'left' in T_boundary: T[:, 0] = T_boundary['left']
            if 'right' in T_boundary: T[:, -1] = T_boundary['right']
        
        if step % max(1, n_steps // 20) == 0:
            T_history.append(T.copy())
    
    return T, T_history


def _thomas_algorithm(a, b, c, d):
    """
    Thomas算法求解三对角线性系统。
    
    系统形式：
        b[0]*x[0] + c[0]*x[1] = d[0]
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
        a[n-1]*x[n-2] + b[n-1]*x[n-1] = d[n-1]
    """
    n = len(d)
    if n == 0:
        return np.array([])
    
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def stiff_thermal_decay(tspan, T0, lambda_stiff, T_ambient, n_steps=1000):
    """
    求解刚性热衰减ODE：
        dT/dt = lambda * (T_ambient - T)
    
    融合 1164_stiff_ode 的核心算法。
    
    当 lambda 很大时（例如快速热扩散），方程变为刚性，
    显式Euler需要极小步长。这里使用解析解和隐式Euler处理刚性。
    
    解析解：
        T(t) = T_ambient + (T0 - T_ambient) * exp(-lambda * t)
    
    隐式Euler离散：
        (T^{n+1} - T^n) / dt = lambda * (T_ambient - T^{n+1})
        => T^{n+1} = (T^n + dt * lambda * T_ambient) / (1 + dt * lambda)
    
    参数：
        tspan: (t0, t1) 时间区间 [s]
        T0: 初始温度 [K]
        lambda_stiff: 衰减系数 [1/s]
        T_ambient: 环境温度 [K]
        n_steps: 步数
    返回：
        t: 时间点
        T: 温度
        T_exact: 精确解
    """
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    T = np.zeros(n_steps + 1)
    T[0] = T0
    
    for i in range(n_steps):
        # 隐式Euler（对刚性稳定）
        T[i + 1] = (T[i] + dt * lambda_stiff * T_ambient) / (1.0 + dt * lambda_stiff)
    
    T_exact = T_ambient + (T0 - T_ambient) * np.exp(-lambda_stiff * (t - t0))
    return t, T, T_exact


def euler_explicit_thermal(tspan, T0, source_func, n_steps=1000):
    """
    显式Euler法求解一般热ODE：dT/dt = source_func(t, T)。
    
    融合 343_euler 的核心算法。
    
    离散格式：
        T^{n+1} = T^n + dt * source_func(t^n, T^n)
    
    参数：
        tspan: (t0, t1)
        T0: 初始条件（标量或向量）
        source_func: callable(t, T) -> dT/dt
        n_steps: 步数
    返回：
        t: 时间点
        T: 解
    """
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    
    T0 = np.asarray(T0, dtype=float)
    if T0.ndim == 0:
        T = np.zeros(n_steps + 1)
        T[0] = T0
        for i in range(n_steps):
            dT = source_func(t[i], T[i])
            T[i + 1] = T[i] + dt * float(dT)
    else:
        m = len(T0)
        T = np.zeros((n_steps + 1, m))
        T[0, :] = T0
        for i in range(n_steps):
            dT = source_func(t[i], T[i, :])
            dT = np.asarray(dT, dtype=float).flatten()
            T[i + 1, :] = T[i, :] + dt * dT
    
    return t, T


def thermal_expansion_density(rho0, T, T0, alpha=ALPHA_THERMAL):
    """
    热膨胀状态方程：
        rho(T) = rho0 * [1 - alpha * (T - T0)]
    
    参数：
        rho0: 参考密度 [kg/m^3]
        T: 温度 [K]（标量或数组）
        T0: 参考温度 [K]
        alpha: 热膨胀系数 [K^-1]
    返回：
        rho: 密度 [kg/m^3]
    """
    T = np.asarray(T, dtype=float)
    dT = T - T0
    rho = rho0 * (1.0 - alpha * dT)
    # 边界保护：密度不能为负
    rho = np.maximum(rho, 100.0)
    return rho


def coupled_thermal_density_evolution(nx, ny, dx, dy, rho0, T0, T_boundary,
                                       dt, n_steps, k=K_THERMAL, H=H_RADIOGENIC,
                                       alpha=ALPHA_THERMAL, rho_ref=3000.0, cp=CP_ROCK):
    """
    耦合热-密度演化计算。
    
    每步：
      1. 用ADI求解热场
      2. 用热膨胀方程更新密度场
      3. 用密度场重新计算热参数（简化模型中固定）
    
    参数：
        nx, ny, dx, dy: 空间网格
        rho0: 初始密度场 (nx, ny) [kg/m^3]
        T0: 初始温度场 (nx, ny) [K]
        T_boundary: 边界温度
        dt: 时间步 [s]
        n_steps: 步数
    返回：
        T_final: 最终温度场
        rho_final: 最终密度场
        history: 每步的 (T, rho) 列表
    """
    T = T0.copy()
    rho = rho0.copy()
    history = []
    
    for step in range(n_steps):
        # 单步ADI热求解
        T_new, _ = adi_thermal_2d(nx, ny, dx, dy, k, H, T_boundary, dt, dt)
        T = T_new
        
        # 更新密度
        rho = thermal_expansion_density(rho_ref, T, 300.0, alpha)
        
        if step % max(1, n_steps // 10) == 0:
            history.append((T.copy(), rho.copy()))
    
    return T, rho, history
