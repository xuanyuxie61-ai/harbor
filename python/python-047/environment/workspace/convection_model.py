"""
convection_model.py
地幔对流-扩散与密度异常演化模块

融合以下种子项目的核心算法：
  - 1068_shallow_water_1d：守恒型Lax-Wendroff格式
  - 901_porous_medium_exact：多孔介质方程的Barenblatt自相似解

物理背景：
  地幔中的密度异常不仅由热膨胀引起，还受到物质对流和化学扩散的控制。
  在简化模型中，我们考虑一维垂向密度异常的对流-扩散方程：
  
      d(rho)/dt + d(u*rho)/dz = D * d^2(rho)/dz^2 + S(z,t)
  
  其中 u 为垂向速度（由Stokes流动解给出），D 为化学扩散系数，S 为源项。
  
  对于多孔介质中的流体迁移（如部分熔融导致密度变化），采用多孔介质方程：
      d(rho)/dt = nabla^2(rho^m)
  其Barenblatt自相似解为验证数值方法提供精确基准。
"""

import numpy as np


MANTLE_DENSITY = 3300.0       # kg/m^3
CHEM_DIFFUSIVITY = 1e-6       # m^2/s
GRAVITY_ACC = 9.81            # m/s^2


def lax_wendroff_density_convection(nz, dz, dt, n_steps,
                                     rho_init, u_field, D_diff, source,
                                     bc_type='periodic'):
    """
    使用Lax-Wendroff格式求解一维密度对流-扩散方程。
    
    融合 1068_shallow_water_1d 的核心算法思想。
    
    守恒型方程：
        d(rho)/dt + d(u*rho)/dz = D * d^2(rho)/dz^2 + S(z,t)
    
    Lax-Wendroff两步格式：
      半步（单元界面）：
          rho_{j+1/2}^{n+1/2} = 0.5*(rho_j^n + rho_{j+1}^n)
                                  - 0.5*dt/dz * (u_{j+1}*rho_{j+1}^n - u_j*rho_j^n)
                                  + 0.5*dt*D/dz^2 * (rho_{j+1}^n - 2*rho_j^n + rho_{j-1}^n)
      全步（单元中心）：
          rho_j^{n+1} = rho_j^n - dt/dz * (u_{j+1/2}*rho_{j+1/2}^{n+1/2} - u_{j-1/2}*rho_{j-1/2}^{n+1/2})
                        + dt*D/dz^2 * (rho_{j+1}^n - 2*rho_j^n + rho_{j-1}^n) + dt*S_j
    
    CFL条件要求：dt <= dz / max|u| 且 dt <= dz^2 / (2*D)。
    
    参数：
        nz: 垂向网格数
        dz: 垂向间距 [m]
        dt: 时间步 [s]
        n_steps: 步数
        rho_init: (nz,) 初始密度异常 [kg/m^3]
        u_field: (nz,) 垂向速度场 [m/s]（正向上）
        D_diff: 扩散系数 [m^2/s]
        source: (nz,) 或 callable 源项 [kg/(m^3 s)]
        bc_type: 'periodic', 'fixed', 'reflective'
    返回：
        rho: (nz,) 最终密度场
        history: 每步密度场列表
    """
    if nz < 3:
        raise ValueError("nz must be >= 3")
    
    rho = np.asarray(rho_init, dtype=float).copy()
    u = np.asarray(u_field, dtype=float)
    if len(rho) != nz or len(u) != nz:
        raise ValueError("Field length mismatch with nz")
    
    # CFL条件检查与调整
    umax = np.max(np.abs(u))
    if umax > 0:
        cfl_adv = dz / umax
    else:
        cfl_adv = 1e10
    if D_diff > 0:
        cfl_diff = dz**2 / (2.0 * D_diff)
    else:
        cfl_diff = 1e10
    
    cfl = min(cfl_adv, cfl_diff)
    if dt > cfl:
        # 自动调整步长以保持稳定性
        sub_steps = int(np.ceil(dt / (0.5 * cfl)))
        actual_dt = dt / sub_steps
    else:
        sub_steps = 1
        actual_dt = dt
    
    history = []
    
    for step in range(n_steps):
        for _ in range(sub_steps):
            rho_half = np.zeros(nz - 1)
            
            # 半步（界面值）
            for j in range(nz - 1):
                # 界面速度取平均
                # 对流通量
                flux_adv = actual_dt / dz * (u[j + 1] * rho[j + 1] - u[j] * rho[j])
                # 扩散项（中心差分）
                if D_diff > 0 and j > 0:
                    flux_diff = actual_dt * D_diff / dz**2 * (rho[j + 1] - 2.0 * rho[j] + rho[j - 1])
                else:
                    flux_diff = 0.0
                
                val = 0.5 * (rho[j] + rho[j + 1]) - 0.5 * flux_adv + 0.5 * flux_diff
                # 数值保护
                rho_half[j] = np.clip(val, -1e4, 1e4)
            
            # 全步
            rho_old = rho.copy()
            for j in range(1, nz - 1):
                # 对流项（用界面值）
                u_half_j = 0.5 * (u[j] + u[j + 1])
                u_half_jm1 = 0.5 * (u[j - 1] + u[j])
                flux = actual_dt / dz * (u_half_j * rho_half[j] - u_half_jm1 * rho_half[j - 1])
                
                # 扩散项
                if D_diff > 0:
                    diff = actual_dt * D_diff / dz**2 * (rho_old[j + 1] - 2.0 * rho_old[j] + rho_old[j - 1])
                else:
                    diff = 0.0
                
                # 源项
                if callable(source):
                    src = actual_dt * source(j * dz, step * dt)
                else:
                    src = actual_dt * source[j]
                
                val = rho_old[j] - flux + diff + src
                rho[j] = np.clip(val, -1e4, 1e4)
            
            # 边界条件
            rho = _apply_bc(rho, bc_type)
        
        if step % max(1, n_steps // 20) == 0:
            history.append(rho.copy())
    
    return rho, history


def _apply_bc(rho, bc_type):
    """应用边界条件。"""
    nz = len(rho)
    if bc_type == 'periodic':
        rho[0] = rho[-2]
        rho[-1] = rho[1]
    elif bc_type == 'fixed':
        # 保持边界不变（已在主循环外设置）
        pass
    elif bc_type == 'reflective':
        rho[0] = rho[1]
        rho[-1] = rho[-2]
    elif bc_type == 'zero_gradient':
        rho[0] = 2.0 * rho[1] - rho[2]
        rho[-1] = 2.0 * rho[-2] - rho[-3]
    return rho


def porous_medium_barenblatt(x, t, m=2.0, C=1.0, delta=0.1):
    """
    多孔介质方程的Barenblatt-Pattle自相似精确解。
    
    融合 901_porous_medium_exact 的核心算法。
    
    多孔介质方程：
        d(u)/dt = nabla^2(u^m)
    
    一维自相似解（Barenblatt, 1952）：
        u(x,t) = (t + delta)^{-beta} * [C - gamma * (x / (t + delta)^{beta})^2]^{1/(m-1)}
    
    其中：
        alpha = 1 / (m - 1)
        beta  = 1 / (m + 1)
        gamma = (m - 1) / (2 * m * (m + 1))
    
    当括号内为负时，u = 0（有限传播速度）。
    
    参数：
        x: 空间坐标或数组 [m]
        t: 时间 [s]
        m: 多孔介质指数 (>1)
        C: 振幅参数
        delta: 时间偏移
    返回：
        u: 解值
        ut: 时间导数
        ux: 空间导数
        uxx: 二阶空间导数
    """
    if m <= 1.0:
        raise ValueError("m must be > 1 for porous medium equation")
    
    x = np.asarray(x, dtype=float)
    t = float(t)
    
    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))
    
    bot = (t + delta)**beta
    xi = x / bot
    factor = C - gamma * xi**2
    
    u = np.zeros_like(x)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)
    
    mask = factor > 0.0
    if np.any(mask):
        u[mask] = (t + delta)**(-beta) * factor[mask]**alpha
        ut[mask] = (2.0 * alpha * beta * gamma * (t + delta)**(-1.0 - 3.0 * beta) * x[mask]**2 * factor[mask]**(alpha - 1.0)
                    - beta * (t + delta)**(-1.0 - beta) * factor[mask]**alpha)
        ux[mask] = (-2.0 * alpha * gamma * (t + delta)**(-3.0 * beta) * x[mask] * factor[mask]**(alpha - 1.0))
        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma**2 * (t + delta)**(-5.0 * beta) * x[mask]**2 * factor[mask]**(alpha - 2.0)
                     - 2.0 * alpha * gamma * (t + delta)**(-3.0 * beta) * factor[mask]**(alpha - 1.0))
    
    return u, ut, ux, uxx


def porous_medium_verification(nz, z_max, t_test, m=2.0, C=1.0, delta=0.1):
    """
    验证数值解与Barenblatt精确解的误差。
    
    返回L2和最大范数误差。
    """
    dz = z_max / (nz - 1)
    z = np.linspace(0, z_max, nz)
    
    u_exact, _, _, _ = porous_medium_barenblatt(z, t_test, m, C, delta)
    
    # 简化的有限差分数值解（显式）
    dt = 0.1 * dz**2
    n_steps = int(t_test / dt)
    u_num = porous_medium_barenblatt(z, 0.0, m, C, delta)[0]
    
    for _ in range(n_steps):
        u_old = u_num.copy()
        for j in range(1, nz - 1):
            # d^2(u^m)/dz^2 的离散
            um_jp = u_old[j + 1]**m
            um_j = u_old[j]**m
            um_jm = u_old[j - 1]**m
            u_num[j] = u_old[j] + dt * (um_jp - 2.0 * um_j + um_jm) / dz**2
        u_num[0] = u_num[1]
        u_num[-1] = u_num[-2]
    
    diff = np.abs(u_num - u_exact)
    l2_error = np.sqrt(np.mean(diff**2))
    linf_error = np.max(diff)
    
    return l2_error, linf_error, u_num, u_exact


def stokes_velocity_profile(z, eta, delta_rho, L_scale):
    """
    简化的Stokes流动垂向速度剖面。
    
    对于密度异常 delta_rho 在粘度 eta 的介质中，
    特征Stokes速度（尺度分析）：
        u_stokes ~ delta_rho * g * L^2 / eta
    
    垂向速度剖面（抛物线型，类似于通道流）：
        u(z) = u_max * (z/L) * (1 - z/L)  对于 0 <= z <= L
    
    参数：
        z: (nz,) 深度坐标 [m]（向下为正，z=0为地表）
        eta: 粘度 [Pa s]
        delta_rho: 密度异常 [kg/m^3]
        L_scale: 特征尺度 [m]
    返回：
        u: (nz,) 垂向速度 [m/s]（正向上，即负深度方向）
    """
    z = np.asarray(z, dtype=float)
    u_max = delta_rho * GRAVITY_ACC * L_scale**2 / (eta + 1e-30)
    # 限制速度防止数值不稳定
    u_max = np.clip(u_max, -1e-2, 1e-2)
    
    u = -u_max * (z / L_scale) * (1.0 - z / L_scale)
    return u


def density_anomaly_evolution_full(z, nz, dt, n_steps,
                                    rho_background, delta_rho_init,
                                    eta, D_diff, source,
                                    L_scale=1e5):
    """
    完整的密度异常演化：对流 + 扩散 + 源项。
    
    参数：
        z: 深度坐标数组 [m]
        nz: 网格数
        dt: 时间步 [s]
        n_steps: 步数
        rho_background: 背景密度 [kg/m^3]
        delta_rho_init: 初始密度异常 [kg/m^3]
        eta: 粘度 [Pa s]
        D_diff: 扩散系数 [m^2/s]
        source: 源项
        L_scale: 特征长度 [m]
    返回：
        rho_final: 最终密度场
        history: 演化历史
    """
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    
    # Stokes速度
    u = stokes_velocity_profile(z, eta, delta_rho_init, L_scale)
    
    # 初始密度异常（叠加在背景上）
    rho_init = np.ones(nz) * rho_background
    # 在中心添加高斯型异常
    z_center = L_scale / 2.0
    sigma = L_scale / 10.0
    rho_init += delta_rho_init * np.exp(-((z - z_center)**2) / (2.0 * sigma**2))
    
    rho_final, history = lax_wendroff_density_convection(
        nz, dz, dt, n_steps, rho_init, u, D_diff, source, bc_type='zero_gradient'
    )
    
    return rho_final, history
