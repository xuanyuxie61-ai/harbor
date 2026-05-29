"""
thermal_process.py
==================
增材制造工艺热-流耦合模拟模块。
整合自：
  - 354_fd1d_advection_lax：一维对流方程 Lax 格式
  - 788_navier_stokes_3d_exact：3D Navier-Stokes 精确解用于熔池流体验证

物理背景：
  激光粉末床熔融（LPBF）过程中，高能激光束扫描金属粉末床，
  产生局部熔池。熔池内的流体流动由表面张力梯度（Marangoni 对流）
  和反冲压力驱动。温度场演化直接影响微观组织、残余应力和变形。

核心方程：
  1. 一维热对流-扩散方程（简化工艺模型）：
       ∂T/∂t + v·∂T/∂z = α·∂²T/∂z² + Q(z,t)/(ρ·c_p)
     其中 v 为扫描速度，α 为热扩散系数，Q 为激光热源。

  2. 3D 不可压缩 Navier-Stokes 方程（熔池流体验证）：
       ∂u/∂t + (u·∇)u = -∇p/ρ + ν·∇²u + f
       ∇·u = 0
     使用 Ethier-Steinman 精确解进行数值验证。
"""

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 1. 一维热对流 Lax 格式 (fd1d_advection_lax 思想)
# =============================================================================

def thermal_advection_lax_1d(T0: np.ndarray, z: np.ndarray, dt: float,
                              n_steps: int, v: float, alpha: float,
                              source: Optional[np.ndarray] = None,
                              bc_type: str = "dirichlet") -> np.ndarray:
    """
    使用 Lax 格式求解一维热对流-扩散方程：
        ∂T/∂t + v·∂T/∂z = α·∂²T/∂z² + S(z,t)

    Lax 格式（对流项）+ 中心差分（扩散项）：
        T_i^{n+1} = 0.5*(T_{i-1}^n + T_{i+1}^n)
                    - (v·dt)/(2·dz)·(T_{i+1}^n - T_{i-1}^n)
                    + (α·dt)/dz²·(T_{i+1}^n - 2·T_i^n + T_{i-1}^n)
                    + dt·S_i^n

    CFL 稳定性条件（对流+扩散）：
        |v|·dt/dz ≤ 1   且   α·dt/dz² ≤ 0.5

    Parameters
    ----------
    T0 : ndarray
        初始温度场。
    z : ndarray
        空间网格（均匀）。
    dt : float
        时间步长。
    n_steps : int
        时间步数。
    v : float
        对流速度（扫描速度，可正可负）。
    alpha : float
        热扩散系数。
    source : ndarray, optional
        热源项 S(z,t)，shape 可为 (nz,) 或常数。
    bc_type : str
        "dirichlet" 或 "neumann" 或 "periodic"。

    Returns
    -------
    T : ndarray
        最终温度场。
    """
    nz = len(z)
    dz = z[1] - z[0]
    if dz <= 0:
        raise ValueError("Spatial grid must be strictly increasing.")

    # CFL 检查
    cfl_adv = abs(v) * dt / dz
    cfl_diff = alpha * dt / (dz * dz)
    if cfl_adv > 1.0 or cfl_diff > 0.5:
        # 自动调整 dt 以满足稳定性
        dt_max1 = dz / (abs(v) + 1e-10)
        dt_max2 = 0.5 * dz * dz / (alpha + 1e-10)
        dt = min(dt, 0.9 * min(dt_max1, dt_max2))
        cfl_adv = abs(v) * dt / dz
        cfl_diff = alpha * dt / (dz * dz)

    T = T0.copy().astype(np.float64)
    T_new = np.zeros_like(T)

    for _ in range(n_steps):
        # 内部点
        for i in range(1, nz - 1):
            adv = -cfl_adv * 0.5 * (T[i+1] - T[i-1])
            diff = cfl_diff * (T[i+1] - 2.0*T[i] + T[i-1])
            lax_avg = 0.5 * (T[i-1] + T[i+1])
            T_new[i] = lax_avg + adv + diff
            if source is not None:
                T_new[i] += dt * source[i]

        # 边界条件
        if bc_type == "dirichlet":
            T_new[0] = T[0]
            T_new[-1] = T[-1]
        elif bc_type == "neumann":
            T_new[0] = T_new[1]
            T_new[-1] = T_new[-2]
        elif bc_type == "periodic":
            # 用 Lax 处理周期性边界
            i = 0
            adv = -cfl_adv * 0.5 * (T[1] - T[-2])
            diff = cfl_diff * (T[1] - 2.0*T[i] + T[-2])
            lax_avg = 0.5 * (T[-2] + T[1])
            T_new[i] = lax_avg + adv + diff
            i = nz - 1
            T_new[i] = T_new[0]
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

        T, T_new = T_new, T

    return T


def gaussian_laser_source(z: np.ndarray, z0: float, power: float,
                          spot_size: float, absorptivity: float = 0.3) -> np.ndarray:
    """
    高斯分布激光热源：
        Q(z) = A · P / (√π · r_s) · exp[-(z-z0)² / r_s²]
    其中 A 为吸收率，P 为激光功率，r_s 为光斑特征半径。
    """
    Q = absorptivity * power / (np.sqrt(np.pi) * spot_size) * \
        np.exp(-((z - z0)**2) / (spot_size**2))
    return Q


# =============================================================================
# 2. 多层沉积热循环模拟
# =============================================================================

def simulate_layer_deposition_thermal(n_layers: int, layer_thickness: float,
                                       scan_speed: float, laser_power: float,
                                       thermal_diffusivity: float,
                                       dt_per_layer: int = 200) -> dict:
    """
    模拟 LPBF 多层沉积过程中的温度场演化。

    每一层：
        1. 在表面 (z=0) 施加移动高斯热源
        2. 用 Lax 格式求解热传导
        3. 记录峰值温度和冷却速率

    Returns
    -------
    dict: {'peak_temps': array, 'cooling_rates': array, 'final_profile': array,
           'depth_grid': array}
    """
    # 深度域：向下为正，总深度覆盖所有层 + 基底
    z_max = n_layers * layer_thickness * 3.0
    nz = 101
    z = np.linspace(0, z_max, nz)
    dz = z[1] - z[0]

    # 初始温度（室温 + 预热）
    T_room = 300.0  # K
    T_preheat = 400.0  # K
    T = np.full(nz, T_preheat, dtype=np.float64)

    peak_temps = []
    cooling_rates = []

    for layer in range(n_layers):
        z_surface = layer * layer_thickness
        # 热源在表面附近移动
        spot = layer_thickness * 2.0
        source = gaussian_laser_source(z, z_surface, laser_power, spot)
        # 将热源归一化为有效温升率 (K/s)
        # 1D 简化：Q_eff = Q / (ρ * cp * A_cross) ，这里直接用标度因子
        # 通过试算调整标度因子使峰值温度处于合理范围（~1000-2000 K）
        source = source * 2.5e3  # 标度到合理的温升速率
        # 时间步长自适应
        dt = 0.5 * min(dz / (abs(scan_speed / 1000.0) + 1e-6),
                        0.5 * dz * dz / (thermal_diffusivity + 1e-10))
        T_before = T.copy()
        T = thermal_advection_lax_1d(T, z, dt, dt_per_layer, scan_speed,
                                      thermal_diffusivity, source=source,
                                      bc_type="neumann")
        peak_temps.append(np.max(T))
        # 冷却速率近似 (K/s)
        cooling = (np.max(T_before) - np.max(T)) / (dt * dt_per_layer)
        cooling_rates.append(cooling)

    return {
        "peak_temps": np.array(peak_temps),
        "cooling_rates": np.array(cooling_rates),
        "final_profile": T,
        "depth_grid": z,
    }


# =============================================================================
# 3. 3D Navier-Stokes 精确解 (navier_stokes_3d_exact 思想)
# =============================================================================

def ethier_steinman_solution(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                              T: float, a: float = np.pi/4.0, d: float = np.pi/2.0,
                              nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """
    Ethier-Steinman 3D 不可压缩 Navier-Stokes 精确解析解。

    速度场：
        u = -a·[exp(ax)·sin(ay+dz) + exp(az)·cos(ax+dy)]·exp(-νd²t)
        v = -a·[exp(ay)·sin(az+dx) + exp(ax)·cos(ay+dz)]·exp(-νd²t)
        w = -a·[exp(az)·sin(ax+dy) + exp(ay)·cos(az+dx)]·exp(-νd²t)

    压力场：
        p = -a²/2 · [exp(2ax) + exp(2ay) + exp(2az)
                     + 2·sin(ax+dy)·cos(az+dx)·exp(a(y+z))
                     + 2·sin(ay+dz)·cos(ax+dy)·exp(a(z+x))
                     + 2·sin(az+dx)·cos(ay+dz)·exp(a(x+y))] · exp(-2νd²t)

    该解满足连续性方程 ∇·u = 0 和动量方程。
    常用于验证数值 NS 求解器。

    Parameters
    ----------
    X, Y, Z : ndarray
        空间坐标网格（可广播）。
    T : float
        时间。
    a, d : float
        解参数。
    nu : float
        运动粘度。

    Returns
    -------
    U, V, W, P : ndarray
        三个速度分量和压力。
    """
    exp_nu = np.exp(-nu * d * d * T)
    exp_2nu = np.exp(-2.0 * nu * d * d * T)

    U = -a * (np.exp(a * X) * np.sin(a * Y + d * Z) +
              np.exp(a * Z) * np.cos(a * X + d * Y)) * exp_nu
    V = -a * (np.exp(a * Y) * np.sin(a * Z + d * X) +
              np.exp(a * X) * np.cos(a * Y + d * Z)) * exp_nu
    W = -a * (np.exp(a * Z) * np.sin(a * X + d * Y) +
              np.exp(a * Y) * np.cos(a * Z + d * X)) * exp_nu

    P = -0.5 * a * a * (
        np.exp(2.0 * a * X) + np.exp(2.0 * a * Y) + np.exp(2.0 * a * Z) +
        2.0 * np.sin(a * X + d * Y) * np.cos(a * Z + d * X) * np.exp(a * (Y + Z)) +
        2.0 * np.sin(a * Y + d * Z) * np.cos(a * X + d * Y) * np.exp(a * (Z + X)) +
        2.0 * np.sin(a * Z + d * X) * np.cos(a * Y + d * Z) * np.exp(a * (X + Y))
    ) * exp_2nu

    return U, V, W, P


def verify_ns_residual_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           T: float, a: float = np.pi/4.0, d: float = np.pi/2.0,
                           nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """
    通过数值差分验证 Ethier-Steinman 解的 NS 方程残差。
    计算：
        R_u = ∂u/∂t + u·∂u/∂x + v·∂u/∂y + w·∂u/∂z + ∂p/∂x/ρ - ν·∇²u
        （类似 R_v, R_w）
        R_cont = ∂u/∂x + ∂v/∂y + ∂w/∂z

    理论上，精确解应使残差为零（仅含数值差分误差）。
    """
    U, V, W, P = ethier_steinman_solution(X, Y, Z, T, a, d, nu)

    dx = X[1, 0, 0] - X[0, 0, 0] if X.ndim >= 3 else 0.1
    dy = Y[0, 1, 0] - Y[0, 0, 0] if Y.ndim >= 3 else 0.1
    dz = Z[0, 0, 1] - Z[0, 0, 0] if Z.ndim >= 3 else 0.1

    # 简化：假设规则网格，使用中心差分
    def grad_x(F):
        dF = np.zeros_like(F)
        dF[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx)
        return dF

    def grad_y(F):
        dF = np.zeros_like(F)
        dF[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy)
        return dF

    def grad_z(F):
        dF = np.zeros_like(F)
        dF[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dz)
        return dF

    def laplacian(F):
        return grad_x(grad_x(F)) + grad_y(grad_y(F)) + grad_z(grad_z(F))

    rho = 1.0
    dUdt = np.zeros_like(U)  # 稳态或单时间快照
    conv_x = U * grad_x(U) + V * grad_y(U) + W * grad_z(U)
    R_u = dUdt + conv_x + grad_x(P) / rho - nu * laplacian(U)

    conv_y = U * grad_x(V) + V * grad_y(V) + W * grad_z(V)
    R_v = dUdt + conv_y + grad_y(P) / rho - nu * laplacian(V)

    conv_z = U * grad_x(W) + V * grad_y(W) + W * grad_z(W)
    R_w = dUdt + conv_z + grad_z(P) / rho - nu * laplacian(W)

    R_cont = grad_x(U) + grad_y(V) + grad_z(W)

    return R_u, R_v, R_w, R_cont


# =============================================================================
# 4. 熔池尺寸估算模型
# =============================================================================

def estimate_melt_pool_size(laser_power: float, scan_speed: float,
                             absorptivity: float, thermal_diffusivity: float,
                             melting_temp: float, ambient_temp: float) -> dict:
    """
    基于 Rosenthal 解析解估算熔池尺寸。

    Rosenthal 移动点热源稳态解：
        T(r) = T_0 + (A·P) / (2·π·k·r) · exp[-v·(r+x) / (2·α)]

    其中 r = √(x²+y²+z²)，k 为热导率，α = k/(ρc) 为热扩散系数。

    熔池边界定义在 T = T_melt 处。
    简化估算熔池半宽 w 和深度 d：
        w ≈ (2·A·P) / (π·e·k·(T_m - T_0)·v)
        d ≈ w / 2

    注意：输入 scan_speed 单位 mm/s，内部转换为 m/s。
    """
    # 典型金属参数（Ti-6Al-4V）
    rho = 4430.0       # kg/m³
    cp = 580.0         # J/(kg·K)
    k = thermal_diffusivity * rho * cp  # W/(m·K)

    delta_T = melting_temp - ambient_temp
    if delta_T < 1.0:
        delta_T = 1.0

    # scan_speed 从 mm/s 转换为 m/s
    v_ms = scan_speed / 1000.0

    # 无量纲 Peclet 数（特征长度 100 μm = 1e-4 m）
    L_char = 1e-4  # m
    Pe = v_ms * L_char / thermal_diffusivity

    # 熔池宽度估算 (m) — 基于 Rosenthal 线热源解析近似
    # w ≈ 2 * A*P / (π * e * k * ΔT * v)  *  修正因子
    width = (2.0 * absorptivity * laser_power) / (np.pi * np.e * k * delta_T * max(v_ms, 1e-6))
    # 添加下界保护
    width = max(width, 1e-6)
    depth = width / (2.0 + min(Pe, 100.0))
    length = width * (1.0 + 0.5 * min(Pe, 100.0))

    return {
        "width_m": width,
        "depth_m": depth,
        "length_m": length,
        "peclet": Pe,
        "thermal_diffusivity": thermal_diffusivity,
    }


# =============================================================================
# 5. 热应力简化模型
# =============================================================================

def thermal_strain_gradient(T_profile: np.ndarray, z: np.ndarray,
                             thermal_expansion: float = 9e-6) -> np.ndarray:
    """
    由温度梯度计算热应变：
        ε_th(z) = α_T · [T(z) - T_ref]

    以及热应变梯度（用于估算热应力）：
        σ_th ≈ E · α_T · ΔT / (1 - ν)
    """
    T_ref = T_profile[-1]  # 远端参考温度
    eps_th = thermal_expansion * (T_profile - T_ref)
    # 数值梯度
    d_eps_dz = np.gradient(eps_th, z)
    return d_eps_dz
