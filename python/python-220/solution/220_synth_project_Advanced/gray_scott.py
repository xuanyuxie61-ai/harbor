"""
gray_scott.py — Gray-Scott 反应扩散系统
============================================
来源项目: 487_gray_scott_pde

本模块实现 2D Gray-Scott 自催化反应扩散系统的时空演化.
该系统是 Turing 模式形成的经典模型:
  U + 2V → 3V (自催化反应)

控制方程:
  ∂U/∂t = Du * ∇²U - U*V² + γ*(1 - U)
  ∂V/∂t = Dv * ∇²V + U*V² - (γ + κ)*V

其中:
  Du, Dv: 扩散系数 (Dv < Du 是 Turing 不稳定性的必要条件)
  γ: 供给率 (feed rate)
  κ: 杀灭率 (kill rate)

空间离散: 9 点 Laplacian 模板 (高精度)
  ∇²u ≈ [1*u_{i-1,j-1} + 4*u_{i-1,j} + 1*u_{i-1,j+1}
         + 4*u_{i,j-1}   - 20*u_{i,j}  + 4*u_{i,j+1}
         + 1*u_{i+1,j-1} + 4*u_{i+1,j} + 1*u_{i+1,j+1}] / (6*dx²)

时间积分: 显式 Euler (满足 CFL 条件)
  U^{n+1} = U^n + dt * (Du * Laplacian(U^n) - U^n*(V^n)² + γ*(1 - U^n))
  V^{n+1} = V^n + dt * (Dv * Laplacian(V^n) + U^n*(V^n)² - (γ+κ)*V^n)

边界条件: 周期性 (环面拓扑)

守恒量 (离散):
  Σ(U + V) * dx * dy ≈ const (在无源项极限下)
"""

import numpy as np
from typing import Tuple, Optional
from config import GrayScottConfig, EPS_NUM


# ============================================================
#  9 点 Laplacian 算子 (周期性边界)
# ============================================================
def laplacian_9pt(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """9 点 Laplacian 模板 (来源: 487_gray_scott_pde)

    对于非均匀网格 (dx ≠ dy):
    L[i,j] = (1/(6*dx*dy)) * [
        (dy/dx)*A[i-1,j-1] + 4*(dy/dx)*A[i-1,j] + (dy/dx)*A[i-1,j+1]
      + 4*(dx/dy)*A[i,j-1] - 20*(dx/dy + dy/dx)*A[i,j] + 4*(dx/dy)*A[i,j+1]
      + (dy/dx)*A[i+1,j-1] + 4*(dy/dx)*A[i+1,j] + (dy/dx)*A[i+1,j+1]
    ]

    当 dx = dy 时简化为:
    L[i,j] = [1*A[i-1,j-1] + 4*A[i-1,j] + 1*A[i-1,j+1]
            + 4*A[i,j-1] - 20*A[i,j] + 4*A[i,j+1]
            + 1*A[i+1,j-1] + 4*A[i+1,j] + 1*A[i+1,j+1]] / (6*dx²)
    """
    # 周期性滚动
    ip1 = np.roll(A, -1, axis=0)  # i+1
    im1 = np.roll(A, 1, axis=0)   # i-1
    jp1 = np.roll(A, -1, axis=1)  # j+1
    jm1 = np.roll(A, 1, axis=1)   # j-1

    # 9 点模板
    if abs(dx - dy) < EPS_NUM:
        dx2 = dx * dy
        lap = (
            1.0 * np.roll(im1, 1, axis=1) + 4.0 * im1 + 1.0 * np.roll(im1, -1, axis=1)
            + 4.0 * np.roll(A, 1, axis=1) - 20.0 * A + 4.0 * jp1
            + 1.0 * np.roll(ip1, 1, axis=1) + 4.0 * ip1 + 1.0 * np.roll(ip1, -1, axis=1)
        ) / (6.0 * dx2)
    else:
        rx = dy / dx  # dy/dx
        ry = dx / dy  # dx/dy
        lap = (
            rx * np.roll(im1, 1, axis=1) + 4.0 * rx * im1 + rx * np.roll(im1, -1, axis=1)
            + 4.0 * ry * np.roll(A, 1, axis=1) - 20.0 * (rx + ry) * A + 4.0 * ry * jp1
            + rx * np.roll(ip1, 1, axis=1) + 4.0 * rx * ip1 + rx * np.roll(ip1, -1, axis=1)
        ) / (6.0 * dx * dy)
    return lap


# ============================================================
#  Gray-Scott 右端项
# ============================================================
def gray_scott_rhs(U: np.ndarray, V: np.ndarray, config: GrayScottConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Gray-Scott 系统右端项

    dU/dt = Du * ∇²U - U*V² + γ*(1 - U)
    dV/dt = Dv * ∇²V + U*V² - (γ + κ)*V

    Args:
        U: 物种 U 浓度场 (nx, ny)
        V: 物种 V 浓度场 (nx, ny)
        config: Gray-Scott 参数

    Returns:
        (dUdt, dVdt): 时间导数
    """
    dx, dy = config.dx, config.dy

    # 扩散项
    lap_U = laplacian_9pt(U, dx, dy)
    lap_V = laplacian_9pt(V, dx, dy)

    # 反应项: U + 2V → 3V
    UV2 = U * V * V  # 自催化反应速率

    # 右端项
    dUdt = config.Du * lap_U - UV2 + config.gamma * (1.0 - U)
    dVdt = config.Dv * lap_V + UV2 - (config.gamma + config.kappa) * V

    return dUdt, dVdt


# ============================================================
#  时间推进
# ============================================================
def gray_scott_step(U: np.ndarray, V: np.ndarray, dt: float,
                    config: GrayScottConfig) -> Tuple[np.ndarray, np.ndarray]:
    """单步显式 Euler 时间推进

    U^{n+1} = U^n + dt * F_U(U^n, V^n)
    V^{n+1} = V^n + dt * F_V(U^n, V^n)

    质量守恒检查:
      Σ(U + V) 应近似守恒 (在无 γ,κ 极限下)
    """
    dUdt, dVdt = gray_scott_rhs(U, V, config)

    U_new = U + dt * dUdt
    V_new = V + dt * dVdt

    # 物理约束: 浓度非负
    U_new = np.maximum(U_new, 0.0)
    V_new = np.maximum(V_new, 0.0)

    # 上界约束 (防止数值爆炸)
    U_new = np.minimum(U_new, 2.0)
    V_new = np.minimum(V_new, 2.0)

    return U_new, V_new


# ============================================================
#  初始化
# ============================================================
def gray_scott_initial_condition(config: GrayScottConfig,
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """初始化 Gray-Scott 系统

    均匀态 (U, V) = (1, 0) + 中心区域小扰动
    扰动使系统偏离不稳定均匀态, 触发 Turing 模式.

    初始条件:
      U(x,y,0) = 1 - 0.5*perturbation * exp(-((x-cx)² + (y-cy)²)/σ²)
      V(x,y,0) = 0.25*perturbation * exp(-((x-cx)² + (y-cy)²)/σ²)

    其中 (cx, cy) 为中心, σ 为扰动宽度.
    """
    rng = np.random.RandomState(seed)
    nx, ny = config.nx, config.ny
    Lx, Ly = config.Lx, config.Ly

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 均匀初态
    U = np.ones((nx, ny)) * config.initial_U
    V = np.ones((nx, ny)) * config.initial_V

    # 中心高斯扰动 (多个随机扰动点)
    cx, cy = Lx / 2.0, Ly / 2.0
    sigma = min(Lx, Ly) / 8.0
    r2 = (X - cx) ** 2 + (Y - cy) ** 2

    V += config.perturbation * np.exp(-r2 / (2.0 * sigma ** 2))
    U -= 0.5 * config.perturbation * np.exp(-r2 / (2.0 * sigma ** 2))

    # 附加小随机扰动 (打破对称性)
    V += config.perturbation * 0.1 * rng.randn(nx, ny)
    V = np.maximum(V, 0.0)

    return U, V


# ============================================================
#  完整模拟
# ============================================================
def run_gray_scott_simulation(config: GrayScottConfig,
                              n_snapshots: int = 5,
                              seed: int = 42) -> dict:
    """运行完整的 Gray-Scott 模拟

    Args:
        config: 参数配置
        n_snapshots: 快照数量
        seed: 随机种子

    Returns:
        dict: 包含 'U_snapshots', 'V_snapshots', 'times', 'mass_history'
    """
    U, V = gray_scott_initial_condition(config, seed)
    dt = 0.8 * config.dt_max  # 安全系数
    n_steps = max(1, int(config.T_final / dt))
    dt = config.T_final / n_steps  # 精确调整

    snapshot_interval = max(1, n_steps // n_snapshots)

    U_snaps = [U.copy()]
    V_snaps = [V.copy()]
    times = [0.0]
    mass_history = []

    for step in range(1, n_steps + 1):
        U, V = gray_scott_step(U, V, dt, config)

        # 守恒量监测
        total_mass = np.sum(U + V) * config.dx * config.dy
        mass_history.append(total_mass)

        if step % snapshot_interval == 0 or step == n_steps:
            U_snaps.append(U.copy())
            V_snaps.append(V.copy())
            times.append(step * dt)

    return {
        'U_snapshots': U_snaps,
        'V_snapshots': V_snaps,
        'times': times,
        'mass_history': mass_history,
        'final_U': U,
        'final_V': V,
    }


# ============================================================
#  反问题: 参数估计目标函数
# ============================================================
def gray_scott_data_misfit(params: np.ndarray, observations: np.ndarray,
                           base_config: GrayScottConfig,
                           obs_times: np.ndarray,
                           obs_locations: np.ndarray) -> float:
    """Gray-Scott 参数反问题的数据失配泛函

    J(θ) = (1/2) * Σ_k ||U(x_k, t_k; θ) - U_obs_k||²

    其中 θ = [Du, Dv, γ, κ] 为待估参数.

    Args:
        params: [Du, Dv, gamma, kappa]
        observations: 观测数据列表
        base_config: 基础配置
        obs_times: 观测时间
        obs_locations: 观测位置

    Returns:
        数据失配值
    """
    config = GrayScottConfig(
        Du=max(params[0], EPS_NUM),
        Dv=max(params[1], EPS_NUM),
        gamma=max(params[2], 0.0),
        kappa=max(params[3], 0.0),
        nx=base_config.nx, ny=base_config.ny,
        Lx=base_config.Lx, Ly=base_config.Ly,
        T_final=obs_times[-1] if len(obs_times) > 0 else base_config.T_final,
    )
    result = run_gray_scott_simulation(config, n_snapshots=len(obs_times))

    misfit = 0.0
    for k, t_obs in enumerate(obs_times):
        if k < len(result['U_snapshots']):
            U_snap = result['U_snapshots'][k]
            for loc in obs_locations:
                ix = int(loc[0] / config.dx) % config.nx
                iy = int(loc[1] / config.dy) % config.ny
                misfit += 0.5 * (U_snap[ix, iy] - observations[k, ix, iy]) ** 2

    return misfit
