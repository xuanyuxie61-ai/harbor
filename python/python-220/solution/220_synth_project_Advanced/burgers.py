"""
burgers.py — 粘性 Burgers 方程求解器
============================================
来源项目: 127_burgers_time_viscous

本模块实现含时粘性 Burgers 方程的有限差分数值求解:
  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

守恒形式:
  ∂u/∂t + ∂f/∂x = ν ∂²u/∂x²,  f(u) = u²/2

Burgers 方程是非线性对流-扩散方程的典型模型:
  - 当 ν → 0 时退化为无粘 Burgers 方程 (可产生激波)
  - 当 ν > 0 时粘性项平滑激波
  - 可通过 Hopf-Cole 变换精确求解

空间离散: 中心差分
  ∂f/∂x ≈ (f(u_{i+1}) - f(u_{i-1})) / (2dx)
  ∂²u/∂x² ≈ (u_{i+1} - 2u_i + u_{i-1}) / dx²

时间积分: 显式前向 Euler
  u_i^{n+1} = u_i^n - dt/(2dx) * (f(u_{i+1}^n) - f(u_{i-1}^n))
             + ν*dt/dx² * (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

稳定性条件 (CFL):
  dt ≤ dx²/(2ν)  (粘性限制)
  dt ≤ dx/max|u|  (对流限制)

边界条件:
  - Dirichlet: u(±L, t) = g(t)
  - Periodic: u(-L, t) = u(L, t)
  - Neumann: ∂u/∂x(±L, t) = 0

Hopf-Cole 精确解 (用于验证):
  u(x,t) = -2ν ∂/∂x ln(φ(x,t))
  其中 φ 满足热传导方程 ∂φ/∂t = ν ∂²φ/∂x²
"""

import numpy as np
from typing import Tuple, Optional, List
from config import BurgersConfig, EPS_NUM


# ============================================================
#  初始条件
# ============================================================
def burgers_initial_condition(x: np.ndarray, ic_type: str = "shock") -> np.ndarray:
    """Burgers 方程初始条件 (来源: 127)

    支持的初始条件类型:
      shock:    u(x,0) = -sign(x)      (激波衰减)
      gaussian: u(x,0) = exp(-x²/σ²)   (高斯脉冲)
      sine:     u(x,0) = sin(πx)        (正弦波)
      ramp:     u(x,0) = max(1-|x|, 0)  (三角脉冲)
      expansion: u(x,0) = -sign(x)      (膨胀波)
    """
    if ic_type == "shock":
        return -np.sign(x)
    elif ic_type == "gaussian":
        sigma = 0.5
        return np.exp(-x ** 2 / (2.0 * sigma ** 2))
    elif ic_type == "sine":
        return np.sin(PI * x)
    elif ic_type == "ramp":
        return np.maximum(1.0 - np.abs(x), 0.0)
    elif ic_type == "expansion":
        return np.sign(x)
    elif ic_type == "spike":
        sigma = 0.1
        return np.exp(-x ** 2 / (2.0 * sigma ** 2)) * 2.0
    else:
        raise ValueError(f"未知初始条件类型: {ic_type}")


# ============================================================
#  通量函数
# ============================================================
def burgers_flux(u: np.ndarray) -> np.ndarray:
    """Burgers 通量: f(u) = u²/2"""
    return 0.5 * u * u


# ============================================================
#  单步时间推进
# ============================================================
def burgers_step(u: np.ndarray, dt: float, dx: float, nu: float,
                 bc_type: str = "dirichlet") -> np.ndarray:
    """显式 Euler 单步推进 (来源: 127_burgers_time_viscous)

    内部点:
      u_i^{n+1} = u_i^n + dt * [ν(u_{i+1}-2u_i+u_{i-1})/dx²
                              - (f(u_{i+1})-f(u_{i-1}))/(2dx)]

    边界处理:
      Dirichlet: u_0, u_{N-1} 固定
      Periodic:  使用 np.roll
      Neumann:   虚拟节点法 (ghost point)
    """
    nx = len(u)
    u_new = np.zeros(nx)

    if bc_type == "periodic":
        # 周期性边界
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        flux_left = burgers_flux(np.roll(u, 1))
        flux_right = burgers_flux(np.roll(u, -1))

        u_new = u + dt * (
            nu * (u_right - 2.0 * u + u_left) / dx**2
            - (flux_right - flux_left) / (2.0 * dx)
        )
    else:
        # 内部点更新
        for i in range(1, nx - 1):
            diffusion = nu * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / dx**2
            advection = (burgers_flux(u[i + 1]) - burgers_flux(u[i - 1])) / (2.0 * dx)
            u_new[i] = u[i] + dt * (diffusion - advection)

        if bc_type == "dirichlet":
            u_new[0] = u[0]
            u_new[-1] = u[-1]
        elif bc_type == "neumann":
            # ∂u/∂x = 0 → ghost point: u_{-1} = u_1
            u_new[0] = u[0] + dt * nu * (2.0 * u[1] - 2.0 * u[0]) / dx**2
            u_new[-1] = u[-1] + dt * nu * (2.0 * u[-2] - 2.0 * u[-1]) / dx**2

    return u_new


# ============================================================
#  完整模拟
# ============================================================
def run_burgers_simulation(config: BurgersConfig,
                           n_snapshots: int = 10) -> dict:
    """运行完整的 Burgers 方程模拟

    Returns:
        dict: x, U_snapshots, times, energy_history
    """
    nx = config.nx
    x = np.linspace(-config.L, config.L, nx)
    dx = config.dx

    u = burgers_initial_condition(x, config.ic_type)
    dt = 0.8 * config.dt_max
    n_steps = max(1, int(config.T_final / dt))
    dt = config.T_final / n_steps

    snapshot_interval = max(1, n_steps // n_snapshots)

    U_snaps = [u.copy()]
    times = [0.0]
    energy_hist = []

    for step in range(1, n_steps + 1):
        u = burgers_step(u, dt, dx, config.nu, config.bc_type)

        # L2 能量: E = ∫ u² dx (应单调递减)
        energy = np.sum(u ** 2) * dx
        energy_hist.append(energy)

        if step % snapshot_interval == 0 or step == n_steps:
            U_snaps.append(u.copy())
            times.append(step * dt)

    return {
        'x': x,
        'U_snapshots': U_snaps,
        'times': times,
        'energy_history': energy_hist,
        'final_u': u,
    }


# ============================================================
#  Hopf-Cole 精确解 (用于验证)
# ============================================================
def hopf_cole_solution(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """Burgers 方程的 Hopf-Cole 精确解

    初始条件: u(x,0) = -tanh(x/(4ν)) (N-wave 衰减)

    精确解:
      u(x,t) = -tanh(x/(4ν)) / (1 + sqrt(1 + t/t_0) * exp(x²/(4ν(t+t_0))))
      其中 t_0 = 1/(2ν) 为特征时间.

    简化形式 (用于 shock 初值):
      u(x,t) = -erf(x / (2*sqrt(2*nu*t))) / (1 + exp(x²/(2*nu)) * ...)
    """
    if t < EPS_NUM:
        return -np.sign(x)

    # Cole-Hopf 变换数值求解
    # φ(x,t) = ∫ exp(-∫_0^ξ u(s,0) ds / (2ν)) * G(x-ξ, t) dξ
    # 其中 G 为热核
    sigma = np.sqrt(2.0 * nu * t)
    # 对于 shock 初值 u(x,0) = -sign(x):
    #   ∫_0^x u(s,0)/(2ν) ds = -|x|/(2ν)
    #   φ = exp(-|x|/(2ν)) * G 的卷积
    u_exact = -np.tanh(x / (4.0 * nu + EPS_NUM)) * np.exp(-x**2 / (4.0 * nu * (t + 1.0/(2.0*nu))))
    # 简化近似 (仅用于定性验证)
    u_exact = -erf_approx(x / (2.0 * sigma + EPS_NUM))
    return u_exact


def erf_approx(z: np.ndarray) -> np.ndarray:
    """误差函数近似 (Abramowitz & Stegun)
    erf(z) ≈ 1 - (a1*t + a2*t² + a3*t³) * exp(-z²)
    其中 t = 1/(1 + 0.47047*|z|)
    """
    sign = np.sign(z)
    z_abs = np.abs(z)
    t = 1.0 / (1.0 + 0.47047 * z_abs)
    a1, a2, a3 = 0.3480242, -0.0958798, 0.7478556
    result = 1.0 - (a1 * t + a2 * t**2 + a3 * t**3) * np.exp(-z_abs**2)
    return sign * result


PI = np.pi
