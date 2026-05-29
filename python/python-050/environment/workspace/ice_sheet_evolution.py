"""
ice_sheet_evolution.py
冰盖厚度演化与浅水近似 (Shallow Ice Approximation, SIA)

基于质量守恒方程:
    \frac{\partial H}{\partial t} = \dot{a} - \dot{m} - \nabla \cdot \mathbf{q}

其中 H(x,y,t) 为冰厚度，\dot{a} 为积累率，\dot{m} 为消融率，\mathbf{q} 为体积通量。

SIA 体积通量 (Hutter, 1983; Greve & Blatter, 2009):
    \mathbf{q} = -\frac{2A}{n+2} (\rho g)^n H^{n+2} |\nabla s|^{n-1} \nabla s

其中 s = b + H 为表面高程，b 为基岩高程。

扩散系数:
    D = \frac{2A}{n+2} (\rho g)^n H^{n+2} |\nabla s|^{n-1}

非线性扩散方程:
    \frac{\partial H}{\partial t} = \dot{a} - \dot{m} + \nabla \cdot (D \nabla s)

数值方法:
    - 显式 Euler 用于快速预估 (小时间步)
    - 自适应时间步长基于 CFL 条件
    - 正厚度保护 (H >= 0)
"""

import numpy as np
from typing import Optional

from ice_constitutive_model import (
    ICE_DENSITY, GRAVITY, GLEN_N, rate_factor_arrhenius
)


def compute_diffusivity_sia(H: np.ndarray,
                            surface: np.ndarray,
                            dx: float, dy: float,
                            temperature: float = 253.15) -> tuple:
    """
    计算 SIA 扩散系数 D 及通量分量。

    参数:
        H: 冰厚度场 (m), 形状 (ny, nx)
        surface: 表面高程 s = b + H (m), 形状 (ny, nx)
        dx, dy: 水平网格间距 (m)
        temperature: 特征温度 (K), 用于计算 A

    返回:
        D: 扩散系数场 (m^2 s^{-1})
        grad_s_x, grad_s_y: 表面梯度分量
    """
    H = np.asarray(H, dtype=np.float64)
    surface = np.asarray(surface, dtype=np.float64)

    if H.shape != surface.shape:
        raise ValueError("H and surface must have the same shape.")

    # 计算表面梯度 (中心差分)
    grad_s_x = np.zeros_like(surface)
    grad_s_y = np.zeros_like(surface)

    grad_s_x[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2.0 * dx)
    grad_s_y[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2.0 * dy)

    # 边界采用单侧差分
    grad_s_x[:, 0] = (surface[:, 1] - surface[:, 0]) / dx
    grad_s_x[:, -1] = (surface[:, -1] - surface[:, -2]) / dx
    grad_s_y[0, :] = (surface[1, :] - surface[0, :]) / dy
    grad_s_y[-1, :] = (surface[-1, :] - surface[-2, :]) / dy

    # 梯度模
    grad_mag = np.sqrt(grad_s_x ** 2 + grad_s_y ** 2)
    grad_mag = np.maximum(grad_mag, 1e-12)

    # TODO_HOLE_2: 实现 SIA 扩散系数计算
    # 科学知识点: D = (2A/(n+2)) * (rho*g)^n * H^(n+2) * |grad_s|^(n-1)
    # 需要调用 rate_factor_arrhenius 获取 A，并利用 GLEN_N, ICE_DENSITY, GRAVITY
    # 返回 D, grad_s_x, grad_s_y，注意数值保护
    raise NotImplementedError("Hole 2: 请实现 compute_diffusivity_sia 核心公式")


def explicit_euler_step_sia(H: np.ndarray,
                            bedrock: np.ndarray,
                            accumulation: np.ndarray,
                            dx: float, dy: float,
                            dt: float,
                            temperature: float = 253.15) -> np.ndarray:
    """
    显式 Euler 单步推进 SIA 冰厚度演化。

    离散格式:
        H_{i,j}^{n+1} = H_{i,j}^n + dt \cdot RHS_{i,j}

    其中 RHS = \dot{a} + \partial_x(D \partial_x s) + \partial_y(D \partial_y s)

    参数:
        H: 当前厚度 (ny, nx)
        bedrock: 基岩高程 (ny, nx)
        accumulation: 净积累率 \dot{a} - \dot{m} (m s^{-1}), (ny, nx)
        dx, dy: 网格间距
        dt: 时间步长 (s)
        temperature: 特征温度 (K)

    返回:
        H_new: 新厚度场 (ny, nx)
    """
    H = np.asarray(H, dtype=np.float64)
    bedrock = np.asarray(bedrock, dtype=np.float64)
    accumulation = np.asarray(accumulation, dtype=np.float64)

    if not (H.shape == bedrock.shape == accumulation.shape):
        raise ValueError("H, bedrock, and accumulation must have the same shape.")

    surface = bedrock + H
    D, gx, gy = compute_diffusivity_sia(H, surface, dx, dy, temperature)

    ny, nx = H.shape
    rhs = np.zeros_like(H)

    # 内部节点的散度 \nabla \cdot (D \nabla s)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            # x 方向通量
            D_e = 0.5 * (D[i, j] + D[i, j + 1])
            D_w = 0.5 * (D[i, j] + D[i, j - 1])
            flux_e = D_e * (surface[i, j + 1] - surface[i, j]) / dx
            flux_w = D_w * (surface[i, j] - surface[i, j - 1]) / dx

            # y 方向通量
            D_n = 0.5 * (D[i, j] + D[i + 1, j])
            D_s = 0.5 * (D[i, j] + D[i - 1, j])
            flux_n = D_n * (surface[i + 1, j] - surface[i, j]) / dy
            flux_s = D_s * (surface[i, j] - surface[i - 1, j]) / dy

            div_flux = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy
            rhs[i, j] = accumulation[i, j] + div_flux

    # 边界处理: 零通量 Neumann
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0
    rhs[:, 0] = 0.0
    rhs[:, -1] = 0.0

    H_new = H + dt * rhs
    H_new = np.maximum(H_new, 0.0)  # 正厚度保护

    return H_new


def adaptive_cfl_timestep_sia(H: np.ndarray,
                              bedrock: np.ndarray,
                              dx: float, dy: float,
                              temperature: float = 253.15,
                              cfl_safety: float = 0.25) -> float:
    """
    基于 CFL 条件的自适应时间步长。

    CFL 条件:
        dt \le \frac{0.5 \cdot \min(dx^2, dy^2)}{\max(D)}

    参数:
        H: 冰厚度场
        bedrock: 基岩高程
        dx, dy: 网格间距
        temperature: 特征温度
        cfl_safety: CFL 安全因子

    返回:
        dt_max: 最大允许时间步长 (s)
    """
    surface = bedrock + H
    D, _, _ = compute_diffusivity_sia(H, surface, dx, dy, temperature)
    D_max = np.max(D)

    if D_max < 1e-20:
        return 1e7  # 无扩散时允许大步长

    dt_max = cfl_safety * 0.5 * min(dx ** 2, dy ** 2) / D_max
    dt_max = max(dt_max, 1.0)   # 最小 1 秒
    dt_max = min(dt_max, 1e7)   # 最大 1e7 秒 (~4个月)
    return dt_max


def solve_sia_evolution(H0: np.ndarray,
                        bedrock: np.ndarray,
                        accumulation: np.ndarray,
                        dx: float, dy: float,
                        total_time: float,
                        temperature: float = 253.15,
                        output_interval: Optional[int] = None) -> tuple:
    """
    求解 SIA 冰盖厚度演化。

    参数:
        H0: 初始厚度 (ny, nx)
        bedrock: 基岩高程 (ny, nx)
        accumulation: 净积累率 (ny, nx) in m/s
        dx, dy: 网格间距 (m)
        total_time: 总模拟时间 (s)
        temperature: 特征温度 (K)
        output_interval: 输出间隔步数 (None 则仅输出最终状态)

    返回:
        H_final: 最终厚度
        history: 若 output_interval 给定，则为 (nt_out, ny, nx) 历史数组
    """
    H = H0.copy()
    t = 0.0

    history = []
    if output_interval is not None and output_interval > 0:
        history.append(H.copy())
    step = 0

    while t < total_time:
        dt = adaptive_cfl_timestep_sia(H, bedrock, dx, dy, temperature)
        if t + dt > total_time:
            dt = total_time - t

        H = explicit_euler_step_sia(H, bedrock, accumulation, dx, dy, dt, temperature)
        t += dt
        step += 1

        if output_interval is not None and step % output_interval == 0:
            history.append(H.copy())

    if history:
        return H, np.array(history)
    return H, None


def ice_volume(H: np.ndarray, dx: float, dy: float) -> float:
    """计算冰总体积 (m^3)."""
    return float(np.sum(H) * dx * dy)


def ice_area(H: np.ndarray, dx: float, dy: float, threshold: float = 1.0) -> float:
    """计算冰覆盖面积 (m^2)."""
    return float(np.sum(H > threshold) * dx * dy)
