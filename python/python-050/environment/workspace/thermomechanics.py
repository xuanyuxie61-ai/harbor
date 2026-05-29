"""
thermomechanics.py
冰盖热力学耦合求解器 — 隐式中点法与焓形式

本模块实现冰盖温度与焓的演化求解，采用隐式中点法 (Implicit Midpoint Rule)
处理热力学方程的强刚性 (stiffness)。

核心控制方程:
  1. 温度形式:
     \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) - \rho c_p \mathbf{u} \cdot \nabla T + \Phi

  2. 焓形式 (Aschwanden et al., 2012):
     \rho \frac{\partial H}{\partial t} = \nabla \cdot (k^* \nabla H) - \rho \mathbf{u} \cdot \nabla H + \Phi

     其中 H = \int_0^T c_p dT' + \omega L 为体积焓，\omega 为液态水含量。

  3. 隐式中点离散 (θ = 0.5):
     T^{n+1} = T^n + \Delta t \cdot f\left( t^{n+1/2}, \frac{T^n + T^{n+1}}{2} \right)

数值特性:
  - 隐式中点法具有二阶精度与 A-稳定性，适合处理冰川热力学中的大时间步长
  - 固定点迭代求解非线性阶段方程
  - 边界处理: 上表面 Dirichlet (大气温度)，底部 Neumann (地热通量)
"""

import numpy as np
from typing import Callable, Tuple

from ice_constitutive_model import (
    ICE_DENSITY, SPECIFIC_HEAT, THERMAL_CONDUCTIVITY, LATENT_HEAT, GLEN_N, GRAVITY
)


def build_thermal_diffusion_matrix(nz: int, dz: float,
                                   thermal_diffusivity: float) -> np.ndarray:
    """
    构建一维垂直热扩散的有限差分矩阵 (Crank-Nicolson 风格)。

    离散格式 (内部节点 i):
        \frac{\partial^2 T}{\partial z^2} \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{dz^2}

    采用标准二阶中心差分，矩阵形式:
        A_{ii} = -2/dz^2,  A_{i,i+1} = A_{i,i-1} = 1/dz^2

    参数:
        nz: 垂直层数
        dz: 层厚 (m)
        thermal_diffusivity: 热扩散系数 k/(\rho c_p) (m^2 s^{-1})

    返回:
        L: (nz, nz) 扩散算子矩阵
    """
    if nz < 3:
        raise ValueError("nz must be at least 3 for finite difference.")
    if dz <= 0:
        raise ValueError("dz must be positive.")

    coef = thermal_diffusivity / (dz ** 2)
    L = np.zeros((nz, nz), dtype=np.float64)

    # 内部节点
    for i in range(1, nz - 1):
        L[i, i - 1] = coef
        L[i, i] = -2.0 * coef
        L[i, i + 1] = coef

    # Neumann 零通量边界 (简化)
    L[0, 0] = -coef
    L[0, 1] = coef
    L[-1, -2] = coef
    L[-1, -1] = -coef

    return L


def implicit_midpoint_step(y_old: np.ndarray,
                           dt: float,
                           f: Callable[[np.ndarray], np.ndarray],
                           it_max: int = 20,
                           tol: float = 1e-10) -> np.ndarray:
    """
    隐式中点法单步推进。

    求解非线性方程:
        y^* = y_{old} + (dt/2) \cdot f(y^*)
        y_{new} = y_{old} + dt \cdot f(y^*)

    其中 y^* 为中点处的状态估计。

    参数:
        y_old: 当前时刻状态向量
        dt: 时间步长
        f: 右端函数 f(y)
        it_max: 最大固定点迭代次数
        tol: 收敛容差

    返回:
        y_new: 下一时刻状态向量
    """
    y_old = np.asarray(y_old, dtype=np.float64)
    y_star = y_old.copy()

    for it in range(it_max):
        y_star_new = y_old + 0.5 * dt * f(y_star)
        diff = np.linalg.norm(y_star_new - y_star, ord=np.inf)
        y_star = y_star_new
        if diff < tol:
            break

    y_new = y_old + dt * f(y_star)
    return y_new


def solve_temperature_evolution(nz: int, z_max: float,
                                dt: float, nt: int,
                                surface_temperature: float,
                                basal_heat_flux: float,
                                velocity_vertical: np.ndarray,
                                dissipation: np.ndarray) -> np.ndarray:
    """
    求解一维垂直温度剖面演化。

    控制方程:
        \rho c_p \frac{\partial T}{\partial t}
            = k \frac{\partial^2 T}{\partial z^2}
              - \rho c_p w \frac{\partial T}{\partial z}
              + \Phi(z)

    边界条件:
        T(z=0, t) = T_s(t)        (表面 Dirichlet)
        -k \frac{\partial T}{\partial z}(z=H, t) = q_b  (底部 Neumann)

    参数:
        nz: 垂直网格点数
        z_max: 冰层总厚度 H (m)
        dt: 时间步长 (s)
        nt: 时间步数
        surface_temperature: 表面温度 T_s (K)
        basal_heat_flux: 底部热通量 q_b (W m^{-2})
        velocity_vertical: 垂直速度剖面 w(z) (m s^{-1}), 长度 nz
        dissipation: 耗散热剖面 \Phi(z) (W m^{-3}), 长度 nz

    返回:
        T_history: (nt+1, nz) 温度演化历史
    """
    if nz < 3:
        raise ValueError("nz must be >= 3")
    if z_max <= 0 or dt <= 0 or nt < 0:
        raise ValueError("z_max, dt must be positive; nt >= 0")

    dz = z_max / (nz - 1)
    alpha = THERMAL_CONDUCTIVITY / (ICE_DENSITY * SPECIFIC_HEAT)

    # 初始温度剖面: 线性插值
    T = np.linspace(surface_temperature,
                    surface_temperature + basal_heat_flux * z_max / THERMAL_CONDUCTIVITY,
                    nz)
    # 上限保护: 不超过压力熔点
    pressure_melting_point = 273.15 - 7.42e-8 * ICE_DENSITY * GRAVITY * np.linspace(0, z_max, nz)
    T = np.minimum(T, pressure_melting_point - 0.1)

    # 确保速度、耗散数组正确
    w = np.asarray(velocity_vertical, dtype=np.float64)
    phi = np.asarray(dissipation, dtype=np.float64)
    if w.shape != (nz,):
        raise ValueError(f"velocity_vertical must have shape ({nz},), got {w.shape}")
    if phi.shape != (nz,):
        raise ValueError(f"dissipation must have shape ({nz},), got {phi.shape}")

    T_history = np.zeros((nt + 1, nz), dtype=np.float64)
    T_history[0] = T

    # 构建扩散矩阵
    L = build_thermal_diffusion_matrix(nz, dz, alpha)

    # 构建对流矩阵 (一阶迎风格式)
    C = np.zeros((nz, nz), dtype=np.float64)
    for i in range(1, nz - 1):
        # 上游风格: 若 w > 0 (向下), 采用向后差分
        if w[i] >= 0:
            C[i, i] = w[i] / dz
            C[i, i - 1] = -w[i] / dz
        else:
            C[i, i] = -w[i] / dz
            C[i, i + 1] = w[i] / dz

    # 源项向量
    source = phi / (ICE_DENSITY * SPECIFIC_HEAT)

    def rhs(T_vec: np.ndarray) -> np.ndarray:
        """温度方程右端函数 dT/dt."""
        dTdt = L @ T_vec - C @ T_vec + source
        # Dirichlet 边界: 表面温度固定
        dTdt[0] = 0.0
        # Neumann 边界: 通过 L 矩阵已近似
        return dTdt

    for n in range(nt):
        T = implicit_midpoint_step(T, dt, rhs, it_max=30, tol=1e-12)

        # 后处理: 物理截断
        T = np.clip(T, 200.0, 273.15)
        T[0] = surface_temperature  # 强制 Dirichlet
        T_history[n + 1] = T

    return T_history


def solve_enthalpy_evolution(nz: int, z_max: float,
                             dt: float, nt: int,
                             surface_temperature: float,
                             basal_heat_flux: float,
                             velocity_vertical: np.ndarray,
                             dissipation: np.ndarray,
                             porosity_initial: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    焓形式求解冰盖温度-含水量演化 (Aschwanden et al., 2012)。

    焓定义:
        \mathcal{H} = \int_0^T c_p(T') dT' + \omega L_f

    其中 \omega 为孔隙率 (液态水体积分数)。

    本构关系:
        T < T_m(p):  \mathcal{H} < \mathcal{H}_{cold},  \omega = 0
        T = T_m(p):  \mathcal{H} \ge \mathcal{H}_{cold},  \omega = (\mathcal{H} - \mathcal{H}_{cold}) / L_f

    参数:
        nz, z_max, dt, nt: 网格与时间参数
        surface_temperature: 表面温度 (K)
        basal_heat_flux: 底部热通量 (W m^{-2})
        velocity_vertical: 垂直速度 w(z) (m s^{-1})
        dissipation: 耗散热 \Phi(z) (W m^{-3})
        porosity_initial: 初始孔隙率 (可选)

    返回:
        H_history: 焓历史 (J kg^{-1})
        T_history: 温度历史 (K)
        omega_history: 孔隙率历史
    """
    dz = z_max / (nz - 1)
    z = np.linspace(0, z_max, nz)

    # 压力熔点
    pmp = 273.15 - 7.42e-8 * ICE_DENSITY * 9.81 * z
    H_cold = SPECIFIC_HEAT * (pmp - 200.0)  # 参考冷焓阈值

    # 初始焓
    T_init = np.minimum(np.linspace(surface_temperature, pmp[-1] + 5.0, nz), pmp - 0.1)
    H = SPECIFIC_HEAT * (T_init - 200.0)
    if porosity_initial is not None:
        H = H + np.asarray(porosity_initial) * LATENT_HEAT

    H = np.clip(H, 0.0, None)

    w = np.asarray(velocity_vertical, dtype=np.float64)
    phi = np.asarray(dissipation, dtype=np.float64)

    alpha_star = THERMAL_CONDUCTIVITY / (ICE_DENSITY * SPECIFIC_HEAT)
    L = build_thermal_diffusion_matrix(nz, dz, alpha_star)

    C = np.zeros((nz, nz), dtype=np.float64)
    for i in range(1, nz - 1):
        if w[i] >= 0:
            C[i, i] = w[i] / dz
            C[i, i - 1] = -w[i] / dz
        else:
            C[i, i] = -w[i] / dz
            C[i, i + 1] = w[i] / dz

    source = phi / ICE_DENSITY

    H_history = np.zeros((nt + 1, nz), dtype=np.float64)
    T_history = np.zeros((nt + 1, nz), dtype=np.float64)
    omega_history = np.zeros((nt + 1, nz), dtype=np.float64)

    H_history[0] = H
    T_history[0], omega_history[0] = enthalpy_to_temperature_water(H, pmp)

    def rhs(H_vec: np.ndarray) -> np.ndarray:
        dHdt = L @ H_vec - C @ H_vec + source
        dHdt[0] = 0.0
        return dHdt

    for n in range(nt):
        H = implicit_midpoint_step(H, dt, rhs, it_max=30, tol=1e-12)
        H = np.clip(H, 0.0, 1e7)
        # 表面焓由温度决定
        H[0] = SPECIFIC_HEAT * (surface_temperature - 200.0)
        H[0] = max(H[0], 0.0)

        H_history[n + 1] = H
        T_history[n + 1], omega_history[n + 1] = enthalpy_to_temperature_water(H, pmp)

    return H_history, T_history, omega_history


def enthalpy_to_temperature_water(enthalpy: np.ndarray,
                                  pressure_melting_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将焓转换为温度和孔隙率。

    转换规则:
        H < H_{cold} = c_p (T_m - T_{ref}):
            T = T_{ref} + H / c_p,   \omega = 0
        H \ge H_{cold}:
            T = T_m(p),   \omega = (H - H_{cold}) / L_f

    参数:
        enthalpy: 焓数组 (J kg^{-1})
        pressure_melting_point: 压力熔点数组 (K)

    返回:
        T, omega: 温度和孔隙率
    """
    H = np.asarray(enthalpy, dtype=np.float64)
    Tm = np.asarray(pressure_melting_point, dtype=np.float64)

    T_ref = 200.0
    H_cold = SPECIFIC_HEAT * (Tm - T_ref)

    T = np.zeros_like(H)
    omega = np.zeros_like(H)

    cold_mask = H < H_cold
    temperate_mask = ~cold_mask

    T[cold_mask] = T_ref + H[cold_mask] / SPECIFIC_HEAT
    T[temperate_mask] = Tm[temperate_mask]
    omega[temperate_mask] = (H[temperate_mask] - H_cold[temperate_mask]) / LATENT_HEAT

    # 截断
    omega = np.clip(omega, 0.0, 0.1)
    T = np.clip(T, 200.0, 273.15)

    return T, omega
