"""
vessel_mechanics.py
血管壁弹性力学与红细胞多体相互作用

融合来源:
- 861_pendulum_nonlinear_ode: 非线性摆精确解（Jacobi椭圆函数）、能量守恒、周期公式
- 345_exm (orbits): N体引力模拟的多体动力学框架

科学背景:
1. 血管壁弹性振动:
   动脉壁由三层结构组成（内膜、中膜、外膜），具有粘弹性。
   在脉动压力下，血管壁经历径向振动，可用非线性振荡器建模。
   小变形时退化为简谐振子；大变形时需考虑非线性恢复力。

2. 红细胞相互作用:
   血液中红细胞（RBC）体积分数约40-45%。RBC之间的相互作用
   通过血浆介导，可近似为多体问题中的短程排斥+长程吸引。
   这种相互作用影响血液的表观粘度和WSS分布。
"""

import numpy as np
from scipy.special import ellipj, ellipk
from typing import Tuple


# ======================================================================
# 来自 861_pendulum_nonlinear_ode 的非线性摆模型
# ======================================================================

class VesselElasticPendulum:
    """
    将血管壁径向振动建模为非线性摆。

    类比映射:
        角位移 θ      ↔  径向位移 ξ = (R - R_0) / R_0
        重力加速度 g  ↔  弹性恢复系数 k_e = E h / (ρ_w R_0²)
        摆长 l        ↔  特征长度 L = R_0
        质量 m        ↔  单位长度壁面质量 m_w

    运动方程:
        ξ'' + (k_e) sin(ξ) = f_ext(t) / m_w

    其中f_ext为脉动压力引起的等效外力。
    """
    def __init__(self, equilibrium_radius: float = 0.005,
                 elastic_modulus_pa: float = 1.0e6,
                 wall_thickness_m: float = 1.0e-3,
                 wall_density_kg_m3: float = 1050.0):
        """
        参数:
            equilibrium_radius: 平衡状态半径 R_0 [m]
            elastic_modulus_pa: 壁面杨氏模量 E [Pa]
            wall_thickness_m: 壁厚 h [m]
            wall_density_kg_m3: 壁面密度 ρ_w [kg/m³]
        """
        self.R0 = equilibrium_radius
        self.E = elastic_modulus_pa
        self.h = wall_thickness_m
        self.rho_w = wall_density_kg_m3
        self.g_eff = self.E * self.h / (self.rho_w * self.R0 ** 2)
        self.mass = 2.0 * np.pi * self.R0 * self.h * self.rho_w  # 单位长度质量

    def deriv(self, t: float, y: np.ndarray, external_force: float = 0.0) -> np.ndarray:
        """
        状态方程右端项。

        状态向量 y = [ξ, ξ']:
            y1' = y2
            y2' = -(g_eff) sin(y1) + f_ext / mass

        参数:
            t: 时间（未显式使用，保持接口一致）
            y: [位移, 速度]
            external_force: 外力 [N/m]
        """
        y = np.asarray(y, dtype=float)
        if len(y) != 2:
            raise ValueError("State vector must have length 2")
        xi = y[0]
        xi_dot = y[1]
        dydt = np.zeros(2)
        dydt[0] = xi_dot
        dydt[1] = -self.g_eff * np.sin(xi) + external_force / (self.mass + 1e-15)
        return dydt

    def exact_solution(self, t: np.ndarray, xi0: float) -> np.ndarray:
        """
        当初始速度为零时的精确解，使用Jacobi椭圆函数。

        精确解（类比非线性摆初始角位移ξ0、零初始速度）:
            ξ(t) = 2 arcsin( sin(ξ0/2) · sn( K(m) - ω_0 t, m ) )

        其中:
            m = sin²(ξ0/2)  （椭圆模数）
            ω_0 = sqrt(g_eff)
            K(m) = 第一类完全椭圆积分
            sn(u, m) = Jacobi椭圆正弦函数

        参数:
            t: 时间数组
            xi0: 初始径向位移

        返回:
            xi: 位移数组
        """
        t = np.atleast_1d(t)
        if abs(xi0) < 1e-12:
            return np.zeros_like(t)

        m = np.sin(xi0 / 2.0) ** 2
        m = np.clip(m, 0.0, 1.0 - 1e-15)
        omega0 = np.sqrt(self.g_eff)
        K = ellipk(m)

        u = K - omega0 * t
        sn, cn, dn, ph = ellipj(u, m)
        xi = 2.0 * np.arcsin(np.clip(np.sin(xi0 / 2.0) * sn, -1.0, 1.0))
        return xi

    def period(self, xi0: float) -> float:
        """
        计算非线性振荡周期。

        公式:
            T = 4 / ω_0 · K(m),  m = sin²(ξ0/2)

        小振幅极限（ξ0 → 0）: T → 2π / ω_0（简谐振子周期）
        """
        if abs(xi0) < 1e-12:
            return 2.0 * np.pi / np.sqrt(self.g_eff)
        m = np.sin(xi0 / 2.0) ** 2
        m = np.clip(m, 0.0, 1.0 - 1e-15)
        K = ellipk(m)
        return 4.0 * K / np.sqrt(self.g_eff)

    def energy(self, xi: float, xi_dot: float) -> float:
        """
        计算总机械能（类比非线性摆能量）。

        公式:
            H = (m_w/2) ξ'² + m_w g_eff (1 - cos(ξ))

        第一项为动能，第二项为弹性势能。
        """
        kinetic = 0.5 * self.mass * xi_dot ** 2
        potential = self.mass * self.g_eff * (1.0 - np.cos(xi))
        return kinetic + potential


def simulate_vessel_oscillation(pendulum: VesselElasticPendulum,
                                t_span: np.ndarray,
                                xi0: float,
                                external_pressure_pa: np.ndarray = None) -> dict:
    """
    模拟血管壁在脉动压力下的振动响应。

    参数:
        pendulum: VesselElasticPendulum实例
        t_span: 时间数组
        xi0: 初始位移
        external_pressure_pa: 可选的外加压强数组（与t_span同长）

    返回:
        包含位移、速度、能量的字典
    """
    n = len(t_span)
    xi = np.zeros(n)
    xi_dot = np.zeros(n)
    energy = np.zeros(n)

    xi[0] = xi0
    xi_dot[0] = 0.0
    energy[0] = pendulum.energy(xi0, 0.0)

    dt = t_span[1] - t_span[0] if n > 1 else 1e-3

    for i in range(1, n):
        f_ext = 0.0
        if external_pressure_pa is not None:
            # 压强转化为线力: F = P * 2πR_0
            f_ext = external_pressure_pa[i] * 2.0 * np.pi * pendulum.R0

        # 半隐式Euler积分（保持能量近似守恒）
        y = np.array([xi[i - 1], xi_dot[i - 1]])
        dydt = pendulum.deriv(t_span[i - 1], y, f_ext)

        xi_dot[i] = xi_dot[i - 1] + dt * dydt[1]
        xi[i] = xi[i - 1] + dt * xi_dot[i]

        # 边界处理：位移不超过物理极限
        xi[i] = np.clip(xi[i], -0.5, 0.5)

        energy[i] = pendulum.energy(xi[i], xi_dot[i])

    return {
        "displacement": xi,
        "velocity": xi_dot,
        "energy": energy,
        "radius": pendulum.R0 * (1.0 + xi)
    }


# ======================================================================
# 来自 345_exm (orbits) 的多体相互作用模型（用于红细胞动力学）
# ======================================================================

def rbc_interaction_force(positions: np.ndarray,
                          radii: float = 2.5e-6,
                          repulsion_strength: float = 1e-12,
                          attraction_strength: float = 1e-14,
                          attraction_range: float = 1e-5) -> np.ndarray:
    """
    计算红细胞之间的等效相互作用力。

    力场模型（简化Lennard-Jones型）:
        F_rep =  k_rep / r^6   （短程排斥，防止重叠）
        F_att = -k_att / r^3   （长程弱吸引，模拟血浆桥联）
        F_total = F_rep + F_att  (当 r < r_cutoff)

    参数:
        positions: (N, 2) 或 (N, 3) 粒子位置 [m]
        radii: 红细胞等效半径 [m]
        repulsion_strength: 排斥力强度
        attraction_strength: 吸引力强度
        attraction_range: 吸引力作用范围 [m]

    返回:
        forces: (N, dim) 每个粒子受到的合力
    """
    positions = np.asarray(positions, dtype=float)
    n_particles, dim = positions.shape
    forces = np.zeros_like(positions)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r < 1e-15 or r > 5.0 * attraction_range:
                continue

            # 排斥项（硬球近似）
            if r < 2.0 * radii:
                f_mag = repulsion_strength * ((2.0 * radii / r) ** 6 - 1.0)
            else:
                f_mag = 0.0

            # 吸引项（血浆介导）
            if r < attraction_range:
                f_mag -= attraction_strength / (r ** 3)

            f_vec = f_mag * r_vec / (r + 1e-15)
            forces[i] -= f_vec
            forces[j] += f_vec

    return forces


def update_rbc_positions_euler(positions: np.ndarray, velocities: np.ndarray,
                               forces: np.ndarray, dt: float,
                               mass_kg: float = 1e-13) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler积分更新红细胞位置。

    参数:
        positions: (N, dim) 当前位置
        velocities: (N, dim) 当前速度
        forces: (N, dim) 受力
        dt: 时间步长
        mass_kg: 单个红细胞质量 [kg]

    返回:
        new_positions, new_velocities
    """
    accel = forces / (mass_kg + 1e-20)
    new_vel = velocities + dt * accel
    new_pos = positions + dt * new_vel
    return new_pos, new_vel


def apparent_viscosity_from_rbc(n_rbc: int, domain_volume: float,
                                base_viscosity: float = 0.0012) -> float:
    """
    基于红细胞数量估算血液表观粘度。

    使用Einstein-Batchelor公式:
        μ_eff = μ_0 (1 + 2.5 φ + 6.2 φ²)

    其中 φ = N_rbc * V_rbc / V_domain
    """
    rbc_volume = 4.0 / 3.0 * np.pi * (2.5e-6) ** 3  # 单个RBC体积 ~ 65 fL
    phi = n_rbc * rbc_volume / (domain_volume + 1e-20)
    phi = np.clip(phi, 0.0, 0.6)
    return base_viscosity * (1.0 + 2.5 * phi + 6.2 * phi * phi)
