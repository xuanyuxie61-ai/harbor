"""
pulse_wave_dynamics.py
动脉压力脉冲波传播模型

融合来源:
- 1061_schroedinger_nonlinear_pde: 非线性薛定谔方程（NLSE）孤子解、有限差分离散、守恒律监测
- 345_exm (waterwave): Lax-Wendroff格式求解一维浅水波方程

科学背景:
心脏周期性收缩产生的压力波以约4-8 m/s的速度沿动脉树传播。
该压力波可建模为：
1. 一维线性化水锤方程（浅水波类比）
2. 非线性薛定谔方程（NLSE）描述的孤子脉冲

一维水锤方程（类比浅水波）:
    ∂A/∂t + ∂Q/∂z = 0                    (质量守恒)
    ∂Q/∂t + ∂(Q²/A)/∂z = -A/ρ ∂p/∂z + f_visc  (动量守恒)

对于小扰动，线性化为:
    ∂p/∂t + (ρ c²/A) ∂Q/∂z = 0
    ∂Q/∂t + (A/ρ) ∂p/∂z = 0

波速 c = sqrt(E h / (2 ρ R))（Moens-Korteweg公式）

NLSE类比:
将压力脉冲视为非线性波包，满足:
    i ∂ψ/∂t + ∂²ψ/∂z² + γ |ψ|² ψ = 0
其中ψ与压力扰动p'相关，γ为非线性系数。
"""

import numpy as np
from typing import Tuple


# ======================================================================
# 来自 345_exm (waterwave) 的 Lax-Wendroff 浅水波格式
# ======================================================================

def shallow_water_lax_wendroff(A: np.ndarray, Q: np.ndarray,
                               dx: float, dt: float,
                               g_eff: float = 1.0,
                               n_steps: int = 1,
                               boundary_type: str = "reflecting") -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Lax-Wendroff格式求解一维浅水波方程（类比动脉压力波）。

    守恒形式:
        U = [A, Q]^T
        F(U) = [Q, Q²/A + 0.5 g A²]^T

    Lax-Wendroff两步格式:
        1. 半步预测（在网格中点）:
           U_{j+1/2}^{n+1/2} = 0.5(U_j^n + U_{j+1}^n) - 0.5(Δt/Δx)(F_{j+1}^n - F_j^n)
        2. 全步校正:
           U_j^{n+1} = U_j^n - (Δt/Δx)(F_{j+1/2}^{n+1/2} - F_{j-1/2}^{n+1/2})

    参数:
        A: 横截面积数组
        Q: 流量数组
        dx: 空间步长
        dt: 时间步长
        g_eff: 等效重力加速度（类比血管弹性）
        n_steps: 推进步数
        boundary_type: 边界条件类型 ("reflecting" 或 "periodic")

    返回:
        A_new, Q_new
    """
    A = np.asarray(A, dtype=float).copy()
    Q = np.asarray(Q, dtype=float).copy()
    n_grid = len(A)

    # Courant数检查
    cfl = dt / dx
    max_speed = np.max(np.sqrt(g_eff * A + 1e-15) + np.abs(Q / (A + 1e-15)))
    if max_speed * cfl > 1.0:
        # 自适应缩小时间步
        dt = 0.9 * dx / (max_speed + 1e-15)
        cfl = dt / dx

    def flux(AA, QQ):
        F1 = QQ
        F2 = QQ ** 2 / (AA + 1e-15) + 0.5 * g_eff * AA ** 2
        return np.array([F1, F2])

    def apply_boundary(AA, QQ):
        if boundary_type == "reflecting":
            AA[0] = AA[1]
            AA[-1] = AA[-2]
            QQ[0] = 0.0  # 入口固定流量或封闭端
            QQ[-1] = QQ[-2]
        elif boundary_type == "periodic":
            AA[0] = AA[-2]
            AA[-1] = AA[1]
            QQ[0] = QQ[-2]
            QQ[-1] = QQ[1]
        return AA, QQ

    for _ in range(n_steps):
        A, Q = apply_boundary(A, Q)

        # 半步预测
        U = np.vstack([A, Q])
        F = np.zeros((2, n_grid))
        for j in range(n_grid):
            F[:, j] = flux(A[j], Q[j])

        U_half = np.zeros((2, n_grid - 1))
        for j in range(n_grid - 1):
            U_half[:, j] = 0.5 * (U[:, j] + U[:, j + 1]) - 0.5 * cfl * (F[:, j + 1] - F[:, j])

        A_half = U_half[0, :]
        Q_half = U_half[1, :]

        # 全步校正
        F_half = np.zeros((2, n_grid - 1))
        for j in range(n_grid - 1):
            F_half[:, j] = flux(A_half[j], Q_half[j])

        U_new = U.copy()
        for j in range(1, n_grid - 1):
            U_new[:, j] = U[:, j] - cfl * (F_half[:, j] - F_half[:, j - 1])

        A = U_new[0, :]
        Q = U_new[1, :]

    return A, Q


def pressure_wave_speed(elastic_modulus_pa: float,
                        wall_thickness_m: float,
                        vessel_radius_m: float,
                        blood_density_kg_m3: float = 1060.0) -> float:
    """
    Moens-Korteweg公式计算动脉压力波传播速度。

    公式:
        c = sqrt( E h / (2 ρ R) )

    参数:
        elastic_modulus_pa: 血管壁杨氏模量 [Pa]
        wall_thickness_m: 壁厚 [m]
        vessel_radius_m: 血管半径 [m]
        blood_density_kg_m3: 血液密度 [kg/m³]

    返回:
        c: 波速 [m/s]
    """
    if vessel_radius_m <= 0 or wall_thickness_m <= 0:
        raise ValueError("Radius and thickness must be positive")
    return np.sqrt(elastic_modulus_pa * wall_thickness_m /
                   (2.0 * blood_density_kg_m3 * vessel_radius_m))


# ======================================================================
# 来自 1061_schroedinger_nonlinear_pde 的 NLSE 孤子模型
# ======================================================================

class NLSEPressurePulse:
    """
    使用非线性薛定谔方程（NLSE）描述动脉压力脉冲的孤子传播。

    将压力扰动表示为复波包:
        p'(z,t) = Re{ ψ(z,t) · exp(i k_0 z - i ω_0 t) }

    ψ满足聚焦型NLSE:
        i ∂ψ/∂t + ∂²ψ/∂z² + γ |ψ|² ψ = 0

    其中:
        γ: 非线性系数（由血管壁非线性弹性决定）
        二阶空间导数用中心差分近似
        时间积分用分步Fourier法或简化的Runge-Kutta
    """
    def __init__(self, nx: int, z_min: float, z_max: float,
                 gamma: float = 0.5, dx: float = None):
        self.nx = nx
        self.z = np.linspace(z_min, z_max, nx)
        self.dz = self.z[1] - self.z[0] if dx is None else dx
        self.gamma = gamma

        # 空间导数离散索引（Neumann边界）
        self.im1 = np.array([1] + list(range(nx - 2)) + [nx - 2])
        self.i = np.arange(nx)
        self.ip1 = np.array([1] + list(range(2, nx)) + [nx - 2])

    def initial_double_soliton(self, z: np.ndarray,
                                amplitude: float = 1.0,
                                c1: float = 2.0, c2: float = 0.5,
                                delta: float = 5.0) -> np.ndarray:
        """
        双孤子初始条件。

        解析形式:
            ψ(z,0) = √α · √(2/γ) · [ e^{iφ₁} sech(ξ₁) + e^{iφ₂} sech(ξ₂) ]
        其中:
            ξ₁ = √α (z - z₀)
            ξ₂ = √α (z - z₀ - δ)
        """
        alpha = amplitude
        prefactor = np.sqrt(alpha) * np.sqrt(2.0 / (self.gamma + 1e-15))

        xi1 = np.sqrt(alpha) * (z - c1 * 0.0)
        xi2 = np.sqrt(alpha) * (z - delta - c2 * 0.0)

        phi1 = 1j * ((c1 / 2.0) * z)
        phi2 = 1j * ((c2 / 2.0) * (z - delta))

        psi = prefactor * (np.exp(phi1) / np.cosh(xi1 + 1e-15) +
                           np.exp(phi2) / np.cosh(xi2 + 1e-15))
        return psi

    def deriv(self, psi: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        计算空间离散后的ODE右端项 dψ/dt。

        dψ_j/dt = i [ (ψ_{j+1} - 2ψ_j + ψ_{j-1})/Δz² + γ |ψ_j|² ψ_j ]
        """
        psi_im1 = psi[self.im1]
        psi_i = psi[self.i]
        psi_ip1 = psi[self.ip1]

        psi_zz = (psi_ip1 - 2.0 * psi_i + psi_im1) / (self.dz ** 2)
        nonlinear = self.gamma * np.abs(psi_i) ** 2 * psi_i
        dpsi_dt = 1j * (psi_zz + nonlinear)
        return dpsi_dt

    def mass_conservation(self, psi: np.ndarray) -> float:
        """
        计算L²质量（守恒量监测）。

        M = ∫ |ψ|² dz ≈ Δz ( -0.5|ψ_1|² + Σ|ψ_j|² - 0.5|ψ_nx|² )
        """
        n = len(psi)
        m = self.dz * (-0.5 * np.abs(psi[0]) ** 2 +
                       np.sum(np.abs(psi[1:n - 1]) ** 2) -
                       0.5 * np.abs(psi[-1]) ** 2)
        return float(np.abs(m))

    def step_rk4(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        四阶Runge-Kutta时间步进。
        """
        k1 = self.deriv(psi)
        k2 = self.deriv(psi + 0.5 * dt * k1)
        k3 = self.deriv(psi + 0.5 * dt * k2)
        k4 = self.deriv(psi + dt * k3)
        return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def evolve(self, psi0: np.ndarray, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        演化NLSE，返回最终波函数和质量守恒监测。

        返回:
            psi_final: 最终波函数
            mass_history: 每步的质量值
        """
        psi = psi0.copy()
        mass_hist = np.zeros(n_steps)

        for n in range(n_steps):
            psi = self.step_rk4(psi, dt)
            mass_hist[n] = self.mass_conservation(psi)

        return psi, mass_hist


def nlse_to_pressure_amplitude(psi: np.ndarray,
                               base_pressure_pa: float = 10000.0,
                               conversion_factor: float = 5000.0) -> np.ndarray:
    """
    将NLSE波函数幅度映射为压力扰动幅值。

    p'(z) = p_base + conversion_factor · |ψ(z)|
    """
    return base_pressure_pa + conversion_factor * np.abs(psi)
