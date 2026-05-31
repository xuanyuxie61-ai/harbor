# -*- coding: utf-8 -*-
"""
wave_propagation.py
===================
一维复合材料杆中应力波传播与动态响应分析模块。

源自种子项目 020_artery_pde（一维双曲 PDE 的血流模型），
改造为固体结构力学中的受迫阻尼波动方程。

科学背景：
---------
考虑含损伤的复合材料杆在冲击/循环载荷下的轴向波动：

  ρ * A * ∂²u/∂t² + c_d * ∂u/∂t = ∂/∂x (EA(x,d) * ∂u/∂x) + f(x,t)

其中：
  u(x,t)     — 轴向位移 [m]
  ρ(x)       — 等效密度 [kg/m³]
  A          — 横截面积 [m²]
  c_d        — 结构阻尼系数 [N·s/m²]
  E(x,d)     — 含损伤弹性模量 [Pa]
  f(x,t)     — 分布体积力 [N/m]

引入状态变量：
  v = ∂u/∂t       — 轴向速度 [m/s]
  ε = ∂u/∂x       — 轴向应变 [-]

化为一阶系统（Method of Lines）：
  ∂ε/∂t = ∂v/∂x
  ρA * ∂v/∂t = ∂(σA)/∂x - c_d * v + f(x,t)
  σ = E(x,d) * ε

对于均匀无损杆（E, ρ 为常数），波动方程简化为：
  ∂²u/∂t² + 2ζω_0 * ∂u/∂t = c² * ∂²u/∂x² + f/(ρA)
其中：
  c = sqrt(E/ρ)           — 纵波波速 [m/s]
  ω_0 = π*c/L            — 基频 [rad/s]
  ζ = c_d / (2ρAω_0)     — 阻尼比 [-]

受迫振动稳态解（分离变量法）：
  设 f(x,t) = F_0 * δ(x-x_0) * sin(ωt)，则稳态位移响应幅值：
    U(x; ω) = Σ_{n=1}^∞ [2/(ρAL)] * [φ_n(x_0)φ_n(x)] / [(ω_n² - ω²)² + (2ζω_nω)²]^{1/2}
  其中 φ_n(x) = sin(nπx/L) 为模态形状，ω_n = nπc/L 为固有频率。

损伤对波传播的影响：
  局部损伤 d(x) 导致波速降低 c_eff(x) = sqrt(E(1-d(x))²/ρ)，
  并引起波反射和散射，可用反射系数近似：
    R ≈ (Z_2 - Z_1) / (Z_2 + Z_1)
  其中 Z_i = ρ_i * c_i 为波阻抗。
"""

import numpy as np
from scipy.integrate import odeint
from typing import Callable, Optional, Tuple


class WaveEquation1D:
    """
    一维波动方程的有限差分离散求解器（Method of Lines）。
    作为 DG 求解器的补充，用于对比验证和特定边界条件问题。
    """

    def __init__(self, L: float, nx: int,
                 rho: float, E: float, A: float = 1.0,
                 damping: float = 0.0):
        """
        Parameters
        ----------
        L : float
            杆长 [m].
        nx : int
            空间网格点数（包含边界）。
        rho, E : float
            密度 [kg/m³] 和弹性模量 [Pa].
        A : float
            横截面积 [m²].
        damping : float
            阻尼系数 c_d [N·s/m²].
        """
        if nx < 3:
            raise ValueError("nx must be >= 3.")
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.rho = rho
        self.E = E
        self.A = A
        self.damping = damping
        self.c = np.sqrt(E / rho)
        self.x = np.linspace(0.0, L, nx)

    def _rhs_fd(self, y: np.ndarray, t: float,
                f_func: Optional[Callable], damage_field: Optional[np.ndarray]) -> np.ndarray:
        """
        有限差分法右端项。
        y = [u_0, ..., u_{nx-1}, v_0, ..., v_{nx-1}]
        """
        nx = self.nx
        u = y[:nx]
        v = y[nx:]

        # 空间离散：中心差分
        dudt = v.copy()

        # 应变 ε = du/dx
        eps = np.zeros(nx)
        eps[1:-1] = (u[2:] - u[:-2]) / (2.0 * self.dx)
        eps[0] = (u[1] - u[0]) / self.dx
        eps[-1] = (u[-1] - u[-2]) / self.dx

        # 应力 σ = E * (1-d)^2 * ε（含损伤退化）
        E_local = self.E * np.ones(nx)
        if damage_field is not None:
            if len(damage_field) != nx:
                raise ValueError("damage_field length must match nx.")
            g_d = (1.0 - np.clip(damage_field, 0.0, 1.0)) ** 2 + 1e-6
            E_local *= g_d

        sigma = E_local * eps

        # ∂σ/∂x 的差分
        dsigma_dx = np.zeros(nx)
        dsigma_dx[1:-1] = (sigma[2:] - sigma[:-2]) / (2.0 * self.dx)
        dsigma_dx[0] = (sigma[1] - sigma[0]) / self.dx
        dsigma_dx[-1] = (sigma[-1] - sigma[-2]) / self.dx

        # 外力
        f_ext = np.zeros(nx)
        if f_func is not None:
            f_ext = f_func(self.x, t)

        # 速度方程
        dvdt = (self.A * dsigma_dx - self.damping * v + f_ext) / (self.rho * self.A)

        # 边界条件：左端固定 u=0，右端自由 σ=0
        dudt[0] = 0.0
        dvdt[0] = 0.0
        dvdt[-1] = 0.0  # 自由端加速度由应力梯度决定，已包含

        return np.concatenate([dudt, dvdt])

    def solve(self, u0: np.ndarray, v0: np.ndarray,
              t_span: Tuple[float, float], nt: int,
              f_func: Optional[Callable] = None,
              damage_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解波动方程。

        Returns
        -------
        t : np.ndarray, shape (nt,)
        u : np.ndarray, shape (nt, nx)
        v : np.ndarray, shape (nt, nx)
        """
        t = np.linspace(t_span[0], t_span[1], nt)
        y0 = np.concatenate([u0, v0])

        def rhs(y, t_val):
            return self._rhs_fd(y, t_val, f_func, damage_field)

        sol = odeint(rhs, y0, t)
        u_hist = sol[:, :self.nx]
        v_hist = sol[:, self.nx:]
        return t, u_hist, v_hist

    def modal_analysis(self, num_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        模态分析：计算均匀杆的前 num_modes 个固有频率和模态形状。

        解析解（两端固定-自由）：
          ω_n = (2n-1) * π * c / (2L)
          φ_n(x) = sin((2n-1) * π * x / (2L))
        """
        n = np.arange(1, num_modes + 1)
        omega_n = (2.0 * n - 1.0) * np.pi * self.c / (2.0 * self.L)
        phi = np.zeros((num_modes, self.nx))
        for i, nn in enumerate(n):
            phi[i, :] = np.sin((2.0 * nn - 1.0) * np.pi * self.x / (2.0 * self.L))
        return omega_n, phi

    def forced_response_amplitude(self, x_force: float, omega: float,
                                   num_modes: int = 20) -> np.ndarray:
        """
        计算在 x_force 处受频率 ω 简谐力作用时的稳态位移幅值分布。

        公式：
          U(x; ω) = Σ_{n=1}^{N_m} [2/(ρAL)] * φ_n(x_f) * φ_n(x) / sqrt((ω_n²-ω²)² + (2ζω_nω)²)
        """
        omega_n, phi = self.modal_analysis(num_modes=num_modes)
        zeta = self.damping / (2.0 * self.rho * self.A * omega_n + 1e-30)
        denom = np.sqrt((omega_n ** 2 - omega ** 2) ** 2 + (2.0 * zeta * omega_n * omega) ** 2)
        coeff = (2.0 / (self.rho * self.A * self.L)) * phi[:, int(x_force / self.L * (self.nx - 1))] / denom
        U = np.sum(coeff[:, None] * phi, axis=0)
        return U


class ImpactLoad:
    """
    冲击载荷模型。
    """

    @staticmethod
    def half_sine_pulse(t: float, F0: float, duration: float) -> float:
        """半正弦冲击脉冲。"""
        if t < 0 or t > duration:
            return 0.0
        return F0 * np.sin(np.pi * t / duration)

    @staticmethod
    def triangular_pulse(t: float, F0: float, duration: float) -> float:
        """三角形冲击脉冲。"""
        if t < 0 or t > duration:
            return 0.0
        if t <= duration / 2.0:
            return 2.0 * F0 * t / duration
        return 2.0 * F0 * (1.0 - t / duration)

    @staticmethod
    def blast_wave(t: float, F0: float, tau_rise: float, tau_decay: float) -> float:
        """
        Friedlander 爆炸波模型：
          p(t) = F0 * (1 - t/tau_decay) * exp(-t/tau_rise)
        """
        if t < 0:
            return 0.0
        return F0 * (1.0 - t / (tau_decay + 1e-30)) * np.exp(-t / (tau_rise + 1e-30))


class WaveReflectionAnalysis:
    """
    应力波在材料界面和损伤区的反射/透射分析。
    """

    @staticmethod
    def reflection_coefficient(rho1: float, c1: float, rho2: float, c2: float) -> float:
        """
        平面纵波垂直入射到平面界面时的反射系数（应力反射）。

        公式：
          Z1 = ρ1 * c1,  Z2 = ρ2 * c2
          R = (Z2 - Z1) / (Z2 + Z1)
          T = 2 * Z2 / (Z2 + Z1)
        """
        Z1 = rho1 * c1
        Z2 = rho2 * c2
        return (Z2 - Z1) / (Z2 + Z1 + 1e-30)

    @staticmethod
    def transmission_coefficient(rho1: float, c1: float, rho2: float, c2: float) -> float:
        """透射系数（应力透射）。"""
        Z1 = rho1 * c1
        Z2 = rho2 * c2
        return 2.0 * Z2 / (Z2 + Z1 + 1e-30)

    @staticmethod
    def damage_reflection_approx(E0: float, rho: float, d_local: float) -> float:
        """
        局部损伤区对平面波的近似反射系数。
        假设损伤区很薄，波速从 c0 = sqrt(E0/ρ) 降至 c_d = sqrt(E0*(1-d)^2/ρ)。
        """
        c0 = np.sqrt(E0 / rho)
        g_d = (1.0 - np.clip(d_local, 0.0, 0.99)) ** 2 + 1e-6
        c_d = c0 * np.sqrt(g_d)
        return WaveReflectionAnalysis.reflection_coefficient(rho, c0, rho, c_d)


if __name__ == "__main__":
    # 自测试
    wave = WaveEquation1D(L=1.0, nx=101, rho=1600.0, E=100e9, A=1e-4, damping=100.0)

    # 模态分析
    omega_n, phi = wave.modal_analysis(num_modes=3)
    print("Natural frequencies (Hz):", omega_n / (2 * np.pi))

    # 受迫响应
    U = wave.forced_response_amplitude(x_force=0.5, omega=omega_n[0] * 0.9)
    print("Forced response max amplitude:", np.max(np.abs(U)))

    # 反射分析
    R = WaveReflectionAnalysis.damage_reflection_approx(E0=100e9, rho=1600.0, d_local=0.3)
    print("Reflection coefficient for d=0.3:", R)
