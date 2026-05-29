"""
hydro_evolution.py
2+1维相对论性粘性流体力学演化

基于种子项目:
- 1368_tumor_pde: 肿瘤反应扩散PDE → QGP能量-动量守恒方程

物理模型:
1. 能量-动量守恒:
   ∂_μ T^{μν} = 0

2. 理想流体能量-动量张量:
   T^{μν}_{ideal} = (ε + P) u^μ u^ν + P g^{μν}

3. 剪切粘滞修正 (Navier-Stokes):
   T^{μν} = T^{μν}_{ideal} + π^{μν}
   π^{μν} = η σ^{μν}
   σ^{μν} = ∇^μ u^ν + ∇^ν u^μ - (2/3) Δ^{μν} ∇·u

4. Bjorken标度:
   τ = √(t² - z²),  η_s = 0.5 ln((t+z)/(t-z))
   在中心快度区假设 η_s = 0，流体四速度 u^μ = (cosh η_s, 0, 0, sinh η_s)

5. 状态方程:
   P = c_s² ε  (对于QGP，c_s² ≈ 1/3)

6. 温度-能量密度关系:
   ε = (π²/30) g_* T⁴  (Stefan-Boltzmann)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class HydroEvolution:
    """
    2+1维Bjorken流体力学演化求解器。
    """

    def __init__(self, eta_s_over_s: float = 0.08,
                 cs2: float = 1.0 / 3.0,
                 g_star: float = 47.5,
                 tau0: float = 0.6,
                 tau_f: float = 10.0,
                 dtau: float = 0.05):
        """
        初始化流体力学参数。

        Parameters
        ----------
        eta_s_over_s : float
            比剪切粘滞系数 η/s (无量纲，KSS界限为 1/4π ≈ 0.08)
        cs2 : float
            声速平方 c_s²
        g_star : float
            有效自由度数目
        tau0 : float
            初始固有时 [fm/c]
        tau_f : float
            终止固有时 [fm/c]
        dtau : float
            时间步长 [fm/c]
        """
        self.eta_s_over_s = eta_s_over_s
        self.cs2 = cs2
        self.g_star = g_star
        self.tau0 = tau0
        self.tau_f = tau_f
        self.dtau = dtau

    def equation_of_state(self, epsilon: np.ndarray) -> np.ndarray:
        """
        状态方程: P = c_s² ε

        Parameters
        ----------
        epsilon : np.ndarray
            能量密度 [GeV/fm³]

        Returns
        -------
        np.ndarray
            压强 [GeV/fm³]
        """
        return self.cs2 * epsilon

    def energy_to_temperature(self, epsilon: np.ndarray) -> np.ndarray:
        """
        从能量密度计算温度 (Stefan-Boltzmann近似)。

        在自然单位 ℏ = c = k_B = 1 中:
        ε [GeV⁴] = (π²/30) g_* T⁴ [GeV⁴]

        转换到 GeV/fm³:
        ε [GeV/fm³] = ε [GeV⁴] / (ℏc)³
        其中 ℏc = 0.1973 GeV·fm

        因此:
        T = [30 ε (ℏc)³ / (π² g_*)]^{1/4}

        Parameters
        ----------
        epsilon : np.ndarray
            能量密度 [GeV/fm³]

        Returns
        -------
        np.ndarray
            温度 [GeV]
        """
        epsilon = np.asarray(epsilon)
        epsilon = np.where(epsilon < 1e-15, 1e-15, epsilon)
        # HOLE 1: Implement Stefan-Boltzmann temperature conversion
        pass

    def temperature_to_energy(self, T: np.ndarray) -> np.ndarray:
        """
        从温度计算能量密度。

        ε [GeV/fm³] = (π²/30) g_* T⁴ [GeV⁴] / (ℏc)³

        Parameters
        ----------
        T : np.ndarray
            温度 [GeV]

        Returns
        -------
        np.ndarray
            能量密度 [GeV/fm³]
        """
        T = np.asarray(T)
        hbarc = 0.1973269804  # GeV·fm
        return (np.pi ** 2 * self.g_star / 30.0) * (T ** 4) / (hbarc ** 3)

    def specific_entropy(self, epsilon: np.ndarray) -> np.ndarray:
        """
        比熵密度 s = (ε + P) / T。

        Parameters
        ----------
        epsilon : np.ndarray
            能量密度

        Returns
        -------
        np.ndarray
            熵密度 [GeV²/fm³] (ℏ = c = k_B = 1)
        """
        T = self.energy_to_temperature(epsilon)
        P = self.equation_of_state(epsilon)
        s = (epsilon + P) / T
        return s

    def bjorken_1d(self, epsilon0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1维Bjorken流体力学解析解（无横向梯度）。

        理想流体: ε(τ) = ε₀ (τ₀/τ)^{1+c_s²}
        粘滞修正: ε(τ) ≈ ε₀ (τ₀/τ)^{1+c_s²} [1 + (4/3) (η/s) (1 - τ₀/τ) / (τ₀ T₀)]

        Parameters
        ----------
        epsilon0 : float
            初始能量密度 [GeV/fm³]

        Returns
        -------
        tau_grid : np.ndarray
            固有时网格
        epsilon : np.ndarray
            能量密度演化
        temperature : np.ndarray
            温度演化
        """
        tau_grid = np.arange(self.tau0, self.tau_f, self.dtau)
        n_tau = len(tau_grid)

        epsilon = np.zeros(n_tau)
        T = np.zeros(n_tau)

        epsilon[0] = epsilon0
        T[0] = self.energy_to_temperature(epsilon0)

        for i in range(1, n_tau):
            tau = tau_grid[i]
            tau_prev = tau_grid[i - 1]

            # 理想部分
            eps_ideal = epsilon0 * (self.tau0 / tau) ** (1.0 + self.cs2)

            # 粘滞修正 (一阶)
            T0 = self.energy_to_temperature(epsilon0)
            visc_corr = 1.0 + (4.0 / 3.0) * self.eta_s_over_s * (
                1.0 - self.tau0 / tau
            ) / (self.tau0 * T0)
            visc_corr = np.clip(visc_corr, 0.5, 5.0)

            epsilon[i] = eps_ideal * visc_corr
            T[i] = self.energy_to_temperature(epsilon[i])

        return tau_grid, epsilon, T

    def evolve_2d(self, x_grid: np.ndarray, y_grid: np.ndarray,
                  epsilon_init: np.ndarray,
                  nx: int = 40, ny: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        2+1维流体力学演化（简化格式：粘性扩散+Bjorken膨胀）。

        演化方程（在横向平面）:
        ∂ε/∂τ + (1+c_s²) ε/τ = D_⊥ ∇²_⊥ ε

        其中 D_⊥ = (η/s) / (τ T s) 为横向扩散系数。

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            横向网格坐标
        epsilon_init : np.ndarray
            初始能量密度分布 (nx, ny)
        nx, ny : int
            网格数

        Returns
        -------
        tau_grid : np.ndarray
            时间网格
        epsilon_history : np.ndarray
            能量密度历史 (nt, nx, ny)
        T_history : np.ndarray
            温度历史
        entropy_history : np.ndarray
            熵密度历史
        """
        tau_grid = np.arange(self.tau0, self.tau_f, self.dtau)
        nt = len(tau_grid)

        epsilon_history = np.zeros((nt, nx, ny))
        T_history = np.zeros((nt, nx, ny))
        entropy_history = np.zeros((nt, nx, ny))

        epsilon_history[0] = epsilon_init
        T_history[0] = self.energy_to_temperature(epsilon_init)
        entropy_history[0] = self.specific_entropy(epsilon_init)

        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0

        for it in range(1, nt):
            tau = tau_grid[it]
            eps_prev = epsilon_history[it - 1].copy()

            # 边界处理：反射边界
            eps_pad = np.pad(eps_prev, ((1, 1), (1, 1)), mode='edge')

            # Laplacian: ∇²ε
            laplacian = (
                (eps_pad[2:, 1:-1] - 2 * eps_pad[1:-1, 1:-1] + eps_pad[:-2, 1:-1]) / dx ** 2 +
                (eps_pad[1:-1, 2:] - 2 * eps_pad[1:-1, 1:-1] + eps_pad[1:-1, :-2]) / dy ** 2
            )

            # 扩散系数 (与温度相关)
            T_prev = T_history[it - 1]
            s_prev = entropy_history[it - 1]
            D = np.zeros_like(T_prev)
            mask = (T_prev > 1e-6) & (s_prev > 1e-15) & (tau > 1e-6)
            D[mask] = self.eta_s_over_s / (tau * T_prev[mask] * s_prev[mask])
            D = np.clip(D, 0.0, 2.0)

            # 时间演化: 显式欧拉
            # dε/dτ = -(1+cs²) ε/τ + D ∇²ε
            damping = -(1.0 + self.cs2) * eps_prev / tau
            diffusion = D * laplacian

            eps_new = eps_prev + self.dtau * (damping + diffusion)
            eps_new = np.clip(eps_new, 1e-6, 1e6)

            epsilon_history[it] = eps_new
            T_history[it] = self.energy_to_temperature(eps_new)
            entropy_history[it] = self.specific_entropy(eps_new)

        return tau_grid, epsilon_history, T_history, entropy_history

    def freezeout_surface(self, tau_grid: np.ndarray,
                          T_history: np.ndarray,
                          T_freezeout: float = 0.154) -> np.ndarray:
        """
        确定逐点冻结面 τ_f(x,y)，即温度降到 T_fo 的时刻。

        Parameters
        ----------
        tau_grid : np.ndarray
            时间网格
        T_history : np.ndarray
            温度历史 (nt, nx, ny)
        T_freezeout : float
            冻结温度 [GeV] (默认154 MeV)

        Returns
        -------
        np.ndarray
            冻结面 (nx, ny)，若未冻结则为 -1
        """
        nt, nx, ny = T_history.shape
        tau_fo = np.full((nx, ny), -1.0)

        for i in range(nx):
            for j in range(ny):
                T_line = T_history[:, i, j]
                # 找第一个低于冻结温度的点
                below = np.where(T_line < T_freezeout)[0]
                if len(below) > 0:
                    idx = below[0]
                    if idx == 0:
                        tau_fo[i, j] = tau_grid[0]
                    else:
                        # 线性插值
                        t1, t2 = tau_grid[idx - 1], tau_grid[idx]
                        T1, T2 = T_line[idx - 1], T_line[idx]
                        if abs(T2 - T1) > 1e-15:
                            tau_fo[i, j] = t1 + (T_freezeout - T1) * (t2 - t1) / (T2 - T1)
                        else:
                            tau_fo[i, j] = t1
        return tau_fo

    def flow_velocity(self, epsilon_history: np.ndarray,
                      x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算横向流速度 (从能量密度梯度近似)。

        在粘性流体中，流速与压力梯度相关:
        u_x ∝ -∂P/∂x ≈ -c_s² ∂ε/∂x

        Parameters
        ----------
        epsilon_history : np.ndarray
            能量密度历史
        x_grid, y_grid : np.ndarray
            网格坐标

        Returns
        -------
        ux, uy : np.ndarray
            横向流速分量 (nt, nx, ny)
        """
        nt, nx, ny = epsilon_history.shape
        ux = np.zeros_like(epsilon_history)
        uy = np.zeros_like(epsilon_history)

        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0

        for it in range(nt):
            eps = epsilon_history[it]
            # 中心差分
            eps_pad_x = np.pad(eps, ((0, 0), (1, 1)), mode='edge')
            eps_pad_y = np.pad(eps, ((1, 1), (0, 0)), mode='edge')

            d_eps_dx = (eps_pad_x[:, 2:] - eps_pad_x[:, :-2]) / (2.0 * dx)
            d_eps_dy = (eps_pad_y[2:, :] - eps_pad_y[:-2, :]) / (2.0 * dy)

            ux[it] = -self.cs2 * d_eps_dx * 0.1  # 比例系数
            uy[it] = -self.cs2 * d_eps_dy * 0.1

        return ux, uy

    def entropy_production(self, tau_grid: np.ndarray,
                           epsilon_history: np.ndarray,
                           ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        """
        计算粘性熵产生率 (基于Rayleigh-Onsager耗散函数)。

        dS/dτ = ∫ d²x τ π^{μν} π_{μν} / (2ηT)

        简化: dS/dτ ∝ η/s · (∇_⊥ · u)² / T

        Parameters
        ----------
        tau_grid : np.ndarray
            时间网格
        epsilon_history : np.ndarray
            能量密度
        ux, uy : np.ndarray
            流速

        Returns
        -------
        np.ndarray
            累积熵产生 (nt,)
        """
        nt, nx, ny = epsilon_history.shape
        S_production = np.zeros(nt)

        dx = 1.0 if nx <= 1 else 1.0  # 归一化
        dy = 1.0 if ny <= 1 else 1.0

        for it in range(1, nt):
            # 散度
            ux_pad = np.pad(ux[it], ((0, 0), (1, 1)), mode='edge')
            uy_pad = np.pad(uy[it], ((1, 1), (0, 0)), mode='edge')
            div_u = ((ux_pad[:, 2:] - ux_pad[:, :-2]) / (2 * dx) +
                     (uy_pad[2:, :] - uy_pad[:-2, :]) / (2 * dy))

            T = self.energy_to_temperature(epsilon_history[it])
            T = np.where(T < 1e-6, 1e-6, T)

            # 局域熵产生率
            local_rate = self.eta_s_over_s * (div_u ** 2) / T
            S_production[it] = S_production[it - 1] + self.dtau * np.sum(local_rate) * dx * dy

        return S_production
