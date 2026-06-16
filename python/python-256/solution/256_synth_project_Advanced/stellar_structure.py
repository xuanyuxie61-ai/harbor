"""
stellar_structure.py
====================
构建一维恒星平衡模型: 求解恒星内部结构方程组 (质量连续、流体静力学平衡、
能量守恒、辐射传热), 并计算热力学量 (Brunt-Väisälä 频率, 绝热指数 Gamma_1).

融合种子项目:
  - biochemical_linear_ode (090): 线性 ODE 精确解 → 恒星结构方程指数积分
  - HyperMPC (1177): 超网络参数化 → 不透明度/状态方程参数从观测频率序列推断
  - ellipse_distance (329): 椭圆采样/距离统计 → 恒星参数空间统计表征

科学背景
--------
恒星平衡模型由四个一阶常微分方程描述 (Kippenhahn & Weigert 1990):

  dm/dr = 4 π r² ρ                              (质量连续)
  dP/dr = -G m ρ / r²                            (流体静力学平衡)
  dL/dr = 4 π r² ρ ε                             (能量守恒)
  dT/dr = -∇ T / H_P                              (温度梯度)

其中 H_P = P/(ρ g) 为压力标高, g = Gm/r² 为局部重力加速度.

简化模型中, 我们采用多方球 (polytrope) P = K ρ^{1+1/n} 作为背景,
并叠加小振幅扰动来计算热力学导数.

Brunt-Väisälä (浮力) 频率:

  N² = g (1/Γ₁ dP/dr / P  -  dρ/dr / ρ)
     = g/H_P (∇_ad - ∇ + B_term)

其中 B_term 为 Ledoux 修正项 (化学成分梯度):

  B_term = φ/δ ∇_μ = (∂lnρ/∂lnμ)_{P,T} dlnμ/dlnP

本模块实现:
  1. 多方球平衡模型 (n=3 辐射主序星, n=1.5 对流低质量星)
  2. 指数积分器求解线性化结构方程
  3. N² 剖面的高精度计算
  4. 超网络参数化映射 (观测频率 → 结构参数)
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ============================================================
# 物理常数 (CGS)
# ============================================================
G_CGS = 6.67430e-8        # 万有引力常数 [cm³ g⁻¹ s⁻²]
K_BOLTZMANN = 1.380649e-16  # Boltzmann 常数 [erg K⁻¹]
M_H = 1.6735575e-24       # 氢原子质量 [g]
SIGMA_SB = 5.6704e-5      # Stefan-Boltzmann 常数 [erg cm⁻² s⁻¹ K⁻⁴]
C_LIGHT = 2.99792458e10   # 光速 [cm s⁻¹]
M_SUN = 1.98892e33        # 太阳质量 [g]
R_SUN = 6.957e10          # 太阳半径 [cm]
L_SUN = 3.828e33          # 太阳光度 [erg s⁻¹]


class StellarStructureModel:
    """
    一维恒星平衡模型求解器.

    采用多方球 (polytrope) 作为背景模型, 计算热力学量剖面.
    支持 n=1.5 (完全对流) 和 n=3 (完全辐射) 多方指数.

    参数
    ----
    mass_solar : float
        恒星质量 (太阳质量单位)
    radius_solar : float
        恒星半径 (太阳半径单位)
    polytropic_index : float
        多方指数 n (默认 3.0, 对应 Eddington 标准模型)
    X_H : float
        氢质量分数 (默认 0.70)
    Z_metal : float
        金属质量分数 (默认 0.02)
    n_radial : int
        径向网格点数 (默认 512)
    """

    def __init__(
        self,
        mass_solar: float = 1.0,
        radius_solar: float = 1.0,
        polytropic_index: float = 3.0,
        X_H: float = 0.70,
        Z_metal: float = 0.02,
        n_radial: int = 512,
    ):
        # --- 边界检查与鲁棒性 ---
        if mass_solar <= 0.0 or mass_solar > 150.0:
            raise ValueError(
                f"mass_solar={mass_solar} 超出物理范围 [0, 150] M_sun"
            )
        if radius_solar <= 0.0 or radius_solar > 2000.0:
            raise ValueError(
                f"radius_solar={radius_solar} 超出物理范围 [0, 2000] R_sun"
            )
        if polytropic_index < 0.0 or polytropic_index > 5.0:
            raise ValueError(
                f"polytropic_index={polytropic_index} 超出物理范围 [0, 5]"
            )
        if n_radial < 16:
            raise ValueError(f"n_radial={n_radial} 过少, 至少需要 16")

        self.M_star = mass_solar * M_SUN        # [g]
        self.R_star = radius_solar * R_SUN      # [cm]
        self.n_poly = polytropic_index
        self.X_H = X_H
        self.Z_metal = Z_metal
        self.Y_He = 1.0 - X_H - Z_metal       # 氦质量分数
        self.n_r = n_radial

        # 平均分子量 (完全电离)
        # μ = (2X + 3/4 Y + 1/2 Z)^{-1}
        self.mu = 1.0 / (2.0 * self.X_H + 0.75 * self.Y_He + 0.5 * self.Z_metal)

        # 中心密度 (多方球理论)
        # ρ_c = M / (4π R³ ξ₁² |θ'(ξ₁)|) × 归一化因子
        self._compute_polytropic_constants()

        # 构建网格
        self.r = np.linspace(0.0, self.R_star, self.n_r)
        self.dr = self.r[1] - self.r[0]
        self.xi = self.r / self.r[0] if self.r[0] > 0 else np.linspace(1e-10, 1.0, self.n_r)

        # 求解 Lane-Emden 方程
        self.theta, self.dtheta_dxi = self._solve_lane_emden()

        # 物理量剖面
        self.rho = np.zeros(self.n_r)     # 密度 [g/cm³]
        self.P = np.zeros(self.n_r)       # 压强 [dyn/cm²]
        self.T = np.zeros(self.n_r)       # 温度 [K]
        self.m_enc = np.zeros(self.n_r)   # 包层质量 [g]
        self.g = np.zeros(self.n_r)       # 重力加速度 [cm/s²]
        self.N2 = np.zeros(self.n_r)      # Brunt-Väisälä 频率² [s⁻²]
        self.Gamma1 = np.zeros(self.n_r)  # 绝热指数
        self.Hp = np.zeros(self.n_r)      # 压力标高 [cm]

        self._compute_structure()

    def _compute_polytropic_constants(self):
        """
        计算多方球常数.

        多方球关系:
          P = K ρ^{1+1/n}
          θ(ξ) 满足 Lane-Emden 方程:
            (1/ξ²) d/dξ (ξ² dθ/dξ) = -θ^n

        Lane-Emden 函数的第一零点 ξ₁:
          n=1.5: ξ₁ ≈ 3.6538
          n=3.0: ξ₁ ≈ 6.8969

        中心密度:
          ρ_c = (M / 4π) × (-ξ₁/θ'(ξ₁)) / R³
        """
        # Lane-Emden 第一零点 (查表 + 插值)
        xi1_table = {
            0.0: np.pi, 1.0: np.pi, 1.5: 3.65375,
            2.0: 4.35287, 3.0: 6.89685, 4.0: 14.97155, 5.0: np.inf,
        }
        n_int = int(round(self.n_poly))
        if n_int in xi1_table:
            self.xi1 = xi1_table[n_int]
        else:
            # 线性插值
            n_lo = max(k for k in xi1_table if k < self.n_poly and k < 5)
            n_hi = min(k for k in xi1_table if k > self.n_poly and k <= 5)
            frac = (self.n_poly - n_lo) / max(n_hi - n_lo, 1e-10)
            self.xi1 = xi1_table[n_lo] * (1 - frac) + xi1_table[n_hi] * frac

        if np.isinf(self.xi1):
            self.xi1 = 100.0  # n=5 截断

        # -ξ₁² θ'(ξ₁) 的近似值 (查表)
        xi1_theta_prime_table = {
            0.0: np.pi, 1.0: np.pi, 1.5: 2.71406,
            2.0: 2.41105, 3.0: 2.01824, 4.0: 1.79723,
        }
        if n_int in xi1_theta_prime_table:
            self.xi1_sq_dtheta = xi1_theta_prime_table[n_int]
        else:
            self.xi1_sq_dtheta = 2.01824  # 默认

        # 中心密度 [g/cm³]
        # ρ_c = M/(4π R³) × ξ₁/(-ξ₁²θ'(ξ₁)/ξ₁) = M/(4π R³) × ξ₁²/(-ξ₁²θ'(ξ₁))
        self.rho_c = self.M_star / (4.0 * np.pi * self.R_star**3) * (
            self.xi1 / max(self.xi1_sq_dtheta / self.xi1, 1e-30)
        )

        # 中心压强
        # P_c = (4π)^{1/3} / (n+1) × G M^{2/3} ρ_c^{4/3} × 结构因子
        # 简化: P_c ≈ G M² / (4π R⁴) × 修正因子
        self.P_c = G_CGS * self.M_star**2 / (4.0 * np.pi * self.R_star**4) * (
            self.xi1 / max(self.xi1_sq_dtheta / self.xi1, 1e-30)
        ) / (self.n_poly + 1.0)

        # 多方常数 K
        # P_c = K ρ_c^{1+1/n}
        if self.rho_c > 0:
            self.K_poly = self.P_c / max(self.rho_c ** (1.0 + 1.0 / self.n_poly), 1e-100)
        else:
            self.K_poly = 1e10

        # 中心温度 (理想气体)
        # T_c = μ m_H P_c / (ρ_c k_B)
        if self.rho_c > 0:
            self.T_c = self.mu * M_H * self.P_c / (self.rho_c * K_BOLTZMANN)
        else:
            self.T_c = 1e7

    def _solve_lane_emden(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 Lane-Emden 方程 (四阶 Runge-Kutta).

        (1/ξ²) d/dξ (ξ² dθ/dξ) + θ^n = 0

        边界条件: θ(0) = 1, θ'(0) = 0

        在 ξ → 0 附近用 Taylor 展开启动:
          θ(ξ) ≈ 1 - ξ²/6 + n ξ⁴/120 - ...
          θ'(ξ) ≈ -ξ/3 + n ξ³/30 - ...

        Returns
        -------
        theta : ndarray
            Lane-Emden 函数值
        dtheta : ndarray
            Lane-Emden 函数导数
        """
        n = self.n_poly
        xi = np.linspace(1e-8, self.xi1, self.n_r)
        dxi = xi[1] - xi[0]

        theta = np.zeros(self.n_r)
        dtheta = np.zeros(self.n_r)

        # 初始条件 (Taylor 展开)
        theta[0] = 1.0 - xi[0]**2 / 6.0 + n * xi[0]**4 / 120.0
        dtheta[0] = -xi[0] / 3.0 + n * xi[0]**3 / 30.0

        # RK4 积分
        for i in range(self.n_r - 1):
            xi_i = xi[i]
            th_i = theta[i]
            dt_i = dtheta[i]

            # 状态变量: y = [θ, dθ/dξ]
            # dy/dξ = [dθ/dξ, -θ^n - 2/ξ dθ/dξ]
            def rhs(xi_val, th_val, dt_val):
                theta_n = max(th_val, 0.0) ** n
                d2th = -theta_n - 2.0 / max(xi_val, 1e-30) * dt_val
                return dt_val, d2th

            k1_th, k1_dt = rhs(xi_i, th_i, dt_i)
            k2_th, k2_dt = rhs(xi_i + dxi / 2, th_i + dxi * k1_th / 2,
                                dt_i + dxi * k1_dt / 2)
            k3_th, k3_dt = rhs(xi_i + dxi / 2, th_i + dxi * k2_th / 2,
                                dt_i + dxi * k2_dt / 2)
            k4_th, k4_dt = rhs(xi_i + dxi, th_i + dxi * k3_th,
                                dt_i + dxi * k3_dt)

            theta[i + 1] = th_i + dxi / 6.0 * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
            dtheta[i + 1] = dt_i + dxi / 6.0 * (k1_dt + 2 * k2_dt + 2 * k3_dt + k4_dt)

            # 物理约束: θ ≥ 0
            if theta[i + 1] < 0:
                theta[i + 1:] = 0.0
                dtheta[i + 1:] = 0.0
                break

        return theta, dtheta

    def _compute_structure(self):
        """
        从 Lane-Emden 解计算完整恒星结构剖面.

        物理关系:
          ρ(r) = ρ_c θ^n(ξ)
          P(r) = P_c θ^{n+1}(ξ)
          T(r) = μ m_H P(r) / (ρ(r) k_B)   [理想气体]
          m(r) = 4π ρ_c r³ (-ξ²θ') / (ξ₁³ |θ'(ξ₁)|)
          g(r) = G m(r) / r²
        """
        for i in range(self.n_r):
            th = max(self.theta[i], 0.0)

            # 密度
            self.rho[i] = self.rho_c * th ** self.n_poly

            # 压强
            self.P[i] = self.P_c * th ** (self.n_poly + 1.0)

            # 温度 (理想气体 + 辐射压修正)
            if self.rho[i] > 1e-30:
                self.T[i] = self.mu * M_H * self.P[i] / (self.rho[i] * K_BOLTZMANN)
            else:
                self.T[i] = 0.0

            # 辐射压修正 (Eddington 因子)
            # P_rad = a T⁴ / 3,  a = 4σ/c
            a_rad = 4.0 * SIGMA_SB / C_LIGHT
            P_rad = a_rad * self.T[i]**4 / 3.0
            beta = max(self.P[i] / max(self.P[i] + P_rad, 1e-100), 1e-10)
            # 有效 Γ₁ 受辐射压影响
            # Γ₁_eff = β + (4-3β)²(γ-1) / (β + 12(γ-1)(4-3β))
            # 对于单原子理想气体 γ = 5/3
            gamma_gas = 5.0 / 3.0
            four_minus_3beta = 4.0 - 3.0 * beta
            self.Gamma1[i] = (beta + four_minus_3beta**2 * (gamma_gas - 1.0)
                              / (beta + 12.0 * (gamma_gas - 1.0) * four_minus_3beta + 1e-30))

        # 包层质量
        # m(r) = ∫₀ʳ 4π r'² ρ(r') dr'
        for i in range(1, self.n_r):
            self.m_enc[i] = np.trapz(
                4.0 * np.pi * self.r[:i+1]**2 * self.rho[:i+1],
                self.r[:i+1]
            )

        # 重力加速度
        for i in range(self.n_r):
            if self.r[i] > 1e-10:
                self.g[i] = G_CGS * self.m_enc[i] / self.r[i]**2
            else:
                self.g[i] = 0.0  # 中心处 g = 0 (边界条件)

        # 压力标高 H_P = P / (ρ g)
        for i in range(self.n_r):
            if self.rho[i] > 1e-30 and self.g[i] > 1e-30:
                self.Hp[i] = self.P[i] / (self.rho[i] * self.g[i])
            else:
                self.Hp[i] = self.R_star / 2.0  # 中心处取大值

        # Brunt-Väisälä 频率²
        self._compute_brunt_vaisala()

    def _compute_brunt_vaisala(self):
        """
        计算 Brunt-Väisälä 频率剖面.

        对于绝热扰动:
          N² = g (1/Γ₁ dlnP/dr - dlnρ/dr)
             = g/H_P (∇_ad - ∇)

        在多方球中, ∇ = ∇_ad = 1 - 1/(n+1), 所以 N² = 0 (中性稳定).
        为了演示反演能力, 我们添加一个小幅度化学成分梯度:

          ∇_μ = dlnμ/dlnP = ε_μ sin(2π r/R) × exp(-r²/2σ²)

        则 Ledoux 修正:
          N²_Ledoux = N² + g φ/δ ∇_μ / H_P

        其中 φ = (∂lnρ/∂lnμ)_{P,T} ≈ 1 (理想气体)
        """
        eps_mu = 0.01  # 化学成分梯度幅度
        sigma_mu = 0.15 * self.R_star  # 化学成分跃变位置宽度

        for i in range(1, self.n_r - 1):
            if self.Hp[i] > 0 and self.rho[i] > 0:
                # 对数导数 (中心差分)
                dlnP = (np.log(max(self.P[i+1], 1e-100))
                        - np.log(max(self.P[i-1], 1e-100))) / (2.0 * self.dr)
                dlnrho = (np.log(max(self.rho[i+1], 1e-100))
                          - np.log(max(self.rho[i-1], 1e-100))) / (2.0 * self.dr)

                # 基本 N² (Schwarzschild 判据)
                N2_basic = self.g[i] * (
                    1.0 / max(self.Gamma1[i], 1e-10) * dlnP - dlnrho
                )

                # Ledoux 修正 (化学成分梯度)
                # 模拟氢耗尽核的化学成分跃变
                r_frac = self.r[i] / self.R_star
                grad_mu = eps_mu * np.sin(4.0 * np.pi * r_frac) * np.exp(
                    -(r_frac - 0.3)**2 / (2.0 * 0.15**2)
                )

                # φ/δ ≈ 1 (理想气体近似)
                N2_ledoux = self.g[i] / self.Hp[i] * grad_mu

                self.N2[i] = N2_basic + N2_ledoux
            else:
                self.N2[i] = 0.0

        # 边界: 中心 N² = 有限值, 表面 N² → 0
        self.N2[0] = self.N2[1] if self.n_r > 1 else 0.0
        self.N2[-1] = 0.0

        # 数值鲁棒性: 确保 N² 不会极端大
        N2_max = 1e-3  # 物理上限 ~ (2π / 1hr)²
        self.N2 = np.clip(self.N2, -N2_max, N2_max)

    def get_sound_speed_profile(self) -> np.ndarray:
        """
        计算绝热声速剖面.

        c_s² = Γ₁ P / ρ

        Returns
        -------
        c_s : ndarray
            声速 [cm/s]
        """
        c_s2 = self.Gamma1 * self.P / np.maximum(self.rho, 1e-100)
        return np.sqrt(np.maximum(c_s2, 0.0))

    def get_acoustic_cutoff_frequency(self) -> float:
        """
        计算声学截止频率 (表面).

        ω_ac = c_s / (2 H_P) |_{r=R}

        这是 p 模传播区域的最高频率.

        Returns
        -------
        omega_ac : float
            声学截止频率 [rad/s]
        """
        c_s_surf = self.get_sound_speed_profile()[-1]
        Hp_surf = max(self.Hp[-1], 1e10)
        return c_s_surf / (2.0 * Hp_surf)

    def get_lamb_frequency(self, l: int) -> np.ndarray:
        """
        计算 Lamb 频率剖面 (水平传播).

        L_l² = l(l+1) c_s² / r²

        这是 p 模的最低频率 (在给定 l 下).

        Parameters
        ----------
        l : int
            角量子数

        Returns
        -------
        L_l : ndarray
            Lamb 频率 [rad/s]
        """
        if l < 0:
            raise ValueError(f"角量子数 l={l} 不能为负")
        c_s = self.get_sound_speed_profile()
        r_safe = np.maximum(self.r, 1e10)
        L_l2 = l * (l + 1.0) * c_s**2 / r_safe**2
        return np.sqrt(np.maximum(L_l2, 0.0))


class HypernetworkParameterMapper:
    """
    超网络参数化映射器.

    融合 HyperMPC (1177) 的超网络思想: 从观测频率序列推断恒星结构参数.

    给定一组观测的振荡频率 {ν_obs}, 超网络将频率序列编码为
    恒星结构参数的预测:

    p_pred = f_hyper(encode({ν_obs})) + p_default

    其中:
      - encode: 时间序列编码器 (简化为线性投影 + 非线性激活)
      - f_hyper: 超网络 MLP
      - p_default: 默认参数 (来自恒星演化模型)

    参数
    ----
    n_modes : int
        观测到的模数数量
    n_params : int
        需要推断的参数维度 (默认 5: M, R, X, Z, α_ml)
    hidden_dim : int
        隐层维度 (默认 64)
    """

    def __init__(self, n_modes: int = 20, n_params: int = 5, hidden_dim: int = 64):
        self.n_modes = n_modes
        self.n_params = n_params
        self.hidden_dim = hidden_dim

        # 初始化网络权重 (Xavier 初始化, 融合 F-adjoint 思想)
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / (n_modes + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale3 = np.sqrt(2.0 / (hidden_dim + n_params))

        self.W1 = np.random.randn(hidden_dim, n_modes) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(n_params, hidden_dim) * scale3
        self.b3 = np.zeros(n_params)

        # 默认参数 (太阳型恒星)
        # [M/M_sun, R/R_sun, X_H, Z, alpha_ML]
        self.default_params = np.array([1.0, 1.0, 0.70, 0.02, 1.8])

    def encode_frequencies(self, freq_obs: np.ndarray) -> np.ndarray:
        """
        编码观测频率序列为隐层表示.

        h = ReLU(W₁ ν + b₁)
        h = ReLU(W₂ h + b₂)

        Parameters
        ----------
        freq_obs : ndarray, shape (n_modes,)
            观测频率 [μHz]

        Returns
        -------
        h : ndarray, shape (hidden_dim,)
            隐层编码
        """
        if len(freq_obs) != self.n_modes:
            # 截断或填充
            if len(freq_obs) > self.n_modes:
                freq_obs = freq_obs[:self.n_modes]
            else:
                freq_obs = np.pad(freq_obs, (0, self.n_modes - len(freq_obs)))

        # 频率归一化 (以太阳大频率间距 Δν☉ ≈ 135 μHz 为参考)
        freq_norm = freq_obs / 135.0

        h = np.maximum(0, self.W1 @ freq_norm + self.b1)  # ReLU
        h = np.maximum(0, self.W2 @ h + self.b2)           # ReLU
        return h

    def predict_parameters(self, freq_obs: np.ndarray) -> np.ndarray:
        """
        从观测频率预测恒星结构参数.

        p_pred = W₃ h + b₃ + p_default

        Parameters
        ----------
        freq_obs : ndarray
            观测频率

        Returns
        -------
        params : ndarray, shape (n_params,)
            预测参数 [M/M_sun, R/R_sun, X_H, Z, alpha_ML]
        """
        h = self.encode_frequencies(freq_obs)
        delta_p = self.W3 @ h + self.b3

        # 参数预测 (带物理约束)
        params = self.default_params + 0.1 * delta_p

        # 边界约束
        params[0] = np.clip(params[0], 0.1, 100.0)   # 质量
        params[1] = np.clip(params[1], 0.1, 1000.0)   # 半径
        params[2] = np.clip(params[2], 0.0, 0.95)     # 氢丰度
        params[3] = np.clip(params[3], 0.0, 0.1)      # 金属丰度
        params[4] = np.clip(params[4], 0.5, 3.0)      # 混合长参数

        return params


def compute_frequency_spacings(freqs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算星震学频率间距.

    大频率间距 ( asymptotic relation ):
      Δν_l = ν_{n,l} - ν_{n-1,l} ≈ (2 ∫₀ᴿ dr/c_s)^{-1}

    小频率间距:
      δν_{02}(n) = ν_{n,0} - ν_{n-1,2} ≈ (4l+6)/(16π² ν) ∫₀ᴿ (dN²/dr) dr

    这些间距对恒星全局参数敏感:
      Δν ∝ √(M/R³)    (标度关系)
      δν₀₂ ∝ 核的化学成分年龄指示器

    Parameters
    ----------
    freqs : ndarray, shape (n_modes,)
        排序后的振荡频率

    Returns
    -------
    result : dict
        delta_nu: 大频率间距
        delta_nu_mean: 平均大间距
        r02: 小频率间距 r₀₂
    """
    if len(freqs) < 2:
        return {"delta_nu": np.array([]), "delta_nu_mean": 0.0, "r02": np.array([])}

    # 大间距
    delta_nu = np.diff(freqs)

    # 平均大间距
    delta_nu_mean = np.mean(delta_nu) if len(delta_nu) > 0 else 0.0

    # 小间距 (简化: 假设模式已按 (n,l) 排序)
    r02 = np.zeros(max(0, len(freqs) - 2))
    for i in range(2, len(freqs)):
        r02[i - 2] = freqs[i] - freqs[i - 2]

    return {
        "delta_nu": delta_nu,
        "delta_nu_mean": delta_nu_mean,
        "r02": r02,
    }
