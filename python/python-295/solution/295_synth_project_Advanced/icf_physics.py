#!/usr/bin/env python3
"""
icf_physics.py
==============
惯性约束聚变 (ICF) 内爆物理核心模块。

物理模型:
  - 二维轴对称 (r-z) 柱坐标可压缩流体方程
  - 理想气体 + 多方状态方程 (EOS)
  - Rankine-Hugoniot 激波跳跃条件
  - Braginskii 粘性与电子热传导
  - 辐射扩散 (多群扩散近似)
  - P_n 球谐矩分解用于内爆对称性度量

核心方程 (轴对称 Euler + 粘性 + 热传导):
  ∂ρ/∂t + ∇·(ρu) = 0
  ∂(ρu)/∂t + ∇·(ρu⊗u) = -∇p + ∇·τ + ρg
  ∂(ρE)/∂t + ∇·[(ρE+p)u] = ∇·(κ∇T) + ∇·(τ·u) + Q_nuc - Q_rad

状态方程 (理想气体):
  p = (γ-1) ρ e_int,  e_int = p/[(γ-1)ρ]
  T = p M / (ρ R_gas)

Rankine-Hugoniot 条件:
  ρ₂/ρ₁ = (γ+1)M_s² / [(γ-1)M_s² + 2]
  p₂/p₁ = 2γ M_s²/(γ+1) - (γ-1)/(γ+1)

Laser 能量沉积 (逆韧致吸收):
  Q_laser = α_IB n_e² / (n_c T_e^{3/2}) × I(r,z) × exp(-∫α_IB ds)

其中 α_IB 为逆韧致吸收系数, n_c = ε₀ m_e ω_l²/e² 为临界密度。
"""

import numpy as np
import math

# =========================================================================
#  物理常数 (CGS)
# =========================================================================
K_B_CGS = 1.380649e-16        # Boltzmann 常数 [erg/K]
M_PROTON = 1.67262192e-24     # 质子质量 [g]
M_ELECTRON = 9.1093837e-28    # 电子质量 [g]
E_CHARGE = 4.80320451e-10     # 电子电荷 [esu]
C_LIGHT = 2.99792458e10       # 光速 [cm/s]
SIGMA_SB = 5.6704e-5          # Stefan-Boltzmann 常数 [erg/(cm² s K⁴)]
H_PLANCK = 6.62607015e-27     # Planck 常数 [erg·s]
AVOGADRO = 6.02214076e23      # Avogadro 常数
R_GAS_CGS = 8.314462618e7     # 气体常数 [erg/(mol·K)]
EPSILON_0 = 8.8541878128e-12  # 真空介电常数 [F/m]

# =========================================================================
#  ICF 典型参数
# =========================================================================
DEFAULT_GAMMA = 5.0 / 3.0           # 多方指数 (单原子理想气体)
DEFAULT_DT_FUEL_MASS = 2.5 * M_PROTON  # DT 燃料平均离子质量
DEFAULT_AB_MASS = 2.0 * 12.0 * M_PROTON  # 烧蚀层 (CH) 平均质量
DEFAULT_LASER_WAVELENGTH = 3.51e-5  # 三倍频 Nd:glass 激光波长 [cm] (351 nm)
DEFAULT_LASER_INTENSITY = 1.0e15    # 激光峰值强度 [W/cm²]
DEFAULT_CAPSULE_RADIUS = 1.0e-2     # 典型靶丸半径 [cm] = 100 μm
DEFAULT_FUEL_DENSITY = 0.25         # 初始 DT 冰层密度 [g/cm³]
DEFAULT_ABLATOR_DENSITY = 1.0       # 烧蚀层密度 [g/cm³]


class ICFPhysics:
    """
    ICF 内爆物理参数与方程封装。
    包含状态方程、激波关系、输运系数与激光耦合。
    """

    def __init__(self, gamma=DEFAULT_GAMMA, mean_mass=DEFAULT_DT_FUEL_MASS,
                 capsule_radius=DEFAULT_CAPSULE_RADIUS,
                 fuel_density=DEFAULT_FUEL_DENSITY):
        """
        初始化 ICF 物理参数。

        参数:
            gamma: 多方指数 γ
            mean_mass: 平均粒子质量 [g]
            capsule_radius: 靶丸初始半径 [cm]
            fuel_density: 初始燃料密度 [g/cm³]
        """
        if gamma <= 1.0:
            raise ValueError(f"多方指数 γ={gamma} 必须 > 1 (热力学稳定性要求)")
        if mean_mass <= 0:
            raise ValueError(f"平均粒子质量必须为正, got {mean_mass}")
        if capsule_radius <= 0:
            raise ValueError(f"靶丸半径必须为正, got {capsule_radius}")
        if fuel_density <= 0:
            raise ValueError(f"燃料密度必须为正, got {fuel_density}")

        self.gamma = gamma
        self.gm1 = gamma - 1.0           # γ - 1
        self.gp1 = gamma + 1.0           # γ + 1
        self.mean_mass = mean_mass
        self.capsule_radius = capsule_radius
        self.fuel_density = fuel_density
        self.cv = K_B_CGS / (self.gm1 * mean_mass)   # 定容比热 [erg/(g·K)]

    # -----------------------------------------------------------------
    #  状态方程
    # -----------------------------------------------------------------
    def pressure(self, rho, e_int):
        """
        理想气体状态方程:
          p = (γ-1) ρ e_int

        参数:
            rho: 密度 [g/cm³]
            e_int: 比内能 [erg/g]
        返回:
            压强 [dyn/cm²]
        """
        rho = np.asarray(rho, dtype=np.float64)
        e_int = np.asarray(e_int, dtype=np.float64)
        # 物理边界: 密度与内能不能为负
        rho = np.maximum(rho, 1.0e-30)
        e_int = np.maximum(e_int, 0.0)
        return self.gm1 * rho * e_int

    def temperature(self, rho, p):
        """
        由密度与压强计算温度:
          T = p M / (ρ R_gas)

        参数:
            rho: 密度 [g/cm³]
            p: 压强 [dyn/cm²]
        返回:
            温度 [K]
        """
        rho = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
        p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
        return p * self.mean_mass / (rho * R_GAS_CGS)

    def sound_speed(self, rho, p):
        """
        等熵声速:
          c_s = sqrt(γ p / ρ)

        参数:
            rho: 密度 [g/cm³]
            p: 压强 [dyn/cm²]
        返回:
            声速 [cm/s]
        """
        rho = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
        p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
        return np.sqrt(self.gamma * p / rho)

    def specific_entropy(self, rho, p):
        """
        比熵 (对数形式):
          s = ln(p / ρ^γ) + const
        用于监测激波不可逆性与 RT 不稳定性混合区。
        """
        rho = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
        p = np.maximum(np.asarray(p, dtype=np.float64), 1.0e-30)
        return np.log(p) - self.gamma * np.log(rho)

    # -----------------------------------------------------------------
    #  Rankine-Hugoniot 激波跳跃
    # -----------------------------------------------------------------
    def rankine_hugoniot(self, mach_s):
        """
        计算正激波后的密度比、压强比、温度比。

        Rankine-Hugoniot 关系:
          ρ₂/ρ₁ = (γ+1) M_s² / [(γ-1) M_s² + 2]
          p₂/p₁ = 2γ M_s² / (γ+1) - (γ-1)/(γ+1)
          T₂/T₁ = [2γ M_s² - (γ-1)] [(γ-1)M_s² + 2] / [(γ+1)² M_s²]

        参数:
            mach_s: 激波 Mach 数 (无量纲, M_s ≥ 1)
        返回:
            dict: {rho_ratio, p_ratio, T_ratio, u_particle}
            u_particle 为波后粒子速度 (相对于波前静止参考系)
        """
        if mach_s < 1.0:
            raise ValueError(f"激波 Mach 数必须 ≥ 1, got {mach_s}")
        ms2 = mach_s ** 2
        gm1, gp1, gam = self.gm1, self.gp1, self.gamma

        rho_ratio = gp1 * ms2 / (gm1 * ms2 + 2.0)
        p_ratio = 2.0 * gam * ms2 / gp1 - gm1 / gp1
        T_ratio = (2.0 * gam * ms2 - gm1) * (gm1 * ms2 + 2.0) / (gp1 ** 2 * ms2)

        # 波后粒子速度 (相对于未扰动介质)
        # u_p = c_s1 × (1 - ρ₁/ρ₂) × M_s  其中 c_s1 为波前声速
        # 此处返回的是 u_p / c_s1 的无量纲形式
        u_particle_norm = (1.0 - 1.0 / rho_ratio) * mach_s

        return {
            'rho_ratio': rho_ratio,
            'p_ratio': p_ratio,
            'T_ratio': T_ratio,
            'u_particle_norm': u_particle_norm
        }

    # -----------------------------------------------------------------
    #  输运系数
    # -----------------------------------------------------------------
    def braginskii_viscosity(self, rho, T, Z_eff=1.0):
        """
        Braginskii 离子粘性 (各向同性部分):
          η_i = 0.96 n_i k_B T τ_i
          τ_i = 3√(m_i) (k_B T)^{3/2} / (4√π n_i Z⁴ e⁴ lnΛ)

        其中 lnΛ 为 Coulomb 对数。

        参数:
            rho: 密度 [g/cm³]
            T: 温度 [K]
            Z_eff: 有效电荷数
        返回:
            η_i: 离子粘性系数 [g/(cm·s)] = [P]
        """
        rho = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
        T = np.maximum(np.asarray(T, dtype=np.float64), 1.0)
        n_i = rho / self.mean_mass  # 离子数密度 [cm⁻³]
        n_e = Z_eff * n_i           # 电子数密度

        # Coulomb 对数 (经典等离子体)
        # lnΛ = 24 - ln(√(n_e[cm⁻³]) / T[eV])
        T_eV = K_B_CGS * T / E_CHARGE  # 温度转换为 eV
        T_eV = np.maximum(T_eV, 1.0e-6)
        ln_lambda = np.maximum(24.0 - np.log(np.sqrt(n_e) / T_eV), 2.0)

        # 离子碰撞时间 τ_i [s]
        tau_i = (3.0 * np.sqrt(self.mean_mass) * (K_B_CGS * T) ** 1.5 /
                 (4.0 * np.sqrt(np.pi) * n_i * Z_eff ** 4 * E_CHARGE ** 4 * ln_lambda))

        # Braginskii 粘性系数
        eta_i = 0.96 * n_i * K_B_CGS * T * tau_i
        return eta_i

    def electron_thermal_conductivity(self, rho, T, Z_eff=1.0):
        """
        Spitzer-Härm 电子热导率:
          κ_e = 3.2 n_e k_B T_e τ_e / m_e × (k_B T_e)^{5/2} 系数
        简化 Spitzer 公式:
          κ_SH ≈ 1.84×10⁻⁵ T_e^{5/2} / (Z lnΛ)  [erg/(s·cm·K)]
        """
        rho = np.maximum(np.asarray(rho, dtype=np.float64), 1.0e-30)
        T = np.maximum(np.asarray(T, dtype=np.float64), 1.0)
        n_i = rho / self.mean_mass
        n_e = Z_eff * n_i

        T_eV = np.maximum(K_B_CGS * T / E_CHARGE, 1.0e-6)
        ln_lambda = np.maximum(24.0 - np.log(np.sqrt(n_e) / T_eV), 2.0)

        # Spitzer-Härm 热导率 [erg/(s·cm·K)]
        kappa_sh = 1.84e-5 * T ** 2.5 / (Z_eff * ln_lambda)

        # 热通量限制 (自由流限制): q ≤ f n_e k_B T_e c_{se}
        # f ≈ 0.1 为通量限制因子
        c_se = np.sqrt(K_B_CGS * T / M_ELECTRON)  # 电子热速度
        q_fs = 0.1 * n_e * K_B_CGS * T * c_se
        # 等效热导率限制
        grad_T_typical = T / self.capsule_radius  # 特征温度梯度
        if grad_T_typical > 0:
            kappa_limit = q_fs / grad_T_typical
            kappa_eff = np.minimum(kappa_sh, kappa_limit)
        else:
            kappa_eff = kappa_sh

        return kappa_eff

    # -----------------------------------------------------------------
    #  激光能量沉积
    # -----------------------------------------------------------------
    def inverse_bremsstrahlung_coeff(self, n_e, T_e, Z_eff=1.0,
                                      lambda_laser=DEFAULT_LASER_WAVELENGTH):
        """
        逆韧致吸收系数 (CGS):
          α_IB = (n_e n_i Z² e⁶ lnΛ) / (3 (2π)^{1/2} ε₀² c m_e^{1/2} (k_B T_e)^{3/2} ω_l²)

        简化为:
          α_IB [cm⁻¹] ≈ C × n_e² Z lnΛ / (T_e^{3/2} λ_l² n_c)

        其中 n_c = π m_e c² / (e² λ_l²) 为临界密度。
        """
        n_e = np.maximum(np.asarray(n_e, dtype=np.float64), 1.0)
        T_e = np.maximum(np.asarray(T_e, dtype=np.float64), 1.0)
        T_eV = np.maximum(K_B_CGS * T_e / E_CHARGE, 1.0e-6)
        n_i = n_e / max(Z_eff, 1.0)

        # Coulomb 对数
        ln_lambda = np.maximum(24.0 - np.log(np.sqrt(n_e) / T_eV), 2.0)

        # 临界密度 [cm⁻³]
        omega_l = 2.0 * np.pi * C_LIGHT / lambda_laser
        n_crit = EPSILON_0 * M_ELECTRON * omega_l ** 2 / E_CHARGE ** 2
        # 单位转换: SI → CGS
        n_crit_cgs = n_crit * 1.0e-6  # m⁻³ → cm⁻³

        # 吸收系数 (简化公式, 量级正确)
        # α_IB ≈ (e⁶ lnΛ)/(12π ε₀³ c m_e² ω_l²) × n_e n_i / T_e^{3/2}
        coeff = (E_CHARGE ** 6 * ln_lambda /
                 (12.0 * np.pi * (1.0 / (4.0 * np.pi)) ** 3 *
                  C_LIGHT * M_ELECTRON ** 2 * omega_l ** 2))
        alpha = coeff * n_e * n_i * Z_eff / (T_e ** 1.5 + 1.0e-30)
        return np.maximum(alpha, 0.0)

    def critical_density(self, lambda_laser=DEFAULT_LASER_WAVELENGTH):
        """
        临界密度 n_c = ε₀ m_e ω_l² / e²
        """
        omega_l = 2.0 * np.pi * C_LIGHT / lambda_laser
        n_crit = EPSILON_0 * M_ELECTRON * omega_l ** 2 / E_CHARGE ** 2
        return n_crit * 1.0e-6  # → cm⁻³

    # -----------------------------------------------------------------
    #  P_n 球谐矩分解 (内爆对称性)
    # -----------------------------------------------------------------
    @staticmethod
    def legendre_mode_decomposition(data_on_sphere, theta_pts, mode_orders=(0, 2, 4, 6)):
        """
        对球面数据进行 P_n Legendre 矩分解。
        用于量化 ICF 内爆对称性偏离程度。

        内爆对称性度量:
          A_n(t) = (2n+1)/2 ∫₀^π f(θ,t) P_n(cos θ) sin θ dθ

        其中 P_n 为 n 阶 Legendre 多项式。
        关键模式: P₂ (椭球不对称), P₄ (四极不对称), P₆ (六极不对称)
        理想内爆要求 |A_n/A₀| < 1% for n ≥ 2。

        参数:
            data_on_sphere: 球面上的数据 f(θ), shape=(N_theta,)
            theta_pts: 极角 θ ∈ [0, π], shape=(N_theta,)
            mode_orders: 需要计算的模式阶数
        返回:
            dict: {mode_order: amplitude}
        """
        data = np.asarray(data_on_sphere, dtype=np.float64)
        theta = np.asarray(theta_pts, dtype=np.float64)
        mu = np.cos(theta)
        sin_theta = np.sin(theta)

        # Gauss-Legendre 积分权重 (简化: 使用梯形法则)
        dmu = np.gradient(mu)
        amplitudes = {}
        for n in mode_orders:
            pn = np.polynomial.legendre.legval(mu, [0] * n + [1])
            amplitude = (2.0 * n + 1.0) / 2.0 * np.sum(data * pn * sin_theta * np.abs(dmu))
            amplitudes[n] = amplitude
        return amplitudes

    # -----------------------------------------------------------------
    #  无量纲参数
    # -----------------------------------------------------------------
    def reynolds_number(self, rho, u, L, eta):
        """Re = ρ u L / η"""
        eta = np.maximum(np.asarray(eta, dtype=np.float64), 1.0e-30)
        return np.asarray(rho) * np.asarray(u) * np.asarray(L) / eta

    def peclet_number(self, rho, u, L, kappa, cp):
        """Pe = ρ u L c_p / κ"""
        kappa = np.maximum(np.asarray(kappa, dtype=np.float64), 1.0e-30)
        return np.asarray(rho) * np.asarray(u) * np.asarray(L) * cp / kappa

    def knudsen_number(self, rho, T, L, Z_eff=1.0):
        """
        Knudsen 数 Kn = λ_mfp / L
        其中 λ_mfp 为离子平均自由程。
        Kn > 0.01 时需要动理学修正。
        """
        n_i = np.maximum(np.asarray(rho, dtype=np.float64) / self.mean_mass, 1.0)
        T = np.maximum(np.asarray(T, dtype=np.float64), 1.0)
        T_eV = np.maximum(K_B_CGS * T / E_CHARGE, 1.0e-6)
        n_e = Z_eff * n_i
        ln_lambda = np.maximum(24.0 - np.log(np.sqrt(n_e) / T_eV), 2.0)
        sigma_ii = np.pi * (E_CHARGE ** 2 / (K_B_CGS * T)) ** 2 * ln_lambda
        lambda_mfp = 1.0 / (np.sqrt(2.0) * n_i * sigma_ii + 1.0e-30)
        return lambda_mfp / np.maximum(np.asarray(L, dtype=np.float64), 1.0e-30)


# =========================================================================
#  便捷函数
# =========================================================================
def create_default_physics():
    """创建默认 ICF 物理参数实例"""
    return ICFPhysics()


def compute_sedov_solution(E0, rho0, gamma, t):
    """
    Sedov-Taylor 自相似解 (点爆炸问题, 用于验证 ICF 内爆激波)。
    r_s(t) = ξ₀ (E₀ t² / ρ₀)^{1/5}

    参数:
        E0: 爆炸能量 [erg]
        rho0: 环境密度 [g/cm³]
        gamma: 多方指数
        t: 时间 [s]
    返回:
        激波半径 [cm], 激波速度 [cm/s]
    """
    if t <= 0:
        return 0.0, 0.0
    xi0 = 1.15167  # γ = 5/3 时的自相似常数
    r_s = xi0 * (E0 * t ** 2 / rho0) ** 0.2
    v_s = 2.0 / 5.0 * r_s / t
    return r_s, v_s


if __name__ == '__main__':
    phys = ICFPhysics()
    print(f"ICF 物理模块初始化成功:")
    print(f"  γ = {phys.gamma:.4f}")
    print(f"  平均粒子质量 = {phys.mean_mass:.4e} g")
    print(f"  靶丸半径 = {phys.capsule_radius:.4e} cm")
    print(f"  c_v = {phys.cv:.4e} erg/(g·K)")

    # Rankine-Hugoniot 测试
    rh = phys.rankine_hugoniot(3.0)
    print(f"\nRankine-Hugoniot (M_s=3):")
    print(f"  ρ₂/ρ₁ = {rh['rho_ratio']:.4f}")
    print(f"  p₂/p₁ = {rh['p_ratio']:.4f}")
    print(f"  T₂/T₁ = {rh['T_ratio']:.4f}")

    # Sedov 解测试
    r_s, v_s = compute_sedov_solution(1e15, 1.0, 5.0 / 3.0, 1e-6)
    print(f"\nSedov-Taylor (t=1μs):")
    print(f"  r_s = {r_s:.4e} cm")
    print(f"  v_s = {v_s:.4e} cm/s")
