"""
energetic_particle_kinetics.py - 高能粒子动力学方程求解器

本模块模拟高能粒子 (fast ions) 与阿尔芬波的共振相互作用。
核心算法融合了以下种子项目的思想:
  - zombie_ode (1434): 多室 ODE 模型与守恒量追踪
  - blip_bio-model (1095): 多尺度 LBM 速度空间离散与碰撞算子
  - asa144 (042): 约束分布重构

物理背景:
  高能粒子在环形磁场中的运动由漂移动力学方程描述:

  ∂f/∂t + v_∥ b·∇f + v_d·∇f + (μ/m)(b·∇B) ∂f/∂v_∥ = C[f] + S

  其中:
    f = f(r, θ, v_∥, μ, t): 分布函数
    v_∥: 平行速度
    μ = m v_⊥²/(2B): 磁矩 (绝热不变量)
    v_d: 漂移速度 (曲率 + ∇B 漂移)
    C[f]: 碰撞算子 (Fokker-Planck)
    S: 源项 (NBI 注入或 ICRF 加热)

  共振条件:
    ω - n Ω_φ - k_∥ v_∥ = 0  (通行粒子)
    ω - n Ω_φ - p Ω_b = 0     (捕获粒子)
  其中 Ω_φ 为环向进动频率，Ω_b 为弹跳频率。

  高能粒子对阿尔芬波的影响通过介电张量修正:
    δω = -ω (β_fast/2) <v_d · δE / (ω - ω_d)>

数值方法:
  速度空间离散: 类 LBM 的离散速度格子
  碰撞算子: 多松弛时间 (MRT) 格式
  源项: 固定分布注入
  守恒量: 粒子数、能量、磁矩

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np


class EnergeticParticleDistribution:
    """
    高能粒子分布函数在 (v_∥, v_⊥) 速度空间中的离散表示。

    使用离散速度格子 (类似 LBM 的 D2Q9 但用于速度空间)。

    速度空间网格:
      v_∥ ∈ [-v_max, v_max], N_∥ 个格子
      v_⊥ ∈ [0, v_max], N_⊥ 个格子

    守恒量:
      N = Σ f_i w_i : 粒子数密度
      E = Σ (1/2) m (v_∥² + v_⊥²) f_i w_i : 能量密度
      M = Σ (m v_⊥²/(2B)) f_i w_i : 磁矩密度
    """

    def __init__(self, n_v_parallel=16, n_v_perp=8, v_max=None,
                 mass=1.67262192e-27, B0=5.0):
        """
        参数:
          n_v_parallel: int, 平行速度方向格子数
          n_v_perp: int, 垂直速度方向格子数
          v_max: float, 最大速度 [m/s]
          mass: float, 粒子质量 [kg]
          B0: float, 参考磁场 [T]
        """
        self.n_vp = n_v_parallel
        self.n_vt = n_v_perp
        self.mass = mass
        self.B0 = B0

        if v_max is None:
            v_max = 3.0e6  # 默认 ~100 keV 氘核
        self.v_max = v_max

        # 速度空间网格
        self.v_parallel = np.linspace(-v_max, v_max, n_v_parallel)
        self.v_perp = np.linspace(0.0, v_max, n_v_perp)
        self.dv_p = self.v_parallel[1] - self.v_parallel[0] if n_v_parallel > 1 else 1.0
        self.dv_t = self.v_perp[1] - self.v_perp[0] if n_v_perp > 1 else 1.0

        # 权重 (v_⊥ 空间需要 2π v_⊥ 因子)
        self.weights = np.zeros((n_v_parallel, n_v_perp))
        for i in range(n_v_parallel):
            for j in range(n_v_perp):
                v_t = self.v_perp[j]
                self.weights[i, j] = self.dv_p * 2.0 * np.pi * v_t * self.dv_t

        # 分布函数 f(v_∥, v_⊥)
        self.f = np.zeros((n_v_parallel, n_v_perp))

        # 碰撞参数
        self.nu_collision = 1.0e2  # 碰撞频率 [s^-1]

    def initialize_maxwellian_fast(self, n_fast, T_fast, v_drift=0.0):
        """
        用移位麦克斯韦分布初始化高能粒子。

        f₀(v_∥, v_⊥) = n_fast (m/(2π T_fast))^{3/2}
                        exp(-m((v_∥ - v_drift)² + v_⊥²)/(2 T_fast))

        参数:
          n_fast: float, 高能粒子数密度 [m^-3]
          T_fast: float, 高能粒子温度 [eV]
          v_drift: float, 平行漂移速度 [m/s]
        """
        T_joule = T_fast * 1.60217663e-19  # eV → J

        for i in range(self.n_vp):
            for j in range(self.n_vt):
                vp = self.v_parallel[i]
                vt = self.v_perp[j]
                v_sq = (vp - v_drift) ** 2 + vt ** 2

                coeff = n_fast * (self.mass / (2.0 * np.pi * T_joule)) ** 1.5
                self.f[i, j] = coeff * np.exp(-self.mass * v_sq / (2.0 * T_joule))

    def conserved_quantities(self):
        """
        计算分布函数的守恒量。

        粒子数密度: N = ∫∫ f(v_∥, v_⊥) 2π v_⊥ dv_∥ dv_⊥
        能量密度: E = ∫∫ (1/2)m(v_∥²+v_⊥²) f 2π v_⊥ dv_∥ dv_⊥
        磁矩密度: M = ∫∫ (m v_⊥²/(2B)) f 2π v_⊥ dv_∥ dv_⊥

        改编自 zombie_conserved 的守恒量追踪思想。

        返回:
          dict, 包含 'density', 'energy', 'momentum', 'magnetic_moment'
        """
        density = 0.0
        energy = 0.0
        momentum = 0.0
        mag_moment = 0.0

        for i in range(self.n_vp):
            for j in range(self.n_vt):
                vp = self.v_parallel[i]
                vt = self.v_perp[j]
                w = self.weights[i, j]
                fval = self.f[i, j]

                v_sq = vp ** 2 + vt ** 2
                density += fval * w
                energy += 0.5 * self.mass * v_sq * fval * w
                momentum += self.mass * vp * fval * w
                mag_moment += (self.mass * vt ** 2 / (2.0 * self.B0)) * fval * w

        return {
            'density': density,
            'energy': energy,
            'momentum': momentum,
            'magnetic_moment': mag_moment,
        }

    def apply_collision_operator_mrt(self, dt):
        """
        施加多松弛时间 (MRT) 碰撞算子。

        改编自 blip_bio-model (1095) 的 MRT-LBM 碰撞。

        碰撞算子在矩空间中:
          Ĉ = -M⁻¹ S M (f - f_eq)

        其中 M 为矩变换矩阵，S 为松弛矩阵。

        对于简化版本，使用 BGK 单松弛:
          f^{n+1} = f^n - ν Δt (f^n - f_eq)

        参数:
          dt: float, 时间步长 [s]
        """
        # 计算当前矩
        conserved = self.conserved_quantities()
        n_density = conserved['density']

        if n_density < 1e-30:
            return  # 密度太低，跳过碰撞

        # 计算等效温度 (从能量)
        T_eff_j = conserved['energy'] / (1.5 * n_density) if n_density > 0 else 1.0
        T_eff_eV = T_eff_j / 1.60217663e-19

        if T_eff_eV < 1.0:
            T_eff_eV = 1.0

        # 平均漂移速度
        v_drift_avg = conserved['momentum'] / (self.mass * n_density) if n_density > 0 else 0.0

        # 平衡分布 (局部麦克斯韦)
        f_eq = np.zeros_like(self.f)
        T_joule = T_eff_eV * 1.60217663e-19
        coeff = n_density * (self.mass / (2.0 * np.pi * T_joule)) ** 1.5

        for i in range(self.n_vp):
            for j in range(self.n_vt):
                vp = self.v_parallel[i]
                vt = self.v_perp[j]
                v_sq = (vp - v_drift_avg) ** 2 + vt ** 2
                f_eq[i, j] = coeff * np.exp(-self.mass * v_sq / (2.0 * T_joule))

        # BGK 碰撞: f → f - ν Δt (f - f_eq)
        nu = self.nu_collision
        relaxation = nu * dt / (1.0 + nu * dt)  # 保证数值稳定
        self.f = self.f - relaxation * (self.f - f_eq)

        # 确保非负
        self.f = np.maximum(self.f, 0.0)

    def compute_drift_frequency(self, r_norm, q_safety, m_mode, n_mode):
        """
        计算高能粒子的漂移共振频率。

        通行粒子漂移频率:
          ω_d = (v_∥² + v_⊥²/2) / (Ω_c R₀) * (n - m/q)

        弹跳频率 (捕获粒子):
          ω_b = π v_∥ / (q R₀ √(2ε))

        共振条件:
          ω - n ω_φ - p ω_b = 0

        参数:
          r_norm: float, 归一化半径
          q_safety: float, 安全因子
          m_mode: int, 极向模数
          n_mode: int, 环向模数

        返回:
          omega_drift: ndarray, shape (n_vp, n_vt), 漂移频率
        """
        R0 = 1.65  # 简化
        epsilon = r_norm  # 反转比

        omega_drift = np.zeros((self.n_vp, self.n_vt))
        omega_ci = 1.60217663e-19 * self.B0 / self.mass  # 简化回旋频率

        for i in range(self.n_vp):
            for j in range(self.n_vt):
                vp = self.v_parallel[i]
                vt = self.v_perp[j]
                v_sq = vp ** 2 + vt ** 2

                # 漂移频率
                v_drift = v_sq / (2.0 * omega_ci * R0) if omega_ci > 0 else 0.0
                k_parallel = (n_mode / R0) * (1.0 - m_mode / (n_mode * q_safety + 1e-30))

                omega_drift[i, j] = v_drift * k_parallel

        return omega_drift

    def wave_particle_power_exchange(self, wave_phi, omega_wave, k_parallel):
        """
        计算波-粒子功率交换。

        P = -∫∫ e φ ω Re[f₁/(ω - k_∥v_∥)] v_⊥ dv_∥ dv_⊥

        简化版本使用共振分母的 Lorentzian 正则化:
          1/(ω - k_∥v_∥) → (ω - k_∥v_∥) / ((ω - k_∥v_∥)² + γ²)

        参数:
          wave_phi: float, 波电势振幅
          omega_wave: float, 波频率 [rad/s]
          k_parallel: float, 平行波数 [1/m]

        返回:
          power: float, 功率交换 [W/m³]
          growth_rate: float, 增长率估计 [s^-1]
        """
        gamma = 0.01 * abs(omega_wave) + 1.0  # 正则化参数

        power = 0.0
        for i in range(self.n_vp):
            for j in range(self.n_vt):
                vp = self.v_parallel[i]
                vt = self.v_perp[j]
                w = self.weights[i, j]

                resonance_denom = omega_wave - k_parallel * vp
                lorentzian = resonance_denom / (resonance_denom ** 2 + gamma ** 2)

                # 功率: 与 ∂f₀/∂v_∥ 有关
                power += self.f[i, j] * w * lorentzian * vp

        power *= wave_phi ** 2 * self.mass

        # 增长率估计
        growth_rate = power / (abs(wave_phi) ** 2 + 1e-30) if abs(wave_phi) > 0 else 0.0

        return power, growth_rate


class MultiSpeciesEPDriver:
    """
    多种高能粒子群体的耦合驱动。

    改编自 zombie_ode (1434) 的多室 ODE 思想。
    不同能量/种类的高能粒子作为独立的"室"(compartment)，
    通过波-粒子相互作用和碰撞相互耦合。

    ODE 系统:
      dN_k/dt = S_k - L_k N_k - Σ_j C_{kj}
      dE_k/dt = P_k - Q_k E_k - Σ_j D_{kj}

    其中 k, j 为不同粒子群体索引。
    守恒量: Σ N_k = const, Σ E_k = const (无源时)
    """

    def __init__(self, n_species=3, v_parallel_bins=16, v_perp_bins=8):
        self.n_species = n_species
        self.species = []
        energies_keV = [30.0, 80.0, 200.0]

        for k in range(n_species):
            e_fast = energies_keV[k % len(energies_keV)] * 1e3
            dist = EnergeticParticleDistribution(
                n_v_parallel=v_parallel_bins,
                n_v_perp=v_perp_bins,
                v_max=np.sqrt(2.0 * e_fast * 1.60217663e-19 / 1.67262192e-27) * 1.2,
            )
            dist.initialize_maxwellian_fast(
                n_fast=1e17,
                T_fast=e_fast,
                v_drift=0.3 * dist.v_max,
            )
            self.species.append(dist)

    def total_conserved(self):
        """
        计算所有群体的总守恒量。
        """
        total = {'density': 0.0, 'energy': 0.0, 'momentum': 0.0, 'magnetic_moment': 0.0}
        for sp in self.species:
            q = sp.conserved_quantities()
            for key in total:
                total[key] += q[key]
        return total

    def evolve_collisions(self, dt):
        """
        对所有群体施加碰撞算子。
        """
        for sp in self.species:
            sp.apply_collision_operator_mrt(dt)

    def summary(self):
        """
        返回各群体状态的摘要。
        """
        lines = []
        for k, sp in enumerate(self.species):
            q = sp.conserved_quantities()
            lines.append(
                f"  群体 {k}: N={q['density']:.3e}, "
                f"E={q['energy']:.3e}, "
                f"P={q['momentum']:.3e}"
            )
        return "\n".join(lines)
