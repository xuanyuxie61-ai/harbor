"""
plasma_config.py - 等离子体物理参数与阿尔芬波模拟配置模块

本模块定义阿尔芬波-高能粒子相互作用模拟所需的全部物理参数，
包括平衡磁场参数、等离子体密度与温度、高能粒子参数，
以及数值离散参数。

物理背景:
  考虑一个简化圆柱形托卡马克平衡，磁场为:
    B = B_φ e_φ + B_θ e_θ
  其中 B_φ 为环向场，B_θ 为极向场。

  安全因子: q(r) = r B_φ / (R B_θ)
  阿尔芬速度: v_A = B / √(μ₀ n_i m_i)
  离子回旋频率: Ω_ci = e B / m_i
  离子惯性长度: d_i = c / ω_pi

  高能粒子参数:
    能量: E_fast = (1/2) m_f v_f²
    纵横比: Λ = v_⊥²/(v_⊥² + v_∥²) (投掷角参数)
    漂移频率: ω_d = (v_⊥²/2 + v_∥²) / (Ω_ci R)

核心公式:
  阿尔芬波色散关系 (理想 MHD):
    ω² = k_∥² v_A²

  含高能粒子修正的色散关系:
    ω² = k_∥² v_A² (1 + δf_EP)
  其中 δf_EP 为高能粒子对介电常数的贡献。

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np


class PlasmaParameters:
    """
    封装阿尔芬波-高能粒子相互作用模拟的物理参数。

    包括磁约束参数、等离子体热背景参数、
    高能粒子 (fast ion) 参数和数值离散参数。
    """

    def __init__(self, **kwargs):
        # ===== 基础物理常数 (SI 单位) =====
        self.mu_0 = 4.0e-7 * np.pi          # 真空磁导率 [H/m]
        self.e_charge = 1.60217663e-19      # 元电荷 [C]
        self.m_proton = 1.67262192e-27      # 质子质量 [kg]
        self.m_deuterium = 2.0 * self.m_proton  # 氘核质量 [kg]
        self.c_light = 2.99792458e8         # 光速 [m/s]
        self.epsilon_0 = 8.8541878128e-12   # 真空介电常数 [F/m]
        self.k_B = 1.380649e-23             # 玻尔兹曼常数 [J/K]

        # ===== 磁平衡参数 =====
        self.B0 = kwargs.get('B0', 5.0)                # 轴上环向磁场 [T]
        self.R0 = kwargs.get('R0', 1.65)               # 大半径 [m]
        self.a_minor = kwargs.get('a_minor', 0.5)      # 小半径 [m]
        self.q0 = kwargs.get('q0', 1.0)                # 轴上安全因子
        self.qa = kwargs.get('qa', 3.0)                # 边界安全因子

        # ===== 热等离子体参数 =====
        self.n_e = kwargs.get('n_e', 5.0e19)           # 电子密度 [m^-3]
        self.T_e = kwargs.get('T_e', 3.0e3)            # 电子温度 [eV]
        self.T_i = kwargs.get('T_i', 2.5e3)            # 离子温度 [eV]
        self.Z_eff = kwargs.get('Z_eff', 1.5)          # 有效电荷数
        self.m_ion = kwargs.get('m_ion', self.m_deuterium)  # 主离子质量 [kg]

        # ===== 高能粒子参数 =====
        self.n_fast = kwargs.get('n_fast', 1.0e17)     # 高能粒子密度 [m^-3]
        self.E_fast = kwargs.get('E_fast', 100.0e3)    # 高能粒子能量 [eV]
        self.m_fast = kwargs.get('m_fast', self.m_proton)   # 高能粒子质量 [kg]
        self.pitch_angle = kwargs.get('pitch_angle', 0.7)   # 特征投掷角 cos(θ)
        self.Z_fast = kwargs.get('Z_fast', 1)          # 高能粒子电荷数

        # ===== 数值离散参数 =====
        self.n_r = kwargs.get('n_r', 64)               # 径向网格点数
        self.n_theta = kwargs.get('n_theta', 32)       # 极向网格点数
        self.n_phi = kwargs.get('n_phi', 16)           # 环向网格点数
        self.fd_order = kwargs.get('fd_order', 4)      # 有限差分阶数
        self.cfl_number = kwargs.get('cfl_number', 0.3) # CFL 数
        self.n_timesteps = kwargs.get('n_timesteps', 200)  # 总时间步数
        self.output_interval = kwargs.get('output_interval', 20)  # 输出间隔

        # ===== 碰撞与耗散参数 =====
        self.eta_resistivity = kwargs.get('eta_resistivity', 1.0e-6)  # 电阻率 [Ω·m]
        self.nu_ion_coll = kwargs.get('nu_ion_coll', 1.0e3)  # 离子碰撞频率 [s^-1]
        self.nu_fast_coll = kwargs.get('nu_fast_coll', 1.0e2)  # 高能粒子碰撞频率 [s^-1]

        # ===== 稳定性分析参数 =====
        self.n_eigenmodes = kwargs.get('n_eigenmodes', 8)   # 计算的本征模数量
        self.stability_tolerance = kwargs.get('stability_tolerance', 1e-10)

        self._validate_parameters()
        self._compute_derived_quantities()

    def _validate_parameters(self):
        """
        验证物理参数的合理性和数值离散参数的边界条件。
        """
        if self.B0 <= 0.0:
            raise ValueError(f"磁场 B0 = {self.B0} T 必须为正")
        if self.R0 <= 0.0:
            raise ValueError(f"大半径 R0 = {self.R0} m 必须为正")
        if self.a_minor <= 0.0 or self.a_minor >= self.R0:
            raise ValueError(f"小半径 a = {self.a_minor} m 必须满足 0 < a < R0")
        if self.n_e <= 0.0:
            raise ValueError(f"电子密度 n_e 必须为正")
        if self.T_e <= 0.0:
            raise ValueError(f"电子温度 T_e 必须为正")
        if self.q0 <= 0.0 or self.qa <= 0.0:
            raise ValueError("安全因子必须为正")
        if self.n_r < 5:
            raise ValueError(f"径向网格点 n_r = {self.n_r} 不足 (至少 5)")
        if self.n_theta < 4:
            raise ValueError(f"极向网格点 n_theta = {self.n_theta} 不足 (至少 4)")
        if self.n_phi < 2:
            raise ValueError(f"环向网格点 n_phi = {self.n_phi} 不足 (至少 2)")
        if self.fd_order not in [2, 4, 6, 8]:
            raise ValueError(f"有限差分阶数 {self.fd_order} 不受支持 (使用 2,4,6,8)")
        if self.cfl_number <= 0.0 or self.cfl_number > 1.0:
            raise ValueError(f"CFL 数 {self.cfl_number} 必须在 (0, 1]")
        if self.pitch_angle < 0.0 or self.pitch_angle > 1.0:
            raise ValueError(f"投掷角余弦 {self.pitch_angle} 必须在 [0, 1]")

    def _compute_derived_quantities(self):
        """
        计算所有导出物理量。

        关键导出量:
          v_A = B₀ / √(μ₀ n_i m_i)  : 阿尔芬速度
          Ω_ci = e B₀ / m_i          : 离子回旋频率
          d_i = c / ω_pi              : 离子惯性长度
          β = 2μ₀ n k_B T / B₀²      : 等离子体比压
          v_fast = √(2 E_fast / m_fast) : 高能粒子速度
          ρ_s = c_s / Ω_ci            : 离子声回旋半径
        """
        # 离子数密度 (准中性条件: n_i ≈ n_e / Z_eff)
        self.n_i = self.n_e / self.Z_eff

        # 阿尔芬速度 [m/s]
        self.v_alfven = self.B0 / np.sqrt(self.mu_0 * self.n_i * self.m_ion)

        # 离子回旋频率 [rad/s]
        self.omega_ci = self.Z_fast * self.e_charge * self.B0 / self.m_ion

        # 离子等离子体频率 [rad/s]
        self.omega_pi = np.sqrt(self.n_i * self.e_charge**2 / (self.epsilon_0 * self.m_ion))

        # 离子惯性长度 [m]
        self.d_i = self.c_light / self.omega_pi

        # 离子声速 [m/s]
        self.c_s = np.sqrt(self.e_charge * (self.T_e + self.T_i) / self.m_ion)

        # 离子声回旋半径 [m]
        self.rho_s = self.c_s / self.omega_ci

        # 热离子速度 [m/s]
        self.v_th_i = np.sqrt(2.0 * self.e_charge * self.T_i / self.m_ion)

        # 等离子体比压 β
        self.beta = 2.0 * self.mu_0 * self.n_e * self.k_B * (self.T_e + self.T_i) / (self.B0 ** 2)

        # 高能粒子速度 [m/s]
        self.v_fast = np.sqrt(2.0 * self.E_fast * self.e_charge / self.m_fast)

        # 高能粒子回旋频率 [rad/s]
        self.omega_cf = self.Z_fast * self.e_charge * self.B0 / self.m_fast

        # 高能粒子拉莫尔半径 [m]
        self.rho_Lf = self.v_fast * np.sqrt(1.0 - self.pitch_angle**2) / self.omega_cf

        # 高能粒子密度比
        self.density_ratio = self.n_fast / self.n_i

        # 高能粒子 β_fast
        self.beta_fast = 2.0 * self.mu_0 * self.n_fast * self.E_fast * self.e_charge / (self.B0 ** 2)

        # 安全因子剖面系数: q(r) = q0 + (qa - q0) * (r/a)^2
        self.q_profile_coeff = self.qa - self.q0

        # 特征阿尔芬时间 [s]: τ_A = R₀ / v_A
        self.tau_alfven = self.R0 / self.v_alfven

        # 径向网格间距 [m]
        self.dr = self.a_minor / max(self.n_r - 1, 1)

        # 极向网格间距 [rad]
        self.dtheta = 2.0 * np.pi / self.n_theta

        # 环向网格间距 [rad]
        self.dphi = 2.0 * np.pi / self.n_phi

        # 最大时间步长由 CFL 条件确定: Δt < CFL * Δr / v_A
        self.dt_max = self.cfl_number * self.dr / self.v_alfven
        self.dt = self.dt_max  # 使用最大允许时间步

        # 模拟总时间 [s]
        self.total_time = self.n_timesteps * self.dt

        # 特征频率: 阿尔芬波频率 f_A = v_A / (2π R₀) [Hz]
        self.f_alfven = self.v_alfven / (2.0 * np.pi * self.R0)

    def q_profile(self, r_norm):
        """
        计算安全因子剖面 q(r)。

        采用简单抛物线模型:
          q(r) = q₀ + (q_a - q₀) * (r/a)²

        更真实的剖面可使用:
          q(r) = q₀ + (q_a - q₀) * [1 - (1 - (r/a)²)²]  (反向剪切)

        参数:
          r_norm: float or ndarray, 归一化半径 r/a ∈ [0, 1]

        返回:
          q: float or ndarray, 安全因子
        """
        r_norm = np.asarray(r_norm, dtype=float)
        q = self.q0 + self.q_profile_coeff * r_norm ** 2
        return q

    def magnetic_shear(self, r_norm):
        """
        计算磁剪切 s(r) = (r/q) * dq/dr。

        磁剪切决定阿尔芬本征模的径向包络宽度:
          Δ_r / a ~ 1 / √(n s)

        在低剪切区域 (如反剪切), 阿尔芬本征模
        可以扩展到大范围，增强与高能粒子的相互作用。

        参数:
          r_norm: float or ndarray, 归一化半径

        返回:
          s: float or ndarray, 磁剪切
        """
        r_norm = np.asarray(r_norm, dtype=float)
        q = self.q_profile(r_norm)
        # dq/dr = 2(q_a - q₀) r/a² → (r/q) dq/dr = 2(q_a - q₀)(r/a)² / q
        dq_dr_norm = 2.0 * self.q_profile_coeff * r_norm
        s = np.where(np.abs(q) > 1e-14, r_norm * dq_dr_norm / q, 0.0)
        return s

    def alfven_continuum_frequency(self, r_norm, n_toroidal=1):
        """
        计算阿尔芬连续谱频率。

        阿尔芬连续谱由磁场非均匀性产生，在环几何中:
          ω_A(r) = |k_∥| v_A = |1/q(r) - m/n| * (n/R₀) * v_A

        对于给定的环向模数 n，连续谱在极向模数 m 上
        形成一组分支，在 q = m/n 的有理面处连续谱间隙
        (Toroidicity-induced Alfvén Eigenmode gap) 出现。

        参数:
          r_norm: float or ndarray, 归一化半径
          n_toroidal: int, 环向模数 n

        返回:
          omega_continuum: ndarray, shape (n_modes, len(r_norm)), 连续谱频率 [rad/s]
        """
        r_norm = np.asarray(r_norm, dtype=float)
        q = self.q_profile(r_norm)

        # 计算多个极向模的连续谱分支
        m_min = 1
        m_max = 6
        n_modes = m_max - m_min + 1
        omega_continuum = np.zeros((n_modes, len(r_norm)))

        for i, m in enumerate(range(m_min, m_max + 1)):
            # k_∥ = (1/R₀)(n - m/q) = (n/R₀)(1 - m/(nq))
            k_parallel = np.abs(n_toroidal / self.R0) * np.abs(1.0 - m / (n_toroidal * q + 1e-30))
            omega_continuum[i] = k_parallel * self.v_alfven

        return omega_continuum

    def summary(self):
        """
        返回等离子体参数的文本摘要。
        """
        lines = [
            "=" * 60,
            "阿尔芬波-高能粒子相互作用模拟参数摘要",
            "=" * 60,
            f"  磁场强度 B₀ = {self.B0:.2f} T",
            f"  大半径 R₀ = {self.R0:.2f} m",
            f"  小半径 a = {self.a_minor:.2f} m",
            f"  反转比 ε = a/R₀ = {self.a_minor/self.R0:.4f}",
            f"  安全因子范围 q₀ = {self.q0:.2f} ~ q_a = {self.qa:.2f}",
            f"  电子密度 n_e = {self.n_e:.2e} m⁻³",
            f"  电子温度 T_e = {self.T_e:.1f} eV",
            f"  离子温度 T_i = {self.T_i:.1f} eV",
            f"  阿尔芬速度 v_A = {self.v_alfven:.2e} m/s",
            f"  离子回旋频率 Ω_ci = {self.omega_ci:.2e} rad/s",
            f"  离子惯性长度 d_i = {self.d_i:.4f} m",
            f"  离子声回旋半径 ρ_s = {self.rho_s:.5f} m",
            f"  等离子体比压 β = {self.beta:.4e}",
            f"  高能粒子密度 n_f = {self.n_fast:.2e} m⁻³",
            f"  高能粒子能量 E_f = {self.E_fast/1e3:.0f} keV",
            f"  高能粒子速度 v_f = {self.v_fast:.2e} m/s",
            f"  高能粒子拉莫尔半径 ρ_Lf = {self.rho_Lf:.5f} m",
            f"  高能粒子密度比 n_f/n_i = {self.density_ratio:.4e}",
            f"  高能粒子 β_f = {self.beta_fast:.4e}",
            f"  阿尔芬特征时间 τ_A = {self.tau_alfven:.2e} s",
            f"  空间离散: {self.n_r} × {self.n_theta} × {self.n_phi} (r × θ × φ)",
            f"  有限差分阶数: {self.fd_order}",
            f"  时间步长 Δt = {self.dt:.2e} s (CFL = {self.cfl_number:.2f})",
            f"  总时间步: {self.n_timesteps}, 总模拟时间 = {self.total_time:.2e} s",
            "=" * 60,
        ]
        return "\n".join(lines)
