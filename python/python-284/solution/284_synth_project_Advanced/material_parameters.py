"""
material_parameters.py — 二维材料异质结物理参数数据库
=====================================================
核心科学问题: 为过渡金属硫族化合物(TMDC)及III-V族二维材料提供
精确的材料参数, 用于构建异质结能带对齐的物理模型.

融合种子项目:
  - 638_lagrange_nd: 多维参数插值框架用于k空间参数化
  - 1059_Glyphosate_crystallization: 分子拓扑学思想用于层状材料建模

物理模型:
  1. 有效质量 m* 与带隙 E_g 的温度依赖:
     E_g(T) = E_g(0) - alpha * T^2 / (T + beta)  [Varshni公式]
  2. 自旋轨道耦合分裂 Delta_SO
  3. 形变势常数 a, b 用于应变修正
  4. 电子亲和能 chi 用于带偏移计算
"""

import numpy as np


# ============================================================
#  物理常数 (SI 单位制, 部分使用 eV 单位)
# ============================================================
HBAR = 1.054571817e-34        # 约化普朗克常数 [J·s]
M0 = 9.1093837015e-31         # 自由电子质量 [kg]
E0 = 1.602176634e-19          # 元电荷 [C], 也用作 eV→J 转换
KB = 1.380649e-23             # 玻尔兹曼常数 [J/K]
KB_EV = KB / E0               # 玻尔兹曼常数 [eV/K]
EPSILON0 = 8.8541878128e-12   # 真空介电常数 [F/m]
A_ANGSTROM = 1e-10            # 埃 → 米


class Material2D:
    """
    二维材料的物理参数封装.

    参数
    ----
    name : str
        材料名称
    eg_0 : float
        0K 带隙 [eV]
    chi : float
        电子亲和能 [eV]
    me_eff : float
        电子有效质量 [m0]
    mh_eff : float
        空穴有效质量 [m0]
    delta_so : float
        自旋-轨道耦合分裂能 [eV]
    epsilon_r : float
        相对介电常数 (面内)
    epsilon_r_z : float
        相对介电常数 (面外)
    a_lattice : float
        晶格常数 [Å]
    d_coeff : float
        层间距 [Å] (范德华间隙)
    varshni_alpha : float
        Varshni 参数 alpha [eV/K]
    varshni_beta : float
        Varshni 参数 beta [K]
    deformation_a : float
        水静力学形变势 a [eV]
    deformation_b : float
        剪切形变势 b [eV]
    mu_e : float
        电子迁移率 [cm²/(V·s)]
    thickness_mono : float
        单层厚度 [Å]
    """
    def __init__(self, name, eg_0, chi, me_eff, mh_eff, delta_so,
                 epsilon_r, epsilon_r_z, a_lattice, d_coeff,
                 varshni_alpha, varshni_beta,
                 deformation_a, deformation_b, mu_e, thickness_mono):
        self.name = name
        self.eg_0 = eg_0
        self.chi = chi
        self.me_eff = me_eff
        self.mh_eff = mh_eff
        self.delta_so = delta_so
        self.epsilon_r = epsilon_r
        self.epsilon_r_z = epsilon_r_z
        self.a_lattice = a_lattice
        self.d_coeff = d_coeff
        self.varshni_alpha = varshni_alpha
        self.varshni_beta = varshni_beta
        self.deformation_a = deformation_a
        self.deformation_b = deformation_b
        self.mu_e = mu_e
        self.thickness_mono = thickness_mono

    def bandgap_temperature(self, T):
        """
        Varshni 公式计算温度依赖的带隙:
            E_g(T) = E_g(0) - alpha * T^2 / (T + beta)

        参数
        ----
        T : float
            温度 [K]

        返回
        ----
        float
            温度 T 时的带隙 [eV]
        """
        T = max(T, 1e-10)  # 避免除零
        return self.eg_0 - self.varshni_alpha * T**2 / (T + self.varshni_beta)

    def conduction_band_edge(self, T=300.0):
        """导带底能量: E_C = -chi (相对真空能级)"""
        return -self.chi

    def valence_band_edge(self, T=300.0):
        """价带顶能量: E_V = -chi - E_g(T)"""
        return -self.chi - self.bandgap_temperature(T)

    def density_of_states_2d_effective(self, T=300.0):
        """
        二维态密度 (每单位面积每单位能量):
            g_2D = m* / (pi * hbar^2)
        对于导带 (电子):
        """
        m_eff_kg = self.me_eff * M0
        return m_eff_kg / (np.pi * HBAR**2)

    def de_broglie_wavelength(self, T=300.0):
        """
        热德布罗意波长:
            lambda_dB = h / sqrt(2 * pi * m* * kB * T)
        """
        m_eff_kg = self.me_eff * M0
        T = max(T, 1e-10)
        h_planck = 2 * np.pi * HBAR
        return h_planck / np.sqrt(2 * np.pi * m_eff_kg * KB * T)

    def bohr_radius_2d(self):
        """
        二维激子玻尔半径 (修正的里德伯公式):
            a_B* = 4 * pi * epsilon_0 * epsilon_r * hbar^2 / (m_r * e^2)
        其中 m_r 是约化质量.
        """
        m_r = (self.me_eff * self.mh_eff) / (self.me_eff + self.mh_eff) * M0
        return 4 * np.pi * EPSILON0 * self.epsilon_r * HBAR**2 / (m_r * E0**2)

    def exciton_binding_energy(self):
        """
        二维激子结合能 (修正氢模型):
            E_bind = m_r * e^4 / (2 * (4*pi*eps_0*eps_r)^2 * hbar^2)
        简化: E_bind = Ry* = (m_r/m0) * 13.6 eV / epsilon_r^2
        """
        m_r_m0 = (self.me_eff * self.mh_eff) / (self.me_eff + self.mh_eff)
        return m_r_m0 * 13.605698 / (self.epsilon_r**2)

    def __repr__(self):
        return (f"Material2D('{self.name}', Eg={self.eg_0:.3f}eV, "
                f"chi={self.chi:.3f}eV, me*={self.me_eff:.4f}m0)")


# ============================================================
#  已知二维材料参数库 (来自文献实验值)
# ============================================================

# MoS2 单层: 直接带隙 ~1.8 eV (K 谷)
MOS2 = Material2D(
    name="MoS2",
    eg_0=1.940, chi=4.0, me_eff=0.47, mh_eff=0.54,
    delta_so=0.150, epsilon_r=7.5, epsilon_r_z=4.0,
    a_lattice=3.16, d_coeff=6.15,
    varshni_alpha=4.5e-4, varshni_beta=230.0,
    deformation_a=-4.0, deformation_b=-1.8,
    mu_e=200.0, thickness_mono=6.15
)

# WSe2 单层: 直接带隙 ~1.65 eV (K 谷)
WSE2 = Material2D(
    name="WSe2",
    eg_0=1.750, chi=4.2, me_eff=0.32, mh_eff=0.42,
    delta_so=0.460, epsilon_r=8.8, epsilon_r_z=5.0,
    a_lattice=3.28, d_coeff=6.30,
    varshni_alpha=3.8e-4, varshni_beta=185.0,
    deformation_a=-3.5, deformation_b=-1.5,
    mu_e=140.0, thickness_mono=6.30
)

# MoSe2 单层
MOSE2 = Material2D(
    name="MoSe2",
    eg_0=1.650, chi=4.15, me_eff=0.45, mh_eff=0.56,
    delta_so=0.180, epsilon_r=7.0, epsilon_r_z=3.8,
    a_lattice=3.29, d_coeff=6.35,
    varshni_alpha=4.0e-4, varshni_beta=200.0,
    deformation_a=-3.8, deformation_b=-1.6,
    mu_e=160.0, thickness_mono=6.35
)

# WS2 单层
WS2 = Material2D(
    name="WS2",
    eg_0=2.010, chi=3.9, me_eff=0.35, mh_eff=0.41,
    delta_so=0.430, epsilon_r=8.0, epsilon_r_z=4.5,
    a_lattice=3.15, d_coeff=6.18,
    varshni_alpha=4.2e-4, varshni_beta=210.0,
    deformation_a=-3.2, deformation_b=-1.4,
    mu_e=180.0, thickness_mono=6.18
)

# hBN 单层: 大带隙绝缘体
HBN = Material2D(
    name="hBN",
    eg_0=6.000, chi=2.0, me_eff=0.80, mh_eff=1.20,
    delta_so=0.010, epsilon_r=3.0, epsilon_r_z=2.5,
    a_lattice=2.50, d_coeff=3.33,
    varshni_alpha=9.0e-4, varshni_beta=1100.0,
    deformation_a=-5.0, deformation_b=-2.5,
    mu_e=0.01, thickness_mono=3.33
)

# GaSe 单层
GASE = Material2D(
    name="GaSe",
    eg_0=2.100, chi=3.8, me_eff=0.20, mh_eff=0.30,
    delta_so=0.230, epsilon_r=10.0, epsilon_r_z=6.0,
    a_lattice=3.76, d_coeff=8.00,
    varshni_alpha=5.0e-4, varshni_beta=180.0,
    deformation_a=-2.8, deformation_b=-1.2,
    mu_e=100.0, thickness_mono=8.00
)


# ============================================================
#  异质结参数计算工具
# ============================================================

def anderson_band_offset(mat1, mat2, T=300.0):
    """
    Anderson 规则计算 Type-II 异质结带偏移.

    电子亲和能规则 (Anderson's rule):
        Delta_Ec = chi_1 - chi_2       (导带偏移)
        Delta_Ev = E_g2(T) - E_g1(T) + Delta_Ec  (价带偏移)

    参数
    ----
    mat1, mat2 : Material2D
        两种二维材料
    T : float
        温度 [K]

    返回
    ----
    delta_Ec : float
        导带偏移 [eV]
    delta_Ev : float
        价带偏移 [eV]
    """
    eg1 = mat1.bandgap_temperature(T)
    eg2 = mat2.bandgap_temperature(T)
    delta_Ec = mat1.chi - mat2.chi
    delta_Ev = eg2 - eg1 + delta_Ec
    return delta_Ec, delta_Ev


def strain_biaxial(material, a_substrate):
    """
    双轴应变计算:
        epsilon_xx = epsilon_yy = (a_sub - a_mat) / a_mat
        epsilon_zz = -2 * (C12/C11) * epsilon_xx  (Poisson 效应)

    参数
    ----
    material : Material2D
        外延层材料
    a_substrate : float
        衬底晶格常数 [Å]

    返回
    ----
    eps_xx : float
        面内应变
    eps_zz : float
        面外应变 (近似, 假设 C12/C11 ≈ 0.5)
    """
    eps_xx = (a_substrate - material.a_lattice) / material.a_lattice
    # 近似泊松比 nu = C12/(C11+C12) ≈ 0.3 → C12/C11 ≈ 0.43
    poisson_ratio = 0.43
    eps_zz = -2 * poisson_ratio * eps_xx / (1 - poisson_ratio)
    return eps_xx, eps_zz


def strain_energy_shift(material, eps_xx, eps_yy):
    """
    形变势理论计算应变导致的带边移动:
        Delta_E_c = a_c * (eps_xx + eps_yy + eps_zz)
        Delta_E_v = a_v * (eps_xx + eps_yy + eps_zz)
                  ± b * sqrt((eps_xx-eps_yy)^2/4 + ...)

    参数
    ----
    material : Material2D
    eps_xx, eps_yy : float
        主应变分量

    返回
    ----
    delta_Ec : float
        导带能量移动 [eV]
    delta_Ev : float
        价带能量移动 [eV] (取绝对值较大的分量)
    """
    eps_hydro = eps_xx + eps_yy  # 面内水静力学应变
    # 面外应变近似 (泊松效应)
    eps_zz = -2 * 0.43 * eps_hydro / (1 - 0.43)
    eps_total = eps_hydro + eps_zz

    # 水静力学形变势
    a_c = material.deformation_a * 0.5   # 导带形变势
    a_v = -material.deformation_a * 0.5  # 价带形变势 (符号相反)

    delta_Ec = a_c * eps_total
    delta_Ev = a_v * eps_total

    # 剪切形变势贡献 (简化)
    if abs(eps_xx - eps_yy) > 1e-12:
        shear_split = material.deformation_b * abs(eps_xx - eps_yy)
        delta_Ev = delta_Ev + shear_split  # 取较大值

    return delta_Ec, delta_Ev


def lattice_mismatch(mat1, mat2):
    """
    晶格失配率:
        f = (a2 - a1) / a1
    """
    return (mat2.a_lattice - mat1.a_lattice) / mat1.a_lattice


def tunneling_effective_mass(mat1, mat2, carrier='electron'):
    """
    隧穿有效质量 (Burt-Foreman 规则):
        1/m_t* = (1/m1* + 1/m2*) / 2
    或更精确地用调和平均:
        m_t* = 2*m1**m2* / (m1* + m2*)
    """
    if carrier == 'electron':
        m1 = mat1.me_eff
        m2 = mat2.me_eff
    else:
        m1 = mat1.mh_eff
        m2 = mat2.mh_eff
    return 2 * m1 * m2 / (m1 + m2)


def screen_coulomb_interaction(mat, r, z_separation):
    """
    Keldysh 屏蔽库仑势 (二维材料修正):
        V(r) = -e^2 / (8 * eps_0 * eps_r * r_0) *
               [H0(r/r_0) - Y0(r/r_0)]
    其中 r_0 是屏蔽长度, H0, Y0 是 Struve 和 Neumann 函数.

    简化为 Yukawa 形式:
        V(r) = -e^2 / (4*pi*eps_0*eps_r) * exp(-r/r_0) / r
    """
    from scipy.special import struve, y0 as neumann_y0

    r_0 = mat.bohr_radius_2d() * 0.5  # Keldysh 屏蔽长度近似
    r = max(r, 1e-15)  # 避免除零
    rho = r / r_0

    # 使用渐近形式避免数值问题
    if rho > 10:
        return 0.0  # 远距离屏蔽
    elif rho < 1e-6:
        # 近距离近似
        return -E0 / (4 * np.pi * EPSILON0 * mat.epsilon_r * r_0)
    else:
        # Keldysh 势
        h0_val = float(struve(0, rho))
        y0_val = float(neumann_y0(rho))
        prefactor = -E0 / (8 * np.pi * EPSILON0 * mat.epsilon_r * r_0)
        return prefactor * (h0_val - y0_val)


def heterostructure_database():
    """
    返回常用异质结组合的参数字典.
    """
    return {
        "MoS2/WSe2": (MOS2, WSE2),
        "MoS2/MoSe2": (MOS2, MOSE2),
        "WSe2/WS2": (WSE2, WS2),
        "MoS2/hBN": (MOS2, HBN),
        "WSe2/GaSe": (WSE2, GASE),
        "MoSe2/WSe2": (MOSE2, WSE2),
    }
