"""
多铁性材料物理常数与材料参数模块
=================================
对应种子项目: 755_mesh_etoe (网格拓扑 → 钙钛矿晶格几何)
              1306_triangle_histogram (三角剖分 → 布里渊区三角网格基础)

物理背景:
    BiFeO3 (BFO) 是最著名的室温多铁性材料, 属于钙钛矿结构 (ABO3).
    其晶格常数 a ≈ 3.96 Å, 在菱形相中 α ≈ 89.3°.
     Néel 温度 T_N ≈ 643 K, Curie 温度 T_C ≈ 1103 K.

本模块提供:
    - 基本物理常数 (真空介电常数、玻尔兹曼常数、旋磁比等)
    - BiFeO3 材料参数 (Landau 系数、梯度系数、电致伸缩系数等)
    - 外场参数 (温度、应力、外加电磁场)
    - 网格参数与单位转换

核心公式:
    Landau 展开系数温度依赖性:
        α_i(T) = α_i^0 * (T - T_C,i)    [Curie-Weiss 律]
    旋磁比:
        γ = g * μ_B / ℏ                  [g ≈ 2 为 Landé 因子]
    磁晶各向异 field:
        H_K = 2 * K_1 / (μ_0 * M_s)
"""

import numpy as np


# ============================================================
# 基本物理常数 (SI 单位制)
# ============================================================

# 真空介电常数 ε₀ (F/m)
EPSILON_0 = 8.8541878128e-12

# 真空磁导率 μ₀ (H/m)
MU_0 = 4.0e-7 * np.pi

# 玻尔兹曼常数 k_B (J/K)
K_BOLTZMANN = 1.380649e-23

# 约化普朗克常数 ℏ (J·s)
HBAR = 1.054571817e-34

# 基本电荷 e (C)
E_CHARGE = 1.602176634e-19

# 玻尔磁子 μ_B (J/T)
MU_BOHR = 9.2740100783e-24

# Landé g 因子 (自由电子近似)
LANDE_G = 2.00231930436256

# 旋磁比 γ₀ = g·μ_B/ℏ (rad/(s·T))
GYROMAGNETIC_RATIO = LANDE_G * MU_BOHR / HBAR

# 光速 c (m/s)
C_LIGHT = 299792458.0


# ============================================================
# BiFeO3 晶格参数
# ============================================================

class BiFeO3Lattice:
    """
    BiFeO3 钙钛矿晶格几何.

    立方钙钛矿原型: a₀ ≈ 3.96 Å
    菱形畸变: α_rhomb ≈ 89.3° (偏离 90° 的角度表征铁电极化)

    菱方→正交转换:
        a_rhomb = a_cubic / sqrt(2 * (1 - cos(α)))
        c_rhomb = a_cubic * sqrt(3 * (1 + 2*cos(α))) / sqrt(1 - cos(α))

    自发极化方向: 沿 [111]_cubic (菱方体对角线)
    |P_s| ≈ 0.95 C/m² (实验值, T = 300 K)
    """

    def __init__(self):
        # 立方晶格常数 (m)
        self.a_cubic = 3.96e-10

        # 菱形角度 (rad)
        self.alpha_rhomb = 89.3 * np.pi / 180.0

        # 菱形晶格常数
        cos_a = np.cos(self.alpha_rhomb)
        self.a_rhomb = self.a_cubic / np.sqrt(2.0 * (1.0 - cos_a))
        self.c_rhomb = (self.a_cubic *
                        np.sqrt(3.0 * (1.0 + 2.0 * cos_a)) /
                        np.sqrt(1.0 - cos_a))

        # 单胞体积 (m³)
        self.V_cell = (self.a_rhomb ** 3) * np.sqrt(
            1.0 - 3.0 * cos_a ** 2 + 2.0 * cos_a ** 3
        )

        # Fe³⁺ 自旋 (S = 5/2, 高自旋 d⁵ 组态)
        self.spin_Fe = 2.5

        # Fe³⁺ 磁矩 (μ_B)
        self.mu_Fe = self.spin_Fe * LANDE_G * MU_BOHR

        # 饱和磁化强度 M_s (A/m), 估算:
        # M_s = n * μ_Fe / V_cell, n=2 formula units per rhomb cell
        self.M_saturation = 2.0 * self.mu_Fe / self.V_cell

        # 自发极化 (C/m²)
        self.P_spontaneous = 0.95

        # [111] 方向单位矢量
        self.n111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


# ============================================================
# BiFeO3 Landau-Ginzburg-Devonshire 参数
# ============================================================

class BiFeO3LGD:
    """
    BiFeO3 的 Landau-Ginzburg-Devonshire 自由能展开参数.

    自由能密度展开:
        F = F_Landau + F_elastic + F_gradient + F_electric + F_magnetic + F_ME

    第二阶 Landau 系数 (温度依赖):
        α₁(T) = α₁⁰ · (T - T_C1)
    其中 T_C1 ≈ 1103 K (铁电 Curie 温度)

    第四阶 Landau 系数:
        α₁₁, α₁₂ (与极化四次方项相关)

    第六阶 Landau 系数:
        α₁₁₁, α₁₁₂ (保证自由能有下界)

    梯度系数:
        G₁₁, G₁₂ (极化梯度能)

    参考:
        - Chen, Y.L. et al., Phys. Rev. B 86, 024106 (2012)
        - Hong, Z. et al., Phys. Rev. B 83, 144107 (2011)
    """

    def __init__(self):
        # ---- 第二阶 Landau 系数 ----
        # α₁⁰ (V·m/(C·K)), 采用文献拟合值
        self.alpha1_0 = 3.2e7

        # Curie 温度 (K)
        self.T_C1 = 1103.0

        # 等效 α₃ (对于菱方相, α₃ = α₁)
        self.alpha3_0 = self.alpha1_0

        # ---- 第四阶 Landau 系数 (J·m⁵/C⁴) ----
        self.alpha11 = -1.2e8
        self.alpha12 = 3.6e8
        self.alpha33 = self.alpha11  # 菱方对称性

        # ---- 第六阶 Landau 系数 (J·m⁹/C⁶) ----
        self.alpha111 = 6.0e9
        self.alpha112 = -2.5e9
        self.alpha123 = -3.0e9
        self.alpha333 = self.alpha111

        # ---- 梯度系数 (J·m³/C²) ----
        self.G11 = 1.6e-10
        self.G12 = 0.8e-10
        self.G44 = 0.8e-10

        # ---- 磁各向异性常数 ----
        # K₁ (J/m³), BiFeO3 的反铁磁各向异性
        self.K1_mag = 3.5e4

        # ---- 磁电耦合系数 ----
        # α_ME (s/m), 线性磁电耦合
        self.alpha_ME_linear = 1.2e-10

        # 双线性-双铁电耦合 γ (J·m⁴/(C²·A))
        self.gamma_ME_biquadratic = 5.0e-4

        # ---- 动力学系数 ----
        # Gilbert 阻尼 α_G (无量纲)
        self.alpha_Gilbert = 0.05

        # LGD 动力学系数 L_P (m/(V·s))
        self.L_P = 1.0e-4

        # ---- 饱和磁化 (A/m) ----
        # BiFeO3 为 G-type 反铁磁体, 但有微弱铁磁分量
        # 有效饱和磁化: M_s ≈ 0 (反铁磁), 微弱铁磁矩 ~0.028 μ_B/Fe
        # 此处使用有效值用于唯象计算
        self.M_saturation = 4.0e4  # A/m (有效值)

        # Néel 温度 (K)
        self.T_Neel = 643.0

        # ---- 电致伸缩系数 (m⁴/C²) ----
        self.Q11 = 0.08
        self.Q12 = -0.025
        self.Q44 = 0.04

        # ---- 弹性柔顺常数 (m²/N) ----
        self.s11 = 5.6e-12
        self.s12 = -1.4e-12
        self.s44 = 14.0e-12

    def alpha1_temperature(self, T):
        """
        计算温度依赖的第二阶 Landau 系数.

        α₁(T) = α₁⁰ · (T - T_C1)

        物理含义:
            T > T_C1: α₁ > 0, 顺电相稳定 (P = 0)
            T < T_C1: α₁ < 0, 铁电相稳定 (P ≠ 0)
            T = T_C1: 二级相变点

        参数:
            T: 温度 (K)

        返回:
            α₁(T) (V·m/C)
        """
        return self.alpha1_0 * (T - self.T_C1)

    def alpha3_temperature(self, T):
        """计算 α₃(T), 菱方相中与 α₁ 等价的系数."""
        return self.alpha3_0 * (T - self.T_C1)

    def effective_alpha(self, T, sigma=0.0):
        """
        有效第二阶系数 (含应力修正).

        α₁* = α₁(T) - (Q₁₁ + 2·Q₁₂) · σ / (s₁₁ + s₁₂)

        其中 σ 为面内应力, 通过弹性柔顺常数耦合.
        应力可通过外延应变调控相结构 (应变工程).

        参数:
            T: 温度 (K)
            sigma: 面内应力 (Pa)

        返回:
            α₁* (V·m/C)
        """
        alpha_T = self.alpha1_temperature(T)
        Q_sum = self.Q11 + 2.0 * self.Q12
        s_sum = self.s11 + self.s12
        if abs(s_sum) < 1e-30:
            return alpha_T
        stress_correction = Q_sum * sigma / s_sum
        return alpha_T - stress_correction


# ============================================================
# 模拟参数配置
# ============================================================

class SimulationConfig:
    """
    模拟计算的全局配置参数.

    空间离散:
        - nx, ny: 网格点数 (小规模可复现实验, 通常 32-128)
        - Lx, Ly: 物理域尺寸 (nm)
        - dx = Lx/(nx-1), dy = Ly/(ny-1)

    时间离散:
        - dt: 时间步长 (s)
        - n_steps: 总步数
        - scheme: 时间积分格式 ('FTCS', 'semi_implicit', 'RK4')

    稳定性条件 (FTCS):
        dt ≤ dx² / (4 · L_P · (G₁₁ + G₁₂))

    边界条件:
        - P_boundary: 'periodic', 'dirichlet', 'neumann'
    """

    def __init__(self):
        # 空间网格
        self.nx = 64
        self.ny = 64
        self.Lx = 100e-9   # 100 nm
        self.Ly = 100e-9   # 100 nm
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        # 时间参数
        self.dt = 1e-13       # 0.1 ps
        self.n_steps = 500
        self.scheme = 'semi_implicit'

        # 温度
        self.temperature = 300.0  # K

        # 外加场
        self.E_ext = np.array([0.0, 0.0, 0.0])   # V/m
        self.H_ext = np.array([0.0, 0.0, 0.0])    # A/m

        # 边界条件
        self.P_boundary = 'periodic'
        self.M_boundary = 'periodic'

        # 随机种子
        self.seed = 285

        # 输出频率
        self.output_interval = 50

        # 有限差分阶数
        self.fd_order = 6  # 6阶精度

    def stability_number(self, lgd_params):
        """
        计算 CFL 稳定性数 (FTCS 格式).

        S = L_P · (2G₁₁ + G₁₂) · dt · (1/dx² + 1/dy²)

        稳定性条件: S ≤ 0.5

        参数:
            lgd_params: BiFeO3LGD 实例

        返回:
            S: CFL 数 (无量纲)
        """
        G_eff = 2.0 * lgd_params.G11 + lgd_params.G12
        inv_dx2 = 1.0 / (self.dx ** 2)
        inv_dy2 = 1.0 / (self.dy ** 2)
        S = lgd_params.L_P * G_eff * self.dt * (inv_dx2 + inv_dy2)
        return S

    def validate_stability(self, lgd_params, verbose=True):
        """
        验证时间步长是否满足稳定性条件.

        若不满足, 自动调整 dt 使 S ≤ 0.4 (留安全余量).

        返回:
            调整后的 dt
        """
        G_eff = 2.0 * lgd_params.G11 + lgd_params.G12
        inv_dx2 = 1.0 / (self.dx ** 2)
        inv_dy2 = 1.0 / (self.dy ** 2)

        if G_eff * (inv_dx2 + inv_dy2) < 1e-30:
            return self.dt

        dt_max = 0.4 / (lgd_params.L_P * G_eff * (inv_dx2 + inv_dy2))

        if self.dt > dt_max:
            if verbose:
                print(f"  [稳定性] dt 从 {self.dt:.3e} 调整为 {dt_max:.3e} s")
            self.dt = dt_max

        return self.dt


# ============================================================
# 单位转换工具
# ============================================================

def eV_to_J(eV):
    """电子伏特 → 焦耳"""
    return eV * E_CHARGE


def J_to_eV(J):
    """焦耳 → 电子伏特"""
    return J / E_CHARGE


def angstrom_to_m(A):
    """埃 → 米"""
    return A * 1e-10


def m_to_angstrom(m):
    """米 → 埃"""
    return m * 1e10


def nm_to_m(nm):
    """纳米 → 米"""
    return nm * 1e-9


def K_to_energy(T):
    """温度 → 热能 (J)"""
    return K_BOLTZMANN * T


def polarization_SI_to_cgs(P_si):
    """极化强度: C/m² → μC/cm²"""
    return P_si * 0.1


def magnetization_SI_to_emu(M_si):
    """磁化强度: A/m → emu/cm³"""
    return M_si * 1e-3
