"""
material_properties.py — 量能器材料物理属性数据库
===================================================

本模块提供高能物理量能器模拟所需的材料物理属性。所有属性基于
PDG (Particle Data Group) Review of Particle Physics 2024 标准。

核心物理量:
  X0  : 辐射长度 (radiation length) [cm]
        — 高能电子经bremsstrahlung损失 1/e 能量所需的平均距离
        — X0 = 716.4 * A / (Z*(Z+1)*ln(283/sqrt(Z))) [g/cm^2]

  Ec  : 临界能量 (critical energy) [MeV]
        — 电离损失率 = 辐射损失率时的电子能量
        — Ec ≈ 610 / (Z + 1.24) [MeV]  (近似公式)

  lambda_I : 核相互作用长度 (nuclear interaction length) [cm]
        — 强子发生非弹性核作用前的平均自由程

  rho : 材料密度 [g/cm^3]

关键公式 (Bethe-Heitler / Rossi):
  dE/dx|_brems = -E / X0
  dE/dx|_ion   = -Ec / X0  (在 E = Ec 处)

Molière 半径:
  R_M = X0 * 21.2 [MeV] / Ec  [cm]
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class MaterialSpec:
    """材料物理属性规范 (不可变数据类)"""
    name: str
    Z: int                    # 原子序数
    A: float                  # 质量数 [g/mol]
    rho: float                # 密度 [g/cm^3]
    X0_cm: float              # 辐射长度 [cm]
    Ec_MeV: float             # 临界能量 [MeV]
    lambda_I_cm: float        # 核相互作用长度 [cm]
    dE_dx_min_MeV_cm: float   # 最小电离 dE/dx [MeV/cm]
    chi_c2: float             # Molière 散射参数 chi_c^2 [rad^2/cm]

    @property
    def moliere_radius_cm(self) -> float:
        """Molière 半径 R_M = X0 * 21.2 MeV / Ec [cm]"""
        return self.X0_cm * 21.2 / self.Ec_MeV

    @property
    def X0_g_cm2(self) -> float:
        """辐射长度以质量厚度表示 [g/cm^2]"""
        return self.X0_cm * self.rho

    @property
    def lambda_I_g_cm2(self) -> float:
        """核作用长度以质量厚度表示 [g/cm^2]"""
        return self.lambda_I_cm * self.rho

    def radiation_loss_rate(self, E_MeV: float) -> float:
        """
        Bremsstrahlung 辐射损失率 [MeV/cm]
        dE/dx|_rad = -E / X0
        """
        if E_MeV < 0.0:
            raise ValueError(f"能量必须非负, 得到 E={E_MeV} MeV")
        return E_MeV / self.X0_cm

    def ionization_loss_rate(self, E_MeV: float) -> float:
        """
        电离损失率 (简化 Bethe-Bloch) [MeV/cm]

        对于 E >> Ec:  dE/dx|_ion ≈ Ec/X0 * [1 + 0.1*ln(E/Ec)]
        对于 E << Ec:  dE/dx|_ion ≈ Ec/X0

        完整 Bethe-Bloch 公式:
        -dE/dx = K*z^2*Z/A * (1/beta^2) * [0.5*ln(2*m_e*c^2*beta^2*gamma^2*T_max/I^2)
                  - beta^2 - delta/2]

        其中 K = 0.307075 MeV cm^2/mol, z=1 (入射电子), I ≈ 16*Z^0.8 eV
        """
        if E_MeV <= 0.0:
            return 0.0
        m_e = 0.511  # 电子静止质量 [MeV]
        gamma = 1.0 + E_MeV / m_e
        beta2 = 1.0 - 1.0 / (gamma * gamma)
        beta2 = max(beta2, 1e-12)  # 防止除零

        # 简化近似：在 Ec 附近匹配
        ratio = E_MeV / self.Ec_MeV
        if ratio < 1e-6:
            return self.Ec_MeV / self.X0_cm
        log_corr = 1.0 + 0.08 * math.log(max(ratio, 1e-10))
        return (self.Ec_MeV / self.X0_cm) * log_corr * min(1.0, 0.5 / math.sqrt(beta2))

    def total_loss_rate(self, E_MeV: float) -> float:
        """总能量损失率 dE/dx = (dE/dx)_ion + (dE/dx)_rad [MeV/cm]"""
        return self.ionization_loss_rate(E_MeV) + self.radiation_loss_rate(E_MeV)

    def critical_depth_X0(self, E0_MeV: float) -> float:
        """
        电磁簇射极大值深度 (Rossi-Greisen)
        t_max = ln(E0/Ec) / ln(2)  [辐射长度单位]

        对于光子入射: t_max += 0.5 * X0
        """
        if E0_MeV <= self.Ec_MeV:
            return 0.0
        return math.log(E0_MeV / self.Ec_MeV) / math.log(2.0)

    def shower_max_depth_cm(self, E0_MeV: float) -> float:
        """簇射极大值的物理深度 [cm]"""
        return self.critical_depth_X0(E0_MeV) * self.X0_cm

    def longitudinal_profile(self, t_X0: float, E0_MeV: float) -> float:
        """
        纵向簇射剖面 Gamma(t) — 单位辐射长度的能量沉积

        使用 Gamma 函数参数化 (Longair parameterization):
        Gamma(t) = b * (b*t)^(a-1) * exp(-b*t) / Gamma(a)

        其中:
          a = ln(E0/Ec) / ln(2)  — 形状参数
          b = 0.5                 — 衰减率 (近似)
          t = depth / X0          — 以辐射长度为单位的深度

        归一化: integral_0^inf Gamma(t) dt = E0
        """
        if E0_MeV <= self.Ec_MeV:
            return 0.0
        if t_X0 < 0.0:
            return 0.0
        a = math.log(E0_MeV / self.Ec_MeV) / math.log(2.0)
        b = 0.5  # 近似衰减常数
        a = max(a, 0.1)  # 防止 a 过小导致奇异

        # Gamma 分布: b * (b*t)^(a-1) * exp(-b*t) / Gamma(a)
        # 乘以 E0 归一化到总能量
        from math import lgamma, exp
        if t_X0 < 1e-15:
            if a > 1.0:
                return 0.0
            elif abs(a - 1.0) < 1e-10:
                return E0_MeV * b * exp(-b * t_X0)
            else:
                return float('inf')  # 奇异

        log_profile = (
            math.log(E0_MeV)
            + (a - 1.0) * math.log(b * t_X0)
            - b * t_X0
            - lgamma(a)
            + math.log(b)
        )
        return math.exp(log_profile)


# =============================================================================
# 标准材料库 (PDG 2024 数据)
# =============================================================================

# 铅钨酸盐 (PbWO4) — CMS ECAL 闪烁晶体
PbWO4 = MaterialSpec(
    name="PbWO4",
    Z=73,                    # 有效原子序数 (加权平均)
    A=183.84,                # 有效质量数
    rho=8.28,                # g/cm^3
    X0_cm=0.89,              # cm — 非常短的辐射长度
    Ec_MeV=7.97,             # MeV
    lambda_I_cm=20.6,        # cm — 核作用长度
    dE_dx_min_MeV_cm=1.55,   # MeV/cm
    chi_c2=0.055,            # rad^2/cm (Molière参数)
)

# 硅 (Si) — ATLAS Strip Tracker / Si-W ECAL 概念
Silicon = MaterialSpec(
    name="Silicon",
    Z=14,
    A=28.085,
    rho=2.329,
    X0_cm=9.37,
    Ec_MeV=40.3,
    lambda_I_cm=45.7,
    dE_dx_min_MeV_cm=3.88,
    chi_c2=0.0035,
)

# 液氩 (LAr) — ATLAS EM Calorimeter
LAr = MaterialSpec(
    name="LAr",
    Z=18,
    A=39.948,
    rho=1.396,
    X0_cm=14.06,
    Ec_MeV=34.4,
    lambda_I_cm=83.6,
    dE_dx_min_MeV_cm=2.14,
    chi_c2=0.0018,
)

# 铁 (Fe) — CMS HCAL 吸收体
Iron = MaterialSpec(
    name="Iron",
    Z=26,
    A=55.845,
    rho=7.874,
    X0_cm=1.76,
    Ec_MeV=20.7,
    lambda_I_cm=16.8,
    dE_dx_min_MeV_cm=14.5,
    chi_c2=0.028,
)

# 铜 (Cu) — 常用吸收体
Copper = MaterialSpec(
    name="Copper",
    Z=29,
    A=63.546,
    rho=8.96,
    X0_cm=1.436,
    Ec_MeV=21.5,
    lambda_I_cm=15.2,
    dE_dx_min_MeV_cm=11.6,
    chi_c2=0.032,
)

# 钨 (W) — 最致密稳定金属, 极短 X0
Tungsten = MaterialSpec(
    name="Tungsten",
    Z=74,
    A=183.84,
    rho=19.3,
    X0_cm=0.35,
    Ec_MeV=8.0,
    lambda_I_cm=9.6,
    dE_dx_min_MeV_cm=9.6,
    chi_c2=0.085,
)

# 塑料闪烁体 (Polystyrene近似)
Polystyrene = MaterialSpec(
    name="Polystyrene",
    Z=6,
    A=12.011,
    rho=1.06,
    X0_cm=42.4,
    Ec_MeV=89.0,
    lambda_I_cm=77.2,
    dE_dx_min_MeV_cm=2.08,
    chi_c2=0.0005,
)


# 材料查找表
MATERIAL_LIBRARY: Dict[str, MaterialSpec] = {
    "PbWO4": PbWO4,
    "Silicon": Silicon,
    "LAr": LAr,
    "Iron": Iron,
    "Copper": Copper,
    "Tungsten": Tungsten,
    "Polystyrene": Polystyrene,
}


def get_material(name: str) -> MaterialSpec:
    """从材料库获取材料属性"""
    if name not in MATERIAL_LIBRARY:
        raise KeyError(
            f"材料 '{name}' 不在库中. 可用: {list(MATERIAL_LIBRARY.keys())}"
        )
    return MATERIAL_LIBRARY[name]


def effective_X0_layered(layers: list, thicknesses: list) -> float:
    """
    计算层状量能器的有效辐射长度

    对于 N 层材料, 每层厚度 d_i, 辐射长度 X0_i:
      1/X0_eff = sum_i (f_i / X0_i)
    其中 f_i = d_i / sum(d_j) 为厚度分数

    或者更精确地, 以质量厚度计算:
      t_X0_total = sum_i (d_i / X0_i)  [辐射长度单位]
    """
    if len(layers) != len(thicknesses):
        raise ValueError("层数与厚度列表长度不匹配")
    if not layers:
        raise ValueError("层列表不能为空")
    total_t = 0.0
    for mat, d in zip(layers, thicknesses):
        if d < 0:
            raise ValueError(f"厚度不能为负: d={d}")
        total_t += d / mat.X0_cm
    if total_t < 1e-30:
        raise ValueError("总辐射长度厚度为零")
    total_d = sum(thicknesses)
    return total_d / total_t
