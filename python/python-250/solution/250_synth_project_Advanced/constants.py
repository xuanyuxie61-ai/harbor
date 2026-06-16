"""
物理常数 (CGS 单位制).

超新星辐射流体模拟所需的物理常数。所有量均采用 CGS 高斯单位制，
以便与标准天体物理文献 (如 Bruenn 1985, Janka 2017) 对照。
"""
from __future__ import annotations
import math

# 基本常数
C_LIGHT       = 2.99792458e10      # cm / s
K_BOLTZMANN   = 1.380649e-16       # erg / K
H_PLANCK      = 6.62607015e-27     # erg * s
H_BAR         = H_PLANCK / (2.0 * math.pi)
M_PROTON      = 1.67262192369e-24  # g
M_ELECTRON    = 9.1093837015e-28   # g
M_NEUTRON     = 1.67492749804e-24  # g
Q_ELECTRON    = 4.80320425e-10     # statC (esu)
N_AVOGADRO    = 6.02214076e23      # 1 / mol

# 引力常数
G_GRAV        = 6.67430e-8         # cm^3 / g / s^2
M_SUN         = 1.98892e33         # g
R_SUN         = 6.957e10           # cm
KM_TO_CM      = 1.0e5
PC_TO_CM      = 3.085677581e18

# 辐射常数
A_RADIATION   = 4.0 * 5.670374419e-5 / C_LIGHT   # erg / cm^3 / K^4
SIGMA_THOMSON = 6.6524587158e-25   # cm^2
SIGMA_SB      = 5.670374419e-5     # erg / cm^2 / s / K^4

# 中微子耦合常数 (费米常数)
G_FERMI       = 1.1663788e-5       # 1 / GeV^2
G_FERMI_CGS   = G_FERMI * 1.6022e-33  # erg * cm^3

# 单位换算
MEV_TO_ERG    = 1.602176634e-6
ERG_TO_MEV    = 1.0 / MEV_TO_ERG

# 典型超新星参数
TYPICAL_CORE_T      = 4.0e10        # K  (proto-neutron star surface)
TYPICAL_CORE_RHO    = 3.0e12        # g / cm^3
TYPICAL_SHOCK_R     = 1.5e7         # cm  (150 km)
TYPICAL_GAIN_RADIUS = 1.0e7         # cm
TYPICAL_SONIC_R     = 3.0e7         # cm

# 安全下限，避免除零 / log(0)
TINY_RHO     = 1.0e-30
TINY_T       = 1.0e4
TINY_P       = 1.0e-30
TINY_E       = 1.0e-30
MACHINE_EPS  = 2.220446049250313e-16


def avogadro_mean_molecular_weight(x_h: float = 0.70, x_he: float = 0.28) -> float:
    """完全电离等离子体的平均分子量 (以质子质量为单位).

    mu = 1 / (2 X_H + 3/4 X_He + 1/2 (1-X_H-X_He))
    """
    x_z = max(0.0, 1.0 - x_h - x_he)
    inv_mu = 2.0 * x_h + 0.75 * x_he + 0.5 * x_z
    return 1.0 / max(inv_mu, 1.0e-6)
