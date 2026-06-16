"""
constants.py — 物理常数与单位系统 (PROJECT_232)

本模块定义高能物理散射计算中使用的基本物理常数。
所有量采用自然单位制 ℏ = c = 1，但保留转换因子以便与实验数据对照。

核心常数:
  ℏ  = 6.582119569e-25 GeV·s       (约化普朗克常数)
  c  = 2.99792458e8 m/s             (光速)
  α  = 1/137.035999084              (精细结构常数)
  m_π = 0.13957 GeV/c²              (带电π介子质量)
  m_K = 0.49368 GeV/c²              (带电K介子质量)
  1 GeV⁻¹ = 0.1973269804e-15 m     (长度换算)

参考: PDG 2022 Review of Particle Physics
"""
import numpy as np


# ---------------------------------------------------------------------------
# 基本物理常数 (自然单位制 ℏ = c = 1，除非特别标注)
# ---------------------------------------------------------------------------
HBAR_GEVS = 6.582119569e-25           # ℏ  [GeV·s]
C_MKS = 2.99792458e8                  # c  [m/s]
ALPHA_EM = 7.2973525693e-3            # 精细结构常数 α ≈ 1/137.036
ALPHA_EM_INV = 1.0 / ALPHA_EM         # α⁻¹
PI = np.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI

# ---------------------------------------------------------------------------
# 强子质量 [GeV]
# ---------------------------------------------------------------------------
MASS_PI_PLUS = 0.13957039             # π± 质量
MASS_PI_ZERO = 0.1349768              # π⁰ 质量
MASS_K_PLUS = 0.493677                # K± 质量
MASS_K_ZERO = 0.497611                # K⁰ 质量
MASS_ETA = 0.547862                   # η 质量
MASS_PROTON = 0.938272                # 质子质量
MASS_NEUTRON = 0.939565               # 中子质量

# ---------------------------------------------------------------------------
# 单位转换因子
# ---------------------------------------------------------------------------
GEV_INV_TO_FM = 0.1973269804          # 1 GeV⁻¹ = 0.1973 fm
FM_TO_GEV_INV = 1.0 / GEV_INV_TO_FM   # 1 fm = 5.068 GeV⁻¹
MB_TO_GEV_INV2 = 0.389379             # 1 mb = 0.3894 GeV⁻²
GEV_INV2_TO_MB = 1.0 / MB_TO_GEV_INV2 # 1 GeV⁻² = 2.568 mb
GEV_INV2_TO_UBARN = GEV_INV2_TO_MB * 1.0e3  # 1 GeV⁻² = 2568 μb

# ---------------------------------------------------------------------------
# 数值容差
# ---------------------------------------------------------------------------
EPS_MACH = np.finfo(np.float64).eps   # 机器精度 ~2.22e-16
TOL_ZERO = 1.0e-14                    # 零判断阈值
TOL_CONVERGE = 1.0e-10                # 迭代收敛阈值
MAX_ITER_DEFAULT = 200                # 默认最大迭代次数

# ---------------------------------------------------------------------------
# 色因子 (SU(3)_c)
# ---------------------------------------------------------------------------
NC = 3                                # 色数
CF = (NC * NC - 1.0) / (2.0 * NC)    # C_F = 4/3 (基本表示)
CA = float(NC)                        # C_A = 3 (伴随表示)
TF = 0.5                              # T_F = 1/2
NF_LIGHT = 3                          # 轻味夸克数 (u,d,s)


def mandelstam_s(p1, p2):
    """
    计算 Mandelstam 变量 s = (p1 + p2)²

    Parameters
    ----------
    p1, p2 : ndarray, shape (4,)
        四动量 (E, px, py, pz)

    Returns
    -------
    float
        s = (E1+E2)² - |p⃗1+p⃗2|²
    """
    p_sum = p1 + p2
    s = p_sum[0] ** 2 - np.sum(p_sum[1:] ** 2)
    return max(s, EPS_MACH)


def mandelstam_t(p1, p3):
    """
    计算 Mandelstam 变量 t = (p1 - p3)²

    Parameters
    ----------
    p1 : ndarray, shape (4,)
        入射粒子四动量
    p3 : ndarray, shape (4,)
        出射粒子四动量

    Returns
    -------
    float
        t = (E1-E3)² - |p⃗1-p⃗3|²
    """
    p_diff = p1 - p3
    t = p_diff[0] ** 2 - np.sum(p_diff[1:] ** 2)
    return t


def mandelstam_u(p1, p4):
    """
    计算 Mandelstam 变量 u = (p1 - p4)²

    约束关系: s + t + u = m1² + m2² + m3² + m4²

    Parameters
    ----------
    p1 : ndarray, shape (4,)
        入射粒子四动量
    p4 : ndarray, shape (4,)
        出射粒子2四动量

    Returns
    -------
    float
        u = (E1-E4)² - |p⃗1-p⃗4|²
    """
    p_diff = p1 - p4
    u = p_diff[0] ** 2 - np.sum(p_diff[1:] ** 2)
    return u


def cm_momentum(s, m1, m2):
    """
    计算质心系中两体系统的动量大小

    公式 (Källén 函数):
      p* = λ^{1/2}(s, m1², m2²) / (2√s)

    其中 λ(x,y,z) = x² + y² + z² - 2xy - 2xz - 2yz

    Parameters
    ----------
    s : float or complex
        质心能量平方 [GeV²]
    m1, m2 : float
        两粒子质量 [GeV]

    Returns
    -------
    float or complex
        质心系动量 [GeV]
    """
    is_complex = isinstance(s, complex) or (hasattr(s, 'dtype') and np.iscomplexobj(s))
    if is_complex:
        # 复数输入: 直接计算 (允许复数 sqrt)
        m1sq = m1 * m1
        m2sq = m2 * m2
        kallen = s * s + m1sq * m1sq + m2sq * m2sq - 2.0 * (s * m1sq + s * m2sq + m1sq * m2sq)
        return np.lib.scimath.sqrt(kallen) / (2.0 * np.lib.scimath.sqrt(s))
    # 实数输入
    if s < (m1 + m2) ** 2:
        return 0.0
    m1sq = m1 * m1
    m2sq = m2 * m2
    kallen = s * s + m1sq * m1sq + m2sq * m2sq - 2.0 * (s * m1sq + s * m2sq + m1sq * m2sq)
    if kallen < 0.0:
        return 0.0
    return np.sqrt(kallen) / (2.0 * np.sqrt(s))


def velocity_factor(s, m1, m2):
    """
    计算速度因子 β (相对论性速度)

    β = √λ(s, m1², m2²) / s

    用于截面的阈值行为分析。当 s → (m1+m2)² 时 β → 0。

    Parameters
    ----------
    s : float
        质心能量平方
    m1, m2 : float
        粒子质量

    Returns
    -------
    float
        β ∈ [0, 1)
    """
    if s <= (m1 + m2) ** 2:
        return 0.0
    m1sq = m1 * m1
    m2sq = m2 * m2
    kallen = s * s + m1sq * m1sq + m2sq * m2sq - 2.0 * (s * m1sq + s * m2sq + m1sq * m2sq)
    if kallen <= 0.0:
        return 0.0
    return np.sqrt(kallen) / s
