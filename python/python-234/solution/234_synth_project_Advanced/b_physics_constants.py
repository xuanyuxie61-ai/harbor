"""
b_physics_constants.py
----------------------
CODATA / PDG 推荐的基本物理常数与 B 介子谱学参数。
本模块为博士级 B 物理衰变链重建提供所有必需的输入常量,
包括 CKM 矩阵元、介子质量 (MeV/c^2)、平均寿命 (ps)、
质量差 Delta m_{d,s} 及宽度差 Delta Gamma_{d,s}。

所有数值以自然单位制 (hbar = c = 1) 为基础,必要时给出单位转换因子。
公式参考:
  - PDG 2024 Review of Particle Physics
  - CKMfitter / UTfit 全局拟合结果 (2024)
  - Schwerbrock, "B Physics phenomenology", arXiv:hep-ph/0607191
"""

from __future__ import annotations
import math
from typing import Tuple

# ---------------------------------------------------------------------------- #
#                          通用物理常数 (自然单位)                            #
# ---------------------------------------------------------------------------- #
HBAR_C           = 197.3269804        # hbar*c, MeV.fm
HBAR_PS          = 6.582119569e-13    # hbar, MeV.ps   (用于 tau -> Gamma 转换)
ALPHA_EM         = 1.0 / 137.035999084  # 电磁精细结构常数 (零动量)
ALPHA_S_MZ       = 0.1179             # 强耦合 alpha_s(m_Z)
SIN2THETA_W      = 0.23122            # 弱混合角 sin^2(theta_W) (MSbar, m_Z)


# ---------------------------------------------------------------------------- #
#                            B 介子质量 (MeV/c^2)                             #
# ---------------------------------------------------------------------------- #
M_B0        = 5279.65       # B^0 质量
M_BP        = 5279.34       # B^+ 质量
M_BS        = 5366.88       # B_s^0 质量
M_BC        = 6274.9        # B_c^+ 质量
M_D0        = 1864.84       # D^0 质量
M_DP        = 1869.66       # D^+ 质量
M_DS        = 1968.35       # D_s^+ 质量
M_PI        = 139.57039     # pi^+ 质量
M_PI0       = 134.9768      # pi^0 质量
M_K         = 493.677       # K^+ 质量
M_K0        = 497.611       # K^0 质量
M_KS        = 497.611       # K_S^0 质量 (近似等于 K^0)
M_KL        = 497.611       # K_L^0 质量
M_JPSI      = 3096.9        # J/psi 质量
M_PHI       = 1019.461      # phi(1020) 质量
M_RHO       = 775.26        # rho(770) 质量
M_OMEGA     = 782.65        # omega(782) 质量
M_DSTAR     = 2010.26       # D*+ 质量
M_DSTAR0    = 2006.85       # D*0 质量


# ---------------------------------------------------------------------------- #
#                            B 介子寿命 (ps) 及总宽度 (MeV)                   #
# ---------------------------------------------------------------------------- #
TAU_B0      = 1.519         # B^0 寿命 (ps)
TAU_BP      = 1.638         # B^+ 寿命 (ps)
TAU_BS      = 1.515         # B_s^0 寿命 (ps)
TAU_BC      = 0.510e-3      # B_c^+ 寿命 (ps)

GAMMA_B0    = HBAR_PS / TAU_B0    # B^0 总宽度 (MeV)
GAMMA_BP    = HBAR_PS / TAU_BP
GAMMA_BS    = HBAR_PS / TAU_BS
GAMMA_BC    = HBAR_PS / TAU_BC


# ---------------------------------------------------------------------------- #
#                       B0-B0bar / Bs-Bsbar 混合参数                         #
# ---------------------------------------------------------------------------- #
DELTA_M_D   = 0.5065        # Delta m_d = m_H - m_L, ps^-1
DELTA_M_S   = 17.757        # Delta m_s, ps^-1
DELTA_GAMMA_D = 0.0         # Delta Gamma_d ≈ 0 (SM 预言)
DELTA_GAMMA_S = 0.087       # Delta Gamma_s, ps^-1
A_SL_D      = -4.8e-4       # 半轻 CP 不对称 S_l^d
A_SL_S      = 2.22e-5       # 半轻 CP 不对称 S_l^s


# ---------------------------------------------------------------------------- #
#                   CKM 矩阵: Wolfenstein 参数 (UTfit 2024)                  #
#  V = R_23(theta_23) R_13(theta_13, delta) R_12(theta_12)                  #
#  近似: V ~ [ 1-l^2/2,  l,        A l^3 (rho_bar - i eta_bar) ]            #
#              [ -l,      1-l^2/2,  A l^2              ]                    #
#              [ A l^3(1-rho_bar-i eta_bar), -A l^2, 1]                    #
# ---------------------------------------------------------------------------- #
LAMBDA_W    = 0.22650       # Wolfenstein lambda = |V_us|
A_W         = 0.790         # A = |V_cb| / lambda^2
RHOBAR_W    = 0.159         # rho_bar
ETABAR_W    = 0.350         # eta_bar (CP 破坏相位的来源)


# ---------------------------------------------------------------------------- #
#                    么正三角形顶点与内角                                     #
#  定义: V_ud V_ub^* + V_cd V_cb^* + V_td V_tb^* = 0                        #
#  顶点: (0,0), (1,0), (rho_bar, eta_bar)                                   #
#  内角: alpha (phi_2), beta (phi_1), gamma (phi_3)                         #
# ---------------------------------------------------------------------------- #
def wolfenstein_to_ckm(
        lam: float = LAMBDA_W,
        a:   float = A_W,
        rb:  float = RHOBAR_W,
        eb:  float = ETABAR_W,
) -> list:
    """
    返回 3x3 CKM 矩阵 (复数列表), 精确到 O(lambda^4)。
    采用 Buras 精确参数化:
        V_ub = A lambda^3 (rho - i eta) / sqrt(1 - A^2 lambda^4)
        V_td = A lambda^3 (1 - rho - i eta)
        V_ts = -A lambda^2 + 0.5 A lambda^4 (1 + 2(rho + i eta))
    """
    lam2 = lam * lam
    lam3 = lam2 * lam
    lam4 = lam2 * lam2
    sqrt14 = math.sqrt(max(1.0 - a * a * lam4, 0.0))
    V = [[0.0 + 0.0j for _ in range(3)] for _ in range(3)]
    V[0][0] = 1.0 - 0.5 * lam2 - 0.125 * lam4
    V[0][1] = lam + 0.0j
    V[0][2] = a * lam3 * (rb - 1j * eb) / sqrt14
    V[1][0] = -lam + 0.0j
    V[1][1] = 1.0 - 0.5 * lam2 - 0.125 * lam4 * (1.0 + 4.0 * a * a)
    V[1][2] = a * lam2 + 0.0j
    V[2][0] = a * lam3 * (1.0 - rb - 1j * eb)
    V[2][1] = -a * lam2 + 0.5 * a * lam4 * (1.0 + 2.0 * (rb + 1j * eb))
    V[2][2] = 1.0 + 0.0j
    return V


def unitarity_triangle_angles(
        rb: float = RHOBAR_W,
        eb: float = ETABAR_W,
) -> Tuple[float, float, float]:
    """
    计算么正三角形的三个内角 (弧度):
        alpha = arg( -V_td V_tb^* / (V_ud V_ub^*) )
        beta  = arg( -V_cd V_cb^* / (V_td V_tb^*) )
        gamma = arg( -V_ud V_ub^* / (V_cd V_cb^*) )
    满足 alpha + beta + gamma = pi。
    """
    alpha = math.atan2(eb, 1.0 - rb)
    beta  = math.atan2(eb, rb)
    gamma = math.atan2(eb, rb)
    # 严格定义:
    # alpha = atan2(eta_bar, 1-rho_bar)
    # beta  = atan2(eta_bar, rho_bar)
    # gamma = pi - alpha - beta
    alpha = math.atan2(eb, 1.0 - rb)
    beta  = math.atan2(eb, rb)
    gamma = math.pi - alpha - beta
    return alpha, beta, gamma


# ---------------------------------------------------------------------------- #
#                     常用衰变道的量子数                                       #
# (电荷 Q, 重子数 B, 奇异数 S, 粲数 C, 底数 B', 同位旋 I, I_3)             #
# ---------------------------------------------------------------------------- #
# (Q, B, S, C, Bprime, I, I3)
QUANTUM_NUMBERS = {
    "B0":     ( 0, 0,  0,  0, -1, 0.5, -0.5),
    "B0bar":  ( 0, 0,  0,  0, +1, 0.5, +0.5),
    "Bp":     (+1, 0,  0,  0, -1, 0.5, +0.5),
    "Bm":     (-1, 0,  0,  0, +1, 0.5, -0.5),
    "Bs":     ( 0, 0, -1,  0, -1, 0.0,  0.0),
    "Dm":     (-1, 0,  0, -1,  0, 0.5, -0.5),
    "Dp":     (+1, 0,  0, +1,  0, 0.5, +0.5),
    "D0":     ( 0, 0,  0, +1,  0, 0.5, +0.5),
    "D0bar":  ( 0, 0,  0, -1,  0, 0.5, -0.5),
    "pip":    (+1, 0,  0,  0,  0, 1.0, +1.0),
    "pim":    (-1, 0,  0,  0,  0, 1.0, -1.0),
    "pi0":    ( 0, 0,  0,  0,  0, 1.0,  0.0),
    "Kp":     (+1, 0, +1,  0,  0, 0.5, +0.5),
    "Km":     (-1, 0, -1,  0,  0, 0.5, -0.5),
    "K0":     ( 0, 0, +1,  0,  0, 0.5, +0.5),
    "K0bar":  ( 0, 0, -1,  0,  0, 0.5, -0.5),
    "KS":     ( 0, 0,  0,  0,  0, 0.5, +0.5),
    "Jpsi":   ( 0, 0,  0,  0,  0, 0.0,  0.0),
    "phi":    ( 0, 0,  0,  0,  0, 0.0,  0.0),
    "rho0":   ( 0, 0,  0,  0,  0, 1.0,  0.0),
}


# ---------------------------------------------------------------------------- #
#                         Källén 运动学函数                                   #
#  lambda(a, b, c) = a^2 + b^2 + c^2 - 2(ab + bc + ca)                     #
#  在三体衰变中用于计算子不变质量平方 s_ij 的允许范围。                     #
# ---------------------------------------------------------------------------- #
def kallen(a: float, b: float, c: float) -> float:
    """Källén 三角形函数, 数值鲁棒的版本。"""
    # 采用重新排列, 避免相近大数相消:
    # lambda = (a - (sqrt(b) + sqrt(c))^2) * (a - (sqrt(b) - sqrt(c))^2)
    if a < 0.0 or b < 0.0 or c < 0.0:
        return 0.0
    sb = math.sqrt(b)
    sc = math.sqrt(c)
    return (a - (sb + sc) ** 2) * (a - (sb - sc) ** 2)


def kallen_sqrt(a: float, b: float, c: float) -> float:
    """sqrt(lambda(a,b,c)), 用于两体动量。返回 0 若参数为负。"""
    val = kallen(a, b, c)
    return math.sqrt(max(val, 0.0))


# ---------------------------------------------------------------------------- #
#                   Blatt-Weisskopf 势垒因子 (L = 0, 1, 2)                  #
#  F_L(z) = 1                                 (L=0)                        #
#  F_L(z) = sqrt(2 z^2 / (z^2 + 1))         (L=1)                        #
#  F_L(z) = sqrt(13 z^4 / (z^4 + 9 z^2 + 9)) (L=2)                        #
#  z = |p| * r, r 为介子半径 (~ 1.5 GeV^-1 for B 衰变)                    #
# ---------------------------------------------------------------------------- #
BLATT_RADIUS_DEFAULT = 1.5   # GeV^-1, 典型强子半径


def blatt_weisskopf(z: float, L: int) -> float:
    """返回 Blatt-Weisskopf 因子 F_L(z), z = |p| * r。"""
    if L < 0:
        raise ValueError("角动量 L 必须非负")
    if L == 0:
        return 1.0
    z2 = z * z
    if L == 1:
        if z2 < 1.0e-20:
            return 0.0
        return math.sqrt(2.0 * z2 / (z2 + 1.0))
    if L == 2:
        if z2 < 1.0e-20:
            return 0.0
        z4 = z2 * z2
        return math.sqrt(13.0 * z4 / (z4 + 9.0 * z2 + 9.0))
    # 通用 L: 使用球贝塞尔函数 j_L(z) 递推
    # F_L(z) = |z j_L(z)| / sqrt(1 + z^2 + ... )
    # 此处仅实现 L <= 2, 高阶抛异常
    raise NotImplementedError(
        f"Blatt-Weisskopf 仅实现到 L=2, 请求 L={L}"
    )


# ---------------------------------------------------------------------------- #
#                   Breit-Wigner 共振线形 (含运行宽度)                       #
#  BW(s) = 1 / (m_r^2 - s - i m_r Gamma_r(s))                              #
#  Gamma_r(s) = Gamma_r * (q/q_r)^L * (m_r/sqrt(s)) * (F_L(q)/F_L(q_r))^2 #
# ---------------------------------------------------------------------------- #
def breit_wigner_running(
        s: complex,
        m_r: float,
        gamma_r: float,
        m_parent: float,
        m_daug1: float,
        m_daug2: float,
        L: int,
        radius: float = BLATT_RADIUS_DEFAULT,
) -> complex:
    """
    计算带运行宽度的相对论性 Breit-Wigner 振幅。
    返回 1 / (m_r^2 - s - i m_r Gamma_r(s))
    """
    def two_body_mom(M_sq: float, m1: float, m2: float) -> float:
        if M_sq <= (m1 + m2) ** 2:
            return 0.0
        return math.sqrt(kallen(M_sq, m1 * m1, m2 * m2)) / (2.0 * math.sqrt(M_sq))

    threshold = (m_daug1 + m_daug2) ** 2
    if s.real < threshold:
        # 低于阈值: 仅保留虚部正则化
        return 1.0 / (m_r * m_r - s)

    q_r = two_body_mom(m_r * m_r, m_daug1, m_daug2)
    q_s = two_body_mom(max(s.real, threshold), m_daug1, m_daug2)
    f_r = blatt_weisskopf(q_r * radius, L)
    f_s = blatt_weisskopf(q_s * radius, L)
    sqrt_s = math.sqrt(max(s.real, threshold))
    if q_r < 1.0e-20 or f_r < 1.0e-20 or sqrt_s < 1.0e-20:
        return 1.0 / (m_r * m_r - s)
    gamma_s = gamma_r * ((q_s / q_r) ** L) * (m_r / sqrt_s) * (f_s / f_r) ** 2
    return 1.0 / (m_r * m_r - s - 1j * m_r * gamma_s)
