"""
cosmo_constants.py -- 宇宙学基本常数、暗能量状态方程与几何量
================================================================
Project 260: 计算宇宙学 -- 暗能量状态方程约束
           高阶有限差分与稳定性分析 (小规模可复现实验)

物理公式索引
-----------
(1)  FLRW 度规:  ds^2 = -c^2 dt^2 + a^2(t)[dr^2/(1-k r^2) + r^2 dOmega^2]
(2)  Friedmann 方程 (第一):
         H^2 = (8 pi G/3) rho - k c^2/a^2 + Lambda/3
(3)  Friedmann 方程 (第二, 加速方程):
         a''/a = -(4 pi G/3)(rho + 3p/c^2) + Lambda/3
(4)  连续性方程:
         rho_dot + 3H(rho + p/c^2) = 0
(5)  CPL 暗能量状态方程:
         w(a) = w0 + wa * (1 - a)
              = w0 + wa * z/(1+z)
(6)  暗能量密度演化 (由连续性方程积分):
         rho_DE(a)/rho_DE(1) = a^{-3(1+w0+wa)} * exp(-3wa(1-a))
(7)  无量纲 Hubble 参数:
         E^2(a) = Om a^{-3} + Or a^{-4} + Ok a^{-2} + ODE * f_DE(a)
(8)  减速度参数:
         q(a) = -1 - (d ln H / d ln a)
              = (1/2) sum_i Omega_i(a) (1 + 3 w_i)
(9)  声速 c_s^2 = dp/rho = w + (1+w) d ln rho / d ln a  (rest frame)
(10) CPL 暗能量声速 (rest frame):
         c_s^2 = w(a) + (1+w(a)) * [3(1+w0+wa)/a - 3wa - 3(1+w0+wa)ln(a)/a ... ]
         简化: c_s^2_DE = w(a) (quintessence 声速)
(11) 声子视界:
         r_s(z) = integral_z^inf c_s(z') / H(z') dz'
         其中 c_s = c / sqrt(3(1+R_b)),  R_b = 3 Omega_b / (4 Omega_gamma)
(12) 共动距离:
         chi(z) = (c/H0) integral_0^z dz' / E(z')
(13) 光度距离:
         d_L(z) = (1+z) chi(z)
(14) 角直径距离:
         d_A(z) = chi(z) / (1+z)
(15) Etherington 互易关系:
         d_L = (1+z)^2 d_A
(16) 体积元:
         dV/(dz dOmega) = chi^2(z) * c / (H0 E(z))
(17) 年龄:
         t(a) = (1/H0) integral_0^a da' / (a' E(a'))

参考
----
  [1] Planck 2018, A&A, 641, A6
  [2] Chevallier & Polarski 2001, IJMPD, 10, 213
  [3] Linder 2003, PRL, 90, 091301
================================================================
"""
import math

# -----------------------------------------------------------------------
#  基本物理常数 (CODATA 2018 + Planck 2018)
# -----------------------------------------------------------------------
H0_KMS_MPC    = 67.36           # Hubble 常数 [km s^{-1} Mpc^{-1}]
h_little      = H0_KMS_MPC / 100.0
c_km_s        = 299792.458      # 光速 [km/s]
c_m_s         = c_km_s * 1e3    # 光速 [m/s]
G_newton      = 6.67430e-11     # 万有引力常数 [m^3 kg^{-1} s^{-2}]
sigma_T       = 6.6524e-29      # Thomson 散射截面 [m^2]
m_p           = 1.6726e-27      # 质子质量 [kg]
T_cmb         = 2.7255          # CMB 温度 [K]
k_B           = 1.380649e-23    # Boltzmann 常数 [J/K]
eV_J          = 1.602176634e-19 # 电子伏 [J]
Mpc_m         = 3.085677581e22  # 1 Mpc [m]

# Hubble 距离和时间
D_H_Mpc       = c_km_s / H0_KMS_MPC
H0_s_inv      = H0_KMS_MPC * 1e3 / Mpc_m
t_H_s         = 1.0 / H0_s_inv
t_H_Gyr       = t_H_s / (3600.0 * 24.0 * 365.25 * 1e9)
H0_inv_Mpc    = 1.0 / (H0_KMS_MPC / c_km_s)  # c/H0 in Mpc, same as D_H

# -----------------------------------------------------------------------
#  宇宙密度参数 (Planck 2018 baseline)
# -----------------------------------------------------------------------
OMEGA_M0      = 0.3153          # 总物质 (CDM + baryons)
OMEGA_B0      = 0.0493          # 重子
OMEGA_CDM0    = OMEGA_M0 - OMEGA_B0
OMEGA_R0      = 4.18e-5         # 辐射 (photons + 3 massless nu)
OMEGA_GAMMA0  = 2.469e-5 / (h_little**2)  # photons
OMEGA_K0      = 0.0             # 空间曲率 (平坦)
OMEGA_DE0     = 1.0 - OMEGA_M0 - OMEGA_R0 - OMEGA_K0

# 扰动参数
SIGMA_8       = 0.831           # 物质起伏振幅
N_S           = 0.9649          # 标量谱指数
A_S           = 2.1e-9          # 原初振幅
K_PIVOT       = 0.05            # 归一化尺度 [h/Mpc]

# 复合/拖拽
Z_DRAG        = 1059.0
R_S_DRAG_MPC  = 147.09
Z_STAR        = 1089.8

# -----------------------------------------------------------------------
#  暗能量状态方程: CPL 参数化 (fiducial = LCDM)
# -----------------------------------------------------------------------
W0_FID        = -1.0
WA_FID        = 0.0

# 扫描范围
W0_SCAN       = (-1.3, -0.7)
WA_SCAN       = (-1.0, 1.0)


# -----------------------------------------------------------------------
#  核心函数
# -----------------------------------------------------------------------

def cpl_eos(a, w0=None, wa=None):
    """
    CPL 状态方程:
        w(a) = w0 + wa * (1 - a)

    参数:
        a  : 尺度因子 (0 < a <= 1)
        w0 : 当前 EoS 参数 (默认 fiducial)
        wa : EoS 演化参数 (默认 fiducial)
    """
    if w0 is None:
        w0 = W0_FID
    if wa is None:
        wa = WA_FID
    a_safe = max(a, 1e-15)
    return w0 + wa * (1.0 - a_safe)


def cpl_density_ratio(a, w0=None, wa=None):
    """
    暗能量密度比 rho_DE(a) / rho_DE(1):

        f(a) = a^{-3(1+w0+wa)} * exp(-3 wa (1-a))

    由连续性方程 rho' = -3(1+w) rho/a 积分得到.
    """
    if w0 is None:
        w0 = W0_FID
    if wa is None:
        wa = WA_FID
    if a <= 0.0:
        return 0.0
    a_safe = max(a, 1e-15)
    log_f = -3.0 * (1.0 + w0 + wa) * math.log(a_safe) - 3.0 * wa * (1.0 - a_safe)
    if log_f > 500:
        return float('inf')
    if log_f < -500:
        return 0.0
    return math.exp(log_f)


def e_squared(a, w0=None, wa=None,
              om=OMEGA_M0, orr=OMEGA_R0, ok=OMEGA_K0):
    """
    E^2(a) = H^2(a)/H0^2:

        = Om a^{-3} + Or a^{-4} + Ok a^{-2} + ODE * f_DE(a)
    """
    if w0 is None:
        w0 = W0_FID
    if wa is None:
        wa = WA_FID
    ode = 1.0 - om - orr - ok
    a_safe = max(a, 1e-15)
    val = (om / a_safe**3 + orr / a_safe**4 + ok / a_safe**2
           + ode * cpl_density_ratio(a_safe, w0, wa))
    return max(val, 0.0)


def e_of_a(a, w0=None, wa=None):
    """E(a) = sqrt(E^2(a))."""
    return math.sqrt(e_squared(a, w0, wa))


def hubble_z(z, w0=None, wa=None):
    """H(z) [km/s/Mpc]."""
    a = 1.0 / (1.0 + z)
    return H0_KMS_MPC * e_of_a(a, w0, wa)


def deceleration_param(a, w0=None, wa=None):
    """
    q(a) = (1/2) * sum_i Omega_i(a) * (1 + 3 w_i(a))

    物质: w=0, 辐射: w=1/3, DE: w=w(a), 曲率: w=-1/3
    """
    if w0 is None:
        w0 = W0_FID
    if wa is None:
        wa = WA_FID
    a_s = max(a, 1e-15)
    E2 = e_squared(a_s, w0, wa)
    if E2 < 1e-30:
        return 0.5
    om_m = OMEGA_M0 / a_s**3
    om_r = OMEGA_R0 / a_s**4
    om_de = (1.0 - OMEGA_M0 - OMEGA_R0) * cpl_density_ratio(a_s, w0, wa)
    w_de = cpl_eos(a_s, w0, wa)
    q = 0.5 * (om_m + (4.0/3.0)*om_r + om_de*(1.0 + 3.0*w_de)) / E2
    return q


def hubble_prime_over_h(a, w0=None, wa=None):
    """
    d ln H / d ln a = -(1 + q(a))

    用于增长方程的阻尼项.
    """
    return -(1.0 + deceleration_param(a, w0, wa))


def omega_m_a(a, w0=None, wa=None):
    """Omega_m(a) = Om0 a^{-3} / E^2(a)."""
    a_s = max(a, 1e-15)
    return OMEGA_M0 / (a_s**3 * e_squared(a_s, w0, wa))


def omega_de_a(a, w0=None, wa=None):
    """Omega_DE(a) = ODE0 * f_DE(a) / E^2(a)."""
    a_s = max(a, 1e-15)
    ode = 1.0 - OMEGA_M0 - OMEGA_R0
    return ode * cpl_density_ratio(a_s, w0, wa) / e_squared(a_s, w0, wa)


def sound_speed_de(a, w0=None, wa=None):
    """
    暗能量 rest-frame 声速 (平方):
        c_s^2 = dp/de = w  (quintessence, c_s^2=1)

    对 CPL: c_s^2 = w(a)  (简化)
    物理单位: c_s^2 以 c^2 为单位.
    """
    w = cpl_eos(a, w0, wa)
    return max(w, -1.0 + 1e-15)  # 防止 phantom 奇点


def baryon_photon_ratio(a, omega_b=OMEGA_B0):
    """
    R_b = 3 Omega_b a / (4 Omega_gamma)

    光子密度: Omega_gamma = 2.469e-5 / h^2
    """
    om_g = 2.469e-5 / (h_little**2)
    return 3.0 * omega_b * a / (4.0 * om_g)


def sound_horizon_integrand(z, w0=None, wa=None):
    """
    声子视界被积函数:
        1 / H(z) * c_s(z)
    c_s = c / sqrt(3(1+R_b))
    返回无量纲量 1 / (sqrt(3(1+R_b)) * E(z))
    """
    a = 1.0 / (1.0 + z)
    R_b = baryon_photon_ratio(a)
    Ez = e_of_a(a, w0, wa)
    return 1.0 / (math.sqrt(3.0 * (1.0 + R_b)) * Ez)


def is_phantom_crossing(w0, wa):
    """
    Phantom 穿越条件:
        (w0 + 1) * (w0 + wa + 1) < 0
    即 w(a) 在 [0,1] 内穿越 -1.
    """
    w_now = w0
    w_early = w0 + wa
    return (w_now + 1.0) * (w_early + 1.0) < 0


def phantom_crossing_scale_factor(w0, wa):
    """
    若 w(a) 穿越 -1, 返回穿越点的尺度因子:
        w0 + wa*(1-a) = -1
        => a_cross = 1 + (w0+1)/wa
    """
    if abs(wa) < 1e-15:
        return None
    a_cross = 1.0 + (w0 + 1.0) / wa
    if 0.0 < a_cross < 1.0:
        return a_cross
    return None


def luminosity_distance_mpc(z, w0=None, wa=None, n_quad=64):
    """
    光度距离 d_L(z) = (1+z) * chi(z).
    chi(z) = (c/H0) integral_0^z dz'/E(z') 用 Gauss-Legendre.
    """
    if z <= 0.0:
        return 0.0
    # 复合 Simpson
    n = n_quad if n_quad % 2 == 0 else n_quad + 1
    dz = z / n
    s = 0.0
    for i in range(n + 1):
        zi = i * dz
        ai = 1.0 / (1.0 + zi)
        Ei = e_of_a(ai, w0, wa)
        if Ei < 1e-30:
            continue
        coeff = 4.0 if (i % 2 == 1) else 2.0
        if i == 0 or i == n:
            coeff = 1.0
        s += coeff / Ei
    chi = D_H_Mpc * dz * s / 3.0
    return (1.0 + z) * chi


def age_of_universe_gyr(z=0.0, w0=None, wa=None, n_quad=64):
    """
    宇宙年龄 t(z) [Gyr]:
        t(z) = (1/H0) integral_0^{1/(1+z)} da / (a E(a))
    """
    if w0 is None:
        w0 = W0_FID
    if wa is None:
        wa = WA_FID
    a_max = 1.0 / (1.0 + z)
    n = n_quad if n_quad % 2 == 0 else n_quad + 1
    da = a_max / n
    s = 0.0
    for i in range(n + 1):
        ai = max(i * da, 1e-10)
        Ei = e_of_a(ai, w0, wa)
        if Ei < 1e-30:
            continue
        val = 1.0 / (ai * Ei)
        coeff = 4.0 if (i % 2 == 1) else 2.0
        if i == 0 or i == n:
            coeff = 1.0
        s += coeff * val
    integral = da * s / 3.0
    return t_H_Gyr * integral


# -----------------------------------------------------------------------
#  快速诊断
# -----------------------------------------------------------------------
if __name__ == '__main__':
    print("=== 宇宙学常数诊断 ===")
    print(f"  H0  = {H0_KMS_MPC:.2f} km/s/Mpc")
    print(f"  Om  = {OMEGA_M0:.4f}")
    print(f"  ODE = {OMEGA_DE0:.4f}")
    print(f"  D_H = {D_H_Mpc:.2f} Mpc")
    print(f"  t_H = {t_H_Gyr:.2f} Gyr")
    print(f"  E(1) = {e_of_a(1.0):.6f}")
    print(f"  q(1) = {deceleration_param(1.0):.4f}")
    print(f"  d_L(z=1) = {luminosity_distance_mpc(1.0):.2f} Mpc")
    print(f"  t(0)  = {age_of_universe_gyr(0.0):.2f} Gyr")
    print(f"  t(z=1)= {age_of_universe_gyr(1.0):.2f} Gyr")
    print(f"  w(a=0.5) = {cpl_eos(0.5, -0.9, -0.3):.4f}")
    print(f"  phantom crossing (w0=-0.9, wa=-0.3): {is_phantom_crossing(-0.9, -0.3)}")
    print(f"  phantom crossing a: {phantom_crossing_scale_factor(-0.9, -0.3)}")
