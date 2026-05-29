"""
special_functions.py
统计力学与电化学中的特殊数学函数

基于种子项目 881_polpak 的核心算法：
- r8_erf: 高精度误差函数（Cody 有理逼近）
- hermite_poly_phys: 物理学家 Hermite 多项式
- laguerre_poly: Laguerre 多项式
- gamma_values, beta, legendre_poly, spherical_harmonic 等

在离子通道问题中的应用：
- erf 用于 Gouy-Chapman 双电层电势解析解
- Hermite 多项式用于速度分布的 Gauss-Hermite 求积
- Laguerre 用于径向分布的展开
- Gamma 函数用于统计力学配分函数
"""

import numpy as np
from scipy.special import factorial


# ---------------------------------------------------------------------------
# 误差函数 erf(x) —— 基于 Cody 有理 Chebyshev 逼近（源自 r8_erf.m）
# ---------------------------------------------------------------------------
def erf_cody(x):
    """
    高精度误差函数 ERF(x)。

    分段有理逼近：
      |x| <= 0.46875:   多项式比值
      0.46875 < |x| <= 4.0:  有理函数 * exp(-x^2)
      |x| > 4.0:  渐近展开
    """
    a = np.array([
        3.16112374387056560E+00,
        1.13864154151050156E+02,
        3.77485237685302021E+02,
        3.20937758913846947E+03,
        1.85777706184603153E-01
    ])
    b = np.array([
        2.36012909523441209E+01,
        2.44024637934444173E+02,
        1.28261652607737228E+03,
        2.84423683343917062E+03
    ])
    c = np.array([
        5.64188496988670089E-01,
        8.88314979438837594E+00,
        6.61191906371416295E+01,
        2.98635138197400131E+02,
        8.81952221241769090E+02,
        1.71204761263407058E+03,
        2.05107837782607147E+03,
        1.23033935479799725E+03,
        2.15311535474403846E-08
    ])
    d = np.array([
        1.57449261107098347E+01,
        1.17693950891312499E+02,
        5.37181101862009858E+02,
        1.62138957456669019E+03,
        3.29079923573345963E+03,
        4.36261909014324716E+03,
        3.43936767414372164E+03,
        1.23033935480374942E+03
    ])
    p = np.array([
        3.05326634961232344E-01,
        3.60344899949804439E-01,
        1.25781726111229246E-01,
        1.60837851487422766E-02,
        6.58749161529837803E-04,
        1.63153871373020978E-02
    ])
    q = np.array([
        2.56852019228982242E+00,
        1.87295284992346047E+00,
        5.27905102951428412E-01,
        6.05183413124413191E-02,
        2.33520497626869185E-03
    ])
    sqrpi = 0.56418958354775628695E+00
    thresh = 0.46875
    xbig = 26.543
    xsmall = 1.11E-16

    xabs = abs(x)
    if xabs <= thresh:
        if xsmall < xabs:
            xsq = xabs * xabs
        else:
            xsq = 0.0
        xnum = a[4] * xsq
        xden = xsq
        for i in range(3):
            xnum = (xnum + a[i]) * xsq
            xden = (xden + b[i]) * xsq
        erfx = x * (xnum + a[3]) / (xden + b[3])
    elif xabs <= 4.0:
        xnum = c[8] * xabs
        xden = xabs
        for i in range(7):
            xnum = (xnum + c[i]) * xabs
            xden = (xden + d[i]) * xabs
        erfx = (xnum + c[7]) / (xden + d[7])
        xsq = np.floor(xabs * 16.0) / 16.0
        delta = (xabs - xsq) * (xabs + xsq)
        erfx = np.exp(-xsq * xsq) * np.exp(-delta) * erfx
        erfx = (0.5 - erfx) + 0.5
        if x < 0.0:
            erfx = -erfx
    else:
        if xbig <= xabs:
            erfx = 1.0 if x > 0.0 else -1.0
        else:
            xsq = 1.0 / (xabs * xabs)
            xnum = p[5] * xsq
            xden = xsq
            for i in range(4):
                xnum = (xnum + p[i]) * xsq
                xden = (xden + q[i]) * xsq
            erfx = xsq * (xnum + p[4]) / (xden + q[4])
            erfx = (sqrpi - erfx) / xabs
            xsq = np.floor(xabs * 16.0) / 16.0
            delta = (xabs - xsq) * (xabs + xsq)
            erfx = np.exp(-xsq * xsq) * np.exp(-delta) * erfx
            erfx = (0.5 - erfx) + 0.5
            if x < 0.0:
                erfx = -erfx
    return erfx


# ---------------------------------------------------------------------------
# Hermite 物理学家多项式 H_n(x)（源自 hermite_poly_phys.m）
# ---------------------------------------------------------------------------
def hermite_phys(n, x):
    """
    计算 Hermite 多项式 H_0(x) ... H_n(x)。

    递推关系：
        H_0(x) = 1
        H_1(x) = 2x
        H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)

    微分方程：
        H''_n - 2x H'_n + 2n H_n = 0

    正交性：
        ∫_{-∞}^{∞} exp(-x^2) H_n(x) H_m(x) dx = sqrt(pi) 2^n n! δ_{nm}
    """
    if n < 0:
        return np.array([])
    cx = np.zeros(n + 1)
    cx[0] = 1.0
    if n == 0:
        return cx
    cx[1] = 2.0 * x
    for i in range(2, n + 1):
        cx[i] = 2.0 * x * cx[i - 1] - 2.0 * (i - 1) * cx[i - 2]
    return cx


# ---------------------------------------------------------------------------
# Laguerre 多项式 L_n(x)（源自 laguerre_poly.m）
# ---------------------------------------------------------------------------
def laguerre_poly(n, x):
    """
    计算 Laguerre 多项式 L_0(x) ... L_n(x)。

    递推关系：
        L_0(x) = 1
        L_1(x) = 1 - x
        n L_n(x) = (2n - 1 - x) L_{n-1}(x) - (n-1) L_{n-2}(x)

    微分方程：
        x L''_n + (1 - x) L'_n + n L_n = 0

    正交性：
        ∫_0^∞ exp(-x) L_n(x) L_m(x) dx = δ_{nm}
    """
    if n < 0:
        return np.array([])
    cx = np.zeros(n + 1)
    cx[0] = 1.0
    if n == 0:
        return cx
    cx[1] = 1.0 - x
    for i in range(2, n + 1):
        cx[i] = (((2.0 * i - 1.0) - x) * cx[i - 1] - (i - 1.0) * cx[i - 2]) / i
    return cx


# ---------------------------------------------------------------------------
# Legendre 多项式 P_n(x)（源自 legendre_poly.m）
# ---------------------------------------------------------------------------
def legendre_poly(n, x):
    """
    Legendre 多项式 P_n(x)。

    递推：
        P_0(x) = 1
        P_1(x) = x
        n P_n(x) = (2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x)

    用于球坐标展开。
    """
    if n < 0:
        return np.array([])
    cx = np.zeros(n + 1)
    cx[0] = 1.0
    if n == 0:
        return cx
    cx[1] = x
    for i in range(2, n + 1):
        cx[i] = ((2.0 * i - 1.0) * x * cx[i - 1] - (i - 1.0) * cx[i - 2]) / i
    return cx


# ---------------------------------------------------------------------------
# Gamma 函数 log（源自 gamma_log_values.m / gamma_values.m）
# ---------------------------------------------------------------------------
def log_gamma_lanczos(z):
    """
    Lanczos 近似计算 ln Γ(z)，适用于 z > 0。

    公式：
        ln Γ(z) ≈ (z - 0.5) ln(z + g - 0.5) - (z + g - 0.5)
                   + ln[ sqrt(2π) * (c_0 + Σ c_k / (z + k)) ]
    """
    if z <= 0:
        raise ValueError("Gamma 函数的 log 要求正实数输入")
    g = 7.0
    coeffs = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])
    z = z - 1.0
    x = coeffs[0]
    for i in range(1, len(coeffs)):
        x = x + coeffs[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + np.log(x) - t + (z + 0.5) * np.log(t)


# ---------------------------------------------------------------------------
# 球谐函数相关（源自 spherical_harmonic.m）
# ---------------------------------------------------------------------------
def spherical_harmonic_norm(l, m):
    """
    球谐函数归一化常数 N_l^m。

    N_l^m = sqrt( (2l+1)/(4π) * (l-m)! / (l+m)! )
    """
    if abs(m) > l:
        return 0.0
    num = np.math.factorial(l - abs(m))
    den = np.math.factorial(l + abs(m))
    return np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) * num / den)


def associated_legendre(l, m, x):
    """
    连带 Legendre 函数 P_l^m(x)。

    采用递推关系计算（源自 legendre_associated.m 思想）。
    """
    if abs(m) > l or abs(x) > 1.0:
        return 0.0
    # 使用 scipy 风格或直接递推（此处用简化递推）
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact = fact + 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


# ---------------------------------------------------------------------------
# 统计力学配分函数相关
# ---------------------------------------------------------------------------
def partition_function_harmonic(omega, T, hbar=1.054571817e-34, kB=1.380649e-23):
    """
    一维谐振子配分函数：
        Z = 1 / (2 sinh(ℏω / 2k_B T))
    """
    x = hbar * omega / (2.0 * kB * T)
    if x < 1e-10:
        # 高温极限
        return 1.0 / (2.0 * x)
    return 1.0 / (2.0 * np.sinh(x))


def debeye_huckel_kappa(ionic_strength, T=300.0, eps_r=78.5):
    """
    Debye-Hückel 屏蔽长度倒数 κ（单位：m^{-1}）。

        κ^2 = (2000 N_A e^2 I) / (ε_0 ε_r k_B T)

    其中 I 为离子强度（mol/L）。
    """
    NA = 6.02214076e23
    e_charge = 1.602176634e-19
    eps0 = 8.854187817e-12
    kB = 1.380649e-23
    kappa_sq = (2000.0 * NA * e_charge ** 2 * ionic_strength) / (eps0 * eps_r * kB * T)
    return np.sqrt(kappa_sq)


def boltzmann_factor(energy, T=300.0):
    """
    Boltzmann 因子 exp(-E / k_B T)。
    能量单位为 J，返回无量纲因子。
    """
    kB = 1.380649e-23
    return np.exp(-energy / (kB * T))
