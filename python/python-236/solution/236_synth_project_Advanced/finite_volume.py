"""
finite_volume.py — 有限体积修正与 Lüscher 形式
================================================
融合种子项目:
  [328_ellipse] : 椭圆积分 (Carlson 形式) → Lüscher zeta 函数
  [305_dist_plot] : 距离函数 → 有限体积几何

物理背景:
  格点 QCD 在有限体积 L^3 x T 中计算. Lüscher 公式将有限体积
  能谱位移与无限体积散射相移联系:

  delta_E = E(L) - E(inf) = -12 * gamma^2 * a_0 / (m * L^3)
                              * exp(-m*L/sqrt(3)) * (1 + O(1/L))

  其中 a_0 为 s-波散射长度, m 为粒子质量.

  Lüscher zeta 函数:
  Z_{00}(s; q^2) = sum_{n in Z^3} (|n|^2 - q^2)^{-s}
  解析延拓到 s=0: 涉及 Epstein zeta 函数.

  在粒子动量参考系中:
  tan(delta_l) = pi^{3/2} * q^{2l+1} * Z_{00}(1; q^2) / ...

核心公式:
  Carlson 椭圆积分 (源自 [328_ellipse]):
  RF(x,y,z) = (1/2) int_0^inf dt / sqrt((t+x)(t+y)(t+z))
  RD(x,y,z) = (3/2) int_0^inf dt / ((t+z)*sqrt((t+x)(t+y)(t+z)))

  Lüscher zeta 函数通过 theta 函数求和:
  Z_{00}(1; q^2) = (1/sqrt(4*pi)) * sum_n 1/(|n|^2 - q^2)
  (需正则化)

  有限体积修正 (指数衰减):
  delta_m(L) / m ~ c * (m*L)^{-3/2} * exp(-m*L)
"""

import numpy as np
from typing import Tuple, Optional, Dict


# ======================================================================
# Carlson 椭圆积分 (源自 [328_ellipse])
# ======================================================================

def carlson_RF(x: complex, y: complex, z: complex,
               tol: float = 1e-12) -> complex:
    """Carlson 对称椭圆积分 RF.

    RF(x,y,z) = (1/2) int_0^inf dt / sqrt((t+x)(t+y)(t+z))

    使用 Carlson 的重复倍增算法 (duplication theorem):
    1. lambda_n = sqrt(x_n*y_n) + sqrt(y_n*z_n) + sqrt(z_n*x_n)
    2. x_{n+1} = (x_n + lambda_n) / 4
    3. 重复直到收敛
    4. RF ≈ 1/sqrt(mean) * (1 + 校正项)

    参数
    ----
    x, y, z : complex
        参数 (非负实部).
    tol : float
        收敛容差.

    返回
    ----
    result : complex
    """
    # 检查退化情况
    if abs(x) < 1e-300 and abs(y) < 1e-300 and abs(z) < 1e-300:
        return complex(np.inf)

    xn, yn, zn = complex(x), complex(y), complex(z)
    for _ in range(100):
        lam = np.sqrt(xn * yn) + np.sqrt(yn * zn) + np.sqrt(zn * xn)
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0
        mu = (xn + yn + zn) / 3.0
        if abs(mu) < 1e-300:
            break
        # 收敛检查
        dx = (mu - xn) / mu
        dy = (mu - yn) / mu
        dz = (mu - zn) / mu
        if max(abs(dx), abs(dy), abs(dz)) < tol:
            break

    # Taylor 展开校正
    mu = (xn + yn + zn) / 3.0
    if abs(mu) < 1e-300:
        return complex(0)
    dx = (mu - xn) / mu
    dy = (mu - yn) / mu
    dz = (mu - zn) / mu

    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz
    # RF ≈ (1/sqrt(mu)) * (1 - e2/10 + e3/14 + e2^2/24 - ...)
    correction = 1.0 - e2 / 10.0 + e3 / 14.0 + e2 ** 2 / 24.0
    return 1.0 / np.sqrt(mu) * correction


def carlson_RD(x: complex, y: complex, z: complex,
               tol: float = 1e-12) -> complex:
    """Carlson 对称椭圆积分 RD.

    RD(x,y,z) = (3/2) int_0^inf dt / ((t+z)*sqrt((t+x)(t+y)(t+z)))
    = RJ(x,y,z,z) 的特殊情况

    使用重复倍增算法:
    RD = 3*sum_n 4^{-n} / ((z_n + lambda_n)*sqrt(...))
    """
    xn, yn, zn = complex(x), complex(y), complex(z)
    sigma = complex(0)
    power4 = 1.0

    for n in range(100):
        lam = np.sqrt(xn * yn) + np.sqrt(yn * zn) + np.sqrt(zn * xn)
        sigma += power4 / ((zn + lam) * np.sqrt(zn + lam + 1e-300))
        power4 /= 4.0
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0
        mu = (xn + yn + zn) / 3.0
        if abs(mu) < 1e-300:
            break
        dx = (mu - xn) / mu
        dy = (mu - yn) / mu
        dz = (mu - zn) / mu
        if max(abs(dx), abs(dy), abs(dz)) < tol:
            break

    mu = (xn + yn + zn) / 3.0
    if abs(mu) < 1e-300:
        return complex(0)
    dx = (mu - xn) / mu
    dy = (mu - yn) / mu
    dz = (mu - zn) / mu

    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz
    # 校正
    correction = 1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 ** 2 / 88.0
    result = 3.0 * sigma + power4 / (mu * np.sqrt(mu)) * correction
    return result


def ellipse_perimeter(a: float, b: float) -> float:
    """椭圆周长 (源自 [328_ellipse]).

    P = 4*a*E(e)  其中 e = sqrt(1 - b^2/a^2)
    E(e) = 完全椭圆积分第二类

    在有限体积中用于计算等能面的几何因子.
    """
    if a <= 0 or b <= 0:
        return 0.0
    if a < b:
        a, b = b, a
    e_sq = 1.0 - (b / a) ** 2
    e_sq = max(0.0, min(e_sq, 1.0 - 1e-15))
    # 用 Gauss-Legendre 积分计算 E(e)
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(32)
    # E(e) = int_0^{pi/2} sqrt(1 - e^2*sin^2(t)) dt
    t = np.pi / 4 * (nodes + 1)
    w = np.pi / 4 * weights
    integrand = np.sqrt(1.0 - e_sq * np.sin(t) ** 2)
    E_e = float(np.sum(w * integrand))
    return 4.0 * a * E_e


# ======================================================================
# Lüscher zeta 函数
# ======================================================================

def luscher_zeta_00(q_sq: float, n_max: int = 50) -> float:
    """Lüscher zeta 函数 Z_{00}(1; q^2) 的截断求和.

    Z_{00}(s; q^2) = sum_{n in Z^3} 1 / (|n|^2 - q^2)^s

    对 s=1, 需要正则化 (条件收敛).
    使用球对称截断 |n| <= n_max.

    参数
    ----
    q_sq : float
        无量纲动量平方.
    n_max : int
        截断半径.

    返回
    ----
    Z : float
    """
    result = 0.0
    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                n_sq = nx ** 2 + ny ** 2 + nz ** 2
                diff = n_sq - q_sq
                if abs(diff) > 1e-10:
                    result += 1.0 / diff
    # 解析部分: 减除发散
    # Z_{00}^{reg} = lim_{s->1} [Z(s) - 4*pi*Lambda^{3-2s}/(3-2s)]
    # 简化: 直接用有限和近似
    return result / (4.0 * np.pi)


def luscher_delta_E(L: float, m: float, a0: float,
                    n_particles: int = 2) -> float:
    """Lüscher 有限体积能移 (两粒子态).

    delta_E = E(L) - 2*m
            ≈ -12 * a0 / (m * L^3) * exp(-m*L/sqrt(3))
                * [1 + O(1/(m*L))]

    参数
    ----
    L : float
        空间尺寸.
    m : float
        粒子质量.
    a0 : float
        s-波散射长度.
    n_particles : int
        粒子数 (默认 2).
    """
    if L <= 0 or m <= 0:
        return 0.0

    mL = m * L
    # 领头阶指数衰减
    prefactor = -12.0 * a0 / (m * L ** 3)
    exponential = np.exp(-mL / np.sqrt(3.0))
    # 次领头修正
    correction = 1.0 + 1.0 / mL + 0.5 / mL ** 2

    return prefactor * exponential * correction


def finite_volume_correction(mass: float, L: float,
                             order: str = 'leading') -> float:
    """单粒子质量的有限体积修正.

    delta_m(L) / m = -3 * sum_{n != 0} int d^3k/(2pi)^3
                      * exp(i*n*L*k) / (2*omega_k)
                    ≈ c * (m*L)^{-3/2} * exp(-m*L)

    参数
    ----
    mass : float
        粒子质量.
    L : float
        空间尺寸.
    order : str
        'leading' 或 'next_to_leading'.

    返回
    ----
    delta_m : float
        质量修正量.
    """
    if mass <= 0 or L <= 0:
        return 0.0

    mL = mass * L
    # 领头阶 (Lüscher 1986)
    coeff = -3.0 / (4.0 * np.pi ** 1.5)
    delta = coeff * mass * (mL) ** (-1.5) * np.exp(-mL)

    if order == 'next_to_leading':
        # NLO 修正
        delta *= (1.0 + 5.0 / (8.0 * mL) - 15.0 / (128.0 * mL ** 2))

    return delta


def infinite_volume_extrap(masses_L: np.ndarray,
                           L_values: np.ndarray,
                           mass: float) -> Dict:
    """有限体积外推到无限体积.

    m(L) = m_inf + c * L^{-3/2} * exp(-m*L)

    通过拟合确定 m_inf.

    返回
    ----
    result : dict
    """
    if len(L_values) < 2:
        return {'m_inf': masses_L[0] if len(masses_L) > 0 else mass,
                'finite_volume_shift': 0.0}

    # 预测有限体积修正
    fv_corrections = np.array([
        finite_volume_correction(mass, L) for L in L_values
    ])

    # 外推: m_inf ≈ mean(m(L) - delta_m(L))
    m_inf_estimates = masses_L - fv_corrections
    m_inf = float(np.mean(m_inf_estimates))

    return {
        'm_inf': m_inf,
        'finite_volume_shift': float(masses_L[-1] - m_inf),
        'corrections': fv_corrections.tolist()
    }
