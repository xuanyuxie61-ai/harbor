"""
球谐函数与矢量球谐函数展开 (spherical_harmonics.py)
=====================================================
为地核发电机模拟提供完整的球谐函数系计算，包括：
  - 连带 Legendre 函数 P_l^m(cos(theta))
  - 标量球谐函数 Y_l^m(theta, phi)
  - 矢量球谐函数（环向 T_l^m，极向 P_l^m）
  - 球谐变换（综合与分析）
  - 球谐系数的能量谱（Mauersberger-Lowes 谱）

数学公式:
  Y_l^m(theta, phi) = N_l^m * P_l^m(cos(theta)) * exp(i*m*phi)
  N_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)

  环向矢量球谐:  T_l^m = r_hat x grad(Y_l^m)
  极向矢量球谐:  P_l^m = r_hat x T_l^m = grad(Y_l^m)
（注：在球面上，T_l^m 给出切向无散分量，P_l^m 给出切向有势分量）
"""

import numpy as np
from typing import Tuple, List


def associated_legendre(l: int, m: int, x: float) -> float:
    """
    计算连带 Legendre 函数 P_l^m(x)，采用标准递推公式。
    要求 |x| <= 1，m >= 0。

    递推关系:
      P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
      P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
      (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    """
    x = float(x)
    if abs(x) > 1.0 + 1e-12:
        raise ValueError("|x| must be <= 1 for Legendre functions")
    x = max(-1.0, min(1.0, x))
    m = abs(m)
    if m > l:
        return 0.0

    # P_m^m
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m:
        return pmm

    # P_{m+1}^m
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    # 递推到 l
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def spherical_harmonic_normalization(l: int, m: int) -> float:
    """计算球谐函数归一化常数 N_l^m。"""
    m = abs(m)
    # 使用对数避免大阶乘溢出
    log_num = 0.0
    for k in range(l - m + 1, l + m + 1):
        log_num += np.log(float(k))
    # (l-m)!/(l+m)! = 1 / [(l-m+1)...(l+m)]
    fact_ratio = np.exp(-log_num) if log_num < 700 else 0.0
    if m == 0:
        fact_ratio = 1.0 / np.math.factorial(l) / np.math.factorial(l)
        fact_ratio = 1.0
        # 更精确的直接计算
        num = 1.0
        den = 1.0
        for k in range(1, l + m + 1):
            den *= k
        for k in range(1, l - m + 1):
            num *= k
        fact_ratio = num / den
    else:
        num = 1.0
        den = 1.0
        for k in range(1, l + m + 1):
            den *= k
        for k in range(1, l - m + 1):
            num *= k
        fact_ratio = num / den

    return np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi) * fact_ratio)


def scalar_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    """
    计算标量球谐函数 Y_l^m(theta, phi)。
    theta : 极角 (0, pi)
    phi   : 方位角 (0, 2pi)
    """
    x = np.cos(theta)
    plm = associated_legendre(l, m, x)
    N = spherical_harmonic_normalization(l, m)
    ylm = N * plm * np.exp(1j * m * phi)
    return ylm


def toroidal_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> Tuple[complex, complex, complex]:
    """
    环向矢量球谐函数 T_l^m = r_hat x grad(Y_l^m)。
    在球坐标系中，其分量为:
      T_r = 0
      T_theta = (1/sin(theta)) * (i*m) * Y_l^m / r
      T_phi   = - d(Y_l^m)/dtheta / r
    这里省略 1/r 因子（纯角向函数）。

    返回 (T_r, T_theta, T_phi)
    """
    ylm = scalar_spherical_harmonic(l, m, theta, phi)

    # 对 theta 的数值导数（中心差分）
    dtheta = 1e-6
    thp = min(np.pi - 1e-8, theta + dtheta)
    thm = max(1e-8, theta - dtheta)
    ylm_p = scalar_spherical_harmonic(l, m, thp, phi)
    ylm_m = scalar_spherical_harmonic(l, m, thm, phi)
    dylm_dtheta = (ylm_p - ylm_m) / (thp - thm)

    T_r = 0.0 + 0.0j
    sin_t = max(1e-15, np.sin(theta))
    T_theta = 1j * m * ylm / sin_t
    T_phi = -dylm_dtheta

    return (T_r, T_theta, T_phi)


def poloidal_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> Tuple[complex, complex, complex]:
    """
    极向矢量球谐函数 P_l^m = grad(Y_l^m)（球面切向部分）。
    分量:
      P_r = 0  (取纯球面切向)
      P_theta = d(Y_l^m)/dtheta / r
      P_phi   = (i*m / sin(theta)) * Y_l^m / r
    这里省略 1/r 因子。

    返回 (P_r, P_theta, P_phi)
    """
    ylm = scalar_spherical_harmonic(l, m, theta, phi)

    dtheta = 1e-6
    thp = min(np.pi - 1e-8, theta + dtheta)
    thm = max(1e-8, theta - dtheta)
    ylm_p = scalar_spherical_harmonic(l, m, thp, phi)
    ylm_m = scalar_spherical_harmonic(l, m, thm, phi)
    dylm_dtheta = (ylm_p - ylm_m) / (thp - thm)

    sin_t = max(1e-15, np.sin(theta))
    P_r = 0.0 + 0.0j
    P_theta = dylm_dtheta
    P_phi = 1j * m * ylm / sin_t

    return (P_r, P_theta, P_phi)


# ---------------------------------------------------------------------------
# 球谐变换（伪谱法）：在均匀经纬网格上分析/综合
# ---------------------------------------------------------------------------
def gauss_legendre_grid(n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Gauss-Legendre 纬向网格点和权重。
    返回 (theta_nodes, weights)。
    """
    # 使用 numpy.polynomial.legendre.leggauss 计算
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # x in [-1, 1], theta = arccos(x)
    theta = np.arccos(x)
    return theta, w


def spherical_harmonic_analysis(field: np.ndarray, l_max: int) -> np.ndarray:
    """
    将标量场 field(theta, phi) 分解为球谐系数。
    field 形状为 (n_theta, n_phi)。
    返回系数数组 coeffs[l, m+l]，其中 l=0..l_max, m=-l..l。
    """
    n_theta, n_phi = field.shape
    theta, w = gauss_legendre_grid(n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    coeffs = np.zeros((l_max + 1, 2 * l_max + 1), dtype=complex)

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            integral = 0.0 + 0.0j
            for it in range(n_theta):
                sin_t = np.sin(theta[it])
                for ip in range(n_phi):
                    ylm_conj = np.conj(scalar_spherical_harmonic(l, m, theta[it], phi[ip]))
                    integral += field[it, ip] * ylm_conj * w[it] * dphi * sin_t
            coeffs[l, m + l_max] = integral

    return coeffs


def spherical_harmonic_synthesis(coeffs: np.ndarray, n_theta: int, n_phi: int) -> np.ndarray:
    """
    由球谐系数 reconstruct 标量场。
    coeffs 形状为 (l_max+1, 2*l_max+1)。
    """
    l_max = coeffs.shape[0] - 1
    theta, _ = gauss_legendre_grid(n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    field = np.zeros((n_theta, n_phi), dtype=complex)
    for it in range(n_theta):
        for ip in range(n_phi):
            val = 0.0 + 0.0j
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    val += coeffs[l, m + l_max] * scalar_spherical_harmonic(l, m, theta[it], phi[ip])
            field[it, ip] = val

    return np.real(field)


# ---------------------------------------------------------------------------
# Mauersberger-Lowes 能量谱
#    R_l = (l+1) * sum_{m=-l}^{l} |g_l^m|^2
#    其中 g_l^m 为磁场球谐系数（高斯系数）
# ---------------------------------------------------------------------------
def mauersberger_lowes_spectrum(coeffs: np.ndarray) -> np.ndarray:
    """
    计算 Mauersberger-Lowes 能量谱 R_l。
    coeffs: 球谐系数数组，形状 (l_max+1, 2*l_max+1)。
    """
    l_max = coeffs.shape[0] - 1
    spectrum = np.zeros(l_max + 1, dtype=float)
    for l in range(l_max + 1):
        energy = 0.0
        for m in range(-l, l + 1):
            c = coeffs[l, m + l_max]
            energy += abs(c) ** 2
        spectrum[l] = (l + 1.0) * energy
    return spectrum


# ---------------------------------------------------------------------------
# 地核发电机专用：偶极子倾角计算
# ---------------------------------------------------------------------------
def dipole_inclination(g10: float, g11: float, h11: float) -> float:
    """
    根据地磁偶极子球谐系数计算磁偶极子倾角。
    g10: l=1, m=0 系数
    g11: l=1, m=1 余弦系数
    h11: l=1, m=1 正弦系数

    偶极子倾角公式:
      tan(I) = 2 * g10 / sqrt(g11^2 + h11^2)
      I = arctan(tan(I))
    """
    denom = np.sqrt(g11 ** 2 + h11 ** 2)
    if denom < 1e-30:
        return np.pi / 2.0 if g10 > 0 else -np.pi / 2.0
    return np.arctan2(2.0 * g10, denom)


def dipole_moment(g10: float, g11: float, h11: float, radius: float) -> float:
    """
    计算地磁偶极矩大小 (单位: A*m^2)。
    公式: mu = (4*pi / mu0) * R^3 * sqrt(g10^2 + g11^2 + h11^2)
    这里 mu0 = 4*pi*1e-7 H/m。
    """
    mu0 = 4.0 * np.pi * 1.0e-7
    return (4.0 * np.pi / mu0) * (radius ** 3) * np.sqrt(g10 ** 2 + g11 ** 2 + h11 ** 2)


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 归一化检查：|Y_0^0|^2 在球面积分 = 1
    theta, w = gauss_legendre_grid(32)
    n_phi = 64
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    integral = 0.0
    for it in range(len(theta)):
        for ip in range(n_phi):
            y = scalar_spherical_harmonic(0, 0, theta[it], phi[ip])
            integral += abs(y) ** 2 * np.sin(theta[it]) * w[it] * dphi
    assert abs(integral - 1.0) < 1e-3, f"Normalization failed: {integral}"

    # Y_1^0 检查
    val = scalar_spherical_harmonic(1, 0, np.pi / 2.0, 0.0)
    assert abs(val - 0.0) < 1e-10  # P_1^0(cos(pi/2)) = 0

    # 倾角检查
    inc = dipole_inclination(1.0, 0.0, 0.0)
    assert abs(inc - np.pi / 2.0) < 1e-10

    print("spherical_harmonics: self-test passed.")


if __name__ == "__main__":
    _self_test()
