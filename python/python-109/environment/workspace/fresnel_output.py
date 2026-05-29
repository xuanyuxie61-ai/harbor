"""
fresnel_output.py
光纤输出端近场 Fresnel 衍射积分计算

融合原项目:
  - 448_fresnel: Fresnel 余弦/正弦积分 C(x), S(x) 的高精度计算

科学背景:
  超连续谱脉冲从光纤端面出射后，在自由空间中经历衍射展宽。
  近场区域的衍射可用 Fresnel 积分精确描述。对于一维狭缝或圆形孔径，
  衍射场分布涉及 Fresnel 积分:
      C(x) = integral_0^x cos(pi*t^2/2) dt
      S(x) = integral_0^x sin(pi*t^2/2) dt

  在轴对称光纤输出中，输出场分布可表示为:
      E(r, z) = (k / (i*z)) * integral_0^a E0(r') * J0(k*r*r'/z) *
                exp(i*k*r'^2 / (2*z)) * r' dr'
  其中 a 为纤芯半径，J0 为 Bessel 函数，k = 2*pi/lambda。
  当 r=0（轴上点）且 z -> 0（近场极限）时，积分退化为 Fresnel 积分形式。

  本模块提供 Fresnel 积分的高精度数值计算及光纤输出端面的衍射场重建。
"""

import numpy as np
from typing import Tuple


def fresnel_integrals(x: float, eps: float = 1e-15) -> Tuple[float, float]:
    """
    计算 Fresnel 积分 C(x) 和 S(x)。

    公式:
        C(x) = integral_0^x cos(pi*t^2/2) dt
        S(x) = integral_0^x sin(pi*t^2/2) dt

    算法（分段处理）:
        1. |x| < 2.5: 级数展开
           C(x) = x * sum_{k=0}^inf (-1)^k * (pi/2)^{2k} * x^{4k} / ((4k)! * (4k+1))
           实际实现使用递推形式以提高数值稳定性。
        2. 2.5 <= |x| < 4.5: 连分数/递推关系
        3. |x| >= 4.5: 渐近展开
           C(x) ~ 0.5 + (f*sin(t0) - g*cos(t0)) / (pi*x)
           S(x) ~ 0.5 - (f*cos(t0) + g*sin(t0)) / (pi*x)
           其中 t = pi*x^2/2, t0 = t mod 2*pi
           f = sum_{k=0}^{20} r_k,  g = sum_{k=0}^{12} s_k

    Parameters
    ----------
    x : float
        自变量。
    eps : float
        级数收敛阈值。

    Returns
    -------
    tuple
        (C, S)
    """
    xa = abs(x)
    px = np.pi * xa
    t = 0.5 * px * xa
    t2 = t * t

    if xa == 0.0:
        return 0.0, 0.0
    elif xa < 2.5:
        # 级数展开 C
        r = xa
        c = r
        for k in range(1, 50):
            r = (-0.5 * r * (4.0 * k - 3.0) / k
                 / (2.0 * k - 1.0) / (4.0 * k + 1.0) * t2)
            c += r
            if abs(r) < abs(c) * eps:
                break
        # 级数展开 S
        s = xa * t / 3.0
        r = s
        for k in range(1, 50):
            r = (-0.5 * r * (4.0 * k - 1.0) / k
                 / (2.0 * k + 1.0) / (4.0 * k + 3.0) * t2)
            s += r
            if abs(r) < abs(s) * eps:
                if x < 0.0:
                    c = -c
                    s = -s
                return float(c), float(s)
        if x < 0.0:
            c = -c
            s = -s
        return float(c), float(s)
    elif xa < 4.5:
        m = int(np.floor(42.0 + 1.75 * t))
        su = 0.0
        c = 0.0
        s_val = 0.0
        f1 = 0.0
        f0 = 1.0e-100
        for k in range(m, -1, -1):
            f = (2.0 * k + 3.0) * f0 / t - f1
            if k == int(k / 2) * 2:
                c += f
            else:
                s_val += f
            su += (2.0 * k + 1.0) * f * f
            f1 = f0
            f0 = f
        q = np.sqrt(su)
        c = c * xa / q
        s_val = s_val * xa / q
        if x < 0.0:
            c = -c
            s_val = -s_val
        return float(c), float(s_val)
    else:
        # 渐近展开
        r = 1.0
        f = 1.0
        for k in range(1, 20):
            r = (-0.25 * r * (4.0 * k - 1.0) * (4.0 * k - 3.0) / t2)
            f += r
        r = 1.0 / (px * xa)
        g = r
        for k in range(1, 12):
            r = (-0.25 * r * (4.0 * k + 1.0) * (4.0 * k - 1.0) / t2)
            g += r
        t0 = t - np.floor(t / (2.0 * np.pi)) * 2.0 * np.pi
        c = 0.5 + (f * np.sin(t0) - g * np.cos(t0)) / px
        s_val = 0.5 - (f * np.cos(t0) + g * np.sin(t0)) / px
        if x < 0.0:
            c = -c
            s_val = -s_val
        return float(c), float(s_val)


def fresnel_diffraction_1d(aperture_field: np.ndarray,
                           x_aperture: np.ndarray,
                           x_observation: np.ndarray,
                           wavelength: float,
                           z: float) -> np.ndarray:
    """
    一维 Fresnel 衍射数值积分（向量化实现）。

    公式（Fresnel-Kirchhoff 近似）:
        E(x, z) = (exp(i*k*z) / sqrt(i*lambda*z)) *
                  integral E0(x') * exp(i*k*(x-x')^2 / (2*z)) dx'

    离散形式（向量化）:
        E(x_m) = C * dx * sum_n E0(x_n) * exp(i*k*(x_m - x_n)^2 / (2*z))

    Parameters
    ----------
    aperture_field : np.ndarray
        孔径场分布（复数组）。
    x_aperture : np.ndarray
        孔径坐标。
    x_observation : np.ndarray
        观察面坐标。
    wavelength : float
        波长（m）。
    z : float
        传播距离（m）。

    Returns
    -------
    np.ndarray
        观察面上的场分布。
    """
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fresnel_diffraction_1d: wavelength and z must be > 0")
    k = 2.0 * np.pi / wavelength
    dx = x_aperture[1] - x_aperture[0]
    prefactor = np.exp(1j * k * z) / np.sqrt(1j * wavelength * z) * dx
    # 向量化：构造 (n_obs, n_aperture) 的相位矩阵
    diff = x_observation[:, np.newaxis] - x_aperture[np.newaxis, :]
    phase = np.exp(1j * k * diff ** 2 / (2.0 * z))
    E_out = prefactor * np.sum(aperture_field[np.newaxis, :] * phase, axis=1)
    return E_out


def fiber_output_diffraction(E_fundamental: np.ndarray,
                              r_fiber: np.ndarray,
                              r_out: np.ndarray,
                              wavelength: float,
                              z: float,
                              core_radius: float) -> np.ndarray:
    """
    轴对称光纤输出场的 Fresnel 衍射（Hankel 变换形式）。

    公式:
        E(r, z) = (2*pi*k / (i*z)) * integral_0^a E0(r') *
                  J0(k*r*r'/z) * exp(i*k*r'^2/(2*z)) * r' dr'

    其中 J0 为零阶 Bessel 函数，a 为纤芯半径。
    当 z 很小时，此积分退化为近场极限 E(r,0) ≈ E0(r)。

    Parameters
    ----------
    E_fundamental : np.ndarray
        光纤端面上的基模场分布（径向）。
    r_fiber : np.ndarray
        光纤端面的径向坐标。
    r_out : np.ndarray
        输出面的径向坐标。
    wavelength : float
        波长（m）。
    z : float
        传播距离（m）。
    core_radius : float
        纤芯半径（m）。

    Returns
    -------
    np.ndarray
        输出面场分布。
    """
    from scipy.special import j0
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fiber_output_diffraction: wavelength and z must be > 0")
    k = 2.0 * np.pi / wavelength
    E_out = np.zeros_like(r_out, dtype=complex)
    dr = r_fiber[1] - r_fiber[0]
    for i, r in enumerate(r_out):
        integrand = E_fundamental * j0(k * r * r_fiber / z) * np.exp(1j * k * r_fiber ** 2 / (2.0 * z)) * r_fiber
        E_out[i] = (2.0 * np.pi * k / (1j * z)) * np.sum(integrand) * dr
    return E_out


def fresnel_number(a: float, wavelength: float, z: float) -> float:
    """
    计算 Fresnel 数。

    公式:
        N_F = a^2 / (lambda * z)

    当 N_F >> 1 时为近场（Fresnel 衍射区），
    当 N_F << 1 时为远场（Fraunhofer 衍射区）。

    Parameters
    ----------
    a : float
        孔径半径（m）。
    wavelength : float
        波长（m）。
    z : float
        传播距离（m）。

    Returns
    -------
    float
        Fresnel 数。
    """
    if wavelength <= 0.0 or z <= 0.0:
        raise ValueError("fresnel_number: wavelength and z must be > 0")
    return a * a / (wavelength * z)
