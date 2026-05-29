"""
integrals_radiation.py
圆形活塞辐射器与声场积分计算

融合原始项目:
  - 300_disk01_integrands (圆盘采样与积分)

科学背景:
  圆形活塞辐射器是扬声器的基本模型.其辐射声压可通过
  Rayleigh积分计算:

      p(\vec{r}) = (j rho0 omega / 2pi) \int_S
                   u_n(\vec{r}') * exp(-j k |\vec{r} - \vec{r}'|)
                   / |\vec{r} - \vec{r}'| dS'

  其中 S 为活塞表面, u_n 为法向振速.

  对于均匀振速 u_n = U0, 远场指向性函数为:
      D(\theta) = 2 J_1(k a sin\theta) / (k a sin\theta)

  本模块实现:
  1. 圆盘上均匀/随机采样 (用于数值积分)
  2. 基于cos_power_int的指向性积分
  3. 活塞辐射阻抗的实部与虚部计算
"""

import numpy as np
import math
from special_functions import cos_power_int


def disk_unit_sample(n, radius=1.0):
    """
    在单位圆盘上均匀随机采样.

    采样策略:
        r = R * sqrt(u),  u~U[0,1]
        theta = 2*pi*v,   v~U[0,1]
        保证面积元均匀.

    参数:
        n: 采样点数
        radius: 圆盘半径

    返回:
        points: (n, 2) 坐标
    """
    rng = np.random.default_rng(42)
    r = radius * np.sqrt(rng.random(n))
    theta = 2.0 * math.pi * rng.random(n)
    points = np.zeros((n, 2), dtype=float)
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)
    return points


def rayleigh_integral_piston(observer, disk_samples, u_n, k, rho0=1.225, c0=343.0):
    """
    通过离散化Rayleigh积分计算活塞辐射声压.

    参数:
        observer: (3,) 观察点坐标 [m]
        disk_samples: (n, 2) 圆盘采样点 (z=0平面)
        u_n: 法向振速幅值 [m/s] 或 (n,) 分布
        k: 波数 [rad/m]
        rho0: 空气密度
        c0: 声速

    返回:
        p: 复声压 [Pa]
    """
    omega = k * c0
    n_samp = disk_samples.shape[0]
    disk_area = math.pi * (np.max(np.linalg.norm(disk_samples, axis=1)) ** 2)
    dS = disk_area / n_samp

    if np.isscalar(u_n):
        u_n = np.full(n_samp, u_n)

    p = 0.0 + 0.0j
    for i in range(n_samp):
        r_prime = np.array([disk_samples[i, 0], disk_samples[i, 1], 0.0])
        R_vec = observer - r_prime
        R = np.linalg.norm(R_vec)
        if R < 1e-6:
            R = 1e-6
        p += u_n[i] * np.exp(-1j * k * R) / R * dS

    p = p * (1j * rho0 * omega / (2.0 * math.pi))
    return p


def piston_directivity_factor(ka, n_points=180):
    """
    计算圆形活塞的指向性因子.

    理论公式:
        D(\theta) = 2 J_1(ka sin\theta) / (ka sin\theta)

    指向性因子 (DI):
        DI = 10 log10( 4pi / \int_0^{2pi}\int_0^{pi/2} D^2(\theta) sin\theta d\theta d\phi )

    为简化,本函数使用数值积分估算.
    """
    from scipy.special import j1

    theta = np.linspace(0, math.pi / 2, n_points)
    dtheta = theta[1] - theta[0]

    D = np.zeros_like(theta)
    for i, th in enumerate(theta):
        s = math.sin(th)
        if ka * s < 1e-6:
            D[i] = 1.0
        else:
            D[i] = 2.0 * j1(ka * s) / (ka * s)

    # 积分 \int D^2 sin\theta d\theta
    integrand = D ** 2 * np.sin(theta)
    integral = np.trapz(integrand, theta)

    if integral < 1e-12:
        return 0.0

    # 轴对称,phi积分给出2pi
    omni_int = 2.0 * math.pi * integral
    di = 10.0 * math.log10(4.0 * math.pi / omni_int)
    return di


def piston_radiation_resistance(ka):
    """
    圆形活塞的辐射阻力比 R_r / (rho0 c0 S).

    理论公式:
        R_r / (rho0 c0 S) = 1 - 2 J_1(2ka) / (2ka)
    """
    from scipy.special import j1
    if ka < 1e-6:
        return (ka ** 2) / 2.0  # 小ka近似
    return 1.0 - 2.0 * j1(2.0 * ka) / (2.0 * ka)


def piston_radiation_reactance(ka):
    """
    圆形活塞的辐射抗比 X_r / (rho0 c0 S).

    理论公式:
        X_r / (rho0 c0 S) = 2 H_1(2ka) / (2ka)
        其中 H_1 为Struve函数.
    """
    from scipy.special import struve
    if ka < 1e-6:
        return 8.0 * ka / (3.0 * math.pi)
    return 2.0 * struve(1, 2.0 * ka) / (2.0 * ka)
