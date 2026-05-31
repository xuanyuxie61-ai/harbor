"""
elliptic_integrals.py
Carlson对称形式不完全椭圆积分

基于种子项目:
- 1274_toms577: Carlson不完全椭圆积分 RF, RC, RD, RJ

物理应用:
1. 在QGP色散关系中，椭圆积分出现在高能部分子的能量损失计算中。
2. 在相对论性重离子碰撞的流体动力学中，椭圆积分用于解析解的构造。
3. 在胶子饱和物理中，椭圆积分描述色玻璃凝聚态中的部分子分布。

数学定义:
  RF(x,y,z) = (1/2) ∫_0^∞ dt / √[(t+x)(t+y)(t+z)]
  RC(x,y)   = (1/2) ∫_0^∞ dt / [(t+x)^{1/2} (t+y)]
  RD(x,y,z) = (3/2) ∫_0^∞ dt / [√(t+x)(t+y) (t+z)^{3/2}]
  RJ(x,y,z,p) = (3/2) ∫_0^∞ dt / [√(t+x)(t+y)(t+z) (t+p)]
"""

import numpy as np
from typing import Tuple


def rf_carlson(x: float, y: float, z: float, errtol: float = 1e-5) -> Tuple[float, int]:
    """
    计算Carlson对称形式不完全椭圆积分 RF(x,y,z)。

    使用迭代平均算法 (B.C. Carlson, 1979):
    λ_n = √(x_n y_n) + √(x_n z_n) + √(y_n z_n)
    x_{n+1} = (x_n + λ_n) / 4
    y_{n+1} = (y_n + λ_n) / 4
    z_{n+1} = (z_n + λ_n) / 4

    收敛后展开到5阶:
    RF ≈ μ^{-1/2} [1 + E2/24 - 3E3/44 + E2²/14 - ...]

    Parameters
    ----------
    x, y, z : float
        非负参数，至多一个为零
    errtol : float
        误差容限

    Returns
    -------
    value : float
        RF(x,y,z)
    ierr : int
        0=成功, 1=参数错误
    """
    if x < 0.0 or y < 0.0 or z < 0.0:
        return 0.0, 1
    if x + y < 1e-30 or x + z < 1e-30 or y + z < 1e-30:
        return 0.0, 1

    ierr = 0
    xn, yn, zn = x, y, z

    for _ in range(100):
        mu = (xn + yn + zn) / 3.0
        xdev = 1.0 - xn / mu
        ydev = 1.0 - yn / mu
        zdev = 1.0 - zn / mu
        eps = max(abs(xdev), abs(ydev), abs(zdev))
        if eps < errtol:
            break
        lam = np.sqrt(xn * yn) + np.sqrt(xn * zn) + np.sqrt(yn * zn)
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0

    e2 = xdev * ydev - zdev * zdev
    e3 = xdev * ydev * zdev
    s = 1.0 + (e2 / 24.0 - 0.1 * e3 + 3.0 * e2 * e2 / 56.0)
    value = s / np.sqrt(mu)
    return value, ierr


def rc_carlson(x: float, y: float, errtol: float = 1e-5) -> Tuple[float, int]:
    """
    计算Carlson退化椭圆积分 RC(x,y)。

    RC(x,y) = (1/2) ∫_0^∞ dt / [(t+x)^{1/2} (t+y)]

    Parameters
    ----------
    x : float
        非负参数
    y : float
        正参数
    errtol : float
        误差容限

    Returns
    -------
    value : float
        RC(x,y)
    ierr : int
        0=成功, 1=参数错误
    """
    if x < 0.0 or y <= 0.0:
        return 0.0, 1

    ierr = 0
    xn, yn = x, y

    for _ in range(100):
        mu = (xn + 2.0 * yn) / 3.0
        xdev = 1.0 - xn / mu
        ydev = 1.0 - yn / mu
        eps = max(abs(xdev), abs(ydev))
        if eps < errtol:
            break
        lam = 2.0 * np.sqrt(xn * yn) + yn
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0

    s = 1.0 + xdev * xdev * (3.0 / 14.0 + xdev * (1.0 / 6.0 + 3.0 * xdev / 22.0))
    value = s / np.sqrt(mu)
    return value, ierr


def rd_carlson(x: float, y: float, z: float, errtol: float = 1e-5) -> Tuple[float, int]:
    """
    计算Carlson椭圆积分 RD(x,y,z)。

    RD(x,y,z) = (3/2) ∫_0^∞ dt / [√(t+x)(t+y) (t+z)^{3/2}]

    Parameters
    ----------
    x, y : float
        非负参数
    z : float
        正参数
    errtol : float
        误差容限

    Returns
    -------
    value : float
        RD(x,y,z)
    ierr : int
        0=成功, 1=参数错误
    """
    if x < 0.0 or y < 0.0 or z <= 0.0:
        return 0.0, 1
    if x + y < 1e-30:
        return 0.0, 1

    ierr = 0
    xn, yn, zn = x, y, z
    sigma = 0.0
    power4 = 1.0

    for _ in range(100):
        mu = (xn + yn + 3.0 * zn) / 5.0
        xdev = 1.0 - xn / mu
        ydev = 1.0 - yn / mu
        zdev = 1.0 - zn / mu
        eps = max(abs(xdev), abs(ydev), abs(zdev))
        if eps < errtol:
            break
        lam = np.sqrt(xn * yn) + np.sqrt(xn * zn) + np.sqrt(yn * zn)
        sigma += power4 / (np.sqrt(zn) * (zn + lam))
        power4 *= 0.25
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0

    ea = xdev * ydev
    eb = zdev * zdev
    ec = ea - eb
    ed = ea - 6.0 * eb
    ef = ed + ec + ec
    s1 = 1.0 + ed * (-0.21428571428571427 + 0.10227272727272728 * ed - 0.12152861952861953 * zdev * ef)
    s2 = zdev * (0.3333333333333333 + zdev * (-0.14285714285714285 + 0.07662337662337662 * zdev))
    s3 = xdev * ydev / zn * 0.3333333333333333 - xdev * ydev * zdev * 0.14285714285714285
    value = 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * np.sqrt(mu))
    return value, ierr


def rj_carlson(x: float, y: float, z: float, p: float,
               errtol: float = 1e-5) -> Tuple[float, int]:
    """
    计算Carlson椭圆积分 RJ(x,y,z,p)。

    RJ(x,y,z,p) = (3/2) ∫_0^∞ dt / [√(t+x)(t+y)(t+z) (t+p)]

    Parameters
    ----------
    x, y, z : float
        非负参数
    p : float
        正参数
    errtol : float
        误差容限

    Returns
    -------
    value : float
        RJ(x,y,z,p)
    ierr : int
        0=成功, 1=参数错误
    """
    if x < 0.0 or y < 0.0 or z < 0.0 or p <= 0.0:
        return 0.0, 1
    if x + y < 1e-30 or x + z < 1e-30 or y + z < 1e-30:
        return 0.0, 1

    ierr = 0
    xn, yn, zn, pn = x, y, z, p
    sigma = 0.0
    power4 = 1.0

    for _ in range(100):
        mu = (xn + yn + zn + 2.0 * pn) / 5.0
        xdev = 1.0 - xn / mu
        ydev = 1.0 - yn / mu
        zdev = 1.0 - zn / mu
        pdev = 1.0 - pn / mu
        eps = max(abs(xdev), abs(ydev), abs(zdev), abs(pdev))
        if eps < errtol:
            break
        lam = np.sqrt(xn * yn) + np.sqrt(xn * zn) + np.sqrt(yn * zn)
        alfa = pn * (np.sqrt(lam + pn) + np.sqrt(lam)) ** 2
        beta = pn * (pn + lam) ** 2 / alfa if alfa > 1e-30 else 0.0
        sigma += power4 * rc_carlson(1.0, beta, errtol)[0]
        power4 *= 0.25
        xn = (xn + lam) / 4.0
        yn = (yn + lam) / 4.0
        zn = (zn + lam) / 4.0
        pn = (pn + lam) / 4.0

    ea = xdev * (ydev + zdev) + ydev * zdev
    eb = xdev * ydev * zdev
    ec = pdev * pdev
    e2 = ea - 3.0 * ec
    e3 = eb + 2.0 * pdev * (ea - ec)
    e4 = (eb + ec * (2.0 * ea - 5.0 * ec)) * pdev
    e5 = ec * ec * pdev * (ea - 3.0 * ec)
    s1 = 1.0 - 0.3 * e2 + 0.1 * e3 + e2 * e2 * (0.21428571428571427 - 0.10227272727272728 * e3)
    s2 = e4 * (0.07142857142857142 - 0.045454545454545456 * e2) + 0.03787878787878788 * e5
    s3 = e2 * e4 * 0.045454545454545456 + e3 * (-0.017316017316017316)
    value = 6.0 * sigma + power4 * (s1 + s2 + s3) / (mu * np.sqrt(mu))
    return value, ierr


class QGPDispersionRelation:
    """
    QGP中的色散关系与部分子能量损失计算。
    """

    @staticmethod
    def gluon_energy_loss(q: float, m_g: float, T: float) -> float:
        """
        计算胶子在QGP中的能量损失 (简化模型)。

        利用RC积分表示色散积分的解析延拓:
        ΔE ∝ T² ∫ dω ω / (ω² - m_g²) RC(1, ω²/T²)

        Parameters
        ----------
        q : float
            胶子动量 [GeV]
        m_g : float
            热胶子质量 [GeV]
        T : float
            温度 [GeV]

        Returns
        -------
        float
            能量损失 [GeV]
        """
        if T < 1e-6 or q < 1e-6:
            return 0.0
        x = (m_g / T) ** 2
        y = (q / T) ** 2
        rc_val, ierr = rc_carlson(x, y)
        if ierr != 0:
            return 0.0
        # 简化模型: ΔE ∝ q T² RC
        delta_E = 0.2 * q * (T ** 2) * rc_val
        return delta_E

    @staticmethod
    def parton_momentum_broadening(k_t: float, q_s: float, L: float) -> float:
        """
        计算部分子横向动量展宽 (BDMPS-Z/LPM效应)。

        ⟨p_⊥²⟩ ∝ q̂ L，其中 q̂ 为输运系数。

        利用RF积分表示非阿贝尔LPM效应:
        ⟨k_⊥²⟩ = q_s² L · RF(1, (k_t/q_s)², 1 + L/L_form)

        Parameters
        ----------
        k_t : float
            横向动量 [GeV]
        q_s : float
            饱和动量 [GeV]
        L : float
            介质长度 [fm]

        Returns
        -------
        float
            动量展宽 [GeV²]
        """
        if q_s < 1e-6 or L < 1e-6:
            return 0.0
        L_form = 1.0 / q_s  # 形成长度
        x, y, z = 1.0, (k_t / q_s) ** 2, 1.0 + L / L_form
        rf_val, ierr = rf_carlson(x, y, z)
        if ierr != 0:
            return 0.0
        p2 = (q_s ** 2) * L * rf_val
        return p2

    @staticmethod
    def dilepton_spectral_function(q: float, T: float, m_l: float) -> float:
        """
        双轻子谱函数 (通过RJ积分表示)。

        在QGP中，双轻子产生率与电磁流关联函数相关:
        dN/d⁴q ∝ Im Π^{em}(q₀, q) · L(q, T)

        其中Lindhard函数可用椭圆积分表示:
        L(q,T) ∝ RJ(1, (q/2T)², (m_l/T)², 1 + q₀/T)

        Parameters
        ----------
        q : float
            三维动量 [GeV]
        T : float
            温度 [GeV]
        m_l : float
            轻子质量 [GeV]

        Returns
        -------
        float
            谱函数值 [arb. units]
        """
        if T < 1e-6:
            return 0.0
        q0 = np.sqrt(q ** 2 + m_l ** 2)
        x = 1.0
        y = (q / (2.0 * T)) ** 2
        z = (m_l / T) ** 2
        p = 1.0 + q0 / T
        rj_val, ierr = rj_carlson(x, y, z, p)
        if ierr != 0:
            return 0.0
        # 谱函数正比于RJ
        rho = np.exp(-q0 / T) * rj_val / (2.0 * np.pi ** 3)
        return rho
