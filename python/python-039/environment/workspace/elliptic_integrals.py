
import numpy as np
from typing import Tuple


def rf_carlson(x: float, y: float, z: float, errtol: float = 1e-5) -> Tuple[float, int]:
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

    @staticmethod
    def gluon_energy_loss(q: float, m_g: float, T: float) -> float:
        if T < 1e-6 or q < 1e-6:
            return 0.0
        x = (m_g / T) ** 2
        y = (q / T) ** 2
        rc_val, ierr = rc_carlson(x, y)
        if ierr != 0:
            return 0.0

        delta_E = 0.2 * q * (T ** 2) * rc_val
        return delta_E

    @staticmethod
    def parton_momentum_broadening(k_t: float, q_s: float, L: float) -> float:
        if q_s < 1e-6 or L < 1e-6:
            return 0.0
        L_form = 1.0 / q_s
        x, y, z = 1.0, (k_t / q_s) ** 2, 1.0 + L / L_form
        rf_val, ierr = rf_carlson(x, y, z)
        if ierr != 0:
            return 0.0
        p2 = (q_s ** 2) * L * rf_val
        return p2

    @staticmethod
    def dilepton_spectral_function(q: float, T: float, m_l: float) -> float:
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

        rho = np.exp(-q0 / T) * rj_val / (2.0 * np.pi ** 3)
        return rho
