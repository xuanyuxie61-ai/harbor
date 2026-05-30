# -*- coding: utf-8 -*-

import numpy as np
from utils import clip_positive


HC = 6.62607015e-34 * 2.99792458e8
NA = 6.02214076e23


def vandermonde_matrix(x, degree):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = degree
    V = np.zeros((n, m + 1))
    for j in range(m + 1):
        V[:, j] = x ** j
    return V


def vandermonde_approx_coef(x, y, degree):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x 与 y 长度不一致")
    if x.size < degree + 1:
        raise ValueError("数据点不足以确定多项式系数")
    V = vandermonde_matrix(x, degree)
    c, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
    return c


def eval_polynomial(c, x):
    c = np.asarray(c)
    x = np.asarray(x)
    if c.size == 0:
        return np.zeros_like(x)
    p = c[-1] * np.ones_like(x, dtype=float)
    for coef in c[-2::-1]:
        p = p * x + coef
    return p



_O3_WAVELENGTH_NM = np.array([
    200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0,
    280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0
])
_O3_CROSS_SECTION_CM2 = np.array([
    3.00e-17, 2.50e-17, 1.80e-17, 1.10e-17, 6.00e-18,
    3.00e-18, 1.20e-18, 4.00e-19, 1.20e-19, 3.50e-20,
    1.00e-20, 3.00e-21, 1.00e-21, 5.00e-22, 3.00e-22, 2.00e-22
])


_O2_WAVELENGTH_NM = np.array([
    200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0
])
_O2_CROSS_SECTION_CM2 = np.array([
    1.0e-20, 5.0e-21, 2.0e-21, 8.0e-22, 3.0e-22,
    1.0e-22, 4.0e-23, 1.5e-23, 5.0e-24
])


class PhotolysisRateCalculator:

    def __init__(self, degree=7):
        self.degree = degree

        self._c_o3 = vandermonde_approx_coef(
            _O3_WAVELENGTH_NM, np.log(clip_positive(_O3_CROSS_SECTION_CM2)), degree
        )

        self._c_o2 = vandermonde_approx_coef(
            _O2_WAVELENGTH_NM, np.log(clip_positive(_O2_CROSS_SECTION_CM2)), degree
        )

    def sigma_o3(self, wavelength_nm, temperature_k=220.0):
        lam = np.asarray(wavelength_nm, dtype=float)
        log_sigma = eval_polynomial(self._c_o3, lam)
        sigma = np.exp(np.clip(log_sigma, -80.0, -10.0))

        sigma *= np.exp(-0.001 * (temperature_k - 220.0))
        return sigma

    def sigma_o2(self, wavelength_nm):
        lam = np.asarray(wavelength_nm, dtype=float)
        log_sigma = eval_polynomial(self._c_o2, lam)
        return np.exp(np.clip(log_sigma, -80.0, -10.0))

    def spectral_actinic_flux(self, z_km, wavelength_nm, solar_zenith_deg,
                              col_o3_cm2, col_o2_cm2, f_top=1e14):
        mu0 = np.cos(np.radians(solar_zenith_deg))
        mu0 = clip_positive(mu0, 1e-3)
        sec_theta = 1.0 / mu0
        s_o3 = self.sigma_o3(wavelength_nm)
        s_o2 = self.sigma_o2(wavelength_nm)
        tau = sec_theta * (col_o3_cm2 * s_o3 + col_o2_cm2 * s_o2)
        tau = np.clip(tau, 0.0, 700.0)
        flux = f_top * np.exp(-tau)
        return flux

    def photolysis_rate_o3(self, z_km, solar_zenith_deg,
                           col_o3_cm2, col_o2_cm2, temperature_k=220.0,
                           lambda_min=200.0, lambda_max=350.0, n_quad=64):

        xi, wi = np.polynomial.legendre.leggauss(n_quad)
        lam = 0.5 * (lambda_max - lambda_min) * xi + 0.5 * (lambda_max + lambda_min)
        factor = 0.5 * (lambda_max - lambda_min)

        sigma = self.sigma_o3(lam, temperature_k)
        phi = np.ones_like(lam)
        flux = self.spectral_actinic_flux(
            z_km, lam, solar_zenith_deg, col_o3_cm2, col_o2_cm2
        )
        integrand = sigma * phi * flux
        J = factor * np.sum(wi * integrand)
        return float(np.clip(J, 0.0, 1e-2))

    def photolysis_rate_o2(self, z_km, solar_zenith_deg,
                           col_o3_cm2, col_o2_cm2,
                           lambda_min=200.0, lambda_max=240.0, n_quad=48):
        xi, wi = np.polynomial.legendre.leggauss(n_quad)
        lam = 0.5 * (lambda_max - lambda_min) * xi + 0.5 * (lambda_max + lambda_min)
        factor = 0.5 * (lambda_max - lambda_min)
        sigma = self.sigma_o2(lam)
        phi = np.ones_like(lam)
        flux = self.spectral_actinic_flux(
            z_km, lam, solar_zenith_deg, col_o3_cm2, col_o2_cm2
        )
        integrand = sigma * phi * flux
        J = factor * np.sum(wi * integrand)
        return float(np.clip(J, 0.0, 1e-8))
