# -*- coding: utf-8 -*-
"""
photolysis_rates.py
平流层光解速率计算模块。

核心物理：光解速率常数 J 定义为

    J_i(z, \theta_0, T) = \int_{\lambda_{\min}}^{\lambda_{\max}}
        \sigma_i(\lambda, T) \, \Phi_i(\lambda) \, F(z, \lambda, \theta_0) \, d\lambda

其中：
  - σ_i(λ,T)：物种 i 的吸收截面 [cm^2]
  - Φ_i(λ)：光解量子产额 [无量纲]
  - F(z,λ,θ_0)：光谱光化通量 [photons cm^{-2} s^{-1} nm^{-1}]

融合来源：
  - 1382_vandermonde_approx_1d: 一维范德蒙德多项式近似吸收截面
"""

import numpy as np
from utils import clip_positive

# 物理常数
HC = 6.62607015e-34 * 2.99792458e8  # J·m
NA = 6.02214076e23


def vandermonde_matrix(x, degree):
    r"""
    构造范德蒙德矩阵：

        V_{ij} = x_i^{j}, \quad j = 0, 1, \dots, m

    Parameters
    ----------
    x : ndarray, shape (n,)
    degree : int
        多项式次数 m。

    Returns
    -------
    V : ndarray, shape (n, m+1)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    m = degree
    V = np.zeros((n, m + 1))
    for j in range(m + 1):
        V[:, j] = x ** j
    return V


def vandermonde_approx_coef(x, y, degree):
    r"""
    最小二乘范德蒙德多项式拟合：求解

        \min_{\mathbf{c}} \| V \mathbf{c} - \mathbf{y} \|_2^2

    其中 V 为范德蒙德矩阵。拟合多项式为

        p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_m x^m

    Parameters
    ----------
    x : ndarray, shape (n,)
    y : ndarray, shape (n,)
    degree : int
        多项式次数 m。

    Returns
    -------
    c : ndarray, shape (m+1,)
        多项式系数（c[0] 为常数项）。
    """
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
    r"""
    Horner 法则求多项式值：

        p(x) = c_0 + c_1 x + \dots + c_m x^m

    Parameters
    ----------
    c : ndarray
    x : float or ndarray

    Returns
    -------
    p : float or ndarray
    """
    c = np.asarray(c)
    x = np.asarray(x)
    if c.size == 0:
        return np.zeros_like(x)
    p = c[-1] * np.ones_like(x, dtype=float)
    for coef in c[-2::-1]:
        p = p * x + coef
    return p


# 预计算的 O3 吸收截面数据（近似，基于 Hartley-Huggins 带）
_O3_WAVELENGTH_NM = np.array([
    200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0,
    280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0
])
_O3_CROSS_SECTION_CM2 = np.array([
    3.00e-17, 2.50e-17, 1.80e-17, 1.10e-17, 6.00e-18,
    3.00e-18, 1.20e-18, 4.00e-19, 1.20e-19, 3.50e-20,
    1.00e-20, 3.00e-21, 1.00e-21, 5.00e-22, 3.00e-22, 2.00e-22
])

# O2 吸收截面（Schumann-Runge 带近似）
_O2_WAVELENGTH_NM = np.array([
    200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0
])
_O2_CROSS_SECTION_CM2 = np.array([
    1.0e-20, 5.0e-21, 2.0e-21, 8.0e-22, 3.0e-22,
    1.0e-22, 4.0e-23, 1.5e-23, 5.0e-24
])


class PhotolysisRateCalculator:
    r"""
    平流层光解速率计算器。

    使用范德蒙德多项式近似吸收截面 σ(λ)，并通过 Beer-Lambert 定律
    计算光谱光化通量：

        F(z, \lambda, \theta_0) = F_{\infty}(\lambda) \,
            \exp\!\left(-\sec\theta_0 \sum_j N_j(z) \sigma_j(\lambda)\right)

    其中 N_j(z) 为物种 j 的垂直柱浓度 [molecules cm^{-2}]。
    """

    def __init__(self, degree=7):
        self.degree = degree
        # 拟合 O3 吸收截面
        self._c_o3 = vandermonde_approx_coef(
            _O3_WAVELENGTH_NM, np.log(clip_positive(_O3_CROSS_SECTION_CM2)), degree
        )
        # 拟合 O2 吸收截面
        self._c_o2 = vandermonde_approx_coef(
            _O2_WAVELENGTH_NM, np.log(clip_positive(_O2_CROSS_SECTION_CM2)), degree
        )

    def sigma_o3(self, wavelength_nm, temperature_k=220.0):
        r"""
        O3 吸收截面，含温度修正（Bass-Paur 近似）：

            \sigma(T) = \sigma(220\,\mathrm{K}) \times
                \exp\!\left[ -0.001 \times (T - 220) \right]

        Parameters
        ----------
        wavelength_nm : float or ndarray
        temperature_k : float

        Returns
        -------
        sigma : float or ndarray [cm^2]
        """
        lam = np.asarray(wavelength_nm, dtype=float)
        log_sigma = eval_polynomial(self._c_o3, lam)
        sigma = np.exp(np.clip(log_sigma, -80.0, -10.0))
        # 温度修正
        sigma *= np.exp(-0.001 * (temperature_k - 220.0))
        return sigma

    def sigma_o2(self, wavelength_nm):
        r"""
        O2 吸收截面。

        Parameters
        ----------
        wavelength_nm : float or ndarray

        Returns
        -------
        sigma : float or ndarray [cm^2]
        """
        lam = np.asarray(wavelength_nm, dtype=float)
        log_sigma = eval_polynomial(self._c_o2, lam)
        return np.exp(np.clip(log_sigma, -80.0, -10.0))

    def spectral_actinic_flux(self, z_km, wavelength_nm, solar_zenith_deg,
                              col_o3_cm2, col_o2_cm2, f_top=1e14):
        r"""
        计算光谱光化通量。

        Beer-Lambert 衰减：

            F(z, \lambda) = F_{\top}(\lambda) \,
                \exp\!\left[-\sec\theta_0 \,
                    \bigl( N_{\mathrm{O}_3}(z)\sigma_{\mathrm{O}_3}(\lambda)
                         + N_{\mathrm{O}_2}(z)\sigma_{\mathrm{O}_2}(\lambda) \bigr)
                \right]

        Parameters
        ----------
        z_km : float
            高度 [km]。
        wavelength_nm : float or ndarray
        solar_zenith_deg : float
            太阳天顶角 [°]。
        col_o3_cm2 : float
            O3 柱浓度 [molecules cm^{-2}]。
        col_o2_cm2 : float
            O2 柱浓度 [molecules cm^{-2}]。
        f_top : float
            顶界入射通量 [photons cm^{-2} s^{-1} nm^{-1}]。

        Returns
        -------
        flux : float or ndarray
        """
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
        r"""
        计算 O3 → O2 + O(1D) 的光解速率 J_O3：

            J_{\mathrm{O}_3} = \int_{200}^{350}
                \sigma_{\mathrm{O}_3}(\lambda, T) \,
                \Phi(\lambda) \,
                F(z, \lambda, \theta_0) \, d\lambda

        其中量子产额 Φ(λ) 在 Hartley 带近似为 1.0。

        Parameters
        ----------
        z_km : float
        solar_zenith_deg : float
        col_o3_cm2 : float
        col_o2_cm2 : float
        temperature_k : float
        lambda_min, lambda_max : float
        n_quad : int
            Gauss-Legendre 积分节点数。

        Returns
        -------
        J : float [s^{-1}]
        """
        # Gauss-Legendre 节点与权重（线性映射到 [lambda_min, lambda_max]）
        xi, wi = np.polynomial.legendre.leggauss(n_quad)
        lam = 0.5 * (lambda_max - lambda_min) * xi + 0.5 * (lambda_max + lambda_min)
        factor = 0.5 * (lambda_max - lambda_min)

        sigma = self.sigma_o3(lam, temperature_k)
        phi = np.ones_like(lam)  # 近似量子产额
        flux = self.spectral_actinic_flux(
            z_km, lam, solar_zenith_deg, col_o3_cm2, col_o2_cm2
        )
        integrand = sigma * phi * flux
        J = factor * np.sum(wi * integrand)
        return float(np.clip(J, 0.0, 1e-2))

    def photolysis_rate_o2(self, z_km, solar_zenith_deg,
                           col_o3_cm2, col_o2_cm2,
                           lambda_min=200.0, lambda_max=240.0, n_quad=48):
        r"""
        计算 O2 → 2O 的光解速率 J_O2（Schumann-Runge 带）。

        Parameters
        ----------
        z_km : float
        solar_zenith_deg : float
        col_o3_cm2 : float
        col_o2_cm2 : float

        Returns
        -------
        J : float [s^{-1}]
        """
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
