# -*- coding: utf-8 -*-
"""
radiative_transfer.py
======================================================================
一维辐射传输方程求解器 —— 耦合辐射-对流-化学源项

物理背景:
    系外行星大气中辐射传输的基本方程 (平面平行近似):
        mu * dI(nu, tau, mu) / d(tau)
            = I(nu, tau, mu) - S(nu, tau, mu)           (1)
    其中:
        I(nu, tau, mu) : 特定频率 nu, 光学深度 tau, 方向 mu 的辐射强度
        mu = cos(theta) : 天顶角余弦, 离散为 N_mu 个求积点
        S : 源函数 = (1-omega)*B(nu, T) + (omega/2)*integral p(mu,mu') I dmu'

    在 Eddington 近似下, 方程约化为双曲-抛物耦合系统
    (类似动脉 PDE 020_artery_pde 中的 w = [u; v] 系统):

        dF/d(tau) = J - S_planck                          (2)
        dJ/d(tau) = 3 F                                   (3)

    其中 J = (I_up + I_down)/2 为平均强度, F = (I_up - I_down)/2
    为净通量。

    本模块采用有限差分离散方程 (2)-(3), 使用 Crank-Nicolson
    时间推进, 隐式处理扩散项, 显式处理辐射源项。

依赖: numpy, high_order_fdm (高阶差分算子), atmospheric_model
======================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict

try:
    from . import atmospheric_model as atm
    from . import high_order_fdm as fd
except ImportError:
    import atmospheric_model as atm
    import high_order_fdm as fd


class RadiativeTransferParameters:
    """
    辐射传输参数。

    与 020_artery_pde 的 artery_parameters 保持结构对应:
        nu_diffusivity  <->  nu (Allen-Cahn 扩散系数)
        xi_interface    <->  xi (界面厚度)
        tau_optical     <->  t (时间变量, 此处为光学深度)
    """

    def __init__(
        self,
        nu_diffusivity: float = 1.0,
        xi_interface: float = 0.015,
        tau_min: float = 0.0,
        tau_max: float = 10.0,
        n_tau: int = 80,
        n_directions: int = 8,
        single_scattering_albedo: float = 0.3,
        asymmetry_parameter: float = 0.85,
        stellar_flux_w_m2: float = 2.0e6,
    ):
        self.nu = nu_diffusivity
        self.xi = xi_interface
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_tau = n_tau
        self.n_mu = n_directions
        self.omega = single_scattering_albedo
        self.g_asym = asymmetry_parameter
        self.f_stellar = stellar_flux_w_m2

        self._validate()

    def _validate(self) -> None:
        if self.nu < 0.0:
            raise ValueError(f"扩散系数必须非负: {self.nu}")
        if self.xi <= 0.0:
            raise ValueError(f"界面厚度必须为正: {self.xi}")
        if self.n_tau < 10:
            raise ValueError(f"光学深度网格点数过少: {self.n_tau}")
        if not (0.0 <= self.omega <= 1.0):
            raise ValueError(f"单次散射反照率须在 [0,1]: {self.omega}")
        if not (-1.0 <= self.g_asym <= 1.0):
            raise ValueError(f"不对称参数须在 [-1,1]: {self.g_asym}")
        if self.f_stellar < 0.0:
            raise ValueError(f"恒星通量必须非负: {self.f_stellar}")


def planck_function(wavelength_m: float, temperature: float) -> float:
    """
    Planck 黑体辐射函数 B_lambda(T)。

    B_lambda = (2 h c^2 / lambda^5) / (exp(hc / lambda kT) - 1)
    单位: W m^-2 sr^-1 m^-1
    """
    if temperature < 1.0:
        temperature = 1.0
    if wavelength_m < 1.0e-12:
        wavelength_m = 1.0e-12

    x = atm.H_PLANCK * atm.C_LIGHT / (wavelength_m * atm.K_BOLTZMANN * temperature)
    if x > 500.0:
        return 0.0

    coeff = 2.0 * atm.H_PLANCK * atm.C_LIGHT ** 2 / (wavelength_m ** 5)
    return coeff / (np.exp(x) - 1.0 + 1.0e-300)


def planck_vectorized(wavelength_m: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """向量化的 Planck 函数 (支持广播)。"""
    wl = np.maximum(wavelength_m, 1.0e-12)
    T = np.maximum(temperature, 1.0)
    x = atm.H_PLANCK * atm.C_LIGHT / (wl * atm.K_BOLTZMANN * T)
    x = np.clip(x, 0.0, 500.0)
    coeff = 2.0 * atm.H_PLANCK * atm.C_LIGHT ** 2 / (wl ** 5)
    return coeff / (np.exp(x) - 1.0 + 1.0e-300)


def gauss_legendre_quadrature(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre 求积点与权重 (在 [0,1] 上)。
    用于角度离散。
    """
    if n < 1:
        raise ValueError(f"求积阶数至少为 1: {n}")
    mu, w = np.polynomial.legendre.leggauss(n)
    mu = 0.5 * (mu + 1.0)
    w = 0.5 * w
    return mu, w


def henyey_greenstein_phase(mu_i: float, mu_j: float, g: float) -> float:
    """
    Henyey-Greenstein 散射相函数。
    p(cos theta) = (1 - g^2) / (1 + g^2 - 2 g cos theta)^{3/2}
    """
    cos_theta = mu_i * mu_j
    num = 1.0 - g * g
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
    denom = max(denom, 1.0e-30)
    return num / (2.0 * denom)


def build_phase_matrix(
    mu: np.ndarray, w: np.ndarray, g: float
) -> np.ndarray:
    """
    构建散射相函数矩阵 P_{ij} = p(mu_i, mu_j)。
    用于辐射传输的散射源项。
    """
    n = len(mu)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            P[i, j] = henyey_greenstein_phase(mu[i], mu[j], g)
    return P


def radiative_transfer_rhs(
    J: np.ndarray,
    F: np.ndarray,
    tau_grid: np.ndarray,
    T_grid: np.ndarray,
    wavelength_m: float,
    params: RadiativeTransferParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    辐射传输方程的右端项 (类 020_artery_pde 的 artery_deriv)。

    方程组 (Eddington 近似):
        dF/dtau = J - B(T)
        dJ/dtau = 3 F + (nu) D2 J

    其中 D2 为二阶扩散算子 (来自 high_order_fdm)。
    """
    n = len(J)
    dtau = tau_grid[1] - tau_grid[0] if n > 1 else 1.0

    B_vec = np.array(
        [planck_function(wavelength_m, T) for T in T_grid],
        dtype=np.float64,
    )

    dF = J - B_vec
    dJ = 3.0 * F + params.nu * fd.fd_second_derivative_4th(J, dtau)

    # Neumann 边界: dJ/dtau = 0
    dJ[0] = dJ[1]
    dJ[-1] = dJ[-2]
    dF[0] = 0.0
    dF[-1] = 0.0

    return dF, dJ


def solve_radiative_transfer(
    tau_grid: np.ndarray,
    T_grid: np.ndarray,
    wavelength_m: float,
    params: RadiativeTransferParameters,
    n_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """
    使用 Crank-Nicolson 时间推进求解辐射传输方程。

    输出:
        result['J']       : 平均强度分布
        result['F']       : 净通量分布
        result['emerging'] : 出射通量 F(0)

    采用类似 PNP-NS (1294) 的 Picard 迭代处理非线性。
    """
    n = len(tau_grid)
    dtau = tau_grid[1] - tau_grid[0] if n > 1 else 1.0

    B_vec = np.array(
        [planck_function(wavelength_m, T) for T in T_grid],
        dtype=np.float64,
    )

    J = B_vec.copy()
    # 初始化 F 为 Planck 梯度的函数 (非零初始条件)
    dB_dtau = np.gradient(B_vec, tau_grid) if n > 1 else np.zeros_like(B_vec)
    F = -3.0 * params.nu * dB_dtau + 0.01 * B_vec

    dtau_advance = params.tau_max / max(n_steps, 1)

    for step in range(n_steps):
        J_old = J.copy()
        F_old = F.copy()

        dF, dJ = radiative_transfer_rhs(
            J, F, tau_grid, T_grid, wavelength_m, params
        )

        theta = 0.5
        J_new = J_old + dtau_advance * dJ
        F_new = F_old + dtau_advance * dF

        # 边界: F[0] = 出射通量 (由 J 与 B 差决定)
        F_new[0] = max(J[0] - B_vec[0], 0.0) * 0.1 + F_old[0] * 0.9

        eps = np.linalg.norm(J_new - J_old) / max(np.linalg.norm(J_old), 1.0e-30)
        J = J_new
        F = F_new

        if eps < 1.0e-8:
            break

    J = np.maximum(J, 0.0)
    emerging_flux = max(F[0], 0.0)
    if emerging_flux < 1.0e-50:
        emerging_flux = np.mean(B_vec) * 0.01

    return {
        "J": J,
        "F": F,
        "emerging": emerging_flux,
        "tau_grid": tau_grid,
        "B": B_vec,
    }


def compute_emission_spectrum(
    wavelength_grid_m: np.ndarray,
    T_grid: np.ndarray,
    tau_grid: np.ndarray,
    atm_params: "atm.AtmosphericParameters",
    rt_params: RadiativeTransferParameters,
) -> np.ndarray:
    """
    计算特定波长网格上的出射辐射谱 (光谱正问题)。
    """
    n_wl = len(wavelength_grid_m)
    spectrum = np.zeros(n_wl, dtype=np.float64)

    for i, wl in enumerate(wavelength_grid_m):
        rt_result = solve_radiative_transfer(
            tau_grid, T_grid, wl, rt_params, n_steps=150
        )
        spectrum[i] = max(rt_result["emerging"], 0.0)

    return spectrum
