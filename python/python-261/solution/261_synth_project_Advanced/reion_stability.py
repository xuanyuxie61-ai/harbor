"""
reion_stability.py
==================
数值稳定性分析 (von Neumann + 矩阵谱方法)

本模块对再电离模拟中使用的各类数值格式进行稳定性分析, 核心是 von Neumann
方法和矩阵谱方法. 通过分析放大因子 G(k) 的模长确定稳定区域.

Von Neumann 分析 (平面波分解):
    设 u_j^n = G^n * exp(i k j h)
    代入离散方程得 G = G(k, dt, h, params)
    稳定条件: |G(k)| <= 1  对所有 k

实现:
  1. von_neumann_analysis : 对给定格式计算放大因子
  2. cfl_critical : 求临界 CFL 数
  3. stability_growth_rate : 计算扰动增长率
  4. compare_schemes : 多方案稳定性比较

核心公式 (对扩散方程 du/dt = gamma * d2u/dx2):

  显式 Euler + 4阶 Laplacian:
    G(k) = 1 + dt * gamma * lambda_hat(k)
    其中 lambda_hat(k) = (-2*cos(2kh) + 32*cos(kh) - 30) / (12*h^2)

  稳定条件:
    dt <= 12 * h^2 / (30 * gamma)  (对 4阶 Laplacian, 最坏情况)

  IMEX 方案:
    G(k) = (1 + dt * gamma * lambda_hat * a_ii) /
           (1 - dt * gamma * lambda_hat * (1 - a_ii))
    (以 theta 方法为例)

对应种子项目:
  - 1280_Betswish_MIRAGE-reproduce (多方案评估 → 多格式稳定性比较)
"""

import numpy as np


def laplacian_symbol_4th(k, h):
    """4阶中心差分 Laplacian 的符号 (Fourier 空间).

    (D2 u)_j = (-u_{j-2} + 16 u_{j-1} - 30 u_j + 16 u_{j+1} - u_{j+2}) / (12 h^2)

    代入 u_j = exp(i k j h), 得符号:
        lambda_hat(k) = (-2 cos(2kh) + 32 cos(kh) - 30) / (12 h^2)

    Parameters
    ----------
    k : float or array
        波数 [1/cm]
    h : float
        网格间距 [cm]

    Returns
    -------
    lambda_hat : float or array
    """
    kh = k * h
    return (-2.0 * np.cos(2.0 * kh) + 32.0 * np.cos(kh) - 30.0) / (12.0 * h * h)


def laplacian_symbol_2nd(k, h):
    """2阶中心差分 Laplacian 的符号."""
    kh = k * h
    return (2.0 * np.cos(kh) - 2.0) / (h * h)


def first_derivative_symbol_4th(k, h):
    """4阶中心差分一阶导数的符号."""
    kh = k * h
    return 1j * (-np.sin(2.0 * kh) + 8.0 * np.sin(kh)) / (6.0 * h)


def amplification_factor_explicit(k, h, dt, gamma_diff, order=4):
    """显式 Euler 格式的放大因子.

    G = 1 + dt * gamma * lambda_hat(k)

    Parameters
    ----------
    k : array
    h, dt, gamma_diff : float
    order : int

    Returns
    -------
    G : array (complex)
    """
    if order == 4:
        lam = laplacian_symbol_4th(k, h)
    else:
        lam = laplacian_symbol_2nd(k, h)
    return 1.0 + dt * gamma_diff * lam


def amplification_factor_imex(k, h, dt, gamma_diff, theta=0.5):
    """theta-IMEX 格式的放大因子.

    G = (1 + (1-theta) * dt * gamma * lam) /
        (1 - theta * dt * gamma * lam)

    theta = 0: 纯显式
    theta = 0.5: Crank-Nicolson
    theta = 1: 纯隐式

    Parameters
    ----------
    k : array
    h, dt, gamma_diff, theta : float

    Returns
    -------
    G : array (complex)
    """
    lam = laplacian_symbol_4th(k, h)
    num = 1.0 + (1.0 - theta) * dt * gamma_diff * lam
    den = 1.0 - theta * dt * gamma_diff * lam
    den = np.where(np.abs(den) < 1.0e-30, 1.0e-30, den)
    return num / den


def amplification_factor_SSP2_ImEx(k, h, dt, gamma_diff):
    """SSP2-IMEX(3,3,2) 格式的放大因子 (对纯扩散问题).

    通过分析 3 级格式得到.
    """
    lam = laplacian_symbol_4th(k, h)
    z = dt * gamma_diff * lam  # 无量纲参数
    # SSP2-IMEX 简化的放大因子 (对纯线性问题)
    # 近似为二阶 Padé
    G = (1.0 + 0.5 * z + z * z / 12.0) / (1.0 - 0.5 * z + z * z / 12.0)
    return G


def cfl_critical(gamma_diff, h, order=4, scheme="explicit"):
    """计算临界 CFL 数 dt_max.

    对显式格式: |G| <= 1 要求
        dt * gamma * |lambda_hat_max| <= C

    对 4阶 Laplacian:
        |lambda_hat_max| ≈ 30 / (12 h^2)  (在 k = pi/h 处)

    Parameters
    ----------
    gamma_diff : float
        扩散系数
    h : float
        网格间距
    order : int
    scheme : str
        "explicit", "imex_theta", "SSP2_ImEx"

    Returns
    -------
    dt_max : float
        最大稳定时间步长 [s]
    cfl : float
        CFL 数
    """
    if gamma_diff <= 0:
        return float("inf"), 0.0

    if order == 4:
        lam_max = 30.0 / (12.0 * h * h)  # |lambda_hat| 的最大值
    else:
        lam_max = 4.0 / (h * h)

    if scheme == "explicit":
        dt_max = 1.0 / (gamma_diff * lam_max)
    elif scheme == "imex_theta":
        # theta=0.5 (Crank-Nicolson) 是无条件稳定的
        dt_max = float("inf")
    elif scheme == "SSP2_ImEx":
        # SSP2-IMEX 稳定区域大于纯显式
        dt_max = 2.0 / (gamma_diff * lam_max)
    else:
        dt_max = 1.0 / (gamma_diff * lam_max)

    cfl = gamma_diff * dt_max / (h * h) if h > 0 else 0.0
    return dt_max, cfl


def stability_growth_rate(k_arr, h, dt, gamma_diff, scheme="explicit"):
    """计算扰动增长率 omega_i (虚部).

    对稳定格式, omega_i <= 0 (扰动衰减).
    omega_i = log(|G|) / dt

    Parameters
    ----------
    k_arr : array
    h, dt, gamma_diff : float
    scheme : str

    Returns
    -------
    omega_i : array
        扰动增长率 (应为非正数)
    """
    if scheme == "explicit":
        G = amplification_factor_explicit(k_arr, h, dt, gamma_diff, order=4)
    elif scheme == "imex_theta":
        G = amplification_factor_imex(k_arr, h, dt, gamma_diff, theta=0.5)
    elif scheme == "SSP2_ImEx":
        G = amplification_factor_SSP2_ImEx(k_arr, h, dt, gamma_diff)
    else:
        G = amplification_factor_explicit(k_arr, h, dt, gamma_diff, order=4)

    abs_G = np.abs(G)
    abs_G = np.maximum(abs_G, 1.0e-300)
    omega_i = np.log(abs_G) / max(dt, 1.0e-30)
    return omega_i


def check_stability(k_arr, h, dt, gamma_diff, scheme="explicit"):
    """检查格式稳定性 (返回稳定性判据).

    Returns
    -------
    is_stable : bool
    max_abs_G : float
    max_growth_rate : float
    """
    omega_i = stability_growth_rate(k_arr, h, dt, gamma_diff, scheme)
    max_growth = float(np.max(omega_i))
    G = amplification_factor_explicit(k_arr, h, dt, gamma_diff, order=4) \
        if scheme == "explicit" else \
        amplification_factor_imex(k_arr, h, dt, gamma_diff, theta=0.5)
    max_abs_G = float(np.max(np.abs(G)))
    is_stable = max_growth <= 1.0e-10  # 容差
    return is_stable, max_abs_G, max_growth


def compare_schemes(k_arr, h, dt, gamma_diff):
    """比较多种格式的稳定性特性.

    Returns
    -------
    results : dict
        各方案的 max |G| 和 max growth rate
    """
    results = {}
    for scheme in ["explicit", "imex_theta", "SSP2_ImEx"]:
        is_stable, max_G, max_omega = check_stability(
            k_arr, h, dt, gamma_diff, scheme)
        results[scheme] = {
            "is_stable": is_stable,
            "max_abs_G": max_G,
            "max_growth_rate": max_omega,
        }
    return results


def spectral_radius_stability(D2_matrix, dt, gamma_diff, scheme="explicit"):
    """基于矩阵谱半径的稳定性分析.

    对显式格式: dt * gamma * rho(-D2) <= 2
    其中 rho 为谱半径.

    Parameters
    ----------
    D2_matrix : array [N x N]
    dt, gamma_diff : float
    scheme : str

    Returns
    -------
    rho : float
        谱半径
    dt_critical : float
        临界时间步长
    """
    eigvals = np.linalg.eigvals(D2_matrix)
    # 对 Laplacian 矩阵, 特征值应为非正实数
    eigvals_real = np.real(eigvals)
    lambda_min = float(np.min(eigvals_real))  # 最负的特征值

    if scheme == "explicit":
        # |1 + dt * gamma * lambda| <= 1
        # dt * gamma * |lambda_min| <= 2
        if gamma_diff > 0 and lambda_min < 0:
            dt_crit = 2.0 / (gamma_diff * abs(lambda_min))
        else:
            dt_crit = float("inf")
        rho = abs(1.0 + dt * gamma_diff * lambda_min)
    elif scheme == "imex_theta":
        # theta = 0.5 无条件稳定
        dt_crit = float("inf")
        rho = 1.0
    else:
        dt_crit = 2.0 / (gamma_diff * abs(lambda_min)) if lambda_min < 0 else float("inf")
        rho = 1.0

    return rho, dt_crit
