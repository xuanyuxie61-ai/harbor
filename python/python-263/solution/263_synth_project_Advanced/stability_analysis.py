# -*- coding: utf-8 -*-
"""
stability_analysis.py
---------------------
日冕 MHD 高阶有限差分格式的稳定性分析.

分析内容
--------
1) von Neumann 稳定性分析 (傅里叶模式分析):
   对线性化 MHD 方程, 代入傅里叶模式 exp(i k x - i omega t),
   得到色散关系 D(omega, k) = 0. 稳定性要求 Im(omega) <= 0.

2) CFL 条件 (Courant-Friedrichs-Lewy):
   dt <= C_cfl * min(h / |v|, h / v_A, h^2 / (2 kappa))

3) 矩阵谱半径: 离散算子特征值实部 <= 0.

4) 数值耗散分析: 人工粘性对高频模式的阻尼率.
"""
from __future__ import annotations
import numpy as np
from solar_constants import VACUUM_PERMEABILITY, BOLTZMANN, PROTON_MASS


def von_neumann_analysis_fd4(v_flow: float, h: float,
                             kappa: float, dt: float) -> dict:
    """对四阶中心差分 + 显式 RK4 进行 von Neumann 分析.

    模型方程: du/dt + v du/dx = kappa d^2 u / dx^2

    代入 u_j^n = G^n exp(i k j h), 得到放大因子 G(k).
    稳定性要求 |G| <= 1 for all k in [-pi/h, pi/h].
    """
    k_grid = np.linspace(-np.pi / h, np.pi / h, 200)
    # 四阶差分对一阶导的修正波数
    k_eff_1 = np.sin(k_grid * h) / h + np.sin(2 * k_grid * h) / (12 * h)
    # 四阶差分对二阶导的修正波数
    k_eff_2 = -(4.0 / 3.0) * np.sin(k_grid * h / 2)**2 / (h**2 / 4) \
        - (1.0 / 3.0) * np.sin(k_grid * h)**2 / h**2

    # RK4 放大因子 (线性)
    # G = 1 + z + z^2/2 + z^3/6 + z^4/24, z = dt * (-i v k_eff + kappa k2_eff)
    z = dt * (-1j * v_flow * k_eff_1 + kappa * k_eff_2)
    G = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    G_mag = np.abs(G)

    return dict(
        k_grid=k_grid,
        G_magnitude=G_mag,
        max_amplification=float(G_mag.max()),
        stable=bool(G_mag.max() <= 1.0 + 1.0e-10),
    )


def cfl_condition(v_flow: np.ndarray, v_alfven: np.ndarray,
                  kappa: np.ndarray, h: np.ndarray,
                  c_cfl: float = 0.5) -> dict:
    """计算 CFL 限制时步.

    dt_adv = C * h / (|v| + v_A)
    dt_diff = C * h^2 / (2 kappa)
    dt_max = min(dt_adv, dt_diff)
    """
    v_total = np.abs(v_flow) + v_alfven
    dt_adv = c_cfl * h / (v_total + 1.0e-12)
    dt_diff = c_cfl * h**2 / (2.0 * kappa + 1.0e-12)
    dt_max = np.minimum(dt_adv, dt_diff)
    return dict(
        dt_adv=float(dt_adv.min()),
        dt_diff=float(dt_diff.min()),
        dt_max=float(dt_max.min()),
        cfl_number=float(c_cfl),
    )


def spectral_radius(A: np.ndarray) -> float:
    """矩阵谱半径 rho(A) = max |lambda_i|.
    稳定性要求 rho(I + dt A) <= 1."""
    evals = np.linalg.eigvals(A)
    return float(np.max(np.abs(evals)))


def dissipation_rate(v_flow: float, kappa: float, h: float,
                     k_mode: np.ndarray) -> np.ndarray:
    """数值耗散率: -Im(omega) 作为波数 k 的函数.

    omega(k) = v k + i kappa k^2 + 人工粘性贡献.
    """
    omega_imag = -kappa * k_mode**2
    # 人工粘性 q ~ c_quad h^2 |dv/dx| 贡献额外耗散
    c_quad = 1.0
    omega_artificial = -c_quad * h**2 * np.abs(k_mode)**3
    return omega_imag + omega_artificial


def grid_reynolds_number(v_flow: float, h: float,
                         nu_visc: float = 1.0e-3) -> float:
    """网格 Reynolds 数: Re_h = |v| h / nu.
    Re_h >> 1 意味着对流主导, 需要人工粘性."""
    return abs(v_flow) * h / (nu_visc + 1.0e-12)


def modified_wavenumber_analysis(h: float) -> dict:
    """分析不同差分格式的修正波数 (分辨率精度):

    - 二阶中心: k_eff h = sin(k h)
    - 四阶中心: k_eff h = (8 sin(k h/2) - sin(k h)) / 6  (近似)
    - 紧凑四阶: k_eff h = 3/2 sin(k h) / (1 + cos(k h))  (Padé)
    """
    k_h = np.linspace(0, np.pi, 100)
    k_eff_2 = np.sin(k_h)
    k_eff_4 = (8.0 * np.sin(k_h / 2.0) - np.sin(k_h)) / 6.0
    k_eff_compact = 1.5 * np.sin(k_h) / (1.0 + np.cos(k_h) + 1.0e-12)

    return dict(
        k_h=k_h,
        k_eff_2=k_eff_2,
        k_eff_4=k_eff_4,
        k_eff_compact=k_eff_compact,
        dispersion_2=float(np.max(np.abs(k_eff_2 - k_h))),
        dispersion_4=float(np.max(np.abs(k_eff_4 - k_h))),
        dispersion_compact=float(np.max(np.abs(k_eff_compact - k_h))),
    )


def stability_summary(z_grid: np.ndarray, v_flow: np.ndarray,
                      b_field: np.ndarray, temperature: np.ndarray,
                      density: np.ndarray, dt: float) -> dict:
    """综合稳定性诊断."""
    h = np.diff(z_grid)
    h_min = h.min()
    from solar_constants import alven_speed, spitzer_conductivity
    v_A = np.array([alven_speed(b, n) for b, n in zip(b_field, density)])
    kappa = np.array([spitzer_conductivity(t) for t in temperature])
    # 对齐到 cell faces (取相邻节点平均)
    v_f = 0.5 * (v_flow[:-1] + v_flow[1:])
    vA_f = 0.5 * (v_A[:-1] + v_A[1:])
    kappa_f = 0.5 * (kappa[:-1] + kappa[1:])

    cfl = cfl_condition(v_f, vA_f, kappa_f, h)
    cfl_ratio = dt / (cfl["dt_max"] + 1.0e-30)

    re_h = np.array([grid_reynolds_number(v, hi) for v, hi in zip(v_f, h)])

    return dict(
        h_min=float(h_min),
        v_A_max=float(v_A.max()),
        kappa_max=float(kappa.max()),
        dt_cfl=cfl["dt_max"],
        dt_ratio=float(cfl_ratio),
        cfl_stable=bool(cfl_ratio <= 1.0),
        re_h_max=float(re_h.max()),
    )
