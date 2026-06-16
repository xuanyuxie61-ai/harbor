"""
norms_utils.py
===================================================================
核反应波函数范数计算与概率流守恒验证模块

映射种子项目:
  - 813_norm_l2: L2 范数计算 → 散射波函数模方积分
  - 815_norm_rms: RMS 范数计算 → 径向波函数均方根归一化
  - 211_continuity_exact: 精确无散度场 → 概率流密度连续性方程验证

核心公式:
  L2 范数:     ||u||_2 = sqrt( integral_0^R |u(r)|^2 dr )
  RMS 范数:    ||u||_rms = sqrt( (1/R) * integral_0^R |u(r)|^2 dr )
  概率流密度:   j(r) = (hbar/m) Im( u* du/dr )
  连续性方程:   div(j) + d(rho)/dt = 0
  径向概率流:   J_l = (hbar*k/m) * |A_l|^2 (对弹性散射守恒)
===================================================================
"""

import numpy as np
from typing import Tuple, Callable, Optional


# ---------- 物理常数 (MeV, fm 单位制) ----------
HBAR_C = 197.3269804        # hbar*c [MeV·fm]
HBAR2_OVER_2M_N = 20.735    # hbar^2/(2*m_n) [MeV·fm^2] 中子约化质量
PROTON_MASS = 938.272       # 质子质量 [MeV/c^2]
NEUTRON_MASS = 939.565      # 中子质量 [MeV/c^2]


def l2_norm_radial(u: np.ndarray, r: np.ndarray) -> float:
    """
    计算径向波函数的 L2 范数:
        ||u||_2 = sqrt( integral_0^R |u(r)|^2 dr )

    使用 Simpson 复合求积公式 (O(h^4) 精度):
        integral ≈ (h/3) [f_0 + 4*f_1 + 2*f_2 + ... + 4*f_{n-1} + f_n]

    参数:
        u: 径向波函数数组 u(r), 可为复数
        r: 径向网格点数组

    返回:
        L2 范数值 (非负实数)
    """
    if len(r) < 3:
        raise ValueError("L2范数计算至少需要3个网格点")
    if len(u) != len(r):
        raise ValueError("波函数与网格点数组长度不匹配")

    # 概率密度 |u(r)|^2
    integrand = np.abs(u) ** 2

    # Simpson 积分 (处理奇数/偶数个区间)
    n = len(r) - 1
    if n % 2 == 0:
        # 标准 Simpson
        integral = _simpson_composite(integrand, r)
    else:
        # 前 n-1 个区间用 Simpson, 最后一个用梯形
        integral = _simpson_composite(integrand[:-1], r[:-1])
        integral += 0.5 * (r[-1] - r[-2]) * (integrand[-2] + integrand[-1])

    return float(np.sqrt(max(integral, 0.0)))


def rms_norm_radial(u: np.ndarray, r: np.ndarray) -> float:
    """
    计算径向波函数的 RMS (均方根) 范数:
        ||u||_rms = sqrt( (1/(R-R0)) * integral_{R0}^R |u(r)|^2 dr )

    物理含义: 核内部区域的平均概率密度开方

    参数:
        u: 径向波函数数组
        r: 径向网格点数组

    返回:
        RMS 范数值
    """
    domain_length = r[-1] - r[0]
    if domain_length <= 0:
        raise ValueError("积分区间长度必须为正")

    l2 = l2_norm_radial(u, r)
    return l2 / np.sqrt(domain_length)


def probability_current_density(
    u: np.ndarray,
    r: np.ndarray,
    mass_reduced: float = NEUTRON_MASS / 2.0
) -> np.ndarray:
    """
    计算径向概率流密度:
        j(r) = (hbar / m_red) * Im( u*(r) * du/dr )

    其中 du/dr 使用中心差分 (O(h^2)):
        du/dr|_i = (u_{i+1} - u_{i-1}) / (2*dr)

    参数:
        u: 复数径向波函数
        r: 径向网格
        mass_reduced: 约化质量 [MeV/c^2]

    返回:
        概率流密度数组 j(r) [1/(fm^2 · c)]
    """
    n = len(u)
    dr = r[1] - r[0] if n > 1 else 1.0
    j = np.zeros(n, dtype=np.float64)

    # 中心差分 (内部点)
    for i in range(1, n - 1):
        dudr = (u[i + 1] - u[i - 1]) / (2.0 * dr)
        j[i] = (HBAR_C / (mass_reduced * HBAR_C)) * np.imag(np.conj(u[i]) * dudr)

    # 边界: 单侧差分
    if n >= 2:
        dudr_0 = (u[1] - u[0]) / dr
        j[0] = (HBAR_C / (mass_reduced * HBAR_C)) * np.imag(np.conj(u[0]) * dudr_0)
        dudr_n = (u[-1] - u[-2]) / dr
        j[-1] = (HBAR_C / (mass_reduced * HBAR_C)) * np.imag(np.conj(u[-1]) * dudr_n)

    return j


def continuity_residual(
    u: np.ndarray,
    r: np.ndarray,
    V_opt: np.ndarray,
    energy: float,
    mass_reduced: float = NEUTRON_MASS / 2.0
) -> float:
    """
    验证概率流连续性方程的残差:
        R_cont = | d(r^2 * j)/dr + (2m/hbar^2) * r^2 * Im(V_opt) * |u|^2 |

    对弹性散射 (V_opt 为实数), 连续性方程简化为:
        d(r^2 * j)/dr = 0

    对含吸收的光学势 (Im(V) < 0):
        d(r^2 * j)/dr = -(2m/hbar^2) * r^2 * |Im(V)| * |u|^2

    参数:
        u: 径向波函数
        r: 径向网格
        V_opt: 光学势 (复数)
        energy: 入射能量 [MeV]
        mass_reduced: 约化质量

    返回:
        连续性方程残差的 L2 范数
    """
    n = len(r)
    if n < 3:
        return 0.0

    dr = r[1] - r[0]
    j = probability_current_density(u, r, mass_reduced)

    # 计算 d(r^2 * j)/dr
    r2j = r ** 2 * j
    dr2j = np.zeros(n)
    for i in range(1, n - 1):
        dr2j[i] = (r2j[i + 1] - r2j[i - 1]) / (2.0 * dr)
    dr2j[0] = (r2j[1] - r2j[0]) / dr
    dr2j[-1] = (r2j[-1] - r2j[-2]) / dr

    # 吸收源项
    imag_V = np.imag(V_opt)
    source = -(2.0 * mass_reduced / HBAR_C ** 2) * r ** 2 * imag_V * np.abs(u) ** 2

    residual = dr2j - source
    return float(np.sqrt(np.mean(residual ** 2)))


def wavefunction_norm_check(
    u: np.ndarray, r: np.ndarray,
    tol: float = 1e-6
) -> dict:
    """
    波函数归一化检查，返回诊断信息

    参数:
        u: 径向波函数
        r: 径向网格
        tol: 容差

    返回:
        包含 L2 范数, RMS 范数, 是否归一化等信息的字典
    """
    l2 = l2_norm_radial(u, r)
    rms = rms_norm_radial(u, r)

    # 均方根半径: <r^2> = integral r^2 |u|^2 dr / integral |u|^2 dr
    integrand_r2 = r ** 2 * np.abs(u) ** 2
    integrand_0 = np.abs(u) ** 2
    dr = r[1] - r[0]

    n_pts = len(r)
    if n_pts % 2 == 0:
        num = _simpson_composite(integrand_r2, r)
        den = _simpson_composite(integrand_0, r)
    else:
        num = _simpson_composite(integrand_r2[:-1], r[:-1])
        num += 0.5 * dr * (integrand_r2[-2] + integrand_r2[-1])
        den = _simpson_composite(integrand_0[:-1], r[:-1])
        den += 0.5 * dr * (integrand_0[-2] + integrand_0[-1])

    rms_radius = np.sqrt(num / den) if den > 1e-30 else 0.0

    return {
        'l2_norm': l2,
        'rms_norm': rms,
        'rms_radius': rms_radius,
        'is_normalized': abs(l2 - 1.0) < tol,
        'n_points': n_pts,
    }


# ---------- 辅助函数 ----------

def _simpson_composite(f: np.ndarray, x: np.ndarray) -> float:
    """
    复合 Simpson 求积:
        integral ≈ (h/3) [f_0 + 4*f_1 + 2*f_2 + 4*f_3 + ... + f_n]
    要求 len(f) 为奇数 (偶数个区间)
    """
    n = len(x) - 1
    h = (x[-1] - x[0]) / n
    s = f[0] + f[-1]
    for i in range(1, n):
        s += (4 if i % 2 == 1 else 2) * f[i]
    return s * h / 3.0


def inner_product_radial(
    u: np.ndarray, v: np.ndarray, r: np.ndarray
) -> complex:
    """
    径向波函数内积:
        <u|v> = integral_0^R u*(r) v(r) dr

    用于检验不同分波的正交性
    """
    integrand = np.conj(u) * v
    n = len(r) - 1
    if n % 2 == 0:
        return complex(_simpson_composite(integrand.real, r) +
                       1j * _simpson_composite(integrand.imag, r))
    else:
        dr = r[1] - r[0]
        re = _simpson_composite(integrand[:-1].real, r[:-1])
        re += 0.5 * dr * (integrand[-2].real + integrand[-1].real)
        im = _simpson_composite(integrand[:-1].imag, r[:-1])
        im += 0.5 * dr * (integrand[-2].imag + integrand[-1].imag)
        return complex(re + 1j * im)
