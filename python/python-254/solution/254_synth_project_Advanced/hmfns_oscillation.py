# -*- coding: utf-8 -*-
"""
hmfns_oscillation.py
====================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

并合产物——超质量中子星 (HMNS) 的微分旋转与 f-模式振荡.
用 Jacobi 椭圆函数给出非线性振荡的解析近似.

f-模式频率
----------
对于非旋转中子星, 基频 f-mode (l=2)::

    f_f-mode ~ 1.6 kHz * (M / 1.4 M_sun)^{1/2} * (R / 10 km)^{-3/2}
                  * sqrt(1 - 2 G M / (R c^2))

准周期振荡 (QPO) 频率与状态方程的相关性 (Bauswein & Stergiophou 2016)::

    f_peak = a_2 * (M_th / M_TOV)^2 + b_2 * (M_th / M_TOV) + c_2

其中 M_th 为热 HMNS 质量, M_TOV 为冷 TOV 极限质量.

Jacobi 椭圆函数解 (非线性单摆)
------------------------------
非线性摆的精确解 (Ochs 2011; Williams 2013)::

    theta(t) = 2 * arcsin( k * sn(omega t + phi_0, chi) )
    thetadot(t) = 2 * k * omega * cn(omega t + phi_0, chi) * dn(...)

其中::

    k = sin(theta_0 / 2)  椭圆模
    omega = sqrt(g / L)
    chi = 1 / k

HMNS 的非线性径向振荡可类比, 以 k 作为非线性度参数.

映射种子项目
-----------
- 860 (pendulum_nonlinear_exact) → Jacobi sn/cn/dn 函数实现,
  精确解析解 (Ochs 2011). 在此被重新用于 HMNS f-mode
  的非线性饱和振幅
"""

from __future__ import annotations
import math
from typing import Tuple, List

from physical_constants import C_LIGHT, G_GRAV, M_SUN, R_NS_TYPICAL


# ---------------------------------------------------------------------------
# Jacobi 椭圆函数 (通过算术几何平均 - AGM)
# ---------------------------------------------------------------------------
def _agm(a: float, b: float, tol: float = 1.0e-14, max_iter: int = 100
         ) -> Tuple[float, List[float]]:
    """算术几何平均 (AGM) 迭代, 同时记录每步的 a_n, b_n, c_n.

    c_n = (a_n - b_n) / 2
    """
    history = []
    for _ in range(max_iter):
        c = (a - b) / 2.0
        history.append((a, b, c))
        if abs(c) < tol * max(abs(a), 1.0):
            break
        a_next = (a + b) / 2.0
        b_next = math.sqrt(a * b)
        a, b = a_next, b_next
    return a, history


def sncndn(u: float, m: float) -> Tuple[float, float, float]:
    """计算 Jacobi 椭圆函数 sn(u|m), cn(u|m), dn(u|m).

    使用 AGM 降阶 (Abramowitz & Stegun 16.4).

    Parameters
    ----------
    u : float  自变量
    m : float  参数 m = k^2  (0 <= m <= 1)
    """
    if m < 0 or m > 1.0 + 1.0e-10:
        raise ValueError(f"m must be in [0, 1]; got {m}")
    m = min(max(m, 0.0), 1.0)
    if m < 1.0e-12:
        return math.sin(u), math.cos(u), 1.0
    if m > 1.0 - 1.0e-12:
        return math.tanh(u), 1.0 / math.cosh(u), 1.0 / math.cosh(u)

    a0 = 1.0
    b0 = math.sqrt(1.0 - m)
    _, history = _agm(a0, b0)
    N = len(history)
    # 向前缩放
    phi = (2.0 ** N) * history[-1][0] * u
    for n in range(N - 1, -1, -1):
        _, _, c_n = history[n]
        a_n = history[n][0]
        # theta_{n-1} = (theta_n + arcsin(c_n/a_n * sin(theta_n))) / 2
        if abs(a_n) < 1.0e-30:
            break
        ratio = c_n / a_n
        ratio = max(min(ratio, 1.0), -1.0)
        try:
            delta = math.asin(ratio * math.sin(phi))
        except ValueError:
            delta = 0.0
        phi = (phi + delta) / 2.0
    sn = math.sin(phi)
    cn = math.cos(phi)
    dn = math.sqrt(max(0.0, 1.0 - m * sn * sn))
    return sn, cn, dn


# ---------------------------------------------------------------------------
# HMNS f-mode 频率估计
# ---------------------------------------------------------------------------
def fmode_frequency(M_g: float, R_cm: float) -> float:
    """基频 f-mode (l=2) 估计 [kHz].

    f ~ 1.6 kHz * sqrt(M / 1.4 M_sun) * (10 km / R)^{3/2}
        * sqrt(1 - 2 G M / (R c^2))
    """
    eta = G_GRAV * M_g / (R_cm * C_LIGHT ** 2)
    if eta >= 0.5:
        return 0.0  # 黑洞, 无 f-mode
    f0_khz = 1.6  # kHz (参考值, 1.4 M_sun, 10 km)
    m_ratio = math.sqrt(M_g / (1.4 * M_SUN))
    r_ratio = (R_NS_TYPICAL / R_cm) ** 1.5
    correction = math.sqrt(max(0.0, 1.0 - 2.0 * eta))
    return f0_khz * m_ratio * r_ratio * correction  # 返回 kHz


def peak_postmerger_frequency(M_th: float, M_TOV: float) -> float:
    """并合后引力波峰值频率 [kHz] (Bauswein-Stergiophou 拟合).

    f_peak = a_2 x^2 + b_2 x + c_2,   x = M_th / M_TOV

    系数 (典型): a_2 = -1.7, b_2 = 7.0, c_2 = -2.0  (kHz)
    """
    a2 = -1.7
    b2 = 7.0
    c2 = -2.0
    x = M_th / M_TOV
    return a2 * x * x + b2 * x + c2


# ---------------------------------------------------------------------------
# HMNS 非线性振荡 (Jacobi 椭圆函数)
# ---------------------------------------------------------------------------
def hmns_radial_oscillation(
    t_s: List[float],
    amplitude_rad: float,
    omega_hz: float,
    k_modulus: float,
) -> Tuple[List[float], List[float]]:
    """HMNS 径向振荡的非线性解析解.

    theta(t) = 2 * arcsin( k * sn(omega t, k^2) )
    thetadot(t) = 2 * k * omega * cn * dn

    Parameters
    ----------
    t_s          : List[float]  时间序列 [s]
    amplitude_rad: float        初始振幅 [rad]
    omega_hz     : float        角频率 [Hz]
    k_modulus    : float        椭圆模 (0 < k < 1)

    Returns
    -------
    (theta, thetadot) : 位移与速度
    """
    if not 0 < k_modulus < 1:
        raise ValueError("k must be in (0, 1)")
    k0 = math.sin(amplitude_rad / 2.0)
    omega = 2.0 * math.pi * omega_hz
    m = k_modulus * k_modulus
    theta = []
    thetadot = []
    for t in t_s:
        sn, cn, dn = sncndn(omega * t, m)
        # 振幅调制
        s_val = k0 * sn
        s_val = max(min(s_val, 1.0), -1.0)
        th = 2.0 * math.asin(s_val)
        # 速度: d/dt arcsin(k sn) = k omega cn dn / sqrt(1 - k^2 sn^2)
        denom = math.sqrt(max(1.0e-30, 1.0 - k0 * k0 * sn * sn))
        thdot = 2.0 * k0 * omega * cn * dn / denom
        theta.append(th)
        thetadot.append(thdot)
    return theta, thetadot


# ---------------------------------------------------------------------------
# 引力波应变 (简化 quadrupole)
# ---------------------------------------------------------------------------
def gw_strain_from_oscillation(
    theta: List[float],
    M_g: float,
    R_cm: float,
    distance_cm: float,
) -> List[float]:
    """从 HMNS 振荡计算引力波无量纲应变 h(t).

    h ~ (4 G / (c^4 D)) * (M R^2 Omega^2) * theta(t)

    简化估计, 忽略角度因子.
    """
    coeff = 4.0 * G_GRAV / (C_LIGHT ** 4 * distance_cm) * M_g * R_cm * R_cm
    return [coeff * th for th in theta]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证 Jacobi 椭圆函数的特殊值."""
    # m=0: sn = sin, cn = cos, dn = 1
    sn, cn, dn = sncndn(1.0, 0.0)
    assert abs(sn - math.sin(1.0)) < 1e-10
    assert abs(cn - math.cos(1.0)) < 1e-10
    assert abs(dn - 1.0) < 1e-10
    # sn^2 + cn^2 = 1
    for m in [0.1, 0.3, 0.5, 0.9]:
        sn, cn, dn = sncndn(2.5, m)
        if abs(sn * sn + cn * cn - 1.0) > 1e-10:
            raise AssertionError(f"sn^2+cn^2 != 1 for m={m}")
    # f-mode 频率范围
    f = fmode_frequency(2.6 * M_SUN, 1.3e6)
    if not 1.0 < f < 5.0:
        raise AssertionError(f"f-mode frequency {f:.2f} kHz out of range")
    return True


if __name__ == "__main__":
    _self_check()
    print("hmfns_oscillation self-check passed.")
    f = fmode_frequency(2.7 * M_SUN, 1.2e6)
    print(f"  f-mode frequency  : {f:.3f} kHz")
    t = [i * 1.0e-4 for i in range(100)]
    theta, _ = hmns_radial_oscillation(t, amplitude_rad=0.1,
                                       omega_hz=f * 1000.0, k_modulus=0.3)
    print(f"  max |theta|       : {max(abs(th) for th in theta):.4f} rad")
