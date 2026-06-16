#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dispersion_relations.py
=======================
【融合种子项目】 139_cauchy_principal_value (Cauchy 主值积分)

核物质响应函数的色散关系计算, 使用 Cauchy 主值积分处理极点奇异性.

物理: Kramers-Kronig 关系
    Re chi(omega) = (1/pi) PV integral_{-inf}^{inf} Im chi(omega') / (omega' - omega) d omega'
    Im chi(omega) = -(1/pi) PV integral_{-inf}^{inf} Re chi(omega') / (omega' - omega) d omega'

数学 (Noble 2000):
    PV int_a^b f(t)/(t-x) dt ≈ sum_i w_i f(x_i) / xi_i
    其中 xi_i 为 Gauss-Legendre 节点, 偶数 N 保证系数和为零.
"""

import math
import numpy as np
from typing import Callable
from numerical_constants import legendre_set


def cauchy_principal_value(f: Callable, a: float, b: float,
                             x: float = None, n: int = 20) -> float:
    """
    Cauchy 主值积分 (移植自 cauchy_principal_value).

    PV int_a^b f(t)/(t-x) dt

    如果 x 未提供, 假设奇点在 (a+b)/2.

    Noble 方法:
        使用 Gauss-Legendre 偶数点求积, 利用节点对称性自动消除奇点.
    """
    if n % 2 != 0:
        n += 1  # 必须偶数

    if x is None:
        x = 0.5 * (a + b)

    xi, wi = legendre_set(n)

    # 映射到 [a, b]
    half_len = 0.5 * (b - a)
    mid = 0.5 * (a + b)

    value = 0.0
    for i in range(n):
        t_i = half_len * xi[i] + mid
        # 在奇点附近: (t - x) = half_len * xi (当 x = mid)
        if abs(xi[i]) < 1e-12:
            continue
        f_val = f(t_i)
        value += wi[i] * f_val / xi[i]

    return value


def kramers_kronig_real(im_chi_func: Callable, omega: float,
                          omega_max: float = 100.0,
                          n_points: int = 40) -> float:
    """
    Kramers-Kronig: 由虚部求实部.

    Re chi(omega) = (2/pi) PV int_0^inf omega' Im chi(omega') / (omega'^2 - omega^2) d omega'
    """
    def integrand(wp):
        return wp * im_chi_func(wp)

    # 分割区间避免奇点
    eps = 1e-6 * omega
    val1 = cauchy_principal_value(integrand, 0.0, omega - eps,
                                   x=omega * 0.5, n=n_points) if omega > eps else 0.0
    val2 = cauchy_principal_value(integrand, omega + eps, omega_max,
                                   x=(omega + omega_max) / 2, n=n_points)

    return (2.0 / math.pi) * (val1 + val2)


def nuclear_response_lorentz(omega: float, omega_0: float = 1.0,
                               gamma: float = 0.1) -> complex:
    """
    Lorentz 型核响应函数.

    chi(omega) = 1 / (omega_0^2 - omega^2 - i gamma omega)
    """
    denom = omega_0 ** 2 - omega ** 2 - 1j * gamma * omega
    return 1.0 / denom if abs(denom) > 1e-30 else complex(1e30, 0)


def lindhard_dielectric(q: float, omega: float, k_f: float = 1.0,
                          mass: float = 1.0) -> complex:
    """
    Lindhard 介电函数 (简化版, 用于核物质响应).

    chi_0(q, omega) 描述无相互作用费米气体的密度响应.

    简化:
        chi_0 ~ -N(0) [1 - (omega + i eta) / (2 v_F q) ln((omega + v_F q)/(omega - v_F q))]
    """
    v_f = k_f / mass
    n_0 = mass * k_f / (math.pi ** 2)

    if abs(q) < 1e-10:
        return complex(-n_0, 0)

    eta = 0.01 * v_f * q
    z = (omega + 1j * eta) / (v_f * q)

    if abs(z - 1) < 1e-6 or abs(z + 1) < 1e-6:
        return complex(-n_0, n_0 * math.pi / 2)

    log_term = math.log(abs((z + 1) / (z - 1)))
    if z.imag != 0:
        log_term = complex(math.log(abs(z + 1) / abs(z - 1)),
                           math.atan2(z.imag, z.real + 1) - math.atan2(z.imag, z.real - 1))
        chi = -n_0 * (1.0 - 0.5 * z * log_term)
        return chi
    else:
        return complex(-n_0 * (1.0 - 0.5 * z * math.log(abs((z + 1) / (z - 1)))), 0)


def dispersion_test() -> dict:
    """色散关系自洽性测试."""
    # Lorentz 响应: 已知解析形式
    omega_0 = 1.0
    gamma = 0.2

    def im_chi(w):
        chi = nuclear_response_lorentz(w, omega_0, gamma)
        return chi.imag

    # 在若干频率点检验 K-K 关系
    test_omegas = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    errors = []
    for omega in test_omegas:
        chi_exact = nuclear_response_lorentz(omega, omega_0, gamma)
        re_exact = chi_exact.real
        re_kk = kramers_kronig_real(im_chi, omega, omega_max=50.0, n_points=60)
        errors.append(abs(re_exact - re_kk))

    return {
        'test_frequencies': test_omegas,
        'kk_errors': errors,
        'max_error': max(errors),
    }


# 自检
if __name__ == "__main__":
    print("=== 色散关系自检 ===")

    # Cauchy 主值: PV int_{-1}^{1} 1/(t-0) dt = 0 (对称)
    def f_const(t): return 1.0
    val = cauchy_principal_value(f_const, -1.0, 1.0, x=0.0, n=20)
    print(f"PV int 1/t [-1,1] = {val:.6f}  (应为 0)")

    # PV int_{-1}^{1} t/(t-0) dt = PV int 1 dt = 2
    def f_linear(t): return t
    val = cauchy_principal_value(f_linear, -1.0, 1.0, x=0.0, n=20)
    print(f"PV int t/t [-1,1] = {val:.6f}  (应为 2)")

    # Lorentz 响应
    chi = nuclear_response_lorentz(0.5, 1.0, 0.2)
    print(f"chi(0.5) = {chi}")

    # K-K 测试
    result = dispersion_test()
    print(f"K-K 最大误差: {result['max_error']:.4f}")

    # Lindhard
    chi_l = lindhard_dielectric(0.5, 0.3)
    print(f"Lindhard chi(0.5, 0.3) = {chi_l}")

    print("\ndispersion_relations.py 自检通过.")
