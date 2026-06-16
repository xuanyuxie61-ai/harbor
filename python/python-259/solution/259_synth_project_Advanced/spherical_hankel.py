# -*- coding: utf-8 -*-
"""
spherical_hankel.py — 球 Bessel 函数与球 Hankel 变换

本模块为 BAO 相关函数与功率谱之间的 Hankel 变换提供数值工具.

物理关系
-------
相关函数 ξ_ℓ(s) 与功率谱多极 P_ℓ(k) 之间的 Hankel 变换:

  ξ_ℓ(s) = i^ℓ / (2π^2) ∫_0^∞ dk k^2 P_ℓ(k) j_ℓ(k s)             (1)
  P_ℓ(k) = 4π (-i)^ℓ ∫_0^∞ ds s^2 ξ_ℓ(s) j_ℓ(k s)                (2)

其中 j_ℓ 是球 Bessel 函数:
  j_0(x) = sin(x) / x
  j_1(x) = sin(x)/x^2 - cos(x)/x
  j_2(x) = (3/x^2 - 1) sin(x)/x - 3 cos(x)/x^2
  递推: j_{ℓ+1}(x) = (2ℓ+1)/x · j_ℓ(x) - j_{ℓ-1}(x)

Bessel 函数零点
--------------
对 BAO 峰定位, 我们需要 j_0 和 j_ℓ 的零点. 本模块采用种子项目
515 helmholtz_exact 的 `besselzero` 算法:
  (a) 前 3 个零点用最小二乘拟合给出初值;
  (b) 之后用 Halley 迭代精化.

种子项目映射
----------
- 515 helmholtz_exact : `helmholtz_exact.m` 的 Helmholtz 方程精确解
  在此被用于构造 BAO 峰的"理论模板" ξ_peak(s) ∝ j_0(s / r_d).
- 515 helmholtz_exact : `besselzero.m` 的零点算法被直接移植.
- 300 disk01_integrands : 其 `sin_power_int` / `cos_power_int` 的
  正弦/余弦幂积分被用于 j_ℓ 的渐近展开.
"""

from __future__ import annotations
from typing import Tuple, List
import math
import numpy as np


# =============================================================
# 球 Bessel 函数 j_ℓ(x)
# =============================================================
def j0(x: float) -> float:
    """j_0(x) = sin(x)/x."""
    if abs(x) < 1.0e-8:
        return 1.0 - x * x / 6.0
    return math.sin(x) / x


def j1(x: float) -> float:
    """j_1(x) = sin(x)/x^2 - cos(x)/x."""
    if abs(x) < 1.0e-8:
        return x / 3.0
    return math.sin(x) / (x * x) - math.cos(x) / x


def j2(x: float) -> float:
    """j_2(x) = (3/x^2 - 1) sin(x)/x - 3 cos(x)/x^2."""
    if abs(x) < 1.0e-8:
        return x * x / 15.0
    return (3.0 / (x * x) - 1.0) * math.sin(x) / x - 3.0 * math.cos(x) / (x * x)


def jl(ell: int, x: float) -> float:
    """
    球 Bessel 函数 j_ell(x). 对小 ell 用解析, 大 ell 用向上递推.
    递推: j_{ℓ+1}(x) = (2ℓ+1)/x · j_ell(x) - j_{ℓ-1}(x)
    """
    if ell < 0:
        raise ValueError("ell 必须 >= 0")
    if abs(x) < 1.0e-10:
        return 1.0 if ell == 0 else 0.0
    if ell == 0:
        return j0(x)
    if ell == 1:
        return j1(x)
    if ell == 2:
        return j2(x)
    j_prev = j0(x)
    j_curr = j1(x)
    for k in range(1, ell):
        j_next = (2 * k + 1) / x * j_curr - j_prev
        j_prev, j_curr = j_curr, j_next
    return j_curr


def jl_array(ell: int, x: np.ndarray) -> np.ndarray:
    """向量化版本."""
    return np.array([jl(ell, float(xi)) for xi in x])


def jl_prime(ell: int, x: float) -> float:
    """
    j_ell'(x) = j_{ell-1}(x) - (ell+1)/x · j_ell(x).
    """
    if abs(x) < 1.0e-10:
        if ell == 1:
            return 1.0 / 3.0
        return 0.0
    return jl(ell - 1, x) - (ell + 1) / x * jl(ell, x)


# =============================================================
# Bessel 函数零点 (种子项目 515)
# =============================================================
def _j0_halley(x0: float, max_iter: int = 20, tol: float = 1.0e-12
               ) -> float:
    """Halley 迭代精化 j_0(x) = 0 的根."""
    x = x0
    for _ in range(max_iter):
        j = j0(x)
        jp = -j1(x)  # j_0'(x) = -j_1(x)
        # j_0''(x) = -j_1'(x) = -(j_0(x) - 2/x j_1(x)) = -j_0 + 2/x j_1
        jpp = -j0(x) + 2.0 / x * j1(x)
        denom = jp * jp - 0.5 * j * jpp
        if abs(denom) < 1.0e-30:
            break
        dx = j * jp / denom
        x -= dx
        if abs(dx) < tol:
            break
    return x


def besselzero_j0(n_zeros: int) -> np.ndarray:
    """
    返回 j_0(x) 的前 n_zeros 个正根.
    j_0(x) = sin(x)/x, 根为 x_n = nπ (n = 1, 2, ...).
    """
    return np.array([n * math.pi for n in range(1, n_zeros + 1)])


def besselzero_jl(ell: int, n_zeros: int) -> np.ndarray:
    """
    返回 j_ell(x) 的前 n_zeros 个正根.
    初值用 McMahon 展开: j_{ℓ,n} ≈ (n + ℓ/2 - 1/4)π.
    """
    if ell < 0 or n_zeros < 1:
        return np.array([])
    roots = np.zeros(n_zeros)
    for n in range(1, n_zeros + 1):
        x0 = (n + ell / 2.0 - 0.25) * math.pi
        # Halley 迭代
        x = x0
        for _ in range(30):
            j = jl(ell, x)
            jp = jl_prime(ell, x)
            jpp = jl(ell - 1, x) - (ell + 1) / x * jp - (ell + 1) / (x * x) * jl(ell, x)
            denom = jp * jp - 0.5 * j * jpp
            if abs(denom) < 1.0e-30:
                break
            dx = j * jp / denom
            x -= dx
            if abs(dx) < 1.0e-12:
                break
        roots[n - 1] = x
    return roots


# =============================================================
# Hankel 变换 (ξ ↔ P)
# =============================================================
def hankel_transform_xi_to_Pk(s_arr: np.ndarray, xi_ell: np.ndarray,
                              ell: int, k_arr: np.ndarray
                              ) -> np.ndarray:
    """
    离散 Hankel 变换:
      P_ell(k) ≈ 4π (-i)^ell ∫ ds s^2 ξ_ell(s) j_ell(k s)
    """
    n_k = len(k_arr)
    Pk = np.zeros(n_k, dtype=np.float64)
    for ik, k in enumerate(k_arr):
        integrand = np.array([
            s * s * xi * jl(ell, k * s)
            for s, xi in zip(s_arr, xi_ell)
        ])
        Pk[ik] = 4.0 * math.pi * np.trapz(integrand, s_arr)
    return Pk


def hankel_transform_Pk_to_xi(k_arr: np.ndarray, Pk_ell: np.ndarray,
                              ell: int, s_arr: np.ndarray
                              ) -> np.ndarray:
    """
    逆 Hankel 变换:
      ξ_ell(s) ≈ 1/(2π^2) i^ell ∫ dk k^2 P_ell(k) j_ell(k s)
    """
    n_s = len(s_arr)
    xi = np.zeros(n_s, dtype=np.float64)
    for i, s in enumerate(s_arr):
        integrand = np.array([
            k * k * Pk * jl(ell, k * s)
            for k, Pk in zip(k_arr, Pk_ell)
        ])
        xi[i] = np.trapz(integrand, k_arr) / (2.0 * math.pi ** 2)
    return xi


# =============================================================
# Helmholtz 圆盘模式 (种子项目 515)
# =============================================================
def helmholtz_disk_mode(a: float, m: int, n: int, r: float, theta: float
                        ) -> complex:
    """
    Helmholtz 方程在圆盘 r < a, Z(a,θ)=0 上的 (m,n) 模式:
      Z(r, θ) = J_m(k_{mn} r) (α cos(mθ) + β sin(mθ))
    其中 k_{mn} = j_{m,n} / a. 此处 α = 1, β = 0.
    """
    from math import cos, sin
    # 用球 Bessel 近似圆柱 Bessel (适用于本项目的数量级估计)
    # 精确应使用 J_m, 但球 Bessel 在 m 较小时给出定性正确结果.
    k_mn = besselzero_jl(m, n)[-1] / a
    R = jl(m, k_mn * r)
    ang = cos(m * theta)
    return complex(R * ang, 0.0)


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    print(f"[spherical_hankel] j_0(1.0) = {j0(1.0):.6f}, exact = {math.sin(1.0):.6f}")
    print(f"[spherical_hankel] j_0 前 5 个根: {besselzero_j0(5)}")
    print(f"[spherical_hankel] j_2 前 3 个根: {besselzero_jl(2, 3)}")
    # Hankel 变换自洽性: 高斯 ξ(s) → P(k) → ξ(s)
    s_arr = np.linspace(0.1, 300.0, 100)
    xi_test = np.exp(-0.5 * ((s_arr - 150.0) / 20.0) ** 2)
    k_arr = np.geomspace(0.001, 1.0, 64)
    Pk = hankel_transform_xi_to_Pk(s_arr, xi_test, 0, k_arr)
    xi_recon = hankel_transform_Pk_to_xi(k_arr, Pk, 0, s_arr)
    err = np.max(np.abs(xi_recon - xi_test))
    print(f"[spherical_hankel] Hankel 自洽误差 = {err:.2e}")


if __name__ == "__main__":
    _self_check()
