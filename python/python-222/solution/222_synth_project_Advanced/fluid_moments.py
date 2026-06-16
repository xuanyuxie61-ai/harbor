# -*- coding: utf-8 -*-
"""
fluid_moments.py
================

相对论流体力学矩方程与 Taylor-Green 涡旋类比。

融合种子项目:
    - 787_navier_stokes_2d_exact: 二维 Navier-Stokes 精确解 (Taylor-Green vortex)

物理背景:
    高能重离子碰撞中, QGP (夸克-胶子等离子体) 的演化可用相对论
    流体力学描述。Parton cascade 的宏观极限给出流体矩方程:

        d T^{mu nu} / d x^nu = 0

    其中能量-动量张量:
        T^{mu nu} = (e + p) u^mu u^nu - p g^{mu nu} + pi^{mu nu}

    e: 能量密度, p: 压强, u^mu: 四速度, pi^{mu nu}: 粘性应力张量

    对 parton 分布 f(x, p), 流体矩为:
        N^mu = int d^3p / E p^mu f
        T^{mu nu} = int d^3p / E p^mu p^nu f

    Taylor-Green 涡旋作为解析基准:
        u = -cos(x) sin(y) exp(-2 nu t)
        v =  sin(x) cos(y) exp(-2 nu t)
        p = -0.25 (cos(2x) + cos(2y)) exp(-4 nu t)

    该解用于验证 parton cascade 的流体极限是否正确。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable
import constants as C


# ======================================================================
# Taylor-Green 涡旋精确解 (来自 787_navier_stokes_2d_exact/uvp_taylor.m)
# ======================================================================
def taylor_green_velocity(nu: float, rho: float, x: float, y: float, t: float) -> Tuple[float, float]:
    """
    Taylor-Green 涡旋速度场:
        u(x,y,t) = -cos(x) sin(y) exp(-2 nu t)
        v(x,y,t) =  sin(x) cos(y) exp(-2 nu t)

    这是 2D 不可压 Navier-Stokes 的精确解。
    """
    decay = math.exp(-2.0 * nu * t)
    u = -math.cos(x) * math.sin(y) * decay
    v = math.sin(x) * math.cos(y) * decay
    return u, v


def taylor_green_pressure(nu: float, rho: float, x: float, y: float, t: float) -> float:
    """
    Taylor-Green 压强场:
        p(x,y,t) = -rho/4 * (cos(2x) + cos(2y)) * exp(-4 nu t)
    """
    decay = math.exp(-4.0 * nu * t)
    p = -rho / 4.0 * (math.cos(2.0 * x) + math.cos(2.0 * y)) * decay
    return p


def taylor_green_vorticity(nu: float, x: float, y: float, t: float) -> float:
    """
    涡量 omega = dv/dx - du/dy:
        omega = -2 sin(x) sin(y) exp(-2 nu t)
    """
    return -2.0 * math.sin(x) * math.sin(y) * math.exp(-2.0 * nu * t)


def taylor_green_residual(nu: float, rho: float, x: float, y: float, t: float,
                            h: float = 0.01) -> Tuple[float, float]:
    """
    Navier-Stokes 残差 (验证精确解):
        R_u = du/dt + u du/dx + v du/dy + (1/rho) dp/dx - nu Laplacian(u)
        R_v = dv/dt + u dv/dx + v dv/dy + (1/rho) dp/dy - nu Laplacian(v)

    对精确解应为 0。
    """
    # 时间导数
    decay = math.exp(-2.0 * nu * t)
    dudt = 2.0 * nu * math.cos(x) * math.sin(y) * decay
    dvdt = -2.0 * nu * math.sin(x) * math.cos(y) * decay

    # 空间导数
    u, v = taylor_green_velocity(nu, rho, x, y, t)
    p = taylor_green_pressure(nu, rho, x, y, t)

    # 中心差分 (分别计算 u 和 v 的各方向导数)
    u_xp, _ = taylor_green_velocity(nu, rho, x+h, y, t)  # u(x+h, y)
    u_xm, _ = taylor_green_velocity(nu, rho, x-h, y, t)  # u(x-h, y)
    u_yp, _ = taylor_green_velocity(nu, rho, x, y+h, t)  # u(x, y+h)
    u_ym, _ = taylor_green_velocity(nu, rho, x, y-h, t)  # u(x, y-h)

    dudx = (u_xp - u_xm) / (2*h)
    dudy = (u_yp - u_ym) / (2*h)

    _, v_xp = taylor_green_velocity(nu, rho, x+h, y, t)  # v(x+h, y)
    _, v_xm = taylor_green_velocity(nu, rho, x-h, y, t)  # v(x-h, y)
    _, v_yp = taylor_green_velocity(nu, rho, x, y+h, t)  # v(x, y+h)
    _, v_ym = taylor_green_velocity(nu, rho, x, y-h, t)  # v(x, y-h)

    dvdx = (v_xp - v_xm) / (2*h)
    dvdy = (v_yp - v_ym) / (2*h)

    p_px = taylor_green_pressure(nu, rho, x+h, y, t)
    p_mx = taylor_green_pressure(nu, rho, x-h, y, t)
    p_py = taylor_green_pressure(nu, rho, x, y+h, t)
    p_my = taylor_green_pressure(nu, rho, x, y-h, t)

    dpdx = (p_px - p_mx) / (2*h)
    dpdy = (p_py - p_my) / (2*h)

    # Laplacian (5 点 stencil)
    u_xx_p, _ = taylor_green_velocity(nu, rho, x+h, y, t)
    u_xx_m, _ = taylor_green_velocity(nu, rho, x-h, y, t)
    u_yy_p, _ = taylor_green_velocity(nu, rho, x, y+h, t)
    u_yy_m, _ = taylor_green_velocity(nu, rho, x, y-h, t)
    u_c, _ = taylor_green_velocity(nu, rho, x, y, t)
    lap_u = (u_xx_p + u_xx_m + u_yy_p + u_yy_m - 4.0 * u_c) / (h * h)

    _, v_xx_p = taylor_green_velocity(nu, rho, x+h, y, t)
    _, v_xx_m = taylor_green_velocity(nu, rho, x-h, y, t)
    _, v_yy_p = taylor_green_velocity(nu, rho, x, y+h, t)
    _, v_yy_m = taylor_green_velocity(nu, rho, x, y-h, t)
    _, v_c = taylor_green_velocity(nu, rho, x, y, t)
    lap_v = (v_xx_p + v_xx_m + v_yy_p + v_yy_m - 4.0 * v_c) / (h * h)

    # 残差
    R_u = dudt + u * dudx + v * dudy + dpdx / rho - nu * lap_u
    R_v = dvdt + u * dvdx + v * dvdy + dpdy / rho - nu * lap_v

    return R_u, R_v


# ======================================================================
# Parton 分布的流体矩
# ======================================================================
def energy_density_moment(fg: List[float], fq: List[float],
                           x_grid: List[float]) -> float:
    """
    能量密度 (parton 分布的第 1 矩):
        e = int_0^1 dx x [f_g(x) + f_q(x)] * Q
    其中 Q 为标度因子。
    """
    n = 32
    from splitting_kernels import gauss_legendre
    from parton_cascade import _interp
    nodes, weights = gauss_legendre(n)
    e = 0.0
    for i in range(n):
        xi = nodes[i]
        fg_i = _interp(x_grid, fg, xi)
        fq_i = _interp(x_grid, fq, xi)
        e += weights[i] * xi * (fg_i + fq_i)
    return e


def pressure_moment(fg: List[float], fq: List[float],
                     x_grid: List[float]) -> float:
    """
    压强 (理想气体近似):
        p = e / 3 (相对论气体)
    """
    e = energy_density_moment(fg, fq, x_grid)
    return e / 3.0


def entropy_density_estimate(e: float, p: float) -> float:
    """
    熵密度 (相对论理想气体):
        s = (e + p) / T
    其中 T ~ (e / a)^(1/4), a = pi^2 g_eff / 30
    """
    g_eff = 2 * (C.N_C**2 - 1) + 7.0/8.0 * 2 * C.N_C * 5  # gluon + 5-flavor quark
    a = C.PI**2 * g_eff / 30.0
    if e <= 0:
        return 0.0
    T = (e / a) ** 0.25
    if T < 1e-15:
        return 0.0
    return (e + p) / T


# ======================================================================
# 相对论 EKT (Eq. of state)
# ======================================================================
def equation_of_state_bag(e: float, B_bag: float = 0.2) -> float:
    """
    Bag 模型状态方程:
        p = (e - 4 B) / 3
    其中 B ~ 0.2 GeV^4 为 bag 常数。
    """
    return max(0.0, (e - 4.0 * B_bag) / 3.0)


# ======================================================================
# Knudsen 数估计 (流体近似有效性)
# ======================================================================
def knudsen_number(mean_free_path: float, system_size: float) -> float:
    """
    Knudsen 数:
        Kn = lambda_mfp / L
    Kn << 1: 流体近似有效
    Kn >> 1: 自由流 (parton cascade 必要)
    """
    if system_size < 1e-15:
        return float('inf')
    return mean_free_path / system_size


def parton_mean_free_path(alpha_s: float, T: float, n_f: int = 5) -> float:
    """
    Parton 平均自由程:
        lambda_mfp ~ 1 / (n * sigma)
    其中 n ~ T^3 (数密度), sigma ~ alpha_s^2 / T^2 (散射截面)
        lambda_mfp ~ 1 / (alpha_s^2 * T)
    """
    if alpha_s < 1e-10 or T < 1e-10:
        return float('inf')
    return 1.0 / (alpha_s * alpha_s * T + C.EPSILON_REG)


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Taylor-Green Vortex Test ===")
    nu = 0.01
    rho = 1.0
    t = 0.5
    x, y = 1.0, 1.0

    u, v = taylor_green_velocity(nu, rho, x, y, t)
    p = taylor_green_pressure(nu, rho, x, y, t)
    w = taylor_green_vorticity(nu, x, y, t)
    print(f"  u = {u:.6f}, v = {v:.6f}")
    print(f"  p = {p:.6f}")
    print(f"  omega = {w:.6f}")

    Ru, Rv = taylor_green_residual(nu, rho, x, y, t, h=0.001)
    print(f"  Residual: R_u = {Ru:.2e}, R_v = {Rv:.2e}")

    print("\n=== Fluid Moments ===")
    from parton_cascade import make_initial_state
    s0 = make_initial_state(12, 10.0)
    e = energy_density_moment(s0.fg, s0.fq, s0.x)
    p = pressure_moment(s0.fg, s0.fq, s0.x)
    s = entropy_density_estimate(e, p)
    print(f"  e = {e:.6f}, p = {p:.6f}, s = {s:.6f}")

    print("\n=== Knudsen Number ===")
    als = C.safe_alpha_s(10.0)
    T = 0.3  # GeV ~ RHIC temperature
    lam = parton_mean_free_path(als, T)
    L = 10.0  # fm
    Kn = knudsen_number(lam, L)
    print(f"  alpha_s = {als:.4f}, T = {T} GeV")
    print(f"  lambda_mfp = {lam:.4f} fm, L = {L} fm")
    print(f"  Kn = {Kn:.4e}")
