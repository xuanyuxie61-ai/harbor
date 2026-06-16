# -*- coding: utf-8 -*-
"""
hadronization_string.py
=======================

Lund 弦强子化模型与色场通量管 Poisson 方程求解。

融合种子项目:
    - 370_fd3d_poisson: 三维 Poisson 有限差分 (CG 求解)
    - 893_polynomial: 多项式运算 (碎裂函数参数化)
    - 807_nonlin_fixed_point: 非线性不动点迭代 (自洽碎裂方程)

物理背景:
    Lund 弦模型: 色弦的能量密度 (单位长度) 为 kappa ~ 1 GeV/fm。
    色场通量管内的电势满足 Poisson 方程:
        -Laplacian Phi = rho_color / epsilon_0
    其中 rho_color 为色荷密度, 与弦的几何构型相关。

    碎裂函数 (Lund symmetric):
        f(z) ~ (1/z) * (1 - z)^a * exp(-b m_T^2 / z)
    其中 z 为 lightcone 动量分数, m_T 为碎裂强子横向质量。

    自洽碎裂方程 (不动点形式):
        D_h^q(z) = int_z^1 (dy/y) K(y) D_h^q(z/y)
    通过不动点迭代求解。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable, Optional
import constants as C
from splitting_kernels import Poly1D, gauss_legendre
from fd_operators import solve_poisson_3d_cg


# ======================================================================
# Lund 碎裂函数 (来自 893_polynomial 的多项式参数化)
# ======================================================================
def lund_fragmentation_function(z: float, a: float = None, b: float = None,
                                 mT2: float = 0.2) -> float:
    """
    Lund symmetric 碎裂函数:
        f(z) = N * (1/z) * (1 - z)^a * exp(-b * mT^2 / z)

    Parameters
    ----------
    z : float
        Lightcone 动量分数 in (0, 1)
    a : float
        Lund 参数 a (default: A_LUND)
    b : float
        Lund 参数 b (default: B_LUND)
    mT2 : float
        横向质量平方 mT^2 = m^2 + pT^2
    """
    if a is None:
        a = C.A_LUND
    if b is None:
        b = C.B_LUND
    if z <= C.Z_CUT or z >= 1.0 - C.Z_CUT:
        return 0.0

    log_f = -math.log(z) + a * math.log(1.0 - z) - b * mT2 / z
    if log_f < -500:
        return 0.0
    return math.exp(log_f)


def lund_fragmentation_polynomial(z: float, coeffs: List[float]) -> float:
    """
    将碎裂函数以多项式基展开:
        D(z) = sum_k c_k * z^k * (1-z)^a * exp(-b/z)
    """
    if z <= C.Z_CUT or z >= 1.0 - C.Z_CUT:
        return 0.0
    poly_val = sum(c * z**k for k, c in enumerate(coeffs))
    return poly_val * (1.0 - z) ** C.A_LUND * math.exp(-C.B_LUND * 0.2 / z)


# ======================================================================
# 碎裂函数矩 (来自 893_polynomial 的 polynomial_value)
# ======================================================================
def fragmentation_moment(n: int, a: float = None, b: float = None,
                          mT2: float = 0.2, nquad: int = 32) -> float:
    """
    碎裂函数的第 n 阶矩:
        <z^n> = int_0^1 dz z^n f(z) / int_0^1 dz f(z)

    用于归一化与 sum rule 检查。
    """
    if a is None:
        a = C.A_LUND
    if b is None:
        b = C.B_LUND

    nodes, weights = gauss_legendre(nquad)
    num = 0.0
    den = 0.0
    for i in range(nquad):
        z = nodes[i]
        f = lund_fragmentation_function(z, a, b, mT2)
        num += weights[i] * (z ** n) * f
        den += weights[i] * f
    return num / max(den, 1e-30)


# ======================================================================
# 色场通量管 Poisson 方程 (来自 370_fd3d_poisson)
# ======================================================================
def flux_tube_charge_density(x: float, y: float, z: float,
                              tube_radius: float = 0.5,
                              tube_length: float = 3.0) -> float:
    """
    色弦通量管的色荷密度 (柱状高斯分布):
        rho(r) = rho_0 * exp(-(x^2 + y^2) / (2 R^2))
               * theta(z) * theta(L - z)

    其中 R 为通量管半径, L 为弦长。
    """
    r2 = x * x + y * y
    if z < 0.01 or z > tube_length - 0.01:
        return 0.0
    return math.exp(-r2 / (2.0 * tube_radius * tube_radius))


def flux_tube_boundary(x: float, y: float, z: float) -> float:
    """Dirichlet 边界条件: 通量管外 Phi = 0"""
    return 0.0


def solve_flux_tube_potential(n_grid: int = 5, tube_radius: float = 0.5,
                                tube_length: float = 3.0) -> Tuple:
    """
    求解通量管内色场电势:
        -Laplacian Phi = g^2 * rho_color
    其中 g 为 QCD 耦合, g^2 = 4 pi alpha_s。

    返回 (Phi_3d, X, Y, Z) 网格与电势。
    """
    alpha_s = C.safe_alpha_s(1.0)
    g2 = 4.0 * C.PI * alpha_s

    extent = tube_radius * 3.0
    f = lambda x, y, z: g2 * flux_tube_charge_density(x, y, z, tube_radius, tube_length)
    g = flux_tube_boundary

    U, X, Y, Z = solve_poisson_3d_cg(
        n_grid, n_grid, n_grid,
        -extent, extent, -extent, extent, 0.0, tube_length,
        f, g,
        max_iter=200, tol=1e-6
    )
    return U, X, Y, Z


def flux_tube_energy_density(Phi: List, X: List[float], Y: List[float], Z: List[float],
                              hx: float, hy: float, hz: float) -> float:
    """
    通量管能量密度积分:
        E = (1/2) int |grad Phi|^2 d^3x
    """
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    energy = 0.0

    for ix in range(1, nx - 1):
        for iy in range(1, ny - 1):
            for iz in range(1, nz - 1):
                dphidx = (Phi[ix+1][iy][iz] - Phi[ix-1][iy][iz]) / (2.0 * hx)
                dphidy = (Phi[ix][iy+1][iz] - Phi[ix][iy-1][iz]) / (2.0 * hy)
                dphidz = (Phi[ix][iy][iz+1] - Phi[ix][iy][iz-1]) / (2.0 * hz)
                grad2 = dphidx**2 + dphidy**2 + dphidz**2
                energy += 0.5 * grad2 * hx * hy * hz

    return energy


# ======================================================================
# 自洽碎裂方程 (来自 807_nonlin_fixed_point)
# ======================================================================
def fragmentation_fixed_point_iteration(D0: Callable[[float], float],
                                         kernel: Callable[[float], float],
                                         z_grid: List[float],
                                         max_iter: int = 50,
                                         tol: float = 1e-8) -> List[float]:
    """
    自洽碎裂方程的不动点迭代:
        D_{n+1}(z) = int_z^1 (dy/y) K(y) D_n(z/y)

    源自 807_nonlin_fixed_point 的迭代思想。
    K(y) 为演化核 (通常是 splitting kernel 的组合)。

    收敛条件: |D_{n+1} - D_n| / |D_n| < tol
    """
    D = [D0(z) for z in z_grid]
    n_z = len(z_grid)
    n_q = 16
    nodes, weights = gauss_legendre(n_q)

    for iteration in range(max_iter):
        D_new = [0.0] * n_z
        for i, zi in enumerate(z_grid):
            s = 0.0
            for k in range(n_q):
                # y in [zi, 1]
                y = zi + (1.0 - zi) * nodes[k]
                if y <= C.Z_CUT or y >= 1.0 - C.Z_CUT:
                    continue
                z_over_y = zi / y
                if z_over_y <= 0.0 or z_over_y >= 1.0:
                    continue
                D_val = _interp_1d(z_grid, D, z_over_y)
                s += weights[k] * kernel(y) * D_val / y
            D_new[i] = s * (1.0 - zi)

        # 归一化
        norm = sum(d * (z_grid[1] - z_grid[0]) for d in D_new) if n_z > 1 else 1.0
        if norm > 0:
            D_new = [d / norm for d in D_new]

        # 收敛检查
        diff = max(abs(D_new[i] - D[i]) for i in range(n_z))
        D = D_new

        if diff < tol:
            break

    return D


def _interp_1d(x_grid: List[float], f: List[float], x: float) -> float:
    if x <= x_grid[0] or x >= x_grid[-1]:
        return 0.0
    for i in range(len(x_grid) - 1):
        if x_grid[i] <= x <= x_grid[i+1]:
            t = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i])
            return f[i] + t * (f[i+1] - f[i])
    return 0.0


# ======================================================================
# 弦碎裂: 迭代产生强子
# ======================================================================
def string_fragmentation_iterative(string_mass: float,
                                     n_hadrons_max: int = 10) -> List[dict]:
    """
    迭代弦碎裂过程:
        1. 从弦端取走一个强子, 携带动量分数 z ~ f(z)
        2. 剩余弦质量 M_rem^2 = (1 - z) * M^2
        3. 重复直到 M_rem < 2 m_pi (阈值)

    碎裂函数: f(z) ~ (1-z)^a / z * exp(-b mT^2 / z)
    """
    hadrons = []
    M2 = string_mass ** 2
    m_pi = C.M_PION

    for i in range(n_hadrons_max):
        if M2 < (2.0 * m_pi) ** 2:
            break

        # 简化: 取 z 的期望值 (由矩给出)
        z_avg = fragmentation_moment(1)
        z_avg = min(max(z_avg, 0.1), 0.9)

        mT2 = m_pi ** 2  # 最低质量态
        m_h = m_pi

        hadrons.append({
            'flavor': 'pi',
            'z': z_avg,
            'mass': m_h,
            'mT2': mT2,
            'rank': i,
        })

        # 剩余质量
        M2 = (1.0 - z_avg) * M2

    return hadrons


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Lund Fragmentation Function ===")
    for z in [0.1, 0.3, 0.5, 0.7, 0.9]:
        f = lund_fragmentation_function(z)
        print(f"  f(z={z}) = {f:.6f}")

    print("\n=== Fragmentation Moments ===")
    for n in range(5):
        mn = fragmentation_moment(n)
        print(f"  <z^{n}> = {mn:.6f}")

    print("\n=== Flux Tube Poisson (small grid) ===")
    U, X, Y, Z = solve_flux_tube_potential(n_grid=3, tube_radius=0.5, tube_length=2.0)
    max_phi = max(U[ix][iy][iz]
                  for ix in range(len(X))
                  for iy in range(len(Y))
                  for iz in range(len(Z)))
    print(f"  Max potential: {max_phi:.6f}")

    print("\n=== Fixed Point Iteration ===")
    z_grid = [0.05 + 0.9 * i / 19 for i in range(20)]
    D0 = lambda z: lund_fragmentation_function(z)
    kernel = lambda y: 0.5 * (1 + (1-y)**2)  # simplified Pqq
    D_fp = fragmentation_fixed_point_iteration(D0, kernel, z_grid, max_iter=20)
    print(f"  D(z) converged, |D[0]| = {abs(D_fp[0]):.4e}")

    print("\n=== String Fragmentation ===")
    hadrons = string_fragmentation_iterative(10.0)
    print(f"  Produced {len(hadrons)} hadrons:")
    for h in hadrons:
        print(f"    {h['flavor']}: z={h['z']:.3f}, rank={h['rank']}")
