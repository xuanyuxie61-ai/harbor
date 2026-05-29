# -*- coding: utf-8 -*-
"""
bcs_gap_solver.py
-----------------
d-波 BCS 能隙方程自洽求解器。

对应种子项目：
  - 897_polynomial_root_bound：多项式根界用于 bracket 搜索
  - 111_box_behnken：Box-Behnken 实验设计用于参数空间探索

物理背景：
  高温超导体（如铜氧化物）的配对对称性为 d_{x^2-y^2} 波：
      Δ_k = Δ_0 * (cos k_x - cos k_y) / 2
  BCS 自洽能隙方程（在 T=0）：
      1 = (U / 2N) Σ_k (cos k_x - cos k_y)^2 / (2 E_k)
  其中 E_k = sqrt(ε_k^2 + Δ_k^2) 为准粒子能量，U 为有效吸引相互作用。
  在有限温度下：
      1 = (U / 2N) Σ_k (cos k_x - cos k_y)^2 tanh(β E_k / 2) / (2 E_k)

核心公式：
  - 准粒子能量：E_k = sqrt(ε_k^2 + Δ_k^2)
  - 能隙方程：
      1 = (U / 2N) Σ_k (cos k_x - cos k_y)^2 * tanh(β E_k/2) / (2 E_k)
  - 临界温度 T_c 由线性化方程确定（Δ→0）：
      1 = (U / 2N) Σ_k (cos k_x - cos k_y)^2 * tanh(β_c ε_k/2) / (2 ε_k)
"""

import numpy as np
from utils import safe_sqrt, fermi_dirac
from bz_integration import integrate_bz_gauss_legendre_2d


def d_wave_form_factor(kx, ky):
    """
    d-波配对形状因子：
        φ_k = (cos k_x - cos k_y) / 2
    """
    return 0.5 * (np.cos(kx) - np.cos(ky))


def quasiparticle_energy(kx, ky, Delta0, t=1.0, tp=0.3, mu=0.0):
    """
    准粒子激发能量：
        ε_k = -2t(cos k_x + cos k_y) + 4t' cos k_x cos k_y - μ
        Δ_k = Δ_0 * φ_k
        E_k = sqrt(ε_k^2 + Δ_k^2)
    """
    eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu
    dk = Delta0 * d_wave_form_factor(kx, ky)
    return safe_sqrt(eps ** 2 + dk ** 2)


def gap_equation_integrand(k_points, Delta0, U, beta, t=1.0, tp=0.3, mu=0.0):
    """
    能隙方程的被积函数：
        f(k) = (U/2) * φ_k^2 * tanh(β E_k / 2) / E_k
    返回各 k 点的被积函数值。
    """
    # TODO: Hole_2 - implement the BCS d-wave gap equation integrand
    raise NotImplementedError("Hole_2: implement f(k) = (U/2) * phi_k^2 * tanh(beta*E_k/2) / E_k")


def gap_equation_rhs(Delta0, U, beta, t=1.0, tp=0.3, mu=0.0, n_k=48):
    """
    计算能隙方程右侧积分值（使用 Gauss-Legendre 张量积）。

    方程形式：RHS = (U/2N) Σ_k φ_k^2 tanh(β E_k/2) / E_k
    """
    if Delta0 < 0:
        Delta0 = abs(Delta0)

    def f(kpts):
        return gap_equation_integrand(kpts, Delta0, U, beta, t, tp, mu)

    val = integrate_bz_gauss_legendre_2d(f, n_per_dim=n_k)
    # 归一化：Gauss-Legendre 在 [-π,π]^2 上的权重和 = (2π)^2
    # 但 integrate_bz_gauss_legendre_2d 已经包含权重，所以 val 就是积分值
    # 需要除以 BZ 体积 (2π)^2 以得到平均值
    bz_vol = (2.0 * np.pi) ** 2
    return val / bz_vol


def solve_gap_self_consistent(U, beta, t=1.0, tp=0.3, mu=0.0,
                               Delta_max=5.0, n_k=48, tol=1e-8, max_iter=200):
    """
    用 Picard 迭代（或混合迭代）自洽求解 d-波能隙方程。

    Parameters
    ----------
    U : float
        有效吸引相互作用强度（U > 0 表示吸引）。
    beta : float
        逆温度 β = 1/(k_B T)。
    t, tp, mu : float
        紧束缚参数。
    Delta_max : float
        能隙上界搜索范围。
    n_k : int
        k 点网格密度。
    tol : float
        自洽收敛阈值。
    max_iter : int
        最大迭代次数。

    Returns
    -------
    Delta0 : float
        自洽能隙幅值。若无解则返回 0.0。
    history : list
        迭代历史。
    converged : bool
    """
    if U <= 0:
        return 0.0, [0.0], True

    history = []
    Delta = 0.5  # 初始猜测
    alpha_mix = 0.3  # 混合系数

    for it in range(max_iter):
        rhs = gap_equation_rhs(Delta, U, beta, t, tp, mu, n_k)
        # 方程为 1 = rhs，因此新 Delta 需满足 rhs(Delta_new) = 1
        # 但 rhs 不是显式关于 Delta 的函数，我们将其视为不动点迭代：
        # Delta_{new} = Delta * rhs(Delta)  ???
        # 实际上，正确的不动点应通过根搜索：令 g(Delta) = rhs(Delta) - 1 = 0
        # 使用简单迭代：若 rhs > 1，需增大 Delta（因为 rhs 随 Delta 增大而减小）
        # 这里用二分搜索求解 g(Delta) = 0
        break

    # 使用二分法直接求解 g(Delta) = gap_equation_rhs(Delta) - 1 = 0
    def g(D):
        return gap_equation_rhs(D, U, beta, t, tp, mu, n_k) - 1.0

    # 确定 bracket
    # 当 Delta=0 时，rhs 最大；当 Delta→∞ 时，rhs→0
    g0 = g(0.0)
    if g0 < 0:
        # 即使 Delta=0 也不满足，说明 U 太小或 T 太高，无能隙
        return 0.0, [0.0], False

    # 寻找上界
    d_hi = Delta_max
    for _ in range(50):
        if g(d_hi) < 0:
            break
        d_hi *= 2.0
        if d_hi > 1e4:
            raise RuntimeError("无法找到能隙方程根的上界。")

    d_lo = 0.0
    history = []
    for it in range(max_iter):
        d_mid = (d_lo + d_hi) * 0.5
        g_mid = g(d_mid)
        history.append(d_mid)
        if abs(g_mid) < tol or (d_hi - d_lo) < tol * max(1.0, d_mid):
            return d_mid, history, True
        if g_mid > 0:
            d_lo = d_mid
        else:
            d_hi = d_mid

    Delta_sc = (d_lo + d_hi) * 0.5
    return Delta_sc, history, False


def compute_critical_temperature(U, t=1.0, tp=0.3, mu=0.0,
                                  beta_max=100.0, n_k=48, tol=1e-6):
    """
    通过线性化能隙方程（Δ→0）估算临界温度 T_c。

    在 Δ→0 时，tanh(β E/2)/E → tanh(β ε/2)/ε，
    方程变为：
        1 = (U/2N) Σ_k φ_k^2 tanh(β_c ε_k/2) / ε_k
    对 β 做二分搜索。
    """
    if U <= 0:
        return 0.0

    def h(beta):
        def f(kpts):
            kx = kpts[:, 0]
            ky = kpts[:, 1]
            phi = d_wave_form_factor(kx, ky)
            eps = (-2.0 * t * (np.cos(kx) + np.cos(ky))
                   + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu)
            eps = np.where(np.abs(eps) < 1e-12, 1e-12, eps)
            tanh_term = np.tanh(beta * eps * 0.5)
            return 0.5 * U * phi ** 2 * tanh_term / eps
        val = integrate_bz_gauss_legendre_2d(f, n_per_dim=n_k)
        bz_vol = (2.0 * np.pi) ** 2
        return val / bz_vol - 1.0

    # 低温极限 beta→∞ 时 h 单调递减
    h0 = h(1e-6)
    if h0 < 0:
        return 0.0  # 即使 T→∞ 也无解
    h_hi = h(beta_max)
    if h_hi > 0:
        # beta_max 仍不足，可能无超导
        return 1.0 / beta_max

    b_lo = 1e-6
    b_hi = beta_max
    for _ in range(80):
        b_mid = (b_lo + b_hi) * 0.5
        h_mid = h(b_mid)
        if abs(h_mid) < tol:
            break
        if h_mid > 0:
            b_lo = b_mid
        else:
            b_hi = b_mid
    beta_c = (b_lo + b_hi) * 0.5
    return 1.0 / beta_c


def box_behnken_parameter_sweep(U_range, T_range, tp_range, mu_range):
    """
    使用 Box-Behnken 设计在四维参数空间 (U, T, t', μ) 中生成实验点，
    用于系统探索超导相图。

    对应种子项目 111_box_behnken。
    """
    from utils import box_behnken
    ranges = np.array([
        [U_range[0], U_range[1]],
        [T_range[0], T_range[1]],
        [tp_range[0], tp_range[1]],
        [mu_range[0], mu_range[1]]
    ], dtype=float)
    design = box_behnken(4, ranges)
    return design


def compute_free_energy(Delta0, U, beta, t=1.0, tp=0.3, mu=0.0, n_k=32):
    """
    计算超导态相对于正常态的自由能差 ΔF = F_s - F_n。

    BCS 自由能（平均场）：
        F = - (1/β) Σ_k [2 ln(2 cosh(β E_k/2))]
            + Σ_k [ε_k - E_k]
            + |Δ_0|^2 / U
    这里计算差值，省略常数项。
    """
    def integrand(kpts):
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu
        phi = d_wave_form_factor(kx, ky)
        dk = Delta0 * phi
        Ek = safe_sqrt(eps ** 2 + dk ** 2)
        Ek = np.maximum(Ek, 1e-14)
        # 自由能密度
        f = -(2.0 / beta) * np.log(2.0 * np.cosh(beta * Ek * 0.5))
        f += eps - Ek
        return f

    val = integrate_bz_gauss_legendre_2d(integrand, n_per_dim=n_k)
    bz_vol = (2.0 * np.pi) ** 2
    F = val / bz_vol + Delta0 ** 2 / U
    return F
