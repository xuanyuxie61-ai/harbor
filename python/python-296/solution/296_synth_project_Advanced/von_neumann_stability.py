# -*- coding: utf-8 -*-
"""
von_neumann_stability.py
========================
高阶有限差分格式的 von Neumann 稳定性分析.

核心算法:
---------
Von Neumann 稳定性分析:
    将数值解展开为 Fourier 模式:
        u_j^n = G^n · exp(i k j Δx)
    代入差分格式, 得到放大因子 G(k Δt, k Δx).
    稳定性条件: |G| ≤ 1 + O(Δt) 对所有 k.

对 FTCS 格式应用于线性扩散 ∂u/∂t = κ ∂²u/∂x²:
    G = 1 - 4 r sin²(k Δx / 2)    (2阶)
    G = 1 - r(2 sin²(θ) - (2/3) sin²(2θ))  (4阶, 近似)
    其中 r = κ Δt / Δx².
    稳定条件: r ≤ 1/2  (2阶), r ≤ ~3/8 (4阶).

对非线性 κ(u) = κ_0 u^{5/2}:
    需用最大 κ 值: r_max = κ(u_max) Δt / Δx²

CFL 条件:
    Δt ≤ C_CFL · Δx² / κ_max    (扩散)
    Δt ≤ C_CFL · Δx / v_max     (对流)

核心来源 (种子项目映射):
- 434_fisher_pde_ftcs: FTCS 稳定性基础
- 338_errors: 数值误差分析
- 816_normal: 随机扰动用于测试稳定性
"""

import math
import cmath

from electron_beam_source import box_muller_single


# ============================================================
# Von Neumann 放大因子 (线性情况)
# ============================================================
def amplification_factor_ftcs_2nd(r, theta):
    """
    2阶 FTCS 放大因子:
        G(θ) = 1 - 2r(1 - cos θ) = 1 - 4r sin²(θ/2)

    参数:
        r    : 扩散数 r = κ Δt / Δx²
        theta: 相位角 θ = k Δx ∈ [0, π]
    返回:
        G : 复数放大因子
    """
    G = 1.0 - 2.0 * r * (1.0 - math.cos(theta))
    return complex(G, 0.0)


def amplification_factor_ftcs_4th(r, theta):
    """
    4阶 FTCS 放大因子 (近似):
        4阶空间离散对 ∂²/∂x²:
        D^(4) = (-u_{j+2} + 16u_{j+1} - 30u_j + 16u_{j-1} - u_{j-2}) / (12Δx²)

        代入 Fourier 模式:
        σ(θ) = (-e^{2iθ} + 16e^{iθ} - 30 + 16e^{-iθ} - e^{-2iθ}) / 12
             = (-2cos(2θ) + 32cos(θ) - 30) / 12

        G = 1 + r · σ(θ)

    参数:
        r    : 扩散数
        theta: 相位角
    返回:
        G : 复数放大因子
    """
    sigma = (-2.0 * math.cos(2.0 * theta) + 32.0 * math.cos(theta) - 30.0) / 12.0
    G = 1.0 + r * sigma
    return complex(G, 0.0)


def max_amplification_ftcs_2nd(r, n_modes=64):
    """
    计算 2阶 FTCS 在所有模式上的最大放大因子.

    参数:
        r     : 扩散数
        n_modes: 分析模式数
    返回:
        |G|_max
    """
    max_G = 0.0
    for m in range(n_modes + 1):
        theta = math.pi * m / n_modes
        G = amplification_factor_ftcs_2nd(r, theta)
        max_G = max(max_G, abs(G))
    return max_G


def max_amplification_ftcs_4th(r, n_modes=64):
    """
    计算 4阶 FTCS 在所有模式上的最大放大因子.
    """
    max_G = 0.0
    for m in range(n_modes + 1):
        theta = math.pi * m / n_modes
        G = amplification_factor_ftcs_4th(r, theta)
        max_G = max(max_G, abs(G))
    return max_G


# ============================================================
# 临界扩散数 (稳定性极限)
# ============================================================
def find_critical_r_2nd(tolerance=1.0e-6, n_modes=128):
    """
    二分法搜索 2阶 FTCS 的临界扩散数 r_crit.
    使得 |G|_max ≤ 1 + tolerance.

    理论值: r_crit = 0.5
    """
    r_lo, r_hi = 0.0, 1.0
    for _ in range(60):
        r_mid = 0.5 * (r_lo + r_hi)
        G_max = max_amplification_ftcs_2nd(r_mid, n_modes)
        if G_max > 1.0 + tolerance:
            r_hi = r_mid
        else:
            r_lo = r_mid
    return 0.5 * (r_lo + r_hi)


def find_critical_r_4th(tolerance=1.0e-6, n_modes=128):
    """
    二分法搜索 4阶 FTCS 的临界扩散数 r_crit.

    理论值约为 0.375 = 3/8.
    """
    r_lo, r_hi = 0.0, 1.0
    for _ in range(60):
        r_mid = 0.5 * (r_lo + r_hi)
        G_max = max_amplification_ftcs_4th(r_mid, n_modes)
        if G_max > 1.0 + tolerance:
            r_hi = r_mid
        else:
            r_lo = r_mid
    return 0.5 * (r_lo + r_hi)


# ============================================================
# CFL 条件计算
# ============================================================
def cfl_diffusion_time_step(kappa_max, dx, cfl_number=0.4, fd_order=2):
    """
    扩散 CFL 时间步长:
        Δt ≤ C_CFL · Δx² / κ_max

    对 2阶: C_CFL ≤ 0.5
    对 4阶: C_CFL ≤ ~0.375

    参数:
        kappa_max : 最大热导率
        dx        : 空间步长
        cfl_number: CFL 数
        fd_order  : FD 阶数
    返回:
        dt_max: 最大允许时间步长
    """
    if kappa_max <= 0:
        return float('inf')
    r_crit = 0.5 if fd_order == 2 else 0.375
    dt_max = cfl_number * r_crit * dx ** 2 / kappa_max
    return dt_max


def cfl_advection_time_step(v_max, dx, cfl_number=0.8):
    """
    对流 CFL 时间步长:
        Δt ≤ C_CFL · Δx / v_max

    参数:
        v_max     : 最大速度
        dx        : 空间步长
        cfl_number: CFL 数
    返回:
        dt_max
    """
    if v_max <= 0:
        return float('inf')
    return cfl_number * dx / v_max


# ============================================================
# 随机扰动稳定性测试 (来自 816_normal)
# ============================================================
def random_perturbation_stability_test(nx, kappa, dx, dt, n_steps,
                                       perturbation_amplitude=1.0e-6,
                                       seed=42):
    """
    随机扰动测试: 在均匀解上施加小随机扰动, 观察增长/衰减.

    步骤:
    1. 初始化 u = u_0 + ε · N(0,1)
    2. 推进 n_steps 步 (线性扩散, 无源)
    3. 计算扰动 L2 范数随时间变化

    参数:
        nx                  : 网格点数
        kappa               : 热导率
        dx                  : 空间步长
        dt                  : 时间步长
        n_steps             : 步数
        perturbation_amplitude: 扰动幅度 ε
        seed                : 随机种子
    返回:
        dict: 包含 'perturbation_norm', 'growth_rate', 'is_stable'
    """
    import random as rng_module
    rng = rng_module.Random(seed)

    # 初始扰动
    u0 = [perturbation_amplitude * box_muller_single(rng) for _ in range(nx)]
    u = list(u0)

    r = kappa * dt / (dx ** 2)
    norms = [math.sqrt(sum(ui ** 2 for ui in u) / nx)]

    for step in range(n_steps):
        u_new = list(u)
        for i in range(1, nx - 1):
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1])
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
        u = u_new
        norm = math.sqrt(sum(ui ** 2 for ui in u) / nx)
        norms.append(norm)

    # 估计增长率
    if norms[0] > 0 and norms[-1] > 0:
        growth_rate = math.log(max(norms[-1], 1.0e-300) / norms[0]) / max(n_steps * dt, 1.0e-300)
    else:
        growth_rate = 0.0

    is_stable = norms[-1] <= norms[0] * 1.01  # 允许 1% 增长

    return {
        "r": r,
        "norms": norms,
        "growth_rate": growth_rate,
        "is_stable": is_stable,
        "initial_norm": norms[0],
        "final_norm": norms[-1],
        "amplification": norms[-1] / max(norms[0], 1.0e-300),
    }


# ============================================================
# 完整稳定性分析
# ============================================================
def full_stability_analysis(kappa_max, dx, dy, fd_orders=(2, 4),
                            n_modes=64, n_random_tests=3):
    """
    执行完整的稳定性分析.

    参数:
        kappa_max  : 最大热导率
        dx, dy     : 空间步长
        fd_orders  : 要分析的 FD 阶数列表
        n_modes    : von Neumann 模式数
        n_random_tests: 随机测试次数
    返回:
        dict: 分析结果
    """
    results = {}

    for order in fd_orders:
        r_crit = find_critical_r_2nd() if order == 2 else find_critical_r_4th()
        dt_max = cfl_diffusion_time_step(kappa_max, dx, 0.8, order)

        # 放大因子谱
        r_test = 0.8 * r_crit  # 稳定区内
        theta_vals = [math.pi * m / n_modes for m in range(n_modes + 1)]
        if order == 2:
            G_vals = [abs(amplification_factor_ftcs_2nd(r_test, th))
                      for th in theta_vals]
        else:
            G_vals = [abs(amplification_factor_ftcs_4th(r_test, th))
                      for th in theta_vals]

        results[order] = {
            "r_critical": r_crit,
            "dt_max": dt_max,
            "G_max_stable": max(G_vals) if G_vals else 0.0,
            "G_spectra": list(zip(theta_vals, G_vals)),
        }

    # 随机扰动测试
    random_tests = []
    for k in range(n_random_tests):
        r_test = 0.3 + 0.1 * k  # 扫描不同扩散数
        test = random_perturbation_stability_test(
            nx=64, kappa=r_test * dx ** 2 / 1.0e-12,
            dx=dx, dt=1.0e-12, n_steps=100, seed=42 + k)
        random_tests.append(test)

    results["random_tests"] = random_tests
    return results


def print_stability_summary(analysis):
    """打印稳定性分析摘要."""
    print("\n" + "=" * 72)
    print("Von Neumann 稳定性分析")
    print("=" * 72)

    for order in [2, 4]:
        if order not in analysis:
            continue
        res = analysis[order]
        print("  FD 阶数 {}:".format(order))
        print("    临界扩散数 r_crit = {:.6f} (理论: {})".format(
            res["r_critical"],
            "0.5" if order == 2 else "~0.375"))
        print("    最大放大因子 (稳定区): {:.6f}".format(res["G_max_stable"]))
        print("    CFL 时间步长上限   : {:.4e} s".format(res["dt_max"]))

    if "random_tests" in analysis:
        print("  随机扰动稳定性测试:")
        for i, test in enumerate(analysis["random_tests"]):
            print("    测试 #{}: r={:.3f}, 放大比={:.4e}, 稳定={}".format(
                i, test["r"], test["amplification"],
                "是" if test["is_stable"] else "否"))

    print("=" * 72)
