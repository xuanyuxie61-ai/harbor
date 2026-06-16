# -*- coding: utf-8 -*-
"""
convergence_benchmark.py
========================
Barenblatt 精确解收敛性验证.

核心算法 (来自 901_porous_medium_exact):
----------------------------------------
多孔介质方程 (PME) 的 Barenblatt 自相似解:
    ∂u/∂t = ∂²/∂x² (u^m)

    精确解:
    u(x,t) = (t+δ)^{-β} · [c - γ(x/(t+δ)^β)²]_{+}^{1/(m-1)}

    其中:
        α = 1/(m-1)
        β = 1/(m+1)
        γ = (m-1)/(2m(m+1))
        c, δ: 参数

    在 fast ignition 中, 能量沉积 PDE 的非线性扩散部分
    κ(u) ∝ u^{5/2} 对应 m = 7/2 的 PME 形式.

收敛阶计算:
    设 E_h 为网格 h 上的误差, 则 p 阶格式:
        E_h ∝ h^p
        p = ln(E_{h1}/E_{h2}) / ln(h1/h2)

核心来源 (种子项目映射):
- 901_porous_medium_exact: Barenblatt 解
- 338_errors: 误差计算
"""

import math


# ============================================================
# Barenblatt 精确解 (来自 901_porous_medium_exact)
# ============================================================
def barenblatt_parameters(c=None, delta=None, m=None):
    """
    返回 Barenblatt 解参数 (来自 901_porous_medium_exact/porous_medium_parameters).

    默认值:
        c     = sqrt(3)/15
        delta = 1/75
        m     = 3
    """
    if c is None:
        c = math.sqrt(3.0) / 15.0
    if delta is None:
        delta = 1.0 / 75.0
    if m is None:
        m = 3.0
    return c, delta, m


def barenblatt_exact(x, t, c=None, delta=None, m=None):
    """
    Barenblatt 精确解:
        u(x,t) = (t+δ)^{-β} · [c - γ(x/(t+δ)^β)²]_{+}^α

    参数:
        x, t  : 空间和时间
        c, delta, m: Barenblatt 参数
    返回:
        u, u_t, u_x, u_xx: 解及其导数
    """
    c, delta, m = barenblatt_parameters(c, delta, m)
    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    xi = x / bot
    factor = c - gamma * xi ** 2

    if factor > 0:
        u = (t + delta) ** (-alpha * (m - 1.0)) * factor ** alpha
        # 简化: u = (t+δ)^{-β} · factor^α
        u = (t + delta) ** (-beta) * factor ** alpha

        # 时间导数
        u_t = (2.0 * alpha * beta * gamma * (t + delta) ** (-1.0 - 3.0 * beta)
               * x ** 2 * factor ** (alpha - 1.0)
               - beta * (t + delta) ** (-1.0 - beta) * factor ** alpha)

        # 空间导数
        u_x = (-2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta)
               * x * factor ** (alpha - 1.0))

        # 二阶导数
        u_xx = (4.0 * (alpha - 1.0) * alpha * gamma ** 2
                * (t + delta) ** (-5.0 * beta) * x ** 2
                * factor ** (alpha - 2.0)
                - 2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta)
                * factor ** (alpha - 1.0))
    else:
        u = 0.0
        u_t = 0.0
        u_x = 0.0
        u_xx = 0.0

    return u, u_t, u_x, u_xx


def barenblatt_residual(x, t, c=None, delta=None, m=None):
    """
    PME 残差: R = u_t - m(m-1)u^{m-2}u_x^2 - m u^{m-1} u_xx

    精确解应使 R ≈ 0.
    """
    c, delta, m = barenblatt_parameters(c, delta, m)
    u, u_t, u_x, u_xx = barenblatt_exact(x, t, c, delta, m)

    if u <= 0:
        return u_t  # 应接近 0

    diffusion = m * (m - 1.0) * u ** (m - 2.0) * u_x ** 2 + m * u ** (m - 1.0) * u_xx
    return u_t - diffusion


# ============================================================
# 收敛性测试框架
# ============================================================
def convergence_rate(errors, grid_sizes):
    """
    计算相邻网格的收敛阶:
        p_i = ln(E_i / E_{i+1}) / ln(h_i / h_{i+1})

    参数:
        errors    : 误差列表 (从粗到细)
        grid_sizes: 对应网格尺寸
    返回:
        rates: 收敛阶列表 (长度 = len(errors) - 1)
    """
    rates = []
    for i in range(len(errors) - 1):
        if errors[i + 1] > 0 and errors[i] > 0 and grid_sizes[i] != grid_sizes[i + 1]:
            rate = math.log(errors[i] / errors[i + 1]) / math.log(
                grid_sizes[i] / grid_sizes[i + 1])
            rates.append(rate)
        else:
            rates.append(0.0)
    return rates


def l2_error_1d(u_num, u_exact, dx):
    """
    L2 误差:
        ||e||_2 = sqrt(Σ (u_num - u_exact)^2 · Δx)
    """
    n = len(u_num)
    err2 = sum((u_num[i] - u_exact[i]) ** 2 for i in range(n)) * dx
    return math.sqrt(err2)


def linf_error_1d(u_num, u_exact):
    """L∞ 误差: max|u_num - u_exact|."""
    return max(abs(u_num[i] - u_exact[i]) for i in range(len(u_num)))


def run_barenblatt_convergence_test(n_levels=4, base_nx=16, T_end=0.1):
    """
    运行 Barenblatt 收敛性测试.

    方法:
    1. 在 t=0 设置 Barenblatt 精确解为初始条件
    2. 用 FTCS 推进到 t = T_end
    3. 与 Barenblatt 精确解比较, 计算 L2 误差
    4. 细化网格, 重复, 估计收敛阶

    参数:
        n_levels: 细化层次数
        base_nx : 最粗网格点数
        T_end   : 终止时间
    返回:
        dict: 包含 'nx_list', 'errors', 'rates'
    """
    nx_list = [base_nx * (2 ** k) for k in range(n_levels)]
    errors = []

    for nx in nx_list:
        Lx = 4.0  # 足够大的域
        dx = Lx / (nx - 1)
        x = [i * dx for i in range(nx)]

        # 初始条件: Barenblatt at t=0
        u0 = [barenblatt_exact(xi, 0.0)[0] for xi in x]

        # 时间步 (满足 CFL)
        m = 3.0
        kappa_max = m * max(max(u0) ** (m - 1.0), 1.0e-10)
        dt = 0.4 * dx ** 2 / max(kappa_max, 1.0e-10)
        nt = max(int(T_end / dt), 10)
        dt = T_end / nt

        # FTCS 时间推进 (PME: ∂u/∂t = ∂²(u^m)/∂x²)
        u = list(u0)
        for step in range(nt):
            u_new = list(u)
            for i in range(1, nx - 1):
                # ∂²(u^m)/∂x² 使用 2阶 FD
                um_ip = max(u[i + 1], 0.0) ** m
                um_i = max(u[i], 0.0) ** m
                um_im = max(u[i - 1], 0.0) ** m
                d2um = (um_ip - 2.0 * um_i + um_im) / (dx ** 2)
                u_new[i] = u[i] + dt * d2um
            # 边界: Dirichlet (u=0)
            u_new[0] = 0.0
            u_new[-1] = 0.0
            # 非负保护
            u = [max(ui, 0.0) for ui in u_new]

        # 精确解 at t=T_end
        u_exact = [barenblatt_exact(xi, T_end)[0] for xi in x]

        # 误差
        err = l2_error_1d(u, u_exact, dx)
        errors.append(err)

    # 收敛阶
    grid_sizes = [Lx / (nx - 1) for nx in nx_list]
    rates = convergence_rate(errors, grid_sizes)

    return {
        "nx_list": nx_list,
        "dx_list": grid_sizes,
        "errors": errors,
        "rates": rates,
        "T_end": T_end,
    }


def print_convergence_summary(result):
    """打印收敛性摘要."""
    print("\n" + "=" * 72)
    print("Barenblatt 精确解收敛性验证")
    print("=" * 72)
    print("  PME 方程: ∂u/∂t = ∂²(u^m)/∂x²  (m=3)")
    print("  终止时间 T = {:.3f}".format(result["T_end"]))
    print("")
    print("  {:>6s}  {:>12s}  {:>12s}  {:>10s}".format(
        "Nx", "Δx", "L2 误差", "收敛阶"))
    print("  " + "-" * 50)
    for i in range(len(result["nx_list"])):
        rate_str = "---" if i == 0 else "{:.3f}".format(result["rates"][i - 1])
        print("  {:6d}  {:12.4e}  {:12.4e}  {:>10s}".format(
            result["nx_list"][i],
            result["dx_list"][i],
            result["errors"][i],
            rate_str))

    if result["rates"]:
        avg_rate = sum(result["rates"]) / len(result["rates"])
        print("")
        print("  平均收敛阶: {:.3f} (理论 2阶 FTCS)".format(avg_rate))
    print("=" * 72)
