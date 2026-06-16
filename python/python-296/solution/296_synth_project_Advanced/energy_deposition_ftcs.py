# -*- coding: utf-8 -*-
"""
energy_deposition_ftcs.py
=========================
Fast Ignition 电子束能量沉积 PDE 求解器.

核心方程 (非线性扩散 + 源项):
------------------------------
∂u/∂t = ∂/∂x (κ(u) ∂u/∂x) + ∂/∂y (κ(u) ∂u/∂y) + S(x, y, t)

其中:
    u(x,y,t)  : 电子能量密度 [J/m^3]
    κ(u)      : Spitzer-Härm 热导率 κ_0 · u^{5/2}  [W/m/K → 适当单位]
    S(x,y,t)  : 快电子束沉积源 [J/(m^3·s)]

离散化方案:
-----------
(1) FTCS (Forward Time Centered Space):
    u^{n+1}_{i,j} = u^n_{i,j} + Δt · (D_x + D_y + S^n_{i,j})
    其中 D_x = (κ_{i+1/2}(u_{i+1}-u_i) - κ_{i-1/2}(u_i-u_{i-1})) / Δx^2

(2) 高阶 (4阶) 空间离散 (来自 vandermonde_fd):
    D_x^{(4)} = (-κ_{i+2}u_{i+2} + 16κ_{i+1}u_{i+1} - 30κ_i u_i
                 + 16κ_{i-1}u_{i-1} - κ_{i-2}u_{i-2}) / (12 Δx^2)

(3) 非线性 κ(u) 在单元界面取算术平均:
    κ_{i+1/2} = 0.5 · (κ(u_i) + κ(u_{i+1}))

核心来源 (种子项目映射):
- 434_fisher_pde_ftcs : FTCS 基本框架
- 901_porous_medium_exact : 非线性扩散方程结构
"""

import math

from vandermonde_fd import uniform_fd_coefficients


# ============================================================
# Spitzer-Härm 热导率 (非线性)
# ============================================================
def spitzer_kappa(u, kappa_0):
    """
    Spitzer-Härm 热导率:
        κ(u) = κ_0 · u^{5/2}

    参数:
        u      : 能量密度 (正)
        kappa_0: 系数
    返回:
        κ(u)
    """
    if u <= 0.0:
        return 0.0
    return kappa_0 * (u ** 2.5)


def spitzer_kappa_derivative(u, kappa_0):
    """
    dκ/du = (5/2) · κ_0 · u^{3/2}
    """
    if u <= 0.0:
        return 0.0
    return 2.5 * kappa_0 * (u ** 1.5)


# ============================================================
# 源项: 快电子束能量沉积
# ============================================================
def gaussian_beam_source(x, y, t, x0, y0, r_beam, tau_pulse,
                         total_energy, t_start=0.0):
    """
    高斯时空束源:
        S(x,y,t) = (E_total / V_eff) · exp(-r^2/r_b^2) · g(t)
    其中:
        r^2 = (x-x_0)^2 + (y-y_0)^2
        g(t) = (2/(τ√π)) · exp(-((t-t_start-τ/2)/(τ/2))^2)  (归一化脉冲)
        V_eff = π · r_b^2 · L_dep  (有效体积)

    参数:
        x, y       : 空间坐标
        t          : 时间
        x0, y0     : 束斑中心
        r_beam     : 束斑半径
        tau_pulse  : 脉冲持续时间
        total_energy: 总能量
        t_start    : 脉冲起始时间
    返回:
        S [J/(m^3·s)]
    """
    # 空间部分
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    spatial = math.exp(-r2 / max(r_beam ** 2, 1.0e-30))

    # 时间部分 (高斯脉冲)
    if tau_pulse <= 0:
        temporal = 0.0
    else:
        t_center = t_start + 0.5 * tau_pulse
        sigma_t = tau_pulse / 4.0  # 4σ 覆盖脉冲宽度
        temporal = math.exp(-((t - t_center) / sigma_t) ** 2)
        temporal /= max(sigma_t * math.sqrt(math.pi), 1.0e-30)

    # 归一化: 总能量 = ∫ S dV dt
    V_eff = math.pi * max(r_beam, 1.0e-15) ** 2 * max(r_beam, 1.0e-15)
    amplitude = total_energy / max(V_eff, 1.0e-30)

    return amplitude * spatial * temporal


# ============================================================
# FTCS 求解器 (来自 434_fisher_pde_ftcs)
# ============================================================
def ftcs_step_1d(u, dx, dt, kappa_0, source_vec, fd_order=2):
    """
    一维 FTCS 单步推进.

    方程: ∂u/∂t = ∂/∂x(κ(u) ∂u/∂x) + S

    离散 (2阶):
        u_i^{n+1} = u_i^n + Δt · [(κ_{i+1/2}(u_{i+1}-u_i)
                                   - κ_{i-1/2}(u_i-u_{i-1})) / Δx^2
                                   + S_i]

    参数:
        u        : 当前解 (长度 nx)
        dx       : 空间步长
        dt       : 时间步长
        kappa_0  : 热导率系数
        source_vec: 源项向量 (长度 nx)
        fd_order : 空间精度 (2 或 4)
    返回:
        u_new: 下一步解
    """
    nx = len(u)
    u_new = list(u)

    if fd_order == 2:
        # 2阶 FTCS
        for i in range(1, nx - 1):
            # 界面热导率 (算术平均)
            kappa_ip = 0.5 * (spitzer_kappa(u[i], kappa_0)
                              + spitzer_kappa(u[i + 1], kappa_0))
            kappa_im = 0.5 * (spitzer_kappa(u[i - 1], kappa_0)
                              + spitzer_kappa(u[i], kappa_0))
            # 扩散通量
            flux_ip = kappa_ip * (u[i + 1] - u[i])
            flux_im = kappa_im * (u[i] - u[i - 1])
            diffusion = (flux_ip - flux_im) / (dx ** 2)
            u_new[i] = u[i] + dt * (diffusion + source_vec[i])
    elif fd_order == 4:
        # 4阶 FD 系数 (来自 vandermonde_fd)
        c = uniform_fd_coefficients(4, 2, 2)
        # c = [-1/12, 4/3, -5/2, 4/3, -1/12]
        for i in range(2, nx - 2):
            # 非线性: 在每个节点计算 κ(u), 再应用 FD
            kappa_vals = [spitzer_kappa(u[i + j], kappa_0) for j in range(-2, 3)]
            u_vals = [u[i + j] for j in range(-2, 3)]
            # 应用 ∂/∂x(κ ∂u/∂x) ≈ Σ c_j · κ_{i+j} · (Laplacian 近似)
            # 简化: 使用 κ(u_i) · ∂^2 u/∂x^2 + (∂κ/∂u)(∂u/∂x)^2
            d2u = sum(c[j + 2] * u_vals[j] for j in range(-2, 3)) / (dx ** 2)
            # 一阶导数 (4阶)
            c1 = uniform_fd_coefficients(4, 2, 1)
            du = sum(c1[j + 2] * u_vals[j] for j in range(-2, 3)) / dx
            kappa_i = spitzer_kappa(u[i], kappa_0)
            dkappa_du = spitzer_kappa_derivative(u[i], kappa_0)
            diffusion = kappa_i * d2u + dkappa_du * du ** 2
            u_new[i] = u[i] + dt * (diffusion + source_vec[i])
    else:
        raise ValueError("fd_order 必须为 2 或 4")

    # 边界条件
    u_new[0] = u_new[1]            # Neumann (零通量)
    u_new[-1] = u_new[-2]          # Neumann

    # 非负保护
    for i in range(nx):
        if u_new[i] < 0.0:
            u_new[i] = 0.0

    return u_new


def ftcs_step_2d(u, dx, dy, dt, kappa_0, source_mat, fd_order=2):
    """
    二维 FTCS 单步推进.

    方程: ∂u/∂t = ∂/∂x(κ(u)∂u/∂x) + ∂/∂y(κ(u)∂u/∂y) + S

    参数:
        u        : 二维解矩阵 (ny × nx)
        dx, dy   : 空间步长
        dt       : 时间步长
        kappa_0  : 热导率系数
        source_mat: 源项矩阵 (ny × nx)
        fd_order : 空间精度
    返回:
        u_new: 下一步解
    """
    ny = len(u)
    nx = len(u[0])
    u_new = [[0.0] * nx for _ in range(ny)]

    hw = 2 if fd_order == 4 else 1

    for j in range(ny):
        for i in range(nx):
            if i < hw or i >= nx - hw or j < hw or j >= ny - hw:
                # 边界: 保持原值 (Neumann 在后续处理)
                u_new[j][i] = u[j][i]
                continue

            # x 方向扩散
            if fd_order == 2:
                kappa_xp = 0.5 * (spitzer_kappa(u[j][i], kappa_0)
                                  + spitzer_kappa(u[j][i + 1], kappa_0))
                kappa_xm = 0.5 * (spitzer_kappa(u[j][i - 1], kappa_0)
                                  + spitzer_kappa(u[j][i], kappa_0))
                diff_x = (kappa_xp * (u[j][i + 1] - u[j][i])
                          - kappa_xm * (u[j][i] - u[j][i - 1])) / (dx ** 2)
            else:
                c = uniform_fd_coefficients(4, 2, 2)
                kappa_vals_x = [spitzer_kappa(u[j][i + k], kappa_0) for k in range(-2, 3)]
                u_vals_x = [u[j][i + k] for k in range(-2, 3)]
                c1 = uniform_fd_coefficients(4, 2, 1)
                du_x = sum(c1[k + 2] * u_vals_x[k] for k in range(-2, 3)) / dx
                d2u_x = sum(c[k + 2] * u_vals_x[k] for k in range(-2, 3)) / (dx ** 2)
                kappa_i = spitzer_kappa(u[j][i], kappa_0)
                dkdu = spitzer_kappa_derivative(u[j][i], kappa_0)
                diff_x = kappa_i * d2u_x + dkdu * du_x ** 2

            # y 方向扩散
            if fd_order == 2:
                kappa_yp = 0.5 * (spitzer_kappa(u[j][i], kappa_0)
                                  + spitzer_kappa(u[j + 1][i], kappa_0))
                kappa_ym = 0.5 * (spitzer_kappa(u[j - 1][i], kappa_0)
                                  + spitzer_kappa(u[j][i], kappa_0))
                diff_y = (kappa_yp * (u[j + 1][i] - u[j][i])
                          - kappa_ym * (u[j][i] - u[j - 1][i])) / (dy ** 2)
            else:
                kappa_vals_y = [spitzer_kappa(u[j + k][i], kappa_0) for k in range(-2, 3)]
                u_vals_y = [u[j + k][i] for k in range(-2, 3)]
                du_y = sum(c1[k + 2] * u_vals_y[k] for k in range(-2, 3)) / dy
                d2u_y = sum(c[k + 2] * u_vals_y[k] for k in range(-2, 3)) / (dy ** 2)
                diff_y = kappa_i * d2u_y + dkdu * du_y ** 2

            u_new[j][i] = u[j][i] + dt * (diff_x + diff_y + source_mat[j][i])

    # Neumann 边界
    for j in range(ny):
        u_new[j][0] = u_new[j][1]
        u_new[j][-1] = u_new[j][-2]
    for i in range(nx):
        u_new[0][i] = u_new[1][i]
        u_new[-1][i] = u_new[-2][i]

    # 非负保护
    for j in range(ny):
        for i in range(nx):
            if u_new[j][i] < 0.0:
                u_new[j][i] = 0.0

    return u_new


# ============================================================
# 完整时间推进
# ============================================================
def run_deposition_simulation_1d(nx, Lx, nt, T_total, kappa_0,
                                 beam_params, fd_order=2):
    """
    运行一维能量沉积模拟.

    参数:
        nx       : 网格点数
        Lx       : 域长度
        nt       : 时间步数
        T_total  : 总时间
        kappa_0  : 热导率系数
        beam_params: 束源参数字典
        fd_order : FD 阶数
    返回:
        dict: 包含 'x', 't', 'u_history', 'total_energy'
    """
    dx = Lx / (nx - 1)
    dt = T_total / max(nt - 1, 1)

    # 网格
    x = [i * dx for i in range(nx)]

    # 初始条件: 零
    u = [0.0] * nx

    # 束参数
    x0 = beam_params.get("x0", 0.3 * Lx)
    r_beam = beam_params.get("r_beam", 10.0e-6)
    tau = beam_params.get("tau_pulse", 10.0e-12)
    E_total = beam_params.get("total_energy", 1.0)
    t_start = beam_params.get("t_start", 0.0)

    # 时间推进
    u_history = [list(u)]
    t_vals = [0.0]
    total_e = [sum(u) * dx]

    for n in range(1, nt):
        t = n * dt
        # 源项
        source = [gaussian_beam_source(x[i], 0.0, t, x0, 0.0,
                                       r_beam, tau, E_total, t_start)
                  for i in range(nx)]
        # FTCS 步进
        u = ftcs_step_1d(u, dx, dt, kappa_0, source, fd_order)
        u_history.append(list(u))
        t_vals.append(t)
        total_e.append(sum(u) * dx)

    return {
        "x": x,
        "t": t_vals,
        "u_history": u_history,
        "total_energy": total_e,
        "dx": dx,
        "dt": dt,
        "nx": nx,
        "nt": nt,
        "fd_order": fd_order,
    }


def run_deposition_simulation_2d(nx, ny, Lx, Ly, nt, T_total, kappa_0,
                                 beam_params, fd_order=2):
    """
    运行二维能量沉积模拟 (小规模可复现).
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T_total / max(nt - 1, 1)

    x = [i * dx for i in range(nx)]
    y = [j * dy for j in range(ny)]

    # 初始条件
    u = [[0.0] * nx for _ in range(ny)]

    x0 = beam_params.get("x0", 0.3 * Lx)
    y0 = beam_params.get("y0", 0.5 * Ly)
    r_beam = beam_params.get("r_beam", 10.0e-6)
    tau = beam_params.get("tau_pulse", 10.0e-12)
    E_total = beam_params.get("total_energy", 1.0)
    t_start = beam_params.get("t_start", 0.0)

    u_snapshots = []
    t_vals = []
    total_e = []
    snapshot_interval = max(1, nt // 5)

    for n in range(nt):
        t = n * dt
        if n % snapshot_interval == 0 or n == nt - 1:
            u_snapshots.append([row[:] for row in u])
            t_vals.append(t)
            total_e.append(sum(sum(row) for row in u) * dx * dy)

        if n < nt - 1:
            source = [[gaussian_beam_source(x[i], y[j], t, x0, y0,
                                            r_beam, tau, E_total, t_start)
                       for i in range(nx)] for j in range(ny)]
            u = ftcs_step_2d(u, dx, dy, dt, kappa_0, source, fd_order)

    return {
        "x": x,
        "y": y,
        "t": t_vals,
        "u_snapshots": u_snapshots,
        "total_energy": total_e,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "fd_order": fd_order,
    }


def print_simulation_summary(result):
    """打印模拟摘要."""
    print("\n" + "=" * 72)
    print("能量沉积 PDE 求解结果 (FTCS)")
    print("=" * 72)
    print("  空间网格: {} × {}".format(
        result.get("nx", "?"),
        result.get("ny", 1)))
    print("  FD 阶数 : {}".format(result["fd_order"]))
    print("  时间步数: {}".format(result["nt"]))
    print("  Δx      : {:.4e} m".format(result["dx"]))
    if "dy" in result:
        print("  Δy      : {:.4e} m".format(result["dy"]))
    print("  Δt      : {:.4e} s".format(result["dt"]))
    if result["total_energy"]:
        print("  初始总能量: {:.4e} J".format(result["total_energy"][0]))
        print("  最终总能量: {:.4e} J".format(result["total_energy"][-1]))
        print("  能量守恒比: {:.6f}".format(
            result["total_energy"][-1] / max(result["total_energy"][0], 1.0e-30)
            if result["total_energy"][0] > 0 else 0.0))
    print("=" * 72)
