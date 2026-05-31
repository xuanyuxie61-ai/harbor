"""
脑血流动力学 — 脑组织氧扩散与反应模块

基于 fisher_pde_ftcs（KPP-Fisher 反应扩散方程），模拟脑实质中氧气从毛细血管
向周围组织的扩散与消耗过程。

科学背景:
- 脑组织氧合遵循反应-扩散方程:
    ∂C/∂t = D ∇²C + λ C (1 - C/C_max) - k_met C
  其中 D 为氧扩散系数，λ 为毛细血管供氧速率，k_met 为代谢消耗速率。
- Krogh 圆柱模型：以毛细血管为中心，半径 R_t 的圆柱区域内氧向组织扩散。
- Michaelis-Menten 型氧消耗:
    V = V_max * C / (K_m + C)
"""

import numpy as np


def oxygen_diffusion_ftcs_1d(C0, nx, nt, t_max, D, lam, k_met, C_max,
                              bc_left_type='dirichlet', bc_left_val=1.0,
                              bc_right_type='neumann', bc_right_val=0.0):
    """
    一维 FTCS 格式求解组织氧扩散-反应方程:
        C_t = D C_xx + λ C(1 - C/C_max) - k_met C

    参数:
        C0: 初始浓度分布函数，输入 x 返回浓度
        nx: 空间节点数
        nt: 时间步数
        t_max: 最大时间
        D: 扩散系数 [mm²/s]
        lam: 供氧反应速率 [1/s]
        k_met: 代谢消耗速率 [1/s]
        C_max: 饱和氧浓度 [mM]

    返回:
        C: (nt+1, nx) 浓度矩阵
        x: 空间坐标
        t: 时间坐标
    """
    xmin = 0.0
    xmax = 1.0  # 归一化 Krogh 圆柱半径
    x = np.linspace(xmin, xmax, nx)
    dx = (xmax - xmin) / (nx - 1)
    dt = t_max / nt

    # CFL 稳定性检查
    if D * dt / dx ** 2 > 0.5:
        # 自适应调整时间步
        dt = 0.45 * dx ** 2 / D
        nt = int(np.ceil(t_max / dt))
        dt = t_max / nt

    C = np.zeros((nt + 1, nx))
    c = C0(x).astype(float)
    C[0, :] = c

    # 空间索引用于中心差分
    Im1 = np.array([0] + list(range(nx - 2)) + [nx - 2])
    I = np.arange(nx)
    Ip1 = np.array([1] + list(range(1, nx - 1)) + [nx - 1])

    for j in range(1, nt + 1):
        d2c_dx2 = (c[Ip1] - 2.0 * c[I] + c[Im1]) / dx ** 2
        reaction = lam * c * (1.0 - c / C_max) - k_met * c
        c_new = c + dt * (D * d2c_dx2 + reaction)

        # 边界条件
        if bc_left_type == 'dirichlet':
            c_new[0] = bc_left_val
        elif bc_left_type == 'neumann':
            c_new[0] = c_new[1] - bc_left_val * dx

        if bc_right_type == 'dirichlet':
            c_new[-1] = bc_right_val
        elif bc_right_type == 'neumann':
            c_new[-1] = c_new[-2] + bc_right_val * dx

        c = np.clip(c_new, 0.0, C_max)
        C[j, :] = c

    t = np.linspace(0, t_max, nt + 1)
    return C, x, t


def oxygen_diffusion_2d_radial(C0, nr, nt, t_max, D, lam, k_met, C_max,
                                R_tissue=0.05, R_cap=0.003):
    """
    二维径向 FTCS 求解组织氧扩散:
        C_t = D (C_rr + (1/r) C_r) + λ C(1 - C/C_max) - k_met C

    采用极坐标径向离散，模拟 Krogh 圆柱模型。
    R_tissue: 组织外半径 [mm]
    R_cap: 毛细血管半径 [mm]
    """
    r = np.linspace(R_cap, R_tissue, nr)
    dr = (R_tissue - R_cap) / (nr - 1)
    dt = t_max / nt

    if D * dt / dr ** 2 > 0.25:
        dt = 0.2 * dr ** 2 / D
        nt = int(np.ceil(t_max / dt))
        dt = t_max / nt

    C = np.zeros((nt + 1, nr))
    c = C0(r).astype(float)
    C[0, :] = c

    for j in range(1, nt + 1):
        c_new = np.zeros_like(c)
        for i in range(1, nr - 1):
            r_i = max(r[i], 1e-10)
            laplacian_r = (c[i + 1] - 2.0 * c[i] + c[i - 1]) / dr ** 2 + \
                          (c[i + 1] - c[i - 1]) / (2.0 * dr * r_i)
            reaction = lam * c[i] * (1.0 - c[i] / C_max) - k_met * c[i]
            c_new[i] = c[i] + dt * (D * laplacian_r + reaction)

        # 内边界（毛细血管壁）: Dirichlet，血氧饱和
        c_new[0] = C_max * 0.95
        # 外边界（组织远端）: Neumann，无通量
        c_new[-1] = c_new[-2]

        c = np.clip(c_new, 0.0, C_max)
        C[j, :] = c

    t = np.linspace(0, t_max, nt + 1)
    return C, r, t


def michaelis_menten_oxygen_consumption(C, V_max, K_m):
    """
    Michaelis-Menten 型氧消耗速率:
        V(C) = V_max * C / (K_m + C)
    """
    C = np.asarray(C, dtype=float)
    C_safe = np.where(C < 0, 0.0, C)
    return V_max * C_safe / (K_m + C_safe + 1e-14)


def krogh_oxygen_tension(r, R_t, R_c, P_c, P_tissue, D_t, M0):
    """
    Krogh 圆柱模型解析解（稳态氧张力分布）:
        P(r) = P_c - (M0 / (4 D_t)) * (r^2 - R_c^2) + (M0 R_t^2 / (2 D_t)) * ln(r / R_c)

    参数:
        r: 距毛细血管中心的径向距离 [mm]
        R_t: 组织圆柱半径 [mm]
        R_c: 毛细血管半径 [mm]
        P_c: 毛细血管壁氧分压 [mmHg]
        P_tissue: 远端组织氧分压 [mmHg]
        D_t: 组织氧扩散系数 [mm²/s]
        M0: 基础氧代谢率 [mL O2/(mL tissue·s)]
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.where(r < R_c, R_c, r)
    term1 = (M0 / (4.0 * D_t)) * (r_safe ** 2 - R_c ** 2)
    term2 = (M0 * R_t ** 2 / (2.0 * D_t)) * np.log(r_safe / R_c)
    P = P_c - term1 + term2
    return P
