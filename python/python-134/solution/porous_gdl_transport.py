#!/usr/bin/env python3
"""
porous_gdl_transport.py
气体扩散层多孔介质传质模块（源自 porous_medium_exact 项目）

将多孔介质方程（非线性扩散 PDE）应用于 PEMFC 气体扩散层（GDL）中的
液态水饱和传输：
    ∂s/∂t = ∇·[ D(s) · ∇s ]  +  S_w

其中 D(s) 为饱和度依赖的毛细扩散系数，具有强非线性：
    D(s) = D_0 · s^α · (1-s)^β · |dP_c/ds| / μ_w

采用 Barenblatt 自相似解思想进行数值求解与验证。
"""

import numpy as np


# ---------------------------------------------------------------------------
# porous_medium_exact 迁移：Barenblatt 自相似解
# ---------------------------------------------------------------------------

def porous_medium_exact(x, t, m, params):
    """
    多孔介质方程 ∂u/∂t = ∇²(u^m) 的 Barenblatt 精确自相似解。
    对应原项目 porous_medium_exact.m。

    解的形式（一维）：
        u(x,t) = t^{-α} · max( 0,  C - γ·(x·t^{-β})² )^{1/(m-1)}

    参数关系：
        α = 1 / (m + 1)
        β = 1 / (2·(m + 1))
        γ = (m - 1) / (2·m·(m + 1))

    在 GDL 水传输中，m 映射为非线性扩散指数（m > 1）。
    """
    x = np.asarray(x, dtype=float)
    t = float(t)
    if t <= 0:
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    alpha = 1.0 / (m + 1.0)
    beta = 1.0 / (2.0 * (m + 1.0))
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))
    C = 1.0  # 归一化常数

    xi = x * t ** (-beta)
    factor = C - gamma * xi ** 2

    u = np.where(factor > 0, t ** (-alpha) * factor ** (1.0 / (m - 1.0)), 0.0)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)

    mask = factor > 1e-12
    if np.any(mask):
        fm = factor[mask]
        u_m = u[mask]
        ut[mask] = (-alpha * u_m / t +
                    t ** (-alpha) * (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0)
                    * (-gamma) * 2.0 * xi[mask] * (-beta) * xi[mask] / t)
        ux[mask] = (t ** (-alpha) * (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0)
                    * (-2.0 * gamma * xi[mask]) * t ** (-beta))
        # 二阶导简化
        uxx[mask] = t ** (-alpha - 2.0 * beta) * (
            (1.0 / (m - 1.0)) * (1.0 / (m - 1.0) - 1.0) * fm ** (1.0 / (m - 1.0) - 2.0)
            * (2.0 * gamma * xi[mask]) ** 2
            + (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0) * (-2.0 * gamma)
        )

    return u, ut, ux, uxx


def porous_medium_residual(x, t, m, params):
    """
    计算多孔介质方程残差 r = u_t - (u^m)_xx，对应原项目 porous_medium_residual.m。
    """
    u, ut, ux, uxx = porous_medium_exact(x, t, m, params)
    # (u^m)_xx = m·(m-1)·u^{m-2}·ux² + m·u^{m-1}·uxx
    um = u ** m
    umx = m * u ** (m - 1) * ux
    umxx = m * (m - 1) * u ** (m - 2) * ux ** 2 + m * u ** (m - 1) * uxx
    r = ut - umxx
    return r


# ---------------------------------------------------------------------------
# GDL 饱和度依赖的毛细扩散系数
# ---------------------------------------------------------------------------

def capillary_diffusivity(s, params):
    """
    计算 GDL 中液态水饱和度的毛细扩散系数 D_cap(s)。

    基于 Leverett J-function 与毛细压力模型：
        P_c(s) = σ·cos(θ)·√(ε/k) · J(s)
        J(s) = 1.417·(1-s) - 2.120·(1-s)² + 1.263·(1-s)³   [Leverett]

    则 D_cap(s) = (K·k_rl(s)/μ_l) · |dP_c/ds|
    """
    s = np.clip(np.asarray(s, dtype=float), 1e-4, 1.0 - 1e-4)

    # 相对渗透率（Corey 模型）
    k_rl = s ** 3.0

    # Leverett J-function 导数
    u = 1.0 - s
    dJ_ds = -1.417 + 4.240 * u - 3.789 * u ** 2

    # 物理参数
    K_abs = 1.0e-12  # 绝对渗透率 [m²]
    mu_l = 3.5e-4    # 液态水粘度 [Pa·s]
    sigma = 0.062    # 表面张力 [N/m]
    theta = 110.0 * np.pi / 180.0  # 接触角 [rad]
    eps = params['epsilon_gdl']

    # 毛细压力梯度
    dPc_ds = sigma * np.cos(theta) * np.sqrt(eps / K_abs) * dJ_ds

    D_cap = (K_abs * k_rl / mu_l) * np.abs(dPc_ds)
    return np.clip(D_cap, 1e-15, 1.0)


def solve_gdl_saturation(params, s_init=None):
    """
    使用有限体积法求解 GDL 中液态水饱和度 s(z,t) 的演化。

    方程：∂s/∂t = ∂/∂z[ D_cap(s) · ∂s/∂z ] + S_w
    其中 S_w 为从催化层渗透进入 GDL 的水源项。

    采用自适应时间步长，保证显式稳定性：dt <= dz² / (2·D_max)
    """
    Nz = max(21, params['Nx'] // 4)
    L_gdl = params['t_gdl']

    dz = L_gdl / (Nz - 1)
    z = np.linspace(0.0, L_gdl, Nz)

    if s_init is None:
        s = np.full(Nz, 0.1)  # 初始低饱和度
    else:
        s = np.clip(s_init, 0.0, 1.0)

    # 自适应时间步长
    D_max_est = 1.0e-6  # D_cap 的上界估计
    dt_stable = 0.4 * dz ** 2 / D_max_est
    t_final = params['t_final']
    Nt = max(int(t_final / dt_stable) + 1, 500)
    dt = t_final / Nt

    # 时间推进（显式，稳定时间步长）
    for n in range(Nt):
        s_old = s.copy()
        D = capillary_diffusivity(s_old, params)

        s_new = s_old.copy()
        for i in range(1, Nz - 1):
            D_e = 0.5 * (D[i] + D[i + 1])
            D_w = 0.5 * (D[i] + D[i - 1])

            flux_e = D_e * (s_old[i + 1] - s_old[i]) / dz
            flux_w = D_w * (s_old[i] - s_old[i - 1]) / dz

            # 源项：催化层产水渗透到 GDL（阴极侧 z=L_gdl 处最强）
            S_w = 0.5 * (z[i] / L_gdl) ** 2 * 1e-4

            s_new[i] = s_old[i] + dt * ((flux_e - flux_w) / dz + S_w)
            s_new[i] = np.clip(s_new[i], 0.0, 1.0)

        # 边界条件
        s_new[0] = 0.05  # 流道侧：低饱和度
        s_new[-1] = 0.6  # 催化层侧：较高饱和度

        s = s_new

    return s, z


def gdl_saturation_profile_barenblatt(z, t, params):
    """
    使用 Barenblatt 自相似解生成 GDL 饱和度参考剖面（用于验证）。
    将 u 映射为 s，通过归一化保证 s ∈ [0,1]。
    """
    m = params['m_porous']
    u, ut, ux, uxx = porous_medium_exact(z, t, m, params)
    # 归一化到 [0, 1]
    u_max = np.max(u) if np.max(u) > 1e-12 else 1.0
    s = np.clip(u / u_max, 0.0, 1.0)
    return s


if __name__ == '__main__':
    p = {'epsilon_gdl': 0.4, 't_gdl': 200e-6, 'Nx': 81, 't_final': 2.0, 'm_porous': 2.5}
    s, z = solve_gdl_saturation(p)
    print("s range:", s.min(), s.max())
