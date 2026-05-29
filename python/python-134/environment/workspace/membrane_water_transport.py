#!/usr/bin/env python3
"""
membrane_water_transport.py
膜内水传输模块（源自 standing_wave_exact + stiff_exact + rk1_implicit 项目）

融合波动方程精确解、刚性 ODE 精确解与隐式向后 Euler 积分，
求解 PEM 膜内一维水含量 λ(z,t) 的时空演化。

控制方程（一维对流-扩散-反应耦合）：
    ∂λ/∂t = ∂/∂z[ D_λ(λ,T) · ∂λ/∂z ] + n_d(λ)·(F/ε)·j(z)/F  -  S_vap(λ,T)

其中：
    D_λ(λ,T) = D_λ_max · f(λ) · exp[ -E_d / (R·T) ]
    f(λ) = 10^{-6} · exp[ 2416·(1/303 - 1/T) ] · (λ/22)       [m²/s]
    n_d(λ) = 2.5·λ/22    电渗拖拽系数
    S_vap = k_vap·(p_sat(T) - p_vap)·(λ - λ_eq)   蒸发/冷凝源项

边界条件：
    z = 0 (阳极侧):  λ = λ_a   (由阳极相对湿度决定)
    z = t_m (阴极侧): λ = λ_c   (由阴极相对湿度与生成水决定)
"""

import numpy as np


# ---------------------------------------------------------------------------
# standing_wave_exact 迁移：水含量波动传播的解析验证解
# ---------------------------------------------------------------------------

def water_content_wave_exact(z, t, params):
    """
    一维波动方程的精确解析解（源自 standing_wave_exact.m）。
    用于验证数值传播特性：
        ∂²u/∂t² = c_w² · ∂²u/∂z²
    精确解：u(z,t) = sin(π·z/L_m) · cos(c_w·π·t/L_m)

    这里将物理意义映射为水含量的小扰动传播。
    """
    L_m = params['t_membrane']
    c_w = 1.0e-3  # 水扰动波速 [m/s]

    z = np.asarray(z, dtype=float)
    t = float(t)
    u = np.sin(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)
    ut = -np.sin(np.pi * z / L_m) * (c_w * np.pi / L_m) * np.sin(c_w * np.pi * t / L_m)
    utt = -np.sin(np.pi * z / L_m) * (c_w * np.pi / L_m) ** 2 * np.cos(c_w * np.pi * t / L_m)
    uz = (np.pi / L_m) * np.cos(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)
    uzz = -(np.pi / L_m) ** 2 * np.sin(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)

    # 边界裁剪
    z = np.clip(z, 0.0, L_m)
    return u, ut, utt, uz, uzz


def water_transport_residual(z, t, params):
    """
    计算波动方程残差 r = u_tt - c_w² * u_zz，对应原项目 standing_wave_residual.m。
    """
    u, ut, utt, uz, uzz = water_content_wave_exact(z, t, params)
    c_w = 1.0e-3
    r = utt - c_w ** 2 * uzz
    return r


# ---------------------------------------------------------------------------
# stiff_exact 迁移：刚性瞬态电化学耦合的精确验证解
# ---------------------------------------------------------------------------

def water_content_stiff_ode(t, lambda0, params):
    """
    刚性 ODE 的精确解（源自 stiff_exact.m）。
    描述膜-电极界面附近水含量的快速瞬态：
        dλ/dt = Λ·[cos(t) - λ]
    精确解：
        λ(t) = Λ·sin(t)/(Λ²+1) + Λ²·cos(t)/(Λ²+1) + μ·exp(-Λ·t)
    其中 μ = λ0 - Λ²/(Λ²+1)。

    Λ 映射为电化学反应速率常数（大数 → 刚性）。
    """
    Lambda = 50.0  # 刚性参数
    mu = lambda0 - Lambda ** 2 / (Lambda ** 2 + 1.0)

    t = np.asarray(t, dtype=float)
    lam = (Lambda * np.sin(t) / (Lambda ** 2 + 1.0) +
           Lambda ** 2 * np.cos(t) / (Lambda ** 2 + 1.0) +
           mu * np.exp(-Lambda * t))
    # 水含量必须在合理范围
    lam = np.clip(lam, 0.0, 22.0)
    return lam


def water_content_stiff_deriv(t, lam, params):
    """
    刚性 ODE 的右端项（源自 stiff_deriv.m）。
    """
    Lambda = 50.0
    dlam = Lambda * (np.cos(t) - lam)
    return dlam


# ---------------------------------------------------------------------------
# rk1_implicit 迁移：隐式向后 Euler 时间积分
# ---------------------------------------------------------------------------

def water_diffusivity(lambda_w, T):
    """
    膜内水扩散系数，Springer 经验公式：
        D_λ = 10^{-6} · exp[2416·(1/303 - 1/T)] · f(λ)
    其中 f(λ) 为分段函数。
    """
    # TODO: Hole_1 待修复 —— 实现 Springer 水扩散系数公式
    # 需根据水含量 lambda_w 和温度 T 计算扩散系数 D_λ。
    # 提示：包含温度依赖的基值 D_base 和水含量分段函数 f(λ)。
    pass


def electro_osmotic_drag_coeff(lambda_w):
    """
    电渗拖拽系数：n_d = 2.5·λ/22 （Springer 模型）。
    """
    return 2.5 * np.clip(lambda_w, 0.0, 22.0) / 22.0


def vapor_source(lambda_w, T, params):
    """
    蒸发/冷凝源项。
    """
    lambda_eq = params['lambda_eq']
    k_vap = 1.0e-2
    return k_vap * (lambda_w - lambda_eq)


def solve_membrane_water_transport(params, j_profile=None):
    """
    使用隐式向后 Euler（源自 rk1_implicit.m）求解膜内水传输 PDE。

    空间：中心差分（二阶）
    时间：隐式一阶（A-stable，适合 stiff 问题）
    """
    Nz = max(21, params['Nx'] // 4)
    Nt = params['Nt']
    t_final = params['t_final']
    T = params['T']
    F = params['F']
    L_m = params['t_membrane']

    dz = L_m / (Nz - 1)
    dt = t_final / Nt

    z = np.linspace(0.0, L_m, Nz)
    t = np.linspace(0.0, t_final, Nt + 1)

    # 初始条件：均匀水含量
    lambda_w = np.full(Nz, params['lambda_eq'])

    # 电流密度分布（若未提供，用线性分布）
    if j_profile is None:
        j_profile = np.linspace(5000.0, 10000.0, Nz)  # A/m²
    j_profile = np.clip(j_profile, 0.0, 50000.0)

    # 存储结果
    lambda_history = np.zeros((Nt + 1, Nz))
    lambda_history[0, :] = lambda_w

    # 隐式时间推进
    for n in range(Nt):
        lam_old = lambda_history[n, :].copy()

        # 使用简单迭代求解非线性隐式系统（Picard 迭代）
        lam_new = lam_old.copy()
        for _ in range(5):
            # 计算变系数
            D = water_diffusivity(lam_new, T)
            n_d = electro_osmotic_drag_coeff(lam_new)

            # 构造三对角矩阵
            a = np.zeros(Nz)
            b = np.zeros(Nz)
            c = np.zeros(Nz)
            rhs = np.zeros(Nz)

            for i in range(Nz):
                if i == 0:
                    # Dirichlet 边界：阳极侧水含量
                    b[i] = 1.0
                    rhs[i] = 3.0  # λ_a ≈ 3（低加湿）
                elif i == Nz - 1:
                    # Dirichlet 边界：阴极侧水含量（较高）
                    b[i] = 1.0
                    rhs[i] = 14.0  # λ_c
                else:
                    D_e = 0.5 * (D[i] + D[i + 1])
                    D_w = 0.5 * (D[i] + D[i - 1])

                    a[i] = -dt * D_w / dz ** 2
                    c[i] = -dt * D_e / dz ** 2
                    b[i] = 1.0 + dt * (D_w + D_e) / dz ** 2 + dt * 0.01
                    rhs[i] = lam_old[i] + dt * (
                        n_d[i] * j_profile[i] / F / 1e4 - vapor_source(lam_new[i], T, params)
                    )

            # 三对角求解 (Thomas algorithm)
            lam_new = thomas_algorithm(a, b, c, rhs)
            lam_new = np.clip(lam_new, 0.0, 22.0)

        lambda_history[n + 1, :] = lam_new

    # 返回最终稳态剖面与时间网格
    return lambda_history[-1, :], t


def thomas_algorithm(a, b, c, d):
    """
    Thomas 算法求解三对角线性系统 A·x = d。
    A 的形式：
        b[0]·x[0] + c[0]·x[1] = d[0]
        a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]
        a[n-1]·x[n-2] + b[n-1]·x[n-1] = d[n-1]
    """
    n = d.size
    cp = c.copy()
    dp = d.copy()
    bp = b.copy()

    # 前向消元
    for i in range(1, n):
        w = a[i] / bp[i - 1]
        bp[i] -= w * cp[i - 1]
        dp[i] -= w * dp[i - 1]

    # 回代
    x = np.zeros(n)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


if __name__ == '__main__':
    p = {'t_membrane': 50e-6, 'T': 353.15, 'F': 96485.0,
         'lambda_eq': 14.0, 'Nx': 81, 'Nt': 200, 't_final': 5.0}
    lam, t = solve_membrane_water_transport(p)
    print("lambda range:", lam.min(), lam.max())
