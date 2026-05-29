"""
gauge_dg_update.py
==================
基于间断 Galerkin（DG）谱方法的规范场 fictitious-time 演化。

原项目映射：274_dg1d_maxwell（1D DG Maxwell 方程求解器）

物理背景
--------
在格点 QCD 中，为了改善算符与强子基态的重叠，常使用“冷却”（cooling）
或梯度流（gradient flow）技术。本模块将 1D DG 框架推广到格点规范场的
 fictitious-time 演化：

    ∂_t U_μ(x, t) = -g_0^2 ∂_{U_μ} S_G[U(t)]

其中 S_G 为 Wilson 作用量。在 SU(2) 情形下，力项可由 staples 计算：

    F_μ(x) = Σ_{ν≠μ} [ staples_{μν}(x) - staples†_{μν}(x) ]

DG 离散化：将每个格点链路视为谱元节点，利用 Lagrange 插值基函数构建
高阶通量（upwind flux），实现高空间精度。

核心公式
--------
DG 弱形式：对每个单元 K，找 U_h ∈ V_h 使得

    ∫_K ∂_t U_h · v dx - ∫_K F(U_h) · ∂_x v dx + ∫_{∂K} F̂ · v ds = 0,  ∀v ∈ V_h

其中数值通量 F̂ 采用 upwind 格式：

    F̂ = (Z^- F^+ + Z^+ F^-) / (Z^- + Z^+)

这里 Z = sqrt(μ/ε) 类比为规范场的“阻抗”。
时间推进采用低存储五阶 Runge-Kutta（LSERK）：

    u^{(i)} = a_i u^{(i-1)} + Δt L(u^{(i-1)})
    u^{n+1} = u^{n} + b_i u^{(i)}
"""

import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger, su2_trace, su2_stereographic_project, su2_stereographic_inverse


# 五阶低存储 RK 系数 (Carpenter & Kennedy, 1994)
_RK4A = np.array([0.0,
                  -567301805773.0 / 1357537059087.0,
                  -2404267990393.0 / 2016746695238.0,
                  -3550918686646.0 / 2091501179385.0,
                  -1275806237668.0 / 842570457699.0])
_RK4B = np.array([1432997174477.0 / 9575080441755.0,
                  5161836677717.0 / 13612068292357.0,
                  1720146321549.0 / 2090206949498.0,
                  3134564353537.0 / 4481467310338.0,
                  2277821191437.0 / 14882151754819.0])
_RK4C = np.array([0.0,
                  1432997174477.0 / 9575080441755.0,
                  2526269341429.0 / 6820363962896.0,
                  2006345519317.0 / 3224310063776.0,
                  2802321613138.0 / 2924317926251.0])


def compute_staple(gauge: GaugeConfig, x: np.ndarray, mu: int) -> np.ndarray:
    """
    计算方向 μ 上点 x 的 staple 和。

    staple_{μ}(x) = Σ_{ν≠μ} [
        U_ν(x) U_μ(x+ν̂) U_ν†(x+μ̂)
      + U_ν†(x-ν̂) U_μ(x-ν̂) U_ν(x-ν̂+μ̂)
    ]
    """
    lat = gauge.lat
    staple = np.zeros((2, 2), dtype=complex)
    for nu in range(4):
        if nu == mu:
            continue
        # 正向 staple
        s1 = gauge.get_link(nu, x)
        s2 = gauge.get_link(mu, lat.neighbor(x, nu, 1))
        s3 = su2_dagger(gauge.get_link(nu, lat.neighbor(x, mu, 1)))
        staple += s1 @ s2 @ s3
        # 反向 staple
        r1 = su2_dagger(gauge.get_link(nu, lat.neighbor(x, nu, -1)))
        r2 = gauge.get_link(mu, lat.neighbor(x, nu, -1))
        r3 = gauge.get_link(nu, lat.neighbor(lat.neighbor(x, nu, -1), mu, 1))
        staple += r1 @ r2 @ r3
    return staple


def compute_force(gauge: GaugeConfig, x: np.ndarray, mu: int) -> np.ndarray:
    """
    计算 Wilson 作用量对 U_μ(x) 的力（投影到 su(2) 代数）。

    力定义为：
        F_μ(x) = ( staple_μ(x) U_μ†(x) - U_μ(x) staple_μ†(x) ) / 2
    这是反厄米的，对应 su(2) 代数元。
    """
    u = gauge.get_link(mu, x)
    st = compute_staple(gauge, x, mu)
    # 投影到 su(2)：取反厄米部分
    tmp = st @ su2_dagger(u)
    force = 0.5 * (tmp - su2_dagger(tmp))
    return force


def dg_gauge_rhs(gauge: GaugeConfig, beta: float = 2.4) -> np.ndarray:
    """
    计算 DG 规范场演化的右端项（fictitious-time derivative）。

    对所有链路计算力项，并以 upwind flux 方式组合相邻格点的影响。
    在简化模型中，我们将每个空间方向视为独立的 1D 谱元链，
    利用 DG 通量计算空间耦合。

    Returns
    -------
    rhs : np.ndarray
        与 gauge.U 同形状的右端项（以投影坐标表示）。
    """
    lat = gauge.lat
    rhs = np.zeros((4, *lat.shape, 3), dtype=float)

    for mu in range(4):
        for idx in range(lat.vol):
            x = lat.index_to_site(idx)
            force = compute_force(gauge, x, mu)
            # 将 su(2) 反厄米矩阵映射到 R^3 力向量
            # force = i f_k σ_k  =>  f_k = -Im(force_{12}) 等
            fvec = np.array([force[0, 1].real,
                             force[0, 1].imag,
                             force[0, 0].imag])
            # DG upwind flux：考虑邻居的力差
            x_plus = lat.neighbor(x, mu, 1)
            force_plus = compute_force(gauge, x_plus, mu)
            fvec_plus = np.array([force_plus[0, 1].real,
                                  force_plus[0, 1].imag,
                                  force_plus[0, 0].imag])
            # upwind 通量（沿正方向传播）
            flux = 0.5 * (fvec_plus - fvec)
            rhs[(mu, *x)] = (beta / 6.0) * fvec + flux

    return rhs


def dg_gauge_evolve(gauge: GaugeConfig, beta: float = 2.4,
                    final_time: float = 1.0, cfl: float = 0.5) -> GaugeConfig:
    """
    使用 DG + LSERK 推进规范场到 fictitious time = final_time。

    Parameters
    ----------
    gauge : GaugeConfig
        初始规范场构型。
    beta : float
        逆耦合常数。
    final_time : float
        演化终止的 fictitious time。
    cfl : float
        CFL 数。

    Returns
    -------
    gauge : GaugeConfig
        演化后的规范场构型。
    """
    lat = gauge.lat
    # 估计时间步长
    dt = cfl * 0.5
    nsteps = max(1, int(np.ceil(final_time / dt)))
    dt = final_time / nsteps

    # 存储残差（投影坐标）
    res = np.zeros((4, *lat.shape, 3), dtype=float)

    for _ in range(nsteps):
        for intrk in range(5):
            rhs = dg_gauge_rhs(gauge, beta)
            res = _RK4A[intrk] * res + dt * rhs
            # 更新规范场：将投影坐标变化映射回 SU(2)
            for mu in range(4):
                for idx in range(lat.vol):
                    x = lat.index_to_site(idx)
                    dq = _RK4B[intrk] * res[(mu, *x)]
                    # 限制步长以保持稳定性
                    norm_dq = np.linalg.norm(dq)
                    if norm_dq > 0.5:
                        dq = dq * (0.5 / norm_dq)
                    u_old = gauge.get_link(mu, x)
                    q_old = su2_stereographic_project(u_old)
                    q_new = q_old + dq
                    # 限制在投影域内
                    norm_q = np.linalg.norm(q_new)
                    if norm_q > 10.0:
                        q_new = q_new * (10.0 / norm_q)
                    u_new = su2_stereographic_inverse(q_new)
                    gauge.set_link(mu, x, u_new)

    return gauge
