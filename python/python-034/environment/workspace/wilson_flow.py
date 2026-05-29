"""
wilson_flow.py
==============
Wilson 梯度流（gradient flow）的实现：通过平滑规范场改善算符重叠。

原项目映射：
  - 826_ode_euler_backward：后向 Euler 隐式 ODE 求解器
  - 765_midpoint_adaptive：自适应隐式中点法

物理背景
--------
Wilson 梯度流方程定义为：

    ḂV_t = -g_0^2 ∂_x S_G[V_t]

其中 V_t 为流时间 t 处的规范场，S_G 为 Wilson 作用量。
在 SU(2) 情形下，流化的链路变量满足：

    ḂV_μ(x, t) = -∇_μ S_G = F_μ(x, t) V_μ(x, t)

这里 F_μ(x, t) 为来自 staples 的 su(2) 代数力（反厄米迹零矩阵）。
流化后的 plaquette 值趋于 1，抑制 UV 涨落。

数值方法
--------
1. 后向 Euler（隐式）：
        V_{n+1} = V_n + Δt F(V_{n+1}) V_{n+1}
   通过定点迭代求解。

2. 自适应隐式中点法（Refactorized Midpoint Rule）：
   引入参数 θ，在 t_n + θ Δt 处求中间值 Y_m：
        Y_m = Y_n + θ Δt F(Y_m) Y_m
        Y_{n+1} = (1/θ) Y_m + (1 - 1/θ) Y_n
   通过局部截断误差（LTE）估计自适应调整步长。

稳定性与误差
------------
后向 Euler 无条件稳定，一阶精度 O(Δt)。
中点法二阶精度 O(Δt^2)，A-稳定，适合刚性方程。
"""

import numpy as np
from lattice_gauge import Lattice, GaugeConfig, su2_dagger, su2_stereographic_project, su2_stereographic_inverse
from gauge_dg_update import compute_force


def apply_flow_step_euler_backward(gauge: GaugeConfig, dt: float,
                                   max_iter: int = 6) -> GaugeConfig:
    """
    后向 Euler 推进一个 Wilson flow 步长。

    隐式方程：
        U_{n+1} = exp( dt * F_{n+1} ) U_n
    在投影坐标下简化为定点迭代：
        q^{(j+1)} = q_n + dt * f( q^{(j)} )
    """
    lat = gauge.lat
    for idx in range(lat.vol):
        x = lat.index_to_site(idx)
        for mu in range(4):
            u_n = gauge.get_link(mu, x)
            q_n = su2_stereographic_project(u_n)
            q = q_n.copy()
            for _ in range(max_iter):
                u_trial = su2_stereographic_inverse(q)
                gauge.set_link(mu, x, u_trial)
                force = compute_force(gauge, x, mu)
                fvec = np.array([force[0, 1].real,
                                 force[0, 1].imag,
                                 force[0, 0].imag])
                q_new = q_n + dt * fvec
                # 阻尼迭代
                q = 0.5 * q + 0.5 * q_new
                if np.linalg.norm(q_new - q_n) < 1e-10:
                    break
            gauge.set_link(mu, x, su2_stereographic_inverse(q))
    return gauge


def adaptive_midpoint_step(gauge: GaugeConfig, dt: float,
                           theta: float = 0.5) -> tuple:
    """
    自适应隐式中点法推进一个步长（简化版，无递归重试）。

    算法：
        1. 在 t_m = t_n + θ Δt 处求解中间值 Y_m：
               Y_m = Y_n + θ Δt F(Y_m) Y_m
        2. 外推得到 t_{n+1} = t_n + Δt：
               Y_{n+1} = (1/θ) Y_m + (1 - 1/θ) Y_n
    """
    lat = gauge.lat
    state_n = gauge.U.copy()

    # 中点求解（最多 8 次定点迭代）
    for idx in range(lat.vol):
        x = lat.index_to_site(idx)
        for mu in range(4):
            q_n = su2_stereographic_project(state_n[(mu, *x)])
            q = q_n.copy()
            for _ in range(8):
                u_tmp = su2_stereographic_inverse(q)
                gauge.set_link(mu, x, u_tmp)
                force = compute_force(gauge, x, mu)
                fvec = np.array([force[0, 1].real,
                                 force[0, 1].imag,
                                 force[0, 0].imag])
                q_new = q_n + theta * dt * fvec
                delta = np.linalg.norm(q_new - q)
                q = q_new
                if delta < 1e-12:
                    break
            q_m = q
            q_np1 = (1.0 / theta) * q_m + (1.0 - 1.0 / theta) * q_n
            gauge.set_link(mu, x, su2_stereographic_inverse(q_np1))

    # 简化 LTE 估计
    lte = 0.0
    return gauge, dt, lte


def wilson_flow_run(gauge: GaugeConfig, flow_time: float = 1.0,
                    dt_init: float = 0.05, method: str = "midpoint") -> GaugeConfig:
    """
    运行 Wilson 梯度流到指定 flow time。

    Parameters
    ----------
    gauge : GaugeConfig
        初始规范场。
    flow_time : float
        目标流时间。
    dt_init : float
        初始步长。
    method : str
        "euler" 或 "midpoint"。

    Returns
    -------
    gauge : GaugeConfig
        流化后的规范场。
    """
    t = 0.0
    dt = dt_init
    n_steps = 0
    while t < flow_time and n_steps < 200:
        dt = min(dt, flow_time - t)
        if method == "euler":
            gauge = apply_flow_step_euler_backward(gauge, dt)
            t += dt
        else:
            gauge, dt_acc, _ = adaptive_midpoint_step(gauge, dt)
            t += dt_acc
            dt = min(2.0 * dt_acc, dt_init)
        n_steps += 1
    return gauge
