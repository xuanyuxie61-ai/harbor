"""
pde_constraint.py
=================

PDE 约束最优控制模块 (热扩散方程)。

融合种子项目:
  - 269_delsq : 五点差分 Laplacian
  - 244_cvt_1d_lumping : CVT 空间离散化

在航天器最优控制中, PDE 约束表现为:
  1. 热扩散方程 (航天器表面温度分布)
  2. 结构热应力约束
  3. 温度路径约束 (防止过热)

数学公式:
---------
1. 热扩散 PDE:
     rho c_p dT/dt = kappa Laplacian(T) + Q
     即 dT/dt = alpha Laplacian(T) + Q/(rho c_p)
     其中 alpha = kappa / (rho c_p) 为热扩散系数

2. 离散形式 (与 Laplacian 算子耦合):
     dT/dt = alpha * (-Delta_h) T + Q_vec

3. 伴随 PDE (向后积分):
     -dlambda_T/dt = alpha * (-Delta_h) lambda_T + dL/dT
     lambda_T(T) = dPhi/dT(T)

4. 耦合项 (温度影响推力):
     T_max(Theta) = T_max_0 * f(T_max_local)
     其中 T_max_local 为局部最高温度
"""

from __future__ import annotations

import math
from typing import List, Tuple

from laplacian_discretizer import LaplacianDiscretizer, make_rectangular_grid
from scientific_constants import ThermalParameters


# ===========================================================================
# 1. 热扩散 PDE 右端函数
# ===========================================================================
def heat_equation_rhs(
    T_field: List[float],
    disc: LaplacianDiscretizer,
    th: ThermalParameters,
    dx: float,
) -> List[float]:
    """计算热扩散方程右端 dT/dt.

    dT/dt = alpha * Laplacian(T) + Q / (rho c_p)
         = -alpha / dx^2 * (-Delta_h T) + Q / (rho c_p)

    Parameters
    ----------
    T_field : List[float]
        温度场 (长度 = 内部节点数 n).
    disc : LaplacianDiscretizer
        Laplacian 离散算子 (5 点差分, 已除 dx^2).
    th : ThermalParameters
        热参数.
    dx : float
        网格间距 [m].

    Returns
    -------
    List[float]
        dT/dt 向量.
    """
    n = disc.size
    if len(T_field) != n:
        raise ValueError(f"温度场长度 {len(T_field)} != 节点数 {n}")

    # -Delta_h T (disc 已经包含负号)
    lap_T = disc.matvec(T_field)

    alpha = th.diffusivity
    source = th.heat_source / (th.density * th.specific_heat)

    rhs = [0.0] * n
    for i in range(n):
        # disc 的 matvec 返回 -Delta_h T, 故需要乘以 -1 得到 Delta_h T
        rhs[i] = -alpha / (dx * dx) * lap_T[i] + source
    return rhs


# ===========================================================================
# 2. PDE 时间积分 (显式 Euler, 带稳定性检查)
# ===========================================================================
def explicit_heat_step(
    T_field: List[float],
    disc: LaplacianDiscretizer,
    th: ThermalParameters,
    dx: float,
    dt: float,
) -> List[float]:
    """显式 Euler 热扩散步.

    稳定性条件 (CFL):  dt <= dx^2 / (4 alpha)  (2D 5 点模板)
    """
    alpha = th.diffusivity
    cfl_limit = dx * dx / (4.0 * alpha)
    if dt > cfl_limit:
        raise ValueError(
            f"显式格式不稳定: dt = {dt:.3e} > CFL limit = {cfl_limit:.3e}. "
            f"请减小 dt 或增大 dx."
        )

    rhs = heat_equation_rhs(T_field, disc, th, dx)
    return [T + dt * r for T, r in zip(T_field, rhs)]


def crank_nicolson_heat_step(
    T_field: List[float],
    disc: LaplacianDiscretizer,
    th: ThermalParameters,
    dx: float,
    dt: float,
    n_jacobi: int = 10,
) -> List[float]:
    """Crank-Nicolson 热扩散步 (隐式, A-稳定).

    (I - 0.5 dt A) T^{n+1} = (I + 0.5 dt A) T^n + dt source
    其中 A = -alpha / dx^2 * (-Delta_h)

    用 Jacobi 迭代求解隐式系统.
    """
    alpha = th.diffusivity
    n = disc.size

    # 构造 A T (不含源项)
    lap_T = disc.matvec(T_field)
    A_T = [-alpha / (dx * dx) * lt for lt in lap_T]

    # RHS = T^n + 0.5 dt A T^n + dt source
    source = th.heat_source / (th.density * th.specific_heat)
    rhs = [
        T_field[i] + 0.5 * dt * A_T[i] + dt * source
        for i in range(n)
    ]

    # Jacobi 迭代求解 (I - 0.5 dt A) T^{n+1} = rhs
    # 近似: 对角元 ~ 1 + 0.5 dt alpha * 4 / dx^2
    diag_coeff = 1.0 + 0.5 * dt * alpha * 4.0 / (dx * dx)
    T_new = [r / diag_coeff for r in rhs]

    for _ in range(n_jacobi):
        lap_T_new = disc.matvec(T_new)
        T_next = [
            (rhs[i] + 0.5 * dt * alpha / (dx * dx) * lap_T_new[i]) / diag_coeff
            for i in range(n)
        ]
        T_new = T_next

    return T_new


# ===========================================================================
# 3. PDE 约束的最优控制: 温度路径约束处理
# ===========================================================================
def thermal_constraint_violation(
    T_field: List[float],
    T_max_allowed: float,
) -> Tuple[float, int]:
    """计算温度约束违反程度.

    g(T) = max(0, max_i(T_i) - T_max_allowed)

    Returns
    -------
    (violation, max_temp_node) : (float, int)
        违反量和最高温度节点索引.
    """
    if not T_field:
        return 0.0, -1
    T_max = max(T_field)
    violation = max(0.0, T_max - T_max_allowed)
    max_node = T_field.index(T_max)
    return violation, max_node


def thermal_penalty(
    T_field: List[float],
    T_max_allowed: float,
    penalty_weight: float = 1e6,
) -> float:
    """温度约束的二次罚函数.

    P(T) = penalty_weight * sum_i max(0, T_i - T_max)^2
    """
    s = 0.0
    for T_i in T_field:
        viol = max(0.0, T_i - T_max_allowed)
        s += viol * viol
    return penalty_weight * s


def thermal_penalty_gradient(
    T_field: List[float],
    T_max_allowed: float,
    penalty_weight: float = 1e6,
) -> List[float]:
    """温度罚函数的梯度 dP/dT_i.

    dP/dT_i = 2 * penalty_weight * max(0, T_i - T_max)
    """
    return [
        2.0 * penalty_weight * max(0.0, T_i - T_max_allowed)
        for T_i in T_field
    ]


# ===========================================================================
# 4. 伴随 PDE (向后积分)
# ===========================================================================
def adjoint_heat_step(
    lambda_T: List[float],
    T_field: List[float],
    disc: LaplacianDiscretizer,
    th: ThermalParameters,
    dx: float,
    dt: float,
    dL_dT: List[float],
) -> List[float]:
    """伴随热方程向后步.

    -dlambda/dt = alpha Laplacian(lambda) + dL/dT
    lambda 向后积分 (dt 取负).
    """
    n = disc.size
    if len(lambda_T) != n or len(dL_dT) != n:
        raise ValueError("向量长度不匹配")

    lap_lam = disc.matvec(lambda_T)
    alpha = th.diffusivity

    # dlambda/dt = -alpha Laplacian(lambda) - dL/dT
    # 向后: lambda^{k} = lambda^{k+1} - dt * dlam/dt
    rhs = [
        -alpha / (dx * dx) * lap_lam[i] - dL_dT[i]
        for i in range(n)
    ]

    return [lambda_T[i] - dt * rhs[i] for i in range(n)]


# ===========================================================================
# 5. 完整 PDE-ODE 耦合模拟
# ===========================================================================
def simulate_coupled_pde_ode(
    T0_field: List[float],
    disc: LaplacianDiscretizer,
    th: ThermalParameters,
    dx: float,
    dt: float,
    n_steps: int,
    method: str = "explicit",
) -> Tuple[List[List[float]], List[float]]:
    """耦合 PDE-ODE 模拟.

    Returns
    -------
    (T_history, max_temp_history) : (List[List[float]], List[float])
        温度场历史和最高温度历史.
    """
    T_curr = list(T0_field)
    T_history = [list(T_curr)]
    max_history = [max(T_curr)]

    for _ in range(n_steps):
        if method == "explicit":
            T_next = explicit_heat_step(T_curr, disc, th, dx, dt)
        else:
            T_next = crank_nicolson_heat_step(T_curr, disc, th, dx, dt)
        T_curr = T_next
        T_history.append(list(T_curr))
        max_history.append(max(T_curr))

    return T_history, max_history


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """PDE 约束模块自检."""
    m = 5
    grid = make_rectangular_grid(m)
    disc = LaplacianDiscretizer(grid)
    th = ThermalParameters()
    dx = 0.01  # 1 cm

    T0 = [300.0] * disc.size  # 初始 300 K
    dt = 0.001  # 1 ms
    n_steps = 10

    print(f"[PDE Constraint] m={m}, n={disc.size}, dx={dx} m")
    print(f"  alpha = {th.diffusivity:.4e} m^2/s")
    print(f"  CFL limit = {dx*dx/(4*th.diffusivity):.4e} s")

    # 显式
    T_hist, max_hist = simulate_coupled_pde_ode(
        T0, disc, th, dx, dt, n_steps, method="explicit"
    )
    print(f"[Explicit] 最终最高温度: {max_hist[-1]:.2f} K")

    # CN
    T_hist2, max_hist2 = simulate_coupled_pde_ode(
        T0, disc, th, dx, dt, n_steps, method="cn"
    )
    print(f"[CN]       最终最高温度: {max_hist2[-1]:.2f} K")

    # 约束检查
    viol, node = thermal_constraint_violation(T_hist[-1], 500.0)
    print(f"[Constraint] violation = {viol:.2e} at node {node}")

    # 罚函数
    pen = thermal_penalty(T_hist[-1], 500.0)
    print(f"[Penalty] = {pen:.2e}")


if __name__ == "__main__":
    self_check()
