"""
costate_shooting.py
===================

两点边值问题的 Broyden 打靶法求解器。

融合种子项目:
  - 120_broyden : Tim Kelley 的 Broyden 拟 Newton 法

在 Pontryagin 最优控制中, PMP 导出的两点边值问题 (TPBVP):
  - 初始状态已知:  x(0) = x_0
  - 终端伴随已知:  lambda(T) = dPhi/dx(T)
  - 需要求解决策变量: lambda(0) (初始伴随)

打靶法思路:
  1. 猜测 lambda(0)
  2. 前向积分状态方程 + 伴随方程 (使用最优控制 u*)
  3. 计算终端误差: F(lambda_0) = lambda_computed(T) - lambda_target(T)
  4. 用 Broyden 法更新 lambda(0), 直到 ||F|| < tol

数学公式:
---------
1. Broyden 更新:
     x_{k+1} = x_k - B_k^{-1} F(x_k)
     其中 B_k 为 Jacobian 的近似, 通过秩-1 更新:
     B_{k+1} = B_k + (Delta F - B_k Delta x) Delta x^T / (Delta x^T Delta x)

2. 打靶函数:
     F(lambda_0) = lambda(T; lambda_0) - dPhi/dx(T)

3. 收敛准则:
     ||F(lambda_0)|| <= atol + rtol * ||F(lambda_0^{(0)})||
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scientific_constants import EARTH, SpacecraftParameters
from state_dynamics import Control, State, rk4_step, euler_step
from hamiltonian_core import (
    Costate,
    costate_derivative,
    hamiltonian,
    optimal_control,
    terminal_costate,
)


# ===========================================================================
# 1. 打靶函数
# ===========================================================================
@dataclass
class ShootingResult:
    """打靶结果."""
    lambda_0: Costate
    trajectory_states: List[State]
    trajectory_costates: List[Costate]
    trajectory_controls: List[Control]
    terminal_residual: float
    n_iterations: int
    converged: bool


def shoot(
    lambda_0_list: List[float],
    x0: State,
    x_target: State,
    sc: SpacecraftParameters,
    T_final: float,
    n_steps: int,
    method: str = "rk4",
) -> List[float]:
    """前向打靶: 从 lambda(0) 积分到 T, 返回终端残差.

    Parameters
    ----------
    lambda_0_list : List[float]
        初始伴随 [lr0, lv0, lm0, ltheta0, lgamma0].
    x0 : State
        初始状态.
    x_target : State
        目标终端状态 (用于计算 terminal_costate).
    sc : SpacecraftParameters
        航天器参数.
    T_final : float
        终端时间 [s].
    n_steps : int
        时间步数.
    method : str
        积分方法.

    Returns
    -------
    List[float]
        终端残差 F(lambda_0) = lambda_computed(T) - lambda_target(T).
    """
    dt = T_final / n_steps
    x_curr = x0
    lam_curr = Costate.from_list(lambda_0_list)

    for k in range(n_steps):
        t = k * dt
        # 最优控制
        u_star = optimal_control(x_curr, lam_curr, sc)

        # 状态积分
        if method == "rk4":
            x_next = rk4_step(x_curr, u_star, sc, dt, t)
        else:
            x_next = euler_step(x_curr, u_star, sc, dt, t)

        # 伴随积分 (显式 Euler, 因 dlam/dt 计算较贵)
        dlam = costate_derivative(x_curr, u_star, lam_curr, sc, t)
        lam_next_list = [
            l + dt * dl for l, dl in zip(lam_curr.as_list(), dlam)
        ]
        lam_next = Costate.from_list(lam_next_list)

        x_curr = x_next
        lam_curr = lam_next

    # 终端残差
    lam_target = terminal_costate(x_curr, x_target)
    residual = [
        lc - lt for lc, lt in zip(lam_curr.as_list(), lam_target.as_list())
    ]
    return residual


# ===========================================================================
# 2. Broyden 法求解 TPBVP (来自 120_broyden)
# ===========================================================================
def broyden_solve(
    x0_init: List[float],
    F_func,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    maxit: int = 50,
    maxdim: int = 10,
) -> Tuple[List[float], int, bool]:
    """Broyden 拟 Newton 法求解非线性方程组 F(x) = 0.

    算法 (来自 120_broyden):
        1. 初始步 stp_0 = -F(x_0)
        2. 迭代:
            x_{k+1} = x_k + stp_k
            F_{k+1} = F(x_{k+1})
            若 ||F_{k+1}|| <= stop_tol, 成功
            计算新方向 (Broyden 秩-1 更新)
        3. 若达到 maxdim, 重启

    Parameters
    ----------
    x0_init : List[float]
        初始猜测.
    F_func : callable
        残差函数 F(x).
    atol, rtol : float
        容差.
    maxit : int
        最大迭代次数.
    maxdim : int
        Broyden 存储维度 (之后重启).

    Returns
    -------
    (x_solution, n_iterations, converged) : (List[float], int, bool)
    """
    n = len(x0_init)
    x = list(x0_init)

    # 初始残差
    fc = F_func(x)
    fnrm = math.sqrt(sum(f * f for f in fc) / n)
    fnrm0 = fnrm
    stop_tol = atol + rtol * fnrm0

    # 存储步方向
    stp = [[0.0] * maxdim for _ in range(n)]
    stp_nrm = [0.0] * maxdim
    for i in range(n):
        stp[i][0] = -fc[i]
    stp_nrm[0] = sum(s * s for s in [stp[i][0] for i in range(n)])

    nbroy = 0
    itc = 0

    while itc < maxit:
        nbroy += 1
        fnrmo = fnrm
        itc += 1

        # 下一步
        for i in range(n):
            x[i] += stp[i][nbroy - 1]

        fc = F_func(x)
        fnrm = math.sqrt(sum(f * f for f in fc) / n)

        # 收敛?
        if fnrm <= stop_tol:
            return x, itc, True

        # 发散?
        if fnrmo <= fnrm:
            # 回溯 (简单步长减半)
            for i in range(n):
                x[i] -= 0.5 * stp[i][nbroy - 1]
            fc = F_func(x)
            fnrm = math.sqrt(sum(f * f for f in fc) / n)
            if fnrm < fnrmo:
                continue
            else:
                return x, itc, False

        # Broyden 更新
        if nbroy + 1 <= maxdim:
            z = list(-f for f in fc)
            if nbroy > 1:
                for kbr in range(nbroy - 1):
                    dot = sum(stp[i][kbr] * z[i] for i in range(n))
                    for i in range(n):
                        z[i] += stp[i][kbr + 1] * dot / stp_nrm[kbr]

            zz = sum(stp[i][nbroy - 1] * z[i] for i in range(n))
            zz /= stp_nrm[nbroy - 1]
            for i in range(n):
                stp[i][nbroy] = z[i] / (1.0 - zz)
            stp_nrm[nbroy] = sum(s * s for s in [stp[i][nbroy] for i in range(n)])
        else:
            # 重启
            for i in range(n):
                stp[i][0] = -fc[i]
            stp_nrm[0] = sum(s * s for s in [stp[i][0] for i in range(n)])
            nbroy = 0

    return x, itc, fnrm <= stop_tol


# ===========================================================================
# 3. 完整 TPBVP 求解器
# ===========================================================================
def solve_tpbbvp(
    x0: State,
    x_target: State,
    sc: SpacecraftParameters,
    T_final: float,
    n_steps: int = 100,
    lambda_0_guess: Optional[List[float]] = None,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    maxit: int = 30,
) -> ShootingResult:
    """求解两点边值问题 (Pontryagin TPBVP).

    Parameters
    ----------
    x0 : State
        初始状态.
    x_target : State
        目标终端状态.
    sc : SpacecraftParameters
        航天器参数.
    T_final : float
        终端时间 [s].
    n_steps : int
        时间步数.
    lambda_0_guess : List[float], optional
        初始伴随猜测 (默认全零).
    atol, rtol : float
        容差.
    maxit : int
        最大 Broyden 迭代.

    Returns
    -------
    ShootingResult
    """
    if lambda_0_guess is None:
        lambda_0_guess = [0.0, 0.0, -1e-4, 0.0, 0.0]

    # 打靶函数 (闭包)
    def F(lam_list: List[float]) -> List[float]:
        return shoot(lam_list, x0, x_target, sc, T_final, n_steps)

    # Broyden 求解
    lam_0_sol, n_iter, converged = broyden_solve(
        lambda_0_guess, F, atol=atol, rtol=rtol, maxit=maxit
    )

    # 最终前向积分, 记录完整轨迹
    dt = T_final / n_steps
    x_curr = x0
    lam_curr = Costate.from_list(lam_0_sol)
    states = [x0]
    costates = [lam_curr]
    controls: List[Control] = []

    for k in range(n_steps):
        t = k * dt
        u_star = optimal_control(x_curr, lam_curr, sc)
        controls.append(u_star)

        x_next = rk4_step(x_curr, u_star, sc, dt, t)
        dlam = costate_derivative(x_curr, u_star, lam_curr, sc, t)
        lam_next = Costate.from_list(
            [l + dt * dl for l, dl in zip(lam_curr.as_list(), dlam)]
        )

        states.append(x_next)
        costates.append(lam_next)
        x_curr = x_next
        lam_curr = lam_next

    # 终端残差
    residual = F(lam_0_sol)
    res_norm = math.sqrt(sum(r * r for r in residual))

    return ShootingResult(
        lambda_0=Costate.from_list(lam_0_sol),
        trajectory_states=states,
        trajectory_costates=costates,
        trajectory_controls=controls,
        terminal_residual=res_norm,
        n_iterations=n_iter,
        converged=converged,
    )


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """打靶法模块自检."""
    sc = SpacecraftParameters()
    x0 = State(
        r=EARTH.radius_mean + 200e3,
        v=7800.0,
        m=sc.total_mass,
        theta=0.0,
        gamma=0.0,
    )
    x_target = State(
        r=EARTH.radius_mean + 400e3,
        v=7670.0,
        m=1800.0,
        theta=0.3,
        gamma=0.0,
    )
    T_final = 500.0
    n_steps = 50

    print("[TPBVP Solver] 打靶法求解:")
    print(f"  初始高度: {(x0.r - EARTH.radius_mean)/1e3:.1f} km")
    print(f"  目标高度: {(x_target.r - EARTH.radius_mean)/1e3:.1f} km")
    print(f"  终端时间: {T_final} s, 步数: {n_steps}")

    result = solve_tpbbvp(x0, x_target, sc, T_final, n_steps, maxit=15)
    print(f"  收敛: {result.converged}")
    print(f"  迭代次数: {result.n_iterations}")
    print(f"  终端残差: {result.terminal_residual:.3e}")
    print(f"  lambda_0: {[f'{c:.3e}' for c in result.lambda_0.as_list()]}")


if __name__ == "__main__":
    self_check()
