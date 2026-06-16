"""
convergence_analyzer.py
=======================

收敛分析与误差评估模块。

在最优控制中, 该模块用于:
  1. 评估打靶法的收敛性
  2. 计算轨迹积分误差
  3. Hamilton 守恒检验
  4. 网格收敛性分析 (h-refinement)

数学公式:
---------
1. 收敛阶 (网格细化):
     若误差 e(h) = C h^p, 则
     p = log(e(h1)/e(h2)) / log(h1/h2)

2. 残差范数:
     ||F(lambda)||_2 = sqrt(sum F_i^2)

3. Hamilton 守恒误差:
     delta_H = max_t |H(t) - H(0)| / |H(0)|

4. 伴随变量连续性:
     delta_lam = ||lambda(T-) - lambda(T+)|| / ||lambda(T+)||
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from scientific_constants import SpacecraftParameters
from state_dynamics import State
from hamiltonian_core import Costate, Control, hamiltonian


# ===========================================================================
# 1. 收敛报告
# ===========================================================================
@dataclass
class ConvergenceReport:
    """收敛分析报告."""
    n_iterations: int
    final_residual: float
    converged: bool
    hamiltonian_variation: float
    estimated_order: float  # 收敛阶
    max_state_jump: float   # 状态最大跳变
    max_costate_jump: float # 伴随最大跳变


# ===========================================================================
# 2. Hamilton 守恒检验
# ===========================================================================
def hamiltonian_conservation(
    states: List[State],
    costates: List[Costate],
    controls: List[Control],
    sc: SpacecraftParameters,
) -> Tuple[List[float], float]:
    """检验 Hamilton 沿轨迹的守恒性.

    Returns
    -------
    (H_values, variation) : (List[float], float)
        H 值和相对变化量.
    """
    H_values = []
    n = min(len(states), len(costates), len(controls))
    for i in range(n):
        H = hamiltonian(states[i], controls[i], costates[i], sc)
        H_values.append(H)

    if not H_values:
        return [], 0.0

    H0 = H_values[0]
    if abs(H0) < 1e-15:
        # H 接近零, 用绝对变化
        delta_H = max(abs(H - H0) for H in H_values)
    else:
        delta_H = max(abs(H - H0) for H in H_values) / abs(H0)

    return H_values, delta_H


# ===========================================================================
# 3. 状态与伴随连续性检验
# ===========================================================================
def state_continuity_check(
    states: List[State],
) -> Tuple[float, int]:
    """检查状态轨迹的连续性 (最大跳变).

    跳变 = ||x_{k+1} - x_k|| / ||x_k||
    """
    max_jump = 0.0
    max_idx = -1
    for k in range(len(states) - 1):
        x1 = states[k].as_list()
        x2 = states[k + 1].as_list()
        norm_x = math.sqrt(sum(xi * xi for xi in x1))
        if norm_x < 1e-15:
            continue
        jump = math.sqrt(sum((x2i - x1i) ** 2 for x1i, x2i in zip(x1, x2))) / norm_x
        if jump > max_jump:
            max_jump = jump
            max_idx = k
    return max_jump, max_idx


def costate_continuity_check(
    costates: List[Costate],
) -> Tuple[float, int]:
    """检查伴随轨迹的连续性."""
    max_jump = 0.0
    max_idx = -1
    for k in range(len(costates) - 1):
        l1 = costates[k].as_list()
        l2 = costates[k + 1].as_list()
        norm_l = math.sqrt(sum(li * li for li in l1))
        if norm_l < 1e-15:
            continue
        jump = math.sqrt(sum((l2i - l1i) ** 2 for l1i, l2i in zip(l1, l2))) / norm_l
        if jump > max_jump:
            max_jump = jump
            max_idx = k
    return max_jump, max_idx


# ===========================================================================
# 4. 网格收敛阶估计
# ===========================================================================
def estimate_convergence_order(
    errors: List[float], step_sizes: List[float]
) -> float:
    """估计收敛阶 p.

    假设 e(h) = C h^p, 则
    p ≈ log(e(h1)/e(h2)) / log(h1/h2)

    使用相邻两对数据, 取平均.
    """
    if len(errors) < 2 or len(step_sizes) < 2:
        return 0.0

    orders = []
    for i in range(len(errors) - 1):
        e1, e2 = errors[i], errors[i + 1]
        h1, h2 = step_sizes[i], step_sizes[i + 1]
        if e1 > 1e-15 and e2 > 1e-15 and h1 != h2:
            p = math.log(e1 / e2) / math.log(h1 / h2)
            orders.append(p)

    if not orders:
        return 0.0
    return sum(orders) / len(orders)


# ===========================================================================
# 5. 完整收敛分析
# ===========================================================================
def analyze_convergence(
    states: List[State],
    costates: List[Costate],
    controls: List[Control],
    sc: SpacecraftParameters,
    n_iterations: int,
    final_residual: float,
    converged: bool,
) -> ConvergenceReport:
    """完整收敛分析."""
    H_values, delta_H = hamiltonian_conservation(states, costates, controls, sc)
    max_state_jump, _ = state_continuity_check(states)
    max_costate_jump, _ = costate_continuity_check(costates)

    # 简化: 假设二阶 (RK4)
    estimated_order = 2.0

    return ConvergenceReport(
        n_iterations=n_iterations,
        final_residual=final_residual,
        converged=converged,
        hamiltonian_variation=delta_H,
        estimated_order=estimated_order,
        max_state_jump=max_state_jump,
        max_costate_jump=max_costate_jump,
    )


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """收敛分析模块自检."""
    from scientific_constants import EARTH

    sc = SpacecraftParameters()

    # 构造简单测试轨迹
    states = [
        State(r=EARTH.radius_mean + 200e3 + k * 1e3, v=7800.0 - k * 0.5, m=2000.0 - k, theta=k * 0.01, gamma=0.0)
        for k in range(10)
    ]
    costates = [Costate(lr=1e-6, lv=1e-3, lm=-1e-4, ltheta=0.0, lgamma=1e-2) for _ in range(10)]
    controls = [Control(thrust=10000.0, alpha=0.1) for _ in range(10)]

    report = analyze_convergence(
        states, costates, controls, sc,
        n_iterations=10, final_residual=1e-5, converged=True
    )
    print(f"[Convergence Analysis]")
    print(f"  迭代次数: {report.n_iterations}")
    print(f"  终端残差: {report.final_residual:.3e}")
    print(f"  收敛: {report.converged}")
    print(f"  Hamilton 变化: {report.hamiltonian_variation:.4e}")
    print(f"  状态最大跳变: {report.max_state_jump:.4e}")
    print(f"  伴随最大跳变: {report.max_costate_jump:.4e}")


if __name__ == "__main__":
    self_check()
