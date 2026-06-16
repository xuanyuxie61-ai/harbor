"""
distributed_hamiltonian.py
==========================

分布式 Hamilton 系统并行评估模块。

融合种子项目:
  - 1218_uw-mad-dash_bagpipe : 分布式训练框架 (RPC / 并行 worker)

在随机最优控制中, 该模块用于:
  1. 并行评估多个不确定性场景的 Hamilton 值
  2. 并行前向积分 (每个 worker 处理一个场景)
  3. 分布式计算期望代价和梯度

数学公式:
---------
1. 期望代价 (Monte Carlo):
     J = E_{xi}[J(xi)] ≈ (1/N) sum_{i=1}^N J(xi_i)

2. 并行梯度:
     nabla J ≈ (1/N) sum_{i=1}^N nabla J(xi_i)

3. Worker 分配:
     Worker k 处理场景 {xi_{k*L+1}, ..., xi_{(k+1)*L}}
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Tuple

from scientific_constants import SpacecraftParameters
from stochastic_sampler import UncertaintyScenario
from state_dynamics import Control, State
from hamiltonian_core import Costate, hamiltonian


# ===========================================================================
# 1. Worker 任务与结果
# ===========================================================================
@dataclass
class WorkerTask:
    """Worker 任务."""
    worker_id: int
    scenarios: List[UncertaintyScenario]
    x0: State
    sc: SpacecraftParameters


@dataclass
class WorkerResult:
    """Worker 结果."""
    worker_id: int
    total_cost: float
    total_hamiltonian: float
    n_scenarios: int


# ===========================================================================
# 2. 单场景评估
# ===========================================================================
def evaluate_single_scenario(
    scenario: UncertaintyScenario,
    x0: State,
    sc: SpacecraftParameters,
    trajectory_states: List[State],
    trajectory_controls: List[Control],
    trajectory_costates: List[Costate],
) -> Tuple[float, float]:
    """评估单个场景的代价和 Hamilton.

    Parameters
    ----------
    scenario : UncertaintyScenario
        不确定性场景 (扰动参数).
    x0 : State
        初始状态.
    sc : SpacecraftParameters
        航天器参数 (会被场景扰动).
    trajectory_* : 标称轨迹.

    Returns
    -------
    (cost, avg_hamiltonian) : (float, float)
    """
    # 扰动参数
    isp_pert = scenario.isp_perturbation
    thrust_pert = scenario.thrust_perturbation

    # 扰动后的航天器参数 (浅拷贝并修改)
    sc_pert = SpacecraftParameters(
        dry_mass=sc.dry_mass,
        fuel_mass=sc.fuel_mass,
        I_sp_vacuum=sc.I_sp_vacuum * isp_pert,
        I_sp_sea=sc.I_sp_sea * isp_pert,
        thrust_max=sc.thrust_max * thrust_pert,
        thrust_min=sc.thrust_min,
        drag_coeff=sc.drag_coeff * scenario.density_perturbation,
        ref_area=sc.ref_area,
        thermal_mass=sc.thermal_mass * scenario.thermal_perturbation,
        emissivity=sc.emissivity,
        biot_number=sc.biot_number,
    )

    # 计算沿轨迹的代价和 Hamilton
    total_cost = 0.0
    total_H = 0.0
    n = min(len(trajectory_states), len(trajectory_controls), len(trajectory_costates))

    for i in range(n):
        x = trajectory_states[i]
        u = trajectory_controls[i]
        lam = trajectory_costates[i]

        # 运行代价 (简化: 推力平方)
        L = 0.5 * (u.thrust / sc_pert.thrust_max) ** 2
        total_cost += L

        # Hamilton
        H = hamiltonian(x, u, lam, sc_pert)
        total_H += H

    avg_H = total_H / max(1, n)
    return total_cost, avg_H


# ===========================================================================
# 3. Worker 函数
# ===========================================================================
def worker_evaluate(task: WorkerTask) -> WorkerResult:
    """Worker: 评估分配的场景子集.

    注意: 这里为简化, 使用标称轨迹. 实际应用中, 每个场景应重新积分.
    """
    total_cost = 0.0
    total_H = 0.0

    # 使用标称轨迹 (简化)
    from state_dynamics import rk4_step
    from hamiltonian_core import optimal_control, costate_derivative

    sc = task.sc
    x_curr = task.x0
    lam_list = [0.0, 0.0, -1e-4, 0.0, 0.0]  # 默认初始伴随
    from hamiltonian_core import Costate
    lam_curr = Costate.from_list(lam_list)

    # 简单积分 50 步
    dt = 10.0
    n_steps = 50
    states = [x_curr]
    controls = []
    costates = [lam_curr]

    for _ in range(n_steps):
        u = optimal_control(x_curr, lam_curr, sc)
        controls.append(u)
        x_next = rk4_step(x_curr, u, sc, dt)
        dlam = costate_derivative(x_curr, u, lam_curr, sc)
        lam_next = Costate.from_list(
            [l + dt * dl for l, dl in zip(lam_curr.as_list(), dlam)]
        )
        states.append(x_next)
        costates.append(lam_next)
        x_curr = x_next
        lam_curr = lam_next

    # 评估每个场景
    for scenario in task.scenarios:
        cost, avg_H = evaluate_single_scenario(
            scenario, task.x0, sc, states, controls, costates
        )
        total_cost += scenario.weight * cost
        total_H += scenario.weight * avg_H

    return WorkerResult(
        worker_id=task.worker_id,
        total_cost=total_cost,
        total_hamiltonian=total_H,
        n_scenarios=len(task.scenarios),
    )


# ===========================================================================
# 4. 分布式评估协调器
# ===========================================================================
def distributed_evaluate(
    scenarios: List[UncertaintyScenario],
    x0: State,
    sc: SpacecraftParameters,
    n_workers: int = 2,
) -> Tuple[float, float]:
    """分布式评估多个场景.

    Parameters
    ----------
    scenarios : List[UncertaintyScenario]
        场景列表.
    x0 : State
        初始状态.
    sc : SpacecraftParameters
        航天器参数.
    n_workers : int
        Worker 数量.

    Returns
    -------
    (expected_cost, expected_hamiltonian) : (float, float)
    """
    if not scenarios:
        return 0.0, 0.0

    # 分配场景到 workers
    n_scenarios = len(scenarios)
    chunk_size = max(1, n_scenarios // n_workers)
    tasks: List[WorkerTask] = []

    for k in range(n_workers):
        start = k * chunk_size
        end = start + chunk_size if k < n_workers - 1 else n_scenarios
        if start >= n_scenarios:
            break
        tasks.append(
            WorkerTask(
                worker_id=k,
                scenarios=scenarios[start:end],
                x0=x0,
                sc=sc,
            )
        )

    # 顺序执行 (避免 multiprocessing 复杂性)
    results: List[WorkerResult] = []
    for task in tasks:
        result = worker_evaluate(task)
        results.append(result)

    # 汇总
    total_cost = sum(r.total_cost for r in results)
    total_H = sum(r.total_hamiltonian for r in results)

    return total_cost, total_H


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """分布式评估模块自检."""
    from stochastic_sampler import generate_scenarios

    scenarios = generate_scenarios(n_scenarios=5, seed_tuple=(100, 200, 300))
    sc = SpacecraftParameters()
    x0 = State(
        r=6.571e6,
        v=7800.0,
        m=sc.total_mass,
        theta=0.0,
        gamma=0.0,
    )

    print("[Distributed Evaluation] 5 个场景, 2 workers:")
    cost, avg_H = distributed_evaluate(scenarios, x0, sc, n_workers=2)
    print(f"  期望代价: {cost:.4e}")
    print(f"  期望 Hamilton: {avg_H:.4e}")


if __name__ == "__main__":
    self_check()
