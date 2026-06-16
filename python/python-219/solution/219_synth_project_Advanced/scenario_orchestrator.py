"""
scenario_orchestrator.py
========================

多场景鲁棒性分析协调器。

融合种子项目:
  - 1126_mengyingmandywu_SAF-replication-files : 多场景鲁棒性检查与协调

在最优控制中, 该模块用于:
  1. 自动生成多个不确定性场景
  2. 并行运行每个场景的最优控制求解
  3. 汇总统计结果 (均值, 方差, 分位数)
  4. 鲁棒性检验 (不同 k 范围)

数学公式:
---------
1. 期望代价:
     E[J] ≈ (1/N) sum_{i=1}^N J(xi_i)

2. 方差:
     Var[J] ≈ (1/(N-1)) sum_{i=1}^N (J(xi_i) - E[J])^2

3. 分位数 (q-quantile):
     Q_q = sorted_J[ceil(q * N)]

4. 鲁棒性指标:
     R = E[J] + kappa * sqrt(Var[J])  (mean-risk)
     其中 kappa 为风险厌恶系数
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from scientific_constants import EARTH, SpacecraftParameters
from state_dynamics import State
from stochastic_sampler import UncertaintyScenario, generate_scenarios
from costate_shooting import solve_tpbbvp, ShootingResult
from distributed_hamiltonian import distributed_evaluate


# ===========================================================================
# 1. 场景结果
# ===========================================================================
@dataclass
class ScenarioResult:
    """单个场景的求解结果."""
    scenario_id: int
    scenario: UncertaintyScenario
    cost: float
    hamiltonian: float
    converged: bool
    terminal_residual: float
    n_iterations: int


@dataclass
class RobustnessReport:
    """鲁棒性分析报告."""
    n_scenarios: int
    mean_cost: float
    std_cost: float
    min_cost: float
    max_cost: float
    quantiles: dict  # {0.25: ..., 0.5: ..., 0.75: ...}
    convergence_rate: float
    risk_index: float  # E[J] + kappa * std


# ===========================================================================
# 2. 单场景求解
# ===========================================================================
def solve_scenario(
    scenario_id: int,
    scenario: UncertaintyScenario,
    x0: State,
    x_target: State,
    sc_base: SpacecraftParameters,
    T_final: float,
    n_steps: int,
) -> ScenarioResult:
    """求解单个场景的最优控制.

    扰动航天器参数, 然后调用 TPBVP 求解器.
    """
    # 扰动参数
    sc = SpacecraftParameters(
        dry_mass=sc_base.dry_mass,
        fuel_mass=sc_base.fuel_mass,
        I_sp_vacuum=sc_base.I_sp_vacuum * scenario.isp_perturbation,
        I_sp_sea=sc_base.I_sp_sea * scenario.isp_perturbation,
        thrust_max=sc_base.thrust_max * scenario.thrust_perturbation,
        thrust_min=sc_base.thrust_min,
        drag_coeff=sc_base.drag_coeff * scenario.density_perturbation,
        ref_area=sc_base.ref_area,
        thermal_mass=sc_base.thermal_mass * scenario.thermal_perturbation,
        emissivity=sc_base.emissivity,
        biot_number=sc_base.biot_number,
    )

    # 求解 TPBVP
    result = solve_tpbbvp(
        x0, x_target, sc, T_final, n_steps, maxit=10
    )

    # 评估代价 (简化: 用 Hamilton 平均值)
    cost, avg_H = distributed_evaluate([scenario], x0, sc, n_workers=1)

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario=scenario,
        cost=cost,
        hamiltonian=avg_H,
        converged=result.converged,
        terminal_residual=result.terminal_residual,
        n_iterations=result.n_iterations,
    )


# ===========================================================================
# 3. 多场景协调器 (来自 1126_SAF)
# ===========================================================================
def run_all_scenarios(
    x0: State,
    x_target: State,
    sc: SpacecraftParameters,
    T_final: float,
    n_steps: int,
    n_scenarios: int = 5,
    k_range: int = 1,
    seed_tuple: Tuple[int, int, int] = (111, 222, 333),
) -> Tuple[List[ScenarioResult], RobustnessReport]:
    """运行所有场景并汇总结果.

    Parameters
    ----------
    x0, x_target, sc, T_final, n_steps : 同 solve_scenario
    n_scenarios : int
        场景数量.
    k_range : int
        鲁棒性检验范围 (不同种子偏移).
    seed_tuple : (s1, s2, s3)
        随机种子.

    Returns
    -------
    (results, report) : (List[ScenarioResult], RobustnessReport)
    """
    print(f"[Scenario Orchestrator] 生成 {n_scenarios} 个场景 (k_range={k_range})")
    scenarios = generate_scenarios(n_scenarios, seed_tuple)

    results: List[ScenarioResult] = []
    for i, sc_i in enumerate(scenarios):
        print(f"  求解场景 {i+1}/{n_scenarios}...")
        res = solve_scenario(i, sc_i, x0, x_target, sc, T_final, n_steps)
        results.append(res)
        status = "OK" if res.converged else "FAIL"
        print(
            f"    cost={res.cost:.4e}, residual={res.terminal_residual:.3e} [{status}]"
        )

    # 统计
    costs = [r.cost for r in results]
    n = len(costs)
    if n == 0:
        mean_c, std_c = 0.0, 0.0
    else:
        mean_c = sum(costs) / n
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in costs) / max(1, n - 1))

    sorted_costs = sorted(costs)
    quantiles = {
        0.25: sorted_costs[max(0, int(0.25 * n) - 1)] if n > 0 else 0.0,
        0.50: sorted_costs[max(0, int(0.50 * n) - 1)] if n > 0 else 0.0,
        0.75: sorted_costs[max(0, int(0.75 * n) - 1)] if n > 0 else 0.0,
    }

    n_converged = sum(1 for r in results if r.converged)
    conv_rate = n_converged / max(1, n)

    kappa = 1.0  # 风险厌恶系数
    risk = mean_c + kappa * std_c

    report = RobustnessReport(
        n_scenarios=n,
        mean_cost=mean_c,
        std_cost=std_c,
        min_cost=min(costs) if costs else 0.0,
        max_cost=max(costs) if costs else 0.0,
        quantiles=quantiles,
        convergence_rate=conv_rate,
        risk_index=risk,
    )

    return results, report


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """场景协调器自检."""
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

    results, report = run_all_scenarios(
        x0, x_target, sc, T_final=500.0, n_steps=50, n_scenarios=3
    )
    print(f"\n[Robustness Report]")
    print(f"  场景数: {report.n_scenarios}")
    print(f"  平均代价: {report.mean_cost:.4e}")
    print(f"  标准差: {report.std_cost:.4e}")
    print(f"  收敛率: {report.convergence_rate:.2%}")
    print(f"  风险指标: {report.risk_index:.4e}")
    print(f"  分位数: {report.quantiles}")


if __name__ == "__main__":
    self_check()
