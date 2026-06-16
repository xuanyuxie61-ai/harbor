"""
main.py
=======

Pontryagin 多阶段随机最优控制框架 - 统一入口。

本框架融合 15 个种子项目的核心算法, 解决能量受限航天器在温度场 PDE
约束下的多目标轨迹优化问题。

科学问题:
---------
基于 Pontryagin 极大值原理, 求解多级火箭在不确定性参数下的最优轨迹,
同时考虑热扩散 PDE 约束 (温度路径约束) 和多目标访问顺序优化。

核心算法融合:
-------------
1.  **1115_inerOsci**      -> 状态动力学 (隐式/显式 ODE 求解)
2.  **120_broyden**        -> 打靶法求解 TPBVP (Broyden 拟 Newton)
3.  **269_delsq**          -> PDE 约束的 Laplacian 离散化
4.  **971_r8bto**          -> 分块 Toeplitz 时间离散算子
5.  **047_asa183**         -> Wichmann-Hill 伪随机数发生器
6.  **699_log_normal_truncated_ab** -> 截断对数正态不确定性建模
7.  **244_cvt_1d_lumping** -> CVT 最优空间量化 (传感器布置)
8.  **440_florida_cvt_pop** -> 密度加权 2D CVT
9.  **624_knapsack_dynamic** -> 动态规划阶段决策
10. **1366_tsp_moler**     -> TSP 多目标路径排序
11. **1218_bagpipe**       -> 分布式并行场景评估
12. **1126_SAF**           -> 多场景鲁棒性协调
13. **912_prime_fermat**   -> 网格维度素性验证
14. **Pontryagin PMP**     -> Hamilton 系统与伴随方程
15. **PDE-constrained OC** -> 热扩散耦合最优控制

运行方式:
---------
    python main.py

输出:
-----
    - 标称最优轨迹求解结果
    - PDE 热扩散耦合模拟
    - CVT 最优传感器布置
    - 多目标路径排序
    - 动态规划阶段决策
    - 多场景鲁棒性分析报告
"""

from __future__ import annotations

import math
import sys
import time
from typing import List, Tuple

# ===========================================================================
# 模块导入
# ===========================================================================
from scientific_constants import (
    EARTH,
    SpacecraftParameters,
    ThermalParameters,
    self_check as constants_check,
)
from stochastic_sampler import (
    generate_scenarios,
    self_check as sampler_check,
)
from laplacian_discretizer import (
    LaplacianDiscretizer,
    make_rectangular_grid,
    self_check as laplacian_check,
)
from block_toeplitz_operator import (
    make_toeplitz_indicator,
    self_check as toeplitz_check,
)
from fermat_primality import (
    is_prime_miller_rabin,
    validate_grid_dimension,
    self_check as fermat_check,
)
from state_dynamics import (
    Control,
    State,
    integrate_trajectory,
    self_check as dynamics_check,
)
from hamiltonian_core import (
    Costate,
    hamiltonian,
    optimal_control,
    self_check as hamiltonian_check,
)
from costate_shooting import (
    solve_tpbbvp,
    self_check as shooting_check,
)
from cvt_quantizer import (
    cvt_1d_lloyd,
    cvt_2d_lloyd,
    gaussian_density_1d,
    gaussian_density_2d,
    self_check as cvt_check,
)
from dynamic_programming import (
    knapsack_dp,
    rocket_stage_selection,
    self_check as dp_check,
)
from path_planner import (
    solve_tsp,
    self_check as path_check,
)
from pde_constraint import (
    simulate_coupled_pde_ode,
    self_check as pde_check,
)
from distributed_hamiltonian import (
    distributed_evaluate,
    self_check as distributed_check,
)
from scenario_orchestrator import (
    run_all_scenarios,
    self_check as orchestrator_check,
)
from boundary_handler import (
    project_state_to_feasible,
    saturate_control,
    self_check as boundary_check,
)
from convergence_analyzer import (
    analyze_convergence,
    self_check as convergence_check,
)


# ===========================================================================
# 主流程
# ===========================================================================
def print_header(title: str) -> None:
    """打印章节标题."""
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def run_module_self_checks() -> None:
    """运行所有模块的自检."""
    print_header("模块自检 (Self-Checks)")
    checks = [
        ("Scientific Constants", constants_check),
        ("Stochastic Sampler", sampler_check),
        ("Laplacian Discretizer", laplacian_check),
        ("Block Toeplitz", toeplitz_check),
        ("Fermat Primality", fermat_check),
        ("State Dynamics", dynamics_check),
        ("Hamiltonian Core", hamiltonian_check),
        ("CVT Quantizer", cvt_check),
        ("Dynamic Programming", dp_check),
        ("Path Planner", path_check),
        ("PDE Constraint", pde_check),
        ("Distributed Hamiltonian", distributed_check),
        ("Boundary Handler", boundary_check),
        ("Convergence Analyzer", convergence_check),
    ]
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            check_func()
            print(f"  [{name}] PASS")
        except Exception as e:
            print(f"  [{name}] FAIL: {e}")


def solve_nominal_trajectory() -> dict:
    """求解标称最优轨迹."""
    print_header("标称最优轨迹求解 (Pontryagin TPBVP)")

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

    print(f"初始状态:")
    print(f"  高度: {(x0.r - EARTH.radius_mean)/1e3:.1f} km")
    print(f"  速度: {x0.v:.1f} m/s")
    print(f"  质量: {x0.m:.1f} kg")
    print(f"终端目标:")
    print(f"  高度: {(x_target.r - EARTH.radius_mean)/1e3:.1f} km")
    print(f"  速度: {x_target.v:.1f} m/s")
    print(f"  质量: {x_target.m:.1f} kg")
    print(f"终端时间: {T_final} s, 步数: {n_steps}")

    result = solve_tpbbvp(x0, x_target, sc, T_final, n_steps, maxit=15)

    print(f"\n求解结果:")
    print(f"  收敛: {result.converged}")
    print(f"  迭代次数: {result.n_iterations}")
    print(f"  终端残差: {result.terminal_residual:.3e}")
    print(f"  初始伴随: {[f'{c:.3e}' for c in result.lambda_0.as_list()]}")

    # 收敛分析
    report = analyze_convergence(
        result.trajectory_states,
        result.trajectory_costates,
        result.trajectory_controls,
        sc,
        result.n_iterations,
        result.terminal_residual,
        result.converged,
    )
    print(f"\n收敛分析:")
    print(f"  Hamilton 变化: {report.hamiltonian_variation:.4e}")
    print(f"  状态最大跳变: {report.max_state_jump:.4e}")
    print(f"  伴随最大跳变: {report.max_costate_jump:.4e}")

    return {
        "result": result,
        "report": report,
        "x0": x0,
        "x_target": x_target,
        "sc": sc,
    }


def run_pde_simulation() -> dict:
    """运行 PDE 热扩散耦合模拟."""
    print_header("PDE 热扩散耦合模拟")

    m_grid = 5
    valid, msg = validate_grid_dimension(m_grid, require_prime=False)
    print(f"网格维度验证: {msg} (valid={valid})")

    grid = make_rectangular_grid(m_grid)
    disc = LaplacianDiscretizer(grid)
    th = ThermalParameters()
    dx = 0.01  # 1 cm

    T0 = [300.0] * disc.size
    dt = 0.001
    n_steps = 10

    print(f"网格: {m_grid}x{m_grid}, 内部节点: {disc.size}")
    print(f"热扩散系数: {th.diffusivity:.4e} m^2/s")
    print(f"时间步: {dt} s, 步数: {n_steps}")

    T_hist, max_hist = simulate_coupled_pde_ode(
        T0, disc, th, dx, dt, n_steps, method="explicit"
    )

    print(f"\n模拟结果:")
    print(f"  初始最高温度: {max_hist[0]:.2f} K")
    print(f"  最终最高温度: {max_hist[-1]:.2f} K")
    print(f"  温度升高: {max_hist[-1] - max_hist[0]:.2f} K")

    return {
        "T_final": T_hist[-1],
        "max_temp": max_hist[-1],
        "disc": disc,
    }


def run_cvt_optimization() -> dict:
    """运行 CVT 最优传感器布置."""
    print_header("CVT 最优传感器布置")

    n_gen = 5
    n_iter = 20
    n_samp = 100

    print(f"1D CVT (Gaussian 密度):")
    gen1d, eng1d = cvt_1d_lloyd(
        n_gen, n_iter, n_samp, gaussian_density_1d, init_mode=2
    )
    print(f"  生成器: {[f'{g:.3f}' for g in gen1d]}")
    print(f"  最终能量: {eng1d[-1]:.6f}")

    print(f"\n2D CVT (Gaussian 密度):")
    gen2d, eng2d = cvt_2d_lloyd(
        n_gen, 10, 400, gaussian_density_2d, domain=(0.0, 1.0, 0.0, 1.0)
    )
    print(f"  生成器: {[(f'{g[0]:.3f}', f'{g[1]:.3f}') for g in gen2d]}")
    print(f"  最终能量: {eng2d[-1]:.6f}")

    return {"gen1d": gen1d, "gen2d": gen2d}


def run_multi_target_planning() -> dict:
    """运行多目标路径规划."""
    print_header("多目标路径规划 (TSP)")

    coords = [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 0.0),
        (2.0, 1.5),
        (1.0, 2.0),
        (0.0, 1.5),
    ]
    print(f"目标坐标: {coords}")

    path, length = solve_tsp(coords, method="hybrid")
    print(f"最优路径: {path}")
    print(f"路径长度: {length:.4f}")

    return {"path": path, "length": length}


def run_stage_decision() -> dict:
    """运行动态规划阶段决策."""
    print_header("动态规划阶段决策 (火箭发动机选择)")

    thrusts = [5000.0, 8000.0, 3000.0, 6000.0]
    fuel_cons = [200, 400, 150, 300]
    fuel_cap = 700
    I_sp = [300.0, 320.0, 280.0, 310.0]

    print(f"发动机推力: {thrusts}")
    print(f"燃料消耗: {fuel_cons}")
    print(f"燃料容量: {fuel_cap}")
    print(f"比冲: {I_sp}")

    sel, dv = rocket_stage_selection(thrusts, fuel_cons, fuel_cap, I_sp)
    print(f"\n选中阶段: {sel}")
    print(f"总 Delta_v: {dv:.2f} m/s")

    return {"selected": sel, "delta_v": dv}


def run_robustness_analysis(nominal_data: dict) -> dict:
    """运行多场景鲁棒性分析."""
    print_header("多场景鲁棒性分析")

    x0 = nominal_data["x0"]
    x_target = nominal_data["x_target"]
    sc = nominal_data["sc"]

    results, report = run_all_scenarios(
        x0, x_target, sc, T_final=500.0, n_steps=50, n_scenarios=3
    )

    print(f"\n鲁棒性报告:")
    print(f"  场景数: {report.n_scenarios}")
    print(f"  平均代价: {report.mean_cost:.4e}")
    print(f"  标准差: {report.std_cost:.4e}")
    print(f"  收敛率: {report.convergence_rate:.2%}")
    print(f"  风险指标: {report.risk_index:.4e}")
    print(f"  分位数: {report.quantiles}")

    return {"results": results, "report": report}


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    """主入口."""
    print()
    print("*" * 70)
    print("*  Pontryagin 多阶段随机最优控制框架")
    print("*  Stochastic Pontryagin Multi-Stage Optimal Control Framework")
    print("*" * 70)
    print()
    print("本框架融合 15 个种子项目的核心算法, 解决能量受限航天器在温度场")
    print("PDE 约束下的多目标轨迹优化问题。")
    print()
    print("科学领域: 数学优化 - 最优控制与 Pontryagin 原理")
    print("难度等级: 博士级")

    t_start = time.time()

    # 1. 模块自检
    run_module_self_checks()

    # 2. 标称轨迹求解
    nominal_data = solve_nominal_trajectory()

    # 3. PDE 热扩散模拟
    pde_data = run_pde_simulation()

    # 4. CVT 传感器布置
    cvt_data = run_cvt_optimization()

    # 5. 多目标路径规划
    path_data = run_multi_target_planning()

    # 6. 阶段决策
    stage_data = run_stage_decision()

    # 7. 鲁棒性分析
    robustness_data = run_robustness_analysis(nominal_data)

    t_end = time.time()

    # 总结
    print_header("执行总结")
    print(f"总耗时: {t_end - t_start:.2f} s")
    print(f"标称轨迹收敛: {nominal_data['result'].converged}")
    print(f"PDE 最终温度: {pde_data['max_temp']:.2f} K")
    print(f"CVT 1D 能量: {cvt_data['gen1d']}")
    print(f"TSP 路径长度: {path_data['length']:.4f}")
    print(f"选中阶段: {stage_data['selected']}")
    print(f"鲁棒性收敛率: {robustness_data['report'].convergence_rate:.2%}")

    print()
    print("*" * 70)
    print("*  所有模块执行完成!")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()
