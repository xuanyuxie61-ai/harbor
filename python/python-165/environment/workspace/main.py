
import numpy as np
import sys


sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from grid_topology import GridTopology
from power_flow import PowerFlowSolver, build_y_bus
from transient_stability import SwingEquation, MultiMachineStability
from optimal_dispatch import EconomicDispatch, UnitCommitmentDP
from load_markov import load_forecast_example
from state_estimation import WeightedLeastSquaresSE, ObservabilityAnalysis
from reliability import LineReliability, VoltageStabilityMargin, ThreePhaseUnbalance
from utils import diff2_center, polynomial_multiply, circle_arc_grid


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    print("=" * 70)
    print("  智能电网潮流优化与暂态稳定性分析综合平台")
    print("  Integrated Smart Grid Power Flow & Transient Stability Platform")
    print("=" * 70)




    print_section("1. 电网拓扑生成与 Delaunay 三角剖分质量评估")

    grid = GridTopology.generate_ring_radial_topology(
        n_ring=6, n_radial=1, r_inner=5.0, r_outer=10.0
    )
    print(f"   节点数: {grid.n_nodes}")
    print(f"   三角单元数: {grid.n_elements}")
    print(f"   边数: {len(grid.get_edge_list())}")

    quality = grid.compute_mesh_quality()
    print(f"   网格质量指标:")
    for k, v in quality.items():
        print(f"      {k}: {v:.6f}")


    extra_nodes = circle_arc_grid(0.0, 0.0, 7.5, 0.0, 360.0, 6)[:-1]
    print(f"   附加中间环节点数: {len(extra_nodes)}")




    print_section("2. 导纳矩阵构建与牛顿-拉夫逊潮流计算")

    edges = grid.get_edge_list()
    n = grid.n_nodes
    n_edges = len(edges)

    r_line = np.full(n_edges, 0.02)
    x_line = np.full(n_edges, 0.08)
    b_shunt = np.full(n_edges, 0.01)

    y_bus = build_y_bus(n, edges, r_line, x_line, b_shunt)


    bus_types = np.zeros(n, dtype=np.int32)
    bus_types[0] = 2

    gen_indices = [1, 4, 8, 11]
    for gi in gen_indices:
        if gi < n:
            bus_types[gi] = 1


    vm0 = np.ones(n, dtype=np.float64)
    va0 = np.zeros(n, dtype=np.float64)
    p_spec = np.zeros(n, dtype=np.float64)
    q_spec = np.zeros(n, dtype=np.float64)


    for i in range(n):
        if bus_types[i] == 0:
            p_spec[i] = -0.06 - 0.02 * ((i * 7) % 5)
            q_spec[i] = -0.02 - 0.01 * ((i * 3) % 4)


    total_load_p = np.sum(np.abs(p_spec))
    n_pv = int(np.sum(bus_types == 1))

    gen_p = total_load_p / max(n_pv, 1) + 0.03
    for i in range(n):
        if bus_types[i] == 1:
            p_spec[i] = gen_p
            q_spec[i] = 0.0

    pf = PowerFlowSolver(y_bus, bus_types, vm0, va0, p_spec, q_spec)
    result_pf = pf.solve(tol=1e-8, max_iter=20)

    print(f"   潮流收敛: {result_pf['converged']}")
    print(f"   迭代次数: {result_pf['iterations']}")
    print(f"   最终失配范数: {result_pf['mismatch']:.2e}")
    print(f"   Slack 节点电压: {result_pf['vm'][0]:.4f} p.u.")
    print(f"   PV 节点电压: {result_pf['vm'][1]:.4f} p.u.")
    print(f"   电压范围: [{result_pf['vm'].min():.4f}, {result_pf['vm'].max():.4f}] p.u.")




    print_section("3. 暂态稳定性时域仿真（摇摆方程）")

    swing = SwingEquation(H=5.0, D=2.0, f_base=50.0,
                          E_prime=1.1, V_inf=1.0, X=0.5)
    P_m = 0.8
    delta_0 = np.arcsin(P_m * swing.X / (swing.E_prime * swing.V_inf))
    omega_0 = swing.omega_s


    res_stable = swing.simulate(
        t_span=(0.0, 5.0), dt=0.01, P_m=P_m,
        delta0=delta_0, omega0=omega_0
    )
    print(f"   稳态运行: δ_0 = {np.degrees(delta_0):.2f}°, ω_s = {swing.omega_s:.2f} rad/s")
    print(f"   无故障仿真终点 δ = {np.degrees(res_stable['delta'][-1]):.2f}°")


    res_fault = swing.simulate(
        t_span=(0.0, 5.0), dt=0.01, P_m=P_m,
        delta0=delta_0, omega0=omega_0,
        fault_time=(0.1, 0.25), X_fault=2.0
    )
    print(f"   故障后仿真终点 δ = {np.degrees(res_fault['delta'][-1]):.2f}°")
    print(f"   暂态稳定判定: {res_fault['stable']}")


    delta_cr = swing.critical_clearing_angle(P_m)
    if delta_cr is not None:
        print(f"   临界切除角 δ_cr = {np.degrees(delta_cr):.2f}°")


    print("   --- 多机系统暂态仿真 ---")
    n_gen = 3
    H_mm = np.array([5.0, 4.0, 6.0])
    D_mm = np.array([2.0, 1.5, 2.5])
    E_mm = np.array([1.1, 1.05, 1.08])
    Y_red = np.array([
        [0.5 - 5.0j, 0.2 + 1.0j, 0.1 + 0.5j],
        [0.2 + 1.0j, 0.6 - 6.0j, 0.15 + 0.8j],
        [0.1 + 0.5j, 0.15 + 0.8j, 0.55 - 5.5j]
    ])
    mm = MultiMachineStability(n_gen, H_mm, D_mm, E_mm, Y_red)
    P_m_mm = np.array([0.8, 0.6, 0.7])
    delta0_mm = np.array([0.3, 0.2, 0.25])
    omega0_mm = np.full(n_gen, mm.omega_s)
    res_mm = mm.simulate(
        t_span=(0.0, 3.0), dt=0.01,
        P_m=P_m_mm, delta0=delta0_mm, omega0=omega0_mm
    )
    print(f"   3机系统仿真完成，最终 δ = {res_mm['delta'][-1]}")




    print_section("4. 最优经济调度与机组组合动态规划")


    a = np.array([0.02, 0.015, 0.025])
    b = np.array([10.0, 12.0, 8.0])
    c = np.array([100.0, 120.0, 90.0])
    p_min = np.array([10.0, 10.0, 5.0])
    p_max = np.array([100.0, 120.0, 80.0])

    ed = EconomicDispatch(a, b, c, p_min, p_max)
    P_demand = 150.0
    res_ed = ed.solve_lambda(P_demand)
    print(f"   负荷需求: {P_demand:.1f} MW")
    print(f"   最优出力: {res_ed['pg']}")
    print(f"   系统 λ: {res_ed['lambda']:.4f} $/MWh")
    print(f"   总发电成本: {res_ed['total_cost']:.2f} $/h")


    uc = UnitCommitmentDP(
        n_gen=3, T=6,
        startup_cost=np.array([50.0, 60.0, 40.0]),
        shutdown_cost=np.array([20.0, 25.0, 15.0]),
        min_up=np.array([2, 2, 1]),
        min_down=np.array([2, 2, 1])
    )
    ed_cost_on = np.array([500.0, 450.0, 480.0, 520.0, 550.0, 510.0])
    res_uc = uc.solve_single_unit_dp(gen_idx=0, ed_cost_on=ed_cost_on, ed_cost_off=0.0)
    print(f"   单机组 DP 最优启停序列: {res_uc['schedule']}")
    print(f"   单机组 DP 最小成本: {res_uc['total_cost']:.2f}")


    demand_series = np.array([120.0, 140.0, 160.0, 150.0, 130.0, 110.0])
    res_agg = uc.solve_aggregated_dp(demand_series, ed)
    print(f"   多机组聚合 DP 可行性: {res_agg['all_feasible']}")


    poly_demo = polynomial_multiply(np.array([1, 1, 1]), np.array([1, 2, 3]))
    print(f"   生成函数卷积示例: {poly_demo}")




    print_section("5. 负荷马尔可夫预测模型")

    lf = load_forecast_example()
    model = lf["model"]
    print(f"   历史负荷样本数: {len(lf['load_series'])}")
    print(f"   马尔可夫状态数: {model.n_states}")
    print(f"   稳态分布: {model.steady_state}")
    print(f"   熵率: {model.entropy_rate():.4f} bits")
    print(f"   10步转移相关度: {model.n_step_correlation(10):.4f}")

    pred = model.predict(current_state=2, n_steps=5)
    print(f"   从状态2出发的5步预测分布: {pred}")




    print_section("6. WLS 状态估计与可观测性分析")


    n_se = 5
    lines_se = [
        (0, 1, 0.02, 0.06, 0.0), (0, 2, 0.08, 0.24, 0.0),
        (1, 2, 0.06, 0.18, 0.0), (1, 3, 0.06, 0.18, 0.0),
        (1, 4, 0.04, 0.12, 0.0), (2, 3, 0.01, 0.03, 0.0),
        (3, 4, 0.08, 0.24, 0.0)
    ]
    edges_se = np.array([[l[0], l[1]] for l in lines_se])
    r_se = np.array([l[2] for l in lines_se])
    x_se = np.array([l[3] for l in lines_se])
    b_se = np.array([l[4] for l in lines_se])
    y_bus_se = build_y_bus(n_se, edges_se, r_se, x_se, b_se)
    bus_types_se = np.array([2, 1, 0, 0, 0], dtype=np.int32)


    p_spec_se = np.array([0.0, 0.4, -0.45, -0.4, -0.6])
    q_spec_se = np.array([0.0, 0.0, -0.15, -0.05, -0.1])
    vm_se0 = np.array([1.06, 1.0, 1.0, 1.0, 1.0])
    va_se0 = np.zeros(n_se)

    pf_se = PowerFlowSolver(y_bus_se, bus_types_se, vm_se0, va_se0, p_spec_se, q_spec_se)
    res_pf_se = pf_se.solve()
    print(f"   潮流预计算收敛: {res_pf_se['converged']} (为状态估计提供初值)")

    se = WeightedLeastSquaresSE(n_se, y_bus_se, bus_types_se)

    measurements = []
    for i in range(n_se):
        measurements.append(('V_mag', i, float(res_pf_se['vm'][i]), 0.01))
    for i in range(n_se):
        if bus_types_se[i] == 0:
            measurements.append(('P_inj', i, float(p_spec_se[i]), 0.02))
            measurements.append(('Q_inj', i, float(q_spec_se[i]), 0.02))

    res_se = se.solve(measurements, res_pf_se['vm'], res_pf_se['va'], tol=1e-6, max_iter=20)
    print(f"   状态估计收敛: {res_se['converged']}")
    print(f"   迭代次数: {res_se['iterations']}")
    print(f"   目标函数 J: {res_se['J']:.6e}")
    print(f"   估计电压范围: [{res_se['vm'].min():.4f}, {res_se['vm'].max():.4f}]")
    print(f"   估计相角范围(deg): [{np.degrees(res_se['va']).min():.2f}, {np.degrees(res_se['va']).max():.2f}]")


    obs = ObservabilityAnalysis(n_se)
    H_test = np.random.randn(2 * n_se, 2 * n_se)
    obs_res = obs.check_observability(H_test)
    print(f"   测试矩阵秩: {obs_res['rank']}/{obs_res['n_states']}")
    print(f"   可观测性: {obs_res['observable']}")
    print(f"   秩亏: {obs_res['deficiency']}")




    print_section("7. 可靠性分析与电压稳定性评估")


    line_lengths = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    rel = LineReliability(line_lengths, lambda0=0.05, mu=8760.0)
    print(f"   线路可用率均值: {rel.availability.mean():.6f}")
    series_r = rel.series_reliability([0, 1, 2, 3])
    print(f"   示例串联路径可靠性: {series_r:.6f}")


    vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
    p_max = vsm.max_power_limit()
    margin = vsm.voltage_margin(P_operating=0.8)
    print(f"   最大功率极限: {p_max:.4f} p.u.")
    print(f"   当前运行裕度: {margin*100:.2f}%")
    pv = vsm.pv_curve(n_points=20)
    print(f"   PV 曲线采样: P_max={pv['P_max']:.4f}, V_nose={pv['V_high'][-1]:.4f}")


    tpu = ThreePhaseUnbalance(
        Va=complex(1.0, 0.0),
        Vb=complex(-0.49, -0.87),
        Vc=complex(-0.51, 0.85)
    )
    print(f"   三相不平衡度: {tpu.unbalance_factor():.4f}%")
    tri = tpu.phasor_triangle_analysis()
    print(f"   相量三角形最小角: {tri['min_angle_deg']:.2f}°")




    print_section("8. 数值工具验证")

    def test_func(x):
        return np.sin(x)

    d2 = diff2_center(test_func, np.pi / 4, h=1e-4)
    print(f"   sin''(π/4) 数值 = {d2:.6f}, 理论 = {-np.sin(np.pi/4):.6f}")

    print("\n" + "=" * 70)
    print("  所有模块执行完毕，系统运行正常。")
    print("=" * 70)


if __name__ == "__main__":
    main()
