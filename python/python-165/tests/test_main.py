"""
main.py
智能电网潮流优化与暂态稳定性分析综合平台 — 统一入口

零参数运行，自动执行：
    1. 环形-径向混合电网拓扑生成与 Delaunay 三角剖分质量评估
    2. 导纳矩阵构建与牛顿-拉夫逊潮流计算
    3. 暂态稳定性时域仿真（单机及多机摇摆方程）
    4. 经济调度（等微增率准则）与机组组合动态规划
    5. 负荷马尔可夫预测与熵率分析
    6. 状态估计（WLS）与可观测性分析
    7. 可靠性分析、电压稳定裕度与三相不平衡度计算
"""

import numpy as np
import sys

# 确保模块路径
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

    # ============================================================
    # 1. 电网拓扑生成与 Delaunay 三角剖分
    # ============================================================
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

    # 利用 circle_arc_grid 生成额外的中间馈线节点并展示
    extra_nodes = circle_arc_grid(0.0, 0.0, 7.5, 0.0, 360.0, 6)[:-1]
    print(f"   附加中间环节点数: {len(extra_nodes)}")

    # ============================================================
    # 2. 导纳矩阵与潮流计算
    # ============================================================
    print_section("2. 导纳矩阵构建与牛顿-拉夫逊潮流计算")

    edges = grid.get_edge_list()
    n = grid.n_nodes
    n_edges = len(edges)
    # 线路参数（典型配电网参数）
    r_line = np.full(n_edges, 0.02)
    x_line = np.full(n_edges, 0.08)
    b_shunt = np.full(n_edges, 0.01)

    y_bus = build_y_bus(n, edges, r_line, x_line, b_shunt)

    # 定义节点类型：0=PQ, 1=PV, 2=Slack
    bus_types = np.zeros(n, dtype=np.int32)
    bus_types[0] = 2  # Slack
    # 设置部分节点为 PV 发电机
    gen_indices = [1, 4, 8, 11]
    for gi in gen_indices:
        if gi < n:
            bus_types[gi] = 1

    # 初始电压与功率注入
    vm0 = np.ones(n, dtype=np.float64)
    va0 = np.zeros(n, dtype=np.float64)
    p_spec = np.zeros(n, dtype=np.float64)
    q_spec = np.zeros(n, dtype=np.float64)

    # 负荷：PQ 节点（除 slack 和 PV 外），小负荷保证收敛
    for i in range(n):
        if bus_types[i] == 0:
            p_spec[i] = -0.06 - 0.02 * ((i * 7) % 5)
            q_spec[i] = -0.02 - 0.01 * ((i * 3) % 4)

    # 计算总负荷
    total_load_p = np.sum(np.abs(p_spec))
    n_pv = int(np.sum(bus_types == 1))
    # PV 节点注入 = 总负荷 / PV数 + 网损裕度
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

    # ============================================================
    # 3. 暂态稳定性分析
    # ============================================================
    print_section("3. 暂态稳定性时域仿真（摇摆方程）")

    swing = SwingEquation(H=5.0, D=2.0, f_base=50.0,
                          E_prime=1.1, V_inf=1.0, X=0.5)
    P_m = 0.8
    delta_0 = np.arcsin(P_m * swing.X / (swing.E_prime * swing.V_inf))
    omega_0 = swing.omega_s

    # 无故障仿真
    res_stable = swing.simulate(
        t_span=(0.0, 5.0), dt=0.01, P_m=P_m,
        delta0=delta_0, omega0=omega_0
    )
    print(f"   稳态运行: δ_0 = {np.degrees(delta_0):.2f}°, ω_s = {swing.omega_s:.2f} rad/s")
    print(f"   无故障仿真终点 δ = {np.degrees(res_stable['delta'][-1]):.2f}°")

    # 三相短路故障仿真（0.1s 故障，0.2s 切除）
    res_fault = swing.simulate(
        t_span=(0.0, 5.0), dt=0.01, P_m=P_m,
        delta0=delta_0, omega0=omega_0,
        fault_time=(0.1, 0.25), X_fault=2.0
    )
    print(f"   故障后仿真终点 δ = {np.degrees(res_fault['delta'][-1]):.2f}°")
    print(f"   暂态稳定判定: {res_fault['stable']}")

    # 临界切除角
    delta_cr = swing.critical_clearing_angle(P_m)
    if delta_cr is not None:
        print(f"   临界切除角 δ_cr = {np.degrees(delta_cr):.2f}°")

    # 多机系统仿真
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

    # ============================================================
    # 4. 经济调度与机组组合
    # ============================================================
    print_section("4. 最优经济调度与机组组合动态规划")

    # 3台机组
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

    # 动态规划机组组合
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

    # 聚合 DP 卷积法
    demand_series = np.array([120.0, 140.0, 160.0, 150.0, 130.0, 110.0])
    res_agg = uc.solve_aggregated_dp(demand_series, ed)
    print(f"   多机组聚合 DP 可行性: {res_agg['all_feasible']}")

    # 多项式卷积演示
    poly_demo = polynomial_multiply(np.array([1, 1, 1]), np.array([1, 2, 3]))
    print(f"   生成函数卷积示例: {poly_demo}")

    # ============================================================
    # 5. 负荷马尔可夫预测
    # ============================================================
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

    # ============================================================
    # 6. 状态估计与可观测性
    # ============================================================
    print_section("6. WLS 状态估计与可观测性分析")

    # 为验证状态估计算法，使用标准 5-bus 测试系统
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

    # 先执行潮流计算获得优质初值，再运行状态估计
    p_spec_se = np.array([0.0, 0.4, -0.45, -0.4, -0.6])
    q_spec_se = np.array([0.0, 0.0, -0.15, -0.05, -0.1])
    vm_se0 = np.array([1.06, 1.0, 1.0, 1.0, 1.0])
    va_se0 = np.zeros(n_se)

    pf_se = PowerFlowSolver(y_bus_se, bus_types_se, vm_se0, va_se0, p_spec_se, q_spec_se)
    res_pf_se = pf_se.solve()
    print(f"   潮流预计算收敛: {res_pf_se['converged']} (为状态估计提供初值)")

    se = WeightedLeastSquaresSE(n_se, y_bus_se, bus_types_se)
    # 构造精确量测（验证算法在无噪声理想条件下的收敛性）
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

    # 可观测性分析
    obs = ObservabilityAnalysis(n_se)
    H_test = np.random.randn(2 * n_se, 2 * n_se)
    obs_res = obs.check_observability(H_test)
    print(f"   测试矩阵秩: {obs_res['rank']}/{obs_res['n_states']}")
    print(f"   可观测性: {obs_res['observable']}")
    print(f"   秩亏: {obs_res['deficiency']}")

    # ============================================================
    # 7. 可靠性、电压稳定与三相不平衡
    # ============================================================
    print_section("7. 可靠性分析与电压稳定性评估")

    # 线路可靠性
    line_lengths = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    rel = LineReliability(line_lengths, lambda0=0.05, mu=8760.0)
    print(f"   线路可用率均值: {rel.availability.mean():.6f}")
    series_r = rel.series_reliability([0, 1, 2, 3])
    print(f"   示例串联路径可靠性: {series_r:.6f}")

    # 电压稳定裕度
    vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
    p_max = vsm.max_power_limit()
    margin = vsm.voltage_margin(P_operating=0.8)
    print(f"   最大功率极限: {p_max:.4f} p.u.")
    print(f"   当前运行裕度: {margin*100:.2f}%")
    pv = vsm.pv_curve(n_points=20)
    print(f"   PV 曲线采样: P_max={pv['P_max']:.4f}, V_nose={pv['V_high'][-1]:.4f}")

    # 三相不平衡
    tpu = ThreePhaseUnbalance(
        Va=complex(1.0, 0.0),
        Vb=complex(-0.49, -0.87),
        Vc=complex(-0.51, 0.85)
    )
    print(f"   三相不平衡度: {tpu.unbalance_factor():.4f}%")
    tri = tpu.phasor_triangle_analysis()
    print(f"   相量三角形最小角: {tri['min_angle_deg']:.2f}°")

    # ============================================================
    # 8. 数值工具验证
    # ============================================================
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

# ================================================================
# 测试用例（31个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: circle_arc_grid 返回形状正确 (n, 2) ----
pts = circle_arc_grid(0.0, 0.0, 5.0, 0.0, 90.0, 5)
assert pts.shape == (5, 2), '[TC01] circle_arc_grid shape FAILED'

# ---- TC02: circle_arc_grid 所有点到圆心距离为 r ----
import numpy as np
pts = circle_arc_grid(1.0, 2.0, 3.0, 0.0, 360.0, 10)
dists = np.sqrt((pts[:, 0] - 1.0)**2 + (pts[:, 1] - 2.0)**2)
assert np.allclose(dists, 3.0, atol=1e-12), '[TC02] circle_arc_grid radius FAILED'

# ---- TC03: polynomial_multiply 卷积正确性 ----
p = np.array([1.0, 2.0, 3.0])
q = np.array([4.0, 5.0])
result = polynomial_multiply(p, q)
expected = np.convolve(p, q)
assert np.allclose(result, expected), '[TC03] polynomial_multiply FAILED'

# ---- TC04: diff2_center sin''(x) = -sin(x) 数值验证 ----
x_val = np.pi / 6.0
d2 = diff2_center(np.sin, x_val, h=1e-4)
assert abs(d2 + np.sin(x_val)) < 1e-6, '[TC04] diff2_center FAILED'

# ---- TC05: triangle_area_2d 直角三角形面积 ----
from utils import triangle_area_2d
a = np.array([0.0, 0.0])
b = np.array([3.0, 0.0])
c = np.array([0.0, 4.0])
area = triangle_area_2d(a, b, c)
assert abs(area - 6.0) < 1e-12, '[TC05] triangle_area_2d FAILED'

# ---- TC06: triangle_angles 内角和为 π ----
from utils import triangle_angles
a = np.array([0.0, 0.0])
b = np.array([1.0, 0.0])
c = np.array([0.5, 0.866])
angles = triangle_angles(a, b, c)
assert abs(np.sum(angles) - np.pi) < 1e-12, '[TC06] triangle_angles sum FAILED'

# ---- TC07: rk4_step 求解 y'=y, y(0)=1 精度验证 ----
from utils import rk4_step
def exp_ode(t, y):
    return np.array([y[0]])
y0 = np.array([1.0])
y1 = rk4_step(exp_ode, 0.0, y0, 0.1)
expected_y1 = np.exp(0.1)
assert abs(y1[0] - expected_y1) < 1e-6, '[TC07] rk4_step FAILED'

# ---- TC08: i4mat_rref 单位矩阵保持不变 ----
from utils import i4mat_rref
I = np.eye(4, dtype=np.int64)
rref = i4mat_rref(I.copy())
assert np.allclose(rref, I), '[TC08] i4mat_rref identity FAILED'

# ---- TC09: GridTopology.generate_ring_radial_topology 确定性输出 ----
import numpy as np
from grid_topology import GridTopology
np.random.seed(42)
g1 = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=3.0, r_outer=6.0)
np.random.seed(42)
g2 = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=3.0, r_outer=6.0)
assert np.allclose(g1.nodes, g2.nodes), '[TC09] GridTopology deterministic FAILED'

# ---- TC10: GridTopology 边数至少为节点数-1（连通性） ----
from grid_topology import GridTopology
import numpy as np
np.random.seed(123)
grid = GridTopology.generate_ring_radial_topology(n_ring=5, n_radial=1, r_inner=4.0, r_outer=8.0)
edges = grid.get_edge_list()
assert len(edges) >= grid.n_nodes - 1, '[TC10] GridTopology connectivity FAILED'

# ---- TC11: GridTopology 网格质量指标非负 ----
from grid_topology import GridTopology
import numpy as np
np.random.seed(99)
grid = GridTopology.generate_ring_radial_topology(n_ring=4, n_radial=1, r_inner=5.0, r_outer=10.0)
quality = grid.compute_mesh_quality()
assert quality['min_angle_deg'] >= 0.0, '[TC11] GridTopology min_angle_deg FAILED'
assert quality['mean_area'] >= 0.0, '[TC11] GridTopology mean_area FAILED'

# ---- TC12: SparseMatrix add + mv 与 to_dense 一致性 ----
from sparse_matrix import SparseMatrix
import numpy as np
sm = SparseMatrix(3)
sm.add(0, 0, 4.0); sm.add(0, 1, -1.0); sm.add(0, 2, -2.0)
sm.add(1, 0, -1.0); sm.add(1, 1, 4.0); sm.add(1, 2, -1.0)
sm.add(2, 0, -2.0); sm.add(2, 1, -1.0); sm.add(2, 2, 4.0)
x = np.array([1.0, 2.0, 3.0])
y_sparse = sm.mv(x)
y_dense = sm.to_dense() @ x
assert np.allclose(y_sparse, y_dense), '[TC12] SparseMatrix mv vs dense FAILED'

# ---- TC13: conjugate_gradient 求解 SPD 系统 Ax = b ----
from sparse_matrix import SparseMatrix, conjugate_gradient
import numpy as np
sm = SparseMatrix(3)
sm.add(0, 0, 4.0); sm.add(0, 1, 1.0)
sm.add(1, 0, 1.0); sm.add(1, 1, 4.0); sm.add(1, 2, 1.0)
sm.add(2, 1, 1.0); sm.add(2, 2, 4.0)
b = np.array([1.0, 2.0, 3.0])
x_cg = conjugate_gradient(sm, b, tol=1e-12)
assert np.allclose(sm.mv(x_cg), b, atol=1e-8), '[TC13] conjugate_gradient FAILED'

# ---- TC14: build_y_bus 2-bus 导纳矩阵验证 ----
from power_flow import build_y_bus
import numpy as np
edges = np.array([[0, 1]], dtype=np.int32)
r_line = np.array([0.01])
x_line = np.array([0.1])
y_bus = build_y_bus(2, edges, r_line, x_line, None)
y_series = 1.0 / complex(0.01, 0.1)
assert abs(y_bus[0, 0] - y_series) < 1e-12, '[TC14] Y_bus diagonal FAILED'
assert abs(y_bus[0, 1] + y_series) < 1e-12, '[TC14] Y_bus off-diagonal FAILED'

# ---- TC15: EconomicDispatch solve_lambda 出力总和等于需求 ----
from optimal_dispatch import EconomicDispatch
import numpy as np
a = np.array([0.02, 0.015, 0.025])
b = np.array([10.0, 12.0, 8.0])
c = np.array([100.0, 120.0, 90.0])
p_min = np.array([10.0, 10.0, 5.0])
p_max = np.array([100.0, 120.0, 80.0])
ed = EconomicDispatch(a, b, c, p_min, p_max)
res = ed.solve_lambda(150.0)
assert abs(res['total_generation'] - 150.0) < 1e-4, '[TC15] EconomicDispatch total FAILED'

# ---- TC16: EconomicDispatch incremental_cost 公式验证 ----
from optimal_dispatch import EconomicDispatch
import numpy as np
a = np.array([0.01, 0.02])
b = np.array([5.0, 8.0])
c = np.array([50.0, 60.0])
p_min = np.array([0.0, 0.0])
p_max = np.array([200.0, 200.0])
ed = EconomicDispatch(a, b, c, p_min, p_max)
p_test = np.array([100.0, 80.0])
ic = ed.incremental_cost(p_test)
assert abs(ic[0] - (2.0 * 0.01 * 100.0 + 5.0)) < 1e-10, '[TC16] incremental_cost FAILED'
assert abs(ic[1] - (2.0 * 0.02 * 80.0 + 8.0)) < 1e-10, '[TC16] incremental_cost FAILED'

# ---- TC17: LoadMarkovModel fit 转移矩阵行和为 1 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
row_sums = model.P.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-9), '[TC17] Markov row sums FAILED'

# ---- TC18: LoadMarkovModel predict 概率分布和为 1 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
pred = model.predict(current_state=2, n_steps=3)
assert abs(pred.sum() - 1.0) < 1e-9, '[TC18] Markov predict sum FAILED'

# ---- TC19: LoadMarkovModel entropy_rate 非负 ----
import numpy as np
from load_markov import LoadMarkovModel
np.random.seed(42)
load_data = np.random.default_rng(42).normal(100, 20, 200)
model = LoadMarkovModel(n_states=5)
model.fit(load_data)
H = model.entropy_rate()
assert H >= 0.0, '[TC19] entropy_rate non-negative FAILED'

# ---- TC20: SwingEquation electrical_power 已知角度验证 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
Pe_90 = swing.electrical_power(np.pi / 2.0)
assert abs(Pe_90 - 1.1 * 1.0 / 0.5 * 1.0) < 1e-12, '[TC20] electrical_power FAILED'

# ---- TC21: SwingEquation critical_clearing_angle 在 δ_0 与 δ_max 之间 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
P_m = 0.8
P_max = 1.1 * 1.0 / 0.5
delta_0 = np.arcsin(P_m / P_max)
delta_cr = swing.critical_clearing_angle(P_m)
assert delta_cr is not None, '[TC21] critical_clearing_angle None FAILED'
assert delta_0 < delta_cr < np.pi - delta_0, '[TC21] critical_clearing_angle range FAILED'

# ---- TC22: SwingEquation simulate 无故障仿真稳定 ----
from transient_stability import SwingEquation
import numpy as np
swing = SwingEquation(H=5.0, D=2.0, E_prime=1.1, V_inf=1.0, X=0.5)
P_m = 0.5
delta_0 = np.arcsin(P_m * swing.X / (swing.E_prime * swing.V_inf))
omega_0 = swing.omega_s
res = swing.simulate(t_span=(0.0, 2.0), dt=0.01, P_m=P_m, delta0=delta_0, omega0=omega_0)
assert res['stable'], '[TC22] SwingEquation stable FAILED'

# ---- TC23: MultiMachineStability simulate 返回正确形状 ----
from transient_stability import MultiMachineStability
import numpy as np
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
res_mm = mm.simulate(t_span=(0.0, 1.0), dt=0.05, P_m=P_m_mm, delta0=delta0_mm, omega0=omega0_mm)
assert len(res_mm['t']) == 21, '[TC23] MultiMachine t length FAILED'
assert res_mm['delta'].shape[1] == 3, '[TC23] MultiMachine delta shape FAILED'
assert res_mm['omega'].shape[1] == 3, '[TC23] MultiMachine omega shape FAILED'

# ---- TC24: LineReliability 可用率在 [0,1] 范围内 ----
from reliability import LineReliability
import numpy as np
line_lengths = np.array([5.0, 10.0, 15.0])
rel = LineReliability(line_lengths, lambda0=0.1, mu=8760.0)
assert np.all(rel.availability >= 0.0), '[TC24] availability >= 0 FAILED'
assert np.all(rel.availability <= 1.0), '[TC24] availability <= 1 FAILED'

# ---- TC25: LineReliability series_reliability ≤ 1 ----
from reliability import LineReliability
import numpy as np
line_lengths = np.array([5.0, 5.0, 5.0, 5.0])
rel = LineReliability(line_lengths, lambda0=0.05, mu=8760.0)
sr = rel.series_reliability([0, 1, 2, 3])
assert 0.0 < sr <= 1.0, '[TC25] series_reliability range FAILED'

# ---- TC26: VoltageStabilityMargin max_power_limit 为正 ----
from reliability import VoltageStabilityMargin
vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
p_max = vsm.max_power_limit()
assert p_max > 0.0, '[TC26] max_power_limit positive FAILED'

# ---- TC27: VoltageStabilityMargin voltage_margin 返回 [0,1] ----
from reliability import VoltageStabilityMargin
vsm = VoltageStabilityMargin(E=1.1, X=0.5, Q=0.2)
margin = vsm.voltage_margin(P_operating=0.5)
assert 0.0 <= margin <= 1.0, '[TC27] voltage_margin range FAILED'

# ---- TC28: ThreePhaseUnbalance 平衡三相系统不平衡度 ≈ 0 ----
from reliability import ThreePhaseUnbalance
import numpy as np
alpha = np.exp(1j * 2.0 * np.pi / 3.0)
Va_bal = complex(1.0, 0.0)
Vb_bal = Va_bal * alpha**2
Vc_bal = Va_bal * alpha
tpu_bal = ThreePhaseUnbalance(Va_bal, Vb_bal, Vc_bal)
uf_bal = tpu_bal.unbalance_factor()
assert uf_bal < 1e-6, '[TC28] balanced unbalance ≈ 0 FAILED'

# ---- TC29: ObservabilityAnalysis 满秩矩阵可观测 ----
from state_estimation import ObservabilityAnalysis
import numpy as np
obs = ObservabilityAnalysis(n_bus=4)
H_full = np.random.default_rng(42).normal(0, 1, (8, 8))
obs_res = obs.check_observability(H_full)
assert obs_res['observable'], '[TC29] observability FAILED'
assert obs_res['deficiency'] == 0, '[TC29] observability deficiency FAILED'

# ---- TC30: diff2_center 边界检查 h 必须为正 ----
import numpy as np
try:
    diff2_center(np.sin, 0.0, h=0.0)
    assert False, '[TC30] diff2_center h=0 should raise FAILED'
except ValueError:
    pass

# ---- TC31: UnitCommitmentDP solve_single_unit_dp 返回6时段调度 ----
from optimal_dispatch import UnitCommitmentDP
import numpy as np
uc = UnitCommitmentDP(
    n_gen=3, T=6,
    startup_cost=np.array([50.0, 60.0, 40.0]),
    shutdown_cost=np.array([20.0, 25.0, 15.0]),
    min_up=np.array([2, 2, 1]),
    min_down=np.array([2, 2, 1])
)
ed_cost_on = np.array([500.0, 450.0, 480.0, 520.0, 550.0, 510.0])
res_uc = uc.solve_single_unit_dp(gen_idx=0, ed_cost_on=ed_cost_on, ed_cost_off=0.0)
assert len(res_uc['schedule']) == 6, '[TC31] UC schedule length FAILED'
assert res_uc['total_cost'] > 0.0, '[TC31] UC total_cost positive FAILED'

print('\n全部 31 个测试通过!\n')
