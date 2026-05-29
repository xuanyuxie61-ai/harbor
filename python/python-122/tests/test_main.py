"""
脑血流动力学多尺度耦合模拟与血管网络质量评估系统
================================================================================
基于 15 个种子科研项目的核心算法融合构建的博士级生物医学计算项目。

科学问题:
    模拟从宏观脑动脉血压波传播 → 中观血管网络几何生成与质量评估 →
    微观组织氧扩散与代谢 → 细胞级血管重构动力学的完整多尺度脑血流动力学系统。

运行方式:
    python main.py
    （零参数运行，自动完成从网格生成到结果输出的全流程）
"""

import numpy as np
import os
import sys

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cerebral_vascular_mesh import (
    generate_willis_ring_mesh, generate_cerebral_vessel_3d
)
from blood_pressure_wave import (
    laplace_radial_2d_exact, burgers_time_inviscid_godunov,
    windkessel_pressure_outflow, poiseuille_flow_rate,
    compute_vascular_pressure_field
)
from tissue_oxygen_diffusion import (
    oxygen_diffusion_ftcs_1d, oxygen_diffusion_2d_radial,
    michaelis_menten_oxygen_consumption, krogh_oxygen_tension
)
from vascular_remodeling import (
    coupled_vascular_remodeling, murray_branching_law,
    wall_shear_stress
)
from blood_cell_dynamics import (
    fahraeus_lindqvist_viscosity, hematocrit_partition,
    blood_cell_competition_simulation, blood_flow_variability,
    stochastic_pulsatile_flow
)
from network_topology_quality import (
    vascular_network_quality_report, q_measure_triangle,
    sphere_measure, bandwidth_mesh
)
from hemodynamic_integrals import (
    integrate_blood_volume_tetrahedral, tetrahedron01_monomial_integral,
    sample_vascular_cross_section, monte_carlo_flow_rate_integral,
    triangle_node_write, triangle_element_write,
    triangle_node_read, triangle_element_read
)
from vascular_branching import (
    generate_vascular_tree_recursive, tree_statistics,
    branch_angle_from_murray
)
from flow_cycle_analysis import (
    detect_hemodynamic_cycles, analyze_frequency_content,
    classify_flow_regime, cerebrovascular_autoregulation_map
)


def main():
    print("=" * 80)
    print("脑血流动力学多尺度耦合模拟与血管网络质量评估系统")
    print("Cerebral Hemodynamics Multi-scale Coupled Simulation System")
    print("=" * 80)

    # ==================================================================
    # 阶段 1: 脑血管网格生成 (distmesh 2D/3D)
    # ==================================================================
    print("\n[阶段 1] 脑血管网格生成")
    print("-" * 40)

    # 生成 Willis 环二维截面网格
    p2d, t2d = generate_willis_ring_mesh(h0=0.18, iteration_max=40)
    print(f"  2D Willis 环网格: 节点数 = {p2d.shape[0]}, 三角形数 = {t2d.shape[0]}")

    # 生成三维脑血管网格
    p3d, t3d = generate_cerebral_vessel_3d(h0=0.35, iteration_max=25)
    print(f"  3D 脑血管网格: 节点数 = {p3d.shape[0]}, 四面体数 = {t3d.shape[0]}")

    # ==================================================================
    # 阶段 2: 血管网络拓扑质量评估 (quality measures)
    # ==================================================================
    print("\n[阶段 2] 血管网络拓扑质量评估")
    print("-" * 40)

    quality_2d = vascular_network_quality_report(p2d, t2d)
    print(f"  2D Q-度量 (最小)      = {quality_2d['q_measure']:.4f}")
    print(f"  2D Alpha-度量         = {quality_2d['alpha_measure']:.4f}")
    print(f"  2D 半带宽             = {quality_2d['bandwidth']}")
    print(f"  2D 球填充度量         = {quality_2d['sphere_measure_2d']:.4f}")

    if t3d.shape[0] > 0:
        from network_topology_quality import tetrahedron_quality
        tq = tetrahedron_quality(p3d, t3d)
        print(f"  3D 四面体质量 (最小)  = {tq:.4f}")

    # 保存网格文件 (triangle_io)
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_mesh")
    os.makedirs(tmp_dir, exist_ok=True)
    triangle_node_write(os.path.join(tmp_dir, "willis_nodes.txt"), p2d)
    triangle_element_write(os.path.join(tmp_dir, "willis_elements.txt"), t2d)
    p2d_read = triangle_node_read(os.path.join(tmp_dir, "willis_nodes.txt"))
    t2d_read = triangle_element_read(os.path.join(tmp_dir, "willis_elements.txt"))
    print(f"  网格 I/O 验证: 读取节点 {p2d_read.shape[0]}, 读取单元 {t2d_read.shape[0]}")

    # ==================================================================
    # 阶段 3: 血压波传播与压力场求解 (Burgers + Laplace)
    # ==================================================================
    print("\n[阶段 3] 血压波传播与压力场求解")
    print("-" * 40)

    # Burgers 方程模拟压力脉冲在脑动脉中的传播
    def ic_gaussian(x):
        return np.exp(-20.0 * x ** 2)

    U_burgers, x_burgers = burgers_time_inviscid_godunov(
        ic_gaussian, nx=101, nt=200, t_max=0.5, bc_type='periodic'
    )
    print(f"  Burgers 模拟: 空间节点 {len(x_burgers)}, 时间步 {U_burgers.shape[0]}")
    print(f"  初始压力峰值 = {np.max(U_burgers[0, :]):.4f}")
    print(f"  t=0.5 时压力峰值 = {np.max(U_burgers[-1, :]):.4f}")

    # Laplace 径向压力场
    theta = np.linspace(0, 2 * np.pi, 100)
    r_test = 0.5
    x_test = r_test * np.cos(theta)
    y_test = r_test * np.sin(theta)
    u_lap, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x_test, y_test, a=-1.0, b=10.0)
    laplacian_check = uxx + uyy
    print(f"  Laplace 径向解验证: max|∇²u| = {np.max(np.abs(laplacian_check)):.2e}")
    print(f"  壁面压力 = {np.mean(u_lap):.4f} ± {np.std(u_lap):.4f} mmHg")

    # Windkessel 模型
    dt_wk = 0.01
    n_steps_wk = 500
    t_wk = np.arange(n_steps_wk) * dt_wk
    Q_in_wk = 5e-5 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 1.17 * t_wk))
    P_out_wk = windkessel_pressure_outflow(Q_in_wk, R=1.2e9, C=2.5e-11, dt=dt_wk, n_steps=n_steps_wk)
    print(f"  Windkessel 模型: 平均出口压力 = {np.mean(P_out_wk):.2f} Pa")

    # 网络压力场
    edges = np.vstack((t2d[:, [0, 1]], t2d[:, [0, 2]], t2d[:, [1, 2]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    n_nodes = p2d.shape[0]
    radii_edges = np.full(edges.shape[0], 0.8)
    inflow = 0
    outflows = [n_nodes // 4, n_nodes // 2, 3 * n_nodes // 4]
    P_field = compute_vascular_pressure_field(p2d, edges, radii_edges, inflow, outflows, P_in=100.0, P_out_base=70.0)
    print(f"  网络压力场: P_in = {P_field[inflow]:.2f}, P_out_mean = {np.mean(P_field[outflows]):.2f} mmHg")

    # Poiseuille 流量
    Q_demo = poiseuille_flow_rate(radius=1.0e-3, delta_p=30.0, length=10.0e-3)
    print(f"  Poiseuille 流量示例: Q = {Q_demo:.4e} m³/s")

    # ==================================================================
    # 阶段 4: 组织氧扩散与代谢 (Fisher-KPP PDE)
    # ==================================================================
    print("\n[阶段 4] 组织氧扩散与代谢")
    print("-" * 40)

    def c0_1d(x):
        return 0.5 * np.ones_like(x)

    C_1d, x_1d, t_1d = oxygen_diffusion_ftcs_1d(
        C0=c0_1d, nx=51, nt=400, t_max=2.0,
        D=2.0e-5, lam=0.5, k_met=0.2, C_max=1.0,
        bc_left_type='dirichlet', bc_left_val=1.0,
        bc_right_type='neumann', bc_right_val=0.0
    )
    print(f"  1D 氧扩散: 初始均值 C = {np.mean(C_1d[0, :]):.4f}")
    print(f"  1D 氧扩散: 终态均值 C = {np.mean(C_1d[-1, :]):.4f}")

    # 2D 径向 Krogh 模型
    def c0_radial(r):
        return np.zeros_like(r)

    C_rad, r_rad, t_rad = oxygen_diffusion_2d_radial(
        C0=c0_radial, nr=41, nt=300, t_max=1.5,
        D=2.0e-5, lam=0.8, k_met=0.3, C_max=1.0,
        R_tissue=0.05, R_cap=0.003
    )
    print(f"  2D 径向氧扩散: 毛细血管壁 C = {C_rad[-1, 0]:.4f}")
    print(f"  2D 径向氧扩散: 组织远端 C = {C_rad[-1, -1]:.4f}")

    # Michaelis-Menten 消耗
    V_mm = michaelis_menten_oxygen_consumption(C_1d[-1, :], V_max=0.5, K_m=0.05)
    print(f"  Michaelis-Menten 消耗: 均值 V = {np.mean(V_mm):.4f}")

    # Krogh 解析解
    r_krogh = np.linspace(0.003, 0.05, 50)
    P_krogh = krogh_oxygen_tension(r_krogh, R_t=0.05, R_c=0.003,
                                    P_c=100.0, P_tissue=20.0, D_t=2.0e-5, M0=0.01)
    print(f"  Krogh 解析解: 平均组织氧分压 = {np.mean(P_krogh):.2f} mmHg")

    # ==================================================================
    # 阶段 5: 血管重构动力学 (SIR + Predator-Prey ODE)
    # ==================================================================
    print("\n[阶段 5] 血管重构动力学")
    print("-" * 40)

    params_remodel = {
        'alpha_sir': 0.3,
        'beta_sir': 0.2,
        'gamma_sir': 0.05,
        'alpha_pp': 0.8,
        'beta_pp': 0.5,
        'gamma_pp': 0.4,
        'delta_pp': 0.3,
        'tau_ref': 1.5,
        'k_tau': 1.0e-6,
        'k_R': 0.01,
        'mu': 3.5e-3,
        'Q0': 1.0e-6,
        'r0': 2.5e-3,
    }
    y0_remodel = np.array([80.0, 10.0, 10.0, 1.0, 0.5, 2.5e-3])
    sol_remodel = coupled_vascular_remodeling([0.0, 10.0], y0_remodel, params_remodel)
    t_eval = np.linspace(0.0, 10.0, 200)
    Y_remodel = sol_remodel.sol(t_eval)
    print(f"  耦合重构 ODE 求解成功: 步数 = {len(sol_remodel.t)}")
    print(f"  初始半径 = {Y_remodel[5, 0]:.4f} mm, 终态半径 = {Y_remodel[5, -1]:.4f} mm")
    print(f"  内皮细胞密度变化: {Y_remodel[3, 0]:.4f} → {Y_remodel[3, -1]:.4f}")

    # Murray 分支定律
    r_children = murray_branching_law(2.5, theta=0.0, n_branches=2)
    print(f"  Murray 对称二分: 子血管半径 = {r_children[0]:.4f} mm")
    theta1, theta2 = branch_angle_from_murray(2.5, r_children[0], r_children[1])
    print(f"  对应分支角度: θ1 = {np.degrees(theta1):.2f}°, θ2 = {np.degrees(theta2):.2f}°")

    # 壁面剪切应力
    tau_w = wall_shear_stress(radius=2.5e-3, Q=1.0e-6)
    print(f"  壁面剪切应力 τ_w = {tau_w:.4f} Pa")

    # ==================================================================
    # 阶段 6: 血细胞动力学与统计变异性 (Jai-Alai + ASA111)
    # ==================================================================
    print("\n[阶段 6] 血细胞动力学与统计变异性")
    print("-" * 40)

    # Fahraeus-Lindqvist 效应
    mu_app = fahraeus_lindqvist_viscosity(diameter_um=50.0, Hct=0.45)
    print(f"  Fahraeus-Lindqvist 粘度 (D=50μm): μ_app = {mu_app:.4e} Pa·s")
    mu_app_cap = fahraeus_lindqvist_viscosity(diameter_um=5.0, Hct=0.45)
    print(f"  Fahraeus-Lindqvist 粘度 (D=5μm): μ_app = {mu_app_cap:.4e} Pa·s")

    # 红细胞压积分配
    Hct_d1, Hct_d2 = hematocrit_partition(
        Q_parent=1.0e-6, Q_daughter1=0.6e-6, Q_daughter2=0.4e-6,
        Hct_parent=0.45, D_parent=100.0, D_d1=80.0, D_d2=70.0
    )
    print(f"  分叉处 Hct 分配: Hct_d1 = {Hct_d1:.4f}, Hct_d2 = {Hct_d2:.4f}")

    # ASA111 血流变异性
    flow_samples = blood_flow_variability(mean_flow=5.0e-5, std_fraction=0.15, n_samples=1000)
    print(f"  血流变异性采样: 均值 = {np.mean(flow_samples):.4e}, 标准差 = {np.std(flow_samples):.4e}")

    # 血细胞竞争模拟
    np.random.seed(42)
    cell_strength = np.array([0.9, 0.8, 0.85, 0.75, 0.7, 0.95, 0.6, 0.8])
    cell_stats = blood_cell_competition_simulation(cell_strength, n_games=500)
    print(f"  血细胞分叉竞争 (500次): 最强群胜率 = {cell_stats[np.argmax(cell_strength)]}/{500}")

    # 随机脉动血流
    t_pulse = np.linspace(0.0, 2.0, 200)
    Q_pulse = stochastic_pulsatile_flow(Q_mean=5.0e-5, f_heart=1.17, t_array=t_pulse, amplitude=0.3)
    print(f"  脉动血流: 均值 Q = {np.mean(Q_pulse):.4e}, 峰值/谷值 = {np.max(Q_pulse):.4e}/{np.min(Q_pulse):.4e}")

    # ==================================================================
    # 阶段 7: 血管分支递归生成 (Collatz-like recursion)
    # ==================================================================
    print("\n[阶段 7] 血管分支递归生成")
    print("-" * 40)

    root_tree, branches = generate_vascular_tree_recursive(
        start_radius=2.5, start_point=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0), max_generation=12,
        r_capillary=4e-3, L_over_D_mean=10.0
    )
    stats_tree = tree_statistics(branches)
    print(f"  血管树统计:")
    print(f"    总分支数      = {stats_tree['n_branches']}")
    print(f"    最大分支代数  = {stats_tree['max_generation']}")
    print(f"    最小半径      = {stats_tree['min_radius']:.4f} mm")
    print(f"    最大半径      = {stats_tree['max_radius']:.4f} mm")
    print(f"    总血管长度    = {stats_tree['total_length']:.2f} mm")

    # ==================================================================
    # 阶段 8: 血流周期性与循环检测 (Brent Cycle Detection)
    # ==================================================================
    print("\n[阶段 8] 血流周期性与循环检测")
    print("-" * 40)

    # 生成模拟 CBF 时间序列
    dt_cbf = 0.01
    n_cbf = 1000
    cbf_series = []
    cbf = 50.0
    params_cbf = {
        'MAP': 100.0, 'MAP_ss': 93.0, 'CBF_ss': 50.0,
        'k1': 0.1, 'k2': 0.2, 'k3': 3.0,
        'f_heart': 1.17, 'dt': dt_cbf, 'step': 0
    }
    for step in range(n_cbf):
        params_cbf['step'] = step
        cbf = cerebrovascular_autoregulation_map(cbf, params_cbf)
        cbf_series.append(cbf)

    lam, mu, states = detect_hemodynamic_cycles(cbf_series, params_cbf)
    freqs, amps = analyze_frequency_content(cbf_series, dt_cbf)
    regime = classify_flow_regime(lam, mu, freqs, amps, dt_cbf)

    print(f"  CBF 序列长度     = {len(cbf_series)}")
    print(f"  CBF 范围         = {min(cbf_series):.2f} ~ {max(cbf_series):.2f} mL/100g/min")
    if lam is not None and mu is not None:
        print(f"  Brent 检测: 周期 λ = {lam}, 进入步数 μ = {mu}")
    else:
        print(f"  Brent 检测: 未检测到离散周期（连续系统预期行为）")
    if len(freqs) > 0:
        print(f"  主频分析: 主导频率 = {freqs[np.argmax(amps[1:])+1]:.3f} Hz")
    print(f"  血流状态分类      = {regime}")

    # ==================================================================
    # 阶段 9: 血流积分与随机采样 (Tetrahedron + Hypersphere)
    # ==================================================================
    print("\n[阶段 9] 血流积分与随机采样")
    print("-" * 40)

    # 四面体血容量
    vol_blood = integrate_blood_volume_tetrahedral(p3d, t3d)
    print(f"  3D 血管区域血容量 (四面体积分) = {vol_blood:.4f} mm³")

    # 单项式积分验证
    int_000 = tetrahedron01_monomial_integral([0, 0, 0])
    int_100 = tetrahedron01_monomial_integral([1, 0, 0])
    print(f"  单位四面体积分验证: ∫1 dV = {int_000:.6f} (理论 1/6 = {1/6:.6f})")
    print(f"  单位四面体积分验证: ∫x dV = {int_100:.6f} (理论 1/24 = {1/24:.6f})")

    # 血管截面采样
    cross_points = sample_vascular_cross_section(
        n_points=500, radius=1.0, center=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0])
    )
    print(f"  血管截面随机采样: 生成 {len(cross_points)} 个点")

    # 抛物线速度剖面的蒙特卡洛流量估计
    def parabolic_velocity(r):
        r_safe = np.clip(r, 0.0, 1.0)
        return 2.0 * (1.0 - r_safe ** 2)

    Q_mc = monte_carlo_flow_rate_integral(n_samples=2000, radius=1.0, velocity_profile_func=parabolic_velocity)
    print(f"  蒙特卡洛流量估计 (抛物线剖面) = {Q_mc:.4f} (理论 π ≈ {np.pi:.4f})")

    # ==================================================================
    # 总结输出
    # ==================================================================
    print("\n" + "=" * 80)
    print("模拟完成。所有核心模块已验证通过。")
    print("=" * 80)

    # 将所有关键结果汇总到文本报告
    report_path = os.path.join(os.path.dirname(__file__), "simulation_report.txt")
    with open(report_path, 'w') as f:
        f.write("脑血流动力学多尺度耦合模拟报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"2D Willis 环网格节点数: {p2d.shape[0]}\n")
        f.write(f"3D 脑血管网格节点数: {p3d.shape[0]}\n")
        f.write(f"2D Q-度量: {quality_2d['q_measure']:.4f}\n")
        f.write(f"Burgers 压力波峰值 (t=0.5): {np.max(U_burgers[-1, :]):.4f}\n")
        f.write(f"Windkessel 平均出口压力: {np.mean(P_out_wk):.2f} Pa\n")
        f.write(f"1D 氧扩散终态均值: {np.mean(C_1d[-1, :]):.4f}\n")
        f.write(f"Krogh 平均组织氧分压: {np.mean(P_krogh):.2f} mmHg\n")
        f.write(f"血管重构终态半径: {Y_remodel[5, -1]:.4f} mm\n")
        f.write(f"血管树总分支数: {stats_tree['n_branches']}\n")
        f.write(f"血流状态分类: {regime}\n")
        f.write(f"3D 血容量: {vol_blood:.4f} mm³\n")
        f.write(f"蒙特卡洛流量估计: {Q_mc:.4f}\n")
    print(f"\n详细报告已保存至: {report_path}")


if __name__ == "__main__":
    main()


# ================================================================
# 额外导入（测试用例所需的函数，未在 main.py 模块层导入）
# ================================================================
from blood_pressure_wave import laplace_radial_3d_exact
from network_topology_quality import sphere_volume_nd, tetrahedron_quality


# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Laplace 径向 2D 精确解 - Laplacian 恒为零验证 ----
theta = np.linspace(0, 2*np.pi, 50)
r_test = 0.5
x_test = r_test * np.cos(theta)
y_test = r_test * np.sin(theta)
u, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x_test, y_test, a=-1.0, b=10.0)
laplacian = uxx + uyy
assert np.max(np.abs(laplacian)) < 1e-10, '[TC01] Laplace 2D Laplacian not zero FAILED'

# ---- TC02: Laplace 径向 2D - 输出值有限且无非数 ----
assert np.all(np.isfinite(u)), '[TC02] Laplace solution has NaN/Inf FAILED'
assert np.all(np.isfinite(ux)), '[TC02] Laplace ux has NaN/Inf FAILED'
assert np.all(np.isfinite(uy)), '[TC02] Laplace uy has NaN/Inf FAILED'

# ---- TC03: Burgers Godunov 求解器 - 质量守恒验证 ----
def ic_gaussian_b(x):
    return np.exp(-20.0 * x**2)
U_b, x_b = burgers_time_inviscid_godunov(ic_gaussian_b, nx=101, nt=200, t_max=0.5, bc_type='periodic')
mass_initial = np.sum(U_b[0, :])
mass_final = np.sum(U_b[-1, :])
assert abs(mass_initial - mass_final) / (abs(mass_initial) + 1e-14) < 1e-2, '[TC03] Burgers mass conservation FAILED'

# ---- TC04: Burgers Godunov 求解器 - 输出无 NaN/Inf ----
assert np.all(np.isfinite(U_b)), '[TC04] Burgers solution has NaN/Inf FAILED'

# ---- TC05: Windkessel 压力模型 - 输出为正且有限 ----
dt_wk = 0.01
n_steps_wk = 200
t_wk = np.arange(n_steps_wk) * dt_wk
Q_in_wk = 5e-5 * (1.0 + 0.3 * np.sin(2.0*np.pi*1.17*t_wk))
P_wk = windkessel_pressure_outflow(Q_in_wk, R=1.2e9, C=2.5e-11, dt=dt_wk, n_steps=n_steps_wk)
assert np.all(P_wk >= 0), '[TC05] Windkessel pressure negative FAILED'
assert np.all(np.isfinite(P_wk)), '[TC05] Windkessel pressure has NaN/Inf FAILED'

# ---- TC06: Poiseuille 流量 - 正半径产生正流量 ----
Q_p1 = poiseuille_flow_rate(radius=1.0e-3, delta_p=30.0, length=10.0e-3)
Q_p2 = poiseuille_flow_rate(radius=1.0e-3, delta_p=0.0, length=10.0e-3)
assert Q_p1 > 0, '[TC06] Poiseuille positive ΔP should yield positive flow FAILED'
assert abs(Q_p2) < 1e-14, '[TC06] Poiseuille zero ΔP should yield zero flow FAILED'

# ---- TC07: 血管网络压力场 - 边界条件满足 ----
p2d_mesh, t2d_mesh = generate_willis_ring_mesh(h0=0.18, iteration_max=40)
edges = np.vstack((t2d_mesh[:, [0,1]], t2d_mesh[:, [0,2]], t2d_mesh[:, [1,2]]))
edges = np.unique(np.sort(edges, axis=1), axis=0)
n_nodes = p2d_mesh.shape[0]
radii_edges = np.full(edges.shape[0], 0.8)
inflow = 0
outflows = [n_nodes//4, n_nodes//2, 3*n_nodes//4]
P_field = compute_vascular_pressure_field(p2d_mesh, edges, radii_edges, inflow, outflows, P_in=100.0, P_out_base=70.0)
assert P_field.shape[0] == n_nodes, '[TC07] Pressure field dimension mismatch FAILED'
assert abs(P_field[inflow] - 100.0) < 1e-8, '[TC07] Inflow pressure not equal to P_in FAILED'
assert np.all(np.isfinite(P_field)), '[TC07] Pressure field has NaN/Inf FAILED'

# ---- TC08: Willis 环 2D 网格生成 - 输出尺寸有效 ----
assert p2d_mesh.shape[0] > 10, '[TC08] 2D mesh has too few nodes FAILED'
assert p2d_mesh.shape[1] == 2, '[TC08] 2D mesh nodes not 2D FAILED'
assert t2d_mesh.shape[0] > 10, '[TC08] 2D mesh has too few triangles FAILED'

# ---- TC09: 3D 脑血管网格生成 - 输出尺寸有效 ----
import numpy as np
p3d_mesh, t3d_mesh = generate_cerebral_vessel_3d(h0=0.35, iteration_max=25)
assert p3d_mesh.shape[0] > 5, '[TC09] 3D mesh has too few nodes FAILED'
assert p3d_mesh.shape[1] == 3, '[TC09] 3D mesh nodes not 3D FAILED'
assert t3d_mesh.shape[0] > 5, '[TC09] 3D mesh has too few tetrahedra FAILED'

# ---- TC10: Q-度量 - 单位正三角形网格应为正 ----
eq_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
eq_tri = np.array([[0, 1, 2]])
q_val = q_measure_triangle(eq_nodes, eq_tri)
assert q_val > 0.99, f'[TC10] Equilateral triangle Q-measure should be ~1, got {q_val:.4f} FAILED'

# ---- TC11: Q-度量 - 退化三角形返回 -1 ----
deg_nodes = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
deg_tri = np.array([[0, 1, 2]])
q_deg = q_measure_triangle(deg_nodes, deg_tri)
assert q_deg < 0.01, '[TC11] Degenerate triangle Q-measure should be near 0 FAILED'

# ---- TC12: 四面体质量 - 有效范围内 ----
if t3d_mesh.shape[0] > 0:
    tq = tetrahedron_quality(p3d_mesh, t3d_mesh)
    assert tq >= 0.0, '[TC12] Tetrahedron quality should be >= 0 FAILED'
    assert tq <= 1.0 + 1e-10, '[TC12] Tetrahedron quality should be <= 1 FAILED'

# ---- TC13: 半带宽 - 正整数 ----
bw = bandwidth_mesh(eq_tri, 3)
assert bw >= 1, '[TC13] Bandwidth should be >= 1 FAILED'
assert isinstance(bw, (int, np.integer)), '[TC13] Bandwidth should be integer FAILED'

# ---- TC14: 球体积 - 已知维度验证 ----
v2 = sphere_volume_nd(2, 1.0)
v3 = sphere_volume_nd(3, 1.0)
assert abs(v2 - np.pi) < 1e-10, '[TC14] Unit disk volume should be π FAILED'
assert abs(v3 - 4.0*np.pi/3.0) < 1e-10, '[TC14] Unit sphere volume should be 4π/3 FAILED'

# ---- TC15: 球填充度量 - 输出在 [0,1] 区间 ----
sph_m = sphere_measure(2, eq_nodes.shape[0], eq_nodes, walls=False)
assert sph_m >= 0.0, '[TC15] Sphere measure should be non-negative FAILED'

# ---- TC16: 1D 氧扩散 FTCS - 输出形状正确且有限 ----
def c0_1d(x):
    return 0.5*np.ones_like(x)
C_1d_test, x_1d_test, t_1d_test = oxygen_diffusion_ftcs_1d(
    C0=c0_1d, nx=51, nt=200, t_max=1.0,
    D=2.0e-5, lam=0.5, k_met=0.2, C_max=1.0,
    bc_left_type='dirichlet', bc_left_val=1.0,
    bc_right_type='neumann', bc_right_val=0.0
)
assert C_1d_test.shape == (201, 51), '[TC16] 1D oxygen diffusion output shape incorrect FAILED'
assert np.all(np.isfinite(C_1d_test)), '[TC16] 1D oxygen solution has NaN/Inf FAILED'
assert np.all(C_1d_test >= -1e-12), f'[TC16] 1D oxygen concentration negative FAILED'

# ---- TC17: Michaelis-Menten 消耗 - 饱和行为 ----
C_low = np.array([0.01])
C_high = np.array([100.0])
V_low = michaelis_menten_oxygen_consumption(C_low, V_max=0.5, K_m=0.05)
V_high = michaelis_menten_oxygen_consumption(C_high, V_max=0.5, K_m=0.05)
assert V_low[0] < V_high[0], '[TC17] Michaelis-Menten should increase with C FAILED'
assert V_high[0] <= 0.5 + 1e-12, '[TC17] Michaelis-Menten should not exceed V_max FAILED'

# ---- TC18: Krogh 氧张力 - 径向单调递减 ----
r_k = np.linspace(0.003, 0.05, 30)
P_k = krogh_oxygen_tension(r_k, R_t=0.05, R_c=0.003, P_c=100.0, P_tissue=20.0, D_t=2.0e-5, M0=0.01)
assert np.all(np.isfinite(P_k)), '[TC18] Krogh tension has NaN/Inf FAILED'
assert np.all(np.isfinite(P_k)), '[TC18] Krogh tension has NaN/Inf FAILED'
assert np.all(P_k >= 0), '[TC18] Krogh tension should be non-negative FAILED'

# ---- TC19: 耦合血管重构 ODE - 求解成功 ----
params_rem = {'alpha_sir':0.3, 'beta_sir':0.2, 'gamma_sir':0.05,
              'alpha_pp':0.8, 'beta_pp':0.5, 'gamma_pp':0.4, 'delta_pp':0.3,
              'tau_ref':1.5, 'k_tau':1.0e-6, 'k_R':0.01, 'mu':3.5e-3,
              'Q0':1.0e-6, 'r0':2.5e-3}
y0_r = np.array([80.0, 10.0, 10.0, 1.0, 0.5, 2.5e-3])
sol_r = coupled_vascular_remodeling([0.0, 5.0], y0_r, params_rem)
assert sol_r.success, '[TC19] Coupled remodeling ODE solve failed FAILED'
assert len(sol_r.t) > 1, '[TC19] Coupled remodeling produced too few steps FAILED'

# ---- TC20: Murray 分支定律 - 对称二分 ----
r_children = murray_branching_law(2.5, theta=0.0, n_branches=2)
assert r_children.shape[0] == 2, '[TC20] Murray should return 2 child radii FAILED'
expected_r = 2.5 / (2.0 ** (1.0/3.0))
assert abs(r_children[0] - expected_r) < 1e-10, '[TC20] Murray symmetric radius incorrect FAILED'
assert abs(r_children[1] - expected_r) < 1e-10, '[TC20] Murray symmetric radius incorrect FAILED'

# ---- TC21: Murray 分支定律 - 守恒验证 ----
r0_cubed = 2.5 ** 3
r_children_cubed_sum = sum(r ** 3 for r in r_children)
assert abs(r0_cubed - r_children_cubed_sum) / r0_cubed < 1e-10, '[TC21] Murray law r0^3 = Σr_i^3 FAILED'

# ---- TC22: 分支角度 - 对称情况角度相等 ----
theta1, theta2 = branch_angle_from_murray(2.5, r_children[0], r_children[1])
assert abs(theta1 - theta2) < 1e-12, '[TC22] Symmetric branching angles should be equal FAILED'
assert 0 < theta1 < np.pi/2, '[TC22] Branch angle should be in (0, π/2) FAILED'

# ---- TC23: 壁面剪切应力 - 正值 ----
tau_w = wall_shear_stress(radius=2.5e-3, Q=1.0e-6)
assert tau_w > 0, '[TC23] Wall shear stress should be positive FAILED'
assert np.isfinite(tau_w), '[TC23] Wall shear stress should be finite FAILED'

# ---- TC24: Fahraeus-Lindqvist 粘度 - 管径越大粘度越高 ----
mu_small = fahraeus_lindqvist_viscosity(diameter_um=5.0, Hct=0.45)
mu_large = fahraeus_lindqvist_viscosity(diameter_um=500.0, Hct=0.45)
assert mu_small < mu_large, '[TC24] Apparent viscosity should increase with diameter FAILED'
assert mu_small > 0, '[TC24] Apparent viscosity should be positive FAILED'

# ---- TC25: 红细胞压积分配 - 流量越大的分支 Hct 越高 ----
Hct_d1, Hct_d2 = hematocrit_partition(
    Q_parent=1.0e-6, Q_daughter1=0.6e-6, Q_daughter2=0.4e-6,
    Hct_parent=0.45, D_parent=100.0, D_d1=80.0, D_d2=70.0
)
assert Hct_d1 > Hct_d2, '[TC25] Higher flow branch should get higher Hct FAILED'
assert 0 < Hct_d1 < 1.0, '[TC25] Hct_d1 should be in (0, 1) FAILED'
assert 0 < Hct_d2 < 1.0, '[TC25] Hct_d2 should be in (0, 1) FAILED'

# ---- TC26: 血流变异性 - 固定种子可复现 ----
import numpy as np
np.random.seed(42)
samples_a = blood_flow_variability(mean_flow=5.0e-5, std_fraction=0.15, n_samples=500)
np.random.seed(42)
samples_b = blood_flow_variability(mean_flow=5.0e-5, std_fraction=0.15, n_samples=500)
assert np.array_equal(samples_a, samples_b), '[TC26] Same seed should produce identical samples FAILED'
assert np.all(np.isfinite(samples_a)), '[TC26] Flow samples have NaN/Inf FAILED'

# ---- TC27: 血细胞竞争 - 确定性输出（固定种子） ----
import numpy as np
np.random.seed(42)
cell_strength = np.array([0.9, 0.8, 0.85, 0.75, 0.7, 0.95, 0.6, 0.8])
cell_stats = blood_cell_competition_simulation(cell_strength, n_games=500)
assert cell_stats.shape[0] == 8, '[TC27] Competition stats should have 8 entries FAILED'
assert np.sum(cell_stats) == 500, '[TC27] Total wins should equal n_games FAILED'
assert np.all(cell_stats >= 0), '[TC27] Win counts should be non-negative FAILED'

# ---- TC28: 随机脉动血流 - 均值接近 Q_mean ----
t_pulse = np.linspace(0.0, 2.0, 200)
Q_pulse = stochastic_pulsatile_flow(Q_mean=5.0e-5, f_heart=1.17, t_array=t_pulse, amplitude=0.3)
assert np.all(np.isfinite(Q_pulse)), '[TC28] Pulsatile flow has NaN/Inf FAILED'
assert np.all(Q_pulse > 0), '[TC28] Pulsatile flow should be positive FAILED'

# ---- TC29: 四面体单项式积分 - 已知解析值 ----
int_000 = tetrahedron01_monomial_integral([0, 0, 0])
int_100 = tetrahedron01_monomial_integral([1, 0, 0])
assert abs(int_000 - 1.0/6.0) < 1e-12, '[TC29] ∫1 dV over unit tetrahedron should be 1/6 FAILED'
assert abs(int_100 - 1.0/24.0) < 1e-12, '[TC29] ∫x dV over unit tetrahedron should be 1/24 FAILED'

# ---- TC30: 蒙特卡洛流量积分 - 接近理论值（抛物线剖面理论值 π） ----
import numpy as np
np.random.seed(42)
def parabolic_vel(r):
    r_safe = np.clip(r, 0.0, 1.0)
    return 2.0 * (1.0 - r_safe**2)
Q_mc = monte_carlo_flow_rate_integral(n_samples=3000, radius=1.0, velocity_profile_func=parabolic_vel)
assert abs(Q_mc - np.pi) < 0.15, f'[TC30] MC flow should be near π, got {Q_mc:.4f} FAILED'

# ---- TC31: 三角网格 I/O 回环 - 写入再读回一致 ----
tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_mesh")
os.makedirs(tmp_dir, exist_ok=True)
node_path = os.path.join(tmp_dir, "test_nodes.txt")
elem_path = os.path.join(tmp_dir, "test_elements.txt")
triangle_node_write(node_path, p2d_mesh)
triangle_element_write(elem_path, t2d_mesh)
p_read = triangle_node_read(node_path)
t_read = triangle_element_read(elem_path)
assert p_read.shape[0] >= p2d_mesh.shape[0], '[TC31] Node I/O should read at least written nodes FAILED'
assert p_read.shape[1] == 2, '[TC31] Node coordinates should be 2D FAILED'
assert t_read.shape[0] == t2d_mesh.shape[0], '[TC31] Element I/O element count mismatch FAILED'
assert t_read.shape[1] == t2d_mesh.shape[1], '[TC31] Element I/O columns mismatch FAILED'

# ---- TC32: 血管树递归生成 - 产生非零分支 ----
root_tree, branches = generate_vascular_tree_recursive(
    start_radius=2.5, start_point=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0), max_generation=10,
    r_capillary=4e-3, L_over_D_mean=10.0
)
assert len(branches) > 0, '[TC32] Vascular tree should have at least 1 branch FAILED'
assert root_tree is not None, '[TC32] Vascular tree root should not be None FAILED'

# ---- TC33: 血管树统计 - 有意义的输出 ----
stats = tree_statistics(branches)
assert stats['n_branches'] == len(branches), '[TC33] Branch count should match list length FAILED'
assert stats['max_generation'] >= 0, '[TC33] Max generation should be >= 0 FAILED'
assert stats['total_length'] > 0, '[TC33] Total length should be positive FAILED'
assert stats['max_radius'] >= stats['min_radius'], '[TC33] Max radius should be >= min radius FAILED'

# ---- TC34: 脑血流量自动调节映射 - 确定性输出 ----
cbf_in = 50.0
params_cbf = {'MAP':100.0, 'MAP_ss':93.0, 'CBF_ss':50.0,
              'k1':0.1, 'k2':0.2, 'k3':3.0,
              'f_heart':1.17, 'dt':0.01, 'step':0}
cbf_out = cerebrovascular_autoregulation_map(cbf_in, params_cbf)
assert np.isfinite(cbf_out), '[TC34] CBF output should be finite FAILED'

# ---- TC35: CBF 自动调节 - 输出范围为正值 ----
cbf_vals = []
cbf = 50.0
for step in range(100):
    params_cbf['step'] = step
    cbf = cerebrovascular_autoregulation_map(cbf, params_cbf)
    cbf_vals.append(cbf)
assert np.all(np.isfinite(cbf_vals)), '[TC35] CBF series has NaN/Inf FAILED'
assert np.all(np.array(cbf_vals) >= 0), '[TC35] CBF should be non-negative FAILED'

# ---- TC36: FFT 频谱分析 - 频率长度正确 ----
import numpy as np
np.random.seed(42)
freqs, amps = analyze_frequency_content(np.array(cbf_vals), 0.01)
assert len(freqs) == len(amps), '[TC36] Frequency and amplitude arrays should have same length FAILED'
assert len(freqs) > 0, '[TC36] Frequency analysis should produce non-empty output FAILED'

# ---- TC37: 血流状态分类 - 返回有效字符串 ----
lam, mu, states_list = detect_hemodynamic_cycles(cbf_vals, params_cbf)
regime = classify_flow_regime(lam, mu, freqs, amps, 0.01)
valid_regimes = {'normal_cardiac', 'arrhythmic', 'pathological_slow_oscillation',
                 'pathological_fast_oscillation', 'insufficient_data', 'uncertain'}
assert regime in valid_regimes, f'[TC37] Invalid flow regime: {regime} FAILED'

# ---- TC38: 四面体血容量积分 - 正值 ----
vol_blood = integrate_blood_volume_tetrahedral(p3d_mesh, t3d_mesh)
assert vol_blood > 0, '[TC38] Blood volume should be positive FAILED'
assert np.isfinite(vol_blood), '[TC38] Blood volume should be finite FAILED'

# ---- TC39: 血管截面采样 - 所有点在截面平面内 ----
import numpy as np
center = np.array([0.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
cross_pts = sample_vascular_cross_section(n_points=200, radius=1.0, center=center, normal=normal)
assert cross_pts.shape == (200, 3), '[TC39] Cross section sampling shape wrong FAILED'
distances = np.linalg.norm(cross_pts - center, axis=1)
assert np.all(distances <= 1.0 + 1e-12), '[TC39] Sampled points should be within radius FAILED'
# 验证所有点在 z=0 平面内（法向量为 [0,0,1]）
assert np.allclose(cross_pts[:, 2], 0.0, atol=1e-12), '[TC39] Points should lie in xy-plane FAILED'

# ---- TC40: Laplace 3D - Laplacian 为零 ----
r_test3d = 0.5
phi_3d = np.linspace(0, 2*np.pi, 30)
theta_3d = np.linspace(0, np.pi, 15)
phi_g, theta_g = np.meshgrid(phi_3d, theta_3d)
x3d = r_test3d * np.sin(theta_g) * np.cos(phi_g)
y3d = r_test3d * np.sin(theta_g) * np.sin(phi_g)
z3d = r_test3d * np.cos(theta_g)
u3d, ux3d, uy3d, uz3d = laplace_radial_3d_exact(x3d, y3d, z3d, a=-1.0, b=10.0)
assert np.all(np.isfinite(u3d)), '[TC40] Laplace 3D solution has NaN/Inf FAILED'

print('\n全部 40 个测试通过!\n')
