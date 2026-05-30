
import numpy as np
import os
import sys


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




    print("\n[阶段 1] 脑血管网格生成")
    print("-" * 40)


    p2d, t2d = generate_willis_ring_mesh(h0=0.18, iteration_max=40)
    print(f"  2D Willis 环网格: 节点数 = {p2d.shape[0]}, 三角形数 = {t2d.shape[0]}")


    p3d, t3d = generate_cerebral_vessel_3d(h0=0.35, iteration_max=25)
    print(f"  3D 脑血管网格: 节点数 = {p3d.shape[0]}, 四面体数 = {t3d.shape[0]}")




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


    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_mesh")
    os.makedirs(tmp_dir, exist_ok=True)
    triangle_node_write(os.path.join(tmp_dir, "willis_nodes.txt"), p2d)
    triangle_element_write(os.path.join(tmp_dir, "willis_elements.txt"), t2d)
    p2d_read = triangle_node_read(os.path.join(tmp_dir, "willis_nodes.txt"))
    t2d_read = triangle_element_read(os.path.join(tmp_dir, "willis_elements.txt"))
    print(f"  网格 I/O 验证: 读取节点 {p2d_read.shape[0]}, 读取单元 {t2d_read.shape[0]}")




    print("\n[阶段 3] 血压波传播与压力场求解")
    print("-" * 40)


    def ic_gaussian(x):
        return np.exp(-20.0 * x ** 2)

    U_burgers, x_burgers = burgers_time_inviscid_godunov(
        ic_gaussian, nx=101, nt=200, t_max=0.5, bc_type='periodic'
    )
    print(f"  Burgers 模拟: 空间节点 {len(x_burgers)}, 时间步 {U_burgers.shape[0]}")
    print(f"  初始压力峰值 = {np.max(U_burgers[0, :]):.4f}")
    print(f"  t=0.5 时压力峰值 = {np.max(U_burgers[-1, :]):.4f}")


    theta = np.linspace(0, 2 * np.pi, 100)
    r_test = 0.5
    x_test = r_test * np.cos(theta)
    y_test = r_test * np.sin(theta)
    u_lap, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x_test, y_test, a=-1.0, b=10.0)
    laplacian_check = uxx + uyy
    print(f"  Laplace 径向解验证: max|∇²u| = {np.max(np.abs(laplacian_check)):.2e}")
    print(f"  壁面压力 = {np.mean(u_lap):.4f} ± {np.std(u_lap):.4f} mmHg")


    dt_wk = 0.01
    n_steps_wk = 500
    t_wk = np.arange(n_steps_wk) * dt_wk
    Q_in_wk = 5e-5 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 1.17 * t_wk))
    P_out_wk = windkessel_pressure_outflow(Q_in_wk, R=1.2e9, C=2.5e-11, dt=dt_wk, n_steps=n_steps_wk)
    print(f"  Windkessel 模型: 平均出口压力 = {np.mean(P_out_wk):.2f} Pa")




    raise NotImplementedError("HOLE_3A: 血管网络压力场计算流程待实现")




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


    def c0_radial(r):
        return np.zeros_like(r)

    C_rad, r_rad, t_rad = oxygen_diffusion_2d_radial(
        C0=c0_radial, nr=41, nt=300, t_max=1.5,
        D=2.0e-5, lam=0.8, k_met=0.3, C_max=1.0,
        R_tissue=0.05, R_cap=0.003
    )
    print(f"  2D 径向氧扩散: 毛细血管壁 C = {C_rad[-1, 0]:.4f}")
    print(f"  2D 径向氧扩散: 组织远端 C = {C_rad[-1, -1]:.4f}")


    V_mm = michaelis_menten_oxygen_consumption(C_1d[-1, :], V_max=0.5, K_m=0.05)
    print(f"  Michaelis-Menten 消耗: 均值 V = {np.mean(V_mm):.4f}")


    r_krogh = np.linspace(0.003, 0.05, 50)
    P_krogh = krogh_oxygen_tension(r_krogh, R_t=0.05, R_c=0.003,
                                    P_c=100.0, P_tissue=20.0, D_t=2.0e-5, M0=0.01)
    print(f"  Krogh 解析解: 平均组织氧分压 = {np.mean(P_krogh):.2f} mmHg")




    print("\n[阶段 5] 血管重构动力学")
    print("-" * 40)




    raise NotImplementedError("HOLE_3B: 血管重构动力学模拟流程待实现")


    r_children = murray_branching_law(2.5, theta=0.0, n_branches=2)
    print(f"  Murray 对称二分: 子血管半径 = {r_children[0]:.4f} mm")
    theta1, theta2 = branch_angle_from_murray(2.5, r_children[0], r_children[1])
    print(f"  对应分支角度: θ1 = {np.degrees(theta1):.2f}°, θ2 = {np.degrees(theta2):.2f}°")


    tau_w = wall_shear_stress(radius=2.5e-3, Q=1.0e-6)
    print(f"  壁面剪切应力 τ_w = {tau_w:.4f} Pa")




    print("\n[阶段 6] 血细胞动力学与统计变异性")
    print("-" * 40)


    mu_app = fahraeus_lindqvist_viscosity(diameter_um=50.0, Hct=0.45)
    print(f"  Fahraeus-Lindqvist 粘度 (D=50μm): μ_app = {mu_app:.4e} Pa·s")
    mu_app_cap = fahraeus_lindqvist_viscosity(diameter_um=5.0, Hct=0.45)
    print(f"  Fahraeus-Lindqvist 粘度 (D=5μm): μ_app = {mu_app_cap:.4e} Pa·s")


    Hct_d1, Hct_d2 = hematocrit_partition(
        Q_parent=1.0e-6, Q_daughter1=0.6e-6, Q_daughter2=0.4e-6,
        Hct_parent=0.45, D_parent=100.0, D_d1=80.0, D_d2=70.0
    )
    print(f"  分叉处 Hct 分配: Hct_d1 = {Hct_d1:.4f}, Hct_d2 = {Hct_d2:.4f}")


    flow_samples = blood_flow_variability(mean_flow=5.0e-5, std_fraction=0.15, n_samples=1000)
    print(f"  血流变异性采样: 均值 = {np.mean(flow_samples):.4e}, 标准差 = {np.std(flow_samples):.4e}")


    np.random.seed(42)
    cell_strength = np.array([0.9, 0.8, 0.85, 0.75, 0.7, 0.95, 0.6, 0.8])
    cell_stats = blood_cell_competition_simulation(cell_strength, n_games=500)
    print(f"  血细胞分叉竞争 (500次): 最强群胜率 = {cell_stats[np.argmax(cell_strength)]}/{500}")


    t_pulse = np.linspace(0.0, 2.0, 200)
    Q_pulse = stochastic_pulsatile_flow(Q_mean=5.0e-5, f_heart=1.17, t_array=t_pulse, amplitude=0.3)
    print(f"  脉动血流: 均值 Q = {np.mean(Q_pulse):.4e}, 峰值/谷值 = {np.max(Q_pulse):.4e}/{np.min(Q_pulse):.4e}")




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




    print("\n[阶段 8] 血流周期性与循环检测")
    print("-" * 40)


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




    print("\n[阶段 9] 血流积分与随机采样")
    print("-" * 40)


    vol_blood = integrate_blood_volume_tetrahedral(p3d, t3d)
    print(f"  3D 血管区域血容量 (四面体积分) = {vol_blood:.4f} mm³")


    int_000 = tetrahedron01_monomial_integral([0, 0, 0])
    int_100 = tetrahedron01_monomial_integral([1, 0, 0])
    print(f"  单位四面体积分验证: ∫1 dV = {int_000:.6f} (理论 1/6 = {1/6:.6f})")
    print(f"  单位四面体积分验证: ∫x dV = {int_100:.6f} (理论 1/24 = {1/24:.6f})")


    cross_points = sample_vascular_cross_section(
        n_points=500, radius=1.0, center=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0])
    )
    print(f"  血管截面随机采样: 生成 {len(cross_points)} 个点")


    def parabolic_velocity(r):
        r_safe = np.clip(r, 0.0, 1.0)
        return 2.0 * (1.0 - r_safe ** 2)

    Q_mc = monte_carlo_flow_rate_integral(n_samples=2000, radius=1.0, velocity_profile_func=parabolic_velocity)
    print(f"  蒙特卡洛流量估计 (抛物线剖面) = {Q_mc:.4f} (理论 π ≈ {np.pi:.4f})")




    print("\n" + "=" * 80)
    print("模拟完成。所有核心模块已验证通过。")
    print("=" * 80)


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
