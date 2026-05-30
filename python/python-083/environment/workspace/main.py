
import numpy as np
import sys
import time


np.random.seed(2024)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    t_start = time.time()
    print("\n" + "=" * 70)
    print("  结构力学：结构拓扑优化与增材制造 — 多尺度协同设计平台")
    print("  Multi-scale Topology Optimization & AM Process Co-design")
    print("=" * 70)




    print_section("0. 有限元网格生成与预处理")

    from fem_core import (generate_rectangular_mesh, triangulation_t3_to_t4,
                          build_vtoe, solve_fem_system, compute_element_stress)

    lx, ly = 10.0, 5.0
    nx, ny = 30, 15
    node_xy, element_node = generate_rectangular_mesh(lx, ly, nx, ny)
    n_nodes = node_xy.shape[0]
    n_elements = element_node.shape[0]
    print(f"  T3 网格: {n_nodes} 节点, {n_elements} 单元")


    node_xy4, element_node4 = triangulation_t3_to_t4(node_xy, element_node)
    print(f"  T4 升级后: {node_xy4.shape[0]} 节点, {element_node4.shape[0]} 单元")


    vtoe_ptr, vtoe = build_vtoe(element_node, n_nodes)
    print(f"  VTOE 映射构建完成，平均节点度 = {len(vtoe)/n_nodes:.2f}")




    print_section("1. SIMP 宏观拓扑优化")

    from topology_optimizer import simp_topology_optimization

    E0 = 110e3
    nu = 0.34
    volfrac = 0.4
    n_dof = n_nodes * 2


    F = np.zeros(n_dof, dtype=np.float64)
    bc_nodes = []
    bc_values = []
    tol = 1e-6


    left_nodes = np.where(node_xy[:, 0] < tol)[0]
    for nd in left_nodes:
        bc_nodes.extend([2*nd, 2*nd+1])
        bc_values.extend([0.0, 0.0])


    right_nodes = np.where(np.abs(node_xy[:, 0] - lx) < tol)[0]
    force_per_node = -5000.0 / max(1, len(right_nodes))
    for nd in right_nodes:
        F[2*nd + 1] += force_per_node

    bc_nodes = np.array(bc_nodes, dtype=np.int32)
    bc_values = np.array(bc_values, dtype=np.float64)

    rho_opt, U_opt, hist_C, hist_V = simp_topology_optimization(
        node_xy, element_node, F, bc_nodes, bc_values,
        E0, nu, volfrac, n_iter=25, r_min=0.3*min(lx/nx, ly/ny),
        plane_stress=True, use_filter=True, use_projection=False)

    print(f"  初始柔度: {hist_C[0]:.4e} N·mm")
    print(f"  最终柔度: {hist_C[-1]:.4e} N·mm")
    print(f"  最终体积分数: {hist_V[-1]:.4f}")
    print(f"  柔度降低比例: {(1 - hist_C[-1]/hist_C[0])*100:.2f}%")




    print_section("2. 微观胞元等效属性均匀化")

    from homogenization import (compute_effective_properties,
                                 effective_property_by_boundary_integral)

    porosity_range = np.linspace(0.0, 0.7, 8)
    print("  孔隙率 | Mori-Tanaka E_eff (GPa) | Self-Consistent E_eff (GPa)")
    print("  " + "-" * 60)
    for f in porosity_range:
        E_mt, nu_mt = compute_effective_properties(E0/1e3, nu, f, "mori_tanaka")
        E_sc, nu_sc = compute_effective_properties(E0/1e3, nu, f, "self_consistent")
        print(f"  {f:.2f}   |      {E_mt:.3f}              |      {E_sc:.3f}")


    k_ratio = effective_property_by_boundary_integral(0.3, n_harmonics=6)
    print(f"\n  边界积分法验证（孔隙率 0.3）: k_eff/k_m = {k_ratio:.4f}")




    print_section("3. CVT 晶格结构生成与打印路径编码")

    from lattice_generator import generate_am_lattice

    lattice = generate_am_lattice(
        domain_bounds=(0.0, lx, 0.0, ly),
        n_cells=80, target_density=0.25,
        use_complex_boundary=True, seed=42)

    print(f"  实际胞元数: {len(lattice['centers'])}")
    print(f"  实际相对密度: {lattice['relative_density']:.4f}")
    print(f"  前 3 个胞元打印路径词长度: {[len(w) for w in lattice['print_path_words'][:3]]}")




    print_section("4. LPBF 热-流耦合工艺模拟")

    from thermal_process import (simulate_layer_deposition_thermal,
                                  ethier_steinman_solution,
                                  estimate_melt_pool_size)


    thermal_result = simulate_layer_deposition_thermal(
        n_layers=5, layer_thickness=0.05, scan_speed=800.0,
        laser_power=200.0, thermal_diffusivity=6.7e-6,
        dt_per_layer=100)

    print(f"  沉积层数: 5")
    print(f"  平均峰值温度: {np.mean(thermal_result['peak_temps']):.1f} K")
    print(f"  最大峰值温度: {np.max(thermal_result['peak_temps']):.1f} K")
    print(f"  平均冷却速率: {np.mean(np.abs(thermal_result['cooling_rates'])):.1e} K/s")


    mp = estimate_melt_pool_size(
        laser_power=200.0, scan_speed=800.0, absorptivity=0.3,
        thermal_diffusivity=6.7e-6, melting_temp=1928.0, ambient_temp=400.0)
    print(f"  估算熔池尺寸: 宽 {mp['width_m']*1e6:.1f} μm, "
          f"深 {mp['depth_m']*1e6:.1f} μm, "
          f"长 {mp['length_m']*1e6:.1f} μm")
    print(f"  Peclet 数: {mp['peclet']:.3f}")


    n_grid = 11
    x = np.linspace(-0.5, 0.5, n_grid)
    y = np.linspace(-0.5, 0.5, n_grid)
    z = np.linspace(-0.5, 0.5, n_grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    U_ns, V_ns, W_ns, P_ns = ethier_steinman_solution(X, Y, Z, T=0.0)
    div_u = np.gradient(U_ns, x, axis=0) + np.gradient(V_ns, y, axis=1) + np.gradient(W_ns, z, axis=2)
    print(f"  NS 精确解验证: max |∇·u| = {np.max(np.abs(div_u)):.2e}")
    print(f"                 max |U| = {np.max(np.abs(U_ns)):.4f}")




    print_section("5. 相场断裂损伤分析")

    from phase_field_damage import compute_crack_driving_force


    G_c = 2.1e3
    l_0 = 0.15

    psi_pos, phi, J_int = compute_crack_driving_force(
        node_xy, element_node, U_opt, E0, nu, G_c, l_0, plane_stress=True)

    n_cracked = np.sum(phi > 0.5)
    print(f"  最大拉伸应变能密度: {np.max(psi_pos):.4e} MPa")
    print(f"  最大相场损伤值: {np.max(phi):.4f}")
    print(f"  损伤单元数 (φ>0.5): {n_cracked} / {n_elements}")
    print(f"  等效 J 积分: {J_int:.4e} N/mm")




    print_section("6. 高维代理模型 (Smolyak 稀疏网格)")

    from surrogate_model import (SmolyakSparseGrid, test_function_oscillatory,
                                  compare_grid_cardinality)

    d_test, q_test = 3, 4
    sg = SmolyakSparseGrid(d_test, q_test)
    sg.sample_function(test_function_oscillatory)
    test_pt = np.array([0.3, 0.5, 0.7])
    approx = sg.interpolate(test_pt)
    exact = test_function_oscillatory(test_pt)
    n_sparse, n_full = compare_grid_cardinality(d_test, q_test)
    print(f"  维度 d={d_test}, 层级 q={q_test}")
    print(f"  稀疏网格采样点数: {n_sparse}")
    print(f"  全张量积采样点数: {n_full}")
    print(f"  压缩比: {n_full / n_sparse:.1f}x")
    print(f"  测试点插值误差: {abs(approx - exact):.6e}")




    print_section("7. 增材制造参数不确定性量化")

    from uncertainty_quantification import (sample_hypercube_uniform,
                                             hypercube_distance_statistics,
                                             generate_am_process_parameters,
                                             parameter_to_physical,
                                             sample_discrete_distribution)

    n_mc = 200
    samples_uq = generate_am_process_parameters(n_mc, seed=55)
    dist_stats = hypercube_distance_statistics(samples_uq)
    print(f"  MC 采样数: {n_mc}")
    print(f"  5D 超立方体点对距离统计:")
    print(f"    均值: {dist_stats['mean']:.4f}, 标准差: {dist_stats['std']:.4f}")
    print(f"    中位数: {dist_stats['median']:.4f}")


    powder_pmf = np.array([0.1, 0.25, 0.35, 0.2, 0.1])
    powder_samples = sample_discrete_distribution(powder_pmf, 1000, seed=66)
    empirical_pmf = np.bincount(powder_samples, minlength=5) / 1000.0
    print(f"  粉末粒径离散分布采样验证:")
    print(f"    理论 PMF: {powder_pmf}")
    print(f"    经验 PMF: {empirical_pmf}")




    print_section("8. 多尺度协同设计总结")


    E_eff_lattice, nu_eff_lattice = compute_effective_properties(
        E0/1e3, nu, lattice['relative_density'], "mori_tanaka")
    print(f"  晶格等效杨氏模量: {E_eff_lattice:.3f} GPa")
    print(f"  晶格等效泊松比: {nu_eff_lattice:.3f}")


    rho_material = 4.43e-9
    macro_volume = lx * ly * 1.0
    macro_mass = macro_volume * volfrac * rho_material * 1e9
    lattice_mass = macro_volume * lattice['relative_density'] * rho_material * 1e9
    print(f"  实体结构估算质量: {macro_mass:.3f} g")
    print(f"  晶格填充估算质量: {lattice_mass:.3f} g")
    print(f"  减重比例: {(1 - lattice_mass/macro_mass)*100:.1f}%")


    max_temp = np.max(thermal_result['peak_temps'])
    temp_stress_factor = max_temp / 1928.0
    print(f"  热应力风险指标: {temp_stress_factor:.3f}")


    overhang_risk = np.mean(rho_opt < 0.1)
    manufacturability_score = max(0.0, 1.0 - overhang_risk - temp_stress_factor * 0.3)
    print(f"  制造可行性综合评分: {manufacturability_score:.3f} (0-1)")

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  计算完成，总耗时: {t_elapsed:.2f} 秒")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
