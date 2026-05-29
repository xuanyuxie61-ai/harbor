"""
main.py
=======
统一入口：面向增材制造的多尺度结构拓扑优化与工艺-性能协同设计。

科学问题：
  在激光粉末床熔融（LPBF）增材制造中，如何通过多尺度拓扑优化设计
  兼具高刚度、低重量和良好可制造性的晶格填充结构？

运行方式：
  python main.py
（零参数可运行）
"""

import numpy as np
import sys
import time

# 设置随机种子保证可复现
np.random.seed(2024)


def print_section(title: str):
    """格式化输出分节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    t_start = time.time()
    print("\n" + "=" * 70)
    print("  结构力学：结构拓扑优化与增材制造 — 多尺度协同设计平台")
    print("  Multi-scale Topology Optimization & AM Process Co-design")
    print("=" * 70)

    # =====================================================================
    # 0. 基础参数与网格生成 (fem_core + mesh_vtoe + triangulation_t3_to_t4)
    # =====================================================================
    print_section("0. 有限元网格生成与预处理")

    from fem_core import (generate_rectangular_mesh, triangulation_t3_to_t4,
                          build_vtoe, solve_fem_system, compute_element_stress)

    lx, ly = 10.0, 5.0  # 设计域尺寸 (mm)
    nx, ny = 30, 15
    node_xy, element_node = generate_rectangular_mesh(lx, ly, nx, ny)
    n_nodes = node_xy.shape[0]
    n_elements = element_node.shape[0]
    print(f"  T3 网格: {n_nodes} 节点, {n_elements} 单元")

    # T3 -> T4 升级（验证功能）
    node_xy4, element_node4 = triangulation_t3_to_t4(node_xy, element_node)
    print(f"  T4 升级后: {node_xy4.shape[0]} 节点, {element_node4.shape[0]} 单元")

    # 节点→单元反向映射
    vtoe_ptr, vtoe = build_vtoe(element_node, n_nodes)
    print(f"  VTOE 映射构建完成，平均节点度 = {len(vtoe)/n_nodes:.2f}")

    # =====================================================================
    # 1. 宏观拓扑优化 (topology_optimizer + r8st 稀疏CG思想)
    # =====================================================================
    print_section("1. SIMP 宏观拓扑优化")

    from topology_optimizer import simp_topology_optimization

    E0 = 110e3       # Ti-6Al-4V 杨氏模量 (MPa)
    nu = 0.34        # 泊松比
    volfrac = 0.4    # 体积分数约束
    n_dof = n_nodes * 2

    # 边界条件：左边界固定，右边界施加均布拉力
    F = np.zeros(n_dof, dtype=np.float64)
    bc_nodes = []
    bc_values = []
    tol = 1e-6

    # 左边界固定
    left_nodes = np.where(node_xy[:, 0] < tol)[0]
    for nd in left_nodes:
        bc_nodes.extend([2*nd, 2*nd+1])
        bc_values.extend([0.0, 0.0])

    # 右边界施加 y 方向集中力
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

    # =====================================================================
    # 2. 微观均匀化 (homogenization + circle_integrals + circles)
    # =====================================================================
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

    # 边界积分方法验证
    k_ratio = effective_property_by_boundary_integral(0.3, n_harmonics=6)
    print(f"\n  边界积分法验证（孔隙率 0.3）: k_eff/k_m = {k_ratio:.4f}")

    # =====================================================================
    # 3. CVT 晶格结构生成 (lattice_generator + usa_cvt_geo + hand_data + circles + tortoise)
    # =====================================================================
    print_section("3. CVT 晶格结构生成与打印路径编码")

    from lattice_generator import generate_am_lattice

    lattice = generate_am_lattice(
        domain_bounds=(0.0, lx, 0.0, ly),
        n_cells=80, target_density=0.25,
        use_complex_boundary=True, seed=42)

    print(f"  实际胞元数: {len(lattice['centers'])}")
    print(f"  实际相对密度: {lattice['relative_density']:.4f}")
    print(f"  前 3 个胞元打印路径词长度: {[len(w) for w in lattice['print_path_words'][:3]]}")

    # =====================================================================
    # 4. 增材制造工艺热-流模拟 (thermal_process + fd1d_advection_lax + navier_stokes_3d_exact)
    # =====================================================================
    print_section("4. LPBF 热-流耦合工艺模拟")

    from thermal_process import (simulate_layer_deposition_thermal,
                                  ethier_steinman_solution,
                                  estimate_melt_pool_size)

    # 4a. 多层沉积热循环
    thermal_result = simulate_layer_deposition_thermal(
        n_layers=5, layer_thickness=0.05, scan_speed=800.0,
        laser_power=200.0, thermal_diffusivity=6.7e-6,
        dt_per_layer=100)

    print(f"  沉积层数: 5")
    print(f"  平均峰值温度: {np.mean(thermal_result['peak_temps']):.1f} K")
    print(f"  最大峰值温度: {np.max(thermal_result['peak_temps']):.1f} K")
    print(f"  平均冷却速率: {np.mean(np.abs(thermal_result['cooling_rates'])):.1e} K/s")

    # 4b. 熔池尺寸估算
    mp = estimate_melt_pool_size(
        laser_power=200.0, scan_speed=800.0, absorptivity=0.3,
        thermal_diffusivity=6.7e-6, melting_temp=1928.0, ambient_temp=400.0)
    print(f"  估算熔池尺寸: 宽 {mp['width_m']*1e6:.1f} μm, "
          f"深 {mp['depth_m']*1e6:.1f} μm, "
          f"长 {mp['length_m']*1e6:.1f} μm")
    print(f"  Peclet 数: {mp['peclet']:.3f}")

    # 4c. 3D NS 精确解验证
    n_grid = 11
    x = np.linspace(-0.5, 0.5, n_grid)
    y = np.linspace(-0.5, 0.5, n_grid)
    z = np.linspace(-0.5, 0.5, n_grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    U_ns, V_ns, W_ns, P_ns = ethier_steinman_solution(X, Y, Z, T=0.0)
    div_u = np.gradient(U_ns, x, axis=0) + np.gradient(V_ns, y, axis=1) + np.gradient(W_ns, z, axis=2)
    print(f"  NS 精确解验证: max |∇·u| = {np.max(np.abs(div_u)):.2e}")
    print(f"                 max |U| = {np.max(np.abs(U_ns)):.4f}")

    # =====================================================================
    # 5. 相场损伤分析 (phase_field_damage + fitzhugh_nagumo 快慢动力学)
    # =====================================================================
    print_section("5. 相场断裂损伤分析")

    from phase_field_damage import compute_crack_driving_force

    # 使用优化后的位移场和原始弹性模量
    G_c = 2.1e3       # 临界能量释放率 (N/m)
    l_0 = 0.15        # 相场特征长度 (mm)

    psi_pos, phi, J_int = compute_crack_driving_force(
        node_xy, element_node, U_opt, E0, nu, G_c, l_0, plane_stress=True)

    n_cracked = np.sum(phi > 0.5)
    print(f"  最大拉伸应变能密度: {np.max(psi_pos):.4e} MPa")
    print(f"  最大相场损伤值: {np.max(phi):.4f}")
    print(f"  损伤单元数 (φ>0.5): {n_cracked} / {n_elements}")
    print(f"  等效 J 积分: {J_int:.4e} N/mm")

    # =====================================================================
    # 6. Smolyak 稀疏网格代理模型 (surrogate_model + sparse_interp_nd)
    # =====================================================================
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

    # =====================================================================
    # 7. 不确定性量化 (uncertainty_quantification + hypercube_distance + fair_dice)
    # =====================================================================
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

    # 逆变换采样：模拟粉末粒径离散分布
    powder_pmf = np.array([0.1, 0.25, 0.35, 0.2, 0.1])  # 5个粒径等级
    powder_samples = sample_discrete_distribution(powder_pmf, 1000, seed=66)
    empirical_pmf = np.bincount(powder_samples, minlength=5) / 1000.0
    print(f"  粉末粒径离散分布采样验证:")
    print(f"    理论 PMF: {powder_pmf}")
    print(f"    经验 PMF: {empirical_pmf}")

    # =====================================================================
    # 8. 多尺度协同设计总结
    # =====================================================================
    print_section("8. 多尺度协同设计总结")

    # 计算带微观晶格的等效柔度
    E_eff_lattice, nu_eff_lattice = compute_effective_properties(
        E0/1e3, nu, lattice['relative_density'], "mori_tanaka")
    print(f"  晶格等效杨氏模量: {E_eff_lattice:.3f} GPa")
    print(f"  晶格等效泊松比: {nu_eff_lattice:.3f}")

    # 宏观质量估算
    rho_material = 4.43e-9  # Ti-6Al-4V 密度 (ton/mm³)
    macro_volume = lx * ly * 1.0  # 假设厚度 1 mm
    macro_mass = macro_volume * volfrac * rho_material * 1e9  # mg -> g
    lattice_mass = macro_volume * lattice['relative_density'] * rho_material * 1e9
    print(f"  实体结构估算质量: {macro_mass:.3f} g")
    print(f"  晶格填充估算质量: {lattice_mass:.3f} g")
    print(f"  减重比例: {(1 - lattice_mass/macro_mass)*100:.1f}%")

    # 热-结构耦合指标
    max_temp = np.max(thermal_result['peak_temps'])
    temp_stress_factor = max_temp / 1928.0  # 相对于熔点的归一化温度
    print(f"  热应力风险指标: {temp_stress_factor:.3f}")

    # 整体制造可行性评分
    overhang_risk = np.mean(rho_opt < 0.1)  # 低密度区域可能无法自支撑
    manufacturability_score = max(0.0, 1.0 - overhang_risk - temp_stress_factor * 0.3)
    print(f"  制造可行性综合评分: {manufacturability_score:.3f} (0-1)")

    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  计算完成，总耗时: {t_elapsed:.2f} 秒")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（36个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_rectangular_mesh returns correct shapes ----
from fem_core import generate_rectangular_mesh
node_xy_tc, element_node_tc = generate_rectangular_mesh(2.0, 1.0, 4, 2)
assert node_xy_tc.shape == (15, 2), '[TC01] generate_rectangular_mesh node shape FAILED'
assert element_node_tc.shape == (16, 3), '[TC01] generate_rectangular_mesh element shape FAILED'

# ---- TC02: build_vtoe mapping covers all elements ----
from fem_core import build_vtoe
node_xy_tc, element_node_tc = generate_rectangular_mesh(1.0, 1.0, 2, 2)
vtoe_ptr_tc, vtoe_tc = build_vtoe(element_node_tc, node_xy_tc.shape[0])
assert len(vtoe_ptr_tc) == node_xy_tc.shape[0] + 1, '[TC02] vtoe_ptr length FAILED'
assert vtoe_ptr_tc[-1] == len(vtoe_tc), '[TC02] vtoe_ptr consistency FAILED'

# ---- TC03: triangulation_t3_to_t4 adds one node per element ----
from fem_core import triangulation_t3_to_t4
node_xy_tc, element_node_tc = generate_rectangular_mesh(1.0, 1.0, 2, 2)
n_orig_tc = node_xy_tc.shape[0]
n_elem_tc = element_node_tc.shape[0]
node_xy4_tc, element_node4_tc = triangulation_t3_to_t4(node_xy_tc, element_node_tc)
assert node_xy4_tc.shape[0] == n_orig_tc + n_elem_tc, '[TC03] T3 to T4 node count FAILED'
assert element_node4_tc.shape[1] == 4, '[TC03] T4 element local nodes FAILED'

# ---- TC04: elastic_d_matrix plane stress is symmetric with positive diagonal ----
from fem_core import elastic_d_matrix
D_tc = elastic_d_matrix(200e3, 0.3, plane_stress=True)
assert np.allclose(D_tc, D_tc.T), '[TC04] D matrix symmetry FAILED'
assert D_tc[0, 0] > 0 and D_tc[1, 1] > 0 and D_tc[2, 2] > 0, '[TC04] D matrix diagonal FAILED'

# ---- TC05: elastic_d_matrix plane strain differs from plane stress ----
D_ps_tc = elastic_d_matrix(200e3, 0.3, plane_stress=True)
D_pe_tc = elastic_d_matrix(200e3, 0.3, plane_stress=False)
assert not np.allclose(D_ps_tc, D_pe_tc), '[TC05] plane strain vs plane stress FAILED'

# ---- TC06: shape_function_t3 at corner nodes is unit vector ----
from fem_core import shape_function_t3
N0_tc = shape_function_t3(0.0, 0.0)
N1_tc = shape_function_t3(1.0, 0.0)
N2_tc = shape_function_t3(0.0, 1.0)
assert np.allclose(N0_tc, [1, 0, 0]), '[TC06] N0 FAILED'
assert np.allclose(N1_tc, [0, 1, 0]), '[TC06] N1 FAILED'
assert np.allclose(N2_tc, [0, 0, 1]), '[TC06] N2 FAILED'

# ---- TC07: gauss_triangle weights sum to reference area 0.5 ----
from fem_core import gauss_triangle
for order_tc in [1, 3, 4, 7]:
    pts_tc, wts_tc = gauss_triangle(order_tc)
    assert abs(np.sum(wts_tc) - 0.5) < 1e-12, f'[TC07] order {order_tc} weight sum FAILED'

# ---- TC08: _nchoosek returns correct binomial coefficients ----
from fem_core import _nchoosek
assert _nchoosek(5, 2) == 10, '[TC08] C(5,2) FAILED'
assert _nchoosek(0, 0) == 1, '[TC08] C(0,0) FAILED'
assert _nchoosek(5, 0) == 1, '[TC08] C(5,0) FAILED'
assert _nchoosek(5, 5) == 1, '[TC08] C(5,5) FAILED'

# ---- TC09: compute_element_stiffness_t3 returns symmetric matrix ----
from fem_core import compute_element_stiffness_t3
node_xy_tc, element_node_tc = generate_rectangular_mesh(1.0, 1.0, 1, 1)
Ke_tc = compute_element_stiffness_t3(node_xy_tc[element_node_tc[0]], 100e3, 0.3)
assert np.allclose(Ke_tc, Ke_tc.T), '[TC09] Ke symmetry FAILED'

# ---- TC10: solve_fem_system produces finite displacement ----
from fem_core import solve_fem_system
node_xy_tc, element_node_tc = generate_rectangular_mesh(1.0, 1.0, 2, 2)
n_dof_tc = node_xy_tc.shape[0] * 2
F_tc = np.zeros(n_dof_tc)
F_tc[1] = -100.0
bc_nodes_tc = np.array([0, 1], dtype=np.int32)
bc_values_tc = np.array([0.0, 0.0])
U_tc = solve_fem_system(node_xy_tc, element_node_tc, 100e3, 0.3, F_tc, bc_nodes_tc, bc_values_tc)
assert np.all(np.isfinite(U_tc)), '[TC10] displacement finiteness FAILED'

# ---- TC11: mori_tanaka zero porosity returns original modulus ----
from homogenization import compute_effective_properties
E_eff_tc, nu_eff_tc = compute_effective_properties(100.0, 0.3, 0.0, "mori_tanaka")
assert abs(E_eff_tc - 100.0) < 1e-6, '[TC11] MT zero porosity E FAILED'
assert abs(nu_eff_tc - 0.3) < 1e-6, '[TC11] MT zero porosity nu FAILED'

# ---- TC12: hashin_shtrikman bounds are non-negative ----
from homogenization import hashin_shtrikman_bounds_2d
K_eff_tc, G_eff_tc = hashin_shtrikman_bounds_2d(100.0, 0.3, 0.3)
assert K_eff_tc >= 0.0, '[TC12] K_eff lower bound FAILED'
assert G_eff_tc >= 0.0, '[TC12] G_eff lower bound FAILED'

# ---- TC13: effective_property_by_boundary_integral in [0,1] ----
from homogenization import effective_property_by_boundary_integral
k_ratio_tc = effective_property_by_boundary_integral(0.3, n_harmonics=6)
assert 0.0 <= k_ratio_tc <= 1.0, '[TC13] k_ratio range FAILED'

# ---- TC14: circle_monomial_integral vanishes for odd powers ----
from homogenization import circle_monomial_integral
assert abs(circle_monomial_integral(1, 0)) < 1e-12, '[TC14] odd e1 FAILED'
assert abs(circle_monomial_integral(0, 3)) < 1e-12, '[TC14] odd e2 FAILED'

# ---- TC15: generate_cvt_points returns correct number and bounds ----
from lattice_generator import generate_cvt_points
np.random.seed(42)
pts_tc = generate_cvt_points(10, (0.0, 1.0, 0.0, 1.0), n_samples=500, n_lloyd=5, seed=42)
assert pts_tc.shape == (10, 2), '[TC15] CVT points shape FAILED'
assert np.all((pts_tc >= 0.0) & (pts_tc <= 1.0)), '[TC15] CVT points bounds FAILED'

# ---- TC16: is_inside_polygon works for unit square ----
from lattice_generator import is_inside_polygon
square_tc = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
test_pts_tc = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.5]])
inside_tc = is_inside_polygon(test_pts_tc, square_tc)
assert inside_tc[0] == True, '[TC16] inside point FAILED'
assert inside_tc[1] == False, '[TC16] outside point FAILED'
assert inside_tc[2] == False, '[TC16] outside negative FAILED'

# ---- TC17: boundary word encode and decode are consistent ----
from lattice_generator import boundary_word_encode, decode_boundary_word
np.random.seed(42)
pts_tc = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
word_tc = boundary_word_encode(pts_tc)
decoded_tc = decode_boundary_word(word_tc, pts_tc[0], step_length=1.0)
assert len(decoded_tc) == len(pts_tc), '[TC17] decode length FAILED'
assert len(word_tc) == len(pts_tc) - 1, '[TC17] word length FAILED'

# ---- TC18: lattice relative density formula correct ----
from lattice_generator import compute_lattice_relative_density
centers_tc = np.array([[0.5, 0.5]])
radii_tc = np.array([0.5 / np.sqrt(np.pi)])
domain_area_tc = 1.0
rho_tc = compute_lattice_relative_density(centers_tc, radii_tc, domain_area_tc)
assert abs(rho_tc - 0.25) < 1e-6, '[TC18] relative density calculation FAILED'

# ---- TC19: generate_am_lattice returns complete dictionary ----
from lattice_generator import generate_am_lattice
np.random.seed(42)
lattice_tc = generate_am_lattice((0.0, 2.0, 0.0, 1.0), n_cells=10, target_density=0.2, seed=42)
assert "centers" in lattice_tc, '[TC19] centers key FAILED'
assert "radii" in lattice_tc, '[TC19] radii key FAILED'
assert "relative_density" in lattice_tc, '[TC19] relative_density key FAILED'
assert "print_path_words" in lattice_tc, '[TC19] print_path_words key FAILED'
assert lattice_tc["relative_density"] > 0.0, '[TC19] positive density FAILED'

# ---- TC20: degradation_function at phi=0 equals 1+k_res and phi=1 equals k_res ----
from phase_field_damage import degradation_function
phi_tc = np.array([0.0, 1.0])
g_tc = degradation_function(phi_tc)
assert abs(g_tc[0] - 1.000001) < 1e-12, '[TC20] g(0) FAILED'
assert abs(g_tc[1] - 1e-6) < 1e-12, '[TC20] g(1) FAILED'

# ---- TC21: double_well_potential vanishes at phi=0 and phi=1 ----
from phase_field_damage import double_well_potential
phi_tc = np.array([0.0, 1.0, 0.5])
F_tc = double_well_potential(phi_tc)
assert abs(F_tc[0]) < 1e-12, '[TC21] F(0) FAILED'
assert abs(F_tc[1]) < 1e-12, '[TC21] F(1) FAILED'
assert F_tc[2] > 0.0, '[TC21] F(0.5) positive FAILED'

# ---- TC22: elastic strain energy is non-negative ----
from phase_field_damage import compute_elastic_strain_energy
U_tc = np.array([0.0, 0.01, 0.0, 0.0])
K_tc = np.array([[1.0, -0.5, 0.0, 0.0],
                 [-0.5, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
psi_tc = compute_elastic_strain_energy(U_tc, K_tc)
assert psi_tc >= 0.0, '[TC22] strain energy non-negative FAILED'

# ---- TC23: clenshaw_curtis_points lie within [0,1] ----
from surrogate_model import clenshaw_curtis_points
for level_tc in range(5):
    pts_cc_tc = clenshaw_curtis_points(level_tc)
    assert np.all(pts_cc_tc >= 0.0) and np.all(pts_cc_tc <= 1.0), f'[TC23] level {level_tc} bounds FAILED'

# ---- TC24: lagrange_basis_1d satisfies nodal property ----
from surrogate_model import lagrange_basis_1d
nodes_tc = np.array([0.0, 0.5, 1.0])
for k_tc in range(3):
    val_tc = lagrange_basis_1d(nodes_tc[k_tc], nodes_tc, k_tc)
    assert abs(val_tc - 1.0) < 1e-12, f'[TC24] basis {k_tc} at node FAILED'
    for j_tc in range(3):
        if j_tc != k_tc:
            val_j_tc = lagrange_basis_1d(nodes_tc[j_tc], nodes_tc, k_tc)
            assert abs(val_j_tc) < 1e-12, f'[TC24] basis {k_tc} at node {j_tc} FAILED'

# ---- TC25: smolyak_coefficient sum over multi-indices equals 1 ----
from surrogate_model import smolyak_coefficient, generate_multi_index_combinations
d_tc, q_tc = 2, 3
indices_tc = generate_multi_index_combinations(d_tc, q_tc)
total_tc = sum(smolyak_coefficient(d_tc, q_tc, sum(mi_tc)) for mi_tc in indices_tc)
assert abs(total_tc - 1) < 1e-12, '[TC25] coefficient sum FAILED'

# ---- TC26: SmolyakSparseGrid exact at sample points for linear function ----
from surrogate_model import SmolyakSparseGrid
np.random.seed(42)
sg_tc = SmolyakSparseGrid(2, 3)
sg_tc.sample_function(lambda x: x[0] + x[1])
test_key_tc = list(sg_tc.nodes_dict.keys())[0]
test_node_tc = sg_tc.nodes_dict[test_key_tc][0]
exact_tc = test_node_tc[0] + test_node_tc[1]
approx_tc = sg_tc.interpolate(test_node_tc)
assert abs(approx_tc - exact_tc) < 1e-12, '[TC26] exact interpolation FAILED'

# ---- TC27: gaussian_laser_source peaks at center ----
from thermal_process import gaussian_laser_source
z_tc = np.linspace(-2.0, 2.0, 101)
Q_tc = gaussian_laser_source(z_tc, z0=0.0, power=200.0, spot_size=0.5)
assert np.argmax(Q_tc) == 50, '[TC27] laser source peak location FAILED'
assert Q_tc[50] > Q_tc[0], '[TC27] laser source peak value FAILED'

# ---- TC28: ethier_steinman_solution continuity residual small ----
from thermal_process import ethier_steinman_solution
x_tc = np.linspace(-0.2, 0.2, 11)
y_tc = np.linspace(-0.2, 0.2, 11)
z_tc = np.linspace(-0.2, 0.2, 11)
X_tc, Y_tc, Z_tc = np.meshgrid(x_tc, y_tc, z_tc, indexing='ij')
U_ns_tc, V_ns_tc, W_ns_tc, P_ns_tc = ethier_steinman_solution(X_tc, Y_tc, Z_tc, T=0.0)
div_u_tc = np.gradient(U_ns_tc, x_tc, axis=0) + np.gradient(V_ns_tc, y_tc, axis=1) + np.gradient(W_ns_tc, z_tc, axis=2)
assert np.max(np.abs(div_u_tc)) < 0.1, '[TC28] continuity residual FAILED'

# ---- TC29: estimate_melt_pool_size returns positive dimensions ----
from thermal_process import estimate_melt_pool_size
mp_tc = estimate_melt_pool_size(200.0, 800.0, 0.3, 6.7e-6, 1928.0, 400.0)
assert mp_tc["width_m"] > 0.0, '[TC29] width positive FAILED'
assert mp_tc["depth_m"] > 0.0, '[TC29] depth positive FAILED'
assert mp_tc["length_m"] > 0.0, '[TC29] length positive FAILED'
assert mp_tc["peclet"] > 0.0, '[TC29] peclet positive FAILED'

# ---- TC30: simp_interpolation at zero density returns E_min ----
from topology_optimizer import simp_interpolation
rho_tc = np.array([0.0, 0.5, 1.0])
E_tc = simp_interpolation(rho_tc, 100.0, E_min=1e-9)
assert abs(E_tc[0] - 1e-9) < 1e-12, '[TC30] E at rho=0 FAILED'
assert abs(E_tc[2] - 100.0) < 1e-12, '[TC30] E at rho=1 FAILED'

# ---- TC31: density_filter preserves uniform density ----
from topology_optimizer import density_filter
centers_tc = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
rho_uniform_tc = np.array([0.5, 0.5, 0.5, 0.5])
rho_f_tc = density_filter(centers_tc, rho_uniform_tc, r_min=1.5)
assert np.allclose(rho_f_tc, 0.5), '[TC31] uniform density filter FAILED'

# ---- TC32: sparse_sym_cg solves identity system exactly ----
from topology_optimizer import sparse_sym_cg
n_cg = 5
rows_cg = np.arange(n_cg, dtype=np.int32)
cols_cg = np.arange(n_cg, dtype=np.int32)
vals_cg = np.ones(n_cg, dtype=np.float64)
b_cg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_cg = sparse_sym_cg(rows_cg, cols_cg, vals_cg, b_cg, n_cg, tol=1e-12)
assert np.allclose(x_cg, b_cg), '[TC32] CG identity system FAILED'

# ---- TC33: discrete distribution sample statistics match PMF ----
from uncertainty_quantification import sample_discrete_distribution
np.random.seed(42)
pmf_tc = np.array([0.2, 0.3, 0.5])
samples_tc = sample_discrete_distribution(pmf_tc, 10000, seed=42)
empirical_tc = np.bincount(samples_tc, minlength=3) / 10000.0
assert abs(empirical_tc[0] - 0.2) < 0.02, '[TC33] pmf bin 0 FAILED'
assert abs(empirical_tc[1] - 0.3) < 0.02, '[TC33] pmf bin 1 FAILED'
assert abs(empirical_tc[2] - 0.5) < 0.02, '[TC33] pmf bin 2 FAILED'

# ---- TC34: hypercube distance statistics are non-negative ----
from uncertainty_quantification import sample_hypercube_uniform, hypercube_distance_statistics
np.random.seed(42)
samples_uq_tc = sample_hypercube_uniform(3, 50, seed=42)
stats_tc = hypercube_distance_statistics(samples_uq_tc)
assert stats_tc["mean"] >= 0.0, '[TC34] mean non-negative FAILED'
assert stats_tc["std"] >= 0.0, '[TC34] std non-negative FAILED'
assert stats_tc["min"] >= 0.0, '[TC34] min non-negative FAILED'

# ---- TC35: parameter_to_physical maps to reasonable ranges ----
from uncertainty_quantification import parameter_to_physical
params_tc = parameter_to_physical(np.array([0.0, 0.5, 1.0, 0.0, 1.0]))
assert 0.9 <= params_tc["laser_power_var"] <= 1.1, '[TC35] laser_power_var FAILED'
assert 0.8 - 1e-12 <= params_tc["powder_size_var"] <= 1.2 + 1e-12, '[TC35] powder_size_var FAILED'

# ---- TC36: compute_element_stress returns finite values ----
from fem_core import compute_element_stress
node_xy_tc, element_node_tc = generate_rectangular_mesh(1.0, 1.0, 2, 2)
n_dof_tc = node_xy_tc.shape[0] * 2
F_tc = np.zeros(n_dof_tc)
F_tc[-1] = -100.0
bc_nodes_tc = np.array([0, 1], dtype=np.int32)
bc_values_tc = np.array([0.0, 0.0])
U_tc = solve_fem_system(node_xy_tc, element_node_tc, 100e3, 0.3, F_tc, bc_nodes_tc, bc_values_tc)
stress_tc = compute_element_stress(node_xy_tc, element_node_tc, U_tc, 100e3, 0.3)
assert np.all(np.isfinite(stress_tc)), '[TC36] stress finiteness FAILED'

print('\n全部 36 个测试通过!\n')
