
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    THETA_12, THETA_23, THETA_13, DELTA_CP,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH,
    EARTH_RADIUS_KM, matter_potential_eV
)
from pmns_matrix import (
    build_pmns_matrix, build_mass_matrix,
    check_unitarity, get_initial_flavor_state,
    jarkslog_invariant
)
from neutrino_hamiltonian import (
    build_vacuum_hamiltonian, build_matter_hamiltonian,
    solve_hamiltonian_eigen, effective_mixing_angles_in_matter,
    msw_resonance_density, hierarchy_discrimination_significance,
    compute_oscillation_wavelengths, mass_sum_bounds,
    log_gamma_pike_hill, fermi_dirac_distribution
)
from matter_profile_fem1d import (
    solve_steady_state_density_1d,
    solve_time_dependent_density_1d
)
from matter_profile_fem2d import (
    solve_steady_state_density_2d,
    compute_bandwidth
)
from monte_carlo_oscillation import (
    monte_carlo_oscillation_probability,
    mc_hierarchy_significance,
    mc_integrate_oscillation_over_spectrum
)
from numerical_integration import (
    midpoint_quad_2d,
    circle_rule,
    oscillation_probability_integral_2d,
    integrate_over_delta_cp,
    adaptive_integral_1d
)
from neutrino_ode_solver import (
    solve_neutrino_oscillation_ode,
    solve_varying_matter_ode,
    solve_euler, solve_rk4,
    r8but_sl, solve_banded_upper_triangular
)
from mesh_utils import (
    evaluate_mesh_quality,
    generate_earth_tetrahedral_mesh,
    mesh_base_one,
    parse_mesh_data
)
from sparse_iterative import (
    power_iteration,
    pagerank_style_matrix,
    find_dominant_oscillation_mode,
    iterative_hierarchy_solver
)
from validation_utils import (
    digit_checksum,
    validate_probability_conservation,
    validate_hermitian,
    validate_eigenvalue_ordering,
    validate_pmns_completeness,
    validate_oscillation_unitarity
)
from data_io import (
    increment_indices,
    convert_index_base,
    write_matrix_file,
    read_matrix_file
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  中微子振荡与质量 Hierarchy 综合分析平台")
    print("  Neutrino Oscillation & Mass Hierarchy Analysis Platform")
    print("=" * 70)




    print_section("1. PMNS 矩阵构造与验证")

    U = build_pmns_matrix()
    M2_NH = build_mass_matrix(hierarchy='normal')
    M2_IH = build_mass_matrix(hierarchy='inverted')

    print(f"PMNS 矩阵 U:")
    for i in range(3):
        row_str = "  ".join(f"{U[i,j]:12.6f}" for j in range(3))
        print(f"  [{row_str}]")

    is_unitary, err = check_unitarity(U)
    print(f"\n幺正性验证: {'通过' if is_unitary else '失败'} (最大误差: {err:.2e})")

    J_cp = jarkslog_invariant(U)
    print(f"Jarlskog 不变量 J_CP = {J_cp:.6e}")




    print_section("2. 中微子质量与振荡波长")

    m_bounds = mass_sum_bounds('normal', m_lightest_eV=0.0)
    print(f"Normal Hierarchy 质量 (m_lightest=0):")
    print(f"  m_1 = {m_bounds['m1']:.6f} eV")
    print(f"  m_2 = {m_bounds['m2']:.6f} eV")
    print(f"  m_3 = {m_bounds['m3']:.6f} eV")
    print(f"  Σm_i = {m_bounds['sum']:.6f} eV")

    E_test = 2.0
    waves = compute_oscillation_wavelengths(E_test)
    print(f"\nE = {E_test} GeV 时的振荡波长:")
    print(f"  L_21 = {waves['L_21']:.2f} km")
    print(f"  L_31 = {waves['L_31']:.2f} km")
    print(f"  L_32 = {waves['L_32']:.2f} km")




    print_section("3. 真空与物质哈密顿量本征值分析")

    H_vac = build_vacuum_hamiltonian(E_test)
    ev_vac, evec_vac, U_mat = solve_hamiltonian_eigen(H_vac)

    print(f"真空哈密顿量本征值 [eV]:")
    for i, ev in enumerate(ev_vac):
        print(f"  E_{i+1} = {ev:.6e} eV")

    is_herm, herm_err = validate_hermitian(H_vac)
    print(f"\n厄米性验证: {'通过' if is_herm else '失败'} (误差: {herm_err:.2e})")

    is_ordered = validate_eigenvalue_ordering(ev_vac)
    print(f"本征值排序验证: {'通过' if is_ordered else '失败'}")


    V_earth = matter_potential_eV(0.5)
    H_mat = build_matter_hamiltonian(E_test, V_earth)
    ev_mat, evec_mat, U_mat_matter = solve_hamiltonian_eigen(H_mat)

    print(f"\n物质哈密顿量本征值 (V = {V_earth:.4e} eV):")
    for i, ev in enumerate(ev_mat):
        print(f"  E_{i+1}^m = {ev:.6e} eV")




    print_section("4. 有限元地球密度剖面 (1D & 2D)")


    r_nodes = np.linspace(0, EARTH_RADIUS_KM, 51)
    rho_1d, r_1d = solve_steady_state_density_1d(r_nodes)
    print(f"1D FEM 节点数: {len(r_1d)}")
    print(f"  中心密度: {rho_1d[0]:.3f} g/cm³")
    print(f"  表面密度: {rho_1d[-1]:.3f} g/cm³")


    rho_prem_center = 13.0885
    rho_prem_surface = 2.6910
    print(f"  PREM 中心密度: {rho_prem_center:.3f} g/cm³")
    print(f"  PREM 表面密度: {rho_prem_surface:.3f} g/cm³")


    rho_2d, nodes_2d, elements_2d = solve_steady_state_density_2d(
        radius_km=EARTH_RADIUS_KM, n_r=10, n_theta=16
    )
    bw = compute_bandwidth(3, elements_2d)
    print(f"\n2D FEM 节点数: {len(nodes_2d)}, 单元数: {len(elements_2d)}")
    print(f"  矩阵半带宽: {bw}")
    print(f"  最小密度: {np.min(rho_2d):.3f} g/cm³")
    print(f"  最大密度: {np.max(rho_2d):.3f} g/cm³")




    print_section("5. 蒙特卡洛振荡概率与参数不确定性")

    mc_result = monte_carlo_oscillation_probability(
        energy_range_gev=(1.0, 10.0),
        baseline_range_km=(100.0, 1300.0),
        n_samples=5000,
        hierarchy='normal',
        param_uncertainties={'theta13': 0.005, 'delta_cp': 0.3},
        seed=42
    )

    print(f"MC 结果 (E∈[1,10] GeV, L∈[100,1300] km, N=5000):")
    print(f"  P(ν_e→ν_e) = {mc_result['P_ee_mean']:.6f} ± {mc_result['P_ee_std']:.6f}")
    print(f"  P(ν_e→ν_μ) = {mc_result['P_em_mean']:.6f} ± {mc_result['P_em_std']:.6f}")
    print(f"  P(ν_e→ν_τ) = {mc_result['P_et_mean']:.6f} ± {mc_result['P_et_std']:.6f}")


    mc_hier = mc_hierarchy_significance(
        energy_gev=2.0, baseline_km=1000.0,
        n_samples=10000, seed=123
    )
    print(f"\nHierarchy 判别 MC 分析:")
    print(f"  NH 正确率: {mc_hier['nh_correct_rate']:.4f}")
    print(f"  IH 正确率: {mc_hier['ih_correct_rate']:.4f}")
    print(f"  NH 置信度: {mc_hier['nh_confidence_sigma']:.2f} σ")
    print(f"  IH 置信度: {mc_hier['ih_confidence_sigma']:.2f} σ")




    print_section("6. 数值积分振荡概率")


    P_avg_ee = oscillation_probability_integral_2d(
        1.0, 10.0, 100.0, 1300.0,
        nx=16, ny=16,
        initial_flavor=0, final_flavor=0,
        hierarchy='normal'
    )
    print(f"2D 积分 P(ν_e→ν_e) 平均值 = {P_avg_ee:.6f}")


    w, ang = circle_rule(12)
    print(f"\n圆积分规则 (12 点):")
    print(f"  权重和 = {np.sum(w):.6f} (应为 1.0)")
    print(f"  角度范围: [{ang[0]:.4f}, {ang[-1]:.4f}] rad")


    def test_func(x):
        return np.sin(x) ** 2
    adaptive_result = adaptive_integral_1d(test_func, 0.0, np.pi, tol=1e-8)
    exact = np.pi / 2.0
    print(f"\n自适应积分验证:")
    print(f"  ∫_0^π sin²(x) dx = {adaptive_result:.10f} (精确值: {exact:.10f})")
    print(f"  误差: {abs(adaptive_result - exact):.2e}")




    print_section("7. 中微子味演化 ODE 求解")

    baseline_test = 1000.0
    E_test = 2.0


    result_euler = solve_neutrino_oscillation_ode(
        E_test, baseline_test, method='euler', n_steps=5000
    )
    print(f"Euler 方法 (5000 步):")
    print(f"  P_ee = {result_euler['prob_final'][0]:.6f}")
    print(f"  P_em = {result_euler['prob_final'][1]:.6f}")
    print(f"  P_et = {result_euler['prob_final'][2]:.6f}")


    result_rk4 = solve_neutrino_oscillation_ode(
        E_test, baseline_test, method='rk4', n_steps=1000
    )
    print(f"\nRK4 方法 (1000 步):")
    print(f"  P_ee = {result_rk4['prob_final'][0]:.6f}")
    print(f"  P_em = {result_rk4['prob_final'][1]:.6f}")
    print(f"  P_et = {result_rk4['prob_final'][2]:.6f}")


    result_exact = solve_neutrino_oscillation_ode(
        E_test, baseline_test, method='matrix_exp'
    )
    print(f"\n矩阵指数法 (精确):")
    print(f"  P_ee = {result_exact['prob_final'][0]:.6f}")
    print(f"  P_em = {result_exact['prob_final'][1]:.6f}")
    print(f"  P_et = {result_exact['prob_final'][2]:.6f}")


    P_mat = np.zeros((3, 3), dtype=np.float64)
    for alpha in range(3):
        res = solve_neutrino_oscillation_ode(
            E_test, baseline_test, method='matrix_exp',
            initial_flavor=['electron', 'muon', 'tau'][alpha]
        )
        P_mat[alpha, :] = res['prob_final']

    is_conserved, cons_err = validate_probability_conservation(P_mat)
    print(f"\n概率守恒验证: {'通过' if is_conserved else '失败'}")
    print(f"  最大误差: {cons_err:.2e}")
    print(f"  概率矩阵:")
    for i in range(3):
        print(f"    [{P_mat[i,0]:.6f}  {P_mat[i,1]:.6f}  {P_mat[i,2]:.6f}]")


    def V_profile(x_km):
        r_ratio = abs(1.0 - x_km / baseline_test)
        return matter_potential_eV(r_ratio)

    result_vary = solve_varying_matter_ode(
        E_test, baseline_test, V_profile, n_steps=2000, method='rk4'
    )
    print(f"\n变物质密度 RK4 (2000 步):")
    print(f"  P_ee = {result_vary['prob_final'][0]:.6f}")
    print(f"  P_em = {result_vary['prob_final'][1]:.6f}")
    print(f"  P_et = {result_vary['prob_final'][2]:.6f}")


    A_test = np.array([[2.0, 1.0, 0.5],
                       [0.0, 3.0, 1.0],
                       [0.0, 0.0, 4.0]], dtype=np.float64)
    b_test = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x_r8but = solve_banded_upper_triangular(A_test, b_test)
    x_exact = np.linalg.solve(A_test, b_test)
    print(f"\nR8BUT 求解器验证:")
    print(f"  R8BUT 解:  [{x_r8but[0]:.6f}, {x_r8but[1]:.6f}, {x_r8but[2]:.6f}]")
    print(f"  精确解:    [{x_exact[0]:.6f}, {x_exact[1]:.6f}, {x_exact[2]:.6f}]")
    print(f"  误差:      {np.max(np.abs(x_r8but - x_exact)):.2e}")




    print_section("8. 三维地球网格质量评估")

    nodes_3d, tetra_3d = generate_earth_tetrahedral_mesh(n_r=4, n_theta=6, n_phi=6)
    if len(tetra_3d) > 0:
        quality = evaluate_mesh_quality(nodes_3d, tetra_3d)
        print(f"地球四面体网格统计:")
        print(f"  节点数: {len(nodes_3d)}")
        print(f"  四面体数: {quality['n_tetra']}")
        print(f"  总体积: {quality['volume_total']:.2f} km³")
        print(f"  Q1 均值: {quality['q1_mean']:.6f}")
        print(f"  Q2 均值: {quality['q2_mean']:.6f}")
    else:
        print("地球四面体网格生成 (简化模式, 节点数较少)")
        print(f"  节点数: {len(nodes_3d)}")


    idx_0based = np.array([0, 1, 2, 3])
    idx_1based = convert_index_base(idx_0based, 0, 1)
    print(f"\n索引转换测试:")
    print(f"  0-based: {idx_0based} -> 1-based: {idx_1based}")




    print_section("9. 稀疏迭代与主导振荡模式")


    dom_mode = find_dominant_oscillation_mode(H_vac)
    print(f"真空主导振荡模式:")
    print(f"  有效能量: {dom_mode['energy']:.6e} eV")
    print(f"  角频率:   {dom_mode['frequency']:.6e} eV")
    print(f"  态向量:   [{dom_mode['state'][0]:.4f}, {dom_mode['state'][1]:.4f}, {dom_mode['state'][2]:.4f}]")


    P_pr = pagerank_style_matrix(np.abs(H_vac), damping=0.85)
    print(f"\nPageRank 风格矩阵 (阻尼=0.85):")
    print(f"  列和范围: [{np.min(np.sum(P_pr, axis=0)):.6f}, {np.max(np.sum(P_pr, axis=0)):.6f}]")


    ev_pi, vec_pi, conv = power_iteration(P_pr, n_iterations=100)
    print(f"  幂迭代收敛: {'是' if conv else '否'}")
    print(f"  主导特征值: {ev_pi:.6f}")


    hier_iter = iterative_hierarchy_solver(2.0, 1000.0)
    print(f"\nHierarchy 迭代判别:")
    print(f"  P_ee (NH) = {hier_iter['P_ee_NH']:.6f}")
    print(f"  P_ee (IH) = {hier_iter['P_ee_IH']:.6f}")
    print(f"  |ΔP| = {hier_iter['delta_P']:.6f}")
    print(f"  判别能力 = {hier_iter['discrimination_power']:.6f}")




    print_section("10. MSW 共振与有效混合角")

    V_test = matter_potential_eV(0.5)
    eff_angles = effective_mixing_angles_in_matter(E_test, V_test)
    print(f"E = {E_test} GeV, V = {V_test:.4e} eV 时的有效混合角:")
    print(f"  θ₁₂^m = {np.rad2deg(eff_angles['theta12_m']):.4f}°")
    print(f"  θ₂₃^m = {np.rad2deg(eff_angles['theta23_m']):.4f}°")
    print(f"  θ₁₃^m = {np.rad2deg(eff_angles['theta13_m']):.4f}°")

    ne_res = msw_resonance_density(E_test)
    print(f"\nMSW 共振电子数密度:")
    print(f"  N_e^res = {ne_res:.4e} cm⁻³")




    print_section("11. 质量 Hierarchy 综合判定")

    sig_NH, hier_NH = hierarchy_discrimination_significance(
        DELTA_M2_31, sigma_dm31=0.03e-3
    )
    sig_IH, hier_IH = hierarchy_discrimination_significance(
        DELTA_M2_31_IH, sigma_dm31=0.03e-3
    )

    print(f"当前实验测量 (PDG 2024):")
    print(f"  Δm²₃₁ (NH) = {DELTA_M2_31:.4e} eV²")
    print(f"  Δm²₃₁ (IH) = {DELTA_M2_31_IH:.4e} eV²")
    print(f"\nHierarchy 显著性 (σ = 0.03×10⁻³ eV²):")
    print(f"  NH 假设显著性: {sig_NH:.2f} σ -> {hier_NH}")
    print(f"  IH 假设显著性: {sig_IH:.2f} σ -> {hier_IH}")

    if sig_NH > 5.0:
        print(f"\n  >> 结论: Normal Hierarchy 在 >5σ 水平被确认")
    elif sig_IH > 5.0:
        print(f"\n  >> 结论: Inverted Hierarchy 在 >5σ 水平被确认")
    else:
        print(f"\n  >> 结论: 需要更多数据以区分 NH 和 IH")




    print_section("12. 特殊函数验证")

    ln_g, fault = log_gamma_pike_hill(5.0)
    print(f"ln Γ(5) = {ln_g:.10f} (精确值: ln 24 = {np.log(24):.10f})")
    print(f"误差: {abs(ln_g - np.log(24)):.2e}, 错误码: {fault}")

    fd = fermi_dirac_distribution(1.0, 0.5, chemical_potential=0.0)
    print(f"\nFermi-Dirac 分布 (E=1.0, T=0.5, μ=0):")
    print(f"  f_FD = {fd:.6f}")




    print_section("13. 数据 I/O 验证")

    test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    write_matrix_file("test_matrix_io.txt", test_data, header="test data")
    read_back = read_matrix_file("test_matrix_io.txt")
    print(f"矩阵读写测试:")
    print(f"  原始: {test_data.tolist()}")
    print(f"  读取: {read_back.tolist()}")
    print(f"  一致: {np.allclose(test_data, read_back)}")
    os.remove("test_matrix_io.txt")




    print("\n" + "=" * 70)
    print("  全部计算任务已完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
