"""
main.py
=======
核壳模型与集体运动耦合计算系统 —— 统一入口

本项目围绕"核物理：原子核壳模型与集体运动"展开，
将 15 个种子项目的核心算法融合为一套完整的博士级科学计算流程。

运行方式：
    python main.py

无需任何命令行参数，程序自动执行从核势构建、单粒子能级求解、
壳模型哈密顿量对角化、集体运动 ODE 演化、核密度扩散、
矩阵元与自能计算、几何采样与统计、能级密度分析到全局能量优化的
完整计算链，并输出所有关键物理量的数值结果。
"""

import numpy as np
import sys
import os
import time

# 将当前目录加入路径，确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from special_functions import (
    spherical_bessel_j, spherical_neumann_n, sine_integral_si,
    associated_legendre, spherical_harmonic_Y, nuclear_form_factor
)
from nuclear_grid import (
    deformed_nuclear_surface_grid, annular_shell_grid,
    cvt_3d_sample, nuclear_volume_cvt_quadrature
)
from potential_deformed import (
    woods_saxon_potential, deformed_woods_saxon,
    build_potential_energy_surface, total_single_particle_potential,
    bilinear_interpolate_2d
)
from hamiltonian_sparse import (
    build_radial_hamiltonian_st, st_to_crs, crs_matvec,
    shell_model_hamiltonian_sparse, lanczos_iteration
)
from radial_solver import (
    compute_radial_wavefunction, find_bound_state_energy,
    solve_all_bound_states, brent_root_find
)
from collective_ode import (
    CollectiveHamiltonian, solve_collective_motion, adiabatic_invariant
)
from density_evolution import (
    ftcs_density_evolution_1d, reaction_source, total_nucleon_number,
    rms_radius, surface_thickness, density_moment
)
from matrix_elements import (
    cauchy_principal_value, self_energy_integral,
    electric_multipole_matrix_element, transition_probability,
    overlap_integral, spectroscopic_factor
)
from nuclear_shape import (
    uniform_sphere_sample, deformed_fermi_sample,
    pairwise_distance_statistics, pair_correlation_function,
    monte_carlo_nuclear_radius, triangular_deformation_analysis
)
from energy_surface import (
    glomin_global_minimize, optimize_nuclear_shape_energy, moment_of_inertia
)
from level_density import (
    bethe_formula, bcs_level_density, log_normal_pdf,
    log_normal_sample, level_spacing_distribution,
    unfolding_spectrum, nuclear_level_density_parameter,
    total_level_density_table
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print(f"\n--- {title} ---")


def main():
    start_time = time.time()

    # ================================================================
    # 全局物理参数设定（以 A ~ 100 的稀土核为例）
    # ================================================================
    A = 100
    Z = 42
    R0 = 1.2
    R = R0 * (A ** (1.0 / 3.0))
    V0 = -50.0      # MeV
    a_ws = 0.65     # fm
    Vso0 = 12.0     # MeV
    beta2 = 0.25
    gamma = 0.3     # rad (~17°)

    print_section("核壳模型与集体运动耦合计算系统")
    print(f"目标核素: A = {A}, Z = {Z}")
    print(f"形变参数: β₂ = {beta2:.3f}, γ = {gamma:.3f} rad")
    print(f"Woods-Saxon 参数: V₀ = {V0:.1f} MeV, R = {R:.3f} fm, a = {a_ws:.2f} fm")

    # ================================================================
    # 1. 特殊函数与球贝塞尔函数计算 (seed: 1084_sine_integral)
    # ================================================================
    print_section("1. 特殊函数计算 —— 核格林函数与形状因子")
    print_subsection("球贝塞尔函数 j_l(kR)")
    k_values = np.linspace(0.1, 5.0, 10)
    for l in [0, 1, 2, 3]:
        j_vals = spherical_bessel_j(l, k_values * R)
        print(f"  l = {l}: max|j_l| = {np.max(np.abs(j_vals)):.6e}")

    print_subsection("正弦积分 Si(x)")
    for x_test in [0.5, 2.0, 8.0, 15.0, 50.0]:
        si_val = sine_integral_si(x_test)
        print(f"  Si({x_test:.1f}) = {si_val:.8f}")

    print_subsection("连带勒让德多项式 P_l^m(cos θ)")
    theta_test = np.pi / 4.0
    x_cos = np.cos(theta_test)
    for l in [2, 3, 4]:
        for m in range(0, l + 1):
            plm = associated_legendre(l, m, x_cos)
            print(f"  P_{l}^{m}({x_cos:.4f}) = {plm:.6f}")

    print_subsection("核形状因子 |F(q)|²")
    for q_test in [0.5, 1.0, 2.0, 3.0]:
        ff = nuclear_form_factor(q_test, A, Z, R0)
        print(f"  q = {q_test:.1f} fm⁻¹: |F(q)|² = {ff:.6e}")

    # ================================================================
    # 2. 核网格生成与 CVT 采样 (seeds: 008_annulus_grid, 250_cvt_3d_sampling)
    # ================================================================
    print_section("2. 核体积网格与 CVT 最优采样")
    print_subsection("变形核表面网格")
    surface_grid, surface_areas = deformed_nuclear_surface_grid(
        beta2, gamma, R, n_theta=20, n_phi=30
    )
    print(f"  表面网格点数: {len(surface_grid)}")
    print(f"  总表面积 (一阶近似): {np.sum(surface_areas):.3f} fm²")

    print_subsection("环形截面网格")
    ann_points, ann_weights = annular_shell_grid(
        n_r=15, n_theta=24, r_inner=0.5, r_outer=R
    )
    print(f"  环形网格点数: {len(ann_points)}")
    print(f"  权重总和 (截面面积): {np.sum(ann_weights):.3f} fm²")

    print_subsection("CVT 三维最优求积点")
    cvt_points, cvt_energy = nuclear_volume_cvt_quadrature(
        A, beta2=beta2, gamma=gamma, n_points=100, n_iter=20
    )
    print(f"  CVT 求积点数: {len(cvt_points)}")
    print(f"  最终 CVT 能量: {cvt_energy[-1]:.6f}")

    # ================================================================
    # 3. 变形 Woods-Saxon 势与二维插值 (seeds: 139_cauchy_principal_value, 1212_test_interp_2d)
    # ================================================================
    print_section("3. 变形 Woods-Saxon 势场与二维插值")
    print_subsection("单粒子势计算")
    params = {
        'V0': V0, 'R0': R, 'a': a_ws,
        'Vso0': Vso0, 'Rso': R, 'aso': a_ws,
        'beta2': beta2, 'gamma': gamma
    }
    for r_test in [2.0, 4.0, 6.0, 8.0]:
        V_total = total_single_particle_potential(r_test, np.pi / 3.0, 0.0, l=2, j=2.5, params=params)
        print(f"  r = {r_test:.1f} fm: V_total = {V_total:.3f} MeV")

    print_subsection("势能量曲面 (β₂, γ) 插值")
    beta_g, gamma_g, E_surf = build_potential_energy_surface(15, 12, V0, R, a_ws)
    # 插值到一个新点
    beta_new = 0.15
    gamma_new = 0.4
    E_interp = bilinear_interpolate_2d(beta_new, gamma_new, beta_g, gamma_g, E_surf)
    print(f"  插值点 (β={beta_new:.2f}, γ={gamma_new:.2f}): E_interp = {E_interp:.3f} MeV")

    # ================================================================
    # 4. 径向哈密顿量构建与稀疏格式 (seed: 1155_st_to_crs)
    # ================================================================
    print_section("4. 径向哈密顿量与稀疏矩阵")
    r_min = 0.05
    r_max = 15.0
    N_r = 300
    r_grid = np.linspace(r_min, r_max, N_r)
    V_grid = woods_saxon_potential(r_grid, V0, R, a_ws)

    print_subsection("ST 格式构建与 CRS 转换")
    nst, ist, jst, Ast = build_radial_hamiltonian_st(r_grid, V_grid, l=2)
    print(f"  ST 非零元个数: {nst}")
    m, n, nz, row, col, val = st_to_crs(nst, ist, jst, Ast)
    print(f"  矩阵维度: {m} × {n}, CRS 非零元: {nz}")

    # 稀疏矩阵-向量乘法测试
    x_test = np.random.randn(m)
    y_test = crs_matvec(row, col, val, x_test)
    print(f"  稀疏矩阵-向量乘法结果范数: {np.linalg.norm(y_test):.6e}")

    # ================================================================
    # 5. 径向薛定谔方程求解与 Brent 寻根 (seed: 1427_zero_brent)
    # ================================================================
    print_section("5. 径向薛定谔方程束缚态求解")
    print_subsection("求解 l = 0, 1, 2 的束缚态")
    all_energies = {}
    all_wavefunctions = {}
    for l_qn in [0, 1, 2]:
        energies, wavefunctions = solve_all_bound_states(
            r_grid, V_grid, l_qn, n_max_states=4,
            E_search_min=-55.0, E_search_max=-1.0
        )
        all_energies[l_qn] = energies
        all_wavefunctions[l_qn] = wavefunctions
        print(f"  l = {l_qn}: 找到 {len(energies)} 个束缚态")
        for idx, E_bnd in enumerate(energies):
            print(f"    n = {idx + 1}, E = {E_bnd:.4f} MeV")

    # ================================================================
    # 6. 壳模型哈密顿量与 Lanczos 迭代 (seeds: 1155_st_to_crs)
    # ================================================================
    print_section("6. 壳模型组态空间与 Lanczos 对角化")
    n_orbitals = 8
    n_particles = 6
    # 从径向解中提取单粒子能级近似值
    sp_energies = np.array([-45.0, -38.0, -32.0, -28.0, -22.0, -18.0, -12.0, -8.0])
    row_sm, col_sm, val_sm, dim_sm = shell_model_hamiltonian_sparse(
        n_particles, n_orbitals, interaction_strength=2.0,
        single_energies=sp_energies
    )
    print(f"  组态空间维数: {dim_sm}")
    print(f"  哈密顿量非零元: {len(val_sm)}")

    eigenvalues = lanczos_iteration(row_sm, col_sm, val_sm, dim_sm, n_iter=20)
    print(f"  Lanczos 最低本征值: {eigenvalues[:5]}")

    # ================================================================
    # 7. 集体运动非线性 ODE 系统 (seed: 091_biochemical_nonlinear_ode)
    # ================================================================
    print_section("7. Bohr-Mottelson 集体运动 ODE 演化")
    ham_coll = CollectiveHamiltonian(mass_number=A, beta_eq=beta2, gamma_eq=0.0)
    t_array, y_array, E_array = solve_collective_motion(
        ham_coll, t_span=(0.0, 50.0), n_steps=2000
    )
    print(f"  初始集体能量: {E_array[0]:.4f} MeV")
    print(f"  最终集体能量: {E_array[-1]:.4f} MeV")
    print(f"  能量相对漂移: {abs(E_array[-1] - E_array[0]) / E_array[0] * 100:.4f} %")

    beta_final = y_array[-1, 0]
    gamma_final = y_array[-1, 1]
    print(f"  初始形变: β = {y_array[0, 0]:.4f}, γ = {y_array[0, 1]:.4f}")
    print(f"  最终形变: β = {beta_final:.4f}, γ = {gamma_final:.4f}")

    I_inv = adiabatic_invariant(y_array, ham_coll, t_array[1] - t_array[0])
    print(f"  β 振动绝热不变量: {I_inv:.6f}")

    # ================================================================
    # 8. 核密度反应-扩散演化 (seed: 434_fisher_pde_ftcs)
    # ================================================================
    print_section("8. 核密度反应-扩散 FTCS 演化")
    # 初始密度：Fermi 分布
    rho0 = 0.16
    a_diff = 0.52
    rho_initial = rho0 / (1.0 + np.exp((r_grid - R) / a_diff))
    A_initial = total_nucleon_number(r_grid, rho_initial)
    print(f"  初始核子数: {A_initial:.2f}")

    rho_final, rho_history, s_param = ftcs_density_evolution_1d(
        r_grid, rho_initial, D=2.5, t_max=10.0, nt=2000,
        rho0=rho0, alpha=0.5, beta=0.1
    )
    A_final = total_nucleon_number(r_grid, rho_final)
    R_rms_final = rms_radius(r_grid, rho_final)
    t_surf = surface_thickness(r_grid, rho_final, rho0)
    print(f"  最终核子数: {A_final:.2f}")
    print(f"  最终均方根半径: {R_rms_final:.3f} fm")
    print(f"  最终表面厚度: {t_surf:.3f} fm")
    print(f"  FTCS 稳定性参数 s = DΔt/Δr² = {s_param:.4f}")

    # 密度多极矩
    Q2 = density_moment(r_grid, rho_final, 2)
    Q4 = density_moment(r_grid, rho_final, 4)
    print(f"  四极矩 M₂: {Q2:.3f} fm²")
    print(f"  十六极矩 M₄: {Q4:.3f} fm⁴")

    # ================================================================
    # 9. Cauchy 主值积分与自能 (seed: 139_cauchy_principal_value)
    # ================================================================
    print_section("9. Cauchy 主值积分与单粒子自能")
    print_subsection("测试主值积分")
    # P ∫_{-1}^{1} cos(x) / x dx = 0 (奇函数)
    def test_func(x):
        return np.cos(x)

    cpv_result = cauchy_principal_value(test_func, -1.0, 1.0, 0.0, n=64)
    print(f"  P ∫_{{-1}}^1 cos(x)/x dx = {cpv_result:.6e} (理论值: 0)")

    # P ∫_{0}^{2} exp(x) / (x - 1) dx
    def test_func2(x):
        return np.exp(x)

    cpv_result2 = cauchy_principal_value(test_func2, 0.0, 2.0, 1.0, n=64)
    print(f"  P ∫_0^2 exp(x)/(x-1) dx = {cpv_result2:.6f}")

    print_subsection("自能计算")
    E_test = -30.0  # MeV
    coupling_sq = np.array([0.5, 1.2, 0.8, 0.3, 2.0])
    E_levels = np.array([-40.0, -35.0, -25.0, -20.0, -15.0])
    sigma_E = self_energy_integral(coupling_sq, E_levels, E_test, n_quad=64)
    print(f"  E = {E_test:.1f} MeV 处自能: Σ(E) = {sigma_E:.4f} MeV")

    # ================================================================
    # 10. 电磁多极矩阵元与跃迁 (seeds: 1084_sine_integral)
    # ================================================================
    print_section("10. 电磁多极跃迁矩阵元")
    if len(all_wavefunctions.get(0, [])) >= 2 and len(all_wavefunctions.get(2, [])) >= 1:
        u_1s = all_wavefunctions[0][0]
        u_1d = all_wavefunctions[2][0]
        me_E2 = electric_multipole_matrix_element(r_grid, u_1s, u_1d, lambda_order=2)
        E_gamma = abs(all_energies[2][0] - all_energies[0][0])
        B_E2, B_W, tau_half = transition_probability(2, me_E2, E_gamma, A, Ji=0.5)
        print(f"  1s → 1d E2 跃迁矩阵元: {me_E2:.4f} e·fm²")
        print(f"  γ 射线能量: {E_gamma:.3f} MeV")
        print(f"  B(E2) = {B_E2:.4e} e²fm⁴")
        print(f"  B(E2; W.u.) = {B_E2 / B_W:.4f}")
        print(f"  估算半寿命: {tau_half:.4e} s")

    # ================================================================
    # 11. 核几何采样与距离统计 (seeds: 884_polygon_distance, 889_polygon_sample, 442_fly_simulation)
    # ================================================================
    print_section("11. 核几何 Monte Carlo 采样与距离统计")
    print_subsection("形变 Fermi 分布采样")
    mc_points = deformed_fermi_sample(500, A, beta2=beta2, gamma=gamma, seed=42)
    print(f"  采样点数: {len(mc_points)}")

    dist_stats = pairwise_distance_statistics(mc_points)
    print(f"  平均核子间距: {dist_stats['mean']:.3f} fm")
    print(f"  间距方差: {dist_stats['variance']:.4f} fm²")
    print(f"  最小/最大间距: {dist_stats['min']:.3f} / {dist_stats['max']:.3f} fm")

    print_subsection("对关联函数 g(r)")
    r_bins, g_r = pair_correlation_function(mc_points, dr=0.3, r_max=10.0)
    peak_idx = np.argmax(g_r)
    print(f"  g(r) 峰值位置: r = {r_bins[peak_idx]:.2f} fm")
    print(f"  g(r) 峰值高度: {g_r[peak_idx]:.3f}")

    print_subsection("Monte Carlo 核半径估算")
    R_eff, t_surf_mc, R_rms_mc = monte_carlo_nuclear_radius(
        A, n_samples=50000, beta2=beta2, gamma=gamma, seed=123
    )
    print(f"  等效半径 (90%): {R_eff:.3f} fm")
    print(f"  表面厚度 (10%-90%): {t_surf_mc:.3f} fm")
    print(f"  均方根半径: {R_rms_mc:.3f} fm")

    print_subsection("核表面三角形剖分")
    S_total, S_var, S_areas = triangular_deformation_analysis(15, 20, beta2, gamma, R)
    print(f"  总表面积: {S_total:.3f} fm²")
    print(f"  三角形面积方差: {S_var:.4f} fm⁴")

    # ================================================================
    # 12. 全局能量优化 (seeds: 471_glomin, 1427_zero_brent)
    # ================================================================
    print_section("12. 核势能面全局优化")

    def nuclear_potential_1d(b):
        # 简化的集体势能
        return (20.0 * (b - 0.2) ** 2
                + 5.0 * b ** 4
                - 2.0 * b ** 3)

    M_est = 100.0
    b_opt, V_opt, calls = glomin_global_minimize(
        -0.5, 0.6, 0.1, M_est, 1e-6, 1e-5, nuclear_potential_1d
    )
    print(f"  glomin 全局最小: β = {b_opt:.5f}, V = {V_opt:.5f} MeV")
    print(f"  函数调用次数: {calls}")

    # 二维势能面优化
    print_subsection("二维 (β, γ) 势能面优化")
    def V_2d(b, g):
        return (15.0 * (b - 0.22) ** 2
                + 8.0 * g ** 2
                + 10.0 * b ** 4
                - 3.0 * b ** 3 * np.cos(3.0 * g))

    beta_opt2, gamma_opt2, E_min2, surf_data = optimize_nuclear_shape_energy(
        V_2d, beta_range=(-0.3, 0.5), gamma_range=(0.0, np.pi / 3.0),
        n_grid_beta=15, n_grid_gamma=12
    )
    print(f"  最优形变: β = {beta_opt2:.4f}, γ = {gamma_opt2:.4f}")
    print(f"  最小势能: {E_min2:.4f} MeV")

    I_perp, I_parallel = moment_of_inertia(beta_opt2, gamma_opt2, A)
    print(f"  转动惯量 I_perp = {I_perp:.3f} MeV⁻¹, I_parallel = {I_parallel:.3f} MeV⁻¹")

    # ================================================================
    # 13. 核能级密度分析 (seed: 698_log_normal)
    # ================================================================
    print_section("13. 核能级密度与统计分布")
    print_subsection("Bethe 公式能级密度")
    E_grid_ld, rho_total, rho_pos, rho_neg = total_level_density_table(A, E_max=20.0, n_points=50)
    print(f"  激发能范围: {E_grid_ld[0]:.1f} - {E_grid_ld[-1]:.1f} MeV")
    print(f"  峰值能级密度: {np.max(rho_total):.3e} MeV⁻¹")
    print(f"  峰值位置: E ≈ {E_grid_ld[np.argmax(rho_total)]:.1f} MeV")

    print_subsection("对数正态分布（能级间距模拟）")
    mu_ln = 0.0
    sigma_ln = 0.5
    spacing_samples = log_normal_sample(mu_ln, sigma_ln, size=1000, seed=42)
    print(f"  采样间距均值: {np.mean(spacing_samples):.4f}")
    print(f"  采样间距标准差: {np.std(spacing_samples):.4f}")

    print_subsection("能谱展开与 Wigner-Dyson 统计")
    # 模拟一组能级（壳模型 + 随机矩阵扰动）
    rng = np.random.default_rng(2024)
    base_levels = np.sort(np.cumsum(np.abs(rng.normal(1.0, 0.3, 200))))
    s_unfolded, N_smooth = unfolding_spectrum(base_levels)
    if len(s_unfolded) > 10:
        mean_s = np.mean(s_unfolded)
        var_s = np.var(s_unfolded)
        print(f"  归一化间距均值: {mean_s:.4f}")
        print(f"  归一化间距方差: {var_s:.4f}")
        print(f"  GOE 理论方差: {4.0 / np.pi - 1.0:.4f}")
        print(f"  Poisson 理论方差: 1.0000")

    # ================================================================
    # 14. 谱学因子与重叠积分
    # ================================================================
    print_section("14. 谱学因子与重叠积分")
    if len(all_wavefunctions.get(0, [])) >= 1:
        u_orb = all_wavefunctions[0][0]
        u_res = all_wavefunctions[0][0] * 0.95  # 模拟轻微形变差异
        # 重新归一化
        norm_res = np.sqrt(np.trapezoid(u_res ** 2, r_grid))
        if norm_res > 0:
            u_res = u_res / norm_res
        S_fac = spectroscopic_factor(r_grid, u_orb, u_res, A - 1, n=1, l=0, j=0.5)
        print(f"  拾取反应谱学因子 S: {S_fac:.4f}")
        overlap = overlap_integral(r_grid, u_orb, u_res)
        print(f"  波函数重叠积分: {overlap:.6f}")

    # ================================================================
    # 15. 结果汇总与文件输出 (seed: 718_matlab_commandline)
    # ================================================================
    print_section("15. 计算结果汇总与文件输出")
    results = {
        'mass_number': A,
        'charge_number': Z,
        'deformation_beta2': beta2,
        'deformation_gamma': gamma,
        'woods_saxon_radius_fm': R,
        'bound_states_l0': len(all_energies.get(0, [])),
        'bound_states_l1': len(all_energies.get(1, [])),
        'bound_states_l2': len(all_energies.get(2, [])),
        'collective_final_beta': float(beta_final),
        'collective_final_gamma': float(gamma_final),
        'density_rms_radius_fm': float(R_rms_final),
        'density_surface_thickness_fm': float(t_surf),
        'optimal_beta': float(beta_opt2),
        'optimal_gamma': float(gamma_opt2),
        'minimum_potential_MeV': float(E_min2),
        'level_density_peak_MeV': float(E_grid_ld[np.argmax(rho_total)]),
        'level_density_peak_value': float(np.max(rho_total)),
        'mc_rms_radius_fm': float(R_rms_mc),
        'mc_surface_thickness_fm': float(t_surf_mc),
        'execution_time_sec': time.time() - start_time
    }

    output_file = 'nuclear_shell_model_results.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  核壳模型与集体运动耦合计算结果\n")
        f.write("=" * 60 + "\n\n")
        for key, value in results.items():
            f.write(f"{key:40s} = {value:12.6f}\n")
        f.write("\n")
        f.write("关键物理量说明:\n")
        f.write("  - deformation_beta2: 四极形变参数 β₂\n")
        f.write("  - deformation_gamma: 形变角 γ (rad)\n")
        f.write("  - bound_states_l*: 各角动量下的束缚态数目\n")
        f.write("  - collective_final_beta/gamma: 集体运动演化终点形变\n")
        f.write("  - density_rms_radius_fm: 密度分布均方根半径\n")
        f.write("  - optimal_beta/gamma: 势能面全局最小对应的形变\n")
        f.write("  - level_density_peak_*: 能级密度峰值信息\n")

    print(f"  结果已写入文件: {output_file}")
    print(f"\n总执行时间: {results['execution_time_sec']:.2f} 秒")
    print("=" * 70)
    print("  计算完成。所有模块运行正常，无报错。")
    print("=" * 70)


if __name__ == '__main__':
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: 球贝塞尔函数 j_0(0) 应接近 1 ----
val = spherical_bessel_j(0, 0.0)
assert abs(val - 1.0) < 1e-6, '[TC01] 球贝塞尔函数 j_0(0) 应接近 1 FAILED'

# ---- TC02: 球贝塞尔函数 j_1 小参数近似 j_1(x)~x/3 ----
x = 1e-6
val = spherical_bessel_j(1, x)
assert abs(val - x / 3.0) < 1e-10, '[TC02] 球贝塞尔函数 j_1 小参数近似 FAILED'

# ---- TC03: 球诺伊曼函数 n_0 小参数近似 n_0(x)~-1/x ----
x = 1e-6
val = spherical_neumann_n(0, x)
assert abs(val + 1.0 / x) < 1e-3, '[TC03] 球诺伊曼函数 n_0 小参数近似 FAILED'

# ---- TC04: 正弦积分 Si(0) = 0 ----
val = sine_integral_si(0.0)
assert abs(val) < 1e-12, '[TC04] 正弦积分 Si(0) = 0 FAILED'

# ---- TC05: 正弦积分大参数趋近 pi/2 ----
val = sine_integral_si(50.0)
assert abs(val - np.pi / 2.0) < 0.02, '[TC05] 正弦积分大参数趋近 pi/2 FAILED'

# ---- TC06: 连带勒让德 P_2^0(1) = 1 ----
val = associated_legendre(2, 0, 1.0)
assert abs(val - 1.0) < 1e-10, '[TC06] 连带勒让德 P_2^0(1) = 1 FAILED'

# ---- TC07: 核形状因子 q=0 时 |F(0)|^2 = 1 ----
val = nuclear_form_factor(0.0, 100, 50)
assert abs(val - 1.0) < 1e-10, '[TC07] 核形状因子 q=0 时应为 1 FAILED'

# ---- TC08: Woods-Saxon 势中心值接近 V0 ----
V = woods_saxon_potential(0.0, -50.0, 5.0, 0.65)
assert abs(V - (-50.0)) < 0.1, '[TC08] Woods-Saxon 势中心值接近 V0 FAILED'

# ---- TC09: Woods-Saxon 势远边界趋近 0 ----
V = woods_saxon_potential(50.0, -50.0, 5.0, 0.65)
assert abs(V) < 1e-6, '[TC09] Woods-Saxon 势远边界趋近 0 FAILED'

# ---- TC10: 双线性插值在网格点上精确恢复 ----
x_grid = np.linspace(0.0, 1.0, 5)
y_grid = np.linspace(0.0, 1.0, 5)
np.random.seed(42)
Z = np.random.rand(5, 5)
val_exact = bilinear_interpolate_2d(x_grid[2], y_grid[3], x_grid, y_grid, Z)
assert abs(val_exact - Z[3, 2]) < 1e-12, '[TC10] 双线性插值在网格点上精确恢复 FAILED'

# ---- TC11: 稀疏矩阵向量乘法满足线性性 ----
r_grid = np.linspace(0.1, 10.0, 30)
V_grid = woods_saxon_potential(r_grid, -50.0, 5.0, 0.65)
nst, ist, jst, Ast = build_radial_hamiltonian_st(r_grid, V_grid, l=0)
m, n, nz, row, col, val = st_to_crs(nst, ist, jst, Ast)
x_vec = np.linspace(1, m, m)
y1 = crs_matvec(row, col, val, x_vec)
y2 = crs_matvec(row, col, val, 2.0 * x_vec)
assert np.allclose(y2, 2.0 * y1), '[TC11] 稀疏矩阵向量乘法线性性 FAILED'

# ---- TC12: 壳模型哈密顿量维度正确 ----
sp_energies = np.array([-10.0, -8.0, -5.0, -2.0])
row_sm, col_sm, val_sm, dim_sm = shell_model_hamiltonian_sparse(4, 4, 1.0, sp_energies)
assert dim_sm == 8, '[TC12] 壳模型哈密顿量维度应为 8 FAILED'

# ---- TC13: Lanczos 本征值为实数且升序 ----
np.random.seed(42)
eigenvalues = lanczos_iteration(row_sm, col_sm, val_sm, dim_sm, n_iter=10)
assert np.all(np.isreal(eigenvalues)), '[TC13] Lanczos 本征值必须为实数 FAILED'
assert eigenvalues[0] <= eigenvalues[-1], '[TC13] Lanczos 本征值应升序排列 FAILED'

# ---- TC14: Brent 寻根法求 sqrt(2) ----
root, calls = brent_root_find(0.0, 2.0, 1e-10, lambda x: x * x - 2.0)
assert abs(root - np.sqrt(2.0)) < 1e-6, '[TC14] Brent 寻根求 sqrt(2) FAILED'
assert calls > 0, '[TC14] Brent 调用次数应大于 0 FAILED'

# ---- TC15: 反应源项在 rho=0 时为 0 ----
src = reaction_source(0.0)
assert abs(src) < 1e-12, '[TC15] 反应源项在 rho=0 时应为 0 FAILED'

# ---- TC16: 总核子数均匀球理论值 ----
R = 5.0
r_grid_u = np.linspace(0, R, 200)
rho0_test = 0.16
rho_u = np.where(r_grid_u <= R, rho0_test, 0.0)
A_calc = total_nucleon_number(r_grid_u, rho_u)
A_theory = 4.0 / 3.0 * np.pi * R ** 3 * rho0_test
assert abs(A_calc - A_theory) / A_theory < 0.05, '[TC16] 总核子数均匀球理论值 FAILED'

# ---- TC17: 均方根半径均匀球理论值 ----
R_rms = rms_radius(r_grid_u, rho_u)
assert abs(R_rms - np.sqrt(3.0 / 5.0) * R) / R < 0.05, '[TC17] 均方根半径均匀球理论值 FAILED'

# ---- TC18: Cauchy 主值积分奇函数近似为 0 ----
cpv = cauchy_principal_value(lambda x: np.cos(x), -1.0, 1.0, 0.0, n=64)
assert abs(cpv) < 0.1, '[TC18] Cauchy 主值奇函数应接近 0 FAILED'

# ---- TC19: 相同归一化波函数重叠积分为 1 ----
r_grid_w = np.linspace(0, 10, 100)
u = np.sin(np.pi * r_grid_w / 10.0)
norm = np.sqrt(np.trapezoid(u ** 2, r_grid_w))
u_norm = u / norm
ov = overlap_integral(r_grid_w, u_norm, u_norm)
assert abs(ov - 1.0) < 1e-6, '[TC19] 相同波函数重叠积分应为 1 FAILED'

# ---- TC20: 对数正态 PDF 必须非负 ----
x = np.linspace(0.1, 5.0, 100)
pdf = log_normal_pdf(x, 0.0, 0.5)
assert np.all(pdf >= 0), '[TC20] 对数正态 PDF 必须非负 FAILED'

# ---- TC21: GOE 能级间距分布 P(0)=0 ----
s = np.array([0.0, 0.5, 1.0, 2.0])
P = level_spacing_distribution(s, regime='goe')
assert abs(P[0]) < 1e-10, '[TC21] GOE 分布 P(0)=0 FAILED'
assert P[1] > 0, '[TC21] GOE 分布 P(0.5)>0 FAILED'

# ---- TC22: 球对称核转动惯量各向同性 ----
I_perp, I_parallel = moment_of_inertia(0.0, 0.0, 100)
assert abs(I_perp - I_parallel) < 1e-10, '[TC22] 球对称核转动惯量各向同性 FAILED'

# ---- TC23: 集体哈密顿量在平衡点势能有限 ----
ham = CollectiveHamiltonian(mass_number=100, beta_eq=0.2, gamma_eq=0.0)
V_eq = ham.potential_energy(0.2, 0.0)
assert np.isfinite(V_eq), '[TC23] 集体哈密顿量在平衡点势能有限 FAILED'

# ---- TC24: FTCS 密度演化近似守恒核子数 ----
r_grid_d = np.linspace(0.05, 15.0, 100)
rho0 = 0.16
R = 5.0
rho_init = rho0 / (1.0 + np.exp((r_grid_d - R) / 0.52))
A0 = total_nucleon_number(r_grid_d, rho_init)
rho_final, _, s = ftcs_density_evolution_1d(r_grid_d, rho_init, D=0.1, t_max=0.1, nt=500)
A1 = total_nucleon_number(r_grid_d, rho_final)
assert abs(A1 - A0) / A0 < 0.1, '[TC24] FTCS 演化核子数近似守恒 FAILED'

# ---- TC25: glomin 全局最小化求抛物线最小值 ----
b_opt, V_opt, calls = glomin_global_minimize(-2.0, 2.0, 0.0, 10.0, 1e-8, 1e-6, lambda x: x ** 2)
assert abs(b_opt) < 0.01, '[TC25] glomin 抛物线最小值点 FAILED'
assert abs(V_opt) < 0.01, '[TC25] glomin 抛物线最小值 FAILED'

# ---- TC26: 核形状能量优化返回值在搜索范围内 ----
def simple_V(b, g):
    return 10.0 * (b - 0.1) ** 2 + 5.0 * g ** 2
beta_opt, gamma_opt, E_min, _ = optimize_nuclear_shape_energy(simple_V, beta_range=(-0.3, 0.5), gamma_range=(0.0, np.pi / 3.0), n_grid_beta=10, n_grid_gamma=8)
assert -0.3 <= beta_opt <= 0.5, '[TC26] 优化 beta 在范围内 FAILED'
assert 0.0 <= gamma_opt <= np.pi / 3.0, '[TC26] 优化 gamma 在范围内 FAILED'

# ---- TC27: 密度零阶矩等于总核子数 ----
M0 = density_moment(r_grid_d, rho_init, 0)
A0_check = total_nucleon_number(r_grid_d, rho_init)
assert abs(M0 - A0_check) < 1e-10, '[TC27] 密度零阶矩等于总核子数 FAILED'

# ---- TC28: 形变 Fermi 采样点数正确 ----
np.random.seed(42)
pts = deformed_fermi_sample(50, 100, beta2=0.0, gamma=0.0, seed=42)
assert len(pts) == 50, '[TC28] 形变 Fermi 采样点数正确 FAILED'

# ---- TC29: 单点距离统计均值为 0 ----
stats = pairwise_distance_statistics(np.array([[0.0, 0.0, 0.0]]))
assert stats['mean'] == 0.0, '[TC29] 单点距离统计均值为 0 FAILED'

# ---- TC30: 跃迁概率物理量必须为正 ----
B, BW, tau = transition_probability(2, 5.0, 1.0, 100, 0.5)
assert B > 0, '[TC30] 跃迁概率 B(E2) 必须为正 FAILED'
assert BW > 0, '[TC30] Weisskopf 单位必须为正 FAILED'
assert tau > 0, '[TC30] 半寿命必须为正 FAILED'

# ---- TC31: 能级密度表输出尺寸与非负性 ----
E_grid, rho_t, rho_p, rho_n = total_level_density_table(100, E_max=20.0, n_points=50)
assert len(E_grid) == 50, '[TC31] 能级密度表点数正确 FAILED'
assert np.all(rho_t >= 0), '[TC31] 能级密度必须非负 FAILED'

# ---- TC32: 均匀球采样点数与范围正确 ----
np.random.seed(42)
pts = uniform_sphere_sample(30, 5.0, seed=42)
assert len(pts) == 30, '[TC32] 均匀球采样点数正确 FAILED'
assert np.all(np.sqrt(np.sum(pts ** 2, axis=1)) <= 5.0 + 1e-10), '[TC32] 均匀球采样点应在球内 FAILED'

# ---- TC33: 绝热不变量非负 ----
ham2 = CollectiveHamiltonian(mass_number=100, beta_eq=0.2)
t_arr, y_arr, E_arr = solve_collective_motion(ham2, t_span=(0.0, 10.0), n_steps=500)
I_inv = adiabatic_invariant(y_arr, ham2, t_arr[1] - t_arr[0])
assert I_inv >= 0, '[TC33] 绝热不变量必须非负 FAILED'

# ---- TC34: 集体运动输出数组形状正确 ----
assert y_arr.shape == (501, 6), '[TC34] 集体运动状态数组形状正确 FAILED'
assert len(t_arr) == 501, '[TC34] 集体运动时间数组长度正确 FAILED'

# ---- TC35: 自能积分结果有限 ----
coupling = np.array([0.5, 1.0])
E_lvls = np.array([-40.0, -30.0])
sigma = self_energy_integral(coupling, E_lvls, -35.0, n_quad=32)
assert np.isfinite(sigma), '[TC35] 自能积分结果有限 FAILED'

# ---- TC36: 电磁多极矩阵元交换对称性 ----
r_grid_m = np.linspace(0, 10, 50)
u1 = np.sin(r_grid_m)
u2 = np.cos(r_grid_m)
me12 = electric_multipole_matrix_element(r_grid_m, u1, u2, 2)
me21 = electric_multipole_matrix_element(r_grid_m, u2, u1, 2)
assert abs(me12 - me21) < 1e-10, '[TC36] 电磁多极矩阵元交换对称性 FAILED'

# ---- TC37: 谱学因子必须非负 ----
r_grid_s = np.linspace(0, 10, 50)
u_s = np.sin(r_grid_s)
norm_s = np.sqrt(np.trapezoid(u_s ** 2, r_grid_s))
u_s = u_s / norm_s
S = spectroscopic_factor(r_grid_s, u_s, u_s, 99, 1, 0, 0.5)
assert S >= 0, '[TC37] 谱学因子必须非负 FAILED'

# ---- TC38: 表面厚度非负 ----
r_grid_t = np.linspace(0, 15, 100)
rho_t = 0.16 / (1.0 + np.exp((r_grid_t - 5.0) / 0.52))
t = surface_thickness(r_grid_t, rho_t, rho0=0.16)
assert t >= 0, '[TC38] 表面厚度必须非负 FAILED'

# ---- TC39: 束缚态求解返回列表长度一致 ----
r_grid_b = np.linspace(0.05, 15.0, 200)
V_grid_b = woods_saxon_potential(r_grid_b, -50.0, 5.0, 0.65)
energies, wfs = solve_all_bound_states(r_grid_b, V_grid_b, l=0, n_max_states=3, E_search_min=-55.0, E_search_max=-1.0)
assert len(energies) <= 3, '[TC39] 束缚态数目不超过上限 FAILED'
assert len(energies) == len(wfs), '[TC39] 能量与波函数列表长度一致 FAILED'

# ---- TC40: 对关联函数非负 ----
np.random.seed(42)
pts_p = uniform_sphere_sample(20, 3.0, seed=42)
r_bins, g_r = pair_correlation_function(pts_p, dr=0.5, r_max=6.0)
assert np.all(g_r >= 0), '[TC40] 对关联函数必须非负 FAILED'

print('\n全部 40 个测试通过!\n')
