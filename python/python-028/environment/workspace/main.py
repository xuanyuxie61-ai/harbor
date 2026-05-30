
import numpy as np
from math import pi
import sys
import os
import time


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




    A = 100
    Z = 42
    R0 = 1.2
    R = R0 * (A ** (1.0 / 3.0))
    V0 = -50.0
    a_ws = 0.65
    Vso0 = 12.0
    beta2 = 0.25
    gamma = 0.3

    print_section("核壳模型与集体运动耦合计算系统")
    print(f"目标核素: A = {A}, Z = {Z}")
    print(f"形变参数: β₂ = {beta2:.3f}, γ = {gamma:.3f} rad")
    print(f"Woods-Saxon 参数: V₀ = {V0:.1f} MeV, R = {R:.3f} fm, a = {a_ws:.2f} fm")




    print_section("1. 特殊函数计算 —— 核格林函数与形状因子")
    print_subsection("球贝塞尔函数 j_l(kR)")
    k_values = np.linspace(0.1, 5.0, 10)
    for l in [0, 1, 2, 3]:
        j_vals = spherical_bessel_j(l, k_values * R)
        print(f"  l = {l}: max|j_l| = {np.max(np.abs(j_vals)):.6e}")

    print_subsection("正弦积分 Si(x)")
    for x_test in [0.5, 2.0, 8.0, 20.0, 50.0]:
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

    beta_new = 0.15
    gamma_new = 0.4
    E_interp = bilinear_interpolate_2d(beta_new, gamma_new, beta_g, gamma_g, E_surf)
    print(f"  插值点 (β={beta_new:.2f}, γ={gamma_new:.2f}): E_interp = {E_interp:.3f} MeV")




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


    x_test = np.random.randn(m)
    y_test = crs_matvec(row, col, val, x_test)
    print(f"  稀疏矩阵-向量乘法结果范数: {np.linalg.norm(y_test):.6e}")




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




    print_section("6. 壳模型组态空间与 Lanczos 对角化")
    n_orbitals = 8
    n_particles = 6

    sp_energies = np.array([-45.0, -38.0, -32.0, -28.0, -22.0, -18.0, -12.0, -8.0])
    row_sm, col_sm, val_sm, dim_sm = shell_model_hamiltonian_sparse(
        n_particles, n_orbitals, interaction_strength=2.0,
        single_energies=sp_energies
    )
    print(f"  组态空间维数: {dim_sm}")
    print(f"  哈密顿量非零元: {len(val_sm)}")

    eigenvalues = lanczos_iteration(row_sm, col_sm, val_sm, dim_sm, n_iter=20)
    print(f"  Lanczos 最低本征值: {eigenvalues[:5]}")




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




    print_section("8. 核密度反应-扩散 FTCS 演化")

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


    Q2 = density_moment(r_grid, rho_final, 2)
    Q4 = density_moment(r_grid, rho_final, 4)
    print(f"  四极矩 M₂: {Q2:.3f} fm²")
    print(f"  十六极矩 M₄: {Q4:.3f} fm⁴")




    print_section("9. Cauchy 主值积分与单粒子自能")
    print_subsection("测试主值积分")

    def test_func(x):
        return np.cos(x)

    cpv_result = cauchy_principal_value(test_func, -1.0, 1.0, 0.0, n=64)
    print(f"  P ∫_{{-1}}^1 cos(x)/x dx = {cpv_result:.6e} (理论值: 0)")


    def test_func2(x):
        return np.exp(x)

    cpv_result2 = cauchy_principal_value(test_func2, 0.0, 2.0, 1.0, n=64)
    print(f"  P ∫_0^2 exp(x)/(x-1) dx = {cpv_result2:.6f}")

    print_subsection("自能计算")
    E_test = -30.0
    coupling_sq = np.array([0.5, 1.2, 0.8, 0.3, 2.0])
    E_levels = np.array([-40.0, -35.0, -25.0, -20.0, -15.0])
    sigma_E = self_energy_integral(coupling_sq, E_levels, E_test, n_quad=64)
    print(f"  E = {E_test:.1f} MeV 处自能: Σ(E) = {sigma_E:.4f} MeV")




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




    print_section("12. 核势能面全局优化")

    def nuclear_potential_1d(b):

        return (20.0 * (b - 0.2) ** 2
                + 5.0 * b ** 4
                - 2.0 * b ** 3)

    M_est = 100.0
    b_opt, V_opt, calls = glomin_global_minimize(
        -0.5, 0.6, 0.1, M_est, 1e-6, 1e-5, nuclear_potential_1d
    )
    print(f"  glomin 全局最小: β = {b_opt:.5f}, V = {V_opt:.5f} MeV")
    print(f"  函数调用次数: {calls}")


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

    rng = np.random.default_rng(2024)
    base_levels = np.sort(np.cumsum(np.abs(rng.normal(1.0, 0.3, 200))))
    s_unfolded, N_smooth = unfolding_spectrum(base_levels)
    if len(s_unfolded) > 10:
        mean_s = np.mean(s_unfolded)
        var_s = np.var(s_unfolded)
        print(f"  归一化间距均值: {mean_s:.4f}")
        print(f"  归一化间距方差: {var_s:.4f}")
        print(f"  GOE 理论方差: {4.0 / pi - 1.0:.4f}")
        print(f"  Poisson 理论方差: 1.0000")




    print_section("14. 谱学因子与重叠积分")
    if len(all_wavefunctions.get(0, [])) >= 1:
        u_orb = all_wavefunctions[0][0]
        u_res = all_wavefunctions[0][0] * 0.95

        norm_res = np.sqrt(np.trapezoid(u_res ** 2, r_grid))
        if norm_res > 0:
            u_res = u_res / norm_res
        S_fac = spectroscopic_factor(r_grid, u_orb, u_res, A - 1, n=1, l=0, j=0.5)
        print(f"  拾取反应谱学因子 S: {S_fac:.4f}")
        overlap = overlap_integral(r_grid, u_orb, u_res)
        print(f"  波函数重叠积分: {overlap:.6f}")




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
