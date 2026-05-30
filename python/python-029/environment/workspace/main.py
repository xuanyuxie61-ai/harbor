
import numpy as np
import warnings
warnings.filterwarnings('ignore')




from optical_potential import OpticalPotentialParameters, build_optical_potential
from chebyshev_schrodinger import solve_radial_schrodinger, riccati_bessel_functions
from s_matrix import (
    compute_cross_sections, scattering_amplitude,
    differential_cross_section, svd_analysis_smatrix,
    transmission_coefficients, compound_formation_cross_section
)
from angular_quadrature import (
    lebedev_rule, integrate_differential_cross_section,
    cartesian_to_spherical
)
from hauser_feshbach import (
    level_density, compound_formation_cross_section as hf_compound_cs,
    decay_width, energy_average_cross_section,
    open_newton_cotes_weights, decay_chain_bdf2,
    width_fluctuation_correction
)
from nuclear_data_io import (
    Nuclide, NuclearDataAggregator,
    generate_nuclear_mass_table, compute_q_value_reaction,
    SphericalShellMesh
)
from special_functions import (
    log_gamma_stirling, gamma_function,
    spherical_bessel_jn_highprecision,
    coulomb_wave_function_series,
    coulomb_phase_shift
)
from orthogonality import (
    l2_inner_product, l2_norm,
    check_orthogonality, gram_schmidt_orthogonalization,
    deformation_coupling_potential, coupling_matrix_element
)
from collective_dynamics import (
    collective_mass_parameter, restoring_force_parameter,
    resonance_energy, damping_width,
    forced_damped_oscillator, giant_resonance_cross_section,
    strength_function_integral, energy_weighted_sum_rule
)
from bifurcation_stability import (
    logistic_attractor, lyapunov_exponent_logistic,
    neutron_multiplication_bifurcation,
    optical_potential_stability_boundary,
    critical_slowing_down_indicator
)
from finite_field_symmetry import (
    gf2_add, gf2_multiply, gf2_poly_string,
    parity_operator_state, isospin_states,
    nuclear_configuration_gf2, time_reversal_symmetry_check,
    shell_model_parity
)
from manifold_learning import (
    sammon_mapping, nuclear_mass_manifold,
    local_linear_embedding, magic_number_detection,
    binding_energy_gradient_flow
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_optical_model_calculation():
    print_section("PART I: 光学模型与分波分析")


    params = OpticalPotentialParameters('n', 56, 26, 14.1)
    print(f"反应系统: {params}")
    print(f"约化质量: {params.mu_MeV:.3f} MeV")
    print(f"波数 k: {params.k:.5f} fm^-1")

    l_max = 8
    S_matrix_dict = {}
    phase_shifts = {}
    wavefunctions = {}

    print(f"\n计算分波 l = 0 ~ {l_max} (含自旋-轨道耦合 j = l ± 1/2)")
    print("-" * 60)
    print(f"{'l':>3} {'j':>5} {'δ_l (rad)':>12} {'|S_l|':>10} {'η_l':>10}")
    print("-" * 60)

    for l in range(l_max + 1):
        js = [l + 0.5] if l == 0 else [l - 0.5, l + 0.5]
        for j in js:
            result = solve_radial_schrodinger(params, l, j, N=70, L=15.0, r_match=10.0)
            key = (l, j)
            S_matrix_dict[key] = result['S_matrix']
            phase_shifts[key] = result['phase_shift']
            wavefunctions[key] = result['u']
            r_grid = result['r']

            delta = result['phase_shift']
            eta = result['absorption']
            print(f"{l:3d} {j:5.1f} {delta:12.6f} {abs(result['S_matrix']):10.6f} {eta:10.6f}")


    xs = compute_cross_sections(S_matrix_dict, params.k, l_max)
    print("\n截面计算结果:")
    print(f"  总截面       σ_tot    = {xs['sigma_total']:.4f} fm² = {xs['sigma_total']*100:.4f} barn")
    print(f"  反应截面     σ_react  = {xs['sigma_reaction']:.4f} fm² = {xs['sigma_reaction']*100:.4f} barn")
    print(f"  弹性截面     σ_el     = {xs['sigma_elastic']:.4f} fm² = {xs['sigma_elastic']*100:.4f} barn")


    consistency = abs(xs['sigma_total'] - xs['sigma_elastic'] - xs['sigma_reaction'])
    print(f"\n光学定理校验 |σ_tot - σ_el - σ_react| = {consistency:.6f} fm²")


    theta = np.linspace(0.05, np.pi - 0.05, 60)
    dsigma = differential_cross_section(theta, S_matrix_dict, params.k, l_max)
    sigma_el_int = 2.0 * np.pi * np.trapezoid(dsigma * np.sin(theta), theta)
    print(f"\n角分布积分校验 σ_el(integrated) = {sigma_el_int:.4f} fm²")


    svd_res = svd_analysis_smatrix(S_matrix_dict, l_max)
    print(f"\nS-矩阵 SVD 分析:")
    print(f"  奇异值: {np.round(svd_res['singular_values'], 6)}")
    print(f"  捕获 99% 能量的秩: {svd_res['rank_99']}")


    T_dict = transmission_coefficients(S_matrix_dict, l_max)
    sigma_cf = compound_formation_cross_section(params, T_dict, l_max)
    print(f"\n复合核形成截面 σ_CF = {sigma_cf:.4f} fm²")


    print("\n波函数正交性校验 (选取前 3 个分波):")
    sample_wf = [wavefunctions[(0, 0.5)], wavefunctions[(1, 0.5)], wavefunctions[(1, 1.5)]]
    overlap, is_ortho = check_orthogonality(sample_wf, r_grid, threshold=0.05)
    print(f"  重叠矩阵最大非对角元: {np.max(np.abs(overlap - np.eye(3))):.6f}")
    print(f"  是否近似正交: {is_ortho}")

    return params, S_matrix_dict, T_dict, r_grid, theta, dsigma


def run_hauser_feshbach(params, T_dict, l_max):
    print_section("PART II: Hauser-Feshbach 复合核统计理论")


    sigma_cf = hf_compound_cs(params, T_dict, l_max, I_target=0.0, I_proj=0.5)
    print(f"复合核形成截面: {sigma_cf:.4f} fm²")


    A_cn = params.target_A + 1
    E_exc = params.E_lab + 8.0
    rho = level_density(A_cn, E_exc, J=2.0)
    print(f"\n复合核 ⁵⁷Fe 激发能 E* ≈ {E_exc:.1f} MeV")
    print(f"能级密度 ρ(E*, J=2) = {rho:.6e} MeV^{-1}")


    widths = decay_width(T_dict, l_max, E_gamma=2.0)
    print(f"\n衰变宽度:")
    print(f"  中子宽度 Γ_n = {widths['neutron']:.4f} MeV")
    print(f"  γ 宽度   Γ_γ = {widths['gamma']:.6f} MeV")
    print(f"  总宽度   Γ_tot = {widths['total']:.4f} MeV")
    print(f"  中子分支比 = {widths['ratio_n']:.4f}")
    print(f"  γ 分支比   = {widths['ratio_gamma']:.4f}")


    def mock_sigma(E):

        return sigma_cf * np.exp(-0.05 * (E - 14.0) ** 2)

    avg_sigma = energy_average_cross_section(10.0, 20.0, 6, mock_sigma)
    print(f"\n能量平均截面 (10-20 MeV, NCO-6): {avg_sigma:.4f} fm²")


    W = width_fluctuation_correction(T_dict, l_max, nu=1.0)
    print(f"宽度涨落修正因子 W = {W:.4f}")


    decay_matrix = np.array([
        [-widths['total'], 0.0, 0.0],
        [widths['neutron'], -0.1, 0.0],
        [widths['gamma'], 0.05, -0.05]
    ])
    N0 = np.array([1.0, 0.0, 0.0])
    t, N_pop = decay_chain_bdf2(N0, decay_matrix, (0.0, 50.0), 200)
    print(f"\n衰变链 BDF2 演化 (t=0→50 fm/c):")
    print(f"  初始复合核布居: {N_pop[0, 0]:.4f}")
    print(f"  最终中子出射道布居: {N_pop[-1, 1]:.4f}")
    print(f"  最终γ衰变道布居: {N_pop[-1, 2]:.4f}")

    return widths


def run_nuclear_data_analysis():
    print_section("PART III: 核数据与质量表分析")


    agg = generate_nuclear_mass_table(range(24, 30), lambda Z: (Z + 15, Z + 35))
    stats = agg.aggregate_by_Z()
    print(f"生成核素数目: {len(agg.nuclides)}")
    for Z in sorted(stats.keys()):
        print(f"  Z={Z}: {stats[Z]['count']} 种同位素, 平均 BE/A = {stats[Z]['BE_mean']/stats[Z]['A_mean']:.3f} MeV")


    Q_val = compute_q_value_reaction(26, 56, 0, 1, 0, 1)
    print(f"\nn + ⁵⁶Fe → n + ⁵⁶Fe 弹性散射 Q 值 ≈ {Q_val:.3f} MeV")

    Q_val2 = compute_q_value_reaction(26, 56, 0, 1, 1, 1)
    print(f"n + ⁵⁶Fe → p + ⁵⁶Mn 电荷交换 Q 值 ≈ {Q_val2:.3f} MeV")


    mesh = SphericalShellMesh(R_max=12.0, n_r=25, n_theta=12, n_phi=24)
    print(f"\n球形壳层网格: {mesh.n_r}×{mesh.n_theta}×{mesh.n_phi}")
    print(f"  顶点数: {mesh.n_vertices}, 体元数: {mesh.n_elements}")


    X_nuc, _ = nuclear_mass_manifold(range(20, 35), range(20, 45))
    magic = magic_number_detection(
        X_nuc[:, 1].astype(int),
        X_nuc[:, 0].astype(int),
        X_nuc[:, 2] * (X_nuc[:, 0] + X_nuc[:, 1])
    )
    print(f"\n检测到的幻数 (Z, N) 对: {magic[:8]}")

    return agg


def run_collective_dynamics():
    print_section("PART IV: 核集体运动与巨共振")

    A = 56
    for lam in [1, 2, 3]:
        B_lam = collective_mass_parameter(A, lam)
        C_lam = restoring_force_parameter(A, lam)
        E_R = resonance_energy(A, lam)
        Gamma = damping_width(A, lam, E_R)
        print(f"λ={lam}: B={B_lam:.1f}, C={C_lam:.1f}, E_R={E_R:.2f} MeV, Γ={Gamma:.2f} MeV")


    lam = 2
    B2 = collective_mass_parameter(A, lam)
    C2 = restoring_force_parameter(A, lam)
    E2 = resonance_energy(A, lam)
    Gamma2 = damping_width(A, lam, E2)
    omega = E2 / 197.3

    from collective_dynamics import time_dependent_multipole_field
    F = lambda t: time_dependent_multipole_field(t, omega, 0.5, lam)
    t, Q, dQ = forced_damped_oscillator((0, 300), 0.0, 0.0, B2, C2, Gamma2, F, n_steps=300)
    print(f"\n四极集体坐标演化:")
    print(f"  最大振幅: {np.max(np.abs(Q)):.6f} fm")
    print(f"  稳态振幅: {Q[-1]:.6f} fm")


    E_range = np.linspace(5, 30, 200)
    sigma_gqr = giant_resonance_cross_section(E_range, A, lam=2)
    S_int = strength_function_integral(E_range, sigma_gqr)
    ewsr = energy_weighted_sum_rule(A, 2)
    print(f"\nGQR(λ=2) 强度积分: {S_int:.2f} mb·MeV")
    print(f"EWSR 理论上限: {ewsr:.2f} (自然单位)")

    return t, Q


def run_stability_analysis():
    print_section("PART V: 非线性稳定性与分岔分析")


    for r in [2.5, 3.2, 3.5, 3.8]:
        lam = lyapunov_exponent_logistic(r)
        attr = logistic_attractor(r)
        print(f"r={r:.1f}: Lyapunov={lam:.4f}, attractor_period={len(attr)}")


    alpha_range = np.linspace(0.05, 1.5, 50)
    eq, stab = neutron_multiplication_bifurcation(alpha_range, lambda_f=0.6, lambda_c=0.3, S=0.02)
    n_stable = np.sum(stab)
    print(f"\n中子增殖分岔分析:")
    print(f"  α 范围: [{alpha_range[0]:.2f}, {alpha_range[-1]:.2f}]")
    print(f"  稳定平衡点比例: {n_stable}/{len(alpha_range)}")


    tau_crit = critical_slowing_down_indicator(0.5, 0.6, 0.3)
    print(f"  临界慢化时间 (α=0.5): {tau_crit:.2f} fm/c")

    return eq, stab


def run_symmetry_algebra():
    print_section("PART VI: 有限域对称性代数")


    p = 0b1011
    q = 0b110
    print(f"GF(2) 多项式 p = {gf2_poly_string(p)}")
    print(f"GF(2) 多项式 q = {gf2_poly_string(q)}")
    print(f"p + q = {gf2_poly_string(gf2_add(p, q))}")
    print(f"p * q = {gf2_poly_string(gf2_multiply(p, q))}")


    for l in range(5):
        print(f"l={l}: 宇称 = {parity_operator_state(l):+d}")


    iso_states = isospin_states(30, 26)
    print(f"\n⁵⁶Fe (N=30, Z=26) 同位旋态:")
    for s in iso_states:
        print(f"  T={s['T']:.1f}, Tz={s['Tz']:.1f}, 多重态维数={s['multiplicity']}")


    configs = nuclear_configuration_gf2(2, 5)
    print(f"\n5 态中填 2 粒子: {len(configs)} 种组态")


    tr = time_reversal_symmetry_check(2.5, configs[0])
    print(f"J=5/2 的 Kramers 简并度: {tr['kramers_degeneracy']}")


    shell_parities = [1, -1, 1, -1, 1]
    for cfg in configs[:4]:
        p_tot = shell_model_parity(cfg, shell_parities)
        print(f"组态 {bin(cfg)}: 总宇称 = {p_tot:+d}")

    return configs


def run_manifold_analysis():
    print_section("PART VII: 核数据流形学习")


    Z_range = range(20, 32)
    N_range = range(20, 40)
    X_nuc, labels = nuclear_mass_manifold(Z_range, N_range)
    print(f"核素数据集: {X_nuc.shape[0]} 个样本, {X_nuc.shape[1]} 维特征")


    X_mean = X_nuc.mean(axis=0)
    X_std = X_nuc.std(axis=0) + 1e-12
    X_norm = (X_nuc - X_mean) / X_std


    Y_sammon, stress = sammon_mapping(X_norm, n_components=2, max_iter=150, alpha=0.2)
    print(f"Sammon 映射最终应力: {stress[-1]:.6f}")
    print(f"低维嵌入范围: x=[{Y_sammon[:,0].min():.3f}, {Y_sammon[:,0].max():.3f}], "
          f"y=[{Y_sammon[:,1].min():.3f}, {Y_sammon[:,1].max():.3f}]")


    if len(X_norm) > 10:
        Y_lle = local_linear_embedding(X_norm, n_neighbors=min(5, len(X_norm)-1), n_components=2)
        print(f"LLE 嵌入完成，形状: {Y_lle.shape}")


    Z_g, N_g, dBZ, dBN = binding_energy_gradient_flow(Z_range, N_range)
    print(f"\n结合能梯度流场计算完成，网格: {Z_g.shape}")
    print(f"  ∂(B/A)/∂Z 范围: [{dBZ[~np.isnan(dBZ)].min():.4f}, {dBZ[~np.isnan(dBZ)].max():.4f}]")
    print(f"  ∂(B/A)/∂N 范围: [{dBN[~np.isnan(dBN)].min():.4f}, {dBN[~np.isnan(dBN)].max():.4f}]")

    return Y_sammon


def run_special_functions_check():
    print_section("PART VIII: 高精度特殊函数校验")


    z_test = [2.5, 5.5, 1.0 + 2.0j]
    for z in z_test:
        gz = gamma_function(z)
        print(f"Γ({z}) = {gz}")


    x = 3.0
    for n in [0, 2, 5]:
        jn = spherical_bessel_jn_highprecision(x, n)
        print(f"j_{n}({x}) = {jn:.10f}")


    F0 = coulomb_wave_function_series(0, 1.0, 2.0)
    print(f"F_0(η=1, ρ=2) = {F0:.6f}")


    sigma_2 = coulomb_phase_shift(2, 1.0)
    print(f"σ_2(η=1) = {sigma_2:.6f} rad")






def main():
    print("\n" + "#" * 70)
    print("#  核反应光学模型与统计理论综合计算平台")
    print("#  PROJECT_29: 博士级 Python 科研代码合成")
    print("#" * 70)


    params, S_dict, T_dict, r_grid, theta, dsigma = run_optical_model_calculation()


    widths = run_hauser_feshbach(params, T_dict, l_max=8)


    agg = run_nuclear_data_analysis()


    t, Q = run_collective_dynamics()


    eq, stab = run_stability_analysis()


    configs = run_symmetry_algebra()


    Y_sammon = run_manifold_analysis()


    run_special_functions_check()

    print("\n" + "#" * 70)
    print("#  全部计算完成，无错误。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
