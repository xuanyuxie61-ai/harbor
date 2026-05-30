
import numpy as np
from math import sqrt, pi, log, exp




from aerosol_microphysics import (
    multimode_lognormal,
    count_mixing_state,
    mixing_state_index,
    bruggeman_effective_medium,
    select_optimal_size_bins,
    extinction_efficiency_small,
)
from mie_scattering import (
    legendre_polynomial_value,
    legendre_coefficients_hg,
    phase_function_hg,
    scattering_asymmetry_parameter,
    expand_phase_function_legendre,
    mie_scattering_cross_section,
)
from radiative_transfer_solver import (
    build_rte_matrix,
    sor_solve,
    compute_radiative_flux,
    compute_heating_rate,
)
from monte_carlo_photon import (
    photon_random_walk_3d,
    estimate_optical_depth_monte_carlo,
)
from atmospheric_mesh import (
    generate_lat_lon_grid,
    compute_distance_table,
    define_atmospheric_layers,
    generate_simple_triangulation,
    compute_mesh_quality_metrics,
)
from inverse_source import (
    inverse_source_location,
    concentration_to_pseudo_distance,
)
from aerosol_activation import (
    kohler_critical_supersaturation,
    activated_fraction_logistic,
    ccn_spectrum_derivative,
    compute_ccn_number_concentration,
)
from statistical_covariance import (
    wishart_variate,
    sample_covariance_matrix,
    eof_analysis,
    aod_covariance_model,
)
from quadrature_engine import (
    integrate_tetrahedron,
    integrate_square,
    integrate_chebyshev1,
    tetrahedron_unit_o24,
    square_minimal_rule,
    chebyshev1_abscissas_weights,
)
from numerical_utils import (
    bisection,
    binomial_coefficient,
    comb_lexicographic,
    rnorm,
    gamma_log_values,
    wilson_hilferty_chi_square,
    safe_acos,
)


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("  气溶胶辐射效应与气候反馈综合分析系统 (PROJECT_59)")
    print("  Aerosol Radiative Effects & Climate Feedback Analysis")
    print("=" * 80)
    np.random.seed(42)




    print_section("1. 气溶胶微物理与光学特性")


    modes = [
        (500.0, 0.15, 1.8),
        (100.0, 1.5, 2.2),
        (2000.0, 0.03, 1.4),
    ]
    r_grid = np.logspace(-3, 1, 200)
    n_dist = multimode_lognormal(r_grid, modes)
    print(f"  粒径范围: {r_grid.min():.4f} ~ {r_grid.max():.2f} μm")
    print(f"  总粒子数浓度: {np.trapezoid(n_dist, r_grid):.2f} #/cm³")


    mixing_counts = count_mixing_state(12, 12, 5)
    chi = mixing_state_index(mixing_counts)
    print(f"  混合态分类: {mixing_counts}")
    print(f"  混合态指数 χ = {chi:.4f} (0=外混, 1=内混)")


    fractions = np.array([0.4, 0.3, 0.2, 0.1])
    m_components = np.array([
        1.53 + 0.0j,
        1.75 + 0.44j,
        1.45 + 0.003j,
        1.53 + 0.008j,
    ], dtype=np.complex128)
    m_eff = bruggeman_effective_medium(fractions, m_components)
    print(f"  等效复折射率 m_eff = {m_eff:.4f}")


    r_bins, N_bins = select_optimal_size_bins(500.0, 0.15, 1.8, 8)
    print(f"  最优粒径分档 (μm): {np.round(r_bins, 4)}")
    print(f"  各档数浓度 (#/cm³): {np.round(N_bins, 2)}")


    wavelength = 0.55
    q_ext_example = extinction_efficiency_small(0.1, wavelength, m_eff)
    print(f"  r=0.1μm, λ=0.55μm 时 Q_ext = {q_ext_example:.4f}")




    print_section("2. Mie 散射与勒让德相函数展开")

    g_asym = 0.65
    coeffs_hg = legendre_coefficients_hg(g_asym, max_l=20)
    print(f"  HG 不对称因子 g = {g_asym}")
    print(f"  前 5 个勒让德系数: {np.round(coeffs_hg[:5], 4)}")


    mu_test = np.linspace(-1.0, 1.0, 1000)
    p_hg = phase_function_hg(mu_test, g_asym)
    norm_check = 0.5 * np.trapezoid(p_hg, mu_test)
    print(f"  HG 相函数归一化积分 = {norm_check:.6f} (理论值 = 1.0)")


    p_reconstructed = expand_phase_function_legendre(mu_test, coeffs_hg)
    recon_error = np.mean(np.abs(p_reconstructed - p_hg))
    print(f"  勒让德重构平均绝对误差 = {recon_error:.6e}")


    test_radii = [0.01, 0.05, 0.1, 0.5, 1.0]
    print(f"  {'r(μm)':>8} {'C_ext(μm²)':>12} {'C_sca(μm²)':>12} {'g':>8}")
    for tr in test_radii:
        c_ext, c_sca, g_val = mie_scattering_cross_section(tr, wavelength, m_eff)
        print(f"  {tr:8.2f} {c_ext:12.4e} {c_sca:12.4e} {g_val:8.4f}")




    print_section("3. 辐射传输方程 SOR 求解")

    num_depth = 8
    num_angle = 4
    tau_total = 1.0
    omega = 0.85
    g_rt = 0.50

    A, b, mu, w = build_rte_matrix(num_depth, num_angle, tau_total, omega, g_rt)
    x0 = np.ones_like(b) * 0.5
    I_solution, iters, residual = sor_solve(A, b, x0, omega_sor=1.3, tol=1e-10, max_iter=5000)
    print(f"  深度层数: {num_depth}, 角度离散数: {num_angle}")
    print(f"  总光学厚度 τ = {tau_total}")
    print(f"  单次散射反照率 ω = {omega}")
    print(f"  SOR 迭代次数: {iters}, 最终残差: {residual:.4e}")


    print(f"  {'层号':>6} {'τ':>8} {'F_up':>10} {'F_down':>10} {'加热率(K/day)':>14}")

    dtau = None
    rho_cp = 1.0e3
    for t in range(num_depth):

        I_layer = None

        F_up = None
        F_down = None

        heating = None
        print(f"  {t:6d} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>14}")




    print_section("4. 三维蒙特卡洛光子传输")

    escaped, absorbed_surf, absorbed_atm, paths = photon_random_walk_3d(
        num_photons=2000,
        max_steps=100,
        extinction_coeff=0.5,
        layer_height=10.0,
        g_asymmetry=0.65,
        albedo=0.92,
        surface_albedo=0.15,
    )
    total = escaped + absorbed_surf + absorbed_atm
    print(f"  模拟光子总数: {total}")
    print(f"  层顶逃逸: {escaped} ({escaped/total*100:.2f}%)")
    print(f"  地表吸收: {absorbed_surf} ({absorbed_surf/total*100:.2f}%)")
    print(f"  大气吸收: {absorbed_atm} ({absorbed_atm/total*100:.2f}%)")
    print(f"  平均光路长度: {np.mean(paths):.3f} km")
    print(f"  光路长度标准差: {np.std(paths):.3f} km")

    tau_mc = estimate_optical_depth_monte_carlo(paths, 10.0)
    print(f"  蒙特卡洛估算有效光学厚度: {tau_mc:.4f}")




    print_section("5. 全球大气网格质量分析")

    nodes = generate_lat_lon_grid(6, 8)
    print(f"  生成全球网格节点数: {nodes.shape[0]}")

    dist_table = compute_distance_table(nodes)
    print(f"  距离矩阵维度: {dist_table.shape}")
    print(f"  最大球面距离: {dist_table.max():.1f} km")



    nodes_2d = nodes.copy()
    nodes_2d[:, 0] *= 111.0
    nodes_2d[:, 1] *= 111.0 * np.cos(np.radians(nodes[:, 0]))


    sub_nodes = nodes_2d[:16]
    tri = generate_simple_triangulation(sub_nodes)
    if tri.shape[0] > 0:
        metrics = compute_mesh_quality_metrics(sub_nodes, tri)
        print(f"  三角化质量指标:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
    else:
        print("  三角化质量指标: (非完全平方网格，跳过)")


    boundaries, mid_pts = define_atmospheric_layers(0.0, 20.0, 10)
    print(f"  大气分层数: 10")
    print(f"  层顶高度: {boundaries[-1]:.1f} km")
    print(f"  层中点: {np.round(mid_pts, 1)}")




    print_section("6. 气溶胶源区反演定位")


    station_pos = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [100.0, 100.0, 0.0],
        [50.0, 50.0, 0.0],
    ])

    true_source = np.array([30.0, 40.0, 0.5])
    Q_src = 1.0e4
    D_diff = 10.0
    L_decay = 200.0
    concentrations = np.zeros(5)
    for i in range(5):
        d = np.linalg.norm(station_pos[i] - true_source) + 1e-6
        concentrations[i] = Q_src / (4.0 * pi * D_diff * d) * exp(-d / L_decay)

    est_source, res_norm = inverse_source_location(
        station_pos, concentrations, Q=Q_src, D_diff=D_diff, L=L_decay
    )
    print(f"  真实源位置: ({true_source[0]:.2f}, {true_source[1]:.2f}, {true_source[2]:.2f}) km")
    print(f"  反演源位置: ({est_source[0]:.2f}, {est_source[1]:.2f}, {est_source[2]:.2f}) km")
    print(f"  定位误差: {np.linalg.norm(est_source - true_source):.2f} km")
    print(f"  残差范数: {res_norm:.4e}")




    print_section("7. 云凝结核 (CCN) 活化动力学")


    s_crit = kohler_critical_supersaturation(
        temperature=298.0,
        surface_tension=0.072,
        molecular_weight_water=0.018,
        density_water=1000.0,
        molecular_weight_solute=0.132,
        density_solute=1760.0,
        vanthoff_factor=3.0,
        dry_radius=0.05e-6,
        mass_solute=(4.0/3.0)*pi*(0.05e-6)**3*1760.0,
    )
    print(f"  临界过饱和度 S_crit = {s_crit*100:.4f}%")

    time_arr = np.linspace(0, 300, 50)
    s_env = 0.3 / 100.0
    f_act = activated_fraction_logistic(time_arr, s_env, s_crit, sigma_g=1.8)
    print(f"  环境过饱和度: {s_env*100:.2f}%")
    print(f"  t=0s  活化分数: {f_act[0]:.4f}")
    print(f"  t=60s 活化分数: {f_act[10]:.4f}")
    print(f"  t=300s 活化分数: {f_act[-1]:.4f}")


    ccn_conc = compute_ccn_number_concentration(
        0.5, N_total=1000.0, r_median=0.08, sigma_g=1.8
    )
    print(f"  S=0.5% 时 CCN 数浓度: {ccn_conc:.1f} cm⁻³")


    dndlns = ccn_spectrum_derivative(0.5/100.0, 1000.0, s_crit, 1.8)
    print(f"  S=0.5% 时 dN/dlnS: {dndlns:.2f} cm⁻³")




    print_section("8. AOD 统计协方差与 EOF 分析")


    stations = np.array([
        [39.9, 116.4],
        [40.7, -74.0],
        [-33.9, 18.4],
        [51.5, -0.1],
        [35.7, 139.7],
        [-23.5, -46.6],
    ])
    Sigma_theory = aod_covariance_model(stations, correlation_length=800.0, sigma_aod=0.20)
    print(f"  站点数: {stations.shape[0]}")
    print(f"  理论AOD协方差矩阵条件数: {np.linalg.cond(Sigma_theory):.2e}")


    S_sample = wishart_variate(Sigma_theory, n=6)
    print(f"  Wishart 样本协方差矩阵迹: {np.trace(S_sample):.4f}")


    eigvals, eigvecs, evr = eof_analysis(Sigma_theory, num_modes=3)
    print(f"  前 3 个 EOF 特征值: {np.round(eigvals, 4)}")
    print(f"  方差解释率: {np.round(evr*100, 2)}%")
    print(f"  累积方差解释率: {np.sum(evr)*100:.2f}%")




    print_section("9. 高维数值积分引擎验证")


    vol_est = integrate_tetrahedron(lambda pts: np.ones(pts.shape[0]))

    print(f"  单位四面体积分 ∫1 dV (规则归一化值≈1.0): {vol_est:.6f}")


    def f_square(pts):
        return pts[:, 0]**2 + pts[:, 1]**2
    sq_est = integrate_square(f_square, deg=6)
    print(f"  正方形积分 ∫(x²+y²)dA (解析=8/3≈2.6667): {sq_est:.6f}, 误差={abs(sq_est-8/3):.2e}")


    n_ch = 32
    x0 = np.cos((2.0*np.arange(1,n_ch+1)-1.0)*pi/(2.0*n_ch))
    cheb_est = (pi / n_ch) * np.sum(x0**2)
    print(f"  切比雪夫积分 ∫x²/sqrt(1-x²)dx (解析=π/2≈1.5708): {cheb_est:.6f}, 误差={abs(cheb_est-pi/2):.2e}")




    print_section("10. 数值工具验证")


    root, it_b = bisection(lambda x: x**3 - 2*x - 5, 2.0, 3.0, tol=1e-12)
    print(f"  二分法求根 x³-2x-5=0: root={root:.10f}, iter={it_b}")


    c_10_3 = binomial_coefficient(10, 3)
    print(f"  C(10,3) = {c_10_3:.0f} (理论=120)")


    comb_sel = comb_lexicographic(20, 5, 1)
    print(f"  comb(20,5,1) = {comb_sel}")


    wh_val = wilson_hilferty_chi_square(5, 0.5)
    print(f"  Wilson-Hilferty χ(5, z=0.5) = {wh_val:.4f}")


    print("\n" + "=" * 80)
    print("  计算流程全部完成，无报错。")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
