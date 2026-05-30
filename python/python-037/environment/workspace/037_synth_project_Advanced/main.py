
import sys
import numpy as np




from utils import (
    gauss_hermite_nodes_weights,
    r8_uniform_01,
    spherical_bessel_j1,
    double_factorial,
)
from wimp_physics import (
    helm_form_factor,
    reduced_mass,
    vmin_recoil,
    differential_rate,
    total_events_in_range,
    annual_modulation_factor,
    annual_modulated_rate,
)
from detector_geometry import (
    Mesh2D,
    create_sample_detector_mesh,
    get_fekete_rule,
    reference_to_physical_t3,
    triangle_area_2d,
)
from detector_field import (
    FEM1DSolver,
    r85_np_fs,
    r85_dif2,
    solve_diffusion_1d,
)
from particle_transport import (
    electron_drift_euler,
    ScintillationODESystem,
    lindhard_quenching_factor,
    ionization_yield,
    energy_deposition_profile,
)
from signal_formation import (
    pwl_approx_1d,
    pwl_interp_1d,
    aberth_ehrlich,
    cr_rc_n_pulse_response,
    shaped_pulse,
    add_electronic_noise,
    extract_pulse_parameters,
)
from monte_carlo_generator import (
    ReproducibleRNG,
    detection_efficiency,
    apply_energy_resolution,
    generate_wimp_events,
    generate_background_events,
)
from annual_modulation import (
    modulation_curve,
    modulation_curve_lissajous,
    bin_events_by_time,
    fit_modulation_amplitude,
    modulation_significance,
    analyze_modulation_by_energy_bins,
)
from event_reconstruction import (
    extract_event_features,
    build_distance_matrix,
    symmetrize_distance_matrix,
    single_linkage_clustering,
    fisher_discriminant,
    apply_discriminant_cut,
    evaluate_background_rejection,
)
from statistical_analysis import (
    glomin_brent,
    poisson_log_likelihood,
    profile_likelihood_ratio,
    confidence_interval_upper_limit,
    sensitivity_curve,
)
from sparse_matrix_utils import (
    SparseMatrixCOO,
    SparseMatrixCSR,
    coo_to_csr,
    csr_to_coo,
    expand_symmetric_coo,
    construct_fem_stiffness_sparse,
    sparse_matvec_timing,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_simulation():
    print("=" * 70)
    print("  暗物质直接探测信号模拟系统")
    print("  WIMP Direct Detection Signal Simulation Framework")
    print("=" * 70)




    print_section("1. 探测器与物理参数配置")


    DETECTOR_RADIUS_M = 0.05
    DETECTOR_THICKNESS_M = 0.02
    TARGET_MASS_KG = 10.0
    EXPOSURE_DAYS = 365.0
    E_MIN_KEV = 0.5
    E_MAX_KEV = 50.0


    M_CHI_GEV = 50.0
    SIGMA_PB = 1.0
    A_MASS = 73
    Z_ATOM = 32


    CATHODE_VOLTAGE = -1000.0
    ANODE_VOLTAGE = 0.0
    EPSILON_GERMANIUM = 16.0

    print(f"  探测器质量: {TARGET_MASS_KG} kg")
    print(f"  曝光时间: {EXPOSURE_DAYS} 天")
    print(f"  WIMP 质量: {M_CHI_GEV} GeV/c^2")
    print(f"  散射截面: {SIGMA_PB} pb")
    print(f"  靶核: Ge-{A_MASS} (Z={Z_ATOM})")




    print_section("2. 探测器几何与三角网格")

    mesh = create_sample_detector_mesh()
    total_area = mesh.total_area()
    print(f"  顶点数: {mesh.n_vertices()}")
    print(f"  三角形数: {mesh.n_triangles()}")
    print(f"  近似面积: {total_area:.4f} cm^2")


    def f_area(x, y):
        return 1.0
    area_integral = mesh.integrate_scalar(f_area, rule_id=2)
    print(f"  Fekete 积分验证面积: {area_integral:.4f} cm^2")




    print_section("3. 内部电场有限元求解")

    n_fem_nodes = 51
    z_nodes = np.linspace(0.0, DETECTOR_THICKNESS_M, n_fem_nodes)

    permittivity = EPSILON_GERMANIUM * 8.854e-12 * np.ones(n_fem_nodes - 1)
    charge_density = np.zeros(n_fem_nodes - 1)

    fem_solver = FEM1DSolver(z_nodes, permittivity, charge_density)
    phi = fem_solver.solve_dirichlet(CATHODE_VOLTAGE, ANODE_VOLTAGE)
    E_field = fem_solver.compute_electric_field(phi)

    print(f"  FEM 节点数: {n_fem_nodes}")
    print(f"  阴极电势: {phi[0]:.2f} V")
    print(f"  阳极电势: {phi[-1]:.2f} V")
    print(f"  平均电场强度: {np.mean(np.abs(E_field)):.2e} V/m")
    print(f"  最大电场强度: {np.max(np.abs(E_field)):.2e} V/m")


    query_z = np.array([0.005, 0.010, 0.015])
    phi_query = fem_solver.evaluate_at_points(phi, query_z)
    print(f"  z=5mm 处电势: {phi_query[0]:.2f} V")
    print(f"  z=10mm 处电势: {phi_query[1]:.2f} V")
    print(f"  z=15mm 处电势: {phi_query[2]:.2f} V")




    print_section("4. WIMP-核子散射物理")

    mu_chi_n = reduced_mass(M_CHI_GEV, A_MASS * 0.938272)
    print(f"  约化质量 μ_{{χN}}: {mu_chi_n:.4f} GeV/c^2")


    test_energies = np.array([1.0, 5.0, 10.0, 20.0, 30.0])
    print("  反冲能量 [keV]  |  Helm F^2(E)")
    print("  " + "-" * 35)
    for e in test_energies:
        ff2 = helm_form_factor(e, A_MASS)
        print(f"  {e:>10.1f}     |  {ff2:.6e}")


    vmin_10kev = vmin_recoil(10.0, M_CHI_GEV, A_MASS)
    print(f"  10 keV 反冲对应最小速度: {vmin_10kev:.2f} km/s")


    print("  反冲能量 [keV]  |  dR/dE [events/(keV·kg·day)]")
    print("  " + "-" * 50)
    for e in test_energies:
        rate = differential_rate(e, M_CHI_GEV, SIGMA_PB, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS)
        print(f"  {e:>10.1f}     |  {rate:.6e}")


    total_signal = total_events_in_range(
        E_MIN_KEV, E_MAX_KEV, M_CHI_GEV, SIGMA_PB, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS
    )
    print(f"  能窗 [{E_MIN_KEV}, {E_MAX_KEV}] keV 内预期 WIMP 事件数: {total_signal:.4f}")


    x_gh, w_gh = gauss_hermite_nodes_weights(16)
    moment2 = np.sum(w_gh * x_gh ** 2)
    print(f"  Gauss-Hermite 二阶矩验证: {moment2:.10f} (理论值: {np.sqrt(np.pi) / 2:.10f})")




    print_section("5. 蒙特卡洛事件产生")

    rng = ReproducibleRNG(seed=20240503)

    n_signal_target = 200
    n_background_target = 300

    signal_events = generate_wimp_events(
        n_signal_target,
        M_CHI_GEV,
        SIGMA_PB,
        A_MASS,
        TARGET_MASS_KG,
        EXPOSURE_DAYS,
        E_MIN_KEV,
        E_MAX_KEV,
        DETECTOR_RADIUS_M,
        DETECTOR_THICKNESS_M,
        rng,
        apply_modulation=True,
    )

    background_events = generate_background_events(
        n_background_target,
        E_MIN_KEV,
        E_MAX_KEV,
        DETECTOR_RADIUS_M,
        DETECTOR_THICKNESS_M,
        rng,
    )

    print(f"  生成 WIMP 信号事件数: {len(signal_events)}")
    print(f"  生成背景事件数: {len(background_events)}")

    if signal_events:
        energies_sig = [ev["energy_obs"] for ev in signal_events]
        print(f"  信号能量范围: [{min(energies_sig):.2f}, {max(energies_sig):.2f}] keV")
        print(f"  信号平均能量: {np.mean(energies_sig):.2f} keV")

    if background_events:
        bg_types = {}
        for ev in background_events:
            t = ev["type"]
            bg_types[t] = bg_types.get(t, 0) + 1
        print(f"  背景组成: {bg_types}")




    print_section("6. 粒子输运与能量沉积")


    def uniform_e_field(r):
        return np.array([0.0, 0.0, np.mean(np.abs(E_field))])

    t_drift, traj_drift = electron_drift_euler(
        uniform_e_field,
        np.array([0.0, 0.0, DETECTOR_THICKNESS_M * 0.5]),
        (0.0, 1.0e-6),
        100,
        mobility=3.0e-4,
    )
    print(f"  电子漂移起点: z={traj_drift[0, 2]:.4e} m")
    print(f"  电子漂移终点: z={traj_drift[-1, 2]:.4e} m")
    print(f"  漂移距离: {abs(traj_drift[-1, 2] - traj_drift[0, 2]):.4e} m")


    scint_sys = ScintillationODESystem(e_dep_norm=1.0)
    y0 = np.array([0.0, 0.0])
    t_pulse, y_pulse = scint_sys.solve_euler(y0, (0.0, 5.0e-6), 500)
    eq_state = scint_sys.equilibrium()
    print(f"  闪烁系统稳态: P*={eq_state[0]:.4e}, Q*={eq_state[1]:.4e}")
    print(f"  脉冲峰值 P: {np.max(y_pulse[:, 0]):.4e}")
    print(f"  脉冲峰值 Q: {np.max(y_pulse[:, 1]):.4e}")


    qf_vals = [lindhard_quenching_factor(e, Z_ATOM, A_MASS) for e in test_energies]
    print("  反冲能量 [keV]  |  Quenching Factor")
    print("  " + "-" * 38)
    for e, qf in zip(test_energies, qf_vals):
        print(f"  {e:>10.1f}     |  {qf:.6f}")


    ne, sig_ne = ionization_yield(10.0, Z_ATOM, A_MASS)
    print(f"  10 keV 反冲电离电子数: {ne:.1f} ± {sig_ne:.1f}")


    z_dep, edep = energy_deposition_profile(10.0, DETECTOR_THICKNESS_M, n_bins=20)
    print(f"  10 keV 沉积分布总能量验证: {np.sum(edep) * (z_dep[1] - z_dep[0]):.4f} keV")




    print_section("7. 探测器信号成形与处理")


    t_signal = np.linspace(0.0, 20.0e-6, 1000)
    h_pulse = cr_rc_n_pulse_response(t_signal, tau_cr=1.0e-6, tau_rc=2.0e-6, n_rc=4)
    print(f"  CR-RC^4 成形器脉冲响应峰值: {np.max(h_pulse):.4e}")
    print(f"  脉冲响应面积: {np.trapezoid(h_pulse, t_signal):.4e}")


    arrival_times = np.array([1.0, 2.5, 4.0, 6.0, 8.0]) * 1.0e-6
    charge_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    V_shaped = shaped_pulse(t_signal, arrival_times, charge_values)
    print(f"  叠加信号峰值: {np.max(V_shaped):.4e} V")


    V_noisy = add_electronic_noise(V_shaped, dt=t_signal[1] - t_signal[0])
    baseline, amplitude, risetime = extract_pulse_parameters(t_signal, V_noisy)
    print(f"  基线: {baseline:.4e} V")
    print(f"  信号幅度: {amplitude:.4e} V")
    print(f"  10%-90% 上升时间: {risetime:.4e} s")




    tau_inv_rc = 1.0 / 2.0e-6
    tau_inv_cr = 1.0 / 1.0e-6

    a = tau_inv_rc
    b = tau_inv_cr
    coeffs = np.array([
        a ** 4 * b,
        4.0 * a ** 3 * b + a ** 4,
        6.0 * a ** 2 * b + 4.0 * a ** 3,
        4.0 * a * b + 6.0 * a ** 2,
        b + 4.0 * a,
        1.0,
    ])
    roots = aberth_ehrlich(coeffs, max_iter=200)
    print(f"  传递函数极点数: {len(roots)}")
    for i, r in enumerate(roots):
        print(f"    极点 {i+1}: s = {r:.4e} [rad/s]")


    t_pwl = np.linspace(0.0, 20.0e-6, 200)
    h_pwl = cr_rc_n_pulse_response(t_pwl)
    xc = np.linspace(0.0, 20.0e-6, 21)
    yc = pwl_approx_1d(len(t_pwl), t_pwl, h_pwl, len(xc), xc)
    xi = np.linspace(0.0, 20.0e-6, 400)
    yi = pwl_interp_1d(len(xc), xc, yc, len(xi), xi)
    y_true = cr_rc_n_pulse_response(xi)
    rmse_pwl = np.sqrt(np.mean((yi - y_true) ** 2))
    print(f"  PWL 近似 RMSE: {rmse_pwl:.4e}")




    print_section("8. 年度调制分析")

    if signal_events:
        t_bins, counts, errors = bin_events_by_time(signal_events, n_bins=12)
        s0_fit, sm_fit, phase_fit, chi2 = fit_modulation_amplitude(t_bins, counts, errors)
        sig = modulation_significance(s0_fit, sm_fit, float(len(signal_events)))

        print(f"  时间分箱数: 12 (月度)")
        print(f"  平均计数率 S0: {s0_fit:.4f} events/bin")
        print(f"  调制振幅 Sm: {sm_fit:.4f} events/bin")
        print(f"  调制相位: {phase_fit:.4f} rad")
        print(f"  拟合 χ²: {chi2:.4f}")
        print(f"  调制显著性: {sig:.4f} σ")


        energy_edges = np.array([0.5, 5.0, 10.0, 20.0, 50.0])
        mod_results = analyze_modulation_by_energy_bins(signal_events, energy_edges)
        print("  分能区调制分析:")
        for res in mod_results:
            if res["s0"] is not None:
                print(f"    [{res['energy_low']:.1f}, {res['energy_high']:.1f}] keV: "
                      f"N={res['n_events']}, S0={res['s0']:.2f}, Sm={res['sm']:.2f}, "
                      f"Sig={res['significance']:.2f}σ")


        t_liss = np.linspace(0.0, 365.25, 100)
        X_liss, Y_liss = modulation_curve_lissajous(t_liss, s0_fit, sm_fit)
        print(f"  Lissajous 参数曲线闭合性检查: |X[0]-X[-1]|={abs(X_liss[0]-X_liss[-1]):.4e}")




    print_section("9. 事件重建与背景甄别")

    if signal_events and background_events:
        X_s = extract_event_features(signal_events)
        X_b = extract_event_features(background_events)


        w_fisher, threshold_fisher, separation = fisher_discriminant(X_s, X_b)
        print(f"  Fisher 分离度: {separation:.4f}")
        print(f"  判别阈值: {threshold_fisher:.4f}")


        eval_result = evaluate_background_rejection(
            signal_events, background_events, w_fisher, threshold_fisher, target_efficiency=0.9
        )
        print(f"  信号效率: {eval_result['signal_efficiency']:.4f}")
        print(f"  背景抑制因子: {eval_result['background_rejection']:.4f}")
        print(f"  纯度: {eval_result['purity']:.4f}")


        n_demo = min(30, len(signal_events))
        X_demo = X_s[:n_demo]
        D_demo = build_distance_matrix(X_demo, weights=np.array([2.0, 1.0, 1.0, 1.0]))
        D_sym = symmetrize_distance_matrix(D_demo)
        linkage_demo, labels_demo = single_linkage_clustering(D_sym)
        n_clusters = len(np.unique(labels_demo))
        print(f"  层次聚类演示: {n_demo} 个事件 → {n_clusters} 个簇")




    print_section("10. 统计推断与灵敏度")


    n_bins_e = 10
    e_edges = np.linspace(E_MIN_KEV, E_MAX_KEV, n_bins_e + 1)
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])

    n_obs = np.zeros(n_bins_e)
    s_pred = np.zeros(n_bins_e)
    b_pred = np.zeros(n_bins_e)

    for ev in signal_events:
        e = ev["energy_obs"]
        idx = int(np.clip(np.digitize(e, e_edges) - 1, 0, n_bins_e - 1))
        n_obs[idx] += 1.0

    for ev in background_events:
        e = ev["energy_obs"]
        idx = int(np.clip(np.digitize(e, e_edges) - 1, 0, n_bins_e - 1))
        n_obs[idx] += 1.0


    for i in range(n_bins_e):
        e_low, e_high = e_edges[i], e_edges[i + 1]
        s_pred[i] = total_events_in_range(
            e_low, e_high, M_CHI_GEV, 1.0, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS
        )

        b_pred[i] = 0.5 * (e_high - e_low) * len(background_events) / (E_MAX_KEV - E_MIN_KEV)

    print(f"  观测计数: {n_obs}")
    print(f"  预期信号 (σ=1pb): {s_pred}")
    print(f"  预期背景: {b_pred}")


    q_mu0 = profile_likelihood_ratio(n_obs, s_pred, b_pred, 0.0)
    print(f"  μ=0 轮廓似然比 q_0: {q_mu0:.4f}")


    mu_90 = confidence_interval_upper_limit(n_obs, s_pred, b_pred)
    print(f"  信号强度 90% CL 上限: μ_90 = {mu_90:.4f}")


    m_chi_scan = np.array([10.0, 30.0, 50.0, 100.0, 200.0, 500.0])
    sigma_90 = sensitivity_curve(
        exposure_kg_day=TARGET_MASS_KG * EXPOSURE_DAYS,
        target_mass_kg=TARGET_MASS_KG,
        background_rate_per_kev_kg_day=0.5,
        e_min_kev=E_MIN_KEV,
        e_max_kev=E_MAX_KEV,
        m_chi_values=m_chi_scan,
    )
    print("  WIMP 质量 [GeV]  |  90% CL σ [pb]")
    print("  " + "-" * 40)
    for m, s in zip(m_chi_scan, sigma_90):
        print(f"  {m:>10.1f}     |  {s:.6e}")


    def neg_logL_mu(mu):
        return -poisson_log_likelihood(n_obs, s_pred, b_pred, mu)

    mu_opt, negLL_opt, ncalls = glomin_brent(
        neg_logL_mu, 0.0, 10.0, 1.0, 100.0, 1.0e-8, 1.0e-8
    )
    print(f"  最大似然估计: μ_hat = {mu_opt:.4f} (logL={-negLL_opt:.2f}, calls={ncalls})")




    print_section("11. 稀疏矩阵与性能测试")


    n_sparse = 1000
    h_sparse = DETECTOR_THICKNESS_M / (n_sparse - 1)
    fem_sparse_coo = construct_fem_stiffness_sparse(n_sparse, h_sparse)
    fem_sparse_csr = fem_sparse_coo.to_csr()
    print(f"  FEM 刚度矩阵维度: {n_sparse} × {n_sparse}")
    print(f"  非零元数: {fem_sparse_csr.nnz()}")
    print(f"  稀疏度: {fem_sparse_csr.nnz() / (n_sparse ** 2):.6f}")

    x_test = np.ones(n_sparse)
    y_test = fem_sparse_csr.matvec(x_test)
    print(f"  稀疏 matvec 结果范数: {np.linalg.norm(y_test):.4e}")

    avg_time = sparse_matvec_timing(fem_sparse_csr, x_test, n_repeat=50)
    print(f"  单次 matvec 平均耗时: {avg_time:.6e} s")


    n_diff = 101
    dx_diff = 0.001
    phi_diff = solve_diffusion_1d(
        n_diff, D=1.0e-4, sigma_a=0.1, source=np.zeros(n_diff),
        dx=dx_diff, bc_left=0.0, bc_right=1.0
    )
    print(f"  扩散方程求解验证: φ(0)={phi_diff[0]:.4e}, φ(L)={phi_diff[-1]:.4e}")




    print_section("模拟完成")
    print("  所有模块运行正常，未检测到错误。")
    print("  本演示涵盖了暗物质直接探测实验的完整模拟链。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_simulation())
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
