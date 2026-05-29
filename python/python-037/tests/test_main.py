r"""
main.py
暗物质直接探测信号模拟——统一入口

本脚本执行完整的暗物质直接探测实验模拟流程：
1. 探测器几何建模与网格生成
2. 内部电场有限元求解（1D FEM + 五对角求解器）
3. WIMP 散射物理计算（微分率、Helm 形状因子）
4. 蒙特卡洛事件生成（信号 + 背景）
5. 粒子输运与信号形成（电子漂移、闪烁脉冲）
6. 成形器传递函数与极点分析
7. 年度调制分析（Lissajous 参数曲线）
8. 事件重建与背景甄别（Fisher 判别 + 层次聚类）
9. 统计推断（轮廓似然、灵敏度曲线）
10. 稀疏矩阵性能测试

运行方式：
    python main.py

无需任何命令行参数。
"""

import sys
import numpy as np

# ========================================================================
# 导入所有子模块
# ========================================================================
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

    # =====================================================================
    # 1. 物理参数与探测器配置
    # =====================================================================
    print_section("1. 探测器与物理参数配置")

    # 探测器参数
    DETECTOR_RADIUS_M = 0.05        # 探测器半径 5 cm
    DETECTOR_THICKNESS_M = 0.02     # 探测器厚度 2 cm
    TARGET_MASS_KG = 10.0           # 靶质量 10 kg
    EXPOSURE_DAYS = 365.0           # 1 年曝光
    E_MIN_KEV = 0.5
    E_MAX_KEV = 50.0

    # WIMP 参数
    M_CHI_GEV = 50.0                # WIMP 质量 50 GeV/c^2
    SIGMA_PB = 1.0                  # WIMP-核子截面 1 pb
    A_MASS = 73                     # 锗-73 靶核
    Z_ATOM = 32                     # 锗原子序数

    # 电学参数
    CATHODE_VOLTAGE = -1000.0       # V
    ANODE_VOLTAGE = 0.0             # V
    EPSILON_GERMANIUM = 16.0        # 相对介电常数

    print(f"  探测器质量: {TARGET_MASS_KG} kg")
    print(f"  曝光时间: {EXPOSURE_DAYS} 天")
    print(f"  WIMP 质量: {M_CHI_GEV} GeV/c^2")
    print(f"  散射截面: {SIGMA_PB} pb")
    print(f"  靶核: Ge-{A_MASS} (Z={Z_ATOM})")

    # =====================================================================
    # 2. 探测器几何与网格
    # =====================================================================
    print_section("2. 探测器几何与三角网格")

    mesh = create_sample_detector_mesh()
    total_area = mesh.total_area()
    print(f"  顶点数: {mesh.n_vertices()}")
    print(f"  三角形数: {mesh.n_triangles()}")
    print(f"  近似面积: {total_area:.4f} cm^2")

    # 在网格上积分常数函数（验证面积）
    def f_area(x, y):
        return 1.0
    area_integral = mesh.integrate_scalar(f_area, rule_id=2)
    print(f"  Fekete 积分验证面积: {area_integral:.4f} cm^2")

    # =====================================================================
    # 3. 内部电场 FEM 求解
    # =====================================================================
    print_section("3. 内部电场有限元求解")

    n_fem_nodes = 51
    z_nodes = np.linspace(0.0, DETECTOR_THICKNESS_M, n_fem_nodes)
    # 假设均匀介电常数和零空间电荷
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

    # 插值到查询点
    query_z = np.array([0.005, 0.010, 0.015])
    phi_query = fem_solver.evaluate_at_points(phi, query_z)
    print(f"  z=5mm 处电势: {phi_query[0]:.2f} V")
    print(f"  z=10mm 处电势: {phi_query[1]:.2f} V")
    print(f"  z=15mm 处电势: {phi_query[2]:.2f} V")

    # =====================================================================
    # 4. WIMP 散射物理计算
    # =====================================================================
    print_section("4. WIMP-核子散射物理")

    mu_chi_n = reduced_mass(M_CHI_GEV, A_MASS * 0.938272)
    print(f"  约化质量 μ_{{χN}}: {mu_chi_n:.4f} GeV/c^2")

    # Helm 形状因子
    test_energies = np.array([1.0, 5.0, 10.0, 20.0, 30.0])
    print("  反冲能量 [keV]  |  Helm F^2(E)")
    print("  " + "-" * 35)
    for e in test_energies:
        ff2 = helm_form_factor(e, A_MASS)
        print(f"  {e:>10.1f}     |  {ff2:.6e}")

    # 最小反冲速度
    vmin_10kev = vmin_recoil(10.0, M_CHI_GEV, A_MASS)
    print(f"  10 keV 反冲对应最小速度: {vmin_10kev:.2f} km/s")

    # 微分事件率
    print("  反冲能量 [keV]  |  dR/dE [events/(keV·kg·day)]")
    print("  " + "-" * 50)
    for e in test_energies:
        rate = differential_rate(e, M_CHI_GEV, SIGMA_PB, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS)
        print(f"  {e:>10.1f}     |  {rate:.6e}")

    # 总预期事件数
    total_signal = total_events_in_range(
        E_MIN_KEV, E_MAX_KEV, M_CHI_GEV, SIGMA_PB, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS
    )
    print(f"  能窗 [{E_MIN_KEV}, {E_MAX_KEV}] keV 内预期 WIMP 事件数: {total_signal:.4f}")

    # Gauss-Hermite 求积验证
    x_gh, w_gh = gauss_hermite_nodes_weights(16)
    moment2 = np.sum(w_gh * x_gh ** 2)
    print(f"  Gauss-Hermite 二阶矩验证: {moment2:.10f} (理论值: {np.sqrt(np.pi) / 2:.10f})")

    # =====================================================================
    # 5. 蒙特卡洛事件产生
    # =====================================================================
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

    # =====================================================================
    # 6. 粒子输运模拟
    # =====================================================================
    print_section("6. 粒子输运与能量沉积")

    # 电子漂移模拟
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

    # 闪烁脉冲 ODE
    scint_sys = ScintillationODESystem(e_dep_norm=1.0)
    y0 = np.array([0.0, 0.0])
    t_pulse, y_pulse = scint_sys.solve_euler(y0, (0.0, 5.0e-6), 500)
    eq_state = scint_sys.equilibrium()
    print(f"  闪烁系统稳态: P*={eq_state[0]:.4e}, Q*={eq_state[1]:.4e}")
    print(f"  脉冲峰值 P: {np.max(y_pulse[:, 0]):.4e}")
    print(f"  脉冲峰值 Q: {np.max(y_pulse[:, 1]):.4e}")

    # Lindhard Quenching Factor
    qf_vals = [lindhard_quenching_factor(e, Z_ATOM, A_MASS) for e in test_energies]
    print("  反冲能量 [keV]  |  Quenching Factor")
    print("  " + "-" * 38)
    for e, qf in zip(test_energies, qf_vals):
        print(f"  {e:>10.1f}     |  {qf:.6f}")

    # 电离产额
    ne, sig_ne = ionization_yield(10.0, Z_ATOM, A_MASS)
    print(f"  10 keV 反冲电离电子数: {ne:.1f} ± {sig_ne:.1f}")

    # 能量沉积分布
    z_dep, edep = energy_deposition_profile(10.0, DETECTOR_THICKNESS_M, n_bins=20)
    print(f"  10 keV 沉积分布总能量验证: {np.sum(edep) * (z_dep[1] - z_dep[0]):.4f} keV")

    # =====================================================================
    # 7. 信号成形与处理
    # =====================================================================
    print_section("7. 探测器信号成形与处理")

    # 成形器脉冲响应
    t_signal = np.linspace(0.0, 20.0e-6, 1000)
    h_pulse = cr_rc_n_pulse_response(t_signal, tau_cr=1.0e-6, tau_rc=2.0e-6, n_rc=4)
    print(f"  CR-RC^4 成形器脉冲响应峰值: {np.max(h_pulse):.4e}")
    print(f"  脉冲响应面积: {np.trapezoid(h_pulse, t_signal):.4e}")

    # 叠加脉冲（模拟 5 个电子团到达）
    arrival_times = np.array([1.0, 2.5, 4.0, 6.0, 8.0]) * 1.0e-6
    charge_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    V_shaped = shaped_pulse(t_signal, arrival_times, charge_values)
    print(f"  叠加信号峰值: {np.max(V_shaped):.4e} V")

    # 添加噪声
    V_noisy = add_electronic_noise(V_shaped, dt=t_signal[1] - t_signal[0])
    baseline, amplitude, risetime = extract_pulse_parameters(t_signal, V_noisy)
    print(f"  基线: {baseline:.4e} V")
    print(f"  信号幅度: {amplitude:.4e} V")
    print(f"  10%-90% 上升时间: {risetime:.4e} s")

    # Aberth-Ehrlich 极点分析
    # CR-RC^4 传递函数分母近似多项式
    # (s + 1/τ_rc)^4 (s + 1/τ_cr) 的展开
    tau_inv_rc = 1.0 / 2.0e-6
    tau_inv_cr = 1.0 / 1.0e-6
    # 构造多项式 p(s) = (s + a)^4 (s + b) = s^5 + (4a+b)s^4 + (6a^2+4ab)s^3 + ...
    a = tau_inv_rc
    b = tau_inv_cr
    coeffs = np.array([
        a ** 4 * b,                       # s^0
        4.0 * a ** 3 * b + a ** 4,       # s^1
        6.0 * a ** 2 * b + 4.0 * a ** 3, # s^2
        4.0 * a * b + 6.0 * a ** 2,      # s^3
        b + 4.0 * a,                     # s^4
        1.0,                             # s^5
    ])
    roots = aberth_ehrlich(coeffs, max_iter=200)
    print(f"  传递函数极点数: {len(roots)}")
    for i, r in enumerate(roots):
        print(f"    极点 {i+1}: s = {r:.4e} [rad/s]")

    # PWL 近似测试
    t_pwl = np.linspace(0.0, 20.0e-6, 200)
    h_pwl = cr_rc_n_pulse_response(t_pwl)
    xc = np.linspace(0.0, 20.0e-6, 21)
    yc = pwl_approx_1d(len(t_pwl), t_pwl, h_pwl, len(xc), xc)
    xi = np.linspace(0.0, 20.0e-6, 400)
    yi = pwl_interp_1d(len(xc), xc, yc, len(xi), xi)
    y_true = cr_rc_n_pulse_response(xi)
    rmse_pwl = np.sqrt(np.mean((yi - y_true) ** 2))
    print(f"  PWL 近似 RMSE: {rmse_pwl:.4e}")

    # =====================================================================
    # 8. 年度调制分析
    # =====================================================================
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

        # 多能量区间分析
        energy_edges = np.array([0.5, 5.0, 10.0, 20.0, 50.0])
        mod_results = analyze_modulation_by_energy_bins(signal_events, energy_edges)
        print("  分能区调制分析:")
        for res in mod_results:
            if res["s0"] is not None:
                print(f"    [{res['energy_low']:.1f}, {res['energy_high']:.1f}] keV: "
                      f"N={res['n_events']}, S0={res['s0']:.2f}, Sm={res['sm']:.2f}, "
                      f"Sig={res['significance']:.2f}σ")

        # Lissajous 参数曲线
        t_liss = np.linspace(0.0, 365.25, 100)
        X_liss, Y_liss = modulation_curve_lissajous(t_liss, s0_fit, sm_fit)
        print(f"  Lissajous 参数曲线闭合性检查: |X[0]-X[-1]|={abs(X_liss[0]-X_liss[-1]):.4e}")

    # =====================================================================
    # 9. 事件重建与背景甄别
    # =====================================================================
    print_section("9. 事件重建与背景甄别")

    if signal_events and background_events:
        X_s = extract_event_features(signal_events)
        X_b = extract_event_features(background_events)

        # Fisher 判别
        w_fisher, threshold_fisher, separation = fisher_discriminant(X_s, X_b)
        print(f"  Fisher 分离度: {separation:.4f}")
        print(f"  判别阈值: {threshold_fisher:.4f}")

        # 应用切割
        eval_result = evaluate_background_rejection(
            signal_events, background_events, w_fisher, threshold_fisher, target_efficiency=0.9
        )
        print(f"  信号效率: {eval_result['signal_efficiency']:.4f}")
        print(f"  背景抑制因子: {eval_result['background_rejection']:.4f}")
        print(f"  纯度: {eval_result['purity']:.4f}")

        # 层次聚类（取前 30 个事件演示）
        n_demo = min(30, len(signal_events))
        X_demo = X_s[:n_demo]
        D_demo = build_distance_matrix(X_demo, weights=np.array([2.0, 1.0, 1.0, 1.0]))
        D_sym = symmetrize_distance_matrix(D_demo)
        linkage_demo, labels_demo = single_linkage_clustering(D_sym)
        n_clusters = len(np.unique(labels_demo))
        print(f"  层次聚类演示: {n_demo} 个事件 → {n_clusters} 个簇")

    # =====================================================================
    # 10. 统计推断
    # =====================================================================
    print_section("10. 统计推断与灵敏度")

    # 构造能谱直方图
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

    # 预期信号（单位截面）
    for i in range(n_bins_e):
        e_low, e_high = e_edges[i], e_edges[i + 1]
        s_pred[i] = total_events_in_range(
            e_low, e_high, M_CHI_GEV, 1.0, A_MASS, TARGET_MASS_KG, EXPOSURE_DAYS
        )
        # 简化背景预期（平坦谱近似）
        b_pred[i] = 0.5 * (e_high - e_low) * len(background_events) / (E_MAX_KEV - E_MIN_KEV)

    print(f"  观测计数: {n_obs}")
    print(f"  预期信号 (σ=1pb): {s_pred}")
    print(f"  预期背景: {b_pred}")

    # 轮廓似然比
    q_mu0 = profile_likelihood_ratio(n_obs, s_pred, b_pred, 0.0)
    print(f"  μ=0 轮廓似然比 q_0: {q_mu0:.4f}")

    # 90% CL 上限
    mu_90 = confidence_interval_upper_limit(n_obs, s_pred, b_pred)
    print(f"  信号强度 90% CL 上限: μ_90 = {mu_90:.4f}")

    # 灵敏度曲线
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

    # 全局优化演示
    def neg_logL_mu(mu):
        return -poisson_log_likelihood(n_obs, s_pred, b_pred, mu)

    mu_opt, negLL_opt, ncalls = glomin_brent(
        neg_logL_mu, 0.0, 10.0, 1.0, 100.0, 1.0e-8, 1.0e-8
    )
    print(f"  最大似然估计: μ_hat = {mu_opt:.4f} (logL={-negLL_opt:.2f}, calls={ncalls})")

    # =====================================================================
    # 11. 稀疏矩阵与 FEM 性能测试
    # =====================================================================
    print_section("11. 稀疏矩阵与性能测试")

    # 构造 FEM 刚度矩阵
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

    # 五对角扩散方程求解
    n_diff = 101
    dx_diff = 0.001
    phi_diff = solve_diffusion_1d(
        n_diff, D=1.0e-4, sigma_a=0.1, source=np.zeros(n_diff),
        dx=dx_diff, bc_left=0.0, bc_right=1.0
    )
    print(f"  扩散方程求解验证: φ(0)={phi_diff[0]:.4e}, φ(L)={phi_diff[-1]:.4e}")

    # =====================================================================
    # 完成
    # =====================================================================
    print_section("模拟完成")
    print("  所有模块运行正常，未检测到错误。")
    print("  本演示涵盖了暗物质直接探测实验的完整模拟链。")
    print("=" * 70)

    return 0


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: spherical_bessel_j1 零值极限 ----
j1_0 = spherical_bessel_j1(0.0)
assert abs(j1_0) < 1e-12, '[TC01] spherical_bessel_j1 零值极限 FAILED'

# ---- TC02: double_factorial 基本值 ----
assert double_factorial(7) == 105.0, '[TC02] double_factorial 基本值 FAILED'

# ---- TC03: gauss_hermite_nodes_weights 二阶矩验证 ----
x_gh, w_gh = gauss_hermite_nodes_weights(16)
moment2 = np.sum(w_gh * x_gh ** 2)
assert abs(moment2 - np.sqrt(np.pi) / 2.0) < 1e-10, '[TC03] Gauss-Hermite 二阶矩验证 FAILED'

# ---- TC04: helm_form_factor 低能极限接近 1 ----
ff2 = helm_form_factor(1.0, 73.0)
assert 0.5 < ff2 <= 1.0, '[TC04] helm_form_factor 低能极限 FAILED'

# ---- TC05: reduced_mass 交换对称性 ----
mu_ab = reduced_mass(50.0, 73.0 * 0.938272)
mu_ba = reduced_mass(73.0 * 0.938272, 50.0)
assert abs(mu_ab - mu_ba) < 1e-12, '[TC05] reduced_mass 交换对称性 FAILED'

# ---- TC06: vmin_recoil 结果为正 ----
vm = vmin_recoil(10.0, 50.0, 73.0)
assert vm > 0.0, '[TC06] vmin_recoil 结果为正 FAILED'

# ---- TC07: differential_rate 非负有限 ----
rate = differential_rate(10.0, 50.0, 1.0, 73.0, 1.0, 365.0)
assert rate >= 0.0 and np.isfinite(rate), '[TC07] differential_rate 非负有限 FAILED'

# ---- TC08: total_events_in_range 非负有限 ----
nevt = total_events_in_range(0.5, 50.0, 50.0, 1.0, 73.0, 10.0, 365.0)
assert nevt >= 0.0 and np.isfinite(nevt), '[TC08] total_events_in_range 非负有限 FAILED'

# ---- TC09: annual_modulation_factor 平均值约等于 1 ----
t_days = np.linspace(0.0, 365.25, 100)
mod_factors = np.array([annual_modulation_factor(t) for t in t_days])
assert abs(np.mean(mod_factors) - 1.0) < 0.01, '[TC09] annual_modulation_factor 平均值 FAILED'

# ---- TC10: get_fekete_rule 返回正确规则度数与节点数 ----
ref_pts, ref_w, deg = get_fekete_rule(2)
assert deg == 6, '[TC10] get_fekete_rule 规则度数 FAILED'
assert len(ref_w) == 7, '[TC10] get_fekete_rule 节点数 FAILED'

# ---- TC11: triangle_area_2d 标准三角形 ----
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area = triangle_area_2d(verts)
assert abs(area - 0.5) < 1e-12, '[TC11] triangle_area_2d 标准三角形 FAILED'

# ---- TC12: Mesh2D 样本网格顶点与三角形数 ----
mesh = create_sample_detector_mesh()
assert mesh.n_vertices() == 13, '[TC12] Mesh2D 顶点数 FAILED'
assert mesh.n_triangles() == 12, '[TC12] Mesh2D 三角形数 FAILED'

# ---- TC13: FEM1DSolver Dirichlet 边界条件 ----
nodes = np.linspace(0.0, 1.0, 11)
eps = np.ones(10)
rho = np.zeros(10)
solver = FEM1DSolver(nodes, eps, rho)
phi = solver.solve_dirichlet(0.0, 1.0)
assert abs(phi[0]) < 1e-10, '[TC13] FEM 左边界 FAILED'
assert abs(phi[-1] - 1.0) < 1e-10, '[TC13] FEM 右边界 FAILED'

# ---- TC14: r85_np_fs 求解线性系统验证 ----
a_r85 = r85_dif2(5)
b_r85 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_r85 = r85_np_fs(5, a_r85, b_r85)
full = np.zeros((5, 5))
for i in range(5):
    full[i, i] = a_r85[0, i]
    if i + 1 < 5:
        full[i, i + 1] = a_r85[1, i]
    if i - 1 >= 0:
        full[i, i - 1] = a_r85[3, i]
residual = np.linalg.norm(full @ x_r85 - b_r85)
assert residual < 1e-10, '[TC14] r85_np_fs 求解验证 FAILED'

# ---- TC15: solve_diffusion_1d 边界条件 ----
phi_diff = solve_diffusion_1d(21, D=1.0, sigma_a=0.1, source=np.zeros(21), dx=0.05, bc_left=0.0, bc_right=1.0)
assert abs(phi_diff[0]) < 1e-10, '[TC15] 扩散方程左边界 FAILED'
assert abs(phi_diff[-1] - 1.0) < 1e-10, '[TC15] 扩散方程右边界 FAILED'

# ---- TC16: electron_drift_euler 恒电场漂移距离 ----
t_drift, r_drift = electron_drift_euler(lambda r: np.array([0.0, 0.0, 1.0e3]), np.array([0.0, 0.0, 0.0]), (0.0, 1.0e-6), 100, mobility=3.0e-4)
expected_z = 3.0e-4 * 1.0e3 * 1.0e-6
assert abs(r_drift[-1, 2] - expected_z) < 1e-12, '[TC16] 电子漂移距离 FAILED'

# ---- TC17: ScintillationODESystem 稳态解非负 ----
scint_sys = ScintillationODESystem(e_dep_norm=1.0)
eq = scint_sys.equilibrium()
assert np.all(eq >= 0.0), '[TC17] 闪烁稳态解非负 FAILED'

# ---- TC18: lindhard_quenching_factor 范围 0 到 1 ----
qf = lindhard_quenching_factor(10.0, 32, 73)
assert 0.0 <= qf <= 1.0, '[TC18] Lindhard QF 范围 FAILED'

# ---- TC19: ionization_yield 非负 ----
ne, sig = ionization_yield(10.0, 32, 73)
assert ne >= 0.0 and sig >= 0.0, '[TC19] 电离产额非负 FAILED'

# ---- TC20: energy_deposition_profile 归一化能量守恒 ----
np.random.seed(42)
z_dep, edep = energy_deposition_profile(10.0, 0.02, n_bins=20)
dz = z_dep[1] - z_dep[0]
total_energy = np.sum(edep) * dz
assert abs(total_energy - 10.0) < 1.0, '[TC20] 能量沉积守恒 FAILED'

# ---- TC21: cr_rc_n_pulse响应非负性 ----
t_pulse = np.linspace(0.0, 20.0e-6, 1000)
h_pulse = cr_rc_n_pulse_response(t_pulse, tau_cr=1.0e-6, tau_rc=2.0e-6, n_rc=4)
assert np.all(h_pulse >= 0.0), '[TC21] CR-RC^n 脉冲响应非负 FAILED'

# ---- TC22: aberth_ehrlich x^3-1=0 求根验证 ----
coeffs_ae = np.array([-1.0, 0.0, 0.0, 1.0])
roots_ae = aberth_ehrlich(coeffs_ae, max_iter=200)
for r in roots_ae:
    assert abs(r**3 - 1.0) < 1e-8, '[TC22] Aberth-Ehrlich 求根 FAILED'

# ---- TC23: pwl_interp_1d 插值精度 ----
xd_pwl = np.array([0.0, 0.5, 1.0])
yd_pwl = np.array([0.0, 0.5, 1.0])
xi_pwl = np.array([0.25, 0.75])
yi_pwl = pwl_interp_1d(3, xd_pwl, yd_pwl, 2, xi_pwl)
assert abs(yi_pwl[0] - 0.25) < 1e-12, '[TC23] PWL 插值精度 FAILED'
assert abs(yi_pwl[1] - 0.75) < 1e-12, '[TC23] PWL 插值精度 FAILED'

# ---- TC24: ReproducibleRNG 可复现性 ----
rng1 = ReproducibleRNG(seed=42)
rng2 = ReproducibleRNG(seed=42)
vals1 = [rng1.uniform() for _ in range(5)]
vals2 = [rng2.uniform() for _ in range(5)]
assert all(abs(v1 - v2) < 1e-12 for v1, v2 in zip(vals1, vals2)), '[TC24] RNG 可复现性 FAILED'

# ---- TC25: detection_efficiency 阈值递增 ----
assert detection_efficiency(0.1) < detection_efficiency(10.0), '[TC25] 探测效率阈值递增 FAILED'

# ---- TC26: modulation_curve 平均值与振幅 ----
t_mod = np.linspace(0.0, 365.25, 100)
s_mod = modulation_curve(t_mod, s0=100.0, sm=5.0)
assert abs(np.mean(s_mod) - 100.0) < 0.1, '[TC26] 调制曲线平均值 FAILED'
assert abs(np.max(s_mod) - 105.0) < 0.1, '[TC26] 调制曲线最大值 FAILED'

# ---- TC27: build_distance_matrix 对称零对角 ----
np.random.seed(42)
X_test = np.random.randn(5, 4)
D_test = build_distance_matrix(X_test)
assert D_test.shape == (5, 5), '[TC27] 距离矩阵形状 FAILED'
assert np.allclose(D_test, D_test.T), '[TC27] 距离矩阵对称性 FAILED'
assert np.all(np.diag(D_test) == 0.0), '[TC27] 距离矩阵对角线 FAILED'

# ---- TC28: fisher_discriminant 分离度为正 ----
np.random.seed(42)
X_sig = np.random.randn(30, 4) + np.array([2.0, 0.0, 0.0, 0.0])
X_bg = np.random.randn(30, 4)
w_fish, thr_fish, sep_fish = fisher_discriminant(X_sig, X_bg)
assert sep_fish > 0.0, '[TC28] Fisher 分离度为正 FAILED'

# ---- TC29: glomin_brent 抛物线最小值 ----
x_min, f_min, calls = glomin_brent(lambda x: (x - 0.3) ** 2, 0.0, 1.0, 0.5, 10.0, 1.0e-10, 1.0e-10)
assert abs(x_min - 0.3) < 0.01, '[TC29] glomin 抛物线最小值 FAILED'
assert f_min < 0.01, '[TC29] glomin 最小值 FAILED'

# ---- TC30: SparseMatrixCSR matvec 验证 ----
coo_test = SparseMatrixCOO(3, 3)
coo_test.add(0, 0, 2.0)
coo_test.add(0, 1, -1.0)
coo_test.add(1, 0, -1.0)
coo_test.add(1, 1, 2.0)
csr_test = coo_test.to_csr()
x_test = np.array([1.0, 2.0, 3.0])
y_test = csr_test.matvec(x_test)
assert abs(y_test[0] - 0.0) < 1e-12, '[TC30] CSR matvec FAILED'
assert abs(y_test[1] - 3.0) < 1e-12, '[TC30] CSR matvec FAILED'

print('\n全部 30 个测试通过!\n')

# main.py 原有函数/类定义，原样保留
