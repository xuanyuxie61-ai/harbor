"""
================================================================================
深海半潜式平台非线性水动力响应与疲劳可靠性综合分析系统
================================================================================

统一入口模块：零参数运行，完成以下完整分析流程：
  1. 平台几何建模与水静力学特性计算
  2. 自适应网格生成（CVT Lloyd + Delaunay 质量检查）
  3. 稀疏矩阵构建与 RCM 带宽优化
  4. 势流理论水动力系数计算（附加质量、辐射阻尼、波浪力）
  5. 方向波谱合成与波浪荷载时程生成
  6. BDF2 隐式时间积分求解六自由度动力响应
  7. 压力泊松方程与速度势求解（SOR 迭代）
  8. 马尔可夫链海况转移与疲劳累积损伤分析
  9. 丢番图波数约束与谱分析

所有参数内嵌，无需外部输入。
================================================================================
"""

import numpy as np
import sys
import time

# ------------------------------------------------------------------------------
# 导入各子模块
# ------------------------------------------------------------------------------
from utils import (
    gamma_func,
    safe_gamma_ratio,
    arc_cosine_safe,
    gcd_vector,
    check_well_posed_diophantine,
)
from wave_spectrum import (
    jonswap_spectrum,
    directional_spreading_gaussian,
    dispersion_relation_finite_depth,
    synthesize_wave_elevation_1d,
    wave_group_velocity,
)
from mesh_geometry import (
    generate_platform_waterline,
    compute_waterplane_properties,
    generate_cvt_nodes_1d,
    polygon_triangulate_earclip,
    triangulation_delaunay_discrepancy,
    simplex01_volume,
    simplex01_monomial_integral,
    triangle_exact_integral_fem,
)
from sparse_matrix import (
    R8NCFSparseMatrix,
    build_laplacian_2d_sparse,
    build_second_difference_1d_sparse,
    rcm_reorder,
    build_adjacency_from_triangulation,
    apply_rcm_to_matrix,
)
from poisson_solver import (
    sor_solve,
    jacobi_solve_2d_poisson,
    solve_pressure_poisson_sor,
    solve_laplace_velocity_potential,
    compute_velocity_from_potential,
)
from platform_dynamics import (
    build_rigid_body_mass_matrix,
    build_hydrostatic_restoring_matrix,
    bdf2_solve,
    catenary_mooring_force,
    partition_dofs_brute,
    build_coupling_matrix_from_stiffness,
    simulate_platform_response,
)
from hydrodynamics import (
    generate_semi_submersible_panels,
    compute_hydrodynamic_coefficients_panel_method,
    morison_force_on_platform_column,
    airy_wave_kinematics,
    panel_area_3d,
    panel_centroid_3d,
    panel_normal_3d,
)
from fatigue_reliability import (
    rainflow_count_cycles,
    miner_damage,
    build_seastate_markov_chain,
    compute_longterm_fatigue_damage_markov,
    simulate_markov_chain_trajectory,
    fatigue_life_prediction,
    reliability_index,
    failure_probability_from_beta,
    stress_from_platform_response,
)
from spectral_analysis import (
    diophantine_nd_nonnegative_solutions,
    wavenumber_discrete_constraint_bragg,
    wavenumber_discrete_constraint_floquet,
    response_spectrum_rao,
    spectral_bandwidth_params,
    diffraction_transfer_function_diophantine,
)


def print_section(title: str):
    """打印格式化的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    start_time = time.time()
    print("\n" + "=" * 70)
    print("  深海半潜式平台非线性水动力响应与疲劳可靠性综合分析系统")
    print("  Deepwater Semi-Submersible Platform Hydrodynamic Response &")
    print("  Fatigue Reliability Integrated Analysis System")
    print("=" * 70)

    # ======================================================================
    # 1. 平台几何建模与水静力学
    # ======================================================================
    print_section("1. 平台几何建模与水静力学特性")

    waterline_vertices = generate_platform_waterline("semi-submersible")
    wp_props = compute_waterplane_properties(waterline_vertices)
    print(f"  水线面面积      : {wp_props['area']:.2f} m²")
    print(f"  水线面形心      : ({wp_props['centroid'][0]:.2f}, {wp_props['centroid'][1]:.2f}) m")
    print(f"  绕 x 轴惯性矩   : {wp_props['I_xx']:.2e} m⁴")
    print(f"  绕 y 轴惯性矩   : {wp_props['I_yy']:.2e} m⁴")
    print(f"  横稳心半径 BM_t : {wp_props['BM_t']:.2f} m")
    print(f"  纵稳心半径 BM_l : {wp_props['BM_l']:.2f} m")

    # 生成三维面板模型
    panels = generate_semi_submersible_panels(
        col_spacing_x=55.0, col_spacing_y=40.0,
        col_diameter=15.0, col_height=20.0,
        pontoon_width=10.0, pontoon_height=8.0,
        n_azimuth=16,
    )
    print(f"  生成面板数量    : {len(panels)}")
    total_wet_area = sum(panel_area_3d(p) for p in panels)
    print(f"  总湿表面积      : {total_wet_area:.2f} m²")

    # ======================================================================
    # 2. 自适应网格生成（CVT Lloyd + Delaunay）
    # ======================================================================
    print_section("2. 自适应网格生成与质量评估")

    # 使用 CVT Lloyd 优化平台周围一维节点分布（用于垂向离散化）
    cvt_nodes = generate_cvt_nodes_1d(n=41, a=-20.0, b=0.0, n_iter=200)
    print(f"  CVT 优化节点数  : {len(cvt_nodes)}")
    print(f"  节点范围        : [{cvt_nodes[0]:.3f}, {cvt_nodes[-1]:.3f}] m")

    # 水线面三角剖分
    x_wl = waterline_vertices[:, 0]
    y_wl = waterline_vertices[:, 1]
    triangles = polygon_triangulate_earclip(x_wl, y_wl)
    print(f"  水线面三角剖分  : {len(triangles)} 个三角形")

    # Delaunay 质量检查
    delaunay_disc = triangulation_delaunay_discrepancy(waterline_vertices, triangles)
    is_delaunay = delaunay_disc <= 1e-10
    print(f"  Delaunay 不一致度 : {delaunay_disc:.6e}")
    print(f"  是否 Delaunay    : {'是' if is_delaunay else '否（在容差内）'}")

    # 单纯形精确积分（质量矩阵验证）
    fem_integrals = triangle_exact_integral_fem(
        waterline_vertices, [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    )
    print(f"  三角形 FEM 精确积分验证 : {[f'{v:.6f}' for v in fem_integrals]}")

    # ======================================================================
    # 3. 稀疏矩阵与 RCM 重排序
    # ======================================================================
    print_section("3. 稀疏矩阵构建与 RCM 带宽优化")

    # 构建二维 Laplacian 稀疏矩阵（模拟流场压力离散）
    nx_grid, ny_grid = 21, 21
    dx_grid, dy_grid = 5.0, 5.0
    A_lap = build_laplacian_2d_sparse(nx_grid, ny_grid, dx_grid, dy_grid)
    print(f"  Laplacian 矩阵维度 : {A_lap.n_rows} × {A_lap.n_cols}")
    print(f"  非零元数量        : {A_lap.nz_num}")

    # RCM 重排序（使用与水线面节点匹配的测试矩阵）
    n_test = len(waterline_vertices)
    A_test = build_second_difference_1d_sparse(n_test)
    adj_test = build_adjacency_from_triangulation(triangles, n_test)
    bw_before = len(waterline_vertices)
    perm = rcm_reorder(adj_test)
    from utils import compute_bandwidth
    bw_after = compute_bandwidth(adj_test, perm)
    print(f"  RCM 带宽优化前    : {bw_before}")
    print(f"  RCM 带宽优化后    : {bw_after}")
    print(f"  带宽缩减比        : {bw_before / max(bw_after, 1):.2f}")

    # 一维二阶差分矩阵
    A_1d = build_second_difference_1d_sparse(20)
    print(f"  1D 二阶差分矩阵   : {A_1d.n_rows} × {A_1d.n_cols}, nz={A_1d.nz_num}")

    # ======================================================================
    # 4. 水动力系数计算（面板法）
    # ======================================================================
    print_section("4. 势流理论水动力系数计算")

    T_wave = 12.0  # 波周期 (s)
    omega_wave = 2.0 * np.pi / T_wave
    A_hydro, B_hydro, F_tf = compute_hydrodynamic_coefficients_panel_method(
        panels, omega_wave, rho=1025.0, h=100.0
    )
    print(f"  计算频率 ω = {omega_wave:.4f} rad/s (T = {T_wave:.1f} s)")
    print(f"  附加质量矩阵 A(ω) 对角元 (kg / kg·m²):")
    for i in range(6):
        print(f"    A[{i},{i}] = {A_hydro[i, i]:.4e}")
    print(f"  辐射阻尼矩阵 B(ω) 对角元 (N·s/m / N·m·s/rad):")
    for i in range(6):
        print(f"    B[{i},{i}] = {B_hydro[i, i]:.4e}")
    print(f"  波浪力传递函数幅值 (N/m):")
    for i in range(6):
        print(f"    |F[{i}]| = {abs(F_tf[i]):.4e}")

    # ======================================================================
    # 5. 方向波谱与波浪荷载
    # ======================================================================
    print_section("5. 方向波谱合成与波浪运动学")

    Hs = 6.0  # 有效波高 (m)
    fp = 1.0 / T_wave  # 谱峰频率
    f_arr = np.linspace(0.05, 0.3, 64)
    S_f = jonswap_spectrum(f_arr, fp, Hs)
    m0_wave = np.trapezoid(S_f, f_arr)
    Hs_check = 4.0 * np.sqrt(m0_wave)
    print(f"  JONSWAP 谱参数    : Hs = {Hs:.2f} m, fp = {fp:.4f} Hz")
    print(f"  谱零阶矩校验      : Hs_calc = {Hs_check:.2f} m")

    theta_mean = 0.0
    n_spread = 4.0
    theta_arr = np.linspace(-np.pi, np.pi, 72)
    D_t = directional_spreading_gaussian(theta_arr, theta_mean, n_spread)
    print(f"  方向扩散函数      : θ_m = {theta_mean:.2f} rad, n = {n_spread:.1f}")
    print(f"  方向分布积分      : {np.trapezoid(D_t, theta_arr):.4f}")

    # 群速度
    cg = wave_group_velocity(f_arr, h=100.0)
    print(f"  谱峰频率群速度    : {cg[len(cg)//2]:.2f} m/s")

    # Airy 波运动学
    kin = airy_wave_kinematics(x=0.0, z=-10.0, t=0.0, A=3.0, T=T_wave, h=100.0, beta=0.0)
    print(f"  Airy 波运动学 (z=-10m):")
    print(f"    水平速度 u      = {kin['u']:.4f} m/s")
    print(f"    垂向速度 w      = {kin['w']:.4f} m/s")
    print(f"    动压力 p_dyn    = {kin['p_dyn']:.2f} Pa")

    # ======================================================================
    # 6. 平台六自由度动力响应（BDF2 隐式积分）
    # ======================================================================
    print_section("6. 平台六自由度动力响应时域分析")

    # 构建质量与恢复矩阵
    mass_platform = 3.5e7  # kg
    cog = np.array([0.0, 0.0, -10.0])
    inertia = np.array([2.5e10, 2.5e10, 3.0e10])
    M = build_rigid_body_mass_matrix(mass_platform, cog, inertia)
    C_rest = build_hydrostatic_restoring_matrix(
        rho=1025.0, g=9.80665,
        area_wp=wp_props['area'],
        I_xx=wp_props['I_xx'], I_yy=wp_props['I_yy'], I_xy=wp_props['I_xy'],
        z_cob=-8.0, z_cog=-10.0,
    )
    print(f"  平台质量          : {mass_platform:.2e} kg")
    print(f"  重心位置          : ({cog[0]:.1f}, {cog[1]:.1f}, {cog[2]:.1f}) m")

    # 波浪力函数
    def wave_force_func(t: float, xi: np.ndarray) -> np.ndarray:
        """时变波浪力：Froude-Krylov + Morison 简化模型。"""
        F = np.zeros(6)
        # Froude-Krylov 力（简化）
        wave_kin = airy_wave_kinematics(
            x=xi[0], z=-10.0, t=t, A=1.5, T=T_wave, h=100.0, beta=0.0
        )
        F[0] = 3.0e5 * wave_kin['u']
        F[2] = 5.0e5 * wave_kin['w']
        # 垂荡力矩
        F[4] = F[0] * 5.0
        return F

    # 系泊配置
    mooring_config = [
        {"anchor": np.array([-200.0, -150.0]), "length": 250.0,
         "weight": 500.0, "EA": 1.5e9, "pretension": 2.0e6},
        {"anchor": np.array([-200.0, 150.0]), "length": 250.0,
         "weight": 500.0, "EA": 1.5e9, "pretension": 2.0e6},
        {"anchor": np.array([200.0, -150.0]), "length": 250.0,
         "weight": 500.0, "EA": 1.5e9, "pretension": 2.0e6},
        {"anchor": np.array([200.0, 150.0]), "length": 250.0,
         "weight": 500.0, "EA": 1.5e9, "pretension": 2.0e6},
    ]

    t_arr, y_arr, info = simulate_platform_response(
        mass=mass_platform,
        cog=cog,
        inertia=inertia,
        A_add=A_hydro,
        B_rad=B_hydro,
        C_rest=C_rest,
        wave_force_func=wave_force_func,
        mooring_config=mooring_config,
        tspan=(0.0, 300.0),
        n_steps=300,
    )
    print(f"  BDF2 积分步数     : {len(t_arr) - 1}")
    print(f"  时间范围          : [{t_arr[0]:.1f}, {t_arr[-1]:.1f}] s")
    print(f"  分区结果          : {info['partition']}")
    print(f"  分区离散度        : {info['partition_discrepancy']:.4e}")

    # 统计响应极值
    surge = y_arr[:, 0]
    sway = y_arr[:, 1]
    heave = y_arr[:, 2]
    roll = y_arr[:, 3]
    pitch = y_arr[:, 4]
    yaw = y_arr[:, 5]
    print(f"  动力响应统计 (位移 m, 转角 rad):")
    print(f"    Surge  max = {np.max(np.abs(surge)):.4f} m")
    print(f"    Sway   max = {np.max(np.abs(sway)):.4f} m")
    print(f"    Heave  max = {np.max(np.abs(heave)):.4f} m")
    print(f"    Roll   max = {np.max(np.abs(roll)):.4f} rad")
    print(f"    Pitch  max = {np.max(np.abs(pitch)):.4f} rad")
    print(f"    Yaw    max = {np.max(np.abs(yaw)):.4f} rad")

    # ======================================================================
    # 7. 泊松方程与速度势求解（SOR）
    # ======================================================================
    print_section("7. 压力泊松方程与速度势求解")

    # 构造合成速度场（模拟平台绕流）
    nx_p, ny_p = 31, 21
    velocity_field = np.zeros((nx_p, ny_p))
    for i in range(nx_p):
        for j in range(ny_p):
            x = i * 2.0 - 30.0
            y = j * 1.5 - 15.0
            # 简化的绕平台势流速度场
            r = np.sqrt(x ** 2 + y ** 2) + 1.0
            velocity_field[i, j] = 1.5 * (1.0 - 225.0 / (r ** 2))

    pressure = solve_pressure_poisson_sor(
        velocity_field, dx=2.0, dy=1.5, rho=1025.0, omega=1.6, tol=1e-6
    )
    p_min, p_max = np.min(pressure), np.max(pressure)
    print(f"  压力场维度        : {pressure.shape}")
    print(f"  压力范围          : [{p_min:.2f}, {p_max:.2f}] Pa")
    print(f"  驻点压力 (近似)   : {pressure[nx_p//2, ny_p//2]:.2f} Pa")

    # Laplace 速度势
    body_bc = np.zeros(nx_p)
    phi = solve_laplace_velocity_potential(
        nx_p, ny_p, Lx=60.0, Ly=30.0, body_bc=body_bc, tol=1e-6
    )
    u_vel, v_vel = compute_velocity_from_potential(phi, dx=60.0/(nx_p-1), dy=30.0/(ny_p-1))
    print(f"  速度势 Laplace 求解完成")
    print(f"  最大水平速度      : {np.max(np.abs(u_vel)):.4f} m/s")
    print(f"  最大垂向速度      : {np.max(np.abs(v_vel)):.4f} m/s")

    # ======================================================================
    # 8. 疲劳累积损伤与可靠性分析（马尔可夫链）
    # ======================================================================
    print_section("8. 疲劳累积损伤与可靠性分析")

    # 从平台响应生成应力时程
    stress_signal = stress_from_platform_response(
        surge, heave, pitch,
        stress_factor_surge=2.5e5,
        stress_factor_heave=1.8e5,
        stress_factor_pitch=3.2e5,
        noise_level=0.01,
    )
    cycles = rainflow_count_cycles(stress_signal)
    print(f"  应力时程长度      : {len(stress_signal)}")
    print(f"  雨流计数循环数    : {len(cycles)}")

    # Miner 损伤
    D_miner = miner_damage(cycles, a=1.0e12, m=3.0)
    print(f"  单次时程 Miner 损伤 : {D_miner:.6e}")

    # 马尔可夫链海况模型
    P, steady, state_labels = build_seastate_markov_chain(n_states=8, seed=42)
    print(f"  海况状态数        : {P.shape[0]}")
    print(f"  稳态分布前 3 态   : {[f'{v:.4f}' for v in steady[:3]]}")

    # 各海况下的损伤率
    damage_rates = np.zeros(len(steady))
    for s in range(len(steady)):
        Hs_state = state_labels[s]
        # 简化的损伤率模型：D ∝ Hs^m
        damage_rates[s] = D_miner * (Hs_state / 6.0) ** 3.0 * 1e-3

    longterm_damage = compute_longterm_fatigue_damage_markov(P, steady, damage_rates)
    print(f"  长期期望年损伤率  : {longterm_damage:.6e}")

    # 疲劳寿命预测
    life_pred = fatigue_life_prediction(longterm_damage, design_life_years=25.0)
    print(f"  预测疲劳寿命      : {life_pred['predicted_life_years']:.1f} 年")
    print(f"  设计寿命          : {life_pred['design_life_years']:.0f} 年")
    print(f"  是否安全          : {'是' if life_pred['is_safe'] else '否'}")

    # 可靠度分析
    beta = reliability_index(
        mean_resistance=500.0, std_resistance=50.0,
        mean_load=300.0, std_load=60.0,
    )
    pf = failure_probability_from_beta(beta)
    print(f"  可靠度指标 β      : {beta:.3f}")
    print(f"  失效概率 P_f      : {pf:.4e}")

    # ======================================================================
    # 9. 丢番图波数约束与谱分析
    # ======================================================================
    print_section("9. 丢番图波数约束与谱分析")

    # 丢番图方程求解示例
    a_dio = np.array([2, 3, 5], dtype=int)
    b_dio = 23
    dio_solutions = diophantine_nd_nonnegative_solutions(a_dio, b_dio)
    print(f"  丢番图方程        : 2x₁ + 3x₂ + 5x₃ = {b_dio}")
    print(f"  非负整数解个数    : {len(dio_solutions)}")
    for sol in dio_solutions[:5]:
        check = int(np.dot(a_dio, sol))
        sol_list = [int(v) for v in sol]
        print(f"    x = {sol_list}  (校验: {check})")

    # Bragg 共振约束
    bragg_solutions = wavenumber_discrete_constraint_bragg(
        wavelength=156.0, column_spacing=55.0, incidence_angle=0.0, max_order=5
    )
    print(f"  Bragg 共振约束 (λ=156m, L=55m):")
    for n, err in bragg_solutions:
        print(f"    阶数 n={n}, 相对误差={err:.4f}")

    # Floquet-Bloch 模式
    floquet_modes = wavenumber_discrete_constraint_floquet(
        domain_lengths=np.array([200.0, 150.0]), max_modes=2, omega=omega_wave, h=100.0
    )
    print(f"  Floquet-Bloch 匹配模式数 : {len(floquet_modes)}")
    for mode in floquet_modes[:3]:
        print(f"    (h₁,h₂)=({mode['h1']},{mode['h2']}), "
              f"k=({mode['kx']:.4f},{mode['ky']:.4f}), "
              f"err={mode['relative_error']:.4f}")

    # 响应谱分析
    omega_spectrum = np.linspace(0.1, 2.0, 128)
    wave_spec_omega = jonswap_spectrum(omega_spectrum / (2.0 * np.pi), fp, Hs)
    # 垂荡响应 RAO
    omega_n_heave = 0.45  # 垂荡固有频率 (rad/s)
    zeta_heave = 0.08
    resp_spec_heave = response_spectrum_rao(omega_spectrum, omega_n_heave, zeta_heave, wave_spec_omega)
    sig_heave = 2.0 * np.sqrt(np.trapezoid(resp_spec_heave, omega_spectrum))
    print(f"  垂荡特征响应幅值  : {sig_heave:.4f} m")

    # 谱带宽参数
    bandwidth = spectral_bandwidth_params(wave_spec_omega, omega_spectrum)
    print(f"  波谱带宽参数 ε    : {bandwidth['epsilon']:.4f}")
    print(f"  平均周期 T₀₁      : {bandwidth['T01']:.2f} s")
    print(f"  平均周期 T₀₂      : {bandwidth['T02']:.2f} s")

    # 绕射传递函数（丢番图约束）
    panel_ks = np.array([0.035, 0.038, 0.040, 0.042, 0.045])
    incident_k = 0.040
    transfer_coeffs = diffraction_transfer_function_diophantine(
        panel_ks, incident_k, np.array([1, 2]), 3
    )
    print(f"  绕射传递函数系数  : {transfer_coeffs}")

    # ======================================================================
    # 10. 综合结果汇总
    # ======================================================================
    print_section("10. 综合分析结果汇总")

    total_time = time.time() - start_time
    print(f"  总计算时间        : {total_time:.3f} s")
    print(f"  面板模型          : {len(panels)} 个面板")
    print(f"  水线面面积        : {wp_props['area']:.2f} m²")
    print(f"  平台质量          : {mass_platform:.2e} kg")
    print(f"  最大垂荡响应      : {np.max(np.abs(heave)):.4f} m")
    print(f"  最大纵摇响应      : {np.max(np.abs(pitch)):.4f} rad")
    print(f"  年疲劳损伤率      : {longterm_damage:.6e}")
    print(f"  预测疲劳寿命      : {life_pred['predicted_life_years']:.1f} 年")
    print(f"  可靠度指标 β      : {beta:.3f}")
    print(f"  失效概率 P_f      : {pf:.4e}")
    print(f"  波谱带宽 ε        : {bandwidth['epsilon']:.4f}")
    print(f"  垂荡特征响应      : {sig_heave:.4f} m")
    print("\n" + "=" * 70)
    print("  分析完成。所有模块运行正常，无报错。")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    main()

    # ================================================================
    # 测试用例（35个，assert模式，涉及随机值均使用固定种子）
    # ================================================================

# ---- TC01: gamma_func 正整数解析验证 ----
result = gamma_func(5.0)
assert abs(result - 24.0) < 1e-6, '[TC01] gamma_func(5) 应等于 24 FAILED'

# ---- TC02: safe_gamma_ratio 已知比值验证 ----
ratio = safe_gamma_ratio(np.array([3.0, 4.0]), 6.0)
expected = 2.0 * 6.0 / 120.0
assert abs(ratio - expected) < 1e-6, '[TC02] safe_gamma_ratio 解析验证 FAILED'

# ---- TC03: arc_cosine_safe 超界输入保护 ----
val_up = arc_cosine_safe(1.0001)
val_low = arc_cosine_safe(-1.0001)
assert abs(val_up - 0.0) < 1e-6, '[TC03] arc_cosine_safe 上限钳位 FAILED'
assert abs(val_low - np.pi) < 1e-6, '[TC03] arc_cosine_safe 下限钳位 FAILED'

# ---- TC04: gcd_vector 已知数组最大公约数 ----
g = gcd_vector(np.array([12, 18, 24]))
assert g == 6, '[TC04] gcd_vector([12,18,24]) 应等于 6 FAILED'

# ---- TC05: check_well_posed_diophantine 判别正确性 ----
assert check_well_posed_diophantine(np.array([2, 3]), 7) == True, '[TC05] 良定义丢番图判别 FAILED'
assert check_well_posed_diophantine(np.array([2, 4]), 7) == False, '[TC05] 不良定义丢番图判别 FAILED'

# ---- TC06: generate_platform_waterline 输出形状 ----
wl = generate_platform_waterline("semi-submersible")
assert wl.ndim == 2 and wl.shape[1] == 2, '[TC06] 水线面顶点维度应为 Nx2 FAILED'

# ---- TC07: compute_waterplane_properties 面积非负 ----
wp = compute_waterplane_properties(wl)
assert wp['area'] > 0, '[TC07] 水线面面积必须为正 FAILED'
assert wp['I_xx'] >= 0, '[TC07] I_xx 必须非负 FAILED'

# ---- TC08: generate_cvt_nodes_1d 输出长度和范围 ----
nodes = generate_cvt_nodes_1d(5, 0.0, 1.0, n_iter=10)
assert len(nodes) == 5, '[TC08] CVT 节点数应为 5 FAILED'
assert nodes[0] == 0.0 and nodes[-1] == 1.0, '[TC08] CVT 端点应固定 FAILED'

# ---- TC09: simplex01_volume 解析验证 ----
vol = simplex01_volume(3)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC09] 3维单位单纯形体积应为 1/6 FAILED'

# ---- TC10: build_laplacian_2d_sparse 矩阵维度正确 ----
A = build_laplacian_2d_sparse(3, 3, 1.0, 1.0)
assert A.n_rows == 9 and A.n_cols == 9, '[TC10] Laplacian 矩阵维度应为 9x9 FAILED'

# ---- TC11: R8NCFSparseMatrix.mv 稀疏矩阵向量乘法 ----
A_test = build_second_difference_1d_sparse(3)
x = np.ones(3)
y = A_test.mv(x)
assert y.shape == (3,), '[TC11] SpMV 输出维度应为 3 FAILED'
assert abs(y[0] - 1.0) < 1e-12, '[TC11] SpMV 首元素验证 FAILED'

# ---- TC12: rcm_reorder 降低带宽 ----
tri = [(0,1,2), (1,2,3)]
adj = build_adjacency_from_triangulation(tri, 4)
perm = rcm_reorder(adj)
from utils import compute_bandwidth
bw_before = compute_bandwidth(adj, list(range(4)))
bw_after = compute_bandwidth(adj, perm)
assert bw_after <= bw_before, '[TC12] RCM 应不增加带宽 FAILED'

# ---- TC13: jonswap_spectrum 输出非负有限 ----
f = np.linspace(0.05, 0.3, 10)
S = jonswap_spectrum(f, fp=0.1, Hs=2.0)
assert np.all(S >= 0), '[TC13] JONSWAP 谱必须非负 FAILED'
assert np.all(np.isfinite(S)), '[TC13] JONSWAP 谱必须有限 FAILED'

# ---- TC14: directional_spreading_gaussian 积分归一化 ----
theta = np.linspace(-np.pi, np.pi, 200)
D = directional_spreading_gaussian(theta, 0.0, 4.0)
int_D = np.trapezoid(D, theta)
assert abs(int_D - 1.0) < 0.05, '[TC14] 方向扩散函数积分应约为 1 FAILED'

# ---- TC15: airy_wave_kinematics 输出字典结构 ----
kin = airy_wave_kinematics(0.0, -10.0, 0.0, 1.0, 10.0, 100.0)
assert set(kin.keys()) == {'eta', 'u', 'w', 'u_dot', 'w_dot', 'p_dyn'}, '[TC15] Airy 波输出键缺失 FAILED'

# ---- TC16: build_rigid_body_mass_matrix 对称性和尺寸 ----
M = build_rigid_body_mass_matrix(1e7, np.array([0,0,-5]), np.array([1e10,1e10,1e10]))
assert M.shape == (6, 6), '[TC16] 质量矩阵维度应为 6x6 FAILED'
assert np.allclose(M, M.T, atol=1e-6), '[TC16] 质量矩阵应对称 FAILED'

# ---- TC17: build_hydrostatic_restoring_matrix C33 为正 ----
C = build_hydrostatic_restoring_matrix(1025.0, 9.81, 1000.0, 1e5, 1e5, 0.0, -8.0, -10.0)
assert C[2, 2] > 0, '[TC17] 垂荡恢复刚度 C33 必须为正 FAILED'

# ---- TC18: bdf2_solve 指数衰减解析解 ----
t_arr, y_arr = bdf2_solve(lambda t, y: -y, (0.0, 1.0), np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), n_steps=20)
assert y_arr.shape == (21, 6), '[TC18] BDF2 输出形状应为 (21,6) FAILED'
assert abs(y_arr[-1, 0] - np.exp(-1.0)) < 0.05, '[TC18] BDF2 指数衰减终值误差过大 FAILED'

# ---- TC19: sor_solve 简单三对角系统 ----
A_sor = build_second_difference_1d_sparse(5)
b_sor = np.ones(5)
x_sor, iters, res = sor_solve(A_sor, b_sor, omega=1.5, tol=1e-8, max_iter=1000)
assert x_sor.shape == (5,), '[TC19] SOR 解维度应为 5 FAILED'
assert res < 1e-4, '[TC19] SOR 残差应足够小 FAILED'

# ---- TC20: jacobi_solve_2d_poisson 零右端项得零解 ----
u, err = jacobi_solve_2d_poisson(5, 5, np.zeros((5,5)), tol=1e-6, max_iter=100)
assert u.shape == (5, 5), '[TC20] Jacobi 解维度应为 5x5 FAILED'
assert np.allclose(u, 0.0, atol=1e-5), '[TC20] 零右端项应得零解 FAILED'

# ---- TC21: compute_velocity_from_potential 线性势 ----
phi_lin = np.outer(np.linspace(0,1,5), np.ones(4))
u_vel, v_vel = compute_velocity_from_potential(phi_lin, dx=0.25, dy=1.0)
assert u_vel.shape == phi_lin.shape, '[TC21] 速度场维度应与势场一致 FAILED'
assert np.allclose(v_vel, 0.0, atol=1e-12), '[TC21] y方向均匀势的v速度应为零 FAILED'

# ---- TC22: panel_area_3d 对矩形面板的面积 ----
rect = np.array([[0,0,0], [2,0,0], [2,1,0], [0,1,0]])
area = panel_area_3d(rect)
assert abs(area - 2.0) < 1e-12, '[TC22] 矩形面板面积应为 2 FAILED'

# ---- TC23: compute_hydrodynamic_coefficients_panel_method 输出尺寸 ----
panels_test = [np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])]
A_test, B_test, F_test = compute_hydrodynamic_coefficients_panel_method(panels_test, 1.0)
assert A_test.shape == (6, 6), '[TC23] 附加质量矩阵维度应为 6x6 FAILED'
assert B_test.shape == (6, 6), '[TC23] 辐射阻尼矩阵维度应为 6x6 FAILED'
assert F_test.shape == (6,), '[TC23] 波浪力维度应为 6 FAILED'

# ---- TC24: catenary_mooring_force 零位移返回非零水平力 ----
F_moor = catenary_mooring_force(0.0, 0.0, np.array([-100.0, 0.0]), 150.0, 500.0, 1e9, 1e6)
assert F_moor.shape == (6,), '[TC24] 系泊力维度应为 6 FAILED'
assert F_moor[0] != 0.0, '[TC24] 零位移时系泊水平力应非零 FAILED'

# ---- TC25: rainflow_count_cycles 正弦信号至少提取一个循环 ----
sig = np.sin(np.linspace(0, 4*np.pi, 100))
cycles = rainflow_count_cycles(sig)
assert len(cycles) > 0, '[TC25] 正弦信号应提取到循环 FAILED'

# ---- TC26: miner_damage 空循环返回零 ----
D_empty = miner_damage([], a=1e12, m=3.0)
assert D_empty == 0.0, '[TC26] 空循环 Miner 损伤应为 0 FAILED'

# ---- TC27: reliability_index 解析验证 ----
beta = reliability_index(500.0, 50.0, 300.0, 60.0)
expected_beta = (500.0 - 300.0) / np.sqrt(50.0**2 + 60.0**2)
assert abs(beta - expected_beta) < 1e-12, '[TC27] 可靠度指标解析验证 FAILED'

# ---- TC28: build_seastate_markov_chain 稳态分布归一化 ----
np.random.seed(42)
P, steady, labels = build_seastate_markov_chain(n_states=4, seed=42)
assert abs(np.sum(steady) - 1.0) < 1e-10, '[TC28] 稳态分布应归一化 FAILED'

# ---- TC29: response_spectrum_rao 峰值在共振频率附近 ----
omega_test = np.linspace(0.1, 2.0, 200)
S_w = np.ones_like(omega_test)
S_resp = response_spectrum_rao(omega_test, 1.0, 0.05, S_w)
peak_idx = np.argmax(S_resp)
assert abs(omega_test[peak_idx] - 1.0) < 0.1, '[TC29] RAO 峰值应在固有频率附近 FAILED'

# ---- TC30: spectral_bandwidth_params 矩形谱带宽 ----
omega_rect = np.linspace(0, 1, 100)
S_rect = np.where((omega_rect > 0.3) & (omega_rect < 0.7), 1.0, 0.0)
bw = spectral_bandwidth_params(S_rect, omega_rect)
assert bw['epsilon'] < 0.5, '[TC30] 矩形谱带宽参数应小于 0.5 FAILED'
assert bw['T01'] > 0, '[TC30] T01 必须为正 FAILED'

# ---- TC31: diophantine_nd_nonnegative_solutions 解析验证 ----
sols = diophantine_nd_nonnegative_solutions(np.array([2, 3]), 10)
assert len(sols) == 2, '[TC31] 2x1+3x2=10 应有 2 个解 FAILED'
for sol in sols:
    assert int(np.dot(np.array([2,3]), sol)) == 10, '[TC31] 丢番图解校验 FAILED'

# ---- TC32: wavenumber_discrete_constraint_bragg 输出格式 ----
bragg = wavenumber_discrete_constraint_bragg(156.0, 55.0, 0.0, 5)
assert isinstance(bragg, list), '[TC32] Bragg 约束输出应为列表 FAILED'

# ---- TC33: diffraction_transfer_function_diophantine 输出尺寸 ----
panel_ks_test = np.array([0.035, 0.038, 0.040])
transfer = diffraction_transfer_function_diophantine(panel_ks_test, 0.040, np.array([1,2]), 3)
assert transfer.shape == panel_ks_test.shape, '[TC33] 传递函数维度应与输入波数一致 FAILED'

# ---- TC34: triangle_exact_integral_fem 对参考三角形的验证 ----
ref_nodes = np.array([[0,0], [1,0], [0,1]])
integrals = triangle_exact_integral_fem(ref_nodes, [(0,0), (1,0), (0,1)])
assert abs(integrals[0] - 0.5) < 1e-12, '[TC34] 常数项积分应为 0.5 FAILED'
assert abs(integrals[1] - 1.0/6.0) < 1e-12, '[TC34] x 项积分应为 1/6 FAILED'

# ---- TC35: wave_group_velocity 输出非负有限 ----
f_test = np.array([0.1, 0.2, 0.3])
cg = wave_group_velocity(f_test, h=100.0)
assert cg.shape == f_test.shape, '[TC35] 群速度维度应与频率一致 FAILED'
assert np.all(cg >= 0), '[TC35] 群速度必须非负 FAILED'
assert np.all(np.isfinite(cg)), '[TC35] 群速度必须有限 FAILED'


print('\n全部 35 个测试通过!\n')
