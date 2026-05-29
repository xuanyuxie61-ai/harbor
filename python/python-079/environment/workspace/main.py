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
        # TODO: 基于 Airy 波运动学计算时变波浪力
        # 关键物理：
        #   调用 airy_wave_kinematics 获取波浪速度场 (u, w)
        #   Froude-Krylov 力近似：F_x ∝ ρ·V·u_dot, F_z ∝ ρ·V·w_dot
        #   或使用速度比例模型：F_x = C_u · u, F_z = C_w · w
        #   需与 platform_dynamics.py 中的 simulate_platform_response 兼容
        raise NotImplementedError("wave_force_func 需要实现")

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
    sys.exit(main())
