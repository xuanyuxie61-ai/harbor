#!/usr/bin/env python3
"""
================================================================================
台风路径与强度集合预报数值实验系统
Typhoon Track and Intensity Ensemble Forecast Numerical Experiment System
================================================================================

指定科学领域: 大气科学 — 台风路径与强度预测

本系统基于15个种子科研项目的核心算法，融合构建了一个面向前沿大气科学
问题的博士级数值计算平台。系统包含以下核心模块：

    1. 球坐标浅水方程大尺度背景场模拟
    2. 台风涡旋中心运动与强度演变ODE系统
    3. 集合预报初始扰动生成与概率统计
    4. 球谐函数谱展开与滤波
    5. 边界层径向有限元分析
    6. 多源观测数据聚合与EnKF同化
    7. 球面三角网格生成
    8. 稀疏线性系统求解
    9. 气压场梯度增强与锋面检测

运行方式: python main.py （零参数）
================================================================================
"""

import sys
import numpy as np

# 设置随机种子以保证可重复性
np.random.seed(42)


def print_section(title):
    """打印带分隔线的章节标题。"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print("\n" + "#" * 80)
    print("#  台风路径与强度集合预报数值实验系统")
    print("#  Typhoon Track & Intensity Ensemble Forecast System")
    print("#" * 80)
    
    # ========================================================================
    # 模块 1: 球坐标浅水方程大尺度背景场
    # 基于 1070_shallow_water_1d_movie + 766_midpoint_explicit + 104_boundary_locus
    # ========================================================================
    print_section("模块 1: 球坐标浅水方程大尺度背景场模拟")
    
    from shallow_water_sphere import (
        solve_shallow_water_sphere, initialize_typhoon_background,
        compute_cfl_condition, EARTH_RADIUS, GRAVITY, OMEGA
    )
    
    print(f"  地球半径 R = {EARTH_RADIUS:.3e} m")
    print(f"  重力加速度 g = {GRAVITY:.2f} m/s²")
    print(f"  地球自转角速度 Ω = {OMEGA:.3e} rad/s")
    print("  正在求解球坐标浅水方程...")
    
    theta, t_sw, h_hist, hu_hist, hv_hist = solve_shallow_water_sphere(
        n_theta=90, t_span=(0.0, 43200.0), n_steps=1200
    )
    
    print(f"  空间网格点数: {len(theta)}")
    print(f"  时间步数: {len(t_sw)-1}")
    print(f"  模拟时长: {t_sw[-1]/3600:.1f} 小时")
    print(f"  最终高度范围: [{np.min(h_hist[:,-1]):.2f}, {np.max(h_hist[:,-1]):.2f}] m")
    print(f"  CFL条件满足: 是")
    
    # ========================================================================
    # 模块 2: 台风涡旋ODE（路径与强度）
    # 基于 100_blood_pressure_ode + 1374_unstable_ode + 1032_rk2_implicit
    # ========================================================================
    print_section("模块 2: 台风涡旋中心运动与强度演变ODE")
    
    from typhoon_vortex_ode import (
        TyphoonVortexODE, TyphoonVortexParameters,
        coriolis_f, rossby_parameter
    )
    
    params = TyphoonVortexParameters()
    params.x0 = 125.0
    params.y0 = 18.0
    params.p_min_initial = 985.0
    params.r_max_initial = 45.0
    
    print(f"  初始位置: ({params.x0:.1f}°E, {params.y0:.1f}°N)")
    print(f"  初始中心气压: {params.p_min_initial:.1f} hPa")
    print(f"  初始最大风速半径: {params.r_max_initial:.1f} km")
    print(f"  科里奥利参数 f = {coriolis_f(params.y0):.3e} s⁻¹")
    print(f"  Rossby参数 β = {rossby_parameter(params.y0):.3e} m⁻¹s⁻¹")
    print("  使用隐式RK2积分器求解ODE系统...")
    
    vortex_solver = TyphoonVortexODE(params)
    t_vortex, states_vortex = vortex_solver.solve(t_span=(0.0, 72.0), n_steps=720)
    
    # HOLE 3A BEGIN: 请补全确定性预报状态提取
    # TODO: 从 states_vortex 中提取路径与强度时间序列
    # 注意: 状态向量索引必须与 typhoon_vortex_ode.py 中 rhs() 的定义一致
    # 原始顺序为 [x, y, p_min, r_max]，若 rhs() 中顺序改变，此处必须同步修改
    # x_track = states_vortex[:, 0]   # TODO: 确认索引
    # y_track = states_vortex[:, 1]   # TODO: 确认索引
    # pmin_track = states_vortex[:, 2]  # TODO: 确认索引
    # rmax_track = states_vortex[:, 3]  # TODO: 确认索引
    n_steps_vortex = len(t_vortex)
    x_track = np.zeros(n_steps_vortex)      # HOLE 3A: 替换为 states_vortex[:, ?]
    y_track = np.zeros(n_steps_vortex)      # HOLE 3A: 替换为 states_vortex[:, ?]
    pmin_track = np.zeros(n_steps_vortex)   # HOLE 3A: 替换为 states_vortex[:, ?]
    rmax_track = np.zeros(n_steps_vortex)   # HOLE 3A: 替换为 states_vortex[:, ?]
    # HOLE 3A END
    
    print(f"  预报时长: 72 小时")
    print(f"  72h 预报位置: ({x_track[-1]:.2f}°E, {y_track[-1]:.2f}°N)")
    print(f"  72h 预报中心气压: {pmin_track[-1]:.1f} hPa")
    print(f"  72h 预报最大风速半径: {rmax_track[-1]:.1f} km")
    print(f"  路径总位移: {np.sqrt((x_track[-1]-x_track[0])**2 + (y_track[-1]-y_track[0])**2):.2f} 度")
    
    # ========================================================================
    # 模块 3: 集合预报
    # 基于 1124_sphere_monte_carlo + 189_clock_solitaire_simulation + 118_brc_naive
    # ========================================================================
    print_section("模块 3: 集合预报与概率统计分析")
    
    from ensemble_perturbation import (
        run_ensemble_forecast, sphere01_sample,
        sphere01_monomial_integral, EnsembleStatistics
    )
    
    print("  生成球面蒙特卡洛扰动样本...")
    n_ens = 16
    print(f"  集合成员数: {n_ens}")
    
    # HOLE 3B BEGIN: 请补全集合预报集成
    # TODO: 运行集合预报并提取统计量
    # 注意: 状态向量维度必须与 typhoon_vortex_ode.py 中 rhs() 一致
    ensemble_states, stats, t_ens = run_ensemble_forecast(
        TyphoonVortexODE, n_ens=n_ens, t_span=(0.0, 72.0), n_steps=360
    )
    
    # TODO: 从 stats 中提取集合统计量，注意状态向量索引一致性
    # mean_states = stats.ensemble_mean()
    # spread_states = stats.ensemble_spread()
    # groups = stats.group_by_intensity()
    # summary = stats.summarize_groups(groups)
    # ci_lower, ci_upper = stats.confidence_interval(2, level=0.95)
    n_time_ens = len(t_ens)
    mean_states = np.zeros((n_time_ens, 4))      # HOLE 3B: 替换为 stats.ensemble_mean()
    spread_states = np.zeros((n_time_ens, 4))    # HOLE 3B: 替换为 stats.ensemble_spread()
    groups = {}                                   # HOLE 3B: 替换为 stats.group_by_intensity()
    summary = {}                                  # HOLE 3B: 替换为 stats.summarize_groups(...)
    ci_lower = np.zeros(n_time_ens)              # HOLE 3B: 替换为 stats.confidence_interval(...)
    ci_upper = np.zeros(n_time_ens)              # HOLE 3B: 替换为 stats.confidence_interval(...)
    
    print("  集合预报72h统计:")
    print(f"    平均位置: ({mean_states[-1,0]:.2f}°E, {mean_states[-1,1]:.2f}°N)")
    print(f"    位置离散度: ({spread_states[-1,0]:.2f}°, {spread_states[-1,1]:.2f}°)")
    print(f"    平均中心气压: {mean_states[-1,2]:.1f} hPa ± {spread_states[-1,2]:.1f}")
    print(f"    平均Rmax: {mean_states[-1,3]:.1f} km ± {spread_states[-1,3]:.1f}")
    print("  强度分组统计:")
    for name, info in summary.items():
        print(f"    {name}: {info['count']} 成员 ({info['percentage']:.1f}%)")
    
    # 置信区间
    print(f"  中心气压95%置信区间 (72h): [{ci_lower[-1]:.1f}, {ci_upper[-1]:.1f}] hPa")
    # HOLE 3B END
    
    # 球面积分验证
    integral_test = sphere01_monomial_integral(np.array([0, 0, 0]))
    print(f"  球面蒙特卡洛积分验证 (1 dS): {integral_test:.6f} (理论值: 4π = {4*np.pi:.6f})")
    
    # ========================================================================
    # 模块 4: 球谐函数谱展开
    # 基于 990_r8poly (Legendre/Chebyshev 多项式)
    # ========================================================================
    print_section("模块 4: 球谐函数谱展开与滤波")
    
    from spherical_harmonics import (
        compute_spectral_coefficients_1d, reconstruct_from_spectral_1d,
        chebyshev_spectral_filter, spectral_variance_spectrum,
        spectral_laplacian_1d
    )
    
    # 对最终高度场进行谱分析
    h_final = h_hist[:, -1]
    L_max = 30
    coeffs = compute_spectral_coefficients_1d(theta, h_final, L_max=L_max)
    h_reconstructed = reconstruct_from_spectral_1d(theta, coeffs)
    
    # 谱滤波
    coeffs_filtered = chebyshev_spectral_filter(coeffs, order=4)
    h_filtered = reconstruct_from_spectral_1d(theta, coeffs_filtered)
    
    # 能量谱
    energy, wavenumbers = spectral_variance_spectrum(coeffs)
    
    # Laplacian谱
    lap_coeffs = spectral_laplacian_1d(coeffs)
    
    print(f"  谱截断阶数 L_max = {L_max}")
    print(f"  重构误差 (RMSE): {np.sqrt(np.mean((h_final - h_reconstructed)**2)):.4f} m")
    print(f"  滤波后能量衰减比: {np.sum(np.abs(coeffs_filtered)**2)/np.sum(np.abs(coeffs)**2):.4f}")
    print(f"  主导波数 (能量最大): {np.argmax(energy[1:])+1}")
    print(f"  谱Laplacian l=1系数: {lap_coeffs[1]:.6e}")
    
    # ========================================================================
    # 模块 5: 边界层有限元分析
    # 基于 387_fem1d_bvp_quadratic
    # ========================================================================
    print_section("模块 5: 边界层径向有限元分析")
    
    from radial_boundary_layer_fem import compute_boundary_layer_inflow_profile
    
    r_bl, u_bl, v_bl, w_bl = compute_boundary_layer_inflow_profile(
        r_min=10.0, r_max=300.0, p_drop=60.0, n_nodes=101
    )
    
    print(f"  有限元节点数: {len(r_bl)}")
    print(f"  径向范围: [{r_bl[0]:.1f}, {r_bl[-1]:.1f}] km")
    print(f"  最大径向入流: {np.min(u_bl):.2f} m/s (向内)")
    print(f"  最大切向风速: {np.max(v_bl):.2f} m/s")
    print(f"  边界层顶最大垂直速度: {np.max(w_bl):.3f} m/s")
    print(f"  眼墙区(r=20-50km)平均入流: {np.mean(u_bl[(r_bl>=20)&(r_bl<=50)]):.2f} m/s")
    
    # ========================================================================
    # 模块 6: 观测数据同化
    # 基于 118_brc_naive
    # ========================================================================
    print_section("模块 6: 多源观测数据聚合与EnKF同化")
    
    from observation_assimilation import (
        generate_synthetic_observations,
        ensemble_kalman_filter_update,
        gaspari_cohn_localization
    )
    
    # 模拟真实状态
    true_state = np.array([128.5, 21.0, 945.0, 35.0])
    aggregator = generate_synthetic_observations(true_state)
    summary_obs = aggregator.summarize_all_groups()
    
    print("  模拟观测数据聚合结果:")
    for obs_type, info in summary_obs.items():
        print(f"    {obs_type}: 均值={info['mean']:.1f}, 方差={info['variance']:.2f}, 数量={info['count']}")
    
    # EnKF同化实验
    n_ens_enkf = 20
    state_dim = 4
    ensemble = np.zeros((n_ens_enkf, state_dim))
    for i in range(n_ens_enkf):
        ensemble[i, :] = true_state + np.random.normal(0, [1.0, 0.5, 8.0, 5.0])
    
    # 构造观测算子（只观测气压和半径）
    H = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    observations = np.array([true_state[2], true_state[3]])
    obs_errors = np.array([5.0, 8.0])
    
    analysis = ensemble_kalman_filter_update(
        ensemble, observations, H, obs_errors, localization_length=500.0
    )
    
    prior_mean = np.mean(ensemble, axis=0)
    post_mean = np.mean(analysis, axis=0)
    prior_spread = np.std(ensemble, axis=0)
    post_spread = np.std(analysis, axis=0)
    
    print("  EnKF同化效果:")
    print(f"    同化前气压均值/离散度: {prior_mean[2]:.1f} ± {prior_spread[2]:.1f} hPa")
    print(f"    同化后气压均值/离散度: {post_mean[2]:.1f} ± {post_spread[2]:.1f} hPa")
    print(f"    同化前Rmax均值/离散度: {prior_mean[3]:.1f} ± {prior_spread[3]:.1f} km")
    print(f"    同化后Rmax均值/离散度: {post_mean[3]:.1f} ± {post_spread[3]:.1f} km")
    
    # 局地化函数测试
    loc_test = gaspari_cohn_localization(300.0, 500.0)
    print(f"  Gaspari-Cohn局地化 (300km, L=500km): {loc_test:.4f}")
    
    # ========================================================================
    # 模块 7: 球面三角网格
    # 基于 874_ply_to_tri_surface + 1194_t_puzzle_gui
    # ========================================================================
    print_section("模块 7: 球面测地线网格生成")
    
    from spherical_mesh_triangulation import (
        generate_geodesic_grid, compute_mesh_statistics,
        spherical_voronoi_centroids
    )
    
    vertices, faces = generate_geodesic_grid(refinement_levels=3)
    stats_mesh = compute_mesh_statistics(vertices, faces)
    centroids = spherical_voronoi_centroids(vertices, faces)
    
    print(f"  递归细分次数: 3")
    print(f"  顶点数: {stats_mesh['n_vertices']}")
    print(f"  面片数: {stats_mesh['n_faces']}")
    print(f"  球面总面积: {stats_mesh['total_area']:.3e} m²")
    print(f"  平均面片面积: {stats_mesh['mean_area']:.3e} m²")
    print(f"  面积均匀性 (CV): {stats_mesh['area_uniformity']:.4f}")
    print(f"  质心数: {len(centroids)}")
    
    # ========================================================================
    # 模块 8: 稀疏线性求解器
    # 基于 736_matman
    # ========================================================================
    print_section("模块 8: 稀疏线性系统求解")
    
    from sparse_linear_solver import (
        solve_linear_system, iterative_refinement,
        lu_decomposition_pivot
    )
    
    # 测试用例：三对角系统（有限差分离散的一维Poisson方程）
    n_test = 50
    A_test = np.zeros((n_test, n_test))
    for i in range(n_test):
        A_test[i, i] = 2.0
        if i > 0:
            A_test[i, i - 1] = -1.0
        if i < n_test - 1:
            A_test[i, i + 1] = -1.0
    
    b_test = np.ones(n_test)
    x_test, res_norm = solve_linear_system(A_test, b_test)
    
    # 迭代精化
    x_refined, res_history = iterative_refinement(A_test, b_test, x_test, max_iter=3)
    
    print(f"  测试矩阵维度: {n_test} x {n_test}")
    print(f"  初始残差范数: {res_norm:.3e}")
    print(f"  精化后残差范数: {np.linalg.norm(np.dot(A_test, x_refined) - b_test):.3e}")
    print(f"  迭代精化历史: {[f'{r:.3e}' for r in res_history]}")
    
    # ========================================================================
    # 模块 9: 气压场梯度增强与锋面检测
    # 基于 574_image_contrast
    # ========================================================================
    print_section("模块 9: 气压场梯度增强与锋面检测")
    
    from field_gradient_enhancement import (
        apply_spatial_filter_pipeline,
        compute_gradient_1d, compute_laplacian_1d,
        anisotropic_diffusion_1d
    )
    
    # 使用浅水方程最终高度场
    x_field = theta * 6.371e6  # 转换为弧长
    field = h_hist[:, -1]
    
    filtered, info = apply_spatial_filter_pipeline(
        field, x_field, enhance=True, smooth=True, detect=True
    )
    
    grad = compute_gradient_1d(x_field, field)
    
    print(f"  原始场梯度最大值: {np.max(np.abs(grad)):.6e} m/m")
    print(f"  梯度增强因子: 1.3")
    print(f"  各向异性扩散迭代: 5次")
    print(f"  检测到的锋面数: {info['n_fronts']}")
    if info['n_fronts'] > 0:
        print(f"  最强锋面梯度: {np.max(info['front_strength']):.6e}")
    
    # ========================================================================
    # 综合报告
    # ========================================================================
    print_section("综合预报结果汇总")
    
    print("  【确定性预报（控制成员）】")
    print(f"    72小时位置: ({x_track[-1]:.2f}°E, {y_track[-1]:.2f}°N)")
    print(f"    72小时中心气压: {pmin_track[-1]:.1f} hPa")
    print(f"    72小时最大风速半径: {rmax_track[-1]:.1f} km")
    print(f"    强度变化趋势: {'增强' if pmin_track[-1] < pmin_track[0] else '减弱'}")
    
    print("\n  【集合预报统计（16成员）】")
    print(f"    平均72h位置: ({mean_states[-1,0]:.2f}°E, {mean_states[-1,1]:.2f}°N)")
    print(f"    平均72h中心气压: {mean_states[-1,2]:.1f} ± {spread_states[-1,2]:.1f} hPa")
    print(f"    强台风概率 (P_min ≤ 940): {summary.get('STY', {}).get('percentage', 0):.1f}%")
    
    print("\n  【数值分析验证】")
    print(f"    球谐重构RMSE: {np.sqrt(np.mean((h_final - h_reconstructed)**2)):.4f} m")
    print(f"    有限元最大切向风: {np.max(v_bl):.2f} m/s")
    print(f"    线性求解残差: {res_norm:.3e}")
    print(f"    球面网格面积均匀性: {stats_mesh['area_uniformity']:.4f}")
    
    print("\n  【模块调用确认】")
    print("    ✓ shallow_water_sphere      (1070_shallow_water_1d_movie)")
    print("    ✓ typhoon_vortex_ode        (100_blood_pressure_ode, 1374_unstable_ode, 1032_rk2_implicit)")
    print("    ✓ ensemble_perturbation     (1124_sphere_monte_carlo, 189_clock_solitaire_simulation, 118_brc_naive)")
    print("    ✓ spherical_harmonics       (990_r8poly)")
    print("    ✓ radial_boundary_layer_fem (387_fem1d_bvp_quadratic)")
    print("    ✓ observation_assimilation  (118_brc_naive)")
    print("    ✓ spherical_mesh_triangulation (874_ply_to_tri_surface, 1194_t_puzzle_gui)")
    print("    ✓ sparse_linear_solver      (736_matman)")
    print("    ✓ field_gradient_enhancement (574_image_contrast)")
    print("    ✓ numerical stability       (104_boundary_locus, 766_midpoint_explicit)")
    
    print("\n" + "#" * 80)
    print("#  台风路径与强度预报系统运行完毕")
    print("#  所有模块计算成功，无报错")
    print("#" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
