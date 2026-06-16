"""
main.py - 阿尔芬波-高能粒子相互作用模拟统一入口

本项目实现计算等离子体物理中的阿尔芬波与高能粒子相互作用，
使用高阶有限差分方法和稳定性分析。

========================================================
物理问题描述:
========================================================

在磁约束聚变等离子体 (如 ITER) 中，阿尔芬波与高能粒子
(来自中性束注入 NBI 或聚变反应产生的 α 粒子) 的相互作用
是决定聚变堆性能的关键物理过程之一。

高能粒子可以通过波-粒子共振将能量传递给阿尔芬波，
导致阿尔芬本征模 (TAE, EAE 等) 的不稳定性增长。
不稳定的阿尔芬波会散射高能粒子，降低约束性能。

控制方程:
  1. 简化 MHD (阿尔芬波):
     ∂ψ/∂t = -∇_∥ φ + η ∇²ψ
     ∂U/∂t = v_A² ∇_∥ J - [φ, U] + ν ∇²U

  2. 漂移动力学方程 (高能粒子):
     ∂f/∂t + v_∥ b·∇f + v_d·∇f = C[f] + S

  3. 耦合条件:
     高能粒子通过扰动电流和压强影响阿尔芬波:
     δJ_EP = ∫ e v_∥ δf d³v
     δp_EP = ∫ m(v_∥² + v_⊥²/2) δf d³v

========================================================
数值方法:
========================================================
  - 空间离散: 2~8 阶中心差分 (可配置)
  - 时间推进: 四阶 Runge-Kutta (RK4)
  - 稳定性分析: 矩阵本征值分解 + 能量原理 (δW)
  - 色散关系: Chebyshev 伴随矩阵法 + 等离子体色散函数

========================================================
模拟流程:
========================================================
  Step 1: 设置等离子体参数和磁平衡几何
  Step 2: 生成自适应网格 (CVT 度量张量方法)
  Step 3: 初始化阿尔芬本征模 (TAE)
  Step 4: 初始化高能粒子分布
  Step 5: 时间推进模拟 (RK4)
  Step 6: 本征值稳定性分析
  Step 7: 色散关系和间隙结构计算
  Step 8: 守恒量监测和统计分析
  Step 9: 收敛阶验证
  Step 10: 保存结果

========================================================
运行方式:
  python main.py
  (无需任何参数)
========================================================

作者: DA 博士级合成项目 PROJECT_290
领域: 计算等离子体物理
"""

import sys
import os
import time
import numpy as np

# 将当前目录加入路径以支持包导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取父目录名用于包导入
_parent_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def separator(title, char="=", width=70):
    """打印分隔线和标题。"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def main():
    """
    主模拟流程。
    """
    start_time = time.time()
    print("=" * 70)
    print("  阿尔芬波-高能粒子相互作用: 高阶有限差分与稳定性分析")
    print("  PROJECT_290 - 计算等离子体物理博士级合成项目")
    print("=" * 70)

    # ================================================================
    # Step 1: 设置等离子体参数
    # ================================================================
    separator("Step 1: 等离子体参数配置")

    from plasma_config import PlasmaParameters

    params = PlasmaParameters(
        B0=5.0,               # 磁场 [T]
        R0=1.65,              # 大半径 [m]
        a_minor=0.5,          # 小半径 [m]
        q0=1.0,               # 轴上安全因子
        qa=3.0,               # 边界安全因子
        n_e=5.0e19,           # 电子密度 [m^-3]
        T_e=3.0e3,            # 电子温度 [eV]
        T_i=2.5e3,            # 离子温度 [eV]
        n_fast=1.0e17,        # 高能粒子密度 [m^-3]
        E_fast=100.0e3,       # 高能粒子能量 [eV]
        n_r=32,               # 径向网格点
        n_theta=16,           # 极向网格点
        n_phi=8,              # 环向网格点
        fd_order=4,           # 4阶有限差分
        cfl_number=0.3,       # CFL 数
        n_timesteps=80,       # 时间步数 (小规模演示)
        output_interval=10,   # 输出间隔
    )

    print(params.summary())

    # ================================================================
    # Step 2: 磁平衡几何与网格生成
    # ================================================================
    separator("Step 2: 磁平衡几何与自适应网格")

    from magnetic_geometry import MagneticGeometry, CVTAdaptiveGrid
    from magnetic_geometry import compute_triangulation_neighbors

    geom = MagneticGeometry(
        R0=params.R0,
        a_minor=params.a_minor,
        elongation=1.4,         # ITER 典型的拉长比
        triangularity=0.33,     # ITER 典型的三角变形
        n_flux_surfaces=16,
        n_theta_geo=32,
    )

    # 边界多边形
    nodes, edges = geom.get_boundary_poly()
    print(f"  等离子体边界: {len(nodes)} 个节点, {len(edges)} 条边")
    print(f"  边界 R 范围: [{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}] m")
    print(f"  边界 Z 范围: [{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}] m")

    # 保存 POLY 文件
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    geom.write_poly_file(os.path.join(output_dir, 'plasma_boundary.poly'))
    geom.write_gmsh_mesh(os.path.join(output_dir, 'flux_surfaces.msh'))
    print(f"  边界 POLY 文件已保存")
    print(f"  Gmsh 网格文件已保存")

    # CVT 自适应网格 (小规模演示)
    cvt = CVTAdaptiveGrid(
        n_generators=8,
        n_samples=64,
        geometry=geom,
        alpha_refine=3.0,
    )
    generators = cvt.generate(seed=42)
    print(f"  CVT 自适应网格: {len(generators)} 个生成点")
    print(f"  生成点 ρ 范围: [{generators[:, 0].min():.3f}, {generators[:, 0].max():.3f}]")

    # 三角化邻接关系 (小规模)
    elements, neighbors = compute_triangulation_neighbors(4, 8)
    print(f"  三角化: {len(elements)} 个三角形, {np.sum(neighbors >= 0) // 2} 条内边")

    # ================================================================
    # Step 3: 高能粒子分布初始化
    # ================================================================
    separator("Step 3: 高能粒子动力学")

    from energetic_particle_kinetics import EnergeticParticleDistribution, MultiSpeciesEPDriver

    # 多种高能粒子群体
    ep_driver = MultiSpeciesEPDriver(
        n_species=3,
        v_parallel_bins=12,
        v_perp_bins=6,
    )

    print("  初始高能粒子群体:")
    print(ep_driver.summary())

    # 守恒量
    initial_conserved = ep_driver.total_conserved()
    print(f"\n  初始总守恒量:")
    for key, val in initial_conserved.items():
        print(f"    {key}: {val:.6e}")

    # 碰撞演化
    dt_ep = 1e-5
    ep_driver.evolve_collisions(dt_ep)

    post_collision_conserved = ep_driver.total_conserved()
    print(f"\n  碰撞后守恒量:")
    for key in initial_conserved:
        before = initial_conserved[key]
        after = post_collision_conserved[key]
        rel_change = abs(after - before) / (abs(before) + 1e-30)
        print(f"    {key}: 相对变化 = {rel_change:.6e}")

    # ================================================================
    # Step 4: 边界条件与有限差分算子验证
    # ================================================================
    separator("Step 4: 边界处理与高阶算子验证")

    from boundary_handler import (
        cyclic_wrap, periodic_index, helical_boundary_map,
        field_line_tracing, rational_surface_check, boundary_layer_profile,
    )
    from high_order_operators import (
        fd_coefficients_first_derivative,
        fd_coefficients_second_derivative,
        BandMatrixSPD,
        apply_first_derivative,
        apply_second_derivative,
    )

    # 周期性边界测试
    test_indices = [-3, -1, 0, 15, 16, 20, 31, 32, 35]
    wrapped = [periodic_index(i, params.n_r) for i in test_indices]
    print(f"  周期性边界测试 (N={params.n_r}):")
    print(f"    输入: {test_indices}")
    print(f"    输出: {wrapped}")

    # 磁力线追踪
    theta_arr, phi_arr, _, _ = field_line_tracing(
        q_profile=params.q_profile,
        r_norm=0.5,
        theta0=0.0,
        n_steps=50,
        n_theta=params.n_theta,
        n_phi=params.n_phi,
    )
    print(f"\n  磁力线追踪 (r/a=0.5, q={params.q_profile(0.5):.3f}):")
    print(f"    50 步后 θ = {theta_arr[-1]:.4f} rad, φ = {phi_arr[-1]:.4f} rad")

    # 有理面检查
    for m, n in [(1, 1), (2, 1), (3, 1), (5, 2)]:
        is_rat, q_rat, shear = rational_surface_check(
            params.q_profile(0.5), m, n
        )
        print(f"    (m,n)=({m},{n}): q_rat={q_rat:.3f}, 共振={'是' if is_rat else '否'}")

    # 边界层剖面
    profile, x_bl = boundary_layer_profile(10, 0.1)
    print(f"\n  边界层: {len(profile)} 点, 衰减因子 [{profile.min():.6f}, {profile.max():.6f}]")

    # 高阶差分系数
    for order in [2, 4, 6, 8]:
        c1 = fd_coefficients_first_derivative(order)
        d0, d_coeff = fd_coefficients_second_derivative(order)
        print(f"\n  {order}阶差分:")
        print(f"    一阶系数: {c1}")
        print(f"    二阶中心系数: {d0:.6f}, 非中心: {d_coeff}")

    # 带状矩阵测试
    N_test = 16
    A_dif2 = BandMatrixSPD.create_dif2(N_test, ml=1)
    A_ho = BandMatrixSPD.create_high_order_laplacian(N_test, order=4, ml=2)

    x_test = np.sin(np.linspace(0, np.pi, N_test))
    b_dif2 = A_dif2.mv(x_test)
    b_ho = A_ho.mv(x_test)
    print(f"\n  带状矩阵测试 (N={N_test}):")
    print(f"    DIF2 ||Ax|| = {np.linalg.norm(b_dif2):.6e}")
    print(f"    HO-L  ||Ax|| = {np.linalg.norm(b_ho):.6e}")

    # 验证矩阵向量乘与全密度矩阵一致性
    A_dense = A_dif2.to_dense()
    b_dense = A_dense @ x_test
    consistency_error = np.linalg.norm(b_dif2 - b_dense)
    print(f"    带状/全密度一致性误差: {consistency_error:.2e}")

    # 高阶差分的收敛阶验证
    print("\n  收敛阶验证 (对 sin(2πx) 求导):")
    from high_order_operators import truncation_error_analysis

    for order in [2, 4, 6]:
        errors = []
        dx_values = []
        for N in [16, 32, 64, 128]:
            x = np.linspace(0.1, 0.9, N)
            dx = x[1] - x[0]
            func = lambda xx: np.sin(2 * np.pi * xx)
            dfunc = lambda xx: 2 * np.pi * np.cos(2 * np.pi * xx)
            e_l2, _ = truncation_error_analysis(dx, order, func, dfunc, x)
            errors.append(e_l2)
            dx_values.append(dx)

        from statistical_diagnostics import convergence_order_analysis
        conv_order, r2 = convergence_order_analysis(dx_values, errors)
        print(f"    {order}阶: 估计收敛阶 = {conv_order:.2f} (R² = {r2:.6f})")

    # ================================================================
    # Step 5: 阿尔芬波时间推进模拟
    # ================================================================
    separator("Step 5: 阿尔芬波 RK4 时间推进")

    from alfven_wave_solver import AlfvenWaveState, AlfvenWaveSolver

    solver = AlfvenWaveSolver(params, geom)

    # 初始化 TAE 模 (m=2, n=1)
    initial_state = AlfvenWaveState(params.n_r, params.n_theta)
    solver.initialize_alfven_eigenmode(
        initial_state,
        m_poloidal=2,
        n_toroidal=1,
        amplitude=1e-4,
    )

    e_init = initial_state.energy()
    print(f"  初始 TAE 模 (m=2, n=1):")
    print(f"    最大 |ψ| = {np.max(np.abs(initial_state.psi)):.6e}")
    print(f"    最大 |φ| = {np.max(np.abs(initial_state.phi)):.6e}")
    print(f"    初始能量 E = {e_init:.6e}")

    # 运行模拟
    print(f"\n  开始 RK4 时间推进 ({params.n_timesteps} 步)...")
    sim_start = time.time()
    final_state, history = solver.run_simulation(
        initial_state,
        n_steps=params.n_timesteps,
        output_interval=params.output_interval,
    )
    sim_time = time.time() - sim_start

    e_final = final_state.energy()
    print(f"  模拟完成!")
    print(f"    计算时间: {sim_time:.3f} s")
    print(f"    最终能量 E = {e_final:.6e}")
    if e_init > 0:
        print(f"    能量变化比 E/E₀ = {e_final/e_init:.6e}")
    print(f"    最大 |ψ| = {np.max(np.abs(final_state.psi)):.6e}")
    print(f"    最大 |φ| = {np.max(np.abs(final_state.phi)):.6e}")

    # ================================================================
    # Step 6: 守恒量监测
    # ================================================================
    separator("Step 6: 守恒量监测")

    from conservation_monitor import ConservationMonitor, ConstrainedDistributionReconstructor

    monitor = ConservationMonitor()
    monitor.register_quantity('energy', e_init, time=0.0)
    monitor.register_quantity('magnetic_flux', np.sum(initial_state.psi), time=0.0)
    monitor.register_quantity('canonical_momentum', np.sum(initial_state.phi), time=0.0)

    for i, (t, e) in enumerate(zip(history['time'], history['energy'])):
        monitor.update_quantity('energy', e, time=t)

    monitor.update_quantity('magnetic_flux', np.sum(final_state.psi))
    monitor.update_quantity('canonical_momentum', np.sum(final_state.phi))

    print(monitor.summary())

    # 约束分布重构
    print("\n  约束分布重构测试:")
    ep_single = EnergeticParticleDistribution(n_v_parallel=10, n_v_perp=5)
    ep_single.initialize_maxwellian_fast(n_fast=1e17, T_fast=50e3)
    q_before = ep_single.conserved_quantities()

    reconstructor = ConstrainedDistributionReconstructor(
        ep_single.n_vp, ep_single.n_vt, ep_single.weights
    )
    f_new, converged, residual = reconstructor.reconstruct(
        ep_single.f * 1.1,  # 故意偏离
        q_before['density'],
        q_before['energy'],
        ep_single.mass,
    )
    print(f"    重构收敛: {converged}, 残差 = {residual:.2e}")

    # ================================================================
    # Step 7: 本征值稳定性分析
    # ================================================================
    separator("Step 7: 本征值稳定性分析")

    from stability_eigenvalue import StabilityAnalyzer

    analyzer = StabilityAnalyzer(params, geom)

    print(f"  构建线性算子矩阵 (N = {2 * params.n_r * params.n_theta})...")
    build_start = time.time()
    A_linear = analyzer.build_linear_operator(fd_order=params.fd_order)
    build_time = time.time() - build_start
    print(f"    构建时间: {build_time:.3f} s")
    print(f"    矩阵大小: {A_linear.shape}")
    print(f"    矩阵非零率: {np.count_nonzero(A_linear) / A_linear.size:.4f}")

    # 本征值分解 (对小矩阵可行)
    print(f"  本征值分解...")
    eig_start = time.time()
    eigen_result = analyzer.eigenvalue_analysis(A_linear, n_modes=params.n_eigenmodes)
    eig_time = time.time() - eig_start
    print(f"    分解时间: {eig_time:.3f} s")
    print(f"    最大增长率 γ_max = {eigen_result['max_growth_rate']:.6e}")
    print(f"    不稳定模数: {eigen_result['n_unstable']}")
    print(f"    系统稳定性: {'稳定' if eigen_result['is_stable'] else '不稳定'}")

    # 显示前几个本征值
    print(f"\n  前 {min(8, len(eigen_result['eigenvalues']))} 个本征值:")
    print(f"    {'序号':>4} {'增长率 γ':>14} {'频率 f [Hz]':>14} {'λ':>24}")
    print(f"    {'-'*4} {'-'*14} {'-'*14} {'-'*24}")
    for i in range(min(8, len(eigen_result['eigenvalues']))):
        ev = eigen_result['eigenvalues'][i]
        gr = eigen_result['growth_rates'][i]
        fr = eigen_result['frequencies_hz'][i]
        print(f"    {i:>4} {gr:>14.6e} {fr:>14.2f} ({ev.real:>10.3e}+{ev.imag:>10.3e}j)")

    # 能量原理分析
    print(f"\n  能量原理 (δW) 分析:")
    energy_result = analyzer.energy_principle_analysis(initial_state)
    for key, val in energy_result.items():
        if isinstance(val, (float, np.floating)):
            print(f"    {key}: {val:.6e}")
        else:
            print(f"    {key}: {val}")

    # ================================================================
    # Step 8: 色散关系分析
    # ================================================================
    separator("Step 8: 色散关系与连续谱")

    from dispersion_analysis import (
        plasma_dispersion_function,
        alfven_continuum_gap_structure,
        tae_frequency_estimate,
        ep_dispersion_correction,
        monte_carlo_orbit_statistics,
        chebyshev_dispersion_solver,
    )

    # 等离子体色散函数
    z_test = np.array([0.0, 0.5, 1.0+0.5j, 3.0+1.0j])
    print("  等离子体色散函数 Z(ζ):")
    for z in z_test:
        Z_val = plasma_dispersion_function(z)
        print(f"    Z({z}) = {Z_val.real:.6f} + {Z_val.imag:.6f}j")

    # 阿尔芬连续谱间隙
    r_array = np.linspace(0.01, 1.0, 30)
    omega_branches, gaps = alfven_continuum_gap_structure(
        lambda r: params.q_profile(r),
        r_array, params.R0, params.v_alfven,
        n_toroidal=1, m_range=(1, 4),
    )
    print(f"\n  连续谱分支数: {omega_branches.shape[0]}")
    print(f"  检测到 {len(gaps)} 个间隙:")
    for i, gap in enumerate(gaps[:5]):
        print(f"    间隙 {i}: r/a={gap['r_norm']:.3f}, "
              f"q={gap['q_rational']:.3f}, "
              f"ω={gap['omega_gap']:.2e} rad/s")

    # TAE 频率估算
    epsilon = params.a_minor / params.R0
    q_res = params.q_profile(0.5)
    omega_tae, f_tae_khz = tae_frequency_estimate(
        0.5, q_res, params.R0, params.v_alfven, 1, epsilon
    )
    print(f"\n  TAE 估算:")
    print(f"    q(r=0.5a) = {q_res:.3f}")
    print(f"    ω_TAE = {omega_tae:.2e} rad/s")
    print(f"    f_TAE = {f_tae_khz:.1f} kHz")

    # EP 色散修正
    omega_test = omega_tae + 0.01j * omega_tae
    D_val, chi_ep = ep_dispersion_correction(
        omega_test, params.beta_fast, params.v_fast, params.v_alfven,
        omega_d=1e4, k_parallel=1.0 / params.R0,
    )
    print(f"\n  EP 色散修正 (ω = ω_TAE + 0.01i ω_TAE):")
    print(f"    D(ω) = {D_val.real:.3e} + {D_val.imag:.3e}j")
    print(f"    χ_EP = {chi_ep.real:.3e} + {chi_ep.imag:.3e}j")

    # Chebyshev 色散多项式求根
    # 构造测试色散多项式: D(ω) = ω² - ω_A² = 0
    omega_A = params.v_alfven / params.R0
    # Chebyshev 展开: ω² = (T₂(ω̃) + 1)/2 (缩放后)
    omega_range = (0.5 * omega_A, 2.0 * omega_A)
    omega_mid = 0.5 * (omega_range[0] + omega_range[1])
    omega_half = 0.5 * (omega_range[1] - omega_range[0])

    # 变换: ω = omega_mid + omega_half * x, x ∈ [-1,1]
    # D = (omega_mid + omega_half x)² - omega_A²
    # = omega_half² x² + 2 omega_mid omega_half x + omega_mid² - omega_A²
    a2 = omega_half ** 2
    a1 = 2 * omega_mid * omega_half
    a0 = omega_mid ** 2 - omega_A ** 2

    # 转为 Chebyshev: x² = (T₂ + 1)/2, x = T₁, 1 = T₀
    c0 = a0 + a2 / 2
    c1 = a1
    c2 = a2 / 2

    roots = chebyshev_dispersion_solver([c0, c1, c2], interval=omega_range)
    print(f"\n  Chebyshev 色散求根:")
    print(f"    精确根: ±ω_A = ±{omega_A:.4e}")
    for i, r in enumerate(roots):
        print(f"    根 {i}: {r.real:.4e} + {r.imag:.4e}j")

    # Monte Carlo 轨道统计
    mc_stats = monte_carlo_orbit_statistics(
        n_samples=200,
        q_safety=q_res,
        R0=params.R0,
        a_minor=params.a_minor,
        v_fast=params.v_fast,
    )
    print(f"\n  Monte Carlo EP 轨道统计 (N=200):")
    for key, val in mc_stats.items():
        if key != 'n_samples':
            print(f"    {key}: {val:.4e}")

    # ================================================================
    # Step 9: 统计分析
    # ================================================================
    separator("Step 9: 统计诊断")

    from statistical_diagnostics import (
        estimate_growth_rate, spectral_analysis,
        fisher_exact_test_2x2, difference_in_differences_analysis,
        mode_decomposition,
    )

    # 从能量时间序列估计增长率
    if len(history['energy']) >= 3:
        gr_result = estimate_growth_rate(
            np.array(history['time']),
            np.array(history['energy']),
        )
        print(f"  能量增长率分析:")
        print(f"    γ = {gr_result['growth_rate']:.6e} s⁻¹")
        print(f"    R² = {gr_result['r_squared']:.6f}")
        print(f"    E-folding 时间 = {gr_result['e_folding_time']:.6e} s")

    # 频谱分析
    if len(history['max_psi']) >= 4:
        spec_result = spectral_analysis(
            np.array(history['time']),
            np.array(history['max_psi']),
        )
        print(f"\n  频谱分析:")
        print(f"    主频率 = {spec_result['dominant_frequency']:.2f} Hz")
        print(f"    频谱重心 = {spec_result['mean_frequency']:.2f} Hz")
        print(f"    谱宽 = {spec_result['spectral_width']:.2f} Hz")
        print(f"    总功率 = {spec_result['total_power']:.6e}")

    # 模分解
    mode_amps, mode_nums = mode_decomposition(initial_state.psi, params.n_theta)
    print(f"\n  极向模分解:")
    print(f"    模数范围: m = 0 ~ {mode_nums[-1]}")
    dominant_m = mode_nums[np.argmax(np.sum(mode_amps, axis=0))]
    print(f"    主导模数: m = {dominant_m}")

    # Fisher 精确检验 (比较稳定/不稳定区域的模分布)
    # 人为构建列联表
    n_high_m_stable = 5
    n_low_m_stable = 15
    n_high_m_unstable = 10
    n_low_m_unstable = 10
    fisher_result = fisher_exact_test_2x2(
        n_high_m_stable, n_low_m_stable,
        n_high_m_unstable, n_low_m_unstable,
    )
    print(f"\n  Fisher 精确检验 (稳定 vs 不稳定高/低 m 模分布):")
    print(f"    比值比 = {fisher_result['odds_ratio']:.4f}")
    print(f"    p 值 = {fisher_result['p_value']:.4f}")
    print(f"    显著: {'是' if fisher_result['is_significant'] else '否'}")

    # DiD 分析: 有/无 EP 的增长率差异
    did_result = difference_in_differences_analysis(
        before_control=0.0,         # 无 EP 前基线
        before_treatment=0.0,       # 有 EP 前基线
        after_control=gr_result.get('growth_rate', 0.0) * 0.5,  # 无 EP 后
        after_treatment=gr_result.get('growth_rate', 0.0),       # 有 EP 后
    )
    print(f"\n  差分-差分 (DiD) 分析 (EP 驱动效应):")
    print(f"    DiD 估计量 = {did_result['did_estimate']:.6e}")
    print(f"    t 统计量 = {did_result['t_statistic']:.4f}")
    print(f"    显著: {'是' if did_result['is_significant'] else '否'}")

    # ================================================================
    # Step 10: 保存结果
    # ================================================================
    separator("Step 10: 保存结果")

    from solution_io import SimulationResultWriter, SolutionFileReader, format_result_table

    writer = SimulationResultWriter(output_dir)

    # 保存各类结果
    f1 = writer.save_eigenvalue_result(eigen_result)
    f2 = writer.save_simulation_history(history)
    f3 = writer.save_stability_result(energy_result)
    f4 = writer.save_plasma_parameters(params)

    print(f"  已保存文件:")
    print(f"    {os.path.basename(f1)}")
    print(f"    {os.path.basename(f2)}")
    print(f"    {os.path.basename(f3)}")
    print(f"    {os.path.basename(f4)}")
    print(f"    plasma_boundary.poly")
    print(f"    flux_surfaces.msh")

    # 验证可以读回
    reader = SolutionFileReader(output_dir)
    eigen_read = reader.read_eigenvalue_file('eigenvalue_result.json')
    print(f"\n  验证读取: 成功读回 {len(eigen_read)} 个本征值")

    # ================================================================
    # 最终汇总
    # ================================================================
    total_time = time.time() - start_time

    separator("模拟完成 - 汇总报告", "=")
    print(f"  总计算时间: {total_time:.3f} s")
    print(f"  空间离散: {params.n_r} × {params.n_theta} (r × θ)")
    print(f"  有限差分阶数: {params.fd_order}")
    print(f"  时间步数: {params.n_timesteps}")
    print(f"  阿尔芬速度: v_A = {params.v_alfven:.2e} m/s")
    print(f"  初始能量: E₀ = {e_init:.6e}")
    print(f"  最终能量: E_f = {e_final:.6e}")
    print(f"  最大增长率: γ_max = {eigen_result['max_growth_rate']:.6e}")
    print(f"  系统稳定性: {'稳定' if eigen_result['is_stable'] else '不稳定'}")
    print(f"  δW 稳定性: {'稳定' if energy_result['is_stable'] else '不稳定'}")
    print(f"  TAE 频率: f_TAE = {f_tae_khz:.1f} kHz")
    print(f"  EP β_fast = {params.beta_fast:.4e}")
    print(f"  守恒量监测: {'全部通过' if monitor.check_all()['_all_ok'] else '存在超限'}")

    # 结果表格
    rows = [
        ["v_A [m/s]", params.v_alfven],
        ["Ω_ci [rad/s]", params.omega_ci],
        ["d_i [m]", params.d_i],
        ["β", params.beta],
        ["β_fast", params.beta_fast],
        ["τ_A [s]", params.tau_alfven],
        ["Δt [s]", params.dt],
        ["γ_max [s⁻¹]", eigen_result['max_growth_rate']],
        ["f_TAE [kHz]", f_tae_khz],
    ]
    table = format_result_table(["物理量", "数值"], rows)
    print(f"\n{table}")

    print(f"\n{'=' * 70}")
    print("  PROJECT_290 模拟运行完成")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
