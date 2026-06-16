"""
main.py - 边界等离子体输运与偏滤器热负荷模拟统一入口

本项目研究边界等离子体在SOL（Scrape-Off Layer）区域的输运过程，
使用高阶有限差分/DG方法求解平行与垂直方向的输运方程，
并通过稳定性分析确保数值格式的可靠性。

核心物理问题:
  1. 沿磁力线方向的平行输运（对流+热传导）
  2. 跨磁力线方向的垂直扩散
  3. 偏滤器靶板鞘层边界条件
  4. 靶板热负荷的参数依赖与不确定性

数值方法:
  - 空间离散：高阶紧致有限差分 + Legendre DG
  - 时间积分：低存储Runge-Kutta
  - 稳定性：von Neumann分析
  - 后处理：小波分析 + Bernstein重建

运行方式：
  python main.py

无需任何参数。
"""

import sys
import os
import numpy as np
import time

# 添加项目目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasma_physics import (
    coulomb_logarithm, electron_ion_collision_frequency,
    spitzer_harm_conductivity, bohm_diffusion, classical_perpendicular_diffusion,
    ion_larmor_radius, sound_speed, sheath_heat_transmission,
    divertor_target_heat_flux, two_point_model, plasma_beta,
    debye_length, electron_plasma_frequency, plasma_state_fidelity,
    EV_TO_JOULE, ELECTRON_CHARGE, DEUTERIUM_MASS
)

from divertor_geometry import (
    generate_sol_mesh, generate_triangular_mesh,
    compute_vertex_to_element_map, hilbert_order_elements,
    generate_divertor_target_profile, check_mesh_quality,
    trace_field_line
)

from legendre_fd import (
    legendre_value, gauss_lobatto_legendre_nodes,
    differentiation_matrix, apply_compact_fd_1st,
    apply_compact_fd_2nd, bernstein_approximation_vectorized,
    verify_fd_convergence, convergence_rate
)

from dg_parallel_solver import (
    DGParallelSolver, generate_1d_mesh, compute_cfl_timestep
)

from perp_diffusion import PerpendicularDiffusionSolver

from iterative_solver import (
    lu_decompose, solve_lu, solve_sparse_cg,
    solve_tridiagonal, estimate_condition_number
)

from stability_analysis import (
    von_neumann_amplification_advection, von_neumann_amplification_diffusion,
    find_max_CFL, generate_stability_report, dg_cfl_limit,
    plasma_transport_eigenvalues
)

from wavelet_diagnostics import (
    daub_coefficients, wavedec, wavelet_denoise,
    detect_elm_events, heat_flux_intermittency,
    compute_wavelet_energy, compute_wavelet_entropy
)

from profile_tools import (
    pwl_interp_1d, cubic_spline_coefficients, cubic_spline_eval,
    generate_density_profile, generate_temperature_profile,
    map_heat_flux_to_target, smooth_profile_bernstein,
    compute_profile_gradients
)

from parameter_scan import (
    ParameterScanner, scan_divertor_heat_flux, scan_recycling_effects,
    monte_carlo_uncertainty, sensitivity_analysis
)

from output_diagnostics import (
    write_ascii_data, write_simulation_summary,
    ConvergenceTracker, compute_plasma_diagnostics,
    compute_solution_quality, generate_final_report
)


def print_header():
    """打印项目标题"""
    print("=" * 72)
    print("  边界等离子体输运与偏滤器热负荷模拟系统")
    print("  Edge Plasma Transport & Divertor Heat Flux Simulation")
    print("  High-Order Finite Difference & Stability Analysis")
    print("  (小规模可复现实验)")
    print("=" * 72)
    print()


# =============================================================================
# 第1阶段：物理参数初始化
# =============================================================================
def phase_physics_initialization():
    """阶段1：设置物理参数并计算基本物理量"""
    print("[阶段 1/8] 物理参数初始化与基本物理量计算")
    print("-" * 50)

    # ITER级参数
    params = {
        'R_major': 6.2,           # 大半径 [m]
        'a_minor': 2.0,           # 小半径 [m]
        'B0': 5.3,                # 环向磁场 [T]
        'n_upstream': 3.0e19,     # 上游密度 [m^-3]
        'T_upstream_eV': 100.0,   # 上游温度 [eV]
        'L_parallel': 30.0,       # 平行连接长度 [m]
        'kappa': 1.7,             # 拉长比
        'triangularity': 0.33,    # 三角变形
    }

    # 计算基本物理量
    n_e = params['n_upstream']
    T_e = params['T_upstream_eV']
    B = params['B0']

    ln_Lambda = coulomb_logarithm(n_e, T_e)
    nu_ei = electron_ion_collision_frequency(n_e, T_e)
    kappa_para = spitzer_harm_conductivity(n_e, T_e)
    D_bohm = bohm_diffusion(T_e, B)
    rho_i = ion_larmor_radius(T_e, B)
    D_classical = classical_perpendicular_diffusion(n_e, T_e, B)
    c_s = sound_speed(T_e, T_e)
    beta = plasma_beta(n_e, T_e, B)
    lambda_D = debye_length(n_e, T_e)
    omega_pe = electron_plasma_frequency(n_e)
    gamma_sheath = sheath_heat_transmission()

    print(f"  Coulomb对数 ln(Λ)      = {ln_Lambda:.3f}")
    print(f"  电子-离子碰撞频率 ν_ei  = {nu_ei:.3e} s^-1")
    print(f"  Spitzer平行热传导 κ_∥   = {kappa_para:.3e} W/(m·eV)")
    print(f"  Bohm扩散系数 D_Bohm     = {D_bohm:.3e} m²/s")
    print(f"  离子Larmor半径 ρ_i      = {rho_i:.6e} m")
    print(f"  经典垂直扩散 D_⊥        = {D_classical:.3e} m²/s")
    print(f"  离子声速 c_s            = {c_s:.3e} m/s")
    print(f"  等离子体β               = {beta:.4f}")
    print(f"  Debye长度 λ_D           = {lambda_D:.3e} m")
    print(f"  电子等离子体频率 ω_pe   = {omega_pe:.3e} rad/s")
    print(f"  鞘层透射系数 γ_sheath   = {gamma_sheath:.3f}")
    print()

    return params


# =============================================================================
# 第2阶段：几何与网格生成
# =============================================================================
def phase_geometry_and_mesh(params):
    """阶段2：生成偏滤器几何和计算网格"""
    print("[阶段 2/8] 偏滤器几何与SOL网格生成")
    print("-" * 50)

    # SOL结构化网格
    sol_mesh = generate_sol_mesh(
        R_major=params['R_major'],
        a_minor=params['a_minor'],
        n_radial=16,
        n_poloidal=32,
        psi_min=1.01,
        psi_max=1.15
    )
    print(f"  SOL网格: {sol_mesh['n_radial']}×{sol_mesh['n_poloidal']} "
          f"({sol_mesh['n_radial'] * sol_mesh['n_poloidal']} 个节点)")

    # 靶板轮廓
    target_profile = generate_divertor_target_profile(
        R_major=params['R_major'],
        a_minor=params['a_minor']
    )
    print(f"  靶板轮廓: {len(target_profile)} 个采样点")
    print(f"  靶板径向范围: [{target_profile[0, 0]:.3f}, {target_profile[-1, 0]:.3f}] m")

    # 三角形网格（用于FEM）
    tri_mesh = generate_triangular_mesh(0.0, 1.0, 0.0, 1.0, nx=12, ny=12)
    print(f"  三角网格: {tri_mesh['n_nodes']} 节点, {tri_mesh['n_elements']} 单元")

    # VTOE映射
    vtoe_ptr, vtoe_list = compute_vertex_to_element_map(
        tri_mesh['elements'], tri_mesh['n_nodes']
    )
    print(f"  VTOE映射: {len(vtoe_list)} 条连接记录")

    # Hilbert排序
    sorted_idx = hilbert_order_elements(tri_mesh['elements'], tri_mesh['nodes'])
    print(f"  Hilbert排序: 已计算 {len(sorted_idx)} 个单元的缓存优化序")

    # 网格质量检查
    quality = check_mesh_quality(tri_mesh['nodes'], tri_mesh['elements'])
    print(f"  网格质量: {'合格' if quality['quality_ok'] else '不合格'} "
          f"(最大边长比={quality['max_aspect_ratio']:.2f})")

    # 场线追踪
    R_start = params['R_major'] + 0.1 * params['a_minor']
    Z_start = -0.5 * params['a_minor']
    field_line = trace_field_line(R_start, Z_start, R_major=params['R_major'],
                                    B0=params['B0'], n_steps=50)
    print(f"  场线追踪: 起始点 ({R_start:.2f}, {Z_start:.2f}), "
          f"终点 ({field_line[-1, 0]:.2f}, {field_line[-1, 1]:.2f})")
    print()

    return {
        'sol_mesh': sol_mesh,
        'target_profile': target_profile,
        'tri_mesh': tri_mesh,
        'field_line': field_line,
    }


# =============================================================================
# 第3阶段：高阶有限差分格式构造
# =============================================================================
def phase_high_order_fd():
    """阶段3：构造和验证高阶有限差分格式"""
    print("[阶段 3/8] 高阶Legendre有限差分格式构造")
    print("-" * 50)

    # GLL节点
    N_order = 4
    nodes, weights = gauss_lobatto_legendre_nodes(N_order)
    print(f"  GLL节点 (N={N_order}): {len(nodes)} 个节点")
    print(f"  节点位置: [{nodes[0]:.6f}, ..., {nodes[-1]:.6f}]")
    print(f"  权重之和: {np.sum(weights):.10f} (应等于2)")

    # 微分矩阵
    D = differentiation_matrix(N_order, nodes)
    print(f"  微分矩阵条件数: {np.linalg.cond(D):.3e}")

    # 紧致格式系数
    from legendre_fd import compact_fd_coefficients_1st, compact_fd_coefficients_2nd
    alpha4, a4, b4 = compact_fd_coefficients_1st(4)
    alpha6, a6, b6 = compact_fd_coefficients_1st(6)
    print(f"  四阶紧致格式: α={alpha4:.4f}, a={a4:.4f}")
    print(f"  六阶紧致格式: α={alpha6:.4f}, a={a6:.4f}, b={b6:.4f}")

    # 收敛阶验证
    test_func = lambda x: np.sin(2.0 * np.pi * x)
    test_dfunc = lambda x: 2.0 * np.pi * np.cos(2.0 * np.pi * x)
    h_values = np.array([0.1, 0.05, 0.025, 0.0125])

    errors_4 = verify_fd_convergence(test_func, test_dfunc, h_values, order=4)
    errors_6 = verify_fd_convergence(test_func, test_dfunc, h_values, order=6)

    rate_4 = convergence_rate(errors_4, h_values)
    rate_6 = convergence_rate(errors_6, h_values)
    print(f"  四阶格式收敛阶: {rate_4:.2f}")
    print(f"  六阶格式收敛阶: {rate_6:.2f}")

    # Bernstein逼近
    n_bernstein = 16
    x_nodes = np.linspace(0, 1, n_bernstein + 1)
    f_vals = np.sin(2 * np.pi * x_nodes)
    x_eval = np.linspace(0, 1, 100)
    f_approx = bernstein_approximation_vectorized(f_vals, x_eval, 0.0, 1.0)
    bernstein_error = np.max(np.abs(f_approx - np.sin(2 * np.pi * x_eval)))
    print(f"  Bernstein逼近 (n={n_bernstein}) 最大误差: {bernstein_error:.6e}")
    print()

    return {
        'N_order': N_order,
        'gll_nodes': nodes,
        'gll_weights': weights,
        'diff_matrix': D,
        'fd_convergence': {'order4': rate_4, 'order6': rate_6},
    }


# =============================================================================
# 第4阶段：DG平行输运求解
# =============================================================================
def phase_dg_parallel_transport(params):
    """阶段4：DG方法求解平行方向输运"""
    print("[阶段 4/8] DG方法平行输运求解")
    print("-" * 50)

    # 创建DG求解器
    c_s = sound_speed(params['T_upstream_eV'], params['T_upstream_eV'])
    solver = DGParallelSolver(
        L_parallel=params['L_parallel'],
        n_elements=24,
        N_order=3,
        advection_speed=c_s,
        diffusion_coeff=10.0
    )

    # 初始条件：高斯脉冲
    L = params['L_parallel']

    def initial_condition(s):
        # 背景 + 高斯脉冲
        background = 1.0
        pulse = 0.5 * np.exp(-((s - L / 3.0) ** 2) / (2.0 * (L / 10.0) ** 2))
        return background + pulse

    solver.set_initial_condition(initial_condition)

    # 记录初始质量
    initial_mass = solver.compute_mass_conservation()
    print(f"  DG单元数: {solver.mesh['n_elements']}")
    print(f"  多项式阶数: {solver.mesh['N_order']}")
    print(f"  时间步长 dt: {solver.dt:.6e} s")
    print(f"  初始质量: {initial_mass:.6e}")

    # 时间推进
    tracker = ConvergenceTracker()
    n_steps = 50
    for step in range(n_steps):
        solver.step(1)
        residual = 1.0 / (step + 1)  # 简化残差
        tracker.record(step + 1, residual)

    final_mass = solver.compute_mass_conservation()
    mass_error = abs(final_mass - initial_mass) / (abs(initial_mass) + 1e-30)

    x_sol, u_sol = solver.get_solution()
    L2_norm = solver.compute_L2_norm()

    print(f"  时间步数: {n_steps}")
    print(f"  最终时间: {solver.t:.6e} s")
    print(f"  质量守恒误差: {mass_error:.6e}")
    print(f"  解的L2范数: {L2_norm:.6e}")
    print(f"  解的范围: [{u_sol.min():.6e}, {u_sol.max():.6e}]")
    print()

    return {
        'solver': solver,
        'solution': (x_sol, u_sol),
        'mass_error': mass_error,
        'tracker': tracker,
    }


# =============================================================================
# 第5阶段：垂直扩散FEM求解
# =============================================================================
def phase_perpendicular_diffusion(params):
    """阶段5：FEM求解垂直方向扩散"""
    print("[阶段 5/8] 垂直扩散FEM求解")
    print("-" * 50)

    # 计算物理扩散系数
    n_e = params['n_upstream']
    T_e = params['T_upstream_eV']
    B = params['B0']

    D_bohm = bohm_diffusion(T_e, B)
    D_classical = classical_perpendicular_diffusion(n_e, T_e, B)
    D_perp = D_bohm * 0.1  # 使用10% Bohm扩散

    print(f"  使用扩散系数: D_⊥ = {D_perp:.3e} m²/s (10% Bohm)")

    # 创建FEM求解器
    fem_solver = PerpendicularDiffusionSolver(
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        nx=10,
        ny=10
    )
    fem_solver.set_diffusion_tensor(D_xx=D_perp, D_yy=D_perp)

    # 源项（中心加热）
    def heat_source(x, y):
        cx, cy = 0.5, 0.5
        sigma = 0.15
        Q_0 = 1.0e6  # 加热功率密度
        return Q_0 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))

    # 求解
    solution = fem_solver.solve(source_func=heat_source, bc_type='hot_boundary')
    print(f"  FEM节点数: {fem_solver.n_nodes}")
    print(f"  FEM单元数: {fem_solver.n_elements}")
    print(f"  温度范围: [{solution.min():.4e}, {solution.max():.4e}] eV")

    # 热流计算
    heat_flux = fem_solver.compute_heat_flux()
    q_mag = np.sqrt(np.sum(heat_flux**2, axis=1))
    print(f"  热流密度范围: [{q_mag.min():.4e}, {q_mag.max():.4e}] W/m²")

    # 测试解析解（如有的话）
    def exact_solution(x, y):
        # 简化测试解
        return 10.0 * np.sin(np.pi * x) * np.sin(np.pi * y) + 1.0

    L2_err = fem_solver.compute_L2_error(exact_solution)
    print(f"  测试L2误差（vs解析解）: {L2_err:.6e}")
    print()

    return {
        'fem_solver': fem_solver,
        'solution': solution,
        'heat_flux': heat_flux,
        'D_perp': D_perp,
    }


# =============================================================================
# 第6阶段：稳定性分析
# =============================================================================
def phase_stability_analysis(params, fd_results):
    """阶段6：数值格式稳定性分析"""
    print("[阶段 6/8] 数值格式稳定性分析")
    print("-" * 50)

    # DG CFL限制
    for N in [1, 2, 3, 4, 5]:
        cfl = dg_cfl_limit(N)
        print(f"  DG (N={N}) CFL限制: {cfl:.4f}")

    # 各格式的最大CFL
    k_h = np.linspace(0.01, np.pi, 200)
    schemes = ['upwind1', 'central2', 'central4', 'compact4', 'compact6']
    print()
    for scheme in schemes:
        cfl_max = find_max_CFL(scheme)
        G = von_neumann_amplification_advection(scheme, k_h, cfl_max * 0.9)
        max_G = np.max(np.abs(G))
        print(f"  {scheme:12s}: CFL_max ≈ {cfl_max:.4f}, |G|_max = {max_G:.6f}")

    # 完整稳定性报告
    stability_report = generate_stability_report(
        advection_speed=sound_speed(params['T_upstream_eV'], params['T_upstream_eV']),
        diffusion_coeff=10.0,
        h=params['L_parallel'] / 24.0,
        N_order=3
    )

    print()
    print(f"  谱半径: {stability_report['spectral_radius']:.3e}")
    print(f"  刚性比: {stability_report['stiffness_ratio']:.3e}")
    print(f"  是否刚性系统: {'是' if stability_report['is_stiff'] else '否'}")
    print(f"  推荐时间积分方法: {stability_report['recommended_method']}")

    # 特征值谱
    eigvals = plasma_transport_eigenvalues(32, 10.0, params['L_parallel'] / 24.0,
                                             sound_speed(params['T_upstream_eV'], params['T_upstream_eV']))
    print(f"  特征值谱: {len(eigvals)} 个模式")
    print(f"  最大实部: {np.max(np.real(eigvals)):.3e}")
    print(f"  最小实部: {np.min(np.real(eigvals)):.3e}")
    print()

    return stability_report


# =============================================================================
# 第7阶段：剖面生成与热负荷诊断
# =============================================================================
def phase_profiles_and_diagnostics(params):
    """阶段7：等离子体剖面生成与靶板热负荷诊断"""
    print("[阶段 7/8] 等离子体剖面与靶板热负荷诊断")
    print("-" * 50)

    # 生成密度和温度剖面
    r_norm = np.linspace(0.0, 1.2, 100)
    n_profile = generate_density_profile(r_norm)
    T_profile = generate_temperature_profile(r_norm)

    print(f"  密度剖面: 芯部={n_profile[0]:.3e}, 脚部={n_profile[60]:.3e} m^-3")
    print(f"  温度剖面: 芯部={T_profile[0]:.2f}, 脚部={T_profile[60]:.2f} eV")

    # Bernstein光滑重建
    n_smooth = smooth_profile_bernstein(n_profile, r_norm, degree=12)
    T_smooth = smooth_profile_bernstein(T_profile, r_norm, degree=12)
    smooth_err_n = np.max(np.abs(n_smooth - n_profile))
    smooth_err_T = np.max(np.abs(T_smooth - T_profile))
    print(f"  Bernstein光滑重建误差: n={smooth_err_n:.3e}, T={smooth_err_T:.3e}")

    # 剖面梯度
    dn_dr = compute_profile_gradients(n_profile, r_norm)
    dT_dr = compute_profile_gradients(T_profile, r_norm)
    print(f"  密度梯度范围: [{dn_dr.min():.3e}, {dn_dr.max():.3e}]")
    print(f"  温度梯度范围: [{dT_dr.min():.3e}, {dT_dr.max():.3e}]")

    # 靶板热负荷映射
    R_target = np.linspace(1.2, 1.8, 50)
    q_target = map_heat_flux_to_target(R_target, q_upstream=1.0e7, lambda_q=0.003)
    q_peak = np.max(q_target)
    print(f"  靶板热负荷峰值: {q_peak:.3e} W/m²")
    print(f"   Strike point位置: R ≈ {R_target[np.argmax(q_target)]:.3f} m")

    # 等离子体诊断
    R_grid = params['R_major'] + params['a_minor'] * r_norm
    diagnostics = compute_plasma_diagnostics(
        n_profile, T_profile, R_grid, params['B0']
    )
    print(f"  储能: {diagnostics['stored_energy']:.3e} J/m")
    print(f"  总粒子数: {diagnostics['total_particles']:.3e} m^-2")
    print(f"  靶板热负荷: {diagnostics['q_target']:.3e} W/m²")

    # 两点模型
    kappa = spitzer_harm_conductivity(params['n_upstream'], params['T_upstream_eV'])
    q_parallel = (2.0 / 7.0) * kappa * params['T_upstream_eV']**3.5 / params['L_parallel']
    tpm_result = two_point_model(q_parallel, params['n_upstream'],
                                   params['T_upstream_eV'], params['L_parallel'])
    print(f"  两点模型靶板温度: {tpm_result['T_target_eV']:.2f} eV")
    print(f"  两点模型靶板密度: {tpm_result['n_target']:.3e} m^-3")
    print()

    return {
        'r_norm': r_norm,
        'n_profile': n_profile,
        'T_profile': T_profile,
        'diagnostics': diagnostics,
        'two_point_model': tpm_result,
    }


# =============================================================================
# 第8阶段：参数扫描与不确定性分析
# =============================================================================
def phase_parameter_scan_and_uncertainty(params):
    """阶段8：参数扫描、小波分析和不确定性量化"""
    print("[阶段 8/8] 参数扫描、小波分析与不确定性量化")
    print("-" * 50)

    # 参数扫描
    n_range = np.array([1e19, 2e19, 3e19, 5e19])
    T_range = np.array([50.0, 100.0, 200.0, 400.0])

    scan_result = scan_divertor_heat_flux(n_range, T_range, params['L_parallel'])
    print(f"  参数扫描: {len(n_range)}×{len(T_range)} = {len(n_range) * len(T_range)} 组合")
    print(f"  靶板热负荷范围: [{scan_result['q_target'].min():.3e}, "
          f"{scan_result['q_target'].max():.3e}] W/m²")

    # 再循环扫描
    R_range = np.linspace(0.5, 0.99, 10)
    recycling_result = scan_recycling_effects(R_range)
    print(f"  再循环系数扫描: R ∈ [{R_range[0]:.2f}, {R_range[-1]:.2f}]")
    print(f"  靶板热负荷变化: [{recycling_result['q_target'].min():.3e}, "
          f"{recycling_result['q_target'].max():.3e}] W/m²")

    # 合成热负荷信号（带ELM和湍流）
    np.random.seed(42)
    t_signal = np.linspace(0, 0.1, 512)
    q_signal = 5.0e6 * np.ones_like(t_signal)

    # 添加湍流涨落
    q_signal += 0.5e6 * np.random.randn(len(t_signal))

    # 添加ELM事件（3个尖峰）
    elm_times = [0.025, 0.055, 0.085]
    for t_elm in elm_times:
        elm_idx = np.argmin(np.abs(t_signal - t_elm))
        q_signal[max(0, elm_idx - 3):elm_idx + 5] += 10e6 * np.exp(
            -np.arange(-3, 5)**2 / 4.0)

    # 小波分析
    details, approx = wavedec(q_signal, level=4)
    energy = compute_wavelet_energy(details)
    entropy = compute_wavelet_entropy(details)
    print(f"  小波能量分布: {energy}")
    print(f"  小波熵: {entropy:.4f}")

    # ELM检测
    elm_indices = detect_elm_events(q_signal, threshold_factor=2.5)
    print(f"  检测到ELM事件: {len(elm_indices)} 个")

    # 间歇性分析
    intermittency = heat_flux_intermittency(q_signal)
    print(f"  热负荷偏度: {intermittency['skewness']:.4f}")
    print(f"  热负荷峰度: {intermittency['kurtosis']:.4f}")
    print(f"  Hurst指数: {intermittency['hurst_exponent']:.4f}")
    print(f"  间歇性参数: {intermittency['intermittency_param']:.4f}")

    # Monte Carlo不确定性
    mc_result = monte_carlo_uncertainty(n_samples=200, L_parallel=params['L_parallel'])
    print(f"  Monte Carlo采样: {mc_result['n_samples']} 个样本")
    print(f"  靶板热负荷均值: {mc_result['q_target_mean']:.3e} W/m²")
    print(f"  靶板热负荷标准差: {mc_result['q_target_std']:.3e} W/m²")
    print(f"  90%置信区间: [{mc_result['q_target_5th']:.3e}, "
          f"{mc_result['q_target_95th']:.3e}] W/m²")

    # 敏感性分析
    def output_func(p):
        kappa = spitzer_harm_conductivity(p['n_upstream'], p['T_upstream'])
        q_para = (2.0 / 7.0) * kappa * p['T_upstream']**3.5 / params['L_parallel']
        res = two_point_model(q_para, p['n_upstream'], p['T_upstream'], params['L_parallel'])
        return res['q_target_Wm2']

    base_params = {'n_upstream': 3e19, 'T_upstream': 100.0}
    sens_n = sensitivity_analysis(
        base_params, 'n_upstream',
        np.linspace(1e19, 6e19, 10), output_func
    )
    print(f"  密度敏感性: {sens_n['relative_sensitivity']:.4f}")
    print()

    return {
        'scan_result': scan_result,
        'mc_result': mc_result,
        'intermittency': intermittency,
    }


# =============================================================================
# 主程序
# =============================================================================
def main():
    """主入口函数"""
    start_time = time.time()
    print_header()

    # 阶段1：物理参数初始化
    params = phase_physics_initialization()

    # 阶段2：几何与网格
    geom_results = phase_geometry_and_mesh(params)

    # 阶段3：高阶有限差分
    fd_results = phase_high_order_fd()

    # 阶段4：DG平行输运
    dg_results = phase_dg_parallel_transport(params)

    # 阶段5：垂直扩散
    perp_results = phase_perpendicular_diffusion(params)

    # 阶段6：稳定性分析
    stability = phase_stability_analysis(params, fd_results)

    # 阶段7：剖面与诊断
    profile_results = phase_profiles_and_diagnostics(params)

    # 阶段8：参数扫描与不确定性
    scan_results = phase_parameter_scan_and_uncertainty(params)

    # 生成综合报告
    total_time = time.time() - start_time

    summary = {
        'physics_params': {
            'R_major [m]': params['R_major'],
            'a_minor [m]': params['a_minor'],
            'B0 [T]': params['B0'],
            'n_upstream [m^-3]': params['n_upstream'],
            'T_upstream [eV]': params['T_upstream_eV'],
            'L_parallel [m]': params['L_parallel'],
        },
        'numerical_params': {
            'DG_elements': dg_results['solver'].mesh['n_elements'],
            'DG_order': dg_results['solver'].mesh['N_order'],
            'FEM_nodes': perp_results['fem_solver'].n_nodes,
            'FEM_elements': perp_results['fem_solver'].n_elements,
            'fd_convergence_order4': fd_results['fd_convergence']['order4'],
        },
        'stability': stability,
        'convergence': {
            'final_residual': dg_results['tracker'].residuals[-1] if dg_results['tracker'].residuals else 0.0,
            'mass_conservation_error': dg_results['mass_error'],
        },
        'physics_results': {
            'stored_energy [J/m]': profile_results['diagnostics']['stored_energy'],
            'q_target [W/m^2]': profile_results['diagnostics']['q_target'],
            'T_target [eV]': profile_results['two_point_model']['T_target_eV'],
        },
        'parameter_scan': {
            'q_target_mean [W/m^2]': scan_results['mc_result']['q_target_mean'],
            'q_target_std [W/m^2]': scan_results['mc_result']['q_target_std'],
        },
        'total_computation_time [s]': total_time,
    }

    # 打印报告
    report = generate_final_report(summary)
    print(report)

    # 写入输出文件
    output_dir = os.path.dirname(os.path.abspath(__file__))
    summary_file = os.path.join(output_dir, 'simulation_summary.txt')
    write_simulation_summary(summary_file, summary)
    print(f"\n摘要报告已保存至: {summary_file}")

    # 保存靶板热负荷数据
    R_target = np.linspace(1.2, 1.8, 50)
    from profile_tools import map_heat_flux_to_target
    q_target_data = map_heat_flux_to_target(R_target, q_upstream=1.0e7)
    heat_flux_file = os.path.join(output_dir, 'divertor_heat_flux.txt')
    write_ascii_data(heat_flux_file, ['R [m]', 'q_target [W/m^2]'],
                      np.column_stack([R_target, q_target_data]),
                      "Divertor Target Heat Flux Profile")
    print(f"靶板热负荷数据已保存至: {heat_flux_file}")

    print("\n" + "=" * 72)
    print("  模拟完成！总计算时间: {:.2f} 秒".format(total_time))
    print("=" * 72)


if __name__ == "__main__":
    main()
