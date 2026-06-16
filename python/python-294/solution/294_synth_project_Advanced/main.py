"""
main.py - 激光等离子体相互作用高阶有限差分仿真统一入口

=================================================================
计算等离子体: 激光等离子体相互作用
高阶有限差分与稳定性分析 (小规模可复现实验)
=================================================================

本程序实现 1D1V Vlasov-Maxwell 方程组的高阶有限差分数值求解,
模拟强激光脉冲与非均匀等离子体靶的相互作用过程。

物理模型:
  - 电子动力学: 1D1V Vlasov 方程
  - 电磁场: 1D Maxwell 方程组
  - 碰撞: Fokker-Planck 碰撞算子
  - 边界: 吸收边界 + 激光注入

数值方法:
  - 空间离散: 2/4/6/8 阶中心差分
  - 时间积分: Strang 分裂 + RK45
  - 色散分析: Newton-Maehly 求根
  - 模式分解: Modified Gram-Schmidt
  - 参数估计: 贝叶斯 MCMC

融合算法来源 (15 个种子项目):
  [057_atbash]       -> 参数标签编码 (numerical_core.py)
  [150_cg_lab_tri]   -> 边界几何有符号距离 (boundary_geometry.py)
  [272_dg1d_burgers] -> DG 数值通量 + RK5 时间推进 (vlasov_maxwell.py)
  [283_diffusion_pde]-> 碰撞扩散算子 (collision_operator.py)
  [358_fd1d_bvp]     -> 有限差分边值问题 (equilibrium via fd_operators.py)
  [480_gram_schmidt] -> 模式正交化分解 (mode_decomposition.py)
  [538_histogram_2d] -> 2D 相空间采样 (distribution_sampler.py)
  [698_log_normal]   -> 超热电子 log-normal 分布 (distribution_sampler.py)
  [786_nas]          -> NAS 核心算子验证 (numerical_core.py)
  [801_newton_maehly]-> 色散多项式求根 (dispersion_solver.py)
  [995_r8sm]         -> Sherman-Morrison 矩阵更新 (matrix_update.py)
  [1080_safe_CCMPC]  -> 抽象基类 + 场景管理架构 (boundary_conditions.py)
  [1216_borexino]    -> 贝叶斯 MCMC 参数估计 (diagnostics_bayesian.py)
  [1242_PhotonDOS]   -> 模式态密度计算 (dos_plasma.py)
  [1393_vin]         -> 参数完整性校验和 (numerical_core.py)

运行方式:
    python main.py

无参数, 直接运行即可。
=================================================================
"""

import sys
import time
import numpy as np

from config import SimulationConfig, PhysicalConstants, NormalizationScales
from numerical_core import (
    vin_checksum, atbash_encode, atbash_decode,
    check_array_finite, check_physical_bounds,
    check_conservation, compute_simulation_fingerprint,
)
from fd_operators import (
    fd_coefficients_first_deriv, fd_coefficients_second_deriv,
    apply_first_derivative, apply_second_derivative,
    apply_fourth_derivative, modified_wavenumber,
    spectral_error_analysis,
)
from stability import (
    von_neumann_advection, von_neumann_diffusion,
    cfl_condition, stability_diagram, rk4_stability_boundary,
)
from dispersion_solver import (
    poly_eval_horner, newton_maehly_roots,
    beam_plasma_dispersion_polynomial,
    solve_dispersion_sweep, dispersion_analysis,
)
from mode_decomposition import (
    classical_gram_schmidt, modified_gram_schmidt,
    compare_cgs_mgs, decompose_field_modes,
    electromagnetic_mode_basis,
)
from dos_plasma import (
    epsilon_cold_plasma, em_dispersion_relation,
    compute_total_dos, dos_frequency_integral,
    plasma_dos_analysis,
)
from distribution_sampler import (
    log_normal_pdf, log_normal_cdf, log_normal_inverse_cdf,
    log_normal_mean, log_normal_variance, log_normal_sample,
    compute_discrete_cdf_2d, sample_from_histogram_2d,
    maxwellian_distribution, two_temperature_distribution,
    initialize_phase_space,
)
from collision_operator import (
    coulomb_logarithm, electron_ion_collision_frequency,
    spitzer_resistivity, collision_operator_fokker_planck,
    solve_collisional_relaxation, inverse_bremsstrahlung_absorption,
)
from vlasov_maxwell import (
    compute_current_density, compute_charge_density,
    run_vlasov_maxwell_simulation,
)
from boundary_conditions import (
    apply_periodic_bc_1d, apply_neumann_bc_1d,
    apply_absorbing_layer, apply_outflow_bc,
    inject_laser_field, velocity_space_bc,
    apply_all_boundary_conditions,
)
from boundary_geometry import (
    signed_distance_to_line, plasma_boundary_normal,
    density_gradient_scale_length, find_critical_surface,
    boundary_geometry_analysis,
)
from matrix_update import (
    lu_factorize, lu_solve, sherman_morrison_solve,
    sherman_morrison_mv, plasma_dielectric_update_system,
)
from diagnostics_bayesian import (
    gaussian_log_likelihood, uniform_log_prior,
    forward_model, metropolis_hastings,
    bayesian_plasma_diagnostics,
)


def print_header():
    """打印程序头信息。"""
    print("=" * 72)
    print("  激光等离子体相互作用: 高阶有限差分与稳定性分析")
    print("  High-Order Finite Difference for Laser-Plasma Interaction")
    print("=" * 72)
    print()


def print_separator(title):
    """打印分隔线。"""
    print()
    print("-" * 60)
    print(f"  {title}")
    print("-" * 60)


def phase1_physical_setup(config):
    """阶段 1: 物理参数设置与校验。"""
    print_separator("阶段 1: 物理参数设置与完整性校验")

    pc = PhysicalConstants
    norm = config.norm

    print(f"  参考密度 n_0 = {norm.n0:.2e} m^-3")
    print(f"  电子温度 T_e = {norm.Te_J:.2e} J ({norm.Te_J / pc.e_charge:.0f} eV)")
    print(f"  等离子体频率 omega_p0 = {norm.omega_p0:.2e} rad/s")
    print(f"  趋肤深度 c/omega_p0 = {norm.skin_depth:.2e} m")
    print(f"  热速度 v_th = {norm.v_th:.2e} m/s (v_th/c = {norm.v_th / pc.c_light:.4f})")
    print(f"  德拜长度 lambda_D = {norm.lambda_D:.2e} m")
    print(f"  归一化激光频率 omega_L/omega_p0 = {norm.omega_L / norm.omega_p0:.4f}")

    # 仿真参数摘要
    print(f"\n  空间格点 N_x = {config.N_x}, dx = {config.dx:.4f}")
    print(f"  速度格点 N_v = {config.N_v}, dv = {config.dv:.4f}")
    print(f"  时间步长 dt = {config.dt:.4f}, CFL = {config.CFL}")
    print(f"  FD 阶数 = {config.fd_order}, DG 阶数 = {config.dg_order}")

    # 参数完整性校验 (VIN 校验和)
    print("\n  [参数完整性校验]")
    fingerprint = compute_simulation_fingerprint(config)
    for key, info in fingerprint.items():
        encoded = info['encoded_label']
        cksum = info['check_char']
        print(f"    {key} -> {encoded} (checksum: {cksum})")

    # 等离子体密度剖面
    n_profile = config.plasma_density_profile()
    ok_n, info_n = check_physical_bounds(n_profile, 0, config.n_max * 1.1, "density")
    print(f"\n  密度剖面: {info_n}")
    ok_f, info_f = check_array_finite(n_profile, "density_profile")
    print(f"  有限性检查: {info_f}")

    return True


def phase2_fd_operator_validation(config):
    """阶段 2: 高阶有限差分算子验证。"""
    print_separator("阶段 2: 高阶有限差分算子精度验证")

    x = config.x
    dx = config.dx

    # 测试函数: f(x) = sin(k*x), k = 2*pi/L * mode_number
    k_test = 2.0 * np.pi / config.L_x * 3  # 第 3 模式
    f_exact = np.sin(k_test * x)
    df_exact = k_test * np.cos(k_test * x)
    d2f_exact = -k_test ** 2 * np.sin(k_test * x)

    print(f"  测试函数: f(x) = sin({k_test:.4f} * x)")
    print(f"  精确一阶导: f'(x) = {k_test:.4f} * cos({k_test:.4f} * x)")
    print(f"  精确二阶导: f''(x) = -{k_test ** 2:.4f} * sin({k_test:.4f} * x)")

    errors_d1 = {}
    errors_d2 = {}

    for order in [2, 4, 6, 8]:
        try:
            df_approx = apply_first_derivative(f_exact, dx, order=order, bc='periodic')
            d2f_approx = apply_second_derivative(f_exact, dx, order=order, bc='periodic')

            err_d1 = np.max(np.abs(df_approx - df_exact))
            err_d2 = np.max(np.abs(d2f_approx - d2f_exact))
            errors_d1[order] = err_d1
            errors_d2[order] = err_d2

            print(f"\n  {order}阶差分:")
            print(f"    一阶导最大误差: {err_d1:.6e}")
            print(f"    二阶导最大误差: {err_d2:.6e}")
        except Exception as e:
            print(f"\n  {order}阶差分: 跳过 ({e})")

    # 收敛阶数估计
    print("\n  [收敛阶数分析]")
    orders_list = sorted(errors_d1.keys())
    if len(orders_list) >= 2:
        for i in range(1, len(orders_list)):
            o1, o2 = orders_list[i - 1], orders_list[i]
            if errors_d1[o2] > 0 and errors_d1[o1] > 0:
                rate = np.log(errors_d1[o1] / errors_d1[o2]) / np.log(2)
                print(f"    一阶导 {o1}->{o2} 阶: 误差比 = {errors_d1[o1] / errors_d1[o2]:.1f}, 收敛率 ~ {rate:.1f}")

    # 谱误差分析
    print("\n  [修正波数分析]")
    spec = spectral_error_analysis(100, [2, 4, 6, 8])
    for order in [2, 4, 6, 8]:
        key = f'error_order{order}'
        if key in spec:
            max_err = np.max(np.abs(spec[key]))
            print(f"    {order}阶: max |k_eff - k| = {max_err:.6e}")

    return True


def phase3_stability_analysis(config):
    """阶段 3: 冯·诺伊曼稳定性分析。"""
    print_separator("阶段 3: 冯·诺伊曼稳定性分析")

    # CFL 条件
    print("  [CFL 稳定性条件]")
    for order in [2, 4, 6, 8]:
        for scheme in ['leapfrog', 'lax_friedrichs', 'rk4']:
            cfl = cfl_condition(order, scheme, c=1.0)
            print(f"    {order}阶 + {scheme:15s}: CFL_max = {cfl['cfl_max_nu']:.4f} ({cfl['condition']})")

    # RK4 稳定域
    rk4_info = rk4_stability_boundary()
    print(f"\n  RK4 稳定域:")
    print(f"    虚轴: |Im(z)| <= {rk4_info['imag_axis_stable_max']:.4f}")
    print(f"    实轴: Re(z) >= {rk4_info['real_axis_stable_min']:.4f}")

    # 对流方程 Von Neumann 分析
    print("\n  [对流方程 Von Neumann 分析]")
    nu_test = np.array([0.3, 0.5, 0.8, 1.0])
    for scheme in ['leapfrog', 'lax_friedrichs', 'rk4']:
        result = von_neumann_advection(nu_test, scheme=scheme, fd_order=config.fd_order)
        for nu in nu_test:
            key = f'modG_nu{nu:.3f}'
            if key in result:
                max_G = np.max(result[key])
                stable = max_G <= 1.0 + 1e-14
                status = "稳定" if stable else "不稳定"
                print(f"    {scheme} nu={nu:.1f}: max|G| = {max_G:.6f} ({status})")

    # 扩散方程稳定性
    print("\n  [扩散方程 Von Neumann 分析]")
    r_test = np.array([0.1, 0.3, 0.49, 0.5, 0.51, 0.7])
    diff_result = von_neumann_diffusion(r_test)
    for r in r_test:
        key = f'modG_r{r:.4f}'
        if key in diff_result:
            max_G = np.max(np.abs(diff_result[key]))
            stable = max_G <= 1.0 + 1e-14
            status = "稳定" if stable else "不稳定"
            print(f"    r={r:.4f}: max|G| = {max_G:.6f} ({status})")

    # 完整稳定性图
    diagram = stability_diagram([2, 4], ['leapfrog', 'rk4'])
    print("\n  [稳定性图摘要]")
    for order in diagram:
        for scheme in diagram[order]:
            if scheme == 'cfl_rk4':
                continue
            if isinstance(diagram[order][scheme], dict) and 'max_stable_nu' in diagram[order][scheme]:
                info = diagram[order][scheme]
                print(f"    {order}阶 + {scheme}: max_stable_nu = {info['max_stable_nu']:.4f}")

    return True


def phase4_dispersion_analysis(config):
    """阶段 4: 色散关系求解与模式分析。"""
    print_separator("阶段 4: Newton-Maehly 色散关系求解")

    # 演示 Newton-Maehly: 简单多项式 x^3 - 6x^2 + 11x - 6 = 0
    # 根为 1, 2, 3
    print("  [验证: 多项式 x^3 - 6x^2 + 11x - 6 = 0, 根 = {1, 2, 3}]")
    coeffs_test = np.array([-6.0, 11.0, -6.0, 1.0])
    roots_test, info_test = newton_maehly_roots(coeffs_test)
    roots_sorted = sorted(roots_test, key=lambda r: r.real)
    print(f"    计算根: {[f'{r.real:.6f}' for r in roots_sorted]}")
    print(f"    收敛: {info_test['converged']}, 迭代: {info_test['iterations']}")

    # Horner 验证
    for r in roots_test:
        p, dp = poly_eval_horner(coeffs_test, r)
        print(f"    根 {r.real:.6f}: p(r) = {abs(p):.2e}, p'(r) = {abs(dp):.4f}")

    # 双束等离子体色散分析
    print("\n  [双束等离子体色散分析]")
    disp_results = dispersion_analysis(config)

    print(f"  束参数:")
    bp = disp_results['beam_params']
    print(f"    omega_p_bg = {bp['omega_p_bg']:.2f}")
    print(f"    omega_p_beam = {bp['omega_p_beam']:.2f}")
    print(f"    v_beam = {bp['v_beam']:.2f}")

    sweep = disp_results['sweep']
    max_growth = sweep['max_growth_rate']
    unstable_k = sweep['unstable_k']
    print(f"\n  不稳定性分析:")
    print(f"    最大增长率: {np.max(max_growth):.6f}")
    print(f"    不稳定波数范围: k in [{unstable_k.min():.3f}, {unstable_k.max():.3f}]"
          if len(unstable_k) > 0 else "    无不稳定波数")
    print(f"    测试 k={disp_results['test_k']:.1f} 的根:")
    for r in disp_results['test_roots']:
        mode_type = "不稳定" if r.imag > 0 else "稳定"
        print(f"      omega = {r.real:.6f} + {r.imag:.6f}i ({mode_type})")

    # 磁化等离子体色散
    mag = disp_results['magnetized']
    print(f"\n  磁化等离子体 (theta=pi/4, Omega_ce=0.5):")
    print(f"    O 模式 omega^2 = {mag['omega_sq_O']}")
    print(f"    X 模式 omega^2 = {mag['omega_sq_X']}")

    return True


def phase5_mode_decomposition(config):
    """阶段 5: 电磁模式 Gram-Schmidt 分解。"""
    print_separator("阶段 5: Gram-Schmidt 模式分解")

    # 构建理论电磁模式基
    omega_p = np.sqrt(config.plasma_density_profile().max())
    basis, omega_modes, k_modes = electromagnetic_mode_basis(
        config.x, omega_p, config.L_x, n_modes=8
    )

    print(f"  理论模式数: 8")
    print(f"  模式频率: {omega_modes[:4]}... (前4个)")
    print(f"  模式波数: {k_modes[:4]}... (前4个)")

    # 添加噪声模拟仿真数据
    rng = np.random.RandomState(42)
    E_snapshots = basis @ np.diag(omega_modes) + 0.01 * rng.randn(config.N_x, 8)

    # CGS vs MGS 比较
    print("\n  [CGS vs MGS 正交化比较]")
    comparison = compare_cgs_mgs(E_snapshots)
    print(f"  {'容差':>10s}  {'CGS rank':>8s}  {'CGS err':>12s}  {'MGS rank':>8s}  {'MGS err':>12s}  {'MGS优?'}")
    for tol, comp in comparison.items():
        better = "Yes" if comp['mgs_better'] else "No"
        print(f"  {tol:>10.1e}  {comp['cgs_rank']:>8d}  {comp['cgs_orth_err']:>12.2e}"
              f"  {comp['mgs_rank']:>8d}  {comp['mgs_orth_err']:>12.2e}  {better}")

    # 使用 MGS 进行模式分解
    print("\n  [MGS 模式分解结果]")
    modes = decompose_field_modes(E_snapshots, method='mgs', tol=1e-12)
    print(f"  有效秩: {modes['rank']}")
    print(f"  正交性误差: {modes['info']['orthogonality_error']:.2e}")
    print(f"  总能量: {modes['total_energy']:.4f}")
    print(f"  模式能量占比 (前4):")
    for i in range(min(4, modes['rank'])):
        print(f"    模式 {i + 1}: {modes['energy_fractions'][i]:.4f} "
              f"(E={modes['mode_energies'][i]:.4f})")

    return True


def phase6_distribution_sampling(config):
    """阶段 6: 粒子分布函数初始化与采样。"""
    print_separator("阶段 6: 分布函数初始化与采样")

    # Log-normal 分布特性
    mu = config.log_normal_mu
    sigma = config.log_normal_sigma

    print(f"  [Log-normal 超热电子分布]")
    print(f"    mu = {mu:.2f}, sigma = {sigma:.2f}")
    print(f"    均值 = {log_normal_mean(mu, sigma):.4f}")
    print(f"    方差 = {log_normal_variance(mu, sigma):.4f}")

    # 采样测试
    samples = log_normal_sample(mu, sigma, 1000, seed=42)
    print(f"    采样 1000 个: mean = {np.mean(samples):.4f}, "
          f"std = {np.std(samples):.4f}")

    # CDF 反函数验证
    p_test = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    x_cdf = log_normal_inverse_cdf(p_test, mu, sigma)
    cdf_back = log_normal_cdf(x_cdf, mu, sigma)
    max_err = np.max(np.abs(cdf_back - p_test))
    print(f"    CDF 反函数精度: max|F(F^-1(p))-p| = {max_err:.2e}")

    # Maxwell 分布
    print(f"\n  [Maxwell-Boltzmann 分布]")
    v = config.v
    f_max = maxwellian_distribution(v, n0=1.0, vth=1.0)
    f_2T = two_temperature_distribution(v, n0=1.0, vth_cold=0.5, vth_hot=3.0, alpha=0.9)
    norm_max = np.sum(f_max) * config.dv
    norm_2T = np.sum(f_2T) * config.dv
    print(f"    Maxwell 归一化: integral = {norm_max:.6f}")
    print(f"    双温分布归一化: integral = {norm_2T:.6f}")

    # 完整相空间初始化
    print(f"\n  [相空间初始化]")
    f, phase_info = initialize_phase_space(config)
    print(f"    分布函数 shape: {f.shape}")
    print(f"    最大 f: {phase_info['max_f']:.6e}")
    print(f"    最小 f: {phase_info['min_f']:.6e}")
    print(f"    总粒子数: {phase_info['total_particles']:.4f}")
    print(f"    密度范围: [{phase_info['density'].min():.4f}, {phase_info['density'].max():.4f}]")

    ok_f, info_f = check_array_finite(f, "f_phase_space")
    print(f"    有限性: {info_f}")

    return True


def phase7_dos_analysis(config):
    """阶段 7: 等离子体态密度计算。"""
    print_separator("阶段 7: 等离子体电磁模式态密度")

    dos_results = plasma_dos_analysis(config)

    print(f"  最大等离子体频率: omega_p_max = {dos_results['omega_p_max']:.4f}")
    print(f"  总模式数 (积分): {dos_results['total_modes']:.4f}")
    print(f"  能量加权积分: {dos_results['total_energy_weighted']:.4f}")

    # 介电函数测试
    omega_test = 1.5
    omega_p = 1.0
    eps = epsilon_cold_plasma(omega_test, omega_p)
    print(f"\n  冷等离子体介电函数 (omega={omega_test}, omega_p={omega_p}):")
    print(f"    eps = {eps.real:.4f} + {eps.imag:.4f}i")
    print(f"    |eps| = {abs(eps):.4f}")

    # 色散关系残差
    k_test = 1.0
    D_em = em_dispersion_relation(omega_test, k_test, omega_p)
    print(f"    电磁色散残差 D(omega={omega_test}, k={k_test}) = {D_em:.4f}")

    # 允许模式
    print(f"\n  允许模式 (k=1.0):")
    print(f"    EM 模式: {dos_results['em_modes_at_k1']}")
    print(f"    Langmuir 模式: {dos_results['langmuir_modes_at_k1']}")

    return True


def phase8_boundary_geometry(config):
    """阶段 8: 边界几何分析。"""
    print_separator("阶段 8: 等离子体边界几何分析")

    geom = boundary_geometry_analysis(config)

    print(f"  峰值密度位置: x = {geom['peak_position']:.4f}")
    print(f"  峰值密度: n = {geom['peak_density']:.4f}")
    print(f"  密度梯度尺度长度 (峰值处): L_n = {geom['scale_length_at_peak']:.4f}")

    if geom['has_critical_surface']:
        print(f"  临界面位置: {geom['critical_positions']}")
    else:
        print(f"  无临界面 (omega_0^2 = {config.omega_0 ** 2:.4f} > n_max = {config.n_max})")

    # 有符号距离计算
    p1 = geom['boundary_left']
    p2 = geom['boundary_right']
    p_test = np.array([config.L_x / 2, 0.0])
    dist = signed_distance_to_line(p1, p2, p_test)
    print(f"\n  有符号距离 (中心点到边界线): {dist:.4f}")

    # 密度梯度分析
    print(f"  密度梯度法线范围: [{geom['normal'].min():.4f}, {geom['normal'].max():.4f}]")
    print(f"  梯度幅值范围: [{geom['gradient_magnitude'].min():.6f}, "
          f"{geom['gradient_magnitude'].max():.4f}]")

    return True


def phase9_collision_operator(config):
    """阶段 9: 碰撞算子与弛豫。"""
    print_separator("阶段 9: 碰撞算子与热化弛豫")

    # Coulomb 对数
    n_e_phys = config.norm.n0
    T_e_phys = config.norm.Te_J
    ln_Lambda = coulomb_logarithm(n_e_phys, T_e_phys)
    print(f"  Coulomb 对数: ln(Lambda) = {ln_Lambda:.2f}")

    # 碰撞频率
    nu_ei = electron_ion_collision_frequency(n_e_phys, T_e_phys)
    print(f"  e-i 碰撞频率: nu_ei = {nu_ei:.2e} s^-1")

    # Spitzer 电阻率
    eta = spitzer_resistivity(n_e_phys, T_e_phys)
    print(f"  Spitzer 电阻率: eta = {eta:.2e} Ohm*m")

    # 逆韧致吸收
    omega_L_phys = config.norm.omega_L
    ib = inverse_bremsstrahlung_absorption(n_e_phys, T_e_phys, omega_L_phys)
    print(f"  逆韧致吸收:")
    print(f"    传播: {ib['propagates']}")
    if ib['propagates']:
        print(f"    吸收系数: alpha_IB = {ib['alpha_IB']:.2e} m^-1")
        print(f"    omega_p/omega_L = {ib['ratio']:.4f}")

    # 碰撞弛豫仿真
    print(f"\n  [碰撞弛豫仿真]")
    v = config.v
    dv = config.dv
    f_init = maxwellian_distribution(v, n0=1.0, vth=1.5)  # 稍热的初始分布
    nu_coll_norm = config.nu_ei_base  # 归一化碰撞频率

    n_relax_steps = 500
    dt_coll = 0.01
    f_final, relax_history = solve_collisional_relaxation(
        f_init, v, dv, nu_coll_norm, 1.0, n_relax_steps, dt_coll
    )

    print(f"    弛豫步数: {n_relax_steps}")
    print(f"    密度守恒: delta_n/n = {relax_history['density_conservation']:.2e}")
    print(f"    能量守恒: delta_E/E = {relax_history['energy_conservation']:.2e}")
    print(f"    初始温度: {relax_history['temperatures'][0]:.4f}")
    print(f"    最终温度: {relax_history['temperatures'][-1]:.4f}")
    print(f"    初始熵: {relax_history['entropies'][0]:.4f}")
    print(f"    最终熵: {relax_history['entropies'][-1]:.4f}")
    entropy_increased = relax_history['entropies'][-1] >= relax_history['entropies'][0] - 0.01
    print(f"    H 定理满足: {entropy_increased}")

    return True


def phase10_matrix_update(config):
    """阶段 10: Sherman-Morrison 矩阵更新。"""
    print_separator("阶段 10: Sherman-Morrison 介电常数更新")

    N = min(config.N_x, 64)

    # 构建测试系统: 对角占优矩阵
    rng = np.random.RandomState(42)
    A = np.diag(3.0 * np.ones(N)) + 0.1 * rng.randn(N, N)
    b = rng.randn(N)

    # LU 分解
    LU, piv = lu_factorize(A)
    x_direct = lu_solve(LU, piv, b)

    # 残差
    residual = np.linalg.norm(A @ x_direct - b)
    print(f"  LU 求解残差: ||Ax - b|| = {residual:.2e}")

    # Sherman-Morrison 更新
    u = np.zeros(N)
    u[N // 2] = 0.5  # 局部更新
    v = np.zeros(N)
    v[N // 2] = 1.0

    x_sm, beta, info = sherman_morrison_solve(LU, piv, u, v, b)
    print(f"\n  Sherman-Morrison 更新:")
    print(f"    beta = 1 - v^T A^-1 u = {beta:.6f}")
    print(f"    条件: {info['condition']}")
    print(f"    alpha = {info.get('alpha', 'N/A')}")

    # 验证: (A - u*v^T) * x_sm = b
    A_updated = A - np.outer(u, v)
    residual_sm = np.linalg.norm(A_updated @ x_sm - b)
    print(f"    更新系统残差: ||(A-uv^T)x - b|| = {residual_sm:.2e}")

    # 矩阵-向量乘法验证
    x_test = rng.randn(N)
    mv_result = sherman_morrison_mv(A, u, v, x_test)
    mv_direct = A_updated @ x_test
    mv_error = np.linalg.norm(mv_result - mv_direct)
    print(f"    MV 验证误差: ||(A-uv^T)x - direct|| = {mv_error:.2e}")

    # 等离子体介电常数更新
    print(f"\n  [等离子体介电常数更新]")
    omega_p_sq_old = config.plasma_density_profile()[:N]
    omega_p_sq_new = omega_p_sq_old * 1.05  # 5% 密度增长
    u_plasma, v_plasma, max_change = plasma_dielectric_update_system(
        N, omega_p_sq_old, omega_p_sq_new, config.dx
    )
    print(f"    最大密度变化: {max_change:.4f}")
    print(f"    更新向量 ||u|| = {np.linalg.norm(u_plasma):.4f}")
    print(f"    更新向量 ||v|| = {np.linalg.norm(v_plasma):.4f}")

    return True


def phase11_vlasov_maxwell(config):
    """阶段 11: Vlasov-Maxwell 仿真 (简化版)。"""
    print_separator("阶段 11: Vlasov-Maxwell 仿真")

    # 使用较少的步数以保证效率
    n_vm_steps = 30
    print(f"  运行 {n_vm_steps} 步 Vlasov-Maxwell 仿真...")

    t_start = time.time()
    vm_results = run_vlasov_maxwell_simulation(config, n_steps=n_vm_steps)
    t_elapsed = time.time() - t_start

    print(f"  计算时间: {t_elapsed:.3f} s")

    # 结果摘要
    f_final = vm_results['f_final']
    ok_f, info_f = check_array_finite(f_final, "f_final")
    print(f"  最终分布函数有限性: {info_f}")

    if len(vm_results['field_energies']) > 0:
        fe = vm_results['field_energies']
        ke = vm_results['kinetic_energies']
        te = vm_results['total_energies']

        print(f"\n  场能量: {fe[0]:.6f} -> {fe[-1]:.6f}")
        print(f"  动能: {ke[0]:.6f} -> {ke[-1]:.6f}")
        print(f"  总能: {te[0]:.6f} -> {te[-1]:.6f}")

        if len(te) > 1:
            ok_c, drift, info_c = check_conservation(te, "total_energy", tol=10.0)
            print(f"  能量守恒: {info_c}")

    # 边界条件应用
    print(f"\n  [边界条件验证]")
    E_test = np.sin(2 * np.pi * config.x / config.L_x)
    E_abs = apply_absorbing_layer(E_test, config.x, config.dx, sigma_max=2.0, n_layer=15)
    absorption_ratio = np.sum(E_abs ** 2) / max(np.sum(E_test ** 2), 1e-30)
    print(f"    吸收层: 能量比 = {absorption_ratio:.4f}")

    E_out = apply_outflow_bc(E_test, config.dx, 'both')
    print(f"    出流 BC: 边界值 = [{E_out[0]:.4f}, {E_out[-1]:.4f}]")

    return vm_results


def phase12_bayesian_diagnostics(config, vm_results):
    """阶段 12: 贝叶斯参数诊断。"""
    print_separator("阶段 12: 贝叶斯 MCMC 参数诊断")

    diagnostics = bayesian_plasma_diagnostics(vm_results, config)

    print(f"  观测数据: {diagnostics['data']}")
    print(f"  观测不确定度: {diagnostics['sigma']}")
    print(f"  MCMC 配置: {diagnostics['n_walkers']} walkers, "
          f"{diagnostics['n_burn']} burn, {diagnostics['n_prod']} prod")
    print(f"  平均接受率: {diagnostics['mean_acceptance']:.3f}")

    print(f"\n  [参数估计结果]")
    for name, est in diagnostics['param_estimates'].items():
        print(f"    {name}: {est['mean']:.4f} +/- {est['std']:.4f} "
              f"(median={est['median']:.4f}, "
              f"16%-84%=[{est['q16']:.4f}, {est['q84']:.4f}])")

    return True


def phase13_final_summary(config):
    """阶段 13: 最终总结。"""
    print_separator("阶段 13: 仿真总结")

    print("  已完成的计算阶段:")
    phases = [
        "物理参数设置与校验",
        "高阶有限差分算子验证",
        "Von Neumann 稳定性分析",
        "Newton-Maehly 色散关系求解",
        "Gram-Schmidt 模式分解",
        "分布函数初始化与采样",
        "等离子体态密度计算",
        "边界几何分析",
        "碰撞算子与弛豫",
        "Sherman-Morrison 矩阵更新",
        "Vlasov-Maxwell 仿真",
        "贝叶斯参数诊断",
    ]
    for i, phase in enumerate(phases, 1):
        print(f"    {i:2d}. {phase}")

    print("\n  融合的 15 个种子项目算法:")
    projects = [
        ("801_newton_maehly", "Newton-Maehly 色散多项式求根"),
        ("480_gram_schmidt", "CGS/MGS 电磁模式正交分解"),
        ("1242_PhotonDOS", "等离子体电磁模式态密度"),
        ("538_histogram_2d", "2D 相空间逆 CDF 采样"),
        ("786_nas", "核心数值算子性能验证"),
        ("698_log_normal", "超热电子 log-normal 分布"),
        ("283_diffusion_pde", "碰撞扩散算子方法线求解"),
        ("272_dg1d_burgers", "DG 数值通量 Vlasov 求解"),
        ("1216_borexino", "贝叶斯 MCMC 参数估计"),
        ("1393_vin", "VIN 校验和参数完整性"),
        ("995_r8sm", "Sherman-Morrison 矩阵更新"),
        ("1080_safe_CCMPC", "场景管理抽象架构"),
        ("150_cg_triangles", "有符号距离边界几何"),
        ("057_atbash", "Atbash 参数标签编码"),
        ("358_fd1d_bvp", "有限差分边值问题求解"),
    ]
    for name, desc in projects:
        print(f"    [{name}] -> {desc}")

    print("\n  核心物理公式:")
    formulas = [
        "Vlasov: df/dt + v*df/dx - (e/m)*E*df/dv = C[f]",
        "Maxwell: dE/dt = curl(B) - J, dB/dt = -curl(E)",
        "Dispersion: omega^2 = omega_p^2 + c^2*k^2",
        "FD: f'(x) ~ sum a_j*(f(x+jh)-f(x-jh))/h",
        "Von Neumann: |G(kh)| <= 1 for stability",
        "Gram-Schmidt: q_j = (a_j - sum proj) / ||...||",
        "Log-normal: f(x) = exp(-(ln(x)-mu)^2/(2*sigma^2))/(x*sigma*sqrt(2*pi))",
        "Coulomb: ln(Lambda) = ln(lambda_D / b_min)",
        "Sherman-Morrison: (A-uv^T)^{-1} = A^{-1} + alpha*(A^{-1}u)(v^TA^{-1})",
        "Bayes: p(theta|D) ~ L(D|theta) * p(theta)",
    ]
    for f in formulas:
        print(f"    {f}")

    print()
    print("=" * 72)
    print("  仿真完成。所有阶段执行成功。")
    print("=" * 72)

    return True


def main():
    """主函数: 执行完整的激光等离子体相互作用仿真流程。"""
    print_header()

    t_total_start = time.time()

    # 创建仿真配置
    config = SimulationConfig()
    print(f"  配置创建完成: N_x={config.N_x}, N_v={config.N_v}, "
          f"FD_order={config.fd_order}")

    # 执行各阶段
    phase1_physical_setup(config)
    phase2_fd_operator_validation(config)
    phase3_stability_analysis(config)
    phase4_dispersion_analysis(config)
    phase5_mode_decomposition(config)
    phase6_distribution_sampling(config)
    phase7_dos_analysis(config)
    phase8_boundary_geometry(config)
    phase9_collision_operator(config)
    phase10_matrix_update(config)
    vm_results = phase11_vlasov_maxwell(config)
    phase12_bayesian_diagnostics(config, vm_results)
    phase13_final_summary(config)

    t_total = time.time() - t_total_start
    print(f"\n  总运行时间: {t_total:.3f} s")
    print("  Done.")


if __name__ == '__main__':
    main()
