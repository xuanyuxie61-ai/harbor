# -*- coding: utf-8 -*-
"""
main.py
=======

磁层粒子输运模拟: 高阶有限差分与稳定性分析
============================================

统一入口: 零参数运行完整的博士级计算实验.

科学问题:
  模拟地球辐射带相对论电子在 (L, E) 相空间中的输运演化,
  求解 2D Fokker-Planck 方程:
    df/dt = (1/G) d/dL (G D_LL df/dL) - d/dE (dot{E} f) + S - f/tau

  使用高阶有限差分 (4阶中心差分 / WENO5 / 紧致格式) 进行空间离散,
  并进行严格的 von Neumann 稳定性分析.

物理过程:
  1. 径向扩散 (ULF 波驱动)
  2. 能量扩散/损失 (波-粒子散射, 同步辐射)
  3. 大气损失 (损失锥)
  4. 磁暴注入 (亚暴源)
  5. 磁顶阴影 (外边界损失)

数值方法:
  - 空间离散: 4阶中心差分 (默认), WENO5 (激波捕获), 紧致格式
  - 时间推进: 显式 Euler (CFL 限制)
  - 边界条件: 损失锥 + 磁顶阴影 + CBF 约束
  - 稳定性: von Neumann 分析 + 矩阵谱分析
  - 随机性: 多项式混沌展开 (PCE) + Ziggurat 采样
  - 磁场混沌: Chirikov 标准映射

项目结构 (14 个 .py 文件):
  1. physical_constants.py   - 物理常数与磁层参数
  2. magnetosphere_grid.py   - (L, E) 相空间网格
  3. high_order_fd.py        - 高阶有限差分格式
  4. boundary_conditions.py  - 物理边界条件
  5. stochastic_sampler.py   - 随机采样 (Ziggurat, PCE, Halton)
  6. stability_analysis.py   - 稳定性分析 (von Neumann, 矩阵)
  7. vlasov_solver.py        - Fokker-Planck 求解器
  8. field_solver.py         - 磁场求解 (偶极 + Chirikov)
  9. phase_space_diagnostics.py - 相空间诊断 (互相关, Lyness积分)
  10. combinatorial_modes.py - 波模式与 L 壳层网络
  11. prime_grid.py          - 素数共振网格
  12. partial_digest_resonance.py - Partial Digest 波模式识别
  13. data_io.py             - 数据 I/O (XYZ, 校验和)
  14. test_particle_orbit.py - 测试粒子轨道积分

种子项目映射 (15 个):
  [1233] DAS 互相关      -> phase_space_diagnostics.py (通量互相关)
  [291]  离散PDF采样      -> stochastic_sampler.py (离散CDF采样)
  [1311] Lyness三角形积分 -> phase_space_diagnostics.py (相空间积分)
  [1260] 安全控制CBF      -> boundary_conditions.py (控制障碍约束)
  [1424] XYZ文件I/O       -> data_io.py (点云数据读写)
  [1433] Ziggurat采样     -> stochastic_sampler.py (正态/指数采样)
  [480]  Gram-Schmidt     -> stability_analysis.py (特征模式正交化)
  [1189] 累积优势网络     -> combinatorial_modes.py (L壳层耦合)
  [071]  银行校验和       -> data_io.py (数据完整性验证)
  [842]  ODE系统(臭氧)    -> vlasov_solver.py (能量演化ODE)
  [853]  Legendre PCE     -> stochastic_sampler.py (随机场展开)
  [848]  Partial Digest    -> partial_digest_resonance.py (波模式识别)
  [136]  组合计数         -> phase_space_diagnostics.py (相空间计数)
  [910]  素数筛           -> prime_grid.py (共振L壳层选择)
  [171]  Chirikov映射     -> field_solver.py (磁力线混沌)

运行方法:
  python main.py

输出:
  - 控制台: 完整的诊断报告
  - 文件: distribution_final.xyz, history.txt (在 output/ 目录)
"""

import sys
import os
import time as time_module
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_constants import (
    R_EARTH, B_EQUATORIAL, E_REST_MeV,
    L_MIN, L_MAX, L_SHELL_NUM,
    ENERGY_MIN_MeV, ENERGY_MAX_MeV, ENERGY_NUM,
    CFL_DEFAULT, PCE_ORDER,
)
from magnetosphere_grid import MagnetosphereGrid
from high_order_fd import (
    central_diff_2nd, central_diff_4th, central_diff2_2nd, central_diff2_4th,
    compact_first_derivative, convergence_test,
)
from boundary_conditions import BoundaryConditions
from stochastic_sampler import (
    ZigguratSampler, DiscreteCDFSampler, PolynomialChaosExpansion,
    halton_sequence, generate_stochastic_diffusion_coefficient,
)
from stability_analysis import (
    von_neumann_analysis_diffusion, von_neumann_analysis_advection,
    compute_cfl_timestep, build_diffusion_matrix,
    matrix_stability_analysis, modified_gram_schmidt,
    comprehensive_stability_check, print_stability_report,
)
from vlasov_solver import VlasovSolver
from field_solver import DipoleField, ChirikovMap, trace_field_line
from phase_space_diagnostics import (
    cross_correlation, phase_space_count, phase_space_moments,
    lyness_triangle_quadrature, integrate_over_triangle,
)
from combinatorial_modes import (
    cyclotron_resonance_condition, enumerate_resonance_modes,
    LShellCouplingNetwork,
)
from prime_grid import sieve_of_eratosthenes, PrimeResonanceGrid
from partial_digest_resonance import WaveModeIdentifier
from data_io import compute_checksum, write_xyz, save_grid_data, load_grid_data, export_simulation_results
from test_particle_orbit import TestParticle, GuidingCenterIntegrator


def print_header():
    """打印项目标题."""
    print("=" * 72)
    print("  磁层粒子输运模拟: 高阶有限差分与稳定性分析")
    print("  Magnetospheric Particle Transport Simulation")
    print("  High-Order Finite Difference & Stability Analysis")
    print("=" * 72)
    print()
    print("科学问题: 地球辐射带相对论电子 (L, E) 相空间输运")
    print("控制方程: 2D Fokker-Planck (漂移动力学) 方程")
    print("数值方法: 高阶有限差分 + von Neumann 稳定性分析")
    print("应用领域: 计算空间物理 / 空间天气预报")
    print()


def section_1_physical_constants():
    """第1部分: 物理常数与参数验证."""
    print("\n" + "=" * 72)
    print("第1部分: 物理常数与磁层参数")
    print("=" * 72)

    print(f"\n[基本常数]")
    print(f"  光速:          c = {2.998e8:.3e} m/s")
    print(f"  电子质量:      m_e = {9.109e-31:.3e} kg")
    print(f"  电子电荷:      e = {1.602e-19:.3e} C")
    print(f"  静止能量:      m_e*c^2 = {E_REST_MeV:.4f} MeV")

    print(f"\n[地球偶极场]")
    print(f"  地球半径:      R_E = {R_EARTH:.3e} m")
    print(f"  赤道磁场:      B_eq = {B_EQUATORIAL:.4e} T = {B_EQUATORIAL*1e9:.1f} nT")

    dipole = DipoleField()
    B_L4 = dipole.field_magnitude(4.0 * R_EARTH, np.pi/2)
    print(f"  L=4 赤道场:    B(4) = {B_L4:.4e} T")
    print(f"  L=6.6 赤道场:  B(6.6) = {dipole.field_magnitude(6.6*R_EARTH, np.pi/2):.4e} T")

    print(f"\n[相空间范围]")
    print(f"  L 范围: [{L_MIN}, {L_MAX}], n_L = {L_SHELL_NUM}")
    print(f"  E 范围: [{ENERGY_MIN_MeV}, {ENERGY_MAX_MeV}] MeV, n_E = {ENERGY_NUM}")

    return dipole


def section_2_grid_construction(dipole):
    """第2部分: 相空间网格构造."""
    print("\n" + "=" * 72)
    print("第2部分: (L, E) 相空间网格构造")
    print("=" * 72)

    # 多种网格类型
    for grid_type in ['uniform', 'log', 'gauss_lobatto']:
        grid = MagnetosphereGrid(grid_type=grid_type)
        info = grid.check_grid_quality()
        print(f"\n  [{grid_type}] n_total = {info['n_total']}, "
              f"L_spacing = [{info['L_min_spacing']:.4f}, {info['L_max_spacing']:.4f}], "
              f"aspect_ratio = {info['aspect_ratio_max']:.2f}")

    # 使用均匀网格作为默认
    grid = MagnetosphereGrid(grid_type='uniform')
    grid.summary()

    # 扩散系数
    D_LL = grid.radial_diffusion_coefficient(Kp=4)
    print(f"\n  径向扩散系数 D_LL (Kp=4):")
    print(f"    L=2: {D_LL[0]:.4e}")
    print(f"    L=4: {D_LL[grid.n_L//2]:.4e}")
    print(f"    L=6: {D_LL[-1]:.4e}")

    return grid, D_LL


def section_3_high_order_fd():
    """第3部分: 高阶有限差分收敛性测试."""
    print("\n" + "=" * 72)
    print("第3部分: 高阶有限差分格式验证")
    print("=" * 72)

    # 收敛性测试
    h_list, errors_2, errors_4 = convergence_test()

    # 紧致格式测试
    print("\n--- 紧致差分测试 ---")
    N = 80
    h = 2*np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    f = np.sin(x)
    df_exact = np.cos(x)

    df_central2 = central_diff_2nd(f, h)
    df_central4 = central_diff_4th(f, h)
    df_compact = compact_first_derivative(f, h)

    err_c2 = np.max(np.abs(df_central2 - df_exact))
    err_c4 = np.max(np.abs(df_central4 - df_exact))
    err_compact = np.max(np.abs(df_compact - df_exact))

    print(f"  N={N}: 2阶误差 = {err_c2:.4e}, 4阶误差 = {err_c4:.4e}, 紧致误差 = {err_compact:.4e}")

    return {'err_c2': err_c2, 'err_c4': err_c4, 'err_compact': err_compact}


def section_4_stability_analysis(grid, D_LL):
    """第4部分: 稳定性分析."""
    print("\n" + "=" * 72)
    print("第4部分: 稳定性分析 (von Neumann + 矩阵)")
    print("=" * 72)

    # von Neumann 分析
    print("\n--- von Neumann 扩散分析 ---")
    for scheme in ['central_2nd', 'central_4th']:
        vn = von_neumann_analysis_diffusion(scheme)
        print(f"  {scheme}: r_max = {vn['r_max']:.4f}")

    # CFL 时间步长
    dt_cfl, dt_diff, dt_adv = compute_cfl_timestep(grid, D_LL)
    print(f"\n--- CFL 条件 ---")
    print(f"  dt_cfl = {dt_cfl:.4e} s ({dt_cfl/3600:.2f} hours)")
    print(f"  dt_diff = {dt_diff:.4e} s")

    # 矩阵稳定性
    print("\n--- 矩阵稳定性 ---")
    A = build_diffusion_matrix(grid, D_LL)
    stab = matrix_stability_analysis(A, dt=dt_cfl * 0.5)
    print(f"  谱半径: {stab['spectral_radius']:.4e}")
    print(f"  最大实部: {stab['max_real_part']:.4e}")
    print(f"  dt_max (Euler): {stab['dt_max_euler']:.4e}")
    print(f"  稳定: {stab.get('stable', True)}")

    # Gram-Schmidt 测试
    print("\n--- Gram-Schmidt 正交化 ---")
    rng = np.random.default_rng(42)
    V = rng.standard_normal((20, 5))
    Q, R, rank = modified_gram_schmidt(V)
    orth_err = np.max(np.abs(Q.T @ Q - np.eye(5)))
    print(f"  秩: {rank}, 正交性误差: {orth_err:.4e}")

    # 综合稳定性检查
    report = comprehensive_stability_check(grid, D_LL, dt_cfl * 0.5)
    print(f"\n  综合稳定性: {'PASS' if report['overall_stable'] else 'FAIL'}")

    return report


def section_5_stochastic_methods(grid):
    """第5部分: 随机方法验证."""
    print("\n" + "=" * 72)
    print("第5部分: 随机方法 (Ziggurat, PCE, Halton)")
    print("=" * 72)

    # Ziggurat 采样
    print("\n--- Ziggurat 采样 ---")
    zs = ZigguratSampler(seed=42)
    samples = zs.sample_normal(10000)
    print(f"  正态采样: mean = {np.mean(samples):.4f}, std = {np.std(samples):.4f}")

    # 离散 CDF 采样
    print("\n--- 离散 CDF 采样 ---")
    energies = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    probs = np.array([0.1, 0.2, 0.35, 0.25, 0.1])
    sampler = DiscreteCDFSampler(energies, probs, seed=42)
    samples = sampler.sample(10000)
    print(f"  采样均值: {np.mean(samples):.4f} MeV (期望: {np.sum(energies*probs):.4f})")

    # PCE
    print("\n--- 多项式混沌展开 ---")
    pce = PolynomialChaosExpansion(order=PCE_ORDER)
    def test_func(xi):
        return np.exp(0.5 * xi)
    coeffs = pce.project(test_func)
    stats = pce.compute_statistics(coeffs)
    print(f"  PCE 系数 (exp(0.5*xi)): {coeffs}")
    print(f"  PCE 均值: {stats['mean']:.4f}")
    print(f"  PCE 方差: {stats['variance']:.4e}")

    # 随机扩散系数
    D_stoch, pce_coeffs = generate_stochastic_diffusion_coefficient(grid, sigma=0.2, seed=42)
    print(f"\n--- 随机扩散系数 ---")
    print(f"  平均 D_LL (L=4): {D_stoch[grid.n_L//2]:.4e}")
    print(f"  PCE 系数数量: {pce_coeffs.shape[1]}")

    # Halton 序列
    print("\n--- Halton 准随机序列 ---")
    points = halton_sequence(10, dim=2)
    print(f"  前 3 点 (2D): {points[:3]}")

    return {'pce_stats': stats}


def section_6_field_and_chirikov(dipole):
    """第6部分: 磁场与 Chirikov 混沌."""
    print("\n" + "=" * 72)
    print("第6部分: 磁场建模与 Chirikov 混沌映射")
    print("=" * 72)

    # 偶极场
    print("\n--- 偶极磁场 ---")
    for L in [2, 4, 6.6]:
        B = dipole.field_magnitude(L * R_EARTH, np.pi/2)
        print(f"  L={L}: B = {B:.4e} T = {B*1e9:.1f} nT")

    # 磁场线追踪
    print("\n--- 磁场线追踪 ---")
    r0 = 4 * R_EARTH
    theta0 = np.pi / 4
    r, theta, phi = trace_field_line(dipole, r0, theta0, 0.0, n_steps=100)
    print(f"  起点: r = {r0/R_EARTH:.1f} R_E, theta = {np.degrees(theta0):.1f} deg")
    print(f"  终点: r = {r[-1]/R_EARTH:.1f} R_E, theta = {np.degrees(theta[-1]):.1f} deg")

    # Chirikov 标准映射
    print("\n--- Chirikov 标准映射 ---")
    for K in [0.5, 0.97, 1.5, 3.0]:
        cmap = ChirikovMap(K=K)
        lyap = cmap.lyapunov_exponent(1.0, 1.0, n_steps=300)
        regime = "规则" if K < 0.97 else ("临界" if K < 1.5 else "混沌")
        print(f"  K = {K:.2f}: Lyapunov = {lyap:.4f} ({regime})")

    # 扩散系数 vs K
    K_vals, D_vals = ChirikovMap().diffusion_coefficient(
        K_range=np.array([0.5, 1.0, 2.0, 3.0]),
        n_samples=10, n_steps=100
    )
    print(f"\n  扩散系数 vs K:")
    for K, D in zip(K_vals, D_vals):
        print(f"    K={K:.1f}: D = {D:.4e}")

    return K_vals, D_vals


def section_7_wave_modes_and_resonance():
    """第7部分: 波模式识别与共振."""
    print("\n" + "=" * 72)
    print("第7部分: 波模式识别与共振分析")
    print("=" * 72)

    # 共振条件
    print("\n--- 回旋共振条件 ---")
    B_L4 = B_EQUATORIAL / 4.0**3
    for n_harm in [-1, 0, 1]:
        k_par, omega_res = cyclotron_resonance_condition(n_harm, B_L4, 1.0)
        print(f"  n={n_harm:+d}: k_par = {k_par:.4e} /m, omega = {omega_res:.4e} rad/s")

    # 模式枚举
    modes = enumerate_resonance_modes(n_max=2)
    print(f"\n  共振模式总数: {len(modes)}")

    # L 壳层耦合网络
    print("\n--- L 壳层耦合网络 ---")
    L_vals = np.linspace(2, 7, 11)
    net = LShellCouplingNetwork(L_vals, alpha=1.0)
    history = net.evolve_cumulative_advantage(n_steps=30)
    diag = net.network_diagnostics()
    print(f"  节点: {diag['n_nodes']}, 边: {diag['n_edges']}")
    print(f"  聚类系数: {diag['clustering_coefficient']:.4f}")
    print(f"  网络密度: {diag['density']:.4f}")

    # 素数共振网格
    print("\n--- 素数共振 L 壳层 ---")
    prime_grid = PrimeResonanceGrid(L_min=2, L_max=8, n_base=31)
    prime_grid.summary()
    weights = prime_grid.get_resonance_weights()
    print(f"  权重范围: [{np.min(weights):.1f}, {np.max(weights):.1f}]")

    # Partial Digest 波模式识别
    print("\n--- Partial Digest 波模式识别 ---")
    identifier = WaveModeIdentifier()
    freq, spec = identifier.generate_synthetic_spectrum()
    identified = identifier.identify_modes(freq, spec)
    print(f"  识别的模式: {[m['mode'] for m in identified]}")
    for m in identified:
        print(f"    {m['mode']}: f = {m['peak']:.3f} Hz, confidence = {m['confidence']:.3f}")

    return diag


def section_8_solver_simulation(grid, D_LL):
    """第8部分: Fokker-Planck 求解与模拟."""
    print("\n" + "=" * 72)
    print("第8部分: Fokker-Planck 求解器模拟")
    print("=" * 72)

    # 创建求解器
    bc = BoundaryConditions(
        L_min=grid.L_min, L_max=grid.L_max, n_L=grid.n_L,
        inner_type='loss_cone', outer_type='magnetopause'
    )
    bc.summary()

    solver = VlasovSolver(grid=grid, bc=bc, Kp=4,
                          fd_order=4, fd_scheme='central')

    print(f"\n  初始状态:")
    print(f"    max f = {np.max(solver.f):.4e}")
    print(f"    min f = {np.min(solver.f):.4e}")
    print(f"    total = {np.sum(solver.f) * grid.dL * grid.dE_avg:.4e}")
    print(f"    dt = {solver.dt:.4e} s")

    # 运行模拟
    print(f"\n--- 运行 Fokker-Planck 模拟 ---")
    total_time = 50 * solver.dt  # 50 步
    results = solver.run(total_time=total_time, output_interval=10*solver.dt)

    # 结果分析
    f_final = results['f_final']
    print(f"\n  最终状态:")
    print(f"    max f = {np.max(f_final):.4e}")
    print(f"    min f = {np.min(f_final):.4e}")
    print(f"    total = {np.sum(f_final) * grid.dL * grid.dE_avg:.4e}")

    # 矩分析
    moments = phase_space_moments(f_final, grid.L, grid.E_MeV)
    print(f"\n  最终矩:")
    print(f"    总粒子数: {moments['total_particles']:.4e}")
    print(f"    峰值 L: {grid.L[np.argmax(moments['density'])]:.2f}")
    print(f"    平均能量: {np.mean(moments['mean_energy']):.4f} MeV")

    return solver, results, moments


def section_9_test_particle():
    """第9部分: 测试粒子轨道."""
    print("\n" + "=" * 72)
    print("第9部分: 测试粒子轨道积分")
    print("=" * 72)

    # 创建粒子
    particle = TestParticle.from_energy_pitch_angle(
        L=4.0, E_MeV=1.0, alpha_eq=np.pi/4
    )
    print(f"\n  初始条件:")
    print(f"    r = {np.linalg.norm(particle.position)/R_EARTH:.2f} R_E")
    print(f"    E = {particle.E_kin_MeV:.3f} MeV")
    print(f"    gamma = {particle.gamma:.4f}")

    # 积分
    integrator = GuidingCenterIntegrator(dt=0.1)
    orbit = integrator.integrate_orbit(particle, n_steps=100, record_interval=10)

    print(f"\n  轨道积分结果:")
    print(f"    轨道点数: {len(orbit['positions'])}")
    print(f"    时间跨度: {orbit['time'][-1]:.2f} s")
    print(f"    初始 mu = {orbit['mu'][0]:.4e} J/T")
    print(f"    最终 mu = {orbit['mu'][-1]:.4e} J/T")
    mu_rel_change = abs(orbit['mu'][-1] - orbit['mu'][0]) / max(abs(orbit['mu'][0]), 1e-30)
    print(f"    mu 相对变化: {mu_rel_change:.4e}")

    return orbit


def section_10_data_io(results, grid):
    """第10部分: 数据输出."""
    print("\n" + "=" * 72)
    print("第10部分: 数据 I/O 与校验")
    print("=" * 72)

    # 校验和
    f_final = results['f_final']
    checksum = compute_checksum(f_final)
    print(f"\n  最终分布校验和: {checksum}")

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 导出结果
    export_simulation_results(results, grid, output_dir=output_dir)

    # 验证
    bin_file = os.path.join(output_dir, 'distribution_final.bin')
    if os.path.exists(bin_file):
        L, E, f_loaded, valid = load_grid_data(bin_file)
        print(f"\n  二进制文件验证: {'PASS' if valid else 'FAIL'}")
        print(f"  数据形状: {f_loaded.shape}")

    return output_dir


def section_11_triangle_quadrature():
    """第11部分: Lyness 三角形积分."""
    print("\n" + "=" * 72)
    print("第11部分: Lyness 三角形积分规则")
    print("=" * 72)

    # 测试函数
    test_cases = [
        (lambda x, y: 1.0, 0.5, "f=1"),
        (lambda x, y: x, 1.0/6.0, "f=x"),
        (lambda x, y: x**2 + y**2, 1.0/6.0, "f=x^2+y^2"),
        (lambda x, y: x*y, 1.0/24.0, "f=x*y"),
    ]

    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    for func, exact, name in test_cases:
        for order in [1, 3, 4, 5]:
            computed = integrate_over_triangle(func, vertices, order=order)
            err = abs(computed - exact)
            print(f"  {name}, order={order}: {computed:.6e} (exact: {exact:.6e}, err: {err:.4e})")

    return True


def print_footer(start_time):
    """打印页脚."""
    elapsed = time_module.time() - start_time
    print("\n" + "=" * 72)
    print("模拟完成!")
    print("=" * 72)
    print(f"  运行时间: {elapsed:.2f} 秒")
    print(f"  总文件数: 14 .py 文件")
    print(f"  种子项目: 15 个全部融入")
    print(f"  应用领域: 计算空间物理 / 磁层粒子输运")
    print()
    print("核心物理:")
    print("  - 2D Fokker-Planck 方程 (L, E) 相空间")
    print("  - 高阶有限差分 (2/4/6阶中心 + WENO5 + 紧致)")
    print("  - von Neumann 稳定性 + CFL 条件")
    print("  - 多项式混沌展开 (不确定性量化)")
    print("  - Chirikov 混沌映射 (磁场随机性)")
    print("  - 素数共振网格 + Partial Digest 波模式识别")
    print()


def main():
    """主函数: 运行完整实验."""
    start_time = time_module.time()
    print_header()

    # 1. 物理常数
    dipole = section_1_physical_constants()

    # 2. 网格构造
    grid, D_LL = section_2_grid_construction(dipole)

    # 3. 高阶有限差分
    fd_results = section_3_high_order_fd()

    # 4. 稳定性分析
    stability = section_4_stability_analysis(grid, D_LL)

    # 5. 随机方法
    stoch_results = section_5_stochastic_methods(grid)

    # 6. 磁场与混沌
    chirikov_results = section_6_field_and_chirikov(dipole)

    # 7. 波模式与共振
    wave_results = section_7_wave_modes_and_resonance()

    # 8. 主模拟
    solver, sim_results, moments = section_8_solver_simulation(grid, D_LL)

    # 9. 测试粒子
    orbit = section_9_test_particle()

    # 10. 数据 I/O
    output_dir = section_10_data_io(sim_results, grid)

    # 11. 三角形积分
    section_11_triangle_quadrature()

    # 页脚
    print_footer(start_time)

    return 0


if __name__ == "__main__":
    sys.exit(main())
