#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 强关联 Hubbard 模型 DQMC 模拟统一入口
==================================================

计算凝聚态: 强关联 Hubbard 模型量子蒙特卡洛
高阶有限差分与稳定性分析 (小规模可复现实验)

本项目融合 15 个种子项目的核心算法:
  [01] 1360_truncated_normal    → 截断正态 HS 辅助场采样
  [02] 1408_wedge_grid          → 三维楔形体布里渊区网格
  [03] 660_legendre_fast_rule   → Gauss-Legendre 虚时积分
  [04] 1014_AAR_disequilibrium  → 热化非平衡诊断
  [05] 1320_triangle_to_fem     → 三角晶格 FEM 映射
  [06] 764_midpoint             → 隐式中点流方程积分器
  [07] 362_fd1d_heat_steady     → 稳态 FD Dyson 方程求解
  [08] 293_disk_grid            → 圆盘费米面采样
  [09] 1235_OptimismPerseveration → 自适应提议分布
  [10] 123_burgers_pde_etdrk4   → ETD 矩阵指数演化算子
  [11] 1352_triangulation_svg   → 晶格拓扑分析
  [12] 798_nested_sequence_display → 嵌套虚时网格
  [13] 218_coordinate_search    → 参数坐标搜索优化
  [14] 301_disk01_monte_carlo   → 圆盘蒙特卡洛积分
  [15] 796_neighbors_to_metis_graph → METIS 图分割

运行:
    python main.py

零参数即可运行完整 DQMC 流水线.
"""

import sys
import os
import numpy as np

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lattice_geometry import (
    triangular_lattice_vectors, reciprocal_vectors,
    build_triangular_cluster, build_kinetic_matrix,
    disk_grid_fibonacci, build_bz_disk_grid,
    wedge_grid_3d, wedge_grid_size,
    tight_binding_dispersion, density_of_states_histogram,
    chemical_potential_half_filling,
    adjacency_to_metis_format, compute_graph_diameter,
    first_brizouin_corner_points,
)
from finite_difference import (
    fd_coefficients_first_derivative,
    fd_coefficients_second_derivative,
    apply_first_derivative, apply_second_derivative,
    build_fd_derivative_matrix,
    solve_steady_state_diffusion_1d,
    midpoint_implicit_integrator,
    richardson_extrapolation,
)
from hubbard_stratonovich import (
    normal_01_pdf, normal_01_cdf,
    truncated_normal_sample, truncated_normal_pdf, truncated_normal_moment,
    compute_hs_alpha, compute_hs_coupling,
    disk01_sample, disk01_area_monte_carlo, disk01_monomial_integral,
    monte_carlo_bz_integral,
)
from greens_function import (
    build_b_matrix, compute_equal_time_green_function,
    compute_g_of_tau, matsubara_frequencies,
    fourier_transform_g_tau,
    gauss_legendre_rule, integrate_gtau_with_gl,
    spectral_function_from_g_iw, dyson_self_energy,
)
from determinant_qmc import DQMCSimulator, NestedImaginaryTimeGrid
from stability_analysis import (
    analyze_stability, check_green_function_idempotency,
    check_particle_number, check_hermiticity,
    cfl_condition_check, von_neumann_stability,
    matrix_exp_error_analysis,
)
from parameter_optimizer import (
    coordinate_search, find_mott_uc,
    optimize_cluster_size, scan_U_values,
)
from thermalization_diagnostics import (
    autocorrelation_function, integrated_autocorrelation_time,
    compute_aar_disequilibrium, gelman_rubin_statistic,
    blocking_analysis, comprehensive_thermalization_check,
)
from observable_estimator import (
    estimate_kinetic_energy, estimate_potential_energy,
    estimate_total_energy, estimate_double_occupancy,
    spin_structure_factor, pairing_correlation,
    high_symmetry_path_kpoints,
    compute_dos_by_kernel_polishing,
)


# ==========================================================================
#  分隔符与输出工具
# ==========================================================================

def banner(title: str, char: str = '=', width: int = 72):
    """打印带分隔的标题."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def sub_banner(title: str, char: str = '-', width: int = 72):
    print()
    print(f"  >> {title}")
    print(f"  {char * (width - 4)}")


# ==========================================================================
#  第一阶段: 晶格与布里渊区构造
# ==========================================================================

def phase1_lattice_construction():
    """阶段 1: 构造三角晶格、布里渊区网格."""
    banner("阶段 1: 晶格几何与布里渊区构造", '=')
    print("  融合种子项目: [02] wedge_grid, [05] triangle_to_fem,")
    print("                [08] disk_grid, [11] triangulation, [15] metis_graph")

    # 1.1 三角晶格
    sub_banner("1.1 三角晶格构造 (4x4 簇)")
    cluster = build_triangular_cluster(Lx=4, Ly=4, a=1.0, periodic=True)
    print(f"  簇尺寸: {cluster['Lx']} x {cluster['Ly']}")
    print(f"  总格点数: Ns = {cluster['Ns']}")
    print(f"  跳跃连接数: {len(cluster['hopping'])}")
    print(f"  配位数 (平均): {2 * len(cluster['hopping']) / cluster['Ns']:.1f}")

    T_mat = build_kinetic_matrix(cluster)
    eigenvalues = np.linalg.eigvalsh(T_mat)
    print(f"  动能矩阵带宽: W = [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    print(f"  带宽: ΔE = {eigenvalues.max() - eigenvalues.min():.4f}")

    # 1.2 图分割 (METIS 格式)
    sub_banner("1.2 晶格图分割 (METIS 格式)")
    xadj, adjncy = adjacency_to_metis_format(cluster['adjacency'])
    print(f"  METIS xadj: {xadj[:8]}... ({len(xadj)} entries)")
    print(f"  METIS adjncy: {len(adjncy)} edges (2E)")
    diameter = compute_graph_diameter(cluster['adjacency'])
    print(f"  图直径: {diameter}")

    # 1.3 布里渊区高对称点
    sub_banner("1.3 第一布里渊区高对称点")
    hsym = first_brizouin_corner_points(a=1.0)
    for name, kvec in hsym.items():
        print(f"  {name:6s} = ({kvec[0]:.4f}, {kvec[1]:.4f})")

    # 1.4 圆盘费米面采样
    sub_banner("1.4 二维圆盘布里渊区采样 (Fibonacci 网格)")
    n_k = 64
    k_pts, k_weights = build_bz_disk_grid(n_k, a=1.0)
    eps_k = tight_binding_dispersion(k_pts[:, 0], k_pts[:, 1])
    print(f"  k 点数: {n_k}")
    print(f"  ε(k) 范围: [{eps_k.min():.4f}, {eps_k.max():.4f}]")

    # 1.5 态密度
    sub_banner("1.5 态密度估计")
    n_k_dos = 400
    k_pts_full = disk_grid_fibonacci(n_k_dos, radius=4 * np.pi / 3.0)
    eps_full = tight_binding_dispersion(k_pts_full[:, 0], k_pts_full[:, 1])
    dos_centers, dos_values = density_of_states_histogram(eps_full, n_bins=80)
    print(f"  DOS 采样 k 点数: {n_k_dos}")
    print(f"  DOS 峰值位置: ε ≈ {dos_centers[np.argmax(dos_values)]:.4f}")

    # 1.6 化学势 (半填充)
    mu = chemical_potential_half_filling(
        temperature=0.1, dos_centers=dos_centers, dos_values=dos_values)
    print(f"  半填充化学势: μ ≈ {mu:.6f}")

    # 1.7 楔形体网格 (3D 布里渊区, 层状 Hubbard)
    sub_banner("1.7 三维楔形体布里渊区网格 (层状 Hubbard)")
    n_wedge = 4
    ng = wedge_grid_size(n_wedge)
    wedge_pts = wedge_grid_3d(n_wedge)
    print(f"  楔形细分: n = {n_wedge}")
    print(f"  3D 网格点数: {ng}")
    print(f"  示例点: ({wedge_pts[0, 0]:.3f}, {wedge_pts[0, 1]:.3f}, {wedge_pts[0, 2]:.3f})")

    return cluster, T_mat, mu, eigenvalues


# ==========================================================================
#  第二阶段: 有限差分与流方程
# ==========================================================================

def phase2_finite_difference():
    """阶段 2: 高阶有限差分模板与虚时流方程."""
    banner("阶段 2: 高阶有限差分与流方程", '=')
    print("  融合种子项目: [06] midpoint, [07] fd1d_heat_steady")

    # 2.1 有限差分模板
    sub_banner("2.1 高阶中心差分模板系数")
    for order in [2, 4, 6, 8]:
        idx, coeffs, hw = fd_coefficients_first_derivative(order)
        coeff_str = ", ".join(f"{c:+8.5f}" for c in coeffs)
        print(f"  {order}阶模板 (hw={hw}): [{coeff_str}]")
        # 验证: Σ c_k k = 1, Σ c_k k^n = 0 for n=0,2,...,p-1
        check_1 = sum(c * i for c, i in zip(coeffs, idx))
        print(f"    验证 Σ c_k k = {check_1:.10f} (应为 1)")

    # 2.2 FD 矩阵构造
    sub_banner("2.2 虚时微分算子矩阵")
    N_tau = 20
    dtau = 0.25
    D1 = build_fd_derivative_matrix(N_tau, dtau, order=4, derivative=1)
    D2 = build_fd_derivative_matrix(N_tau, dtau, order=4, derivative=2)
    print(f"  一阶导数矩阵 D1: {D1.shape}, ||D1||_F = {np.linalg.norm(D1, 'fro'):.4f}")
    print(f"  二阶导数矩阵 D2: {D2.shape}, ||D2||_F = {np.linalg.norm(D2, 'fro'):.4f}")

    # 2.3 测试: 对解析函数求导
    sub_banner("2.3 FD 精度验证 (f(τ) = sin(2πτ/β))")
    beta_test = 5.0
    tau_test = np.linspace(0, beta_test, 61)
    f_exact = np.sin(2 * np.pi * tau_test / beta_test)
    df_exact = (2 * np.pi / beta_test) * np.cos(2 * np.pi * tau_test / beta_test)

    for order in [2, 4, 6]:
        df_fd = apply_first_derivative(f_exact, tau_test[1] - tau_test[0],
                                       order=order, boundary='periodic')
        error = np.max(np.abs(df_fd - df_exact))
        print(f"  {order}阶 FD: max |error| = {error:.4e}")

    # 2.4 稳态扩散方程 (Dyson 方程类比)
    sub_banner("2.4 一维稳态扩散方程 (Dyson 实空间类比)")
    kappa_func = lambda x: 1.0 + 0.5 * np.sin(np.pi * x)
    source_func = lambda x: np.ones_like(x)
    x_sol, u_sol = solve_steady_state_diffusion_1d(
        N=50, a=0.0, b=1.0, u_left=0.0, u_right=0.0,
        kappa_func=kappa_func, source_func=source_func)
    print(f"  网格点数: {len(x_sol)}")
    print(f"  解的范围: u ∈ [{u_sol.min():.6f}, {u_sol.max():.6f}]")
    print(f"  最大值位置: x = {x_sol[np.argmax(u_sol)]:.4f}")

    # 2.5 隐式中点法积分流方程
    sub_banner("2.5 隐式中点法: 虚时流方程积分")
    # 简化的 Wegner 流方程: dH/dλ = -[H, [H, ω]]
    def flow_rhs(lam, H_flat):
        Ns = 4
        H = H_flat.reshape(Ns, Ns)
        # 简化: 趋向对角化
        omega = np.diag(np.arange(Ns, dtype=float))
        comm1 = H @ omega - omega @ H
        comm2 = comm1 @ H - H @ comm1
        return -comm2.flatten()

    H0 = np.array([[2.0, 0.5, 0.1, 0.0],
                   [0.5, 1.5, 0.3, 0.1],
                   [0.1, 0.3, 1.0, 0.2],
                   [0.0, 0.1, 0.2, 0.5]])
    t_span = (0.0, 1.0)
    t_flow, H_flow = midpoint_implicit_integrator(
        flow_rhs, t_span, H0.flatten(), n_steps=20)
    H_final = H_flow[-1].reshape(4, 4)
    off_diag = np.linalg.norm(H_final - np.diag(np.diag(H_final)))
    print(f"  流参数范围: λ ∈ [0, 1]")
    print(f"  初始非对角元: ||H_off|| = {np.linalg.norm(H0 - np.diag(np.diag(H0))):.6f}")
    print(f"  最终非对角元: ||H_off|| = {off_diag:.6f}")
    print(f"  对角化程度: {off_diag / max(np.linalg.norm(H0), 1e-15):.4e}")

    # 2.6 Richardson 外推
    sub_banner("2.6 Richardson 外推 (Δτ → 0 极限)")
    f_coarse = 1.0 + 0.01  # h 的 2 阶近似
    f_fine = 1.0 + 0.0025  # h/2 的 2 阶近似
    f_extrap = richardson_extrapolation(f_coarse, f_fine, order=2)
    print(f"  粗步长结果: {f_coarse:.6f}")
    print(f"  细步长结果: {f_fine:.6f}")
    print(f"  外推结果:   {f_extrap:.6f}")


# ==========================================================================
#  第三阶段: HS 变换与蒙特卡洛积分
# ==========================================================================

def phase3_hs_and_monte_carlo():
    """阶段 3: HS 变换、截断正态采样、蒙特卡洛积分."""
    banner("阶段 3: Hubbard-Stratonovich 变换与蒙特卡洛积分", '=')
    print("  融合种子项目: [01] truncated_normal, [14] disk01_monte_carlo")

    # 3.1 截断正态分布
    sub_banner("3.1 截断正态分布采样")
    a_cut, b_cut = -3.0, 3.0
    samples = truncated_normal_sample(a_cut, b_cut, size=10000,
                                      mean=0.0, std=1.0)
    print(f"  截断范围: [{a_cut}, {b_cut}]")
    print(f"  样本数: {len(samples)}")
    print(f"  样本均值: {np.mean(samples):.6f} (理论: 0)")
    print(f"  样本标准差: {np.std(samples):.6f} (理论: ~0.97)")
    print(f"  样本范围: [{samples.min():.4f}, {samples.max():.4f}]")

    # 3.2 截断正态矩
    sub_banner("3.2 截断正态分布各阶矩")
    for n in range(1, 7):
        moment = truncated_normal_moment(n, a_cut, b_cut)
        print(f"  E[X^{n}] = {moment:.8f}")

    # 3.3 HS 耦合常数
    sub_banner("3.3 HS 耦合常数 α(U, Δτ)")
    for U_val in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for dtau in [0.05, 0.1, 0.2]:
            alpha = compute_hs_alpha(dtau, U_val)
            approx = np.sqrt(dtau * U_val)
            print(f"  U={U_val:5.1f}, Δτ={dtau:.2f}: α={alpha:.6f} "
                  f"(√(ΔτU)={approx:.4f})")

    # 3.4 圆盘蒙特卡洛积分
    sub_banner("3.4 布里渊区圆盘蒙特卡洛积分")
    test_func = lambda k: np.cos(k[0]) * np.cos(k[1])
    estimate, se = monte_carlo_bz_integral(test_func, n_points=5000,
                                            radius=np.pi)
    print(f"  测试函数: f(k) = cos(kx) cos(ky)")
    print(f"  积分区域: 圆盘 R = π")
    print(f"  蒙特卡洛估计: {estimate:.6f} ± {se:.6f}")

    # 3.5 圆盘单项式积分
    sub_banner("3.5 圆盘单项式精确积分")
    for p, q in [(0, 0), (2, 0), (0, 2), (2, 2), (4, 0)]:
        val = disk01_monomial_integral(p, q, radius=1.0)
        print(f"  ∫∫ x^{p} y^{q} dA = {val:.8f}")

    return samples


# ==========================================================================
#  第四阶段: DQMC 模拟
# ==========================================================================

def phase4_dqmc_simulation(T_mat, mu, eigenvalues):
    """阶段 4: 运行 DQMC 模拟."""
    banner("阶段 4: 行列式量子蒙特卡洛 (DQMC) 模拟", '=')
    print("  融合种子项目: [09] OptimismPerseveration, [10] ETD RK4,")
    print("                [12] nested_sequence_display")

    # 4.1 物理参数
    U = 4.0
    beta = 4.0
    L = 20
    dtau = beta / L
    Ns = T_mat.shape[0]
    print(f"\n  物理参数:")
    print(f"    U/t = {U}")
    print(f"    β·t = {beta}")
    print(f"    Δτ·t = {dtau:.4f}")
    print(f"    Ns = {Ns}")
    print(f"    μ/t = {mu:.4f}")

    # 4.2 嵌套虚时网格
    sub_banner("4.1 嵌套虚时网格 (多尺度)")
    nested_grid = NestedImaginaryTimeGrid(beta=beta, max_level=4)
    print(nested_grid.hierarchy_info())
    common = nested_grid.get_common_points(1, 3)
    print(f"\n  Level 1 与 Level 3 的公共点: {len(common)} 个")

    # 4.3 B 矩阵构造与矩阵指数
    sub_banner("4.2 B 矩阵与矩阵指数精度")
    V_test = np.zeros(Ns)
    B_test = build_b_matrix(T_mat - mu * np.eye(Ns), V_test, dtau, method='eigendecomp')
    print(f"  B 矩阵尺寸: {B_test.shape}")
    print(f"  ||B||_F = {np.linalg.norm(B_test, 'fro'):.6f}")
    print(f"  κ(B) = {np.linalg.cond(B_test):.4e}")

    # 矩阵指数方法对比
    A_test = -dtau * (T_mat - mu * np.eye(Ns))
    method_errors = matrix_exp_error_analysis(A_test,
                                              methods=['taylor_10', 'taylor_20', 'eigendecomp'])
    for method, info in method_errors.items():
        print(f"  {method:15s}: rel_err = {info['relative_error']:.4e}")

    # 4.4 运行 DQMC
    sub_banner("4.3 DQMC 主循环 (小规模)")
    sim = DQMCSimulator(
        T_mat=T_mat, U=U, mu=mu, beta=beta, L=L,
        n_sweeps=50, n_therm=20, seed=12345,
        hs_field_type='discrete'
    )
    results = sim.run()

    print(f"\n  === DQMC 结果 ===")
    print(f"  平均能量:       E = {results['mean_energy']:.6f} ± {results['std_energy']:.6f}")
    print(f"  平均双占据:     D = {results['mean_double_occ']:.6f} ± {results['std_double_occ']:.6f}")
    print(f"  平均符号:       ⟨s⟩ = {results['mean_sign']:.4f}")
    print(f"  平均接受率:     α = {results['mean_accept_rate']:.4f}")

    # 4.5 格林函数检查
    sub_banner("4.4 格林函数自洽性检查")
    G_test, sign_test = compute_equal_time_green_function(
        T_mat - mu * np.eye(Ns), sim.hs_config, dtau, U)
    idemp_check = check_green_function_idempotency(G_test)
    herm_check = check_hermiticity(G_test)
    part_check = check_particle_number(G_test, target_density=1.0)

    print(f"  幂等性残差: {idemp_check['relative_residual']:.4e}")
    print(f"  厄米性残差: {herm_check['relative_residual']:.4e}")
    print(f"  粒子数密度: n = {part_check['n_total']:.4f} (目标: {part_check['target_density']})")
    print(f"  粒子数守恒: {'是' if part_check['is_conserved'] else '否'}")

    # 4.6 虚时格林函数 G(τ)
    sub_banner("4.5 虚时格林函数 G(τ)")
    n_tau_show = 5
    tau_vals = np.linspace(0, beta, n_tau_show)
    G_tau_trace = []
    for tau_val in tau_vals:
        # 简化: 直接用矩阵指数计算
        l = int(round(tau_val / dtau))
        l = min(l, L)
        B_prod = np.eye(Ns)
        if l > 0:
            hs_alpha = compute_hs_alpha(dtau, U)
            for m in range(l):
                V_slice = sim.hs_config[m % L] * hs_alpha * np.sign(U)
                B_m = build_b_matrix(T_mat - mu * np.eye(Ns), V_slice, dtau)
                B_prod = B_m @ B_prod
            G_tau = (-1) ** l * B_prod @ G_test
        else:
            G_tau = G_test
        G_tau_trace.append(np.trace(G_tau).real / Ns)

    print(f"  {'τ':>8s} {'tr G(τ)/Ns':>14s}")
    for tau_val, g_val in zip(tau_vals, G_tau_trace):
        print(f"  {tau_val:8.4f} {g_val:14.6f}")

    # 4.7 Matsubara 频率
    sub_banner("4.6 Matsubara 频率与傅里叶变换")
    n_max = 5
    omega_n = matsubara_frequencies(n_max, beta)
    print(f"  费米子 Matsubara 频率 (n=-{n_max}...{n_max}):")
    for n_idx, w in enumerate(omega_n[:5]):
        print(f"    ω_{n_idx - n_max} = {w:.4f}")

    return results, sim, G_test


# ==========================================================================
#  第五阶段: 稳定性分析
# ==========================================================================

def phase5_stability_analysis(T_mat, mu, results):
    """阶段 5: 数值稳定性分析."""
    banner("阶段 5: 数值稳定性分析", '=')

    # 5.1 CFL 条件
    sub_banner("5.1 CFL 稳定性条件")
    delta_x = 1.0  # 晶格常数
    dtau = 0.1
    for scheme in ['explicit_1d', 'explicit_2d', 'explicit_3d', 'implicit']:
        cfl = cfl_condition_check(dtau, delta_x, diffusion_coeff=1.0, scheme=scheme)
        status = '稳定' if cfl['is_stable'] else '不稳定'
        print(f"  {scheme:15s}: CFL = {cfl['cfl_number']:.4f} "
              f"(limit = {cfl['cfl_limit']:.4f}) [{status}]")

    # 5.2 von Neumann 稳定性
    sub_banner("5.2 von Neumann 稳定性分析")
    k_values = np.linspace(0, np.pi, 51)
    vn = von_neumann_stability(dtau, delta_x, k_values, diffusion_coeff=1.0)
    print(f"  放大因子范围: [{vn['min_amplification']:.4f}, {vn['max_amplification']:.4f}]")
    print(f"  稳定性: {'稳定' if vn['is_stable'] else '不稳定'}")

    # 5.3 矩阵指数精度
    sub_banner("5.3 矩阵指数 exp(-Δτ H) 精度分析")
    Ns = T_mat.shape[0]
    A = -dtau * (T_mat - mu * np.eye(Ns))
    method_comparison = matrix_exp_error_analysis(A)
    for method, info in method_comparison.items():
        print(f"  {method:15s}: rel_err = {info['relative_error']:.4e}")

    return True


# ==========================================================================
#  第六阶段: 热化诊断
# ==========================================================================

def phase6_thermalization_diagnostics(results):
    """阶段 6: 热化与非平衡诊断."""
    banner("阶段 6: 热化诊断 (AAR Disequilibrium)", '=')
    print("  融合种子项目: [04] GlacierWeilin AAR disequilibrium")

    # 6.1 自相关时间
    sub_banner("6.1 积分自相关时间")
    for obs in ['energy', 'double_occ']:
        key = obs + '_series'
        if key in results:
            info = integrated_autocorrelation_time(results[key])
            print(f"  {obs:15s}: τ_int = {info['tau_int']:.2f}, "
                  f"N_eff = {info['n_eff']:.1f}, "
                  f"效率 = {info['efficiency']:.4f}")

    # 6.2 AAR 非平衡度
    sub_banner("6.2 AAR 非平衡度诊断")
    for obs in ['energy', 'double_occ', 'sign']:
        key = obs + '_series'
        if key in results:
            aar_info = compute_aar_disequilibrium(results[key])
            status = '已平衡' if aar_info['is_equilibrated'] else '未平衡'
            print(f"  {obs:15s}: AAR = {aar_info['aar']:.4f}, "
                  f" Diseq = {aar_info['disequilibrium']:.4f}, "
                  f" 趋势 = {aar_info['trend_slope']:.4e} [{status}]")

    # 6.3 阻塞分析
    sub_banner("6.3 阻塞分析 (统计误差估计)")
    if 'energy_series' in results:
        blocking = blocking_analysis(results['energy_series'])
        if len(blocking['standard_errors']) > 0:
            print(f"  块大小范围: [{blocking['block_sizes'][0]}, "
                  f"{blocking['block_sizes'][-1]}]")
            print(f"  最终标准误差: {blocking['standard_errors'][-1]:.6f}")

    # 6.4 Gelman-Rubin 多链诊断 (模拟)
    sub_banner("6.4 Gelman-Rubin 多链诊断 (模拟)")
    if 'energy_series' in results:
        series = results['energy_series']
        n = len(series)
        chains = [series[:n // 3], series[n // 3:2 * n // 3], series[2 * n // 3:]]
        chains = [c for c in chains if len(c) > 5]
        if len(chains) >= 2:
            gr = gelman_rubin_statistic(chains)
            print(f"  R̂ = {gr['R_hat']:.4f} (收敛阈值: < 1.1)")
            print(f"  收敛: {'是' if gr['converged'] else '否'}")

    # 6.5 综合报告
    sub_banner("6.5 综合热化诊断报告")
    report = comprehensive_thermalization_check(results)
    for obs, info in report.items():
        is_therm = info['is_thermalized']
        print(f"  {obs:15s}: {'已热化' if is_therm else '未完全热化'}")

    return report


# ==========================================================================
#  第七阶段: 参数优化
# ==========================================================================

def phase7_parameter_optimization(T_mat):
    """阶段 7: 参数搜索与 Mott 转变定位."""
    banner("阶段 7: 参数空间优化", '=')
    print("  融合种子项目: [13] coordinate_search")

    # 7.1 坐标搜索寻找 Mott U_c
    sub_banner("7.1 坐标搜索: Mott 转变临界 U_c")
    U_c_est = find_mott_uc(t=1.0, beta=10.0, U_range=(0.5, 16.0))
    print(f"  估计 Mott 临界点: U_c ≈ {U_c_est:.4f} t")

    # 7.2 U 扫描
    sub_banner("7.2 U/t 扫描 (金属-绝缘体转变)")
    U_values = np.linspace(0.5, 12.0, 12)
    scan_results = scan_U_values(U_values, t=1.0, beta=10.0, n_sites=4)
    print(f"  {'U/t':>6s} {'E/Ns':>10s} {'D':>10s} {'κ':>10s}")
    for i in range(len(U_values)):
        print(f"  {U_values[i]:6.2f} {scan_results['energy'][i]:10.6f} "
              f"{scan_results['double_occ'][i]:10.6f} "
              f"{scan_results['compressibility'][i]:10.6f}")

    # 7.3 簇大小优化
    sub_banner("7.3 最优簇大小")
    Ns_opt = optimize_cluster_size(target_condition=1e8, max_Ns=100)
    print(f"  最优簇大小: Ns ≈ {Ns_opt}")

    return U_c_est


# ==========================================================================
#  第八阶段: 可观测量计算
# ==========================================================================

def phase8_observables(cluster, T_mat, mu, G):
    """阶段 8: 物理可观测量最终计算."""
    banner("阶段 8: 物理可观测量", '=')
    print("  融合种子项目: [03] legendre_fast_rule")

    Ns = T_mat.shape[0]
    U = 4.0

    # 8.1 能量
    sub_banner("8.1 能量分解")
    energy = estimate_total_energy(G, T_mat, U)
    print(f"  动能:   E_kin = {energy['E_kinetic']:.6f}")
    print(f"  势能:   E_pot = {energy['E_potential']:.6f}")
    print(f"  总能:   E     = {energy['E_total']:.6f}")
    print(f"  每格点: E/Ns  = {energy['E_total'] / Ns:.6f}")

    # 8.2 双占据
    sub_banner("8.2 双占据")
    D_mean, D_sites = estimate_double_occupancy(G)
    print(f"  平均双占据: D = {D_mean:.6f}")
    print(f"  各格点双占据范围: [{D_sites.min():.6f}, {D_sites.max():.6f}]")

    # 8.3 高对称路径 k 点
    sub_banner("8.3 高对称路径 k 点")
    k_path, distances = high_symmetry_path_kpoints(a=1.0, n_points_per_segment=10)
    print(f"  路径: Γ → M → K → Γ")
    print(f"  k 点总数: {len(k_path)}")
    print(f"  路径总长度: {distances[-1]:.4f}")

    # 8.4 态密度 (核平滑)
    sub_banner("8.4 态密度 (核平滑)")
    eigenvalues = np.linalg.eigvalsh(T_mat)
    omega = np.linspace(eigenvalues.min() - 0.5, eigenvalues.max() + 0.5, 100)
    dos = compute_dos_by_kernel_polishing(eigenvalues, omega, eta=0.1)
    print(f"  能量网格: {len(omega)} 点")
    print(f"  DOS 范围: [{dos.min():.6f}, {dos.max():.6f}]")
    print(f"  DOS 峰值: ε ≈ {omega[np.argmax(dos)]:.4f}")

    # 8.5 Gauss-Legendre 积分
    sub_banner("8.5 Gauss-Legendre 虚时积分")
    for n_gl in [4, 8, 16, 32]:
        nodes, weights = gauss_legendre_rule(n_gl, 0.0, 5.0)
        # 测试: ∫_0^5 τ² dτ = 125/3 ≈ 41.6667
        test_integral = np.sum(weights * nodes ** 2)
        exact = 125.0 / 3.0
        err = abs(test_integral - exact)
        print(f"  {n_gl:2d}点 GL: ∫τ² dτ = {test_integral:.10f} "
              f"(error = {err:.4e})")

    return energy, D_mean


# ==========================================================================
#  总结
# ==========================================================================

def print_summary(results, U_c, energy, D_mean):
    """打印最终总结."""
    banner("项目总结", '=')
    print("""
  本项目实现了强关联 Hubbard 模型在三角晶格上的
  行列式量子蒙特卡洛 (DQMC) 模拟流水线, 包含:

  [阶段 1] 三角晶格构造、布里渊区离散化、METIS 图分割
  [阶段 2] 高阶有限差分模板、稳态 Dyson 方程、流方程积分
  [阶段 3] Hubbard-Stratonovich 变换、截断正态采样
  [阶段 4] DQMC 核心采样 (自适应提议、嵌套虚时网格)
  [阶段 5] 数值稳定性分析 (CFL、von Neumann、矩阵指数)
  [阶段 6] 热化诊断 (AAR 非平衡、自相关、Gelman-Rubin)
  [阶段 7] 参数优化 (Mott 转变定位、坐标搜索)
  [阶段 8] 可观测量估计 (能量、双占据、态密度)

  关键结果:
""")
    print(f"  Mott 转变临界点: U_c ≈ {U_c:.2f} t")
    print(f"  模拟能量:        E/Ns = {energy['E_total'] / 16.0:.6f}")
    print(f"  双占据:          D = {D_mean:.6f}")
    if 'mean_sign' in results:
        print(f"  平均符号:        ⟨s⟩ = {results['mean_sign']:.4f}")
        print(f"  平均接受率:      α = {results['mean_accept_rate']:.4f}")

    print("""
  15 个种子项目全部融入:
    [01] 1360_truncated_normal      → 截断正态 HS 场采样
    [02] 1408_wedge_grid            → 3D 布里渊区楔形网格
    [03] 660_legendre_fast_rule     → Gauss-Legendre 虚时积分
    [04] 1014_AAR_disequilibrium    → 热化非平衡诊断
    [05] 1320_triangle_to_fem       → 三角晶格 FEM 映射
    [06] 764_midpoint               → 隐式中点流方程积分
    [07] 362_fd1d_heat_steady       → 稳态 FD 求解器
    [08] 293_disk_grid              → 圆盘费米面采样
    [09] 1235_OptimismPerseveration → 自适应提议分布
    [10] 123_burgers_pde_etdrk4     → ETD 矩阵指数
    [11] 1352_triangulation         → 晶格拓扑分析
    [12] 798_nested_sequence_display → 嵌套虚时网格
    [13] 218_coordinate_search      → 坐标搜索优化
    [14] 301_disk01_monte_carlo     → 圆盘 MC 积分
    [15] 796_neighbors_to_metis     → METIS 图分割

  所有阶段完成 ✓
""")


# ==========================================================================
#  主函数
# ==========================================================================

def main():
    """主入口: 零参数运行完整 DQMC 流水线."""
    print("\n" + "=" * 72)
    print("  强关联 Hubbard 模型量子蒙特卡洛模拟系统")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("  PROJECT_268 - 博士级合成项目")
    print("=" * 72)

    np.random.seed(42)

    # 阶段 1: 晶格构造
    cluster, T_mat, mu, eigenvalues = phase1_lattice_construction()

    # 阶段 2: 有限差分
    phase2_finite_difference()

    # 阶段 3: HS 变换与 MC
    hs_samples = phase3_hs_and_monte_carlo()

    # 阶段 4: DQMC 模拟
    results, sim, G = phase4_dqmc_simulation(T_mat, mu, eigenvalues)

    # 阶段 5: 稳定性分析
    phase5_stability_analysis(T_mat, mu, results)

    # 阶段 6: 热化诊断
    therm_report = phase6_thermalization_diagnostics(results)

    # 阶段 7: 参数优化
    U_c = phase7_parameter_optimization(T_mat)

    # 阶段 8: 可观测量
    energy, D_mean = phase8_observables(cluster, T_mat, mu, G)

    # 总结
    print_summary(results, U_c, energy, D_mean)


if __name__ == '__main__':
    main()
