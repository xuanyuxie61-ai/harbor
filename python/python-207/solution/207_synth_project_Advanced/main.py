"""
main.py — 统一入口: 随机热传导方程的不确定性量化与置信/预测带构建

本项目解决的前沿科学问题
========================
对于具有随机扩散系数的一维热传导方程:

    ∂u/∂t = ∂/∂x [ κ(x,ω) · ∂u/∂x ],  x ∈ [0,L], t ∈ [0,T]
    u(0,t) = u_L,  u(L,t) = u_R
    u(x,0) = u₀

其中 κ(x,ω) = κ₀ · exp(σ_ln · Z(x,ω)) 为对数正态随机场,
Z(x,ω) 为零均值平稳高斯场, 协方差核 C(r) = σ²·exp(-r²/(2ℓ²)).

目标: 构建温度场 u(x,t) 的:
  1. 点态置信区间 (Pointwise CI)
  2. 同时置信带 (Simultaneous Confidence Band)
  3. 点态预测区间 (Pointwise PI)
  4. 同时预测带 (Simultaneous Prediction Band)

算法流水线
==========
  uq_parameters           参数定义与验证
  → covariance_cholesky   协方差核 + Cholesky 分解 (AS 6/7)
  → kl_expansion          KL 展开 + 随机场生成
  → cdf_discrete_sampler  分层 MC 采样
  → stochastic_heat_implicit  隐式 FDM 求解 (种子 361)
  → lax_wendroff_advection    对流传播 (种子 355)
  → statistical_aggregator    MC 统计量聚合
  → confidence_band_calibrator 置信带校准 (种子 807+809)
  → prediction_band_builder   预测带构建
  → boundary_word_extractor   边界拓扑分析 (种子 106)
  → maple_boundary_geometry   几何分析 (种子 714)
  → mesh_vertex_to_element    FEM 网格 (种子 756+414)
  → cube_exactness_quadrature 求积精确度 (种子 231+164)
  → tsp_confidence_path       采样路径优化 (种子 1364)
  → luhn_checksum_validator   计算完整性校验 (种子 704)
  → bifurcation_time_delay    时滞系统分析 (种子 1085)
  → fem2d_scalar_field        2D FEM 场求解
"""

import numpy as np
import sys
import os

# 确保包导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uq_parameters import UQProblemParameters
from covariance_cholesky import CovarianceKernel, cholesky_as006, build_precision_from_covariance
from kl_expansion import KLExpansion
from cdf_discrete_sampler import stratified_kl_sampling, DiscreteCDFSampler1D
from stochastic_heat_implicit import solve_stochastic_heat_batch
from lax_wendroff_advection import solve_advection_lw
from statistical_aggregator import (
    compute_mc_statistics, compute_standardized_residuals,
    compute_max_statistics, bootstrap_max_statistics
)
from confidence_band_calibrator import ConfidenceBandCalibrator
from prediction_band_builder import PredictionBandBuilder
from boundary_word_extractor import analyze_confidence_region_topology, word_parity, word_reverse
from maple_boundary_geometry import analyze_confidence_region_geometry, compute_exact_area
from mesh_vertex_to_element import build_vtoe, create_2d_triangulation, assemble_fem_laplacian_2d
from cube_exactness_quadrature import chebyshev1_exactness_test, compute_expectation_via_quadrature
from tsp_confidence_path import optimize_mc_sampling_order
from luhn_checksum_validator import ComputationIntegrityValidator
from bifurcation_time_delay import simulate_bifurcation, parameter_sensitivity, lyapunov_exponent_estimate
from nonlinear_rootfinder import regula_falsi, fixed_point_iteration, calibrate_coverage_critical_value


def print_header(title):
    """打印带装饰的标题."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title):
    """打印节标题."""
    print(f"\n--- {title} ---")


def main():
    """主函数: 执行完整的 UQ 流水线."""

    print_header("随机热传导方程的不确定性量化与置信/预测带构建")
    print("Scientific Problem: UQ for Stochastic Parabolic PDEs")
    print("Focus: Confidence Intervals & Prediction Intervals")

    # ==================================================================
    # Phase 1: 参数定义与验证
    # ==================================================================
    print_section("Phase 1: 参数定义与验证")
    params = UQProblemParameters()
    print(params.summary())

    # ==================================================================
    # Phase 2: 协方差核构造与 Cholesky 分解
    # ==================================================================
    print_section("Phase 2: 协方差核构造与 Cholesky 分解")
    kernel = CovarianceKernel(
        sigma=params.sigma_kappa,
        length_scale=params.correlation_length
    )
    cov_matrix = kernel.evaluate(params.x)
    L_chol, nullty, ifault = cholesky_as006(cov_matrix, params.cholesky_jitter)

    print(f"  协方差矩阵: {cov_matrix.shape[0]}×{cov_matrix.shape[1]}")
    print(f"  Cholesky 分解: ifault={ifault}, nullty={nullty}")

    precision, _, _, cond_est = build_precision_from_covariance(
        cov_matrix, params.cholesky_jitter
    )
    print(f"  条件数估计: {cond_est:.4e}")
    print(f"  精度矩阵: {precision.shape}")

    # ==================================================================
    # Phase 3: KL 展开与随机场生成
    # ==================================================================
    print_section("Phase 3: Karhunen-Loève 展开")
    kl = KLExpansion(
        kernel=kernel,
        x_grid=params.x,
        n_modes=params.n_kl_modes,
        gc_nodes=params.gc_nodes,
        gc_weights=params.gc_weights
    )
    print(f"  KL 模式数: {kl.n_modes}")
    print(f"  保留能量: {kl.energy_retained*100:.2f}%")
    print(f"  前5个特征值: {kl.eigenvalues[:5]}")

    # ==================================================================
    # Phase 4: 分层 MC 采样
    # ==================================================================
    print_section("Phase 4: 分层 Monte Carlo 采样")
    rng = np.random.default_rng(42)
    xi_matrix = stratified_kl_sampling(
        n_modes=params.n_kl_modes,
        n_samples=params.n_mc,
        rng=rng
    )
    print(f"  KL 系数矩阵: {xi_matrix.shape}")
    print(f"  系数均值: {np.mean(xi_matrix, axis=0)[:3]}")
    print(f"  系数标准差: {np.std(xi_matrix, axis=0)[:3]}")

    # CDF 采样器验证
    pdf_test = np.ones(10) / 10
    sampler = DiscreteCDFSampler1D(pdf_test)
    indices, u_vals = sampler.sample_with_dithering(100, rng)
    print(f"  CDF 分层采样: 100 样本, 区间分布 {np.bincount(indices, minlength=10)}")

    # ==================================================================
    # Phase 5: 随机场实现生成
    # ==================================================================
    print_section("Phase 5: 随机扩散系数场生成")
    kappa_fields = kl.generate_batch(
        xi_matrix,
        kappa_0=params.kappa_0,
        sigma_kappa=params.sigma_kappa
    )
    print(f"  扩散系数场: {kappa_fields.shape}")
    print(f"  κ 范围: [{np.min(kappa_fields):.4e}, {np.max(kappa_fields):.4e}]")
    print(f"  κ 均值: {np.mean(kappa_fields):.4e}")
    print(f"  κ > 0 比例: {np.mean(kappa_fields > 0)*100:.1f}%")

    # ==================================================================
    # Phase 6: 隐式 FDM 求解
    # ==================================================================
    print_section("Phase 6: 随机热传导方程求解 (隐式 FDM)")
    solutions = solve_stochastic_heat_batch(kappa_fields, params)
    print(f"  解集合: {solutions.shape}")
    print(f"  最终温度范围: [{np.min(solutions[:, -1, :]):.2f}, {np.max(solutions[:, -1, :]):.2f}] K")

    # ==================================================================
    # Phase 7: Lax-Wendroff 对流传播 (辅助)
    # ==================================================================
    print_section("Phase 7: Lax-Wendroff 对流传播测试")
    u_init_adv = np.zeros(params.nx)
    u_init_adv[params.nx//4:params.nx//2] = 1.0
    u_adv_hist, cfl = solve_advection_lw(
        u_init_adv, c=0.5, dx=params.dx, dt=params.dt, nt=50
    )
    print(f"  CFL 数: {cfl:.4f}")
    print(f"  对流后最大值: {np.max(u_adv_hist[-1]):.4f}")

    # ==================================================================
    # Phase 8: MC 统计量聚合
    # ==================================================================
    print_section("Phase 8: Monte Carlo 统计量聚合")
    stats = compute_mc_statistics(solutions)
    print(f"  均值场范围: [{np.min(stats['mean']):.2f}, {np.max(stats['mean']):.2f}]")
    print(f"  标准差场范围: [{np.min(stats['std']):.4e}, {np.max(stats['std']):.4e}]")
    print(f"  标准误差场均值: {np.mean(stats['stderr']):.4e}")

    residuals = compute_standardized_residuals(
        solutions, stats['mean'], stats['std']
    )
    max_stats = compute_max_statistics(residuals)
    print(f"  极大统计量: mean={np.mean(max_stats):.4f}, std={np.std(max_stats):.4f}")

    # ==================================================================
    # Phase 9: 置信带校准
    # ==================================================================
    print_section("Phase 9: 置信带校准 (Regula Falsi + 不动点迭代)")
    calibrator = ConfidenceBandCalibrator(solutions, alpha=params.alpha_sim)
    cal_result = calibrator.calibrate(n_bootstrap=params.bootstrap_n, seed=42)

    print(f"  校准方法: {cal_result['method']}")
    print(f"  临界值 c_α: {cal_result['critical_value']:.6f}")
    print(f"  覆盖率误差: {cal_result['coverage_error']:.6e}")
    print(f"  有效独立检验数 N_eff: {cal_result['n_eff']:.1f}")
    print(f"  候选临界值:")
    for name, c in cal_result['candidates'].items():
        print(f"    {name}: {c:.6f}")

    ci_lower, ci_upper, c_final = calibrator.build_confidence_band()
    print(f"\n  同时置信带:")
    print(f"    临界值: {c_final:.6f}")
    print(f"    带宽均值: {np.mean(ci_upper - ci_lower):.4f}")

    # ==================================================================
    # Phase 10: 预测区间构建
    # ==================================================================
    print_section("Phase 10: 预测区间构建")
    pred_builder = PredictionBandBuilder(solutions, alpha=params.alpha_pi)
    pi_lower, pi_upper, t_val = pred_builder.pointwise_prediction_interval()
    pb_lower, pb_upper, c_pred = pred_builder.simultaneous_prediction_band(
        n_bootstrap=params.bootstrap_n, seed=42
    )

    width_info = pred_builder.compare_ci_pi_widths(c_final)
    print(f"  t 临界值: {t_val:.4f}")
    print(f"  预测因子 √(1+1/N): {width_info['prediction_factor']:.6f}")
    print(f"  点态 PI 宽度均值: {np.mean(pi_upper - pi_lower):.4f}")
    print(f"  同时 PB 宽度均值: {np.mean(pb_upper - pb_lower):.4f}")
    print(f"  PI/CI 宽度比: {width_info['mean_width_ratio']:.2f}")

    # ==================================================================
    # Phase 11: 求积精确度检验
    # ==================================================================
    print_section("Phase 11: Chebyshev 求积精确度检验")
    cheb_errors = chebyshev1_exactness_test(params.quadrature_order, 2 * params.quadrature_order)
    max_err = max(cheb_errors.values())
    print(f"  求积阶数: {params.quadrature_order}")
    print(f"  最大相对误差 (到 {2*params.quadrature_order} 次): {max_err:.4e}")

    # 用求积计算随机期望
    def test_func(xi):
        """简单的测试函数: 第一个 KL 系数的平方."""
        return xi[0] ** 2 if len(xi) > 0 else 0.0

    expectation, n_eval = compute_expectation_via_quadrature(
        test_func, n_modes=1, quad_order=params.quadrature_order
    )
    print(f"  E[ξ₁²] via 求积: {expectation:.6f} (理论值: 1.0)")

    # ==================================================================
    # Phase 12: FEM 2D 辅助计算
    # ==================================================================
    print_section("Phase 12: 2D FEM 网格与刚度矩阵")
    x_2d = np.linspace(0, 0.5, 11)
    y_2d = np.linspace(0, 0.5, 11)
    nodes_xy, etov = create_2d_triangulation(x_2d, y_2d)
    vtoe_ptr, vtoe = build_vtoe(etov, nodes_xy.shape[0])

    print(f"  2D 网格: {nodes_xy.shape[0]} 顶点, {etov.shape[1]} 单元")
    print(f"  VTOE 总条目: {len(vtoe)}")

    kappa_2d = np.full(nodes_xy.shape[0], params.kappa_0)
    K_stiff = assemble_fem_laplacian_2d(nodes_xy, etov, kappa_2d)
    print(f"  刚度矩阵: {K_stiff.shape}")
    print(f"  刚度矩阵条件数: {np.linalg.cond(K_stiff):.4e}")

    # ==================================================================
    # Phase 13: 边界字与几何分析
    # ==================================================================
    print_section("Phase 13: 置信区域边界拓扑与几何分析")
    # 构造 2D 均值场 (使用最终时刻的解作为 "空间场")
    mean_final = stats['mean'][-1, :]  # shape (nx,)
    # 创建一个 2D 伪场用于边界分析
    field_2d = np.outer(mean_final, np.ones(20))
    ci_lower_2d = np.outer(ci_lower[-1, :], np.ones(20))
    ci_upper_2d = np.outer(ci_upper[-1, :], np.ones(20))

    topo_info = analyze_confidence_region_topology(field_2d, ci_lower_2d, ci_upper_2d)
    print(f"  边界点数: {topo_info['n_boundary_points']}")
    print(f"  环绕数: {topo_info['winding_number']}")
    print(f"  面积分数: {topo_info['area_fraction']:.4f}")

    # 字操作演示
    if topo_info.get('boundary_word_length', 0) > 0:
        test_word = ['N', 'E', 'E', 'S', 'S', 'W', 'W', 'N']
        print(f"  测试字: {test_word}")
        print(f"  字奇偶性: {word_parity(test_word)}")
        print(f"  字反转: {word_reverse(test_word)}")

    geom_info = analyze_confidence_region_geometry(field_2d, np.mean(mean_final))
    print(f"  分形维数: {geom_info['fractal_dimension']:.4f}")
    print(f"  边界长度: {geom_info['boundary_length']:.2f}")

    # ==================================================================
    # Phase 14: TSP 采样路径优化
    # ==================================================================
    print_section("Phase 14: MC 采样路径优化 (TSP)")
    # 使用最终时刻的解
    final_solutions = solutions[:, -1, :]
    opt_order, orig_cost, opt_cost = optimize_mc_sampling_order(final_solutions)
    improvement = (1.0 - opt_cost / max(orig_cost, 1e-30)) * 100
    print(f"  原始路径代价: {orig_cost:.4f}")
    print(f"  优化路径代价: {opt_cost:.4f}")
    print(f"  改进: {improvement:.1f}%")

    # ==================================================================
    # Phase 15: 时滞系统分岔分析
    # ==================================================================
    print_section("Phase 15: 时滞系统分岔与不确定性分析")
    tau_test = np.linspace(0.5, 1.5, 10)
    final_states = parameter_sensitivity(tau_test, b_fixed=1.35, x0=0.01, tmax=20.0)
    print(f"  τ 扫描: {len(tau_test)} 点")
    print(f"  终态范围: [{np.min(final_states):.4f}, {np.max(final_states):.4f}]")
    print(f"  终态标准差: {np.std(final_states):.4f}")

    # Lyapunov 指数
    lambda_max = lyapunov_exponent_estimate(tau=0.95, b=1.35, x0=0.01)
    print(f"  Lyapunov 指数 (τ=0.95, b=1.35): {lambda_max:.4f}")

    # ==================================================================
    # Phase 16: 计算完整性校验
    # ==================================================================
    print_section("Phase 16: Luhn 计算完整性校验")
    validator = ComputationIntegrityValidator()
    cd1 = validator.register_statistic('mc_solutions_final', solutions[:, -1, :])
    cd2 = validator.register_statistic('mean_field', stats['mean'][-1, :])
    cd3 = validator.register_statistic('ci_lower', ci_lower[-1, :])
    print(validator.summary())

    # 验证
    valid1, msg1 = validator.verify_statistic('mc_solutions_final', solutions[:, -1, :])
    print(f"  mc_solutions_final 校验: {msg1}")
    valid2, msg2 = validator.verify_statistic('mean_field', stats['mean'][-1, :])
    print(f"  mean_field 校验: {msg2}")

    # ==================================================================
    # 汇总
    # ==================================================================
    print_header("计算汇总")
    print(f"  MC 样本量:          {params.n_mc}")
    print(f"  KL 模式数:          {params.n_kl_modes}")
    print(f"  空间节点:           {params.nx}")
    print(f"  时间步:             {params.nt}")
    print(f"  Fourier 数:         {params.fourier_number:.4f}")
    print(f"  置信水平:           {params.coverage_target}")
    print(f"  CI 临界值:          {c_final:.6f}")
    print(f"  PI t-临界值:        {t_val:.4f}")
    print(f"  同时 PB 临界值:     {c_pred:.6f}")
    print(f"  ECH N_eff:          {cal_result['n_eff']:.1f}")
    print(f"  Lyapunov 指数:      {lambda_max:.4f}")
    print(f"  计算完整性:         {'通过' if valid1 and valid2 else '失败'}")
    print()
    print("✓ 所有阶段执行完成")

    return {
        'params': params,
        'stats': stats,
        'calibration': cal_result,
        'ci': (ci_lower, ci_upper, c_final),
        'pi': (pi_lower, pi_upper, t_val),
        'pb': (pb_lower, pb_upper, c_pred),
    }


if __name__ == '__main__':
    result = main()
