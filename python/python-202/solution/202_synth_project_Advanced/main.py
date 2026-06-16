"""
不确定性量化: 随机配置方法 — 统一入口
================================================
Uncertainty Quantification via Stochastic Collocation Methods

本项目实现了一个完整的随机配置法框架, 用于求解参数化 PDE 中的
不确定性量化问题。核心科学问题:

  给定随机椭圆方程:
    -d/dx[κ(x,ω) du/dx] = f(x),  x ∈ (0,1)
    u(0) = u(1) = 0

  其中扩散系数 κ(x,ω) 是随机场, 通过 Karhunen-Loève 展开参数化。

  目标: 计算解 u(x,ω) 的统计矩 (均值、方差) 和灵敏度。

方法:
  1. KL 展开参数化随机场
  2. Smolyak 稀疏网格构造配置点
  3. 在每个配置点求解确定性 PDE
  4. 加权组合计算统计矩
  5. PC 展开计算 Sobol 灵敏度指数
  6. 自适应细化优化配置点分布

作者: Sci-Project-Synthesis
版本: 1.0
"""

import sys
import os
import numpy as np

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polynomial_utils import (
    gauss_legendre, gauss_hermite, hermite_evaluate, legendre_evaluate,
    tensor_product_quadrature, compute_gaussian_quadrature
)
from sparse_grid import (
    sparse_grid_nodes_weights, multi_index_set_smolyak,
    smolyak_coefficients, estimate_sparse_grid_size, estimate_full_grid_size
)
from stochastic_field import KarhunenLoeveExpansion, RandomFieldSampler
from elliptic_solver import EllipticSolver1D, solve_parametric_elliptic
from advection_solver import AdvectionDiffusionSolver1D, solve_stochastic_advection_diffusion
from moment_computation import MomentCalculator
from basis_conversion import PolynomialChaosBasis, BasisConverter
from sensitivity_analysis import SobolAnalyzer
from adaptive_refinement import AdaptiveRefinement
from particle_transport import StochasticParticleTransport, FokkerPlanckSolver
from stiff_stochastic_ode import StiffStochasticODE
from response_surface import NaturalCubicSpline, ResponseSurfaceReconstructor
from domain_mapper import StochasticDomainMapper
from sparse_operators import SparseMatrixCSR, BlockToeplitzSolver, CovarianceOperator
from classification_surrogate import SolutionRegimeClassifier
from calendar_converter import MultiScaleCoupling, PeriodicBoundaryHandler


def section_header(title: str) -> None:
    """打印节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def sub_header(title: str) -> None:
    """打印子标题。"""
    print(f"\n--- {title} ---")


# ================================================================
# 第一部分: 稀疏网格构造与验证
# ================================================================
def demo_sparse_grid():
    """演示稀疏网格构造。"""
    section_header("第一部分: Smolyak 稀疏网格构造")

    D = 3  # 随机维度
    level = 4  # Smolyak level

    # 多指标集
    mi = multi_index_set_smolyak(D, level)
    coeffs = smolyak_coefficients(D, level, mi)
    print(f"  维度 D={D}, Level q={level}")
    print(f"  多指标数: {len(mi)}")
    print(f"  非零系数数: {np.sum(np.abs(coeffs) > 1e-15)}")

    # 稀疏网格 vs 全张量积
    sg_size = estimate_sparse_grid_size(D, level)
    fg_size = estimate_full_grid_size(D, level)
    print(f"  稀疏网格原始点数: {sg_size}")
    print(f"  全张量积点数: {fg_size}")
    print(f"  节省比: {fg_size / max(sg_size, 1):.1f}x")

    # 构造稀疏网格
    nodes, weights = sparse_grid_nodes_weights(D, level, rule_type='gauss')
    print(f"  去重后稀疏网格点数: {nodes.shape[0]}")
    print(f"  权重和: {np.sum(weights):.10f} (应为 2^D={2**D})")
    print(f"  节点范围: [{nodes.min():.4f}, {nodes.max():.4f}]")

    # 验证: 积分常数函数应得 2^D ([-1,1]^D 上的体积)
    integral_const = np.sum(weights)
    print(f"  ∫ 1 dξ = {integral_const:.10f} (精确: {2**D})")

    # 验证: 积分 ξ_d² (均匀测度, ∫_{-1}^1 ξ² dξ = 2/3, 乘以其他维度体积 2^{D-1})
    for d in range(D):
        integral_x2 = np.sum(weights * nodes[:, d] ** 2)
        exact_x2 = (2.0 / 3.0) * (2 ** (D - 1))
        print(f"  ∫ ξ_{d}² dξ = {integral_x2:.10f} (精确: {exact_x2:.6f})")

    return nodes, weights


# ================================================================
# 第二部分: 随机场 KL 展开
# ================================================================
def demo_kl_expansion():
    """演示 Karhunen-Loève 展开。"""
    section_header("第二部分: Karhunen-Loève 随机场展开")

    # 空间网格
    N_x = 51
    x = np.linspace(0.0, 1.0, N_x)

    # KL 展开参数
    sigma2 = 0.5  # 协方差幅度
    l_c = 0.2     # 相关长度
    K = 8         # 截断阶数

    kl = KarhunenLoeveExpansion(
        spatial_grid=x,
        covariance_scale=sigma2,
        correlation_length=l_c,
        n_terms=K,
        kernel_type='squared_exponential'
    )

    print(f"  {kl}")
    print(f"  能量捕获比: {kl.energy_ratio:.6f}")
    print(f"  截断方差: {kl.truncated_variance():.6f}")

    # 特征值分布
    print(f"  特征值 (前5): ", end="")
    for k in range(min(5, K)):
        print(f"{kl.eigenvalues[k]:.6f} ", end="")
    print()

    # 评估随机场 (零实现 = 均值场)
    xi_zero = np.zeros(K)
    kappa_mean = kl.evaluate_field(xi_zero, mean_value=1.0, log_transform=True)
    print(f"  均值场: min={kappa_mean.min():.6f}, max={kappa_mean.max():.6f}")

    # 评估随机场 (非零实现)
    xi_sample = np.array([0.5, -0.3, 0.8, -0.1, 0.2, -0.4, 0.6, -0.2])
    kappa_sample = kl.evaluate_field(xi_sample, mean_value=1.0, log_transform=True)
    print(f"  样本场: min={kappa_sample.min():.6f}, max={kappa_sample.max():.6f}, "
          f"mean={kappa_sample.mean():.6f}")

    # 采样器
    sampler = RandomFieldSampler(kl)
    mc_samples = sampler.monte_carlo_sample(100, seed=42)
    lhs_samples = sampler.latin_hypercube_sample(100, seed=42)
    print(f"  MC 样本: shape={mc_samples.shape}, "
          f"mean norm={np.mean(np.linalg.norm(mc_samples, axis=1)):.4f}")
    print(f"  LHS 样本: shape={lhs_samples.shape}, "
          f"mean norm={np.mean(np.linalg.norm(lhs_samples, axis=1)):.4f}")

    return kl, x


# ================================================================
# 第三部分: 随机椭圆方程求解
# ================================================================
def demo_elliptic_solver(kl, x):
    """演示随机椭圆方程求解。"""
    section_header("第三部分: 随机椭圆方程 -d/dx[κ(x) du/dx] = f(x)")

    solver = EllipticSolver1D(
        n_spatial=51,
        domain_length=1.0,
        boundary_left=0.0,
        boundary_right=0.0,
        averaging='harmonic'
    )

    print(f"  空间网格: {solver.n_spatial} 点, h = {solver.h:.6f}")

    # 确定性求解 (均值场)
    K = kl.n_terms
    xi_zero = np.zeros(K)
    u_det = solve_parametric_elliptic(solver, kl, xi_zero)
    print(f"  确定性解 (ξ=0): max|u| = {np.max(np.abs(u_det)):.6f}")

    # 多个配置点求解
    n_config = 20
    rng = np.random.RandomState(42)
    xi_configs = rng.randn(n_config, K) * 0.5  # 缩小范围避免极端值

    solutions = np.zeros((n_config, solver.n_spatial))
    for i in range(n_config):
        solutions[i] = solve_parametric_elliptic(solver, kl, xi_configs[i])
        # 数值检查
        assert np.all(np.isfinite(solutions[i])), f"Non-finite solution at config {i}"

    print(f"  配置点求解: {n_config} 个实现")
    print(f"  解的范围: [{solutions.min():.6f}, {solutions.max():.6f}]")

    # 通量和能量范数
    kappa_mean = kl.evaluate_field(xi_zero, mean_value=1.0, log_transform=True)
    flux = solver.compute_flux(u_det, kappa_mean)
    energy = solver.compute_energy_norm(u_det, kappa_mean)
    print(f"  通量范围: [{flux.min():.6f}, {flux.max():.6f}]")
    print(f"  能量范数: {energy:.6f}")

    # 残差
    residual = solver.residual_error(u_det, kappa_mean)
    print(f"  残差范数: {np.linalg.norm(residual):.2e}")

    return solver, solutions, xi_configs


# ================================================================
# 第四部分: 统计矩计算
# ================================================================
def demo_moment_computation(solutions, weights_approx):
    """演示统计矩计算。"""
    section_header("第四部分: 统计矩计算")

    calc = MomentCalculator(solutions, weights_approx)

    mean = calc.compute_mean()
    var = calc.compute_variance(mean)
    std = calc.compute_std(mean)

    print(f"  均值: max = {np.max(mean):.6f}, min = {np.min(mean):.6f}")
    print(f"  方差: max = {np.max(var):.6f}")
    print(f"  标准差: max = {np.max(std):.6f}")

    skew = calc.compute_skewness(mean, std)
    kurt = calc.compute_kurtosis(mean, std)
    print(f"  偏度: max = {np.max(np.abs(skew)):.6f}")
    print(f"  超额峰度: max = {np.max(np.abs(kurt)):.6f}")

    # 置信区间
    lower, upper = calc.compute_confidence_interval(0.95, mean, std)
    print(f"  95% 置信区间宽度: max = {np.max(upper - lower):.6f}")

    # 超越概率
    threshold = 0.01
    prob = calc.compute_probability_exceedance(threshold, mean, std)
    print(f"  P(u > {threshold}): mean = {np.mean(prob):.4f}")

    # 所有矩
    moments = calc.compute_all_moments()
    print(f"  变异系数 (CV): max = {np.max(moments['cv']):.6f}")

    return moments


# ================================================================
# 第五部分: 多项式混沌基转换
# ================================================================
def demo_basis_conversion():
    """演示 PC 基转换。"""
    section_header("第五部分: 多项式混沌基转换")

    D = 2
    p = 3

    # Hermite 基
    pc_hermite = PolynomialChaosBasis(D, p, 'hermite')
    print(f"  Hermite PC 基: D={D}, p={p}, n_basis={pc_hermite.n_basis}")

    # Legendre 基
    pc_legendre = PolynomialChaosBasis(D, p, 'legendre')
    print(f"  Legendre PC 基: D={D}, p={p}, n_basis={pc_legendre.n_basis}")

    # 基转换矩阵
    T_h2l = BasisConverter.hermite_to_legendre_1d(p, n_quad=30)
    print(f"  Hermite→Legendre 转换矩阵: shape={T_h2l.shape}")
    print(f"  矩阵条件数: {np.linalg.cond(T_h2l):.4f}")

    T_l2h = BasisConverter.legendre_to_hermite_1d(p, n_quad=30)
    print(f"  Legendre→Hermite 转换矩阵: shape={T_l2h.shape}")

    # 验证: T_h2l × T_l2h ≈ I
    product = T_h2l @ T_l2h
    identity_error = np.linalg.norm(product - np.eye(p + 1))
    print(f"  ||T_H2L × T_L2H - I|| = {identity_error:.6f}")

    # 基函数求值
    xi_test = np.array([[0.0, 0.0], [0.5, -0.5], [-0.3, 0.7]])
    psi_h = pc_hermite.evaluate_basis(xi_test)
    psi_l = pc_legendre.evaluate_basis(xi_test)
    print(f"  Hermite 基函数在 (0,0): Ψ_0={psi_h[0, 0]:.4f}")
    print(f"  Legendre 基函数在 (0,0): Ψ_0={psi_l[0, 0]:.4f}")

    return pc_hermite, pc_legendre


# ================================================================
# 第六部分: 灵敏度分析
# ================================================================
def demo_sensitivity_analysis(pc_basis):
    """演示 Sobol 灵敏度分析。"""
    section_header("第六部分: Sobol 全局灵敏度分析")

    D = pc_basis.D
    n_basis = pc_basis.n_basis

    # 合成 PC 系数 (模拟已计算的系数)
    rng = np.random.RandomState(123)
    # 使第一个维度更重要
    coeffs = rng.randn(n_basis) * 0.1
    for j, alpha in enumerate(pc_basis.multi_indices):
        if alpha[0] > 0:
            coeffs[j] *= 3.0  # 维度 0 更重要
        if sum(alpha) == 0:
            coeffs[j] = 1.0  # 均值

    norms_sq = pc_basis._basis_norms_squared()

    analyzer = SobolAnalyzer(coeffs, pc_basis.multi_indices, norms_sq, D)

    var = analyzer.total_variance()
    print(f"  总方差: {float(np.mean(var)):.6f}")

    S1 = analyzer.first_order_sobol()
    ST = analyzer.total_order_sobol()

    print(f"  一阶 Sobol 指数:")
    for d in range(D):
        print(f"    S_{d} = {float(np.mean(S1[d])):.6f}")

    print(f"  总效应 Sobol 指数:")
    for d in range(D):
        print(f"    S_T{d} = {float(np.mean(ST[d])):.6f}")

    d_eff = analyzer.effective_dimension()
    print(f"  有效维度: {d_eff:.4f}")

    important = analyzer.important_dimensions(threshold=0.01)
    print(f"  重要维度 (阈值0.01): {important}")

    summary = analyzer.summary()
    print(f"  摘要: variance_mean={summary['total_variance_mean']:.6f}, "
          f"d_eff={summary['effective_dimension']:.4f}")

    return analyzer


# ================================================================
# 第七部分: 随机对流扩散方程
# ================================================================
def demo_advection_diffusion():
    """演示随机对流扩散方程。"""
    section_header("第七部分: 随机对流扩散方程")

    solver = AdvectionDiffusionSolver1D(
        n_spatial=51,
        domain_length=1.0,
        diffusion_coeff=0.01,
        time_end=0.5,
        cfl_number=0.4
    )

    print(f"  空间网格: {solver.n_spatial} 点")
    print(f"  扩散系数: ν = {solver.nu}")

    # 不同速度下的求解
    velocities = [0.1, 0.5, 1.0, 2.0]
    for c in velocities:
        u_final, time_hist = solver.solve(c)
        l2 = solver.compute_l2_norm(u_final)
        h1 = solver.compute_h1_seminorm(u_final)
        Pe = solver.peclet_number(c)
        print(f"  c={c:.1f}: ||u||₂={l2:.6f}, |u|₁={h1:.6f}, Pe_h={Pe:.4f}")

    # 随机速度集成
    rng = np.random.RandomState(42)
    vel_samples = 0.5 + 0.3 * rng.randn(15)
    final_sols, l2_norms = solve_stochastic_advection_diffusion(solver, vel_samples)
    print(f"  随机速度集成: {len(vel_samples)} 个样本")
    print(f"  L² 范数: mean={np.mean(l2_norms):.6f}, std={np.std(l2_norms):.6f}")

    return solver


# ================================================================
# 第八部分: 随机刚性 ODE
# ================================================================
def demo_stiff_ode():
    """演示随机刚性 ODE 系统。"""
    section_header("第八部分: 随机刚性 ODE (Lindberg 问题)")

    ode_solver = StiffStochasticODE(
        n_equations=3,
        time_end=0.5,
        n_timesteps=200,
        newton_tol=1e-8,
        newton_max_iter=15
    )

    # 确定性求解
    time, sol = ode_solver.solve_lindberg()
    print(f"  时间: [0, {ode_solver.T}]")
    print(f"  终态: u₁={sol[-1, 0]:.6f}, u₂={sol[-1, 1]:.6f}, u₃={sol[-1, 2]:.6f}")
    print(f"  守恒检查: u₁+u₂+u₃ = {np.sum(sol[-1]):.10f}")

    # 刚性比
    S = ode_solver.stiffness_ratio(sol[10])
    print(f"  刚性比 (t=10Δt): {S:.2e}")

    # 随机反应速率
    rng = np.random.RandomState(42)
    rate_samples = np.zeros((10, 3))
    rate_samples[:, 0] = 0.04 * np.exp(0.1 * rng.randn(10))  # k1 随机
    rate_samples[:, 1] = 1e4 * np.exp(0.05 * rng.randn(10))  # k2 随机
    rate_samples[:, 2] = 3e7 * np.ones(10)                     # k3 固定

    t_mean, mean_sol, std_sol = ode_solver.solve_stochastic_lindberg(rate_samples)
    print(f"  随机集成: {len(rate_samples)} 个样本")
    print(f"  均值终态: [{mean_sol[-1, 0]:.6f}, {mean_sol[-1, 1]:.6f}, {mean_sol[-1, 2]:.6f}]")
    print(f"  标准差终态: [{std_sol[-1, 0]:.6f}, {std_sol[-1, 1]:.6f}, {std_sol[-1, 2]:.6f}]")

    return ode_solver


# ================================================================
# 第九部分: 粒子输运与 Fokker-Planck
# ================================================================
def demo_particle_transport():
    """演示随机粒子输运。"""
    section_header("第九部分: 随机粒子输运与 Fokker-Planck 方程")

    # 粒子输运
    pt = StochasticParticleTransport(
        initial_position=0.0,
        diffusion_coeff=0.01,
        time_end=1.0,
        n_timesteps=500
    )

    # 单轨迹
    time, pos = pt.simulate_trajectory(velocity=0.5, seed=42)
    print(f"  单轨迹: X(T) = {pos[-1]:.4f} (c=0.5)")

    # 系综
    velocities = np.array([0.2, 0.5, 0.8, 1.0, 1.5])
    ens = pt.simulate_ensemble(velocities, seed=42)
    print(f"  系综均值终位: {ens['overall_mean'][-1]:.4f}")
    print(f"  系综方差终值: {ens['overall_var'][-1]:.6f}")

    D_eff = pt.compute_dispersion_coefficient(velocities)
    print(f"  有效分散系数: D_eff = {D_eff:.6f}")

    # Fokker-Planck 求解
    fp = FokkerPlanckSolver(
        n_spatial=101,
        x_min=-1.0,
        x_max=1.0,
        time_end=0.5,
        n_timesteps=100
    )
    x_fp, pdf = fp.solve(velocity=0.2, diffusion=0.05)
    print(f"  Fokker-Planck PDF: max = {np.max(pdf):.4f}, "
          f"∫p dx = {np.sum(pdf) * (x_fp[1] - x_fp[0]):.4f}")

    return pt


# ================================================================
# 第十部分: 响应面重构
# ================================================================
def demo_response_surface():
    """演示响应面重构。"""
    section_header("第十部分: 自然三次样条响应面重构")

    # 测试函数
    def test_func(xi):
        return np.sin(3 * xi) * np.exp(-xi ** 2)

    # 配置点
    nodes = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    values = test_func(nodes)

    # 样条拟合
    recon = ResponseSurfaceReconstructor()
    spline = recon.reconstruct_1d(nodes, values)

    # 评估
    xi_fine = np.linspace(-1.5, 1.5, 100)
    u_fine = test_func(xi_fine)
    u_spline = spline.evaluate(xi_fine)

    error = np.max(np.abs(u_fine - u_spline))
    print(f"  最大插值误差: {error:.2e}")
    print(f"  样条曲率: {spline.curvature():.6f}")

    # 统计矩
    stats = recon.compute_statistics_from_spline(spline, n_quad=50)
    print(f"  E[u] = {stats['mean']:.6f}")
    print(f"  Var[u] = {stats['variance']:.6f}")
    print(f"  Std[u] = {stats['std']:.6f}")

    # 导数
    du_spline = spline.evaluate_derivative(xi_fine)
    du_exact = 3.0 * np.cos(3 * xi_fine) * np.exp(-xi_fine ** 2) - \
               2 * xi_fine * np.sin(3 * xi_fine) * np.exp(-xi_fine ** 2)
    deriv_error = np.max(np.abs(du_exact - du_spline))
    print(f"  导数最大误差: {deriv_error:.2e}")

    return spline


# ================================================================
# 第十一部分: 域映射
# ================================================================
def demo_domain_mapping():
    """演示随机域映射。"""
    section_header("第十一部分: 随机域映射")

    mapper = StochasticDomainMapper(
        reference_domain=(0.0, 1.0),
        mapping_type='affine'
    )

    # 仿射映射
    xi_hat = np.linspace(0.0, 1.0, 21)

    # 不同随机边界
    for a, b in [(0.0, 1.0), (-0.1, 1.1), (0.1, 0.9)]:
        x, J = mapper.affine_map_1d(xi_hat, a, b)
        quality = mapper.compute_mapping_distortion(J)
        print(f"  边界 [{a}, {b}]: J={J[0]:.4f}, "
              f"条件数={quality['condition_number']:.4f}, "
              f"微分同胚={quality['is_diffeomorphism']}")

    # 多项式映射
    coeffs = np.array([0.1, 0.8, 0.2])  # x = 0.1 + 0.8ξ̂ + 0.2ξ̂²
    x_poly, J_poly = mapper.polynomial_map_1d(xi_hat, coeffs)
    quality_poly = mapper.compute_mapping_distortion(J_poly)
    print(f"  多项式映射: J ∈ [{J_poly.min():.4f}, {J_poly.max():.4f}], "
          f"微分同胚={quality_poly['is_diffeomorphism']}")

    # PDE 系数变换
    kappa_phys = np.ones_like(xi_hat) * 2.0
    source_phys = np.ones_like(xi_hat) * 1.0
    kappa_ref, source_ref = mapper.transform_pde_coefficients(
        kappa_phys, J_poly, source_phys
    )
    print(f"  变换后 κ: [{kappa_ref.min():.4f}, {kappa_ref.max():.4f}]")
    print(f"  变换后 f: [{source_ref.min():.4f}, {source_ref.max():.4f}]")

    return mapper


# ================================================================
# 第十二部分: 稀疏算子
# ================================================================
def demo_sparse_operators():
    """演示稀疏算子。"""
    section_header("第十二部分: 稀疏算子与协方差求解")

    n = 20

    # 三对角 CSR
    diag = 4.0 * np.ones(n)
    lower = -np.ones(n - 1)
    upper = -np.ones(n - 1)
    csr = SparseMatrixCSR.from_tridiagonal(lower, diag, upper)

    print(f"  CSR 矩阵: {n}×{n}, nnz={csr.nnz()}, "
          f"稀疏度={csr.sparsity_ratio():.4f}")

    # SpMV
    x = np.ones(n)
    y = csr.matvec(x)
    print(f"  SpMV: ||Ax|| = {np.linalg.norm(y):.6f}")

    # 求解
    b = np.ones(n)
    x_sol = csr.solve(b)
    residual = np.linalg.norm(csr.matvec(x_sol) - b)
    print(f"  求解残差: {residual:.2e}")

    # 对角分离格式
    from sparse_operators import DiagonalSeparatedStorage
    dia = DiagonalSeparatedStorage.from_tridiagonal(lower, diag, upper)
    inv_d = dia.jacobi_preconditioner()
    print(f"  Jacobi 预处理: max(inv_diag) = {np.max(inv_d):.4f}")

    # Block Toeplitz (协方差矩阵)
    def cov_func(r):
        return np.exp(-r / 0.3)

    cov_op = CovarianceOperator(cov_func, n_points=10, grid_spacing=0.1)
    bt_solver = cov_op.build_toeplitz_solver()
    A_dense = bt_solver.to_dense()
    print(f"  Block Toeplitz: 10×10, 条件数={np.linalg.cond(A_dense):.4f}")

    # 采样
    samples = cov_op.sample_field(n_samples=3, seed=42)
    print(f"  随机场样本: {samples.shape}")
    print(f"  样本均值: {np.mean(samples, axis=0)[:5]}")

    return csr


# ================================================================
# 第十三部分: 分类代理与自适应细化
# ================================================================
def demo_classification_adaptive():
    """演示分类代理和自适应细化。"""
    section_header("第十三部分: 解行为分类与自适应细化")

    # 分类器
    D = 2
    classifier = SolutionRegimeClassifier(D, max_order=3, regularization=1e-3)

    # 生成训练数据
    rng = np.random.RandomState(42)
    xi_train = rng.uniform(-1.5, 1.5, (50, D))
    y_train = np.where(xi_train[:, 0] > 0, 1.0, -1.0)  # 简单分界

    history = classifier.fit(xi_train, y_train, n_epochs=50, learning_rate=0.1)
    print(f"  训练: {len(xi_train)} 样本, 最终准确率={history['accuracy'][-1]:.4f}")

    # 预测
    xi_test = rng.uniform(-1.5, 1.5, (20, D))
    preds = classifier.predict(xi_test)
    print(f"  预测: {np.sum(preds > 0)} 正类, {np.sum(preds < 0)} 负类")

    # 边界点检测
    boundary_pts = classifier.find_boundary_points(xi_test, n_select=5)
    print(f"  边界点: {boundary_pts.shape}")

    # 自适应细化
    refiner = AdaptiveRefinement(
        dimension=D, initial_level=2,
        refinement_threshold=1e-3, max_level=6
    )

    # 模拟误差指示器
    nodes_test = rng.uniform(-1, 1, (30, D))
    surplus = np.abs(np.sin(3 * nodes_test[:, 0])) * np.exp(-nodes_test[:, 1] ** 2)

    marked = refiner.dorfler_marking(surplus)
    print(f"  Dörfler 标记: {np.sum(marked)}/{len(marked)} 点被标记")

    # CVT 密度
    density = refiner.compute_cvt_density(nodes_test, surplus)
    print(f"  CVT 密度: [{density.min():.6f}, {density.max():.6f}]")

    # Lloyd 迭代
    new_points = refiner.lloyd_cvt_refinement(
        nodes_test, density, n_new_points=10, n_iterations=15
    )
    print(f"  Lloyd CVT: 生成 {new_points.shape[0]} 个新点")

    # 收敛速率估计
    n_list = [10, 30, 100, 300, 1000]
    e_list = [1.0, 0.3, 0.1, 0.03, 0.01]
    rate = refiner.estimate_convergence_rate(n_list, e_list)
    print(f"  估计收敛速率: r = {rate:.4f}")

    return classifier, refiner


# ================================================================
# 第十四部分: 多尺度耦合与周期处理
# ================================================================
def demo_multiscale_periodic():
    """演示多尺度耦合和周期边界。"""
    section_header("第十四部分: 多尺度耦合与周期边界处理")

    # 多尺度耦合
    ms = MultiScaleCoupling(n_fast_steps=100, n_slow_steps=10, epsilon=0.01)

    t = np.linspace(0, 0.1, 200)
    u_coupled = ms.coupled_solution(t, fast_freq=1.0, fast_amp=0.1,
                                     slow_decay=5.0, coupling_strength=0.3)
    u_homo = ms.homogenized_solution(t, slow_decay=5.0)

    err = ms.phase_error(t, u_coupled, u_homo)
    print(f"  多尺度耦合: 最大相位误差 = {err:.6f}")
    print(f"  耦合解范围: [{u_coupled.min():.6f}, {u_coupled.max():.6f}]")
    print(f"  均匀化解范围: [{u_homo.min():.6f}, {u_homo.max():.6f}]")

    # 周期边界处理
    pb = PeriodicBoundaryHandler(domain_length=1.0)
    x = np.linspace(0, 1.0, 64, endpoint=False)
    u_periodic = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)

    coeffs = pb.fourier_coefficients(u_periodic, x)
    energy = pb.spectral_energy(coeffs)
    print(f"  傅里叶系数: 前3个 = {np.abs(coeffs[:3])}")
    print(f"  谱能量 (前3): {energy[:3]}")
    print(f"  总谱能量: {np.sum(energy):.6f}")

    # 重构
    u_recon = pb.reconstruct_from_fourier(coeffs, n_modes=5, x=x)
    recon_error = np.max(np.abs(u_periodic - u_recon))
    print(f"  傅里叶重构误差: {recon_error:.2e}")

    return ms, pb


# ================================================================
# 第十五部分: 完整 UQ 工作流集成
# ================================================================
def demo_full_uq_workflow():
    """演示完整的 UQ 工作流。"""
    section_header("第十五部分: 完整 UQ 工作流集成")

    print("  执行完整流程:")
    print("  1. 定义随机场 → KL 展开")
    print("  2. 构造稀疏网格配置点")
    print("  3. 配置点处求解 PDE")
    print("  4. 计算统计矩")
    print("  5. 灵敏度分析")
    print("  6. 响应面重构")
    print("  7. 收敛验证")

    # 参数设置
    D = 2       # 随机维度
    level = 3   # 稀疏网格 level
    N_x = 31    # 空间点数

    # Step 1: KL 展开
    x = np.linspace(0.0, 1.0, N_x)
    kl = KarhunenLoeveExpansion(x, covariance_scale=0.3,
                                 correlation_length=0.3, n_terms=D,
                                 kernel_type='squared_exponential')
    print(f"\n  Step 1: KL 展开 K={D}, 能量比={kl.energy_ratio:.4f}")

    # Step 2: 稀疏网格
    nodes, weights = sparse_grid_nodes_weights(D, level, rule_type='gauss')
    n_config = nodes.shape[0]
    print(f"  Step 2: 稀疏网格 level={level}, 配置点数={n_config}")

    # Step 3: PDE 求解
    solver = EllipticSolver1D(n_spatial=N_x, domain_length=1.0)
    solutions = np.zeros((n_config, N_x))
    for i in range(n_config):
        kappa = kl.evaluate_field(nodes[i], mean_value=1.0, log_transform=True)
        solutions[i] = solver.solve(kappa)
    print(f"  Step 3: 求解 {n_config} 个配置点 PDE 完成")

    # Step 4: 统计矩
    calc = MomentCalculator(solutions, weights)
    mean = calc.compute_mean()
    var = calc.compute_variance(mean)
    print(f"  Step 4: E[u] max={np.max(mean):.6f}, Var[u] max={np.max(var):.6f}")

    # Step 5: PC 投影与灵敏度
    pc = PolynomialChaosBasis(D, max_order=2, basis_type='legendre')
    coeffs = pc.compute_pc_coefficients(
        solutions, nodes, weights
    )
    norms_sq = pc._basis_norms_squared()
    analyzer = SobolAnalyzer(coeffs, pc.multi_indices, norms_sq, D)
    S1 = analyzer.first_order_sobol()
    ST = analyzer.total_order_sobol()
    print(f"  Step 5: Sobol S1 = [{float(np.mean(S1[0])):.4f}, {float(np.mean(S1[1])):.4f}]")
    print(f"          Sobol ST = [{float(np.mean(ST[0])):.4f}, {float(np.mean(ST[1])):.4f}]")

    # Step 6: 响应面重构
    recon = ResponseSurfaceReconstructor()
    # 取空间中间点的响应面
    mid_idx = N_x // 2
    spline = recon.reconstruct_1d(nodes[:, 0], solutions[:, mid_idx])
    stats = recon.compute_statistics_from_spline(spline, n_quad=30)
    print(f"  Step 6: 响应面统计 E[u]={stats['mean']:.6f}, "
          f"Var[u]={stats['variance']:.6f}")

    # Step 7: 收敛验证 — 对比不同 level
    print(f"  Step 7: 收敛研究")
    errors = []
    n_points_list = []
    for lv in [2, 3, 4]:
        nd, wt = sparse_grid_nodes_weights(D, lv, rule_type='gauss')
        n_pts = nd.shape[0]
        n_points_list.append(n_pts)

        sols_lv = np.zeros((n_pts, N_x))
        for i in range(n_pts):
            kappa_lv = kl.evaluate_field(nd[i], mean_value=1.0, log_transform=True)
            sols_lv[i] = solver.solve(kappa_lv)

        calc_lv = MomentCalculator(sols_lv, wt)
        mean_lv = calc_lv.compute_mean()

        # 与最细网格对比
        if lv == 4:
            ref_mean = mean_lv
        errors.append(mean_lv)

    if len(errors) >= 2:
        for lv_idx, lv in enumerate([2, 3, 4]):
            if lv_idx < len(errors) - 1:
                err = np.linalg.norm(errors[lv_idx] - errors[-1])
                print(f"    level={lv}: N={n_points_list[lv_idx]}, "
                      f"error vs level 4 = {err:.6e}")

    print("\n  ✓ 完整 UQ 工作流执行完毕")


# ================================================================
# 主函数
# ================================================================
def main():
    """
    主函数: 运行所有演示模块。

    零参数可运行, 自动执行完整的不确定性量化计算流程。
    """
    print("=" * 70)
    print("  不确定性量化: 随机配置方法 (Stochastic Collocation UQ)")
    print("  PROJECT 202 — 博士级科研代码合成项目")
    print("=" * 70)
    print(f"  NumPy 版本: {np.__version__}")
    print(f"  随机种子: 42 (可复现)")

    np.random.seed(42)

    # 运行各模块
    nodes, weights = demo_sparse_grid()
    kl, x = demo_kl_expansion()
    solver, solutions, xi_configs = demo_elliptic_solver(kl, x)

    # 为矩量计算构造近似权重 (归一化)
    w_approx = np.ones(solutions.shape[0]) / solutions.shape[0]
    moments = demo_moment_computation(solutions, w_approx)

    pc_h, pc_l = demo_basis_conversion()
    analyzer = demo_sensitivity_analysis(pc_h)
    demo_advection_diffusion()
    demo_stiff_ode()
    demo_particle_transport()
    demo_response_surface()
    demo_domain_mapping()
    demo_sparse_operators()
    classifier, refiner = demo_classification_adaptive()
    demo_multiscale_periodic()
    demo_full_uq_workflow()

    section_header("项目运行完成")
    print("  所有 15 个模块已成功执行。")
    print("  本项目融合了以下种子项目的核心算法:")
    print("    1. DeepHistoPathology → 解行为分类 (classification_surrogate)")
    print("    2. circle_integrals → Gamma函数积分 (polynomial_utils)")
    print("    3. ToponymGeoreferencing → 域映射 (domain_mapper)")
    print("    4. matrix_assemble_parfor → 并行矩阵组装 (sparse_grid)")
    print("    5. calpak → 基转换/多周期耦合 (calendar_converter)")
    print("    6. md → 粒子输运 (particle_transport)")
    print("    7. fd1d_poisson → 椭圆求解器 (elliptic_solver)")
    print("    8. r8bto → Block Toeplitz (sparse_operators)")
    print("    9. lindberg_ode → 刚性随机ODE (stiff_stochastic_ode)")
    print("   10. cvt_1d_sampling → CVT自适应 (adaptive_refinement)")
    print("   11. fd1d_advection_ftcs → 对流扩散 (advection_solver)")
    print("   12. AccMLBio → 弱监督分类 (classification_surrogate)")
    print("   13. r8sr → 稀疏存储 (sparse_operators)")
    print("   14. BoolQ → 多项式分类 (classification_surrogate)")
    print("   15. interp_ncs → 样条插值 (response_surface)")
    print("\n  所有计算已完成, 无报错。")


if __name__ == '__main__':
    main()
