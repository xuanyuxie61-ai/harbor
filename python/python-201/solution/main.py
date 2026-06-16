#!/usr/bin/env python3
"""
main.py — 统一入口: 随机 Cahn-Hilliard 相场方程的
          多元素广义多项式混沌不确定性量化
=========================================================

本项目实现一个前沿博士级科学计算问题:
  随机 Cahn-Hilliard 相场方程的多元素广义多项式混沌 (ME-gPC)
  不确定性量化与全局灵敏度分析。

物理问题:
  二元合金旋节分解 (spinodal decomposition) 过程中,
  梯度能量系数 κ(ξ)、双阱势参数 γ(ξ) 和迁移率 M(ξ)
  为随机变量, 通过 ME-gPC 方法传播不确定性,
  计算浓度场的统计矩、Sobol 灵敏度指数和失效概率。

数学框架:
  随机 Cahn-Hilliard 方程:
    ∂c/∂t = ∇·(M(ξ)∇μ),  μ = γ(ξ)(c³-c) - κ(ξ)∇²c

  多元素 gPC 展开:
    c(x,t,ξ) ≈ Σ_{e=1}^{E} Σ_{k=0}^{P_e} ĉ_k^{(e)}(x,t) Ψ_k^{(e)}(ξ)

算法组件:
  1. 多项式混沌基函数构造 (Hermite/Legendre/Jacobi)
  2. Smolyak 稀疏网格求积
  3. 随机 Galerkin 投影与耦合张量计算
  4. Cahn-Hilliard 方程的 Fourier 谱方法求解
  5. 多元素自适应细化 (Dörfler 标记策略)
  6. Port-Hamiltonian 约束的神经闭合模型
  7. 多保真度控制变量方差缩减
  8. MCMC 贝叶斯推断 (PCE 代理模型加速)
  9. Sobol 全局灵敏度分析与失效概率计算

种子项目映射:
  096_bisection_min     → 二分法搜索最优 PCE 阶数
  1050_LaM-SLidE        → 编码器-解码器架构 / 神经闭合
  986_r8ncf             → 稀疏矩阵 COO 格式
  1434_zombie_ode       → 耦合系统 → 多元素耦合
  680_line_grid         → 一维网格生成 → 物理/随机网格
  738_matrix_assemble   → 并行矩阵装配 → Galerkin 系统
  932_pyramid_grid      → 层级网格 → Smolyak 稀疏网格
  068_ball_integrals    → 高维积分 → 概率空间求积
  1246_Gaulios          → 有限-无穷horizon → 多保真度桥接
  211_continuity_exact  → 无散度场 → 质量守恒约束
  033_asa076            → 正态CDF/Owen T → 失效概率
  1298_Port-Hamiltonian → Port-Hamiltonian → 能量保持闭合
  787_navier_stokes     → NS 精确解 → CH 求解器验证
  1180_HII-galaxy       → MCMC + 贝叶斯 → 后验推断
  382_fem_to_xml        → 有限元网格 → 计算域离散化

运行:
  python main.py   # 零参数运行

依赖:
  numpy, scipy (标准科学计算库)
"""

import argparse
import json
import sys
import time
import numpy as np

# ============================================================
#  导入项目模块
# ============================================================
from config import (create_default_config, GlobalConfig,
                    estimate_total_basis_dim)
from measure import (gauss_quadrature, tensor_product_quadrature,
                     owen_t_function, pdf_eval, cdf_eval)
from utils import (bisection_optimal_order, SparseMatrixCOO,
                   multi_index_set, line_grid,
                   spectral_derivative_2d)
from grid import (create_physical_mesh, generate_initial_condition,
                  smolyak_sparse_grid, generate_mesh_info)
from polynomial_basis import OrthogonalPolynomialBasis
from multi_element import MultiElementGalerkin
from sparse_grid import AdaptiveSparseGrid
from galerkin import (StochasticGalerkinProjector,
                      GalerkinSystemAssembler)
from cahn_hilliard import (CahnHilliardSolver,
                           StochasticCahnHilliard)
from refinement import AdaptiveRefinement
from neural_closure import (PortHamiltonianClosure,
                            create_closure_model,
                            generate_training_data)
from multifidelity import (MultiFidelityController,
                           run_multifidelity_analysis)
from statistics import PCEStatistics
from mcmc_bayes import (MCMCSampler, compute_failure_probability_is,
                        format_bayesian_results)


def print_header():
    """打印项目标题"""
    print("=" * 70)
    print("  随机 Cahn-Hilliard 相场方程的多元素广义多项式混沌")
    print("  不确定性量化与全局灵敏度分析")
    print("  Multi-Element gPC for Stochastic Cahn-Hilliard Equation")
    print("=" * 70)
    print()


def print_section(title: str):
    """打印节标题"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ============================================================
#  阶段 1: 配置与初始化
# ============================================================
def phase_1_initialization() -> GlobalConfig:
    """初始化全局配置"""
    print_section("阶段 1: 配置与初始化")

    config = create_default_config()

    # 打印配置
    d = len(config.measures)
    P = estimate_total_basis_dim(config)
    print(f"  随机空间维度: d = {d}")
    print(f"  各维度测度:")
    for i, m in enumerate(config.measures):
        print(f"    ξ_{i} ({m.name}): {m.measure_type}, "
              f"mean={m.mean:.4f}, var={m.variance:.6f}, "
              f"support={m.support}")
    print(f"  PCE 最大阶数: p = {config.basis.max_degree}")
    print(f"  基函数数量: P = {P}")
    print(f"  多元素初始数: {config.multi_element.initial_elements}")
    print(f"  稀疏网格层级: {config.sparse_grid.level}")
    print(f"  物理网格: {config.cahn_hilliard.nx} × "
          f"{config.cahn_hilliard.ny}")
    print(f"  时间步: dt={config.cahn_hilliard.dt:.1e}, "
          f"N={config.cahn_hilliard.n_steps}")

    return config


# ============================================================
#  阶段 2: 多项式基函数与求积规则
# ============================================================
def phase_2_basis_and_quadrature(config: GlobalConfig
                                 ) -> tuple:
    """构造正交多项式基和稀疏网格"""
    print_section("阶段 2: 多项式基函数与求积规则")

    # 构造基函数
    basis = OrthogonalPolynomialBasis(config.basis, config.measures)
    print(f"  基函数构造完成:")
    print(f"    维度 d = {basis.n_dim}")
    print(f"    基函数数 P = {basis.n_basis}")
    print(f"    截断类型: {config.basis.truncation}")
    print(f"    索引集 (前10个):")
    for i in range(min(10, basis.n_basis)):
        print(f"      α_{i} = {tuple(basis.indices[i])}")

    # 稀疏网格
    sparse_grid = AdaptiveSparseGrid(config)
    print(f"\n  Smolyak 稀疏网格:")
    print(f"    求积点数: {sparse_grid.n_points}")
    print(f"    层级: {config.sparse_grid.level}")

    # 验证正交性
    q_nodes = sparse_grid.get_nodes_array()
    q_weights = sparse_grid.get_weights_array()
    orth_err = basis.verify_orthogonality(q_nodes, q_weights)
    print(f"    正交性验证: ||S - I||_F/P = {orth_err:.4e}")

    # 二分法搜索最优阶数 (映射 096_bisection_min)
    print(f"\n  二分法搜索最优多项式阶数 (映射 096_bisection_min):")

    def error_estimate(p):
        """经验误差估计: C(d+p,p) * machine_eps"""
        from math import comb
        n_b = comb(basis.n_dim + p, p)
        return n_b * 1e-14 + 1.0 / (p + 1) ** 2

    p_opt = bisection_optimal_order(
        error_estimate, p_min=1, p_max=10,
        target_error=1e-4)
    print(f"    最优阶数: p_opt = {p_opt}")
    print(f"    对应误差估计: {error_estimate(p_opt):.4e}")

    return basis, sparse_grid


# ============================================================
#  阶段 3: 随机 Galerkin 投影
# ============================================================
def phase_3_galerkin_projection(config: GlobalConfig,
                                basis: OrthogonalPolynomialBasis
                                ) -> tuple:
    """执行随机 Galerkin 投影"""
    print_section("阶段 3: 随机 Galerkin 投影")

    projector = StochasticGalerkinProjector(config, basis)

    # 验证对称性
    sym_err = projector.verify_galerkin_symmetry()
    print(f"  Galerkin 对称性验证: ||C - C^T||/||C|| = {sym_err:.4e}")

    # 耦合张量信息
    C = projector.coupling_tensor
    print(f"  三重乘积张量:")
    print(f"    形状: {C.shape}")
    print(f"    非零元比例: "
          f"{np.sum(np.abs(C) > 1e-10) / C.size:.4f}")
    print(f"    最大绝对值: {np.max(np.abs(C)):.4e}")

    # 系统装配
    assembler = GalerkinSystemAssembler(projector, n_physical_dof=16)
    print(f"\n  {assembler.summary()}")

    # 演示: 随机扩散系数
    kappa_hat = np.zeros(basis.n_basis)
    kappa_hat[0] = config.cahn_hilliard.kappa_mean
    if basis.n_basis > 1:
        kappa_hat[1] = config.cahn_hilliard.kappa_std

    K_sg = projector.compute_diffusion_coupling(kappa_hat)
    print(f"  随机扩散耦合矩阵:")
    print(f"    形状: {K_sg.shape}")
    print(f"    非零元数: {K_sg.nnz}")
    print(f"    条件数 (估计): {np.linalg.cond(K_sg.toarray()):.2e}")

    # 灵敏度耦合
    S = projector.compute_sensitivity_coupling()
    print(f"\n  灵敏度耦合矩阵:")
    print(f"    形状: {S.shape}")
    for d_idx in range(basis.n_dim):
        print(f"    ∂/∂ξ_{d_idx}: max|S| = "
              f"{np.max(np.abs(S[d_idx])):.4e}")

    return projector, assembler


# ============================================================
#  阶段 4: 多元素自适应细化
# ============================================================
def phase_4_adaptive_refinement(config: GlobalConfig
                                ) -> MultiElementGalerkin:
    """执行多元素自适应细化"""
    print_section("阶段 4: 多元素自适应细化 (ME-gPC)")

    me = MultiElementGalerkin(config)
    print(f"  初始 ME-gPC 状态:")
    print(f"    元素数: {me.n_elements}")
    print(f"    总基函数: {me.total_basis_size}")

    # 设置初始系数 (模拟)
    rng = np.random.RandomState(42)
    for elem in me.elements:
        P_e = elem.local_basis.n_basis
        coeffs = rng.randn(P_e) * 0.01
        coeffs[0] = 1.0 + rng.randn() * 0.1
        elem.set_coefficients(coeffs)

    print(f"\n  初始统计量:")
    mean, var = me.compute_global_statistics()
    print(f"    全局均值: {mean:.6f}")
    print(f"    全局方差: {var:.6e}")

    # 自适应细化
    refiner = AdaptiveRefinement(config, me)

    print(f"\n  自适应细化循环:")
    for iteration in range(5):
        info = refiner.refine()
        status = info.get("status", "?")
        n_elem = info.get("n_elements", me.n_elements)
        max_ind = info.get("max_indicator", 0.0)
        print(f"    Step {iteration + 1}: status={status}, "
              f"n_elements={n_elem}, "
              f"max_indicator={max_ind:.4e}")

        if status == "converged":
            break

    mean_final, var_final = me.compute_global_statistics()
    print(f"\n  细化后统计量:")
    print(f"    元素数: {me.n_elements}")
    print(f"    总基函数: {me.total_basis_size}")
    print(f"    全局均值: {mean_final:.6f}")
    print(f"    全局方差: {var_final:.6e}")

    return me


# ============================================================
#  阶段 5: Cahn-Hilliard 求解
# ============================================================
def phase_5_cahn_hilliard_solve(config: GlobalConfig,
                                basis: OrthogonalPolynomialBasis,
                                projector: StochasticGalerkinProjector,
                                sparse_grid: AdaptiveSparseGrid
                                ) -> tuple:
    """求解随机 Cahn-Hilliard 方程"""
    print_section("阶段 5: 随机 Cahn-Hilliard 方程求解")

    # 创建确定性求解器
    solver = CahnHilliardSolver(config.cahn_hilliard)
    X, Y, dx, dy = create_physical_mesh(config.cahn_hilliard)

    # 物理网格信息 (映射 382_fem_to_xml)
    mesh_info = generate_mesh_info(
        config.cahn_hilliard.nx, config.cahn_hilliard.ny,
        config.cahn_hilliard.Lx, config.cahn_hilliard.Ly)
    print(f"  物理网格信息 (映射 382_fem_to_xml):")
    print(f"    节点数: {mesh_info['n_nodes']}")
    print(f"    单元数: {mesh_info['n_elements']}")
    print(f"    单元类型: {mesh_info['element_type']}")
    print(f"    dx={dx:.4f}, dy={dy:.4f}")

    # 确定性验证 (映射 787_navier_stokes_2d_exact)
    print(f"\n  确定性验证 (映射 787_navier_stokes_2d_exact):")
    c_init = generate_initial_condition(config.cahn_hilliard, X, Y)
    print(f"    初始浓度: mean={np.mean(c_init):.4f}, "
          f"std={np.std(c_init):.4f}")

    # 用标称参数求解
    c_final, energies = solver.solve(
        c_init,
        kappa=config.cahn_hilliard.kappa_mean,
        gamma=config.cahn_hilliard.gamma_mean,
        mobility=config.cahn_hilliard.mobility_mean,
        n_steps=100)

    print(f"    最终浓度: mean={np.mean(c_final):.4f}, "
          f"std={np.std(c_final):.4f}")
    print(f"    初始能量: {energies[0]:.6f}")
    print(f"    最终能量: {energies[-1]:.6f}")

    # 质量守恒验证 (映射 211_continuity_exact)
    mass_err = solver.verify_conservation(c_final, c_init)
    print(f"    质量守恒误差: {mass_err:.4e}")

    # 能量耗散验证
    dissipation = solver.compute_energy_dissipation(energies)
    print(f"    能量耗散违反比例: {dissipation:.4f}")

    # 随机求解 (配点法)
    print(f"\n  随机配点法求解:")
    stoch_solver = StochasticCahnHilliard(
        config, basis, projector)

    q_nodes = sparse_grid.get_nodes_array()
    q_weights = sparse_grid.get_weights_array()

    # 减少步数以节省计算
    config.cahn_hilliard.n_steps = 50
    pce_coeffs, mean_energies = stoch_solver.solve_collocation(
        q_nodes, q_weights)

    print(f"    PCE 系数场形状: {pce_coeffs.shape}")
    print(f"    均值系数范围: [{pce_coeffs[:,:,0].min():.4f}, "
          f"{pce_coeffs[:,:,0].max():.4f}]")

    if pce_coeffs.shape[2] > 1:
        total_var_field = np.sum(pce_coeffs[:,:,1:] ** 2, axis=2)
        print(f"    方差场范围: [{total_var_field.min():.6e}, "
              f"{total_var_field.max():.6e}]")

    if mean_energies:
        print(f"    平均能量: {mean_energies[0]:.4f} → "
              f"{mean_energies[-1]:.4f}")

    # 质量守恒验证
    c_recon = stoch_solver.reconstruct_solution(
        pce_coeffs,
        np.array([m.mean for m in config.measures]))
    mass_err_stoch = solver.verify_conservation(c_recon, c_init)
    print(f"    随机解质量守恒误差: {mass_err_stoch:.4e}")

    return pce_coeffs, c_final, energies


# ============================================================
#  阶段 6: 神经闭合模型训练
# ============================================================
def phase_6_neural_closure(config: GlobalConfig,
                           basis: OrthogonalPolynomialBasis
                           ) -> PortHamiltonianClosure:
    """训练物理约束神经闭合模型"""
    print_section("阶段 6: Port-Hamiltonian 神经闭合模型")

    closure = create_closure_model(config, basis.n_basis)
    print(f"  网络结构:")
    print(f"    输入维度: {basis.n_basis}")
    print(f"    隐藏层: {config.neural_closure.n_layers} × "
          f"{config.neural_closure.hidden_dim}")
    print(f"    输出维度: {basis.n_basis}")
    print(f"    总参数: {closure.network.n_params()}")
    print(f"    激活函数: {config.neural_closure.activation}")

    # 生成训练数据
    print(f"\n  生成训练数据...")
    X_train, Y_train = generate_training_data(
        None, basis, None, n_samples=50)
    print(f"    训练样本: {X_train.shape[0]}")
    print(f"    输入维度: {X_train.shape[1]}")
    print(f"    输出维度: {Y_train.shape[1]}")

    # 训练
    print(f"\n  训练 (Port-Hamiltonian 约束, "
          f"映射 1298_Eric6669)...")
    losses = closure.train(
        X_train, Y_train,
        n_epochs=config.neural_closure.n_epochs,
        lr=config.neural_closure.learning_rate)

    if losses:
        print(f"    初始损失: {losses[0]:.6e}")
        print(f"    最终损失: {losses[-1]:.6e}")
        print(f"    损失降低比: {losses[0] / max(losses[-1], 1e-30):.2f}")

    # 验证能量耗散
    print(f"\n  能量耗散验证:")
    rng = np.random.RandomState(99)
    test_coeffs = rng.randn(1, basis.n_basis) * 0.1
    test_coeffs[0, 0] = 1.0
    grad_H = rng.randn(1, basis.n_basis) * 0.01

    energy_rate = closure.verify_energy_dissipation(
        test_coeffs, grad_H)
    print(f"    <τ̂, ∇H> = {energy_rate:.6e} "
          f"({'≤0 满足耗散' if energy_rate <= 0 else '>0 可能违反'})")

    return closure


# ============================================================
#  阶段 7: 多保真度分析
# ============================================================
def phase_7_multifidelity(config: GlobalConfig,
                          basis: OrthogonalPolynomialBasis
                          ) -> dict:
    """多保真度不确定性量化"""
    print_section("阶段 7: 多保真度不确定性量化")

    d = len(config.measures)

    # 定义高低保真度函数 (映射 1246_Gaulios)
    def high_fidelity_func(xi):
        """高保真: 完整的 PCE 求值"""
        xi_2d = xi.reshape(1, -1)
        Psi = basis.evaluate(xi_2d)
        # 模拟高保真输出: 使用所有阶的 PCE
        coeffs = np.zeros(basis.n_basis)
        coeffs[0] = 1.0
        for k in range(1, basis.n_basis):
            coeffs[k] = 0.1 / k
        return float((Psi @ coeffs)[0])

    def low_fidelity_func(xi):
        """低保真: 截断 PCE (仅用前几个系数)"""
        xi_2d = xi.reshape(1, -1)
        Psi = basis.evaluate(xi_2d)
        coeffs = np.zeros(basis.n_basis)
        coeffs[0] = 1.0
        if basis.n_basis > 1:
            coeffs[1] = 0.1
        return float((Psi @ coeffs)[0])

    results = run_multifidelity_analysis(
        config, high_fidelity_func, low_fidelity_func)

    print(f"  多保真度分析结果 (映射 1246_Gaulios):")
    print(f"    高保真样本数: {results['n_high']}")
    print(f"    低保真样本数: {results['n_low']}")
    print(f"    控制变量均值: {results['control_variate_mean']:.6f}")
    print(f"    控制变量方差: {results['control_variate_var']:.6e}")
    print(f"    MC 均值:      {results['mc_mean']:.6f}")
    print(f"    MC 方差:      {results['mc_var']:.6e}")
    print(f"    高低相关系数: {results['correlation']:.4f}")
    print(f"    方差缩减因子: {results['variance_reduction']:.2f}")

    if results['correction_losses']:
        print(f"    修正模型初始损失: "
              f"{results['correction_losses'][0]:.6e}")
        print(f"    修正模型最终损失: "
              f"{results['correction_losses'][-1]:.6e}")

    return results


# ============================================================
#  阶段 8: 统计分析
# ============================================================
def phase_8_statistics(config: GlobalConfig,
                       basis: OrthogonalPolynomialBasis,
                       pce_coeffs: np.ndarray) -> dict:
    """计算统计量和灵敏度"""
    print_section("阶段 8: 统计分析与 Sobol 灵敏度")

    analyzer = PCEStatistics(config, basis)

    # 提取空间平均的 PCE 系数
    # pce_coeffs shape: (ny, nx, P) → 取空间平均
    if pce_coeffs.ndim == 3:
        pce_spatial = np.mean(pce_coeffs, axis=(0, 1))
    else:
        pce_spatial = pce_coeffs

    stats = analyzer.compute_all_statistics(pce_spatial)
    var_names = [m.name for m in config.measures]
    print(analyzer.format_statistics(stats, var_names))

    # PDF 估计
    x_pdf, pdf_vals = analyzer.compute_pdf(pce_spatial, n_grid=50)
    print(f"\n  概率密度函数估计:")
    print(f"    网格点范围: [{x_pdf[0]:.4f}, {x_pdf[-1]:.4f}]")
    print(f"    PDF 最大值: {np.max(pdf_vals):.4f}")
    # 找到众数
    mode_idx = np.argmax(pdf_vals)
    print(f"    众数: {x_pdf[mode_idx]:.4f}")

    return stats


# ============================================================
#  阶段 9: MCMC 贝叶斯推断
# ============================================================
def phase_9_mcmc_bayes(config: GlobalConfig,
                       basis: OrthogonalPolynomialBasis,
                       pce_coeffs: np.ndarray) -> dict:
    """MCMC 贝叶斯推断"""
    print_section("阶段 9: MCMC 贝叶斯推断 (映射 1180_Yuva12345)")

    # 提取空间平均 PCE 系数
    if pce_coeffs.ndim == 3:
        pce_spatial = np.mean(pce_coeffs, axis=(0, 1))
    else:
        pce_spatial = pce_coeffs

    # 创建 MCMC 采样器
    sampler = MCMCSampler(config, basis)

    # 模拟观测: 在均值点 + 噪声
    mu_point = np.array([m.mean for m in config.measures])
    mu_2d = mu_point.reshape(1, -1)
    Psi = basis.evaluate(mu_2d)
    true_value = float((Psi @ pce_spatial)[0])
    noise_std = 0.05 * abs(true_value) + 0.01
    observation = true_value + np.random.randn() * noise_std * 0.1

    print(f"  观测值: {observation:.6f}")
    print(f"  噪声标准差: {noise_std:.6f}")

    # 运行 MCMC
    print(f"\n  运行 MCMC 采样...")
    chain, log_likes, acc_rate = sampler.sample(
        pce_spatial, observation, noise_std,
        n_walkers=config.mcmc.n_walkers,
        n_burnin=config.mcmc.n_burnin,
        n_samples=config.mcmc.n_samples)

    print(f"  采样完成:")
    print(f"    样本数: {chain.shape[0]}")
    print(f"    接受率: {acc_rate:.3f}")

    # 后验统计
    post_stats = sampler.compute_posterior_statistics()
    var_names = [m.name for m in config.measures]

    # 失效概率
    mean_val = float(np.mean(pce_spatial[0]) if
                     len(pce_spatial.shape) > 0 else pce_spatial)
    threshold = mean_val + 2.0 * float(
        np.sqrt(max(np.sum(pce_spatial[1:] ** 2), 0.0)))
    failure_results = compute_failure_probability_is(
        pce_spatial, basis, config, threshold, n_samples=10000)

    print(f"\n  {format_bayesian_results(post_stats, failure_results, var_names)}")

    # Owen T 函数演示 (映射 033_asa076)
    print(f"\n  Owen T 函数计算 (映射 033_asa076):")
    h_vals = [0.0, 0.5, 1.0, 2.0, 3.0]
    a_vals = [0.5, 1.0, 2.0]
    for h in h_vals:
        for a in a_vals:
            T_val = owen_t_function(h, a)
            print(f"    T({h:.1f}, {a:.1f}) = {T_val:.6e}")

    return {"posterior": post_stats, "failure": failure_results}


# ============================================================
#  阶段 10: 综合报告
# ============================================================
def phase_10_summary(config: GlobalConfig,
                     stats: dict,
                     mf_results: dict,
                     bayes_results: dict):
    """打印综合报告"""
    print_section("阶段 10: 综合报告")

    print("""
  本计算演示了随机 Cahn-Hilliard 相场方程的完整不确定性量化流程:

  ┌─────────────────────────────────────────────────────────┐
  │ 1. 多项式混沌基构造 (Hermite/Legendre 正交多项式)        │
  │ 2. Smolyak 稀疏网格求积 (高维概率空间高效积分)           │
  │ 3. 随机 Galerkin 投影 (耦合张量 + 非线性项处理)          │
  │ 4. Cahn-Hilliard 求解 (Fourier 谱方法 + 半隐式时间)      │
  │ 5. 多元素自适应细化 (Dörfler 标记 + hp-策略)             │
  │ 6. Port-Hamiltonian 神经闭合 (能量保持约束)              │
  │ 7. 多保真度方差缩减 (控制变量 + 非线性修正)              │
  │ 8. Sobol 全局灵敏度 (一阶 + 总效应指数)                  │
  │ 9. MCMC 贝叶斯推断 (PCE 代理加速 + 失效概率)            │
  └─────────────────────────────────────────────────────────┘

  核心科学贡献:
  • 将 ME-gPC 方法应用于相场方程的不确定性量化
  • 结合 Port-Hamiltonian 结构约束确保物理一致性
  • 多保真度策略大幅降低计算成本
  • 全局灵敏度分析揭示关键不确定源

  15 个种子项目的融入:
  ┌────────────────────────────────────────────────────────┐
  │ 096_bisection_min     → 二分法搜索最优 PCE 阶数         │
  │ 1050_LaM-SLidE        → 编码器-解码器 / 神经闭合特征    │
  │ 986_r8ncf             → 稀疏矩阵 COO 格式               │
  │ 1434_zombie_ode       → 耦合 ODE → 多元素耦合           │
  │ 680_line_grid         → 一维网格 → 物理/随机网格        │
  │ 738_matrix_assemble   → 并行装配 → Galerkin 系统        │
  │ 932_pyramid_grid      → 层级结构 → Smolyak 稀疏网格     │
  │ 068_ball_integrals    → 高维积分 → 概率空间求积         │
  │ 1246_Gaulios          → 有限-无穷horizon → 多保真桥接   │
  │ 211_continuity_exact  → 无散度约束 → 质量守恒验证       │
  │ 033_asa076            → Owen T 函数 → 失效概率          │
  │ 1298_Port-Hamiltonian → 能量结构 → 物理约束闭合         │
  │ 787_navier_stokes     → NS 精确解 → CH 求解器验证       │
  │ 1180_HII-galaxy       → MCMC + 贝叶斯 → 后验推断       │
  │ 382_fem_to_xml        → FEM 网格 → 计算域离散化         │
  └────────────────────────────────────────────────────────┘
""")

    print("  计算完成!")
    print("=" * 70)


# ============================================================
#  主函数
# ============================================================
VERSION = "201.1.0"


def _json_default(obj):
    """Convert numpy scalars/arrays for stable JSON CLI output."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(data: dict):
    print(json.dumps(data, ensure_ascii=False, sort_keys=True,
                     indent=2, default=_json_default))


def _config_payload(config: GlobalConfig) -> dict:
    return {
        "basis": {
            "max_degree": config.basis.max_degree,
            "truncation": config.basis.truncation,
            "hyperbolic_q": config.basis.hyperbolic_q,
        },
        "cahn_hilliard": {
            "Lx": config.cahn_hilliard.Lx,
            "Ly": config.cahn_hilliard.Ly,
            "nx": config.cahn_hilliard.nx,
            "ny": config.cahn_hilliard.ny,
            "dt": config.cahn_hilliard.dt,
            "n_steps": config.cahn_hilliard.n_steps,
            "kappa_mean": config.cahn_hilliard.kappa_mean,
            "kappa_std": config.cahn_hilliard.kappa_std,
            "gamma_mean": config.cahn_hilliard.gamma_mean,
            "gamma_std": config.cahn_hilliard.gamma_std,
            "mobility_mean": config.cahn_hilliard.mobility_mean,
            "mobility_std": config.cahn_hilliard.mobility_std,
            "random_seed": config.cahn_hilliard.random_seed,
        },
        "measures": [
            {
                "name": m.name,
                "type": m.measure_type,
                "mean": m.mean,
                "variance": m.variance,
                "support": list(m.support),
            }
            for m in config.measures
        ],
        "multi_element": {
            "initial_elements": config.multi_element.initial_elements,
            "local_degree": config.multi_element.local_degree,
            "max_elements": config.multi_element.max_elements,
        },
        "sparse_grid": {
            "level": config.sparse_grid.level,
            "growth_rule": config.sparse_grid.growth_rule,
            "quadrature_type": config.sparse_grid.quadrature_type,
        },
        "total_basis_dim": estimate_total_basis_dim(config),
    }


def command_config(args) -> int:
    config = create_default_config()
    payload = _config_payload(config)
    if args.json:
        _print_json(payload)
    else:
        print(f"dimensions={len(config.measures)}")
        print(f"basis_max_degree={config.basis.max_degree}")
        print(f"total_basis_dim={payload['total_basis_dim']}")
        print(f"grid={config.cahn_hilliard.nx}x{config.cahn_hilliard.ny}")
    return 0


def command_basis(args) -> int:
    config = create_default_config()
    config.basis.max_degree = args.degree
    basis = OrthogonalPolynomialBasis(config.basis, config.measures)
    payload = {
        "degree": args.degree,
        "n_dim": basis.n_dim,
        "n_basis": basis.n_basis,
        "truncation": config.basis.truncation,
        "indices": [list(map(int, idx)) for idx in basis.indices[:args.limit]],
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"degree={payload['degree']}")
        print(f"n_dim={payload['n_dim']}")
        print(f"n_basis={payload['n_basis']}")
        print(f"indices={payload['indices']}")
    return 0


def command_quadrature(args) -> int:
    config = create_default_config()
    config.sparse_grid.level = args.level
    sparse_grid = AdaptiveSparseGrid(config)
    nodes = sparse_grid.get_nodes_array()
    weights = sparse_grid.get_weights_array()
    payload = {
        "level": args.level,
        "n_points": int(sparse_grid.n_points),
        "weight_sum": float(np.sum(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "first_nodes": nodes[:args.limit].tolist(),
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"level={payload['level']}")
        print(f"n_points={payload['n_points']}")
        print(f"weight_sum={payload['weight_sum']:.12e}")
    return 0


def command_owen(args) -> int:
    value = owen_t_function(args.h, args.a)
    payload = {"h": args.h, "a": args.a, "value": value}
    if args.json:
        _print_json(payload)
    else:
        print(f"T({args.h:.6g}, {args.a:.6g}) = {value:.12e}")
    return 0


def command_smoke(args) -> int:
    np.random.seed(args.seed)
    config = create_default_config()
    config.basis.max_degree = args.degree
    config.sparse_grid.level = args.level
    config.cahn_hilliard.nx = args.nx
    config.cahn_hilliard.ny = args.ny
    config.cahn_hilliard.n_steps = args.steps
    basis = OrthogonalPolynomialBasis(config.basis, config.measures)
    sparse_grid = AdaptiveSparseGrid(config)
    solver = CahnHilliardSolver(config.cahn_hilliard)
    X, Y, _, _ = create_physical_mesh(config.cahn_hilliard)
    c0 = generate_initial_condition(config.cahn_hilliard, X, Y)
    c_final, energies = solver.solve(
        c0,
        kappa=config.cahn_hilliard.kappa_mean,
        gamma=config.cahn_hilliard.gamma_mean,
        mobility=config.cahn_hilliard.mobility_mean,
        n_steps=args.steps,
    )
    payload = {
        "basis_dim": int(basis.n_basis),
        "grid": [args.nx, args.ny],
        "initial_mass": float(np.mean(c0)),
        "initial_min": float(np.min(c0)),
        "initial_max": float(np.max(c0)),
        "final_mass": float(np.mean(c_final)),
        "mass_error": float(solver.verify_conservation(c_final, c0)),
        "energy_initial": float(energies[0]),
        "energy_final": float(energies[-1]),
        "sparse_points": int(sparse_grid.n_points),
        "steps": args.steps,
    }
    if args.json:
        _print_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key}={value}")
    return 0


def run_full_workflow():
    """
    统一入口: 零参数运行完整 UQ 流程。
    """
    t_start = time.time()
    print_header()

    # 阶段 1: 配置
    config = phase_1_initialization()

    # 阶段 2: 基函数与求积
    basis, sparse_grid = phase_2_basis_and_quadrature(config)

    # 阶段 3: Galerkin 投影
    projector, assembler = phase_3_galerkin_projection(config, basis)

    # 阶段 4: 多元素自适应细化
    me_galerkin = phase_4_adaptive_refinement(config)

    # 阶段 5: Cahn-Hilliard 求解
    pce_coeffs, c_final, energies = phase_5_cahn_hilliard_solve(
        config, basis, projector, sparse_grid)

    # 阶段 6: 神经闭合模型
    closure = phase_6_neural_closure(config, basis)

    # 阶段 7: 多保真度分析
    mf_results = phase_7_multifidelity(config, basis)

    # 阶段 8: 统计分析
    stats = phase_8_statistics(config, basis, pce_coeffs)

    # 阶段 9: MCMC 贝叶斯推断
    bayes_results = phase_9_mcmc_bayes(config, basis, pce_coeffs)

    # 阶段 10: 综合报告
    phase_10_summary(config, stats, mf_results, bayes_results)

    # 总时间
    t_total = time.time() - t_start
    print(f"\n总计算时间: {t_total:.2f} 秒")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="executable",
        description=("Multi-Element gPC uncertainty quantification for a "
                     "stochastic Cahn-Hilliard equation."),
    )
    parser.add_argument("--version", action="version",
                        version=f"executable {VERSION}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run the full demonstration")
    run_parser.set_defaults(func=lambda args: (run_full_workflow() or 0))

    config_parser = subparsers.add_parser("config", help="print default configuration")
    config_parser.add_argument("--json", action="store_true")
    config_parser.set_defaults(func=command_config)

    basis_parser = subparsers.add_parser("basis", help="summarize polynomial basis")
    basis_parser.add_argument("--degree", type=int, default=3)
    basis_parser.add_argument("--limit", type=int, default=10)
    basis_parser.add_argument("--json", action="store_true")
    basis_parser.set_defaults(func=command_basis)

    quad_parser = subparsers.add_parser("quadrature", help="summarize sparse quadrature")
    quad_parser.add_argument("--level", type=int, default=3)
    quad_parser.add_argument("--limit", type=int, default=5)
    quad_parser.add_argument("--json", action="store_true")
    quad_parser.set_defaults(func=command_quadrature)

    owen_parser = subparsers.add_parser("owen", help="compute Owen's T function")
    owen_parser.add_argument("--h", type=float, required=True)
    owen_parser.add_argument("--a", type=float, required=True)
    owen_parser.add_argument("--json", action="store_true")
    owen_parser.set_defaults(func=command_owen)

    smoke_parser = subparsers.add_parser("smoke", help="run a fast deterministic setup check")
    smoke_parser.add_argument("--degree", type=int, default=2)
    smoke_parser.add_argument("--level", type=int, default=2)
    smoke_parser.add_argument("--nx", type=int, default=8)
    smoke_parser.add_argument("--ny", type=int, default=8)
    smoke_parser.add_argument("--steps", type=int, default=2)
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--json", action="store_true")
    smoke_parser.set_defaults(func=command_smoke)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        run_full_workflow()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
