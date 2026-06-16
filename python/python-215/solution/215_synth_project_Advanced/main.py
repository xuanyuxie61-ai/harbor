#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py — 统一入口: 多目标拓扑优化催化反应器

科学问题:
  设计一个一维催化反应器结构 (密度分布 ρ(x)),
  同时优化三个相互竞争的目标:
    f1: 最小化结构柔度 (最大化刚度)
    f2: 最小化热应力方差 (均匀温度分布)
    f3: 最大化生化转化效率

  使用 NSGA-II 多目标进化算法求解 Pareto 前沿.

运行方式:
    python main.py

作者: Sci-Project-Synthesis (DA workflow)
"""

import numpy as np
import sys
import time

# 项目内模块
from objective_functions import MultiObjectiveEvaluator, verify_quadrature_accuracy
from nsga2_optimizer import NSGA2Optimizer
from pareto_core import (
    extract_pareto_front, fast_non_dominated_sort,
    crowding_distance, spacing_metric, generational_distance,
    hypervolume_2d
)
from stochastic_robustness import (
    verify_ou_statistics, fidelity_witness, fidelity_matrix,
    UncertaintyQuantifier, gpc_expansion, gpc_project, hermite_polynomial
)
from quadrature_engine import (
    pyramid_unit_nodes_weights, pyramid_monomial_integral,
    disk_monomial_integral, disk_sample_uniform,
    gauss_legendre_1d, gauss_legendre_2d, integrate_2d
)
from adaptive_rk_integrator import (
    rk45_adaptive, ornstein_uhlenbeck_em, ou_analytical_moments,
    butchers_rk4, butchers_dormand_prince
)
from cvt_design_sampler import (
    cvt_lloyd_sampling, halton_sequence, sieve_of_eratosthenes,
    fresnel_cos, fresnel_sin, fresnel_phase,
    sample_design_space
)
from coupled_pde_system import (
    MaterialModel, BiochemicalKinetics,
    solve_thermal_field, coupled_solve, analyze_matrix_structure,
    fem1d_assemble
)
from pram_parallel import PRAMConfig, ParallelEvaluator, tile_configurations


def print_separator(title: str = "", width: int = 70):
    """打印分隔线."""
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * width)
        print("=" * max(pad, 1) + f" {title} " + "=" * max(pad, 1))
        print("=" * width)
    else:
        print("=" * width)


def main():
    """
    主流程:
      Phase 1: 验证基础数值模块 (积分、ODE、矩阵)
      Phase 2: 构建多目标优化问题
      Phase 3: NSGA-II 进化求解 Pareto 前沿
      Phase 4: 分析 Pareto 前沿质量
      Phase 5: 鲁棒性验证与保真度评估
    """
    t_start = time.time()

    print_separator("催化反应器多目标拓扑优化 — 博士级科学计算项目")
    print("""
    科学问题:
      设计一维催化反应器的密度分布 ρ(x) ∈ [0,1]^n,
      同时优化:
        f1: 结构柔度    (最小化 → 最大化刚度)
        f2: 热应力方差  (最小化 → 均匀温度分布)
        f3: 负转化率    (最小化 → 最大化生化转化效率)

      约束: 体积分数 ≤ 0.5
      方法: NSGA-II + 顺序耦合 PDE + 随机鲁棒性
    """)

    # ==================================================================
    # Phase 1: 基础数值模块验证
    # ==================================================================
    print_separator("Phase 1: 基础数值模块验证")

    # 1.1 积分验证
    print("\n[1.1] 数值积分精度验证:")
    quad_results = verify_quadrature_accuracy()
    for k, v in quad_results.items():
        print(f"  {k:30s} = {v:.8e}")

    # 1.2 Gauss 节点
    print("\n[1.2] Gauss-Legendre 节点 (5阶):")
    nodes, weights = gauss_legendre_1d(5)
    for i in range(5):
        print(f"  x_{i+1} = {nodes[i]:+.8f},  w_{i+1} = {weights[i]:.8f}")
    print(f"  权重之和 = {np.sum(weights):.12f} (应为 2.0)")

    # 1.3 ODE 验证 — 自适应 RK45
    print("\n[1.3] 自适应 RK45 求解验证 (指数衰减):")

    def exp_decay(t, y):
        return np.array([-y[0]])

    t_span = (0, 5)
    y0 = np.array([1.0])
    t_ode, y_ode = rk45_adaptive(exp_decay, t_span, y0, tol=1e-8)
    y_exact = np.exp(-t_ode)
    err_ode = np.max(np.abs(y_ode[:, 0] - y_exact))
    print(f"  最大绝对误差: {err_ode:.2e}")
    print(f"  步数: {len(t_ode)}")

    # 1.4 OU 过程验证
    print("\n[1.4] Ornstein-Uhlenbeck 过程统计验证:")
    ou_stats = verify_ou_statistics(
        theta=2.0, mu=0.0, sigma=1.0, x0=1.0,
        tmax=3.0, n_steps=5000, n_paths=100
    )
    t_end = ou_stats['t'][-1]
    mean_ana = ou_stats['mean_analytical'][-1]
    mean_num = ou_stats['mean_numerical'][-1]
    var_ana = ou_stats['var_analytical'][-1]
    var_num = ou_stats['var_numerical'][-1]
    print(f"  t = {t_end:.1f}:")
    print(f"    均值: 解析 = {mean_ana:.6f}, 数值 = {mean_num:.6f}")
    print(f"    方差: 解析 = {var_ana:.6f}, 数值 = {var_num:.6f}")

    # 1.5 生化动力学
    print("\n[1.5] 生化反应动力学 (Michaelis-Menten):")
    kinetics = BiochemicalKinetics()
    T_test = 350.0
    k_test = kinetics.rate_constant(T_test)
    r_test = kinetics.reaction_rate(0.5, T_test)
    print(f"  T = {T_test} K: k(T) = {k_test:.4e} 1/s")
    print(f"  c = 0.5 mol/m³: r = {r_test:.4e} mol/(m³·s)")

    # 1.6 矩阵分析
    print("\n[1.6] FEM 刚度矩阵结构分析:")
    material = MaterialModel()
    n_elem = 20
    K_test, F_test, x_test = fem1d_assemble(
        n_elem, 1.0, material, np.ones(n_elem)
    )
    mat_info = analyze_matrix_structure(K_test)
    for k, v in mat_info.items():
        print(f"  {k}: {v}")

    # 1.7 CVT 采样
    print("\n[1.7] CVT 采样收敛 (Lloyd 算法):")
    gen, energy_hist = cvt_lloyd_sampling(
        n_generators=10, n_samples=2000, n_iterations=30, dimension=1
    )
    print(f"  初始能量: {energy_hist[0]:.4f}")
    print(f"  最终能量: {energy_hist[-1]:.4f}")
    print(f"  能量比:   {energy_hist[-1]/energy_hist[0]:.4f}")
    print(f"  生成点:   {np.sort(gen.ravel())}")

    # 1.8 素数 & Halton 序列
    print("\n[1.8] 素数筛法 & Halton 序列:")
    primes = sieve_of_eratosthenes(50)
    print(f"  ≤50 的素数: {primes}")
    halton_pts = halton_sequence(5, 2)
    print(f"  Halton 2D 前5点:\n{halton_pts}")

    # 1.9 Fresnel 积分
    print("\n[1.9] Fresnel 积分验证:")
    x_f = np.array([0.5, 1.0, 1.5, 2.0])
    C_val = fresnel_cos(x_f)
    S_val = fresnel_sin(x_f)
    for i, x in enumerate(x_f):
        print(f"  x={x:.1f}: C(x)={C_val[i]:.6f}, S(x)={S_val[i]:.6f}")

    # ==================================================================
    # Phase 2: 多目标优化
    # ==================================================================
    print_separator("Phase 2: 多目标优化问题构建")

    n_elements = 15  # 设计变量维度
    L_domain = 1.0   # 反应器长度

    print(f"\n  设计变量维度: {n_elements}")
    print(f"  反应器长度:   {L_domain} m")
    print(f"  体积分数约束: ≤ 0.5")

    evaluator = MultiObjectiveEvaluator(
        n_elements=n_elements,
        L=L_domain,
        volume_fraction_max=0.5,
        use_robust=False,  # 确定性评估以加速
        seed=42
    )

    # 测试单个设计
    print("\n  单设计评估测试 (均匀 ρ=0.5):")
    rho_test = np.full(n_elements, 0.5)
    res_test = evaluator.evaluate_deterministic(rho_test)
    for k, v in res_test.items():
        print(f"    {k:25s} = {v:.6f}")

    # ==================================================================
    # Phase 3: NSGA-II 进化求解
    # ==================================================================
    print_separator("Phase 3: NSGA-II 进化求解 Pareto 前沿")

    lower = np.full(n_elements, 0.05)
    upper = np.full(n_elements, 0.95)

    # PRAM 并行配置
    pram_config = PRAMConfig(n_workers=2, block_size=10, strategy='CREW')
    print(f"\n  PRAM 配置: workers={pram_config.n_workers}, "
          f"block_size={pram_config.block_size}, "
          f"strategy={pram_config.strategy}")

    tiles = tile_configurations(30, max_tile_size=10)
    print(f"  分块方案数: {len(tiles)}")

    pop_size = 30
    n_gens = 10

    print(f"\n  种群大小:  {pop_size}")
    print(f"  进化代数:  {n_gens}")

    optimizer = NSGA2Optimizer(
        pop_size=pop_size,
        n_generations=n_gens,
        eta_c=20.0,
        eta_m=20.0,
        crossover_prob=0.9,
        lower=lower,
        upper=upper,
        seed=42
    )

    def eval_wrapper(rho):
        return evaluator.get_objective_vector(rho, use_robust=False)

    print("\n  开始 NSGA-II 进化...")
    t_opt_start = time.time()
    pareto_set, pareto_front = optimizer.optimize(eval_wrapper)
    t_opt_end = time.time()

    print(f"\n  优化耗时: {t_opt_end - t_opt_start:.2f} s")
    print(f"  总评估次数: {evaluator.eval_count}")
    print(f"  Pareto 前沿大小: {len(pareto_front)}")

    # ==================================================================
    # Phase 4: Pareto 前沿分析
    # ==================================================================
    print_separator("Phase 4: Pareto 前沿质量分析")

    # 4.1 前沿展示
    print("\n[4.1] Pareto 前沿 (归一化目标值):")
    print(f"  {'#':>3s}  {'f1 柔度':>10s}  {'f2 热方差':>10s}  {'f3 转化率':>10s}")
    print(f"  {'-'*3:>3s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")
    for i in range(min(len(pareto_front), 10)):
        f = pareto_front[i]
        print(f"  {i+1:3d}  {f[0]:10.4f}  {f[1]:10.4f}  {f[2]:10.4f}")

    # 4.2 质量指标
    print("\n[4.2] Pareto 前沿质量指标:")
    sp = spacing_metric(pareto_front)
    print(f"  间距指标 (Spacing):   {sp:.6f}")

    # 理想点和 nadir
    f_ideal = np.min(pareto_front, axis=0)
    f_nadir = np.max(pareto_front, axis=0)
    print(f"  理想点 (utopia):  {f_ideal}")
    print(f"  Nadir 点:         {f_nadir}")

    # 2D 超体积 (f1 vs f2)
    if len(pareto_front) >= 2:
        ref_2d = f_nadir[:2] * 1.1
        hv = hypervolume_2d(pareto_front[:, :2], ref_2d)
        print(f"  2D 超体积 (f1-f2):  {hv:.6f}")

    # 4.3 保真度 witness
    print("\n[4.3] 保真度 Witness (各解与理想解的接近度):")
    fw_vals = fidelity_matrix(pareto_front, f_ideal, f_nadir)
    for i in range(min(len(fw_vals), 10)):
        print(f"  解 {i+1}: F_w = {fw_vals[i]:.6f}")
    print(f"  平均保真度: {np.mean(fw_vals):.6f}")

    # ==================================================================
    # Phase 5: 鲁棒性 & 补充验证
    # ==================================================================
    print_separator("Phase 5: 鲁棒性验证与补充分析")

    # 5.1 鲁棒评估最佳解
    print("\n[5.1] 最优解的鲁棒性评估 (含制造不确定性):")
    best_idx = np.argmin(np.sum(
        (pareto_front - f_ideal) / (f_nadir - f_ideal + 1e-10), axis=1
    ))
    rho_best = pareto_set[best_idx]
    print(f"  最佳解索引: {best_idx}")
    print(f"  最佳密度分布: {np.round(rho_best, 3)}")

    res_det = evaluator.evaluate_deterministic(rho_best)
    print(f"\n  确定性目标:")
    print(f"    f1 = {res_det['f1_compliance']:.6f}")
    print(f"    f2 = {res_det['f2_thermal_var']:.6f}")
    print(f"    f3 = {res_det['f3_neg_yield']:.6f}")
    print(f"    转化率 = {res_det['yield']:.4f}")
    print(f"    体积分数 = {res_det['volume_fraction']:.4f}")

    # 5.2 gPC 分析
    print("\n[5.2] gPC (广义多项式混沌) 不确定性展开:")
    uq = UncertaintyQuantifier(sigma_ou=0.05, theta_ou=2.0,
                               beta_risk=2.0, n_mc=20, seed=42)
    rho_pert = uq.generate_perturbations(rho_best, n_realizations=20)
    f1_samples = np.array([
        evaluator.evaluate_deterministic(rp)['f1_compliance']
        for rp in rho_pert
    ])
    xi_samples = np.random.default_rng(42).standard_normal(20)
    gpc_coeffs = gpc_project(f1_samples, xi_samples, max_order=3)
    print(f"  gPC 系数 (0-3阶): {np.round(gpc_coeffs, 4)}")
    print(f"  f1 样本均值: {np.mean(f1_samples):.6f}")
    print(f"  f1 样本标准差: {np.std(f1_samples):.6f}")
    f1_rob, f1_std = uq.robust_objective(f1_samples)
    print(f"  鲁棒目标 (μ+βσ): {f1_rob:.6f}  (β={uq.beta_risk})")

    # 5.3 温度场可视化 (文本)
    print("\n[5.3] 最优设计的温度场:")
    result = coupled_solve(n_elements, L_domain,
                           MaterialModel(), BiochemicalKinetics(),
                           rho_best)
    x_grid = result['x']
    T_grid = result['T']
    c_grid = result['c']
    print(f"  {'x':>6s}  {'T(K)':>8s}  {'c(mol/m³)':>10s}")
    stride = max(1, len(x_grid) // 8)
    for i in range(0, len(x_grid), stride):
        print(f"  {x_grid[i]:6.3f}  {T_grid[i]:8.2f}  {c_grid[i]:10.6f}")

    # ==================================================================
    # 总结
    # ==================================================================
    t_total = time.time() - t_start
    print_separator("计算完成 — 总结")
    print(f"""
  科学问题: 催化反应器多目标拓扑优化
  设计变量: {n_elements} 维密度分布 ρ ∈ [0,1]^{n_elements}
  目标数:   3 (柔度、热应力方差、转化率)
  优化方法: NSGA-II (种群={pop_size}, 代数={n_gens})
  Pareto 前沿大小: {len(pareto_front)}

  关键结果:
    - 理想点 (utopia): {np.round(f_ideal, 4)}
    - Nadir 点:        {np.round(f_nadir, 4)}
    - 平均保真度:      {np.mean(fw_vals):.4f}
    - 间距指标:        {sp:.4f}
    - 最优转化率:      {res_det['yield']:.4f}
    - 最优体积分数:    {res_det['volume_fraction']:.4f}

  总计算时间: {t_total:.2f} s

  融合的种子项目:
    931 (金字塔求积)  →  高阶数值积分引擎
    946 (2D求积)      →  二维 Gauss 积分
    294 (圆盘积分)    →  圆盘域解析积分
    1221 (SGF)        →  催化剂退化建模
    910 (素数)        →  Halton 低差异序列
    839 (OU过程)      →  随机鲁棒性分析
    091 (生化ODE)     →  反应动力学
    737 (矩阵分析)    →  FEM 矩阵结构分析
    906 (PRAM)        →  并行评估协调
    972 (Butcher表)   →  自适应 RK 积分
    909 (捕食-被捕食) →  Pareto 竞争动态
    385 (FEM1D)       →  有限元离散化
    246 (CVT采样)     →  最优设计空间采样
    448 (Fresnel)     →  波场品质目标
    1109 (Fidelity)   →  Pareto 前沿保真度
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
