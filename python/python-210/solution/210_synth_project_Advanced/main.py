#!/usr/bin/env python3
"""
main.py - 不确定性量化：可靠性分析与失效概率计算

本程序实现一个完整的结构可靠性分析流程，融合以下核心算法：

1. FORM (一阶可靠性方法): Hasofer-Lind 迭代寻找设计验算点
2. SORM (二阶可靠性方法): Breitung 曲率修正
3. 自适应重要性采样: Lloyd-CVT 生成最优采样点
4. 多项式混沌展开: Chebyshev 配点 + Sobol 灵敏度
5. 压缩感知: L1 稀疏灵敏度识别
6. 随机有限元: KL 展开 + FEM 反应扩散
7. 渐近尾概率: Fresnel 积分 + Laplace 近似
8. 数据驱动校准: MLE + BIC 模型选择

科学问题:
  计算含随机参数的结构系统的失效概率 P_f = P(g(u) <= 0),
  其中 g 为非线性极限状态函数, u 为标准正态空间中的随机向量。

种子项目 (15个):
  014_approx_chebyshev   → PCE 基函数构建
  377_fem_neumann        → 随机 FEM
  247_cvt_2d_lumping     → CVT-Lloyd 自适应采样
  855_pdflib             → 概率分布库
  345_exm                → 动力系统可靠性
  241_cvt_3_movie        → 拒绝采样
  468_geometry           → 失效面几何分析
  252_cvt_box            → CVT 能量优化
  448_fresnel            → Fresnel 渐近
  918_prob               → 完备分布工具
  802_newton_rc          → Newton 逆向通信
  1165_GEMS              → 数据校准
  182_circle_distance    → 球面采样
  223_counterfeit        → 压缩感知
  820_numgrid            → 空间网格
"""

import sys
import os
import numpy as np

# 确保当前目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_variables import NormalRV, LognormalRV, GammaRV, WeibullRV, std_normal_cdf
from limit_states import (LinearLimitState, NonlinearLimitState,
                          QuarticLimitState, SeriesSystemLimitState,
                          StressStrengthLimitState, ExponentialCosLimitState)
from reliability import FORMSolver, sorm_breitung, NewtonRaphsonRC
from adaptive_sampling import AdaptiveImportanceSampler, cvt_energy, cvt_lloyd_2d
from polynomial_chaos import (PolynomialChaosExpansion, chebyshev_nodes,
                               divided_differences, newton_interp_eval)
from asymptotic import fresnel_integrals, laplace_asymptotic_pf, mills_ratio
from stochastic_fem import KarhunenLoeveExpansion, StochasticReactionDiffusion
from sensitivity import (SobolSensitivity, CompressedSensingSensitivity,
                          PearsonCorrelation, TornadoSensitivity)
from spatial_mesh import numgrid, FailureDomainAnalyzer, sample_in_region
from calibration import DistributionFitter, ModelSelection, KernelDensityReliability


def _sep(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def run_form_analysis(lsf, n_dim, name=""):
    """FORM 分析"""
    _sep(f"FORM 一阶可靠性分析 {name}")
    solver = FORMSolver(lsf, n_dim, tol=1e-8, max_iter=200)
    solver.solve()

    print(f"  可靠度指标 beta   = {solver.beta:.6f}")
    print(f"  失效概率 P_f      = {solver.pf:.6e}")
    print(f"  设计验算点 u*     = [{', '.join(f'{v:.4f}' for v in solver.u_star)}]")
    print(f"  方向余弦 alpha    = [{', '.join(f'{v:.4f}' for v in solver.alpha)}]")
    print(f"  收敛              = {solver.converged}")
    print(f"  迭代次数          = {len(solver.history['beta']) - 1}")
    imp = solver.importance_factors()
    if imp is not None:
        print(f"  重要性因子 α²     = [{', '.join(f'{v:.4f}' for v in imp)}]")
    return solver


def run_sorm_analysis(lsf, u_star, beta, name=""):
    """SORM 分析"""
    _sep(f"SORM 二阶可靠性修正 {name}")
    pf_sorm, curvatures, correction = sorm_breitung(lsf, u_star, beta)
    pf_form = std_normal_cdf(-beta)

    print(f"  FORM P_f          = {pf_form:.6e}")
    print(f"  SORM P_f          = {pf_sorm:.6e}")
    print(f"  修正因子          = {correction:.6f}")
    print(f"  主曲率            = [{', '.join(f'{k:.4f}' for k in curvatures[:5])}]")
    return pf_sorm


def run_adaptive_is(lsf, n_dim, form_u_star=None):
    """自适应重要性采样"""
    _sep("自适应 CVT 重要性采样")
    ais = AdaptiveImportanceSampler(lsf, n_dim, n_generators=max(15, 3 * n_dim), rng_seed=42)
    result = ais.adaptive_run(n_samples=15000, max_outer=3, cov_target=0.05, cvt_iter=15,
                               form_design_point=form_u_star)

    print(f"  失效概率 P_f      = {result['pf']:.6e}")
    print(f"  变异系数 COV      = {result['cov']:.4f}")
    print(f"  CVT 生成点数      = {ais.n_generators}")
    print(f"  外层迭代          = {len(result['pf_history'])}")

    # CVT 能量收敛
    if ais.energy_history is not None and len(ais.energy_history) > 0:
        print(f"  CVT 初始能量      = {ais.energy_history[0]:.4f}")
        print(f"  CVT 最终能量      = {ais.energy_history[-1]:.4f}")
    return result


def run_pce_analysis(lsf, n_dim):
    """多项式混沌展开"""
    _sep("多项式混沌展开 (PCE) 代理模型")
    degree = min(3, max(2, 6 - n_dim))
    pce = PolynomialChaosExpansion(n_dim, degree)
    pce.fit_collocation(lsf.evaluate, n_quad=degree + 2)

    print(f"  维度              = {n_dim}")
    print(f"  多项式阶数        = {degree}")
    print(f"  PCE 项数          = {pce.n_terms}")
    print(f"  PCE 均值 E[Y]     = {pce.mean():.6f}")
    print(f"  PCE 方差 Var[Y]   = {pce.variance():.6f}")

    S1 = pce.sobol_indices()
    ST = pce.total_sobol_indices()
    print(f"  一阶 Sobol S_i    = [{', '.join(f'{s:.4f}' for s in S1)}]")
    print(f"  全阶 Sobol S_Ti   = [{', '.join(f'{s:.4f}' for s in ST)}]")
    return pce


def run_fresnel_analysis():
    """Fresnel 积分验证"""
    _sep("Fresnel 积分与渐近尾概率")
    test_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    print(f"  {'x':>6s}  {'C(x)':>12s}  {'S(x)':>12s}")
    for x in test_vals:
        c, s = fresnel_integrals(x)
        print(f"  {x:6.1f}  {c:12.8f}  {s:12.8f}")

    # Mill's ratio
    print(f"\n  Mill's 比率 R(x):")
    for x in [1.0, 2.0, 3.0, 5.0]:
        r = mills_ratio(x)
        print(f"    R({x:.1f}) = {r:.8f}")


def run_stochastic_fem():
    """随机有限元"""
    _sep("随机有限元: KL 展开 + 反应扩散")

    # KL 展开
    kl = KarhunenLoeveExpansion(
        n_elements=30, domain_length=1.0,
        sigma_field=0.3, correlation_length=0.2, n_terms=10
    )
    print(f"  KL 展开项数       = {kl.n_terms}")
    cum_var = kl.explained_variance_ratio()
    print(f"  方差解释 (前5项)  = [{', '.join(f'{v:.4f}' for v in cum_var[:5])}]")

    # 随机反应扩散
    srd = StochasticReactionDiffusion(n_elements=20)
    c_array = [0.1, -0.5, 0.0, 0.0]
    t_det, w_det = srd.solve_deterministic(c_array, t_final=0.5, n_steps=50)
    print(f"\n  确定性反应扩散:")
    print(f"    最终时间        = {t_det[-1]:.3f}")
    print(f"    最大 |w|        = {np.max(np.abs(w_det[-1])):.6f}")

    # MC 求解
    t_vals = np.linspace(0, 0.3, 20)
    mean_w, std_w = srd.solve_stochastic_mc(kl, c_array, n_mc=10,
                                              t_final=0.3, n_steps=20)
    print(f"\n  随机 MC (10 样本):")
    print(f"    最终 mean(|w|)  = {np.mean(np.abs(mean_w[-1])):.6f}")
    print(f"    最终 max(std)   = {np.max(std_w[-1]):.6f}")
    return kl


def run_sensitivity_analysis(lsf, n_dim):
    """灵敏度分析"""
    _sep("灵敏度分析")

    # Pearson 相关
    rng = np.random.default_rng(42)
    n_samp = 2000
    u_samp = rng.standard_normal((n_samp, n_dim))
    g_vals = lsf.evaluate(u_samp)
    rho = PearsonCorrelation.compute(u_samp, g_vals)
    print(f"  Pearson 相关:     [{', '.join(f'{r:.4f}' for r in rho)}]")

    # 龙卷风灵敏度
    tornado = TornadoSensitivity.compute(lsf, n_dim, delta=0.5)
    print(f"  龙卷风 OAT:       [{', '.join(f'{t:.4f}' for t in tornado)}]")

    # 压缩感知
    cs = CompressedSensingSensitivity(n_dim, n_measurements=max(2 * n_dim, 8), rng_seed=42)
    cs.build_sensing_matrix()
    obs = cs.finite_difference_sensitivity(lsf, delta=0.5)
    s_sparse = cs.l1_sensitivity_recovery(obs, epsilon=0.01)
    print(f"  L1 稀疏灵敏度:    [{', '.join(f'{s:.4f}' for s in s_sparse)}]")


def run_spatial_analysis(lsf, n_dim):
    """失效域几何分析"""
    _sep("失效域几何与空间网格")

    # numgrid 网格
    if n_dim == 2:
        G, x, y, pts = numgrid('D', 30)
        n_active = int(np.max(G))
        print(f"  圆盘网格 (n=30):  有效点 = {n_active}")

    # 失效域分析
    analyzer = FailureDomainAnalyzer(lsf, n_dim)
    fail_vol = analyzer.estimate_failure_volume(n_samples=3000, bound=4.0)
    boundary_dist = analyzer.estimate_boundary_distance(n_directions=50)
    centroid = analyzer.estimate_failure_centroid(n_samples=3000, bound=4.0)

    print(f"  失效域体积估计    = {fail_vol:.4f}")
    print(f"  边界距离 (beta)   = {boundary_dist:.4f}")
    print(f"  失效域重心        = [{', '.join(f'{c:.4f}' for c in centroid)}]")


def run_calibration_demo():
    """数据驱动校准"""
    _sep("数据驱动模型校准")

    # 生成模拟数据 (来自 Gamma 分布)
    rng = np.random.default_rng(42)
    true_alpha, true_beta = 3.0, 2.0
    data = rng.gamma(true_alpha, 1.0 / true_beta, size=200)

    results, best = ModelSelection.compare_distributions(data)
    print(f"  真实分布: Gamma(alpha={true_alpha}, beta={true_beta})")
    print(f"  样本量: {len(data)}")
    print(f"\n  模型比较 (BIC):")
    for name, info in results.items():
        print(f"    {name:12s}: BIC = {info['bic']:.2f},  params = {info['params']}")
    print(f"\n  最佳模型: {best}")

    # Gamma MLE
    alpha_hat, beta_hat = DistributionFitter.fit_gamma_mle(data)
    print(f"\n  Gamma MLE: alpha = {alpha_hat:.4f} (真值 {true_alpha})")
    print(f"             beta  = {beta_hat:.4f} (真值 {true_beta})")

    # KDE
    kde = KernelDensityReliability(data)
    print(f"  KDE 带宽 h        = {kde.h:.4f}")


def run_newton_rc_demo():
    """Newton 逆向通信演示"""
    _sep("Newton 逆向通信求解器 (种子项目 802)")

    # 求解 x^2 + y^2 = 1, x - y = 0 → (1/sqrt(2), 1/sqrt(2))
    def F(x):
        return np.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - x[1]])

    solver = NewtonRaphsonRC(2, tol=1e-10, max_iter=50)
    x = np.array([0.5, 0.8])
    fx = F(x)
    ido = 0

    for _ in range(100):
        ido, x = solver.step(ido, x, fx)
        if ido == 0:
            print(f"  收敛!  x = [{x[0]:.10f}, {x[1]:.10f}]")
            print(f"  精确解: [{1 / np.sqrt(2):.10f}, {1 / np.sqrt(2):.10f}]")
            break
        elif ido > 0:
            fx = F(x)
        else:
            print(f"  未收敛")
            break


def run_chebyshev_demo():
    """Chebyshev 逼近演示"""
    _sep("Chebyshev 逼近与差商插值 (种子项目 014)")

    # 逼近 f(x) = exp(-x^2) 在 [-2, 2] 上
    a, b, n = -2.0, 2.0, 12
    nodes = chebyshev_nodes(a, b, n)
    f_vals = np.exp(-nodes ** 2)

    # 差商
    dd = divided_differences(nodes, f_vals)

    # 在密集网格上评估
    x_test = np.linspace(a, b, 51)
    y_interp = newton_interp_eval(nodes, dd, x_test)
    y_exact = np.exp(-x_test ** 2)
    maxerr = np.max(np.abs(y_interp - y_exact))

    print(f"  函数: f(x) = exp(-x^2), 区间 [{a}, {b}], {n} 个 Chebyshev 节点")
    print(f"  最大插值误差      = {maxerr:.6e}")
    print(f"  Chebyshev 节点    = [{', '.join(f'{x:.3f}' for x in nodes[:5])}, ...]")


def main():
    """主入口：完整结构可靠性分析流程"""
    print("=" * 72)
    print("  PROJECT 210: 不确定性量化 — 可靠性分析与失效概率计算")
    print("  Uncertainty Quantification: Reliability Analysis &")
    print("  Failure Probability Computation")
    print("=" * 72)

    np.random.seed(42)

    # ==================================================================
    # 1. 问题定义
    # ==================================================================
    _sep("问题定义")
    n_dim = 2  # 应力-强度模型为 2D

    distributions = [
        LognormalRV(mu_x=1.0, sigma_x=0.2, name='X1_Strength'),
        NormalRV(mu=0.0, sigma=1.0, name='X2_Load'),
    ]
    print(f"  随机变量数        = {n_dim}")
    for i, d in enumerate(distributions):
        print(f"    {d.name:20s}: mean = {d.mean():.4f}, std = {d.std():.4f}")

    # 使用应力-强度干涉模型作为主问题 (有解析解, 用于验证算法)
    # g(u) = (mu_R - mu_S) + sigma_R*u_1 - sigma_S*u_2 (在标准正态空间中)
    # 精确 beta = (mu_R - mu_S)/sqrt(sigma_R^2 + sigma_S^2)
    lsf = StressStrengthLimitState(mu_R=10.0, sigma_R=2.0, mu_S=6.0, sigma_S=1.5)
    print(f"\n  极限状态函数: 应力-强度干涉模型")
    print(f"  g(u) = (mu_R + sigma_R*u_1) - (mu_S + sigma_S*u_2)")
    print(f"  mu_R=10, sigma_R=2, mu_S=6, sigma_S=1.5")
    print(f"  精确 beta = (10-6)/sqrt(4+2.25) = {4.0/np.sqrt(4+2.25):.6f}")
    print(f"  失效准则: g(u) ≤ 0")

    # ==================================================================
    # 2. FORM 分析
    # ==================================================================
    form_solver = run_form_analysis(lsf, n_dim, "(非线性)")

    # ==================================================================
    # 3. SORM 修正
    # ==================================================================
    pf_sorm = run_sorm_analysis(lsf, form_solver.u_star, form_solver.beta)

    # ==================================================================
    # 4. 自适应重要性采样
    # ==================================================================
    ais_result = run_adaptive_is(lsf, n_dim, form_solver.u_star)

    # ==================================================================
    # 5. 多项式混沌展开
    # ==================================================================
    pce = run_pce_analysis(lsf, n_dim)

    # ==================================================================
    # 6. 灵敏度分析
    # ==================================================================
    run_sensitivity_analysis(lsf, n_dim)

    # ==================================================================
    # 7. Fresnel 积分与渐近分析
    # ==================================================================
    run_fresnel_analysis()

    # ==================================================================
    # 8. 空间网格与失效域分析
    # ==================================================================
    run_spatial_analysis(lsf, n_dim)

    # ==================================================================
    # 9. 随机有限元
    # ==================================================================
    run_stochastic_fem()

    # ==================================================================
    # 10. 数据驱动校准
    # ==================================================================
    run_calibration_demo()

    # ==================================================================
    # 11. 基础算法演示
    # ==================================================================
    run_newton_rc_demo()
    run_chebyshev_demo()

    # ==================================================================
    # 12. 多基准测试
    # ==================================================================
    _sep("多基准极限状态函数 FORM 对比")
    benchmarks = [
        ("线性 (3D)", LinearLimitState(np.ones(3), c0=3.0), 3),
        ("应力-强度", StressStrengthLimitState(), 2),
        ("非线性 (2D)", NonlinearLimitState(c0=5.0, lam=2.0, kappa=0.5), 2),
        ("四次 (3D)", QuarticLimitState(c0=5.0), 3),
        ("指数余弦 (4D)", ExponentialCosLimitState(n_dim=4, c0=3.0), 4),
    ]
    print(f"  {'问题':20s}  {'beta':>10s}  {'P_f':>14s}  {'精确beta':>10s}")
    for name, blsf, dim in benchmarks:
        solver = FORMSolver(blsf, dim, tol=1e-8, max_iter=200)
        solver.solve()
        exact_b = getattr(blsf, 'exact_beta', lambda: None)()
        eb_str = f"{exact_b:.6f}" if exact_b is not None else "N/A"
        print(f"  {name:20s}  {solver.beta:10.6f}  {solver.pf:14.6e}  {eb_str:>10s}")

    # ==================================================================
    # 汇总
    # ==================================================================
    _sep("分析汇总")
    print(f"  最终失效概率估计 (FORM):   P_f = {form_solver.pf:.6e}")
    print(f"  最终失效概率估计 (SORM):   P_f = {pf_sorm:.6e}")
    print(f"  最终失效概率估计 (AIS):    P_f = {ais_result['pf']:.6e}")
    print(f"  可靠度指标 beta (FORM):    β   = {form_solver.beta:.6f}")
    print(f"\n  所有 15 个种子项目算法均已集成。")
    print(f"  程序正常结束。")


if __name__ == '__main__':
    main()
