"""
main.py — 稀疏 PCE 恢复博士级计算项目主入口
=============================================
项目主题:
    L1 正则化稀疏多项式混沌展开恢复
    ——面向随机椭圆 PDE 的不确定性量化

问题描述:
    考虑随机椭圆 PDE:
        -∇·(a(x,ω) ∇u(x,ω)) = f(x)     in Ω = (0,1)
        u(0,ω) = u(1,ω) = 0
    其中扩散系数 a(x,ω) 为随机场, 具有稀疏 PCE 展开:
        a(x,ω) = a_0(x) + ∑_{|α|≤p} a_α(x) Ψ_α(ω)
    仅少数系数 a_α 显著非零. 本程序从有限观测 y 中恢复稀疏系数:
        min_c  0.5 ||Ψ c - y||²  +  λ ||c||_1
    其中 Ψ 为 PCE 基矩阵, c 为待恢复的稀疏系数向量.

算法流程:
    1. 构造多维稀疏多项式基 (Legendre / Jacobi / 双曲交叉)
    2. 生成随机观测 (点观测 + 线积分 + 面观测)
    3. 求解 FEM 正演问题 (1D 二次 BVP)
    4. 追踪 LASSO 正则化路径 (warm-start)
    5. 多算法对比 (ISTA / FISTA / Heavy-ball / 隐式中点)
    6. 支撑集识别 (水平集持续同调 + Otsu 阈值)
    7. GAN 学习结构化稀疏先验并指导加权恢复
    8. 图结构稀疏性分析 (Bellman-Ford)
    9. 蒙特卡洛 + Feynman-Kac 概率验证
   10. 综合诊断报告

科学公式贯穿全程:
    - 三项递推 (Legendre / Jacobi)
    - Galerkin 投影
    - 软阈值近端算子
    - Nesterov 加速与重启
    - 惯性振子动力学
    - Feynman-Kac 概率表示
    - 持续同调支撑集识别
    - Bellman-Ford 图依赖分析
"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_basis import (multi_index_set, pce_basis_eval, legendre_eval,
                          jacobi_eval, jacobi_exactness_degree,
                          legendre_product_integral)
from measurement_operator import (sample_hypercube, build_observation_operator,
                                  check_rip_constant, line_segment_points)
from fem_forward import (solve_bvp_1d, l2_error, h1_semi_error, max_error,
                         assemble_stiffness_1d, fem1d_nodes, fem1d_connectivity)
from l1_ista import (ista_l1, fista_l1, heavy_ball_proximal,
                     implicit_midpoint_proximal, soft_threshold,
                     lipschitz_constant, dual_certificate)
from regularization_path import (solve_regularization_path, select_lambda_bic,
                                 select_lambda_gcv, find_path_kinks,
                                 compute_lambda_max)
from monte_carlo_feynman_kac import (disk01_positive_area,
                                     monomial_integral_disk,
                                     feynman_kac_poisson_2d,
                                     mc_expected_loss)
from levelset_support import (support_by_persistence, adaptive_threshold,
                              track_support_evolution, stability_estimate,
                              contour_tree_1d)
from graph_sparsity import (build_sparsity_graph, support_connected_components,
                            graph_total_variation, hierarchical_ordering,
                            check_downward_closed, shortest_dependency_path,
                            bellman_ford)
from gan_sparse_prior import (SparsePriorGenerator, generate_training_supports,
                              weighted_lasso_with_prior)
from diagnostics import (relative_error, support_metrics, full_diagnostic_report,
                         print_report, compare_algorithms, convergence_rate)


def banner(title):
    """终端横幅."""
    width = 64
    print('\n' + '=' * width)
    print(title.center(width))
    print('=' * width)


def main():
    np.random.seed(214)
    print("Project 214: L1-Regularized Sparse PCE Recovery")
    print("         for Stochastic Elliptic PDEs")
    print("=" * 64)

    # ==================================================================
    # 阶段 1: 多项式基构造
    # ==================================================================
    banner("阶段 1: 稀疏多项式基构造")
    d = 3  # 随机变量维数
    p = 4  # 总阶上界

    # 全阶截断 vs 双曲交叉
    indices_full = multi_index_set(d, p, hyperbolic_cross=False)
    indices_hc = multi_index_set(d, p, hyperbolic_cross=True, q=0.6)
    print(f"  维数 d = {d},  总阶 p = {p}")
    print(f"  全阶截断基大小 M_full = {len(indices_full)}")
    print(f"  双曲交叉基大小 M_hc  = {len(indices_hc)} (q=0.6)")

    # Legendre 正交性验证
    orth_err = 0.0
    for pp in range(4):
        for qq in range(4):
            val = legendre_product_integral(pp, qq, n_quad=32)
            expected = 2.0 / (2 * pp + 1) if pp == qq else 0.0
            orth_err += (val - expected) ** 2
    print(f"  Legendre 正交性验证误差 (16 对): {np.sqrt(orth_err):.3e}")

    # Jacobi 精确性
    exact_deg = jacobi_exactness_degree(0.5, 0.5, 8)
    print(f"  8 点 Gauss-Jacobi (α=β=0.5) 精确阶: {exact_deg} "
          f"(理论 = 15)")

    # ==================================================================
    # 阶段 2: 观测算子构造
    # ==================================================================
    banner("阶段 2: 观测算子与采样")
    indices = indices_hc  # 采用双曲交叉基
    M = len(indices)

    segments = [([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])]
    polygons = [[[-1, -1], [1, -1], [1, 1], [-1, 1]]]

    config = dict(d=d, n_point=80, segments=segments, polygons=polygons, seed=42)
    A = build_observation_operator(config, indices, basis='legendre')
    print(f"  观测算子 A ∈ R^{{{A.shape[0]} × {A.shape[1]}}}")
    print(f"    - 点观测: 80")
    print(f"    - 线观测: {len(segments)}")
    print(f"    - 面观测: {len(polygons)}")

    # RIP 常数 (经验)
    delta_3 = check_rip_constant(A, s=3, n_test=20, seed=0)
    print(f"  经验 RIP 常数 δ_3 = {delta_3:.4f}")

    # ==================================================================
    # 阶段 3: 合成真值与观测
    # ==================================================================
    banner("阶段 3: 合成真解与观测数据")
    rng = np.random.RandomState(214)
    # 稀疏真解: 5 个非零分量
    n_active = 5
    active_idx = rng.choice(M, size=n_active, replace=False)
    c_true = np.zeros(M)
    c_true[active_idx] = rng.randn(n_active) * 3.0
    print(f"  真解维度: {M}")
    print(f"  真解稀疏度: {n_active} (非零指标 = {sorted(active_idx)})")

    y_clean = A @ c_true
    noise_std = 0.05 * np.linalg.norm(y_clean) / np.sqrt(len(y_clean))
    y = y_clean + noise_std * rng.randn(len(y_clean))
    print(f"  观测向量 y ∈ R^{len(y)},  噪声水平 σ = {noise_std:.4e}")
    support_true = set(active_idx)

    # ==================================================================
    # 阶段 4: 正则化路径追踪
    # ==================================================================
    banner("阶段 4: 正则化路径追踪")
    lambdas, solutions, supports, bic_scores = solve_regularization_path(
        A, y, n_lambda=25, lam_min_ratio=1e-3, max_iter_per=800, tol=1e-7)
    print(f"  λ 网格: [{lambdas.min():.4e}, {lambdas.max():.4e}], "
          f"共 {len(lambdas)} 个")
    lam_bic, idx_bic = select_lambda_bic(lambdas, bic_scores)
    lam_gcv, idx_gcv = select_lambda_gcv(A, y, lambdas, solutions)
    print(f"  BIC 最优 λ = {lam_bic:.4e}  (支撑集大小 = {len(supports[idx_bic])})")
    print(f"  GCV 最优 λ = {lam_gcv:.4e}  (支撑集大小 = {len(supports[idx_gcv])})")

    # 路径折点
    kinks = find_path_kinks(lambdas, [len(s) for s in supports])
    print(f"  检测到的路径折点数: {len(kinks)}")

    # 选择 λ (取 BIC 结果)
    lam = lam_bic
    print(f"  选用 λ = {lam:.4e}")

    # ==================================================================
    # 阶段 5: 多算法对比求解
    # ==================================================================
    banner("阶段 5: 多算法对比求解")
    results = {}

    x_ista, info_ista = ista_l1(A, y, lam, max_iter=2000, tol=1e-9)
    results['ISTA'] = (x_ista, info_ista)

    x_fista, info_fista = fista_l1(A, y, lam, max_iter=2000, tol=1e-9)
    results['FISTA'] = (x_fista, info_fista)

    x_hb, info_hb = heavy_ball_proximal(A, y, lam, mass=0.1, damping=0.85,
                                         max_iter=2000, tol=1e-9)
    results['Heavy-Ball'] = (x_hb, info_hb)

    x_imp, info_imp = implicit_midpoint_proximal(A, y, lam, h=0.5 / lipschitz_constant(A),
                                                  max_iter=1500)
    results['Implicit-Midpoint'] = (x_imp, info_imp)

    # 对比报告
    summary = compare_algorithms(results, c_true)
    print(f"  {'Algorithm':<22} {'RelErr':>9} {'Sparsity':>9} {'Iter':>6}")
    print("  " + "-" * 50)
    for s in summary:
        print(f"  {s['algorithm']:<22} {s['rel_error']:>9.3e} "
              f"{s['sparsity']:>9} {s['iterations']:>6}")

    # 选最优算法 (按相对误差)
    best_algo = min(summary, key=lambda s: s['rel_error'])['algorithm']
    x_best, info_best = results[best_algo]
    print(f"\n  >>> 最优算法: {best_algo}")

    # ==================================================================
    # 阶段 6: 支撑集识别 (水平集持续同调)
    # ==================================================================
    banner("阶段 6: 支撑集识别 (水平集方法)")
    # 持续同调
    support_pers = support_by_persistence(x_best, min_persistence_ratio=0.1)
    # Otsu 自适应阈值
    tau_otsu = adaptive_threshold(x_best, method='otsu')
    support_otsu = set(np.where(np.abs(x_best) > tau_otsu)[0])
    # MAD 阈值
    tau_mad = adaptive_threshold(x_best, method='median_abs_deviation')
    support_mad = set(np.where(np.abs(x_best) > tau_mad)[0])

    print(f"  持续同调支撑集: |S| = {len(support_pers)}")
    print(f"  Otsu 阈值支撑集: |S| = {len(support_otsu)}, τ = {tau_otsu:.4e}")
    print(f"  MAD 阈值支撑集: |S| = {len(support_mad)}, τ = {tau_mad:.4e}")

    # 支撑集精度
    for name, sup in [('Persistence', support_pers),
                      ('Otsu', support_otsu), ('MAD', support_mad)]:
        m = support_metrics(sup, support_true)
        print(f"  {name}: precision={m['precision']:.3f}, "
              f"recall={m['recall']:.3f}, F1={m['f1']:.3f}")

    # 持续同调树
    births, lifetimes, order = contour_tree_1d(np.abs(x_best))
    print(f"  前 5 大持续分支: {lifetimes[:5]}")

    # 支撑集稳定性
    stab = stability_estimate(x_best, noise_level=0.01 * np.max(np.abs(x_best)),
                              n_trials=50)
    stable = np.sum(stab > 0.8)
    print(f"  稳定性分析: 高稳定指标 (>80%) 数量 = {stable}")

    # ==================================================================
    # 阶段 7: GAN 结构化稀疏先验
    # ==================================================================
    banner("阶段 7: GAN 结构化稀疏先验")
    train_supports = generate_training_supports(M, n_samples=200,
                                                decay_rate=0.7, seed=7)
    print(f"  训练支撑集: {train_supports.shape}")
    gan = SparsePriorGenerator(M, dim_k=8, hidden=16, seed=77)
    # 训练 GAN
    n_epochs = 30
    for epoch in range(n_epochs):
        batch_idx = rng.choice(len(train_supports), size=32, replace=False)
        loss_d = gan.train_step(train_supports[batch_idx], lr=5e-3)
    print(f"  GAN 训练 {n_epochs} 轮完成")
    # 生成权重
    w_prior = gan.generate_weight_vector(n_samples=100)
    print(f"  先验权重: min={w_prior.min():.3f}, "
          f"mean={w_prior.mean():.3f}, max={w_prior.max():.3f}")

    # 加权 LASSO
    x_weighted = weighted_lasso_with_prior(A, y, lam, w_prior,
                                            max_iter=2000, tol=1e-9)
    err_weighted = relative_error(x_weighted, c_true)
    err_best = relative_error(x_best, c_true)
    print(f"  加权 LASSO 重构误差: {err_weighted:.4e}")
    print(f"  标准 FISTA 重构误差: {err_best:.4e}")
    print(f"  先验增益: {(err_best - err_weighted) / max(err_best, 1e-14) * 100:.2f}%")

    # ==================================================================
    # 阶段 8: 图结构稀疏性 (Bellman-Ford)
    # ==================================================================
    banner("阶段 8: 图结构稀疏性分析")
    edges, adj = build_sparsity_graph(indices, threshold=0.5)
    print(f"  稀疏图: {len(indices)} 节点, {len(edges) // 2} 条边")

    sup_final = set(np.where(np.abs(x_best) > tau_otsu)[0])
    comps = support_connected_components(sup_final, adj)
    print(f"  支撑集连通分量数: {len(comps)}")
    for k, comp in enumerate(comps):
        print(f"    分量 {k}: 大小 = {len(comp)}")

    # 图 TV
    gtv = graph_total_variation(x_best, adj)
    print(f"  图总变差 ||x||_GTV = {gtv:.4e}")

    # 层次结构
    levels = hierarchical_ordering(indices)
    print(f"  多指标分层: {sorted(levels.keys())} 层")
    is_closed, _ = check_downward_closed(sup_final, indices)
    print(f"  支撑集向下封闭性: {'是' if is_closed else '否'}")

    # Bellman-Ford 依赖路径
    dep_path = shortest_dependency_path(sup_final, adj)
    print(f"  最短依赖路径: {len(dep_path)} 节点")

    # Bellman-Ford 负环检测 (构造带权图)
    n_v = len(indices)
    weighted_edges = [(u, v, 1.0 - 0.1 * rng.randn()) for (u, v) in
                      [(i, j) for i in range(n_v) for j in range(n_v)
                       if abs(i - j) == 1][:20]]
    _, _, neg_cycle = bellman_ford(n_v, weighted_edges, source=0)
    print(f"  Bellman-Ford 负环检测: {'存在' if neg_cycle else '不存在'}")

    # ==================================================================
    # 阶段 9: 1D FEM 正演 + 误差估计
    # ==================================================================
    banner("阶段 9: 1D 随机椭圆 PDE FEM 正演")
    # 确定性扩散系数
    def a_func(x):
        return 1.0 + 0.5 * np.sin(2 * np.pi * x)

    def f_func(x):
        return np.ones_like(x)

    n_elem = 32
    nodes, u_fem, K_fem, F_fem = solve_bvp_1d(n_elem, a_func, f_func)
    # 参考解 (细网格)
    nodes_fine, u_fine, _, _ = solve_bvp_1d(128, a_func, f_func)
    # 插值到细网格比较
    u_coarse_on_fine = np.interp(nodes_fine, nodes, u_fem)
    err_l2 = l2_error(nodes_fine, u_coarse_on_fine, u_fine)
    err_h1 = h1_semi_error(nodes_fine, u_coarse_on_fine, u_fine)
    err_max = max_error(u_coarse_on_fine, u_fine)
    print(f"  单元数: {n_elem}  |  节点数: {len(nodes)}")
    print(f"  L2  误差 (vs 128 单元): {err_l2:.4e}")
    print(f"  H1  误差              : {err_h1:.4e}")
    print(f"  L∞  误差              : {err_max:.4e}")

    # ==================================================================
    # 阶段 10: 蒙特卡洛 + Feynman-Kac 概率验证
    # ==================================================================
    banner("阶段 10: 蒙特卡洛 + Feynman-Kac 验证")
    # 单位圆盘正象限面积 (应为 π/4 ≈ 0.7854)
    area_mc = disk01_positive_area(n_sample=20000, seed=214)
    print(f"  正象限圆盘面积 MC 估计: {area_mc:.4f} "
          f"(理论 π/4 = {np.pi / 4:.4f})")

    # 单项式积分
    for (a, b) in [(1, 0), (0, 1), (2, 0), (1, 1)]:
        val = monomial_integral_disk(a, b, n_sample=20000, seed=214)
        print(f"    ∫ x^{a} y^{b} dA = {val:.4f}")

    # Feynman-Kac 求解 Poisson 方程: -½Δu = 1,  u|∂D = 0
    # 精确解: u(x) = 1 - |x|²
    fk_val = feynman_kac_poisson_2d(lambda x: 1.0,
                                    lambda x: 0.0,
                                    [0.0, 0.0],
                                    n_paths=800, dt=2e-3,
                                    max_steps=2000, seed=214)
    fk_exact = 1.0  # u(0) = 1
    print(f"  Feynman-Kac 在 (0,0) 处的值: {fk_val:.4f} "
          f"(精确 = {fk_exact})")
    print(f"  相对误差: {abs(fk_val - fk_exact) / fk_exact:.4e}")

    # MC 期望损失验证
    mean_loss, se_loss = mc_expected_loss(A, y_clean, x_best,
                                          n_samples=300,
                                          noise_std=noise_std, seed=214)
    print(f"  期望损失 E[||Ax*-y||²/m] = {mean_loss:.4e} ± {se_loss:.4e}")

    # ==================================================================
    # 阶段 11: 综合诊断
    # ==================================================================
    banner("阶段 11: 综合诊断报告")
    report = full_diagnostic_report(x_best, c_true, A, y, lam, info_best,
                                    algo_name=best_algo,
                                    support_true=support_true)
    print_report(report)

    print("\n" + "=" * 64)
    print("PROJECT 214 完成. 所有模块运行正常.")
    print("科学问题: 随机椭圆 PDE 的 L1 稀疏 PCE 恢复")
    print("算法: FISTA / Heavy-ball / Implicit-midpoint + GAN 先验")
    print("=" * 64)


if __name__ == '__main__':
    main()
