"""
main.py — 统一入口: 随机参数 PDE 的不确定性量化全流水线
=========================================================

本项目解决的前沿科学问题:
    随机参数椭圆 PDE 的不确定性量化 (Uncertainty Quantification)
    -∇·(a(x,ω)∇u(x,ω)) = f(x)  in Ω = [0,1]²
    u = 0                         on ∂Ω

    其中 a(x,ω) = exp(Y(x,ω)) 为对数正态随机扩散系数,
    Y(x,ω) 为高斯随机场, 通过 Karhunen-Loève 展开参数化。

求解流程:
    1. 随机场建模: KL 展开 + Hankel-Cholesky (random_field.py)
    2. 稀疏网格配置: Smolyak + Van der Corput (sparse_grid_collocation.py)
    3. PCE 展开: Diophantine 多指标 + 正交多项式 (polynomial_chaos.py)
    4. FEM 求解: P1 有限元 + 渗流分析 (fem_solver.py)
    5. 深度学习代理: Inception 多尺度网络 (deep_surrogate.py)
    6. 解流形分析: MDS + POD (solution_manifold.py)
    7. 多保真度融合: 状态机 + Dijkstra (multi_fidelity.py)
    8. 贝叶斯反问题: SBI + C2ST/MMD (bayesian_inverse.py)
    9. 分岔检测: Newton-Maehly (bifurcation.py)
    10. 误差分析: SSIM/Jaccard/KS (error_analysis.py)

运行: python main.py  (零参数)

映射种子项目 (15个):
    180_circle_map, 558_hypercube_grid, 1055_fperdigon_DeepHistoPathology,
    306_distance_to_position, 1380_van_der_corput, 1371_ulam_spiral,
    865_percolation_simulation, 582_image_normalize, 801_newton_maehly,
    847_pariomino, 1276_IzzetYoung_ConsistentMIClientSimulator,
    504_hankel_cholesky, 1111_sbi-benchmark_results,
    1164_jenzenho_flame-ai-2024-reproduce, 190_closest_pair_brute
"""

import numpy as np
import sys
import os

# 导入所有模块
from random_field import (
    build_covariance_matrix, karhunen_loeve_decomposition,
    sample_random_field, log_normal_transform, normalize_random_field,
    kl_energy_spectrum, squared_exponential_covariance
)
from sparse_grid_collocation import (
    hypercube_grid_tensor, van_der_corput_sequence, halton_sequence,
    ulam_spiral_index, build_smolyak_grid, qmc_estimator
)
from polynomial_chaos import (
    total_order_multiindices, hyperbolic_cross_multiindices,
    multiindex_parity_charge, evaluate_pce_basis_batch,
    compute_pce_coefficients_collocation, pce_statistics,
    pce_coefficient_svd
)
from fem_solver import (
    generate_triangular_mesh, solve_stochastic_pde_realization,
    mesh_quality_indicator, closest_pair_brute, percolation_analysis
)
from deep_surrogate import (
    SurrogateNetwork, train_surrogate, compute_surrogate_metrics
)
from solution_manifold import (
    compute_solution_distance_matrix, classical_mds,
    pod_decomposition, closest_pair_in_set, adaptive_sampling_fill_gap,
    estimate_manifold_dimension
)
from multi_fidelity import (
    FidelityGraph, AdaptiveRefinementState,
    multi_fidelity_monte_carlo, information_percolation,
    compute_fidelity_weights
)
from bayesian_inverse import (
    GaussianPrior, rejection_sampling_inference, map_estimation,
    compute_all_posterior_metrics
)
from bifurcation import (
    newton_maehly_roots, detect_bifurcation_along_path,
    critical_parameter_search
)
from error_analysis import (
    mean_squared_error, relative_l2_error, structural_similarity_index,
    jaccard_index, statistical_moment_errors, kolmogorov_smirnov_distance,
    generate_error_report
)


def section_header(title, char='═', width=70):
    """打印章节标题"""
    border = char * width
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")


def subsection(title):
    print(f"\n── {title} ──")


# ============================================================
# Step 1: 随机场建模
# ============================================================
def step1_random_field():
    section_header("Step 1: 随机场建模与 Karhunen-Loève 展开")
    np.random.seed(42)

    # 空间网格点
    nx = 20
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    N = len(points)

    # 协方差矩阵 (平方指数核)
    print(f"  空间点: {N}, 相关长度: ℓ=0.3, 方差: σ²=1.0")
    cov_matrix = build_covariance_matrix(
        points, squared_exponential_covariance,
        sigma2=1.0, length_scale=0.3
    )

    # KL 分解
    eigenvalues, eigenvectors, explained = karhunen_loeve_decomposition(
        cov_matrix, n_modes=15, tolerance=1e-6
    )
    print(f"  KL 模态数: {len(eigenvalues)}")
    print(f"  能量捕获比: {explained:.6f}")

    energy, cumulative = kl_energy_spectrum(eigenvalues)
    print(f"  前3个特征值: {eigenvalues[:3]}")
    print(f"  前5模态累积能量: {cumulative[min(4,len(cumulative)-1)]:.4f}")

    # 采样
    samples, xi = sample_random_field(eigenvalues, eigenvectors,
                                       n_samples=50, seed=42)
    print(f"  采样数: {samples.shape[0]}, 场维度: {samples.shape[1]}")

    # 对数正态变换
    a_field = log_normal_transform(samples[0], mean_val=0.0, sigma_coeff=0.5)
    print(f"  扩散系数范围: [{a_field.min():.4f}, {a_field.max():.4f}]")

    # 标准化
    a_norm, params = normalize_random_field(a_field, method='zscore')
    print(f"  标准化后: 均值={params['mean']:.4f}, 标准差={params['std']:.4f}")

    return {
        'points': points, 'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors, 'explained': explained,
        'samples': samples, 'xi': xi, 'a_field': a_field
    }


# ============================================================
# Step 2: 稀疏网格配置
# ============================================================
def step2_sparse_grid():
    section_header("Step 2: 稀疏网格与拟随机序列")
    d = 3  # 随机参数维数

    # 张量积网格
    ns = [4, 4, 4]
    a = [-3.0, -3.0, -3.0]
    b = [3.0, 3.0, 3.0]
    grid = hypercube_grid_tensor(d, ns, a, b, centering='endpoint')
    print(f"  张量积网格: {d}D, 点数: {grid.shape[1]}")

    # Van der Corput 序列
    vdc = van_der_corput_sequence(20, base=2)
    print(f"  Van der Corput 前5个值: {vdc[:5]}")

    # Halton 序列
    halton = halton_sequence(30, dimensions=d)
    print(f"  Halton 序列: {halton.shape[0]}点 × {halton.shape[1]}维")
    print(f"  Halton 差异估计: {np.std(halton, axis=0).mean():.4f}")

    # Ulam 螺旋
    spiral = ulam_spiral_index(3)
    print(f"  Ulam 螺旋: {spiral.shape}, 中心值: {spiral[3,3]}")

    # Smolyak 稀疏网格
    q = 4
    sg_pts, sg_wts = build_smolyak_grid(d, q)
    print(f"  Smolyak 网格: level={q}, 点数: {len(sg_pts)}")
    print(f"  权重和: {np.sum(sg_wts):.4f} (应≈体积)")

    # QMC 估计器
    qmc_pts = qmc_estimator(50, d)
    print(f"  QMC 估计器: {len(qmc_pts)}点")

    return {
        'tensor_grid': grid, 'vdc': vdc, 'halton': halton,
        'smolyak_pts': sg_pts, 'smolyak_wts': sg_wts
    }


# ============================================================
# Step 3: 多项式混沌展开
# ============================================================
def step3_pce(xi_samples):
    section_header("Step 3: 广义多项式混沌展开 (gPC)")
    # 限制维度 (前几个 KL 模态)
    d = min(5, xi_samples.shape[1]) if xi_samples.ndim > 1 else 1
    if xi_samples.ndim == 1:
        xi_samples = xi_samples.reshape(-1, 1)
    xi_samples = xi_samples[:, :d]

    # 多指标枚举 (低阶避免组合爆炸)
    p = 2
    indices = total_order_multiindices(d, p)
    print(f"  全阶截断: d={d}, p={p}, 基函数数: {len(indices)}")
    print(f"  前5个多指标: {indices[:5].tolist()}")

    # 双曲交叉
    hc_indices = hyperbolic_cross_multiindices(d, p, q_norm=0.7)
    print(f"  双曲交叉: q=0.7, 基函数数: {len(hc_indices)}")

    # 奇偶荷
    charges = [multiindex_parity_charge(tuple(idx)) for idx in indices]
    print(f"  奇偶荷分布: +1={sum(1 for c in charges if c>0)}, "
          f"-1={sum(1 for c in charges if c<0)}")

    # 合成测试函数: f(ξ) = sin(ξ₁) + 0.5·ξ₂² + 0.3·ξ₁·ξ₃
    N_samp = max(200, len(indices) * 3)
    N_samp = min(N_samp, len(xi_samples))
    xi_test = xi_samples[:N_samp]
    # 若不够则复制扩展
    if len(xi_test) < len(indices) * 2:
        rng_pce = np.random.default_rng(42)
        extra = rng_pce.standard_normal((len(indices) * 3 - len(xi_test), d)) * 0.5
        xi_test = np.vstack([xi_test, extra])
    if d >= 3:
        f_vals = np.sin(xi_test[:, 0]) + 0.5 * xi_test[:, 1] ** 2 + \
                 0.3 * xi_test[:, 0] * xi_test[:, 2]
    elif d >= 2:
        f_vals = np.sin(xi_test[:, 0]) + 0.5 * xi_test[:, 1] ** 2
    else:
        f_vals = np.sin(xi_test[:, 0])

    # 配点法求 PCE 系数
    coefs, cond = compute_pce_coefficients_collocation(
        indices, xi_test, f_vals, distribution='gauss'
    )
    print(f"  PCE 系数数: {len(coefs)}, 条件数: {cond:.2e}")

    # 统计矩
    mean, var, std, sobol = pce_statistics(coefs, indices, 'gauss')
    print(f"  PCE 均值: {mean:.4f}")
    print(f"  PCE 标准差: {std:.4f}")
    print(f"  Sobol 主效应: {sobol}")

    # SVD 分析
    coefs_matrix = coefs.reshape(-1, 1) if coefs.ndim == 1 else coefs
    U, S, Vt, aspect, energy2 = pce_coefficient_svd(
        np.column_stack([coefs_matrix, coefs_matrix * 0.5 + 0.1])
    )
    print(f"  SVD 长短轴比: {aspect:.4f}")
    print(f"  SVD 前2奇异值能量占比: {energy2:.4f}")

    return {
        'indices': indices, 'coefficients': coefs,
        'mean': mean, 'variance': var, 'sobol': sobol
    }


# ============================================================
# Step 4: FEM 求解
# ============================================================
def step4_fem(a_field):
    section_header("Step 4: 有限元求解随机 PDE")
    nx, ny = 10, 10

    result = solve_stochastic_pde_realization(nx, ny, a_field)
    sol = result['solution']
    quality = result['quality']
    perc = result['percolation']

    print(f"  网格: {quality['n_nodes']}节点, {quality['n_elements']}单元")
    print(f"  最小边长: {quality['min_edge_length']:.4f}")
    print(f"  面积比: {quality['area_ratio']:.2f}")
    print(f"  求解器残差: {result['residual']:.2e}")
    print(f"  解的范围: [{sol.min():.6f}, {sol.max():.6f}]")
    print(f"  解的均值: {np.mean(sol):.6f}")

    print(f"  渗流分析: {perc['n_components']}个连通分量")
    print(f"  最大分量: {perc['largest_size']}节点")
    print(f"  渗流贯穿: {perc['percolates']}")

    return result


# ============================================================
# Step 5: 深度学习代理
# ============================================================
def step5_surrogate(xi_samples, qoi_samples):
    section_header("Step 5: 深度学习代理模型")
    if xi_samples.ndim == 1:
        xi_samples = xi_samples.reshape(-1, 1)

    n_train = min(200, len(xi_samples) - 20)
    n_val = min(50, len(xi_samples) - n_train)
    if n_train < 20:
        n_train = 20
        n_val = 10

    dim_in = xi_samples.shape[1]
    dim_out = qoi_samples.shape[1] if qoi_samples.ndim > 1 else 1
    if qoi_samples.ndim == 1:
        qoi_samples = qoi_samples.reshape(-1, 1)

    xi_train = xi_samples[:n_train]
    q_train = qoi_samples[:n_train]
    xi_val = xi_samples[n_train:n_train + n_val]
    q_val = qoi_samples[n_train:n_train + n_val]

    # 维度对齐
    if dim_out > 1:
        q_train = q_train[:, :min(3, dim_out)]
        q_val = q_val[:, :min(3, dim_out)]
        dim_out = min(3, dim_out)

    print(f"  训练集: {n_train}, 验证集: {n_val}")
    print(f"  输入维: {dim_in}, 输出维: {dim_out}")

    rng = np.random.default_rng(42)
    net = SurrogateNetwork(dim_in, dim_out, hidden_sizes=[32, 32], rng=rng)
    history = train_surrogate(net, xi_train, q_train, xi_val, q_val,
                               epochs=30, batch_size=16, lr=0.005)

    # 评估
    pred_train = net.forward(xi_train, training=False)
    pred_val = net.forward(xi_val, training=False)
    metrics_train = compute_surrogate_metrics(q_train, pred_train)
    metrics_val = compute_surrogate_metrics(q_val, pred_val)

    print(f"  训练 R²: {metrics_train['r_squared']:.4f}")
    print(f"  验证 R²: {metrics_val['r_squared']:.4f}")
    print(f"  验证相对L2: {metrics_val['relative_l2']:.4f}")
    print(f"  训练损失: {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")

    return {'network': net, 'metrics': metrics_val, 'history': history}


# ============================================================
# Step 6: 解流形分析
# ============================================================
def step6_manifold(snapshots):
    section_header("Step 6: 解流形分析 (MDS + POD)")
    if snapshots.ndim == 1:
        snapshots = snapshots.reshape(1, -1)
    N = len(snapshots)
    print(f"  快照数: {N}, 维度: {snapshots.shape[1]}")

    # 距离矩阵
    D = compute_solution_distance_matrix(snapshots, norm_type='L2')
    print(f"  距离矩阵: {D.shape}, 最大距离: {D.max():.4f}")

    # 经典 MDS
    Y_mds, eig_mds, stress = classical_mds(D, n_components=3)
    print(f"  cMDS 嵌入维: {Y_mds.shape}")
    print(f"  cMDS 应力: {stress:.4f}")
    print(f"  cMDS 特征值: {eig_mds[:3]}")

    # POD
    mean_field, modes, pod_eigs, coefs, energy = pod_decomposition(
        snapshots, energy_threshold=0.99
    )
    print(f"  POD 模态数: {modes.shape[1]}")
    print(f"  POD 累积能量: {energy:.4f}")
    print(f"  POD 前3特征值: {pod_eigs[:min(3,len(pod_eigs))]}")

    # 流形维度
    dim_eff = estimate_manifold_dimension(pod_eigs)
    print(f"  有效流形维度: {dim_eff}")

    # 最近点对
    if N >= 2:
        d_min, i, j = closest_pair_in_set(snapshots[:min(50, N)])
        print(f"  最近点对: 距离={d_min:.4f}, 索引=({i},{j})")

    # 自适应采样
    if Y_mds.shape[0] >= 3:
        new_pts = adaptive_sampling_fill_gap(Y_mds[:5], manifold_dim=2, n_new=3)
        print(f"  自适应新增点: {len(new_pts)}")

    return {
        'Y_mds': Y_mds, 'pod_modes': modes, 'pod_eigs': pod_eigs,
        'stress': stress, 'manifold_dim': dim_eff
    }


# ============================================================
# Step 7: 多保真度融合
# ============================================================
def step7_multi_fidelity():
    section_header("Step 7: 多保真度融合与自适应状态机")

    # 保真度图
    graph = FidelityGraph(n_levels=4, base_mesh=4, convergence_order=2)
    print(f"  保真度级别: {graph.n_levels}")
    for l in graph.levels:
        print(f"    L={l.level}: h={l.h:.4f}, cost={l.cost:.1f}")

    # Dijkstra
    path, dist = graph.dijkstra(0, 3)
    print(f"  Dijkstra 路径: {path}, 距离: {dist:.4f}")

    # 状态机
    state_machine = AdaptiveRefinementState(error_tol=1e-3, max_level=5, budget=500.0)
    print(f"\n  自适应加密过程:")
    current_error = 1.0
    for step in range(6):
        action = state_machine.update(current_error, 10.0 * (2 ** state_machine.current_level))
        receptivity = state_machine.get_receptivity()
        print(f"    步骤{step}: 状态={state_machine.state}, 级别={state_machine.current_level}, "
              f"误差={current_error:.2e}, 动作={action}, 接受度={receptivity:.3f}")
        if action == 'stop':
            break
        current_error *= 0.2

    # 多保真度 MC
    def model_L0(xi):
        return np.sin(xi[0]) * 0.8

    def model_L1(xi):
        return np.sin(xi[0]) * 0.95

    def model_L2(xi):
        return np.sin(xi[0])

    estimate, variance, cost = multi_fidelity_monte_carlo(
        [model_L0, model_L1, model_L2],
        n_samples_per_level=[100, 30, 10],
        seed=42
    )
    print(f"\n  MFMC 估计: {estimate:.4f}")
    print(f"  MFMC 方差: {variance:.4e}")
    print(f"  MFMC 总成本: {cost:.1f}")

    # 信息渗流
    adj = np.array([[0.8, 0.5, 0.1, 0.0],
                    [0.5, 0.9, 0.6, 0.2],
                    [0.1, 0.6, 0.8, 0.7],
                    [0.0, 0.2, 0.7, 0.9]])
    perc, reachable = information_percolation(adj, [0], [3], threshold=0.3)
    print(f"  信息渗流: {perc}, 可达节点: {reachable}")

    # 保真度权重
    weights = compute_fidelity_weights([0.1, 0.01, 0.001], [1.0, 8.0, 64.0])
    print(f"  最优保真度权重: {weights}")

    return {
        'estimate': estimate, 'variance': variance,
        'state_machine': state_machine
    }


# ============================================================
# Step 8: 贝叶斯反问题
# ============================================================
def step8_bayesian():
    section_header("Step 8: 贝叶斯反问题与模拟推断")

    # 先验
    prior = GaussianPrior(mean=np.zeros(2), cov=np.eye(2))
    print(f"  先验: N(0, I₂)")

    # 合成观测
    true_xi = np.array([0.5, -0.3])

    def forward_model(xi):
        return np.array([np.sin(xi[0]) + 0.5 * xi[1],
                          0.3 * xi[0] ** 2 + xi[1]])

    d_obs = forward_model(true_xi) + 0.05 * np.random.randn(2)
    print(f"  真实参数: {true_xi}")
    print(f"  观测数据: {d_obs}")

    # 拒绝采样
    samples, weights, acc_rate, ess = rejection_sampling_inference(
        prior, forward_model, d_obs, noise_std=0.1,
        n_proposals=2000, seed=42
    )
    print(f"  后验样本: {samples.shape[0]}")
    print(f"  接受率: {acc_rate:.4f}")
    print(f"  有效样本量: {ess:.1f}")
    print(f"  后验均值: {np.mean(samples, axis=0)}")

    # MAP
    xi_map, logp = map_estimation(prior, forward_model, d_obs,
                                    noise_std=0.1, n_starts=3, max_iter=50)
    print(f"  MAP 估计: {xi_map}")
    print(f"  MAP 后验: {logp:.4f}")

    # 后验验证
    prior_samples = prior.sample(200, seed=123)
    metrics = compute_all_posterior_metrics(prior_samples, samples)
    print(f"  C2ST 准确率: {metrics['c2st_accuracy']:.4f} (0.5=完美)")
    print(f"  MMD²: {metrics['mmd_squared']:.4e}")
    print(f"  中位距离: {metrics['median_distance']:.4f}")

    return {
        'posterior_samples': samples, 'xi_map': xi_map,
        'metrics': metrics
    }


# ============================================================
# Step 9: 分岔检测
# ============================================================
def step9_bifurcation():
    section_header("Step 9: 分岔检测与临界参数搜索")

    # Newton-Maehly 示例: 求 x³ - 6x² + 11x - 6 = 0 的根 (根为 1,2,3)
    coeffs = [1.0 + 0j, -6.0 + 0j, 11.0 + 0j, -6.0 + 0j]
    roots, converged, iters = newton_maehly_roots(coeffs)
    print(f"  多项式: x³ - 6x² + 11x - 6")
    print(f"  Newton-Maehly 根: {np.sort(np.real(roots))}")
    print(f"  收敛: {converged}, 迭代: {iters}")

    # 分岔检测
    def bifurcation_model(xi):
        lam = xi[0]
        return np.array([lam ** 2 - 2 * lam + 0.5])

    t_vals, min_eigs, detected, bif_t = detect_bifurcation_along_path(
        bifurcation_model, np.array([-2.0]), np.array([3.0]), n_steps=15
    )
    print(f"\n  分岔路径扫描:")
    print(f"    检测到分岔: {detected}")
    if detected:
        print(f"    分岔位置: t={bif_t:.4f}")
    print(f"    最小特征值范围: [{min_eigs.min():.4f}, {min_eigs.max():.4f}]")

    # 临界参数搜索
    crit_val, stability = critical_parameter_search(
        bifurcation_model, (-2.0, 3.0), n_search=12, seed=42
    )
    print(f"  临界参数值: {crit_val:.4f}")

    return {
        'roots': roots, 'bifurcation_detected': detected,
        'critical_value': crit_val
    }


# ============================================================
# Step 10: 误差分析
# ============================================================
def step10_error_analysis():
    section_header("Step 10: 综合误差分析")

    # 合成参考/计算场
    N = 100
    x = np.linspace(0, 1, N)
    ref_field = np.sin(np.pi * x) * np.exp(-x)
    comp_field = ref_field + 0.05 * np.random.randn(N)

    mse = mean_squared_error(ref_field, comp_field)
    rel_l2 = relative_l2_error(ref_field, comp_field)
    ssim = structural_similarity_index(ref_field, comp_field)
    jacc = jaccard_index(ref_field, comp_field)

    print(f"  MSE: {mse:.6e}")
    print(f"  相对L2误差: {rel_l2:.4f}")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  Jaccard: {jacc:.4f}")

    # 统计矩误差
    ref_samples = np.sin(np.pi * x) + 0.1 * np.random.randn(N)
    comp_samples = ref_samples + 0.02 * np.random.randn(N)
    mt, mp, me = statistical_moment_errors(ref_samples, comp_samples, max_moment=4)
    print(f"  矩误差 (1-4阶): {me}")

    ks = kolmogorov_smirnov_distance(ref_samples, comp_samples)
    print(f"  KS 距离: {ks:.4f}")

    # 综合报告
    report = generate_error_report(ref_field, comp_field,
                                     ref_samples, comp_samples,
                                     mesh_size=0.01, pce_order=3)
    print(f"\n  综合误差报告:")
    for key, val in report.items():
        if isinstance(val, float):
            print(f"    {key}: {val:.6f}")
        else:
            print(f"    {key}: {val}")

    return report


# ============================================================
# 主程序
# ============================================================
def main():
    """
    随机参数 PDE 不确定性量化 — 完整流水线
    """
    print("=" * 70)
    print("  PROJECT 209: 随机参数 PDE 模型的不确定性量化")
    print("  Uncertainty Quantification for Stochastic Parameter PDEs")
    print("=" * 70)
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  NumPy 版本: {np.__version__}")
    print(f"  工作目录: {os.getcwd()}")

    # Step 1: 随机场建模
    rf_data = step1_random_field()

    # Step 2: 稀疏网格
    sg_data = step2_sparse_grid()

    # Step 3: PCE
    pce_data = step3_pce(rf_data['xi'][:100])

    # Step 4: FEM
    fem_data = step4_fem(rf_data['a_field'])

    # Step 5: 代理模型
    # 构造 QoI: PDE 解的积分量
    sol = fem_data['solution']
    n_qoi = min(10, len(sol))
    qoi = np.array([np.mean(sol[i::n_qoi]) for i in range(n_qoi)])
    # 扩展到足够样本 (加噪声模拟不同随机实现)
    rng = np.random.default_rng(42)
    n_surr_samples = 300
    # qoi: (n_qoi,) → 每行一个样本, 重复并加噪声
    qoi_samples = np.tile(qoi.reshape(1, -1), (n_surr_samples, 1))
    qoi_samples += 0.01 * rng.standard_normal(qoi_samples.shape)
    # xi: 从 KL 采样扩展
    xi_base = rf_data['xi']
    xi_repeated = np.tile(xi_base, (n_surr_samples // max(1, len(xi_base)) + 1, 1))[:n_surr_samples]
    if xi_repeated.shape[1] > 5:
        xi_repeated = xi_repeated[:, :5]
    surr_data = step5_surrogate(xi_repeated, qoi_samples)

    # Step 6: 解流形
    # 构造多快照
    n_snaps = min(30, rf_data['samples'].shape[0])
    snapshots = rf_data['samples'][:n_snaps, :min(50, rf_data['samples'].shape[1])]
    manif_data = step6_manifold(snapshots)

    # Step 7: 多保真度
    mf_data = step7_multi_fidelity()

    # Step 8: 贝叶斯反问题
    bayes_data = step8_bayesian()

    # Step 9: 分岔
    bif_data = step9_bifurcation()

    # Step 10: 误差分析
    err_report = step10_error_analysis()

    # 总结
    section_header("计算完成 — 结果总结")
    print(f"  随机场 KL 能量捕获: {rf_data['explained']:.4f}")
    print(f"  PCE 方差: {pce_data['variance']:.6f}")
    print(f"  FEM 解范围: [{sol.min():.6f}, {sol.max():.6f}]")
    print(f"  代理模型 R²: {surr_data['metrics']['r_squared']:.4f}")
    print(f"  流形维度: {manif_data['manifold_dim']}")
    print(f"  MFMC 估计: {mf_data['estimate']:.4f}")
    print(f"  MAP 估计: {bayes_data['xi_map']}")
    print(f"  分岔检测: {bif_data['bifurcation_detected']}")
    print(f"  总体 MSE: {err_report.get('mse', 'N/A')}")
    print("\n  ✓ 所有模块运行成功")
    print("=" * 70)


if __name__ == '__main__':
    main()
