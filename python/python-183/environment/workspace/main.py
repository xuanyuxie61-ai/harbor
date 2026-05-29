r"""
main.py
================================================================================
基于时空结构方程模型与间断 Galerkin 方法的高维因果推断网络分析系统

统一入口，零参数运行。
本程序整合 15 个种子项目的核心算法，在"数据科学：因果推断与结构方程"
领域构建一个面向前沿科学问题的博士级计算框架。

运行方式:
    python main.py
r"""

import numpy as np
import sys
import os

# 将当前目录加入路径以导入各模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparse_sem_matrix import (
    sample_covariance, graphical_lasso, threshold_precision,
    dense_to_csr, extract_causal_skeleton
)
from dg_causal_solver import solve_causal_diffusion_dg
from markov_causal_chain import (
    build_causal_markov_chain, canonical_form,
    absorption_probabilities_and_times, intervene_do_state
)
from pagerank_causal_rank import (
    adjacency_from_edges, build_google_matrix, power_method_rank,
    identify_confounders_by_rank
)
from toeplitz_time_inverse import (
    sample_autocovariance, toeplitz_matrix,
    fiedler_toeplitz_inverse, lag_causal_strength, yule_walker_solve
)
from gaussian_causal_test import owen_t_function, partial_correlation_test
from causal_ode_dynamics import (
    rk4_integrate, simulate_intervention_diffusion,
    monte_carlo_causal_distance, ball_unit_sample_nd
)
from spherical_causal_field import (
    sphere_llq_grid_points, project_to_spherical_harmonics,
    spherical_laplacian_spectrum
)
from pyramid_integrator import (
    integrate_pyramid, integrate_causal_effect_parameter_space,
    integrate_on_3d_causal_region
)
from causal_mesh_interpolator import (
    interpolate_mesh_field, polygon_contains_point,
    integrate_field_over_mesh
)
from geometry_utils import (
    generate_icosphere_nodes, compute_face_normals,
    compute_vertex_normals, stla_string
)
from time_series_utils import (
    align_time_series, cross_correlation, find_peak_lag,
    granger_causality_f_stat
)


def section_header(name: str):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print("=" * 70)


def run_sem_sparse_precision():
    r"""
    模块 1: 稀疏精度矩阵估计（Graphical Lasso）
    基于种子项目 510_hb_to_st 的稀疏矩阵思想。
    r"""
    section_header("模块 1: 稀疏精度矩阵与因果骨架提取")
    np.random.seed(42)
    p = 12
    n = 800
    # 构造真实 DAG 的精度矩阵
    Theta_true = np.eye(p) * 2.0
    true_edges = [(0, 1, 0.5), (1, 2, 0.4), (2, 3, 0.3),
                  (0, 3, 0.25), (4, 5, 0.6), (5, 6, 0.35),
                  (7, 8, 0.45), (8, 9, 0.3), (9, 10, 0.4),
                  (10, 11, 0.3), (0, 6, 0.2), (3, 9, 0.15)]
    for i, j, w in true_edges:
        Theta_true[i, j] = w
        Theta_true[j, i] = w

    Sigma_true = np.linalg.inv(Theta_true)
    X = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)
    S = sample_covariance(X)
    Theta_est = graphical_lasso(S, lam=0.06, max_iter=300, verbose=False)
    Theta_sparse = threshold_precision(Theta_est, eps=4e-3)
    edges, _ = extract_causal_skeleton(Theta_sparse)

    # CSR 转换
    data, indices, indptr = dense_to_csr(Theta_sparse)
    print(f"  变量维度 p={p}, 样本量 n={n}")
    print(f"  真实因果边数: {len(true_edges)}, 估计因果边数: {len(edges)}")
    print(f"  精度矩阵非零元比例: {np.count_nonzero(Theta_sparse) / (p * p) * 100:.2f}%")
    print(f"  CSR 数据长度: {len(data)}")
    return Theta_sparse, edges


def run_dg_causal_diffusion():
    r"""
    模块 2: 因果效应时空传播的 DG 求解
    基于种子项目 275_dg1d_poisson 的间断 Galerkin 方法。
    r"""
    section_header("模块 2: 因果扩散方程的间断 Galerkin 求解")
    t_hist, u_hist = solve_causal_diffusion_dg(
        nel=10, nsteps=120, dt=0.0005, K=0.5
    )
    print(f"  空间单元数: 10, 时间步数: 120")
    print(f"  终态最大绝对值: {np.max(np.abs(u_hist[-1])):.6e}")
    print(f"  终态 L2 能量: {np.sqrt(np.sum(u_hist[-1] ** 2)):.6e}")
    return t_hist, u_hist


def run_markov_causal_chain(edges):
    r"""
    模块 3: 因果马尔可夫链与 do-干预分析
    基于种子项目 1200_tennis_matrix 的状态转移矩阵。
    r"""
    section_header("模块 3: 因果马尔可夫链与干预效应")
    p = 12
    P, trans, absorb = build_causal_markov_chain(p, edges, n_states_per_var=3)
    P_canon, Q, R, state_map = canonical_form(P, trans, absorb)
    B, t = absorption_probabilities_and_times(Q, R)

    # do-干预：固定变量 0 为最高状态
    P_do = intervene_do_state(P, state_idx=0, new_value_state=2, n_states_per_var=3)
    P_canon_do, Q_do, R_do, _ = canonical_form(P_do, trans, absorb)
    B_do, t_do = absorption_probabilities_and_times(Q_do, R_do)
    effect_norm = np.linalg.norm(B_do - B, 'fro')

    print(f"  总状态数: {P.shape[0]}, 瞬态: {len(trans)}, 吸收态: {len(absorb)}")
    print(f"  期望到达时间范围: [{t.min():.3f}, {t.max():.3f}]")
    print(f"  do(变量0=高状态) 的因果效应 (Frobenius 范数): {effect_norm:.4f}")
    return B, t


def run_pagerank_causal_rank(edges, n):
    r"""
    模块 4: 因果网络 PageRank 排序
    基于种子项目 844_pagerank 的幂迭代算法。
    r"""
    section_header("模块 4: CausalRank 与混淆变量识别")
    A = adjacency_from_edges(edges, n, use_weights=True)
    G = build_google_matrix(A, alpha=0.85)
    pi = power_method_rank(G, max_iter=300)
    confounders = identify_confounders_by_rank(edges, n, top_k=3)
    print(f"  CausalRank (前 6): {pi[:6].round(4)}")
    print(f"  识别出的潜在混淆变量 (节点, 分数):")
    for node, score in confounders:
        print(f"    节点 {node}: {score:.4f}")
    return pi, confounders


def run_toeplitz_time_analysis():
    r"""
    模块 5: Toeplitz 自协方差矩阵快速求逆与时滞因果强度
    基于种子项目 1263_toeplitz_inverse 的 Fiedler 算法。
    r"""
    section_header("模块 5: Toeplitz 时间矩阵与 AR 系数估计")
    np.random.seed(11)
    n_series = 128
    phi_true = np.array([0.65, -0.25])
    eps = np.random.randn(n_series)
    x = np.zeros(n_series)
    x[0] = eps[0]
    x[1] = phi_true[0] * x[0] + eps[1]
    for t in range(2, n_series):
        x[t] = phi_true[0] * x[t - 1] + phi_true[1] * x[t - 2] + eps[t]

    max_lag = 12
    gamma = sample_autocovariance(x, max_lag)
    phi_est = yule_walker_solve(gamma, p=2)

    T = toeplitz_matrix(max_lag + 1, gamma)
    T_inv = fiedler_toeplitz_inverse(T)
    C = lag_causal_strength(T_inv)

    print(f"  真实 AR(2) 系数: {phi_true}")
    print(f"  估计 AR(2) 系数: {phi_est.round(4)}")
    print(f"  滞后因果强度 (前 6): {C[:6].round(4)}")
    return phi_est, C


def run_gaussian_causal_test(Theta_sparse, n):
    r"""
    模块 6: Owen T 函数与偏相关系数检验
    基于种子项目 033_asa076 的高斯求积计算。
    r"""
    section_header("模块 6: 高斯因果假设检验 (Owen T + 偏相关)")
    # TODO [Hole 3] 请补全高斯因果检验的数据准备与调用逻辑：
    # 1. 从 Theta_sparse 获取维度 p
    # 2. 生成合成数据：Sigma = inv(Theta_sparse + c * I)，从 N(0, Sigma) 采样 n 个样本
    # 3. 计算样本协方差矩阵 S
    # 4. 估计检验用精度矩阵 Theta_est_test = inv(S + c' * I)
    # 5. 调用 partial_correlation_test(Theta_est_test, n, alpha_level=0.05)
    # 6. 统计显著边数 n_edges
    # 注意：常数 c 和 c' 的选取需与稀疏精度矩阵的正定性假设一致
    raise NotImplementedError("Hole 3: 高斯因果检验数据准备待实现")

    pvals, reject = None, None
    n_edges = 0

    # Owen T 函数测试
    t_val = owen_t_function(1.0, 0.5)
    print(f"  Owen T(1.0, 0.5) = {t_val:.8f}")
    print(f"  显著条件依赖边数 (alpha=0.05): {n_edges}")
    return pvals, reject


def run_causal_ode_dynamics():
    r"""
    模块 7: 因果效应动态演化与蒙特卡洛距离估计
    基于种子项目 1036_rk4 (Runge-Kutta) 与 066_ball_distance (球采样)。
    r"""
    section_header("模块 7: 因果 ODE 动力学与 MC 距离估计")
    np.random.seed(13)
    p = 6
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            A[i, j] = 0.25 * np.random.randn()
    np.fill_diagonal(A, -0.6)
    B = np.eye(p) * 0.9
    y0 = np.zeros(p)

    t, y = simulate_intervention_diffusion(
        A, B, y0, (0.0, 2.0), n_steps=180,
        intervention_time=0.6, intervention_idx=2, intervention_magnitude=2.5
    )
    mu, var = monte_carlo_causal_distance(A, B, y0, (0.0, 1.5), n_steps=120, n_samples=80)

    print(f"  系统维度: {p}")
    print(f"  干预后最大状态幅值: {np.max(np.abs(y)):.4f}")
    print(f"  终态能量: {np.sum(y[-1] ** 2):.4f}")
    print(f"  MC 因果距离估计: mean={mu:.4f}, var={var:.6f}")
    return t, y


def run_spherical_causal_field():
    r"""
    模块 8: 球面因果场离散化与调和展开
    基于种子项目 1122_sphere_llq_grid 的球面网格生成。
    r"""
    section_header("模块 8: 球面因果场与调和分析")
    points = sphere_llq_grid_points(r=1.0, pc=np.zeros(3), lat_num=5, long_num=10)
    field = np.zeros(len(points))
    for i, pt in enumerate(points):
        r = np.linalg.norm(pt)
        if r > 1e-12:
            theta = np.arccos(np.clip(pt[2] / r, -1.0, 1.0))
            phi = np.arctan2(pt[1], pt[0])
            field[i] = np.exp(-1.5 * (theta - np.pi / 3.0) ** 2) * np.cos(3 * phi)

    coeffs = project_to_spherical_harmonics(field, points, l_max=4)
    lambdas = spherical_laplacian_spectrum(4)
    print(f"  球面网格点数: {len(points)}")
    print(f"  球面调和系数 (前 8): {coeffs[:8].round(4)}")
    print(f"  Laplace-Beltrami 特征值 (l=0..4): {lambdas.round(2)}")
    return points, coeffs


def run_pyramid_integration():
    r"""
    模块 9: 高维因果效应数值积分
    基于种子项目 937_pyramid_witherden_rule 的求积规则。
    r"""
    section_header("模块 9: 因果参数空间数值积分")
    # 测试 1: 金字塔多项式积分
    val1 = integrate_pyramid(lambda x, y, z: x * x + y * y + 2.0 * z, precision=4)
    # 测试 2: 高维因果效应参数空间期望
    val2 = integrate_causal_effect_parameter_space(
        lambda theta: np.exp(-np.sum(theta ** 2)), dim=4, n_samples=800
    )
    # 测试 3: 长方体区域积分
    val3 = integrate_on_3d_causal_region(
        lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y) * z ** 2,
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), precision=4
    )
    print(f"  金字塔多项式积分: {val1:.6f}")
    print(f"  4D 因果效应期望估计: {val2:.6f}")
    print(f"  单位立方体因果场积分: {val3:.6f}")
    return val1, val2, val3


def run_mesh_interpolation():
    r"""
    模块 10: 三角网格插值与多边形区域判定
    基于种子项目 425_ffmatlib 与 109_boundary_word_right。
    r"""
    section_header("模块 10: 因果场网格插值与空间区域判定")
    points = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
    ])
    triangles = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
    ])
    field = np.array([0.0, 2.0, 4.0, 2.0, 3.0])
    query = np.array([[0.6, 0.4], [0.2, 0.7]])
    interp = interpolate_mesh_field(points, triangles, field, query)
    total = integrate_field_over_mesh(points, triangles, field)

    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside = polygon_contains_point(poly, np.array([0.5, 0.5]))

    print(f"  查询点插值结果: {interp}")
    print(f"  网格上因果场积分: {total:.4f}")
    print(f"  点 (0.5,0.5) 在正方形区域内? {inside}")
    return interp, total


def run_geometry_processing():
    r"""
    模块 11: 三维几何处理与 STL 格式
    基于种子项目 1425_xyzf_display 与 1296_tri_surface_to_stla。
    r"""
    section_header("模块 11: 三维因果曲面几何处理")
    pts, faces = generate_icosphere_nodes(radius=1.0, subdivisions=1)
    fnormals = compute_face_normals(pts, faces)
    vnormals = compute_vertex_normals(pts, faces)
    stl = stla_string(pts, faces[:2])
    print(f"  二十面体球节点数: {pts.shape[0]}, 面片数: {faces.shape[0]}")
    print(f"  面法向量一致性检查 (与径向点积均值): {np.mean(np.sum(fnormals * pts[faces[:, 0]], axis=1)):.4f}")
    print(f"  STL 字符串长度: {len(stl)} chars")
    return pts, faces


def run_time_series_analysis():
    r"""
    模块 12: 时间序列对齐、互相关与 Granger 因果检验
    基于种子项目 135_calpak 的时间处理思想。
    r"""
    section_header("模块 12: 时间序列因果时滞分析")
    np.random.seed(23)
    n = 250
    t = np.linspace(0, 20, n)
    x = np.sin(0.8 * t) + 0.15 * np.random.randn(n)
    # y 受 x 滞后 6 步的影响
    lag_effect = 6
    y = np.cos(0.8 * t) + 0.4 * np.roll(x, lag_effect) + 0.15 * np.random.randn(n)
    y[:lag_effect] = y[lag_effect]

    grid = np.linspace(0, 20, 60)
    x_alg = align_time_series(t, x, grid)
    y_alg = align_time_series(t, y, grid)

    lags, ccf = cross_correlation(x_alg, y_alg, max_lag=12)
    peak_lag, peak_val = find_peak_lag(lags, ccf)
    F_stat, pval = granger_causality_f_stat(x, y, max_lag=4)

    print(f"  序列长度: {n}")
    print(f"  互相关峰值滞后: {peak_lag} (理论值 ~{lag_effect * (60 / n):.1f} 网格步)")
    print(f"  互相关峰值: {peak_val:.4f}")
    print(f"  Granger 因果检验: F={F_stat:.3f}, p={pval:.4f}")
    return peak_lag, F_stat


def main():
    print("\n" + "#" * 70)
    print("#  基于时空结构方程模型与间断 Galerkin 方法的高维因果推断网络分析系统")
    print("#" * 70)
    print("\n科学领域: 数据科学 — 因果推断与结构方程 (Causal Inference & SEM)")
    print("项目编号: PROJECT_183")
    print("语言: Python 3")

    # 执行各模块
    Theta_sparse, edges = run_sem_sparse_precision()
    t_hist, u_hist = run_dg_causal_diffusion()
    B, t = run_markov_causal_chain(edges)
    pi, confounders = run_pagerank_causal_rank(edges, n=Theta_sparse.shape[0])
    phi_est, C = run_toeplitz_time_analysis()
    pvals, reject = run_gaussian_causal_test(Theta_sparse, n=800)
    t_ode, y_ode = run_causal_ode_dynamics()
    pts_sph, coeffs_sph = run_spherical_causal_field()
    v1, v2, v3 = run_pyramid_integration()
    interp, total = run_mesh_interpolation()
    pts_geo, faces_geo = run_geometry_processing()
    peak_lag, F_stat = run_time_series_analysis()

    # 综合评估
    section_header("综合评估与数值验证")
    print("  各模块运行状态: OK")
    print(f"  因果骨架边数: {len(edges)}")
    print(f"  关键混淆变量数: {len(confounders)}")
    print(f"  DG 扩散能量耗散比: {np.sum(u_hist[0] ** 2) / (np.sum(u_hist[-1] ** 2) + 1e-12):.3f}")
    print(f"  球面调和截断阶数: l_max=4")
    print(f"  高维积分维度: 4D")
    print("\n" + "#" * 70)
    print("#  所有模块执行完毕，无报错。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
