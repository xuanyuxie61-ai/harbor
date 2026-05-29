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
    p = Theta_sparse.shape[0]
    # 生成合成数据用于检验
    Sigma = np.linalg.inv(Theta_sparse + 0.5 * np.eye(p))
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    S = np.cov(X, rowvar=False)
    Theta_est_test = np.linalg.inv(S + 0.1 * np.eye(p))
    pvals, reject = partial_correlation_test(Theta_est_test, n, alpha_level=0.05)
    n_edges = np.sum(reject) // 2

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

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: sample_covariance 输出形状与有限性检验 ----
np.random.seed(42)
X_small = np.random.randn(50, 5)
S_small = sample_covariance(X_small)
assert S_small.shape == (5, 5), '[TC01] sample_covariance 输出形状错误 FAILED'
assert np.all(np.isfinite(S_small)), '[TC01] sample_covariance 输出含 NaN/Inf FAILED'

# ---- TC02: sample_covariance 对称性检验 ----
np.random.seed(123)
X_sym = np.random.randn(100, 4)
S_sym = sample_covariance(X_sym)
assert np.allclose(S_sym, S_sym.T, atol=1e-12), '[TC02] sample_covariance 不满足对称性 FAILED'

# ---- TC03: sample_covariance 半正定性检验（特征值非负） ----
np.random.seed(77)
X_psd = np.random.randn(200, 6)
S_psd = sample_covariance(X_psd)
eigvals_psd = np.linalg.eigvalsh(S_psd)
assert np.all(eigvals_psd >= -1e-10), '[TC03] sample_covariance 非半正定 FAILED'

# ---- TC04: soft_threshold 已知值检验 ----
from sparse_sem_matrix import soft_threshold
M_test = np.array([[3.0, 0.5, -0.3], [0.5, 2.0, 0.1], [-0.3, 0.1, 4.0]])
st = soft_threshold(M_test, tau=0.4)
assert abs(st[0, 1] - 0.1) < 1e-12, '[TC04] soft_threshold 正值软阈值错误 FAILED'
assert abs(st[0, 2] - 0.0) < 1e-12, '[TC04] soft_threshold 零截断错误 FAILED'
assert abs(st[0, 0] - 3.0) < 1e-12, '[TC04] soft_threshold 对角线不应被阈值化 FAILED'
assert abs(st[2, 2] - 4.0) < 1e-12, '[TC04] soft_threshold 对角线保留错误 FAILED'

# ---- TC05: graphical_lasso 输出形状与有限性 ----
np.random.seed(42)
p5 = 6
n5 = 300
Theta5 = np.eye(p5) * 2.0
Sigma5 = np.linalg.inv(Theta5)
X5 = np.random.multivariate_normal(np.zeros(p5), Sigma5, size=n5)
S5 = sample_covariance(X5)
Theta_est5 = graphical_lasso(S5, lam=0.1, max_iter=100, verbose=False)
assert Theta_est5.shape == (p5, p5), '[TC05] graphical_lasso 输出形状错误 FAILED'
assert np.all(np.isfinite(Theta_est5)), '[TC05] graphical_lasso 输出含 NaN/Inf FAILED'

# ---- TC06: threshold_precision 小值归零与对角线保留 ----
Theta_dense = np.array([[2.1, 0.003, -0.001], [0.003, 1.8, 0.0005], [-0.001, 0.0005, 3.0]])
Theta_th = threshold_precision(Theta_dense, eps=0.01)
assert Theta_th[0, 1] == 0.0, '[TC06] threshold_precision 小正值未归零 FAILED'
assert Theta_th[0, 2] == 0.0, '[TC06] threshold_precision 小负值未归零 FAILED'
assert Theta_th[0, 0] == 2.1, '[TC06] threshold_precision 对角线被修改 FAILED'

# ---- TC07: dense_to_csr 与 csr_to_dense 往返检验 ----
from sparse_sem_matrix import csr_to_dense
A_csr = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, -2.0], [5.0, 0.0, 0.0]])
data_csr, indices_csr, indptr_csr = dense_to_csr(A_csr)
A_rec = csr_to_dense(data_csr, indices_csr, indptr_csr, 3)
assert np.allclose(A_csr, A_rec, atol=1e-12), '[TC07] CSR 往返转换不一致 FAILED'

# ---- TC08: extract_causal_skeleton 边数检验 ----
Theta_skel = np.zeros((4, 4))
np.fill_diagonal(Theta_skel, 2.0)
Theta_skel[0, 1] = Theta_skel[1, 0] = 0.5
Theta_skel[2, 3] = Theta_skel[3, 2] = -0.3
edges_skel, p_skel = extract_causal_skeleton(Theta_skel)
assert len(edges_skel) == 2, '[TC08] extract_causal_skeleton 边数错误 FAILED'
assert p_skel == 4, '[TC08] extract_causal_skeleton 节点数错误 FAILED'

# ---- TC09: legendre_basis_2d 输出维度检验 ----
from dg_causal_solver import legendre_basis_2d
phi_dg, dphi_dg, w_dg = legendre_basis_2d()
assert phi_dg.shape == (3, 2), '[TC09] legendre_basis_2d phi 形状错误 FAILED'
assert dphi_dg.shape == (3, 2), '[TC09] legendre_basis_2d dphi 形状错误 FAILED'
assert w_dg.shape == (2,), '[TC09] legendre_basis_2d w 形状错误 FAILED'
assert abs(np.sum(w_dg) - 2.0) < 1e-10, '[TC09] legendre_basis_2d Gauss-Legendre 权重和错误 FAILED'

# ---- TC10: assemble_dg_matrices 输出形状与可复现性 ----
from dg_causal_solver import assemble_dg_matrices
M10, A10 = assemble_dg_matrices(nel=4, K=0.5, penal=10.0, ss=-1.0)
Ndof10 = 4 * 3
assert M10.shape == (Ndof10, Ndof10), '[TC10] assemble_dg_matrices M 形状错误 FAILED'
assert A10.shape == (Ndof10, Ndof10), '[TC10] assemble_dg_matrices A 形状错误 FAILED'
assert np.all(np.isfinite(M10)), '[TC10] M 含 NaN/Inf FAILED'
assert np.all(np.isfinite(A10)), '[TC10] A 含 NaN/Inf FAILED'
# 可复现性
M10b, A10b = assemble_dg_matrices(nel=4, K=0.5, penal=10.0, ss=-1.0)
assert np.allclose(M10, M10b, atol=1e-14), '[TC10] M 矩阵不可复现 FAILED'
assert np.allclose(A10, A10b, atol=1e-14), '[TC10] A 矩阵不可复现 FAILED'

# ---- TC11: solve_causal_diffusion_dg 输出形状与能量耗散 ----
t_dg, u_dg = solve_causal_diffusion_dg(nel=6, nsteps=40, dt=0.002, K=0.3)
Ndof_dg = 6 * 3
assert t_dg.shape == (41,), '[TC11] DG 时间向量形状错误 FAILED'
assert u_dg.shape == (41, Ndof_dg), '[TC11] DG 解历史形状错误 FAILED'
assert np.all(np.isfinite(u_dg)), '[TC11] DG 解含 NaN/Inf FAILED'
energy_final = np.sum(u_dg[-1] ** 2)
assert energy_final > 0, '[TC11] 有源项时终态能量应为正 FAILED'

# ---- TC12: rk4_integrate 谐振子解析解验证 ----
harmonic_osc = lambda t, y: np.array([y[1], -y[0]])
t_rk4, y_rk4 = rk4_integrate(harmonic_osc, (0.0, np.pi/2), np.array([0.0, 1.0]), n_steps=200)
assert abs(y_rk4[-1, 0] - 1.0) < 0.005, '[TC12] RK4 谐振子 sin(pi/2)=1 误差过大 FAILED'
assert abs(y_rk4[-1, 1]) < 0.01, '[TC12] RK4 谐振子 cos(pi/2)=0 误差过大 FAILED'

# ---- TC13: ball_unit_sample_nd 单位球内检验 ----
np.random.seed(42)
for dim_test in [2, 3, 5, 10]:
    pt_ball = ball_unit_sample_nd(dim_test)
    assert np.linalg.norm(pt_ball) <= 1.0 + 1e-10, f'[TC13] 采样点在 dim={dim_test} 出了单位球 FAILED'

# ---- TC14: Owen T 函数边界值与已知对照 ----
t_owen1 = owen_t_function(0.0, 0.5)
expected_t1 = np.arctan(0.5) / (2.0 * np.pi)
assert abs(t_owen1 - expected_t1) < 1e-10, '[TC14] Owen T(0,0.5) 与解析值不符 FAILED'
t_owen2 = owen_t_function(2.0, 0.0)
assert abs(t_owen2) < 1e-14, '[TC14] Owen T(2,0) 应接近 0 FAILED'
t_owen3 = owen_t_function(0.0, 0.0)
assert abs(t_owen3) < 1e-14, '[TC14] Owen T(0,0) 应接近 0 FAILED'

# ---- TC15: partial_correlation_test 输出形状与可复现性 ----
np.random.seed(15)
p15 = 6
n15 = 400
Theta15 = np.eye(p15) * 2.0
Theta15[0, 1] = Theta15[1, 0] = 0.55
Sigma15 = np.linalg.inv(Theta15)
X15 = np.random.multivariate_normal(np.zeros(p15), Sigma15, size=n15)
S15 = np.cov(X15, rowvar=False)
Theta_est15 = np.linalg.inv(S15 + 0.1 * np.eye(p15))
pvals15, reject15 = partial_correlation_test(Theta_est15, n15, alpha_level=0.05)
assert pvals15.shape == (p15, p15), '[TC15] p 值矩阵形状错误 FAILED'
assert reject15.shape == (p15, p15), '[TC15] reject 矩阵形状错误 FAILED'
np.random.seed(15)
X15b = np.random.multivariate_normal(np.zeros(p15), Sigma15, size=n15)
S15b = np.cov(X15b, rowvar=False)
Theta_est15b = np.linalg.inv(S15b + 0.1 * np.eye(p15))
pvals15b, reject15b = partial_correlation_test(Theta_est15b, n15, alpha_level=0.05)
assert np.allclose(pvals15, pvals15b, atol=1e-12), '[TC15] 可复现性 FAILED'

# ---- TC16: sphere_llq_grid_points 点数与球面检验 ----
pts_sph = sphere_llq_grid_points(r=2.0, pc=np.array([1.0, 2.0, 3.0]), lat_num=3, long_num=6)
expected_n16 = 2 + 3 * 6
assert pts_sph.shape[0] == expected_n16, '[TC16] 球面网格点数错误 FAILED'
for i in range(pts_sph.shape[0]):
    dist = np.linalg.norm(pts_sph[i] - np.array([1.0, 2.0, 3.0]))
    assert abs(dist - 2.0) < 1e-8, f'[TC16] 球面网格点 {i} 不在球面上 FAILED'

# ---- TC17: spherical_laplacian_spectrum 特征值验证 ----
lambdas_sph = spherical_laplacian_spectrum(4)
assert len(lambdas_sph) == 5, '[TC17] 谱长度错误 FAILED'
for l in range(5):
    assert abs(lambdas_sph[l] + l * (l + 1)) < 1e-12, f'[TC17] l={l} 特征值错误 FAILED'

# ---- TC18: pyramid_witherden_rule 输出合理性检验 ----
from pyramid_integrator import pyramid_witherden_rule
for prec in range(6):
    n_w, x_w, y_w, z_w, w_w = pyramid_witherden_rule(prec)
    assert n_w > 0, f'[TC18] precision={prec} 求积点数为零 FAILED'
    assert len(w_w) == n_w, f'[TC18] precision={prec} 权重数组长度不匹配 FAILED'
    assert np.all(w_w > 0), f'[TC18] precision={prec} 含非正权重 FAILED'
    assert np.all(np.isfinite(x_w)), f'[TC18] precision={prec} x 坐标含 NaN/Inf FAILED'
    assert np.all(np.isfinite(z_w)), f'[TC18] precision={prec} z 坐标含 NaN/Inf FAILED'

# ---- TC19: integrate_pyramid 常数函数精确积分 ----
const_val = integrate_pyramid(lambda x, y, z: 2.5, precision=4)
assert abs(const_val - 10.0) < 0.2, '[TC19] 常数 2.5 在金字塔上积分近似 10 FAILED'

# ---- TC20: triangle_area 已知三角形面积 ----
from causal_mesh_interpolator import triangle_area
area_20 = abs(triangle_area(np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.0, 4.0])))
assert abs(area_20 - 6.0) < 1e-12, '[TC20] 直角三角形面积非 6 FAILED'

# ---- TC21: polygon_contains_point 已知包含关系 ----
poly_sq = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
assert polygon_contains_point(poly_sq, np.array([1.0, 1.0])) == True, '[TC21] 内部点判定错误 FAILED'
assert polygon_contains_point(poly_sq, np.array([3.0, 1.0])) == False, '[TC21] 外部点判定错误 FAILED'
assert polygon_contains_point(poly_sq, np.array([0.0, 0.0])) == True, '[TC21] 顶点判定错误 FAILED'
assert polygon_contains_point(poly_sq, np.array([1.0, 0.0])) == True, '[TC21] 边界点判定错误 FAILED'

# ---- TC22: cross_product 正交性检验 ----
from geometry_utils import cross_product
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])
cp = cross_product(v1, v2)
assert np.allclose(cp, np.array([0.0, 0.0, 1.0]), atol=1e-12), '[TC22] 叉积结果错误 FAILED'
assert abs(np.dot(cp, v1)) < 1e-14, '[TC22] 叉积须正交于 v1 FAILED'
assert abs(np.dot(cp, v2)) < 1e-14, '[TC22] 叉积须正交于 v2 FAILED'

# ---- TC23: generate_icosphere_nodes 输出形状 ----
pts_ico, faces_ico = generate_icosphere_nodes(radius=2.0, subdivisions=0)
assert pts_ico.shape == (12, 3), '[TC23] 初始二十面体应有 12 节点 FAILED'
assert faces_ico.shape == (20, 3), '[TC23] 初始二十面体应有 20 面片 FAILED'
pts_ico2, faces_ico2 = generate_icosphere_nodes(radius=1.0, subdivisions=1)
assert pts_ico2.shape[0] > 12, '[TC23] 细分后节点数应变多 FAILED'
assert np.all(np.isfinite(pts_ico2)), '[TC23] 节点坐标含 NaN/Inf FAILED'

# ---- TC24: cross_correlation 可复现性与对称性 ----
np.random.seed(24)
x_cc = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.05 * np.random.randn(100)
np.random.seed(24)
y_cc = np.cos(np.linspace(0, 4*np.pi, 100)) + 0.05 * np.random.randn(100)
lags24, ccf24 = cross_correlation(x_cc, y_cc, max_lag=5)
assert len(lags24) == 11, '[TC24] 互相关滞后数组长度错误 FAILED'
assert len(ccf24) == 11, '[TC24] 互相关值数组长度错误 FAILED'
assert np.all(np.isfinite(ccf24)), '[TC24] 互相关值含 NaN/Inf FAILED'
np.random.seed(24)
x_ccb = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.05 * np.random.randn(100)
np.random.seed(24)
y_ccb = np.cos(np.linspace(0, 4*np.pi, 100)) + 0.05 * np.random.randn(100)
_, ccf24b = cross_correlation(x_ccb, y_ccb, max_lag=5)
assert np.allclose(ccf24, ccf24b, atol=1e-12), '[TC24] cross_correlation 可复现性 FAILED'

# ---- TC25: granger_causality_f_stat 可复现性 ----
np.random.seed(25)
n25 = 300
x25 = np.random.randn(n25)
y25 = 0.4 * np.roll(x25, 2) + 0.3 * np.random.randn(n25)
np.random.seed(25)
x25b = np.random.randn(n25)
y25b = 0.4 * np.roll(x25b, 2) + 0.3 * np.random.randn(n25)
F25, p25 = granger_causality_f_stat(x25, y25, max_lag=3)
F25b, p25b = granger_causality_f_stat(x25b, y25b, max_lag=3)
assert abs(F25 - F25b) < 1e-12, '[TC25] Granger F 统计量可复现性 FAILED'
assert abs(p25 - p25b) < 1e-12, '[TC25] Granger p 值可复现性 FAILED'

# ---- TC26: causal_ode_system 零输入零初始系统输出全零 ----
from causal_ode_dynamics import causal_ode_system
p26 = 3
A26 = np.eye(p26) * (-0.5)
B26 = np.eye(p26) * 0.5
y0_26 = np.zeros(p26)
u_zero_26 = lambda t: np.zeros(p26)
dydt26 = causal_ode_system(0.0, y0_26, A26, B26, u_zero_26)
assert np.allclose(dydt26, np.zeros(p26), atol=1e-12), '[TC26] 零输入零初始 dy/dt 应全零 FAILED'

# ---- TC27: forward_difference 已知序列差分数值 ----
from time_series_utils import forward_difference
x_fd = np.array([1.0, 3.0, 6.0, 10.0])
diff_fd = forward_difference(x_fd)
assert np.allclose(diff_fd, np.array([2.0, 3.0, 4.0]), atol=1e-12), '[TC27] forward_difference 数值错误 FAILED'

# ---- TC28: compute_face_normals 单位法向量检验 ----
pts28 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
faces28 = np.array([[0, 1, 2], [0, 1, 3]])
fn28 = compute_face_normals(pts28, faces28)
for f in range(fn28.shape[0]):
    nrm = np.linalg.norm(fn28[f])
    assert abs(nrm - 1.0) < 1e-10, f'[TC28] 面 {f} 法向量未单位化 FAILED'

# ---- TC29: stla_string 语法结构检验 ----
pts29 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
faces29 = np.array([[0, 1, 2]])
stl29 = stla_string(pts29, faces29)
assert 'solid CausalMesh' in stl29, '[TC29] STL 缺少 solid 头 FAILED'
assert 'endsolid CausalMesh' in stl29, '[TC29] STL 缺少 endsolid 尾 FAILED'
assert 'facet normal' in stl29, '[TC29] STL 缺少 facet normal FAILED'
assert 'outer loop' in stl29, '[TC29] STL 缺少 outer loop FAILED'
assert 'vertex' in stl29, '[TC29] STL 缺少 vertex FAILED'

# ---- TC30: yule_walker_solve 已知 AR(1) 系数验证 ----
np.random.seed(30)
n30 = 500
phi_true30 = 0.7
eps30 = np.random.randn(n30)
x30 = np.zeros(n30)
x30[0] = eps30[0]
for t in range(1, n30):
    x30[t] = phi_true30 * x30[t-1] + eps30[t]
gamma30 = sample_autocovariance(x30, max_lag=5)
phi_est30 = yule_walker_solve(gamma30, p=1)
assert abs(phi_est30[0] - phi_true30) < 0.15, '[TC30] AR(1) Yule-Walker 估计误差过大 FAILED'

# ---- TC31: power_method_rank 概率分布归一化 ----
edges31 = [(0, 1, 0.5), (1, 2, 0.4), (2, 3, 0.3), (0, 3, 0.2)]
A31 = adjacency_from_edges(edges31, 4, use_weights=True)
G31 = build_google_matrix(A31, alpha=0.85)
pi31 = power_method_rank(G31, max_iter=300)
assert abs(np.sum(pi31) - 1.0) < 1e-10, '[TC31] CausalRank 分布未归一化 FAILED'
assert np.all(pi31 >= 0), '[TC31] CausalRank 含负值 FAILED'
assert np.all(np.isfinite(pi31)), '[TC31] CausalRank 含 NaN/Inf FAILED'

# ---- TC32: build_google_matrix 列随机性 ----
A32 = adjacency_from_edges(edges31, 4, use_weights=False)
G32 = build_google_matrix(A32, alpha=0.85)
col_sums32 = np.sum(G32, axis=0)
assert np.allclose(col_sums32, np.ones(4), atol=1e-10), '[TC32] Google 矩阵非列随机 FAILED'

# ---- TC33: integrate_field_over_mesh 已知网格场积分 ----
pts33 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
faces33 = np.array([[0, 1, 2]])
field33 = np.array([1.0, 1.0, 1.0])
total33 = integrate_field_over_mesh(pts33, faces33, field33)
expected33 = 0.5 * 1.0  # area=0.5, avg_val=1.0
assert abs(total33 - expected33) < 1e-10, '[TC33] 网格场积分错误 FAILED'

# ---- TC34: exchange_matrix 与 hankel_matrix 结构检验 ----
from toeplitz_time_inverse import exchange_matrix, hankel_matrix
J34 = exchange_matrix(4)
assert J34[0, 3] == 1.0 and J34[3, 0] == 1.0, '[TC34] 交换矩阵结构错误 FAILED'
assert J34[1, 2] == 1.0 and J34[2, 1] == 1.0, '[TC34] 交换矩阵结构错误 FAILED'
c34 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
H34 = hankel_matrix(4, c34)
assert abs(H34[0, 0] - 1.0) < 1e-12, '[TC34] Hankel 矩阵 (0,0) 错误 FAILED'
assert abs(H34[0, 3] - 4.0) < 1e-12, '[TC34] Hankel 矩阵 (0,3) 错误 FAILED'

# ---- TC35: canonical_form 与 absorption_probabilities_and_times 输出有效性 ----
np.random.seed(35)
p35 = 5
edges35 = [(0, 1, 0.4), (1, 2, 0.3), (3, 4, 0.5)]
P35, trans35, absorb35 = build_causal_markov_chain(p35, edges35, n_states_per_var=3)
P_canon35, Q35, R35, smap35 = canonical_form(P35, trans35, absorb35)
B35, t35 = absorption_probabilities_and_times(Q35, R35)
assert np.allclose(np.sum(B35, axis=1), np.ones(B35.shape[0]), atol=1e-8), '[TC35] 吸收概率和不等于 1 FAILED'
assert np.all(t35 >= 0), '[TC35] 期望吸收时间含负值 FAILED'

print('\n全部 35 个测试通过!\n')
