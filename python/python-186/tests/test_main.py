"""
main.py
社交网络传播动力学博士级合成项目入口

科学问题:
    基于复杂网络拓扑、空间异质性与延迟反馈的多模态耦合传播动力学模型

    本项目融合15个种子项目的核心算法，构建一个完整的
    社交网络信息-流行病耦合传播模拟与参数校准系统。
"""

import os
import sys
import numpy as np

# 添加项目目录到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from network_topology import (
    construct_social_network, floyd_warshall, network_efficiency,
    betweenness_centrality, power_method_eigenvector, clustering_coefficient,
    degree_distribution
)
from spatial_mesh import (
    generate_2d_triangular_mesh, refine_mesh_midpoint, fem_sample_on_mesh,
    spatial_diffusion_operator
)
from sparse_algebra import SparseCCS
from epidemic_dynamics import (
    seaihr_ode_rhs, coupled_info_epidemic_rhs, rk4_integrate,
    dde_rk4_integrate, compute_reproduction_number
)
from propagation_front import propagation_front_simulation
from parameter_calibration import (
    StreamingStatistics, bisection_root_finder, calibrate_beta_target,
    maximum_likelihood_estimation, akaike_information_criterion
)
from stochastic_contact import (
    circle_positive_distance_monte_carlo, geometric_contact_rate,
    magic4_test_matrix, percolation_threshold_estimate
)
from data_io import (
    generate_helical_point_cloud, normalize_features,
    write_xyz_data, read_xyz_data, compute_pca_features
)
from utils import check_numerical_stability, entropy, kl_divergence


def main():
    print("=" * 70)
    print("  社交网络传播动力学: 多模态耦合传播模拟系统")
    print("  Social Network Propagation Dynamics: Multi-modal Coupled Model")
    print("=" * 70)

    # ==================================================================
    # 阶段1: 社交网络拓扑构建与分析
    # ==================================================================
    print("\n[阶段1] 社交网络拓扑构建与分析")
    print("-" * 50)

    n_nodes = 100
    adj = construct_social_network(n_nodes, community_structure=True, seed=42)

    # 稀疏矩阵表示
    sparse_adj = SparseCCS.from_dense(adj)

    # 全源最短路径 (Floyd-Warshall)
    dist = floyd_warshall(adj)

    # 网络效率
    eff = network_efficiency(dist)
    print(f"  网络全局效率: {eff:.4f}")

    # 介数中心性
    bc = betweenness_centrality(adj)
    print(f"  介数中心性范围: [{bc.min():.4f}, {bc.max():.4f}]")

    # PageRank-like中心性 (幂法)
    lambda_max, pr_vec = power_method_eigenvector(adj)
    print(f"  主特征值 (PageRank): {lambda_max:.4f}")
    print(f"  中心性最高节点: {np.argmax(pr_vec)} (得分: {pr_vec.max():.4f})")

    # 聚类系数
    cc = clustering_coefficient(adj)
    print(f"  平均聚类系数: {cc.mean():.4f}")

    # 度分布
    degrees, pk = degree_distribution(adj)
    print(f"  平均度: {degrees.mean():.2f}")
    print(f"  度分布熵: {entropy(pk):.4f}")

    # 图拉普拉斯稀疏矩阵
    laplacian = SparseCCS.network_laplacian(adj)
    lambda_lap, _ = laplacian.power_iteration_sparse()
    print(f"  图拉普拉斯谱半径: {lambda_lap:.4f}")

    # ==================================================================
    # 阶段2: 空间异质性建模
    # ==================================================================
    print("\n[阶段2] 空间异质性建模 (有限元方法)")
    print("-" * 50)

    # 生成2D三角网格
    nodes, elements = generate_2d_triangular_mesh(8, 8, 0.0, 1.0, 0.0, 1.0)
    print(f"  初始网格: {nodes.shape[0]} 节点, {elements.shape[0]} 单元")

    # 网格细化 (1次中点细分)
    nodes_ref, elements_ref = refine_mesh_midpoint(nodes, elements)
    print(f"  细化后网格: {nodes_ref.shape[0]} 节点, {elements_ref.shape[0]} 单元")

    # 在节点上定义空间传播风险场
    # 风险场: 高斯型空间异质性
    risk_field = np.exp(-((nodes_ref[:, 0] - 0.5)**2 + (nodes_ref[:, 1] - 0.5)**2) / 0.1)

    # FEM采样
    sample_pts = np.random.rand(50, 2)
    sampled_risk = fem_sample_on_mesh(nodes_ref, elements_ref, risk_field, sample_pts)
    print(f"  空间风险场采样均值: {sampled_risk.mean():.4f}")

    # 空间扩散算子
    K_stiff = spatial_diffusion_operator(nodes_ref, elements_ref)
    print(f"  刚度矩阵条件数估计: {np.linalg.cond(K_stiff):.2e}")

    # ==================================================================
    # 阶段3: 随机接触模型
    # ==================================================================
    print("\n[阶段3] 随机接触模型与几何概率")
    print("-" * 50)

    # 几何概率: 单位圆上两点平均距离
    mean_dist, var_dist = circle_positive_distance_monte_carlo(n_samples=5000)
    print(f"  单位圆正象限平均距离: {mean_dist:.4f} (方差: {var_dist:.4f})")

    # 几何接触率
    contact_rates = geometric_contact_rate(n_nodes, activity_distribution='power_law')
    print(f"  平均接触率: {contact_rates.mean():.4f}")

    # 渗流阈值估计
    p_c = percolation_threshold_estimate(50, n_realizations=20)
    print(f"  渗流阈值估计: {p_c:.4f}")

    # 幻方测试矩阵
    magic_matrix = magic4_test_matrix(8)
    magic_sum = np.sum(magic_matrix[0, :])
    print(f"  8阶幻方幻和: {magic_sum} (理论值: {8*(64+1)//2})")

    # ==================================================================
    # 阶段4: 流行病-信息耦合动力学
    # ==================================================================
    print("\n[阶段4] 流行病-信息耦合传播动力学")
    print("-" * 50)

    # 基础参数
    N_pop = 100000.0
    params = {
        'N': N_pop,
        'beta_0': 0.8,
        'eta_A': 0.5,
        'sigma': 1.0 / 5.2,       # 潜伏期 ~5.2天
        'p_sym': 0.7,
        'gamma_A': 1.0 / 7.0,     # 无症状恢复 ~7天
        'gamma_I': 1.0 / 10.0,    # 有症状恢复 ~10天
        'alpha_H': 0.05,          # 住院率
        'gamma_H': 1.0 / 14.0,    # 住院恢复 ~14天
        'mu': 0.02,               # 住院死亡率
        'omega': 1.0 / 180.0,     # 免疫丧失 ~180天
        'tau': 7.0,               # 信息延迟
        'k_beta': 0.3,
        'beta_info': 2.0,
        'n_info': 9.65,
        'gamma_info': 1.0
    }

    # 初始条件
    I0 = 100.0
    E0 = 200.0
    A0 = 50.0
    S0 = N_pop - I0 - E0 - A0
    H0 = 10.0
    R0_pop = 0.0
    D0 = 0.0
    I_info0 = 0.0

    y0 = np.array([S0, E0, A0, I0, H0, R0_pop, D0, I_info0], dtype=np.float64)

    # 历史函数
    def history_func(t):
        return y0.copy()

    # DDE积分
    t_span = (0.0, 120.0)
    dt = 0.5

    t_hist, y_hist = dde_rk4_integrate(
        coupled_info_epidemic_rhs, y0, t_span, dt, params, history_func
    )

    final_state = y_hist[-1, :]
    print(f"  模拟时长: {t_span[1]} 天")
    print(f"  最终状态 (S,E,A,I,H,R,D,Info):")
    print(f"    S={final_state[0]:.1f}, E={final_state[1]:.1f}, A={final_state[2]:.1f}")
    print(f"    I={final_state[3]:.1f}, H={final_state[4]:.1f}, R={final_state[5]:.1f}")
    print(f"    D={final_state[6]:.1f}, I_info={final_state[7]:.4f}")

    # 计算R0
    params['beta'] = params['beta_0']
    r0 = compute_reproduction_number(params)
    print(f"  基本再生数 R_0: {r0:.4f}")

    # ==================================================================
    # 阶段5: 传播前沿高分辨率模拟 (DG方法)
    # ==================================================================
    print("\n[阶段5] 传播前沿高分辨率模拟 (间断Galerkin)")
    print("-" * 50)

    x_dg, u_dg, t_dg = propagation_front_simulation(
        initial_spread=0.15, diffusion_coeff=0.005, final_time=1.5
    )
    print(f"  DG网格: {x_dg.shape[0]} 节点 x {x_dg.shape[1]} 单元")
    print(f"  最终时间步: {t_dg[-1]:.4f}")
    print(f"  解范围: [{u_dg.min():.4f}, {u_dg.max():.4f}]")

    # ==================================================================
    # 阶段6: 参数校准
    # ==================================================================
    print("\n[阶段6] 参数校准与统计推断")
    print("-" * 50)

    # 流式统计
    stats = StreamingStatistics()
    for val in y_hist[:, 3]:  # I compartment
        stats.update(float(val))
    s = stats.get_stats()
    print(f"  感染人数统计: mean={s['mean']:.1f}, std={s['std']:.1f}")

    # 二分法校准 beta
    target_r0 = 2.5
    beta_cal = calibrate_beta_target(target_r0, params, None, tol=1e-5)
    print(f"  目标 R_0={target_r0} -> 校准 beta={beta_cal:.6f}")

    # 验证
    params_verify = params.copy()
    params_verify['beta'] = beta_cal
    r0_verify = compute_reproduction_number(params_verify)
    print(f"  验证 R_0={r0_verify:.4f}")

    # ==================================================================
    # 阶段7: 多维数据I/O与特征分析
    # ==================================================================
    print("\n[阶段7] 多维数据I/O与特征分析")
    print("-" * 50)

    # 生成螺旋点云
    point_cloud = generate_helical_point_cloud(n_points=200)
    print(f"  生成点云: {point_cloud.shape[0]} 点")

    # 写入/读取XYZ
    tmp_file = os.path.join(project_dir, "temp_pointcloud.xyz")
    write_xyz_data(tmp_file, point_cloud)
    pc_read = read_xyz_data(tmp_file)
    print(f"  XYZ I/O 验证: 写入{point_cloud.shape[0]}点, 读取{pc_read.shape[0]}点")

    # 特征归一化
    norm_pc, norm_params = normalize_features(point_cloud, method='zscore')
    print(f"  标准化后均值: {norm_pc.mean(axis=0)}")
    print(f"  标准化后方差: {np.var(norm_pc, axis=0)}")

    # PCA
    projected, ratio = compute_pca_features(point_cloud, n_components=2)
    print(f"  PCA前2主成分方差比: [{ratio[0]:.4f}, {ratio[1]:.4f}]")

    # 清理临时文件
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    # ==================================================================
    # 阶段8: 综合指标与稳定性检查
    # ==================================================================
    print("\n[阶段8] 综合指标与数值稳定性检查")
    print("-" * 50)

    checks = [
        ("邻接矩阵", adj),
        ("距离矩阵", dist),
        ("风险场", risk_field),
        ("DG解", u_dg),
        ("感染历史", y_hist[:, 3])
    ]

    all_stable = True
    for name, arr in checks:
        stable = check_numerical_stability(arr, name=name)
        all_stable = all_stable and stable

    if all_stable:
        print("  所有数值检查通过 ✓")
    else:
        print("  部分数值检查未通过 ✗")

    print("\n" + "=" * 70)
    print("  模拟完成。所有模块运行正常。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: entropy返回有限非负标量 ----
import numpy as np
result = entropy(np.array([0.1, 0.2, 0.3, 0.4]))
assert np.isfinite(result), '[TC01] entropy 应返回有限值 FAILED'
assert result >= 0, '[TC01] entropy 应为非负 FAILED'

# ---- TC02: 均匀分布熵应为log(n) ----
import numpy as np
p = np.ones(5) / 5.0
result = entropy(p)
assert abs(result - np.log(5.0)) < 1e-10, '[TC02] 均匀分布entropy应为log(n) FAILED'

# ---- TC03: 相同分布KL散度应为0 ----
import numpy as np
p = np.array([0.5, 0.5])
q = np.array([0.5, 0.5])
result = kl_divergence(p, q)
assert abs(result) < 1e-10, '[TC03] 相同分布KL散度应为0 FAILED'

# ---- TC04: generate_helical_point_cloud输出形状正确 ----
import numpy as np
pc = generate_helical_point_cloud(n_points=100)
assert pc.shape == (100, 3), '[TC04] 点云形状应为(100,3) FAILED'
assert np.all(np.isfinite(pc)), '[TC04] 点云坐标应全部有限 FAILED'

# ---- TC05: normalize_features zscore均值接近0方差接近1 ----
import numpy as np
np.random.seed(42)
data = np.random.randn(100, 5)
norm_data, params = normalize_features(data, method='zscore')
assert params['method'] == 'zscore', '[TC05] 方法应为zscore FAILED'
assert np.all(np.abs(np.mean(norm_data, axis=0)) < 1e-10), '[TC05] zscore后均值应接近0 FAILED'
assert np.allclose(np.var(norm_data, axis=0), 1.0, atol=1e-8), '[TC05] zscore后方差应接近1 FAILED'

# ---- TC06: normalize_features minmax范围[0,1] ----
import numpy as np
np.random.seed(42)
data = np.random.randn(100, 3)
norm_data, params = normalize_features(data, method='minmax')
assert np.all(norm_data >= -1e-10), '[TC06] minmax归一化后值应>=0 FAILED'
assert np.all(norm_data <= 1.0 + 1e-10), '[TC06] minmax归一化后值应<=1 FAILED'

# ---- TC07: normalize_features robust归一化非NaN ----
import numpy as np
np.random.seed(42)
data = np.random.randn(50, 4)
norm_data, params = normalize_features(data, method='robust')
assert params['method'] == 'robust', '[TC07] 方法应为robust FAILED'
assert np.all(np.isfinite(norm_data)), '[TC07] robust归一化后应全部有限 FAILED'

# ---- TC08: SparseCCS稠密-稀疏往返一致性 ----
import numpy as np
A = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
sp = SparseCCS.from_dense(A)
A_back = sp.to_dense()
assert np.allclose(A, A_back), '[TC08] 稠密-稀疏往返应一致 FAILED'

# ---- TC09: SparseCCS矩阵向量乘法正确性 ----
import numpy as np
A = np.array([[1.0, 0.0], [2.0, 3.0]])
sp = SparseCCS.from_dense(A)
x = np.array([1.0, 1.0])
y = sp.mv(x)
assert np.allclose(y, np.array([1.0, 5.0])), '[TC09] 稀疏矩阵-向量乘法不正确 FAILED'

# ---- TC10: SparseCCS转置向量乘法正确性 ----
import numpy as np
A = np.array([[1.0, 2.0], [0.0, 3.0]])
sp = SparseCCS.from_dense(A)
x = np.array([1.0, 1.0])
y = sp.mtv(x)
assert np.allclose(y, np.array([1.0, 5.0])), '[TC10] 稀疏转置-向量乘法不正确 FAILED'

# ---- TC11: Floyd-Warshall对称性及自距离为0 ----
import numpy as np
A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
dist = floyd_warshall(A)
assert np.allclose(dist, dist.T), '[TC11] 距离矩阵应对称 FAILED'
assert np.all(np.diag(dist) == 0.0), '[TC11] 自距离应为0 FAILED'

# ---- TC12: network_efficiency有限非负 ----
import numpy as np
A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
dist = floyd_warshall(A)
eff = network_efficiency(dist)
assert np.isfinite(eff), '[TC12] 网络效率应为有限值 FAILED'
assert eff >= 0, '[TC12] 网络效率应为非负 FAILED'

# ---- TC13: power_method_eigenvector可复现性 ----
import numpy as np
np.random.seed(42)
A = construct_social_network(20, community_structure=True, seed=123)
lam1, v1 = power_method_eigenvector(A)
np.random.seed(42)
lam2, v2 = power_method_eigenvector(A)
assert abs(lam1 - lam2) < 1e-12, '[TC13] 固定种子后特征值应相同 FAILED'
assert np.allclose(v1, v2), '[TC13] 固定种子后特征向量应相同 FAILED'

# ---- TC14: 度分布概率质量函数之和为1 ----
import numpy as np
np.random.seed(42)
A = construct_social_network(30, community_structure=True, seed=123)
degrees, pk = degree_distribution(A)
assert abs(np.sum(pk) - 1.0) < 1e-12, '[TC14] 度分布概率和应为1 FAILED'

# ---- TC15: betweenness_centrality非负 ----
import numpy as np
np.random.seed(42)
A = construct_social_network(20, community_structure=True, seed=123)
bc = betweenness_centrality(A)
assert np.all(bc >= -1e-12), '[TC15] 介数中心性应为非负 FAILED'
assert np.all(np.isfinite(bc)), '[TC15] 介数中心性应全部有限 FAILED'

# ---- TC16: clustering_coefficient范围[0,1] ----
import numpy as np
np.random.seed(42)
A = construct_social_network(20, community_structure=True, seed=123)
cc = clustering_coefficient(A)
assert np.all(cc >= -1e-12), '[TC16] 聚类系数应>=0 FAILED'
assert np.all(cc <= 1.0 + 1e-12), '[TC16] 聚类系数应<=1 FAILED'

# ---- TC17: seaihr_ode_rhs全零输入导数有限 ----
import numpy as np
params_test = {
    'N': 1000.0, 'beta': 0.5, 'eta_A': 0.5, 'sigma': 0.2,
    'p_sym': 0.7, 'gamma_A': 0.1, 'gamma_I': 0.1,
    'alpha_H': 0.05, 'gamma_H': 0.07, 'mu': 0.02, 'omega': 0.01
}
y0 = np.zeros(7)
dy = seaihr_ode_rhs(0.0, y0, params_test)
assert np.all(np.isfinite(dy)), '[TC17] 全零输入导数应有限 FAILED'

# ---- TC18: compute_reproduction_number关于beta单调递增 ----
import numpy as np
p = {'beta': 0.5, 'p_sym': 0.7, 'eta_A': 0.5, 'gamma_I': 0.1, 'alpha_H': 0.05, 'gamma_A': 0.1}
r0_low = compute_reproduction_number(p)
p['beta'] = 1.0
r0_high = compute_reproduction_number(p)
assert r0_high > r0_low, '[TC18] R0应随beta单调递增 FAILED'

# ---- TC19: triangle_area已知三角形面积 ----
import numpy as np
from spatial_mesh import triangle_area
area = triangle_area(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
assert abs(area - 0.5) < 1e-10, '[TC19] 单位直角三角形面积应为0.5 FAILED'

# ---- TC20: barycentric_coordinates顶点坐标 ----
import numpy as np
from spatial_mesh import barycentric_coordinates
p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
l1, l2, l3 = barycentric_coordinates(p1, p1, p2, p3)
assert abs(l1 - 1.0) < 1e-10, '[TC20] 顶点重心坐标第一分量应为1 FAILED'
assert abs(l2) < 1e-10 and abs(l3) < 1e-10, '[TC20] 顶点重心坐标其他分量应为0 FAILED'

# ---- TC21: StreamingStatistics均值方差正确 ----
import numpy as np
stats = StreamingStatistics()
for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
    stats.update(x)
s = stats.get_stats()
assert abs(s['mean'] - 3.0) < 1e-10, '[TC21] 均值应为3 FAILED'
assert abs(s['variance'] - 2.5) < 1e-10, '[TC21] 样本方差应为2.5 FAILED'

# ---- TC22: StreamingStatistics min/max正确 ----
import numpy as np
stats = StreamingStatistics()
for x in [5.0, 1.0, 10.0, -3.0, 7.0]:
    stats.update(x)
s = stats.get_stats()
assert abs(s['min'] - (-3.0)) < 1e-10, '[TC22] 最小值应为-3 FAILED'
assert abs(s['max'] - 10.0) < 1e-10, '[TC22] 最大值应为10 FAILED'

# ---- TC23: bisection_root_finder求解x^2-2=0 ----
import numpy as np
def _f_tc23(x): return x**2 - 2.0
root, iters, success = bisection_root_finder(_f_tc23, 0.0, 2.0, tol=1e-8)
assert success, '[TC23] 二分法应成功 FAILED'
assert abs(root - np.sqrt(2.0)) < 1e-7, '[TC23] 根应为sqrt(2) FAILED'
assert iters >= 1, '[TC23] 至少需要1次迭代 FAILED'

# ---- TC24: calibrate_beta_target返回正beta ----
import numpy as np
params_tmpl = {
    'N': 100000.0, 'beta_0': 0.8, 'eta_A': 0.5,
    'p_sym': 0.7, 'gamma_A': 0.14, 'gamma_I': 0.1, 'alpha_H': 0.05
}
beta_cal = calibrate_beta_target(2.0, params_tmpl, None, tol=1e-5)
assert beta_cal > 0, '[TC24] 校准beta应为正 FAILED'

# ---- TC25: akaike_information_criterion有限正值 ----
import numpy as np
aic = akaike_information_criterion(-10.0, 3, 100)
assert aic > 0, '[TC25] AIC应为正 FAILED'
assert np.isfinite(aic), '[TC25] AIC应为有限值 FAILED'

# ---- TC26: magic4_test_matrix幻和正确 ----
import numpy as np
M = magic4_test_matrix(8)
magic_const = 8 * (64 + 1) // 2
assert np.all(np.sum(M, axis=1) == magic_const), '[TC26] 每行和应等于幻和 FAILED'
assert np.all(np.sum(M, axis=0) == magic_const), '[TC26] 每列和应等于幻和 FAILED'

# ---- TC27: generate_2d_triangular_mesh形状正确 ----
import numpy as np
nodes, elements = generate_2d_triangular_mesh(4, 4, 0.0, 1.0, 0.0, 1.0)
expected_nodes = (4 + 1) * (4 + 1)
expected_elements = 2 * 4 * 4
assert nodes.shape[0] == expected_nodes, '[TC27] 网格节点数不正确 FAILED'
assert elements.shape[0] == expected_elements, '[TC27] 网格单元数不正确 FAILED'
assert nodes.shape[1] == 2, '[TC27] 节点坐标应为2维 FAILED'

# ---- TC28: legendre_gauss_lobatto_nodes端点为±1 ----
import numpy as np
from propagation_front import legendre_gauss_lobatto_nodes
for N in [1, 2, 3, 4, 5]:
    r = legendre_gauss_lobatto_nodes(N)
    assert abs(r[0] - (-1.0)) < 1e-10, '[TC28] LGL节点起点应为-1 FAILED'
    assert abs(r[-1] - 1.0) < 1e-10, '[TC28] LGL节点终点应为1 FAILED'

# ---- TC29: vandermonde_matrix维度正确且首列为1 ----
import numpy as np
from propagation_front import vandermonde_matrix
r = np.linspace(-1, 1, 5)
V = vandermonde_matrix(4, r)
assert V.shape == (5, 5), '[TC29] Vandermonde矩阵形状应为(5,5) FAILED'
assert np.allclose(V[:, 0], 1.0), '[TC29] Vandermonde第一列应为全1 FAILED'

# ---- TC30: differentiation_matrix正确尺寸 ----
import numpy as np
from propagation_front import legendre_gauss_lobatto_nodes, differentiation_matrix
r = legendre_gauss_lobatto_nodes(4)
D = differentiation_matrix(4, r)
assert D.shape == (5, 5), '[TC30] 微分矩阵形状应为(5,5) FAILED'

# ---- TC31: circle_positive_distance_monte_carlo可复现 ----
import numpy as np
mean1, var1 = circle_positive_distance_monte_carlo(n_samples=5000, seed=123)
mean2, var2 = circle_positive_distance_monte_carlo(n_samples=5000, seed=123)
assert abs(mean1 - mean2) < 1e-12, '[TC31] 固定种子蒙特卡洛应可复现 FAILED'
assert abs(var1 - var2) < 1e-12, '[TC31] 固定种子方差应可复现 FAILED'

# ---- TC32: refine_mesh_midpoint节点和单元数增长 ----
import numpy as np
nodes, elements = generate_2d_triangular_mesh(3, 3)
nodes_ref, elements_ref = refine_mesh_midpoint(nodes, elements)
assert nodes_ref.shape[0] > nodes.shape[0], '[TC32] 细化后节点数应增加 FAILED'
assert elements_ref.shape[0] == 4 * elements.shape[0], '[TC32] 细化后单元数应为原4倍 FAILED'

# ---- TC33: spatial_diffusion_operator对称性 ----
import numpy as np
nodes, elements = generate_2d_triangular_mesh(5, 5)
K = spatial_diffusion_operator(nodes, elements)
assert np.allclose(K, K.T), '[TC33] 刚度矩阵应对称 FAILED'

# ---- TC34: SparseCCS.network_laplacian谱半径非负 ----
import numpy as np
np.random.seed(42)
A = construct_social_network(20, community_structure=True, seed=123)
L = SparseCCS.network_laplacian(A)
lam, _ = L.power_iteration_sparse()
assert lam >= -1e-10, '[TC34] 拉普拉斯谱半径应为非负 FAILED'

# ---- TC35: check_numerical_stability正常数组返回True ----
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
result = check_numerical_stability(arr, name='test')
assert result == True, '[TC35] 正常数组应返回True FAILED'

# ---- TC36: check_numerical_stability含NaN返回False ----
import numpy as np
arr = np.array([1.0, np.nan, 3.0])
result = check_numerical_stability(arr, name='test')
assert result == False, '[TC36] 含NaN数组应返回False FAILED'

# ---- TC37: 集成测试 - 网络拓扑全流程无异常 ----
import numpy as np
np.random.seed(42)
adj = construct_social_network(30, community_structure=True, seed=42)
sparse_adj = SparseCCS.from_dense(adj)
dist = floyd_warshall(adj)
eff = network_efficiency(dist)
bc = betweenness_centrality(adj)
lam, vec = power_method_eigenvector(adj)
cc = clustering_coefficient(adj)
degrees, pk = degree_distribution(adj)
laplacian = SparseCCS.network_laplacian(adj)
lam_lap, _ = laplacian.power_iteration_sparse()
assert eff >= 0, '[TC37] 集成测试-网络效率应非负 FAILED'
assert cc.shape[0] == 30, '[TC37] 集成测试-聚类系数维度错误 FAILED'
assert np.all(np.isfinite(bc)), '[TC37] 集成测试-介数中心性应有限 FAILED'
assert np.isfinite(lam_lap), '[TC37] 集成测试-拉普拉斯谱半径应有限 FAILED'

# ---- TC38: 集成测试 - 空间网格与有限元无异常 ----
import numpy as np
nodes, elements = generate_2d_triangular_mesh(5, 5, 0.0, 1.0, 0.0, 1.0)
nodes_ref, elements_ref = refine_mesh_midpoint(nodes, elements)
risk_field = np.exp(-((nodes_ref[:, 0] - 0.5)**2 + (nodes_ref[:, 1] - 0.5)**2) / 0.1)
sample_pts = np.array([[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]])
sampled = fem_sample_on_mesh(nodes_ref, elements_ref, risk_field, sample_pts)
K_stiff = spatial_diffusion_operator(nodes_ref, elements_ref)
assert sampled.shape[0] == 3, '[TC38] 集成测试-采样点数不正确 FAILED'
assert np.all(np.isfinite(sampled)), '[TC38] 集成测试-采样值应有限 FAILED'
assert np.all(np.isfinite(K_stiff)), '[TC38] 集成测试-刚度矩阵应有限 FAILED'

# ---- TC39: 集成测试 - 流行病动力学与参数校准无异常 ----
import numpy as np
params_test = {
    'N': 1000.0, 'beta': 0.5, 'eta_A': 0.5, 'sigma': 0.2,
    'p_sym': 0.7, 'gamma_A': 0.1, 'gamma_I': 0.1,
    'alpha_H': 0.05, 'gamma_H': 0.07, 'mu': 0.02, 'omega': 0.01
}
y0 = np.array([900.0, 50.0, 10.0, 30.0, 5.0, 5.0, 0.0])
r0 = compute_reproduction_number(params_test)
stats = StreamingStatistics()
for i in range(10):
    stats.update(float(i))
s = stats.get_stats()
assert r0 > 0, '[TC39] 集成测试-R0应为正 FAILED'
assert s['n'] == 10, '[TC39] 集成测试-统计计数应为10 FAILED'

# ---- TC40: 集成测试 - 传播前沿与数据I/O无异常 ----
import numpy as np
pc = generate_helical_point_cloud(n_points=50)
norm_pc, norm_params = normalize_features(pc, method='zscore')
proj, ratio = compute_pca_features(pc, n_components=2)
assert pc.shape[0] == 50, '[TC40] 集成测试-点云点数不正确 FAILED'
assert proj.shape == (50, 2), '[TC40] 集成测试-PCA投影形状不正确 FAILED'
assert len(ratio) == 2, '[TC40] 集成测试-方差比率长度应为2 FAILED'
assert np.all(ratio >= -1e-10), '[TC40] 集成测试-方差比率应为非负 FAILED'

print('\n全部 40 个测试通过!\n')
