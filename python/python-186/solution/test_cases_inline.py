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
