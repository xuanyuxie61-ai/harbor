# ---- TC01: chebyshev_grid 返回正确数量的节点 ----
x = chebyshev_grid(10)
assert len(x) == 11, '[TC01] chebyshev_grid 节点数 FAILED'

# ---- TC02: chebyshev_grid 节点均在 [-1, 1] 区间内 ----
x = chebyshev_grid(15)
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC02] chebyshev_grid 范围 FAILED'

# ---- TC03: chebyshev_diff_matrix 输出形状正确 ----
D = chebyshev_diff_matrix(8)
assert D.shape == (9, 9), '[TC03] chebyshev_diff_matrix 形状 FAILED'

# ---- TC04: chebyshev_diff_matrix 行和为零 ----
D = chebyshev_diff_matrix(12)
assert np.allclose(np.sum(D, axis=1), 0.0, atol=1e-10), '[TC04] chebyshev_diff_matrix 行和 FAILED'

# ---- TC05: chebyshev_barycentric_interpolate 在节点处插值准确 ----
import numpy as np
from chebyshev_pricing import chebyshev_barycentric_interpolate
x_grid = chebyshev_grid(10)
v = np.sin(np.pi * x_grid)
v_interp = chebyshev_barycentric_interpolate(x_grid, v, x_grid)
assert np.allclose(v_interp, v, atol=1e-12), '[TC05] chebyshev重心插值 FAILED'

# ---- TC06: circle01_monomial_integral 奇指数返回零 ----
val = circle01_monomial_integral(np.array([1, 2]))
assert val == 0.0, '[TC06] circle01 奇指数积分 FAILED'

# ---- TC07: circle01_monomial_integral 零指数解析解(积分值为2π) ----
val = circle01_monomial_integral(np.array([0, 0]))
assert abs(val - 2.0 * np.pi) < 1e-10, '[TC07] circle01 (0,0) 积分 FAILED'

# ---- TC08: spectral_var_cvar 返回字典含必要键 ----
np.random.seed(42)
sample_returns = np.random.randn(200) * 0.02
result = spectral_var_cvar(sample_returns, alpha=0.05, n_cheb=32)
assert 'VaR' in result and 'CVaR' in result and 'mean' in result and 'std' in result, '[TC08] spectral_var_cvar 返回键 FAILED'

# ---- TC09: spectral_var_cvar 的 CVaR 不超过 VaR（尾部条件期望性质）----
np.random.seed(42)
sample_returns = np.random.randn(200) * 0.02
result = spectral_var_cvar(sample_returns, alpha=0.05, n_cheb=32)
assert result['CVaR'] <= result['VaR'] + 1e-12, '[TC09] CVaR <= VaR FAILED'

# ---- TC10: sphere_distance1 同点距离为零 ----
from spherical_embedding import sphere_distance1
d = sphere_distance1(0.5, 0.3, 0.5, 0.3, r=1.0)
assert abs(d) < 1e-12, '[TC10] sphere_distance1 同点距离 FAILED'

# ---- TC11: spherical_diversity_index 对相反点取最大值2.0 ----
points = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
div = spherical_diversity_index(points)
assert abs(div - 2.0) < 1e-12, '[TC11] spherical_diversity_index 最大值 FAILED'

# ---- TC12: angular_distance_matrix 对角线为零 ----
xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
ad = angular_distance_matrix(xyz)
assert np.allclose(np.diag(ad), 0.0, atol=1e-12), '[TC12] angular_distance_matrix 对角线 FAILED'

# ---- TC13: build_asset_digraph 孤立节点获得自环 ----
np.random.seed(42)
corr_test = np.eye(4)
adj = build_asset_digraph(4, threshold=0.5, corr=corr_test)
assert np.all(np.diag(adj) > 0), '[TC13] build_asset_digraph 孤立节点自环 FAILED'

# ---- TC14: pagerank_systemic_risk 得分和为1 ----
adj_test = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=float)
scores = pagerank_systemic_risk(adj_test, damping=0.85)
assert abs(np.sum(scores) - 1.0) < 1e-10, '[TC14] pagerank 得分和 FAILED'

# ---- TC15: markowitz_min_variance 权重和为1且非负 ----
Sigma = np.diag([0.04, 0.09, 0.16, 0.25])
result = markowitz_min_variance(Sigma)
w = result['weights']
assert abs(np.sum(w) - 1.0) < 1e-12, '[TC15] markowitz 权重和 FAILED'
assert np.all(w >= -1e-12), '[TC15] markowitz 权重非负 FAILED'

# ---- TC16: risk_parity_weights 返回权重和为1 ----
Sigma = np.diag([0.04, 0.09, 0.16, 0.25])
result = risk_parity_weights(Sigma, max_iter=2000, tol=1e-10)
assert abs(np.sum(result['weights']) - 1.0) < 1e-12, '[TC16] risk_parity 权重和 FAILED'

# ---- TC17: herfindahl_risk_concentration 等贡献时为1/n ----
n_test = 5
rc_equal = np.ones(n_test) / n_test
h = herfindahl_risk_concentration(rc_equal)
assert abs(h - 1.0/n_test) < 1e-12, '[TC17] herfindahl 等贡献 FAILED'

# ---- TC18: effective_number_of_bets 等贡献时为n ----
n_test = 5
rc_equal = np.ones(n_test) / n_test
enb = effective_number_of_bets(rc_equal)
assert abs(enb - n_test) < 1e-6, '[TC18] effective_number_of_bets FAILED'

# ---- TC19: simplex_lattice_points 格点数符合组合公式 ----
from math import comb
n_dim, t_val = 4, 5
pts = simplex_lattice_points(n_dim, t_val)
expected_count = comb(n_dim + t_val - 1, t_val)
assert len(pts) == expected_count, '[TC19] simplex_lattice_points 格点数 FAILED'
assert np.all(pts.sum(axis=1) == t_val), '[TC19] simplex_lattice_points 行和 FAILED'

# ---- TC20: simplex_volume 二维单位单纯形体积为1/2 ----
import math
np.math = math
pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
vol = simplex_volume(pts)
assert abs(vol - 0.5) < 1e-12, '[TC20] simplex_volume FAILED'

# ---- TC21: coupled_market_dynamics 输出维度正确 ----
n_assets = 3
y = np.zeros(6)
K2 = np.zeros((3, 3))
m = np.ones(3)
dy = coupled_market_dynamics(y, 0.0, k1=1.0, K2=K2, gamma=0.5, m=m)
assert len(dy) == 6, '[TC21] coupled_market_dynamics 输出维度 FAILED'

# ---- TC22: caesar_perturb 输出形状不变 ----
np.random.seed(42)
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
perturbed = caesar_perturb(data, k=3)
assert len(perturbed) == len(data), '[TC22] caesar_perturb 长度 FAILED'

# ---- TC23: matrix_interpolation_upsample 输出形状为输入的2倍 ----
A = np.arange(16, dtype=float).reshape(4, 4)
up = matrix_interpolation_upsample(A, factor=2)
assert up.shape == (8, 8), '[TC23] matrix_interpolation_upsample 形状 FAILED'

# ---- TC24: r8mat_condition_number 单位矩阵条件数为1 ----
I = np.eye(5)
cond = r8mat_condition_number(I)
assert abs(cond - 1.0) < 1e-10, '[TC24] r8mat_condition_number 单位阵 FAILED'

# ---- TC25: generate_synthetic_data 返回正确的键与形状 ----
np.random.seed(42)
data = generate_synthetic_data(n_assets=6, n_days=100, seed=42)
assert 'returns' in data and 'mu' in data and 'sigma' in data and 'corr' in data, '[TC25] generate_synthetic_data 键 FAILED'
assert data['returns'].shape == (100, 6), '[TC25] generate_synthetic_data returns 形状 FAILED'
assert len(data['mu']) == 6, '[TC25] generate_synthetic_data mu 长度 FAILED'
assert data['corr'].shape == (6, 6), '[TC25] generate_synthetic_data corr shape FAILED'

# ---- TC26: 集成测试——完整 main() 无错误运行 ----
import sys
np.random.seed(42)
ret = main()
assert ret == 0, '[TC26] main() 返回码 FAILED'
