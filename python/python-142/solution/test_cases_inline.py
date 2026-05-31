# ---- TC01: gauss_legendre_nodes_weights 节点数与权重和 ----
from utils import gauss_legendre_nodes_weights
x, w = gauss_legendre_nodes_weights(10)
assert len(x) == 10 and len(w) == 10, '[TC01] 节点/权重长度错误 FAILED'
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC01] 权重和不等于 2 FAILED'
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC01] 节点不在 [-1,1] 内 FAILED'

# ---- TC02: logistic_transform 区间映射 ----
from utils import logistic_transform
z = np.array([-100.0, -5.0, 0.0, 5.0, 100.0])
y = logistic_transform(z, a=0.0, b=1.0)
assert np.all(y >= 0.0) and np.all(y <= 1.0), '[TC02] 映射越界 FAILED'
assert y[2] > 0.49 and y[2] < 0.51, '[TC02] z=0 未映射到 0.5 FAILED'

# ---- TC03: safe_sqrt 负数返回 0 ----
from utils import safe_sqrt
x = np.array([-5.0, -0.01, 0.0, 4.0, 9.0])
y = safe_sqrt(x)
assert abs(y[0] - 0.0) < 1e-15, '[TC03] 负数未返回 0 FAILED'
assert abs(y[2] - 0.0) < 1e-15, '[TC03] 零输入错误 FAILED'
assert abs(y[3] - 2.0) < 1e-15, '[TC03] sqrt(4) 应为 2 FAILED'
assert abs(y[4] - 3.0) < 1e-15, '[TC03] sqrt(9) 应为 3 FAILED'

# ---- TC04: normal_cdf(0) ≈ 0.5 ----
assert abs(normal_cdf(np.array([0.0]))[0] - 0.5) < 1e-10, '[TC04] normal_cdf(0) 偏离 0.5 FAILED'
x_test = np.array([-3.0, 0.0, 3.0])
cdf_vals = normal_cdf(x_test)
assert cdf_vals[0] < 0.01, '[TC04] CDF(-3) 应接近 0 FAILED'
assert cdf_vals[2] > 0.99, '[TC04] CDF(3) 应接近 1 FAILED'

# ---- TC05: is_positive_definite 单位阵 ----
from utils import is_positive_definite
I = np.eye(5)
assert is_positive_definite(I), '[TC05] 单位阵应为正定 FAILED'
M_sing = np.ones((4, 4))
assert not is_positive_definite(M_sing), '[TC05] 奇异阵应判定为非正定 FAILED'

# ---- TC06: nearest_correlation_matrix 对角线全为 1 ----
np.random.seed(42)
A_bad = np.random.randn(6, 6)
A_bad = A_bad @ A_bad.T
A_bad = A_bad / np.max(np.abs(A_bad))
R = nearest_correlation_matrix(A_bad)
assert R.shape == (6, 6), '[TC06] 输出维度错误 FAILED'
assert np.allclose(np.diag(R), 1.0, atol=1e-6), '[TC06] 对角线不全为 1 FAILED'
eigvals_R = np.linalg.eigvalsh(R)
assert np.all(eigvals_R > -1e-8), '[TC06] 输出非半正定 FAILED'

# ---- TC07: tridiagonal_solve 对角系统精确解 ----
from utils import tridiagonal_solve
n_tri = 5
b_diag = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
a_sub = np.array([0.0, 0.0, 0.0, 0.0])
c_sup = np.array([0.0, 0.0, 0.0, 0.0])
x_exact = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
d_rhs = b_diag * x_exact  # 纯对角: b_i * x_i = d_i
x_tri = tridiagonal_solve(a_sub, b_diag, c_sup, d_rhs)
assert np.allclose(x_tri, x_exact, atol=1e-12), '[TC07] 对角系统求解错误 FAILED'

# ---- TC08: modified_gram_schmidt Q 列正交归一 ----
np.random.seed(42)
A_test = np.random.randn(8, 5)
Q, R, rank = modified_gram_schmidt(A_test)
assert Q.shape[1] == rank, '[TC08] Q 列数与秩不匹配 FAILED'
assert rank <= 5, '[TC08] 秩超过列数 FAILED'
QtQ = Q.T @ Q
assert np.allclose(QtQ, np.eye(rank), atol=1e-10), '[TC08] Q 列非正交 FAILED'

# ---- TC09: orthogonalize_credit_factors 载荷行范式 ≤ 1 ----
np.random.seed(42)
raw = np.random.randn(20, 6)
orth = orthogonalize_credit_factors(raw, method="mgs")
row_norms = np.sqrt(np.sum(orth**2, axis=1))
assert np.all(row_norms <= 1.0 + 1e-10), '[TC09] 载荷行范式超过 1 FAILED'
assert orth.shape[0] == 20, '[TC09] 输出行数错误 FAILED'

# ---- TC10: portfolio_weight_grid 权重和为 1 ----
np.random.seed(42)
weights = portfolio_weight_grid(3, 5)
assert weights.ndim == 2 and weights.shape[1] == 3, '[TC10] 权重矩阵维度错误 FAILED'
assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12), '[TC10] 权重和不等于 1 FAILED'
assert np.all(weights >= -1e-12), '[TC10] 权重存在负值 FAILED'

# ---- TC11: correlation_simplex_grid 特征值和等于 n_factors ----
n_f = 3
e_grids = correlation_simplex_grid(n_f, 4)
assert len(e_grids) > 0, '[TC11] 特征值网格为空 FAILED'
for ev in e_grids:
    assert abs(np.sum(ev) - n_f) < 1e-10, '[TC11] 特征值和不等于 n_factors FAILED'

# ---- TC12: cg_ne_solve 用 Helmert 矩阵精确恢复 ----
np.random.seed(42)
n_h = 15
H = helmert_matrix(n_h)
x_true = np.random.randn(n_h)
b_h = H @ x_true
x_sol, iters, res = cg_ne_solve(H, b_h, tol=1e-12)
err = np.linalg.norm(x_sol - x_true)
assert err < 1e-8, '[TC12] CG-NE 解误差过大 FAILED'
assert iters > 0, '[TC12] CG-NE 未迭代 FAILED'

# ---- TC13: clenshaw_curtis_nodes 范围正确 ----
cc = clenshaw_curtis_nodes(16, a=0.0, b=10.0)
assert len(cc) == 16, '[TC13] Clenshaw-Curtis 节点数错误 FAILED'
assert np.all(cc >= 0.0) and np.all(cc <= 10.0), '[TC13] CC 节点越界 FAILED'

# ---- TC14: interpolate_credit_curve 节点精确恢复 ----
t_data = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
p_data = np.array([0.005, 0.012, 0.025, 0.038, 0.055, 0.072, 0.095])
p_check = interpolate_credit_curve(t_data, p_data, t_data, method="linear")
assert np.allclose(p_check, p_data, atol=1e-12), '[TC14] 线性插值节点恢复失败 FAILED'

# ---- TC15: sphere_llt_grid_points 点数正确 ----
lat, lon = 3, 4
xyz_pts = sphere_llt_grid_points(1.0, np.zeros(3), lat, lon)
assert xyz_pts.shape[0] == 2 + lat * lon, '[TC15] LLT 网格点数错误 FAILED'
assert xyz_pts.shape[1] == 3, '[TC15] LLT 网格坐标维度错误 FAILED'

# ---- TC16: spherical_voronoi_areas 面积非负 ----
areas_v, faces_v = spherical_voronoi_areas(xyz_pts)
assert np.all(areas_v >= -1e-12), '[TC16] Voronoi 面积存在负值 FAILED'
assert len(areas_v) == xyz_pts.shape[0], '[TC16] 面积数量与点数不匹配 FAILED'

# ---- TC17: simulate_default_contagion 状态在 [0,1] 内 ----
np.random.seed(42)
t_ode, y_ode = simulate_default_contagion(
    initial_default_rate=0.05,
    initial_pressure=0.1,
    initial_buffer=0.5,
    t_max=2.0,
    n_steps=200,
    theta=0.5
)
assert len(t_ode) == 201, '[TC17] 时间步数错误 FAILED'
assert np.all(y_ode[:, 0] >= -1e-12) and np.all(y_ode[:, 0] <= 1.0 + 1e-12), '[TC17] 违约强度越界 FAILED'
assert np.all(y_ode[:, 1] >= -1e-12) and np.all(y_ode[:, 1] <= 1.0 + 1e-12), '[TC17] 传染压力越界 FAILED'
assert np.all(y_ode[:, 2] >= -1e-12) and np.all(y_ode[:, 2] <= 1.0 + 1e-12), '[TC17] 缓冲水平越界 FAILED'

# ---- TC18: network_cascade_intensity 网络效应增强 ----
adj_test = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
local_test = np.array([0.1, 0.2, 0.15])
net_test = network_cascade_intensity(adj_test, local_test, 0.1)
assert np.all(net_test >= local_test - 1e-12), '[TC18] 网络效应未增强 FAILED'
assert np.all(net_test <= 1.0 + 1e-12), '[TC18] 网络增强强度越界 FAILED'

# ---- TC19: solve_ill_bvp_fd 边界条件满足 ----
x_bvp, y_bvp = solve_ill_bvp_fd(epsilon=0.01, n_nodes=80, bc_left=2.0, bc_right=1.0)
assert np.isclose(y_bvp[0], 2.0, atol=1e-6), '[TC19] 左边界条件不满足 FAILED'
assert np.isclose(y_bvp[-1], 1.0, atol=1e-6), '[TC19] 右边界条件不满足 FAILED'
assert len(x_bvp) == 80, '[TC19] BVP 节点数错误 FAILED'

# ---- TC20: default_probability_from_structural 在 [0,1] 内 ----
pd_s1 = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 5.0)
pd_s2 = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 10.0)
pd_s3 = default_probability_from_structural(30.0, 0.05, 0.2, 100.0, 5.0)
assert 0.0 <= pd_s1 <= 1.0, '[TC20] PD_5y 越界 FAILED'
assert 0.0 <= pd_s2 <= 1.0, '[TC20] PD_10y 越界 FAILED'
assert 0.0 <= pd_s3 <= 1.0, '[TC20] PD_v0<barrier 越界 FAILED'
assert pd_s1 < pd_s2, '[TC20] 5年PD应小于10年PD FAILED'

# ---- TC21: integrate_standard_triangle 常数函数面积 ----
f_const = lambda x, y: 1.0
tri_area = integrate_standard_triangle(f_const, rule=7)
assert abs(tri_area - 0.5) < 1e-8, '[TC21] 标准三角形面积不为 0.5 FAILED'

# ---- TC22: cauchy_principal_value 已知解析值 ----
f_cpv = lambda t: t**2
cpv_val = cauchy_principal_value(f_cpv, -1.0, 2.0, 0.5, n=64)
# 解析: int_{-1}^2 t^2/(t-0.5) = int_{-1}^2 (t+0.5 + 0.25/(t-0.5)) dt
# = [t^2/2 + 0.5t]_{-1}^2 + 0.25*ln|(2-0.5)/(-1-0.5)|
# = (2+1-0.5+0.5) + 0.25*ln(1.5/1.5) = 3 + 0 = 3
expected_cpv = 3.0
assert abs(cpv_val - expected_cpv) < 1e-6, '[TC22] CPV 积分值错误 FAILED'

# ---- TC23: knapsack_01_dp 简单已知解 ----
from portfolio_knapsack import knapsack_01_dp
vals = np.array([60, 100, 120], dtype=float)
wts = np.array([10, 20, 30], dtype=int)
cap = 50
max_val, sel = knapsack_01_dp(vals, wts, cap)
assert abs(max_val - 220.0) < 1e-10, '[TC23] 背包最优值应为 220 FAILED'
assert np.sum(wts * sel) <= cap, '[TC23] 背包重量超限 FAILED'

# ---- TC24: generate_credit_portfolio_data 数据结构正确 ----
np.random.seed(42)
data = generate_credit_portfolio_data(n_assets=10, seed=42)
assert "expected_returns" in data, '[TC24] 缺少 expected_returns FAILED'
assert "capital_charges" in data, '[TC24] 缺少 capital_charges FAILED'
assert "pd_values" in data, '[TC24] 缺少 pd_values FAILED'
assert len(data["expected_returns"]) == 10, '[TC24] 资产数量错误 FAILED'
assert np.all(data["pd_values"] >= 0) and np.all(data["pd_values"] <= 1), '[TC24] PD 越界 FAILED'

# ---- TC25: factor_covariance_from_loadings 对角线为 1 ----
np.random.seed(42)
B_test = np.random.randn(15, 4)
B_test = B_test / (np.linalg.norm(B_test, axis=1, keepdims=True) + 1e-12) * 0.6
corr_mat = factor_covariance_from_loadings(B_test, np.ones(4))
assert np.allclose(np.diag(corr_mat), 1.0, atol=1e-6), '[TC25] 协方差矩阵对角线不为 1 FAILED'
assert corr_mat.shape == (15, 15), '[TC25] 协方差矩阵维度错误 FAILED'

# ---- TC26: cholesky_with_pivot 重构验证 ----
np.random.seed(42)
C_test = np.random.randn(6, 6)
C_test = C_test @ C_test.T
C_test = C_test / np.max(np.abs(C_test))
np.fill_diagonal(C_test, 1.0)
L_chol = cholesky_with_pivot(C_test)
assert L_chol is not None, '[TC26] Cholesky 分解返回 None FAILED'
recon = L_chol @ L_chol.T
assert np.allclose(recon, C_test, atol=1e-8), '[TC26] Cholesky 重构失败 FAILED'

# ---- TC27: helmert_matrix 正交性 ----
H8 = helmert_matrix(8)
assert H8.shape == (8, 8), '[TC27] Helmert 矩阵维度错误 FAILED'
HtH = H8 @ H8.T
assert np.allclose(HtH, np.eye(8), atol=1e-10), '[TC27] Helmert 矩阵非正交 FAILED'

# ---- TC28: arc_length_parameterize 弧长参数 ----
pts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
t_param, s = arc_length_parameterize(pts)
assert len(t_param) == 4, '[TC28] 参数长度错误 FAILED'
assert np.isclose(t_param[0], 0.0), '[TC28] 起点参数不为 0 FAILED'
assert np.isclose(t_param[-1], 1.0), '[TC28] 终点参数不为 1 FAILED'
assert np.all(np.diff(t_param) >= -1e-12), '[TC28] 参数非单调 FAILED'
assert np.all(np.diff(s) >= -1e-12), '[TC28] 累积距离非单调 FAILED'

# ---- TC29: build_regional_default_correlation 输出结构 ----
xyz_r, areas_r, adj_r, corr_r = build_regional_default_correlation(n_regions=8, base_correlation=0.3)
n_r = xyz_r.shape[0]
assert corr_r.shape == (n_r, n_r), '[TC29] 相关性矩阵维度错误 FAILED'
assert np.allclose(np.diag(corr_r), 1.0, atol=1e-5), '[TC29] 相关性矩阵对角线不为 1 FAILED'
assert adj_r.shape == (n_r, n_r), '[TC29] 邻接矩阵维度错误 FAILED'
assert areas_r.shape == (n_r,), '[TC29] 面积向量维度错误 FAILED'
assert abs(np.sum(areas_r) - 1.0) < 1e-10, '[TC29] 面积权重和不等于 1 FAILED'

# ---- TC30: credit_portfolio_optimization 返回有效选择 ----
np.random.seed(42)
d30 = generate_credit_portfolio_data(n_assets=12, seed=42)
sel30, met30 = credit_portfolio_optimization(
    d30["expected_returns"],
    d30["capital_charges"],
    d30["pd_values"],
    d30["lgd_values"],
    d30["ead_values"],
    total_capital=400.0,
    var_limit=1500.0,
    resolution=400
)
assert np.any(sel30), '[TC30] 未选择任何资产 FAILED'
assert met30["total_capital"] <= 400.0 * 1.01, '[TC30] 资本约束违反 FAILED'
assert met30["total_risk"] <= 1500.0 * 1.01, '[TC30] 风险约束违反 FAILED'
assert met30["total_return"] > 0, '[TC30] 总收益非正 FAILED'
