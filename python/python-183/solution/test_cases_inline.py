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
