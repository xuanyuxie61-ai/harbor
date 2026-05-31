# ---- TC01: QuasiperiodicDynamics exact_solution 输出形状正确 ----
qpd_test = QuasiperiodicPreferenceDynamics()
t = np.linspace(0, 5, 20)
y = qpd_test.exact_solution(t)
assert y.shape == (20, 4), '[TC01] exact_solution shape FAILED'

# ---- TC02: QuasiperiodicDynamics exact_solution 在 t=0 解析验证 ----
y0 = qpd_test.exact_solution(np.array([0.0]))
assert abs(y0[0, 0] - 2.0) < 1e-10, '[TC02] exact_solution y0[0] FAILED'
assert abs(y0[0, 2] - (-(1.0 + np.pi**2))) < 1e-10, '[TC02] exact_solution y0[2] FAILED'

# ---- TC03: QuasiperiodicDynamics integrate_ode 数值解收敛 ----
t_eval = np.linspace(0, 10, 50)
y_ode = qpd_test.integrate_ode(t_eval)
y_exact = qpd_test.exact_solution(t_eval)
err = np.linalg.norm(y_exact - y_ode, axis=1).mean()
assert err < 1.0, '[TC03] integrate_ode convergence FAILED'

# ---- TC04: QuasiperiodicDynamics temporal_modulation 输出非负 ----
mod = qpd_test.temporal_modulation(np.array([0.0, 1.0, 2.0]), base_preference=1.0, amplitude=0.05)
assert np.all(mod >= 0.1), '[TC04] temporal_modulation non-negative FAILED'

# ---- TC05: LaguerrePolynomialChaos evaluate 已知值 L_0, L_1, L_2 ----
lpc_test = LaguerrePolynomialChaos(max_degree=5)
vals = lpc_test.evaluate(2, np.array([0.0]))
assert abs(vals[0, 0] - 1.0) < 1e-10, '[TC05] L_0(0) FAILED'
assert abs(vals[0, 1] - 1.0) < 1e-10, '[TC05] L_1(0) FAILED'
# L_2(x) = (x^2 - 4x + 2)/2, L_2(0) = 1.0
assert abs(vals[0, 2] - 1.0) < 1e-10, '[TC05] L_2(0) FAILED'

# ---- TC06: LaguerrePolynomialChaos quadrature_rule 权重和为1 ----
nodes, weights = lpc_test.quadrature_rule(8)
assert abs(np.sum(weights) - 1.0) < 1e-6, '[TC06] quadrature_rule weights sum FAILED'

# ---- TC07: LaguerrePolynomialChaos exponential_product_table 对称性 ----
exp_table = lpc_test.exponential_product_table(0.3)
assert exp_table.shape == (6, 6), '[TC07] exp_table shape FAILED'
assert np.allclose(exp_table, exp_table.T), '[TC07] exp_table symmetry FAILED'

# ---- TC08: LaguerrePolynomialChaos propagate_uncertainty 非负方差 ----
ratings = np.array([1.0, 2.5, 3.0, 4.5, 5.0])
var_val = lpc_test.propagate_uncertainty(ratings, 0.3)
assert var_val >= 0.0, '[TC08] propagate_uncertainty non-negative FAILED'

# ---- TC09: CordicEngine cossin 在 π/2 处精度 ----
cordic_test = CordicEngine(n_iter=24)
cos_v, sin_v = cordic_test.cossin(np.pi / 2.0)
assert abs(sin_v - 1.0) < 1e-5, '[TC09] cossin sin(π/2) FAILED'
assert abs(cos_v - 0.0) < 1e-5, '[TC09] cossin cos(π/2) FAILED'

# ---- TC10: CordicEngine exp_cordic vs numpy ----
exp_v = cordic_test.exp_cordic(2.0)
assert abs(exp_v - np.exp(2.0)) < 0.5, '[TC10] exp_cordic accuracy FAILED'

# ---- TC11: CordicEngine log_cordic vs numpy ----
log_v = cordic_test.log_cordic(np.e)
assert abs(log_v - 1.0) < 1e-3, '[TC11] log_cordic ln(e) FAILED'

# ---- TC12: CordicEngine sqrt_cordic 精度 ----
sqrt_v = cordic_test.sqrt_cordic(4.0)
assert abs(sqrt_v - 2.0) < 1e-3, '[TC12] sqrt_cordic FAILED'

# ---- TC13: CordicEngine sqrt_cordic 边界: x=0, x<0 ----
assert cordic_test.sqrt_cordic(0.0) == 0.0, '[TC13] sqrt_cordic zero FAILED'
assert np.isnan(cordic_test.sqrt_cordic(-1.0)), '[TC13] sqrt_cordic negative FAILED'

# ---- TC14: CordicEngine exp_cordic 边界: x过大, x过小 ----
assert cordic_test.exp_cordic(800.0) == float('inf'), '[TC14] exp_cordic overflow FAILED'
assert cordic_test.exp_cordic(-800.0) == 0.0, '[TC14] exp_cordic underflow FAILED'

# ---- TC15: TruncatedNormalRatingModel mean 在 [a,b] 内 ----
trunc_test = TruncatedNormalRatingModel(mu=3.0, sigma=1.2, a=1.0, b=5.0)
m = trunc_test.mean()
assert 1.0 <= m <= 5.0, '[TC15] truncated mean in bounds FAILED'

# ---- TC16: TruncatedNormalRatingModel variance 非负 ----
v = trunc_test.variance()
assert v >= 0.0, '[TC16] truncated variance non-negative FAILED'

# ---- TC17: TruncatedNormalRatingModel sample 可复现性 ----
import numpy as np
np.random.seed(42)
s1 = trunc_test.sample(200)
np.random.seed(42)
s2 = trunc_test.sample(200)
assert np.allclose(s1, s2), '[TC17] sample reproducibility FAILED'

# ---- TC18: TruncatedNormalRatingModel sample 在 [a,b] 内 ----
np.random.seed(42)
samples = trunc_test.sample(1000)
assert np.all(samples >= 1.0) and np.all(samples <= 5.0), '[TC18] sample in bounds FAILED'

# ---- TC19: TruncatedNormalRatingModel pdf 非负 ----
x_vals = np.linspace(1.0, 5.0, 50)
pdf_vals = trunc_test.pdf(x_vals)
assert np.all(pdf_vals >= 0.0), '[TC19] pdf non-negative FAILED'

# ---- TC20: TruncatedNormalRatingModel expected_rating 非破坏性 ----
mu_orig = trunc_test.mu
er = trunc_test.expected_rating(3.5, clip=True)
assert trunc_test.mu == mu_orig, '[TC20] expected_rating non-destructive FAILED'
assert 1.0 <= er <= 5.0, '[TC20] expected_rating bounds FAILED'

# ---- TC21: HeatDiffusionRecommender diffuse 输出形状 ----
R_test = np.array([[3.0, np.nan, 4.0], [np.nan, 2.0, np.nan], [5.0, np.nan, np.nan]])
heat_test = HeatDiffusionRecommender(alpha=0.05, n_steps=5)
R_diff = heat_test.diffuse(R_test)
assert R_diff.shape == (3, 3), '[TC21] diffuse shape FAILED'
assert np.all(R_diff >= 1.0) and np.all(R_diff <= 5.0), '[TC21] diffuse range FAILED'

# ---- TC22: HeatDiffusionRecommender diffuse 保持已知值 ----
R_diff2 = heat_test.diffuse(R_test)
assert abs(R_diff2[0, 0] - 3.0) < 0.1, '[TC22] diffuse preserve known FAILED'

# ---- TC23: FemEmbeddingInterpolator _tetrahedron_volume 已知四面体 ----
fem_test = FemEmbeddingInterpolator(latent_dim=6)
tetra = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = fem_test._tetrahedron_volume(tetra)
assert abs(vol - 1.0/6.0) < 1e-10, '[TC23] tetrahedron_volume FAILED'

# ---- TC24: FemEmbeddingInterpolator _barycentric_coords 和为1 ----
lam = fem_test._barycentric_coords(np.array([0.25, 0.25, 0.25]), tetra)
assert abs(np.sum(lam) - 1.0) < 1e-10, '[TC24] barycentric sum to 1 FAILED'

# ---- TC25: RobustGaussianSolver solve_gauss vs numpy ----
np.random.seed(42)
solver_test = RobustGaussianSolver()
M = np.random.randn(10, 10)
A_spd = M.T @ M + np.eye(10)
b = np.ones(10)
x_gauss = solver_test.solve_gauss(A_spd, b)
x_np = np.linalg.solve(A_spd, b)
assert np.linalg.norm(x_gauss - x_np) < 1e-8, '[TC25] solve_gauss accuracy FAILED'

# ---- TC26: RobustGaussianSolver solve_plu vs numpy ----
x_plu = solver_test.solve_plu(A_spd, b)
assert np.linalg.norm(x_plu - x_np) < 1e-8, '[TC26] solve_plu accuracy FAILED'

# ---- TC27: RobustGaussianSolver plu_decomposition 重构验证 ----
P, L, U = solver_test.plu_decomposition(A_spd)
reconstructed = P.T @ L @ U
assert np.allclose(reconstructed, A_spd, atol=1e-8), '[TC27] PLU reconstruction FAILED'

# ---- TC28: RobustGaussianSolver determinant 精度 ----
det_gauss = solver_test.determinant(A_spd)
det_np = np.linalg.det(A_spd)
assert abs(det_gauss - det_np) < max(1e-6, abs(det_np) * 1e-6), '[TC28] determinant FAILED'

# ---- TC29: RobustGaussianSolver inverse 精度 ----
inv_gauss = solver_test.inverse(A_spd)
assert np.allclose(inv_gauss @ A_spd, np.eye(10), atol=1e-6), '[TC29] inverse FAILED'

# ---- TC30: GeometricSampler sample_unit_circle 单位范数 ----
np.random.seed(42)
geom_test = GeometricSampler(n_samples=5000)
pts = geom_test.sample_unit_circle(100)
norms = np.linalg.norm(pts, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10), '[TC30] unit circle norm FAILED'

# ---- TC31: GeometricSampler circle_monomial_integral 解析值 ----
ival = geom_test.circle_monomial_integral(np.array([2, 2]))
# Analytical: I = 2 * Γ(1.5)*Γ(1.5) / Γ(3) = 2 * (√π/2)² / 2 = π/4
assert abs(ival - np.pi / 4.0) < 1e-10, '[TC31] monomial integral FAILED'

# ---- TC32: GeometricSampler circle_monomial_integral 奇次幂为零 ----
ival_odd = geom_test.circle_monomial_integral(np.array([1, 2]))
assert abs(ival_odd) < 1e-10, '[TC32] monomial integral odd FAILED'

# ---- TC33: GeometricSampler plane_tetrahedron_intersect 已知相交 ----
tetra_geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
plane_p = np.array([0.2, 0.2, 0.2])
plane_n = np.array([1.0, 1.0, 1.0])
n_int, pts_int = geom_test.plane_tetrahedron_intersect(plane_p, plane_n, tetra_geom)
assert n_int > 0, '[TC33] plane_tetrahedron_intersect FAILED'

# ---- TC34: GeometricSampler parallelogram_area_3d 已知面积 ----
para_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
area_para = geom_test.parallelogram_area_3d(para_pts)
assert abs(area_para - 1.0) < 1e-10, '[TC34] parallelogram_area_3d FAILED'

# ---- TC35: GeometricSampler quadrilateral_area_3d 非负 ----
quad_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
area_quad = geom_test.quadrilateral_area_3d(quad_pts)
assert area_quad >= 0.0, '[TC35] quadrilateral_area_3d non-negative FAILED'

# ---- TC36: HilbertLSH xyz_to_h 确定性 ----
hilbert_test = HilbertLSH(order=4)
h1 = hilbert_test.xyz_to_h(0, 0, 0)
h2 = hilbert_test.xyz_to_h(0, 0, 0)
assert h1 == h2, '[TC36] xyz_to_h determinism FAILED'

# ---- TC37: HilbertLSH hash_vectors 输出形状 ----
np.random.seed(42)
vecs = np.random.randn(10, 3)
hashes = hilbert_test.hash_vectors(vecs)
assert hashes.shape == (10,), '[TC37] hash_vectors shape FAILED'
assert np.all(hashes >= 0), '[TC37] hash_vectors non-negative FAILED'

# ---- TC38: StringSimilarityEngine levenshtein_distance 已知值 ----
str_sim = StringSimilarityEngine()
d1 = str_sim.levenshtein_distance("kitten", "sitting")
assert d1 == 3, '[TC38] levenshtein kitten/sitting FAILED'

# ---- TC39: StringSimilarityEngine levenshtein_distance 相同字符串 ----
d2 = str_sim.levenshtein_distance("abc", "abc")
assert d2 == 0, '[TC39] levenshtein identical FAILED'

# ---- TC40: StringSimilarityEngine levenshtein_distance 空字符串 ----
d3 = str_sim.levenshtein_distance("", "abc")
assert d3 == 3, '[TC40] levenshtein empty FAILED'

# ---- TC41: StringSimilarityEngine similarity 归一化 ----
sim_val = str_sim.similarity("abc", "abc")
assert abs(sim_val - 1.0) < 1e-10, '[TC41] similarity identical FAILED'
sim_val2 = str_sim.similarity("abc", "xyz")
assert 0.0 <= sim_val2 <= 1.0, '[TC41] similarity range FAILED'

# ---- TC42: StringSimilarityEngine compute_similarity_matrix 对称性 ----
strs = ["量子热力学", "有限元混沌", "扩散熵流形"]
S = str_sim.compute_similarity_matrix(strs)
assert S.shape == (3, 3), '[TC42] similarity matrix shape FAILED'
assert np.allclose(S, S.T), '[TC42] similarity matrix symmetry FAILED'
assert np.allclose(np.diag(S), 1.0), '[TC42] similarity matrix diagonal FAILED'

# ---- TC43: TriangulationBoundaryDetector detect_boundary_edges 已知三角剖分 ----
tri_test = TriangulationBoundaryDetector()
triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
be = tri_test.detect_boundary_edges(triangles)
assert be.shape[1] == 2, '[TC43] boundary_edges shape FAILED'
assert be.shape[0] > 0, '[TC43] boundary_edges empty FAILED'

# ---- TC44: TriangulationBoundaryDetector detect_boundary 返回节点集 ----
np.random.seed(42)
pts_2d = np.random.randn(20, 2)
bnodes = tri_test.detect_boundary(pts_2d)
assert len(bnodes) > 0, '[TC44] detect_boundary non-empty FAILED'
assert all(isinstance(n, (int, np.integer)) for n in bnodes), '[TC44] boundary node types FAILED'

# ---- TC45: TriangulationBoundaryDetector detect_boundary 退化为空点 ----
pts_few = np.array([[0.0, 0.0], [1.0, 1.0]])
bn_few = tri_test.detect_boundary(pts_few)
assert bn_few == {0, 1}, '[TC45] detect_boundary degenerate FAILED'

# ---- TC46: AggregateStatistics group_users_by_embedding 标签范围 ----
np.random.seed(42)
agg_test = AggregateStatistics()
emb = np.random.randn(30, 4)
labels = agg_test.group_users_by_embedding(emb, n_clusters=4)
assert len(np.unique(labels)) <= 4, '[TC46] group clustering n_clusters FAILED'
assert labels.shape == (30,), '[TC46] group clustering shape FAILED'

# ---- TC47: AggregateStatistics compute_group_statistics 统计量结构 ----
R_mat = np.random.RandomState(42).uniform(1, 5, (30, 10))
stats = agg_test.compute_group_statistics(R_mat, labels)
assert len(stats) > 0, '[TC47] group statistics non-empty FAILED'
for gid, s in stats.items():
    assert 'mean' in s and 'std' in s and 'min' in s and 'max' in s and 'count' in s, '[TC47] group stat keys FAILED'
    assert s['min'] <= s['mean'] <= s['max'], '[TC47] group stat monotonicity FAILED'

# ---- TC48: AggregateStatistics running_aggregate 正确性 ----
vals = [('a', 1.0), ('b', 2.0), ('a', 3.0), ('b', 4.0)]
r_stats = agg_test.running_aggregate(vals)
assert r_stats['a']['count'] == 2, '[TC48] running_aggregate count FAILED'
assert abs(r_stats['a']['mean'] - 2.0) < 1e-10, '[TC48] running_aggregate mean FAILED'

# ---- TC49: AggregateStatistics average_embeddings 结果形状 ----
np.random.seed(42)
embs = np.random.randn(5, 3)
avg_emb = agg_test.average_embeddings(embs)
assert avg_emb.shape == (3,), '[TC49] average_embeddings shape FAILED'

# ---- TC50: 集成测试: generate_synthetic_data 输出结构 ----
np.random.seed(42)
R_obs, R_full, P, Q, item_meta = generate_synthetic_data(n_users=30, n_items=20)
assert R_obs.shape == (30, 20), '[TC50] generate_synthetic_data R_obs shape FAILED'
assert R_full.shape == (30, 20), '[TC50] generate_synthetic_data R_full shape FAILED'
assert P.shape == (30, 8), '[TC50] generate_synthetic_data P shape FAILED'
assert Q.shape == (20, 8), '[TC50] generate_synthetic_data Q shape FAILED'
assert len(item_meta) == 20, '[TC50] generate_synthetic_data item_meta FAILED'
assert np.all(R_full >= 1.0) and np.all(R_full <= 5.0), '[TC50] R_full range FAILED'

# ---- TC51: 集成测试: 全流程不崩溃 ----
np.random.seed(42)
R_obs2, R_full2, P2, Q2, item_meta2 = generate_synthetic_data(n_users=20, n_items=15)
trunc_model2 = TruncatedNormalRatingModel(mu=3.0, sigma=1.2, a=1.0, b=5.0)
trunc_model2.mean()
heat2 = HeatDiffusionRecommender(alpha=0.05, n_steps=3)
R_diff2 = heat2.diffuse(R_obs2)
fem2 = FemEmbeddingInterpolator(latent_dim=6)
R_fem2 = fem2.interpolate(R_diff2)
qpd2 = QuasiperiodicPreferenceDynamics()
t_eval2 = np.linspace(0, 10, 20)
y_exact2 = qpd2.exact_solution(t_eval2)
y_ode2 = qpd2.integrate_ode(t_eval2)
cordic2 = CordicEngine(n_iter=24)
cos2, sin2 = cordic2.cossin(np.pi / 6.0)
lpc2 = LaguerrePolynomialChaos(max_degree=5)
observed2 = ~np.isnan(R_obs2)
unc2 = lpc2.propagate_uncertainty(R_obs2[observed2], 0.3)
hilbert2 = HilbertLSH(order=4)
h_u2 = hilbert2.hash_vectors(P2[:, :3])
h_i2 = hilbert2.hash_vectors(Q2[:, :3])
nn2 = hilbert2.approximate_nn(h_u2, h_i2, top_k=3)
str_sim2 = StringSimilarityEngine()
S2 = str_sim2.compute_similarity_matrix(item_meta2[:5])
solver2 = RobustGaussianSolver()
M2 = np.eye(10) * 2.0
x2 = solver2.solve_plu(M2, np.ones(10))
geom2 = GeometricSampler(n_samples=1000)
mc2 = geom2.monte_carlo_circle_integral(lambda x, y: x**2 + y**2, 500)
tri2 = TriangulationBoundaryDetector()
bn2 = tri2.detect_boundary(P2[:, :2])
agg2 = AggregateStatistics()
labels2 = agg2.group_users_by_embedding(P2, n_clusters=3)
gstats2 = agg2.compute_group_statistics(R_diff2, labels2)
assert True, '[TC51] full pipeline no crash PASSED'

# ---- TC52: CordicEngine log_cordic 边界 x<=0 ----
log_neg = cordic_test.log_cordic(-1.0)
assert log_neg == float('-inf'), '[TC52] log_cordic negative FAILED'
log_zero = cordic_test.log_cordic(0.0)
assert log_zero == float('-inf'), '[TC52] log_cordic zero FAILED'

# ---- TC53: LaguerrePolynomialChaos quadrature_rule 节点递增 ----
nodes_8, _ = lpc_test.quadrature_rule(8)
assert np.all(np.diff(nodes_8) > 0), '[TC53] quadrature nodes monotonic FAILED'

# ---- TC54: GeometricSampler positive_circle_distance_stats 非负 ----
np.random.seed(42)
mu_d, var_d = geom_test.positive_circle_distance_stats(200)
assert mu_d >= 0.0, '[TC54] distance mean non-negative FAILED'
assert var_d >= 0.0, '[TC54] distance variance non-negative FAILED'

# ---- TC55: RobustGaussianSolver solve_gauss 奇异矩阵不崩溃 ----
A_sing = np.array([[1.0, 2.0], [1.0, 2.0]])
b_sing = np.array([3.0, 3.0])
x_sing = solver_test.solve_gauss(A_sing, b_sing)
assert x_sing is not None, '[TC55] singular solve_gauss no crash FAILED'
