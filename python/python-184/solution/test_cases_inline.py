# ---- TC01: generate_synthetic_series 返回三元组 ----
t1, s1, a1 = generate_synthetic_series(n=64, seed=42)
assert isinstance(t1, np.ndarray) and isinstance(s1, np.ndarray) and isinstance(a1, np.ndarray), '[TC01] generate_synthetic_series 返回类型 FAILED'
assert len(t1) == 64 and len(s1) == 64 and len(a1) == 64, '[TC01] 序列长度校验 FAILED'

# ---- TC02: generate_synthetic_series 可复现性 ----
import numpy as np
np.random.seed(123)
t2a, s2a, a2a = generate_synthetic_series(n=64, seed=123)
np.random.seed(123)
t2b, s2b, a2b = generate_synthetic_series(n=64, seed=123)
assert np.allclose(t2a, t2b) and np.allclose(s2a, s2b) and np.allclose(a2a, a2b), '[TC02] generate_synthetic_series 可复现性 FAILED'

# ---- TC03: generate_synthetic_series 输出值均为有限值 ----
t3, s3, a3 = generate_synthetic_series(n=64, seed=7)
assert np.all(np.isfinite(t3)) and np.all(np.isfinite(s3)), '[TC03] generate_synthetic_series 有限值校验 FAILED'

# ---- TC04: ARPredictor fit 返回自身且系数有限 ----
ar4 = ARPredictor(order=4)
_, s4, _ = generate_synthetic_series(n=128, seed=99)
result4 = ar4.fit(s4)
assert result4 is ar4, '[TC04] ARPredictor fit 返回值 FAILED'
assert len(ar4.ar_coefs) == 4, '[TC04] AR 系数个数 FAILED'
assert np.all(np.isfinite(ar4.ar_coefs)), '[TC04] AR 系数有限性 FAILED'

# ---- TC05: ARPredictor predict 输出形状正确 ----
ar5 = ARPredictor(order=3)
_, s5, _ = generate_synthetic_series(n=128, seed=42)
ar5.fit(s5)
pred5 = ar5.predict(s5, steps=10)
assert len(pred5) == 10, '[TC05] ARPredictor predict 输出长度 FAILED'
assert np.all(np.isfinite(pred5)), '[TC05] ARPredictor predict 有限性 FAILED'

# ---- TC06: ARPredictor forecast_interval 上下界关系 ----
ar6 = ARPredictor(order=4)
_, s6, _ = generate_synthetic_series(n=128, seed=55)
ar6.fit(s6)
pred6, lower6, upper6 = ar6.forecast_interval(s6, steps=8, confidence=0.95)
assert len(pred6) == 8 and len(lower6) == 8 and len(upper6) == 8, '[TC06] forecast_interval 输出形状 FAILED'
assert np.all(lower6 <= pred6) and np.all(pred6 <= upper6), '[TC06] forecast_interval 上下界关系 FAILED'

# ---- TC07: ARPredictor stability_analysis 返回字典含预期键 ----
ar7 = ARPredictor(order=4)
_, s7, _ = generate_synthetic_series(n=128, seed=77)
ar7.fit(s7)
stab7 = ar7.stability_analysis()
assert isinstance(stab7, dict), '[TC07] stability_analysis 返回类型 FAILED'
assert 'stable' in stab7 and 'modes' in stab7, '[TC07] stability_analysis 缺键 FAILED'

# ---- TC08: ToeplitzSolver solve_yule_walker 有限系数 ----
import numpy as np
np.random.seed(42)
ts8 = ToeplitzSolver()
autocorr8 = np.array([1.0, 0.5, 0.2, 0.05])
ar8, k8 = ts8.solve_yule_walker(autocorr8)
assert len(ar8) == 3 and len(k8) == 3, '[TC08] solve_yule_walker 输出长度 FAILED'
assert np.all(np.isfinite(ar8)) and np.all(np.isfinite(k8)), '[TC08] solve_yule_walker 有限性 FAILED'

# ---- TC09: ToeplitzSolver schur_cohn_stability ----
ts9 = ToeplitzSolver()
k_stable9 = np.array([0.3, -0.5, 0.2])
assert ts9.schur_cohn_stability(k_stable9) == True, '[TC09] schur_cohn_stability 稳定判定 FAILED'
k_unstable9 = np.array([0.3, 1.5, 0.2])
assert ts9.schur_cohn_stability(k_unstable9) == False, '[TC09] schur_cohn_stability 不稳定判定 FAILED'

# ---- TC10: ToeplitzSolver solve_toeplitz 输出有限且形状正确 ----
ts10 = ToeplitzSolver()
t10 = np.array([2.0, 1.0, 0.5])
T10 = ts10.autocorr_to_toeplitz(t10)
b10 = T10 @ np.array([1.0, 2.0, 3.0])
x10 = ts10.solve_toeplitz(t10, b10)
assert len(x10) == 3, '[TC10] solve_toeplitz 输出长度 FAILED'
assert np.all(np.isfinite(x10)), '[TC10] solve_toeplitz 有限性 FAILED'

# ---- TC11: GraphAnomalyDetector detect 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
gad11 = GraphAnomalyDetector(k_neighbors=3)
X11 = np.random.randn(50, 4)
scores11 = gad11.detect(X11, method='pagerank')
assert len(scores11) == 50, '[TC11] detect 输出长度 FAILED'
assert np.all(scores11 >= 0.0) and np.all(scores11 <= 1.0), '[TC11] detect 范围约束 FAILED'

# ---- TC12: GraphAnomalyDetector connected_components 正整数 ----
import numpy as np
np.random.seed(42)
gad12 = GraphAnomalyDetector(k_neighbors=3)
X12 = np.random.randn(30, 3)
gad12.build_graph(X12)
n_comp12, labels12 = gad12.connected_components()
assert n_comp12 >= 1, '[TC12] connected_components 计数 FAILED'
assert len(labels12) == 30, '[TC12] connected_components 标签长度 FAILED'

# ---- TC13: GraphAnomalyDetector spectral_anomaly_score 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
gad13 = GraphAnomalyDetector(k_neighbors=5)
X13 = np.random.randn(40, 3)
gad13.build_graph(X13)
spec13 = gad13.spectral_anomaly_score()
assert len(spec13) == 40, '[TC13] spectral_anomaly_score 长度 FAILED'
assert np.all(spec13 >= 0.0) and np.all(spec13 <= 1.0), '[TC13] spectral_anomaly_score 范围 FAILED'

# ---- TC14: RBFReconstructor fit+predict 有限输出 ----
import numpy as np
np.random.seed(42)
rbf14 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
centers14 = np.linspace(0, 10, 20).reshape(-1, 1)
values14 = np.sin(centers14[:, 0])
rbf14.fit(centers14, values14)
points14 = np.linspace(0, 10, 50).reshape(-1, 1)
pred14 = rbf14.predict(points14)
assert len(pred14) == 50, '[TC14] RBF predict 输出长度 FAILED'
assert np.all(np.isfinite(pred14)), '[TC14] RBF predict 有限性 FAILED'

# ---- TC15: RBFReconstructor reconstruct_series 输出形状 ----
import numpy as np
np.random.seed(42)
rbf15 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
t_obs15 = np.array([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0])
v_obs15 = np.sin(t_obs15)
t_all15 = np.linspace(0, 10, 51)
recon15 = rbf15.reconstruct_series(t_obs15, v_obs15, t_all15)
assert len(recon15) == 51, '[TC15] reconstruct_series 输出长度 FAILED'
assert np.all(np.isfinite(recon15)), '[TC15] reconstruct_series 有限性 FAILED'

# ---- TC16: RBFReconstructor anomaly_score 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
rbf16 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
t16 = np.linspace(0, 10, 60)
s16 = np.sin(t16) + 0.1 * np.random.randn(60)
scores16 = rbf16.anomaly_score(t16, s16)
assert len(scores16) == 60, '[TC16] anomaly_score 长度 FAILED'
assert np.all(scores16 >= 0.0) and np.all(scores16 <= 1.0), '[TC16] anomaly_score 范围 FAILED'

# ---- TC17: PDE1DHeatExplicit solve 有限输出 ----
heat17 = PDE1DHeatExplicit(kappa=0.5)
x17 = np.linspace(0, 1, 31)
u0_17 = np.sin(np.pi * x17)
u17 = heat17.solve(x17, u0_17, dt=0.001, n_steps=50)
assert len(u17) == 31, '[TC17] PDE1DHeat solve 输出长度 FAILED'
assert np.all(np.isfinite(u17)), '[TC17] PDE1DHeat solve 有限性 FAILED'

# ---- TC18: ReactionDiffusion1D solve 输出形状 ----
rd18 = ReactionDiffusion1D(D=0.3, rho=0.1, K=2.0, mu=0.05, c_s=0.5)
x18 = np.linspace(0, 1, 25)
u0_18 = 0.5 + 0.3 * np.sin(2 * np.pi * x18)
u18 = rd18.solve(x18, u0_18, dt=0.005, n_steps=30, scheme='heun')
assert len(u18) == 25, '[TC18] ReactionDiffusion1D solve 输出长度 FAILED'
assert np.all(np.isfinite(u18)), '[TC18] ReactionDiffusion1D solve 有限性 FAILED'

# ---- TC19: BiochemicalODE 守恒量非负 ----
bio19 = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
y19 = np.array([1.0, 2.0, 0.0, 0.0])
h19 = bio19.conserved_quantities(y19)
assert len(h19) == 2, '[TC19] conserved_quantities 长度 FAILED'
assert np.all(h19 >= 0.0), '[TC19] 守恒量非负 FAILED'

# ---- TC20: BiochemicalODE integrate_rk4 初始/最终守恒近似 ----
bio20 = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
y0_20 = np.array([1.0, 2.0, 0.0, 0.0])
t20, y20 = bio20.integrate_rk4(y0_20, (0.0, 5.0), 100)
h_init20 = bio20.conserved_quantities(y0_20)
h_final20 = bio20.conserved_quantities(y20[-1])
assert y20.shape == (101, 4), '[TC20] integrate_rk4 输出形状 FAILED'
assert np.all(np.abs(h_final20 - h_init20) < 1e-2), '[TC20] 守恒量偏差过大 FAILED'

# ---- TC21: ExtendedBrusselator integrate 输出形状 ----
bruss21 = ExtendedBrusselator(a=1.0, b=2.8)
y0_21 = np.array([1.0, 1.0])
t21, y21 = bruss21.integrate(y0_21, (0.0, 10.0), 100)
assert y21.shape == (101, 2), '[TC21] ExtendedBrusselator integrate 输出形状 FAILED'
assert np.all(np.isfinite(y21)), '[TC21] ExtendedBrusselator integrate 有限性 FAILED'

# ---- TC22: ExtendedBrusselator lyapunov_exponent 有限标量 ----
bruss22 = ExtendedBrusselator(a=1.0, b=2.8)
y0_22 = np.array([1.0, 1.0])
lyap22 = bruss22.lyapunov_exponent_numerical(y0_22, (0.0, 10.0), 200)
assert np.isfinite(lyap22), '[TC22] Lyapunov 指数有限性 FAILED'

# ---- TC23: GenHermiteQuadrature compute_rule 权重和 ≈ √π ----
gh23 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=16)
nodes23, weights23 = gh23.compute_rule()
assert len(nodes23) == 16 and len(weights23) == 16, '[TC23] compute_rule 输出长度 FAILED'
assert np.abs(np.sum(weights23) - np.sqrt(np.pi)) < 1e-10, '[TC23] 权重和校验 FAILED'

# ---- TC24: GenHermiteQuadrature integrate 常数函数 ----
gh24 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=20)
gh24.compute_rule()
I_const24 = gh24.integrate(lambda x: np.ones_like(x))
assert np.abs(I_const24 - np.sqrt(np.pi)) < 1e-10, '[TC24] 常数函数积分 FAILED'

# ---- TC25: GenHermiteQuadrature integrate x^2 ----
gh25 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=30)
gh25.compute_rule()
I_x2_25 = gh25.integrate(lambda x: x**2)
expected25 = 0.5 * np.sqrt(np.pi)
assert np.abs(I_x2_25 - expected25) < 1e-6, '[TC25] x^2 积分 FAILED'

# ---- TC26: EmbeddingGeometry delay_embed 形状正确 ----
geom26 = EmbeddingGeometry(embedding_dim=3, delay=2)
s26 = np.sin(np.linspace(0, 4*np.pi, 100))
X26 = geom26.delay_embed(s26)
expected26 = 100 - (3 - 1) * 2
assert X26.shape == (expected26, 3), '[TC26] delay_embed 形状 FAILED'

# ---- TC27: EmbeddingGeometry embedding_dimension_estimate 正整数 ----
geom27 = EmbeddingGeometry(embedding_dim=3, delay=2)
s27 = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.05 * np.random.randn(200)
import numpy as np
np.random.seed(42)
dim27 = geom27.embedding_dimension_estimate(s27, max_dim=6, threshold=0.2)
assert isinstance(dim27, (int, np.integer)), '[TC27] embedding_dimension_estimate 类型 FAILED'
assert dim27 > 0, '[TC27] embedding_dimension_estimate 非正 FAILED'

# ---- TC28: NumericalRobustness next_float > x (for x > 0) ----
nr28 = NumericalRobustness()
x28 = 1.0
nx28 = nr28.next_float(x28)
assert nx28 > x28, '[TC28] next_float 单调性 FAILED'

# ---- TC29: NumericalRobustness prev_float < x (for x > 0) ----
nr29 = NumericalRobustness()
x29 = 1.0
px29 = nr29.prev_float(x29)
assert px29 < x29, '[TC29] prev_float 单调性 FAILED'

# ---- TC30: NumericalRobustness threshold_by_quantile_root 在 [0,1] ----
nr30 = NumericalRobustness()
scores30 = np.random.rand(200)
import numpy as np
np.random.seed(42)
th30 = nr30.threshold_by_quantile_root(scores30, target_fpr=0.1)
assert 0.0 <= th30 <= 1.0, '[TC30] threshold_by_quantile_root 范围 FAILED'

# ---- TC31: ManufacturedVerification verify_heat_fem 返回预期键 ----
mms31 = ManufacturedVerification(kappa=0.5)
x31 = np.linspace(0.0, 1.0, 21)
res31 = mms31.verify_heat_fem(x31, dt=0.001, n_steps=50)
assert 'l2_error' in res31 and 'relative_l2' in res31, '[TC31] verify_heat_fem 缺键 FAILED'
assert res31['l2_error'] > 0, '[TC31] L2 误差非正 FAILED'

# ---- TC32: ManufacturedVerification convergence_study 估计收敛阶 ----
mms32 = ManufacturedVerification(kappa=0.5)
conv32 = mms32.convergence_study(n_grids=[11, 21, 41], dt=0.0005, n_steps=50)
assert 'estimated_spatial_order' in conv32, '[TC32] convergence_study 缺键 FAILED'
assert np.isfinite(conv32['estimated_spatial_order']), '[TC32] 估计收敛阶有限性 FAILED'

# ---- TC33: DiscretePatternKernel binarize 输出 0/1 ----
dpk33 = DiscretePatternKernel(window_size=5)
s33 = np.sin(np.linspace(0, 4*np.pi, 80)) + 0.2 * np.random.randn(80)
import numpy as np
np.random.seed(42)
binary33 = dpk33.binarize(s33)
assert np.all((binary33 == 0) | (binary33 == 1)), '[TC33] binarize 输出非 0/1 FAILED'

# ---- TC34: DiscretePatternKernel entropy_rate 有限值 ----
dpk34 = DiscretePatternKernel(window_size=5)
s34 = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.2 * np.random.randn(100)
import numpy as np
np.random.seed(42)
binary34 = dpk34.binarize(s34)
ent34 = dpk34.entropy_rate(binary34)
assert np.isfinite(ent34), '[TC34] entropy_rate 有限性 FAILED'
assert 0.0 <= ent34 <= 2.0, '[TC34] entropy_rate 范围 FAILED'

# ---- TC35: TriangularFeatureIntegrator verify_monomial 高精度 ----
tfi35 = TriangularFeatureIntegrator(order=5)
verif35 = tfi35.verify_monomial(2, 1)
assert verif35['error'] < 1e-10, '[TC35] verify_monomial 精度 FAILED'

# ---- TC36: TriangularFeatureIntegrator extract_triangular_features 输出 [0,1] ----
tfi36 = TriangularFeatureIntegrator(order=5)
triplets36 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
feat36 = tfi36.extract_triangular_features(triplets36)
assert len(feat36) == 3, '[TC36] extract_triangular_features 长度 FAILED'
assert np.all(feat36 >= 0.0) and np.all(feat36 <= 1.0), '[TC36] extract_triangular_features 范围 FAILED'

# ---- TC37: main 集成测试：结果字典含预期键 ----
assert isinstance(results, dict), '[TC37] main 返回类型 FAILED'
expected_keys = ['ar_coefs', 'stability', 'forecast', 'rbf_recon_error', 'pde_smooth_l2',
                 'bio_conservation_error', 'lyapunov_exponent', 'predictive_variance',
                 'embedding_dimension', 'condition_number', 'mms_l2_error', 'entropy_rate',
                 'precision', 'recall', 'f1_score', 'auc_roc', 'elapsed_time']
for k in expected_keys:
    assert k in results, f'[TC37] main 结果缺键 {k} FAILED'

# ---- TC38: main 集成测试：性能指标在 [0,1] ----
assert 0.0 <= results['precision'] <= 1.0, '[TC38] precision 范围 FAILED'
assert 0.0 <= results['recall'] <= 1.0, '[TC38] recall 范围 FAILED'
assert 0.0 <= results['f1_score'] <= 1.0, '[TC38] f1_score 范围 FAILED'
assert 0.0 <= results['auc_roc'] <= 1.0, '[TC38] auc_roc 范围 FAILED'

# ---- TC39: main 集成测试：forecast 是有限值 ----
assert np.all(np.isfinite(results['forecast'])), '[TC39] forecast 有限性 FAILED'

# ---- TC40: main 集成测试：elapsed_time > 0 ----
assert results['elapsed_time'] > 0, '[TC40] elapsed_time 非正 FAILED'
