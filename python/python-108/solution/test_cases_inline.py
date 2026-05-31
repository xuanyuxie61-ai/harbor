# ---- TC01: CVV 数据结构基本操作正确性 ----
cvv = CVV([3, 5, 4], dtype=float)
for i, n in enumerate([3, 5, 4]):
    cvv.set_row(i, np.arange(n, dtype=float))
assert cvv.size() == (3, 12), '[TC01] CVV 尺寸不正确 FAILED'
assert np.allclose(cvv.get_row(1), np.array([0., 1., 2., 3., 4.])), '[TC01] CVV 行数据不正确 FAILED'

# ---- TC02: CVV 索引访问正确性 ----
cvv2 = CVV([2, 3], dtype=int)
cvv2.set_row(0, np.array([10, 20]))
cvv2.set_row(1, np.array([30, 40, 50]))
assert cvv2.iget(0, 1) == 20, '[TC02] CVV iget 不正确 FAILED'
assert cvv2.iget(1, 2) == 50, '[TC02] CVV iget 不正确 FAILED'

# ---- TC03: MicrocavityGeometry 弧长解析验证 ----
geom = MicrocavityGeometry(R_major=10.0e-6, r_minor=1.5e-6)
arc_len = geom.arc_length_trapezoidal(0.0, 2 * np.pi, n=2000)
theory = 2.0 * np.pi * geom.r_minor
assert abs(arc_len - theory) / theory < 1e-4, '[TC03] 弧长相对误差过大 FAILED'

# ---- TC04: 网格生成尺寸正确性 ----
nodes, elements, markers = geom.generate_cross_section_mesh(nr=5, n_theta=8)
assert nodes.shape == (40, 2), '[TC04] 节点数不正确 FAILED'
assert elements.shape[1] == 3, '[TC04] 单元不是三角形 FAILED'

# ---- TC05: SphereQuadrature 球面面积 = 4π ----
sq = SphereQuadrature()
A = sq.sphere01_area()
assert abs(A - 4.0 * np.pi) < 1e-12, '[TC05] 球面积与 4π 不匹配 FAILED'

# ---- TC06: SphereQuadrature 奇次单项式积分为 0（对称性） ----
assert abs(sq.monomial_integral_exact((1, 0, 0))) < 1e-14, '[TC06] 奇次单项式积分应为 0 FAILED'

# ---- TC07: SphereQuadrature (0,0,0) 积分 = 4π ----
I_000 = sq.monomial_integral_exact((0, 0, 0))
assert abs(I_000 - 4.0 * np.pi) < 1e-12, '[TC07] (0,0,0) 积分应为 4π FAILED'

# ---- TC08: SphereQuadrature (2,0,0) 精确值验证 ----
I_200 = sq.monomial_integral_exact((2, 0, 0))
assert abs(I_200 - 4.0 * np.pi / 3.0) < 1e-12, '[TC08] (2,0,0) 积分应为 4π/3 FAILED'

# ---- TC09: SphereQuadrature 均匀采样点落在单位球面上 ----
np.random.seed(42)
pts = sq.uniform_sample(100, seed=99)
norms = np.sqrt(np.sum(pts**2, axis=1))
assert np.all(np.abs(norms - 1.0) < 1e-14), '[TC09] 采样点不在单位球面上 FAILED'

# ---- TC10: Vandermonde2D 常数积分 = 矩形面积 ----
vq = Vandermonde2DQuadrature()
t = 2
n_needed = (t + 1) * (t + 2) // 2
xs = np.linspace(0.1, 0.9, n_needed)
ys = np.linspace(0.1, 0.9, n_needed)
w = vq.compute_weights(xs, ys, t, rect_a=0.0, rect_b=1.0, rect_c=0.0, rect_d=1.0)
val = vq.integrate(np.ones(n_needed), w)
assert abs(val - 1.0) < 1e-10, '[TC10] Vandermonde2D 对 1 积分应为 1.0 FAILED'

# ---- TC11: ErrorNorms L¹ 范数——已知数组 ----
err = ErrorNorms()
arr = np.array([1.0, -2.0, 3.0])
l1 = err.l1_norm_discrete(arr)
assert abs(l1 - 6.0) < 1e-12, '[TC11] L1 范数不正确 FAILED'

# ---- TC12: ErrorNorms L² 范数——已知数组 ----
l2 = err.l2_norm_discrete(arr)
assert abs(l2 - np.sqrt(14.0)) < 1e-12, '[TC12] L2 范数不正确 FAILED'

# ---- TC13: ErrorNorms L∞ 范数 ----
linf = err.linf_norm(arr)
assert abs(linf - 3.0) < 1e-12, '[TC13] L∞ 范数不正确 FAILED'

# ---- TC14: ErrorNorms 离散 L¹ 范数（带体积元） ----
vals = np.array([2.0, 3.0])
vols = np.array([0.5, 0.5])
l1_vol = err.l1_norm_discrete(vals, vols)
assert abs(l1_vol - 2.5) < 1e-12, '[TC14] 带体积元 L1 范数不正确 FAILED'

# ---- TC15: ErrorNorms 收敛阶估计——已知二阶 ----
e = np.array([0.01, 0.0025, 0.000625])
h = np.array([0.1, 0.05, 0.025])
orders = err.convergence_order(e, h)
assert np.all(np.abs(orders - 2.0) < 0.1), '[TC15] 收敛阶估计不正确 FAILED'

# ---- TC16: ErrorNorms Richardson 外推——二阶修正 ----
uh = np.array([1.0 + 0.04])
uh2 = np.array([1.0 + 0.01])
extrap = err.richardson_extrapolation(uh, uh2, p=2.0)
assert abs(extrap[0] - 1.0) < 1e-12, '[TC16] Richardson 外推不正确 FAILED'

# ---- TC17: BisectionSolver 求 x²-2=0 的根 ----
bisect = BisectionSolver(max_iter=200, tol=1e-12)
root, iters = bisect.solve(lambda x: x**2 - 2, 1.0, 2.0)
assert abs(root - np.sqrt(2.0)) < 1e-10, '[TC17] 二分法求 √2 不准确 FAILED'
assert iters > 0, '[TC17] 二分迭代次数应为正 FAILED'

# ---- TC18: BisectionSolver 同号区间触发 ValueError ----
try:
    bisect.solve(lambda x: x**2 + 1, 1.0, 2.0)
    assert False, '[TC18] 应触发 ValueError FAILED'
except ValueError:
    pass

# ---- TC19: CollatzPolynomial 次数正确 ----
poly = CollatzPolynomial(np.array([1, 0, 1, 1]))
assert poly.degree() == 3, '[TC19] 多项式次数不正确 FAILED'

# ---- TC20: CollatzPolynomial 序列在最大步数内终止 ----
seq = poly.sequence(max_steps=30)
assert len(seq) <= 30, '[TC20] 序列未在最大步数内终止 FAILED'

# ---- TC21: CollatzPolynomial smooth_analog |x|<1 收敛到 0 ----
fp = CollatzPolynomial.smooth_analog(0.5, max_iter=50)
assert abs(fp) < 1e-10, '[TC21] 平滑映射小初值应收敛到 0 FAILED'

# ---- TC22: ResonanceEigenSolver 灵敏度 dλ/dn 公式 ----
solver = ResonanceEigenSolver(R_major=10.0e-6, n_nominal=3.47)
S = solver.sensitivity_dlambda_dn(m=140, lambda_nm=1550.0, n_eff=3.47)
assert abs(S - 1550.0 / 3.47) < 1e-10, '[TC22] 灵敏度公式不正确 FAILED'

# ---- TC23: 谐振条件残差符号验证 ----
res = solver.resonance_condition(m=140, lambda_nm=1500.0, n_eff=3.47)
assert isinstance(res, float), '[TC23] 谐振条件残差应为浮点数 FAILED'
assert np.isfinite(res), '[TC23] 谐振条件残差应有限 FAILED'

# ---- TC24: 幂迭代最大本征值——对角占优矩阵 ----
np.random.seed(42)
n_test = 20
A_test = np.diag(np.linspace(1.0, 10.0, n_test))
lam, vec, it_ev = solver.power_iteration_eigenvalue(A_test, max_iter=500, tol=1e-10)
assert abs(lam - 10.0) < 1e-6, '[TC24] 幂迭代最大本征值不正确 FAILED'

# ---- TC25: 随机变量生成器固定种子可复现 ----
np.random.seed(42)
rng1 = RandomVariateGenerator(seed=42)
s1 = rng1.normal(0.0, 1.0, size=100)
rng2 = RandomVariateGenerator(seed=42)
s2 = rng2.normal(0.0, 1.0, size=100)
assert np.allclose(s1, s2), '[TC25] 种子 42 下随机数不可复现 FAILED'

# ---- TC26: HypersphereSampler 采样点在超球面上 ----
np.random.seed(42)
hyper = HypersphereSampler(dim=5, seed=42)
pts = hyper.sample(50)
norms = np.linalg.norm(pts, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-14), '[TC26] 超球面采样点不在球面上 FAILED'

# ---- TC27: HypersphereSampler 角度统计数值范围 ----
np.random.seed(42)
stats = hyper.angle_statistics(n_pairs=200)
assert 0.0 <= stats['mean_abs_cos'] <= 1.0, '[TC27] mean_abs_cos 超出 [0,1] FAILED'
assert stats['mean_abs_cos'] >= 0.0, '[TC27] mean_abs_cos 应为非负 FAILED'

# ---- TC28: BandedMatrixSolver 对角求解残差 ----
bs = BandedMatrixSolver(n=5, ml=0, mu=0)
for i in range(5):
    bs.set_element(i, i, float(i + 1) * 2.0)
bs.factorize_np()
b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
x = bs.solve(b)
Ax = bs.matvec(x)
assert np.linalg.norm(Ax - b) < 1e-14, '[TC28] 带状对角求解残差过大 FAILED'
assert np.allclose(x, np.ones(5)), '[TC28] 对角求解预期 x=[1,1,1,1,1] FAILED'

# ---- TC29: LaplacianStencils 5点格式解析验证 ----
nx_l = ny_l = 21
L_l = 2.0
x_arr = np.linspace(0, L_l, nx_l)
y_arr = np.linspace(0, L_l, ny_l)
X, Y = np.meshgrid(x_arr, y_arr)
u = np.sin(np.pi * X / L_l) * np.sin(np.pi * Y / L_l)
h_l = L_l / (nx_l - 1)
Lu = LaplacianStencils.laplacian5_2d(u, h_l)
Lu_theory = -2.0 * (np.pi / L_l)**2 * u
interior = Lu[1:-1, 1:-1] - Lu_theory[1:-1, 1:-1]
assert np.sqrt(np.mean(interior**2)) < 0.05, '[TC29] 5点 Laplacian 误差过大 FAILED'

# ---- TC30: LaplacianStencils 9点 torus 对常数为零 ----
u_const = np.ones((10, 10))
Lu9 = LaplacianStencils.laplacian9_torus(u_const, h=0.1)
assert np.max(np.abs(Lu9)) < 1e-14, '[TC30] 9点 torus Laplacian 对常数应输出 0 FAILED'

# ---- TC31: SensorResponseSurrogate 训练后预测输出数值范围 ----
surr = SensorResponseSurrogate()
surr.train(n_points=100)
pred = surr.predict(0.0, 1.00)
assert np.isfinite(pred), '[TC31] 代理模型预测应有限 FAILED'
assert isinstance(pred, float), '[TC31] 预测应为浮点数 FAILED'

# ---- TC32: SensorResponseSurrogate 可复现性（相同种子） ----
surr2 = SensorResponseSurrogate()
surr2.train(n_points=100)
pred2 = surr2.predict(0.0, 1.00)
# 同种子、同训练数据，预测应相同
assert abs(pred - pred2) < 1e-10, '[TC32] 相同种子的训练结果应一致 FAILED'

# ---- TC33: ErrorNorms compute_quality_metrics 动态范围正值 ----
field = np.linspace(0.1, 10.0, 100)
metrics = err.compute_quality_metrics(field)
assert metrics['max'] > metrics['min'], '[TC33] 质量指标 max/min 顺序错误 FAILED'
assert metrics['dynamic_range_db'] > 0, '[TC33] 动态范围应为正值 FAILED'

# ---- TC34: ErrorNorms relative_l2_error 零残差返回 0 ----
A_ref = np.array([1.0, 2.0, 3.0])
rel = err.relative_l2_error(A_ref, A_ref)
assert rel < 1e-14, '[TC34] 零残差相对 L2 误差应为 0 FAILED'

# ---- TC35: LaplacianStencils 3点非均匀1D——已知解析解 ----
x_1d = np.array([0.0, 0.5, 1.2, 2.0])
u_1d = x_1d ** 2
Lu3 = LaplacianStencils.laplacian3_uneven_1d(u_1d, x_1d)
# u'' = 2，内部点应近似
assert abs(Lu3[1] - 2.0) < 1e-10, '[TC35] 3点非均匀 Laplacian 不正确 FAILED'
assert abs(Lu3[2] - 2.0) < 1e-10, '[TC35] 3点非均匀 Laplacian 不正确 FAILED'

# ---- TC36: GaussLegendreTensor 对 f(x,y)=1 积分为面积 ----
gt = GaussLegendreTensor()
f_one = lambda x, y: 1.0
approx_one = gt.tensor_quad_2d(f_one, 0.0, 1.0, 0.0, 1.0, m=4)
assert abs(approx_one - 1.0) < 1e-14, '[TC36] GaussLegendre 对 1 积分应为 1.0 FAILED'

# ---- TC37: MonteCarloUQ 参数扰动键完整性 ----
np.random.seed(42)
mc = MonteCarloUQ(n_samples=5, seed=42)
base = {"R_major": 10.0e-6, "r_minor": 1.5e-6, "n_ring": 3.47}
std = {"R_major": 0.05e-6, "r_minor": 0.02e-6, "n_ring": 0.01}
p = mc.parameter_perturbation(base, std)
assert set(p.keys()) == set(base.keys()), '[TC37] 扰动参数字典键不正确 FAILED'
assert p["n_ring"] > 0, '[TC37] 扰动后折射率应 > 0 FAILED'

# ---- TC38: MonteCarloUQ MC 传播输出结构正确 ----
np.random.seed(42)
mc2 = MonteCarloUQ(n_samples=30, seed=42)
base2 = {"R_major": 10.0e-6, "r_minor": 1.5e-6, "n_ring": 3.47}
std2 = {"R_major": 0.05e-6, "r_minor": 0.02e-6, "n_ring": 0.01}
fwd = lambda p: {"lambda_res_nm": 2.0 * np.pi * p["R_major"] * p["n_ring"] / 120 * 1e9}
summary = mc2.run_mc_propagation(base2, std2, fwd)
assert summary['n_success'] > 0, '[TC38] MC 应有成功样本 FAILED'
assert 'lambda_res_nm' in summary, '[TC38] 输出应包含 lambda_res_nm FAILED'
assert summary['lambda_res_nm']['std'] >= 0, '[TC38] std 应为非负 FAILED'

# ---- TC39: ErrorNorms residual_norm 一致性 ----
residual = np.array([0.1, -0.2, 0.3])
res_norm = err.residual_norm(residual)
l2_direct = np.sqrt(np.sum(residual**2))
assert abs(res_norm - l2_direct) < 1e-14, '[TC39] residual_norm 与直接计算不一致 FAILED'

# ---- TC40: BandedMatrixSolver matvec 对角矩阵乘法 ----
bs2 = BandedMatrixSolver(n=4, ml=0, mu=0)
for i in range(4):
    bs2.set_element(i, i, float(i + 1))
bs2.factorize_np()
x_vec = np.array([2.0, 3.0, 4.0, 5.0])
b_vec = bs2.matvec(x_vec)
assert abs(b_vec[0] - 2.0) < 1e-14, '[TC40] matvec row0 不正确 FAILED'
assert abs(b_vec[1] - 6.0) < 1e-14, '[TC40] matvec row1 不正确 FAILED'
assert abs(b_vec[2] - 12.0) < 1e-14, '[TC40] matvec row2 不正确 FAILED'
assert abs(b_vec[3] - 20.0) < 1e-14, '[TC40] matvec row3 不正确 FAILED'
