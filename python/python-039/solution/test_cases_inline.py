# ---- TC01: Woods-Saxon密度非负有限 ----
geo_test = NuclearGeometry(mass_number_a=197, mass_number_b=197, radius_param=1.12, diffuseness=0.54, nucleon_cross_section=4.2)
r_test = np.array([0.0, 5.0, 10.0, 20.0])
rho_vals = geo_test.woods_saxon_density(r_test)
assert np.all(rho_vals >= 0.0) and np.all(np.isfinite(rho_vals)), '[TC01] Woods-Saxon密度非负有限 FAILED'

# ---- TC02: 厚度函数积分近似归一到质量数 ----
x_t = np.linspace(-15.0, 15.0, 80)
y_t = np.linspace(-15.0, 15.0, 80)
X_t, Y_t = np.meshgrid(x_t, y_t, indexing='ij')
T_A = geo_test.thickness_function(X_t, Y_t, 'A')
dx_t = x_t[1] - x_t[0]
dy_t = y_t[1] - y_t[0]
integral_TA = np.sum(T_A) * dx_t * dy_t
assert 150.0 < integral_TA < 220.0, '[TC02] 厚度函数积分归一性 FAILED'

# ---- TC03: 小碰撞参数下N_part和N_coll为正 ----
np_test, nc_test = geo_test.compute_npart_ncoll(2.0, x_t, y_t)
assert np_test > 0.0 and nc_test > 0.0, '[TC03] 小b时N_part/N_coll为正 FAILED'

# ---- TC04: 偏心距在合理范围[0,1] ----
eps2_test, eps4_test = geo_test.eccentricity(5.0, x_t, y_t)
assert 0.0 <= eps2_test <= 1.0 and abs(eps4_test) <= 1.0, '[TC04] 偏心距范围 FAILED'

# ---- TC05: tortoise边界字输出长度正确 ----
bx_test, by_test = geo_test.tortoise_boundary_word(n_segments=32)
assert len(bx_test) == 32 and len(by_test) == 32, '[TC05] 边界字长度 FAILED'

# ---- TC06: 直角三角形面积正确 ----
mesh_test = MeshGenerator(max_area=1.0, min_angle=20.0)
pts_right = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
tri_right = np.array([[0, 1, 2]])
area_right = mesh_test.triangle_area(pts_right, tri_right)
assert abs(area_right[0] - 6.0) < 1e-10, '[TC06] 直角三角形面积 FAILED'

# ---- TC07: 等边三角形质量为1 ----
pts_equilateral = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]])
tri_eq = np.array([[0, 1, 2]])
quality_eq = mesh_test.triangle_quality(pts_equilateral, tri_eq)
assert abs(quality_eq[0] - 1.0) < 1e-10, '[TC07] 等边三角形质量 FAILED'

# ---- TC08: 状态方程线性P=cs2*epsilon ----
hydro_test = HydroEvolution(eta_s_over_s=0.08, cs2=1.0/3.0, g_star=47.5, tau0=0.6, tau_f=2.0, dtau=0.1)
eps_test = np.array([1.0, 5.0, 10.0])
P_test = hydro_test.equation_of_state(eps_test)
assert np.allclose(P_test, hydro_test.cs2 * eps_test), '[TC08] 状态方程线性 FAILED'

# ---- TC09: 温度-能量密度互逆 ----
T_from_eps = hydro_test.energy_to_temperature(eps_test)
eps_back = hydro_test.temperature_to_energy(T_from_eps)
assert np.allclose(eps_test, eps_back, rtol=1e-5), '[TC09] 温度能量互逆 FAILED'

# ---- TC10: Bjorken 1D能量单调不增 ----
tau_bj, eps_bj, T_bj = hydro_test.bjorken_1d(15.0)
assert np.all(np.diff(eps_bj) <= 1e-10), '[TC10] Bjorken能量单调不增 FAILED'

# ---- TC11: RF椭圆积分对称性 ----
rf_123, _ = rf_carlson(1.0, 2.0, 3.0)
rf_213, _ = rf_carlson(2.0, 1.0, 3.0)
assert abs(rf_123 - rf_213) < 1e-6, '[TC11] RF对称性 FAILED'

# ---- TC12: RC退化解析解 RC(x,x)=1/sqrt(x) ----
rc_xx, _ = rc_carlson(4.0, 4.0)
assert abs(rc_xx - 1.0/np.sqrt(4.0)) < 1e-6, '[TC12] RC退化解析解 FAILED'

# ---- TC13: 零温下胶子能量损失为0 ----
delta_E_zero = QGPDispersionRelation.gluon_energy_loss(10.0, 0.5, 0.0)
assert delta_E_zero == 0.0, '[TC13] 零温能量损失 FAILED'

# ---- TC14: 混合RNG输出在[0,1) ----
rng_test = MiddleSquareHybrid(seed=12345, d=4)
rands = np.array([rng_test.random() for _ in range(100)])
assert np.all(rands >= 0.0) and np.all(rands < 1.0), '[TC14] RNG范围 FAILED'

# ---- TC15: 固定种子可复现 ----
rng_a = MiddleSquareHybrid(seed=99999, d=4)
rng_b = MiddleSquareHybrid(seed=99999, d=4)
vals_a = [rng_a.random() for _ in range(20)]
vals_b = [rng_b.random() for _ in range(20)]
assert vals_a == vals_b, '[TC15] 固定种子可复现 FAILED'

# ---- TC16: Bell数 B(0)=1, B(1)=1 ----
bell_test = CombinatorialPhysics.bell_numbers(5)
assert bell_test[0] == 1 and bell_test[1] == 1, '[TC16] Bell数边界 FAILED'

# ---- TC17: Stirling数 S(n,n)=1 ----
s_test = CombinatorialPhysics.stirling_numbers_second_kind(8, 8)
assert s_test[8, 8] == 1 and s_test[5, 5] == 1 and s_test[1, 1] == 1, '[TC17] Stirling数S(n,n)=1 FAILED'

# ---- TC18: HistogramAnalysis均值与numpy一致 ----
np.random.seed(42)
data_hist = np.random.normal(5.0, 2.0, 500)
hist_test = HistogramAnalysis(data_hist, n_bins=30, range_limits=(0.0, 10.0))
assert abs(hist_test.mean() - np.mean(data_hist)) < 1e-10, '[TC18] 直方图均值 FAILED'

# ---- TC19: 从v2反推偏心距线性响应 ----
flow_test = FlowHarmonicDecomposition()
eps_est = flow_test.eccentricity_from_flow(0.05, response_coeff=0.18)
assert abs(eps_est - 0.05/0.18) < 1e-10, '[TC19] 偏心距线性响应 FAILED'

# ---- TC20: RREFSolver求解3x3线性系统 ----
solver_test = RREFSolver()
A_tc = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
b_tc = np.array([8.0, -11.0, -3.0])
x_tc = solver_test.solve(A_tc, b_tc)
assert np.allclose(A_tc @ x_tc, b_tc), '[TC20] RREF求解3x3 FAILED'

# ---- TC21: 单位矩阵行列式为1 ----
det_I = solver_test.determinant(np.eye(4))
assert abs(det_I - 1.0) < 1e-10, '[TC21] 单位矩阵行列式 FAILED'

# ---- TC22: Hamming编解码自洽 ----
data_ham = np.array([1, 0, 1, 1])
cw = HammingErrorDetection.encode(data_ham)
decoded_ham, _ = HammingErrorDetection.decode(cw)
assert np.array_equal(decoded_ham, data_ham), '[TC22] Hamming编解码 FAILED'

# ---- TC23: Hamming校验线性系统正确解 ----
check_pass = HammingErrorDetection.check_linear_system(A_tc, x_tc, b_tc)
assert check_pass, '[TC23] Hamming校验正确解 FAILED'

# ---- TC24: Wilson-Dirac矩阵形状正确 ----
dirac_test = LatticeDiracSolver(mass=0.1, lattice_size=4)
D_test = dirac_test.wilson_dirac_matrix()
assert D_test.shape == (16, 16), '[TC24] Dirac矩阵形状 FAILED'

# ---- TC25: 非中心t分位数搜索中位数约等于0(delta=0) ----
t_q_test, iters_test = NonCentralTDistribution.quantile_search(0.5, 30.0, delta=0.0, a=-5.0, b=5.0)
assert abs(t_q_test) < 0.5, '[TC25] 非中心t中位数 FAILED'

# ---- TC26: v2显著性零信号为零显著性 ----
inf_test = QGPStatisticalInference()
t_v2_zero, sig_v2_zero = inf_test.v2_significance(0.0, 0.01, 0.0)
assert sig_v2_zero == 0.0, '[TC26] 零v2显著性 FAILED'

# ---- TC27: 二次优化器求抛物线最小值 ----
opt_test = QuadraticOptimizer(max_iter=50)
f_parabola = lambda x: (x - 3.0)**2 + 1.0
x_opt, iters_opt, f_opt = opt_test.optimize(f_parabola, 0.0, 2.0, 5.0)
assert abs(x_opt - 3.0) < 0.1 and abs(f_opt - 1.0) < 0.1, '[TC27] 二次优化抛物线 FAILED'

# ---- TC28: SVD方差比总和为1 ----
np.random.seed(42)
X_svd = np.random.randn(20, 10)
svd_test = EventSVDAnalyzer(n_components=5)
svd_test.fit(X_svd)
ratios = svd_test.explained_variance_ratio()
assert abs(np.sum(ratios) - 1.0) < 1e-10, '[TC28] SVD方差比总和 FAILED'

# ---- TC29: 分裂概率边界为零 ----
cascade_test = PartonCascade(alpha_s=0.3, q0=1.0)
p_boundary = cascade_test.splitting_probability(0.0, 1.0, 'gg')
assert p_boundary == 0.0, '[TC29] 分裂概率边界 FAILED'

# ---- TC30: 事件采样方位角固定种子可复现 ----
np.random.seed(42)
rng_s1 = MiddleSquareHybrid(seed=77777, d=4)
sampler_s1 = QGPEventSampler(rng_s1)
phi_s1 = sampler_s1.sample_azimuthal_angle(v2=0.05, n_samples=100)
rng_s2 = MiddleSquareHybrid(seed=77777, d=4)
sampler_s2 = QGPEventSampler(rng_s2)
phi_s2 = sampler_s2.sample_azimuthal_angle(v2=0.05, n_samples=100)
assert np.allclose(phi_s1, phi_s2), '[TC30] 方位角采样可复现 FAILED'
