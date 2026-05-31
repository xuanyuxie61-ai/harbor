# ---- TC01: generate_conformal_array 球面点半径一致 ----
pos_conf = generate_conformal_array(1.0, 2, 4)
r_dist = np.linalg.norm(pos_conf, axis=1)
assert np.allclose(r_dist, 1.0), '[TC01] generate_conformal_array 球面点半径一致 FAILED'

# ---- TC02: sphere_llt_grid_line_count 公式验证 ----
lines = sphere_llt_grid_line_count(2, 4)
assert lines == 24, '[TC02] sphere_llt_grid_line_count 公式验证 FAILED'

# ---- TC03: generate_planar_array 矩形口径单元数 ----
pos = generate_planar_array(4, 4, 0.5, 0.5, aperture_type='rectangular')
assert pos.shape == (16, 3), '[TC03] generate_planar_array 矩形口径单元数 FAILED'

# ---- TC04: generate_planar_array 圆形口径过滤 ----
pos_c = generate_planar_array(4, 4, 0.5, 0.5, aperture_type='circular')
assert pos_c.shape[0] <= 16 and pos_c.shape[1] == 3, '[TC04] generate_planar_array 圆形口径过滤 FAILED'

# ---- TC05: generate_conformal_array 输出尺寸 ----
pos_conf2 = generate_conformal_array(1.0, 2, 4)
assert pos_conf2.shape == (10, 3), '[TC05] generate_conformal_array 输出尺寸 FAILED'

# ---- TC06: tet_mesh_quality_metrics 返回结构 ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
tetra = np.array([[0, 1, 2, 3]])
q = tet_mesh_quality_metrics(nodes, tetra)
assert 'quality1' in q and 'min' in q['quality1'], '[TC06] tet_mesh_quality_metrics 返回结构 FAILED'

# ---- TC07: ArrayFactorCalculator 方向图输出尺寸 ----
pos = generate_planar_array(4, 4, 0.5, 0.5)
afc = ArrayFactorCalculator(pos, frequency_hz=3e9)
theta = np.linspace(-np.pi / 2, np.pi / 2, 37)
phi = np.zeros_like(theta)
af = afc.compute_array_factor(theta, phi)
assert af.shape == theta.shape, '[TC07] ArrayFactorCalculator 方向图输出尺寸 FAILED'

# ---- TC08: ArrayFactorCalculator 总方向图峰值在波束指向 ----
afc.apply_steering(np.pi / 6, 0.0)
pat_db = afc.compute_total_pattern_db(theta, phi, normalize=True)
assert np.max(pat_db) >= -0.1, '[TC08] ArrayFactorCalculator 总方向图峰值 FAILED'

# ---- TC09: ArrayFactorCalculator 方向性系数为正 ----
D = afc.directivity(theta_grid=31, phi_grid=31)
assert D > 0.0 and np.isfinite(D), '[TC09] ArrayFactorCalculator 方向性系数为正 FAILED'

# ---- TC10: MutualCouplingMatrix 阻抗矩阵对称 ----
mc = MutualCouplingMatrix(pos[:4], frequency_hz=3e9)
Z = mc.build_impedance_matrix()
assert Z.shape == (4, 4) and np.allclose(Z, Z.T), '[TC10] MutualCouplingMatrix 阻抗矩阵对称 FAILED'

# ---- TC11: near_field_e_field 返回形状正确 ----
obs = np.array([[0.5, 0.0, 0.1], [0.0, 0.5, 0.1]])
curr = np.ones(pos.shape[0], dtype=complex)
E = near_field_e_field(pos, curr, obs, 3e9)
assert E.shape == (2, 3), '[TC11] near_field_e_field 返回形状正确 FAILED'

# ---- TC12: bivariate_normal_cdf 返回值在概率区间 ----
p_bound = bivariate_normal_cdf(1.0, 1.0, 0.5)
assert 0.0 <= p_bound <= 1.0, '[TC12] bivariate_normal_cdf 返回值在概率区间 FAILED'

# ---- TC13: bivariate_normal_cdf 独立性分解 ----
p2 = bivariate_normal_cdf(0.0, 0.0, 0.0)
assert abs(p2 - 0.25) < 1e-3, '[TC13] bivariate_normal_cdf 独立性分解 FAILED'

# ---- TC14: RandomWalkPhaseNoise 理论均方位移 ----
rw = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
msd = rw.theoretical_msd(500)
assert abs(msd - 500 * 0.02 ** 2) < 1e-10, '[TC14] RandomWalkPhaseNoise 理论均方位移 FAILED'

# ---- TC15: RandomWalkPhaseNoise 可复现性 ----
np.random.seed(42)
rw1 = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
t1, x1, m1 = rw1.simulate(10, 100)
np.random.seed(42)
rw2 = RandomWalkPhaseNoise(step_delta=0.02, seed=42)
t2, x2, m2 = rw2.simulate(10, 100)
assert np.allclose(x1, x2) and np.allclose(m1, m2), '[TC15] RandomWalkPhaseNoise 可复现性 FAILED'

# ---- TC16: sample_spatial_fading 输出尺寸与范围 ----
np.random.seed(42)
gains = sample_spatial_fading(n_samples=200, correlation_length=0.3, seed=42)
assert gains.shape == (200,) and np.all(gains > 0), '[TC16] sample_spatial_fading 输出尺寸与范围 FAILED'

# ---- TC17: sidelobe_level_cdf 概率范围 ----
prob = sidelobe_level_cdf(-10.0, n_elements=64)
assert 0.0 <= prob <= 1.0, '[TC17] sidelobe_level_cdf 概率范围 FAILED'

# ---- TC18: RK12Solver 常数 ODE 精确保持 ----
def const_ode(t, y):
    return np.zeros_like(y)


solver = RK12Solver(const_ode)
t, y, e = solver.solve((0.0, 1.0), np.array([2.0, 3.0]), 10)
assert np.allclose(y[-1], np.array([2.0, 3.0])), '[TC18] RK12Solver 常数 ODE 精确保持 FAILED'

# ---- TC19: KeplerPerturbedArrayCoupling Hamiltonian 有限 ----
kep = KeplerPerturbedArrayCoupling(delta=0.015, e=0.6)
H = kep.conserved_quantity(kep.y0)
assert np.isfinite(H), '[TC19] KeplerPerturbedArrayCoupling Hamiltonian 有限 FAILED'

# ---- TC20: LangfordBeamPhaseDynamics derivative 输出维度 ----
lang = LangfordBeamPhaseDynamics()
dydt = lang.derivative(0.0, lang.y0)
assert dydt.shape == (3,), '[TC20] LangfordBeamPhaseDynamics derivative 输出维度 FAILED'

# ---- TC21: DigitalPhaseShifter 量化相位周期性 ----
dps = DigitalPhaseShifter(bits=4)
qphase = dps.quantize_phase(np.array([0.0, 2 * np.pi, 4 * np.pi]))
assert np.allclose(qphase, qphase[0]), '[TC21] DigitalPhaseShifter 量化相位周期性 FAILED'

# ---- TC22: hamming_distance_matrix_gray 相邻距离为1 ----
dg = hamming_distance_matrix_gray(7)
adj = [int(dg[i, i + 1]) for i in range(7)]
assert all(v == 1 for v in adj), '[TC22] hamming_distance_matrix_gray 相邻距离为1 FAILED'

# ---- TC23: integrate_square_minimal 常数函数精确 ----
val = integrate_square_minimal(lambda x, y: np.ones_like(x), deg=3)
assert abs(val - 4.0) < 1e-12, '[TC23] integrate_square_minimal 常数函数精确 FAILED'

# ---- TC24: integrate_square_minimal x2+y2 解析验证 ----
val2 = integrate_square_minimal(lambda x, y: x ** 2 + y ** 2, deg=3)
assert abs(val2 - 8.0 / 3.0) < 1e-10, '[TC24] integrate_square_minimal x2+y2 解析验证 FAILED'

# ---- TC25: NewtonInterpolator1D 节点处精确恢复 ----
xd = np.array([0.0, 1.0, 2.0, 3.0])
yd = xd ** 2
interp = NewtonInterpolator1D(xd, yd)
yi = interp.evaluate(xd)
assert np.allclose(yi, yd), '[TC25] NewtonInterpolator1D 节点处精确恢复 FAILED'

# ---- TC26: convergence_rate_residual 几何级数 ----
res = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
rate = convergence_rate_residual(res)
assert abs(rate - 0.5) < 0.05, '[TC26] convergence_rate_residual 几何级数 FAILED'

# ---- TC27: safe_inverse_sqrt 非正输入鲁棒性 ----
result = safe_inverse_sqrt(np.array([-1.0, 0.0, 4.0]))
assert np.all(np.isfinite(result)) and abs(result[2] - 0.5) < 1e-10, '[TC27] safe_inverse_sqrt 非正输入鲁棒性 FAILED'

# ---- TC28: rotation_matrix_z 正交性验证 ----
Rz = rotation_matrix_z(np.pi / 3)
assert np.allclose(Rz @ Rz.T, np.eye(3)), '[TC28] rotation_matrix_z 正交性验证 FAILED'

# ---- TC29: cube_distance_pdf_exact 非负性 ----
dvals = np.array([0.1, 0.5, 1.0, 1.3, 1.6])
pdf_vals = cube_distance_pdf_exact(dvals)
assert np.all(pdf_vals >= 0), '[TC29] cube_distance_pdf_exact 非负性 FAILED'

# ---- TC30: filename_increment 基本递增 ----
fn = filename_increment("sim_009.txt")
assert fn == "sim_010.txt", '[TC30] filename_increment 基本递增 FAILED'
