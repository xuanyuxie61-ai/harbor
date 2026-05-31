# ---- TC01: sincn_fun在零点返回1 ----
result = sincn_fun(np.array([0.0]))
assert np.abs(result[0] - 1.0) < 1e-10, '[TC01] sincn_fun在零点返回1 FAILED'

# ---- TC02: sincn_fun在整数点返回0 ----
result = sincn_fun(np.array([1.0, 2.0, -1.0]))
assert np.all(np.abs(result) < 1e-10), '[TC02] sincn_fun在整数点返回0 FAILED'

# ---- TC03: alnorm(0)接近0.5 ----
result = alnorm(0.0)
assert np.abs(result - 0.5) < 1e-4, '[TC03] alnorm(0)接近0.5 FAILED'

# ---- TC04: alnorm(2)接近理论值0.97725 ----
result = alnorm(2.0)
assert np.abs(result - 0.97725) < 1e-4, '[TC04] alnorm(2)接近理论值 FAILED'

# ---- TC05: cisi的Si(1)接近已知值 ----
ci_val, si_val = cisi(np.array([1.0]))
assert np.abs(si_val[0] - 0.946083070367183) < 1e-4, '[TC05] cisi的Si(1) FAILED'

# ---- TC06: binomial_coefficient(20,10)等于184756 ----
result = binomial_coefficient(20, 10)
assert abs(result - 184756.0) < 0.5, '[TC06] binomial_coefficient(20,10) FAILED'

# ---- TC07: chebyshev_to_monomial_matrix形状正确且可逆 ----
M = chebyshev_to_monomial_matrix(5)
assert M.shape == (6, 6), '[TC07] chebyshev_to_monomial_matrix形状 FAILED'
assert abs(np.linalg.det(M)) > 1e-10, '[TC07] chebyshev_to_monomial_matrix行列式 FAILED'

# ---- TC08: point_in_polygon对矩形内点返回True ----
from mesh_builder import point_in_polygon
inside = point_in_polygon([0, 10, 10, 0], [0, 0, 10, 10], 5, 5)
assert inside == True, '[TC08] point_in_polygon对矩形内点 FAILED'

# ---- TC09: point_in_polygon对矩形外点返回False ----
from mesh_builder import point_in_polygon
outside = point_in_polygon([0, 10, 10, 0], [0, 0, 10, 10], 15, 5)
assert outside == False, '[TC09] point_in_polygon对矩形外点 FAILED'

# ---- TC10: generate_depth_grid边界与单调性 ----
z_grid = generate_depth_grid(1000.0, 51, stretch_power=2.0)
assert z_grid[0] == 0.0, '[TC10] generate_depth_grid首点 FAILED'
assert abs(z_grid[-1] - 1000.0) < 1e-6, '[TC10] generate_depth_grid末点 FAILED'
assert np.all(np.diff(z_grid) > 0), '[TC10] generate_depth_grid单调性 FAILED'

# ---- TC11: build_initial_field高斯束峰值在声源深度处 ----
z = np.linspace(0, 100, 101)
u = build_initial_field(z, z_s=50.0, source_type='gaussian', k0=1.0, w0=5.0)
assert np.argmax(np.abs(u)) == 50, '[TC11] build_initial_field峰值位置 FAILED'

# ---- TC12: source_power_normalization后能量为1 ----
z = np.linspace(0, 100, 101)
u = build_initial_field(z, z_s=50.0, source_type='gaussian', k0=1.0, w0=5.0)
u_norm = source_power_normalization(u, z)
power = np.trapezoid(np.abs(u_norm)**2, z)
assert abs(power - 1.0) < 1e-6, '[TC12] source_power_normalization能量 FAILED'

# ---- TC13: disk_uniform_sample所有点在圆盘内 ----
np.random.seed(42)
samples = disk_uniform_sample(1000, radius=1.0, seed=42)
radii = np.linalg.norm(samples, axis=1)
assert np.all(radii <= 1.0 + 1e-10), '[TC13] disk_uniform_sample越界 FAILED'

# ---- TC14: hammersley_sequence值域在0到1之间 ----
from source_field import hammersley_sequence
seq = hammersley_sequence(0, 100, 3)
assert np.all(seq >= 0.0) and np.all(seq <= 1.0), '[TC14] hammersley_sequence值域 FAILED'

# ---- TC15: OceanEnvironment声速最小值不低于1400 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
z_test = np.linspace(0, 4000, 41)
c_vals = env.sound_speed(z_test)
assert np.all(c_vals >= 1400.0), '[TC15] OceanEnvironment声速最小值 FAILED'

# ---- TC16: OceanEnvironment吸收系数非负 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
alpha = env.absorption_db_per_km()
assert alpha >= 0.0, '[TC16] OceanEnvironment吸收系数 FAILED'

# ---- TC17: ModalConstraintSolver正确求解Diophantine不等式 ----
solver = ModalConstraintSolver()
solutions = solver.solve_inequality_integer(a=2.0, b=10.0)
assert solutions == [0, 1, 2, 3, 4], '[TC17] ModalConstraintSolver求解 FAILED'

# ---- TC18: ReceiverArray Dolph-Chebyshev权重和为1 ----
z_vla = np.linspace(50.0, 1500.0, 16)
r_vla = np.full_like(z_vla, 10000.0)
vla = ReceiverArray(r_vla, z_vla)
w = vla.dolph_chebyshev_weights(sidelobe_db=-25)
assert abs(np.sum(w) - 1.0) < 1e-10, '[TC18] Dolph-Chebyshev权重和 FAILED'

# ---- TC19: VolumeScatteringModel散射强度非负 ----
scat = VolumeScatteringModel(sigma0=1e-6, z0=50.0, alpha=0.3, Lambda=100.0)
z_test = np.array([10.0, 50.0, 100.0])
sv = scat.scattering_strength_linear(z_test)
assert np.all(sv >= 0.0), '[TC19] VolumeScatteringModel散射强度 FAILED'

# ---- TC20: triangle_monomial_integral在参考三角形上积分x*y ----
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
I_tri = triangle_monomial_integral(1, 1, tri)
assert abs(I_tri - 1.0/24.0) < 1e-12, '[TC20] triangle_monomial_integral FAILED'

# ---- TC21: BoundaryConditionHandler海面边界条件将首点置零 ----
env = OceanEnvironment(c0=1500.0, z_axis=1000.0, B=1000.0, epsilon=0.0057, depth_max=4000.0, frequency=100.0)
z_grid = generate_depth_grid(4000.0, 51, stretch_power=2.0)
r_grid = generate_range_grid(1000.0, 100.0)
mesh = PEMesh(r_grid, z_grid, env)
bc = BoundaryConditionHandler(env, mesh)
u = np.ones(51, dtype=np.complex128)
u_out = bc.apply_surface_bc(u)
assert u_out[0] == 0.0, '[TC21] apply_surface_bc首点 FAILED'

# ---- TC22: safe_divide对零除数返回fill_value ----
from utils import safe_divide
result = safe_divide(5.0, 0.0, fill_value=999.0)
assert result == 999.0, '[TC22] safe_divide对零除数返回fill_value FAILED'

# ---- TC23: sincu_fun在零点返回1 ----
from special_functions import sincu_fun
result = sincu_fun(np.array([0.0]))
assert np.abs(result[0] - 1.0) < 1e-10, '[TC23] sincu_fun在零点返回1 FAILED'

# ---- TC24: 三角形数公式正确 ----
from utils import triangle_number
result = triangle_number(10)
assert result == 55, '[TC24] triangle_number(10) FAILED'

# ---- TC25: 集成测试main返回结果包含关键字段 ----
result = main()
assert 'U' in result, '[TC25] main结果缺少U FAILED'
assert 'tl_coh' in result, '[TC25] main结果缺少tl_coh FAILED'
assert 'kr' in result, '[TC25] main结果缺少kr FAILED'
