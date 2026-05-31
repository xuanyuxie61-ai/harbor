
# ---- TC01: gamma_lanczos 对正整数解析验证 ----
assert abs(gamma_lanczos(1.0) - 1.0) < 1e-10, '[TC01] gamma_lanczos(1.0) 应等于 1 FAILED'
assert abs(gamma_lanczos(2.0) - 1.0) < 1e-10, '[TC01] gamma_lanczos(2.0) 应等于 1 FAILED'
assert abs(gamma_lanczos(3.0) - 2.0) < 1e-10, '[TC01] gamma_lanczos(3.0) 应等于 2 FAILED'

# ---- TC02: lngamma_lanczos 非法输入返回错误码 ----
val, ier = lngamma_lanczos(-1.0)
assert ier == 1, '[TC02] lngamma_lanczos 对负数应返回错误码 1 FAILED'

# ---- TC03: factorial_ratio 数值稳定性 ----
fr = factorial_ratio(10, 5)
expected = 30240.0
assert abs(fr - expected) < 1e-6, '[TC03] factorial_ratio(10,5) 应等于 30240 FAILED'

# ---- TC04: associated_legendre_normalized 边界值为有限值 ----
p00 = associated_legendre_normalized(0, 0, 0.5)
assert np.isfinite(p00), '[TC04] P_00(0.5) 应为有限值 FAILED'

# ---- TC05: generate_asteroid_cross_section 输出形状正确 ----
np.random.seed(42)
poly = generate_asteroid_cross_section(n_points=32)
assert poly.shape == (32, 2), '[TC05] 二维截面应返回 (32,2) 形状 FAILED'

# ---- TC06: ear_clip_triangulation 输出面片数为 n-2 ----
tri = ear_clip_triangulation(poly)
assert tri.shape[0] == poly.shape[0] - 2, '[TC06] 三角面片数应为 n-2 FAILED'

# ---- TC07: polyhedron_volume_and_com 对单位立方体解析验证 ----
cube_v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
cube_f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=int)
vol, com = polyhedron_volume_and_com(cube_v, cube_f)
assert abs(vol - 1.0) < 1e-10, '[TC07] 单位立方体体积应为 1.0 FAILED'
assert np.linalg.norm(com - np.array([0.5, 0.5, 0.5])) < 1e-10, '[TC07] 单位立方体质心应为 (0.5,0.5,0.5) FAILED'

# ---- TC08: surface_area 对简单三角形解析验证 ----
tri_v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
tri_f = np.array([[0,1,2]], dtype=int)
area = surface_area(tri_v, tri_f)
assert abs(area - 0.5) < 1e-10, '[TC08] 直角三角形面积应为 0.5 FAILED'

# ---- TC09: revolve_to_3d 输出顶点与面片形状 ----
simple_poly = np.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=float)
v3d, f3d = revolve_to_3d(simple_poly, z_scale=1.0)
assert v3d.ndim == 2 and v3d.shape[1] == 3, '[TC09] 三维顶点应为 (N,3) FAILED'
assert f3d.ndim == 2 and f3d.shape[1] == 3, '[TC09] 面片应为 (M,3) FAILED'

# ---- TC10: compute_stokes_coefficients_from_shape 输出尺寸与参考半径为正 ----
tet_v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
tet_f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
c_coeff, s_coeff, r_ref = compute_stokes_coefficients_from_shape(tet_v, tet_f, 1000.0, n_max=4)
assert c_coeff.shape == (5, 5), '[TC10] C系数形状应为 (n_max+1, n_max+1) FAILED'
assert s_coeff.shape == (5, 5), '[TC10] S系数形状应为 (n_max+1, n_max+1) FAILED'
assert r_ref > 0, '[TC10] 参考半径应为正 FAILED'

# ---- TC11: SphericalHarmonicGravity.potential 返回有限值 ----
sh = SphericalHarmonicGravity(1.0, 1.0, c_coeff, s_coeff, n_max=4)
pot = sh.potential(np.array([2.0, 0.0, 0.0]))
assert np.isfinite(pot), '[TC11] 引力势应为有限值 FAILED'

# ---- TC12: SphericalHarmonicGravity.acceleration 输出为三维向量 ----
acc = sh.acceleration(np.array([2.0, 0.0, 0.0]))
assert acc.shape == (3,), '[TC12] 加速度应为 3 维向量 FAILED'

# ---- TC13: SphericalHarmonicGravity.gradient_fd 与 acceleration 数值一致 ----
acc_fd = sh.gradient_fd(np.array([2.0, 0.0, 0.0]))
assert np.linalg.norm(acc - acc_fd) < 0.1, '[TC13] 数值梯度与解析加速度应接近 FAILED'

# ---- TC14: polyhedron_gravity_potential 返回有限值 ----
tet_pot = polyhedron_gravity_potential(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
assert np.isfinite(tet_pot), '[TC14] 多面体引力势应为有限值 FAILED'

# ---- TC15: polyhedron_gravity_acceleration 输出为三维向量 ----
tet_acc = polyhedron_gravity_acceleration(np.array([2.0, 2.0, 2.0]), tet_v * 1000, tet_f, 2000.0)
assert tet_acc.shape == (3,), '[TC15] 多面体加速度应为 3 维向量 FAILED'

# ---- TC16: combined_gravity_model 输出为三维向量 ----
a_comb = combined_gravity_model(np.array([2.0, 0.0, 0.0]), tet_v, tet_f, 1.0, 1.0, c_coeff, s_coeff, n_max=4, density=2000.0)
assert a_comb.shape == (3,), '[TC16] 组合引力加速度应为 3 维向量 FAILED'

# ---- TC17: solve_internal_potential_2d 势场全为有限值 ----
phi, xc, yc = solve_internal_potential_2d(nx=8, ny=8, r_max=100.0)
assert np.all(np.isfinite(phi)), '[TC17] 内部引力势应全为有限值 FAILED'
assert phi.shape == (8, 8), '[TC17] 势场形状应为 (ny,nx) FAILED'

# ---- TC18: internal_gravity_from_potential 输出尺寸与势场一致 ----
gx, gy = internal_gravity_from_potential(phi, xc, yc)
assert gx.shape == phi.shape, '[TC18] gx 形状应与 phi 相同 FAILED'
assert gy.shape == phi.shape, '[TC18] gy 形状应与 phi 相同 FAILED'

# ---- TC19: newton_cotes_integrate 解析验证 sin^2 ----
nc_res = newton_cotes_integrate(lambda t: np.sin(t) ** 2, 0.0, np.pi, n=5, n_sub=10)
assert abs(nc_res - np.pi / 2.0) < 1e-6, '[TC19] Newton-Cotes 积分 sin^2 应接近 pi/2 FAILED'

# ---- TC20: OrbitalDynamics 确定性积分输出形状正确 ----
def grav_dummy(pos):
    return -pos / (np.linalg.norm(pos) ** 3)
dyn = OrbitalDynamics(grav_dummy, beta_srp=0.0, perturbation_std=0.0)
state0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
t_arr, states = dyn.integrate_deterministic(state0, (0.0, 1.0), n_steps=10)
assert states.shape == (11, 6), '[TC20] 确定性积分状态矩阵应为 (n_steps+1, 6) FAILED'

# ---- TC21: OrbitSensitivityAnalysis 设计矩阵形状正确 ----
def mock_obj(params):
    return -np.sum(params ** 2)
analy = OrbitSensitivityAnalysis(['a','e','i'], np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0]]), mock_obj)
design, responses, main_effects = analy.run_analysis()
assert design.shape[1] == 3, '[TC21] 设计矩阵列数应为 3 FAILED'
assert responses.shape[0] == design.shape[0], '[TC21] 响应值数量应与设计矩阵行数相同 FAILED'

# ---- TC22: optimize_orbit_binary_backtrack 找到全局最优解 ----
best_bits, best_score = optimize_orbit_binary_backtrack(3, lambda bits: float(np.sum(bits)))
assert best_bits.shape == (3,), '[TC22] 最优二进制向量长度应为 3 FAILED'
assert best_score == 3.0, '[TC22] 最大化 sum(bits) 应得到 3 FAILED'

# ---- TC23: compute_orbit_quality_score 权重单调性验证 ----
def lt(p): return 1.0
def dv(p): return 0.0
def coll(p): return 0.0
score1 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([1.0, -1.0, -1.0]))
score2 = compute_orbit_quality_score(np.array([1.0, 0.1, 0.1]), lt, dv, coll, weights=np.array([2.0, -1.0, -1.0]))
assert score2 > score1, '[TC23] 增大 lifetime 权重应提高评分 FAILED'

# ---- TC24: ball_distance_stats 均值在合理范围且方差非负 ----
np.random.seed(42)
mu, var = ball_distance_stats(n_samples=2000, seed=42)
assert 0.9 < mu < 1.2, '[TC24] 单位球距离均值应在合理范围内 FAILED'
assert var >= 0, '[TC24] 方差应非负 FAILED'

# ---- TC25: collision_probability_surface 范围约束在 [0,1] ----
p_coll = collision_probability_surface(np.array([10.0, 10.0, 10.0]), tet_v, tet_f, safe_distance=0.5, position_uncertainty=0.1)
assert 0.0 <= p_coll <= 1.0, '[TC25] 碰撞概率应在 [0,1] 范围内 FAILED'

# ---- TC26: region_connectivity_analysis 对全连通图返回 1 个分量 ----
adj = np.ones((5, 5), dtype=int) - np.eye(5, dtype=int)
conn = region_connectivity_analysis(adj)
assert conn['n_components'] == 1, '[TC26] 全连通图应只有 1 个连通分量 FAILED'
assert conn['total_nodes'] == 5, '[TC26] 总节点数应为 5 FAILED'

# ---- TC27: s_word_count 字符串单词计数正确 ----
wc = s_word_count("v 1.0 2.0 3.0")
assert wc == 4, '[TC27] 字符串单词数应为 4 FAILED'

# ---- TC28: generate_synthetic_asteroid_pointcloud 输出形状正确 ----
np.random.seed(42)
pc = generate_synthetic_asteroid_pointcloud(n_theta=8, n_phi=8)
assert pc.shape == (64, 3), '[TC28] 合成点云形状应为 (64,3) FAILED'
