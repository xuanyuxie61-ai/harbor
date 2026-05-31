# ---- TC01: sigmoid(0) 精确等于 0.5 ----
result = sigmoid(0.0)
assert abs(result - 0.5) < 1e-12, '[TC01] sigmoid(0) 应精确等于 0.5 FAILED'

# ---- TC02: sigmoid 极大值数值稳定性 ----
result = sigmoid(1000.0)
assert np.isfinite(result) and result > 0.999999, '[TC02] sigmoid 极大值应趋近于 1 且不溢出 FAILED'

# ---- TC03: sigmoid 对称性验证 s(-x) = 1 - s(x) ----
x_test = np.array([-2.0, -1.0, 0.5, 3.0])
left = sigmoid(-x_test)
right = 1.0 - sigmoid(x_test)
assert np.allclose(left, right, atol=1e-12), '[TC03] sigmoid 对称性 s(-x)=1-s(x) FAILED'

# ---- TC04: Beverton-Holt S=0 时补充量为 0 ----
R0 = beverton_holt(0.0, 2.0, 0.3)
assert abs(R0) < 1e-12, '[TC04] Beverton-Holt 在 S=0 时补充量应为 0 FAILED'

# ---- TC05: Beverton-Holt 大 S 渐近值趋近 alpha/beta ----
R_large = beverton_holt(1e6, 2.0, 0.3)
assert abs(R_large - 2.0 / 0.3) < 1e-4, '[TC05] Beverton-Holt 大 S 渐近值应为 alpha/beta FAILED'

# ---- TC06: Ricker 在 S=0 时补充量为 0 ----
R0_ricker = ricker_recruitment(0.0, 2.0, 0.3)
assert abs(R0_ricker) < 1e-12, '[TC06] Ricker 在 S=0 时补充量应为 0 FAILED'

# ---- TC07: Ricker 最优亲体量处取最大值解析验证 ----
alpha, beta = 2.0, 0.3
S_opt = 1.0 / beta
R_max = ricker_recruitment(S_opt, alpha, beta)
expected_max = alpha / (np.e * beta)
assert abs(R_max - expected_max) < 1e-10, '[TC07] Ricker 最优亲体量处最大值解析验证 FAILED'

# ---- TC08: Sigmoid-Allee 补充量非负性 ----
S_vals = np.linspace(0.0, 10.0, 11)
R_allee = sigmoid_allee_recruitment(S_vals, 2.0, 0.3, 1.5, 8.0)
assert np.all(R_allee >= 0.0), '[TC08] Sigmoid-Allee 补充量应始终非负 FAILED'

# ---- TC09: recruitment_derivative bh 模式解析验证 ----
S_test = 3.0
dR = recruitment_derivative(S_test, 2.0, 0.3, 1.5, 8.0, model_type='bh')
expected_bh = 2.0 / ((1.0 + 0.3 * S_test) ** 2)
assert abs(dR - expected_bh) < 1e-10, '[TC09] recruitment_derivative bh 模式解析验证 FAILED'

# ---- TC10: Vandermonde 插值在节点处精确回代 ----
nodes = np.array([0.1, 0.25, 0.4, 0.6, 0.85])
values = 10.0 + 20.0 * nodes - 15.0 * (nodes ** 2) + 5.0 * (nodes ** 3)
interp_at_nodes = vandermonde_interp_1d(nodes, values, nodes)
max_err = np.max(np.abs(interp_at_nodes - values))
assert max_err < 1e-10, '[TC10] Vandermonde 插值节点回代误差 FAILED'

# ---- TC11: pvand 求解 Vandermonde 系统验证 ----
n_v = 5
alpha_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
c_true = np.array([5.0, 0.0, 5.0, 0.0, 0.0])
V = np.vander(alpha_v, N=n_v, increasing=True)
b_v = V.T @ c_true
x_sol = pvand(n_v, alpha_v, b_v)
residual = np.linalg.norm(V.T @ x_sol - b_v)
assert residual < 1e-10, '[TC11] pvand 求解 Vandermonde 系统残差 FAILED'

# ---- TC12: bidim_vandermonde_solve 输出尺寸与输入一致 ----
alpha_b = np.array([1.0, 2.0])
beta_b = np.array([3.0, 4.0])
b_b = np.array([1.0, 2.0, 3.0, 4.0])
x_b = bidim_vandermonde_solve(2, alpha_b, beta_b, b_b)
assert len(x_b) == 4, '[TC12] bidim_vandermonde_solve 输出长度应为 n*n FAILED'

# ---- TC13: sphere_stereograph_inverse 投影点 z 坐标为 1 ----
pts_plane = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])
inv_pts = sphere_stereograph_inverse(pts_plane)
norm_inv = np.linalg.norm(inv_pts, axis=1)
assert np.allclose(norm_inv, 1.0, atol=1e-10), '[TC13] 逆投影点应位于单位球面 FAILED'

# ---- TC14: divergence_free_velocity 输出为有限值 ----
X_test = np.array([0.2, 0.5, 0.8])
Y_test = np.array([0.3, 0.6, 0.9])
U_test, V_test = divergence_free_velocity(3, X_test, Y_test, 1.0)
assert np.all(np.isfinite(U_test)) and np.all(np.isfinite(V_test)), '[TC14] 无散度速度场输出应全为有限值 FAILED'

# ---- TC15: integrate_cube_domain 常数函数精确积分 ----
cube_int = integrate_cube_domain(lambda pts: 2.0, degree=3, scale=1.0)
# 立方体 [-1,1]^3 体积为 8, 常数 2 积分应为 16
assert abs(cube_int - 16.0) < 1e-10, '[TC15] 立方体域常数函数积分应精确 FAILED'

# ---- TC16: integrate_pyramid_domain 常数函数非负 ----
pyr_int = integrate_pyramid_domain(lambda pts: 1.0, degree=3)
assert pyr_int > 0.0, '[TC16] 金字塔域常数正函数积分应大于 0 FAILED'

# ---- TC17: 二十面体顶点数为 12 且位于单位球面 ----
vertices = icosahedron_vertices()
norm_v = np.linalg.norm(vertices, axis=1)
assert len(vertices) == 12, '[TC17] 二十面体顶点数应为 12 FAILED'
assert np.allclose(norm_v, 1.0, atol=1e-10), '[TC17] 二十面体顶点应位于单位球面 FAILED'

# ---- TC18: 球面立体投影正反变换可逆 ----
proj = sphere_stereograph(vertices)
recovered = sphere_stereograph_inverse(proj)
recon_err = np.max(np.linalg.norm(vertices - recovered, axis=1))
assert recon_err < 1e-10, '[TC18] 球面立体投影正反变换可逆性 FAILED'

# ---- TC19: Schaefer-Gordon 稳态 E=0 时生物量等于承载力 ----
B0 = schaefer_gordon_steady_state(0.0, 0.4, 1000.0, 0.01)
assert abs(B0 - 1000.0) < 1e-10, '[TC19] Schaefer-Gordon E=0 时 B 应等于 K FAILED'

# ---- TC20: Schaefer-Gordon 过度捕捞时生物量为 0 ----
B_over = schaefer_gordon_steady_state(100.0, 0.4, 1000.0, 0.01)
assert abs(B_over) < 1e-10, '[TC20] Schaefer-Gordon 过度捕捞时 B 应为 0 FAILED'

# ---- TC21: 最短路径重构包含源点 ----
connectivity = np.array([
    [0.0, 2.5, 4.0, np.inf, 3.0],
    [2.5, 0.0, 1.5, 3.5, np.inf],
    [4.0, 1.5, 0.0, 2.0, 4.5],
    [np.inf, 3.5, 2.0, 0.0, 1.8],
    [3.0, np.inf, 4.5, 1.8, 0.0]
], dtype=float)
dist_path, predecessor = mpa_network_optimize(5, connectivity, source_patch=0)
path_to_0 = reconstruct_path(predecessor, 0)
assert path_to_0[0] == 0, '[TC21] 最短路径重构应包含源点 FAILED'

# ---- TC22: Allen-Cahn 能量泛函非负性 ----
x_ac = np.linspace(0.0, 10.0, 101)
u_ac = np.tanh((x_ac - 3.0) / (np.sqrt(2.0) * 0.5))
energy = energy_functional(u_ac, 10.0 / 100.0, 0.1, 0.5)
assert energy >= 0.0, '[TC22] Allen-Cahn 能量泛函应非负 FAILED'

# ---- TC23: fishery_forcing 输出尺寸与输入一致 ----
u_forcing = np.array([0.5, -0.3, 1.0, -1.0])
force = fishery_forcing(0.0, u_forcing, 25.0, 0.3, 0.01, 1000.0)
assert len(force) == len(u_forcing), '[TC23] fishery_forcing 输出尺寸应与输入一致 FAILED'

# ---- TC24: 一维蒙特卡洛积分常数函数精确验证 ----
np.random.seed(42)
mc_result = monte_carlo_integral_1d(lambda x: 3.0, 0.0, 1.0, 100, method='mc')
assert abs(mc_result - 3.0) < 1e-10, '[TC24] MC 积分常数函数应精确 FAILED'

# ---- TC25: Hammersley 序列输出尺寸正确 ----
pts = hammersley_sequence(0, 4, m=3, n=1000)
assert pts.shape == (5, 3), '[TC25] Hammersley 序列输出尺寸应为 (5, 3) FAILED'

# ---- TC26: 线积分常数函数精确验证 ----
line_int = integrate_line_profile(lambda z: 5.0, 0.0, 10.0, order=3)
assert abs(line_int - 50.0) < 1e-10, '[TC26] 线积分常数函数应精确 FAILED'

# ---- TC27: 栖息地聚类输出标签在合理范围内 ----
np.random.seed(42)
n_stations = 40
env_data = np.random.randn(n_stations, 4)
zones, _, zone_stats = cluster_habitat_zones(n_stations, env_data, n_zones=3)
assert np.all((zones >= 0) & (zones < 3)), '[TC27] 聚类标签应在 [0, n_zones) 范围内 FAILED'
assert zone_stats['inertia'] >= 0.0, '[TC27] 聚类惯性应非负 FAILED'
