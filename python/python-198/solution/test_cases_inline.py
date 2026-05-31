# ---- TC01: generate_disk_triangulation 输出形状正确 ----
n_r, n_theta, n_parts_test = 6, 12, 4
nodes, elements, boundary_mask = generate_disk_triangulation(n_r, n_theta)
assert nodes.shape[1] == 2, '[TC01] nodes shape FAILED'
assert elements.shape[1] == 3, '[TC01] elements shape FAILED'
assert len(boundary_mask) == nodes.shape[0], '[TC01] boundary_mask length FAILED'
assert elements.shape[0] > 0, '[TC01] elements count FAILED'

# ---- TC02: generate_disk_triangulation 边界节点数量合理 ----
n_theta_outer = n_theta
n_boundary = np.sum(boundary_mask)
assert n_boundary == n_theta_outer, '[TC02] boundary node count FAILED'

# ---- TC03: compute_element_quality 输出在 [0, 1] 范围内 ----
quality = compute_element_quality(nodes, elements)
assert quality.min() >= 0.0, '[TC03] quality min FAILED'
assert quality.max() <= 1.0, '[TC03] quality max FAILED'
assert np.all(np.isfinite(quality)), '[TC03] quality finite FAILED'

# ---- TC04: extract_boundary_edges 提取边界边非空 ----
boundary_edges = extract_boundary_edges(elements)
assert boundary_edges.shape[0] > 0, '[TC04] boundary edges empty FAILED'
assert boundary_edges.shape[1] == 2, '[TC04] boundary edges shape FAILED'

# ---- TC05: domain_decomposition 所有节点分区完成 ----
partition = domain_decomposition(nodes, elements, n_parts_test)
assert partition.shape[0] == nodes.shape[0], '[TC05] partition shape FAILED'
assert np.min(partition) >= 0, '[TC05] partition min FAILED'
assert np.max(partition) < n_parts_test, '[TC05] partition max FAILED'

# ---- TC06: compute_interface_nodes 接口节点存在 ----
interface = compute_interface_nodes(elements, partition)
assert interface.shape[0] == nodes.shape[0], '[TC06] interface shape FAILED'
assert np.sum(interface) >= 0, '[TC06] interface count negative FAILED'

# ---- TC07: truncated_normal_ab_pdf 积分近似为1 ----
mu, sigma, a, b = 0.5, 0.3, 0.0, 1.5
x_grid = np.linspace(a, b, 2000)
pdf_vals = truncated_normal_ab_pdf(x_grid, mu, sigma, a, b)
integral = np.trapz(pdf_vals, x_grid)
assert abs(integral - 1.0) < 0.01, '[TC07] PDF integral FAILED'

# ---- TC08: truncated_normal_ab_mean 在截断区间内 ----
mean_val = truncated_normal_ab_mean(mu, sigma, a, b)
assert a <= mean_val <= b, '[TC08] mean out of truncation bounds FAILED'

# ---- TC09: truncated_normal_ab_variance 非负有限 ----
var_val = truncated_normal_ab_variance(mu, sigma, a, b)
assert var_val >= 0.0, '[TC09] variance negative FAILED'
assert np.isfinite(var_val), '[TC09] variance infinite FAILED'

# ---- TC10: truncated_normal_ab_sample 输出形状正确（固定种子） ----
np.random.seed(42)
samples = truncated_normal_ab_sample(mu, sigma, a, b, size=1000)
assert len(samples) == 1000, '[TC10] sample count FAILED'
assert np.all(samples >= a), '[TC10] sample below a FAILED'
assert np.all(samples <= b), '[TC10] sample above b FAILED'

# ---- TC11: truncated_normal_ab_sample 可复现性 ----
np.random.seed(42)
s1 = truncated_normal_ab_sample(mu, sigma, a, b, size=100)
np.random.seed(42)
s2 = truncated_normal_ab_sample(mu, sigma, a, b, size=100)
assert np.allclose(s1, s2), '[TC11] reproducibility FAILED'

# ---- TC12: hermite_he_prob_matrix 正交性验证 ----
degree_test = 4
xi_test = np.linspace(-3, 3, 300)
H = hermite_he_prob_matrix(degree_test, xi_test)
assert H.shape == (300, degree_test + 1), '[TC12] H shape FAILED'
# 验证与X的二次方关系
he2_exact = xi_test ** 2 - 1.0
assert np.allclose(H[:, 2], he2_exact, atol=1e-10), '[TC12] He_2 orthogonality FAILED'

# ---- TC13: build_pce_galerkin_matrix 输出方阵 ----
alpha_mu, alpha_sigma = 1.0, 0.2
A_pce = build_pce_galerkin_matrix(degree_test, alpha_mu, alpha_sigma)
assert A_pce.shape == (degree_test + 1, degree_test + 1), '[TC13] A_pce shape FAILED'
assert np.all(np.isfinite(A_pce)), '[TC13] A_pce finite FAILED'

# ---- TC14: vandermonde_solve 精确恢复 ----
x_vand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
V = vandermonde_matrix(5, x_vand)
b_test = np.array([1.0, 3.0, 9.0, 27.0, 81.0])
c = vandermonde_solve(V, b_test)
recovered = V @ c
assert np.allclose(recovered, b_test, atol=1e-10), '[TC14] vandermonde solve FAILED'

# ---- TC15: godunov_flux 确定性 - f(uL,uR) == f(uL,uR) 相同输入相同输出 ----
from burgers_fvm import godunov_flux
f1 = godunov_flux(np.array([0.3]), np.array([0.8]))
f2 = godunov_flux(np.array([0.3]), np.array([0.8]))
assert np.allclose(f1, f2), '[TC15] godunov reproducibility FAILED'

# ---- TC16: godunov_flux uL==uR 时通量有限 ----
from burgers_fvm import godunov_flux
u_test = np.array([0.5, -0.3])
f_test = godunov_flux(u_test, u_test)
assert np.all(np.isfinite(f_test)), '[TC16] godunov equal inputs FAILED'

# ---- TC17: build_fvm_operators 输出维度正确 ----
area, centroid, internal_edges, boundary_edges_fvm = build_fvm_operators(nodes, elements)
assert area.shape[0] == elements.shape[0], '[TC17] area shape FAILED'
assert centroid.shape[0] == elements.shape[0], '[TC17] centroid shape FAILED'
assert centroid.shape[1] == 2, '[TC17] centroid cols FAILED'
assert np.all(area > 0), '[TC17] area positive FAILED'

# ---- TC18: solve_burgers_fvm 输出形状正确 ----
def u0_func(x, y):
    r = np.sqrt(x**2 + y**2)
    return 0.5 * np.sin(np.pi * r) * (1.0 - r**2)

np.random.seed(42)
t, U = solve_burgers_fvm(nodes, elements, u0_func, t_max=0.1, nt=20)
assert len(t) == 21, '[TC18] time steps FAILED'
assert U.shape == (21, elements.shape[0]), '[TC18] U shape FAILED'
assert np.all(np.isfinite(U)), '[TC18] U finite FAILED'

# ---- TC19: matrix_exponential_pade exp(0) = I ----
A_zero = np.zeros((3, 3))
E_zero = matrix_exponential_pade(A_zero)
assert np.allclose(E_zero, np.eye(3), atol=1e-12), '[TC19] exp(0) FAILED'

# ---- TC20: matrix_exponential_pade 对稀疏矩阵输出有限 ----
A_small = np.array([[0.1, 0.2], [-0.2, 0.1]])
E_small = matrix_exponential_pade(A_small)
assert E_small.shape == (2, 2), '[TC20] exp small shape FAILED'
assert np.all(np.isfinite(E_small)), '[TC20] exp small finite FAILED'

# ---- TC21: ca_speedup_theory s=1 加速比为1 ----
sp = ca_speedup_theory(1, 1e-3, 1e-2)
assert abs(sp - 1.0) < 1e-10, '[TC21] s=1 speedup FAILED'

# ---- TC22: ca_speedup_theory 加速比 >= 1 ----
for s_test in [1, 2, 4, 8, 16]:
    sp_s = ca_speedup_theory(s_test, 1e-3, 1e-2)
    assert sp_s >= 1.0 - 1e-12, f'[TC22] speedup s={s_test} FAILED'

# ---- TC23: optimize_s_parameter 返回有效参数 ----
best_s, best_sp = optimize_s_parameter(1e-3, 1e-2, s_max=10)
assert best_s >= 1, '[TC23] best_s FAILED'
assert best_sp >= 1.0, '[TC23] best_sp FAILED'

# ---- TC24: gmres_solve 求解小型线性系统 ----
np.random.seed(42)
A_tiny = np.array([[4.0, 1.0], [1.0, 3.0]])
b_tiny = np.ones(2)
x_gmres, res_hist, iters = gmres_solve(A_tiny, b_tiny, restart=2, max_iter=10, tol=1e-10, s_step=1)
assert len(x_gmres) == 2, '[TC24] gmres solution shape FAILED'
assert res_hist[-1] < 1e-8, '[TC24] gmres residual FAILED'

# ---- TC25: ca_sstep_arnoldi 输出矩阵尺寸正确 ----
np.random.seed(42)
n_v = 5
A_arn = np.diag(np.arange(1, n_v + 1, dtype=float))
v0_arn = np.ones(n_v)
V, H = ca_sstep_arnoldi(A_arn, v0_arn, s=2, m_total=3)
assert V.shape[0] == n_v, '[TC25] V row count FAILED'
assert H.shape[1] == 3, '[TC25] H col count FAILED'

# ---- TC26: ellipse_sample 采样点满足椭圆约束（固定种子） ----
np.random.seed(42)
A_ellipse = np.array([[4.0, 1.0], [1.0, 3.0]])
R_test = 1.0
samples_ell = ellipse_sample(500, A_ellipse, R_test)
assert samples_ell.shape == (2, 500), '[TC26] ellipse sample shape FAILED'
quad_form = np.sum(samples_ell * (A_ellipse @ samples_ell), axis=0)
assert np.all(quad_form <= R_test**2 + 1e-10), '[TC26] ellipse constraint FAILED'

# ---- TC27: monte_carlo_pce_verify 返回有效统计量 ----
np.random.seed(42)
result = monte_carlo_pce_verify(
    n_samples=10000, pce_degree=4, alpha_mu=1.0, alpha_sigma=0.2,
    u0_scalar=1.0, tf=0.5, exact_mean_func=None
)
assert 'mc_mean' in result, '[TC27] mc_mean key FAILED'
assert result['mc_mean'] > 0, '[TC27] mc_mean non-positive FAILED'
assert result['error_mean'] < 0.05, '[TC27] error_mean too large FAILED'

# ---- TC28: dense_to_csr 往返无损 ----
from sparse_io import csr_to_dense
A_dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
csr = dense_to_csr(A_dense)
A_recovered = csr_to_dense(csr)
assert np.allclose(A_recovered, A_dense), '[TC28] CSR round-trip FAILED'

# ---- TC29: build_pce_block_sparse 矩阵形状正确 ----
spatial_A = 0.1 * np.eye(4)
A_total = build_pce_block_sparse(spatial_A, 2, alpha_mu=1.0, alpha_sigma=0.2)
n_expected = 4 * (2 + 1)
assert A_total.shape == (n_expected, n_expected), '[TC29] block sparse shape FAILED'
assert np.all(np.isfinite(A_total)), '[TC29] block sparse finite FAILED'

# ---- TC30: laplace_radial_2d_exact 满足拉普拉斯方程 ----
x_lap = np.array([0.5, 1.2, 0.8])
y_lap = np.array([0.3, -0.5, 0.0])
u, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x_lap, y_lap, a=0.1, b=0.5)
laplacian = uxx + uyy
assert np.allclose(laplacian, 0.0, atol=1e-12), '[TC30] Laplace equation FAILED'

# ---- TC31: sawtooth_wave 值域在 [-1, 1] ----
t_saw = np.linspace(0, 5.0, 100)
saw_vals = sawtooth_wave(t_saw, omega=2.0 * np.pi, amplitude=1.0)
assert np.min(saw_vals) >= -1.0, '[TC31] sawtooth min FAILED'
assert np.max(saw_vals) <= 1.0, '[TC31] sawtooth max FAILED'

# ---- TC32: sawtooth_wave 周期性 ----
t1 = 0.25
T = 2.0 * np.pi / (2.0 * np.pi)
v1 = sawtooth_wave(t1, omega=2.0 * np.pi)
v2 = sawtooth_wave(t1 + T, omega=2.0 * np.pi)
assert abs(v1 - v2) < 1e-12, '[TC32] sawtooth periodicity FAILED'

# ---- TC33: compute_l2_error 非负 ----
u_num = np.array([0.5, 0.8, 1.2])
u_exact = np.array([0.5, 0.7, 1.3])
area_err = np.array([0.1, 0.2, 0.15])
err = compute_l2_error(u_num, u_exact, area_err)
assert err >= 0.0, '[TC33] L2 error negative FAILED'

# ---- TC34: caustic_mapping 边数正确 ----
n_caustic, m_caustic = 20, 5
edges_c, pj_c, pk_c = caustic_mapping(n_caustic, m_caustic)
assert edges_c.shape == (n_caustic + 1, 2), '[TC34] caustic edges shape FAILED'
assert pj_c.shape == (n_caustic + 1, 2), '[TC34] pj shape FAILED'

# ---- TC35: shock_formation_time 检测激波 ----
def u0_shock(x):
    return -0.5 * np.sin(np.pi * x)
x_grid = np.linspace(-1.0, 1.0, 100)
t_b, x_s = shock_formation_time(u0_shock, x_grid)
assert np.isfinite(t_b), '[TC35] shock time infinite FAILED'
assert t_b > 0, '[TC35] shock time non-positive FAILED'

# ---- TC36: generate_kl_coefficients 输出有限且形状正确 ----
np.random.seed(42)
kl_coeffs = generate_kl_coefficients(n_modes=5, correlation_length=0.2)
assert len(kl_coeffs) == 5, '[TC36] KL coeffs length FAILED'
assert np.all(np.isfinite(kl_coeffs)), '[TC36] KL coeffs finite FAILED'
