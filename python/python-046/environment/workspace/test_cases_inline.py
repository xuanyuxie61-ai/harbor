# ---- TC01: compute_triangle_area 直角三角形面积 ----
from utils import compute_triangle_area
area = compute_triangle_area(np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.0, 4.0]))
assert abs(area - 6.0) < 1e-10, '[TC01] compute_triangle_area 直角三角形面积 FAILED'

# ---- TC02: normalize_vector 返回单位长度 ----
v = np.array([3.0, 4.0])
u = normalize_vector(v)
assert abs(np.linalg.norm(u) - 1.0) < 1e-10, '[TC02] normalize_vector 返回单位长度 FAILED'

# ---- TC03: clip_to_range 边界约束 ----
from utils import clip_to_range
x = clip_to_range(5.0, 0.0, 1.0)
assert x == 1.0, '[TC03] clip_to_range 边界约束 FAILED'

# ---- TC04: rotation_matrix_3d 正交性验证 ----
from utils import rotation_matrix_3d
R = rotation_matrix_3d(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
assert np.allclose(R @ R.T, np.eye(3)), '[TC04] rotation_matrix_3d 正交性验证 FAILED'

# ---- TC05: FaultMesh 规则网格节点数量 ----
fm = FaultMesh(length=10.0, width=5.0, strike_deg=0.0, dip_deg=90.0, num_strike=2, num_dip=2, adaptivity=False)
assert fm.num_nodes == 9, '[TC05] FaultMesh 规则网格节点数量 FAILED'

# ---- TC06: FaultMesh 元素面积总和等于断层面积 ----
fm = FaultMesh(length=10.0, width=5.0, strike_deg=0.0, dip_deg=90.0, num_strike=2, num_dip=2, adaptivity=False)
total_area = np.sum(fm.element_areas())
assert abs(total_area - 50.0) < 1e-6, '[TC06] FaultMesh 元素面积总和 FAILED'

# ---- TC07: SurfaceGrid 网格点数 ----
sg = SurfaceGrid(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=3, ny=4)
assert sg.num_points == 12, '[TC07] SurfaceGrid 网格点数 FAILED'

# ---- TC08: composite_trapezoidal 积分 4/(1+x^2) 近似 pi ----
result = composite_trapezoidal(lambda x: 4.0 / (1.0 + x ** 2), 0.0, 1.0, 10001)
assert abs(result - np.pi) < 1e-8, '[TC08] composite_trapezoidal 积分 FAILED'

# ---- TC09: gauss_legendre_integral 指数函数精确积分 ----
result = gauss_legendre_integral(lambda x: np.exp(x), -1.0, 1.0, n=10)
expected = np.exp(1.0) - np.exp(-1.0)
assert abs(result - expected) < 1e-12, '[TC09] gauss_legendre_integral 指数函数 FAILED'

# ---- TC10: integrate_over_triangle 线性函数 ----
p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
result = integrate_over_triangle(lambda x, y: x + y, p1, p2, p3, order=7)
assert abs(result - 1.0 / 3.0) < 1e-10, '[TC10] integrate_over_triangle 线性函数 FAILED'

# ---- TC11: legendre_polynomial_values P0和P1验证 ----
x = np.array([-0.5, 0.0, 0.5])
P = legendre_polynomial_values(len(x), 1, x)
assert np.allclose(P[:, 0], np.ones(len(x))), '[TC11] legendre_polynomial_values P0 FAILED'
assert np.allclose(P[:, 1], x), '[TC11] legendre_polynomial_values P1 FAILED'

# ---- TC12: hermite_probabilist_values_array He0和He2 ----
x = np.array([0.0, 1.0])
H = hermite_probabilist_values_array(2, x)
assert np.allclose(H[:, 0], np.ones(2)), '[TC12] hermite_probabilist_values_array He0 FAILED'
assert abs(H[0, 2] - (-1.0)) < 1e-10, '[TC12] hermite_probabilist_values_array He2(0) FAILED'

# ---- TC13: FEM1DBasis interpolate_1d 线性插值 ----
node_x = np.array([0.0, 0.5, 1.0])
node_v = np.array([1.0, 2.0, 1.0])
eval_x = np.array([0.25, 0.75])
interp = FEM1DBasis.interpolate_1d(node_x, node_v, eval_x)
assert abs(interp[0] - 1.75) < 1e-10, '[TC13] FEM1DBasis interpolate_1d 0.25 FAILED'
assert abs(interp[1] - 1.75) < 1e-10, '[TC13] FEM1DBasis interpolate_1d 0.75 FAILED'

# ---- TC14: FEMElasticity2D stiffness matrix symmetry ----
fe_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
fe_elements = np.array([[0, 1, 2], [1, 3, 2]])
fem = FEMElasticity2D(fe_nodes, fe_elements, E=200e9, nu=0.3)
K = fem.assemble_stiffness_matrix(use_sparse=False)
assert np.allclose(K, K.T), '[TC14] FEMElasticity2D stiffness symmetry FAILED'

# ---- TC15: InSARForwardModel LOS projection scalar ----
insar = InSARForwardModel(los_vector=np.array([1.0, 0.0, 0.0]), wavelength=0.056)
disp = np.array([2.0, 3.0, 4.0])
los = insar.project_to_los(disp)
assert abs(los - 2.0) < 1e-10, '[TC15] InSARForwardModel LOS projection FAILED'

# ---- TC16: ElasticHalfspacePoisson1D FD error bound ----
H = 20e3
mu = 30e9
solver = ElasticHalfspacePoisson1D(H, mu, nx=101)
z_fd, u_fd = solver.solve(lambda z: np.sin(np.pi * z / H), u_bottom=0.0)
u_exact = (H / np.pi) ** 2 * np.sin(np.pi * z_fd / H) / mu
error = np.max(np.abs(u_fd - u_exact))
assert error < 1.0, '[TC16] ElasticHalfspacePoisson1D FD error FAILED'

# ---- TC17: RateStateFriction steady_state theta formula ----
rs = RateStateFriction(a=0.015, b=0.020, Dc=0.01, sigma_n=100e6, mu0=0.6, V0=1e-6, k=1e9, V_pl=1e-9, radiation_damping=False)
V_ss = 1e-9
theta_ss, tau_ss = rs.steady_state_solution(V_ss)
assert abs(theta_ss - rs.Dc / V_ss) < 1e-10, '[TC17] RateStateFriction theta_ss FAILED'

# ---- TC18: RateStateFriction dstate_dt vanishes at steady state ----
dtheta = rs.dstate_dt(V_ss, theta_ss)
assert abs(dtheta) < 1e-10, '[TC18] RateStateFriction dstate_dt steady FAILED'

# ---- TC19: build_laplacian_1d shape ----
L1d = build_laplacian_1d(5, h=1.0)
assert L1d.shape == (3, 5), '[TC19] build_laplacian_1d shape FAILED'

# ---- TC20: build_laplacian_2d shape ----
L2d = build_laplacian_2d(3, 4, hx=1.0, hy=1.0)
assert L2d.shape == (12, 12), '[TC20] build_laplacian_2d shape FAILED'

# ---- TC21: tikhonov_solve identity system ----
from regularization import tikhonov_solve
G = np.eye(3)
W = np.eye(3)
d = np.array([1.0, 2.0, 3.0])
L = np.eye(3)
m, cov = tikhonov_solve(G, W, d, 0.01, L)
assert np.allclose(m, d, atol=0.01), '[TC21] tikhonov_solve identity FAILED'

# ---- TC22: CCSMatrix multiply_vector matches dense ----
A_dense = np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]])
A_ccs = CCSMatrix.from_dense(A_dense)
x = np.array([1.0, 2.0, 3.0])
y_ccs = A_ccs.multiply_vector(x)
y_dense = A_dense @ x
assert np.allclose(y_ccs, y_dense), '[TC22] CCSMatrix multiply_vector FAILED'

# ---- TC23: matrix_chain_optimal_order known result ----
dims = [10, 20, 5, 30, 8]
min_cost, split = matrix_chain_optimal_order(dims)
assert min_cost == 2600, '[TC23] matrix_chain_optimal_order FAILED'

# ---- TC24: NelderMeadOptimizer quadratic minimum ----
from inversion_core import NelderMeadOptimizer
quad_obj = lambda x: np.sum((x - np.array([1.0, -2.0])) ** 2)
nm = NelderMeadOptimizer(tol=1e-8, max_iter=500)
x_opt, f_opt, n_eval = nm.optimize(quad_obj, np.array([0.0, 0.0]))
assert np.allclose(x_opt, np.array([1.0, -2.0]), atol=1e-4), '[TC24] NelderMeadOptimizer quadratic FAILED'

# ---- TC25: FaultSlipInversion compute_misfit zero residual ----
G = np.eye(5)
W = np.eye(5)
d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
inv = FaultSlipInversion(G, W, d, 0.1)
misfit = inv.compute_misfit(d)
assert abs(misfit) < 1e-10, '[TC25] FaultSlipInversion zero misfit FAILED'

# ---- TC26: mixed_legendre_hermite_basis_2d output shape ----
x = np.array([0.0, 0.5])
y = np.array([1.0, 2.0])
B = mixed_legendre_hermite_basis_2d(x, y, n_leg=2, n_herm=2)
assert B.shape == (2, 9), '[TC26] mixed_legendre_hermite_basis_2d shape FAILED'

# ---- TC27: InSARForwardModel add_noise reproducibility ----
np.random.seed(42)
insar = InSARForwardModel()
d = np.zeros(10)
d_noisy1 = insar.add_noise(d, sigma=0.01, atmospheric=False)
np.random.seed(42)
d_noisy2 = insar.add_noise(d, sigma=0.01, atmospheric=False)
assert np.allclose(d_noisy1, d_noisy2), '[TC27] add_noise reproducibility FAILED'

# ---- TC28: wrap_to_pi periodic wrap ----
from utils import wrap_to_pi
angle = wrap_to_pi(3.5 * np.pi)
assert abs(angle - (-0.5 * np.pi)) < 1e-10, '[TC28] wrap_to_pi FAILED'
