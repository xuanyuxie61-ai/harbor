# ---- TC01: 双阱势在 phi=0 时为 0.25 ----
pf_test = PhaseFieldModel(nx=5, ny=5, dx=0.1, dy=0.1, epsilon=0.01)
W0 = pf_test.double_well_potential(np.array([0.0]))
assert np.abs(W0[0] - 0.25) < 1e-12, '[TC01] 双阱势在 phi=0 时为 0.25 FAILED'

# ---- TC02: 双阱势导数在平衡态 phi=+-1 处为零 ----
pf_test = PhaseFieldModel(nx=5, ny=5, dx=0.1, dy=0.1, epsilon=0.01)
dW = pf_test.double_well_derivative(np.array([1.0, -1.0]))
assert np.max(np.abs(dW)) < 1e-12, '[TC02] 双阱势导数在平衡态为零 FAILED'

# ---- TC03: 插值函数输出范围在 [0,1] ----
pf_test = PhaseFieldModel(nx=5, ny=5, dx=0.1, dy=0.1, epsilon=0.01)
h_vals = pf_test.interpolation_function(np.array([-2.0, 0.0, 2.0]))
assert np.all(h_vals >= 0.0) and np.all(h_vals <= 1.0), '[TC03] 插值函数输出范围 FAILED'

# ---- TC04: 五点差分 Laplacian 对常数场为零 ----
pf_test = PhaseFieldModel(nx=7, ny=7, dx=0.1, dy=0.1, epsilon=0.01)
const_field = np.ones((7, 7))
lap = pf_test.laplacian_5point(const_field)
assert np.max(np.abs(lap)) < 1e-12, '[TC04] Laplacian 对常数场为零 FAILED'

# ---- TC05: 梯度模长非负 ----
np.random.seed(42)
tracker = InterfaceTracker(nx=7, ny=7, dx=0.1, dy=0.1)
phi_test = np.random.rand(7, 7)
grad_mag = tracker.compute_gradient_magnitude(phi_test)
assert np.all(grad_mag >= 0.0), '[TC05] 梯度模长非负 FAILED'

# ---- TC06: Gauss-Legendre 3点权重和为 2 ----
nodes, weights = GaussQuadrature.gauss_legendre_3point()
assert np.abs(np.sum(weights) - 2.0) < 1e-12, '[TC06] Gauss-Legendre 3点权重和 FAILED'

# ---- TC07: 超立方体单项式积分解析验证 ----
val = HypercubeIntegrals.monomial_integral(3, np.array([1, 2, 1]))
assert np.abs(val - 1.0/12.0) < 1e-12, '[TC07] 超立方体单项式积分 FAILED'

# ---- TC08: DST 与 IDST 互为逆变换 ----
f = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
b = SineTransform.dst_1d(f)
f_recon = SineTransform.idst_1d(b)
assert np.max(np.abs(f - f_recon)) < 1e-12, '[TC08] DST-IDST 逆变换 FAILED'

# ---- TC09: 矩阵对数范数对稳定矩阵返回负值 ----
A = np.array([[-2.0, 1.0], [0.5, -3.0]])
mu2 = LogarithmicNorm.log_norm(A, p=2)
assert mu2 < 0.0, '[TC09] 稳定矩阵对数范数为负 FAILED'

# ---- TC10: 伴矩阵法求多项式根验证 ----
coeffs = np.array([6.0, -5.0, 1.0])
roots = CompanionMatrixEigenvalue.find_roots(coeffs, basis='power')
roots_sorted = np.sort(np.real(roots))
assert np.max(np.abs(roots_sorted - np.array([2.0, 3.0]))) < 1e-10, '[TC10] 伴矩阵求根验证 FAILED'

# ---- TC11: Logistic 映射不动点验证 ----
x_fp = BifurcationAnalysis.logistic_map(0.6, 2.5)
for _ in range(100):
    x_fp = BifurcationAnalysis.logistic_map(x_fp, 2.5)
assert np.abs(x_fp - 0.6) < 1e-10, '[TC11] Logistic 不动点验证 FAILED'

# ---- TC12: 显式 Euler 对线性ODE yprime=-y 的衰减验证 ----
def f_decay(t, y):
    return -y
t_e, y_e = ODESolver.explicit_euler(f_decay, 0.0, np.array([1.0]), 1.0, 0.01)
assert y_e[-1][0] < y_e[0][0] and y_e[-1][0] > 0.0, '[TC12] Euler 线性ODE衰减验证 FAILED'

# ---- TC13: 固相分数输出范围 [0,1] ----
thermal = ThermalTransportSolver(nx=5, ny=5, dx=0.1, dy=0.1, dt=0.001)
h = thermal.solid_fraction(np.array([-1.5, 0.0, 1.5]))
assert np.all(h >= 0.0) and np.all(h <= 1.0), '[TC13] 固相分数范围 FAILED'

# ---- TC14: 均匀相场的界面张力接近零 ----
ns = NavierStokesSolver(nx=7, ny=7, dx=0.1, dy=0.1, dt=0.001)
phi_uniform = np.ones((7, 7))
Fx, Fy = ns.compute_surface_tension_force(phi_uniform)
assert np.max(np.abs(Fx)) < 1e-10 and np.max(np.abs(Fy)) < 1e-10, '[TC14] 均匀相场界面张力为零 FAILED'

# ---- TC15: Dirichlet BC 正确施加边界值 ----
nx_fem, ny_fem = 5, 5
x_fem = np.linspace(0, 1, nx_fem)
y_fem = np.linspace(0, 1, ny_fem)
fem = FEM2DSerene(nx_fem, ny_fem, x_fem, y_fem)
K = np.ones((nx_fem*ny_fem, nx_fem*ny_fem))
F = np.zeros(nx_fem*ny_fem)
K_bc, F_bc = fem.apply_dirichlet_bc(K, F, [0], [5.0])
assert K_bc[0,0] == 1.0 and F_bc[0] == 5.0, '[TC15] Dirichlet BC 施加 FAILED'

# ---- TC16: 等边三角形质量为1 ----
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.5, np.sqrt(3.0)/2.0])
q = TriangleGridTopology.triangle_quality(p1, p2, p3)
assert np.abs(q - 1.0) < 1e-12, '[TC16] 等边三角形质量为1 FAILED'

# ---- TC17: 动态规划网格分配总和正确 ----
mesh_adapt = MeshAdaptation(nx=10, ny=10, x_max=1.0, y_max=1.0)
err_funcs = [lambda n: 1.0/max(n,1)**2, lambda n: 0.5/max(n,1)**2]
dist = mesh_adapt.dynamic_programming_mesh_distribution(20, err_funcs, 2)
assert sum(dist) == 20, '[TC17] 动态规划网格分配总和 FAILED'

# ---- TC18: 显式步进对零rhs保持场不变 ----
stepper = PhaseFieldTimeStepper(dt=0.01, dx=0.1, dy=0.1, epsilon=0.01, tau=1.0)
phi_test = np.ones((5, 5))
phi_new = stepper.explicit_step(phi_test, lambda p: np.zeros_like(p))
assert np.max(np.abs(phi_new - phi_test)) < 1e-12, '[TC18] 显式步进零rhs不变 FAILED'

# ---- TC19: Gauss-Seidel 求解一维泊松方程收敛 ----
x_gs, u_gs, it_gs = GaussSeidelPoisson.solve_1d_poisson_gs(
    n_intervals=20, a=0.0, b=1.0, ua=0.0, ub=1.0,
    force_func=lambda x: 0.0, max_iter=10000, tol=1e-6
)
assert it_gs < 10000, '[TC19] GS 求解一维泊松方程收敛 FAILED'

# ---- TC20: Logistic Lyapunov 指数在稳定区为负 ----
lyap = BifurcationAnalysis.lyapunov_exponent_logistic(2.5, n_iter=5000)
assert lyap < 0.0, '[TC20] Logistic Lyapunov 稳定区为负 FAILED'

# ---- TC21: 标准正态数组输出形状匹配 ----
np.random.seed(42)
arr = RandomNumberGenerator.standard_normal_array((3, 4))
assert arr.shape == (3, 4), '[TC21] 标准正态数组形状 FAILED'

# ---- TC22: 复合 Simpson 对常数函数精确积分 ----
val = CompositeQuadrature.composite_simpson(lambda x: 5.0, 0.0, 1.0, 4)
assert np.abs(val - 5.0) < 1e-12, '[TC22] 复合 Simpson 常数函数精确 FAILED'

# ---- TC23: CFL条件返回正时间步长 ----
dt_cfl = LinearStabilityAnalysis.cfl_condition_1d_advection_diffusion(v=1.0, D=0.1, dx=0.01)
assert dt_cfl > 0.0, '[TC23] CFL条件返回正数 FAILED'

# ---- TC24: 谱方法 Poisson 求解器输出有限值 ----
solver_sp = SpectralPoissonSolver(nx=5, ny=5, Lx=1.0, Ly=1.0)
f_test = np.ones((5, 5))
u_test = solver_sp.solve_2d_poisson_dirichlet(f_test)
assert np.all(np.isfinite(u_test)), '[TC24] 谱方法 Poisson 求解器输出有限值 FAILED'

# ---- TC25: setup_simulation_domain 返回包含必需键的字典 ----
params = setup_simulation_domain()
assert isinstance(params, dict) and 'nx' in params and 'dt' in params, '[TC25] setup_simulation_domain 返回字典 FAILED'
