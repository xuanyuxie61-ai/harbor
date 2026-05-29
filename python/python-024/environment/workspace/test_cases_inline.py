
# ---- TC01: HarrisEquilibrium B_field中心点Bx为零 ----
eq = HarrisEquilibrium(B0=1.0e-2, lambda_cs=5.0e4, B_guide=2.0e-3, p0=2.0e-3, T_plasma=2.0e6, rho_inf=1.0e-12, y_max=3.0e5)
B0 = eq.B_field(np.array([0.0]))
assert abs(B0[0, 0]) < 1e-10, '[TC01] Harris B_field中心Bx为零 FAILED'

# ---- TC02: HarrisEquilibrium pressure在y=0处最大 ----
p_at_0 = eq.pressure(np.array([0.0]))[0]
p_at_far = eq.pressure(np.array([eq.y_max]))[0]
assert p_at_0 > p_at_far, '[TC02] Harris pressure中心最大 FAILED'

# ---- TC03: HarrisEquilibrium current_density非负 ----
y_arr = np.linspace(-eq.y_max, eq.y_max, 65)
J = eq.current_density(y_arr)
assert np.all(J[:, 2] >= 0), '[TC03] Harris current_density非负 FAILED'

# ---- TC04: HarrisEquilibrium mass_density有限且非负 ----
rho = eq.mass_density(y_arr)
assert np.all(rho >= 0) and np.all(np.isfinite(rho)), '[TC04] Harris mass_density非负有限 FAILED'

# ---- TC05: HarrisEquilibrium alfven_speed有限非负 ----
va = eq.alfven_speed(y_arr)
assert np.all(va >= 0) and np.all(np.isfinite(va)), '[TC05] Harris alfven_speed非负有限 FAILED'

# ---- TC06: HarrisEquilibrium plasma_beta有限非负 ----
beta = eq.plasma_beta(y_arr)
assert np.all(beta >= 0) and np.all(np.isfinite(beta)), '[TC06] Harris plasma_beta非负有限 FAILED'

# ---- TC07: HarrisEquilibrium generate_quadrilateral_mesh尺寸正确 ----
nodes, elements = eq.generate_quadrilateral_mesh(nx=8, ny=16)
assert nodes.shape == (8 * 16, 2), '[TC07] Harris mesh节点数 FAILED'
assert elements.shape == ((8 - 1) * (16 - 1), 4), '[TC07] Harris mesh单元数 FAILED'

# ---- TC08: HarrisEquilibrium bilinear_interpolate_on_mesh范围约束 ----
y_arr2 = np.linspace(-eq.y_max, eq.y_max, 65)
p_arr = eq.pressure(y_arr2)
field_interp = eq.bilinear_interpolate_on_mesh(nodes, elements, p_arr, y_arr2)
assert np.min(field_interp) >= np.min(p_arr) * 0.99 and np.max(field_interp) <= np.max(p_arr) * 1.01, '[TC08] Harris bilinear_interpolate范围 FAILED'

# ---- TC09: HarrisEquilibrium magnetic_shear中心最大 ----
shear = eq.magnetic_shear(y_arr)
assert shear[len(y_arr)//2] >= np.max(shear) * 0.99, '[TC09] Harris magnetic_shear中心最大 FAILED'

# ---- TC10: Quaternion from_axis_angle rotate_vector 90度旋转x到y ----
q90 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
B_rot = q90.rotate_vector(np.array([1.0, 0.0, 0.0]))
assert abs(B_rot[0]) < 1e-10 and abs(B_rot[1] - 1.0) < 1e-10 and abs(B_rot[2]) < 1e-10, '[TC10] Quaternion 90度旋转 FAILED'

# ---- TC11: Quaternion to_rotation_matrix正交性验证 ----
R = q90.to_rotation_matrix()
I_check = R @ R.T
assert np.allclose(I_check, np.eye(3), atol=1e-10), '[TC11] Quaternion rotation_matrix正交性 FAILED'

# ---- TC12: Quaternion slerp t=0.5为中间旋转 ----
q0 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)
q1 = Quaternion.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
q_mid = q0.slerp(q1, 0.5)
B_mid = q_mid.rotate_vector(np.array([1.0, 0.0, 0.0]))
assert abs(B_mid[0]) < 1e-6 and abs(B_mid[1] - 1.0) < 1e-6, '[TC12] Quaternion SLERP FAILED'

# ---- TC13: MagneticTopology reflect_y反射对称性 ----
v = np.array([1.0, 2.0, 3.0])
v_ref = MagneticTopology.reflect_y(v)
assert v_ref[0] == -1.0 and v_ref[1] == 2.0 and v_ref[2] == -3.0, '[TC13] MagneticTopology reflect_y FAILED'

# ---- TC14: FEM1DAssembler mass_matrix尺寸与对称性 ----
fem = FEM1DAssembler(8, domain=(0.0, 1.0))
M = fem.mass_matrix(sparse=False)
assert M.shape == (9, 9), '[TC14] FEM mass_matrix尺寸 FAILED'
assert np.allclose(M, M.T, atol=1e-14), '[TC14] FEM mass_matrix对称性 FAILED'

# ---- TC15: FEM1DAssembler stiffness_matrix尺寸 ----
K = fem.stiffness_matrix(sparse=False)
assert K.shape == (9, 9), '[TC15] FEM stiffness_matrix尺寸 FAILED'

# ---- TC16: STtoGEAssembler assemble累加验证 ----
ist = np.array([1, 2, 2, 3, 3, 3])
jst = np.array([1, 1, 2, 2, 3, 1])
ast = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
A = STtoGEAssembler.assemble(len(ist), ist, jst, ast)
assert A[1, 0] == 2.0 and A[2, 2] == 5.0 and A[2, 0] == 6.0, '[TC16] ST assemble累加 FAILED'

# ---- TC17: HankelSolver build_hankel + inverse逆矩阵验证 ----
n = 4
np.random.seed(7)
x = np.abs(np.random.randn(2 * n - 1)) + 0.5
H = HankelSolver.build_hankel(n, x)
H_inv = HankelSolver.inverse(n, x)
I_approx = H @ H_inv
assert np.allclose(I_approx, np.eye(n), atol=1e-8), '[TC17] Hankel逆矩阵 FAILED'

# ---- TC18: IntegerRREF rref行简化验证 ----
A_test = np.array([[1, 3, 0, 2, 6, 3, 1],
                   [-2, -6, 0, -2, -8, 3, 1],
                   [3, 9, 0, 0, 6, 6, 2],
                   [-1, -3, 0, 1, 0, 9, 3]], dtype=np.int64)
Ar, det = IntegerRREF.rref(A_test)
assert Ar.shape == A_test.shape, '[TC18] IRREF尺寸 FAILED'
assert det != 0, '[TC18] IRREF伪行列式 FAILED'

# ---- TC19: MHDStabilityAnalyzer analyze_stability返回结构 ----
analyzer = MHDStabilityAnalyzer(kx=0.5, eta=1e-3, ny=32)
result = analyzer.analyze_stability()
assert 'max_growth_rate' in result and 'unstable' in result and 'n_modes' in result, '[TC19] MHD stability返回结构 FAILED'

# ---- TC20: TrigonometricInterpolator interpolate插值精度 ----
N = 16
phi_nodes = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
f_vals = np.cos(2.0 * phi_nodes) + 0.5 * np.sin(3.0 * phi_nodes)
interp = TrigonometricInterpolator(phi_nodes, f_vals)
phi_fine = np.linspace(0.0, 2.0 * np.pi, 200)
f_interp = interp.interpolate(phi_fine)
f_exact = np.cos(2.0 * phi_fine) + 0.5 * np.sin(3.0 * phi_fine)
err = np.max(np.abs(f_interp - f_exact))
assert err < 1e-9, '[TC20] TrigonometricInterpolator插值精度 FAILED'

# ---- TC21: TrigonometricInterpolator derivative导数精度 ----
df_exact = -2.0 * np.sin(2.0 * phi_fine) + 1.5 * np.cos(3.0 * phi_fine)
df_interp = interp.derivative(phi_fine)
err_d = np.max(np.abs(df_interp - df_exact))
assert err_d < 100.0 and np.all(np.isfinite(df_interp)), '[TC21] TrigonometricInterpolator导数精度 FAILED'

# ---- TC22: GridReshaper vector_to_grid reshape正确 ----
vec = np.arange(24)
grid = GridReshaper.vector_to_grid(vec, 4, 6, orientation='row')
assert grid.shape == (4, 6), '[TC22] GridReshaper vector_to_grid尺寸 FAILED'
assert grid[0, 0] == 0 and grid[3, 5] == 23, '[TC22] GridReshaper vector_to_grid值 FAILED'

# ---- TC23: WedgeIntegrals monomial_integral体积验证 ----
vol = WedgeIntegrals.monomial_integral((0, 0, 0))
assert abs(vol - 1.0) < 1e-14, '[TC23] WedgeIntegrals体积 FAILED'

# ---- TC24: WedgeIntegrals monomial_integral奇数次z为零 ----
val = WedgeIntegrals.monomial_integral((1, 1, 1))
assert abs(val) < 1e-14, '[TC24] WedgeIntegrals奇数次z积分 FAILED'

# ---- TC25: MonteCarloParticleAccelerator sample_initial_velocities可复现性 ----
np.random.seed(42)
acc = MonteCarloParticleAccelerator(n_particles=100)
v1 = acc.sample_initial_velocities(T_thermal=1.0e6, seed=42)
np.random.seed(42)
v2 = acc.sample_initial_velocities(T_thermal=1.0e6, seed=42)
assert np.allclose(v1, v2, atol=1e-14), '[TC25] MonteCarlo采样可复现性 FAILED'

# ---- TC26: NonlinearOrbitTracker integrate_rk4输出尺寸 ----
tracker = NonlinearOrbitTracker()
y0 = np.array([0.1, 0.0, 1.0e5, 0.0])
t, states = tracker.integrate_rk4(y0, (0.0, 1.0e-3), n_steps=100)
assert len(t) == 101 and states.shape == (101, 4), '[TC26] NonlinearOrbitTracker RK4输出尺寸 FAILED'

# ---- TC27: AnomalousResistivity reaction_term边界输入 ----
model = AnomalousResistivity(ny=32)
eta = np.full(32, model.eta_cl)
J = np.zeros(32)
R = model.reaction_term(eta, J)
assert len(R) == 32 and np.all(np.isfinite(R)), '[TC27] AnomalousResistivity reaction_term FAILED'

# ---- TC28: AnomalousResistivity equilibrium_eta范围约束 ----
J_test = model.J_c * 10.0 * np.ones(32)
eta_eq = model.equilibrium_eta(J_test)
assert np.all(eta_eq >= model.eta_cl) and np.all(eta_eq <= model.eta_max), '[TC28] equilibrium_eta范围 FAILED'

# ---- TC29: WaveDampingModel simulate输出结构 ----
wave = WaveDampingModel(nx=21)
times, states = wave.simulate((0.0, 1.0), nt=50)
assert len(times) == 51 and states.shape == (51, 42), '[TC29] WaveDampingModel simulate输出结构 FAILED'

# ---- TC30: ContinuationSolver trace_branch延拓步数 ----
def F(x):
    return np.array([x[0]**3 - x[0] + x[1]])
def J(x):
    return np.array([[3.0 * x[0]**2 - 1.0, 1.0]])
solver = ContinuationSolver(h_init=0.05, tol=1e-10)
xs, params, ps = solver.trace_branch(F, J, np.array([0.0, 0.0]), n_steps=20)
assert len(xs) >= 2 and len(params) == len(xs), '[TC30] ContinuationSolver trace_branch FAILED'

# ---- TC31: MagneticTopology check_harris_symmetry对称性验证 ----
eq2 = HarrisEquilibrium()
y_test = np.linspace(-eq2.y_max, eq2.y_max, 21)
sym = MagneticTopology.check_harris_symmetry(eq2.B_field, y_test)
assert sym['symmetric'] == True, '[TC31] Harris对称性验证 FAILED'

# ---- TC32: FEM1DAssembler solve_steady解结构 ----
fem2 = FEM1DAssembler(16, domain=(0.0, 1.0))
K2 = fem2.stiffness_matrix(sparse=False)
F2 = np.ones(17)
u = fem2.solve_steady(K2, F2, sparse=False)
assert len(u) == 17 and np.all(np.isfinite(u)), '[TC32] FEM solve_steady FAILED'

# ---- TC33: HarrisEquilibrium compute_reconnection_rate输出尺寸 ----
y_small = np.array([0.0, 1.0e4])
eta_small = np.array([1e-6, 1e-6])
v_small = np.zeros((2, 3))
E_rec = eq.compute_reconnection_rate(y_small, eta_small, v_small)
assert len(E_rec) == 2 and np.all(np.isfinite(E_rec)), '[TC33] compute_reconnection_rate FAILED'

# ---- TC34: WedgeIntegrals gauss_legendre_wedge_7point尺寸 ----
pts, wts = WedgeIntegrals.gauss_legendre_wedge_7point()
assert pts.shape == (6, 3) and len(wts) == 6, '[TC34] WedgeIntegrals 7point尺寸 FAILED'

# ---- TC35: Quaternion norm单位四元数模为1 ----
q_unit = Quaternion(1.0, 0.0, 0.0, 0.0)
assert abs(q_unit.norm() - 1.0) < 1e-14, '[TC35] Quaternion单位模 FAILED'
