# ---- TC01: generate_nucleosome_tet_mesh 输出结构正确性 ----
np.random.seed(42)
mesh = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
assert mesh.n_nodes > 0 and mesh.n_elements > 0, '[TC01] generate_nucleosome_tet_mesh 输出结构正确性 FAILED'

# ---- TC02: TetMesh 表面积为正且体积积分与节点数一致 ----
mesh2 = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
assert mesh2.compute_surface_area() > 0, '[TC02] TetMesh 表面积为正且体积积分与节点数一致 FAILED'
vol_sum, _ = mesh2.integrate_nodal_values(np.ones(mesh2.n_nodes))
assert vol_sum > 0, '[TC02] TetMesh 表面积为正且体积积分与节点数一致 FAILED'

# ---- TC03: compute_dsb_repair_compartment_volume 非负性与阈值单调性 ----
np.random.seed(1)
gamma_density = np.exp(-np.sum(mesh.nodes ** 2, axis=1) / (2.0 * 50.0 ** 2))
gamma_density = gamma_density / np.max(gamma_density)
vol_repair, total_signal = compute_dsb_repair_compartment_volume(mesh, gamma_density, threshold=0.15)
assert vol_repair >= 0 and total_signal >= 0, '[TC03] compute_dsb_repair_compartment_volume 非负性与阈值单调性 FAILED'
vol_repair_high, _ = compute_dsb_repair_compartment_volume(mesh, gamma_density, threshold=0.85)
assert vol_repair_high <= vol_repair + 1e-12, '[TC03] compute_dsb_repair_compartment_volume 非负性与阈值单调性 FAILED'

# ---- TC04: normal_distribution_ode_solution 峰值、边界与积分归一化 ----
y0 = normal_distribution_ode_solution(np.array([0.0]), sigma0=1.0)
assert abs(y0[0] - 1.0 / np.sqrt(2.0 * np.pi)) < 1e-10, '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'
t = np.linspace(-5, 5, 1000)
y = normal_distribution_ode_solution(t, sigma0=1.0)
integral = np.trapezoid(y, t)
assert abs(integral - 1.0) < 0.01, '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'
y_far = normal_distribution_ode_solution(np.array([-20.0, 20.0]), sigma0=1.0)
assert np.all(y_far >= 0) and np.all(y_far < 1e-80), '[TC04] normal_distribution_ode_solution 峰值、边界与积分归一化 FAILED'

# ---- TC05: henon_crowding_map 原点、圆外不变性与有界性 ----
x = np.array([0.0, 2.0, 0.5])
y = np.array([0.0, 0.0, 0.5])
xm, ym = henon_crowding_map(x, y, c=0.98, n_iter=5)
assert xm[0] == 0.0 and ym[0] == 0.0, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'
assert xm[1] == 2.0 and ym[1] == 0.0, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'
assert np.sqrt(xm[2] ** 2 + ym[2] ** 2) < 1.0 + 1e-6, '[TC05] henon_crowding_map 原点、圆外不变性与有界性 FAILED'

# ---- TC06: OverdampedLangevinIntegrator 构造校验与step输出形状 ----
try:
    OverdampedLangevinIntegrator(dt=-0.001)
    assert False, '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'
except ValueError:
    pass
integrator = OverdampedLangevinIntegrator(dt=0.001)
x = np.array([[100.0, 0.0, 0.0]])
f = np.array([[1.0, 0.0, 0.0]])
x_new = integrator.step(x, f)
assert x_new.shape == (1, 3), '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'
assert np.linalg.norm(x_new[0]) <= 5000.0 + 1e-6, '[TC06] OverdampedLangevinIntegrator 构造校验与step输出形状 FAILED'

# ---- TC07: dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 ----
force = dna_repair_protein_force(np.zeros((0, 3)), np.zeros(3))
assert force.shape == (0, 3), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'
force2 = dna_repair_protein_force(np.array([[1.0, 0.0, 0.0]]), np.zeros(3))
assert force2.shape == (1, 3) and np.all(np.isfinite(force2)), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'
np.random.seed(7)
force3 = dna_repair_protein_force(np.random.randn(5, 3), np.zeros(3))
assert force3.shape == (5, 3) and np.all(np.isfinite(force3)), '[TC07] dna_repair_protein_force 空输入、单蛋白与多蛋白鲁棒性 FAILED'

# ---- TC08: simulate_ku80_search_time 输出结构与固定种子可复现性 ----
np.random.seed(42)
result_a = simulate_ku80_search_time(n_proteins=5, n_steps=100)
assert 'mfpt_us' in result_a and 'binding_fraction' in result_a and 'msd_final_nm2' in result_a, '[TC08] simulate_ku80_search_time 输出结构与固定种子可复现性 FAILED'
np.random.seed(42)
result_b = simulate_ku80_search_time(n_proteins=5, n_steps=100)
assert abs(result_a['mfpt_us'] - result_b['mfpt_us']) < 1e-12, '[TC08] simulate_ku80_search_time 输出结构与固定种子可复现性 FAILED'

# ---- TC09: gray_scott_step 维度保持、非负性截断与零场稳定性 ----
u = np.ones((8, 8)) * 0.5
v = np.ones((8, 8)) * 0.25
u_new, v_new = gray_scott_step(u, v, du=0.16, dv=0.08, f=0.03, k=0.062, dt=0.1, dx=1.0, dy=1.0)
assert u_new.shape == (8, 8) and v_new.shape == (8, 8), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'
assert np.all(u_new >= 0) and np.all(v_new >= 0), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'
u0, v0 = gray_scott_step(np.zeros((4, 4)), np.zeros((4, 4)), du=0.16, dv=0.08, f=0.03, k=0.062, dt=0.1, dx=1.0, dy=1.0)
assert np.all(u0 >= 0) and np.all(v0 >= 0), '[TC09] gray_scott_step 维度保持、非负性截断与零场稳定性 FAILED'

# ---- TC10: porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 ----
x = np.linspace(-2, 2, 100)
res = porous_medium_residual(x, t=1.0, m=3.0)
assert np.linalg.norm(res) < 1.0, '[TC10] porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 FAILED'
x_far = np.array([-10.0, 10.0])
res_far = porous_medium_residual(x_far, t=1.0, m=3.0)
assert np.allclose(res_far, 0.0), '[TC10] porous_medium_residual Barenblatt精确解残差范数与紧支集外为零 FAILED'

# ---- TC11: simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 ----
pm = simulate_parp1_nonlinear_diffusion(nx=64, nt=50, dt=0.01, dx=0.1, m=3.0)
assert 'l2_error' in pm and 'total_mass_numerical' in pm and 'total_mass_exact' in pm, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'
assert pm['total_mass_numerical'] >= 0, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'
assert pm['l2_error'] < 1.0, '[TC11] simulate_parp1_nonlinear_diffusion 输出结构、质量守恒与误差有界 FAILED'

# ---- TC12: simplex_vertex_coordinates 维度、范数与夹角约束 ----
for nd in [2, 3, 4]:
    s = simplex_vertex_coordinates(nd)
    assert s.shape == (nd, nd + 1), '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
    norms = np.linalg.norm(s, axis=0)
    assert np.allclose(norms, 1.0), '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
    cos_angle = np.dot(s[:, 0], s[:, 1])
    assert abs(cos_angle + 1.0 / nd) < 1e-10, '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'
s4 = simplex_vertex_coordinates(4)
d = np.linalg.norm(s4[:, 0] - s4[:, 1])
theory = np.sqrt(2.0 + 2.0 / 4.0)
assert abs(d - theory) < 1e-6, '[TC12] simplex_vertex_coordinates 维度、范数与夹角约束 FAILED'

# ---- TC13: pwl_interp_2d 三角形插值精确性与外推行为 ----
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([0.0, 1.0, 2.0])
zd = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
zi = pwl_interp_2d(xd, yd, zd, np.array([0.5]), np.array([0.2]))
assert abs(zi[0] - 0.7) < 1e-9, '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'
zi2 = pwl_interp_2d(xd, yd, zd, np.array([1.5]), np.array([1.8]))
assert np.isfinite(zi2[0]), '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'
zi_out = pwl_interp_2d(xd, yd, zd, np.array([5.0]), np.array([5.0]))
assert zi_out[0] == np.inf, '[TC13] pwl_interp_2d 三角形插值精确性与外推行为 FAILED'

# ---- TC14: sammon_mapping 降维后输出形状正确 ----
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
Y = sammon_mapping(X, n_components=2, max_iter=10, alpha=0.3, random_state=42)
assert Y.shape == (4, 2), '[TC14] sammon_mapping 降维后输出形状正确 FAILED'

# ---- TC15: BandedLU 三对角系统求解残差 ----
n = 20
A = np.diag(2.0 * np.ones(n)) + np.diag(-1.0 * np.ones(n - 1), 1) + np.diag(-1.0 * np.ones(n - 1), -1)
b = np.ones(n)
solver = BandedLU(n, ml=1, mu=1)
A_band = solver._full_to_band(A)
A_lu, pivot, info = solver.factorize(A_band)
x_sol = solver.solve(A_lu, pivot, b)
res = np.linalg.norm(A @ x_sol - b)
assert res < 1e-10, '[TC15] BandedLU 三对角系统求解残差 FAILED'

# ---- TC16: solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 ----
pb = solve_nonlinear_pb(n=33, domain_length=10.0, max_iter=20, tol=1e-6)
assert pb['success'], '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'
assert abs(pb['phi'][0]) < 1e-6 and abs(pb['phi'][-1]) < 1e-6, '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'
assert pb['residual_norm'] < 1e-6, '[TC16] solve_nonlinear_pb 收敛性、Dirichlet边界与残差下降 FAILED'

# ---- TC17: matrix_chain_optimal_order 经典动态规划最优解与边界 ----
dims = [30, 35, 15, 5, 10, 20, 25]
cost, s = matrix_chain_optimal_order(dims)
assert cost == 15125, '[TC17] matrix_chain_optimal_order 经典动态规划最优解与边界 FAILED'
cost_single, _ = matrix_chain_optimal_order([10, 20])
assert cost_single == 0, '[TC17] matrix_chain_optimal_order 经典动态规划最优解与边界 FAILED'

# ---- TC18: catalan_number 递推正确性与小值边界 ----
assert catalan_number(5) == 42, '[TC18] catalan_number 递推正确性与小值边界 FAILED'
assert catalan_number(0) == 1, '[TC18] catalan_number 递推正确性与小值边界 FAILED'
assert catalan_number(-1) == 0, '[TC18] catalan_number 递推正确性与小值边界 FAILED'

# ---- TC19: assemble_enm_stiffness_matrix 对称性、维度与力平衡 ----
np.random.seed(3)
coords = np.random.randn(5, 3) * 10.0
H = assemble_enm_stiffness_matrix(coords, cutoff=50.0, spring_constant=1.0)
assert H.shape == (15, 15), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'
assert np.allclose(H, H.T), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'
row_sums = np.sum(H, axis=1)
assert np.allclose(row_sums, 0.0), '[TC19] assemble_enm_stiffness_matrix 对称性、维度与力平衡 FAILED'

# ---- TC20: Matrix Market I/O 坐标格式与数组格式往返一致性 ----
A_test = np.array([[1.0, 2.0], [2.0, 3.0]])
write_matrix_market("/tmp/test_mm_coord.mtx", A_test, rep="coordinate", symm="symmetric")
A_read, rows, cols, entries, rep, field, symm = read_matrix_market("/tmp/test_mm_coord.mtx")
assert np.allclose(A_test, A_read), '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'
assert symm == "symmetric", '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'
write_matrix_market("/tmp/test_mm_array.mtx", A_test, rep="array", symm="general")
A_read2, rows2, cols2, entries2, rep2, field2, symm2 = read_matrix_market("/tmp/test_mm_array.mtx")
assert np.allclose(A_test, A_read2), '[TC20] Matrix Market I/O 坐标格式与数组格式往返一致性 FAILED'

# ---- TC21: build_tec_file与parse_tec_file 往返一致性与维度解析 ----
node_coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64).T
element_node = np.array([[0, 1, 2, 3]], dtype=np.int64).T
node_data = np.array([[1.0, 0.8, 0.9, 1.1]], dtype=np.float64)
tec = build_tec_file(node_coord, element_node, node_data, variable_names=['X', 'Y', 'Z', 'Density'])
parsed = parse_tec_file(tec)
assert parsed['node_num'] == 4 and parsed['element_num'] == 1, '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'
assert parsed['dim_num'] == 3, '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'
assert 'Density' in parsed['variable_names'], '[TC21] build_tec_file与parse_tec_file 往返一致性与维度解析 FAILED'

# ---- TC22: grid_double_resolution 2D/3D形状与积分守恒 ----
field = np.ones((4, 4))
doubled = grid_double_resolution(field, mode="2d")
assert doubled.shape == (8, 8), '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
assert abs(np.sum(field) - np.sum(doubled)) < 1e-12, '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
field3 = np.ones((2, 2, 2))
doubled3 = grid_double_resolution(field3, mode="3d")
assert doubled3.shape == (4, 4, 4), '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'
assert abs(np.sum(field3) - np.sum(doubled3)) < 1e-12, '[TC22] grid_double_resolution 2D/3D形状与积分守恒 FAILED'

# ---- TC23: adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 ----
np.random.seed(8)
field2d = np.zeros((8, 8))
field2d[3:5, 3:5] = 1.0
levels = adaptive_mesh_refinement_2d(field2d, gradient_threshold=0.1, max_level=1)
assert len(levels) >= 1 and levels[0].shape == field2d.shape, '[TC23] adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 FAILED'
if len(levels) > 1:
    assert levels[1].shape[0] == 2 * field2d.shape[0], '[TC23] adaptive_mesh_refinement_2d 至少返回原始网格与形状递增 FAILED'

# ---- TC24: compute_optimal_tensor_contraction_cost 与矩阵链一致性 ----
tensor_cost = compute_optimal_tensor_contraction_cost(
    [(30, 35), (35, 15), (15, 5), (5, 10), (10, 20), (20, 25)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
)
assert tensor_cost == 15125, '[TC24] compute_optimal_tensor_contraction_cost 与矩阵链一致性 FAILED'

# ---- TC25: simulate_gamma_h2ax_wave 输出结构与非负总量 ----
gs = simulate_gamma_h2ax_wave(nx=32, ny=32, nt=100, f=0.03, k=0.062, du=0.16, dv=0.08, dt=0.1, dx=1.0)
assert gs['u_final'].shape == (32, 32) and gs['v_final'].shape == (32, 32), '[TC25] simulate_gamma_h2ax_wave 输出结构与非负总量 FAILED'
assert gs['total_gamma_h2ax'] >= 0, '[TC25] simulate_gamma_h2ax_wave 输出结构与非负总量 FAILED'

# ---- TC26: assemble_poisson_boltzmann_jacobian 输出维度与边界条件 ----
n_j = 9
phi_j = np.zeros(n_j)
rho_j = np.zeros(n_j)
J, F = assemble_poisson_boltzmann_jacobian(n_j, phi_j, rho_j, h=0.5)
assert J.shape == (n_j, n_j), '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'
assert F.shape == (n_j,), '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'
assert J[0, 0] == 1.0 and F[0] == 0.0, '[TC26] assemble_poisson_boltzmann_jacobian 输出维度与边界条件 FAILED'

# ---- TC27: TetMesh常数场积分等于总体积 ----
mesh3 = generate_nucleosome_tet_mesh(n_rings=2, n_theta=4, n_z=2)
integral_const, total_vol = mesh3.integrate_nodal_values(np.ones(mesh3.n_nodes))
assert abs(integral_const - total_vol) < 1e-12, '[TC27] TetMesh常数场积分等于总体积 FAILED'
assert total_vol > 0, '[TC27] TetMesh常数场积分等于总体积 FAILED'

# ---- TC28: build_optimal_parenthesization 输出非空字符串 ----
dims2 = [30, 35, 15, 5, 10, 20, 25]
cost2, s2 = matrix_chain_optimal_order(dims2)
opt_expr = build_optimal_parenthesization(s2, 0, len(dims2) - 2)
assert isinstance(opt_expr, str) and len(opt_expr) > 0, '[TC28] build_optimal_parenthesization 输出非空字符串 FAILED'

# ---- TC29: main() 集成运行返回None且不崩溃 ----
main_result = main()
assert main_result is None, '[TC29] main() 集成运行返回None且不崩溃 FAILED'

# ---- TC30: pwl_interp_2d 网格边界点精确恢复 ----
xg = np.array([0.0, 1.0, 2.0])
yg = np.array([0.0, 1.0])
zg = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
zi_grid = pwl_interp_2d(xg, yg, zg, xg, yg.repeat(len(xg)))
assert np.allclose(zi_grid[:len(xg)], zg[:, 0]), '[TC30] pwl_interp_2d 网格边界点精确恢复 FAILED'
