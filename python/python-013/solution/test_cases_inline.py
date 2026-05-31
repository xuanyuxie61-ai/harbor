# ---- TC01: TriangularLattice 基本属性验证 ----
lat = TriangularLattice(4, 4, a=1.0)
assert lat.nsites == 16, '[TC01] TriangularLattice 基本属性验证 FAILED'
assert len(lat.neighbors[0]) == 6, '[TC01] TriangularLattice 基本属性验证 FAILED'
assert lat.a == 1.0, '[TC01] TriangularLattice 基本属性验证 FAILED'

# ---- TC02: hex_grid_in_brillouin_zone 生成有效点 ----
lat2 = TriangularLattice(6, 6, a=1.0)
pts = hex_grid_in_brillouin_zone(3, lat2.bz_vertices)
assert pts.shape[1] == 2, '[TC02] hex_grid_in_brillouin_zone 生成有效点 FAILED'
assert len(pts) >= 1, '[TC02] hex_grid_in_brillouin_zone 生成有效点 FAILED'

# ---- TC03: build_hubbard_hamiltonian 返回厄米矩阵 ----
H = build_hubbard_hamiltonian(2, [[1], [0]], 1.0, 4.0, 0.0)
assert H.shape == (16, 16), '[TC03] build_hubbard_hamiltonian 返回厄米矩阵 FAILED'
assert np.allclose(H, H.T.conj()), '[TC03] build_hubbard_hamiltonian 返回厄米矩阵 FAILED'

# ---- TC04: exact_diagonalization 本征值为实数且有序 ----
evals, evecs = exact_diagonalization(H)
assert np.allclose(evals.imag, 0, atol=1e-10), '[TC04] exact_diagonalization 本征值为实数且有序 FAILED'
assert np.all(np.diff(evals) >= -1e-12), '[TC04] exact_diagonalization 本征值为实数且有序 FAILED'

# ---- TC05: compute_ground_state_properties 输出有限值 ----
props = compute_ground_state_properties(2, [[1], [0]], 1.0, 4.0, 0.0)
assert 'E0' in props and 'double_occupancy' in props, '[TC05] compute_ground_state_properties 输出有限值 FAILED'
assert np.isfinite(props['E0']) and np.isfinite(props['double_occupancy']), '[TC05] compute_ground_state_properties 输出有限值 FAILED'

# ---- TC06: thermal_average beta=0 等于算符对角元平均 ----
D = np.diag([0.0, 1.0, 1.0, 2.0])
evals2 = np.array([0.0, 1.0, 1.0, 2.0])
evecs2 = np.eye(4)
val = thermal_average(evals2, evecs2, D, beta=0.0)
expected = np.mean(np.diag(D))
assert np.isclose(val, expected), '[TC06] thermal_average beta=0 等于算符对角元平均 FAILED'

# ---- TC07: lebedev_sphere_grid 权重和近似 4pi ----
x, y, z, w = lebedev_sphere_grid(50)
assert len(w) > 0, '[TC07] lebedev_sphere_grid 权重和近似 4pi FAILED'
assert np.isclose(np.sum(w), 4 * np.pi, rtol=0.05), '[TC07] lebedev_sphere_grid 权重和近似 4pi FAILED'

# ---- TC08: brillouin_zone_area 为正 ----
lat3 = TriangularLattice(4, 4, a=1.0)
area = brillouin_zone_area(lat3.bz_vertices)
assert area > 0, '[TC08] brillouin_zone_area 为正 FAILED'

# ---- TC09: newton_divided_differences 对线性函数精确 ----
xd = np.array([0.0, 1.0, 2.0])
yd = 2.0 * xd + 1.0
dif = newton_divided_differences(xd, yd)
assert np.isclose(dif[0], 1.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'
assert np.isclose(dif[1], 2.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'
assert np.isclose(dif[2], 0.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'

# ---- TC10: evaluate_divided_difference 节点精确命中 ----
xv = np.array([0.0, 1.0, 2.0])
yv = evaluate_divided_difference(xd, dif, xv)
assert np.allclose(yv, yd), '[TC10] evaluate_divided_difference 节点精确命中 FAILED'

# ---- TC11: dyson_equation 零自能返回原格林函数 ----
G0_test = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
Sigma_zero = np.zeros_like(G0_test)
G_test = dyson_equation(G0_test, Sigma_zero)
assert np.allclose(G_test, G0_test), '[TC11] dyson_equation 零自能返回原格林函数 FAILED'

# ---- TC12: build_matsubara_green 形状正确 ----
K_test = np.array([[0.0, -1.0], [-1.0, 0.0]])
G0_mats = build_matsubara_green(2, K_test, mu=0.0, beta=2.0, U=0.0, n_max=2, sigma=0)
assert G0_mats.shape == (5, 2, 2), '[TC12] build_matsubara_green 形状正确 FAILED'

# ---- TC13: pade_spectral_function 求和规则在合理范围 ----
omega_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) * np.pi
g_iw = 1.0 / (1j * omega_n + 0.5)
omega_real = np.linspace(-5, 5, 100)
A_pade = pade_spectral_function(omega_n, g_iw, omega_real, eta=0.1)
sum_rule = np.trapezoid(A_pade, omega_real)
assert 0.5 < sum_rule < 2.0, '[TC13] pade_spectral_function 求和规则在合理范围 FAILED'

# ---- TC14: spectral_moments M0 非负 ----
moments = spectral_moments(A_pade, omega_real, max_moment=2)
assert moments['M_0'] >= 0, '[TC14] spectral_moments M0 非负 FAILED'

# ---- TC15: collatz_stopping_time 基础值 ----
assert collatz_stopping_time(1) == 0, '[TC15] collatz_stopping_time 基础值 FAILED'
assert collatz_stopping_time(27) == 111, '[TC15] collatz_stopping_time 基础值 FAILED'

# ---- TC16: self_consistent_iteration 线性问题收敛 ----
def linear_update(x):
    return np.array([0.5 * x[0] + 1.0])
x_sc, it_sc, res_sc = self_consistent_iteration(linear_update, np.array([0.0]), tol=1e-10, max_iter=100, mixing="simple", alpha=0.5)
assert it_sc < 100, '[TC16] self_consistent_iteration 线性问题收敛 FAILED'
assert np.isclose(x_sc[0], 2.0, atol=1e-6), '[TC16] self_consistent_iteration 线性问题收敛 FAILED'

# ---- TC17: iteration_complexity_index 单调残差 ----
res_mono = [1.0, 0.5, 0.25, 0.125]
idx = iteration_complexity_index(res_mono)
assert idx >= 0, '[TC17] iteration_complexity_index 单调残差 FAILED'

# ---- TC18: disordered_hubbard_parameters U 为正 ----
np.random.seed(42)
eps, U_vals = disordered_hubbard_parameters(10, W=2.0, U_base=4.0, U_var=0.5)
assert np.all(U_vals > 0), '[TC18] disordered_hubbard_parameters U 为正 FAILED'
assert len(eps) == 10, '[TC18] disordered_hubbard_parameters U 为正 FAILED'

# ---- TC19: thermal_spin_configuration 自旋单位模长 ----
np.random.seed(42)
spins = thermal_spin_configuration(10, beta=1.0)
norms = np.sqrt(np.sum(spins ** 2, axis=1))
assert np.allclose(norms, 1.0), '[TC19] thermal_spin_configuration 自旋单位模长 FAILED'

# ---- TC20: square_surface_sample 形状正确 ----
np.random.seed(42)
boundary_pts = square_surface_sample(20)
assert boundary_pts.shape == (20, 2), '[TC20] square_surface_sample 形状正确 FAILED'

# ---- TC21: DQMCConfig 参数计算正确 ----
cfg = DQMCConfig(nsites=4, beta=2.0, U=4.0, t=1.0, dtau=0.1)
assert cfg.L >= 1, '[TC21] DQMCConfig 参数计算正确 FAILED'
assert cfg.nsites == 4, '[TC21] DQMCConfig 参数计算正确 FAILED'

# ---- TC22: build_kinetic_matrix 对称且对角元为零 ----
K_mat = build_kinetic_matrix(4, [[1, 3], [0, 2], [1, 3], [0, 2]], 1.0)
assert np.allclose(K_mat, K_mat.T), '[TC22] build_kinetic_matrix 对称且对角元为零 FAILED'
assert np.allclose(np.diag(K_mat), 0.0), '[TC22] build_kinetic_matrix 对称且对角元为零 FAILED'

# ---- TC23: sawtooth_wave 值域约束 ----
t_vals = np.linspace(0, 10, 100)
saw_vals = np.array([sawtooth_wave(ti, 1.0) for ti in t_vals])
assert np.all(saw_vals >= -0.5) and np.all(saw_vals < 0.5), '[TC23] sawtooth_wave 值域约束 FAILED'

# ---- TC24: doublon_dynamics_hubbard 输出非负 ----
t_dd, y_dd = doublon_dynamics_hubbard(U=4.0, t_hop=1.0, beta=2.0, t_max=5.0)
assert np.all(y_dd >= 0), '[TC24] doublon_dynamics_hubbard 输出非负 FAILED'
assert t_dd[0] == 0.0, '[TC24] doublon_dynamics_hubbard 输出非负 FAILED'

# ---- TC25: compute_dos_tetrahedron 输出有限且长度正确 ----
lat4 = TriangularLattice(4, 4, a=1.0)
kpts = lat4.reciprocal_lattice_points()
t = 1.0
energies = np.zeros(len(kpts))
for i, k in enumerate(kpts):
    kx, ky = k
    energies[i] = -2.0 * t * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky))
omega_grid = np.linspace(-6.0, 3.0, 50)
dos = compute_dos_tetrahedron(kpts, energies, omega_grid, eta=0.1)
assert len(dos) == len(omega_grid), '[TC25] compute_dos_tetrahedron 输出有限且长度正确 FAILED'
assert np.all(np.isfinite(dos)), '[TC25] compute_dos_tetrahedron 输出有限且长度正确 FAILED'

# ---- TC26: trinity_triangle_tiling_brillouin_zone 返回正确形状 ----
triangles, weights = trinity_triangle_tiling_brillouin_zone(kpts, lat4.bz_vertices)
assert triangles.shape[1] == 3, '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'
assert len(weights) == len(triangles), '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'
assert np.isclose(np.sum(weights), 1.0), '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'

# ---- TC27: sawtooth_drive_matrix 对角结构 ----
np.random.seed(42)
V_list = sawtooth_drive_matrix(4, amplitude=1.0, omega=1.0, times=np.array([0.0, 1.0]))
assert len(V_list) == 2, '[TC27] sawtooth_drive_matrix 对角结构 FAILED'
assert np.allclose(V_list[0], np.diag(np.diag(V_list[0]))), '[TC27] sawtooth_drive_matrix 对角结构 FAILED'

# ---- TC28: build_hubbard_hamiltonian U=0 能谱对称中心在零 ----
H0 = build_hubbard_hamiltonian(2, [[1], [0]], 1.0, 0.0, 0.0)
evals0, _ = exact_diagonalization(H0)
assert np.isclose(np.sum(evals0), 0.0, atol=1e-10), '[TC28] build_hubbard_hamiltonian U=0 能谱对称中心在零 FAILED'

# ---- TC29: run_dqmc 输出结构正确 ----
np.random.seed(42)
cfg = DQMCConfig(nsites=4, beta=1.0, U=2.0, t=1.0, dtau=0.2)
res_dqmc = run_dqmc(cfg, [[1, 3], [0, 2], [1, 3], [0, 2]], n_warmup=10, n_measure=20)
assert 'double_occupancy' in res_dqmc, '[TC29] run_dqmc 输出结构正确 FAILED'
assert 'kinetic_energy' in res_dqmc, '[TC29] run_dqmc 输出结构正确 FAILED'
assert np.isfinite(res_dqmc['double_occupancy']), '[TC29] run_dqmc 输出结构正确 FAILED'
