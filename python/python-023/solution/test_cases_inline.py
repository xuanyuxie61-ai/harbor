from dispersion_relation import plasma_dispersion_function, d_plasma_dispersion_function, solve_whistler_dispersion
from distribution_models import kappa_distribution_3d, escape_probability, survival_probability, noncentral_beta_tail
from fractal_magnetic_field import generate_fractal_flux_tubes, compute_fractal_dimension, map_fractal_to_magnetic_field
from matrix_exponential_solver import matrix_exponential_pade, evolve_diffusion_operator
from moment_integrator import integrate_2d_velocity_space
from particle_orbit import boris_push
from pce_expansion import legendre_polynomial_normalized, enumerate_multi_indices
from phase_space_lagrange import barycentric_interpolate, chebyshev_nodes, lagrange_phase_space_reconstruction
from resonance_voronoi import detect_resonant_particles, voronoi_nearest_neighbor
from sparse_assembler import coo_to_csr, csr_matvec, estimate_condition_number
from file_sequence_processor import generate_filename_sequence, compute_temporal_correlation
from quasilinear_diffusion import assemble_ql_diffusion_matrix, compute_ql_diffusion_coefficients

# ---- TC01: plasma_dispersion_function 零输入返回有限复数 ----
z0 = plasma_dispersion_function(0.0)
assert np.isfinite(z0), '[TC01] plasma_dispersion_function 零输入返回有限复数 FAILED'

# ---- TC02: plasma_dispersion_function 大参数使用渐进展开 ----
z_large = plasma_dispersion_function(100.0 + 0j)
assert np.isfinite(z_large), '[TC02] plasma_dispersion_function 大参数使用渐进展开 FAILED'

# ---- TC03: d_plasma_dispersion_function 满足导数恒等式 Zp=-2(1+z*Z) ----
z_test = 0.5 + 0.5j
Zp = d_plasma_dispersion_function(z_test)
Z = plasma_dispersion_function(z_test)
assert np.abs(Zp + 2.0*(1.0 + z_test*Z)) < 1e-10, '[TC03] d_plasma_dispersion_function 满足导数恒等式 FAILED'

# ---- TC04: solve_whistler_dispersion 返回有限复频率 ----
test_params = {'q_e': 1.602176634e-19, 'm_e': 9.10938356e-31, 'c': 2.99792458e8, 'eps0': 8.854187817e-12, 'B0': 100e-9, 'n0': 1e7, 'Omega_e': 1.7588e4, 'omega_pe': 5.64e7, 'v_te': 5.93e6}
omega_sol = solve_whistler_dispersion(1e-2, test_params)
assert omega_sol is None or (np.isscalar(omega_sol) and np.isfinite(omega_sol.real) and np.isfinite(omega_sol.imag)), '[TC04] solve_whistler_dispersion 返回有限复频率 FAILED'

# ---- TC05: kappa_distribution_3d 输出非负 ----
vx = np.array([0.0, 1e6, -1e6])
vy = np.array([0.0, 0.0, 1e6])
vz = np.array([0.0, 0.0, 0.0])
f_kappa = kappa_distribution_3d(vx, vy, vz, n0=1e7, v_th=5e6, kappa=4.0)
assert np.all(f_kappa >= 0), '[TC05] kappa_distribution_3d 输出非负 FAILED'

# ---- TC06: escape_probability t=0 时返回 0 ----
assert escape_probability(0.0, 1.0) == 0.0, '[TC06] escape_probability t=0 时返回 0 FAILED'

# ---- TC07: survival_probability t=0 时返回 1 ----
assert survival_probability(0.0, 1.0) == 1.0, '[TC07] survival_probability t=0 时返回 1 FAILED'

# ---- TC08: generate_fractal_flux_tubes 输出形状为 (n_points, 3) ----
np.random.seed(42)
points = generate_fractal_flux_tubes(100)
assert points.shape == (100, 3), '[TC08] generate_fractal_flux_tubes 输出形状为 (n_points, 3) FAILED'

# ---- TC09: compute_fractal_dimension 返回值在 [0, 3] 范围内 ----
np.random.seed(42)
pts = generate_fractal_flux_tubes(500)
D_box = compute_fractal_dimension(pts)
assert 0.0 <= D_box <= 3.0, '[TC09] compute_fractal_dimension 返回值在 [0, 3] 范围内 FAILED'

# ---- TC10: matrix_exponential_pade 零矩阵返回单位矩阵 ----
A_zero = np.zeros((3, 3))
E = matrix_exponential_pade(A_zero)
assert np.allclose(E, np.eye(3)), '[TC10] matrix_exponential_pade 零矩阵返回单位矩阵 FAILED'

# ---- TC11: matrix_exponential_pade 对角矩阵返回 diag(exp(d)) ----
A_diag = np.diag([1.0, 2.0, 3.0])
E_diag = matrix_exponential_pade(A_diag)
expected = np.diag(np.exp([1.0, 2.0, 3.0]))
assert np.allclose(E_diag, expected, atol=1e-8), '[TC11] matrix_exponential_pade 对角矩阵返回 diag(exp(d)) FAILED'

# ---- TC12: evolve_diffusion_operator 零算子保持初始条件不变 ----
f0 = np.array([1.0, 2.0, 3.0])
A0 = np.zeros((3, 3))
f_result = evolve_diffusion_operator(A0, f0, 0.1, 5)
assert np.allclose(f_result, f0), '[TC12] evolve_diffusion_operator 零算子保持初始条件不变 FAILED'

# ---- TC13: integrate_2d_velocity_space 常数正被积函数积分结果为正 ----
v_p = np.linspace(0.0, 1.0, 5)
v_pl = np.linspace(-1.0, 1.0, 5)
VP_mesh, VPL_mesh = np.meshgrid(v_p, v_pl, indexing='ij')
integrand = np.ones_like(VP_mesh) * VP_mesh
result = integrate_2d_velocity_space(v_pl, v_p, integrand)
assert result > 0 and np.isfinite(result), '[TC13] integrate_2d_velocity_space 常数正被积函数积分结果为正 FAILED'

# ---- TC14: boris_push 纯磁场中保持速度模长守恒 ----
x = np.array([0.0, 0.0, 0.0])
v = np.array([1e6, 0.0, 0.0])
B = np.array([0.0, 0.0, 100e-9])
E = np.array([0.0, 0.0, 0.0])
x_new, v_new = boris_push(x, v, 1.602176634e-19, 9.10938356e-31, B, E, 1e-9)
assert np.abs(np.linalg.norm(v_new) - np.linalg.norm(v)) < 1e-3 * np.linalg.norm(v), '[TC14] boris_push 纯磁场中保持速度模长守恒 FAILED'

# ---- TC15: legendre_polynomial_normalized n=0 时恒为 sqrt(0.5) ----
val = legendre_polynomial_normalized(0, 0.5)
assert np.abs(val - np.sqrt(0.5)) < 1e-10, '[TC15] legendre_polynomial_normalized n=0 时恒为 sqrt(0.5) FAILED'

# ---- TC16: enumerate_multi_indices N=2 P=3 返回 4 个和恰好为 3 的多指标 ----
indices = enumerate_multi_indices(2, 3)
assert len(indices) == 4, '[TC16] enumerate_multi_indices N=2 P=3 返回 4 个和恰好为 3 的多指标 FAILED'

# ---- TC17: barycentric_interpolate 在节点上精确恢复函数值 ----
nodes = np.array([0.0, 1.0, 2.0, 3.0])
values = np.array([1.0, 3.0, 5.0, 7.0])
val_at_node = barycentric_interpolate(nodes, values, 1.0)
assert np.abs(val_at_node - 3.0) < 1e-10, '[TC17] barycentric_interpolate 在节点上精确恢复函数值 FAILED'

# ---- TC18: chebyshev_nodes 生成 n 个点且全部落在 [a, b] 区间内 ----
cheb = chebyshev_nodes(-1.0, 1.0, 8)
assert len(cheb) == 8 and np.all(cheb >= -1.0) and np.all(cheb <= 1.0), '[TC18] chebyshev_nodes 生成 n 个点且全部落在 [a, b] 区间内 FAILED'

# ---- TC19: detect_resonant_particles 空色散解返回空列表 ----
v_grid_test = np.array([[1e6, 0.0, 1e6], [0.0, 0.0, -1e6]])
empty_omega = np.array([])
res_empty = detect_resonant_particles(v_grid_test, empty_omega, test_params)
assert len(res_empty) == 0, '[TC19] detect_resonant_particles 空色散解返回空列表 FAILED'

# ---- TC20: voronoi_nearest_neighbor 返回距离非负 ----
qpts = np.array([[0.0, 0.0], [1.0, 1.0]])
ctr = np.array([[0.0, 0.0], [2.0, 2.0], [0.5, 0.5]])
idx, dists = voronoi_nearest_neighbor(qpts, ctr)
assert np.all(dists >= 0), '[TC20] voronoi_nearest_neighbor 返回距离非负 FAILED'

# ---- TC21: coo_to_csr 转换后 csr_matvec 与稠密乘法结果一致 ----
data = np.array([1.0, 2.0, 3.0])
row = np.array([0, 1, 2])
col = np.array([0, 1, 2])
csr_data, csr_indices, csr_indptr = coo_to_csr(data, row, col, 3, 3)
x_vec = np.array([1.0, 2.0, 3.0])
y_csr = csr_matvec(csr_data, csr_indices, csr_indptr, x_vec)
y_dense = np.diag([1.0, 2.0, 3.0]) @ x_vec
assert np.allclose(y_csr, y_dense), '[TC21] coo_to_csr 转换后 csr_matvec 与稠密乘法结果一致 FAILED'

# ---- TC22: estimate_condition_number 单位矩阵条件数接近 1 ----
np.random.seed(42)
cond_I = estimate_condition_number(np.eye(4), n_iter=5)
assert np.abs(cond_I - 1.0) < 0.1, '[TC22] estimate_condition_number 单位矩阵条件数接近 1 FAILED'

# ---- TC23: generate_filename_sequence 生成正确数量的字符串文件名 ----
fnames = generate_filename_sequence("test.dat", 5)
assert len(fnames) == 5 and all(isinstance(f, str) for f in fnames), '[TC23] generate_filename_sequence 生成正确数量的字符串文件名 FAILED'

# ---- TC24: compute_temporal_correlation lag=0 时自相关严格为 1 ----
np.random.seed(42)
series = np.random.randn(20)
corr = compute_temporal_correlation(series)
assert np.abs(corr[0] - 1.0) < 1e-10, '[TC24] compute_temporal_correlation lag=0 时自相关严格为 1 FAILED'

# ---- TC25: noncentral_beta_tail 输出非负且有限 ----
v_test = np.linspace(0.0, 1e7, 10)
tail_vals = noncentral_beta_tail(v_test, v_max=1e7)
assert np.all(tail_vals >= 0) and np.all(np.isfinite(tail_vals)), '[TC25] noncentral_beta_tail 输出非负且有限 FAILED'

# ---- TC26: map_fractal_to_magnetic_field 返回可调用对象且输出为 3 维向量 ----
np.random.seed(42)
pts_frac = generate_fractal_flux_tubes(50)
B_func = map_fractal_to_magnetic_field(pts_frac, 100e-9)
B_at_origin = B_func(np.array([0.0, 0.0, 0.0]))
assert callable(B_func) and B_at_origin.shape == (3,), '[TC26] map_fractal_to_magnetic_field 返回可调用对象且输出为 3 维向量 FAILED'

# ---- TC27: assemble_ql_diffusion_matrix 输出矩阵与向量形状正确 ----
v_p_test = np.linspace(0.01, 1e7, 4)
v_pl_test = np.linspace(-1e7, 1e7, 4)
omega_test = np.array([[1e-5, complex(1e4, -1e2)]])
A_mat, rhs_vec = assemble_ql_diffusion_matrix(v_pl_test, v_p_test, omega_test, test_params)
assert A_mat.shape == (16, 16) and rhs_vec.shape == (16,), '[TC27] assemble_ql_diffusion_matrix 输出矩阵与向量形状正确 FAILED'

# ---- TC28: compute_ql_diffusion_coefficients 扩散系数非负 ----
D_par, D_perp, D_cross = compute_ql_diffusion_coefficients(v_pl_test, v_p_test, omega_test, test_params)
assert np.all(D_par >= 0) and np.all(D_perp >= 0), '[TC28] compute_ql_diffusion_coefficients 扩散系数非负 FAILED'

# ---- TC29: lagrange_phase_space_reconstruction 输出非负 ----
np.random.seed(42)
f_grid_test = np.random.rand(4, 4)
f_rec = lagrange_phase_space_reconstruction(v_pl_test, v_p_test, f_grid_test, test_params)
assert np.all(f_rec >= 0), '[TC29] lagrange_phase_space_reconstruction 输出非负 FAILED'

# ---- TC30: run_simulation 完整流程返回包含 moments 的字典 ----
results = run_simulation()
assert isinstance(results, dict) and 'moments' in results and 'D_matrix' in results, '[TC30] run_simulation 完整流程返回包含 moments 的字典 FAILED'
