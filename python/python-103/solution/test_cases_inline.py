np.random.seed(42)
from sparse_solver import ax_crs

# ---- TC01: jacobi_polynomial P0恒为1 ----
x_test = np.array([-0.5, 0.0, 0.5])
v_test = jacobi_polynomial(3, 0, -0.5, -0.5, x_test)
assert np.allclose(v_test[:, 0], 1.0), '[TC01] jacobi_polynomial P0恒为1 FAILED'

# ---- TC02: jacobi_quadrature_rule权重和等于零阶矩 ----
x_q, w_q = jacobi_quadrature_rule(8, -0.5, -0.5)
assert np.isclose(np.sum(w_q), np.pi), '[TC02] jacobi_quadrature_rule权重和 FAILED'

# ---- TC03: spectral_expand_pulse重构尺寸匹配 ----
t_test = np.linspace(-1, 1, 64)
A_test = np.exp(-(t_test / 0.3) ** 2)
coeffs, A_recon = spectral_expand_pulse(t_test, A_test, alpha_jac=-0.5, beta_jac=-0.5, n_modes=16)
assert coeffs.size == 16 and A_recon.size == t_test.size, '[TC03] spectral_expand_pulse重构尺寸匹配 FAILED'

# ---- TC04: dispersion_operator_spectral输出尺寸匹配 ----
coeffs_test = np.ones(8, dtype=complex)
disp_coeffs = dispersion_operator_spectral(coeffs_test, -0.5, -0.5, 8, -20e-27, 0.1e-39, 1.0)
assert disp_coeffs.size == 8, '[TC04] dispersion_operator_spectral输出尺寸匹配 FAILED'

# ---- TC05: triangle_area解析值验证 ----
area = triangle_area(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0]))
assert np.isclose(area, 1.0), '[TC05] triangle_area解析值验证 FAILED'

# ---- TC06: create_fiber_triangulation输出结构正确 ----
nodes, triangles, boundary_flags = create_fiber_triangulation(2e-6, 10e-6, n_theta=8, n_radial_core=2, n_radial_clad=2)
assert nodes.ndim == 2 and nodes.shape[1] == 2, '[TC06] create_fiber_triangulation输出结构正确 FAILED'
assert triangles.ndim == 2 and triangles.shape[1] == 3, '[TC06] create_fiber_triangulation输出结构正确 FAILED'
assert boundary_flags.size == nodes.shape[0], '[TC06] create_fiber_triangulation输出结构正确 FAILED'

# ---- TC07: identify_boundary_nodes检测到外边界 ----
boundary_nodes = identify_boundary_nodes(triangles, nodes.shape[0])
assert np.sum(boundary_nodes) >= 8, '[TC07] identify_boundary_nodes检测到外边界 FAILED'

# ---- TC08: compute_nonlinear_coefficient公式解析验证 ----
gamma_test = compute_nonlinear_coefficient(2.6e-20, 1.0e15, 1.0e-12)
c_light = 2.99792458e8
expected_gamma = 2.6e-20 * 1.0e15 / (c_light * 1.0e-12)
assert np.isclose(gamma_test, expected_gamma), '[TC08] compute_nonlinear_coefficient公式解析验证 FAILED'

# ---- TC09: pwl_product_integral常数函数解析验证 ----
x_pwl = np.linspace(0.0, 1.0, 5)
f_pwl = np.ones(5) * 3.0
g_pwl = np.ones(5) * 4.0
integral = pwl_product_integral(0.0, 1.0, x_pwl, f_pwl, x_pwl, g_pwl)
assert np.isclose(integral, 12.0), '[TC09] pwl_product_integral常数函数解析验证 FAILED'

# ---- TC10: pulse_inner_product交换对称性 ----
t_pwl = np.linspace(-1e-12, 1e-12, 64)
A1 = np.exp(-(t_pwl / 0.5e-12) ** 2)
A2 = np.exp(-((t_pwl - 0.2e-12) / 0.6e-12) ** 2)
ip12 = pulse_inner_product(t_pwl, A1 + 0j, A2 + 0j)
ip21 = pulse_inner_product(t_pwl, A2 + 0j, A1 + 0j)
assert np.isclose(ip12, np.conj(ip21)), '[TC10] pulse_inner_product交换对称性 FAILED'

# ---- TC11: raman_response_convolution输出尺寸匹配 ----
t_r = np.linspace(-1e-12, 3e-12, 256)
A_r = np.exp(-(t_r / 0.5e-12) ** 2) + 0j
h_R = np.ones_like(t_r)
conv = raman_response_convolution(t_r, A_r, h_R)
assert conv.size == t_r.size, '[TC11] raman_response_convolution输出尺寸匹配 FAILED'

# ---- TC12: hyperball01_sample样本在单位球内 ----
np.random.seed(42)
samples = hyperball01_sample(3, 500)
normss = np.linalg.norm(samples, axis=0)
assert np.all(normss <= 1.0 + 1e-12), '[TC12] hyperball01_sample样本在单位球内 FAILED'

# ---- TC13: sphere01_quad_llm常数函数积分接近4π ----
integral, n_eval = sphere01_quad_llm(lambda x: 1.0, h=0.5)
assert np.isclose(integral, 4.0 * np.pi, atol=0.5), '[TC13] sphere01_quad_llm常数函数积分接近4π FAILED'

# ---- TC14: brownian_motion_simulation固定种子可复现 ----
traj1 = brownian_motion_simulation(2, 101, 1e-3, 1.0, seed=42)
traj2 = brownian_motion_simulation(2, 101, 1e-3, 1.0, seed=42)
assert np.allclose(traj1, traj2), '[TC14] brownian_motion_simulation固定种子可复现 FAILED'

# ---- TC15: generate_ase_noise G<=1时返回全零 ----
t_ase = np.linspace(0.0, 1e-12, 64)
noise = generate_ase_noise(t_ase, n_sp=2.0, G=1.0, h_nu=1e-19, bw=1e12, seed=42)
assert np.all(noise == 0.0), '[TC15] generate_ase_noise G<=1时返回全零 FAILED'

# ---- TC16: bose_einstein_distribution概率和为1 ----
probs, n_arr = bose_einstein_distribution(n_avg=3.0, n_max=30)
assert np.isclose(np.sum(probs), 1.0), '[TC16] bose_einstein_distribution概率和为1 FAILED'

# ---- TC17: ax_crs稀疏矩阵向量乘解析验证 ----
a_crs = np.array([1.0, 2.0, 3.0])
ia_crs = np.array([0, 1, 1])
ja_crs = np.array([0, 0, 1])
x_vec = np.array([1.0, 1.0])
y_vec = ax_crs(a_crs, ia_crs, ja_crs, x_vec, 2, 3)
assert np.allclose(y_vec, np.array([1.0, 5.0])), '[TC17] ax_crs稀疏矩阵向量乘解析验证 FAILED'

# ---- TC18: mgmres求解三对角系统精确 ----
n = 32
h = 1.0 / (n + 1)
rows = []
cols = []
vals = []
for i in range(n):
    rows.append(i); cols.append(i); vals.append(2.0 / (h ** 2) + 0.1)
    if i > 0:
        rows.append(i); cols.append(i - 1); vals.append(-1.0 / (h ** 2))
    if i < n - 1:
        rows.append(i); cols.append(i + 1); vals.append(-1.0 / (h ** 2))
nz = len(vals)
a_crs_m = np.array(vals, dtype=float)
ia_crs_m = np.array(rows, dtype=int)
ja_crs_m = np.array(cols, dtype=int)
x_exact = np.sin(np.pi * np.linspace(0, 1, n))
rhs = ax_crs(a_crs_m, ia_crs_m, ja_crs_m, x_exact, n, nz)
x0 = np.zeros(n)
x_sol = mgmres(a_crs_m, ia_crs_m, ja_crs_m, x0, rhs, n, nz, itr_max=100, mr=20, tol_abs=1e-12, tol_rel=1e-10, verbose=False)
rel_err = np.linalg.norm(x_sol - x_exact) / np.linalg.norm(x_exact)
assert rel_err < 1e-6, '[TC18] mgmres求解三对角系统精确 FAILED'

# ---- TC19: build_dispersion_matrix_crs输出结构正确 ----
a_d, ia_d, ja_d, nz_d = build_dispersion_matrix_crs(16, 1e-15, -20e-27, 0.1e-39, beta4=0.0)
assert nz_d > 0 and a_d.size == nz_d, '[TC19] build_dispersion_matrix_crs输出结构正确 FAILED'

# ---- TC20: magic_matrix幻和正确 ----
M = magic_matrix(5)
magic_sum = np.sum(M[0, :])
assert np.allclose(np.sum(M, axis=1), magic_sum) and np.allclose(np.sum(M, axis=0), magic_sum), '[TC20] magic_matrix幻和正确 FAILED'

# ---- TC21: caesar_shift_phase能量守恒 ----
np.random.seed(42)
spectrum = np.random.randn(64) + 1j * np.random.randn(64)
shifted = caesar_shift_phase(spectrum, 8)
assert np.isclose(np.sum(np.abs(spectrum) ** 2), np.sum(np.abs(shifted) ** 2)), '[TC21] caesar_shift_phase能量守恒 FAILED'

# ---- TC22: four_fifths_search返回有效组合 ----
result = four_fifths_search(20, exponent=2)
assert result is not None and len(result) == 5, '[TC22] four_fifths_search返回有效组合 FAILED'

# ---- TC23: wdm_channel_search返回正确长度 ----
channels, fwm = wdm_channel_search(n_channels=4)
assert len(channels) == 4 and fwm >= 0.0, '[TC23] wdm_channel_search返回正确长度 FAILED'

# ---- TC24: raman_response_function因果性验证 ----
t_raman = np.linspace(-1e-12, 3e-12, 512)
h_r = raman_response_function(t_raman)
assert np.all(h_r[t_raman < 0] == 0.0), '[TC24] raman_response_function因果性验证 FAILED'

# ---- TC25: dispersion_operator零频验证 ----
omega_test = np.zeros(10)
D_test = dispersion_operator(omega_test, 0.2, -20e-27, 0.1e-39)
assert np.allclose(D_test, -0.1), '[TC25] dispersion_operator零频验证 FAILED'

# ---- TC26: soliton_order输出有限正数 ----
t_sol = np.linspace(-5e-12, 5e-12, 256)
A0_sol = np.sqrt(1000.0) / np.cosh(t_sol / 100e-15)
N_sol, L_D, L_NL = soliton_order(A0_sol, t_sol, 1.5e-3, -20e-27)
assert np.isfinite(N_sol) and N_sol > 0.0, '[TC26] soliton_order输出有限正数 FAILED'

# ---- TC27: temporal_width零输入返回0 ----
t_zero = np.zeros(64)
w_zero = temporal_width(t_zero, t_zero)
assert w_zero == 0.0, '[TC27] temporal_width零输入返回0 FAILED'

# ---- TC28: spectral_width对称脉冲非零 ----
t_sym = np.linspace(-5e-12, 5e-12, 256)
A_sym = np.exp(-(t_sym / 1e-12) ** 2)
sw = spectral_width(t_sym, A_sym)
assert sw > 0.0 and np.isfinite(sw), '[TC28] spectral_width对称脉冲非零 FAILED'

# ---- TC29: ssfm_solve基本传播输出有限 ----
t_ss = np.linspace(-2e-12, 2e-12, 256)
A0_ss = np.exp(-(t_ss / 0.5e-12) ** 2) + 0j
A_final, z_hist, A_hist = ssfm_solve(A0_ss, t_ss, 0.01, 10, 0.2e-3, -20e-27, 0.1e-39, 1.5e-3)
assert A_final.size == A0_ss.size and np.all(np.isfinite(A_final)), '[TC29] ssfm_solve基本传播输出有限 FAILED'

# ---- TC30: dream_mcmc输出尺寸正确 ----
np.random.seed(42)
logL_test = lambda p: -0.5 * np.sum(p ** 2)
logP_test = lambda p: 0.0 if np.all((p >= -1.0) & (p <= 1.0)) else -np.inf
z, fit, rate = dream_mcmc(logL_test, logP_test, par_num=2, chain_num=3, gen_num=20, limits=np.array([[-1.0, -1.0], [1.0, 1.0]]), seed=42)
assert z.shape == (2, 3, 20), '[TC30] dream_mcmc输出尺寸正确 FAILED'
assert 0.0 <= rate <= 1.0, '[TC30] dream_mcmc接受率范围 FAILED'
