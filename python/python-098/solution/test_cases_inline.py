# ---- TC01: horner_eval 验证多项式 p(x)=1-2x+3x^2-0.5x^3 在 x=2 的值 ----
val = horner_eval([1.0, -2.0, 3.0, -0.5], 2.0)
expected = 1.0 - 2.0*2.0 + 3.0*(2.0**2) - 0.5*(2.0**3)
assert abs(val - expected) < 1e-12, '[TC01] horner_eval 多项式求值 FAILED'

# ---- TC02: legendre_polynomials_array 输出形状为 (m, n+1) ----
x_arr = np.array([-0.5, 0.0, 0.5])
Px = legendre_polynomials_array(5, x_arr)
assert Px.shape == (3, 6), '[TC02] legendre_polynomials_array 输出形状 FAILED'

# ---- TC03: design_hologram_phase_2d 输出尺寸匹配输入网格 ----
x_g = np.linspace(-1e-6, 1e-6, 8)
y_g = np.linspace(-1e-6, 1e-6, 6)
coeffs = np.array([[1.0, 0.5], [0.3, 0.0]])
phase_2d = design_hologram_phase_2d(x_g, y_g, coeffs, Lx=2e-6, Ly=2e-6)
assert phase_2d.shape == (8, 6), '[TC03] design_hologram_phase_2d 输出尺寸 FAILED'

# ---- TC04: chebyshev_nodes 关于原点对称 ----
nodes = chebyshev_nodes(8)
assert np.allclose(nodes, -nodes[::-1], atol=1e-12), '[TC04] chebyshev_nodes 对称性 FAILED'

# ---- TC05: square01_monomial_integral [2,3] 解析值为 1/12 ----
val = square01_monomial_integral([2, 3])
assert abs(val - 1.0/12.0) < 1e-12, '[TC05] square01_monomial_integral 解析验证 FAILED'

# ---- TC06: squaresym_monomial_integral 奇指数积分为零 ----
val = squaresym_monomial_integral([1, 2])
assert val == 0.0, '[TC06] squaresym_monomial_integral 奇函数为零 FAILED'

# ---- TC07: integrate_2d_gauss_legendre 常数函数积分等于面积 ----
val = integrate_2d_gauss_legendre(lambda x, y: 2.0, xlim=(0.0, 3.0), ylim=(0.0, 4.0), n=4)
assert abs(val - 24.0) < 1e-12, '[TC07] integrate_2d_gauss_legendre 常数函数 FAILED'

# ---- TC08: lloyd_relaxation_square 固定种子结果可复现 ----
np.random.seed(42)
g1 = lloyd_relaxation_square(n_generators=16, n_steps=5, seed=42)
np.random.seed(42)
g2 = lloyd_relaxation_square(n_generators=16, n_steps=5, seed=42)
assert np.allclose(g1, g2, atol=1e-12), '[TC08] lloyd_relaxation_square 可复现性 FAILED'

# ---- TC09: compute_voronoi_areas_square 面积和近似等于总面积 ----
np.random.seed(42)
gens = lloyd_relaxation_square(n_generators=16, n_steps=3, seed=42)
areas = compute_voronoi_areas_square(gens, n_samples=50000, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
assert abs(np.sum(areas) - 4.0) < 0.1, '[TC09] compute_voronoi_areas_square 面积和 FAILED'

# ---- TC10: spherical_harmonic_y Y_0^0 为常数 1/sqrt(4π) ----
y00 = spherical_harmonic_y(0, 0, 0.5, 0.0)
assert abs(y00 - 1.0/np.sqrt(4.0*np.pi)) < 1e-12, '[TC10] spherical_harmonic_y Y_0^0 FAILED'

# ---- TC11: scattering_coefficients_mie ka=0 时系数全为零 ----
a_mie = scattering_coefficients_mie(3, 1.0, 0.0, 3.48**2, 1.0)
assert np.allclose(a_mie, 0.0, atol=1e-12), '[TC11] scattering_coefficients_mie 零半径 FAILED'

# ---- TC12: effective_medium_profile 线性剖面端点值正确 ----
n0 = effective_medium_profile(0.0, 3.48, 1.0, 1.0, 'linear')
n1 = effective_medium_profile(1.0, 3.48, 1.0, 1.0, 'linear')
assert abs(n0 - 1.0) < 1e-12 and abs(n1 - 3.48) < 1e-12, '[TC12] effective_medium_profile 端点值 FAILED'

# ---- TC13: pendulum_period_small_angle 解析公式验证 ----
T = pendulum_period_small_angle(g=9.81, l=0.1)
T_expected = 2.0 * np.pi * np.sqrt(0.1 / 9.81)
assert abs(T - T_expected) < 1e-12, '[TC13] pendulum_period_small_angle 解析验证 FAILED'

# ---- TC14: pendulum_period_elliptic 大角度周期大于小角度周期 ----
T_small = pendulum_period_small_angle(g=9.81, l=0.1)
T_large = pendulum_period_elliptic(theta0=np.pi/3.0, g=9.81, l=0.1)
assert T_large > T_small, '[TC14] pendulum_period_elliptic 大角度周期 FAILED'

# ---- TC15: effective_phase_shift_nonlinear 零光强等于线性相位 ----
params = {'omega0': 1.0, 'omega': 1.0, 'gamma': 0.05, 'kappa': 0.1, 'I_sat': 1.0}
phi0 = effective_phase_shift_nonlinear(0.0, params)
detuning = 1.0**2 - 1.0**2
phi_linear = np.arctan2(detuning, 0.05 * 1.0)
assert abs(phi0 - phi_linear) < 1e-12, '[TC15] effective_phase_shift_nonlinear 零光强 FAILED'

# ---- TC16: duffing_resonator 返回二维状态向量 ----
dy = duffing_resonator(0.0, np.array([0.1, 0.0]), alpha=1.0, beta=0.1, gamma=0.05, omega=1.0, F=1.0)
assert dy.shape[0] == 2, '[TC16] duffing_resonator 返回维度 FAILED'

# ---- TC17: rk4_integrate 常数ODE du/dt=3 精确积分 ----
t, y = rk4_integrate(lambda t, y: np.array([3.0]), (0.0, 1.0), np.array([0.0]), n_steps=10)
assert abs(y[-1, 0] - 3.0) < 1e-12, '[TC17] rk4_integrate 常数ODE FAILED'

# ---- TC18: rk23_integrate 误差估计非负 ----
def f_rk23(t, y):
    return np.array([-y[0]])
t, y, e = rk23_integrate(f_rk23, (0.0, 1.0), np.array([1.0]), n_steps=20)
assert np.all(e >= -1e-15), '[TC18] rk23_integrate 误差非负 FAILED'

# ---- TC19: angular_spectrum_propagate 输出形状与输入一致 ----
field = np.ones((16, 16), dtype=complex)
out = angular_spectrum_propagate(field, k0=1.0, z=1.0, dx=0.1, dy=0.1)
assert out.shape == field.shape, '[TC19] angular_spectrum_propagate 输出形状 FAILED'

# ---- TC20: build_scattering_operator 输出矩阵维度为 N×N ----
phase_p = np.zeros((8, 8))
S, x_c = build_scattering_operator(n_pixels=8, aperture_size=10e-6, wavelength=633e-9, phase_profile=phase_p)
assert S.shape == (64, 64), '[TC20] build_scattering_operator 输出维度 FAILED'

# ---- TC21: svd_compress_scattering 高秩近似误差不大于低秩 ----
np.random.seed(42)
S_test = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
_, _, _, _, m64 = svd_compress_scattering(S_test, rank=64)
_, _, _, _, m8 = svd_compress_scattering(S_test, rank=8)
assert m64['relative_error'] <= m8['relative_error'] + 1e-12, '[TC21] svd_compress_scattering 误差单调性 FAILED'

# ---- TC22: apply_scattering_operator 满足线性 S*(a*u) = a*S*u ----
np.random.seed(42)
S_test = np.random.randn(16, 16) + 1j * np.random.randn(16, 16)
u = np.random.randn(16) + 1j * np.random.randn(16)
a = 2.0 + 1.0j
lhs = apply_scattering_operator(S_test, a * u)
rhs = a * apply_scattering_operator(S_test, u)
assert np.allclose(lhs, rhs, atol=1e-12), '[TC22] apply_scattering_operator 线性 FAILED'

# ---- TC23: gerchberg_saxton_iteration 误差单调不增 ----
target = np.ones((8, 8), dtype=float)
init = np.zeros((8, 8), dtype=float)
phase_gs, err_hist = gerchberg_saxton_iteration(target, init, n_iter=10)
assert all(err_hist[i] >= err_hist[i+1] - 1e-12 for i in range(len(err_hist)-1)), '[TC23] gerchberg_saxton_iteration 单调性 FAILED'

# ---- TC24: generate_unit_cube_mesh 节点数等于 nx*ny*nz ----
nodes, elements, elem_types = generate_unit_cube_mesh(nx=3, ny=4, nz=5)
assert nodes.shape[0] == 3*4*5, '[TC24] generate_unit_cube_mesh 节点数 FAILED'

# ---- TC25: detect_dimension 正确识别3D点云 ----
pts_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 1.0]])
dim = detect_dimension(pts_3d)
assert dim == 3, '[TC25] detect_dimension 3D识别 FAILED'

# ---- TC26: compute_point_cloud_stats 空点云返回零重心 ----
stats = compute_point_cloud_stats(np.zeros((0, 3)))
assert np.allclose(stats['centroid'], np.zeros(3), atol=1e-12), '[TC26] compute_point_cloud_stats 空点云 FAILED'

# ---- TC27: craps_exact_probability 精确值为 244/495 ----
p = craps_exact_probability()
assert abs(p - 244.0/495.0) < 1e-12, '[TC27] craps_exact_probability 精确值 FAILED'

# ---- TC28: binary_to_phase_config 与 phase_to_binary_config 二电平可逆 ----
np.random.seed(42)
phases = np.random.uniform(-np.pi, np.pi, 20)
config = phase_to_binary_config(phases, phase_levels=2)
phases_back = binary_to_phase_config(config, phase_levels=2)
config2 = phase_to_binary_config(phases_back, phase_levels=2)
assert np.array_equal(config, config2), '[TC28] phase_to_binary 可逆性 FAILED'

# ---- TC29: validate_hologram_config 全零配置满足默认约束 ----
cfg = np.zeros((8, 8), dtype=int)
valid, violations = validate_hologram_config(cfg, constraints={'max_ones': 100, 'min_ones': 0})
assert valid and len(violations) == 0, '[TC29] validate_hologram_config 全零配置 FAILED'

# ---- TC30: verify_monomial_integrals 数值积分高精度验证 ----
max_err = verify_monomial_integrals(max_order=4)
assert max_err < 1e-12, '[TC30] verify_monomial_integrals 精度 FAILED'
