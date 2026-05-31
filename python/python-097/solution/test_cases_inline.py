# ---- TC01: generate_prime_grid_steps返回值均为大于1的整数 ----
nx_p, ny_p, nz_p = generate_prime_grid_steps(20, 20, 20)
for p in [nx_p, ny_p, nz_p]:
    has_divisor = any(p % d == 0 for d in range(2, int(np.sqrt(p)) + 1))
    assert p > 1 and not has_divisor, '[TC01] generate_prime_grid_steps返回值均为大于1的整数 FAILED'

# ---- TC02: rms_error对零差异返回0 ----
arr = np.array([1.0, 2.0, 3.0, 4.0])
assert rms_error(arr, arr) == 0.0, '[TC02] rms_error对零差异返回0 FAILED'

# ---- TC03: convergence_rate估计二阶收敛 ----
errors = [0.04, 0.01, 0.0025]
res = [0.2, 0.1, 0.05]
rates = convergence_rate(errors, res)
assert len(rates) == 2, '[TC03] convergence_rate估计二阶收敛 FAILED'
assert abs(rates[0] - 2.0) < 0.1, '[TC03] convergence_rate估计二阶收敛 FAILED'

# ---- TC04: check_energy_conservation对恒定能量返回通过 ----
W_const = [1.0, 1.0, 1.0, 1.0]
t_const = [0.0, 1.0, 2.0, 3.0]
P_const = [0.0, 0.0, 0.0]
ec = check_energy_conservation(W_const, t_const, P_const, tol=1e-2)
assert ec['conserved'] == True, '[TC04] check_energy_conservation对恒定能量返回通过 FAILED'

# ---- TC05: quality_factor对正损耗返回有限正Q ----
Q_val = quality_factor(1e9, 1e-6, 1e-3)
assert np.isfinite(Q_val) and Q_val > 0.0, '[TC05] quality_factor对正损耗返回有限正Q FAILED'

# ---- TC06: wavenumber_frequency_relation解析验证 ----
k_test = wavenumber_frequency_relation(2.0*np.pi*1e9, EPSILON_0, MU_0)
k_expected = 2.0*np.pi*1e9 / C_0
assert abs(k_test - k_expected) < 1e-3, '[TC06] wavenumber_frequency_relation解析验证 FAILED'

# ---- TC07: generate_rectangular_grid返回有效网格对象 ----
grid_test = generate_rectangular_grid(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5)
assert grid_test.dx > 0 and grid_test.cell_volume() > 0, '[TC07] generate_rectangular_grid返回有效网格对象 FAILED'

# ---- TC08: CylindricalCavity体积与表面积公式验证 ----
cav = CylindricalCavity(radius=0.05, height=0.10)
V = cav.volume()
A = cav.surface_area()
V_exp = np.pi * 0.05**2 * 0.10
A_exp = 2.0*np.pi*0.05**2 + 2.0*np.pi*0.05*0.10
assert abs(V - V_exp) < 1e-12 and abs(A - A_exp) < 1e-12, '[TC08] CylindricalCavity体积与表面积公式验证 FAILED'

# ---- TC09: CylindricalCavity TM010截止频率为正值 ----
f_tm010 = cav.tm_cutoff_frequency(0, 1, 0)
assert f_tm010 > 0.0, '[TC09] CylindricalCavity TM010截止频率为正值 FAILED'

# ---- TC10: CircleSegmentDielectric面积公式验证 ----
seg = CircleSegmentDielectric(r=0.04, theta=np.pi/3.0, height=0.05, epsilon_r=10.0)
A_seg = seg.area()
A_seg_exp = 0.04**2 * (np.pi/3.0 - np.sin(np.pi/3.0)) / 2.0
assert abs(A_seg - A_seg_exp) < 1e-15, '[TC10] CircleSegmentDielectric面积公式验证 FAILED'

# ---- TC11: lagrange_value_1d对常数函数精确重构 ----
xd_const = np.array([0.0, 1.0, 2.0, 3.0])
yd_const = np.array([5.0, 5.0, 5.0, 5.0])
xi_const = np.array([0.5, 1.5, 2.5])
yi_const = lagrange_value_1d(xd_const, yd_const, xi_const)
assert np.allclose(yi_const, 5.0), '[TC11] lagrange_value_1d对常数函数精确重构 FAILED'

# ---- TC12: lagrange_value_1d对线性函数精确重构 ----
xd_lin = np.array([0.0, 1.0, 2.0])
yd_lin = np.array([0.0, 2.0, 4.0])
xi_lin = np.array([0.5, 1.5])
yi_lin = lagrange_value_1d(xd_lin, yd_lin, xi_lin)
assert np.allclose(yi_lin, np.array([1.0, 3.0])), '[TC12] lagrange_value_1d对线性函数精确重构 FAILED'

# ---- TC13: hankel_inverse_fiedler逆矩阵验证A*A_inv≈I ----
n_h = 3
x_hinv = np.array([2.0, -1.0, 3.0, 0.5, 1.0])
A_inv, A_mat = hankel_inverse_fiedler(n_h, x_hinv)
I_approx = A_mat @ A_inv
assert np.linalg.norm(I_approx - np.eye(n_h)) < 0.1, '[TC13] hankel_inverse_fiedler逆矩阵验证A*A_inv≈I FAILED'

# ---- TC14: antenna_array_impedance_matrix自阻抗为实数50 ----
Z_mat = antenna_array_impedance_matrix(4, 0.5)
assert Z_mat.shape == (4, 4) and abs(Z_mat[0, 0] - 50.0) < 1e-10, '[TC14] antenna_array_impedance_matrix自阻抗为实数50 FAILED'

# ---- TC15: gauss_legendre_1d对常数函数精确积分 ----
nodes_gl, weights_gl = gauss_legendre_1d(3)
integral_const = np.sum(weights_gl * 1.0)
assert abs(integral_const - 2.0) < 1e-14, '[TC15] gauss_legendre_1d对常数函数精确积分 FAILED'

# ---- TC16: FDTD3DEngine初始化后场全为零 ----
grid_fdtd = generate_rectangular_grid(0.0, 0.01, 5, 0.0, 0.01, 5, 0.0, 0.01, 5)
eps_fdtd = np.ones((5, 5, 5)) * EPSILON_0
mu_fdtd = np.ones((5, 5, 5)) * MU_0
sig_fdtd = np.zeros((5, 5, 5))
engine_test = FDTD3DEngine(grid_fdtd, eps_fdtd, mu_fdtd, sig_fdtd, cfl_factor=0.5)
assert np.allclose(engine_test.Ex, 0.0) and np.allclose(engine_test.Hx, 0.0), '[TC16] FDTD3DEngine初始化后场全为零 FAILED'

# ---- TC17: stability_analysis_2d_scalar数值频率不超过理论频率 ----
omega_num, omega_ex = stability_analysis_2d_scalar(10.0, 10.0, 0.01, 0.01, 1e-11, C_0)
assert omega_num <= omega_ex + 1e-6, '[TC17] stability_analysis_2d_scalar数值频率不超过理论频率 FAILED'

# ---- TC18: sphere_distance_stats均值接近理论值4/3 ----
np.random.seed(42)
stats = sphere_distance_stats(n_samples=5000)
assert abs(stats['mean'] - 4.0/3.0) < 0.05, '[TC18] sphere_distance_stats均值接近理论值4/3 FAILED'

# ---- TC19: PMLBoundary3D反射系数估计为正且小于1 ----
grid_pml = generate_rectangular_grid(0.0, 0.01, 10, 0.0, 0.01, 10, 0.0, 0.01, 10)
pml = PMLBoundary3D(grid_pml, pml_thickness=3, reflection_coeff=1e-5)
R_est = pml.compute_reflection_estimate()
assert 0.0 < R_est < 1.0, '[TC19] PMLBoundary3D反射系数估计为正且小于1 FAILED'

# ---- TC20: HarmonicSource初始化属性正确 ----
src = HarmonicSource(amplitude=1.0, frequency=1e9, t0=1e-9, tau=0.5e-9, position=(2, 2, 2), component='Ez')
assert src.amplitude == 1.0 and src.component == 'Ez', '[TC20] HarmonicSource初始化属性正确 FAILED'

# ---- TC21: interpolate_material_profile线性模式对分段常数精确 ----
z_test = np.array([0.0, 0.5, 1.0])
eps_test = np.array([EPSILON_0, EPSILON_0, 2.0*EPSILON_0])
z_query = np.array([0.25, 0.75])
eps_interp = interpolate_material_profile(z_test, eps_test, z_query, method='linear')
assert abs(eps_interp[0] - EPSILON_0) < 1e-15, '[TC21] interpolate_material_profile线性模式对分段常数精确 FAILED'

# ---- TC22: assign_material_properties返回正确形状数组 ----
grid_mat = generate_rectangular_grid(0.0, 0.01, 5, 0.0, 0.01, 5, 0.0, 0.01, 5)
cav_mat = CylindricalCavity(radius=0.005, height=0.01)
eps_mat, mu_mat, sig_mat = assign_material_properties(grid_mat, [cav_mat])
assert eps_mat.shape == (5, 5, 5) and mu_mat.shape == (5, 5, 5) and sig_mat.shape == (5, 5, 5), '[TC22] assign_material_properties返回正确形状数组 FAILED'

# ---- TC23: power_flow_pagerank_analysis输出归一化和为1 ----
np.random.seed(42)
Ex_r = np.random.randn(4, 4, 4) * 1e-3
Ey_r = np.random.randn(4, 4, 4) * 1e-3
Ez_r = np.random.randn(4, 4, 4) * 1e-3
Hx_r = np.random.randn(4, 4, 4) * 1e-3
Hy_r = np.random.randn(4, 4, 4) * 1e-3
Hz_r = np.random.randn(4, 4, 4) * 1e-3
rank_field = power_flow_pagerank_analysis((Ex_r, Ey_r, Ez_r), (Hx_r, Hy_r, Hz_r), 0.001, 0.001, 0.001)
assert abs(np.sum(rank_field) - 1.0) < 1e-10, '[TC23] power_flow_pagerank_analysis输出归一化和为1 FAILED'

# ---- TC24: compute_cavity_modes_2d返回指定数量的模式 ----
np.random.seed(42)
nx_m, ny_m = 6, 6
dx_m, dy_m = 0.01, 0.01
eps_m = np.ones((nx_m, ny_m)) * EPSILON_0
mu_m = np.ones((nx_m, ny_m)) * MU_0
modes = compute_cavity_modes_2d(nx_m, ny_m, dx_m, dy_m, eps_m, mu_m, n_modes=2, max_iter=100)
assert len(modes) == 2, '[TC24] compute_cavity_modes_2d返回指定数量的模式 FAILED'

# ---- TC25: CylindricalCavity te_cutoff_frequency为正值 ----
f_te111 = cav.te_cutoff_frequency(1, 1, 1)
assert f_te111 > 0.0, '[TC25] CylindricalCavity te_cutoff_frequency为正值 FAILED'

# ---- TC26: CircleSegmentDielectric centroid距离非负 ----
centroid = seg.centroid()
assert centroid[0] >= 0.0, '[TC26] CircleSegmentDielectric centroid距离非负 FAILED'

# ---- TC27: 生成网格后cell_volume为正 ----
assert grid_test.cell_volume() > 0.0, '[TC27] 生成网格后cell_volume为正 FAILED'

# ---- TC28: integrate_field_energy_quadrature对零场返回零 ----
E_zero = (np.zeros((3,3,3)), np.zeros((3,3,3)), np.zeros((3,3,3)))
H_zero = (np.zeros((3,3,3)), np.zeros((3,3,3)), np.zeros((3,3,3)))
eps_z = np.ones((3,3,3)) * EPSILON_0
mu_z = np.ones((3,3,3)) * MU_0
W_quad_zero = integrate_field_energy_quadrature(E_zero, H_zero, eps_z, mu_z, 0.001, 0.001, 0.001, order=2)
assert abs(W_quad_zero) < 1e-30, '[TC28] integrate_field_energy_quadrature对零场返回零 FAILED'

# ---- TC29: rms_error对已知差异计算正确 ----
a_arr = np.array([0.0, 4.0, 8.0])
b_arr = np.array([1.0, 5.0, 9.0])
rms_val = rms_error(a_arr, b_arr)
assert abs(rms_val - 1.0) < 1e-12, '[TC29] rms_error对已知差异计算正确 FAILED'

# ---- TC30: check_energy_conservation对能量增长返回不通过 ----
W_grow = [1.0, 2.0, 4.0, 8.0]
t_grow = [0.0, 1.0, 2.0, 3.0]
P_grow = [0.0, 0.0, 0.0]
ec_grow = check_energy_conservation(W_grow, t_grow, P_grow, tol=1e-6)
assert ec_grow['conserved'] == False, '[TC30] check_energy_conservation对能量增长返回不通过 FAILED'
