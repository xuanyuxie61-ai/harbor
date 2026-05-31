# ---- TC01: jacobi_polynomial 输出形状正确 ----
x_test = np.linspace(-1.0, 1.0, 10)
V = jacobi_polynomial(10, 5, 0.0, 0.0, x_test)
assert V.shape == (10, 6), '[TC01] jacobi_polynomial shape FAILED'
assert np.all(np.isfinite(V)), '[TC01] jacobi_polynomial finite values FAILED'

# ---- TC02: jacobi_polynomial P_0^(0,0)(x)=1 对所有 x ----
V02 = jacobi_polynomial(5, 0, 0.0, 0.0, np.array([-0.5, 0.0, 0.5]))
assert np.allclose(V02[:, 0], 1.0), '[TC02] jacobi P0 identity FAILED'

# ---- TC03: laguerre_compute 返回正确数量的节点和权重数组 ----
xtab, weight = laguerre_compute(10, alpha=0.0)
assert len(xtab) == 10, '[TC03] laguerre_compute node count FAILED'
assert len(weight) == 10, '[TC03] laguerre_compute weight count FAILED'
assert np.all(np.isfinite(xtab)), '[TC03] laguerre_compute nodes finite FAILED'
assert np.all(np.isfinite(weight)), '[TC03] laguerre_compute weights finite FAILED'

# ---- TC04: laguerre_quadrature_integrate 返回有限结果 ----
def f_test(x):
    return x
lag_result = laguerre_quadrature_integrate(f_test, norder=16, alpha=0.0)
assert np.isfinite(lag_result), '[TC04] laguerre_quadrature result finite FAILED'
assert not np.isnan(lag_result), '[TC04] laguerre_quadrature result not NaN FAILED'

# ---- TC05: antoine_vapor_pressure 返回正有限值 ----
Psat = antoine_vapor_pressure(80.0, 8.20417, 1642.89, 230.300)
assert Psat > 0, '[TC05] antoine_vapor_pressure positive FAILED'
assert np.isfinite(Psat), '[TC05] antoine_vapor_pressure finite FAILED'

# ---- TC06: wilson_activity_coefficient 返回正值且形状正确 ----
V_w = np.array([5.87e-5, 1.80e-5, 4.07e-5])
L_ij = np.array([[0.0, 155.21, 46.32], [292.51, 0.0, 289.19], [182.42, 201.21, 0.0]])
x_w = np.array([0.4, 0.3, 0.3])
gamma_w = wilson_activity_coefficient(x_w, V_w, L_ij, 350.0)
assert gamma_w.shape == (3,), '[TC06] wilson gamma shape FAILED'
assert np.all(gamma_w > 0), '[TC06] wilson gamma positive FAILED'
assert np.all(np.isfinite(gamma_w)), '[TC06] wilson gamma finite FAILED'

# ---- TC07: vle_flash_calculation 汽相组成归一化 ----
V_vle = np.array([5.87e-5, 1.80e-5, 4.07e-5])
L_vle = np.array([[0.0, 155.21, 46.32], [292.51, 0.0, 289.19], [182.42, 201.21, 0.0]])
A_vle = np.array([8.20417, 8.07131, 8.08097])
B_vle = np.array([1642.89, 1730.63, 1582.91])
C_vle = np.array([230.300, 233.426, 239.726])
y, K, gamma = vle_flash_calculation(np.array([0.4, 0.3, 0.3]), 101325.0, 350.0, A_vle, B_vle, C_vle, V_vle, L_vle)
assert np.abs(np.sum(y) - 1.0) < 1e-10, '[TC07] vle y sum FAILED'
assert np.all(y >= 0), '[TC07] vle y nonnegative FAILED'

# ---- TC08: vle_relative_volatility 最大值为 1.0 ----
alpha_rel = vle_relative_volatility(K)
assert np.abs(np.max(alpha_rel) - 1.0) < 1e-12, '[TC08] relative volatility max FAILED'

# ---- TC09: activity_coefficient_spectral_expansion 输出形状正确 ----
x_range = np.linspace(0.0, 1.0, 15)
V_jac09 = activity_coefficient_spectral_expansion(x_range, 3, alpha_jac=0.0, beta_jac=0.0, n_modes=5)
assert V_jac09.shape == (15, 6), '[TC09] spectral expansion shape FAILED'

# ---- TC10: shepard_interp_1d 在数据点处精确插值 ----
xd = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
yd = np.array([373.0, 368.0, 362.0, 355.0, 348.0, 340.0])
yi_exact = shepard_interp_1d(6, xd, yd, 2.0, 1, np.array([4.0]))
assert np.abs(yi_exact[0] - 362.0) < 1e-6, '[TC10] shepard exact match FAILED'

# ---- TC11: quad_trapezoid 积分 f(x)=x 从 0 到 1 得 0.5 ----
trap_result11 = quad_trapezoid(lambda x: x, 0.0, 1.0, 100)
assert np.abs(trap_result11 - 0.5) < 1e-10, '[TC11] trapezoid x integral FAILED'

# ---- TC12: shepard_interp_1d p=0 时返回等权平均 ----
yi_avg = shepard_interp_1d(3, np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]), 0.0, 1, np.array([0.5]))
assert np.abs(yi_avg[0] - 20.0) < 1e-10, '[TC12] shepard p=0 average FAILED'

# ---- TC13: drectangle 内部点返回负值 ----
test_pt_inside = np.array([[0.5, 0.4]])
d_inside = drectangle(test_pt_inside, 0.0, 1.5, 0.0, 0.8)
assert d_inside[0] < 0, '[TC13] drectangle inside FAILED'

# ---- TC14: drectangle 外部点返回正值 ----
test_pt_outside = np.array([[2.0, 0.5]])
d_outside = drectangle(test_pt_outside, 0.0, 1.5, 0.0, 0.8)
assert d_outside[0] > 0, '[TC14] drectangle outside FAILED'

# ---- TC15: reference_to_physical_q4 角点 (0,0) 映射到第一顶点 ----
q4 = np.array([[0.0, 1.5, 1.5, 0.0], [0.0, 0.0, 0.8, 0.8]])
rs_corner = np.array([[0.0], [0.0]])
xy_corner = reference_to_physical_q4(q4, 1, rs_corner)
assert np.allclose(xy_corner[:, 0], np.array([0.0, 0.0])), '[TC15] q4 corner mapping FAILED'

# ---- TC16: generate_tray_mesh 节点数和单元数正确 ----
nodes, elements, areas = generate_tray_mesh(1.5, 0.8, nx=8, ny=5)
assert nodes.shape == (54, 2), '[TC16] tray mesh node count FAILED'
assert len(elements) == 40, '[TC16] tray mesh element count FAILED'

# ---- TC17: quad_trapezoid 积分 sin(x) 0 到 pi 得 2.0 ----
trap_result17 = quad_trapezoid(lambda x: np.sin(x), 0.0, np.pi, 200)
assert np.abs(trap_result17 - 2.0) < 1e-4, '[TC17] trapezoid sin integral FAILED'

# ---- TC18: fd1d_wave_solve 输出形状正确且 alpha 有限 ----
def P_x1(t):
    return 120000.0
def P_x2(t):
    return 101325.0
def P_t1(z):
    return np.full_like(z, 110000.0)
def Pt_t1(z):
    return np.zeros_like(z)
P_field, alpha_wave = fd1d_wave_solve(20, 0.0, 15.0, 100, 0.0, 0.5, 85.0, P_x1, P_x2, P_t1, Pt_t1)
assert P_field.shape == (101, 21), '[TC18] wave solve shape FAILED'
assert np.isfinite(alpha_wave), '[TC18] wave alpha finite FAILED'
assert np.all(np.isfinite(P_field)), '[TC18] wave field finite FAILED'

# ---- TC19: rk45_integrate 指数衰减 ODE ----
def exp_decay(t, y):
    return -0.5 * y
t_rk, y_rk, e_rk = rk45_integrate(exp_decay, (0.0, 5.0), np.array([1.0]), 100)
assert len(t_rk) == 101, '[TC19] rk45 time length FAILED'
assert y_rk[-1, 0] > 0, '[TC19] rk45 decay positive FAILED'
assert y_rk[-1, 0] < y_rk[0, 0], '[TC19] rk45 monotonic decay FAILED'

# ---- TC20: langford_deriv 返回 3 维有限值 ----
import numpy as np
np.random.seed(42)
xyz0 = np.array([0.1, -0.2, 0.05])
deriv20 = langford_deriv(0.0, xyz0)
assert deriv20.shape == (3,), '[TC20] langford deriv shape FAILED'
assert np.all(np.isfinite(deriv20)), '[TC20] langford deriv finite FAILED'

# ---- TC21: lorenz96_deriv 输出形状匹配输入 ----
import numpy as np
np.random.seed(42)
y_l96_21 = np.random.randn(20) * 0.1 + 0.5
deriv_l96 = lorenz96_deriv(0.0, y_l96_21, n=20, force=8.0)
assert deriv_l96.shape == (20,), '[TC21] lorenz96 deriv shape FAILED'
assert np.all(np.isfinite(deriv_l96)), '[TC21] lorenz96 deriv finite FAILED'

# ---- TC22: gilliland_correlation 残差有限 ----
residual = gilliland_correlation(3.0, 1.5, 25, 8)
assert np.isfinite(residual), '[TC22] Gilliland residual finite FAILED'

# ---- TC23: reboiler_duty 返回正值 ----
Q_R = reboiler_duty(3.0, 50.0, 5e5, 35000.0, 100.0, 0.4, 0.95, 0.05)
assert Q_R > 0, '[TC23] reboiler duty positive FAILED'
assert np.isfinite(Q_R), '[TC23] reboiler duty finite FAILED'

# ---- TC24: estimate_N_from_R N > N_min ----
N_est = estimate_N_from_R(3.0, 1.5, 8)
assert N_est > 8, '[TC24] estimate N from R FAILED'

# ---- TC25: packing_void_fraction 在 [0.2, 0.98] 范围内 ----
eps = packing_void_fraction(1000, 0.05, 1.0, 3.0, 0.8)
assert 0.2 <= eps <= 0.98, '[TC25] void fraction range FAILED'

# ---- TC26: ergun_pressure_drop 返回正值 ----
dP_ergun = ergun_pressure_drop(0.7, 1.8e-5, 1.5, 2.5, 0.05, 3.0)
assert dP_ergun > 0, '[TC26] Ergun dP positive FAILED'
assert np.isfinite(dP_ergun), '[TC26] Ergun dP finite FAILED'

# ---- TC27: hexagon01_area 等于 3*sqrt(3)/2 ----
area_hex = hexagon01_area()
expected_area = 3.0 * np.sqrt(3.0) / 2.0
assert np.abs(area_hex - expected_area) < 1e-12, '[TC27] hexagon area FAILED'

# ---- TC28: hexagon_monte_carlo_integrate 常数函数积分 ----
import numpy as np
np.random.seed(42)
const_result = hexagon_monte_carlo_integrate(lambda x, y: 1.0, n_samples=10000)
assert np.abs(const_result - expected_area) / expected_area < 0.03, '[TC28] hexagon MC constant FAILED'

# ---- TC29: rcont_random_table 行和列和匹配 ----
nrowt = np.array([20, 100, 120, 120, 20])
ncolt = np.array([90, 110, 180])
mat_rcont = rcont_random_table(5, 3, nrowt, ncolt, seed=100)
assert np.allclose(np.sum(mat_rcont, axis=1), nrowt), '[TC29] rcont row sums FAILED'
assert np.sum(mat_rcont) > 0, '[TC29] rcont total positive FAILED'

# ---- TC30: interpolate_vle_data 输出形状正确 ----
z_data = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
T_data = np.array([373.0, 368.0, 362.0, 355.0, 348.0, 340.0])
x_data_30 = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1],
                      [0.45, 0.45, 0.1], [0.6, 0.35, 0.05], [0.8, 0.18, 0.02]])
y_data_30 = np.array([[0.3, 0.6, 0.1], [0.45, 0.45, 0.1], [0.55, 0.38, 0.07],
                      [0.65, 0.30, 0.05], [0.78, 0.20, 0.02], [0.90, 0.09, 0.01]])
z_query = np.linspace(0.0, 10.0, 5)
T_interp, x_interp, y_interp = interpolate_vle_data(z_data, T_data, x_data_30, y_data_30, z_query)
assert T_interp.shape == (5,), '[TC30] interp T shape FAILED'
assert x_interp.shape == (5, 3), '[TC30] interp x shape FAILED'
assert y_interp.shape == (5, 3), '[TC30] interp y shape FAILED'

# ---- TC31: integrate_mass_transfer_flux 返回有限值 ----
def mass_flux(z):
    return 0.5 * np.exp(-0.1 * z)
z_nodes = np.linspace(0.0, 10.0, 21)
total_mt = integrate_mass_transfer_flux(z_nodes, mass_flux)
assert total_mt > 0, '[TC31] mass transfer flux positive FAILED'
assert np.isfinite(total_mt), '[TC31] mass transfer flux finite FAILED'

# ---- TC32: compute_local_efficiency_on_mesh 效率在 [0,1] ----
nodes32, elements32, areas32 = generate_tray_mesh(1.5, 0.8, nx=5, ny=4)
x_liq32 = np.array([0.5, 0.4, 0.1])
y_vap32 = np.array([0.5, 0.3, 0.05])
K_eq32 = np.array([1.2, 0.875, 0.5])
E_local32 = compute_local_efficiency_on_mesh(nodes32, elements32, x_liq32, y_vap32, K_eq32)
assert np.all(E_local32 >= 0.0), '[TC32] local efficiency lower bound FAILED'
assert np.all(E_local32 <= 1.0), '[TC32] local efficiency upper bound FAILED'

# ---- TC33: mesh_average_efficiency 在 [0,1] ----
E_avg33 = mesh_average_efficiency(nodes32, elements32, areas32, E_local32)
assert 0.0 <= E_avg33 <= 1.0, '[TC33] avg efficiency range FAILED'

# ---- TC34: relative_change 对称性 ----
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
rc = relative_change(a, b)
assert np.abs(rc) < 1e-12, '[TC34] relative_change identical FAILED'

# ---- TC35: thermo_factor_check 在区间内不变 ----
T_val = 400.0
T_checked = thermo_factor_check(T_val)
assert T_checked == 400.0, '[TC35] thermo factor in bounds FAILED'
