
# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================

from utils import safe_divide
from rayleigh_plesset_solver import gas_pressure_adiabatic
from surface_integrals import chebyshev2_nodes_weights

# ---- TC01: safe_divide 正常除法结果正确 ----
result = safe_divide(10.0, 2.0)
assert abs(result - 5.0) < 1e-12, '[TC01] safe_divide 正常除法结果正确 FAILED'

# ---- TC02: safe_divide 除以零返回默认值 ----
result = safe_divide(10.0, 0.0)
assert abs(result - 0.0) < 1e-12, '[TC02] safe_divide 除以零返回默认值 FAILED'

# ---- TC03: van_der_waals_pressure 返回值为有限正数 ----
n_g = 1.0
V = 1.0
T_g = 300.0
p_vdw = van_der_waals_pressure(n_g, V, T_g)
assert np.isfinite(p_vdw) and p_vdw > 0, '[TC03] van_der_waals_pressure 返回值为有限正数 FAILED'

# ---- TC04: gas_pressure_adiabatic 解析验证 ----
R_test = 1e-5
R0_test = 1e-5
p_g0_test = 101325.0
p_adiabatic = gas_pressure_adiabatic(R_test, R0_test, p_g0_test)
assert abs(p_adiabatic - p_g0_test) < 1.0, '[TC04] gas_pressure_adiabatic 解析验证 FAILED'

# ---- TC05: critical_nucleation_radius_bisection 返回正半径 ----
R_crit = critical_nucleation_radius_bisection(101325.0, 2338.0, 0.0728, 998.0, R_min=1e-9, R_max=1e-3)
assert R_crit > 0 and R_crit < 1e-3, '[TC05] critical_nucleation_radius_bisection 返回正半径 FAILED'

# ---- TC06: ellipse_condition_number 单位矩阵条件数为1 ----
I2 = np.eye(2)
cond_I = ellipse_condition_number(I2)
assert abs(cond_I - 1.0) < 1e-10, '[TC06] ellipse_condition_number 单位矩阵条件数为1 FAILED'

# ---- TC07: bubble_ellipse_shape 输出形状正确 ----
V_c = np.array([0.0, 0.0])
A_shape = np.array([[2.0, 0.3], [0.3, 1.5]])
pts = bubble_ellipse_shape(V_c, A_shape, 1e-5, num_points=50)
assert pts.shape == (2, 50), '[TC07] bubble_ellipse_shape 输出形状正确 FAILED'

# ---- TC08: bubble_volume_quadrature 球体体积解析验证 ----
def r_sphere(theta, phi):
    return 1.0e-5
vol_sphere = bubble_volume_quadrature(r_sphere, theta_nodes=8, phi_nodes=8)
vol_exact = (4.0/3.0) * np.pi * (1.0e-5)**3
assert abs(vol_sphere - vol_exact) / vol_exact < 0.05, '[TC08] bubble_volume_quadrature 球体体积解析验证 FAILED'

# ---- TC09: kinetic_energy_integral 非负性 ----
E_kin = kinetic_energy_integral(1e-5, 100.0, 998.0, theta_nodes=8)
assert E_kin >= 0, '[TC09] kinetic_energy_integral 非负性 FAILED'

# ---- TC10: chebyshev2_nodes_weights 权重和等于精确积分值 ----
x_cheb, w_cheb = chebyshev2_nodes_weights(16, a=-1.0, b=1.0)
assert abs(np.sum(w_cheb) - np.pi / 2.0) < 1e-10, '[TC10] chebyshev2_nodes_weights 权重和等于精确积分值 FAILED'

# ---- TC11: phi_mq 对称性验证 ----
r_val = 2.0
r0_val = 1.0
phi_pos = phi_mq(r_val, r0_val)
phi_neg = phi_mq(-r_val, r0_val)
assert abs(phi_pos - phi_neg) < 1e-12, '[TC11] phi_mq 对称性验证 FAILED'

# ---- TC12: rbf_weights 可解性验证 ----
np.random.seed(42)
m_rbf = 2
nd_rbf = 5
xd_rbf = np.random.rand(m_rbf, nd_rbf)
pd_rbf = np.random.rand(nd_rbf)
r0_rbf = 0.5
w_rbf = rbf_weights(m_rbf, nd_rbf, xd_rbf, r0_rbf, phi_mq, pd_rbf)
assert len(w_rbf) == nd_rbf and np.all(np.isfinite(w_rbf)), '[TC12] rbf_weights 可解性验证 FAILED'

# ---- TC13: nucleation_barrier_energy 符号验证 ----
delta_G = nucleation_barrier_energy(101325.0, 2338.0, 0.0728)
assert delta_G > 0, '[TC13] nucleation_barrier_energy 符号验证 FAILED'

# ---- TC14: nucleation_rate 非负性 ----
J = nucleation_rate(101325.0, 2338.0, 0.0728, 293.15)
assert J >= 0, '[TC14] nucleation_rate 非负性 FAILED'

# ---- TC15: coupled_residual 输出尺寸为8 ----
U_test = np.array([1e-5, 0.0, 293.15, 1e-12, 0.0, 0.0, 0.0, 0.0])
params_test = {'p_inf': 101325.0, 'sigma': 0.0728, 'rho': 998.0, 'mu': 1.002e-3, 'R0': 1e-5, 'p_g0': 101325.0, 'gamma': 1.4, 'R_eq': 1e-5, 'c_sound': 1482.0}
F_res = coupled_residual(U_test, params_test)
assert F_res.shape == (8,), '[TC15] coupled_residual 输出尺寸为8 FAILED'

# ---- TC16: solve_coupled_picard 返回8维有限解 ----
U0_picard = np.array([1e-5, 0.0, 293.15, 1e-12, 0.0, 0.0, 0.0, 0.0])
U_pic, conv_pic, _ = solve_coupled_picard(U0_picard, params_test, max_iter=50, tol=1e-8)
assert U_pic.shape == (8,) and np.isfinite(U_pic[0]), '[TC16] solve_coupled_picard 返回8维有限解 FAILED'

# ---- TC17: bubble_energy_budget 总能量非负 ----
energies = bubble_energy_budget(1e-5, 100.0, 101325.0, 2338.0, 0.0728, 998.0, 1482.0)
assert energies['total'] >= 0, '[TC17] bubble_energy_budget 总能量非负 FAILED'

# ---- TC18: collapse_efficiency 范围约束 ----
eta = collapse_efficiency(50e-6, 1e-9, 101325.0, 2338.0, 0.0728, 998.0, 1482.0)
assert 0.0 <= eta <= 1.0, '[TC18] collapse_efficiency 范围约束 FAILED'

# ---- TC19: chaotic_microfragmentation 可复现性 ----
np.random.seed(42)
x_frag1, lyap1 = chaotic_microfragmentation(num_points=100, iterations=50)
np.random.seed(42)
x_frag2, lyap2 = chaotic_microfragmentation(num_points=100, iterations=50)
assert np.allclose(x_frag1, x_frag2) and abs(lyap1 - lyap2) < 1e-12, '[TC19] chaotic_microfragmentation 可复现性 FAILED'

# ---- TC20: fragmentation_dimension 范围约束 ----
np.random.seed(42)
x_frag, _ = chaotic_microfragmentation(num_points=200, iterations=100)
D_box = fragmentation_dimension(x_frag)
assert 0.0 <= D_box <= 2.0, '[TC20] fragmentation_dimension 范围约束 FAILED'

# ---- TC21: generate_square_mesh 节点数正确 ----
nodes_mesh, elems_mesh = generate_square_mesh(0.0, 1.0, 0.5)
assert len(nodes_mesh) == 9, '[TC21] generate_square_mesh 节点数正确 FAILED'

# ---- TC22: fem_matrices_2d 矩阵形状一致 ----
nodes_fem, elems_fem = generate_square_mesh(0.0, 1.0, 0.5)
M_fem, K_fem = fem_matrices_2d(nodes_fem, elems_fem, 1482.0, 998.0)
assert M_fem.shape == K_fem.shape and M_fem.shape[0] == len(nodes_fem), '[TC22] fem_matrices_2d 矩阵形状一致 FAILED'

# ---- TC23: energy_spectrum_analysis 输出频率非负 ----
np.random.seed(42)
R_hist = np.sin(2 * np.pi * np.linspace(0, 1, 64)) + 1e-5
v_hist = np.zeros(64)
dt_spec = 1e-9
freqs, R_fft, v_fft = energy_spectrum_analysis(R_hist, v_hist, dt_spec)
assert np.all(freqs >= 0), '[TC23] energy_spectrum_analysis 输出频率非负 FAILED'

# ---- TC24: disk01_sample 点在单位圆内 ----
np.random.seed(42)
disk_pts = disk01_sample(100)
norms = np.sqrt(np.sum(disk_pts**2, axis=0))
assert np.all(norms <= 1.0 + 1e-10), '[TC24] disk01_sample 点在单位圆内 FAILED'

# ---- TC25: full_deck_nucleation_stats 统计量范围正确 ----
np.random.seed(42)
p_range = np.linspace(50000.0, 150000.0, 5)
stats = full_deck_nucleation_stats(20, p_range, 2338.0, 0.0728, 293.15, surface_area=1e-4)
assert stats['min'] <= stats['max'] and stats['mean'] >= stats['min'] and stats['mean'] <= stats['max'], '[TC25] full_deck_nucleation_stats 统计量范围正确 FAILED'

print('\n全部 25 个测试通过!\n')
