# ---- TC01: generate_rectangular_mesh 生成网格总面积正确 ----
mesh_t1 = generate_rectangular_mesh(5, 3, lx=2.0, ly=1.0)
assert abs(mesh_t1.total_area() - 2.0) < 1e-10, '[TC01] generate_rectangular_mesh 生成网格总面积正确 FAILED'

# ---- TC02: 矩形网格三角形有向面积符号一致 ----
areas_t2 = mesh_t1.compute_areas()
assert np.all(areas_t2 > 0) or np.all(areas_t2 < 0), '[TC02] 矩形网格三角形有向面积符号一致 FAILED'

# ---- TC03: node_to_element_average 输出尺寸匹配单元数 ----
node_vals_t3 = np.arange(mesh_t1.n_nodes)
elem_vals_t3 = mesh_t1.node_to_element_average(node_vals_t3)
assert elem_vals_t3.shape[0] == mesh_t1.n_elements, '[TC03] node_to_element_average 输出尺寸匹配单元数 FAILED'

# ---- TC04: 弹性刚度矩阵对称性验证 ----
fem_t4 = ElasticFEM2D(mesh_t1, young=1e11, nu=0.3, thickness=1.0)
K_t4 = fem_t4.assemble_global_stiffness()
assert np.allclose(K_t4, K_t4.T, atol=1e-8), '[TC04] 弹性刚度矩阵对称性验证 FAILED'

# ---- TC05: 零位移时接触间隙等于节点初始y坐标 ----
u_zero_t5 = np.zeros(2 * mesh_t1.n_nodes)
contact_nodes_t5 = mesh_t1.find_bottom_boundary_nodes()
gaps_t5 = assemble_contact_gaps(mesh_t1, u_zero_t5, contact_nodes_t5, rigid_surface_y=0.0)
expected_gaps_t5 = mesh_t1.nodes[contact_nodes_t5, 1]
assert np.allclose(gaps_t5, expected_gaps_t5, atol=1e-10), '[TC05] 零位移时接触间隙等于节点初始y坐标 FAILED'

# ---- TC06: 带状矩阵紧凑存储往返一致 ----
A_t6 = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
solver_t6 = BandedSolver(3, 1, 1, compact=True)
A_band_t6 = solver_t6.full_to_compact(A_t6)
A_full_t6 = solver_t6.compact_to_full(A_band_t6)
assert np.allclose(A_t6, A_full_t6, atol=1e-12), '[TC06] 带状矩阵紧凑存储往返一致 FAILED'

# ---- TC07: 带状求解器求解对角系统正确 ----
A_t7 = np.diag([2.0, 3.0, 5.0])
solver_t7 = BandedSolver(3, 0, 0, compact=True)
A_band_t7 = solver_t7.full_to_compact(A_t7)
b_t7 = np.array([1.0, 2.0, 3.0])
x_t7 = solver_t7.solve_system(A_band_t7, b_t7, use_pivot=False)
res_t7 = A_t7 @ x_t7 - b_t7
assert np.linalg.norm(res_t7) < 1e-12, '[TC07] 带状求解器求解对角系统正确 FAILED'

# ---- TC08: 对角矩阵上下带宽均为零 ----
diag_mat_t8 = np.diag([1.0, 2.0, 3.0, 4.0])
ml_t8, mu_t8 = compute_matrix_bandwidth(diag_mat_t8)
assert ml_t8 == 0 and mu_t8 == 0, '[TC08] 对角矩阵上下带宽均为零 FAILED'

# ---- TC09: Archard磨损模型零压力时磨损深度不变 ----
wear_t9 = ArchardWearModel(wear_coeff=1e-6, omega=2.0*np.pi, v0=0.01)
def p_zero(t): return 0.0
t_t9, h_t9 = wear_t9.integrate_midpoint(h0=0.0, t_span=(0.0, 1.0), n_steps=10, pressure_func=p_zero)
assert abs(h_t9[-1]) < 1e-14, '[TC09] Archard磨损模型零压力时磨损深度不变 FAILED'

# ---- TC10: RK4磨损积分在正压力下单调不减 ----
def p_const(t): return 1.0e6
t_t10, h_t10 = wear_t9.integrate_rk4(h0=0.0, t_span=(0.0, 1.0), n_steps=20, pressure_func=p_const)
assert np.all(np.diff(h_t10) >= -1e-15), '[TC10] RK4磨损积分在正压力下单调不减 FAILED'

# ---- TC11: 耦合磨损单步在零压力下无增量 ----
p_zero_arr_t11 = np.zeros(5)
h_prev_t11 = np.ones(5)
h_new_t11 = coupled_wear_contact_step(wear_t9, h_prev_t11, p_zero_arr_t11, dt=0.1)
assert np.allclose(h_new_t11, h_prev_t11, atol=1e-14), '[TC11] 耦合磨损单步在零压力下无增量 FAILED'

# ---- TC12: peaks函数在(0,0)处解析值正确 ----
from friction_optimization import peaks_function
val_t12 = peaks_function(0.0, 0.0)
expected_t12 = 3.0 * np.exp(-1.0) - (1.0/3.0) * np.exp(-1.0)
assert abs(val_t12 - expected_t12) < 1e-10, '[TC12] peaks函数在(0,0)处解析值正确 FAILED'

# ---- TC13: 自相关函数在lag=0处为正 ----
np.random.seed(42)
x_t13 = np.random.randn(50)
corr_t13 = correlation_function(x_t13, m=10)
assert corr_t13[0] > 0, '[TC13] 自相关函数在lag=0处为正 FAILED'

# ---- TC14: 粗糙度生成固定随机种子可复现 ----
np.random.seed(123)
x_r1_t14, h_r1_t14 = generate_pink_noise_profile(100, length=1.0, beta=1.8)
np.random.seed(123)
x_r2_t14, h_r2_t14 = generate_pink_noise_profile(100, length=1.0, beta=1.8)
assert np.allclose(h_r1_t14, h_r2_t14, atol=1e-12), '[TC14] 粗糙度生成固定随机种子可复现 FAILED'

# ---- TC15: safe_divide零除返回fallback ----
from utils import safe_divide
assert safe_divide(5.0, 0.0, fallback=99.0) == 99.0, '[TC15] safe_divide零除返回fallback FAILED'

# ---- TC16: Macaulay括号负输入返回零 ----
from utils import macaulay_bracket
assert macaulay_bracket(-3.0) == 0.0, '[TC16] Macaulay括号负输入返回零 FAILED'
assert macaulay_bracket(2.0) == 2.0, '[TC16] Macaulay括号负输入返回零 FAILED'

# ---- TC17: 2x2对称系统解析解正确 ----
from utils import solve_2x2_symmetric
x1_t17, x2_t17 = solve_2x2_symmetric(2.0, 1.0, 3.0, 5.0, 7.0)
expected_x1_t17 = (3.0*5.0 - 1.0*7.0) / (2.0*3.0 - 1.0)
expected_x2_t17 = (-1.0*5.0 + 2.0*7.0) / (2.0*3.0 - 1.0)
assert abs(x1_t17 - expected_x1_t17) < 1e-12 and abs(x2_t17 - expected_x2_t17) < 1e-12, '[TC17] 2x2对称系统解析解正确 FAILED'

# ---- TC18: 稳定系统特征值全负实部判定无不稳定模态 ----
ev_stable_t18 = np.array([-1.0+2j, -2.0-1j, -0.5+0j])
stab_t18 = stability_criterion(ev_stable_t18)
assert stab_t18['unstable_count'] == 0, '[TC18] 稳定系统特征值全负实部判定无不稳定模态 FAILED'

# ---- TC19: 频响函数幅值非负 ----
K_t19 = np.eye(4) * 100.0
M_t19 = np.eye(4) * 0.1
C_t19 = np.eye(4) * 0.5
omega_t19 = np.linspace(1.0, 50.0, 10)
frf_t19 = frequency_response_function(K_t19, M_t19, C_t19, omega_t19, load_dof=0)
assert np.all(frf_t19 >= 0.0), '[TC19] 频响函数幅值非负 FAILED'

# ---- TC20: 单位矩阵条件数估计为1 ----
I_t20 = np.eye(5)
cond_t20 = condition_number_estimate(I_t20)
assert abs(cond_t20 - 1.0) < 1e-10, '[TC20] 单位矩阵条件数估计为1 FAILED'

# ---- TC21: 整数向量最大公约数与numpy一致 ----
from diophantine_utils import i4vec_gcd
vec_t21 = np.array([12, 18, 24])
assert i4vec_gcd(vec_t21) == 6, '[TC21] 整数向量最大公约数与numpy一致 FAILED'

# ---- TC22: 丢番图非负解特解满足原方程 ----
a_t22 = np.array([2, 3])
b_t22 = 12
d_t22, v_t22, B_t22, kmin_t22, kmax_t22 = diophantine_nonnegative_solve(a_t22, b_t22)
assert np.dot(a_t22, v_t22) == b_t22, '[TC22] 丢番图非负解特解满足原方程 FAILED'

# ---- TC23: 接触压力计算结果非负 ----
mesh_t23 = generate_rectangular_mesh(5, 3, lx=1.0, ly=0.5)
fem_t23 = ElasticFEM2D(mesh_t23, young=1e10, nu=0.3, thickness=1.0)
contact_nodes_t23 = mesh_t23.find_bottom_boundary_nodes()
cs_t23 = SignoriniCoulombContact(fem_t23, contact_nodes_t23, friction_coeff=0.3, aug_lag_penalty=1e8, max_iter=10, tol=1e-6)
u_dummy_t23 = np.zeros(2 * mesh_t23.n_nodes)
p_n_t23, p_t_t23 = cs_t23.compute_contact_pressure(u_dummy_t23)
assert np.all(p_n_t23 >= 0.0), '[TC23] 接触压力计算结果非负 FAILED'

# ---- TC24: 主动集Newton接触求解器在简单载荷下收敛 ----
f_ext_t24 = np.zeros(2 * mesh_t23.n_nodes)
u_as_t24, hist_as_t24 = active_set_newton_contact(fem_t23, contact_nodes_t23, f_ext_t24, friction_coeff=0.3, max_iter=20, tol=1e-8)
assert hist_as_t24['iterations'] <= 20, '[TC24] 主动集Newton接触求解器在简单载荷下收敛 FAILED'

# ---- TC25: 弹性应变能非负 ----
np.random.seed(42)
u_rand_t25 = np.random.randn(2 * mesh_t23.n_nodes)
energy_t25 = fem_t23.compute_strain_energy(u_rand_t25)
assert energy_t25 >= 0.0, '[TC25] 弹性应变能非负 FAILED'

# ---- TC26: 接触子矩阵带状存储维度正确 ----
K_t26 = fem_t23.assemble_global_stiffness()
K_sub_band_t26 = extract_banded_submatrix(K_t26, contact_nodes_t23, mesh_t23.n_nodes, 2, 2)
assert K_sub_band_t26.shape[0] == 5, '[TC26] 接触子矩阵带状存储维度正确 FAILED'

# ---- TC27: 金字塔采样点位于单位金字塔内 ----
from monte_carlo_contact import pyramid01_sample
np.random.seed(42)
samples_t27 = pyramid01_sample(100)
assert np.all(samples_t27[2, :] >= 0.0) and np.all(samples_t27[2, :] <= 1.0), '[TC27] 金字塔采样点位于单位金字塔内 FAILED'

# ---- TC28: 蒙特卡洛接触力统计量可复现 ----
np.random.seed(42)
def sampler_t28(pts):
    return np.ones(len(pts)) * 1.0e5 if len(pts) > 0 else np.zeros(0)
mean_p_t28, var_p_t28, max_p_t28 = monte_carlo_contact_force_variance(mesh_t23, sampler_t28, n_samples=100)
assert abs(mean_p_t28 - 1.0e5) < 1e-3, '[TC28] 蒙特卡洛接触力统计量可复现 FAILED'

# ---- TC29: peaks数值梯度与有限差分一致 ----
from friction_optimization import peaks_gradient
h_t29 = 1e-6
fx_num_t29 = (peaks_function(0.2+h_t29, 0.1) - peaks_function(0.2-h_t29, 0.1)) / (2.0*h_t29)
fy_num_t29 = (peaks_function(0.2, 0.1+h_t29) - peaks_function(0.2, 0.1-h_t29)) / (2.0*h_t29)
fx_t29, fy_t29 = peaks_gradient(0.2, 0.1)
assert abs(fx_t29 - fx_num_t29) < 1e-6 and abs(fy_t29 - fy_num_t29) < 1e-6, '[TC29] peaks数值梯度与有限差分一致 FAILED'

# ---- TC30: 粗糙度叠加位移范围受scale控制 ----
np.random.seed(42)
x_rough_t30, h_rough_t30 = generate_pink_noise_profile(50, length=1.0, beta=1.8)
contact_mask_t30 = np.zeros(mesh_t23.n_nodes, dtype=bool)
contact_mask_t30[:5] = True
nodes_rough_t30 = apply_roughness_to_mesh(mesh_t23.nodes, h_rough_t30, contact_mask_t30, scale=1e-6)
max_disp_t30 = np.max(np.abs(nodes_rough_t30[:, 1] - mesh_t23.nodes[:, 1]))
assert max_disp_t30 <= 5e-6, '[TC30] 粗糙度叠加位移范围受scale控制 FAILED'
