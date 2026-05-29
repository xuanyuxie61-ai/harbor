# ---- TC01: gamma_func 正整数解析验证 ----
result = gamma_func(5.0)
assert abs(result - 24.0) < 1e-6, '[TC01] gamma_func(5) 应等于 24 FAILED'

# ---- TC02: safe_gamma_ratio 已知比值验证 ----
ratio = safe_gamma_ratio(np.array([3.0, 4.0]), 6.0)
expected = 2.0 * 6.0 / 120.0
assert abs(ratio - expected) < 1e-6, '[TC02] safe_gamma_ratio 解析验证 FAILED'

# ---- TC03: arc_cosine_safe 超界输入保护 ----
val_up = arc_cosine_safe(1.0001)
val_low = arc_cosine_safe(-1.0001)
assert abs(val_up - 0.0) < 1e-6, '[TC03] arc_cosine_safe 上限钳位 FAILED'
assert abs(val_low - np.pi) < 1e-6, '[TC03] arc_cosine_safe 下限钳位 FAILED'

# ---- TC04: gcd_vector 已知数组最大公约数 ----
g = gcd_vector(np.array([12, 18, 24]))
assert g == 6, '[TC04] gcd_vector([12,18,24]) 应等于 6 FAILED'

# ---- TC05: check_well_posed_diophantine 判别正确性 ----
assert check_well_posed_diophantine(np.array([2, 3]), 7) == True, '[TC05] 良定义丢番图判别 FAILED'
assert check_well_posed_diophantine(np.array([2, 4]), 7) == False, '[TC05] 不良定义丢番图判别 FAILED'

# ---- TC06: generate_platform_waterline 输出形状 ----
wl = generate_platform_waterline("semi-submersible")
assert wl.ndim == 2 and wl.shape[1] == 2, '[TC06] 水线面顶点维度应为 Nx2 FAILED'

# ---- TC07: compute_waterplane_properties 面积非负 ----
wp = compute_waterplane_properties(wl)
assert wp['area'] > 0, '[TC07] 水线面面积必须为正 FAILED'
assert wp['I_xx'] >= 0, '[TC07] I_xx 必须非负 FAILED'

# ---- TC08: generate_cvt_nodes_1d 输出长度和范围 ----
nodes = generate_cvt_nodes_1d(5, 0.0, 1.0, n_iter=10)
assert len(nodes) == 5, '[TC08] CVT 节点数应为 5 FAILED'
assert nodes[0] == 0.0 and nodes[-1] == 1.0, '[TC08] CVT 端点应固定 FAILED'

# ---- TC09: simplex01_volume 解析验证 ----
vol = simplex01_volume(3)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC09] 3维单位单纯形体积应为 1/6 FAILED'

# ---- TC10: build_laplacian_2d_sparse 矩阵维度正确 ----
A = build_laplacian_2d_sparse(3, 3, 1.0, 1.0)
assert A.n_rows == 9 and A.n_cols == 9, '[TC10] Laplacian 矩阵维度应为 9x9 FAILED'

# ---- TC11: R8NCFSparseMatrix.mv 稀疏矩阵向量乘法 ----
A_test = build_second_difference_1d_sparse(3)
x = np.ones(3)
y = A_test.mv(x)
assert y.shape == (3,), '[TC11] SpMV 输出维度应为 3 FAILED'
assert abs(y[0] - 1.0) < 1e-12, '[TC11] SpMV 首元素验证 FAILED'

# ---- TC12: rcm_reorder 降低带宽 ----
tri = [(0,1,2), (1,2,3)]
adj = build_adjacency_from_triangulation(tri, 4)
perm = rcm_reorder(adj)
from utils import compute_bandwidth
bw_before = compute_bandwidth(adj, list(range(4)))
bw_after = compute_bandwidth(adj, perm)
assert bw_after <= bw_before, '[TC12] RCM 应不增加带宽 FAILED'

# ---- TC13: jonswap_spectrum 输出非负有限 ----
f = np.linspace(0.05, 0.3, 10)
S = jonswap_spectrum(f, fp=0.1, Hs=2.0)
assert np.all(S >= 0), '[TC13] JONSWAP 谱必须非负 FAILED'
assert np.all(np.isfinite(S)), '[TC13] JONSWAP 谱必须有限 FAILED'

# ---- TC14: directional_spreading_gaussian 积分归一化 ----
theta = np.linspace(-np.pi, np.pi, 200)
D = directional_spreading_gaussian(theta, 0.0, 4.0)
int_D = np.trapezoid(D, theta)
assert abs(int_D - 1.0) < 0.05, '[TC14] 方向扩散函数积分应约为 1 FAILED'

# ---- TC15: airy_wave_kinematics 输出字典结构 ----
kin = airy_wave_kinematics(0.0, -10.0, 0.0, 1.0, 10.0, 100.0)
assert set(kin.keys()) == {'eta', 'u', 'w', 'u_dot', 'w_dot', 'p_dyn'}, '[TC15] Airy 波输出键缺失 FAILED'

# ---- TC16: build_rigid_body_mass_matrix 对称性和尺寸 ----
M = build_rigid_body_mass_matrix(1e7, np.array([0,0,-5]), np.array([1e10,1e10,1e10]))
assert M.shape == (6, 6), '[TC16] 质量矩阵维度应为 6x6 FAILED'
assert np.allclose(M, M.T, atol=1e-6), '[TC16] 质量矩阵应对称 FAILED'

# ---- TC17: build_hydrostatic_restoring_matrix C33 为正 ----
C = build_hydrostatic_restoring_matrix(1025.0, 9.81, 1000.0, 1e5, 1e5, 0.0, -8.0, -10.0)
assert C[2, 2] > 0, '[TC17] 垂荡恢复刚度 C33 必须为正 FAILED'

# ---- TC18: bdf2_solve 指数衰减解析解 ----
t_arr, y_arr = bdf2_solve(lambda t, y: -y, (0.0, 1.0), np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), n_steps=20)
assert y_arr.shape == (21, 6), '[TC18] BDF2 输出形状应为 (21,6) FAILED'
assert abs(y_arr[-1, 0] - np.exp(-1.0)) < 0.05, '[TC18] BDF2 指数衰减终值误差过大 FAILED'

# ---- TC19: sor_solve 简单三对角系统 ----
A_sor = build_second_difference_1d_sparse(5)
b_sor = np.ones(5)
x_sor, iters, res = sor_solve(A_sor, b_sor, omega=1.5, tol=1e-8, max_iter=1000)
assert x_sor.shape == (5,), '[TC19] SOR 解维度应为 5 FAILED'
assert res < 1e-4, '[TC19] SOR 残差应足够小 FAILED'

# ---- TC20: jacobi_solve_2d_poisson 零右端项得零解 ----
u, err = jacobi_solve_2d_poisson(5, 5, np.zeros((5,5)), tol=1e-6, max_iter=100)
assert u.shape == (5, 5), '[TC20] Jacobi 解维度应为 5x5 FAILED'
assert np.allclose(u, 0.0, atol=1e-5), '[TC20] 零右端项应得零解 FAILED'

# ---- TC21: compute_velocity_from_potential 线性势 ----
phi_lin = np.outer(np.linspace(0,1,5), np.ones(4))
u_vel, v_vel = compute_velocity_from_potential(phi_lin, dx=0.25, dy=1.0)
assert u_vel.shape == phi_lin.shape, '[TC21] 速度场维度应与势场一致 FAILED'
assert np.allclose(v_vel, 0.0, atol=1e-12), '[TC21] y方向均匀势的v速度应为零 FAILED'

# ---- TC22: panel_area_3d 对矩形面板的面积 ----
rect = np.array([[0,0,0], [2,0,0], [2,1,0], [0,1,0]])
area = panel_area_3d(rect)
assert abs(area - 2.0) < 1e-12, '[TC22] 矩形面板面积应为 2 FAILED'

# ---- TC23: compute_hydrodynamic_coefficients_panel_method 输出尺寸 ----
panels_test = [np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])]
A_test, B_test, F_test = compute_hydrodynamic_coefficients_panel_method(panels_test, 1.0)
assert A_test.shape == (6, 6), '[TC23] 附加质量矩阵维度应为 6x6 FAILED'
assert B_test.shape == (6, 6), '[TC23] 辐射阻尼矩阵维度应为 6x6 FAILED'
assert F_test.shape == (6,), '[TC23] 波浪力维度应为 6 FAILED'

# ---- TC24: catenary_mooring_force 零位移返回非零水平力 ----
F_moor = catenary_mooring_force(0.0, 0.0, np.array([-100.0, 0.0]), 150.0, 500.0, 1e9, 1e6)
assert F_moor.shape == (6,), '[TC24] 系泊力维度应为 6 FAILED'
assert F_moor[0] != 0.0, '[TC24] 零位移时系泊水平力应非零 FAILED'

# ---- TC25: rainflow_count_cycles 正弦信号至少提取一个循环 ----
sig = np.sin(np.linspace(0, 4*np.pi, 100))
cycles = rainflow_count_cycles(sig)
assert len(cycles) > 0, '[TC25] 正弦信号应提取到循环 FAILED'

# ---- TC26: miner_damage 空循环返回零 ----
D_empty = miner_damage([], a=1e12, m=3.0)
assert D_empty == 0.0, '[TC26] 空循环 Miner 损伤应为 0 FAILED'

# ---- TC27: reliability_index 解析验证 ----
beta = reliability_index(500.0, 50.0, 300.0, 60.0)
expected_beta = (500.0 - 300.0) / np.sqrt(50.0**2 + 60.0**2)
assert abs(beta - expected_beta) < 1e-12, '[TC27] 可靠度指标解析验证 FAILED'

# ---- TC28: build_seastate_markov_chain 稳态分布归一化 ----
np.random.seed(42)
P, steady, labels = build_seastate_markov_chain(n_states=4, seed=42)
assert abs(np.sum(steady) - 1.0) < 1e-10, '[TC28] 稳态分布应归一化 FAILED'

# ---- TC29: response_spectrum_rao 峰值在共振频率附近 ----
omega_test = np.linspace(0.1, 2.0, 200)
S_w = np.ones_like(omega_test)
S_resp = response_spectrum_rao(omega_test, 1.0, 0.05, S_w)
peak_idx = np.argmax(S_resp)
assert abs(omega_test[peak_idx] - 1.0) < 0.1, '[TC29] RAO 峰值应在固有频率附近 FAILED'

# ---- TC30: spectral_bandwidth_params 矩形谱带宽 ----
omega_rect = np.linspace(0, 1, 100)
S_rect = np.where((omega_rect > 0.3) & (omega_rect < 0.7), 1.0, 0.0)
bw = spectral_bandwidth_params(S_rect, omega_rect)
assert bw['epsilon'] < 0.5, '[TC30] 矩形谱带宽参数应小于 0.5 FAILED'
assert bw['T01'] > 0, '[TC30] T01 必须为正 FAILED'

# ---- TC31: diophantine_nd_nonnegative_solutions 解析验证 ----
sols = diophantine_nd_nonnegative_solutions(np.array([2, 3]), 10)
assert len(sols) == 2, '[TC31] 2x1+3x2=10 应有 2 个解 FAILED'
for sol in sols:
    assert int(np.dot(np.array([2,3]), sol)) == 10, '[TC31] 丢番图解校验 FAILED'

# ---- TC32: wavenumber_discrete_constraint_bragg 输出格式 ----
bragg = wavenumber_discrete_constraint_bragg(156.0, 55.0, 0.0, 5)
assert isinstance(bragg, list), '[TC32] Bragg 约束输出应为列表 FAILED'

# ---- TC33: diffraction_transfer_function_diophantine 输出尺寸 ----
panel_ks_test = np.array([0.035, 0.038, 0.040])
transfer = diffraction_transfer_function_diophantine(panel_ks_test, 0.040, np.array([1,2]), 3)
assert transfer.shape == panel_ks_test.shape, '[TC33] 传递函数维度应与输入波数一致 FAILED'

# ---- TC34: triangle_exact_integral_fem 对参考三角形的验证 ----
ref_nodes = np.array([[0,0], [1,0], [0,1]])
integrals = triangle_exact_integral_fem(ref_nodes, [(0,0), (1,0), (0,1)])
assert abs(integrals[0] - 0.5) < 1e-12, '[TC34] 常数项积分应为 0.5 FAILED'
assert abs(integrals[1] - 1.0/6.0) < 1e-12, '[TC34] x 项积分应为 1/6 FAILED'

# ---- TC35: wave_group_velocity 输出非负有限 ----
f_test = np.array([0.1, 0.2, 0.3])
cg = wave_group_velocity(f_test, h=100.0)
assert cg.shape == f_test.shape, '[TC35] 群速度维度应与频率一致 FAILED'
assert np.all(cg >= 0), '[TC35] 群速度必须非负 FAILED'
assert np.all(np.isfinite(cg)), '[TC35] 群速度必须有限 FAILED'
