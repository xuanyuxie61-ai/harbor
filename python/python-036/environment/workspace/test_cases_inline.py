# ---- TC01: PMNS 矩阵幺正性验证 ----
U = build_pmns_matrix()
is_unitary, err = check_unitarity(U)
assert is_unitary, '[TC01] PMNS 矩阵幺正性验证 FAILED'
assert err < 1e-8, '[TC01] PMNS 矩阵幺正性误差过大 FAILED'

# ---- TC02: 质量矩阵为对角矩阵 ----
M2_NH = build_mass_matrix(hierarchy='normal')
assert M2_NH.shape == (3, 3), '[TC02] 质量矩阵形状 FAILED'
assert np.allclose(M2_NH, np.diag(np.diag(M2_NH))), '[TC02] 质量矩阵非对角 FAILED'
M2_IH = build_mass_matrix(hierarchy='inverted')
assert M2_IH[2, 2] < 0, '[TC02] 反质量矩阵对角元符号 FAILED'

# ---- TC03: Jarlskog 不变量范围 ----
U = build_pmns_matrix()
J = jarkslog_invariant(U)
assert -0.5 < J < 0.5, '[TC03] Jarlskog 不变量范围 FAILED'

# ---- TC04: 真空哈密顿量为厄米矩阵 ----
H_vac = build_vacuum_hamiltonian(2.0)
is_herm, herm_err = validate_hermitian(H_vac)
assert is_herm, '[TC04] 真空哈密顿量厄米性 FAILED'
assert herm_err < 1e-10, '[TC04] 真空哈密顿量厄米误差过大 FAILED'

# ---- TC05: 物质哈密顿量本征值升序排列 ----
V_test = matter_potential_eV(0.5)
H_mat = build_matter_hamiltonian(2.0, V_test)
ev_mat, evec_mat, U_mat_matter = solve_hamiltonian_eigen(H_mat)
is_ordered = validate_eigenvalue_ordering(ev_mat)
assert is_ordered, '[TC05] 物质哈密顿量本征值排序 FAILED'

# ---- TC06: Normal Hierarchy 质量为正且递增 ----
bounds = mass_sum_bounds('normal', m_lightest_eV=0.0)
assert bounds['m1'] >= 0, '[TC06] m1 为负 FAILED'
assert bounds['m2'] > bounds['m1'], '[TC06] m2 不大于 m1 FAILED'
assert bounds['m3'] > bounds['m2'], '[TC06] m3 不大于 m2 FAILED'
assert bounds['sum'] > 0, '[TC06] 质量和非正 FAILED'

# ---- TC07: 振荡波长为正 ----
waves = compute_oscillation_wavelengths(2.0)
assert waves['L_21'] > 0, '[TC07] L_21 非正 FAILED'
assert waves['L_31'] > 0, '[TC07] L_31 非正 FAILED'
assert waves['L_32'] > 0, '[TC07] L_32 非正 FAILED'

# ---- TC08: ODE 矩阵指数法概率守恒 ----
res_exact = solve_neutrino_oscillation_ode(2.0, 1000.0, method='matrix_exp')
prob_sum = np.sum(res_exact['prob_final'])
assert abs(prob_sum - 1.0) < 1e-6, '[TC08] ODE 矩阵指数法概率守恒 FAILED'

# ---- TC09: Euler 和 RK4 方法概率和接近 1 ----
res_euler = solve_neutrino_oscillation_ode(2.0, 1000.0, method='euler', n_steps=5000)
res_rk4 = solve_neutrino_oscillation_ode(2.0, 1000.0, method='rk4', n_steps=1000)
assert abs(np.sum(res_euler['prob_final']) - 1.0) < 1e-3, '[TC09] Euler 概率和 FAILED'
assert abs(np.sum(res_rk4['prob_final']) - 1.0) < 1e-6, '[TC09] RK4 概率和 FAILED'

# ---- TC10: R8BUT 求解器与 numpy 结果一致 ----
A_test = np.array([[2.0, 1.0, 0.5], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]], dtype=np.float64)
b_test = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x_r8but = solve_banded_upper_triangular(A_test, b_test)
x_exact = np.linalg.solve(A_test, b_test)
assert np.allclose(x_r8but, x_exact, atol=1e-10), '[TC10] R8BUT 求解精度 FAILED'

# ---- TC11: 1D FEM 稳态密度在边界范围内 ----
r_nodes = np.linspace(0, EARTH_RADIUS_KM, 51)
rho_1d, r_1d = solve_steady_state_density_1d(r_nodes)
assert len(rho_1d) == len(r_nodes), '[TC11] 1D FEM 输出长度 FAILED'
assert np.all(rho_1d >= 0), '[TC11] 1D FEM 密度为负 FAILED'
assert rho_1d[0] > rho_1d[-1], '[TC11] 1D FEM 中心密度不大于表面密度 FAILED'

# ---- TC12: 2D FEM 密度求解输出结构正确 ----
rho_2d, nodes_2d, elements_2d = solve_steady_state_density_2d(radius_km=100.0, n_r=5, n_theta=8)
bw = compute_bandwidth(3, elements_2d)
assert len(rho_2d) == len(nodes_2d), '[TC12] 2D FEM 节点数匹配 FAILED'
assert bw > 0, '[TC12] 带宽非正 FAILED'
assert np.all(rho_2d >= 0), '[TC12] 2D FEM 密度为负 FAILED'

# ---- TC13: 四面体网格质量评估返回正确结构 ----
nodes_3d, tetra_3d = generate_earth_tetrahedral_mesh(n_r=4, n_theta=6, n_phi=6)
if len(tetra_3d) > 0:
    quality = evaluate_mesh_quality(nodes_3d, tetra_3d)
    assert 'q1_mean' in quality, '[TC13] 网格质量字典结构 FAILED'
    assert quality['n_tetra'] == len(tetra_3d), '[TC13] 网格质量四面体数不匹配 FAILED'
    assert quality['volume_total'] > 0, '[TC13] 网格总体积非正 FAILED'

# ---- TC14: 索引转换与增量正确性 ----
idx = np.array([0, 1, 2, 3])
idx_1 = convert_index_base(idx, 0, 1)
assert np.array_equal(idx_1, np.array([1, 2, 3, 4])), '[TC14] 索引 0->1 转换 FAILED'
idx_0 = convert_index_base(idx_1, 1, 0)
assert np.array_equal(idx_0, idx), '[TC14] 索引 1->0 转换 FAILED'
idx_inc = increment_indices(idx, 5)
assert np.array_equal(idx_inc, np.array([5, 6, 7, 8])), '[TC14] 索引增量 FAILED'

# ---- TC15: 矩阵文件读写一致性 ----
test_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
write_matrix_file('test_io_tmp.txt', test_mat, header='test')
read_mat = read_matrix_file('test_io_tmp.txt')
assert np.allclose(test_mat, read_mat), '[TC15] 矩阵文件读写一致性 FAILED'
import os
os.remove('test_io_tmp.txt')

# ---- TC16: 圆积分规则权重和为 1 ----
w, ang = circle_rule(12)
assert abs(np.sum(w) - 1.0) < 1e-14, '[TC16] 圆积分权重和 FAILED'
assert ang[0] == 0.0, '[TC16] 圆积分起始角度 FAILED'
assert ang[-1] < 2 * np.pi, '[TC16] 圆积分终止角度超界 FAILED'

# ---- TC17: 2D 中点法则积分常数函数 ----
def f_const(x, y):
    return 2.0
I_const = midpoint_quad_2d(4, 4, 0.0, 1.0, 0.0, 2.0, f_const)
assert abs(I_const - 4.0) < 1e-14, '[TC17] 2D 中点法则常数积分 FAILED'

# ---- TC18: 自适应 Simpson 积分 sin^2 精度 ----
def f_sin2(x):
    return np.sin(x) ** 2
I_adaptive = adaptive_integral_1d(f_sin2, 0.0, np.pi, tol=1e-8)
assert abs(I_adaptive - np.pi / 2.0) < 1e-6, '[TC18] 自适应积分精度 FAILED'

# ---- TC19: 蒙特卡洛振荡概率在物理范围内 ----
np.random.seed(42)
mc_result = monte_carlo_oscillation_probability(
    energy_range_gev=(1.0, 5.0),
    baseline_range_km=(100.0, 500.0),
    n_samples=200,
    hierarchy='normal',
    param_uncertainties={},
    seed=42
)
assert 0.0 <= mc_result['P_ee_mean'] <= 1.0, '[TC19] MC P_ee 均值超界 FAILED'
assert mc_result['P_ee_std'] >= 0, '[TC19] MC P_ee 标准差为负 FAILED'
assert abs(mc_result['P_ee_mean'] + mc_result['P_em_mean'] + mc_result['P_et_mean'] - 1.0) < 0.1, '[TC19] MC 概率和偏离 1 FAILED'

# ---- TC20: MC hierarchy 显著性统计量合理 ----
np.random.seed(42)
mc_hier = mc_hierarchy_significance(2.0, 1000.0, n_samples=5000, seed=123)
assert 0.0 <= mc_hier['nh_correct_rate'] <= 1.0, '[TC20] NH 正确率超界 FAILED'
assert 0.0 <= mc_hier['ih_correct_rate'] <= 1.0, '[TC20] IH 正确率超界 FAILED'
assert mc_hier['nh_confidence_sigma'] > 0, '[TC20] NH 置信度非正 FAILED'

# ---- TC21: MSW 共振电子数密度为正 ----
ne_res = msw_resonance_density(2.0)
assert ne_res > 0, '[TC21] MSW 共振密度非正 FAILED'

# ---- TC22: Fermi-Dirac 分布值在 [0,1] 内 ----
fd = fermi_dirac_distribution(1.0, 0.5, chemical_potential=0.0)
assert 0.0 <= fd <= 1.0, '[TC22] Fermi-Dirac 分布超界 FAILED'

# ---- TC23: log Gamma(5) 近似与精确值一致 ----
ln_g, fault = log_gamma_pike_hill(5.0)
assert fault == 0, '[TC23] log Gamma 错误码 FAILED'
assert abs(ln_g - np.log(24)) < 1e-6, '[TC23] log Gamma(5) 精度 FAILED'

# ---- TC24: 幂迭代主导特征值精度 ----
M_diag = np.diag([1.0, 2.0, 3.0])
ev_pi, vec_pi, conv = power_iteration(M_diag, n_iterations=200, seed=42)
assert abs(ev_pi - 3.0) < 1e-6, '[TC24] 幂迭代主导特征值 FAILED'
assert conv, '[TC24] 幂迭代未收敛 FAILED'

# ---- TC25: PageRank 风格矩阵每列和为 1 ----
H_test = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)
P_pr = pagerank_style_matrix(H_test, damping=0.85)
col_sums = np.sum(P_pr, axis=0)
assert np.allclose(col_sums, 1.0, atol=1e-10), '[TC25] PageRank 列和不为 1 FAILED'

# ---- TC26: hierarchy 判别显著性为正且类型正确 ----
sig_NH, hier_NH = hierarchy_discrimination_significance(DELTA_M2_31, sigma_dm31=0.03e-3)
assert sig_NH > 0, '[TC26] NH 显著性非正 FAILED'
assert hier_NH == 'normal', '[TC26] NH hierarchy 类型判断 FAILED'
sig_IH, hier_IH = hierarchy_discrimination_significance(DELTA_M2_31_IH, sigma_dm31=0.03e-3)
assert hier_IH == 'inverted', '[TC26] IH hierarchy 类型判断 FAILED'

# ---- TC27: 有效混合角在物质中处于物理范围 ----
eff_angles = effective_mixing_angles_in_matter(2.0, matter_potential_eV(0.5))
assert 0.0 <= eff_angles['theta12_m'] <= np.pi / 2, '[TC27] 有效混合角12超界 FAILED'
assert 0.0 <= eff_angles['theta13_m'] <= np.pi / 2, '[TC27] 有效混合角13超界 FAILED'
assert 0.0 <= eff_angles['theta23_m'] <= np.pi / 2, '[TC27] 有效混合角23超界 FAILED'

# ---- TC28: 主导振荡模式频率非负 ----
dom_mode = find_dominant_oscillation_mode(H_vac)
assert dom_mode['frequency'] >= 0, '[TC28] 主导模式频率为负 FAILED'
assert np.linalg.norm(dom_mode['state']) > 0.99, '[TC28] 主导模式态向量未归一化 FAILED'

# ---- TC29: 变物质密度 ODE 概率守恒 ----
def V_const(x_km):
    return 1e-13
result_vary = solve_varying_matter_ode(2.0, 100.0, V_const, n_steps=500, method='rk4')
prob_sum_vary = np.sum(result_vary['prob_final'])
assert abs(prob_sum_vary - 1.0) < 1e-3, '[TC29] 变物质 ODE 概率守恒 FAILED'

# ---- TC30: 迭代 hierarchy 求解器返回合理差异 ----
hier_iter = iterative_hierarchy_solver(2.0, 1000.0)
assert hier_iter['delta_P'] >= 0, '[TC30] delta_P 为负 FAILED'
assert 0.0 <= hier_iter['P_ee_NH'] <= 1.0, '[TC30] P_ee_NH 超界 FAILED'
assert 0.0 <= hier_iter['P_ee_IH'] <= 1.0, '[TC30] P_ee_IH 超界 FAILED'
