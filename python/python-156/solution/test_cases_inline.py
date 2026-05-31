# ---- TC01: main() 返回结果字典包含所有预期键 ----
assert isinstance(results, dict), '[TC01] 结果应为字典 FAILED'
expected_keys = ['Z_nodes', 'T_fem', 'Y_F_fem', 'Y_O_fem', 'chi_fd',
                 'ignition_results', 'turb_stats', 'area_ratio', 'S_n',
                 'p_3d', 't_3d', 'validation', 'errors_ms']
for key in expected_keys:
    assert key in results, f'[TC01] 缺少键 {key} FAILED'

# ---- TC02: Z_nodes 严格单调递增且范围 [0,1] ----
Z_nodes = results['Z_nodes']
assert np.all(np.diff(Z_nodes) > 0), '[TC02] Z_nodes 必须单调递增 FAILED'
assert abs(Z_nodes[0]) < 1e-12, '[TC02] Z_nodes[0] 应为 0 FAILED'
assert abs(Z_nodes[-1] - 1.0) < 1e-12, '[TC02] Z_nodes[-1] 应为 1 FAILED'

# ---- TC03: T_fem 所有值有限且在合理温度范围内 ----
T_fem = results['T_fem']
assert np.all(np.isfinite(T_fem)), '[TC03] T_fem 存在非有限值 FAILED'
assert np.all(T_fem >= 200.0), '[TC03] T_fem 存在低于 200K 的值 FAILED'
assert np.all(T_fem <= 3500.0), '[TC03] T_fem 存在高于 3500K 的值 FAILED'

# ---- TC04: Y_F_fem 所有值在 [0,1] 范围内 ----
Y_F_fem = results['Y_F_fem']
assert np.all(np.isfinite(Y_F_fem)), '[TC04] Y_F_fem 存在非有限值 FAILED'
assert np.all(Y_F_fem >= -1e-6), '[TC04] Y_F_fem 存在负值 FAILED'
assert np.all(Y_F_fem <= 1.0 + 1e-6), '[TC04] Y_F_fem 存在超过1的值 FAILED'

# ---- TC05: Y_O_fem 所有值在 [0,1] 范围内 ----
Y_O_fem = results['Y_O_fem']
assert np.all(np.isfinite(Y_O_fem)), '[TC05] Y_O_fem 存在非有限值 FAILED'
assert np.all(Y_O_fem >= -1e-6), '[TC05] Y_O_fem 存在负值 FAILED'
assert np.all(Y_O_fem <= 1.0 + 1e-6), '[TC05] Y_O_fem 存在超过1的值 FAILED'

# ---- TC06: chi_fd 所有值非负且有限 ----
chi_fd = results['chi_fd']
assert np.all(np.isfinite(chi_fd)), '[TC06] chi_fd 存在非有限值 FAILED'
assert np.all(chi_fd >= 0.0), '[TC06] chi_fd 存在负值 FAILED'

# ---- TC07: ignition_results 包含有效的正临界 Damköhler 数 ----
ig = results['ignition_results']
assert ig['critical_Damkohler'] > 0, '[TC07] 临界 Damköhler 数必须为正 FAILED'
assert ig['converged'], '[TC07] WDK 算法应收敛 FAILED'

# ---- TC08: validation 整体通过 ----
val = results['validation']
assert val['overall_valid'], '[TC08] 模拟验证应整体通过 FAILED'

# ---- TC09: 数值解相对 L² 误差小 ----
errs = results['errors_ms']
assert errs['relative_L2'] < 1.0, '[TC09] 相对 L² 误差应小于 1 FAILED'

# ---- TC10: area_ratio 为正数 ----
assert results['area_ratio'] > 0, '[TC10] 皱褶因子应为正数 FAILED'

# ---- TC11: S_n 为有限正值 ----
S_n = results['S_n']
assert np.isfinite(S_n) and S_n > 0, '[TC11] S_n 应为有限正值 FAILED'

# ---- TC12: 3D 网格有正数节点和单元 ----
assert len(results['p_3d']) > 0, '[TC12] 3D 网格节点数应为正 FAILED'
assert len(results['t_3d']) > 0, '[TC12] 3D 网格单元数应为正 FAILED'

# ---- TC13: scalar_dissipation_rate 在 Z_st 处等于 chi_st ----
chi_test = scalar_dissipation_rate(Z_STOICHIOMETRIC, 2.0)
assert abs(chi_test - 2.0) < 1e-6, '[TC13] χ(Z_st) 应等于 χ_st FAILED'

# ---- TC14: mixture_molecular_weight 在 Z=0 处返回氧化剂分子量 ----
W0 = mixture_molecular_weight(0.0)
assert abs(W0 - 28.97e-3) < 1e-6, '[TC14] W(0) 应等于 M_OX FAILED'

# ---- TC15: arrhenius_rate_constant 随温度单调递增 ----
k1 = arrhenius_rate_constant(500.0, A=1.0, Ea=5000.0)
k2 = arrhenius_rate_constant(1500.0, A=1.0, Ea=5000.0)
assert k2 > k1, '[TC15] Arrhenius 速率常数应随温度递增 FAILED'

# ---- TC16: adiabatic_flame_temperature 返回有限正值 ----
T_ad_test, phi_test = adiabatic_flame_temperature(0.05, 0.232, 300.0)
assert np.isfinite(T_ad_test) and T_ad_test > 300.0, '[TC16] 绝热火焰温度应为大于初温的有限值 FAILED'

# ---- TC17: markstein_length Le=1 时等于 δ_L ----
L_M_eq = markstein_length(1.0, alpha_diff=2.0e-5, S_L=0.4)
delta_L = 2.0e-5 / 0.4
assert abs(L_M_eq - delta_L) < 1e-12, '[TC17] Le=1 时 L_M 应等于 δ_L FAILED'

# ---- TC18: curved_flame_speed 在零曲率时等于层流速度 ----
S_n_zero = curved_flame_speed(0.4, 0.0, 1.0)
assert abs(S_n_zero - 0.4) < 1e-12, '[TC18] 零曲率时 S_n 应等于 S_L FAILED'

# ---- TC19: flame_front_surface_area 返回正面积和比率 ----
area_test, ratio_test = flame_front_surface_area(B=0.01, L=0.03, w=0.003, Ka=0.0)
assert area_test > 0, '[TC19] 火焰表面积应为正 FAILED'
assert ratio_test > 0, '[TC19] 皱褶因子应为正 FAILED'

# ---- TC20: Darrieus-Landau 增长率在 k=0 时为零 ----
sigma_dl_zero = darrieus_landau_growth_rate(0.0)
assert abs(sigma_dl_zero) < 1e-12, '[TC20] k=0 时 D-L 增长率应为零 FAILED'

# ---- TC21: poly_eval 对多项式 x^2 - 4 正确求值 ----
from ignition_polynomial import poly_eval, wdk_roots
c_test = np.array([-4.0, 0.0, 1.0], dtype=complex)
assert abs(poly_eval(c_test, 2.0)) < 1e-12, '[TC21] P(2)=0 for x²-4 FAILED'
assert abs(poly_eval(c_test, 0.0) + 4.0) < 1e-12, '[TC21] P(0)=-4 for x²-4 FAILED'

# ---- TC22: wdk_roots 对 x^2 - 4 找到 ±2 ----
roots_test, conv = wdk_roots(c_test)
assert conv, '[TC22] WDK 应对简单二次多项式收敛 FAILED'
found_p2 = any(abs(r - 2.0) < 1e-6 for r in roots_test)
found_m2 = any(abs(r + 2.0) < 1e-6 for r in roots_test)
assert found_p2 and found_m2, '[TC22] WDK 应找到根 ±2 FAILED'

# ---- TC23: compute_errors 返回正确键且零误差 ----
err_test = compute_errors(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
                          np.array([0.0, 0.5, 1.0]))
assert 'L2_error' in err_test, '[TC23] compute_errors 缺少 L2_error FAILED'
assert 'Linf_error' in err_test, '[TC23] compute_errors 缺少 Linf_error FAILED'
assert abs(err_test['L2_error']) < 1e-12, '[TC23] 相同输入的 L² 误差应为零 FAILED'

# ---- TC24: polynomial_multiply 正确计算卷积 ----
from stoichiometric_polynomial import polynomial_multiply
p_test = polynomial_multiply(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
assert len(p_test) == 3, '[TC24] 乘积长度应为 3 FAILED'
assert abs(p_test[0] - 1.0) < 1e-12, '[TC24] 系数[0] 应为 1 FAILED'
assert abs(p_test[1] - 2.0) < 1e-12, '[TC24] 系数[1] 应为 2 FAILED'
assert abs(p_test[2] - 1.0) < 1e-12, '[TC24] 系数[2] 应为 1 FAILED'

# ---- TC25: chicken_egg_shape 在 x=0 处达到最大半径 B/2 ----
x_mid = np.array([0.0])
r_mid = chicken_egg_shape(0.01, 0.03, 0.0, x_mid)
assert abs(r_mid[0] - 0.005) < 1e-12, '[TC25] x=0 处 r 应等于 B/2 FAILED'

# ---- TC26: mass_conservation_checksum 验证总和为 1 ----
from conservation_validator import mass_conservation_checksum
Y_dict = {'fuel': np.array([0.0, 0.5, 1.0]), 'oxidizer': np.array([1.0, 0.5, 0.0])}
cs, cs_err = mass_conservation_checksum(Y_dict)
assert np.max(np.abs(cs - 1.0)) < 1e-12, '[TC26] 质量分数和应恒为 1 FAILED'

# ---- TC27: domain_decomposition_1d 正确分区 ----
Z_test = np.linspace(0.0, 1.0, 20)
sub_idx, sub_bounds = domain_decomposition_1d(Z_test, 4)
assert len(sub_idx) == 4, '[TC27] 应有 4 个子域 FAILED'
assert sub_idx[0][0] == 0, '[TC27] 第一个子域应从索引 0 开始 FAILED'
assert sub_idx[-1][1] == 20, '[TC27] 最后一个子域应到索引 20 FAILED'

# ---- TC28: flamelet_boundary_conditions 返回正确结构 ----
bc_test = flamelet_boundary_conditions()
assert 'T_left' in bc_test and 'T_right' in bc_test, '[TC28] 边界条件缺少温度键 FAILED'
assert abs(bc_test['T_ad'] - 2226.0) < 1.0, '[TC28] 绝热温度应约为 2226 K FAILED'
