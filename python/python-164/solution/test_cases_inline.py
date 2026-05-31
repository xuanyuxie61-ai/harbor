# 补充导入 main.py 中未导入但测试用例需要的函数
from butler_volmer import butler_volmer_current
from ripening_model import kelvin_solubility, ripening_rate, gauss_legendre_integral_exactness
from diffusion_solver import r83_cr_fa, r83_cr_sl, thomas_algorithm
from carbon_corrosion import carbon_mass_loss_rate, numerical_flux_godunov
from ecsa_calculator import power_method_eigenvalue
from catalyst_optimizer import golden_section_search
from morphology_evolution import ubvec_next_gray, mandelbrot_like_escape_time
from ccl_grid import hypercube_grid

# ---- TC01: butler_volmer_current 返回标量且有限 ----
j_bv = butler_volmer_current(0.05, 1e-4, 0.5, 0.5, 4, 353.15)
assert isinstance(j_bv, float), '[TC01] butler_volmer_current 返回值非标量 FAILED'
assert np.isfinite(j_bv), '[TC01] butler_volmer_current 返回非有限值 FAILED'
assert j_bv > 0, '[TC01] butler_volmer_current 正过电位应返回正电流 FAILED'

# ---- TC02: butler_volmer_current 零过电位近似零电流 ----
j_zero = butler_volmer_current(0.0, 1e-4, 0.5, 0.5, 4, 353.15)
assert abs(j_zero) < 1e-6, '[TC02] butler_volmer_current 零过电位应接近零电流 FAILED'

# ---- TC03: exchange_current_density 返回正值 ----
j0_val = exchange_current_density(353.15, 1.2)
assert j0_val > 0, '[TC03] exchange_current_density 应返回正值 FAILED'
assert np.isfinite(j0_val), '[TC03] exchange_current_density 返回非有限值 FAILED'

# ---- TC04: exchange_current_density 温度升高交换电流增大 ----
j0_lo = exchange_current_density(333.15, 1.2)
j0_hi = exchange_current_density(353.15, 1.2)
assert j0_hi > j0_lo, '[TC04] 温度升高交换电流密度应增大 FAILED'

# ---- TC05: solve_overpotential_muller 返回有限值 ----
p = orr_kinetic_parameters()
eta_m = solve_overpotential_muller(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                    p['alpha_a'], p['alpha_c'], p['n'], 353.15)
assert np.isfinite(eta_m), '[TC05] Muller 法过电位应有限 FAILED'
assert abs(eta_m) < 2.0, '[TC05] Muller 法过电位应在 [-2, 2] 范围内 FAILED'

# ---- TC06: solve_overpotential_wdk 返回有限值 ----
eta_w = solve_overpotential_wdk(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                 p['alpha_a'], p['alpha_c'], p['n'], 353.15)
assert np.isfinite(eta_w), '[TC06] WDK 法过电位应有限 FAILED'
assert abs(eta_w) < 2.0, '[TC06] WDK 法过电位应在 [-2, 2] 范围内 FAILED'

# ---- TC07: orr_kinetic_parameters 返回正确结构的字典 ----
p2 = orr_kinetic_parameters(353.15)
assert 'alpha_a' in p2 and 'alpha_c' in p2 and 'n' in p2, '[TC07] orr_kinetic_parameters 缺少必要键 FAILED'
assert p2['n'] == 4, '[TC07] 电子转移数应为 4 FAILED'
assert 0 < p2['alpha_a'] < 1 and 0 < p2['alpha_c'] < 1, '[TC07] 传递系数应在 (0,1) 内 FAILED'

# ---- TC08: generate_ccl_parameter_grid 返回正确形状 ----
pg = generate_ccl_parameter_grid()
assert pg['num_points'] == int(np.prod(pg['ns'])), '[TC08] 网格点数与 ns 乘积不一致 FAILED'
assert pg['grid'].shape[0] == len(pg['names']), '[TC08] 网格维度与名称列表长度不一致 FAILED'

# ---- TC09: sample_operating_condition 返回正确字段 ----
cond = sample_operating_condition(pg, 0)
assert 'temperature_K' in cond, '[TC09] 操作条件缺少 temperature_K FAILED'
assert 333.15 <= cond['temperature_K'] <= 353.15, '[TC09] 温度超出范围 FAILED'

# ---- TC10: effective_diffusivity 返回正值且在合理范围 ----
D_eff = effective_diffusivity(2.1e-5, 0.4)
assert D_eff > 0, '[TC10] 有效扩散系数应大于零 FAILED'
assert D_eff < 2.1e-5, '[TC10] 有效扩散系数应小于体相扩散系数 FAILED'

# ---- TC11: effective_diffusivity 边界 epsilon=0 返回极小正值 ----
D_min = effective_diffusivity(2.1e-5, 0.0)
assert D_min >= 1e-15, '[TC11] epsilon=0 时有效扩散系数应不小于保护值 FAILED'

# ---- TC12: solve_diffusion_tridiagonal 返回正确的输出形状和技术 ----
x_grid, C_tri = solve_diffusion_tridiagonal(1e-9, 100.0, 10e-6, 1.2, N=51)
assert len(x_grid) == 51 and len(C_tri) == 51, '[TC12] 扩散解输出长度应为 51 FAILED'
assert abs(C_tri[0] - 1.2) < 1e-10, '[TC12] 左边界 Dirichlet C=C_0 未满足 FAILED'
assert C_tri[-1] >= 0, '[TC12] 膜侧浓度不应为负 FAILED'

# ---- TC13: solve_diffusion_banded 与三对角法一致 ----
_, C_band = solve_diffusion_banded(1e-9, 100.0, 10e-6, 1.2, N=51)
diff_max = np.max(np.abs(C_tri - C_band))
assert diff_max < 1e-6, '[TC13] 三对角法与带状 LU 分解结果不一致 FAILED'

# ---- TC14: thomas_algorithm 求解已知三对角系统 ----
# 系统: 2x0 - x1 = 1, -xi-1 + 2xi - xi+1 = 0, -xn-2 + 2xn-1 = 0
n_test = 10
lower_t = -np.ones(n_test - 1)
diag_t = 2.0 * np.ones(n_test)
upper_t = -np.ones(n_test - 1)
rhs_t = np.zeros(n_test)
rhs_t[0] = 1.0
x_thomas = thomas_algorithm(lower_t, diag_t, upper_t, rhs_t)
assert np.all(np.isfinite(x_thomas)), '[TC14] Thomas 算法应返回有限值 FAILED'
assert x_thomas[0] > 0, '[TC14] Thomas 算法解不合理 FAILED'

# ---- TC15: r83_cr_fa + r83_cr_sl 求解简单三对角系统 ----
n_cr = 5
a_cr = np.zeros((3, n_cr))
a_cr[0, :] = -1.0    # 上对角线
a_cr[1, :] = 2.0     # 对角线
a_cr[2, :] = -1.0    # 下对角线
b_cr = np.ones(n_cr)
a_cr_factored = r83_cr_fa(n_cr, a_cr)
x_cr = r83_cr_sl(n_cr, a_cr_factored, b_cr)
assert len(x_cr) == n_cr, '[TC15] 循环约化解长度应为 n FAILED'
assert np.all(np.isfinite(x_cr)), '[TC15] 循环约化解应有限 FAILED'

# ---- TC16: kelvin_solubility 随半径增大而减小（单调性） ----
C_r1 = kelvin_solubility(2e-9, 2.5, 9.09e-6, 353.15, 1e-6)
C_r2 = kelvin_solubility(5e-9, 2.5, 9.09e-6, 353.15, 1e-6)
assert C_r1 > C_r2, '[TC16] 曲率半径越小溶解度应越大 FAILED'

# ---- TC17: critical_radius 返回正值 ----
rc = critical_radius(2.5, 9.09e-6, 353.15, 2e-6, 1e-6)
assert rc > 0, '[TC17] 临界半径应大于零 FAILED'
assert np.isfinite(rc), '[TC17] 临界半径应有限 FAILED'

# ---- TC18: ripening_rate 正负与颗粒尺寸关系 ----
# 小于临界半径的颗粒溶解 (rate<0)，大于的熟化长大 (rate>0)
rc_val = critical_radius(2.5, 9.09e-6, 353.15, 2e-6, 1e-6)
rate_small = ripening_rate(rc_val * 0.5, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15)
rate_large = ripening_rate(rc_val * 2.0, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15)
assert rate_small <= 0, '[TC18] 小于临界半径的颗粒应溶解 (rate<=0) FAILED'
assert rate_large >= 0, '[TC18] 大于临界半径的颗粒应长大 (rate>=0) FAILED'

# ---- TC19: evolve_size_distribution 返回正确形状和无负值 ----
import numpy as np
np.random.seed(42)
radii_init = np.random.lognormal(np.log(4e-9), 0.15, 10)
radii_init = np.clip(radii_init, 1e-9, 15e-9)
hist = evolve_size_distribution(radii_init, 1e-12, 9.09e-6, 1e-6, 2e-6, 2.5, 353.15, 3600, 10)
assert hist.shape[0] == 11 and hist.shape[1] == 10, '[TC19] 演化历史形状错误 FAILED'
assert np.all(hist >= 0.5e-9), '[TC19] 颗粒半径不应低于物理下限 0.5 nm FAILED'

# ---- TC20: lsw_analytical_r3 返回正值且>初始半径 ----
r_lsw = lsw_analytical_r3(500 * 3600, 4e-9, 2.5, 1e-12, 9.09e-6, 1e-6, 353.15)
assert r_lsw > 4e-9, '[TC20] LSW 熟化后半径应大于初始半径 FAILED'
assert np.isfinite(r_lsw), '[TC20] LSW 预测半径应有限 FAILED'

# ---- TC21: disk_distance_stats_monte_carlo 可复现性 (固定种子) ----
np.random.seed(42)
mu1, var1 = disk_distance_stats_monte_carlo(radii_init, radii_init)
np.random.seed(42)
mu2, var2 = disk_distance_stats_monte_carlo(radii_init, radii_init)
assert abs(mu1 - mu2) < 1e-15, '[TC21] 固定随机种子应产生相同结果 FAILED'
assert mu1 >= 0, '[TC21] 平均距离不应为负 FAILED'

# ---- TC22: moment_size_distribution 一阶矩等于均值 ----
m1 = moment_size_distribution(radii_init, k=1)
assert abs(m1 - np.mean(radii_init)) < 1e-15, '[TC22] 一阶矩应等于算术均值 FAILED'

# ---- TC23: moment_size_distribution k=0 应返回 1 ----
m0 = moment_size_distribution(radii_init, k=0)
assert abs(m0 - 1.0) < 1e-15, '[TC23] 零阶矩应等于 1 FAILED'

# ---- TC24: gauss_legendre_integral_exactness 线性函数精确积分 ----
nodes = np.array([-0.577350269189626, 0.577350269189626])
weights = np.array([1.0, 1.0])
integral = gauss_legendre_integral_exactness(lambda t: 2.0 * t + 1.0, 2, weights, nodes, a=0, b=1)
assert abs(integral - 1.0) < 1e-12, '[TC24] 两点高斯积分应对线性函数精确 FAILED'

# ---- TC25: corrosion_current_density 非负 ----
j_corr = corrosion_current_density(1.0, T=353.15)
assert j_corr >= 0, '[TC25] 腐蚀电流密度不应为负 FAILED'
assert np.isfinite(j_corr), '[TC25] 腐蚀电流密度应有限 FAILED'

# ---- TC26: corrosion_current_density 低电位下为零 ----
j_corr_low = corrosion_current_density(0.1, E_corr_0=0.207)
assert j_corr_low == 0.0, '[TC26] 低于平衡电位时腐蚀电流应为零 FAILED'

# ---- TC27: corrosion_front_velocity 非负 ----
v_f = corrosion_front_velocity(0.8, 353.15)
assert v_f >= 0, '[TC27] 腐蚀前沿速度不应为负 FAILED'

# ---- TC28: solve_corrosion_propagation 返回正确形状且值非负 ----
nx_c = 21
dx_c = 10e-6 / (nx_c - 1)
u0_c = np.ones(nx_c) * 200.0
U_c = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='godunov')
assert U_c.shape == (51, nx_c), '[TC28] 腐蚀传播结果形状错误 FAILED'
assert np.all(U_c >= 0), '[TC28] 碳比表面积不应为负 FAILED'

# ---- TC29: solve_corrosion_propagation Lax-Wendroff 格式可运行 ----
U_lw = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='lax_wendroff')
assert U_lw.shape == (51, nx_c), '[TC29] Lax-Wendroff 格式应返回正确形状 FAILED'
assert np.all(np.isfinite(U_lw)), '[TC29] Lax-Wendroff 格式应产生有限值 FAILED'

# ---- TC30: structural_integrity_loss 在 [0, 1] 范围内 ----
loss = structural_integrity_loss(150.0, 200.0)
assert 0.0 <= loss <= 1.0, '[TC30] 结构完整性损失应在 [0, 1] 内 FAILED'
assert abs(loss - 0.25) < 1e-10, '[TC30] 150/200 损失应为 0.25 FAILED'

# ---- TC31: carbon_mass_loss_rate 返回负值（质量损失） ----
rate_mass = carbon_mass_loss_rate(10.0, 1.0)
assert rate_mass < 0, '[TC31] 碳质量损失速率应为负 FAILED'

# ---- TC32: ecsa_from_size_distribution 返回正值 ----
radii_ecsa = np.array([2e-9, 3e-9, 4e-9, 5e-9])
ecsa_val = ecsa_from_size_distribution(radii_ecsa)
assert ecsa_val > 0, '[TC32] ECSA 应为正值 FAILED'

# ---- TC33: ECSA 随粒径增大而减小 ----
radii_small = np.array([2e-9, 2.5e-9, 3e-9])
radii_large = np.array([4e-9, 5e-9, 6e-9])
ecsa_s = ecsa_from_size_distribution(radii_small)
ecsa_l = ecsa_from_size_distribution(radii_large)
assert ecsa_s > ecsa_l, '[TC33] 小粒径 ECSA 应大于大粒径 ECSA FAILED'

# ---- TC34: ecsa_loss_kinetics 单调衰减 ----
ecsa0 = 50.0
ecsa_t1 = ecsa_loss_kinetics(ecsa0, 100)
ecsa_t2 = ecsa_loss_kinetics(ecsa0, 200)
assert 0 <= ecsa_t2 < ecsa_t1 <= ecsa0, '[TC34] ECSA 应随时间单调衰减 FAILED'

# ---- TC35: voltage_loss_from_ecsa 单调性 ----
dV_50 = voltage_loss_from_ecsa(0.5)
dV_80 = voltage_loss_from_ecsa(0.8)
assert dV_50 > dV_80, '[TC35] ECSA 保留越少电压损失应越大 FAILED'

# ---- TC36: total_ecsa_loss_model 衰减行为 ----
ecsa_tot = total_ecsa_loss_model(100, 50.0)
assert 0 <= ecsa_tot <= 50.0, '[TC36] 综合 ECSA 应在 [0, 初始值] 内 FAILED'

# ---- TC37: build_stability_jacobian + stability_analysis_max_eigenvalue ----
J_s = build_stability_jacobian(3, [1e-4, 2e-4, 5e-5],
                                np.array([[-1e-4, 2e-5, 0],
                                          [3e-5, -2e-4, 1e-5],
                                          [0, 4e-5, -5e-5]]))
lam_s, stab_s = stability_analysis_max_eigenvalue(J_s)
assert stab_s in ('stable', 'unstable', 'critical'), '[TC37] 稳定性判断必须是 stable/unstable/critical 之一 FAILED'
assert np.isfinite(lam_s), '[TC37] 主导特征值应有限 FAILED'

# ---- TC38: power_method_eigenvalue 对角矩阵精确特征值 ----
A_diag = np.diag([3.0, 1.0, 2.0])
lam_pm, _, _ = power_method_eigenvalue(A_diag, it_max=100, tol=1e-12)
assert abs(lam_pm - 3.0) < 1e-6, '[TC38] 幂法对角矩阵最大特征值应为 3.0 FAILED'

# ---- TC39: golden_section_search 找到已知最小值 ----
def f_test(x):
    return (x - 0.3) ** 2
a_gs, b_gs, it_gs, x_opt_gs, f_opt_gs = golden_section_search(f_test, 0.0, 1.0, n_max=100, x_tol=1e-10)
assert abs(x_opt_gs - 0.3) < 1e-6, '[TC39] 黄金分割应找到 x*=0.3 FAILED'
assert abs(f_opt_gs) < 1e-10, '[TC39] 最小值应接近零 FAILED'
assert it_gs > 0, '[TC39] 黄金分割应至少迭代一次 FAILED'

# ---- TC40: power_performance 单调递增且有界 ----
P_01 = power_performance(0.01)
P_02 = power_performance(0.02)
assert 0 <= P_01 < P_02 <= 1.0, '[TC40] 功率应随负载单调递增且在 [0, 1] 内 FAILED'

# ---- TC41: catalyst_cost 线性关系 ----
cost1 = catalyst_cost(0.1, area_active=250.0, price_pt=50.0)
cost2 = catalyst_cost(0.2, area_active=250.0, price_pt=50.0)
assert abs(cost2 / cost1 - 2.0) < 1e-10, '[TC41] 催化剂成本应与负载成正比 FAILED'

# ---- TC42: optimize_catalyst_loading 返回合理区间内结果 ----
L_opt, J_opt, info = optimize_catalyst_loading(L_min=0.02, L_max=0.5)
assert 0.02 <= L_opt <= 0.5, '[TC42] 最优负载应在搜索区间内 FAILED'
assert info['iterations'] > 0, '[TC42] 优化应至少执行一次迭代 FAILED'

# ---- TC43: sensitivity_analysis 返回有限值 ----
sens = sensitivity_analysis(L_opt)
assert np.isfinite(sens), '[TC43] 敏感性分析应返回有限值 FAILED'

# ---- TC44: SparseAssembler 创建矩阵正确形状 ----
assembler = SparseAssembler(n_interior=10, n_boundary=2)
A_full, A_band = assembler.assemble_diffusion_reaction_matrix(1e-9, 100.0, 1e-7)
assert A_full.shape == (10, 10), '[TC44] 组装稠密矩阵形状应为 (10,10) FAILED'
assert A_band.shape[1] == 10, '[TC44] 带状矩阵列数应为 10 FAILED'

# ---- TC45: SparseAssembler 对角占优检查 ----
is_dd, ratio = assembler.check_diagonal_dominance(A_full)
assert isinstance(is_dd, bool), '[TC45] 对角占优判断应为布尔值 FAILED'
assert ratio > 0, '[TC45] 对角占优比应为正 FAILED'

# ---- TC46: harwell_boeing_metadata 返回正确结构 ----
meta = harwell_boeing_metadata(10, 10, 28)
assert meta['nrow'] == 10 and meta['ncol'] == 10, '[TC46] Harwell-Boeing 元数据维度不正确 FAILED'

# ---- TC47: box_counting_dimension 圆形约为 1.0 ----
theta_circ = np.linspace(0, 2 * np.pi, 200)
x_circ = np.cos(theta_circ)
y_circ = np.sin(theta_circ)
D_f_circ = box_counting_dimension(x_circ, y_circ)
assert 0.5 <= D_f_circ <= 1.5, '[TC47] 圆的分形维数应在 0.5-1.5 范围内 FAILED'

# ---- TC48: enumerate_catalyst_surface_states 输出形状正确 ----
states = enumerate_catalyst_surface_states(4, max_states=16)
assert states.shape == (16, 4), '[TC48] 4位点16状态枚举形状应为 (16,4) FAILED'
# 相邻状态汉明距离为1 (格雷码性质)
hamming = np.sum(np.abs(states[0] - states[1]))
assert hamming == 1, '[TC48] 格雷码相邻状态汉明距离应为 1 FAILED'

# ---- TC49: ubvec_next_gray 遍历所有状态 ----
t_vec = np.zeros(4, dtype=int)
all_vecs = [t_vec.copy()]
for _ in range(15):
    t_vec = ubvec_next_gray(t_vec)
    all_vecs.append(t_vec.copy())
# 检查是否遍历了16个不同状态
unique_vecs = set(tuple(v) for v in all_vecs)
assert len(unique_vecs) == 16, '[TC49] ubvec_next_gray 应遍历 16 个不同状态 FAILED'

# ---- TC50: morphology_degradation_index 在 [0, 1] 内 ----
mdi = morphology_degradation_index(1.8, 1.5, 0.2)
assert 0.0 <= mdi <= 1.0, '[TC50] 形貌退化指数应在 [0, 1] 内 FAILED'

# ---- TC51: pore_network_connectivity_map 返回有效连通性值 ----
conn_map, _, _ = pore_network_connectivity_map(n_grid=16, max_iter=30)
assert conn_map.shape == (16, 16), '[TC51] 连通性图形状应为 (16,16) FAILED'
assert np.all((conn_map >= 0) & (conn_map <= 1)), '[TC51] 连通性应在 [0, 1] 内 FAILED'

# ---- TC52: effective_surface_area_fractal 返回正值 ----
A_eff_f = effective_surface_area_fractal(50.0, 1e-6, 1e-9, 1.8)
assert A_eff_f > 0, '[TC52] 分形有效比表面积应为正 FAILED'
assert np.isfinite(A_eff_f), '[TC52] 分形有效比表面积应有限 FAILED'

# ---- TC53: mandelbrot_like_escape_time 在 Mandelbrot 集内返回 max_iter ----
escape_in = mandelbrot_like_escape_time(0.0, 0.0, max_iter=30)
assert escape_in == 30, '[TC53] z=0 应在 Mandelbrot 集内 (未逃逸) FAILED'
escape_out = mandelbrot_like_escape_time(2.0, 0.0, max_iter=30)
assert escape_out < 30, '[TC53] c=2 应快速逃逸 FAILED'

# ---- TC54: carbon_mass_loss_rate 零电流返回零 ----
rate_zero = carbon_mass_loss_rate(0.0, 1.0)
assert rate_zero == 0.0, '[TC54] 零腐蚀电流应返回零质量损失 FAILED'

# ---- TC55: pt_dissolution_parameters 返回正确字段字典 ----
pt_p = pt_dissolution_parameters()
assert 'gamma' in pt_p and 'D_Pt2' in pt_p and 'V_m' in pt_p, '[TC55] pt_dissolution_parameters 缺少必要字段 FAILED'
assert pt_p['gamma'] > 0 and pt_p['D_Pt2'] > 0, '[TC55] Pt 物理参数必须为正 FAILED'

# ---- TC56: solve_corrosion_propagation MacCormack 格式可运行 ----
U_mc = solve_corrosion_propagation(u0_c, nx_c, 50, dx_c, 360.0, 1e-12, 1e-5, 0.4, method='maccormack')
assert U_mc.shape == (51, nx_c), '[TC56] MacCormack 格式应返回正确形状 FAILED'
assert np.all(np.isfinite(U_mc)), '[TC56] MacCormack 格式应产生有限值 FAILED'

# ---- TC57: hypercube_grid 返回正确维度的网格 ----
ng = hypercube_grid(m=2, n=4, ns=[2, 2], a=[0.0, 0.0], b=[1.0, 1.0], c=[1, 1])
assert ng.shape == (2, 4), '[TC57] hypercube_grid 2D×2×2 应返回 (2,4) FAILED'
assert np.all((ng >= 0) & (ng <= 1)), '[TC57] 网格点应在 [0,1] 范围内 FAILED'

# ---- TC58: structural_integrity_loss 无损失时返回零 ----
loss_none = structural_integrity_loss(200.0, 200.0)
assert abs(loss_none) < 1e-15, '[TC58] 无碳损失时完整性损失应为零 FAILED'

# ---- TC59: numerical_flux_godunov 正速度取左侧通量 ----
flux_pos = numerical_flux_godunov(3.0, 5.0, v=1.0)
assert abs(flux_pos - 3.0) < 1e-15, '[TC59] v>0 时 Godunov 通量应取左侧值 FAILED'

# ---- TC60: numerical_flux_godunov 负速度取右侧通量 ----
flux_neg = numerical_flux_godunov(3.0, 5.0, v=-1.0)
assert abs(flux_neg - 5.0) < 1e-15, '[TC60] v<0 时 Godunov 通量应取右侧值 FAILED'
