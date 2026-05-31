# ---- TC01: generate_cylindrical_mesh 输出节点和单元形状 ----
nodes, elements = generate_cylindrical_mesh(R=0.15, H=3.0, Nr=8, Nz=20)
assert nodes.shape == (189, 2), '[TC01] 节点形状应为 (189,2) FAILED'
assert elements.shape == (160, 4), '[TC01] 单元形状应为 (160,4) FAILED'

# ---- TC02: compute_jacobian_2d 返回正值 ----
nodes2, elements2 = generate_cylindrical_mesh(R=0.1, H=2.0, Nr=4, Nz=5)
j0 = compute_jacobian_2d(nodes2, elements2[0])
assert j0 > 0, '[TC02] 首个单元 Jacobian 应为正 FAILED'
assert np.isfinite(j0), '[TC02] Jacobian 应为有限值 FAILED'

# ---- TC03: mesh_boundary_segments 返回二维数组 ----
nodes3, elements3 = generate_cylindrical_mesh(R=0.1, H=1.0, Nr=3, Nz=3)
boundary = mesh_boundary_segments(elements3)
assert boundary.ndim == 2, '[TC03] 边界应为二维数组 FAILED'
assert boundary.shape[1] == 2, '[TC03] 边界每段应有2列 FAILED'
assert boundary.shape[0] > 0, '[TC03] 应有边界段 FAILED'

# ---- TC04: boundary_perturb 输出维度为 (2n, 2) ----
import numpy as np
np.random.seed(42)
p4 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
q4 = boundary_perturb(p4, mu=0.1, seed=42)
assert q4.shape == (2 * p4.shape[0], 2), '[TC04] 扰动后形状应为 (2n,2) FAILED'

# ---- TC05: mesh_quality_report 包含所有必要键 ----
nodes5, elements5 = generate_cylindrical_mesh(R=0.1, H=1.0, Nr=2, Nz=2)
report5 = mesh_quality_report(nodes5, elements5)
assert 'jacobian_min' in report5, '[TC05] 缺少 jacobian_min FAILED'
assert 'jacobian_mean' in report5, '[TC05] 缺少 jacobian_mean FAILED'
assert 'jacobian_negative_count' in report5, '[TC05] 缺少 jacobian_negative_count FAILED'
assert report5['jacobian_negative_count'] == 0, '[TC05] 负Jacobian单元数应为0 FAILED'

# ---- TC06: legendre_nodes_weights sin积分精度 ----
x_gl, w_gl = legendre_nodes_weights(32, a=0.0, b=np.pi)
integral_sin = np.sum(w_gl * np.sin(x_gl))
assert abs(integral_sin - 2.0) < 1e-10, '[TC06] ∫_0^π sin(x) dx 应≈2.0 FAILED'

# ---- TC07: gauss_legendre_integral 解析验证——常数函数 ----
val7 = gauss_legendre_integral(lambda x: np.ones_like(x), -1.0, 1.0, n=32)
assert abs(val7 - 2.0) < 1e-10, '[TC07] ∫_{-1}^1 1 dx 应=2.0 FAILED'

# ---- TC08: alpert_log_integral 权重和近似区间长度 ----
def f_alpert_const(x):
    return np.ones_like(x)
val8 = alpert_log_integral(f_alpert_const, h=1.0, rule_idx=5)
assert abs(val8 - 1.0) < 0.05, '[TC08] Alpert 规则权重和应≈1.0 FAILED'

# ---- TC09: chebyshev_coefficients 与 chebyshev_eval 一致性——cos函数逼近 ----
def f_cheb(x):
    return np.cos(np.pi * 0.5 * (x + 1))
coef9 = chebyshev_coefficients(f_cheb, n=16)
y_cheb9 = chebyshev_eval(0.0, coef9)
assert abs(y_cheb9) < 0.01, '[TC09] Chebyshev cos(π/2) 应≈0.0 FAILED'

# ---- TC10: sparse_grid_gauss_legendre 二维高斯积分近似π ----
def f_sparse(x):
    return np.exp(-(x[0]**2 + x[1]**2))
val10, n10 = sparse_grid_gauss_legendre(dim=2, k=4, f=f_sparse, a=-2.0, b=2.0)
assert abs(val10 - np.pi) < 0.2, '[TC10] 稀疏网格 2D 高斯积分应≈π FAILED'
assert n10 > 0, '[TC10] 函数求值次数应为正 FAILED'

# ---- TC11: fixed_point_iteration 求解 x=cos(x) 收敛 ----
np.random.seed(42)
g_cos = lambda x: np.array([np.cos(x[0])])
x_fp, res_fp, it_fp, conv_fp = fixed_point_iteration(
    g_cos, x0=np.array([0.5]), tol=1e-12, max_iter=200,
    bounds=(np.array([-2.0]), np.array([2.0]))
)
assert conv_fp, '[TC11] 定点迭代应收敛 FAILED'
assert abs(x_fp[0] - np.cos(x_fp[0])) < 1e-10, '[TC11] 解应满足 x=cos(x) FAILED'

# ---- TC12: newton_solver 求解 x²-2=0 得到 √2 ----
f_newt = lambda x: np.array([x[0]**2 - 2.0])
j_newt = lambda x: np.array([[2.0 * x[0]]])
x_nt, res_nt, it_nt, conv_nt = newton_solver(
    f_newt, j_newt, x0=np.array([1.5]), tol=1e-12
)
assert conv_nt, '[TC12] Newton 法应收敛 FAILED'
assert abs(x_nt[0]**2 - 2.0) < 1e-10, '[TC12] 解应满足 x²=2 FAILED'

# ---- TC13: reactor_algebraic_residual 返回二维数组 ----
params13 = {'u_g_in': 0.05, 'j_in': 0.05 * 0.25}
state13 = np.array([0.20, 0.003])
f_res13 = reactor_algebraic_residual(state13, params13)
assert f_res13.shape == (2,), '[TC13] 残差应为 (2,) FAILED'
assert np.all(np.isfinite(f_res13)), '[TC13] 残差应为有限值 FAILED'

# ---- TC14: reactor_jacobian 返回 (2,2) 形状 ----
J14 = reactor_jacobian(np.array([0.25, 0.01]), {'u_g_in': 0.05, 'j_in': 0.0125})
assert J14.shape == (2, 2), '[TC14] Jacobian 应为 (2,2) FAILED'

# ---- TC15: HartmannFlow velocity 在中心处有限且对称 ----
np.random.seed(42)
hart = HartmannFlow(G=1.0, Ha=2.0, Re=10.0, Rm=6.0)
u0 = hart.velocity(0.0)
assert np.isfinite(u0), '[TC15] 中心速度应为有限值 FAILED'
up = hart.velocity(0.5)
um = hart.velocity(-0.5)
assert abs(up - um) < 1e-12, '[TC15] 速度应对称 FAILED'

# ---- TC16: HartmannFlow residual_check 残差范数有限 ----
np.random.seed(42)
hart16 = HartmannFlow(G=1.0, Ha=2.0, Re=10.0, Rm=6.0)
y_hart = np.linspace(-0.9, 0.9, 9)
ur16, br16 = hart16.residual_check(y_hart)
assert np.all(np.isfinite(ur16)), '[TC16] 速度残差应为有限值 FAILED'
assert np.all(np.isfinite(br16)), '[TC16] 磁场残差应为有限值 FAILED'

# ---- TC17: interphase_momentum_exchange 返回有限值 ----
np.random.seed(42)
M_gl = interphase_momentum_exchange(
    alpha_g=0.25, u_g=0.05, u_l=0.01,
    rho_l=800.0, mu_l=0.002, d_b=5e-3
)
assert np.isfinite(M_gl), '[TC17] 相间动量交换应为有限值 FAILED'

# ---- TC18: effective_viscosity_slurry 单调递增 ----
mu_eff0 = effective_viscosity_slurry(mu_l=0.002, alpha_s=0.0)
mu_eff1 = effective_viscosity_slurry(mu_l=0.002, alpha_s=0.2)
mu_eff2 = effective_viscosity_slurry(mu_l=0.002, alpha_s=0.4)
assert mu_eff0 < mu_eff1 < mu_eff2, '[TC18] 有效粘度应随 alpha_s 递增 FAILED'
assert mu_eff0 == 0.002, '[TC18] alpha_s=0 时应为纯液粘 FAILED'

# ---- TC19: Schiller_Naumann_CD 高Re下为常数0.44 ----
cd_low = schiller_naumann_cd(np.array([1.0]))
cd_high = schiller_naumann_cd(np.array([2000.0]))
assert cd_low[0] > 0.44, '[TC19] 低Re下 CD 应 >0.44 FAILED'
assert abs(cd_high[0] - 0.44) < 1e-12, '[TC19] 高Re下 CD 应=0.44 FAILED'

# ---- TC20: poisson_nucleation_events 固定种子可复现 ----
np.random.seed(42)
t1, w1, n1 = poisson_nucleation_events(lambda_rate=10.0, t_end=1.0, event_num=20, seed=131)
np.random.seed(42)
t2, w2, n2 = poisson_nucleation_events(lambda_rate=10.0, t_end=1.0, event_num=20, seed=131)
assert np.array_equal(t1, t2), '[TC20] 相同种子应产生相同成核时间 FAILED'
assert n1 == 20, '[TC20] 事件数应为20 FAILED'

# ---- TC21: breakage_frequency_lehr 返回正值 ----
np.random.seed(42)
g_lehr = breakage_frequency_lehr(V=1e-7, C_B=0.5, sigma=0.072, rho_l=800.0)
assert g_lehr > 0, '[TC21] 破裂频率应为正 FAILED'
assert np.isfinite(g_lehr), '[TC21] 破裂频率应为有限值 FAILED'

# ---- TC22: coalescence_kernel_prince_blanch 返回非负有限值 ----
np.random.seed(42)
Q_pb = coalescence_kernel_prince_blanch(
    V_i=1e-7, V_j=2e-7, epsilon=0.1, sigma=0.072, rho_l=800.0
)
assert Q_pb >= 0, '[TC22] 聚并核应为非负 FAILED'
assert np.isfinite(Q_pb), '[TC22] 聚并核应为有限值 FAILED'

# ---- TC23: wheeler_algorithm 2节点返回正确形状并权重非负 ----
np.random.seed(42)
m_init = np.array([1.0, 1.5e-7, 5e-14, 3e-20])
xi23, wi23 = wheeler_algorithm(m_init, n_nodes=2)
assert xi23.shape == (2,), '[TC23] 节点应为 (2,) FAILED'
assert wi23.shape == (2,), '[TC23] 权重应为 (2,) FAILED'
assert np.all(wi23 >= 0), '[TC23] 权重应为非负 FAILED'
assert np.sum(wi23) > 0, '[TC23] 权重和应为正 FAILED'

# ---- TC24: qmom_integrate_pbe 输出形状正确 ----
np.random.seed(42)
m0_init24 = np.array([1.0, 1.5e-7, 5e-14, 3e-20])
t_arr24, m_hist24 = qmom_integrate_pbe(
    m0_init24, (0.0, 0.5), dt=0.01, n_nodes=2,
    rho_l=800.0, sigma=0.072, epsilon=0.05
)
assert m_hist24.shape[1] == 4, '[TC24] 矩历史应有4列 FAILED'
assert t_arr24.shape[0] == m_hist24.shape[0], '[TC24] 时间与矩历史行数应一致 FAILED'
assert np.all(np.isfinite(m_hist24)), '[TC24] 矩历史应为有限值 FAILED'

# ---- TC25: knapsack_brute_force 自洽性——空集价值为0 ----
vals25 = np.array([12.0, 10.0, 8.0, 6.0, 5.0])
wts25 = np.array([4.0, 3.0, 3.0, 2.0, 2.0])
vmax, wmax, smax = knapsack_brute_force(vals25, wts25, capacity=8.0)
assert vmax >= 0, '[TC25] 最大价值应为非负 FAILED'
assert wmax <= 8.0, '[TC25] 总重量不应超过容量 FAILED'
assert np.all((smax == 0) | (smax == 1)), '[TC25] 选择应为0/1 FAILED'
assert vmax >= 12.0, '[TC25] 至少能装入价值12的物品 FAILED'

# ---- TC26: diophantine_bounded_solutions 找到 2x+3y+4z=10 的解 ----
np.random.seed(42)
sols26 = diophantine_bounded_solutions(np.array([2, 3, 4]), 10, np.array([5, 5, 5]))
assert sols26.shape[0] > 0, '[TC26] 应有非平凡解 FAILED'
for sol in sols26:
    assert np.dot(np.array([2, 3, 4]), sol) == 10, f'[TC26] 解{sol}不满足方程 FAILED'

# ---- TC27: catalyst_value_per_segment 返回正值 ----
np.random.seed(42)
W_cat27 = np.array([5.0, 5.0, 5.0])
T_seg27 = np.array([523.0, 543.0, 563.0])
vals27, wts27 = catalyst_value_per_segment(W_cat27, T_seg27, Q_gas=0.01)
assert vals27.shape == (3,), '[TC27] 价值应为 (3,) FAILED'
assert np.all(vals27 >= 0), '[TC27] 价值应为非负 FAILED'
assert np.all(np.isfinite(vals27)), '[TC27] 价值应为有限值 FAILED'

# ---- TC28: optimize_catalyst_loading brute_force 返回字典 ----
np.random.seed(42)
T_prof28 = np.linspace(523.0, 573.0, 5)
result28 = optimize_catalyst_loading(
    W_total=30.0, n_segments=5, T_profile=T_prof28, Q_gas=0.01, method='brute_force'
)
assert 'max_value' in result28, '[TC28] 缺 max_value FAILED'
assert 'selection' in result28, '[TC28] 缺 selection FAILED'

# ---- TC29: detq_orthogonal 旋转矩阵行列式=1 ----
np.random.seed(42)
theta = np.pi / 3.0
Q_ortho = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
d_detq, fault = detq_orthogonal(Q_ortho)
assert fault == 0, '[TC29] 正交矩阵检测不应失败 FAILED'
assert abs(abs(d_detq) - 1.0) < 0.001, '[TC29] 正交矩阵 |det| 应≈1 FAILED'

# ---- TC30: check_mesh_transformation_orthogonality 识别正交矩阵 ----
theta30 = np.pi / 4.0
Jac30 = np.array([[np.cos(theta30), -np.sin(theta30)], [np.sin(theta30), np.cos(theta30)]])
is_ortho30, det30, err30 = check_mesh_transformation_orthogonality(Jac30)
assert is_ortho30, '[TC30] 旋转矩阵应被识别为正交 FAILED'
assert err30 < 0.1, '[TC30] 正交误差应很小 FAILED'

# ---- TC31: power_iteration_eigenvector 收敛且特征值合理 ----
np.random.seed(42)
A31 = np.array([[0.5, 0.5], [0.3, 0.7]])
lam31, vec31, it31, conv31 = power_iteration_eigenvector(A31, max_iter=200, tol=1e-10, damping=0.85)
assert conv31, '[TC31] 幂法应收敛 FAILED'
assert np.isfinite(lam31), '[TC31] 特征值应为有限 FAILED'

# ---- TC32: steady_state_concentration_solver 输出向量正确 ----
np.random.seed(42)
K32 = np.array([[2.0, -1.0], [-0.5, 1.5]])
b32 = np.array([1.0, 1.0])
c32, res32, it32, conv32 = steady_state_concentration_solver(K32, b32, alpha_relax=0.8)
assert c32.shape == (2,), '[TC32] 浓度解应为 (2,) FAILED'
assert conv32, '[TC32] 稳态求解应收敛 FAILED'
r32 = K32 @ c32 - b32
assert np.max(np.abs(r32)) < 1e-5, '[TC32] 解应满足 KC≈b FAILED'

# ---- TC33: estimate_condition_number 返回正值 ----
cond33 = estimate_condition_number(np.array([[2.0, -1.0], [-0.5, 1.5]]))
assert cond33 >= 1.0, '[TC33] 条件数应≥1 FAILED'
assert np.isfinite(cond33), '[TC33] 条件数应为有限 FAILED'

# ---- TC34: generate_inlet_conditions 固定种子可复现统计 ----
np.random.seed(42)
inlet34a = generate_inlet_conditions(50, T_mean=523.0, T_std=3.0,
                                     yCO_mean=0.30, yH2_mean=0.60, y_std=0.015,
                                     Q_mean=0.01, Q_std=0.0005, seed=131)
np.random.seed(42)
inlet34b = generate_inlet_conditions(50, T_mean=523.0, T_std=3.0,
                                     yCO_mean=0.30, yH2_mean=0.60, y_std=0.015,
                                     Q_mean=0.01, Q_std=0.0005, seed=131)
assert inlet34a['statistics']['T_mean'] == inlet34b['statistics']['T_mean'], '[TC34] 相同种子温度均值应相同 FAILED'
assert 'T' in inlet34a, '[TC34] 缺少 T FAILED'
assert 'yCO' in inlet34a, '[TC34] 缺少 yCO FAILED'

# ---- TC35: generate_perturbed_profile 形状不变 ----
np.random.seed(42)
base35 = np.sin(np.linspace(0, np.pi, 20))
pert35 = generate_perturbed_profile(base35, sigma_perturb=0.05, seed=131)
assert pert35.shape == base35.shape, '[TC35] 扰动后形状应不变 FAILED'
assert np.all(pert35 >= 0), '[TC35] 扰动后值应为非负 FAILED'

# ---- TC36: date_to_jdn 与 jdn_to_date 相互可逆 ----
np.random.seed(42)
jdn36 = date_to_jdn(2024, 5, 6)
y, m, d = jdn_to_date(jdn36)
assert (y, m, d) == (2024, 5, 6), '[TC36] 日期往返应一致 FAILED'

# ---- TC37: is_leap_year_gregorian 已知闰年判断 ----
assert is_leap_year_gregorian(2024), '[TC37] 2024应为闰年 FAILED'
assert not is_leap_year_gregorian(2023), '[TC37] 2023不应为闰年 FAILED'
assert not is_leap_year_gregorian(1900), '[TC37] 1900不应为闰年 FAILED'
assert is_leap_year_gregorian(2000), '[TC37] 2000应为闰年 FAILED'

# ---- TC38: reactor_operation_timeline 返回正值天数 ----
np.random.seed(42)
tl38 = reactor_operation_timeline((2024, 1, 1), (2024, 12, 31))
assert tl38['total_days'] == 366, '[TC38] 2024年应有366天 FAILED'
assert tl38['max_cycles'] > 0, '[TC38] 最大批次循环应为正 FAILED'

# ---- TC39: operating_calendar_year 可用率在 [0,1] 范围内 ----
np.random.seed(42)
cal39 = operating_calendar_year(2024, scheduled_downtime_days=[(3, 15), (6, 20)])
assert 0.0 <= cal39['availability'] <= 1.0, '[TC39] 可用率应在 [0,1] FAILED'
assert cal39['total_days'] == 366, '[TC39] 2024年应有366天 FAILED'
assert cal39['downtime_days'] == 2, '[TC39] 应有2个停机日 FAILED'

# ---- TC40: two_fluid_momentum_residual 返回有限值 ----
np.random.seed(42)
res_g40, res_l40 = two_fluid_momentum_residual(
    alpha_g=0.25, u_g=0.05, u_l=0.01, p=2.5e6,
    rho_g=20.0, rho_l=800.0, mu_eff=0.003, g_vec=-9.81, d_b=5e-3, dx=0.01, dy=0.01
)
assert np.isfinite(res_g40), '[TC40] 气相残差应为有限 FAILED'
assert np.isfinite(res_l40), '[TC40] 液相残差应为有限 FAILED'

# ---- TC41: moment_source_qmom 返回4个分量 ----
np.random.seed(42)
m41 = np.array([1.0, 1.5e-7, 5e-14, 3e-20])
xi41, wi41 = wheeler_algorithm(m41, n_nodes=2)
S41 = moment_source_qmom(m41, xi41, wi41, rho_l=800.0, sigma=0.072, epsilon=0.1)
assert S41.shape == (4,), '[TC41] 源项应为 (4,) FAILED'
assert np.all(np.isfinite(S41)), '[TC41] 源项应为有限值 FAILED'

# ---- TC42: gauss_legendre_integral x² 从 -1 到 1 = 2/3 ----
val42 = gauss_legendre_integral(lambda x: x**2, -1.0, 1.0, n=8)
assert abs(val42 - 2.0/3.0) < 1e-10, '[TC42] ∫_{-1}^1 x² dx 应=2/3 FAILED'

# ---- TC43: Newton 法对线性系统一步收敛 ----
f43 = lambda x: np.array([x[0] - 2.0, x[1] - 3.0])
j43 = lambda x: np.array([[1.0, 0.0], [0.0, 1.0]])
x43, res43, it43, conv43 = newton_solver(f43, j43, x0=np.array([0.0, 0.0]), tol=1e-12)
assert conv43, '[TC43] 线性系统Newton应收敛 FAILED'
assert abs(x43[0] - 2.0) < 1e-10, '[TC43] x[0]应≈2 FAILED'
assert abs(x43[1] - 3.0) < 1e-10, '[TC43] x[1]应≈3 FAILED'

# ---- TC44: 固定点迭代对压缩映射收敛（g(x)=0.5x+1） ----
np.random.seed(42)
g44 = lambda x: np.array([0.5 * x[0] + 1.0])
x44, res44, it44, conv44 = fixed_point_iteration(
    g44, x0=np.array([0.0]), tol=1e-12, max_iter=200
)
assert conv44, '[TC44] 压缩映射应收敛 FAILED'
assert abs(x44[0] - 2.0) < 1e-10, '[TC44] 不动点应=2 FAILED'
