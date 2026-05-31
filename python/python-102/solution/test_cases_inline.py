# ---- TC01: norm_l2 对标量数组返回有限非负值 ----
ca = ConvergenceAnalysis()
x = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
l2 = ca.norm_l2(x)
assert np.isfinite(l2) and l2 >= 0.0, '[TC01] norm_l2 应返回有限非负值 FAILED'

# ---- TC02: norm_l2 带权重时与直接计算一致 ----
w = np.array([2.0, 1.0, 1.0])
l2_w = ca.norm_l2(x, weights=w)
expected = np.sqrt(2 * 1 + 1 * 4 + 1 * 9)
assert abs(l2_w - expected) < 1e-12, '[TC02] norm_l2 带权重计算 FAILED'

# ---- TC03: norm_linfty 返回最大值 ----
diff = np.array([0.3, -1.5, 0.8, -0.2], dtype=np.float64)
max_val = ca.norm_linfty(diff)
assert abs(max_val - 1.5) < 1e-12, '[TC03] norm_linfty 应返回 1.5 FAILED'

# ---- TC04: norm_linfty 返回最大值及位置 ----
pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float64)
max_val2, max_pt = ca.norm_linfty(diff, sample_points=pts)
assert abs(max_val2 - 1.5) < 1e-12, '[TC04] norm_linfty 带位置 FAILED'
assert np.allclose(max_pt, [1, 1]), '[TC04] 最大值位置应为 (1,1) FAILED'

# ---- TC05: norm_h1_semi 返回有限非负值 ----
gx = np.array([0.1, 0.2, 0.3])
gy = np.array([0.4, 0.5, 0.6])
h1 = ca.norm_h1_semi(gx, gy)
assert np.isfinite(h1) and h1 >= 0.0, '[TC05] norm_h1_semi 应返回有限非负值 FAILED'

# ---- TC06: box_distance_stats 均值非负且解析近似一致性 ----
np.random.seed(42)
mu_bd, var_bd, m2_bd = ca.box_distance_stats(10000, 1.0e-6, 2.0e-6, 3.0e-6)
assert mu_bd > 0.0 and var_bd > 0.0, '[TC06] box_distance_stats 均值和方差应为正 FAILED'
mu_ana = ca.box_distance_analytical(1.0e-6, 2.0e-6, 3.0e-6)
rel_diff = abs(mu_bd - mu_ana) / mu_ana
assert rel_diff < 0.2, '[TC06] MC 均值与解析近似偏差应 < 20% FAILED'

# ---- TC07: estimate_convergence_rate 对纯幂律返回精确阶数 ----
h_test = np.array([0.1, 0.05, 0.025, 0.0125])
err_test = 2.0 * h_test ** 2.0
p_est, C_est, r2_est = ca.estimate_convergence_rate(err_test, h_test)
assert abs(p_est - 2.0) < 0.01, '[TC07] 收敛阶应接近 2.0 FAILED'
assert r2_est > 0.999, '[TC07] R² 应 > 0.999 FAILED'

# ---- TC08: richardson_extrapolation 保守性 ----
f_h, f_h2, f_h4 = 0.90, 0.95, 0.975
f_ext = ca.richardson_extrapolation(f_h, f_h2, f_h4, order=2)
assert f_ext > f_h2, '[TC08] Richardson 外推值应大于中网格值 FAILED'

# ---- TC09: gci_calculation 对无变化解返回近零 ----
gci_zero = ca.gci_calculation(1.0, 1.0, 1.0, r=2.0, p=2.0)
assert gci_zero < 1e-14, '[TC09] GCI 对无变化解应接近 0 FAILED'

# ---- TC10: mc_convergence_test 累积均值误差渐减 ----
np.random.seed(42)
mc_samp = np.random.randn(3000)
cm, se = ca.mc_convergence_test(mc_samp, batch_size=100)
assert len(cm) == 30, '[TC10] MC batch 数应为 30 FAILED'
assert se[-1] < se[0], '[TC10] 标准误差应收敛 FAILED'

# ---- TC11: evaluate_phase_error 对零误差返回零 L2 误差 ----
xg = np.linspace(0, 1, 10)
yg = np.linspace(0, 1, 10)
phi_ex = np.ones((10, 10))
err_dict = ca.evaluate_phase_error(phi_ex, phi_ex.copy(), xg, yg)
assert err_dict['L2_error'] < 1e-14, '[TC11] 零误差的 L2 应为 0 FAILED'
assert err_dict['Linf_error'] < 1e-14, '[TC11] 零误差的 Linf 应为 0 FAILED'

# ---- TC12: MaxwellFEM2D 网格生成输出尺寸正确 ----
fem = MaxwellFEM2D(wavelength=1.55e-6)
nodes_m, elements_m = fem.build_rectangular_mesh(5, 5, (-1e-6, 1e-6), (-1e-6, 1e-6))
n_expect = (2 * 5 - 1) * (2 * 5 - 1)
e_expect = 2 * (5 - 1) * (5 - 1)
assert nodes_m.shape == (n_expect, 2), '[TC12] 节点数不正确 FAILED'
assert elements_m.shape == (e_expect, 6), '[TC12] 单元数不正确 FAILED'

# ---- TC13: MaxwellFEM2D quad_points_t6 权重之和为 1 ----
qp, wq = MaxwellFEM2D.quad_points_t6()
assert abs(np.sum(wq) - 1.0) < 1e-14, '[TC13] 积分权重之和应为 1 FAILED'

# ---- TC14: MaxwellFEM2D shape_t6 在角点评估基函数 ----
N_c, dNdr_c, dNds_c = fem.shape_t6(0.0, 0.0)
assert abs(N_c[0] - 1.0) < 1e-14, '[TC14] N1(0,0) 应为 1 FAILED'
assert abs(N_c[1]) < 1e-14, '[TC14] N2(0,0) 应为 0 FAILED'
assert abs(N_c[2]) < 1e-14, '[TC14] N3(0,0) 应为 0 FAILED'

# ---- TC15: MaxwellFEM2D epsilon_profile 纳米柱内返回 n_si² ----
eps_inside = fem.epsilon_profile(0.0, 0.0, (0.0, 0.0), (0.3e-6, 0.6e-6))
assert abs(eps_inside - fem.eps_si) < 1e-14, '[TC15] 纳米柱中心 eps 应为 eps_si FAILED'

# ---- TC16: MaxwellFEM2D epsilon_profile 外部返回 n_air² ----
eps_outside = fem.epsilon_profile(2.0e-6, 0.0, (0.0, 0.0), (0.3e-6, 0.6e-6))
assert abs(eps_outside - fem.eps_air) < 1e-14, '[TC16] 纳米柱外部 eps 应为 eps_air FAILED'

# ---- TC17: PhaseQuadrature 积分规则有正权重 ----
from phase_quadrature import quadrilateral_witherden_rule
n_q, xu, yu, w = quadrilateral_witherden_rule(15)
assert n_q <= 12, '[TC17] 超过 7 阶应降级到 7 阶 (12 点) FAILED'
assert np.all(w > 0), '[TC17] 积分权重应为正 FAILED'
assert np.all(np.isfinite(w)), '[TC17] 积分权重应全有限 FAILED'
assert abs(np.sum(w) - 1.0) < 0.1, '[TC17] 权重之和应约等于 1 FAILED'

# ---- TC18: PhaseQuadrature 相位延迟非负且振幅在 [0,1] ----
pq = PhaseQuadrature(wavelength=1.55e-6)
avg_ph, trans = pq.integrate_phase_delay(0.0, 0.0, 0.3e-6, 0.6e-6, 1.0e-6)
assert avg_ph >= 0.0, '[TC18] 相位延迟应为非负 FAILED'
assert 0.0 <= trans <= 1.0, '[TC18] 传输振幅应在 [0,1] 内 FAILED'

# ---- TC19: PhaseQuadrature 极化率标量非负 ----
alpha = pq.integrate_polarizability(0.0, 0.0, 0.3e-6, 0.6e-6)
assert alpha >= 0.0, '[TC19] 极化率应为非负 FAILED'

# ---- TC20: PhaseQuadrature 坐标映射正确 ----
x_map, y_map = pq.map_to_pillar(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.0, 0.0, 0.3e-6, 0.6e-6)
assert abs(x_map[0]) < 1e-15, '[TC20] 中心映射到 x=0 FAILED'
assert abs(y_map[0]) < 1e-15, '[TC20] 中心映射到 y=0 FAILED'

# ---- TC21: MultipoleExtractor Levi-Civita 符号 ----
eps_012 = MultipoleExtractor._levi_civita(0, 1, 2)
assert eps_012 == 1, '[TC21] ε_012 应为 1 FAILED'
eps_021 = MultipoleExtractor._levi_civita(0, 2, 1)
assert eps_021 == -1, '[TC21] ε_021 应为 -1 FAILED'
eps_001 = MultipoleExtractor._levi_civita(0, 0, 1)
assert eps_001 == 0, '[TC21] ε_001 应为 0 FAILED'

# ---- TC22: MultipoleExtractor 辐射功率非负 ----
me = MultipoleExtractor(wavelength=1.55e-6)
p_test = np.array([1e-18, 0.0, 0.0], dtype=np.complex128)
m_test = np.array([0.0, 0.0, 0.0], dtype=np.complex128)
powers = me.radiation_powers(p_test, m_test)
assert powers['P_dipole_electric'] > 0.0, '[TC22] 电偶极辐射功率应 > 0 FAILED'
assert powers['P_total'] > 0.0, '[TC22] 总辐射功率应 > 0 FAILED'

# ---- TC23: MetasurfaceCVT 密度函数非负 ----
grid = MetasurfaceCVT(region=(-5.0e-6, 5.0e-6, -5.0e-6, 5.0e-6))
rho_center = grid.density_function(np.array([0.0]), np.array([0.0]))
assert np.all(rho_center >= 1.0), '[TC23] 密度函数在原点应 >= 1 FAILED'

# ---- TC24: ProcessSampler 三角形采样重心接近几何重心 ----
np.random.seed(42)
ps = ProcessSampler(seed=42)
tri_pts = ps.uniform_in_triangle(2000, np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
centroid = np.mean(tri_pts, axis=0)
expected_ct = np.array([1.0 / 3.0, 1.0 / 3.0])
assert np.linalg.norm(centroid - expected_ct) < 0.05, '[TC24] 采样重心应接近 (1/3, 1/3) FAILED'

# ---- TC25: ProcessSampler 圆环采样在 [r1, r2] 内 ----
np.random.seed(42)
ann_pts = ProcessSampler.uniform_in_annulus(500, 1.0, 2.0, center=(0.0, 0.0))
r = np.sqrt(ann_pts[:, 0] ** 2 + ann_pts[:, 1] ** 2)
assert np.all(r >= 1.0) and np.all(r <= 2.0), '[TC25] 圆环点应在 [1,2] 内 FAILED'

# ---- TC26: ProcessSampler 高度误差场产生有限值 ----
np.random.seed(42)
x_g = np.linspace(-5e-6, 5e-6, 64)
y_g = np.linspace(-5e-6, 5e-6, 64)
h_field = ps.generate_height_error_field(x_g, y_g, sigma_h=5e-9, correlation_length=0.8e-6)
assert np.all(np.isfinite(h_field)), '[TC26] 高度误差场应全有限 FAILED'
assert not np.any(np.isnan(h_field)), '[TC26] 高度误差场不应含 NaN FAILED'

# ---- TC27: TopologyOptimizer 离散相位均匀分布 ----
opt = TopologyOptimizer(n_levels=8)
dp = opt.discrete_phases
assert len(dp) == 8, '[TC27] 应有 8 个离散相位 FAILED'
assert abs(dp[0] - 0.0) < 1e-14, '[TC27] 第一个相位应为 0 FAILED'
assert abs(dp[-1] - 2.0 * np.pi * 7 / 8) < 1e-14, '[TC27] 最后一个相位应为 2π*7/8 FAILED'

# ---- TC28: TopologyOptimizer DP 量化返回正确长度 ----
np.random.seed(42)
target_ph = np.linspace(0.0, 2.0 * np.pi, 20)
quantized, err = opt.quantize_phase_dp(target_ph, weights=np.ones(20))
assert len(quantized) == 20, '[TC28] 量化输出长度应为 20 FAILED'
assert err >= 0.0, '[TC28] 量化误差应为非负 FAILED'

# ---- TC29: TopologyOptimizer 纯相位 DP 量化误差有界 ----
target_ph2 = np.mod(np.linspace(0, 8 * np.pi, 50), 2.0 * np.pi)
quantized2, err2 = opt.quantize_phase_dp(target_ph2)
assert err2 < 100.0, '[TC29] DP 量化误差应有界 FAILED'

# ---- TC30: UncertaintyQuantify hermite_gauss_rule 返回奇函数点对称 ----
x_h, w_h = UncertaintyQuantify.hermite_gauss_rule(5)
assert len(x_h) == 5, '[TC30] 应有 5 个积分点 FAILED'
assert abs(np.sum(x_h * w_h)) < 1e-14, '[TC30] 对称积分点和为 0 FAILED'

# ---- TC31: UncertaintyQuantify level_to_order_open ----
uq = UncertaintyQuantify(dim_num=3, level_max=4)
assert uq.level_to_order_open(0) == 1, '[TC31] level 0 → order 1 FAILED'
assert uq.level_to_order_open(1) == 3, '[TC31] level 1 → order 3 FAILED'
assert uq.level_to_order_open(2) == 5, '[TC31] level 2 → order 5 FAILED'

# ---- TC32: WavefrontTracer 相位图构建不崩溃 ----
x_w = np.linspace(-2e-6, 2e-6, 21)
y_w = np.linspace(-2e-6, 2e-6, 21)
Xw, Yw = np.meshgrid(x_w, y_w, indexing='ij')
phase_simple = np.zeros((21, 21))
tracer = WavefrontTracer(x_w, y_w, phase_simple)
adj = tracer.build_graph()
assert len(adj) == 21 * 21, '[TC32] 图应有 441 个顶点 FAILED'

# ---- TC33: WavefrontTracer 传播方向计算 ----
dir_vec = tracer.phase_gradient_direction(0.0, 0.0)
assert np.isfinite(dir_vec[0]) and np.isfinite(dir_vec[1]), '[TC33] 方向向量应有限 FAILED'

# ---- TC34: PhaseSurface 悬链面相位非负 ----
ps_surf = PhaseSurface(x_w, y_w)
phi_cat_test = ps_surf.catenoid_phase_profile(a_param=2.0e6, center=(0.0, 0.0))
assert np.all(phi_cat_test >= 0.0), '[TC34] 悬链面相位应非负 FAILED'

# ---- TC35: PhaseSurface 螺旋面相位范围正确 ----
phi_hel_test = ps_surf.helicoid_phase_profile(a_param=1.0, center=(0.0, 0.0))
assert np.isfinite(phi_hel_test).all(), '[TC35] 螺旋面相位应全有限 FAILED'
assert np.max(phi_hel_test) <= np.pi + 1e-6, '[TC35] 螺旋面相位范围 ≤ π FAILED'
assert np.min(phi_hel_test) >= -np.pi - 1e-6, '[TC35] 螺旋面相位范围 ≥ -π FAILED'

# ---- TC36: PhaseSurface 表面能量非负 ----
W_test = ps_surf.surface_energy(phi_cat_test)
assert W_test >= 0.0, '[TC36] Willmore 能量应为非负 FAILED'

# ---- TC37: PhaseSurface 平均曲率计算不产生 NaN ----
H_cat = ps_surf.compute_mean_curvature(phi_cat_test)
assert not np.any(np.isnan(H_cat)), '[TC37] 平均曲率不应含 NaN FAILED'

# ---- TC38: PhaseSurface 高斯曲率计算不产生 NaN ----
K_cat = ps_surf.compute_gaussian_curvature(phi_cat_test)
assert not np.any(np.isnan(K_cat)), '[TC38] 高斯曲率不应含 NaN FAILED'

# ---- TC39: PhaseSurface minimal_surface_smooth 能量降低 ----
phi_disc_test = np.zeros((21, 21))
for i in range(21):
    for j in range(21):
        phi_disc_test[i, j] = np.floor((i + j) / 4.0) * (np.pi / 4)
phi_smooth_test = ps_surf.minimal_surface_smooth(phi_disc_test, lambda_fidelity=0.05, max_iter=30, dt=0.05)
W_before = ps_surf.surface_energy(phi_disc_test)
W_after = ps_surf.surface_energy(phi_smooth_test)
assert W_after <= W_before * 1.1, '[TC39] 平滑后能量不应大幅增加 FAILED'

# ---- TC40: laplacian_smooth 不修改边界 ----
phi_lap = phi_disc_test.copy()
phi_lap_s = ps_surf.laplacian_smooth(phi_lap, n_iter=10)
assert np.allclose(phi_lap_s[0, :], phi_disc_test[0, :]), '[TC40] 边界不应修改 FAILED'
assert np.allclose(phi_lap_s[-1, :], phi_disc_test[-1, :]), '[TC40] 边界不应修改 FAILED'
