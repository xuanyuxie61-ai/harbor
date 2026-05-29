from aerosol_microphysics import lognormal_size_distribution
from quadrature_engine import chebyshev1_abscissas_weights

# ---- TC01: 对数正态分布返回值非负 ----
r_test = np.array([0.01, 0.1, 1.0])
n_r = lognormal_size_distribution(r_test, 1000.0, 0.15, 1.8)
assert np.all(n_r >= 0), '[TC01] 对数正态分布返回值非负 FAILED'

# ---- TC02: 多模态对数正态分布输出形状与输入一致 ----
r_grid = np.logspace(-3, 1, 50)
modes = [(500.0, 0.15, 1.8), (100.0, 1.5, 2.2)]
n_total = multimode_lognormal(r_grid, modes)
assert n_total.shape == r_grid.shape, '[TC02] 多模态对数正态分布输出形状与输入一致 FAILED'

# ---- TC03: 混合态分类网格总数等于 m*n ----
m, n, C = 12, 12, 5
counts = count_mixing_state(m, n, C)
assert np.sum(counts) == m * n, '[TC03] 混合态分类网格总数等于 m*n FAILED'

# ---- TC04: 混合态指数在 [0,1] 范围内 ----
chi = mixing_state_index(counts)
assert 0.0 <= chi <= 1.0, '[TC04] 混合态指数在 [0,1] 范围内 FAILED'

# ---- TC05: Bruggeman 有效介质结果有限 ----
fracs = np.array([0.4, 0.3, 0.2, 0.1])
m_comp = np.array([1.53+0.0j, 1.75+0.44j, 1.45+0.003j, 1.53+0.008j], dtype=np.complex128)
m_eff = bruggeman_effective_medium(fracs, m_comp)
assert np.isfinite(m_eff) and np.imag(m_eff) >= 0, '[TC05] Bruggeman 有效介质结果有限 FAILED'

# ---- TC06: 消光效率小参数近似为有限正数 ----
q_ext = extinction_efficiency_small(0.05, 0.55, m_eff)
assert np.isfinite(q_ext) and q_ext >= 0, '[TC06] 消光效率小参数近似为有限正数 FAILED'

# ---- TC07: HG 相函数勒让德系数 g=0 时仅 a0=1 ----
coeffs_0 = legendre_coefficients_hg(0.0, max_l=5)
assert coeffs_0[0] == 1.0 and np.allclose(coeffs_0[1:], 0.0), '[TC07] HG 相函数勒让德系数 g=0 时仅 a0=1 FAILED'

# ---- TC08: HG 相函数数值归一化积分等于 1 ----
mu_test = np.linspace(-1.0, 1.0, 1000)
p_hg = phase_function_hg(mu_test, 0.65)
norm_hg = 0.5 * np.trapezoid(p_hg, mu_test)
assert abs(norm_hg - 1.0) < 0.01, '[TC08] HG 相函数数值归一化积分等于 1 FAILED'

# ---- TC09: 勒让德展开各向同性散射解析验证 ----
coeffs_iso = legendre_coefficients_hg(0.0, max_l=10)
p_iso = expand_phase_function_legendre(mu_test, coeffs_iso)
assert np.allclose(p_iso, 1.0/(4.0*pi), atol=1e-10), '[TC09] 勒让德展开各向同性散射解析验证 FAILED'

# ---- TC10: Mie 散射截面返回三个有限值 ----
c_ext, c_sca, g_val = mie_scattering_cross_section(0.1, 0.55, m_eff)
assert np.isfinite(c_ext) and np.isfinite(c_sca) and np.isfinite(g_val), '[TC10] Mie 散射截面返回三个有限值 FAILED'

# ---- TC11: RTE 矩阵维度正确 ----
A, b, mu, w = build_rte_matrix(4, 4, 1.0, 0.85, 0.5)
assert A.shape == (16, 16) and b.shape[0] == 16 and mu.shape[0] == 4 and w.shape[0] == 4, '[TC11] RTE 矩阵维度正确 FAILED'

# ---- TC12: SOR 求解收敛后残差小于容差 ----
x0 = np.ones_like(b) * 0.5
x_sol, iters, residual = sor_solve(A, b, x0, omega_sor=1.3, tol=1e-8, max_iter=5000)
assert residual < 1e-5, '[TC12] SOR 求解收敛后残差小于容差 FAILED'

# ---- TC13: 加热率计算为有限值 ----
hr = compute_heating_rate(10.0, 5.0, 0.1, 1.0e3)
assert np.isfinite(hr), '[TC13] 加热率计算为有限值 FAILED'

# ---- TC14: 蒙特卡洛光子传输守恒 ----
np.random.seed(42)
escaped, absorbed_surf, absorbed_atm, paths = photon_random_walk_3d(
    num_photons=500, max_steps=50, extinction_coeff=0.5, layer_height=10.0,
    g_asymmetry=0.65, albedo=0.92, surface_albedo=0.15)
assert escaped + absorbed_surf + absorbed_atm == 500, '[TC14] 蒙特卡洛光子传输守恒 FAILED'

# ---- TC15: 蒙特卡洛估算光学厚度非负 ----
tau_mc = estimate_optical_depth_monte_carlo(paths, 10.0)
assert tau_mc >= 0.0, '[TC15] 蒙特卡洛估算光学厚度非负 FAILED'

# ---- TC16: 全球网格生成节点数正确 ----
nodes = generate_lat_lon_grid(6, 8)
assert nodes.shape == (48, 2), '[TC16] 全球网格生成节点数正确 FAILED'

# ---- TC17: 球面距离矩阵对称且对角线为 0 ----
dist = compute_distance_table(nodes)
assert np.allclose(dist, dist.T) and np.allclose(np.diag(dist), 0.0), '[TC17] 球面距离矩阵对称且对角线为 0 FAILED'

# ---- TC18: 大气分层边界单调递增 ----
bounds, mids = define_atmospheric_layers(0.0, 20.0, 10)
assert np.all(np.diff(bounds) > 0) and len(mids) == 10, '[TC18] 大气分层边界单调递增 FAILED'

# ---- TC19: 源区反演返回三维位置 ----
stations = np.array([[0.0,0.0,0.0],[100.0,0.0,0.0],[0.0,100.0,0.0],[100.0,100.0,0.0],[50.0,50.0,0.0]])
conc = np.array([1.0, 0.5, 0.5, 0.25, 0.7])
source_pos, res_norm = inverse_source_location(stations, conc, Q=1.0, D_diff=10.0, L=200.0)
assert source_pos.shape == (3,) and np.isfinite(res_norm), '[TC19] 源区反演返回三维位置 FAILED'

# ---- TC20: Köhler 临界过饱和度在合理范围内 ----
s_crit = kohler_critical_supersaturation(
    298.0, 0.072, 0.018, 1000.0, 0.132, 1760.0, 3.0, 0.05e-6,
    (4.0/3.0)*pi*(0.05e-6)**3*1760.0)
assert 0.0 < s_crit < 0.5, '[TC20] Köhler 临界过饱和度在合理范围内 FAILED'

# ---- TC21: Logistic 活化分数始终在 [0,1] ----
t_arr = np.linspace(0, 300, 50)
f_act = activated_fraction_logistic(t_arr, 0.3/100.0, s_crit, 1.8)
assert np.all(f_act >= 0.0) and np.all(f_act <= 1.0), '[TC21] Logistic 活化分数始终在 [0,1] FAILED'

# ---- TC22: CCN 数浓度非负 ----
ccn = compute_ccn_number_concentration(0.5, 1000.0, 0.08, 1.8)
assert ccn >= 0.0, '[TC22] CCN 数浓度非负 FAILED'

# ---- TC23: AOD 协方差矩阵对称正定 ----
stations_aod = np.array([[39.9,116.4],[40.7,-74.0],[-33.9,18.4]])
Sigma_aod = aod_covariance_model(stations_aod, 800.0, 0.20)
assert np.allclose(Sigma_aod, Sigma_aod.T) and np.all(np.linalg.eigvalsh(Sigma_aod) > 0), '[TC23] AOD 协方差矩阵对称正定 FAILED'

# ---- TC24: EOF 方差解释率在 [0,1] 内且降序 ----
eigvals, eigvecs, evr = eof_analysis(Sigma_aod, num_modes=2)
assert np.all(evr >= 0.0) and np.all(evr <= 1.0) and np.all(np.diff(evr) <= 0.0), '[TC24] EOF 方差解释率在 [0,1] 内且降序 FAILED'

# ---- TC25: 四面体积分常函数 1 返回归一化值 1 ----
vol = integrate_tetrahedron(lambda pts: np.ones(pts.shape[0]))
assert abs(vol - 1.0) < 0.01, '[TC25] 四面体积分常函数 1 返回归一化值 1 FAILED'

# ---- TC26: 正方形积分常函数 1 返回面积 4 ----
area_sq = integrate_square(lambda pts: np.ones(pts.shape[0]), deg=6)
assert abs(area_sq - 4.0) < 0.01, '[TC26] 正方形积分常函数 1 返回面积 4 FAILED'

# ---- TC27: 二分法求根 Wallis 方程精度 ----
root, it_b = bisection(lambda x: x**3 - 2*x - 5, 2.0, 3.0, tol=1e-10)
assert abs(root - 2.0945514815) < 1e-6 and it_b > 0, '[TC27] 二分法求根 Wallis 方程精度 FAILED'

# ---- TC28: 二项式系数 C(10,3) 等于 120 ----
assert abs(binomial_coefficient(10, 3) - 120.0) < 1e-10, '[TC28] 二项式系数 C(10,3) 等于 120 FAILED'

# ---- TC29: 组合字典序选取返回 p 个元素 ----
comb = comb_lexicographic(20, 5, 1)
assert len(comb) == 5 and all(1 <= c <= 20 for c in comb), '[TC29] 组合字典序选取返回 p 个元素 FAILED'

# ---- TC30: Wilson-Hilferty 返回值有限正数 ----
wh = wilson_hilferty_chi_square(5, 0.5)
assert np.isfinite(wh) and wh > 0, '[TC30] Wilson-Hilferty 返回值有限正数 FAILED'

# ---- TC31: safe_acos 边界外输入不报错 ----
assert np.isfinite(safe_acos(1.0001)) and np.isfinite(safe_acos(-1.0001)), '[TC31] safe_acos 边界外输入不报错 FAILED'

# ---- TC32: Wishart 样本协方差矩阵对称正定 ----
np.random.seed(42)
Sigma_test = np.array([[1.0,0.5],[0.5,1.0]])
S_wish = wishart_variate(Sigma_test, n=2)
assert np.allclose(S_wish, S_wish.T) and np.all(np.linalg.eigvalsh(S_wish) > 0), '[TC32] Wishart 样本协方差矩阵对称正定 FAILED'

# ---- TC33: 样本协方差矩阵对角线非负 ----
np.random.seed(42)
data_test = np.random.randn(10, 3)
S_sample = sample_covariance_matrix(data_test)
assert np.all(np.diag(S_sample) >= 0.0), '[TC33] 样本协方差矩阵对角线非负 FAILED'

# ---- TC34: 切比雪夫节点和权重长度匹配 ----
x_c, w_c = chebyshev1_abscissas_weights(16, -1.0, 1.0)
assert len(x_c) == 16 and len(w_c) == 16, '[TC34] 切比雪夫节点和权重长度匹配 FAILED'

# ---- TC35: 高斯-勒让德相函数积分不对称因子解析验证 ----
g_asym_test = 0.65
g_num = scattering_asymmetry_parameter(g_asym_test, num_points=400)
assert abs(g_num - g_asym_test) < 0.05, '[TC35] 高斯-勒让德相函数积分不对称因子解析验证 FAILED'
