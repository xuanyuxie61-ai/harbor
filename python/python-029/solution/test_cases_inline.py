# ---- TC01: OpticalPotentialParameters 基本属性验证 ----
params = OpticalPotentialParameters('n', 56, 26, 14.1)
assert params.target_A == 56 and params.target_Z == 26, '[TC01] OpticalPotentialParameters 基本属性验证 FAILED'
assert params.k > 0 and params.mu_MeV > 0, '[TC01] OpticalPotentialParameters 基本属性验证 FAILED'

# ---- TC02: 光学势返回复数数组且实部为负 ----
params = OpticalPotentialParameters('n', 56, 26, 14.1)
r = np.linspace(0.1, 15.0, 50)
U = build_optical_potential(r, params, l=0)
assert np.iscomplexobj(U), '[TC02] 光学势返回复数数组且实部为负 FAILED'
assert np.all(np.real(U) < 0), '[TC02] 光学势返回复数数组且实部为负 FAILED'

# ---- TC03: Riccati-Bessel 函数输出尺寸正确 ----
Rl, Rlp, Sl, Slp = riccati_bessel_functions(3.0, 5)
assert len(Rl) == 6 and len(Rlp) == 6 and len(Sl) == 6 and len(Slp) == 6, '[TC03] Riccati-Bessel 函数输出尺寸正确 FAILED'

# ---- TC04: Gamma(5.5) 解析值校验 ----
g = gamma_function(5.5)
assert abs(g.real - 52.342777) < 0.1, '[TC04] Gamma(5.5) 解析值校验 FAILED'

# ---- TC05: ln Gamma(10) 为有限实数 ----
lg = log_gamma_stirling(10.0)
assert np.isfinite(lg.real), '[TC05] ln Gamma(10) 为有限实数 FAILED'

# ---- TC06: GF(2) 加法满足交换律 ----
a = gf2_add(0b1011, 0b110)
b = gf2_add(0b110, 0b1011)
assert a == b, '[TC06] GF(2) 加法满足交换律 FAILED'

# ---- TC07: GF(2) 乘法满足结合律 ----
a = gf2_multiply(gf2_multiply(0b11, 0b101), 0b111)
b = gf2_multiply(0b11, gf2_multiply(0b101, 0b111))
assert a == b, '[TC07] GF(2) 乘法满足结合律 FAILED'

# ---- TC08: 宇称算符本征值正确性 ----
assert parity_operator_state(2) == 1, '[TC08] 宇称算符本征值正确性 FAILED'
assert parity_operator_state(3) == -1, '[TC08] 宇称算符本征值正确性 FAILED'

# ---- TC09: 同位旋态最低 T 不小于 |Tz| ----
states = isospin_states(30, 26)
assert all(s['T'] >= abs(s['Tz']) for s in states), '[TC09] 同位旋态最低 T 不小于 |Tz| FAILED'

# ---- TC10: 核组态数目为 C(5,2)=10 ----
configs = nuclear_configuration_gf2(2, 5)
assert len(configs) == 10, '[TC10] 核组态数目为 C(5,2)=10 FAILED'

# ---- TC11: 56Fe 结合能为正 ----
nuc = Nuclide(26, 56)
assert nuc.binding_energy() > 0, '[TC11] 56Fe 结合能为正 FAILED'

# ---- TC12: 56Fe 中子分离能为正 ----
nuc = Nuclide(26, 56)
assert nuc.neutron_separation_energy() > 0, '[TC12] 56Fe 中子分离能为正 FAILED'

# ---- TC13: L² 内积满足共轭对称性 ----
r = np.linspace(0.01, 5.0, 100)
f = np.sin(r) + 1j * np.cos(r)
g = np.exp(-r)
inner_fg = l2_inner_product(f, g, r)
inner_gf = l2_inner_product(g, f, r)
assert abs(inner_fg - np.conj(inner_gf)) < 1e-10, '[TC13] L² 内积满足共轭对称性 FAILED'

# ---- TC14: Gram-Schmidt 正交化后单位范数 ----
r = np.linspace(0.01, 5.0, 100)
u1 = np.sin(r) * np.exp(-r)
u2 = np.cos(r) * np.exp(-r)
ortho = gram_schmidt_orthogonalization([u1, u2], r)
assert abs(l2_norm(ortho[0], r) - 1.0) < 1e-6, '[TC14] Gram-Schmidt 正交化后单位范数 FAILED'

# ---- TC15: 穿透系数范围约束在 [0,1] ----
S_dict = {(0, 0.5): 0.9 * np.exp(0.2j), (1, 0.5): 0.8 * np.exp(0.1j), (1, 1.5): 0.85 * np.exp(0.15j)}
T = transmission_coefficients(S_dict, 1)
assert all(0.0 <= v <= 1.0 for v in T.values()), '[TC15] 穿透系数范围约束在 [0,1] FAILED'

# ---- TC16: 光学定理关系 2*σ_tot = σ_el + σ_react ----
xs = compute_cross_sections(S_dict, 0.5, 1)
assert abs(2.0 * xs['sigma_total'] - xs['sigma_elastic'] - xs['sigma_reaction']) < 1e-10, '[TC16] 光学定理关系 2*σ_tot = σ_el + σ_react FAILED'

# ---- TC17: 微分截面非负性 ----
theta = np.linspace(0.1, np.pi - 0.1, 50)
dsigma = differential_cross_section(theta, S_dict, 0.5, 1)
assert np.all(dsigma >= 0), '[TC17] 微分截面非负性 FAILED'

# ---- TC18: SVD 奇异值非负且降序 ----
svd_res = svd_analysis_smatrix(S_dict, 1)
s = svd_res['singular_values']
assert np.all(s >= 0), '[TC18] SVD 奇异值非负且降序 FAILED'
assert np.all(np.diff(s) <= 1e-12), '[TC18] SVD 奇异值非负且降序 FAILED'

# ---- TC19: Logistic 映射 r=2.0 不动点解析验证 ----
attr = logistic_attractor(2.0, warm_up=1000, max_iter=2000)
assert len(attr) == 1 and abs(attr[0] - 0.5) < 1e-4, '[TC19] Logistic 映射 r=2.0 不动点解析验证 FAILED'

# ---- TC20: Lyapunov 指数稳定区域为负 ----
lam = lyapunov_exponent_logistic(2.5)
assert lam < 0, '[TC20] Lyapunov 指数稳定区域为负 FAILED'

# ---- TC21: 能级密度为正 ----
rho = level_density(56, 10.0, 2.0)
assert rho > 0, '[TC21] 能级密度为正 FAILED'

# ---- TC22: Newton-Cotes Open 权重和等于区间长度 ----
x, w = open_newton_cotes_weights(5, 0.0, 10.0)
assert abs(np.sum(w) - 10.0) < 1e-10, '[TC22] Newton-Cotes Open 权重和等于区间长度 FAILED'

# ---- TC23: BDF2 衰变链布居数非负 ----
M = np.array([[-0.5, 0.0], [0.5, -0.1]])
N0 = np.array([1.0, 0.0])
t, N = decay_chain_bdf2(N0, M, (0.0, 10.0), 50)
assert np.all(N >= -1e-12), '[TC23] BDF2 衰变链布居数非负 FAILED'

# ---- TC24: 集体质量参数为正 ----
B = collective_mass_parameter(56, 2)
assert B > 0, '[TC24] 集体质量参数为正 FAILED'

# ---- TC25: 巨共振截面峰值靠近共振能量 ----
E_R = resonance_energy(56, 2)
E_range = np.linspace(E_R - 5, E_R + 5, 200)
sigma = giant_resonance_cross_section(E_range, 56, 2)
idx_peak = np.argmax(sigma)
assert abs(E_range[idx_peak] - E_R) < 1.5, '[TC25] 巨共振截面峰值靠近共振能量 FAILED'

# ---- TC26: Lebedev 权重归一化和为 1 ----
x_l, y_l, z_l, w_l = lebedev_rule(26)
assert abs(np.sum(w_l) - 1.0) < 1e-12, '[TC26] Lebedev 权重归一化和为 1 FAILED'

# ---- TC27: 球坐标与笛卡尔坐标互逆转换 ----
from angular_quadrature import spherical_to_cartesian
theta = np.array([0.5, 1.0, 1.5])
phi = np.array([0.0, 1.0, 2.0])
x, y, z = spherical_to_cartesian(theta, phi)
theta2, phi2 = cartesian_to_spherical(x, y, z)
assert np.allclose(theta, theta2) and np.allclose(phi, phi2), '[TC27] 球坐标与笛卡尔坐标互逆转换 FAILED'

# ---- TC28: 半整数自旋 Kramers 简并度为 2 ----
tr = time_reversal_symmetry_check(2.5, 0b0011)
assert tr['kramers_degeneracy'] == 2, '[TC28] 半整数自旋 Kramers 简并度为 2 FAILED'

# ---- TC29: 阻尼宽度随激发能单调增加 ----
G1 = damping_width(56, 2, 5.0)
G2 = damping_width(56, 2, 15.0)
assert G2 > G1, '[TC29] 阻尼宽度随激发能单调增加 FAILED'

# ---- TC30: Coulomb 相移 eta=0 时为零 ----
sigma_0 = coulomb_phase_shift(2, 0.0)
assert abs(sigma_0) < 1e-12, '[TC30] Coulomb 相移 eta=0 时为零 FAILED'

# ---- TC31: 复合核形成截面非负 ----
params31 = OpticalPotentialParameters('n', 56, 26, 14.1)
S_dict31 = {(0, 0.5): 0.9 * np.exp(0.2j), (1, 0.5): 0.8 * np.exp(0.1j), (1, 1.5): 0.85 * np.exp(0.15j)}
T31 = transmission_coefficients(S_dict31, 1)
sigma_cf = compound_formation_cross_section(params31, T31, 1)
assert sigma_cf >= 0, '[TC31] 复合核形成截面非负 FAILED'

# ---- TC32: 球形网格顶点数计算正确 ----
mesh = SphericalShellMesh(R_max=10.0, n_r=20, n_theta=10, n_phi=20)
assert mesh.n_vertices == 20 * 10 * 20, '[TC32] 球形网格顶点数计算正确 FAILED'

# ---- TC33: 弹性散射 Q 值近似为 0 ----
Q_val = compute_q_value_reaction(26, 56, 0, 1, 0, 1)
assert abs(Q_val) < 1.0, '[TC33] 弹性散射 Q 值近似为 0 FAILED'

# ---- TC34: 宽度涨落修正因子不大于 1 ----
W = width_fluctuation_correction(T31, 1, nu=1.0)
assert W <= 1.0 and W > 0, '[TC34] 宽度涨落修正因子不大于 1 FAILED'

# ---- TC35: 球形 Bessel j0(0) 解析值为 1 ----
j0 = spherical_bessel_jn_highprecision(1e-12, 0)
assert abs(j0 - 1.0) < 1e-6, '[TC35] 球形 Bessel j0(0) 解析值为 1 FAILED'
