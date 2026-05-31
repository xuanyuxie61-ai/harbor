# ---- TC01: Lattice site_index 与 index_to_site 互逆 ----
lat_tc = Lattice(4, 4, 4, 8)
x_test = np.array([1, 2, 3, 4])
idx = lat_tc.site_index(x_test)
x_back = lat_tc.index_to_site(idx)
assert np.array_equal(x_back, np.mod(x_test, lat_tc.dims)), '[TC01] Lattice site_index 与 index_to_site 互逆 FAILED'

# ---- TC02: su2_identity 迹为2且是单位阵 ----
from lattice_gauge import su2_identity, su2_trace
I2 = su2_identity()
assert abs(su2_trace(I2) - 2.0) < 1e-12, '[TC02] su2_identity 迹为2 FAILED'
assert np.allclose(I2, np.eye(2, dtype=complex)), '[TC02] su2_identity 是单位阵 FAILED'

# ---- TC03: su2_stereographic_project/inverse 互逆 ----
from lattice_gauge import su2_stereographic_project, su2_stereographic_inverse
q_test = np.array([0.5, -0.3, 0.8])
u_rec = su2_stereographic_inverse(q_test)
q_rec = su2_stereographic_project(u_rec)
assert np.allclose(q_rec, q_test, atol=1e-10), '[TC03] su2_stereographic_project/inverse 互逆 FAILED'

# ---- TC04: GaugeConfig 恒等场 average_plaquette 为 2.0 ----
lat_tc2 = Lattice(2, 2, 2, 4)
gauge_id = GaugeConfig(lat_tc2)
avg_plaq = gauge_id.average_plaquette()
assert abs(avg_plaq - 2.0) < 1e-12, '[TC04] GaugeConfig 恒等场 average_plaquette 为 2.0 FAILED'

# ---- TC05: GaugeConfig wilson_action 对恒等场解析验证 ----
beta_tc = 2.4
S_w = gauge_id.wilson_action(beta_tc)
expected_S = beta_tc * (1.0 - 1.0) * lat_tc2.vol * 6.0
assert abs(S_w - expected_S) < 1e-9, '[TC05] GaugeConfig wilson_action 解析验证 FAILED'

# ---- TC06: lagrange_interpolate 对线性函数精确恢复 ----
xpol = np.array([0.0, 1.0, 2.0, 3.0])
ypol = 2.0 * xpol + 1.0
val = lagrange_interpolate(xpol, ypol, 1.5)
assert abs(val - 4.0) < 1e-12, '[TC06] lagrange_interpolate 对线性函数精确恢复 FAILED'

# ---- TC07: correlator_effective_mass 对纯指数精确 ----
nt_tc = 8
m_true = 0.5
corr_exp = np.exp(-m_true * np.arange(nt_tc))
m_eff = correlator_effective_mass(corr_exp, dt=1)
assert np.allclose(m_eff[:nt_tc - 1], m_true, atol=1e-12), '[TC07] correlator_effective_mass 对纯指数精确 FAILED'

# ---- TC08: calccf 与 spline_eval 对三次多项式精确恢复 ----
xi = np.array([0.0, 1.0, 2.0, 3.0])
c_vals = np.zeros((2, 4))
c_vals[0, :] = xi ** 3
c_vals[1, :] = 3.0 * xi ** 2
breaks, coefs = calccf(xi, c_vals)
t_test = 1.5
spline_val = spline_eval(breaks, coefs, t_test)
assert abs(spline_val - t_test ** 3) < 1e-10, '[TC08] calccf 与 spline_eval 对三次多项式精确 FAILED'

# ---- TC09: gevp_solve 对已知本征值正确 ----
from variational_spectrum import gevp_solve
A = np.array([[4.0, 1.0], [1.0, 3.0]])
B = np.eye(2)
lam, vec = gevp_solve(A, B)
expected_lam = np.linalg.eigvals(np.linalg.solve(B, A)).real
expected_lam.sort()
lam_sorted = np.sort(lam)
assert np.allclose(lam_sorted, expected_lam, atol=1e-8), '[TC09] gevp_solve 对已知本征值 FAILED'

# ---- TC10: hooke_jeeves 最小化二次函数 ----
from variational_spectrum import hooke_jeeves
def quad_func(x):
    return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2
xbest, fbest = hooke_jeeves(quad_func, np.array([0.0, 0.0]), rho=0.5, eps=1e-6, itermax=500)
assert abs(fbest) < 1e-3, '[TC10] hooke_jeeves 最小化二次函数 FAILED'
assert np.linalg.norm(xbest - np.array([2.0, -1.0])) < 1e-2, '[TC10] hooke_jeeves 最优位置 FAILED'

# ---- TC11: lattice_coupling_from_beta 解析验证 ----
g0 = lattice_coupling_from_beta(6.0)
assert abs(g0 - 1.0) < 1e-12, '[TC11] lattice_coupling_from_beta 解析验证 FAILED'

# ---- TC12: beta_function_su3 对 g=0 返回零 ----
from reaction_rg import beta_function_su3
assert abs(beta_function_su3(0.0, nf=2)) < 1e-15, '[TC12] beta_function_su3 对 g=0 返回零 FAILED'

# ---- TC13: alpha_s_running 输出为正有限 ----
mu_tc = np.array([1.0, 2.0, 3.0])
alpha_vals = alpha_s_running(mu_tc, lambda_qcd=0.3, nf=2)
assert np.all(np.isfinite(alpha_vals)), '[TC13] alpha_s_running 输出为正有限 FAILED'
assert np.all(alpha_vals > 0), '[TC13] alpha_s_running 输出为正 FAILED'

# ---- TC14: rg_step_matrix 是上三角且对角为 -0.5 ----
from reaction_rg import rg_step_matrix
R = rg_step_matrix(5)
assert np.allclose(np.diag(R), -0.5 * np.ones(5)), '[TC14] rg_step_matrix 对角元 FAILED'
assert np.all(R[np.triu_indices(5, k=1)] == 0.0), '[TC14] rg_step_matrix 上三角 FAILED'

# ---- TC15: r8pp_fa 与 r8pp_sl 解线性方程组 ----
from matrix_algebra import r8pp_fa, r8pp_sl, dense_to_packed
A_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
A_pack = dense_to_packed(A_dense)
r_fac, info = r8pp_fa(2, A_pack)
assert info == 0, '[TC15] r8pp_fa 分解失败 FAILED'
b_vec = np.array([1.0, 2.0])
x_sol = r8pp_sl(2, r_fac, b_vec)
x_expected = np.linalg.solve(A_dense, b_vec)
assert np.allclose(x_sol, x_expected, atol=1e-10), '[TC15] r8pp_fa 与 r8pp_sl 解线性方程组 FAILED'

# ---- TC16: pion_decay_constant_from_dynamics GMOR 解析验证 ----
mpi_test = 139.0
fpi = pion_decay_constant_from_dynamics(mpi_test, B0=2700.0, mq=0.0035)
mq_mev = 0.0035 * 1e3
fpi_expected = np.sqrt((2.0 * mq_mev * 2700.0) / (mpi_test ** 2))
assert abs(fpi - fpi_expected) < 1e-6, '[TC16] pion_decay_constant_from_dynamics GMOR 解析验证 FAILED'

# ---- TC17: chiral_condensate_from_reaction 零π场返回正确值 ----
c_test = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
sigma = chiral_condensate_from_reaction(np.array([0.0, 1.0]), c_test, f0=0.092)
expected_sigma = 0.5 * (1.0 + 1.0) - 0.0 / (0.092 ** 2)
assert abs(sigma[0] - expected_sigma) < 1e-12, '[TC17] chiral_condensate_from_reaction 零π场 FAILED'

# ---- TC18: point_source 只在指定位置非零 ----
lat_src = Lattice(2, 2, 2, 4)
src = point_source(lat_src, np.array([1, 0, 1, 2]), spin=0)
assert src[(1, 0, 1, 2, 0)] == 1.0, '[TC18] point_source 指定位置为1 FAILED'
assert np.sum(np.abs(src)) == 1.0, '[TC18] point_source 其他位置为零 FAILED'

# ---- TC19: WilsonDiracOperator.apply 对零场返回零 ----
lat_wd = Lattice(2, 2, 2, 4)
gauge_wd = GaugeConfig(lat_wd)
np.random.seed(42)
gauge_wd.randomize()
wd_op = WilsonDiracOperator(lat_wd, gauge_wd, mass=0.1)
psi_zero = np.zeros((*lat_wd.shape, 2), dtype=complex)
out_zero = wd_op.apply(psi_zero)
assert np.allclose(out_zero, 0.0), '[TC19] WilsonDiracOperator.apply 对零场返回零 FAILED'

# ---- TC20: solve_chiral_oscillator 输出尺寸正确 ----
t_ch, y_ch = solve_chiral_oscillator(np.array([0.5, 0.0]), (0.0, 5.0), mu_chiral=1.5)
assert len(t_ch) == 500, '[TC20] solve_chiral_oscillator 时间网格长度 FAILED'
assert y_ch.shape == (2, 500), '[TC20] solve_chiral_oscillator 解轨迹形状 FAILED'

# ---- TC21: QuarkMesonReactionNetwork.rates 非负 ----
network = QuarkMesonReactionNetwork()
rates = network.rates(np.array([1.0, 1.0, 0.5, 0.2]))
assert np.all(rates >= 0), '[TC21] QuarkMesonReactionNetwork.rates 非负 FAILED'

# ---- TC22: packed_to_dense 与 dense_to_packed 互逆 ----
from matrix_algebra import packed_to_dense
M_dense = np.array([[2.0, 1.0], [1.0, 3.0]])
M_pack = dense_to_packed(M_dense)
M_dense2 = packed_to_dense(2, M_pack)
assert np.allclose(M_dense, M_dense2), '[TC22] packed_to_dense 与 dense_to_packed 互逆 FAILED'

# ---- TC23: su2_random 生成 SU(2) 矩阵行列式为1 ----
from lattice_gauge import su2_random
np.random.seed(42)
U_rand = su2_random()
det_U = np.linalg.det(U_rand)
assert abs(abs(det_U) - 1.0) < 1e-10, '[TC23] su2_random 行列式为1 FAILED'

# ---- TC24: decay_constant_integral 返回值非负有限 ----
fpi_val = decay_constant_integral(0.2, 0.2, lattice_spacing=1.0)
assert np.isfinite(fpi_val) and fpi_val >= 0, '[TC24] decay_constant_integral 返回值非负有限 FAILED'

# ---- TC25: self_energy_integral 返回值非负有限 ----
sigma_val = self_energy_integral(0.1, cutoff=np.pi)
assert np.isfinite(sigma_val) and sigma_val >= 0, '[TC25] self_energy_integral 返回值非负有限 FAILED'
