# ---- TC01: chebyshev_nodes 返回形状正确，边界值为 ±1 ----
import numpy as np
x_nodes = chebyshev_nodes(10)
assert len(x_nodes) == 11, '[TC01] 节点数应为 n+1 FAILED'
assert abs(x_nodes[0] - 1.0) < 1e-14, '[TC01] 首节点应为 1.0 FAILED'
assert abs(x_nodes[-1] - (-1.0)) < 1e-14, '[TC01] 末节点应为 -1.0 FAILED'

# ---- TC02: spectral_differentiation_matrix 尺寸正确 ----
D1_test = spectral_differentiation_matrix(8)
assert D1_test.shape == (9, 9), '[TC02] 谱微分矩阵形状应为 (n+1, n+1) FAILED'

# ---- TC03: 谱一阶微分精度 — D sin(pi*x) ≈ pi*cos(pi*x) ----
x_test = chebyshev_nodes(32)
D_test = spectral_differentiation_matrix(32)
u_test = np.sin(np.pi * x_test)
du_numerical = D_test @ u_test
du_exact = np.pi * np.cos(np.pi * x_test)
err_max = np.max(np.abs(du_numerical - du_exact))
assert err_max < 1e-4, '[TC03] 一阶谱微分最大误差应 < 1e-4 FAILED'

# ---- TC04: chebyshev_analyze + chebyshev_synthesize 往返 ----
from chebyshev_spectral import chebyshev_synthesize
x32 = chebyshev_nodes(32)
f_vals = np.sin(np.pi * x32)
coef = chebyshev_analyze(f_vals)
f_back = chebyshev_synthesize(coef)
err_roundtrip = np.max(np.abs(f_back - f_vals))
assert err_roundtrip < 1e-12, '[TC04] 分析-合成往返误差应 < 1e-12 FAILED'

# ---- TC05: clenshaw_evaluate — T_0(x) = 1 ----
from chebyshev_spectral import clenshaw_evaluate
coef_t0 = np.array([1.0])
val_t0 = clenshaw_evaluate(coef_t0, 0.5)
assert abs(val_t0 - 1.0) < 1e-14, '[TC05] T_0(0.5) 应为 1.0 FAILED'

# ---- TC06: clenshaw_evaluate — T_2(x) = 2x^2-1 ----
coef_t2 = np.array([0.0, 0.0, 1.0])
x_eval = np.array([0.0, 0.5, 1.0])
vals = clenshaw_evaluate(coef_t2, x_eval)
expected_t2 = 2.0 * x_eval**2 - 1.0
err_t2 = np.max(np.abs(vals - expected_t2))
assert err_t2 < 1e-14, '[TC06] T_2 评估误差应 < 1e-14 FAILED'

# ---- TC07: chebyshev_derivative_series — d/dx T_1(x) = 1·T_0(x) ----
coef_t1 = np.array([0.0, 1.0])   # coefficients for T_1 = x
dcoef = chebyshev_derivative_series(coef_t1)
assert len(dcoef) == 2, '[TC07] 导数系数长度应相同 FAILED'
assert abs(dcoef[0] - 1.0) < 1e-14, '[TC07] d/dx T_1 应为 T_0 FAILED'
assert abs(dcoef[1]) < 1e-14, '[TC07] 高阶系数应为 0 FAILED'

# ---- TC08: chebyshev_l2_norm — 非负 ----
norm_test = chebyshev_l2_norm(np.array([1.0, 0.5, 0.2]))
assert norm_test >= 0.0, '[TC08] L2 范数应 >= 0 FAILED'
assert np.isfinite(norm_test), '[TC08] L2 范数应为有限值 FAILED'

# ---- TC09: r83t_dif2 + thomas_solve — 求解已知系统 ----
r83t_test = r83t_dif2(10)
d_ones = np.ones(10, dtype=np.float64)
x_thomas = thomas_solve(r83t_test, d_ones)
res_thomas = r83t_mv(r83t_test, x_thomas) - d_ones
err_thomas = np.max(np.abs(res_thomas))
assert err_thomas < 1e-12, '[TC09] Thomas求解残差应 < 1e-12 FAILED'

# ---- TC10: jacobi_solve 对 DIF2 系统收敛 ----
r83t_jac = r83t_dif2(15)
d_jac = np.ones(15, dtype=np.float64)
x_exact = thomas_solve(r83t_jac, d_jac)
x_jac, info_jac = jacobi_solve(r83t_jac, d_jac, x0=x_exact*0.9, tol=1e-10, max_iter=20000)
err_jac = np.max(np.abs(x_jac - x_exact))
assert err_jac < 1e-5, '[TC10] Jacobi迭代误差应 < 1e-5 FAILED'

# ---- TC11: conjugate_gradient_solve 残差下降 ----
r83t_cg = r83t_dif2(12)
d_cg = np.ones(12, dtype=np.float64)
x_cg_test, info_cg_test = conjugate_gradient_solve(r83t_cg, d_cg, tol=1e-10)
res_cg = r83t_mv(r83t_cg, x_cg_test) - d_cg
err_cg = np.max(np.abs(res_cg))
assert err_cg < 1e-8, '[TC11] CG求解残差应 < 1e-8 FAILED'

# ---- TC12: discrete_energy_norm — 非负 ----
x_energy = chebyshev_nodes(16)
D2_test = spectral_differentiation_matrix(16)
D2_test = D2_test @ D2_test
u_energy = np.sin(np.pi * x_energy)
E_test = discrete_energy_norm(u_energy, D2_test, dx_weight=2.0/16)
assert E_test >= 0.0, '[TC12] 离散能量范数应 >= 0 FAILED'

# ---- TC13: enforce_dirichlet — 边界值正确设置 ----
u_bc_test = np.ones(20, dtype=np.float64)
u_bc = enforce_dirichlet(u_bc_test, 0.0, 0.0)
assert abs(u_bc[0]) < 1e-14, '[TC13] 右边界应为 0 FAILED'
assert abs(u_bc[-1]) < 1e-14, '[TC13] 左边界应为 0 FAILED'

# ---- TC14: smooth_initial_condition — 输出无 NaN/Inf，形状正确 ----
x_ic = chebyshev_nodes(20)
u_gauss = smooth_initial_condition(x_ic, case="gaussian")
u_sine = smooth_initial_condition(x_ic, case="sine")
u_poly = smooth_initial_condition(x_ic, case="poly")
assert not np.any(np.isnan(u_gauss)), '[TC14] Gaussian初始条件含NaN FAILED'
assert not np.any(np.isinf(u_gauss)), '[TC14] Gaussian初始条件含Inf FAILED'
assert np.all(u_gauss >= 0.0), '[TC14] Gaussian初始条件应 >= 0 FAILED'

# ---- TC15: relative_l2_error — 相同输入误差为 0 ----
u_identical = np.array([1.0, 2.0, 3.0])
err_zero = relative_l2_error(u_identical, u_identical)
assert err_zero < 1e-14, '[TC15] 相同向量相对L2误差应为 0 FAILED'

# ---- TC16: monte_carlo_statistic — 返回字典结构正确 ----
np.random.seed(42)
samples_mc = np.random.randn(100)
stats = monte_carlo_statistic(samples_mc)
assert 'mean' in stats, '[TC16] monte_carlo_statistic 缺失 mean FAILED'
assert 'std' in stats, '[TC16] monte_carlo_statistic 缺失 std FAILED'
assert 'variance' in stats, '[TC16] monte_carlo_statistic 缺失 variance FAILED'
assert isinstance(stats['mean'], float), '[TC16] mean 应为 float FAILED'

# ---- TC17: chirikov_orbit — 小K值下能量漂移有限 ----
np.random.seed(42)
orbit_test, energy_test = chirikov_orbit(n_steps=50, K=0.1)
drift_test = np.max(np.abs(energy_test - energy_test[0])) / (abs(energy_test[0]) + 1e-15)
assert drift_test < 0.5, '[TC17] 小K值Chirikov能量漂移应 < 0.5 FAILED'

# ---- TC18: fermat_is_prime — 已知素数与合数 ----
assert fermat_is_prime(2, k=5) == True, '[TC18] 2 应为素数 FAILED'
assert fermat_is_prime(3, k=5) == True, '[TC18] 3 应为素数 FAILED'
assert fermat_is_prime(4, k=5) == False, '[TC18] 4 应为合数 FAILED'
assert fermat_is_prime(1, k=5) == False, '[TC18] 1 不是素数 FAILED'
assert fermat_is_prime(17, k=5) == True, '[TC18] 17 应为素数 FAILED'

# ---- TC19: enumerate_grlex_indices — 形状正确，全为非负 ----
indices = enumerate_grlex_indices(d=2, p=2)
assert indices.shape[1] == 2, '[TC19] 多指标集维度应为 d FAILED'
assert np.all(indices >= 0), '[TC19] 多指标集所有元素应 >= 0 FAILED'
assert np.all(np.sum(indices, axis=1) <= 2), '[TC19] 总度数应 <= p FAILED'

# ---- TC20: chebyshev_interpolate — 在原节点处恢复原值 ----
x_orig = chebyshev_nodes(16)
f_orig = np.sin(2.0 * np.pi * x_orig)
coef_orig = chebyshev_analyze(f_orig)
f_interp = chebyshev_interpolate(coef_orig, x_orig)
err_interp = np.max(np.abs(f_interp - f_orig))
assert err_interp < 1e-12, '[TC20] 插值在原始节点处误差应 < 1e-12 FAILED'

# ---- TC21: cvt_1d_lloyd — 固定种子可复现 ----
np.random.seed(42)
cvt1 = cvt_1d_lloyd(n_generators=6, n_samples=2000, it_num=10, domain=(-1.0, 1.0), rho_func=None, seed=123)
np.random.seed(42)
cvt2 = cvt_1d_lloyd(n_generators=6, n_samples=2000, it_num=10, domain=(-1.0, 1.0), rho_func=None, seed=123)
diff_cvt = np.max(np.abs(cvt1 - cvt2))
assert diff_cvt < 1e-14, '[TC21] CVT固定种子应可复现 FAILED'

# ---- TC22: gpc_mean_variance — 已知系数统计量 ----
indices_gpc = enumerate_grlex_indices(d=1, p=2)
coefs_gpc = np.array([0.5, 0.3, 0.1])
mean_gpc, var_gpc = gpc_mean_variance(coefs_gpc, indices_gpc)
assert abs(mean_gpc - 0.5) < 1e-14, '[TC22] gPC均值应为首系数 FAILED'
assert abs(var_gpc - (0.3**2 + 0.1**2)) < 1e-14, '[TC22] gPC方差应为非零系数平方和 FAILED'

# ---- TC23: adaptive_truncation_error_analysis — 截断误差单调不增 ----
coef_trunc_test = np.array([1.0, 0.5, 0.2, 0.05, 0.01])
n_req, errs_trunc = adaptive_truncation_error_analysis(coef_trunc_test, threshold=1e-12)
assert n_req > 0, '[TC23] 所需模式数应 > 0 FAILED'
assert np.all(np.diff(errs_trunc) <= 1e-14), '[TC23] 截断误差应单调不增 FAILED'

# ---- TC24: check_solution_stability — NaN/Inf 检测 ----
assert check_solution_stability(np.array([1.0, 2.0, 3.0])) == True, '[TC24] 正常解应判断为稳定 FAILED'
assert check_solution_stability(np.array([1.0, np.nan, 3.0])) == False, '[TC24] NaN解应判断为不稳定 FAILED'
assert check_solution_stability(np.array([1.0, np.inf, 3.0])) == False, '[TC24] Inf解应判断为不稳定 FAILED'

# ---- TC25: gauss_seidel_solve 对 DIF2 系统收敛 ----
np.random.seed(42)
r83t_gs = r83t_dif2(10)
d_gs = np.ones(10, dtype=np.float64)
x_exact_gs = thomas_solve(r83t_gs, d_gs)
x_gs_test, info_gs = gauss_seidel_solve(r83t_gs, d_gs, x0=x_exact_gs*0.9, tol=1e-10, max_iter=20000)
err_gs = np.max(np.abs(x_gs_test - x_exact_gs))
assert err_gs < 1e-5, '[TC25] Gauss-Seidel误差应 < 1e-5 FAILED'
