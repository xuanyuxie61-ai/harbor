# ---- TC01: Carlson RF(1,2,0) 返回有限正值 ----
rf_val = carlson_rf(1.0, 2.0, 0.0)
assert rf_val > 0 and np.isfinite(rf_val), '[TC01] Carlson RF should be positive finite FAILED'

# ---- TC02: Jacobi sn/cn/dn 恒等式 sn²+cn²=1 ----
sn, cn, dn = jacobi_sncndn(0.5, 0.5)
assert abs(sn ** 2 + cn ** 2 - 1.0) < 1e-12, '[TC02] Jacobi sn^2+cn^2=1 FAILED'

# ---- TC03: Gauss AGM 介于 a 和 b 之间 ----
a, b = 1.0, np.sqrt(2.0)
agm_val, iters = gauss_agm(a, b)
assert min(a, b) - 1e-15 <= agm_val <= max(a, b) + 1e-15, '[TC03] AGM should be between a and b FAILED'

# ---- TC04: drug_protein_binding_fraction 输出在 (0,1] 范围内 ----
fu = drug_protein_binding_fraction(1e5, 1e-6, n_sites=1)
assert 0.0 < fu <= 1.0, '[TC04] Free fraction should be in (0,1] FAILED'

# ---- TC05: effective_diffusion_coefficient 返回正值 ----
D_eff = effective_diffusion_coefficient(1e-9, 1e-10, np.pi / 4.0)
assert D_eff > 0 and np.isfinite(D_eff), '[TC05] Effective diffusion should be positive finite FAILED'

# ---- TC06: r8vec_uniform_01_sorted_exponential 输出有序且在 [0,1] ----
u_sorted = r8vec_uniform_01_sorted_exponential(30)
assert np.all(np.diff(u_sorted) >= 0), '[TC06] Sorted exponential must be monotonic non-decreasing FAILED'
assert u_sorted[0] >= 0 and u_sorted[-1] <= 1.0, '[TC06] Values must be in [0,1] FAILED'

# ---- TC07: sample_physiological_params 固定种子可复现 ----
np.random.seed(42)
p1 = sample_physiological_params(20, np.array([70.0, 1.2, 0.05]), np.array([0.15, 0.20, 0.30]))
np.random.seed(42)
p2 = sample_physiological_params(20, np.array([70.0, 1.2, 0.05]), np.array([0.15, 0.20, 0.30]))
assert np.allclose(p1, p2), '[TC07] Reproducibility with fixed seed FAILED'

# ---- TC08: CC 求积权重水平 3 之和为 2.0 ----
w = cc_weights(3)
assert abs(w.sum() - 2.0) < 1e-12, '[TC08] CC weights level 3 sum should be 2.0 FAILED'

# ---- TC09: square_monomial_integral 精确值 ----
smi = square_monomial_integral(0.0, 1.0, 0.0, 1.0, 2, 0)
assert abs(smi - 1.0 / 3.0) < 1e-12, '[TC09] Integral of x^2 over [0,1] should be 1/3 FAILED'

# ---- TC10: sparse_grid_integrate ∫ x² dx over [-1,1] = 2/3 ----
sg_val = sparse_grid_integrate(lambda p: p[0] ** 2, 1, 3)
assert abs(sg_val - 2.0 / 3.0) < 1e-10, '[TC10] Sparse grid 1D x^2 integral should be 2/3 FAILED'

# ---- TC11: sparse_grid_integrate ∫ 1 dV over [-1,1]^3 = 8.0 ----
sg_vol = sparse_grid_integrate(lambda p: 1.0, 3, 2)
assert abs(sg_vol - 8.0) < 1e-10, '[TC11] Sparse grid 3D volume should be 8 FAILED'

# ---- TC12: DD Laplacian 特征值全部为正 ----
lam_dd = laplacian_dd_eigenvalues(15, 1.0)
assert np.all(lam_dd > 0), '[TC12] DD Laplacian eigenvalues must all be positive FAILED'

# ---- TC13: NN Laplacian 行列式近似为零（奇异） ----
A_nn = laplacian_1d_nn(10, 1.0)
assert abs(np.linalg.det(A_nn)) < 1e-6, '[TC13] NN Laplacian determinant should be ~0 (singular) FAILED'

# ---- TC14: laplacian_apply 对二次函数精确成立 ----
n14 = 10
L14 = 1.0
x14 = np.linspace(0.0, L14, n14 + 2)[1:-1]
u14 = x14 * (1.0 - x14)
Au14 = laplacian_apply(u14, bc_type="DD", L=L14)
assert np.allclose(Au14, 2.0 * np.ones_like(u14), atol=1e-12), '[TC14] Laplacian of x(1-x) should be 2 exact FAILED'

# ---- TC15: Feynman-Kac 1D 蒙特卡洛结果有限且在边界值之间 ----
np.random.seed(42)
mc_fk = feynman_kac_1d(0.5, 1.0, 2.0, 0.0, 1.0, h=0.0005, n_trajectories=1000)
assert np.isfinite(mc_fk), '[TC15] Feynman-Kac 1D result must be finite FAILED'
assert 0.0 <= mc_fk <= 1.0, '[TC15] Feynman-Kac 1D result should be in [0,1] FAILED'

# ---- TC16: tough_exact 在 t=0 应返回 [1,1,1,1] ----
y_exact_0 = tough_exact(0.0)
assert np.allclose(y_exact_0, [1.0, 1.0, 1.0, 1.0], atol=1e-12), '[TC16] tough_exact(0) should be [1,1,1,1] FAILED'

# ---- TC17: rk4_step 对 y'=y 精确近似 exp(0.1) ----
y_rk4 = rk4_step(lambda t, y: y, 0.0, np.array([1.0]), 0.1)
assert abs(y_rk4[0] - np.exp(0.1)) < 1e-6, '[TC17] RK4 step for exp growth FAILED'

# ---- TC18: implicit_trapezoidal_step 对 y'=y 稳定 ----
y_trap = implicit_trapezoidal_step(lambda t, y: y, 0.0, np.array([1.0]), 0.1)
assert abs(y_trap[0] - 1.105) < 0.01, '[TC18] Implicit trapezoidal step result out of range FAILED'

# ---- TC19: PBPK_ODE_System rhs 输出长度为 7 ----
system = PBPK_ODE_System()
r19 = system.rhs(0.0, np.zeros(7))
assert len(r19) == 7, '[TC19] PBPK rhs must have 7 compartments FAILED'
assert np.all(np.isfinite(r19)), '[TC19] PBPK rhs must be finite FAILED'

# ---- TC20: Rosenbrock 全局最小值 f(1,1)=0 ----
assert rosenbrock(np.array([1.0, 1.0])) < 1e-12, '[TC20] Rosenbrock(1,1) should be ~0 FAILED'

# ---- TC21: Himmelblau 函数在 (0,0) 处值为 170 ----
h_val = himmelblau(np.array([0.0, 0.0]))
assert abs(h_val - 170.0) < 1e-10, '[TC21] Himmelblau(0,0) should be 170 FAILED'

# ---- TC22: enzyme_kinetics_polynomial_substrate 无抑制剂精确值 ----
v_enz = enzyme_kinetics_polynomial_substrate(5.0, 10.0, 5.0, Ki=1e10, I=0.0)
assert abs(v_enz - 5.0) < 1e-10, '[TC22] MM kinetics S=Km should give Vmax/2 FAILED'

# ---- TC23: chebyshev_nodes 输出正确数量和范围 ----
ch_nodes = chebyshev_nodes(12, -1.0, 1.0)
assert len(ch_nodes) == 12, '[TC23] Chebyshev nodes count FAILED'
assert np.all(ch_nodes >= -1.0) and np.all(ch_nodes <= 1.0), '[TC23] Chebyshev nodes range FAILED'

# ---- TC24: Lagrange 插值在节点处精确 ----
n24 = 8
cnodes24 = chebyshev_nodes(n24, 0.0, np.pi)
cvals24 = np.sin(cnodes24)
y_back24 = lagrange_interpolate(cnodes24, cvals24, cnodes24)
assert np.allclose(y_back24, cvals24, atol=1e-12), '[TC24] Lagrange interpolation must be exact at nodes FAILED'

# ---- TC25: piecewise_linear_interpolate 对单调递增数据保持单调性 ----
x_pl = np.linspace(0.0, 10.0, 10)
y_pl = x_pl ** 2
x_eval = np.linspace(0.0, 10.0, 50)
y_eval = piecewise_linear_interpolate(x_pl, y_pl, x_eval)
assert np.all(np.diff(y_eval) >= -1e-15), '[TC25] Piecewise linear interp must preserve monotonicity FAILED'

# ---- TC26: knapsack_01 已知最优解 ----
w26 = np.array([2.0, 3.0, 4.0, 5.0])
v26 = np.array([3.0, 4.0, 5.0, 6.0])
mx26, sel26 = knapsack_01(w26, v26, 5.0)
assert abs(mx26 - 7.0) < 1e-10, '[TC26] Knapsack max value FAILED'
assert sorted(sel26) == [0, 1], '[TC26] Knapsack selected items FAILED'

# ---- TC27: Luhn checksum 计算正确 ----
cs27 = luhn_checksum([1, 2, 3, 4, 5])
assert cs27 == 9, '[TC27] Luhn checksum of [1,2,3,4,5] should be 9 FAILED'

# ---- TC28: luhn_is_valid 对有效序列返回 True ----
assert luhn_is_valid([1, 2, 3, 4, 5, 9]), '[TC28] Luhn valid sequence FAILED'

# ---- TC29: safe_divide 除零返回默认值 ----
sd_val = safe_divide(5.0, 0.0, 999.0)
assert sd_val == 999.0, '[TC29] safe_divide(5,0) should return default 999 FAILED'

# ---- TC30: concentration_binning 输出长度正确 ----
fake_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
counts30, edges30, centers30 = concentration_binning(fake_c, n_bins=4)
assert len(counts30) == 4, '[TC30] Bin counts length FAILED'
assert len(edges30) == 5, '[TC30] Bin edges length FAILED'
assert len(centers30) == 4, '[TC30] Bin centers length FAILED'

# ---- TC31: softplus 满足 softplus(0) = log(2) ----
assert abs(softplus(0.0) - np.log(2.0)) < 1e-12, '[TC31] softplus(0) should be log(2) FAILED'

# ---- TC32: clip_concentration 将负值钳制到下界 ----
clipped = clip_concentration(-1e-20, C_min=1e-12)
assert clipped >= 1e-12, '[TC32] clip_concentration must clamp to C_min FAILED'

# ---- TC33: r8vec_normal_01_sorted 输出有序 ----
z33 = r8vec_normal_01_sorted(25, method="exponential")
assert np.all(np.diff(z33) >= 0), '[TC33] Normal sorted must be monotonic non-decreasing FAILED'

# ---- TC34: ellipsoid_tri_surface 输出正确维度 ----
v34, t34 = ellipsoid_tri_surface(1.0, 2.0, 3.0, 10, 10)
assert v34.ndim == 2 and v34.shape[1] == 3, '[TC34] Ellipsoid vertices shape FAILED'
assert t34.ndim == 2 and t34.shape[1] == 3, '[TC34] Ellipsoid triangles shape FAILED'

# ---- TC35: inside_ellipsoid 原点在单位球内 ----
import numpy as np
assert inside_ellipsoid(np.array([0.0, 0.0, 0.0]), 1.0, 1.0, 1.0), '[TC35] Origin should be inside unit sphere FAILED'
assert not inside_ellipsoid(np.array([10.0, 0.0, 0.0]), 1.0, 1.0, 1.0), '[TC35] Far point should be outside unit sphere FAILED'

# ---- TC36: CVT Lloyd 可复现且所有点在椭球内 ----
np.random.seed(42)
cvt_pts = cvt_ellipsoid_lloyd(1.0, 1.0, 1.0, 6, n_samples=2000, n_iterations=5, seed=42)
assert cvt_pts.shape == (6, 3), '[TC36] CVT output shape FAILED'
all_inside_36 = all(inside_ellipsoid(p, 1.0, 1.0, 1.0) for p in cvt_pts)
assert all_inside_36, '[TC36] All CVT points must be inside ellipsoid FAILED'

# ---- TC37: sparse_grid_integrate ∫ x²y² dA over [-1,1]² = 4/9 ----
sg_xy = sparse_grid_integrate(lambda p: p[0] ** 2 * p[1] ** 2, 2, 3)
assert abs(sg_xy - 4.0 / 9.0) < 1e-10, '[TC37] Sparse grid 2D x^2 y^2 integral should be 4/9 FAILED'

# ---- TC38: laplacian_1d_pp 周期性矩阵尺寸正确 ----
A_pp38 = laplacian_1d_pp(8, 1.0)
assert A_pp38.shape == (8, 8), '[TC38] Periodic Laplacian shape FAILED'
assert np.all(np.isfinite(A_pp38)), '[TC38] Periodic Laplacian must be finite FAILED'

# ---- TC39: sample_drug_arrival_times 输出有序且非负 ----
np.random.seed(42)
arr_times = sample_drug_arrival_times(50, mean_interval=10.0, cv=0.2)
assert np.all(np.diff(arr_times) >= 0), '[TC39] Drug arrival times must be sorted FAILED'
assert arr_times[0] >= 0, '[TC39] Drug arrival times must be nonnegative FAILED'

# ---- TC40: solve_tissue_concentration_profile 返回三个非空数组 ----
x40, C_exact40, C_fd40 = solve_tissue_concentration_profile(20, 0.01, D_eff=1e-9, clearance_rate=0.01, influx=1.0)
assert len(x40) == 20, '[TC40] Tissue profile x length FAILED'
assert len(C_exact40) == 20, '[TC40] Tissue profile C_exact length FAILED'
assert len(C_fd40) == 20, '[TC40] Tissue profile C_fd length FAILED'
assert np.all(C_exact40 >= 0), '[TC40] Tissue concentrations must be nonnegative FAILED'
