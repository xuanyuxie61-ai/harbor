# ---- TC01: 单位正方形有向面积为1 ----
square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
area_sq = polygon_area_2d(square)
assert abs(area_sq - 1.0) < 1e-10, '[TC01] 单位正方形有向面积为1 FAILED'

# ---- TC02: 单位正方形四边形面积为1 ----
quad_sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
area_quad = quadrilateral_area(quad_sq)
assert abs(area_quad - 1.0) < 1e-10, '[TC02] 单位正方形四边形面积为1 FAILED'

# ---- TC03: 标准正方形为凸四边形 ----
assert quadrilateral_is_convex(quad_sq) == True, '[TC03] 标准正方形为凸四边形 FAILED'

# ---- TC04: 网格法估计单位正方形面积比例精确为1 ----
est_grid = area_estimate_grid(square, 1.0, 1.0, 20)
assert abs(est_grid - 1.0) < 1e-10, '[TC04] 网格法估计单位正方形面积比例精确为1 FAILED'

# ---- TC05: QMC估计单位正方形面积比例接近1 ----
est_qmc = area_estimate_qmc(square, 1.0, 1.0, 2000)
assert abs(est_qmc - 1.0) < 0.1, '[TC05] QMC估计单位正方形面积比例接近1 FAILED'

# ---- TC06: Mandelbrot原点属于集合逃逸时间为count_max+1 ----
esc = mandelbrot_escape_time(0.0, 0.0, count_max=30)
assert esc == 31, '[TC06] Mandelbrot原点属于集合逃逸时间为count_max+1 FAILED'

# ---- TC07: 散射强度场输出在[0,1]范围内 ----
Xg, Yg = np.meshgrid(np.linspace(-1.0, 1.0, 11), np.linspace(-1.0, 1.0, 11))
strength = compute_scattering_strength(Xg, Yg, count_max=20)
assert np.all(strength >= 0.0) and np.all(strength <= 1.0), '[TC07] 散射强度场输出在[0,1]范围内 FAILED'

# ---- TC08: 分形孔隙度场输出在[0,1]范围内 ----
porosity = fractal_porosity_field(21, 21, fractal_dim=1.8, rng=np.random.default_rng(42))
assert np.all(porosity >= 0.0) and np.all(porosity <= 1.0), '[TC08] 分形孔隙度场输出在[0,1]范围内 FAILED'

# ---- TC09: IFS分形点集形状为(n_points, 2) ----
pts_ifs = ifs_leaf_fractal(n_points=500, rng=np.random.default_rng(42))
assert pts_ifs.shape == (500, 2), '[TC09] IFS分形点集形状为(n_points, 2) FAILED'

# ---- TC10: GLL权重和应为2 ----
pts_gll, wts_gll = gauss_lobatto_legendre_points_weights(5)
assert abs(np.sum(wts_gll) - 2.0) < 1e-10, '[TC10] GLL权重和应为2 FAILED'

# ---- TC11: 谱元法GLL积分低阶多项式精确 ----
errors_quad = test_quadrature_exactness(max_degree=3)
assert all(err < 1e-12 for err in errors_quad), '[TC11] 谱元法GLL积分低阶多项式精确 FAILED'

# ---- TC12: 混合积分Legendre乘积解析验证 ----
val_mix = monomial_integral_mixed(2, np.array([1, 1]), np.zeros(2), np.zeros(2), np.array([0, 0]))
assert abs(val_mix - 4.0) < 1e-12, '[TC12] 混合积分Legendre乘积解析验证 FAILED'

# ---- TC13: Helmholtz矩阵为复数方阵 ----
c_const = np.full(21, 3000.0)
A_h = build_helmholtz_matrix_1d(21, 10.0, c_const, 2.0 * np.pi * 10.0)
assert A_h.shape == (21, 21) and np.iscomplexobj(A_h), '[TC13] Helmholtz矩阵为复数方阵 FAILED'

# ---- TC14: Helmholtz求解波场有限 ----
u_h, res_h = solve_helmholtz_1d(21, 10.0, c_const, 2.0 * np.pi * 10.0, 10, source_amp=1e6, max_iter=50, tol=1e-6)
assert np.all(np.isfinite(u_h)), '[TC14] Helmholtz求解波场有限 FAILED'

# ---- TC15: 驻波精确解解析验证 ----
u_sw, ut_sw, utt_sw, ux_sw, uxx_sw = standing_wave_exact(np.pi / 2, 0.0, c=0.5)
assert abs(u_sw - 1.0) < 1e-12, '[TC15] 驻波精确解解析验证 FAILED'

# ---- TC16: 驻波精确解残差为零 ----
res_sw = standing_wave_residual(np.linspace(0.0, np.pi, 11), 0.5, c=0.5)
assert np.max(np.abs(res_sw)) < 1e-12, '[TC16] 驻波精确解残差为零 FAILED'

# ---- TC17: 双弹簧参数返回正确字段 ----
params_sp = spring_double_parameters(m1=3.0, m2=5.0, k1=2.0, k2=8.0)
assert params_sp['m1'] == 3.0 and params_sp['k2'] == 8.0, '[TC17] 双弹簧参数返回正确字段 FAILED'

# ---- TC18: 双弹簧平衡点导数为零 ----
params_eq = spring_double_parameters(m1=3.0, m2=5.0, k1=2.0, k2=8.0, y0=np.array([0.0, 0.0, 0.0, 0.0]))
dydt_eq = spring_double_deriv(0.0, np.array([0.0, 0.0, 0.0, 0.0]), params_eq)
assert np.all(np.abs(dydt_eq) < 1e-12), '[TC18] 双弹簧平衡点导数为零 FAILED'

# ---- TC19: RK4积分常数ODE精确解 ----
def const_deriv(t, y):
    return np.array([2.0])
t_rk4, y_rk4 = rk4_integrate(const_deriv, (0.0, 1.0), np.array([0.0]), 100)
assert abs(y_rk4[-1, 0] - 2.0) < 1e-10, '[TC19] RK4积分常数ODE精确解 FAILED'

# ---- TC20: 相同数据misfit为零 ----
data = np.array([1.0, 2.0, 3.0])
misfit_same, _ = compute_misfit(data, data)
assert abs(misfit_same) < 1e-12, '[TC20] 相同数据misfit为零 FAILED'

# ---- TC21: 均匀介质一维旅行时解析验证 ----
v_uniform = np.full(101, 1000.0)
dx_uniform = 10.0
tt_uniform = tomography_traveltime_1d(v_uniform, dx_uniform, 0, [50])
assert abs(tt_uniform[0] - 50.0 * dx_uniform / 1000.0) < 1e-10, '[TC21] 均匀介质一维旅行时解析验证 FAILED'

# ---- TC22: 超几何分布PMF概率和为1 ----
w_vals = np.arange(0, 51)
pw = urn_two_color_pdf(w_vals, 50, np.array([100, 100]))
assert abs(np.sum(pw) - 1.0) < 1e-10, '[TC22] 超几何分布PMF概率和为1 FAILED'

# ---- TC23: Urn抽样总数等于draw_num ----
draw_colors = urn_sample(100, 10, 2, np.array([60, 40]), rng=np.random.default_rng(42))
assert np.sum(draw_colors) == 10, '[TC23] Urn抽样总数等于draw_num FAILED'

# ---- TC24: 贝叶斯后验采样接受率在合理范围 ----
import numpy as np
rng_post = np.random.default_rng(42)
def prior_sampler():
    return rng_post.normal(3000.0, 200.0)
def likelihood(v):
    return np.exp(-0.5 * ((v - 3100.0) / 100.0) ** 2)
samples_post, rate_post = bayesian_posterior_sample(likelihood, prior_sampler, n_samples=100, rng=rng_post)
assert 0.0 < rate_post <= 1.0 and len(samples_post) > 0, '[TC24] 贝叶斯后验采样接受率在合理范围 FAILED'

# ---- TC25: 速度模型输出形状和范围正确 ----
vel, x_c, z_c = build_velocity_model(21, 11, use_cvt=False, rng=np.random.default_rng(42))
assert vel.shape == (11, 21) and np.all(vel >= 1000.0) and np.all(vel <= 8000.0), '[TC25] 速度模型输出形状和范围正确 FAILED'

# ---- TC26: 二维随机速度扰动输出形状正确 ----
pert = generate_random_velocity_perturbation(31, 21, theta=2.0, sigma=0.1, dx=1.0, rng=np.random.default_rng(42))
assert pert.shape == (21, 31), '[TC26] 二维随机速度扰动输出形状正确 FAILED'

# ---- TC27: 地震波正演输出形状正确 ----
def source_fn_zero(t):
    return 0.0
c_wave = np.full(21, 3000.0)
u_hist, t_wave = seismic_wave_rk4_1d(21, 10.0, 50, 0.001, c_wave, source_fn_zero, 0, boundary='reflecting')
assert u_hist.shape == (51, 21), '[TC27] 地震波正演输出形状正确 FAILED'

# ---- TC28: CVT优化后generator在定义域内 ----
gx, gz = cvt_optimize(4, (0.0, 1.0), (0.0, 1.0), n_steps=2, sample_num=100, rng=np.random.default_rng(42))
assert np.all((gx >= 0.0) & (gx <= 1.0)) and np.all((gz >= 0.0) & (gz <= 1.0)), '[TC28] CVT优化后generator在定义域内 FAILED'

# ---- TC29: MC面积估计固定种子可复现 ----
rng_mc1 = np.random.default_rng(123)
rng_mc2 = np.random.default_rng(123)
est_mc1 = area_estimate_mc(square, 1.0, 1.0, 500, rng=rng_mc1)
est_mc2 = area_estimate_mc(square, 1.0, 1.0, 500, rng=rng_mc2)
assert abs(est_mc1 - est_mc2) < 1e-12, '[TC29] MC面积估计固定种子可复现 FAILED'

