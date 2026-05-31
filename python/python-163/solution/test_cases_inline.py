# ---- TC01: darcy_velocity_scalar 基本计算 ----
from thm_model import darcy_velocity_scalar, biot_modulus, strain_tensor_2d
q = darcy_velocity_scalar(k=1e-14, mu=1e-3, dp_dx=-100.0, rho_f=1000.0, g=9.81, dz_sign=0.0)
assert isinstance(q, float), '[TC01] 返回值应为标量 FAILED'
assert np.isfinite(q), '[TC01] 结果应有限 FAILED'
assert q > 0, '[TC01] 负压力梯度应产生正通量 FAILED'

# ---- TC02: effective_heat_capacity 已知值计算 ----
rho_c_eff = effective_heat_capacity(phi=0.15, rho_f=1000.0, cp_f=4180.0, rho_r=2700.0, cp_r=850.0)
rho_c_expected = 0.15 * 1000.0 * 4180.0 + 0.85 * 2700.0 * 850.0
assert abs(rho_c_eff - rho_c_expected) < 0.01, '[TC02] 有效热容计算错误 FAILED'

# ---- TC03: effective_thermal_conductivity 已知值计算 ----
lam_eff = effective_thermal_conductivity(phi=0.15, lam_f=0.6, lam_r=2.5)
lam_expected = 0.15 * 0.6 + 0.85 * 2.5
assert abs(lam_eff - lam_expected) < 1e-12, '[TC03] 有效导热系数计算错误 FAILED'

# ---- TC04: biot_modulus 已知值计算 ----
M = biot_modulus(phi=0.15, K_f=2.222e9, alpha=0.8, K_s=50e9)
assert M > 0, '[TC04] Biot模量应为正 FAILED'
assert np.isfinite(M), '[TC04] Biot模量应有限 FAILED'

# ---- TC05: thermal_diffusivity 已知值计算 ----
kappa = thermal_diffusivity(lambda_eff=2.215, rho_eff=2500.0, cp_eff=1000.0)
kappa_expected = 2.215 / (2500.0 * 1000.0)
assert abs(kappa - kappa_expected) < 1e-15, '[TC05] 热扩散率计算错误 FAILED'

# ---- TC06: strain_tensor_2d 对称性验证 ----
exx, ezz, exz = strain_tensor_2d(dux_dx=0.001, duz_dz=0.002, dux_dz=0.0005, duz_dx=0.0005)
exx2, ezz2, exz2 = strain_tensor_2d(dux_dx=0.001, duz_dz=0.002, dux_dz=0.0005, duz_dx=0.0005)
assert exx == exx2, '[TC06] 应变张量应确定性 FAILED'
assert exz == 0.5 * (0.0005 + 0.0005), '[TC06] 剪应变计算错误 FAILED'

# ---- TC07: fluid_density_temperature 范围约束 ----
params_test = THMParameters()
rho_fluid = fluid_density_temperature(p=20e6, T=423.15, params=params_test)
assert 500.0 <= rho_fluid <= 1500.0, '[TC07] 流体密度超出物理范围 FAILED'
rho_cold = fluid_density_temperature(p=20e6, T=273.15, params=params_test)
assert rho_cold > rho_fluid, '[TC07] 冷流体密度应大于热流体 FAILED'

# ---- TC08: fluid_viscosity_temperature 单调递减 ----
mu_hot = fluid_viscosity_temperature(400.0)
mu_cold = fluid_viscosity_temperature(300.0)
assert mu_hot < mu_cold, '[TC08] 粘度应随温度升高而减小 FAILED'
assert 1e-4 <= mu_hot <= 5e-3, '[TC08] 粘度裁剪范围错误 FAILED'

# ---- TC09: lambert_w_approx W(1.0) 解析验证 ----
w1 = lambert_w_approx(1.0, branch=0)
assert abs(w1 - 0.56714329) < 0.01, '[TC09] Lambert W(1.0) 近似精度不足 FAILED'

# ---- TC10: lambert_w_approx W(-1/e) = -1.0 ----
w_neg = lambert_w_approx(-np.exp(-1.0), branch=0)
assert abs(w_neg - (-1.0)) < 0.01, '[TC10] Lambert W(-1/e) 应等于 -1.0 FAILED'

# ---- TC11: lambert_w_approx W(0)=0 (n_iter=0避免Halley迭代log(0)) ----
w0 = lambert_w_approx(0.0, branch=0, n_iter=0)
assert abs(w0) < 1e-10, '[TC11] Lambert W(0.0) 应等于 0.0 FAILED'

# ---- TC12: injection_well_pressure 基本计算 ----
p_inj_test = injection_well_pressure(m_dot=5.0, k_inj=1e-14, mu_inj=1e-3, h_inj=100.0, r_e=500.0, r_w=0.1, p_res=20e6)
assert p_inj_test > 20e6, '[TC12] 注入压力应大于储层压力 FAILED'
assert np.isfinite(p_inj_test), '[TC12] 注入压力应有限 FAILED'

# ---- TC13: wellbore_pressure_drop_lambert 基本计算 ----
dp = wellbore_pressure_drop_lambert(m_dot=5.0, D_well=0.2, L_well=1500.0, T_well=400.0, rho_f=950.0, mu_ref=1e-3)
assert dp > 0, '[TC13] 井筒压降应为正 FAILED'
assert np.isfinite(dp), '[TC13] 井筒压降应有限 FAILED'

# ---- TC14: l2_error 相同数组误差为零 ----
u = np.array([1.0, 2.0, 3.0])
err = l2_error(u, u.copy())
assert err < 1e-15, '[TC14] 相同数组L2误差应为零 FAILED'

# ---- TC15: l2_error 不同数组正误差 ----
err_diff = l2_error(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
assert err_diff > 0.5, '[TC15] 不同数组应有正误差 FAILED'

# ---- TC16: newton_raphson 求解 x^3-2=0 ----
root, conv, it = newton_raphson(lambda x: x**3 - 2.0, lambda x: 3.0*x**2, x0=1.5)
assert conv, '[TC16] Newton-Raphson 应收敛 FAILED'
assert abs(root - 2.0**(1.0/3.0)) < 1e-10, '[TC16] x^3-2=0 根值不正确 FAILED'

# ---- TC17: bisection 求解 x^3-2=0 ----
root2, conv2, it2 = bisection(lambda x: x**3 - 2.0, 1.0, 2.0)
assert conv2, '[TC17] 二分法应收敛 FAILED'
assert abs(root2 - 2.0**(1.0/3.0)) < 1e-10, '[TC17] 二分法根值不正确 FAILED'

# ---- TC18: convergence_rate 已知收敛率 ----
errs = np.array([0.1, 0.025, 0.00625])
hs = np.array([1.0, 0.5, 0.25])
rates = convergence_rate(errs, hs)
assert len(rates) == 2, '[TC18] 应收敛率数量错误 FAILED'
assert abs(rates[0] - 2.0) < 0.01, '[TC18] 收敛率应约为2.0 (O(h^2)) FAILED'

# ---- TC19: richardson_extrapolation ----
from convergence_analysis import richardson_extrapolation
vals = np.array([4.0, 3.5])
hs_rich = np.array([1.0, 0.5])
ext, err_est = richardson_extrapolation(vals, hs_rich, p_expected=2.0)
assert np.isfinite(ext), '[TC19] Richardson外推值应有限 FAILED'

# ---- TC20: CubicSplineInterpolator 插值精度 ----
from spline_properties import build_temperature_spline_property
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_data = x_data ** 2
spline = build_temperature_spline_property(x_data, y_data, "test")
y_interp = spline.evaluate(3.5)
assert abs(y_interp - 12.25) < 0.05, '[TC20] 样条插值 x^2 在3.5处应接近12.25 FAILED'

# ---- TC21: 默认样条可用性 ----
spline_k = default_rock_thermal_conductivity_spline()
val_400K = spline_k.evaluate(400.0)
assert 1.5 < val_400K < 3.5, '[TC21] 400K时岩石导热系数应在合理范围 FAILED'

spline_mu = default_fluid_viscosity_spline()
val_mu_300K = spline_mu.evaluate(300.0)
assert 1e-4 < val_mu_300K < 2e-3, '[TC21] 300K时流体粘度应在合理范围 FAILED'

# ---- TC22: Triangle 正三角形质量 ----
tri_eq = Triangle(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]]))
assert tri_eq.quality > 0.99, '[TC22] 正三角形质量应接近1.0 FAILED'
assert tri_eq.is_well_formed(), '[TC22] 正三角形应为良态 FAILED'

# ---- TC23: Triangle 退化三角形 ----
tri_deg = Triangle(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]))
assert tri_deg.quality == 0.0, '[TC23] 退化三角形质量应为0 FAILED'
assert not tri_deg.is_well_formed(), '[TC23] 退化三角形应为非良态 FAILED'

# ---- TC24: quadrature_rules 对常数函数精确积分 ----
n_pts, xq, yq, zq, wq = hexahedron_witherden_rule(5)
integral_1 = np.sum(wq)
assert abs(integral_1 - 1.0) < 1e-12, '[TC24] 单位立方体上1的积分应为1.0 FAILED'

# ---- TC25: quadrature_rules 多项式积分 ----
integral_poly_test = np.sum(wq * (xq**2 * yq**2 * zq**2))
assert abs(integral_poly_test - 1.0/27.0) < 1e-12, '[TC25] x^2y^2z^2在[0,1]^3上积分应为1/27 FAILED'

# ---- TC26: orthogonal_fit 输出结构和有限性验证 ----
x_fit = np.linspace(0.0, 500.0, 64)
y_fit = np.sin(x_fit / 100.0)
fit = fit_permeability_field_1d(x_fit, y_fit, n_terms=5)
y_pred = fit.evaluate(x_fit)
assert len(y_pred) == len(x_fit), '[TC26] 拟合输出长度应等于输入长度 FAILED'
assert np.all(np.isfinite(y_pred)), '[TC26] 拟合输出应全部有限 FAILED'
assert fit.n_terms == 5, '[TC26] 项数应为5 FAILED'

# ---- TC27: fracture_aperture_markov_evolution 可复现性 ----
np.random.seed(42)
hist1 = fracture_aperture_markov_evolution(a_initial=1e-4, thermal_cycles=50, delta_a=1e-5, closure_prob=0.3, opening_prob=0.4)
np.random.seed(42)
hist2 = fracture_aperture_markov_evolution(a_initial=1e-4, thermal_cycles=50, delta_a=1e-5, closure_prob=0.3, opening_prob=0.4)
assert np.allclose(hist1, hist2), '[TC27] 固定种子应产生相同裂缝演化 FAILED'
assert hist1[-1] >= 0, '[TC27] 裂缝孔径应为非负 FAILED'

# ---- TC28: effective_permeability_from_fracture_network ----
aps = np.array([1e-4, 2e-4, 5e-5])
k_eff_frac_test = effective_permeability_from_fracture_network(aps, fracture_density=2.0, matrix_perm=1e-14)
assert k_eff_frac_test >= 1e-14, '[TC28] 有效渗透率应不小于基质渗透率 FAILED'
assert k_eff_frac_test <= 1e-12, '[TC28] 有效渗透率不应超过上界 FAILED'

# ---- TC29: mc_integral_thermal_energy 可复现性 ----
np.random.seed(42)
mean1, std1 = mc_integral_thermal_energy(n_samples=2000, T_mean=400.0, T_std=20.0, rho_eff=2500.0, cp_eff=1000.0, volume=1e7)
np.random.seed(42)
mean2, std2 = mc_integral_thermal_energy(n_samples=2000, T_mean=400.0, T_std=20.0, rho_eff=2500.0, cp_eff=1000.0, volume=1e7)
assert abs(mean1 - mean2) < 1e-10, '[TC29] 固定种子MC应可复现 FAILED'

# ---- TC30: ConvergenceMonitor 收敛检测 ----
monitor = ConvergenceMonitor(tol=1e-4, max_iter=100)
conv, reason = monitor.check(residual=1e-5)
assert conv, '[TC30] 低于容差的残差应判定收敛 FAILED'
assert "residual below tolerance" in reason.lower(), '[TC30] 收敛原因不正确 FAILED'

# ---- TC31: ConvergenceMonitor 发散检测 ----
monitor2 = ConvergenceMonitor(tol=1e-8, max_iter=100)
monitor2.check(residual=0.001)
conv2, reason2 = monitor2.check(residual=0.1)
assert conv2, '[TC31] 残差突增十倍应检测到发散 FAILED'

# ---- TC32: THMParameters 属性一致性 ----
params_check = THMParameters()
lam_check = params_check.lame_lambda()
mu_check = params_check.lame_mu()
assert lam_check > 0 and mu_check > 0, '[TC32] Lamé参数应为正 FAILED'
E_check = mu_check * (3.0 * lam_check + 2.0 * mu_check) / (lam_check + mu_check)
assert abs(E_check - params_check.young_modulus) / params_check.young_modulus < 0.01, '[TC32] 弹性参数不一致 FAILED'

# ---- TC33: SimulationIO 写入与读取回环 ----
import tempfile, os as _os
io_test = SimulationIO(output_dir=_os.path.join(tempfile.gettempdir(), "test_thm_io"))
test_field = np.array([[1.0, 2.0], [3.0, 4.0]])
io_test.write_field("test_field.txt", test_field, "test")
read_back = io_test.read_matrix_file(_os.path.join(io_test.output_dir, "test_field.txt"))
assert read_back.shape == (2, 2), '[TC33] 读回矩阵形状错误 FAILED'
assert abs(np.sum(read_back) - 10.0) < 1e-10, '[TC33] 读回矩阵值错误 FAILED'
# Cleanup
import shutil
shutil.rmtree(io_test.output_dir, ignore_errors=True)

# ---- TC34: darcy_velocity_scalar 零渗透率 ----
q_zero_k = darcy_velocity_scalar(k=0.0, mu=1e-3, dp_dx=-100.0, rho_f=1000.0, g=0.0)
assert q_zero_k == 0.0, '[TC34] 零渗透率应产生零通量 FAILED'

# ---- TC35: linf_error 计算 ----
from convergence_analysis import linf_error
err_inf = linf_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]))
assert abs(err_inf - 0.5) < 1e-15, '[TC35] L∞误差应为0.5 FAILED'
