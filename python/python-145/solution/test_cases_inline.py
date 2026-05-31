# ---- TC01: log_normal_pdf 基础值验证 ----
pdf_vals_check = sf.log_normal_pdf(np.array([1.0]), 0.0, 1.0)
assert np.isfinite(pdf_vals_check).all() and pdf_vals_check[0] > 0.0, '[TC01] log_normal_pdf(1.0) 应返回正值 FAILED'

# ---- TC02: log_normal_pdf 负值输入应返回0 ----
pdf_zero = sf.log_normal_pdf(np.array([-1.0, 0.0, -5.0]), 0.0, 1.0)
assert np.allclose(pdf_zero, 0.0), '[TC02] log_normal_pdf 对非正输入应返回 0 FAILED'

# ---- TC03: log_normal_cdf 渐近性质 ----
cdf_small = sf.log_normal_cdf(np.array([1e-10]), 0.0, 1.0)
cdf_large = sf.log_normal_cdf(np.array([1e10]), 0.0, 1.0)
assert cdf_small[0] < 1e-6, '[TC03] log_normal_cdf(极小值) 应接近 0 FAILED'
assert cdf_large[0] > 0.9999, '[TC03] log_normal_cdf(极大值) 应接近 1 FAILED'

# ---- TC04: log_normal_cdf_inv 正值与单调性 ----
import numpy as np
np.random.seed(42)
p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
inv_vals = sf.log_normal_cdf_inv(p_test, 0.0, 1.0)
# 逆CDF 应返回正值且单调递增
assert np.all(inv_vals > 0.0), '[TC04] log_normal_cdf_inv 应返回正值 FAILED'
assert np.all(np.diff(inv_vals) > 0.0), '[TC04] log_normal_cdf_inv 应单调递增 FAILED'

# ---- TC05: normal_01_cdf_inv 对称性 ----
p_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
z_vals = sf.normal_01_cdf_inv(p_vals)
assert np.allclose(z_vals[0], -z_vals[-1], atol=1e-10), '[TC05] normal_01_cdf_inv 对称性失败 FAILED'
assert np.isclose(z_vals[2], 0.0, atol=1e-10), '[TC05] normal_01_cdf_inv(0.5) 应为 0 FAILED'

# ---- TC06: lambert_w 上分支基本值 ----
lw_vals = sf.lambert_w(np.array([0.0, 1.0, np.e]), branch=0)
assert np.isclose(lw_vals[0], 0.0, atol=1e-10), '[TC06] W_0(0) 应为 0 FAILED'
assert lw_vals[1] > 0.5 and lw_vals[2] > 0.9, '[TC06] W_0(1) 和 W_0(e) 应为正 FAILED'

# ---- TC07: lambert_w 恒等式验证 ----
test_x = np.array([0.5, 2.0, 5.0])
w_vals = sf.lambert_w(test_x, branch=0)
for idx, (wv, xv) in enumerate(zip(w_vals, test_x)):
    reconstructed = wv * np.exp(wv)
    assert np.isclose(reconstructed, xv, atol=1e-6), f'[TC07] W_0({xv}) 恒等式失败 FAILED'

# ---- TC08: hep_coefficients 系数对称性 ----
c4 = pc.hep_coefficients(4)
c5 = pc.hep_coefficients(5)
assert len(c4) == 5 and len(c5) == 6, '[TC08] hep_coefficients 返回长度错误 FAILED'
# He_4(0) = 3 (偶次), He_5(0) = 0 (奇次)
assert np.isclose(c4[0], 3.0, atol=1e-10), '[TC08] He_4 常数项应为 3 FAILED'
assert np.isclose(c5[0], 0.0, atol=1e-10), '[TC08] He_5 常数项应为 0 FAILED'

# ---- TC09: hep_value 与系数一致性 ----
c3 = pc.hep_coefficients(3)
x_test_h = np.array([0.0, 0.5, 1.0, 2.0])
h3_vals = pc.hep_value(x_test_h, 3)
# He_3(x) = x^3 - 3x
h3_expected = x_test_h**3 - 3.0 * x_test_h
assert np.allclose(h3_vals, h3_expected, atol=1e-10), '[TC09] He_3(x) 值不匹配 FAILED'

# ---- TC10: hep_value 递推一致性 ----
x_check = np.array([0.0, -1.0, 2.0, -3.0, 5.0])
for n in range(0, 6):
    hv = pc.hep_value(x_check, n)
    assert np.isfinite(hv).all(), f'[TC10] He_{n}(x) 产生非有限值 FAILED'

# ---- TC11: hep_values 输出形状 ----
x_in = np.array([0.5, 1.0, 1.5])
v_all = pc.hep_values(x_in, 5)
assert v_all.shape == (3, 6), '[TC11] hep_values 输出形状错误 FAILED'
assert np.allclose(v_all[:, 0], 1.0), '[TC11] He_0 应全为 1 FAILED'

# ---- TC12: hermite_product_polynomial_value 基本验证 ----
np.random.seed(42)
xi_prod = np.random.randn(10, 2)
prod_val = pc.hermite_product_polynomial_value(2, [2, 1], xi_prod)
assert prod_val.shape == (10,), '[TC12] 乘积多项式输出形状错误 FAILED'
assert np.isfinite(prod_val).all(), '[TC12] 乘积多项式产生非有限值 FAILED'

# ---- TC13: generate_multi_indices 数量验证 ----
mi_2_2 = pc.generate_multi_indices(2, 2)
# N = (2+2)!/(2!*2!) = 6
assert mi_2_2.shape[0] == 6 and mi_2_2.shape[1] == 2, '[TC13] d=2,p=2 应有 6 个指标 FAILED'
mi_2_3 = pc.generate_multi_indices(2, 3)
assert mi_2_3.shape[0] == 10, '[TC13] d=2,p=3 应有 10 个指标 FAILED'

# ---- TC14: polynomial_chaos_expand 确定性测试 ----
np.random.seed(123)
mi_test = pc.generate_multi_indices(2, 2)
coeffs_test = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
xi_test = np.random.randn(100, 2)
pc_samples = pc.polynomial_chaos_expand(coeffs_test, mi_test, xi_test)
assert pc_samples.shape == (100,), '[TC14] PC 展开输出形状错误 FAILED'
# 均值应接近 coeffs[0]
assert abs(np.mean(pc_samples) - coeffs_test[0]) < 0.5, '[TC14] PC 展开均值偏差过大 FAILED'

# ---- TC15: sobol_sensitivity 归一化验证 ----
mi_sobol = pc.generate_multi_indices(2, 2)
coeffs_sobol = np.ones(mi_sobol.shape[0], dtype=float) * 0.1
coeffs_sobol[0] = 1.0
total_var, sobol = pc.sobol_sensitivity(coeffs_sobol, mi_sobol)
assert total_var > 0.0, '[TC15] Sobol 总方差应为正 FAILED'
assert np.all(sobol >= 0.0) and np.all(sobol <= 1.0), '[TC15] Sobol 指标应在 [0,1] 内 FAILED'

# ---- TC16: lorenz96_parameters 基本调用 ----
n_l, f_l, p_l, t0_l, y0_l, ts_l = sd.lorenz96_parameters(n=4, force=8.0)
assert n_l == 4 and abs(f_l - 8.0) < 1e-10, '[TC16] lorenz96_parameters 参数不匹配 FAILED'
assert y0_l.shape == (4,), '[TC16] y0 形状错误 FAILED'

# ---- TC17: lorenz96_deriv 恒定解检验 ----
np.random.seed(42)
y_const = 8.0 * np.ones(4)
dy = sd.lorenz96_deriv(0.0, y_const, force=8.0)
# 当所有 y_i = F 时，dy_i = (F-F)*F - F + F = 0
assert np.allclose(dy, 0.0, atol=1e-10), '[TC17] Lorenz-96 恒定解导数应为零 FAILED'

# ---- TC18: duffing_parameters 基本调用 ----
a_d, b_d, g_d, d_d, o_d, t0_d, y0_d, ts_d = sd.duffing_parameters()
assert abs(a_d - 1.0) < 1e-10 and abs(b_d - 5.0) < 1e-10, '[TC18] duffing 默认参数不匹配 FAILED'
assert y0_d.shape == (2,), '[TC18] duffing y0 形状错误 FAILED'

# ---- TC19: duffing_deriv 零位移导数 ----
dy_d = sd.duffing_deriv(0.0, np.array([0.0, 0.0]))
# dy1/dt = y2 = 0, dy2/dt = -δ*0 - α*0 - β*0 + γ*cos(0) = γ
assert np.isclose(dy_d[0], 0.0, atol=1e-10), '[TC19] Duffing x=0 时 dx/dt 应为 0 FAILED'
assert dy_d[1] > 0.0, '[TC19] Duffing x=0 时 dv/dt 应为正值 FAILED'

# ---- TC20: oregonator_parameters 基本调用 ----
e1_o, e2_o, q_o, f_o, t0_o, y0_o, ts_o = sd.oregonator_parameters()
assert y0_o.shape == (3,), '[TC20] Oregonator y0 形状错误 FAILED'
assert e1_o > 0.0 and e2_o > 0.0, '[TC20] Oregonator 无量纲参数应为正 FAILED'

# ---- TC21: oregonator_deriv 稳态解验证 ----
e1, e2, q_o21, f_o21, _, _, _ = sd.oregonator_parameters()
y_ss = np.array([0.0, 0.0, 0.0])
dy_ss = sd.oregonator_deriv(0.0, y_ss, e1, e2, q_o21, f_o21)
assert dy_ss.shape == (3,), '[TC21] Oregonator 导数输出形状错误 FAILED'
assert np.isfinite(dy_ss).all(), '[TC21] Oregonator 导数产生非有限值 FAILED'

# ---- TC22: multi_factor_coupling 输出非负性 ----
np.random.seed(42)
lorenz_y = np.random.randn(8)
duffing_y = np.array([0.5, -0.2])
oregonator_y = np.array([1.0, 0.5, 0.8])
sigma_c = sd.multi_factor_coupling(0.0, lorenz_y, duffing_y, oregonator_y, n_factors=3)
assert sigma_c.shape == (3,), '[TC22] 耦合输出形状错误 FAILED'
assert np.all(sigma_c >= 0.0), '[TC22] 波动率耦合必须非负 FAILED'

# ---- TC23: rk2_step 简单 ODE 测试 ----
def exp_deriv(t, y):
    return y
import numpy as np
y_rk2 = ts.rk2_step(exp_deriv, 0.0, np.array([1.0]), 0.1)
assert y_rk2.shape == (1,), '[TC23] rk2_step 输出形状错误 FAILED'
assert y_rk2[0] > 0.0, '[TC23] rk2_step 指数增长验证失败 FAILED'

# ---- TC24: rk3_step 简单 ODE 测试 ----
y_rk3 = ts.rk3_step(exp_deriv, 0.0, np.array([1.0]), 0.1)
assert y_rk3.shape == (1,), '[TC24] rk3_step 输出形状错误 FAILED'
assert y_rk3[0] > 0.0, '[TC24] rk3_step 指数增长验证失败 FAILED'

# ---- TC25: rk23_integrate 输出形状 ----
np.random.seed(42)
def linear_deriv(t, y):
    return np.array([-y[0]])
t_rk, y_rk, e_rk = ts.rk23_integrate(linear_deriv, (0.0, 1.0), np.array([1.0]), n_steps=50)
assert t_rk.shape == (51,) and y_rk.shape == (51, 1) and e_rk.shape == (51, 1), '[TC25] rk23 输出形状错误 FAILED'
assert np.isfinite(y_rk).all(), '[TC25] rk23 产生非有限值 FAILED'

# ---- TC26: triangle_area 已知面积验证 ----
p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
area = fem.triangle_area(p1, p2, p3)
assert np.isclose(area, 0.5, atol=1e-10), '[TC26] 单位直角三角形面积应为 0.5 FAILED'

# ---- TC27: generate_rectangular_grid 输出形状 ----
node_xy, elem_node = fem.generate_rectangular_grid(3, 3, xl=0.0, xr=1.0, yb=0.0, yt=1.0)
expected_nodes = (2*3-1) * (2*3-1)  # 25
expected_elems = (3-1) * (3-1) * 2  # 8
assert node_xy.shape == (expected_nodes, 2), f'[TC27] 节点数应为 {expected_nodes} FAILED'
assert elem_node.shape == (expected_elems, 6), f'[TC27] 单元数应为 {expected_elems} FAILED'

# ---- TC28: get_quad_rule_triangle 权重和 ----
w3, xy3 = fem.get_quad_rule_triangle(3)
assert np.isclose(np.sum(w3), 0.5, atol=1e-10), '[TC28] 三角形积分权重和应为 0.5 FAILED'

# ---- TC29: basis_11_t6 节点求值 ----
node_test = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[0.5,0.0],[0.5,0.5],[0.0,0.5]], dtype=float)
# 在顶点1处，基函数1应为1
b1, _, _ = fem.basis_11_t6(node_test, 1, np.array([0.0, 0.0]))
assert np.isclose(b1, 1.0, atol=1e-10), '[TC29] T6 基函数在对应节点应为 1 FAILED'

# ---- TC30: assemble_fem_matrices 对称性 ----
def k_zero(x, y): return 0.0
A_sym, M_sym = fem.assemble_fem_matrices(node_xy, elem_node, element_order=6, k_coef=k_zero, nq=3)
diff_A = np.linalg.norm((A_sym - A_sym.T).toarray())
assert diff_A < 1e-10, '[TC30] 刚度矩阵不对称 FAILED'
diff_M = np.linalg.norm((M_sym - M_sym.T).toarray())
assert diff_M < 1e-10, '[TC30] 质量矩阵不对称 FAILED'

# ---- TC31: shepard_interp_2d 精确插值 ----
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([0.0, 0.0, 0.0])
zd = np.array([0.0, 1.0, 4.0])
xi_s = np.array([0.0, 1.0, 2.0])
yi_s = np.array([0.0, 0.0, 0.0])
zi_s = ycc.shepard_interp_2d(xd, yd, zd, 2.0, xi_s, yi_s)
assert np.allclose(zi_s, zd, atol=1e-10), '[TC31] Shepard 在数据点上应精确插值 FAILED'

# ---- TC32: horner_eval 基本验证 ----
c_poly = np.array([1.0, 2.0, 3.0])  # 1 + 2x + 3x^2
p0 = ycc.horner_eval(c_poly, np.array([0.0]))
assert np.isclose(p0, 1.0, atol=1e-10), '[TC32] Horner p(0) 应为 c0 FAILED'
p1 = ycc.horner_eval(c_poly, np.array([1.0]))
assert np.isclose(p1, 6.0, atol=1e-10), '[TC32] Horner p(1) 应为 6 FAILED'

# ---- TC33: fit_yield_polynomial 基本拟合 ----
mats = np.linspace(0.5, 10.0, 8)
yields_true = 0.05 + 0.01 * mats
c_fit, res, cond = ycc.fit_yield_polynomial(mats, yields_true, degree=1)
assert np.isclose(c_fit[0], 0.05, atol=1e-6), '[TC33] 多项式拟合截距错误 FAILED'
assert np.isclose(c_fit[1], 0.01, atol=1e-6), '[TC33] 多项式拟合斜率错误 FAILED'

# ---- TC34: extract_curve_features 单调曲线 ----
m_feat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_feat = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
feat = ycc.extract_curve_features(m_feat, y_feat)
assert feat['start'] is not None and feat['end'] is not None, '[TC34] 特征提取应返回起止点 FAILED'

# ---- TC35: bond_price_from_forward 基本定价 ----
T_test = np.linspace(0.0, 5.0, 51)
f_const = 0.05 * np.ones_like(T_test)
P_1y = tsp.bond_price_from_forward(f_const, T_test, 0.0, 1.0)
P_2y = tsp.bond_price_from_forward(f_const, T_test, 0.0, 2.0)
assert np.isclose(P_1y, np.exp(-0.05), atol=1e-6), '[TC35] 恒定利率债券定价错误 FAILED'
assert np.isclose(P_2y, np.exp(-0.10), atol=1e-6), '[TC35] 恒定利率债券定价错误 FAILED'

# ---- TC36: zero_yield_from_forward 基本收益率 ----
zy_1y = tsp.zero_yield_from_forward(f_const, T_test, 0.0, 1.0)
zy_2y = tsp.zero_yield_from_forward(f_const, T_test, 0.0, 2.0)
assert np.isclose(zy_1y, 0.05, atol=1e-6), '[TC36] 恒定利率零息收益率应为 0.05 FAILED'
assert np.isclose(zy_2y, 0.05, atol=1e-6), '[TC36] 恒定利率零息收益率应为 0.05 FAILED'

# ---- TC37: bond_price_from_forward 单调性 ----
T_s = np.linspace(0.0, 10.0, 101)
f_pos = 0.03 * np.ones_like(T_s)
P_3y = tsp.bond_price_from_forward(f_pos, T_s, 0.0, 3.0)
P_5y = tsp.bond_price_from_forward(f_pos, T_s, 0.0, 5.0)
assert P_3y > P_5y, '[TC37] 正利率下债券价格应随期限递减 FAILED'

# ---- TC38: instantaneous_short_rate 基本调用 ----
r_short = tsp.instantaneous_short_rate(f_const, T_test)
assert r_short > 0.0, '[TC38] 短期利率应为正 FAILED'

# ---- TC39: st_to_coo / coo_to_st 往返 ----
import numpy as np
from scipy import sparse as sp
mat_small = sp.coo_matrix(([1.0, 2.0, 3.0], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
ist_s, jst_s, ast_s = sla.coo_to_st(mat_small)
mat_back = sla.st_to_coo(ist_s, jst_s, ast_s, (3, 3))
assert np.allclose(mat_back.toarray(), mat_small.toarray()), '[TC39] COO-ST 往返失败 FAILED'

# ---- TC40: estimate_bandwidth 对角矩阵 ----
mat_diag = sp.eye(5, format='coo')
bw = sla.estimate_bandwidth(mat_diag)
assert bw == 0, '[TC40] 对角矩阵带宽应为 0 FAILED'

# ---- TC41: sparse_matvec 基本验证 ----
A_test = sp.eye(3, format='csr') * 2.0
x_test_sla = np.array([1.0, 2.0, 3.0])
y_test_sla = sla.sparse_matvec(A_test, x_test_sla)
assert np.allclose(y_test_sla, np.array([2.0, 4.0, 6.0])), '[TC41] 稀疏矩阵向量乘法错误 FAILED'

# ---- TC42: HJMMultiFactorModel 初始化 ----
import hjm_model as hjm
model = hjm.HJMMultiFactorModel(n_factors=3, sigma0=0.02, pc_degree=2, pc_dim=2)
assert model.n_factors == 3, '[TC42] HJM 模型因子数错误 FAILED'
assert len(model.kappa) >= 3, '[TC42] HJM 模型 kappa 长度错误 FAILED'

# ---- TC43: HJMMultiFactorModel volatility_structure ----
np.random.seed(42)
lorenz_y = np.random.randn(8)
duffing_y = np.array([0.5, -0.2])
oregonator_y = np.array([1.0, 0.5, 0.8])
vol = model.volatility_structure(0.0, 1.0, lorenz_y, duffing_y, oregonator_y)
assert vol.shape == (3,), '[TC43] 波动率结构输出形状错误 FAILED'
assert np.all(vol >= 0.0) and np.all(vol <= 1.0), '[TC43] 波动率应在 [0,1] 内 FAILED'

# ---- TC44: musiela_drift 非负性 ----
def sigma_const(t, s):
    return 0.02
alpha = tsp.musiela_drift([sigma_const], 2.0, t=0.0)
assert alpha >= 0.0, '[TC44] HJM 漂移项应为非负 FAILED'

# ---- TC45: write_st_file / read_st_file 往返 ----
from scipy import sparse as sp
mat_w = sp.coo_matrix(([2.0, 3.0], ([0, 1], [0, 1])), shape=(2, 2))
ist_w, jst_w, ast_w = sla.coo_to_st(mat_w)
temp_file = "/tmp/test_st_rw_45.st"
sla.write_st_file(temp_file, 2, 2, len(ast_w), ist_w, jst_w, ast_w)
m_r, n_r, nst_r, ist_r, jst_r, ast_r = sla.read_st_file(temp_file)
assert m_r == 2 and n_r == 2, '[TC45] ST 文件读写维度错误 FAILED'
mat_r = sla.st_to_coo(ist_r, jst_r, ast_r, (m_r, n_r))
assert np.allclose(mat_r.toarray(), mat_w.toarray()), '[TC45] ST 文件读写往返失败 FAILED'
import os; os.remove(temp_file)
