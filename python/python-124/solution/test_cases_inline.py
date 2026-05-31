# ---- TC01: BoneGeometry构建 - 验证节点和单元数量 ----
from bone_geometry import BoneGeometry
geom = BoneGeometry(width=20.0, height=30.0, cortical_thickness=2.5, nx=9, ny=9)
assert geom.node_num == 289, '[TC01] node_num mismatch FAILED'
assert geom.element_num == 128, '[TC01] element_num mismatch FAILED'
assert geom.node_xy.shape == (2, 289), '[TC01] node_xy shape mismatch FAILED'

# ---- TC02: BoneGeometry - classify_nodes返回正确类型 ----
cortical, trabecular = geom.classify_nodes()
assert isinstance(cortical, np.ndarray), '[TC02] cortical should be ndarray FAILED'
assert len(cortical) + len(trabecular) == geom.node_num, '[TC02] node classification total mismatch FAILED'
assert len(cortical) > 0, '[TC02] should have cortical nodes FAILED'
assert len(trabecular) > 0, '[TC02] should have trabecular nodes FAILED'

# ---- TC03: BoneGeometry - 单元面积总和等于矩形面积 ----
total_area = np.sum(geom.element_area)
assert abs(total_area - 20.0 * 30.0) < 1e-10, '[TC03] total element area should equal rectangle area FAILED'
assert np.all(geom.element_area > 0), '[TC03] all element areas must be positive FAILED'

# ---- TC04: BoneGeometry - compute_half_bandwidth为正整数 ----
hbw = geom.compute_half_bandwidth()
assert hbw > 0, '[TC04] half_bandwidth should be positive FAILED'
assert isinstance(hbw, (int, np.integer)), '[TC04] half_bandwidth should be integer FAILED'

# ---- TC05: BoneGeometry - nx/ny必须为奇数 ----
try:
    BoneGeometry(width=20.0, height=30.0, nx=8, ny=9)
    assert False, '[TC05] should raise ValueError for even nx FAILED'
except ValueError:
    pass

# ---- TC06: point_line_distance_signed - 已知距离验证 ----
from bone_geometry import point_line_distance_signed
p1 = np.array([0.0, 0.0])
p2 = np.array([20.0, 0.0])
p = np.array([10.0, 5.0])
dist = point_line_distance_signed(p1, p2, p)
assert abs(dist - 5.0) < 1e-12, '[TC06] signed distance should be 5.0 FAILED'
# 对称性: 交换p1,p2应得负值
dist_rev = point_line_distance_signed(p2, p1, p)
assert abs(dist_rev + 5.0) < 1e-12, '[TC06] reversed line should give -5.0 FAILED'

# ---- TC07: TrabecularMicrostructure - 孔隙率在[0,1]区间 ----
import numpy as np
np.random.seed(42)
from microstructure_model import TrabecularMicrostructure
micro = TrabecularMicrostructure(grid_size=15, pattern_seed=42)
assert 0.0 <= micro.porosity <= 1.0, '[TC07] porosity should be in [0,1] FAILED'

# ---- TC08: TrabecularMicrostructure - 有效模量为正值 ----
assert micro.effective_modulus > 0, '[TC08] effective_modulus should be positive FAILED'
report = micro.generate_microstructure_report()
assert 'porosity' in report, '[TC08] report should contain porosity FAILED'
assert 'effective_young_modulus_GPa' in report, '[TC08] report should contain effective_young_modulus_GPa FAILED'

# ---- TC09: get_pentomino_matrix - 合法/非法名称 ----
from microstructure_model import get_pentomino_matrix, rotate_matrix_90, flip_matrix
mat = get_pentomino_matrix('X')
assert mat.shape == (3, 3), '[TC09] X pentomino should be 3x3 FAILED'
assert np.sum(mat) == 5, '[TC09] X pentomino should have 5 cells FAILED'
try:
    get_pentomino_matrix('Q')
    assert False, '[TC09] should raise ValueError for unknown name FAILED'
except ValueError:
    pass

# ---- TC10: rotate_matrix_90 - 4次旋转后还原 ----
mat = get_pentomino_matrix('L')
rotated = rotate_matrix_90(mat, k=4)
assert np.array_equal(mat, rotated), '[TC10] 4x90 rotation should return original FAILED'

# ---- TC11: BoneDensityField - 中心求值 ----
from density_field import BoneDensityField, csevl, inits
bdf = BoneDensityField(nx_cheb=8, ny_cheb=8)
val_center = bdf.evaluate(0.0, 0.0)
assert np.isfinite(val_center), '[TC11] center density should be finite FAILED'
assert val_center > 0, '[TC11] center density should be positive FAILED'

# ---- TC12: csevl - 已知切比雪夫级数求值 ----
# 切比雪夫级数: f(x) = c0/2 + Σ c_k T_k(x); 系数cs=[c0,c1,...]
# cs=[2.0,0,0] => f(x) = 1.0*T0(x) = 1.0
coeffs = np.array([2.0, 0.0, 0.0])
val = csevl(0.5, coeffs)
assert abs(val - 1.0) < 1e-12, '[TC12] csevl with only c0=2 should return 1.0 FAILED'
# cs=[0,1,0] => f(x) = 1.0*T1(x) = x = 0.3
coeffs2 = np.array([0.0, 1.0, 0.0])
val2 = csevl(0.3, coeffs2)
assert abs(val2 - 0.3) < 1e-12, '[TC12] csevl with only c1 should return x FAILED'

# ---- TC13: inits - 系数截断 ----
coeffs = np.array([1.0, 0.5, 1e-15, 0.0])
n = inits(coeffs, eta=1e-12)
assert n == 2, '[TC13] inits should return 2 for trailing small coeffs FAILED'

# ---- TC14: BoneDensityField.elastic_modulus_from_density - 单调性 ----
bdf = BoneDensityField(nx_cheb=4, ny_cheb=4)
E1 = bdf.elastic_modulus_from_density(0.3)
E2 = bdf.elastic_modulus_from_density(0.6)
assert E2 > E1, '[TC14] elastic modulus should be monotonic in density FAILED'
E0 = bdf.elastic_modulus_from_density(0.0)
assert E0 == 0.0, '[TC14] E(0) should be 0 FAILED'
E_max = bdf.elastic_modulus_from_density(1.0)
assert abs(E_max - 17000.0) < 1e-6, '[TC14] E(1) should be E0=17000 FAILED'

# ---- TC15: gegenbauer_rule - Legendre情形的权值和 ----
from quadrature_engine import gegenbauer_rule
x_q, w_q = gegenbauer_rule(n=5, lambda_param=0.5, a=0.0, b=1.0)
assert abs(np.sum(w_q) - 1.0) < 1e-12, '[TC15] Legendre weights on [0,1] should sum to 1 FAILED'
assert len(x_q) == 5, '[TC15] should have 5 nodes FAILED'

# ---- TC16: triangle_gauss_rule - 权值和为0.5 ----
from quadrature_engine import triangle_gauss_rule
x_tri, y_tri, w_tri = triangle_gauss_rule(order=3)
assert abs(np.sum(w_tri) - 0.5) < 1e-12, '[TC16] triangle weights should sum to 0.5 FAILED'
assert np.all(w_tri > 0), '[TC16] all triangle weights should be positive FAILED'

# ---- TC17: tetrahedron_unit_monomial_integral - 解析值验证 ----
from quadrature_engine import tetrahedron_unit_monomial_integral
val = tetrahedron_unit_monomial_integral(0, 0, 0, 0)
assert abs(val - 1.0/6.0) < 1e-15, '[TC17] integral of 1 over unit tetrahedron = 1/6 FAILED'
val_xy = tetrahedron_unit_monomial_integral(1, 1, 0, 0)
assert abs(val_xy - 1.0/120.0) < 1e-15, '[TC17] integral of x*y = 1/120 FAILED'

# ---- TC18: comp_next - 生成正确数量的组合 ----
from quadrature_engine import comp_next
a = np.zeros(3, dtype=int)
more, h, t = False, 0, 0
count = 0
for _ in range(50):
    a, more, h, t = comp_next(4, 3, a, more, h, t)
    count += 1
    if not more:
        break
assert count == 15, '[TC18] compositions of 4 into 3 parts should be 15 FAILED'

# ---- TC19: elastic_matrix_plane_stress - 属性验证 ----
from fem_core import elastic_matrix_plane_stress
D = elastic_matrix_plane_stress(E=200.0, nu=0.3)
assert D.shape == (3, 3), '[TC19] D matrix should be 3x3 FAILED'
assert D[0, 0] > D[0, 1], '[TC19] diagonal terms should dominate FAILED'
assert D[0, 0] > 0, '[TC19] D(0,0) should be positive FAILED'

# ---- TC20: t6_basis_functions - 单位分解性质 ----
from fem_core import t6_basis_functions
phi, _, _ = t6_basis_functions(0.3, 0.3)
assert abs(np.sum(phi) - 1.0) < 1e-12, '[TC20] partition of unity FAILED'
phi2, _, _ = t6_basis_functions(1.0/3.0, 1.0/3.0)
assert abs(np.sum(phi2) - 1.0) < 1e-12, '[TC20] partition of unity at centroid FAILED'

# ---- TC21: r8blt_sl - 求解对角占优下三角系统 ----
from fem_core import r8blt_sl
n, ml = 3, 2
a_band = np.array([[2.0, 3.0, 4.0], [0.5, 0.3, 0.0], [0.1, 0.0, 0.0]])
b = np.array([2.0, 1.0, 3.0])
x = r8blt_sl(n, ml, a_band, b)
assert abs(x[0] - 1.0) < 1e-12, '[TC21] x[0] should be 1.0 FAILED'

# ---- TC22: BoneRemodelingODE - 稳态密度在合法范围 ----
from bone_remodeling_ode import BoneRemodelingODE
ode = BoneRemodelingODE(k_form=0.05, k_res=0.03, U_ref=0.5, rho_min=0.01, rho_max=1.8)
rho_ss = ode.steady_state_density(U=1.0)
assert 0.01 <= rho_ss <= 1.8, '[TC22] steady state density in bounds FAILED'
rho_ss2 = ode.steady_state_density(U=0.0)
assert abs(rho_ss2 - 0.01) < 1e-12, '[TC22] rho_ss at U=0 should be rho_min FAILED'

# ---- TC23: BoneRemodelingODE - remodeling_rate符号正确性 ----
rate_form = ode.remodeling_rate(rho=0.5, U=1.0)
assert rate_form >= 0, '[TC23] high U should cause formation (positive rate) FAILED'
rate_res = ode.remodeling_rate(rho=0.5, U=0.1)
assert rate_res <= 0, '[TC23] low U should cause resorption (negative rate) FAILED'

# ---- TC24: BoneRemodelingODE - 精确解验证 ----
t_test = np.array([0.0, 10.0, 200.0])
rho_exact = ode.exact_solution_linear(t_test, rho0=1.0, A=0.5, B=0.1)
assert abs(rho_exact[0] - 1.0) < 1e-12, '[TC24] rho(0) should be initial value FAILED'
assert abs(rho_exact[-1] - 5.0) < 1e-6, '[TC24] rho(infty) should approach A/B=5.0 FAILED'

# ---- TC25: golden_section_search - 寻找已知极小值 ----
from parameter_optimization import golden_section_search, gradient_descent
def f_quad(x):
    return (x - 3.0) ** 2 + 2.0
a, b, it, nf = golden_section_search(f_quad, 0.0, 10.0, max_iter=50, x_tol=1e-10)
x_opt = (a + b) / 2.0
assert abs(x_opt - 3.0) < 1e-6, '[TC25] golden section should find minimum at x=3 FAILED'
assert it > 0, '[TC25] should take at least one iteration FAILED'

# ---- TC26: gradient_descent - 寻找已知极小值 ----
def fp_quad(x):
    return 2.0 * (x - 3.0)
x_gd, it_gd = gradient_descent(fp_quad, x0=0.0, gamma=0.1, max_iter=10000)
assert abs(x_gd - 3.0) < 1e-5, '[TC26] gradient descent should find minimum at x=3 FAILED'

# ---- TC27: matrix_exponential_pade - 零矩阵返回单位阵 ----
from numerical_diagnostics import matrix_exponential_pade, matrix_exponential_taylor
A_zero = np.zeros((2, 2))
E_zero = matrix_exponential_pade(A_zero, order=3)
assert np.allclose(E_zero, np.eye(2), atol=1e-14), '[TC27] exp(0) should be I FAILED'

# ---- TC28: matrix_exponential_pade vs Taylor - 小矩阵一致性 ----
A_small = np.array([[0.1, 0.02], [0.03, 0.15]])
E_pade = matrix_exponential_pade(A_small, order=3)
E_taylor = matrix_exponential_taylor(A_small, terms=50)
diff = np.linalg.norm(E_pade - E_taylor, ord='fro')
assert diff < 1e-5, '[TC28] Pade and Taylor should be consistent for small A FAILED'

# ---- TC29: polynomial_value_horner - 已知结果 ----
from numerical_diagnostics import polynomial_value_horner, polynomial_value_direct
# p(x) = 1 + 2x + 3x^2  =>  p(2) = 1 + 4 + 12 = 17
coeffs_poly = np.array([1.0, 2.0, 3.0])
val_horner = polynomial_value_horner(2.0, coeffs_poly)
assert abs(val_horner - 17.0) < 1e-12, '[TC29] Horner: p(2)=1+2*2+3*4=17 FAILED'
val_direct = polynomial_value_direct(2.0, coeffs_poly)
assert abs(val_direct - 17.0) < 1e-12, '[TC29] Direct: p(2)=1+2*2+3*4=17 FAILED'

# ---- TC30: quadratic_roots_stable - 病态问题精度更高 ----
from numerical_diagnostics import quadratic_roots_standard, quadratic_roots_stable
a, b, c = 1.0, -2.0, 1.0 - 1e-12
r1_std, r2_std = quadratic_roots_standard(a, b, c)
r1_stb, r2_stb = quadratic_roots_stable(a, b, c)
true_r1 = 1.0 + 1e-6
true_r2 = 1.0 - 1e-6
err_std = max(abs(r1_std - true_r1), abs(r2_std - true_r2))
err_stb = max(abs(r1_stb - true_r1), abs(r2_stb - true_r2))
assert err_stb < 1e-9, '[TC30] stable method should have high accuracy FAILED'

# ---- TC31: condition_number_analysis - 单位矩阵条件数 ----
from numerical_diagnostics import condition_number_analysis
I_mat = np.eye(3)
analysis = condition_number_analysis(I_mat)
assert analysis['rank'] == 3, '[TC31] identity matrix rank should be 3 FAILED'
assert analysis['cond_2'] < 2.0, '[TC31] identity condition number should be ~1 FAILED'
assert not analysis['is_singular'], '[TC31] identity should not be singular FAILED'

# ---- TC32: l2_error_estimate - 精确匹配时误差为0 ----
from quadrature_engine import l2_error_estimate
uh = np.array([1.0, 2.0, 3.0])
uexact = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, 0.3, 0.2])
areas = np.array([1.0, 1.0, 1.0])
err = l2_error_estimate(uh, uexact, weights, areas)
assert err < 1e-15, '[TC32] L2 error of exact match should be zero FAILED'

# ---- TC33: build_trabecular_field - 输出形状和范围 ----
from microstructure_model import build_trabecular_field
cortical_mask = np.array([True, True, False, False], dtype=bool)
density = build_trabecular_field(2, 2, cortical_mask, seed_offset=0)
assert len(density) == 4, '[TC33] density field length should be 4 FAILED'
assert density[0] > 0, '[TC33] cortical density should be positive FAILED'
assert np.all(density > 0), '[TC33] all densities should be positive FAILED'

# ---- TC34: BoneDensityField - evaluate_batch形状正确 ----
bdf = BoneDensityField(nx_cheb=4, ny_cheb=4)
xy_batch = np.array([[10.0, 5.0, 15.0], [15.0, 20.0, 10.0]])
vals = bdf.evaluate_batch(xy_batch)
assert vals.shape == (3,), '[TC34] batch evaluation shape should be (3,) FAILED'
assert np.all(np.isfinite(vals)), '[TC34] all batch values should be finite FAILED'

# ---- TC35: 可复现性 - 固定种子后两次构造结果相同 ----
np.random.seed(123)
micro1 = TrabecularMicrostructure(grid_size=15, pattern_seed=123)
np.random.seed(123)
micro2 = TrabecularMicrostructure(grid_size=15, pattern_seed=123)
assert abs(micro1.porosity - micro2.porosity) < 1e-15, '[TC35] reproducibility with fixed seed FAILED'
assert abs(micro1.effective_modulus - micro2.effective_modulus) < 1e-15, '[TC35] reproducibility of effective modulus FAILED'
