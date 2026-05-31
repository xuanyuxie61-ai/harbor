# ---- TC01: RMSD 零值 - 相同坐标返回 0 ----
import numpy as np
np.random.seed(42)
test_coords = np.random.randn(10, 3)
result = compute_rmsd(test_coords, test_coords)
assert abs(result) < 1e-14, '[TC01] RMSD of identical coords should be 0 FAILED'

# ---- TC02: RMSD 非负性 ----
import numpy as np
np.random.seed(43)
c1 = np.random.randn(10, 3)
c2 = c1 + np.random.randn(10, 3) * 0.5
result = compute_rmsd(c1, c2)
assert result >= 0.0, '[TC02] RMSD should be non-negative FAILED'

# ---- TC03: RMSD 交换对称性 ----
import numpy as np
np.random.seed(44)
c1 = np.random.randn(10, 3)
c2 = c1 + np.random.randn(10, 3) * 0.5
r1 = compute_rmsd(c1, c2)
r2 = compute_rmsd(c2, c1)
assert abs(r1 - r2) < 1e-14, '[TC03] RMSD symmetry FAILED'

# ---- TC04: 回转半径正值 ----
import numpy as np
np.random.seed(45)
coords = np.random.randn(15, 3)
rg = compute_radius_of_gyration(coords)
assert rg > 0.0, '[TC04] Radius of gyration should be positive FAILED'

# ---- TC05: 天然接触分数范围 [0,1] ----
import numpy as np
np.random.seed(46)
n_coords = np.random.randn(12, 3)
c_coords = n_coords + np.random.randn(12, 3) * 2.0
q_val = compute_native_contact_fraction(c_coords, n_coords, contact_cutoff=1.0, native_cutoff=1.5)
assert 0.0 <= q_val <= 1.0, '[TC05] Q should be in [0,1] FAILED'

# ---- TC06: 端到端距离非负 ----
import numpy as np
np.random.seed(47)
coords = np.random.randn(10, 3)
ee = compute_end_to_end_distance(coords)
assert ee >= 0.0, '[TC06] End-to-end distance should be non-negative FAILED'

# ---- TC07: 二面角函数可用性检查（用端到端距离替代） ----
import numpy as np
np.random.seed(48)
coords = np.array([[0.,0.,0.],[1.,0.,0.],[2.,1.,0.],[3.,1.,1.]])
ee_val = compute_end_to_end_distance(coords)
d = np.linalg.norm(coords[-1] - coords[0])
assert abs(ee_val - d) < 1e-14, '[TC07] End-to-end should equal |r_N - r_1| FAILED'

# ---- TC08: Chebyshev 插值一致性（使用已有函数） ----
import numpy as np
np.random.seed(49)
n_order = 12
a_dom, b_dom = -1.0, 1.0
x_nodes = np.cos(np.pi * (2.0 * np.arange(1, n_order + 1) - 1.0) / (2.0 * n_order))
f_vals = np.sin(np.pi * x_nodes)
from chebyshev_pes import chebyshev_coefficients, chebyshev_interpolant
coeffs = chebyshev_coefficients(a_dom, b_dom, n_order, lambda x: np.sin(np.pi * x))
xq = np.linspace(a_dom, b_dom, 100)
feval = chebyshev_interpolant(a_dom, b_dom, n_order, coeffs, xq)
max_err = np.max(np.abs(feval - np.sin(np.pi * xq)))
assert max_err < 1e-6, '[TC08] Chebyshev interpolation accuracy FAILED'

# ---- TC09: Chebyshev 导数有限性 ----
import numpy as np
np.random.seed(50)
from chebyshev_pes import chebyshev_coefficients, chebyshev_derivative
n_order = 16
coeffs = chebyshev_coefficients(-2.0, 2.0, n_order, lambda x: x**3 - 3.0*x)
xq = np.linspace(-2.0, 2.0, 80)
deriv = chebyshev_derivative(-2.0, 2.0, n_order, coeffs, xq)
assert np.all(np.isfinite(deriv)), '[TC09] Chebyshev derivative should be finite FAILED'

# ---- TC10: Legendre 积分节点/权重（通过 FEM solver） ----
import numpy as np
from fem1d_pmethod_solver import legendre_com
nodes, weights = legendre_com(10)
assert abs(np.sum(weights) - 2.0) < 1e-12, '[TC10] Legendre weights should sum to 2 FAILED'

# ---- TC11: Legendre 积分节点范围 [-1,1] ----
import numpy as np
from fem1d_pmethod_solver import legendre_com
nodes, weights = legendre_com(15)
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC11] Legendre nodes should be in [-1,1] FAILED'

# ---- TC12: 球面单项式积分解析值 ----
import numpy as np
val = sphere01_monomial_integral(0, 0, 0)
assert abs(val - 4.0 * np.pi) < 1e-10, '[TC12] Sphere integral of 1 should be 4π FAILED'

# ---- TC13: 球面奇次幂积分为 0 ----
val = sphere01_monomial_integral(1, 0, 0)
assert abs(val) < 1e-14, '[TC13] Odd exponent integral should be 0 FAILED'

# ---- TC14: 二十面体形状检查 ----
from sphere_quad import icosahedron_shape
verts, faces = icosahedron_shape()
assert verts.shape[0] == 12, '[TC14] Icosahedron should have 12 vertices FAILED'
assert faces.shape[0] >= 1, '[TC14] Icosahedron should have faces FAILED'

# ---- TC15: 球面求积权重和为 4π ----
import numpy as np
pts, w = sphere01_quad_icos1c(n_subdivide=2)
assert abs(np.sum(w) - 4.0 * np.pi) < 1e-10, '[TC15] Sphere quad weights should sum to 4π FAILED'

# ---- TC16: 球面 MC 积分 seed 可复现 ----
import numpy as np
pts1, w1 = sphere01_quad_mc(n_samples=5000, seed=42)
pts2, w2 = sphere01_quad_mc(n_samples=5000, seed=42)
assert np.allclose(pts1, pts2), '[TC16] MC sphere quad should be reproducible with same seed FAILED'

# ---- TC17: 立方体求积权重和为体积 ----
import numpy as np
nodes, wts = cube_rule(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, order_1d=3)
assert abs(np.sum(wts) - 1.0) < 1e-12, '[TC17] Cube quad weights should sum to volume=1 FAILED'

# ---- TC18: 立方积分解析精度 (阶数为3规则) ----
err_dict = test_cube_rule_precision(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, max_degree=2)
max_e = max(err_dict.values())
assert max_e < 1e-10, '[TC18] Cube rule should be exact for degree <= 2 FAILED'

# ---- TC19: Sylvester 结式确定性 ----
import numpy as np
p = np.array([1.0, 0.0, -2.0, 0.0])
q = np.array([1.0, -1.0])
res = polynomial_resultant_sylvester(p, q)
assert abs(res) > 1e-10, '[TC19] Sylvester resultant of x^3-2x and x-1 should be non-zero FAILED'

# ---- TC20: 临界点检测返回结构 ----
import numpy as np
poly = np.array([1.0, 0.0, -3.0, 0.0, 0.0])
crit = analyze_potential_landscape_criticality(poly)
assert 'critical_points' in crit, '[TC20] Critical point analysis should return dict FAILED'

# ---- TC21: Sigmoid 截断范围 [0,1] ----
import numpy as np
r_test = np.linspace(0.0, 5.0, 50)
S = smooth_cutoff_function(r_test, r_cut=2.5, width=0.5)
assert np.all(S >= 0.0) and np.all(S <= 1.0), '[TC21] Sigmoid cutoff should be in [0,1] FAILED'

# ---- TC22: Sigmoid 截断单调性 ----
import numpy as np
r_test = np.linspace(0.0, 5.0, 50)
S = smooth_cutoff_function(r_test, r_cut=2.5, width=0.5)
assert np.all(np.diff(S) <= 1e-12), '[TC22] Sigmoid cutoff should be non-increasing FAILED'

# ---- TC23: 介电函数范围检查 ----
import numpy as np
r_test = np.linspace(0.5, 4.0, 50)
eps = dielectric_switch_function(r_test, r_in=1.5, r_out=3.0, eps_in=4.0, eps_out=80.0)
assert np.all(eps >= 4.0 - 1e-10) and np.all(eps <= 80.0 + 1e-10), '[TC23] Dielectric function range FAILED'

# ---- TC24: 弹性网络矩阵对称性 ----
import numpy as np
np.random.seed(51)
coords_enm = np.random.randn(10, 3)
gamma = build_elastic_network_matrix(coords_enm, cutoff=2.0, spring_constant=1.0)
assert np.allclose(gamma, gamma.T), '[TC24] ENM matrix should be symmetric FAILED'

# ---- TC25: ENM 矩阵行和为0 ----
import numpy as np
np.random.seed(52)
coords_enm = np.random.randn(8, 3)
gamma = build_elastic_network_matrix(coords_enm, cutoff=2.0, spring_constant=1.0)
row_sums = np.sum(gamma, axis=1)
assert np.all(np.abs(row_sums) < 1e-12), '[TC25] ENM matrix row sums should be 0 FAILED'

# ---- TC26: R8SS 矩阵向量乘等价性 ----
import numpy as np
np.random.seed(53)
dense_mat = np.random.randn(6, 6)
dense_mat = dense_mat @ dense_mat.T
na, diag, a_r8ss = r8ss_from_dense(dense_mat)
x_vec = np.ones(6)
b_skyline = r8ss_mv(6, na, diag, a_r8ss, x_vec)
b_direct = dense_mat @ x_vec
assert np.allclose(b_skyline, b_direct), '[TC26] R8SS mat-vec should equal dense mat-vec FAILED'

# ---- TC27: R8SS 往返转换 ----
import numpy as np
np.random.seed(54)
from sparse_hessian import r8ss_to_r8ge
dense_mat = np.random.randn(6, 6)
dense_mat = dense_mat @ dense_mat.T
na, diag, a_r8ss = r8ss_from_dense(dense_mat)
dense_recovered = r8ss_to_r8ge(6, na, diag, a_r8ss)
assert np.allclose(dense_recovered, dense_mat), '[TC27] R8SS roundtrip FAILED'

# ---- TC28: MSF 非负 ----
import numpy as np
np.random.seed(55)
coords_enm = np.random.randn(8, 3)
gamma = build_elastic_network_matrix(coords_enm, cutoff=2.0, spring_constant=1.0)
msf = compute_mean_square_fluctuation(gamma, kT=1.0)
assert np.all(msf >= 0.0), '[TC28] MSF should be non-negative FAILED'

# ---- TC29: 正常模式特征值非负 ----
import numpy as np
np.random.seed(56)
coords_enm = np.random.randn(10, 3)
gamma = build_elastic_network_matrix(coords_enm, cutoff=2.0, spring_constant=1.0)
eigvals_nma, _ = normal_mode_analysis(gamma, n_modes=6)
assert np.all(eigvals_nma >= -1e-12), '[TC29] NMA eigenvalues should be non-negative FAILED'

# ---- TC30: Gamma 不完全函数范围 [0,1] ----
val_gam, fault = gammds(2.0, 3.0)
assert 0.0 <= val_gam <= 1.0, '[TC30] Normalized incomplete gamma should be in [0,1] FAILED'

# ---- TC31: Gamma CDF 单调性 ----
import numpy as np
x_vals = np.linspace(0.1, 10.0, 50)
cdf_vals = gamma_cdf(x_vals, shape=2.5, scale=1.2)
assert np.all(np.diff(cdf_vals) >= -1e-14), '[TC31] Gamma CDF should be non-decreasing FAILED'

# ---- TC32: 贪心划分覆盖所有元素 ----
import numpy as np
np.random.seed(57)
workloads = np.array([100, 85, 120, 95, 110, 75, 130, 90])
part = partition_greedy(workloads)
s0 = np.sum(workloads[part == 0])
s1 = np.sum(workloads[part == 1])
assert abs(s0 + s1 - np.sum(workloads)) < 1e-10, '[TC32] Partition should cover all items FAILED'

# ---- TC33: 自由能景观划分 bin 数 ----
import numpy as np
energies_test = np.linspace(0.0, 10.0, 1000)
ranges = partition_free_energy_landscape(energies_test, n_bins=4)
assert len(ranges) == 4, '[TC33] Should have 4 energy bins FAILED'

# ---- TC34: 四面体边界计数非负 ----
import numpy as np
tet_elements = np.array([[0,1,2,3],[4,5,6,7],[0,1,2,4]], dtype=int)
n_bn, n_bf, _ = tet_mesh_boundary_count(tet_elements)
assert n_bn >= 0 and n_bf >= 0, '[TC34] Boundary counts should be non-negative FAILED'

# ---- TC35: 力切换函数范围 [0,1] ----
import numpy as np
r_test = np.linspace(1.0, 4.0, 50)
S_switch, dS = force_switching(r_test, r_on=1.5, r_off=3.0)
assert np.all(S_switch >= 0.0) and np.all(S_switch <= 1.0), '[TC35] Force switching should be in [0,1] FAILED'

# ---- TC36: 临界点类型检查 ----
import numpy as np
poly = np.poly1d([1.0, 0.0, -2.0, 0.0, 0.1])
crit = analyze_potential_landscape_criticality(poly.coeffs)
for cp_type in crit['types']:
    assert cp_type in ('minimum', 'maximum', 'degenerate'), '[TC36] Invalid critical point type FAILED'

# ---- TC37: 停留时间统计输出结构 ----
import numpy as np
np.random.seed(60)
dwell = np.random.default_rng(60).gamma(shape=2.5, scale=1.2, size=200)
stats = metastable_state_residence_time_distribution(dwell)
assert 'mean' in stats and 'gamma_shape' in stats and 'half_life' in stats, '[TC37] Residence stats should have required keys FAILED'

# ---- TC38: 卡方 p 值范围 ----
pval = chi_square_pvalue(5.2, 3)
assert 0.0 <= pval <= 1.0, '[TC38] Chi-square p-value should be in [0,1] FAILED'

# ---- TC39: 多项式交点检测 ----
import numpy as np
poly1 = np.array([1.0, 0.0, -2.0])
poly2 = np.array([1.0, -1.0, -1.0])
intersections = detect_bifurcation_points(poly1, poly2, x_range=(-2.0, 2.0))
for x_pt in intersections:
    val1 = np.polyval(poly1, x_pt)
    val2 = np.polyval(poly2, x_pt)
    assert abs(val1 - val2) < 1e-8, '[TC39] Intersection point values should be equal FAILED'

# ---- TC40: Sigmoid 高阶导数无 NaN ----
import numpy as np
r_test = np.array([1.0, 2.0, 3.0])
d2 = smooth_cutoff_derivative(r_test, r_cut=2.5, width=0.5, order=2)
assert np.all(np.isfinite(d2)), '[TC40] Sigmoid 2nd derivative should be finite FAILED'

# ---- TC41: 四面体三维表面积非负 ----
import numpy as np
np.random.seed(61)
pts = np.random.randn(15, 3) * 2.0
tet_elements = np.array([[0,1,2,3],[4,5,6,7],[0,1,4,5]], dtype=int)
boundary_faces = tet_mesh_boundary_set(tet_elements)
area, volume = compute_surface_area_and_volume(pts, boundary_faces)
assert area >= 0.0, '[TC41] Surface area should be non-negative FAILED'

# ---- TC42: Kramers 速率近似正值 ----
rate = kramers_rate_approximation(barrier_height=5.0, kT=1.0, D=0.5, curvature_top=-2.0, curvature_bottom=3.0)
assert rate > 0.0, '[TC42] Kramers rate should be positive FAILED'

# ---- TC43: NMR 序参数范围 [0,1] ----
import numpy as np
np.random.seed(62)
vec = np.random.randn(3)
vec = vec / np.linalg.norm(vec)
s2 = compute_nmr_order_parameter(vec, n_subdivide=2)
assert 0.0 <= s2 <= 1.0, '[TC43] NMR order parameter S^2 should be in [0,1] FAILED'

# ---- TC44: 反应坐标网格输出维度 ----
import numpy as np
grid_nodes, grid_elements = generate_reaction_coordinate_grid(0.0, 1.0, 5, 0.0, 2.0, 5)
assert grid_nodes.shape[1] == 2, '[TC44] Grid nodes should have 2 columns FAILED'
assert grid_nodes.shape[0] == 25, '[TC44] Grid nodes count should be 5*5=25 FAILED'

# ---- TC45: 稳定性: Chebyshev 高次导数有界 ----
import numpy as np
from chebyshev_pes import chebyshev_coefficients, chebyshev_derivative
n_order = 30
coeffs = chebyshev_coefficients(-3.0, 3.0, n_order, lambda x: np.exp(-x**2))
xq = np.linspace(-3.0, 3.0, 200)
deriv = chebyshev_derivative(-3.0, 3.0, n_order, coeffs, xq)
assert np.all(np.abs(deriv) < 100.0), '[TC45] Chebyshev derivative should be bounded FAILED'

# ---- TC46: 自由能剖面拟合可运行 ----
import numpy as np
np.random.seed(63)
x_vals = np.linspace(0.1, 0.9, 15)
fe_vals = 2.0 * (x_vals - 0.5)**2 + 0.1 * np.sin(10.0 * x_vals)
a_ch, b_ch, coeffs_ch = fit_free_energy_profile(x_vals, fe_vals, order=12)
assert a_ch < b_ch, '[TC46] Domain should have a < b FAILED'
assert len(coeffs_ch) == 12, '[TC46] Should have 12 Chebyshev coefficients FAILED'

# ---- TC47: 多项式结式根的等价值 ----
import numpy as np
from polynomial_analysis import polynomial_resultant_roots
p = np.array([1.0, -5.0, 6.0])
q = np.array([1.0, -3.0, 2.0])
r1 = polynomial_resultant_sylvester(p, q)
r2 = polynomial_resultant_roots(p, q)
assert abs(r1 - r2) < 1e-8, '[TC47] Sylvester and root-product resultants should match FAILED'

# ---- TC48: 球面积分 x^2 解析值 ----
val_x2 = sphere01_monomial_integral(2, 0, 0)
assert abs(val_x2 - 4.0 * np.pi / 3.0) < 1e-10, '[TC48] ∫x² on S² should be 4π/3 FAILED'

# ---- TC49: distmesh simpqual 范围 [0,1] ----
import numpy as np
p_test = np.array([[0.,0.],[1.,0.],[0.5,0.866]], dtype=np.float64)
t_test = np.array([[0,1,2]], dtype=np.int32)
qual = simpqual(p_test, t_test)
assert np.all(qual >= 0.0) and np.all(qual <= 1.0), '[TC49] Simpqual should be in [0,1] FAILED'

# ---- TC50: Sylvester 矩阵维度正确 ----
import numpy as np
p = np.array([1.0, -3.0, 2.0])
q = np.array([1.0, -1.0])
S = sylvester_matrix(p, q)
assert S.shape == (3, 3), '[TC50] Sylvester matrix of deg2 and deg1 should be 3x3 FAILED'
