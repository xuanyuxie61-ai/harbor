# ---- TC01: BandedMatrix 构造与索引存取 ----
import numpy as np
bm = BandedMatrix(5, 1, 1)
bm.set_entry(2, 2, 7.0)
assert bm.get_entry(2, 2) == 7.0, '[TC01] BandedMatrix set/get entry FAILED'
assert isinstance(bm.get_entry(0, 0), float), '[TC01] BandedMatrix entry type FAILED'

# ---- TC02: BandedMatrix PLU 分解与求解，残差为零 ----
import numpy as np
bm2 = BandedMatrix(8, 1, 1)
for i in range(8):
    bm2.set_entry(i, i, 2.0)
    if i > 0:
        bm2.set_entry(i, i-1, -1.0)
    if i < 7:
        bm2.set_entry(i, i+1, -1.0)
info = bm2.plu_factor()
assert info == 0, '[TC02] BandedMatrix PLU factorization FAILED'
b2 = np.ones(8)
x2 = bm2.solve(b2)
A_dense = np.zeros((8, 8))
for i in range(8):
    A_dense[i, i] = 2.0
    if i > 0:
        A_dense[i, i-1] = -1.0
    if i < 7:
        A_dense[i, i+1] = -1.0
res = np.linalg.norm(A_dense @ x2 - b2)
assert res < 3e-12, '[TC02] BandedMatrix solve residual FAILED'

# ---- TC03: BandedMatrix 行列式验证 ----
import numpy as np
bm3 = BandedMatrix(4, 1, 1)
for i in range(4):
    bm3.set_entry(i, i, 2.0)
    if i > 0:
        bm3.set_entry(i, i-1, -1.0)
    if i < 3:
        bm3.set_entry(i, i+1, -1.0)
bm3.plu_factor()
det_val = bm3.determinant()
A3 = np.zeros((4, 4))
for i in range(4):
    A3[i, i] = 2.0
    if i > 0:
        A3[i, i-1] = -1.0
    if i < 3:
        A3[i, i+1] = -1.0
expected_det = np.linalg.det(A3)
assert abs(det_val - expected_det) < 1e-10, '[TC03] BandedMatrix determinant FAILED'

# ---- TC04: SymmetricToeplitzSolver 基本求解 ----
import numpy as np
first_row = np.array([4.0, 1.0, 0.5, 0.0])
ts = SymmetricToeplitzSolver(first_row)
b_t = np.array([1.0, 0.0, 0.0, 0.0])
x_t = ts.solve_general(b_t)
recon = ts.matvec(x_t)
assert np.linalg.norm(recon - b_t) < 1e-12, '[TC04] SymmetricToeplitz solve FAILED'

# ---- TC05: build_tridiagonal_banded 构造验证 ----
import numpy as np
btb = build_tridiagonal_banded(6, -1.0, 2.0, -1.0)
assert btb.get_entry(0, 0) == 2.0, '[TC05] tridiagonal diag FAILED'
assert btb.get_entry(1, 0) == -1.0, '[TC05] tridiagonal lower FAILED'
assert btb.get_entry(0, 1) == -1.0, '[TC05] tridiagonal upper FAILED'

# ---- TC06: ocp_graphite 单调性验证 (sto 增大则电压降低) ----
ocp1 = ocp_graphite(0.2)
ocp2 = ocp_graphite(0.8)
assert ocp2 < ocp1, '[TC06] ocp_graphite monotonicity FAILED'
assert 0.0 < ocp1 < 1.5, '[TC06] ocp_graphite range FAILED'
assert 0.0 < ocp2 < 1.0, '[TC06] ocp_graphite range FAILED'

# ---- TC07: ocp_lco 单调性验证 (sto 增大则电压降低) ----
ocp_l1 = ocp_lco(0.3)
ocp_l2 = ocp_lco(0.9)
assert ocp_l1 > ocp_l2, '[TC07] ocp_lco monotonicity FAILED'
assert 3.5 < ocp_l1 < 4.5, '[TC07] ocp_lco range FAILED'
assert 3.5 < ocp_l2 < 4.5, '[TC07] ocp_lco range FAILED'

# ---- TC08: butler_volmer_flux 过电位为零时通量为零 ----
j0_test = exchange_current_density(1000.0, 25000.0, 30555.0, 1e-4, 298.15)
flux0 = butler_volmer_flux(0.0, j0_test, 298.15)
assert abs(flux0) < 1e-12, '[TC08] butler_volmer_flux zero eta FAILED'

# ---- TC09: exchange_current_density 正值与温度敏感性 ----
j0_low = exchange_current_density(1000.0, 20000.0, 30555.0, 1e-4, 298.15)
j0_high = exchange_current_density(1000.0, 20000.0, 30555.0, 1e-4, 318.15)
assert j0_low > 0, '[TC09] exchange_current_density positivity FAILED'
assert j0_high > j0_low, '[TC09] exchange_current_density temperature sensitivity FAILED'

# ---- TC10: gauss_legendre_nodes_weights 权重和为 2 ----
import numpy as np
nodes, wts = gauss_legendre_nodes_weights(10)
assert abs(np.sum(wts) - 2.0) < 1e-12, '[TC10] Gauss-Legendre weights sum FAILED'
assert len(nodes) == 10, '[TC10] Gauss-Legendre node count FAILED'
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC10] Gauss-Legendre node range FAILED'

# ---- TC11: log_gamma 已知解析值验证 ----
import numpy as np
lg5 = log_gamma(5.0)
assert abs(lg5 - np.log(24.0)) < 1e-6, '[TC11] log_gamma(5) FAILED'
lg1 = log_gamma(1.0)
assert abs(lg1 - 0.0) < 1e-12, '[TC11] log_gamma(1) FAILED'

# ---- TC12: incomplete_beta_ratio 边界值 ----
ib0 = incomplete_beta_ratio(0.0, 2.0, 3.0)
assert abs(ib0) < 1e-12, '[TC12] incomplete_beta_ratio at x=0 FAILED'
ib1 = incomplete_beta_ratio(1.0, 2.0, 3.0)
assert abs(ib1 - 1.0) < 1e-12, '[TC12] incomplete_beta_ratio at x=1 FAILED'

# ---- TC13: triangle_unit_rule 权重和为 0.5 ----
import numpy as np
xi, eta, w = triangle_unit_rule(order=3)
tri_sum = np.sum(w)
assert abs(tri_sum - 0.5) < 1e-12, '[TC13] triangle unit rule weight sum FAILED'
assert np.all(xi >= 0) and np.all(eta >= 0), '[TC13] triangle unit rule xi/eta positivity FAILED'
assert np.all(xi + eta <= 1.0), '[TC13] triangle unit rule xi+eta<=1 FAILED'

# ---- TC14: Catalan 数一致性与整型输出 ----
c0 = catalan_number(0)
c3 = catalan_number(3)
c5 = catalan_number(5)
assert isinstance(c0, int), '[TC14] catalan_number type FAILED'
assert c0 == 1, '[TC14] catalan_number(0) FAILED'
assert c3 == catalan_number(3), '[TC14] catalan_number reproducibility FAILED'
assert c5 == catalan_number(5), '[TC14] catalan_number reproducibility FAILED'

# ---- TC15: effective_diffusivity_bruggeman 范围检查 ----
d_eff = effective_diffusivity_bruggeman(0.4)
assert 0.0 < d_eff < 1.0, '[TC15] Bruggeman diffusivity range FAILED'
d_unit = effective_diffusivity_bruggeman(1.0)
assert abs(d_unit - 1.0) < 1e-12, '[TC15] Bruggeman diffusivity at epsilon=1 FAILED'

# ---- TC16: cubic_spline_coeffs 插值精度 ----
import numpy as np
x_sp = np.linspace(0.0, 1.0, 5)
y_sp = x_sp ** 3
a, b, c, d = cubic_spline_coeffs(x_sp, y_sp, bc_type="natural")
sp_val = a[1] + b[1] * (0.5 - x_sp[1]) + c[1] * (0.5 - x_sp[1])**2 + d[1] * (0.5 - x_sp[1])**3
assert abs(sp_val - 0.125) < 0.02, '[TC16] cubic_spline_coeffs accuracy FAILED'

# ---- TC17: lu_factor_dense + lu_solve_dense 线性方程组求解 ----
import numpy as np
A_lu = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b_lu = np.array([5.0, 5.0, 3.0])
P, LU = lu_factor_dense(A_lu.copy())
x_lu = lu_solve_dense(P, LU, b_lu)
assert np.linalg.norm(A_lu @ x_lu - b_lu) < 1e-12, '[TC17] LU factorization/solve FAILED'

# ---- TC18: muller_root 求 x^3-2x-5=0 的根 ----
import numpy as np
def poly_test(x):
    return x**3 - 2*x - 5
root = muller_root(poly_test, 1.0, 2.0, 3.0)
assert abs(poly_test(root)) < 1e-8, '[TC18] muller_root residual FAILED'
assert 2.0 < root < 2.5, '[TC18] muller_root value range FAILED'

# ---- TC19: rk2_integrate 指数衰减精度 ----
import numpy as np
def exp_decay(t, y):
    return np.array([-0.5 * y[0]])
t_rk, y_rk = rk2_integrate(exp_decay, np.array([1.0]), (0.0, 2.0), 100)
assert abs(y_rk[-1, 0] - np.exp(-1.0)) < 0.01, '[TC19] rk2_integrate accuracy FAILED'

# ---- TC20: cooley_tukey_fft 与 numpy.fft 比较 ----
import numpy as np
np.random.seed(42)
sig = np.random.randn(64)
fft_ours = cooley_tukey_fft(sig.copy())
fft_ref = np.fft.fft(sig)
assert np.linalg.norm(fft_ours - fft_ref) < 1e-10, '[TC20] cooley_tukey_fft accuracy FAILED'

# ---- TC21: compute_impedance_spectrum 基本输出 ----
import numpy as np
np.random.seed(42)
N_test = 128
t_test = np.linspace(0.0, 1.0, N_test)
dt_test = t_test[1] - t_test[0]
I_test = np.ones(N_test) + 0.1 * np.sin(2 * np.pi * 10.0 * t_test)
V_test = 4.0 + 0.05 * np.sin(2 * np.pi * 10.0 * t_test + 0.2)
freqs, Z = compute_impedance_spectrum(I_test, V_test, dt_test)
assert len(freqs) == len(Z), '[TC21] impedance spectrum length mismatch FAILED'
assert len(freqs) > 0, '[TC21] impedance spectrum empty FAILED'
assert np.all(np.isfinite(Z)), '[TC21] impedance spectrum finite FAILED'
assert np.all(freqs >= 0), '[TC21] impedance spectrum freq non-negative FAILED'

# ---- TC22: lognormal_psd 固定种子可复现性 ----
import numpy as np
np.random.seed(42)
r1 = lognormal_psd(100, mu_ln=-12.0, sigma_ln=0.5)
np.random.seed(42)
r2 = lognormal_psd(100, mu_ln=-12.0, sigma_ln=0.5)
assert np.allclose(r1, r2), '[TC22] lognormal_psd reproducibility FAILED'
assert np.all(r1 >= 1e-7), '[TC22] lognormal_psd r_min constraint FAILED'
assert np.all(r1 <= 20e-6), '[TC22] lognormal_psd r_max constraint FAILED'

# ---- TC23: cluster_particles 输出形状与标签 ----
import numpy as np
np.random.seed(42)
radii_test = lognormal_psd(200, mu_ln=-12.0, sigma_ln=0.5)
labels, centers = cluster_particles(radii_test, n_classes=4)
assert len(labels) == 200, '[TC23] cluster_particles label count FAILED'
assert len(centers) == 4, '[TC23] cluster_particles center count FAILED'
assert np.all((labels >= 0) & (labels < 4)), '[TC23] cluster_particles label range FAILED'
assert np.all(centers > 0), '[TC23] cluster_particles center positivity FAILED'

# ---- TC24: brownian_step_3d 固定种子的确定性输出与球反射 ----
import numpy as np
from stochastic_li_transport import brownian_step_3d, reflect_sphere
np.random.seed(42)
pos = np.array([0.0, 0.0, 0.0])
pos1 = brownian_step_3d(pos, 1e-4, 1e-12)
assert pos1.shape == (3,), '[TC24] brownian_step_3d shape FAILED'
assert np.any(np.abs(pos1 - pos) > 1e-12), '[TC24] brownian_step_3d static FAILED'
pos_out = np.array([1.5e-7, 0.0, 0.0])
pos_ref = reflect_sphere(pos_out, np.array([0.0, 0.0, 0.0]), 1e-7)
assert np.linalg.norm(pos_ref) <= 1e-7 + 1e-12, '[TC24] reflect_sphere FAILED'

# ---- TC25: feynman_kac_particle_diffusion 固定种子，输出类型与范围 ----
import numpy as np
np.random.seed(42)
mean_c, std_c = feynman_kac_particle_diffusion(
    radius=1e-7, D_s=1e-12, surface_concentration=25000.0,
    n_paths=200, dt=1e-3, t_max=0.05
)
assert isinstance(mean_c, float), '[TC25] Feynman-Kac mean type FAILED'
assert isinstance(std_c, float), '[TC25] Feynman-Kac std type FAILED'
assert mean_c > 0, '[TC25] Feynman-Kac mean positivity FAILED'
assert std_c >= 0, '[TC25] Feynman-Kac std non-negative FAILED'

# ---- TC26: first_passage_time_monte_carlo 固定种子，输出范围 ----
import numpy as np
np.random.seed(42)
mfpt, mfpt_std = first_passage_time_monte_carlo(
    radius=1e-7, D_s=1e-12, start_radius=0.0, n_paths=200, dt=1e-4
)
assert mfpt > 0, '[TC26] MFPT positivity FAILED'
assert mfpt_std >= 0, '[TC26] MFPT std non-negative FAILED'

# ---- TC27: stochastic_electrolyte_walk 固定种子，输出形状与范围 ----
import numpy as np
np.random.seed(42)
positions_test = stochastic_electrolyte_walk(
    n_particles=200, length=75e-6, D_e=4e-11,
    dt=2e-4, n_steps=300
)
assert len(positions_test) == 200, '[TC27] stochastic_electrolyte_walk count FAILED'
assert np.all((positions_test >= 0) & (positions_test <= 75e-6)), '[TC27] stochastic_electrolyte_walk range FAILED'

# ---- TC28: concentration_variance_from_walk 输出非负 ----
import numpy as np
np.random.seed(42)
pos_tmp = np.random.uniform(0, 75e-6, 500)
var_conc = concentration_variance_from_walk(pos_tmp, length=75e-6, n_bins=10)
assert var_conc >= 0, '[TC28] concentration_variance non-negative FAILED'

# ---- TC29: d_ocp_dT 熵系数符号验证 ----
from electrochemistry import d_ocp_dT_graphite, d_ocp_dT_lco
dTg = d_ocp_dT_graphite(0.5)
assert dTg < 0, '[TC29] d_ocp_dT_graphite sign FAILED'
dTl = d_ocp_dT_lco(0.5)
assert dTl > 0, '[TC29] d_ocp_dT_lco sign FAILED'

# ---- TC30: TemperatureDependentProperty 插值与单调性 ----
import numpy as np
temps = np.linspace(273.15, 350.0, 10)
vals = 1e-12 * np.exp(5000.0 / 8.314 * (1.0/298.15 - 1.0/temps))
prop = TemperatureDependentProperty(temps, vals)
v298 = prop.eval(298.15)
assert v298 > 0, '[TC30] TemperatureDependentProperty positivity FAILED'
v320 = prop.eval(320.0)
assert v320 > v298, '[TC30] TemperatureDependentProperty monotonicity FAILED'

# ---- TC31: psd_confidence_interval 输出顺序 ----
import numpy as np
np.random.seed(42)
r_test = lognormal_psd(100, mu_ln=-12.0, sigma_ln=0.2)
ci_lo, ci_hi = psd_confidence_interval(r_test, confidence=0.95)
assert ci_lo < ci_hi, '[TC31] psd_confidence_interval order FAILED'
assert ci_lo >= 1e-7, '[TC31] psd_confidence_interval lo bound FAILED'

# ---- TC32: Basis 函数 T3 在节点处插值 ----
from fem_assembler import basis_t3
N, _, _ = basis_t3(0.0, 0.0)
assert abs(N[0] - 1.0) < 1e-12, '[TC32] basis_t3 N1 at origin FAILED'
assert abs(N[1]) < 1e-12, '[TC32] basis_t3 N2 at origin FAILED'
assert abs(N[2]) < 1e-12, '[TC32] basis_t3 N3 at origin FAILED'
N2, _, _ = basis_t3(1.0, 0.0)
assert abs(N2[1] - 1.0) < 1e-12, '[TC32] basis_t3 N2 at (1,0) FAILED'
N3, _, _ = basis_t3(0.0, 1.0)
assert abs(N3[2] - 1.0) < 1e-12, '[TC32] basis_t3 N3 at (0,1) FAILED'

# ---- TC33: jacobian_t3 单位三角形面积 ----
import numpy as np
from fem_assembler import jacobian_t3
ref_tri_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
J, detJ = jacobian_t3(ref_tri_coords)
assert abs(detJ - 1.0) < 1e-12, '[TC33] jacobian_t3 det for unit triangle FAILED'

# ---- TC34: Polygon2D 面积与包含判断 ----
import numpy as np
sq = Polygon2D(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
assert abs(sq.area() - 1.0) < 1e-12, '[TC34] Polygon2D area FAILED'
assert sq.contains(0.5, 0.5), '[TC34] Polygon2D contains interior FAILED'
assert not sq.contains(-0.1, 0.5), '[TC34] Polygon2D contains exterior FAILED'
cx, cy = sq.centroid()
assert abs(cx - 0.5) < 1e-12, '[TC34] Polygon2D centroid x FAILED'
assert abs(cy - 0.5) < 1e-12, '[TC34] Polygon2D centroid y FAILED'

# ---- TC35: rotate_complex 旋转恒等 ----
import numpy as np
from geometry_engine import rotate_complex
z = np.array([1.0 + 0.0j, 0.0 + 1.0j])
z_rot = rotate_complex(z, np.pi / 2)
assert abs(z_rot[0].real) < 1e-12, '[TC35] rotate_complex real FAILED'
assert abs(z_rot[0].imag - 1.0) < 1e-12, '[TC35] rotate_complex imag FAILED'

# ---- TC36: greedy_thermal_protocol 输出非空 ----
import numpy as np
greedy_I, greedy_t = greedy_thermal_protocol(
    target_capacity=1500.0,
    current_options=np.array([5.0, 10.0, 20.0, 30.0, 50.0]),
    duration_step=30.0,
    thermal_model_func=None,
    max_temp=318.15
)
assert len(greedy_I) > 0, '[TC36] greedy_thermal_protocol empty FAILED'
assert len(greedy_t) == len(greedy_I), '[TC36] greedy_thermal_protocol length mismatch FAILED'
assert np.all(np.array(greedy_I) > 0), '[TC36] greedy_thermal_protocol current positivity FAILED'
assert np.all(np.array(greedy_t) > 0), '[TC36] greedy_thermal_protocol duration positivity FAILED'

# ---- TC37: cluster_protocol_segments 输出形状 ----
import numpy as np
cur_test = np.array([5.0, 10.0, 10.0, 20.0, 20.0, 20.0, 5.0])
seg_labels, seg_centers = cluster_protocol_segments(cur_test, n_segments=3)
assert len(seg_labels) == len(cur_test), '[TC37] cluster_protocol_segments label count FAILED'
assert len(seg_centers) == 3, '[TC37] cluster_protocol_segments center count FAILED'

# ---- TC38: beta_mixture_psd 固定种子输出范围 ----
import numpy as np
np.random.seed(42)
bm_radii = beta_mixture_psd(50, 2.0, 5.0, 5.0, 2.0, mix=0.5)
assert len(bm_radii) == 50, '[TC38] beta_mixture_psd length FAILED'
assert np.all(bm_radii >= 1e-7), '[TC38] beta_mixture_psd r_min FAILED'
assert np.all(bm_radii <= 20e-6), '[TC38] beta_mixture_psd r_max FAILED'

# ---- TC39: battery_cell_geometry 区域分类 ----
geo_test = BatteryCellGeometry(
    total_width=1.0, total_height=0.5,
    neg_cc_width=0.05, neg_elec_width=0.30,
    sep_width=0.05, pos_elec_width=0.30,
    pos_cc_width=0.05
)
assert geo_test.classify_point(0.025, 0.25) == "neg_cc", '[TC39] classify neg_cc FAILED'
assert geo_test.classify_point(0.2, 0.25) == "neg_elec", '[TC39] classify neg_elec FAILED'
assert geo_test.classify_point(0.375, 0.25) == "separator", '[TC39] classify separator FAILED'
assert geo_test.classify_point(0.55, 0.25) == "pos_elec", '[TC39] classify pos_elec FAILED'
assert geo_test.classify_point(0.72, 0.25) == "pos_cc", '[TC39] classify pos_cc FAILED'
regions = geo_test.get_all_regions()
assert len(regions) >= 5, '[TC39] get_all_regions count FAILED'

# ---- TC40: mesh generator 结构化三角网格输出形状 ----
import numpy as np
nx_m, ny_m = 10, 5
geo_m = BatteryCellGeometry(
    total_width=1.0, total_height=0.5,
    neg_cc_width=0.05, neg_elec_width=0.30,
    sep_width=0.05, pos_elec_width=0.30,
    pos_cc_width=0.05
)
nodes_m, elements_m, region_tags_m = generate_structured_triangle_mesh(nx_m, ny_m, geo_m)
assert nodes_m.ndim == 2 and nodes_m.shape[1] == 2, '[TC40] mesh nodes shape FAILED'
assert elements_m.ndim == 2 and elements_m.shape[1] == 3, '[TC40] mesh elements shape FAILED'
assert len(region_tags_m) == len(elements_m), '[TC40] mesh region_tags length FAILED'
boundary_m = build_boundary_mask(nodes_m, elements_m)
assert boundary_m.ndim == 1, '[TC40] boundary_mask dimension FAILED'
quality_m = compute_element_quality(nodes_m, elements_m)
assert np.all((quality_m >= 0.0) & (quality_m <= 1.0)), '[TC40] element quality range FAILED'

# ---- TC41: compute_heat_generation 输出非负 ----
import numpy as np
tags_test = np.array([0, 0, 1, 2, 3, 3, 4])
overpot_test = np.full(7, 0.03)
flux_test = np.full(7, 5.0)
Q_test = compute_heat_generation(tags_test, 30.0, overpot_test, flux_test)
assert len(Q_test) == 7, '[TC41] compute_heat_generation length FAILED'
assert np.all(np.isfinite(Q_test)), '[TC41] compute_heat_generation finite FAILED'

# ---- TC42: fibonacci_seeding_2d 输出形状与包围盒 ----
import numpy as np
pts_fib = fibonacci_seeding_2d(50, (0.0, 1.0, 0.0, 0.5))
assert len(pts_fib) == 50, '[TC42] fibonacci_seeding_2d count FAILED'
assert pts_fib.ndim == 2 and pts_fib.shape[1] == 2, '[TC42] fibonacci_seeding_2d shape FAILED'
assert np.all(pts_fib[:, 0] >= 0.0) and np.all(pts_fib[:, 0] <= 1.0), '[TC42] fibonacci x range FAILED'
assert np.all(pts_fib[:, 1] >= 0.0) and np.all(pts_fib[:, 1] <= 0.5), '[TC42] fibonacci y range FAILED'


