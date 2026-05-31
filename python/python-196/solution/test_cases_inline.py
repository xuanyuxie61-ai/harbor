# ---- TC01: hypersphere_surface_area dim=1 returns 2.0 ----
s1 = hypersphere_surface_area(1)
assert abs(s1 - 2.0) < 1e-12, '[TC01] dim=1 surface area should be 2.0 FAILED'

# ---- TC02: hypersphere_surface_area dim=2 equals 2*pi ----
s2 = hypersphere_surface_area(2)
assert abs(s2 - 2.0 * np.pi) < 1e-12, '[TC02] dim=2 surface area should be 2*pi FAILED'

# ---- TC03: hypersphere_surface_area dim=3 equals 4*pi ----
s3 = hypersphere_surface_area(3)
assert abs(s3 - 4.0 * np.pi) < 1e-12, '[TC03] dim=3 surface area should be 4*pi FAILED'

# ---- TC04: hypersphere_volume dim=2 equals pi ----
v2 = hypersphere_volume(2)
assert abs(v2 - np.pi) < 1e-12, '[TC04] dim=2 volume should be pi FAILED'

# ---- TC05: hypersphere_volume consistency V_m = S_m / m ----
for d in [2, 3, 4, 5]:
    sd = hypersphere_surface_area(d)
    vd = hypersphere_volume(d)
    assert abs(vd - sd / d) < 1e-12, f'[TC05] V_{d} = S_{d}/{d} FAILED'

# ---- TC06: check_positive_definite_symmetric on identity matrix ----
I2 = np.eye(2)
assert check_positive_definite_symmetric(I2), '[TC06] Identity should be SPD FAILED'

# ---- TC07: check_positive_definite_symmetric rejects non-symmetric matrix ----
ns = np.array([[1, 2], [3, 4]], dtype=float)
assert not check_positive_definite_symmetric(ns), '[TC07] Non-symmetric should be rejected FAILED'

# ---- TC08: rotation_matrix_2d orthogonality (R^T R = I) ----
R = rotation_matrix_2d(np.pi / 3.0)
assert np.allclose(R @ R.T, np.eye(2), atol=1e-14), '[TC08] Rotation matrix should be orthogonal FAILED'

# ---- TC09: rotation_matrix_2d determinant equals 1 ----
assert abs(np.linalg.det(R) - 1.0) < 1e-14, '[TC09] Rotation matrix det should be 1 FAILED'

# ---- TC10: dilation_matrix_2d returns correct scaling ----
D = dilation_matrix_2d(2.0, 3.0)
assert D[0, 0] == 2.0 and D[1, 1] == 3.0 and D[0, 1] == 0.0 and D[1, 0] == 0.0, '[TC10] Dilation matrix FAILED'

# ---- TC11: affine_transform_2d identity ----
import numpy as np
from mesh_transform import affine_transform_2d
pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
transformed = affine_transform_2d(pts)
assert np.allclose(transformed, pts, atol=1e-14), '[TC11] Identity affine transform FAILED'

# ---- TC12: alnorm_cdf at 0 returns 0.5 ----
assert abs(alnorm_cdf(0.0) - 0.5) < 1e-14, '[TC12] Phi(0) should be 0.5 FAILED'

# ---- TC13: alnorm_cdf symmetry Phi(-x) = 1 - Phi(x) ----
for x in [0.5, 1.0, 2.0, 3.0]:
    assert abs(alnorm_cdf(-x) - (1.0 - alnorm_cdf(x))) < 1e-14, f'[TC13] Phi(-{x}) symmetry FAILED'

# ---- TC14: alnorm_cdf upper tail ----
p_lower = alnorm_cdf(1.0, upper=False)
p_upper = alnorm_cdf(1.0, upper=True)
assert abs(p_lower + p_upper - 1.0) < 1e-14, '[TC14] Upper + lower tails should sum to 1 FAILED'

# ---- TC15: log_normal_pdf positive at x>0 ----
pdf_val = log_normal_pdf(np.array([1.0]), 0.0, 1.0)
assert pdf_val[0] > 0, '[TC15] LogNormal PDF at x=1 should be positive FAILED'

# ---- TC16: log_normal_pdf zero at x<=0 ----
pdf_zero = log_normal_pdf(np.array([0.0]), 0.0, 1.0)
assert pdf_zero[0] == 0.0, '[TC16] LogNormal PDF at x=0 should be 0 FAILED'

# ---- TC17: log_normal_mean formula ----
mu, sigma = 2.0, 0.5
expected_mean = np.exp(mu + 0.5 * sigma ** 2)
assert abs(log_normal_mean(mu, sigma) - expected_mean) < 1e-12, '[TC17] LogNormal mean formula FAILED'

# ---- TC18: log_normal_variance formula ----
expected_var = (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)
assert abs(log_normal_variance(mu, sigma) - expected_var) < 1e-12, '[TC18] LogNormal variance formula FAILED'

# ---- TC19: generate_task_set returns correct count with seed reproducibility ----
import numpy as np
tasks_a = generate_task_set(n_tasks=10, seed=42)
tasks_b = generate_task_set(n_tasks=10, seed=42)
assert len(tasks_a) == 10, '[TC19] Task set should have 10 tasks FAILED'
assert len(tasks_b) == 10, '[TC19] Task set B should have 10 tasks FAILED'
for i in range(10):
    assert abs(tasks_a[i].base_flops - tasks_b[i].base_flops) < 1e-9, f'[TC19] Task {i} reproducibility FAILED'

# ---- TC20: binomial_coeff C(5,2)=10 ----
from utils import binomial_coeff
assert binomial_coeff(5, 2) == 10, '[TC20] C(5,2) should be 10 FAILED'

# ---- TC21: binomial_coeff C(n,0)=1, C(n,n)=1 ----
assert binomial_coeff(10, 0) == 1, '[TC21] C(10,0) should be 1 FAILED'
assert binomial_coeff(10, 10) == 1, '[TC21] C(10,10) should be 1 FAILED'

# ---- TC22: vandermonde_quadrature_weights exact for polynomial degree < n ----
import numpy as np
n_q = 4
x_q = np.linspace(0.0, 1.0, n_q)
w_q = vandermonde_quadrature_weights(n_q, 0.0, 1.0, x_q)
# Test exact integration of x^k for k=0..n-1
for k in range(n_q):
    est = np.sum(w_q * (x_q ** k))
    true = 1.0 / (k + 1)
    assert abs(est - true) < 1e-12, f'[TC22] Vandermonde quadrature for x^{k} FAILED'

# ---- TC23: pyramid_monomial_integral zero for odd exponents ----
val_odd = pyramid_monomial_integral([1, 0, 0])
assert val_odd == 0.0, '[TC23] Odd exponent x integral should be 0 FAILED'

# ---- TC24: pyramid_monomial_integral for z^0 gives volume ----
vol_pyr = pyramid_monomial_integral([0, 0, 0])
assert abs(vol_pyr - pyramid_volume()) < 1e-12, '[TC24] Integral of 1 over pyramid should be volume FAILED'

# ---- TC25: pyramid_volume equals 4/3 ----
assert abs(pyramid_volume() - 4.0 / 3.0) < 1e-12, '[TC25] Pyramid volume should be 4/3 FAILED'

# ---- TC26: composite_quadrature_2d sin product integral ----
import numpy as np
f_sin = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
val_2d = composite_quadrature_2d(f_sin, 0.0, 1.0, 0.0, 1.0, 8, 8)
true_2d_ref = 4.0 / (np.pi ** 2)
assert abs(val_2d - true_2d_ref) < 1e-5, '[TC26] Composite 2D sin integral FAILED'

# ---- TC27: estimate_quadrature_error returns finite non-negative ----
err_est = estimate_quadrature_error(f_sin, 0.0, 1.0, 0.0, 1.0, 4, 8)
assert err_est >= 0.0, '[TC27] Quadrature error estimate should be non-negative FAILED'
assert np.isfinite(err_est), '[TC27] Quadrature error estimate should be finite FAILED'

# ---- TC28: ellipse_area known case ----
A_test = np.array([[2.0, 0.0], [0.0, 2.0]])
r_test = 2.0
area_test = ellipse_area(A_test, r_test)
expected_area = np.pi * r_test * r_test / np.sqrt(np.linalg.det(A_test))
assert abs(area_test - expected_area) < 1e-12, '[TC28] Ellipse area FAILED'

# ---- TC29: ellipse_sample points lie within ellipse ----
import numpy as np
np.random.seed(42)
A_ell = np.array([[4.0, 1.0], [1.0, 3.0]])
samples_ell = ellipse_sample(500, A_ell, 1.5, rng=np.random.default_rng(42))
for i in range(500):
    xv = samples_ell[:, i]
    qf = xv @ A_ell @ xv
    assert qf <= 1.5 ** 2 + 1e-10, f'[TC29] Point {i} outside ellipse FAILED'

# ---- TC30: hypersphere01_monomial_integral zero for odd exponent ----
val_odd_sph = hypersphere01_monomial_integral(3, [1, 0, 0])
assert val_odd_sph == 0.0, '[TC30] Odd monomial on sphere should integrate to 0 FAILED'

# ---- TC31: hypersphere_monte_carlo_integral of constant 1 gives area (reproducible) ----
import numpy as np
np.random.seed(42)
def ones_fn(x):
    return 1.0
val_sph, err_sph = hypersphere_monte_carlo_integral(3, 3000, ones_fn, rng=np.random.default_rng(42))
area_s3 = hypersphere_surface_area(3)
assert abs(val_sph - area_s3) < 0.15, f'[TC31] Sphere MC integral of 1 should be ~{area_s3:.4f} FAILED'

# ---- TC32: hypercube_distance_stats reproducibility ----
import numpy as np
np.random.seed(42)
mu1, var1 = hypercube_distance_stats(4, 2000, rng=np.random.default_rng(42))
np.random.seed(42)
mu2, var2 = hypercube_distance_stats(4, 2000, rng=np.random.default_rng(42))
assert abs(mu1 - mu2) < 1e-14, '[TC32] Hypercube distance stats mu reproducibility FAILED'
assert abs(var1 - var2) < 1e-14, '[TC32] Hypercube distance stats var reproducibility FAILED'

# ---- TC33: cheby_nodes lie within [a,b] ----
xn = cheby_nodes(-1.0, 1.0, 50)
assert np.all(xn >= -1.0) and np.all(xn <= 1.0), '[TC33] Cheby nodes should be within [-1,1] FAILED'

# ---- TC34: cheby_nodes endpoints for n=2 ----
xn2 = cheby_nodes(-1.0, 1.0, 2)
assert abs(max(xn2) - min(xn2)) > 0.0, '[TC34] Cheby nodes for n=2 should be distinct FAILED'

# ---- TC35: divided_differences + newton_interp_eval reproduce data points ----
import numpy as np
xd_t = np.array([0.0, 0.5, 1.0])
yd_t = np.array([1.0, 2.0, 5.0])
dd_t = divided_differences(xd_t, yd_t)
y_back = newton_interp_eval(xd_t, dd_t, xd_t)
assert np.allclose(y_back, yd_t, atol=1e-12), '[TC35] Newton interpolation should reproduce data FAILED'

# ---- TC36: PerformanceSurrogate chebyshev train and predict ----
import numpy as np
np.random.seed(42)
def f_test(x):
    return np.sin(2.0 * np.pi * x)
surr = PerformanceSurrogate(model_type='chebyshev')
surr.train((0.0, 1.0), f_test, n_nodes=15)
x_pred = np.array([0.25, 0.75])
y_pred = surr.predict(x_pred)
for i in range(len(x_pred)):
    assert np.isfinite(y_pred[i]), f'[TC36] Chebyshev surrogate prediction at x={x_pred[i]} FAILED'

# ---- TC37: PerformanceSurrogate least_squares train and predict ----
surr_lsq = PerformanceSurrogate(model_type='least_squares')
surr_lsq.train((0.0, 1.0), f_test, n_nodes=20, m_poly=8)
y_lsq_pred = surr_lsq.predict(x_pred)
for i in range(len(x_pred)):
    assert np.isfinite(y_lsq_pred[i]), f'[TC37] LSQ surrogate prediction at x={x_pred[i]} FAILED'

# ---- TC38: build_rectangular_mesh correct node count ----
nodes_m, elems_m = build_rectangular_mesh(5, 7)
assert nodes_m.shape == (2, 35), f'[TC38] Mesh should have 5*7=35 nodes, got {nodes_m.shape} FAILED'
assert elems_m.shape[1] == 2 * 4 * 6, f'[TC38] Mesh should have 2*4*6=48 elements FAILED'

# ---- TC39: polygon_surface_quality equilateral triangle quality ~1 ----
import numpy as np
eq_nodes = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, np.sqrt(3.0)/2.0]])
eq_elems = np.array([[1, 2, 3]]).T
q, qmin, qmean = polygon_surface_quality(eq_nodes, eq_elems)
assert abs(q[0] - 1.0) < 1e-10, f'[TC39] Equilateral triangle quality should be 1, got {q[0]} FAILED'

# ---- TC40: Processor effective_performance decreases with utilization ----
proc_test = Processor(0, 'CPU', peak_gflops=100.0, memory_bw_gb_s=10.0,
                      power_idle_w=10.0, power_peak_w=100.0,
                      thermal_resistance_k_w=0.5, position_xy=[0.0, 0.0])
proc_test.utilization = 0.0
perf0 = proc_test.effective_performance()
proc_test.utilization = 1.0
perf1 = proc_test.effective_performance()
assert perf1 < perf0, '[TC40] Effective performance should decrease with utilization FAILED'

# ---- TC41: HeterogeneousPlatform build_default_platform has 4 processors ----
platform_test = HeterogeneousPlatform(ambient_temp=300.0)
platform_test.build_default_platform()
assert len(platform_test.processors) == 4, f'[TC41] Default platform should have 4 processors FAILED'
types = [p.proc_type for p in platform_test.processors]
assert types.count('CPU') == 2, '[TC41] Should have 2 CPUs FAILED'
assert types.count('GPU') == 1, '[TC41] Should have 1 GPU FAILED'
assert types.count('FPGA') == 1, '[TC41] Should have 1 FPGA FAILED'

# ---- TC42: HeterogeneousPlatform comm_matrix symmetric ----
cm = platform_test.comm_latency_matrix
assert np.allclose(cm, cm.T, atol=1e-14), '[TC42] Communication matrix should be symmetric FAILED'

# ---- TC43: greedy_partition_load_balance balanced output ----
weights = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
assign_p, loads = greedy_partition_load_balance(weights, 2)
assert len(assign_p) == 5, '[TC43] Assignment should have 5 entries FAILED'
assert len(loads) == 2, '[TC43] Should have 2 bins FAILED'
assert abs(sum(loads) - sum(weights)) < 1e-12, '[TC43] Total load should be preserved FAILED'

# ---- TC44: reversi_greedy_move corner preference ----
import numpy as np
np.random.seed(42)
board = np.zeros((8, 8), dtype=int)
board[0, 0] = 2  # occupy one corner
move_vals = np.random.rand(8, 8)
i_m, j_m = reversi_greedy_move(board, 1, move_vals)
assert i_m == 0 and j_m == 7, f'[TC44] Should pick corner (0,7), got ({i_m},{j_m}) FAILED'

# ---- TC45: solve_task_mapping_ilp assigns each task exactly once ----
import numpy as np
np.random.seed(42)
cost_mat = np.random.rand(6, 3)
assign_ilp, cost_ilp = solve_task_mapping_ilp(6, 3, cost_mat, max_solutions=5)
assert assign_ilp is not None, '[TC45] ILP should return an assignment FAILED'
assert len(set(assign_ilp)) <= 3, '[TC45] Should use at most 3 processors FAILED'
for i in range(6):
    assert 0 <= assign_ilp[i] < 3, f'[TC45] Task {i} assignment out of range FAILED'

# ---- TC46: rref_matrix on identity returns identity ----
import numpy as np
I3 = np.eye(3, dtype=float)
rref_I, det_I = rref_matrix(I3)
assert np.allclose(rref_I, I3, atol=1e-12), '[TC46] RREF of identity should be identity FAILED'

# ---- TC47: rref_matrix on 2x2 singular matrix produces zero row ----
M_sing = np.array([[1.0, 2.0], [2.0, 4.0]])
rref_sing, det_sing = rref_matrix(M_sing)
assert np.allclose(rref_sing[1, :], 0.0, atol=1e-12) or np.allclose(rref_sing[0, :], 0.0, atol=1e-12), '[TC47] RREF of singular matrix should have a zero row FAILED'

# ---- TC48: fem2d_poisson_solve small mesh returns finite solution ----
import numpy as np
def src_small(x, y):
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
def ex_small(x, y):
    u = np.sin(np.pi * x) * np.sin(np.pi * y) + x
    dudx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 1.0
    dudy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return u, dudx, dudy
u_fem, nodes_fem, elems_fem, el2_fem, eh1_fem = fem2d_poisson_solve(
    5, 5, src_small, ex_small, xl=0.0, xr=1.0, yb=0.0, yt=1.0, conductivity=1.0
)
assert np.all(np.isfinite(u_fem)), '[TC48] FEM solution should be finite FAILED'
assert el2_fem >= 0.0, '[TC48] L2 error should be non-negative FAILED'
assert eh1_fem >= 0.0, '[TC48] H1 error should be non-negative FAILED'
assert u_fem.shape[0] == 25, f'[TC48] Solution should have 25 nodes FAILED'

# ---- TC49: extract_gradient_at_nodes returns finite gradients ----
grad = extract_gradient_at_nodes(u_fem, nodes_fem, elems_fem)
assert grad.shape == (2, 25), f'[TC49] Gradient shape should be (2, 25), got {grad.shape} FAILED'
assert np.all(np.isfinite(grad)), '[TC49] Gradient should be finite FAILED'

# ---- TC50: adaptive_refinement_markers returns correct shape ----
markers = adaptive_refinement_markers(nodes_fem, elems_fem, grad, threshold_ratio=0.3)
assert markers.shape[0] == elems_fem.shape[1], '[TC50] Marker count should match element count FAILED'
assert np.sum(markers) > 0, '[TC50] At least one element should be marked FAILED'

# ---- TC51: refine_marked_elements increases node count ----
new_nodes, new_elems = refine_marked_elements(nodes_fem, elems_fem, markers)
assert new_nodes.shape[1] > nodes_fem.shape[1], '[TC51] Refinement should increase node count FAILED'
assert new_elems.shape[1] > elems_fem.shape[1], '[TC51] Refinement should increase element count FAILED'

# ---- TC52: schedule_tasks_greedy produces valid schedule (integration) ----
import numpy as np
np.random.seed(196)
tasks_sched = generate_task_set(n_tasks=6, seed=196)
plat_sched = HeterogeneousPlatform(ambient_temp=300.0)
plat_sched.build_default_platform()
surr_sched = PerformanceSurrogate(model_type='chebyshev')
def dummy_perf(x):
    return 1.0 + 0.2 * np.sin(3.0 * x)
surr_sched.train((0.0, 1.0), dummy_perf, n_nodes=8)
sched, metrics_sched = schedule_tasks_greedy(
    tasks_sched, plat_sched, surrogate=surr_sched,
    alpha_makespan=0.6, alpha_energy=0.3, alpha_reliability=0.1
)
assert 'makespan' in metrics_sched, '[TC52] Metrics should contain makespan FAILED'
assert metrics_sched['makespan'] > 0, '[TC52] Makespan should be positive FAILED'
total_assigned = sum(len(v) for v in sched.values())
assert total_assigned == len(tasks_sched), f'[TC52] All {len(tasks_sched)} tasks should be scheduled FAILED'

# ---- TC53: antithetic_variates_integral bounded result (reproducible) ----
import numpy as np
np.random.seed(42)
def sq_sum(x):
    return np.sum(x ** 2)
res_av, err_av = antithetic_variates_integral(3, 500, sq_sum, rng=np.random.default_rng(42))
assert np.isfinite(res_av), '[TC53] Antithetic variates result should be finite FAILED'
assert err_av >= 0, '[TC53] Antithetic variates error should be non-negative FAILED'

# ---- TC54: uniform_in_sphere01_map all points within unit sphere ----
import numpy as np
np.random.seed(42)
pts_sph = uniform_in_sphere01_map(3, 100, rng=np.random.default_rng(42))
norms = np.linalg.norm(pts_sph, axis=0)
assert np.all(norms <= 1.0 + 1e-12), '[TC54] All points should be within unit sphere FAILED'

# ---- TC55: least_squares_fit residual non-negative ----
import numpy as np
np.random.seed(42)
xd_fit = np.linspace(0.0, 1.0, 10)
yd_fit = 2.0 * xd_fit + 1.0 + 0.01 * np.random.randn(10)
c_fit, res_fit = least_squares_fit(xd_fit, yd_fit, 3)
assert res_fit >= 0.0, '[TC55] LSQ residual should be non-negative FAILED'
assert c_fit.shape == (3,), f'[TC55] Coefficient shape should be (3,), got {c_fit.shape} FAILED'

# ---- TC56: poly_value evaluates correctly at endpoints ----
y0 = poly_value(c_fit, np.array([xd_fit[0]]))
yN = poly_value(c_fit, np.array([xd_fit[-1]]))
assert np.isfinite(y0[0]) and np.isfinite(yN[0]), '[TC56] Poly value should be finite at endpoints FAILED'
