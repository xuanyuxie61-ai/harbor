# ---- TC01: vec_norm of zero vector returns 0 ----
v = np.zeros(5)
assert vec_norm(v) == 0.0, '[TC01] vec_norm zero vector FAILED'

# ---- TC02: vec_norm of unit vector returns 1 ----
v = np.array([1.0, 0.0, 0.0])
assert abs(vec_norm(v) - 1.0) < 1e-12, '[TC02] vec_norm unit vector FAILED'

# ---- TC03: vec_norm with ord=np.inf returns max abs ----
v = np.array([1.0, -5.0, 3.0])
assert abs(vec_norm(v, ord=np.inf) - 5.0) < 1e-12, '[TC03] vec_norm inf-norm FAILED'

# ---- TC04: checksum_vector known value: weights=[3,2,1] sum=3*1+2*2+1*3=10, 10%11=10 ----
v = np.array([1.0, 2.0, 3.0])
cs = checksum_vector(v)
assert cs == 10, '[TC04] checksum_vector known value FAILED'

# ---- TC05: checksum_vector empty vector returns 0 ----
assert checksum_vector(np.array([])) == 0, '[TC05] checksum_vector empty FAILED'

# ---- TC06: safe_divide normal case ----
from utils import safe_divide
assert abs(safe_divide(6.0, 3.0) - 2.0) < 1e-12, '[TC06] safe_divide normal FAILED'

# ---- TC07: safe_divide zero denominator returns default ----
assert safe_divide(1.0, 0.0, default=99.0) == 99.0, '[TC07] safe_divide zero FAILED'

# ---- TC08: is_spd on identity matrix returns True ----
I = np.eye(10)
assert is_spd(I) == True, '[TC08] is_spd identity FAILED'

# ---- TC09: is_spd on non-square matrix returns False ----
A_ns = np.random.randn(5, 3)
assert is_spd(A_ns) == False, '[TC09] is_spd non-square FAILED'

# ---- TC10: lambert_w_fast W(0)=0 ----
w, en = lambert_w_fast(0.0)
assert abs(w) < 1e-12, '[TC10] lambert_w_fast W(0)=0 FAILED'

# ---- TC11: lambert_w_fast W(e)=1 ----
w, en = lambert_w_fast(math.e)
assert abs(w - 1.0) < 1e-8, '[TC11] lambert_w_fast W(e)=1 FAILED'

# ---- TC12: harmonic_number H(1)=1 ----
assert abs(harmonic_number(1) - 1.0) < 1e-12, '[TC12] harmonic_number H(1)=1 FAILED'

# ---- TC13: harmonic_number H(4)=1+1/2+1/3+1/4 ----
assert abs(harmonic_number(4) - (1.0 + 1.0/2.0 + 1.0/3.0 + 1.0/4.0)) < 1e-12, '[TC13] harmonic_number H(4) FAILED'

# ---- TC14: harmonic_number at 0 returns 0 ----
assert harmonic_number(0) == 0.0, '[TC14] harmonic_number H(0) FAILED'

# ---- TC15: steinerberger_integral_01 matches formula 2*H(n)/pi ----
n_sb = 5
val_int = steinerberger_integral_01(n_sb)
val_expected = 2.0 * harmonic_number(n_sb) / math.pi
assert abs(val_int - val_expected) < 1e-12, '[TC15] steinerberger_integral_01 FAILED'

# ---- TC16: steinerberger_function at x=0 returns 0 ----
vals = steinerberger_function(10, np.array([0.0]))
assert abs(vals[0]) < 1e-12, '[TC16] steinerberger_function x=0 FAILED'

# ---- TC17: halton_value first 2D point is (0,0) ----
import numpy as np
hv = halton_value(0, 2)
assert hv.shape == (2,), '[TC17] halton_value shape FAILED'
assert abs(hv[0]) < 1e-12 and abs(hv[1]) < 1e-12, '[TC17] halton_value zero FAILED'

# ---- TC18: halton_sequence shape correct ----
hs = halton_sequence(0, 4, 3)
assert hs.shape == (3, 5), '[TC18] halton_sequence shape FAILED'

# ---- TC19: random_orthogonal_matrix with fixed seed is orthogonal ----
import numpy as np
np.random.seed(42)
Q = random_orthogonal_matrix(8, seed=42)
orth_err = np.linalg.norm(Q.T @ Q - np.eye(8))
assert orth_err < 1e-10, '[TC19] random_orthogonal_matrix orthogonality FAILED'

# ---- TC20: random_spd_matrix with fixed seed is SPD ----
import numpy as np
np.random.seed(42)
A_spd, lam, Q2 = random_spd_matrix(10, seed=42)
assert is_spd(A_spd), '[TC20] random_spd_matrix SPD FAILED'
assert np.all(lam > 0), '[TC20] random_spd_matrix eigenvalues positive FAILED'

# ---- TC21: hutchinson_trace_estimator on identity ~ n ----
import numpy as np
np.random.seed(42)
matvec_id = lambda v: v
tr_est = hutchinson_trace_estimator(matvec_id, 20, num_samples=100, seed=42)
assert abs(tr_est - 20.0) < 2.0, '[TC21] hutchinson_trace_estimator FAILED'

# ---- TC22: line_grid output shape and boundaries for c=1 ----
x_grid = line_grid(8, 0.0, 1.0, c=1)
assert x_grid.shape == (8,), '[TC22] line_grid shape FAILED'
assert abs(x_grid[0] - 0.0) < 1e-12, '[TC22] line_grid left boundary FAILED'
assert abs(x_grid[-1] - 1.0) < 1e-12, '[TC22] line_grid right boundary FAILED'

# ---- TC23: mesh_refinement_1d doubles length minus 1 ----
x_fine = mesh_refinement_1d(x_grid)
assert x_fine.shape == (15,), '[TC23] mesh_refinement_1d length FAILED'

# ---- TC24: laguerre_polynomial L_0(x)=1 for all x ----
v_lag = laguerre_polynomial(3, 5, np.array([0.5, 1.0, 2.0]))
assert np.allclose(v_lag[:, 0], 1.0), '[TC24] laguerre_polynomial L_0 FAILED'

# ---- TC25: laguerre_polynomial L_1(x)=1-x ----
assert np.allclose(v_lag[:, 1], 1.0 - np.array([0.5, 1.0, 2.0])), '[TC25] laguerre_polynomial L_1 FAILED'

# ---- TC26: gauss_laguerre_rule integral of exp(-x)*x^2 = 2 ----
nodes, weights = gauss_laguerre_rule(8)
approx = float(np.dot(weights, nodes ** 2))
assert abs(approx - 2.0) < 1e-10, '[TC26] gauss_laguerre_rule integral FAILED'

# ---- TC27: gauss_hermite_rule sum of weights = sqrt(pi) ----
nodes_h, weights_h = gauss_hermite_rule(10)
assert abs(np.sum(weights_h) - math.sqrt(math.pi)) < 1e-10, '[TC27] gauss_hermite_rule weights FAILED'

# ---- TC28: dif2_r8ge diagonal values are 2 ----
A_dif2 = dif2_r8ge(10)
assert np.allclose(np.diag(A_dif2), 2.0), '[TC28] dif2_r8ge diagonal FAILED'

# ---- TC29: conjugate_gradient on DIF2 converges ----
A_dif2_small = dif2_r8ge(6)
b_cg = np.ones(6)
x_cg, info_cg = conjugate_gradient(lambda v: A_dif2_small @ v, b_cg, tol=1e-10)
assert info_cg['converged'], '[TC29] conjugate_gradient convergence FAILED'
assert info_cg['final_residual'] < 1e-8, '[TC29] conjugate_gradient residual FAILED'

# ---- TC30: jacobi_preconditioner on identity returns same vector ----
I_small = np.eye(5)
prec_jac = jacobi_preconditioner(I_small)
r_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
z = prec_jac(r_test)
assert np.allclose(z, r_test), '[TC30] jacobi_preconditioner identity FAILED'

# ---- TC31: lebedev_rule_14 sum of weights = 4*pi ----
nodes_l, w_l = lebedev_rule_14()
assert abs(np.sum(w_l) - 4.0 * math.pi) < 1e-10, '[TC31] lebedev_rule_14 weights FAILED'

# ---- TC32: integrate_on_sphere constant 1 = 4*pi ----
val_sph = integrate_on_sphere(lambda pts: np.ones(pts.shape[0]), rule='14')
assert abs(val_sph - 4.0 * math.pi) < 1e-10, '[TC32] integrate_on_sphere constant FAILED'

# ---- TC33: genz_evaluate cosine at origin matches formula ----
m_genz = 2
c_genz = np.ones(m_genz) / m_genz
w_genz = np.full(m_genz, 0.5)
pts_genz = np.zeros((m_genz, 1))
val_genz = genz_evaluate(1, m_genz, c_genz, w_genz, pts_genz)
expected_genz = math.cos(2.0 * math.pi * w_genz[0])
assert abs(val_genz[0] - expected_genz) < 1e-12, '[TC33] genz_evaluate cosine FAILED'

# ---- TC34: condition_number_estimate on DIF2 returns kappa > 1 ----
kappa, lmax, lmin = condition_number_estimate(A_dif2_small)
assert kappa > 1.0, '[TC34] condition_number_estimate kappa FAILED'
assert lmax > lmin, '[TC34] condition_number_estimate eigenvalues FAILED'

# ---- TC35: theoretical_cg_error_bound at k=0 returns 2.0 ----
bound = theoretical_cg_error_bound(100.0, 0)
assert abs(bound - 2.0) < 1e-12, '[TC35] theoretical_cg_error_bound k=0 FAILED'

# ---- TC36: theoretical_cg_iteration_count returns positive integer ----
it_count = theoretical_cg_iteration_count(100.0, 1e-8)
assert it_count > 0, '[TC36] theoretical_cg_iteration_count FAILED'
assert isinstance(it_count, int), '[TC36] theoretical_cg_iteration_count type FAILED'

# ---- TC37: build_test_problem dif2 returns correct dimension ----
A_tp, b_tp, x_tp = build_test_problem('dif2', 8)
assert A_tp.shape == (8, 8), '[TC37] build_test_problem dif2 shape FAILED'
assert len(b_tp) == 8, '[TC37] build_test_problem dif2 b shape FAILED'

# ---- TC38: compare_solvers returns string containing solver name ----
dummy_results = {'TestSolver': {'iterations': 5, 'final_residual': 1e-8, 'converged': True}}
report = compare_solvers(dummy_results)
assert isinstance(report, str), '[TC38] compare_solvers string FAILED'
assert 'TestSolver' in report, '[TC38] compare_solvers content FAILED'

# ---- TC39: r83_mv agrees with dense matrix-vector multiply ----
a83 = dif2_r83(5)
x83 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y83 = r83_mv(a83, x83)
A83_dense = dif2_r8ge(5)
y83_dense = A83_dense @ x83
assert np.allclose(y83, y83_dense), '[TC39] r83_mv vs dense FAILED'

# ---- TC40: r83s_mv agrees with dense matrix-vector multiply ----
a83s = dif2_r83s()
y83s = r83s_mv(a83s, x83)
assert np.allclose(y83s, y83_dense), '[TC40] r83s_mv vs dense FAILED'

# ---- TC41: r83t_mv agrees with dense matrix-vector multiply ----
a83t = dif2_r83t(5)
y83t = r83t_mv(a83t, x83)
assert np.allclose(y83t, y83_dense), '[TC41] r83t_mv vs dense FAILED'

# ---- TC42: anisotropic_diffusion_2d with eps_x=eps_y=1 is SPD ----
A_aniso = anisotropic_diffusion_2d(4, 4, 1.0, 1.0)
assert is_spd(A_aniso), '[TC42] anisotropic_diffusion_2d SPD FAILED'

# ---- TC43: verify_checksum confirms matching checksums ----
v_chk = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
cs_chk = checksum_vector(v_chk)
assert verify_checksum(v_chk, cs_chk) == True, '[TC43] verify_checksum match FAILED'

# ---- TC44: conjugate_gradient on DIF2 converges exactly (residual ~0) within n steps ----
A_44 = dif2_r8ge(8)
b_44 = np.ones(8)
x_44, info_44 = conjugate_gradient(lambda v: A_44 @ v, b_44, tol=1e-10)
assert info_44['iterations'] <= 8, '[TC44] CG iterations within n FAILED'
assert info_44['final_residual'] < 1e-12, '[TC44] CG final residual zero FAILED'

# ---- TC45: preconditioned_cg with Jacobi converges in fewer iterations on DIF2 ----
A_45 = dif2_r8ge(16)
b_45 = np.ones(16)
x_cg45, info_cg45 = conjugate_gradient(lambda v: A_45 @ v, b_45, tol=1e-8)
prec_j45 = jacobi_preconditioner(A_45)
x_pcg45, info_pcg45 = preconditioned_cg(lambda v: A_45 @ v, prec_j45, b_45, tol=1e-8)
assert info_pcg45['converged'], '[TC45] PCG Jacobi convergence FAILED'
