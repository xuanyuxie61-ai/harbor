# ---- TC01: linear_solver - gauss_elimination_partial_pivot solves 2x2 system exactly ----
import numpy as np
A_small = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
b_small = np.array([5.0, 8.0], dtype=np.float64)
x_gauss = gauss_elimination_partial_pivot(A_small, b_small)
assert x_gauss.shape == (2,), '[TC01] gauss_elimination_partial_pivot output shape FAILED'
assert np.abs(np.dot(A_small, x_gauss)[0] - b_small[0]) < 1e-12, '[TC01] gauss_elimination solves Ax=b FAILED'
assert np.abs(np.dot(A_small, x_gauss)[1] - b_small[1]) < 1e-12, '[TC01] gauss_elimination consistency FAILED'

# ---- TC02: linear_solver - plu_decomposition PA = LU identity ----
A_plu = np.array([[4.0, 3.0, 2.0], [3.0, 5.0, 1.0], [2.0, 1.0, 6.0]], dtype=np.float64)
P, L, U = plu_decomposition(A_plu)
LU = L @ U
PA = P @ A_plu
assert np.max(np.abs(LU - PA)) < 1e-10, '[TC02] PLU decomposition PA != LU FAILED'
assert np.allclose(np.diag(L), 1.0), '[TC02] L must be unit lower triangular FAILED'
assert np.allclose(np.triu(U), U), '[TC02] U must be upper triangular FAILED'

# ---- TC03: linear_solver - solve_plu produces correct solution ----
b_plu = np.array([1.0, 2.0, 3.0], dtype=np.float64)
x_plu = solve_plu(P, L, U, b_plu)
assert np.max(np.abs(A_plu @ x_plu - b_plu)) < 1e-10, '[TC03] solve_plu Ax=b FAILED'

# ---- TC04: linear_solver - condition_number_estimate for identity ----
I_n = np.eye(5, dtype=np.float64)
cond_I = condition_number_estimate(I_n)
assert np.abs(cond_I - 1.0) < 1e-10, '[TC04] condition number of identity must be 1 FAILED'

# ---- TC05: linear_solver - gauss_elimination raises ValueError for singular matrix ----
A_sing = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
b_sing = np.array([3.0, 6.0], dtype=np.float64)
raised = False
try:
    gauss_elimination_partial_pivot(A_sing, b_sing)
except ValueError:
    raised = True
assert raised, '[TC05] gauss_elimination should raise ValueError for singular matrix FAILED'

# ---- TC06: mode_analysis - bernstein_basis n=0 is all ones ----
x_bern0 = np.array([0.0, 0.5, 1.0])
B0 = bernstein_basis(0, x_bern0)
assert B0.shape == (3, 1), '[TC06] bernstein_basis n=0 shape FAILED'
assert np.allclose(B0, 1.0), '[TC06] bernstein_basis n=0 should be all 1 FAILED'

# ---- TC07: mode_analysis - bernstein_basis partition of unity ----
x_pu = np.linspace(0.0, 1.0, 10)
B5 = bernstein_basis(5, x_pu)
assert np.allclose(np.sum(B5, axis=1), 1.0), '[TC07] bernstein_basis partition of unity FAILED'

# ---- TC08: mode_analysis - bernstein_approximate exact for constant function ----
f_vals_const = np.array([3.0, 3.0, 3.0, 3.0])
x_eval_const = np.linspace(0.0, 1.0, 20)
y_const = bernstein_approximate(f_vals_const, 0.0, 1.0, 3, x_eval_const)
assert np.allclose(y_const, 3.0), '[TC08] bernstein_approximate constant function FAILED'

# ---- TC09: mode_analysis - bernstein_approximate end-point interpolation ----
f_end = np.array([1.0, 2.0, 3.0, 4.0])
y_0 = bernstein_approximate(f_end, 0.0, 3.0, 3, np.array([0.0]))
y_3 = bernstein_approximate(f_end, 0.0, 3.0, 3, np.array([3.0]))
assert np.abs(y_0[0] - 1.0) < 1e-10, '[TC09] bernstein end-point interpolation at a FAILED'
assert np.abs(y_3[0] - 4.0) < 1e-10, '[TC09] bernstein end-point interpolation at b FAILED'

# ---- TC10: mode_analysis - jacobi_polynomial n=0 is identically 1 ----
x_j0 = np.array([-1.0, 0.0, 0.5, 1.0])
P_j0 = jacobi_polynomial(0, 0.5, -0.3, x_j0)
assert np.allclose(P_j0, 1.0), '[TC10] jacobi_polynomial n=0 must be 1 FAILED'

# ---- TC11: mode_analysis - jacobi_polynomial n=1 linear formula ----
x_j1 = np.array([0.0, 0.5, 1.0])
P_j1 = jacobi_polynomial(1, 0.0, 0.0, x_j1)
alpha = 0.0; beta = 0.0
expected = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x_j1
assert np.allclose(P_j1, expected), '[TC11] jacobi_polynomial n=1 FAILED'

# ---- TC12: joint_spectrum - normalized_sinc at 0 is 1 ----
x_sinc0 = np.array([0.0])
s0 = normalized_sinc(x_sinc0)
assert np.abs(s0[0] - 1.0) < 1e-14, '[TC12] normalized_sinc(0) must be 1 FAILED'

# ---- TC13: joint_spectrum - normalized_sinc at integer is 0 ----
x_sinc_int = np.array([1.0, 2.0, 3.0])
s_int = normalized_sinc(x_sinc_int)
assert np.max(np.abs(s_int)) < 1e-14, '[TC13] normalized_sinc(n) must be 0 for n!=0 FAILED'

# ---- TC14: joint_spectrum - pump_envelope_gaussian peak value ----
omega_p0 = 2.0 * np.pi * 2.99792458e8 / 405.0e-9
alpha_peak = pump_envelope_gaussian(np.array([omega_p0]), omega_p0, omega_p0 * 0.015)
assert np.abs(alpha_peak[0] - 1.0) < 1e-14, '[TC14] pump_envelope_gaussian peak must be 1 FAILED'

# ---- TC15: joint_spectrum - pump_envelope_gaussian decay ----
sigma_gauss = omega_p0 * 0.01
omega_off = omega_p0 + 3.0 * sigma_gauss
alpha_off = pump_envelope_gaussian(np.array([omega_off]), omega_p0, sigma_gauss)
assert alpha_off[0] < 0.02, '[TC15] pump_envelope_gaussian 3-sigma decay FAILED'

# ---- TC16: joint_spectrum - phase_matching_function with zero mismatch is 1 ----
dk_zero = np.array([0.0])
phi_pm = phase_matching_function(dk_zero, 0.01)
assert np.abs(phi_pm[0] - 1.0) < 1e-14, '[TC16] phase_matching_function(0) must be 1 FAILED'

# ---- TC17: joint_spectrum - compute_jsa output shape and normalization ----
c = 2.99792458e8
lambda_p0 = 405.0e-9
omega_p0_local = 2.0 * np.pi * c / lambda_p0

def sellmeier_p(omega):
    lam = 2.0 * np.pi * c / (omega + 1e-20)
    lam_um = np.abs(lam) * 1e6
    n2 = 2.19229 + 0.83547 / (1.0 - 0.04970 / (lam_um ** 2)) - 0.01696 * lam_um ** 2
    return np.sqrt(np.maximum(1.0, n2))

def sellmeier_s(omega):
    lam = 2.0 * np.pi * c / (omega + 1e-20)
    lam_um = np.abs(lam) * 1e6
    n2 = 2.10468 + 0.89342 / (1.0 - 0.04456 / (lam_um ** 2)) - 0.01020 * lam_um ** 2
    return np.sqrt(np.maximum(1.0, n2))

n_omega_t = 8
omega_s_t = np.linspace(omega_p0_local * 0.48, omega_p0_local * 0.52, n_omega_t)
omega_i_t = np.linspace(omega_p0_local * 0.48, omega_p0_local * 0.52, n_omega_t)
jsa_t = compute_jsa(omega_s_t, omega_i_t, omega_p0_local, omega_p0_local * 0.015, 10.0e-3, 9.5e-6,
                    sellmeier_p, sellmeier_s, sellmeier_p)
assert jsa_t.shape == (n_omega_t, n_omega_t), '[TC17] compute_jsa output shape FAILED'
norm_jsa = np.sum(np.abs(jsa_t) ** 2)
assert np.abs(norm_jsa - 1.0) < 1e-12, '[TC17] compute_jsa normalization FAILED'

# ---- TC18: joint_spectrum - schmidt_decomposition_jsa purity in [0,1] ----
lambdas_t, u_t, v_t, K_t, purity_t = schmidt_decomposition_jsa(jsa_t)
assert 0.0 <= purity_t <= 1.0, '[TC18] purity must be in [0,1] FAILED'
assert np.abs(np.sum(lambdas_t) - 1.0) < 1e-12, '[TC18] Schmidt coefficients must sum to 1 FAILED'

# ---- TC19: joint_spectrum - schmidt_decomposition_jsa K >= 1 ----
assert K_t >= 1.0 - 1e-10, '[TC19] Schmidt number K must be >= 1 FAILED'

# ---- TC20: entanglement_metrics - concurrence_from_purity pure state is 0 ----
C_pure = concurrence_from_purity(1.0)
assert np.abs(C_pure) < 1e-12, '[TC20] concurrence of pure state must be 0 FAILED'

# ---- TC21: entanglement_metrics - concurrence_from_purity maximally mixed is sqrt(2) ----
C_mixed = concurrence_from_purity(0.0)
assert np.abs(C_mixed - np.sqrt(2.0)) < 1e-12 or np.abs(C_mixed - 1.0) < 1e-12, '[TC21] concurrence bounds FAILED'

# ---- TC22: entanglement_metrics - von_neumann_entropy_schmidt single mode is 0 ----
S_single = von_neumann_entropy_schmidt(np.array([1.0]))
assert np.abs(S_single) < 1e-12, '[TC22] entropy of single Schmidt mode must be 0 FAILED'

# ---- TC23: entanglement_metrics - von_neumann_entropy_schmidt maximally mixed ----
n_modes = 16
lambdas_flat = np.ones(n_modes) / n_modes
S_flat = von_neumann_entropy_schmidt(lambdas_flat)
assert np.abs(S_flat - np.log2(n_modes)) < 1e-10, '[TC23] entropy of flat spectrum FAILED'

# ---- TC24: entanglement_metrics - chsh_parameter with known correlation matrix ----
corr_t = np.array([[495, 5, 470, 30],
                    [5, 495, 30, 470],
                    [470, 30, 480, 20],
                    [30, 470, 20, 480]], dtype=np.float64)
S_chsh = chsh_parameter(corr_t)
assert S_chsh > 0.0, '[TC24] CHSH parameter must be positive FAILED'
assert S_chsh <= 2.0 * np.sqrt(2.0) + 1e-10, '[TC24] CHSH parameter must not exceed 2*sqrt(2) FAILED'

# ---- TC25: entanglement_metrics - chsh_parameter with uncorrelated counts ----
corr_uncorr = np.ones((4, 4), dtype=np.float64) * 100.0
S_uncorr = chsh_parameter(corr_uncorr)
assert np.abs(S_uncorr) < 1e-10, '[TC25] CHSH for uncorrelated should be 0 FAILED'

# ---- TC26: entanglement_metrics - state_fidelity_target singlet ----
jsa_fid = np.diag(np.ones(8)) * (1.0 / np.sqrt(8.0))
F_singlet = state_fidelity_target(jsa_fid, target_type="singlet")
assert 0.0 <= F_singlet <= 1.0, '[TC26] singlet fidelity must be in [0,1] FAILED'

# ---- TC27: network_coupling - build_coupling_digraph output shape ----
n_stages_t = 3
np.random.seed(42)
A_net_t = build_coupling_digraph(n_stages_t, coupling_strength=0.1)
assert A_net_t.shape == (2 * n_stages_t, 2 * n_stages_t), '[TC27] coupling digraph shape FAILED'

# ---- TC28: network_coupling - adjacency_to_transition column sum is 1 or 0 ----
T_net_t = adjacency_to_transition(A_net_t)
col_sums = np.sum(np.abs(T_net_t), axis=0)
assert np.all((col_sums < 1e-14) | (np.abs(col_sums - 1.0) < 1e-10)), '[TC28] transition matrix column sums FAILED'

# ---- TC29: network_coupling - transitive_closure_digraph is reflexive for self-loops ----
C_closure = transitive_closure_digraph(A_net_t)
diag_C = np.diag(C_closure)
assert np.all(diag_C == 1), '[TC29] transitive closure diagonal (self-reachable) FAILED'

# ---- TC30: parameter_optimizer - gray_code_subsets count is 2^n ----
n_gray = 4
subs_gray, iadds_gray = gray_code_subsets(n_gray)
assert len(subs_gray) == 2 ** n_gray, '[TC30] gray_code_subsets count must be 2^n FAILED'

# ---- TC31: parameter_optimizer - diophantine_nd_nonnegative known solution ----
a_dio = np.array([3, 5], dtype=int)
b_dio = 16
sols_dio = diophantine_nd_nonnegative(a_dio, b_dio)
assert sols_dio.shape[0] >= 1, '[TC31] diophantine must have at least 1 solution for a=[3,5],b=16 FAILED'

# ---- TC32: parameter_optimizer - diophantine all solutions satisfy a·x=b ----
for i in range(min(sols_dio.shape[0], 10)):
    assert np.dot(a_dio, sols_dio[i, :]) == b_dio, '[TC32] diophantine solution must satisfy a·x=b FAILED'

# ---- TC33: parameter_optimizer - subset_sum_backtrack_all finds correct subsets ----
v_ss = np.array([2, 3, 5, 7], dtype=int)
target_ss = 10
ss_sols = subset_sum_backtrack_all(target_ss, v_ss)
found = False
for sol in ss_sols:
    if np.sum(sol * v_ss) == target_ss:
        found = True
        break
assert found and len(ss_sols) > 0, '[TC33] subset_sum_backtrack must find solution for 2+3+5=10 FAILED'

# ---- TC34: quantum_evolution - spdc_derivative output shape ----
gamma_t = np.array([1.0e6, 1.0e6, 1.0e10], dtype=np.float64)
y_t = np.array([1.0, 0.5, 2.0], dtype=np.complex128)
kappa_t = 5.0e6
dydt = spdc_derivative(0.0, y_t, gamma_t, lambda t: kappa_t, lambda t: np.zeros(3, dtype=np.complex128))
assert dydt.shape == (3,), '[TC34] spdc_derivative output shape FAILED'

# ---- TC35: quantum_evolution - robertson_like_conservation non-negative ----
y_rob = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 1.0]], dtype=np.complex128)
C_rob = robertson_like_conservation(y_rob)
assert C_rob.shape == (2,), '[TC35] robertson_like_conservation output shape FAILED'
assert np.all(C_rob >= 0.0), '[TC35] conserved quantity must be non-negative FAILED'

# ---- TC36: quantum_evolution - backward_euler_spdc produces correct shapes ----
np.random.seed(42)
y0_t = np.array([0.0, 0.0, 1.0e3], dtype=np.complex128)
t_span_t = (0.0, 1.0e-6)
n_steps_t = 10
t_ode, y_ode = backward_euler_spdc(
    y0=y0_t, t_span=t_span_t, n_steps=n_steps_t,
    gamma=gamma_t, kappa_func=lambda t: 5.0e6,
    f_noise=lambda t: np.zeros(3, dtype=np.complex128),
    newton_tol=1e-10, max_newton=20
)
assert t_ode.shape == (n_steps_t + 1,), '[TC36] backward_euler t array shape FAILED'
assert y_ode.shape == (n_steps_t + 1, 3), '[TC36] backward_euler y array shape FAILED'

# ---- TC37: phase_space_integral - fibonacci_sequence correctness ----
from phase_space_integral import fibonacci_sequence
fib = fibonacci_sequence(10)
assert fib[0] == 0, '[TC37] fibonacci F0 must be 0 FAILED'
assert fib[1] == 1, '[TC37] fibonacci F1 must be 1 FAILED'
assert fib[9] == 34, '[TC37] fibonacci F9 must be 34 FAILED'

# ---- TC38: phase_space_integral - lattice_rule_2d_periodic integrates constant ----
np.random.seed(42)
I_const = lattice_rule_2d_periodic(lambda x: 1.0, 8)
assert np.abs(I_const - 1.0) < 1e-10, '[TC38] lattice rule constant integration must be 1 FAILED'

# ---- TC39: mode_analysis - spherical_harmonic_basis output dimensions ----
theta_t = np.linspace(0.0, np.pi, 10)
phi_t = np.linspace(0.0, 2.0 * np.pi, 10)
Yr, Yi = spherical_harmonic_basis(l_max=2, theta=theta_t, phi=phi_t)
n_modes_expected = (2 + 1) ** 2
assert Yr.shape == (10, n_modes_expected), '[TC39] spherical_harmonic_basis real part shape FAILED'
assert Yi.shape == (10, n_modes_expected), '[TC39] spherical_harmonic_basis imag part shape FAILED'

# ---- TC40: pump_propagation - burgers_like_pump_solution output shape ----
import numpy as np
np.random.seed(42)
z_burg = np.linspace(-1.0, 1.0, 21)
t_burg = np.linspace(0.01, 0.1, 5)
U_burg = burgers_like_pump_solution(nu_eff=0.01 / np.pi, z_grid=z_burg, t_grid=t_burg)
assert U_burg.shape == (21, 5), '[TC40] burgers_like_pump_solution output shape FAILED'

# ---- TC41: linear_solver - condition_number singular matrix is huge/inf ----
A_bad = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
cond_bad = condition_number_estimate(A_bad)
assert cond_bad > 1e5 or np.isinf(cond_bad), '[TC41] condition number of singular matrix must be large FAILED'

# ---- TC42: joint_spectrum - phase_mismatch returns ndarray ----
c = 2.99792458e8
omega_p0_local = 2.0 * np.pi * c / 405.0e-9
dk_test = phase_mismatch(np.array([omega_p0_local * 0.5]), np.array([omega_p0_local * 0.5]),
                          omega_p0_local, 9.5e-6, sellmeier_p, sellmeier_s, sellmeier_p)
assert isinstance(dk_test, np.ndarray), '[TC42] phase_mismatch must return ndarray FAILED'
assert np.isfinite(dk_test[0]), '[TC42] phase_mismatch must be finite FAILED'

# ---- TC43: entanglement_metrics - hom_visibility V in [0,1] ----
jsa_r = np.ones((4, 4), dtype=np.float64) / 4.0
jsa_i = np.zeros((4, 4), dtype=np.float64)
delay_g = np.linspace(-1.0e-12, 1.0e-12, 21)
om_s = np.linspace(1.0, 2.0, 4)
om_i = np.linspace(1.0, 2.0, 4)
R_tau, V_hom = hom_visibility(jsa_r, jsa_i, delay_g, om_s, om_i)
assert 0.0 <= V_hom <= 1.0, '[TC43] HOM visibility must be in [0,1] FAILED'

# ---- TC44: linear_solver - gauss_elimination_partial_pivot handles matrix with right-hand side matrix ----
A_multi = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
B_multi = np.array([[5.0, 1.0], [8.0, -1.0]], dtype=np.float64)
X_multi = gauss_elimination_partial_pivot(A_multi, B_multi)
assert X_multi.shape == (2, 2), '[TC44] gauss_elimination multi-RHS shape FAILED'
assert np.max(np.abs(A_multi @ X_multi - B_multi)) < 1e-10, '[TC44] gauss_elimination multi-RHS FAILED'

# ---- TC45: phase_space_integral - triangle_unit_sample_random numbers in unit triangle ----
from phase_space_integral import triangle_unit_sample_random
import numpy as np
np.random.seed(42)
pts_tri = triangle_unit_sample_random(100)
assert pts_tri.shape == (100, 2), '[TC45] triangle sampling shape FAILED'
assert np.all(pts_tri[:, 0] >= 0.0), '[TC45] triangle sampling xi1 >= 0 FAILED'
assert np.all(pts_tri[:, 1] >= 0.0), '[TC45] triangle sampling xi2 >= 0 FAILED'
assert np.all(pts_tri[:, 0] + pts_tri[:, 1] <= 1.0 + 1e-12), '[TC45] triangle xi1+xi2 <= 1 FAILED'

# ---- TC46: mode_analysis - bernstein_basis at x=0 gives [1,0,...,0] ----
B_at_0 = bernstein_basis(4, np.array([0.0]))
assert np.abs(B_at_0[0, 0] - 1.0) < 1e-14, '[TC46] bernstein_basis B0,4(0)=1 FAILED'
assert np.allclose(B_at_0[0, 1:], 0.0), '[TC46] bernstein_basis B_{>0},4(0)=0 FAILED'

# ---- TC47: mode_analysis - bernstein_basis at x=1 gives [0,...,0,1] ----
B_at_1 = bernstein_basis(4, np.array([1.0]))
assert np.abs(B_at_1[0, -1] - 1.0) < 1e-14, '[TC47] bernstein_basis B4,4(1)=1 FAILED'
assert np.allclose(B_at_1[0, :-1], 0.0), '[TC47] bernstein_basis B_{<4},4(1)=0 FAILED'

# ---- TC48: joint_spectrum - compute_jsa is not all-zero ----
assert np.any(np.abs(jsa_t) > 0.0), '[TC48] compute_jsa should produce non-zero output FAILED'

# ---- TC49: network_coupling - network_photon_number_evolution output shape ----
np.random.seed(42)
n_stages_ev = 2
A_ev = build_coupling_digraph(n_stages_ev)
n_init = np.zeros(2 * n_stages_ev)
n_init[0] = 100.0
src = np.zeros((n_stages_ev, 2 * n_stages_ev))
n_hist = network_photon_number_evolution(n_stages_ev, n_init, src, A_ev)
assert n_hist.shape == (n_stages_ev + 1, 2 * n_stages_ev), '[TC49] photon evolution shape FAILED'
assert np.all(n_hist >= 0.0), '[TC49] photon numbers must be non-negative FAILED'

# ---- TC50: pump_propagation - solve_pump_envelope_fem simple linear case ----
import numpy as np
np.random.seed(42)
c = 2.99792458e8
lambda_p0 = 405.0e-9
omega_p0_local = 2.0 * np.pi * c / lambda_p0

def sellmeier_p(omega):
    lam = 2.0 * np.pi * c / (omega + 1e-20)
    lam_um = np.abs(lam) * 1e6
    n2 = 2.19229 + 0.83547 / (1.0 - 0.04970 / (lam_um ** 2)) - 0.01696 * lam_um ** 2
    return np.sqrt(np.maximum(1.0, n2))

k_p_fem = sellmeier_p(omega_p0_local) * omega_p0_local / c
A_fem = solve_pump_envelope_fem(
    n_nodes=5,
    z_domain=(0.0, 5.0e-3),
    k_p=k_p_fem,
    alpha_p=0.0,
    gamma_eff=lambda z, A: 0.0,
    source_spdc=lambda z: 0.0,
    nonlinear_tol=1e-8,
    max_iter=30
)
assert A_fem.shape == (5,), '[TC50] FEM pump propagation output shape FAILED'
assert np.abs(A_fem[0] - 1.0e4) < 1e-8, '[TC50] FEM Dirichlet BC at inlet FAILED'
assert np.all(np.isfinite(A_fem)), '[TC50] FEM output must be finite FAILED'
