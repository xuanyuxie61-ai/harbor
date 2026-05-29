# ---- TC01: polyiamond_hexagon_mesh generates valid topology ----
nodes, elements, boundary_nodes = polyiamond_hexagon_mesh(1, scale=1.0)
assert nodes.shape[0] >= 3, '[TC01] polyiamond_hexagon_mesh FAILED'
assert elements.shape[1] == 3, '[TC01] polyiamond_hexagon_mesh FAILED'
assert len(boundary_nodes) > 0, '[TC01] polyiamond_hexagon_mesh FAILED'

# ---- TC02: magic_square row and column sums equal magic constant ----
n = 5
M = magic_square(n)
magic_sum = n * (n ** 2 + 1) // 2
assert np.all(np.sum(M, axis=1) == magic_sum), '[TC02] magic_square FAILED'
assert np.all(np.sum(M, axis=0) == magic_sum), '[TC02] magic_square FAILED'

# ---- TC03: generate_perforated_plate_mesh excludes holes ----
nodes_p, elements_p, boundary_p = generate_perforated_plate_mesh(4, 4, hole_positions=[(0.5, 0.5)], hole_radius=0.2, Lx=1.0, Ly=1.0)
assert nodes_p.shape[0] > 0, '[TC03] generate_perforated_plate_mesh FAILED'
assert elements_p.shape[0] > 0, '[TC03] generate_perforated_plate_mesh FAILED'
for nidx in range(nodes_p.shape[0]):
    xn, yn = nodes_p[nidx]
    assert (xn - 0.5) ** 2 + (yn - 0.5) ** 2 >= 0.2 ** 2 - 1e-10, '[TC03] generate_perforated_plate_mesh FAILED'

# ---- TC04: integrate_triangle_wandzura exact for constant function ----
unit_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
I_const = integrate_triangle_wandzura(lambda x, y: 1.0, unit_tri, degree=2)
assert abs(I_const - 0.5) < 1e-10, '[TC04] integrate_triangle_wandzura FAILED'

# ---- TC05: integrate_hexahedron_witherden exact for constant ----
I_hex_const = integrate_hexahedron_witherden(lambda x, y, z: 1.0, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), degree=3)
assert abs(I_hex_const - 1.0) < 1e-10, '[TC05] integrate_hexahedron_witherden FAILED'

# ---- TC06: t3_shape_functions sum to unity ----
N, dN_dxi, dN_deta = t3_shape_functions(0.3, 0.2)
assert abs(np.sum(N) - 1.0) < 1e-10, '[TC06] t3_shape_functions FAILED'

# ---- TC07: r83v_mv produces correct matrix-vector product ----
n_r = 5
a_r = np.ones(n_r - 1)
b_r = 2.0 * np.ones(n_r)
c_r = np.ones(n_r - 1)
x_r = np.arange(1, n_r + 1, dtype=float)
y_r = r83v_mv(n_r, a_r, b_r, c_r, x_r)
expected_r = np.array([4.0, 8.0, 12.0, 16.0, 14.0])
assert np.allclose(y_r, expected_r), '[TC07] r83v_mv FAILED'

# ---- TC08: r83v_fs solves tridiagonal system with small residual ----
n_fs = 10
a_fs = -1.0 * np.ones(n_fs - 1)
b_fs = 2.0 * np.ones(n_fs)
c_fs = -1.0 * np.ones(n_fs - 1)
rhs_fs = np.ones(n_fs)
x_sol_fs = r83v_fs(n_fs, a_fs, b_fs, c_fs, rhs_fs)
res_fs = np.linalg.norm(r83v_mv(n_fs, a_fs, b_fs, c_fs, x_sol_fs) - rhs_fs)
assert res_fs < 1e-8, '[TC08] r83v_fs FAILED'

# ---- TC09: r83v_cg converges for SPD tridiagonal system ----
n_cg = 20
a_cg = -1.0 * np.ones(n_cg - 1)
b_cg = 2.0 * np.ones(n_cg)
c_cg = -1.0 * np.ones(n_cg - 1)
np.random.seed(42)
rhs_cg = np.random.rand(n_cg)
x_cg, it_cg, res_cg = r83v_cg(n_cg, a_cg, b_cg, c_cg, rhs_cg, tol=1e-10)
assert res_cg < 1e-8, '[TC09] r83v_cg FAILED'

# ---- TC10: build_beam_tridiagonal produces symmetric system ----
a_beam, b_beam, c_beam, rhs_beam, h_beam = build_beam_tridiagonal(10, EI=1.0, L=1.0, load_type='uniform')
assert np.allclose(a_beam, c_beam), '[TC10] build_beam_tridiagonal FAILED'
assert h_beam > 0, '[TC10] build_beam_tridiagonal FAILED'

# ---- TC11: frequency_response_function peak near resonance ----
omega_f = np.linspace(0.1, 20.0, 1000)
omega_n_f = 10.0
zeta_f = 0.05
H_f = frequency_response_function(omega_f, omega_n_f, zeta_f)
peak_idx_f = np.argmax(np.abs(H_f))
assert abs(omega_f[peak_idx_f] - omega_n_f) < 0.5, '[TC11] frequency_response_function FAILED'

# ---- TC12: psd_response non-negative for all frequencies ----
omega_p = np.linspace(0.1, 20.0, 500)
S_y_p = psd_response(omega_p, omega_n=10.0, zeta=0.02, psd_input=lambda o: np.ones_like(o))
assert np.all(S_y_p >= 0), '[TC12] psd_response FAILED'

# ---- TC13: first_passage_probability within [0, 1] ----
Pf_fp, nu_fp = first_passage_probability(sigma_y=1.0, threshold=2.0, omega_0=10.0, T_duration=3600.0, method='poisson')
assert 0.0 <= Pf_fp <= 1.0, '[TC13] first_passage_probability FAILED'
assert nu_fp >= 0, '[TC13] first_passage_probability FAILED'

# ---- TC14: cauchy_principal_value finite for smooth function ----
cpv_val = cauchy_principal_value(lambda t: np.sin(t), 0.0, np.pi, np.pi / 2.0, n=20)
assert np.isfinite(cpv_val), '[TC14] cauchy_principal_value FAILED'

# ---- TC15: lngamma value matches known log Gamma(5.5) ----
lg_val, ier_val = lngamma(5.5)
assert ier_val == 0, '[TC15] lngamma FAILED'
assert abs(lg_val - 3.957813) < 1e-4, '[TC15] lngamma FAILED'

# ---- TC16: gamma_function Gamma(5) = 24 ----
g5_val = gamma_function(5.0)
assert abs(g5_val - 24.0) < 1e-8, '[TC16] gamma_function FAILED'

# ---- TC17: golden_section_search finds minimum of parabola ----
x_min_gs, f_min_gs, it_gs = golden_section_search(lambda x: (x - 2.0) ** 2, -5.0, 5.0, n_iter=50, x_tol=1e-10)
assert abs(x_min_gs - 2.0) < 1e-6, '[TC17] golden_section_search FAILED'
assert f_min_gs < 1e-10, '[TC17] golden_section_search FAILED'

# ---- TC18: normal_cdf monotonic and bounded ----
x_cdf = np.array([-5.0, 0.0, 5.0])
cdf_vals = normal_cdf(x_cdf)
assert np.all(np.diff(cdf_vals) > 0), '[TC18] normal_cdf FAILED'
assert 0.0 <= cdf_vals[0] <= 0.01, '[TC18] normal_cdf FAILED'
assert 0.99 <= cdf_vals[-1] <= 1.0, '[TC18] normal_cdf FAILED'

# ---- TC19: complete_elliptic_K increasing in m ----
m_ell = np.array([0.1, 0.5, 0.9])
K_ell = complete_elliptic_k(m_ell)
assert np.all(np.diff(K_ell) > 0), '[TC19] complete_elliptic_k FAILED'

# ---- TC20: nonlinear_vibration_period hardening T < T_linear ----
T_nl_val, T_lin_val = nonlinear_vibration_period(amplitude=0.01, omega_linear=100.0, alpha_nonlin=1e6)
assert T_nl_val < T_lin_val, '[TC20] nonlinear_vibration_period FAILED'

# ---- TC21: elliptical_hole_stress_concentration SCF >= 1 ----
theta_arr = np.linspace(0, 2 * np.pi, 50)
s_theta_val, kt_val = elliptical_hole_stress_concentration(a=0.2, b=0.1, sigma_inf=100e6, theta=theta_arr)
assert kt_val >= 1.0, '[TC21] elliptical_hole_stress_concentration FAILED'

# ---- TC22: legendre_polynomial orthonormality on Gauss quadrature ----
from pce_expansion import legendre_polynomial
xq_leg, wq_leg = np.polynomial.legendre.leggauss(5)
L_leg = legendre_polynomial(3, xq_leg)
for i_leg in range(4):
    for j_leg in range(i_leg + 1, 4):
        dot_leg = np.sum(wq_leg * L_leg[i_leg] * L_leg[j_leg])
        assert abs(dot_leg) < 1e-10, '[TC22] legendre_polynomial FAILED'

# ---- TC23: generate_multi_indices count matches expected for n=3,p=2 ----
indices_test = generate_multi_indices(3, 2)
expected_count_23 = 10
assert indices_test.shape[0] == expected_count_23, '[TC23] generate_multi_indices FAILED'

# ---- TC24: evaluate_pce_basis norms equal 1 for orthonormal Legendre ----
np.random.seed(42)
xi_test = np.random.uniform(-1, 1, (10, 2))
psi_test, _, norms_test = evaluate_pce_basis(2, 2, xi_test)
assert np.allclose(norms_test, 1.0), '[TC24] evaluate_pce_basis FAILED'

# ---- TC25: pce_moments mean equals first coefficient ----
coeffs_test = np.array([3.0, 0.5, -0.2, 1.0, 0.0, 0.3])
indices_pce = generate_multi_indices(2, 2)
mean_pce, var_pce = pce_moments(coeffs_test, indices_pce)
assert abs(mean_pce - 3.0) < 1e-10, '[TC25] pce_moments FAILED'
assert var_pce >= 0, '[TC25] pce_moments FAILED'

# ---- TC26: kl_expansion_1d modes are normalized ----
x_kl = np.linspace(0, 1.0, 50)
eigvals_kl, modes_kl = kl_expansion_1d(3, length=1.0, correlation_length=0.3, x_coords=x_kl)
for i_kl in range(3):
    norm_sq_kl = np.trapezoid(modes_kl[i_kl] ** 2, x_kl)
    assert abs(norm_sq_kl - 1.0) < 0.1, '[TC26] kl_expansion_1d FAILED'

# ---- TC27: cvt_energy decreases after one Lloyd iteration ----
np.random.seed(42)
from cvt_sampling import cvt_energy, cvt_iterate
gens_cvt = np.random.rand(5, 2)
samples_cvt = np.random.rand(1000, 2)
E0_cvt, _ = cvt_energy(gens_cvt, samples_cvt)
new_gens_cvt, diff_cvt, _ = cvt_iterate(gens_cvt, samples_cvt)
E1_cvt, _ = cvt_energy(new_gens_cvt, samples_cvt)
assert E1_cvt <= E0_cvt + 1e-10, '[TC27] cvt_iterate FAILED'

# ---- TC28: generate_cvt_samples reproducible with fixed seed ----
np.random.seed(42)
samples1 = generate_cvt_samples(dim=2, n_gen=10, it_max=5, seed=42)
np.random.seed(42)
samples2 = generate_cvt_samples(dim=2, n_gen=10, it_max=5, seed=42)
assert np.allclose(samples1, samples2), '[TC28] generate_cvt_samples FAILED'

# ---- TC29: map_stochastic_to_sphere yields unit directions ----
np.random.seed(42)
xi_samples = np.random.randn(20, 3)
directions, radii = map_stochastic_to_sphere(xi_samples)
norms_dir = np.linalg.norm(directions, axis=1)
assert np.allclose(norms_dir, 1.0), '[TC29] map_stochastic_to_sphere FAILED'
assert np.all(radii >= 0), '[TC29] map_stochastic_to_sphere FAILED'

# ---- TC30: matrix export and read round-trip preserves values ----
A_test = np.array([[1.0, 2.0], [3.0, 4.0]])
import tempfile, os
from matrix_exporter import export_matrix_market
fd, tmpfile = tempfile.mkstemp(suffix='.mtx')
os.close(fd)
export_matrix_market(tmpfile, A_test, symmetry='general')
A_read = read_matrix_market(tmpfile)
assert np.allclose(A_read, A_test), '[TC30] matrix export/read FAILED'
os.remove(tmpfile)

# ---- TC31: modal_analysis_tridiagonal eigenvalues positive ----
n_ma = 15
a_ma = -1.0 * np.ones(n_ma - 1)
b_ma = 2.0 * np.ones(n_ma)
c_ma = -1.0 * np.ones(n_ma - 1)
eigvals_tri, eigvecs_tri = modal_analysis_tridiagonal(n_ma, a_ma, b_ma, c_ma, n_modes=3)
assert np.all(eigvals_tri > 0), '[TC31] modal_analysis_tridiagonal FAILED'

# ---- TC32: assemble_triangular_fem_matrices produces symmetric K ----
nodes_t = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
elems_t = np.array([[0, 1, 2], [1, 3, 2]])
K_t, M_t = assemble_triangular_fem_matrices(nodes_t, elems_t, material_thickness=1.0, young_modulus=1.0, poisson_ratio=0.3, density=1.0)
assert np.allclose(K_t, K_t.T), '[TC32] assemble_triangular_fem_matrices FAILED'
assert np.allclose(M_t, M_t.T), '[TC32] assemble_triangular_fem_matrices FAILED'

# ---- TC33: faces_average counts consistent ----
elem_stress = np.array([1.0, 2.0, 3.0])
elems_avg = np.array([[0, 1, 2], [1, 2, 3]])
nodal_avg = faces_average(elem_stress, elems_avg, 4)
assert nodal_avg[1] == 1.5, '[TC33] faces_average FAILED'

# ---- TC34: triangulation_quad exact for linear function ----
nodes_lin = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_lin = np.array([[0, 1, 2]])
node_vals_lin = nodes_lin[:, 0] + nodes_lin[:, 1]
I_lin, area_lin = triangulation_quad(nodes_lin, elems_lin, node_vals_lin)
assert abs(I_lin - (2.0 / 3.0) * area_lin) < 1e-10, '[TC34] triangulation_quad FAILED'

# ---- TC35: FORM reliability beta non-negative for simple linear limit state ----
g_lin_lambda = lambda u: 3.0 - np.sum(u)
dg_lin_lambda = lambda u: -np.ones_like(u)
beta_form, u_star, Pf_form = form_reliability(g_lin_lambda, dg_lin_lambda, u0=np.zeros(2), dim=2, max_iter=50, tol=1e-8)
assert beta_form >= 0, '[TC35] form_reliability FAILED'
assert 0.0 <= Pf_form <= 1.0, '[TC35] form_reliability FAILED'
