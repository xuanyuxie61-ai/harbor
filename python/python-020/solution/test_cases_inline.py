# ---- TC01: magnetic_length returns correct value for B=10, m*=1 ----
lB_val = magnetic_length(10.0, 1.0)
assert abs(lB_val - np.sqrt(0.1)) < 1e-12, '[TC01] magnetic_length FAILED'

# ---- TC02: cyclotron_frequency returns correct value for B=10, m*=1 ----
omega_val = cyclotron_frequency(10.0, 1.0)
assert abs(omega_val - 10.0) < 1e-12, '[TC02] cyclotron_frequency FAILED'

# ---- TC03: landau_level_energy for n=0 equals 0.5*hbar*omega_c ----
E0 = landau_level_energy(0, 10.0, 1.0)
assert abs(E0 - 5.0) < 1e-12, '[TC03] landau_level_energy FAILED'

# ---- TC04: filling_factor returns finite positive value ----
nu_val = filling_factor(8, 10.0, 50.0, 1.0)
assert nu_val > 0 and np.isfinite(nu_val), '[TC04] filling_factor FAILED'

# ---- TC05: fermi_dirac at T->0 behaves as step function around mu ----
f_low = fermi_dirac(0.5, 1.0, 1e-10)
f_high = fermi_dirac(1.5, 1.0, 1e-10)
assert f_low > 0.999 and f_high < 0.001, '[TC05] fermi_dirac step FAILED'

# ---- TC06: gram_schmidt_qr produces orthonormal Q columns ----
np.random.seed(42)
V_gs = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
Q_gs, R_gs = gram_schmidt_qr(V_gs)
for i in range(3):
    for j in range(3):
        dot = np.vdot(Q_gs[:, i], Q_gs[:, j])
        target = 1.0 if i == j else 0.0
        assert abs(dot - target) < 1e-10, '[TC06] gram_schmidt_qr FAILED'

# ---- TC07: condition_number of rank-1 2x2 matrix is inf ----
sing = np.array([[1.0, 2.0], [2.0, 4.0]])
assert condition_number(sing) == np.inf, '[TC07] condition_number singular FAILED'

# ---- TC08: safe_exp handles very large input without overflow ----
from utils import safe_exp
big_result = safe_exp(1000.0)
assert np.isfinite(big_result), '[TC08] safe_exp overflow FAILED'

# ---- TC09: safe_log handles zero without crash or nan ----
from utils import safe_log
log_result = safe_log(0.0)
assert np.isfinite(log_result), '[TC09] safe_log zero FAILED'

# ---- TC10: landau_orbital_wavefunction raises ValueError for invalid n ----
try:
    landau_orbital_wavefunction(-1, 0, np.array([1.0+0j]), 1.0)
    assert False, '[TC10] landau_orbital exception FAILED'
except ValueError:
    pass

# ---- TC11: landau_degeneracy returns positive value ----
Nphi = landau_degeneracy(10.0, 50.0, 1.0)
assert Nphi > 0, '[TC11] landau_degeneracy FAILED'

# ---- TC12: density_of_states_landau is non-negative everywhere ----
E_range = np.linspace(0, 10, 50)
dos = density_of_states_landau(E_range, 10.0, 1.0, gamma=0.05)
assert np.all(dos >= 0), '[TC12] DOS negative FAILED'

# ---- TC13: laughlin_wavefunction with return_log gives complex scalar ----
np.random.seed(42)
z_laugh = np.array([1.0+0j, 0.5+0.5j])
log_psi = laughlin_wavefunction(z_laugh, 3, 1.0, return_log=True)
assert isinstance(log_psi, (complex, np.complexfloating)), '[TC13] laughlin log type FAILED'

# ---- TC14: pair_correlation_function g(r) is non-negative ----
np.random.seed(42)
z_g2 = np.array([1.0+0j, -1.0+0j, 0.0+1.0j, 0.0-1.0j])
r_e, g_r, r_c = pair_correlation_function(z_g2, 3, 1.0, r_bins=10)
assert np.all(g_r >= 0), '[TC14] pair_correlation negative FAILED'

# ---- TC15: hammersley_sequence output shape is (m, N) ----
pts = hammersley_sequence(0, 49, 3)
assert pts.shape == (3, 50), '[TC15] hammersley shape FAILED'

# ---- TC16: radical_inverse produces identical results for same input ----
from monte_carlo_sampler import radical_inverse
phi_a = radical_inverse(7, 3)
phi_b = radical_inverse(7, 3)
assert abs(phi_a - phi_b) < 1e-15, '[TC16] radical_inverse FAILED'

# ---- TC17: cg_ne_solve satisfies normal equation A^T A x = A^T b ----
A_cg = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
b_cg = np.array([4.0, 5.0, 3.0])
x_cg, conv_cg, res_cg = cg_ne_solve(A_cg, b_cg, tol=1e-10)
assert np.allclose(A_cg.T @ A_cg @ x_cg, A_cg.T @ b_cg, atol=1e-8), '[TC17] CGNE FAILED'

# ---- TC18: newton_solve finds sqrt(2) from initial guess 1.5 ----
def F_n(x): return np.array([x[0]**2 - 2.0])
def J_n(x): return np.array([[2.0*x[0]]])
x_n, conv_n, _ = newton_solve(F_n, J_n, np.array([1.5]), tol=1e-8)
assert conv_n and abs(x_n[0] - np.sqrt(2.0)) < 1e-6, '[TC18] Newton FAILED'

# ---- TC19: euler_integrate approximates exp(-t) within tolerance ----
def ode_decay(t, y): return np.array([-y[0]])
t_eu, y_eu = euler_integrate(ode_decay, (0.0, 1.0), np.array([1.0]), n_steps=500)
err_eu = np.max(np.abs(y_eu[:, 0] - np.exp(-t_eu)))
assert err_eu < 0.01, '[TC19] Euler FAILED'

# ---- TC20: chiral_luttinger_dispersion is linear in k ----
k_lut = np.array([1.0, 2.0, 3.0])
eps_lut = chiral_luttinger_dispersion(k_lut, 2.0, g_factor=0.5)
assert np.allclose(eps_lut, k_lut * 4.0), '[TC20] Luttinger FAILED'

# ---- TC21: fisher_kpp_exact_solution at t=0, x=0 equals 1/(1+a)^2 ----
u_fk = fisher_kpp_exact_solution(0.0, 0.0, a=1.0)
assert abs(u_fk - 0.25) < 1e-12, '[TC21] fisher_kpp exact FAILED'

# ---- TC22: fisher_kpp_fd_solve produces bounded solution ----
u_fd = fisher_kpp_fd_solve(20, -4.0, 4.0, 20, 0.0, 1.0, D=1.0, r=1.0, K=1.0)
assert u_fd.shape[0] > 0 and u_fd.shape[1] > 0, '[TC22] fisher_kpp shape FAILED'
assert np.all(u_fd >= 0) and np.all(u_fd <= 2.0), '[TC22] fisher_kpp bounds FAILED'

# ---- TC23: Lindblad evolution preserves trace of density matrix ----
H_lb = np.array([[1.0, 0.3], [0.3, -0.5]], dtype=complex)
rho0_lb = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
L_lb = np.array([[0.0, 0.05], [0.0, 0.0]], dtype=complex)
times_lb, rhos_lb = density_matrix_evolution_lindblad(rho0_lb, H_lb, [L_lb], (0.0, 1.0), 50)
tr_final = np.trace(rhos_lb[-1]).real
assert abs(tr_final - 1.0) < 1e-6, '[TC23] Lindblad trace FAILED'

# ---- TC24: fd1d_wave_solve satisfies boundary conditions ----
def ux1(t): return 0.0
def ux2(t): return 0.0
def ut1(x): return np.sin(np.pi * x)
def utt1(x): return np.zeros_like(x)
u_wv = fd1d_wave_solve(20, 0.0, 1.0, 40, 0.0, 1.0, 1.0, ux1, ux2, ut1, utt1)
assert u_wv.shape == (41, 21), '[TC24] wave shape FAILED'
assert abs(u_wv[-1, 0]) < 1e-6 and abs(u_wv[-1, -1]) < 1e-6, '[TC24] wave boundary FAILED'

# ---- TC25: find_nearest_neighbors returns correct indices and distances ----
m_nn, nr_nn, ns_nn = 2, 3, 2
R_nn = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
S_nn = np.array([[0.1, 1.9], [0.0, 0.0]])
idx_nn, dist_nn = find_nearest_neighbors(m_nn, nr_nn, R_nn, ns_nn, S_nn)
assert idx_nn[0] == 0 and idx_nn[1] == 2, '[TC25] nearest neighbor idx FAILED'
assert abs(dist_nn[0] - 0.1) < 1e-12, '[TC25] nearest neighbor dist FAILED'

# ---- TC26: hyperball_distance_stats is reproducible with fixed seed ----
mu1, var1, _ = hyperball_distance_stats(2, 50, seed=42)
mu2, var2, _ = hyperball_distance_stats(2, 50, seed=42)
assert abs(mu1 - mu2) < 1e-12 and abs(var1 - var2) < 1e-12, '[TC26] hyperball repro FAILED'

# ---- TC27: wedge01_integral for (0,0,0) equals unit volume 1 ----
vol = wedge01_integral([0, 0, 0])
assert abs(vol - 1.0) < 1e-12, '[TC27] wedge01 volume FAILED'

# ---- TC28: 2D Gauss-Legendre integrates x^2*y^2 exactly on [0,1]^2 ----
def f28(x, y): return x**2 * y**2
val28 = multidimensional_gauss_legendre(f28, 2, 4, [(0.0, 1.0), (0.0, 1.0)])
assert abs(val28 - 1.0/9.0) < 1e-12, '[TC28] 2D Gauss FAILED'

# ---- TC29: flux_quantization_phase for n_phi=3, n_e=1 gives phase=2*pi/3 ----
phase, charge = flux_quantization_phase(3, 1, m_laughlin=3)
assert abs(phase - 2.0*np.pi/3.0) < 1e-12, '[TC29] flux phase FAILED'
assert abs(charge - 1.0/3.0) < 1e-12, '[TC29] flux charge FAILED'

# ---- TC30: orbital_evolution_parameters returns arrays of length n_points ----
tau, theta, x2, y2, z2 = orbital_evolution_parameters(0.1, 30.0, 20.0, n_points=60)
assert len(tau) == 60 and len(theta) == 60, '[TC30] orbital length FAILED'

# ---- TC31: qmc_integration of constant function equals domain_volume ----
pts_qmc = hammersley_sequence(0, 99, 2)
def f_const(pt): return 1.0
I_qmc = qmc_integration(f_const, pts_qmc, domain_volume=3.5)
assert abs(I_qmc - 3.5) < 1e-12, '[TC31] QMC constant FAILED'

# ---- TC32: two_point_correlation g2 is non-negative ----
np.random.seed(42)
z_corr = np.array([0.5+0j, -0.5+0j, 0.0+0.5j, 0.0-0.5j])
r_e2, g2, r_c2 = two_point_correlation(z_corr, 1.0, r_bins=10)
assert np.all(g2 >= 0), '[TC32] two_point_correlation FAILED'

# ---- TC33: spectral stiffness matrix is symmetric ----
K_mat = build_spectral_stiffness_matrix(6, domain=(0.0, 1.0))
assert np.allclose(K_mat, K_mat.T), '[TC33] stiffness symmetry FAILED'

# ---- TC34: Lagrange basis equals identity at nodes ----
nodes = np.array([0.0, 0.5, 1.0])
phi_nodes = local_basis_1d_lagrange(3, nodes, nodes)
assert np.allclose(phi_nodes, np.eye(3)), '[TC34] Lagrange Kronecker FAILED'

# ---- TC35: coulomb_interaction_2d is finite at r=0 due to cutoff ----
V0 = coulomb_interaction_2d(0.0, epsilon_r=12.0)
assert np.isfinite(V0), '[TC35] Coulomb r=0 FAILED'

# ---- TC36: edge_state_density_of_states is non-negative ----
omega_test = np.linspace(-1.0, 1.0, 20)
dos_edge = edge_state_density_of_states(omega_test, 1.0, 10.0, T=0.01)
assert np.all(dos_edge >= 0), '[TC36] edge DOS negative FAILED'

# ---- TC37: tknn_conductivity matches chern_number_from_berry_curvature ----
Omega_dummy = np.ones((2, 2))
C_dummy = chern_number_from_berry_curvature(Omega_dummy, 1.0, 1.0)
sigma_dummy = tknn_conductivity(np.sum(Omega_dummy), 1.0, 1.0)
assert abs(sigma_dummy - C_dummy) < 1e-12, '[TC37] TKNN consistency FAILED'

# ---- TC38: berry_connection returns finite for normalized states ----
u_a = np.array([1.0, 0.0])
u_b = np.array([0.0, 1.0])
A_conn = berry_connection(u_a, u_b, 0.1)
assert np.isfinite(A_conn), '[TC38] berry_connection FAILED'

# ---- TC39: quasihole_wavefunction with return_log returns complex ----
np.random.seed(42)
z_qh = np.array([1.0+0j, 0.5+0.5j])
log_psi_qh = quasihole_wavefunction(z_qh, 0.2+0.1j, 3, 1.0, return_log=True)
assert isinstance(log_psi_qh, (complex, np.complexfloating)), '[TC39] quasihole type FAILED'

# ---- TC40: hypercube_surface_distance_stats reproducible with fixed seed ----
mu1_c, var1_c, _ = hypercube_surface_distance_stats(2, 50, seed=42)
mu2_c, var2_c, _ = hypercube_surface_distance_stats(2, 50, seed=42)
assert abs(mu1_c - mu2_c) < 1e-12, '[TC40] hypercube repro FAILED'
