# ---- TC01: BinaryBlackHole chirp mass analytic check ----
bbh = BinaryBlackHole(m1_msun=36.0, m2_msun=29.0)
geom_to_msun = bbh.MSUN_SI * bbh.G_SI / bbh.C_SI**3
expected_Mc = (36.0 * 29.0)**(3.0/5.0) / (65.0)**(1.0/5.0)
assert abs(bbh.M_c / geom_to_msun - expected_Mc) < 1e-6, '[TC01] BinaryBlackHole chirp mass FAILED'

# ---- TC02: effective_spin equals weighted average ----
chi = effective_spin(30.0, 20.0, 0.5, -0.3)
expected_chi = (30.0*0.5 + 20.0*(-0.3)) / 50.0
assert abs(chi - expected_chi) < 1e-12, '[TC02] effective_spin FAILED'

# ---- TC03: final_mass_spin mass conservation ----
M_f, a_f = final_mass_spin(36.0, 29.0, 0.3, -0.2)
assert M_f < 65.0 and M_f > 0, '[TC03] final mass out of range FAILED'
assert a_f >= 0.0 and a_f <= 0.99, '[TC03] final spin out of range FAILED'

# ---- TC04: solve_qnm_frequencies contains (2,2,0) with positive real part ----
qnm = solve_qnm_frequencies(l_max=3, n_overtones=1, M=1.0, a=0.0)
assert (2, 2, 0) in qnm, '[TC04] QNM (2,2,0) missing FAILED'
assert qnm[(2, 2, 0)].real > 0, '[TC04] QNM real part not positive FAILED'

# ---- TC05: gravitational_wave_luminosity returns finite values ----
lum, lum_dimless = gravitational_wave_luminosity(qnm, M=65.0, a=0.0)
assert np.isfinite(lum), '[TC05] luminosity not finite FAILED'
assert np.isfinite(lum_dimless), '[TC05] luminosity dimless not finite FAILED'

# ---- TC06: conformal_factor approaches 1 far from sources ----
masses = np.array([1.0, 1.0])
positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
psi_far = conformal_factor_brill_lindquist(100.0, 100.0, 0.0, masses, positions)
assert abs(psi_far - 1.0) < 0.05, '[TC06] conformal factor far field FAILED'

# ---- TC07: full_imrphenom_waveform output length matches input ----
t_test = np.linspace(-1.0, 0.0, 100)
h_p, h_c = full_imrphenom_waveform(t_test, 36.0, 29.0, 400.0)
assert len(h_p) == 100 and len(h_c) == 100, '[TC07] waveform output length FAILED'
assert np.all(np.isfinite(h_p)) and np.all(np.isfinite(h_c)), '[TC07] waveform contains non-finite FAILED'

# ---- TC08: shifted_legendre_polynomial P0 is all ones ----
x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
leg = shifted_legendre_polynomial(x, n_max=0)
assert np.allclose(leg[:, 0], 1.0), '[TC08] Legendre P0 not all ones FAILED'

# ---- TC09: shifted_legendre_polynomial P1 at 0.5 is zero ----
leg2 = shifted_legendre_polynomial(np.array([0.5]), n_max=1)
assert abs(leg2[0, 1]) < 1e-12, '[TC09] Legendre P1 at 0.5 not zero FAILED'

# ---- TC10: waveform_inner_product_chebyshev symmetry ----
def h1(f): return np.ones_like(f)
def h2(f): return 2.0 * np.ones_like(f)
ip1 = waveform_inner_product_chebyshev(h1, h2, 1.0, 10.0, n=16)
ip2 = waveform_inner_product_chebyshev(h2, h1, 1.0, 10.0, n=16)
assert abs(ip1 - ip2) < 1e-10, '[TC10] inner product symmetry FAILED'

# ---- TC11: compute_extrinsic_curvature trace-free for flat psi ----
psi_test = np.ones((5, 5))
K = compute_extrinsic_curvature(psi_test, h=0.1)
assert np.allclose(K['trace'], 0.0), '[TC11] extrinsic curvature trace not zero FAILED'

# ---- TC12: solve_initial_data_brill_lindquist psi is finite ----
idata = solve_initial_data_brill_lindquist(nx=9, ny=9, h=0.5)
assert np.all(np.isfinite(idata['psi'])), '[TC12] initial data psi not finite FAILED'

# ---- TC13: antenna_pattern_functions reproducible with same args ----
arm1 = np.array([1.0, 0.0, 0.0])
arm2 = np.array([0.0, 1.0, 0.0])
Fp1, Fc1 = antenna_pattern_functions(0.5, 0.3, 0.1, arm1, arm2)
Fp2, Fc2 = antenna_pattern_functions(0.5, 0.3, 0.1, arm1, arm2)
assert abs(Fp1 - Fp2) < 1e-15 and abs(Fc1 - Fc2) < 1e-15, '[TC13] antenna pattern not reproducible FAILED'

# ---- TC14: network_snr is zero for zero signal ----
h_p_zero = np.zeros(10)
h_c_zero = np.zeros(10)
rho_zero = network_snr([0.5, 0.3], [0.2, 0.1], h_p_zero, h_c_zero, noise_psd=1.0)
assert rho_zero == 0.0, '[TC14] network SNR for zero signal not zero FAILED'

# ---- TC15: compute_sky_position returns unit vector ----
network = get_standard_detector_network()
positions = np.array([d['position'] for d in network])
dt = np.array([0.0, 0.0, 0.0])
lat, lon, n_vec = compute_sky_position_from_time_delays(positions, dt, radius=1.0)
assert abs(np.linalg.norm(n_vec) - 1.0) < 1e-6, '[TC15] sky position not unit vector FAILED'

# ---- TC16: GWPrior.log_prior returns -inf for out-of-bound mass ----
prior = GWPrior(m_min=5.0, m_max=100.0)
bad_params = {'m1': 1.0, 'm2': 30.0, 'a1': 0.0, 'a2': 0.0, 'D_L': 200.0, 'inclination': 0.5, 'phi_c': 0.0, 'psi': 0.0, 't_c': 0.0}
assert prior.log_prior(bad_params) == -np.inf, '[TC16] prior out-of-bounds not -inf FAILED'

# ---- TC17: metropolis_hastings reproducible with fixed seed ----
np.random.seed(42)
def dummy_log_post(p):
    return -0.5 * ((p['m1'] - 30.0)**2 + (p['m2'] - 25.0)**2)
init_p = {'m1': 32.0, 'm2': 27.0, 'a1': 0.0, 'a2': 0.0, 'D_L': 400.0, 'inclination': 0.0, 'phi_c': 0.0, 'psi': 0.0, 't_c': 0.0}
s1, _ = metropolis_hastings(dummy_log_post, init_p, n_steps=100, step_sizes={'m1': 1.0, 'm2': 1.0})
np.random.seed(42)
s2, _ = metropolis_hastings(dummy_log_post, init_p, n_steps=100, step_sizes={'m1': 1.0, 'm2': 1.0})
assert s1[-1]['m1'] == s2[-1]['m1'] and s1[-1]['m2'] == s2[-1]['m2'], '[TC17] MCMC not reproducible FAILED'

# ---- TC18: marginal_posterior_mass_plane evidence positive ----
def log_post_simple(m1, m2):
    return -0.5 * ((m1 - 30.0)**2 + (m2 - 25.0)**2)
pts, post, ev = marginal_posterior_mass_plane(log_post_simple, (20.0, 40.0), (15.0, 35.0), order=3)
assert ev > 0, '[TC18] evidence not positive FAILED'

# ---- TC19: euclidean_gcd matches manual calculation ----
assert euclidean_gcd(48, 18) == 6, '[TC19] euclidean_gcd FAILED'

# ---- TC20: rational_approximation recovers exact fraction ----
p, q = rational_approximation(0.75, max_denominator=100)
assert abs(p / q - 0.75) < 1e-12, '[TC20] rational_approximation FAILED'

# ---- TC21: check_finite returns True for finite array ----
assert check_finite(np.array([1.0, 2.0, 3.0]), "test") == True, '[TC21] check_finite FAILED'

# ---- TC22: run_stability_tests returns dict with all_pass ----
results = run_stability_tests()
assert isinstance(results, dict), '[TC22] stability tests return type FAILED'
assert 'all_pass' in results, '[TC22] stability tests missing all_pass FAILED'

# ---- TC23: evolve_binary_orbit energy decreases due to radiation ----
t_orb, traj, energy = evolve_binary_orbit(m1=1.0, m2=1.0, initial_separation=10.0, t_span=(0.0, 50.0), n_steps=1000)
assert energy[-1] < energy[0], '[TC23] orbit energy not decreasing FAILED'

# ---- TC24: get_standard_detector_network has three detectors ----
net = get_standard_detector_network()
assert len(net) == 3, '[TC24] detector network size not 3 FAILED'

# ---- TC25: BinaryBlackHole parameter dict round-trip ----
bbh_orig = BinaryBlackHole(m1_msun=40.0, m2_msun=30.0, a1=0.5, a2=-0.3)
params = bbh_orig.to_parameter_dict()
bbh_recon = BinaryBlackHole.from_parameter_dict(params)
assert abs(bbh_recon.m1_msun - bbh_orig.m1_msun) < 1e-12, '[TC25] parameter dict round-trip m1 FAILED'
assert abs(bbh_recon.m2_msun - bbh_orig.m2_msun) < 1e-12, '[TC25] parameter dict round-trip m2 FAILED'
assert abs(bbh_recon.a1 - bbh_orig.a1) < 1e-12, '[TC25] parameter dict round-trip a1 FAILED'

# ---- TC26: spherical_distance symmetric in arguments ----
from detector import spherical_distance
d1 = spherical_distance(0.1, 0.2, 0.3, 0.4, radius=1.0)
d2 = spherical_distance(0.3, 0.4, 0.1, 0.2, radius=1.0)
assert abs(d1 - d2) < 1e-12, '[TC26] spherical_distance not symmetric FAILED'

# ---- TC27: log_normal_pdf analytic normalization via quadrature ----
from bayesian import log_normal_pdf
x_grid = np.linspace(0.01, 20.0, 2000)
dx = x_grid[1] - x_grid[0]
pdf_vals = np.array([log_normal_pdf(x, mu=1.5, sigma=0.5) for x in x_grid])
norm_approx = np.trapezoid(pdf_vals, x_grid)
assert abs(norm_approx - 1.0) < 0.05, '[TC27] log_normal_pdf normalization FAILED'

# ---- TC28: teukolsky_potential finite at large radius ----
from teukolsky import teukolsky_potential
V_far = teukolsky_potential(100.0, M=1.0, a=0.0, omega=0.3, m=2)
assert np.isfinite(V_far), '[TC28] teukolsky_potential not finite at large r FAILED'

# ---- TC29: burgers_godunov conserves total mass for periodic BC ----
from utils import burgers_godunov
x_burg, U_burg = burgers_godunov(lambda x: np.sin(2*np.pi*x), nx=64, nt=50, t_max=0.2, bc_type='periodic')
conservation_err = abs(np.sum(U_burg[-1, :]) - np.sum(U_burg[0, :]))
assert conservation_err < 1e-10, '[TC29] burgers_godunov conservation FAILED'

# ---- TC30: implicit_midpoint_integrator energy conservation for harmonic oscillator ----
from utils import implicit_midpoint_integrator
def harm_deriv(t, y):
    return np.array([y[1], -y[0]])
t_int, y_int = implicit_midpoint_integrator(harm_deriv, (0.0, 10.0), np.array([1.0, 0.0]), n_steps=500)
E = 0.5 * y_int[:, 1]**2 + 0.5 * y_int[:, 0]**2
drift = np.max(np.abs(E - E[0]))
assert drift < 1e-10, '[TC30] implicit midpoint energy conservation FAILED'
