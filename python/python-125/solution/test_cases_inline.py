# ---- TC01: simpson_integration analytic validation ∫_0^1 sin²(πx) dx = 0.5 ----
def f_sin2(x):
    return np.sin(np.pi * x) ** 2
result_simpson = simpson_integration(f_sin2, 0.0, 1.0, n=1000)
assert abs(result_simpson - 0.5) < 1e-6, '[TC01] simpson_integration sin²(πx) FAILED'

# ---- TC02: trapezoidal_integration returns finite non-negative for non-negative y ----
x_trap = np.linspace(0.0, 1.0, 100)
y_trap = np.sin(np.pi * x_trap) ** 2
result_trap = trapezoidal_integration(x_trap, y_trap)
assert np.isfinite(result_trap), '[TC02] trapezoidal_integration not finite FAILED'
assert result_trap >= 0.0, '[TC02] trapezoidal_integration negative FAILED'

# ---- TC03: relative_error exact match returns zero ----
assert relative_error(1.5, 1.5) == 0.0, '[TC03] relative_error exact match FAILED'

# ---- TC04: erf_approx(0) ≈ 0 ----
assert abs(erf_approx(0.0)) < 1e-6, '[TC04] erf_approx(0) FAILED'

# ---- TC05: erf_approx(1.0) ≈ 0.8427 ----
erf1 = erf_approx(1.0)
assert abs(erf1 - 0.8427007929) < 0.005, '[TC05] erf_approx(1) FAILED'

# ---- TC06: log_gamma_lanczos(5) ≈ ln(24) ----
lg5 = log_gamma_lanczos(5.0)
assert abs(lg5 - np.log(24.0)) < 1e-6, '[TC06] log_gamma_lanczos(5) FAILED'

# ---- TC07: spherical_harmonic_y Y_0^0 = 1/√(4π) ----
Y00 = spherical_harmonic_y(0, 0, 0.0, 0.0)
expected_Y00 = 1.0 / np.sqrt(4.0 * np.pi)
assert abs(Y00.real - expected_Y00) < 1e-10, '[TC07] spherical_harmonic Y_0^0 FAILED'
assert abs(Y00.imag) < 1e-10, '[TC07] spherical_harmonic imag part FAILED'

# ---- TC08: legendre_polynomial_value P_0(x)=1 for all x ----
x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
V_leg = legendre_polynomial_value(0, x_test)
assert np.allclose(V_leg[:, 0], 1.0), '[TC08] legendre P_0 FAILED'

# ---- TC09: legendre_polynomial_value P_1(x)=x ----
V_leg1 = legendre_polynomial_value(1, x_test)
assert np.allclose(V_leg1[:, 1], x_test), '[TC09] legendre P_1 identity FAILED'

# ---- TC10: dog_receptive_field returns ndarray of correct shape ----
x_rf, y_rf = np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)
X_rf, Y_rf = np.meshgrid(x_rf, y_rf)
rf = dog_receptive_field(X_rf, Y_rf, A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
assert isinstance(rf, np.ndarray), '[TC10] dog_receptive_field type FAILED'
assert rf.shape == (32, 32), '[TC10] dog_receptive_field shape FAILED'

# ---- TC11: dog_receptive_field center value = A_c - A_s ----
rf_center = dog_receptive_field(np.array([[0.0]]), np.array([[0.0]]), A_c=1.0, sigma_c=0.15, A_s=0.5, sigma_s=0.40)
assert abs(rf_center[0, 0] - 0.5) < 1e-10, '[TC11] dog_receptive_field center FAILED'

# ---- TC12: gauss_legendre_quadrature weights sum to 2 ----
x_gl, w_gl = gauss_legendre_quadrature(16)
assert abs(np.sum(w_gl) - 2.0) < 1e-10, '[TC12] gauss_legendre weights sum FAILED'

# ---- TC13: generate_hexagonal_photoreceptor_array correct count ----
pts_hex = generate_hexagonal_photoreceptor_array(1.0, 2)
assert pts_hex.shape[0] == 1 + 6 + 12, '[TC13] hex array count FAILED'
assert pts_hex.shape[1] == 2, '[TC13] hex array shape FAILED'

# ---- TC14: hexagon_moment_integral symmetry M(2,0) = M(0,2) ----
hex_verts = np.array([
    [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0], [-0.5, np.sqrt(3.0)/2.0],
    [-1.0, 0.0], [-0.5, -np.sqrt(3.0)/2.0], [0.5, -np.sqrt(3.0)/2.0]
])
M20 = hexagon_moment_integral(2, 0, hex_verts)
M02 = hexagon_moment_integral(0, 2, hex_verts)
assert abs(M20 - M02) < 1e-10, '[TC14] hexagon moment symmetry FAILED'

# ---- TC15: convex_hull_2d returns non-empty result ----
hull = convex_hull_2d(pts_hex)
assert len(hull) >= 3, '[TC15] convex_hull_2d result FAILED'

# ---- TC16: evaluate_mesh_quality returns valid quality metrics ----
triangles = delaunay_triangulation_2d(pts_hex)
quality = evaluate_mesh_quality(pts_hex, triangles)
assert isinstance(quality, dict), '[TC16] mesh quality type FAILED'
assert 0.0 <= quality['alpha_ave'] <= 1.0 + 1e-10, '[TC16] mesh quality alpha range FAILED'
assert quality['num_triangles'] > 0, '[TC16] mesh quality num_triangles FAILED'

# ---- TC17: horner_polynomial_eval constant polynomial ----
p_const = horner_polynomial_eval(np.array([3.0]), np.array([0.0, 1.0, -5.0]))
assert np.allclose(p_const, 3.0), '[TC17] horner constant FAILED'

# ---- TC18: trig_interp_basis τ_k(0) = 1 ----
tau_zero = trig_interp_basis(np.array([0.0]), 7)
assert abs(tau_zero[0] - 1.0) < 1e-12, '[TC18] trig_interp_basis at origin FAILED'

# ---- TC19: encoding_efficiency zero spike train gives zero info ----
spikes_zero = np.zeros(1000)
stim_const = np.ones(1000)
eff_zero = encoding_efficiency(spikes_zero, stim_const, 0.001)
assert eff_zero['info_rate_bits'] == 0.0, '[TC19] encoding_efficiency zero FAILED'

# ---- TC20: faure_generate output in [0,1] range ----
faure_pts = faure_generate(3, 20, skip=50)
assert np.all(faure_pts >= 0.0) and np.all(faure_pts <= 1.0), '[TC20] faure_generate range FAILED'
assert faure_pts.shape == (3, 20), '[TC20] faure_generate shape FAILED'

# ---- TC21: generate_inhomogeneous_poisson_spikes seed reproducibility ----
rate_fn = lambda t: 30.0 + 20.0 * np.sin(2.0 * np.pi * 5.0 * t)
spikes1 = generate_inhomogeneous_poisson_spikes(rate_fn, 0.0, 1.0, dt=0.001, max_rate=55.0, seed=42)
spikes2 = generate_inhomogeneous_poisson_spikes(rate_fn, 0.0, 1.0, dt=0.001, max_rate=55.0, seed=42)
assert np.array_equal(spikes1, spikes2), '[TC21] poisson spikes reproducibility FAILED'

# ---- TC22: band_lu_factorize and band_lu_solve on identity-like diagonal matrix ----
ml_test, mu_test, n_test = 2, 2, 10
A_test = np.zeros((ml_test + mu_test + 1, n_test))
A_test[mu_test, :] = 1.0
A_lu, info = band_lu_factorize(A_test.copy(), n_test, ml_test, mu_test)
assert info == 0, '[TC22] band_lu_factorize info FAILED'
b_test = np.ones(n_test)
x_test2 = band_lu_solve(A_lu, b_test, n_test, ml_test, mu_test)
assert np.allclose(x_test2, 1.0), '[TC22] band_lu_solve identity FAILED'

# ---- TC23: build_retinal_network_matrix correct shapes ----
A1b, A2m, A3m, A4m = build_retinal_network_matrix(20, 5, connectivity_radius=2.0)
assert A1b.shape == (5, 20), '[TC23] A1_band shape FAILED'
assert A2m.shape == (20, 5), '[TC23] A2 shape FAILED'
assert A3m.shape == (5, 20), '[TC23] A3 shape FAILED'
assert A4m.shape == (5, 5), '[TC23] A4 shape FAILED'

# ---- TC24: sinusoidal_grating output in [0,1] range ----
grating_test = sinusoidal_grating(32, 32, spatial_freq=3.0, orientation=np.pi/4.0)
assert grating_test.shape == (32, 32), '[TC24] sinusoidal_grating shape FAILED'
assert np.all(grating_test >= 0.0) and np.all(grating_test <= 1.0), '[TC24] sinusoidal_grating range FAILED'

# ---- TC25: gaussian_blob center is maximum ----
blob_test = gaussian_blob(31, 31, sigma_x=0.3, sigma_y=0.3, center_x=0.0, center_y=0.0)
assert blob_test.shape == (31, 31), '[TC25] gaussian_blob shape FAILED'
assert abs(blob_test.max() - 1.0) < 1e-10, '[TC25] gaussian_blob center max FAILED'

# ---- TC26: white_noise_stimulus reproducibility ----
noise1 = white_noise_stimulus(16, 16, seed=42)
noise2 = white_noise_stimulus(16, 16, seed=42)
assert np.array_equal(noise1, noise2), '[TC26] white_noise reproducibility FAILED'

# ---- TC27: integrate_photocurrent_clenshaw_curtis constant function ----
def const_fn(x):
    return 3.0
integral_const = integrate_photocurrent_clenshaw_curtis(const_fn, -1.0, 1.0, n=32)
assert abs(integral_const - 6.0) < 1e-6, '[TC27] clenshaw_curtis constant FAILED'

# ---- TC28: compute_synaptic_efficacy on zero field returns zero total_efficacy ----
zero_v = np.zeros((8, 8))
eff_syn = compute_synaptic_efficacy(zero_v, threshold=0.1)
assert eff_syn['total_efficacy'] == 0.0, '[TC28] synaptic efficacy zero FAILED'
assert eff_syn['active_fraction'] == 0.0, '[TC28] synaptic active_fraction FAILED'

# ---- TC29: solve_light_adaptation_steady_state returns finite values and correct shape ----
src = np.zeros((8, 8))
C_steady, iters, err = solve_light_adaptation_steady_state(
    8, 8, 10.0, 10.0, 10.0, 10.0, src, epsilon=1e-8, max_iter=5000
)
assert C_steady.shape == (8, 8), '[TC29] light adaptation shape FAILED'
assert np.all(np.isfinite(C_steady)), '[TC29] light adaptation finite FAILED'
assert iters > 0, '[TC29] light adaptation iterations FAILED'

# ---- TC30: decompose_rf_with_legendre_basis returns expected number of coefficients ----
def rf_test(x):
    return np.exp(-x**2 / 0.1)
coeffs_rf = decompose_rf_with_legendre_basis(rf_test, max_degree=8, n_quad=32)
assert len(coeffs_rf) == 9, '[TC30] decompose_rf coeff count FAILED'
assert np.all(np.isfinite(coeffs_rf)), '[TC30] decompose_rf finite FAILED'

# ---- TC31: reconstruct_rf_from_legendre_coeffs returns finite values ----
x_rec = np.linspace(-1.0, 1.0, 50)
rf_rec = reconstruct_rf_from_legendre_coeffs(coeffs_rf, x_rec)
assert len(rf_rec) == 50, '[TC31] reconstruct_rf shape FAILED'
assert np.all(np.isfinite(rf_rec)), '[TC31] reconstruct_rf finite FAILED'

# ---- TC32: compute_bipolar_response_convolution returns scalar ----
grating_small = sinusoidal_grating(16, 16, spatial_freq=2.0, orientation=0.0)
rf_params_test = {'A_c': 1.0, 'sigma_c': 0.15, 'A_s': 0.5, 'sigma_s': 0.40}
response_bp = compute_bipolar_response_convolution(grating_small, rf_params_test, grid_spacing=0.1)
assert isinstance(response_bp, float), '[TC32] bipolar response type FAILED'
assert np.isfinite(response_bp), '[TC32] bipolar response finite FAILED'

# ---- TC33: spherical_harmonic Y_1^0(π/2, 0) = 0 (cos(π/2)=0) ----
Y10 = spherical_harmonic_y(1, 0, np.pi/2.0, 0.0)
assert abs(Y10.real) < 1e-10, '[TC33] spherical_harmonic Y_1^0 FAILED'

# ---- TC34: drifting_grating correct shape ----
drift_test = drifting_grating(8, 8, n_frames=10, spatial_freq=2.0, temporal_freq=4.0, orientation=0.0, dt=0.01)
assert drift_test.shape == (10, 8, 8), '[TC34] drifting_grating shape FAILED'

# ---- TC35: matrix_condition_number_estimate on identity returns near 1 ----
I_mat = np.eye(5)
cond_I = matrix_condition_number_estimate(I_mat)
assert 0.5 < cond_I < 2.0, '[TC35] condition number identity FAILED'

# ---- TC36: sample_neural_parameter_space returns dict with correct keys ----
param_ranges_test = {
    'freq': (0.5, 10.0),
    'contrast': (0.1, 1.0),
}
np.random.seed(42)
samples_test = sample_neural_parameter_space(n_samples=10, param_ranges=param_ranges_test, skip=50)
assert 'freq' in samples_test, '[TC36] param space key FAILED'
assert len(samples_test['freq']) == 10, '[TC36] param space length FAILED'

# ---- TC37: explore_synaptic_combinations correct shape ----
combos_test = explore_synaptic_combinations(n_synapses=10, subset_size=3, n_combinations=5)
assert combos_test.shape == (5, 3), '[TC37] explore combos shape FAILED'

# ---- TC38: evaluate_tuning_function returns finite array ----
tune_vals = evaluate_tuning_function('x^2', np.array([0.0, 1.0, 2.0]))
assert np.all(np.isfinite(tune_vals)), '[TC38] evaluate_tuning finite FAILED'

# ---- TC39: simpson_integration returns float scalar ----
result_s_type = simpson_integration(lambda x: x, 0.0, 1.0, n=100)
assert isinstance(result_s_type, float), '[TC39] simpson return type FAILED'

# ---- TC40: trig_interpolate_spike_pattern returns correct length ----
st_times = np.array([0.0, 0.5])
st_vals = np.array([1.0, 0.5])
t_eval_test = np.linspace(0.0, 2.0, 20)
interp_res = trig_interpolate_spike_pattern(st_times, st_vals, t_eval_test, period=2.0)
assert len(interp_res) == 20, '[TC40] trig_interpolate shape FAILED'
assert np.all(np.isfinite(interp_res)), '[TC40] trig_interpolate finite FAILED'

# ---- TC41: analyze_spike_train basic stats for simple spike train ----
simple_spikes = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
stats_test = analyze_spike_train(simple_spikes, 1.0, n_bins=10)
assert stats_test['n_spikes'] == 5, '[TC41] analyze_spike_train count FAILED'
assert stats_test['mean_rate'] == 5.0, '[TC41] analyze_spike_train rate FAILED'

# ---- TC42: simulate_rgc_spike_train seed reproducibility ----
np.random.seed(42)
bp_test = np.random.random(100)
t_test = np.linspace(0.0, 0.1, 100)
spikes_a = simulate_rgc_spike_train(bp_test, t_test, seed=42)
np.random.seed(42)
bp_test2 = np.random.random(100)
spikes_b = simulate_rgc_spike_train(bp_test2, t_test, seed=42)
assert np.array_equal(spikes_a, spikes_b), '[TC42] rgc spike reproducibility FAILED'

# ---- TC43: simulate_synaptic_transmission returns dict with expected keys ----
syn_result = simulate_synaptic_transmission(
    nx=16, ny=16, n_steps=200,
    Du=0.16, Dv=0.08, F=0.035, K=0.060,
    dt=0.5, dx=0.5, dy=0.5,
    initial_condition='localized', boundary='periodic'
)
assert 'final_U' in syn_result, '[TC43] synaptic dict U FAILED'
assert 'final_V' in syn_result, '[TC43] synaptic dict V FAILED'
assert 'n_steps' in syn_result, '[TC43] synaptic dict n_steps FAILED'

# ---- TC44: solve_phototransduction_rk4 integration returns finite values ----
def light_zero(t):
    return 0.0
y0_test = np.array([0.0, 0.5, 0.1])
params_test = {
    'alpha_pde': 2.0, 'beta_pde': 5.0, 'alpha_gc_max': 10.0,
    'K_gc': 0.1, 'n_gc': 4.0, 'gamma': 1.0, 'eta': 1.0,
    'g_max': 1.0, 'K_cGMP': 0.05, 'V_m': -40.0, 'E_Ca': 40.0,
}
t_arr, y_arr = solve_phototransduction_rk4(light_zero, y0_test, (0.0, 0.1), 0.001, params_test)
assert len(t_arr) > 0, '[TC44] rk4 timesteps FAILED'
assert np.all(np.isfinite(y_arr)), '[TC44] rk4 finite FAILED'
assert y_arr.shape[1] == 3, '[TC44] rk4 3 vars FAILED'

# ---- TC45: log_gamma_lanczos returns finite for positive input ----
lg_pos = log_gamma_lanczos(3.5)
assert np.isfinite(lg_pos), '[TC45] log_gamma finite FAILED'
assert lg_pos > 0.0, '[TC45] log_gamma positive FAILED'

# ---- TC46: spherical_harmonic returns complex type ----
Y_22 = spherical_harmonic_y(2, 1, np.pi/3.0, np.pi/4.0)
assert isinstance(Y_22, complex), '[TC46] spherical_harmonic type FAILED'

# ---- TC47: simpson_integration of x^2 from 0 to 1 = 1/3 ----
result_x2 = simpson_integration(lambda x: x**2, 0.0, 1.0, n=1000)
assert abs(result_x2 - 1.0/3.0) < 1e-8, '[TC47] simpson x^2 FAILED'

# ---- TC48: gauss_legendre_quadrature nodes are within [-1, 1] ----
x_gl32, w_gl32 = gauss_legendre_quadrature(32)
assert np.all(x_gl32 >= -1.0) and np.all(x_gl32 <= 1.0), '[TC48] gauss_legendre nodes range FAILED'
assert np.all(w_gl32 > 0.0), '[TC48] gauss_legendre weights positive FAILED'

# ---- TC49: horner_polynomial_eval quadratic x^2 ----
p_quad = horner_polynomial_eval(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0]))
assert np.allclose(p_quad, np.array([0.0, 1.0, 4.0, 9.0])), '[TC49] horner quadratic FAILED'

# ---- TC50: build_retinal_network_matrix A4 is square ----
_, _, _, A4_test = build_retinal_network_matrix(15, 5, connectivity_radius=2.0)
assert A4_test.shape == (5, 5), '[TC50] A4 square FAILED'

# ---- TC51: solve_cbb_system returns vector of correct length ----
np.random.seed(123)
A1b_cbb, A2_cbb, A3_cbb, A4_cbb = build_retinal_network_matrix(10, 3, connectivity_radius=2.0)
b_cbb = np.random.random(13)
x_cbb = solve_cbb_system(A1b_cbb, A2_cbb, A3_cbb, A4_cbb, b_cbb, 10, 3, ml=2, mu=2)
assert len(x_cbb) == 13, '[TC51] solve_cbb length FAILED'
assert np.all(np.isfinite(x_cbb)), '[TC51] solve_cbb finite FAILED'

# ---- TC52: faure_generate different dimensions ----
faure_5d = faure_generate(5, 8, skip=20)
assert faure_5d.shape == (5, 8), '[TC52] faure 5d shape FAILED'
assert np.all(faure_5d >= 0.0) and np.all(faure_5d <= 1.0), '[TC52] faure 5d range FAILED'

# ---- TC53: white_noise_stimulus returns values in [0,1] ----
noise_range = white_noise_stimulus(20, 20, seed=123)
assert np.all(noise_range >= 0.0) and np.all(noise_range <= 1.0), '[TC53] white_noise range FAILED'

# ---- TC54: drifting_grating values in [0,1] ----
drift_range = drifting_grating(8, 8, n_frames=5, spatial_freq=2.0, temporal_freq=4.0, orientation=0.0, dt=0.01)
assert np.all(drift_range >= 0.0) and np.all(drift_range <= 1.0), '[TC54] drifting_grating range FAILED'

# ---- TC55: matrix_condition_number_estimate returns finite ----
A_rand = np.random.random((4, 4)) + 10.0 * np.eye(4)
cond_rand = matrix_condition_number_estimate(A_rand)
assert np.isfinite(cond_rand), '[TC55] condition number finite FAILED'
assert cond_rand > 0.0, '[TC55] condition number positive FAILED'

# ---- TC56: compute_synaptic_efficacy on non-zero field returns positive total_efficacy ----
v_nonzero = np.ones((8, 8)) * 0.5
eff_nonzero = compute_synaptic_efficacy(v_nonzero, threshold=0.1)
assert eff_nonzero['total_efficacy'] > 0.0, '[TC56] synaptic efficacy positive FAILED'
assert eff_nonzero['active_fraction'] > 0.0, '[TC56] synaptic active_fraction positive FAILED'
assert eff_nonzero['peak_concentration'] == 0.5, '[TC56] synaptic peak concentration FAILED'

# ---- TC57: simpson_integration of cos from 0 to π/2 = 1 ----
result_cos = simpson_integration(lambda x: np.cos(x), 0.0, np.pi/2.0, n=1000)
assert abs(result_cos - 1.0) < 1e-7, '[TC57] simpson cos FAILED'

# ---- TC58: erf_approx is odd function: erf(-x) = -erf(x) ----
erf_neg1 = erf_approx(-1.0)
erf_pos1 = erf_approx(1.0)
assert abs(erf_neg1 + erf_pos1) < 1e-6, '[TC58] erf odd symmetry FAILED'
