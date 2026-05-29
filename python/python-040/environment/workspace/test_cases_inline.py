# ---- TC01: ZPrimeModel stores attributes correctly ----
model = ZPrimeModel(mass=3000.0, total_width=90.0, gq_coupling=0.2, gl_coupling=0.1, gq_axial=0.05, gl_axial=0.02, chi=0.01)
assert model.mass == 3000.0, '[TC01] ZPrimeModel mass FAILED'
assert model.gl_coupling == 0.1, '[TC01] ZPrimeModel gl_coupling FAILED'

# ---- TC02: breit_wigner_propagator finite at resonance ----
s_peak = model.mass ** 2
prop = breit_wigner_propagator(s_peak, model)
assert np.isfinite(prop) and abs(prop) > 0, '[TC02] BW propagator finite FAILED'

# ---- TC03: breit_wigner_propagator returns zero for non-physical s ----
prop_zero = breit_wigner_propagator(-100.0, model)
assert prop_zero == 0.0 + 0.0j, '[TC03] BW propagator non-physical FAILED'

# ---- TC04: dilepton_cross_section shape and non-negative ----
s_test = np.array([model.mass ** 2])
ct_test = np.linspace(-0.9, 0.9, 5)
dsigma = dilepton_cross_section(s_test, ct_test, model, include_sm=False)
assert dsigma.shape == (1, 5), '[TC04] cross section shape FAILED'
assert np.all(dsigma >= 0), '[TC04] cross section positivity FAILED'

# ---- TC05: eft_contact_interaction zero for invalid inputs ----
delta_sigma = eft_contact_interaction(s=1e6, eta_ll=1.0, eta_rr=0.0, eta_lr=0.0, Lambda=0.0)
assert delta_sigma == 0.0, '[TC05] EFT invalid input FAILED'

# ---- TC06: decay_width_dilepton positive ----
gamma_ll = decay_width_dilepton(model)
assert gamma_ll > 0, '[TC06] dilepton width positive FAILED'

# ---- TC07: width_consistency_check returns bool ----
check = width_consistency_check(model)
assert isinstance(check, (bool, np.bool_)), '[TC07] width consistency type FAILED'

# ---- TC08: scattering_amplitude_matrix shape and complex dtype ----
s_vals = np.linspace(8e6, 9e6, 5)
amp = scattering_amplitude_matrix(s_vals, model)
assert amp.shape == (5, 9), '[TC08] amplitude matrix shape FAILED'
assert np.iscomplexobj(amp), '[TC08] amplitude matrix complex FAILED'

# ---- TC09: chi_square_signal non-negative ----
obs = np.array([10.0, 12.0, 11.0])
bkg = np.array([9.0, 9.5, 10.0])
sig = np.array([1.0, 2.0, 1.0])
unc = np.array([1.0, 1.0, 1.0])
chi2 = chi_square_signal(obs, bkg, sig, unc)
assert chi2 >= 0, '[TC09] chi-square non-negative FAILED'

# ---- TC10: exclusion_limit_at_95cl finite for positive luminosity ----
limit = exclusion_limit_at_95cl(5.0, 20.0, 3000.0)
assert np.isfinite(limit) and limit >= 0, '[TC10] exclusion limit finite FAILED'

# ---- TC11: c8mat_fss solves identity system exactly ----
n = 4
a_id = np.eye(n, dtype=complex)
b_mat = np.ones((n, 2), dtype=complex)
x_sol = c8mat_fss(n, a_id, 2, b_mat)
assert np.allclose(x_sol, b_mat), '[TC11] c8mat_fss identity FAILED'

# ---- TC12: r8row_sort_quick_a sorts by first column ----
mat = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 0.5], [5.0, 1.5], [4.0, 2.5]])
sorted_mat = r8row_sort_quick_a(5, 2, mat.copy())
assert np.all(np.diff(sorted_mat[:, 0]) >= -1e-12), '[TC12] row sort monotonic FAILED'

# ---- TC13: r8utt_sl returns finite solution ----
a_toep = np.array([2.0, -1.0, 0.5])
b_toep = np.array([4.0, 2.0, 1.0])
x_toep = r8utt_sl(3, a_toep, b_toep)
assert np.isfinite(x_toep).all(), '[TC13] Toeplitz solution finite FAILED'

# ---- TC14: detector_deconvolution_toeplitz preserves array size ----
obs = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
psf = np.array([0.6, 0.3, 0.1])
deconv = detector_deconvolution_toeplitz(obs, psf, regularization=1e-4)
assert deconv.shape == obs.shape, '[TC14] deconvolution shape FAILED'

# ---- TC15: advection_diffusion_energy_deposit returns matching shapes ----
x_dep, e_dep = advection_diffusion_energy_deposit(nx=51, nt=100, c=0.5, diff_coeff=0.001)
assert x_dep.shape == (51,) and e_dep.shape == (51,), '[TC15] advection shapes FAILED'

# ---- TC16: news_edge_detector raises ValueError on 1D input ----
try:
    news_edge_detector(np.array([1.0, 2.0, 3.0]))
    assert False, '[TC16] edge detector 1D rejection FAILED'
except ValueError:
    pass

# ---- TC17: detector_hit_map reproducible with fixed seed ----
np.random.seed(123)
hm1, em1 = detector_hit_map(n_pixels=32, noise_level=0.01, seed=123)
np.random.seed(123)
hm2, em2 = detector_hit_map(n_pixels=32, noise_level=0.01, seed=123)
assert np.allclose(hm1, hm2), '[TC17] hit map reproducibility FAILED'

# ---- TC18: aperiodic_detector_geometry returns 2D coordinates ----
coords = aperiodic_detector_geometry(nmax=1, scale=1.0)
assert coords.shape[1] == 2, '[TC18] geometry 2D FAILED'
assert np.isfinite(coords).all(), '[TC18] geometry finite FAILED'

# ---- TC19: estimate_momentum_from_curvature positive for circular arc ----
theta = np.linspace(0, np.pi/4, 10)
R = 1.0
x_arc = R * np.cos(theta)
y_arc = R * np.sin(theta)
p_est = estimate_momentum_from_curvature(x_arc, y_arc, magnetic_field=3.8)
assert p_est > 0, '[TC19] momentum positive FAILED'

# ---- TC20: hermite_cubic_spline interpolates exactly at nodes ----
xn = np.array([0.0, 1.0, 2.0])
fn = np.array([0.0, 1.0, 4.0])
dn = np.array([0.0, 2.0, 4.0])
f_out, d_out, s_out, t_out = hermite_cubic_spline(xn, fn, dn, xn)
assert np.allclose(f_out, fn), '[TC20] hermite exact nodes FAILED'

# ---- TC21: fem1d_track_fit returns consistent node and coefficient lengths ----
tlen = np.linspace(0, 1, 15)
dedx = 2.0 * np.exp(-tlen)
node_x, node_c = fem1d_track_fit(tlen, dedx, n_nodes=8)
assert len(node_x) == 8 and len(node_c) == 8, '[TC21] FEM lengths FAILED'

# ---- TC22: particle_id_from_dedx returns valid particle string ----
pid = particle_id_from_dedx(np.array([1.5, 1.6, 1.55]), momentum=10.0)
assert pid in ('electron', 'muon', 'pion', 'proton', 'unknown'), '[TC22] PID string FAILED'

# ---- TC23: svd_low_rank_approximation error decreases with higher rank ----
np.random.seed(42)
mat = np.random.rand(10, 8)
_, _, _, approx1 = svd_low_rank_approximation(mat, rank=1)
_, _, _, approx3 = svd_low_rank_approximation(mat, rank=3)
err1 = np.linalg.norm(mat - approx1, 'fro')
err3 = np.linalg.norm(mat - approx3, 'fro')
assert err3 <= err1 + 1e-12, '[TC23] SVD rank monotonicity FAILED'

# ---- TC24: singular_value_entropy within [0, 1] ----
s_arr = np.array([5.0, 3.0, 1.0, 0.1])
entropy = singular_value_entropy(s_arr)
assert 0.0 <= entropy <= 1.0, '[TC24] entropy range FAILED'

# ---- TC25: pca_denoise preserves matrix shape ----
np.random.seed(42)
mat_noisy = np.random.rand(8, 8)
denoised = pca_denoise(mat_noisy, variance_threshold=0.9)
assert denoised.shape == mat_noisy.shape, '[TC25] PCA shape FAILED'

# ---- TC26: signal_background_discriminator separates signal and background ----
np.random.seed(42)
hit_maps = []
labels = []
for _ in range(10):
    hm, _ = detector_hit_map(n_pixels=16, noise_level=0.01, seed=42)
    hit_maps.append(hm)
    labels.append(1)
for _ in range(10):
    hm = np.random.exponential(0.1, (16, 16))
    hit_maps.append(hm)
    labels.append(0)
labels = np.array(labels)
basis, scores = signal_background_discriminator(hit_maps, labels, n_components=3)
assert basis.shape[0] == 3, '[TC26] discriminator basis shape FAILED'
assert len(scores) == 20, '[TC26] discriminator scores length FAILED'

# ---- TC27: resonance_peak_finder detects planted peak ----
masses = np.arange(100, 200, 1.0)
counts = np.ones_like(masses) * 10.0
counts[50] = 100.0
pm, ph, ps = resonance_peak_finder(masses, counts, window_width=5.0)
assert ph > 0, '[TC27] peak height positive FAILED'

# ---- TC28: bsm_cross_section_interp_2d within data extrema ----
mg = np.array([1000.0, 2000.0, 3000.0])
cg = np.array([0.1, 0.2])
cs_tab = np.outer(1.0/mg**2, cg**2) * 1000.0
cs_val = bsm_cross_section_interp_2d(mg, cg, cs_tab, query_mass=1500.0, query_coupling=0.15)
assert np.min(cs_tab) <= cs_val <= np.max(cs_tab), '[TC28] interp 2D range FAILED'

# ---- TC29: expected_signal_yield scales linearly with luminosity ----
n1 = expected_signal_yield(0.1, 1000.0, efficiency=1.0, branching_ratio=1.0)
n2 = expected_signal_yield(0.1, 2000.0, efficiency=1.0, branching_ratio=1.0)
assert np.isclose(n2, 2.0 * n1), '[TC29] yield linearity FAILED'

# ---- TC30: discovery_potential positive for positive signal ----
z_vals = discovery_potential(np.array([0.1]), np.array([10.0]), np.array([3000.0]), np.array([0.05]))
assert z_vals[0] > 0, '[TC30] discovery potential positive FAILED'

# ---- TC31: reconstruct_invariant_mass symmetric under swap ----
m_a = reconstruct_invariant_mass(500.0, 0.5, 0.3, 480.0, -0.4, 3.5)
m_b = reconstruct_invariant_mass(480.0, -0.4, 3.5, 500.0, 0.5, 0.3)
assert np.isclose(m_a, m_b), '[TC31] invariant mass symmetry FAILED'

# ---- TC32: cl_s_limit returns boolean ----
excluded = cl_s_limit(25, 20.0, 10.0, 0.95)
assert isinstance(excluded, (bool, np.bool_)), '[TC32] CLs type FAILED'

# ---- TC33: run_full_analysis returns dict with required keys ----
zp_p = {'mass': 2000.0, 'total_width': 60.0, 'gq_coupling': 0.2}
res = run_full_analysis(zp_p, luminosity_fb=1000.0, n_bins=20)
req_keys = ('zp_mass', 'peak_mass', 'exclusion_limit_sigma_95_pb', 'discovery_potential_z')
assert all(k in res for k in req_keys), '[TC33] full analysis keys FAILED'

# ---- TC34: format_physics_summary returns non-empty string ----
summary = format_physics_summary(res)
assert isinstance(summary, str) and len(summary) > 0, '[TC34] summary string FAILED'

# ---- TC35: flame_ode_solve saturates towards 1 ----
t_fl, y_fl = flame_ode_solve((0.0, 200.0), y0=0.01, delta=0.01, n_steps=5000)
assert y_fl[-1] > 0.9 and y_fl[-1] <= 1.0, '[TC35] flame saturation FAILED'

# ---- TC36: electromagnetic_shower_profile non-negative with peak ----
depths = np.linspace(0, 15, 100)
prof = electromagnetic_shower_profile(depths, E0=100.0, Ec=0.008)
assert np.all(prof >= 0), '[TC36] shower profile non-negative FAILED'
assert np.max(prof) > 0, '[TC36] shower profile has peak FAILED'

# ---- TC37: burgers_hadronization_pde output shape consistency ----
x_b, t_b, u_b = burgers_hadronization_pde(nx=64, nt=5, viscosity=0.03, t_max=0.3)
assert u_b.shape == (len(t_b), len(x_b)), '[TC37] Burgers shape FAILED'

# ---- TC38: hadronization_energy_spectrum conserves energy approximately ----
np.random.seed(42)
energies = hadronization_energy_spectrum(parton_energy=500.0, n_particles=50)
assert np.sum(energies) <= 500.0 * 1.01, '[TC38] energy conservation FAILED'
assert np.all(energies >= 0), '[TC38] energy positivity FAILED'

# ---- TC39: pwl_interp_2d_scattered exact at data points ----
dp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
dv = np.array([0.0, 1.0, 1.0, 2.0])
qp = np.array([[0.0, 0.0], [1.0, 1.0]])
iv = pwl_interp_2d_scattered(dp, dv, qp)
assert np.allclose(iv, np.array([0.0, 2.0]), atol=1e-6), '[TC39] PWL exact nodes FAILED'

# ---- TC40: knapsack_channel_selection returns boolean selection ----
sig_y = np.array([5.0, 8.0, 2.0])
bkg_y = np.array([50.0, 40.0, 80.0])
lumi = np.array([500.0, 500.0, 1000.0])
total_sig, selected = knapsack_channel_selection(sig_y, bkg_y, lumi, max_lumi=1000.0)
assert isinstance(total_sig, (float, np.floating)), '[TC40] knapsack sig type FAILED'
assert selected.dtype == bool, '[TC40] knapsack dtype FAILED'
