# ---- TC01: density_profile returns physically reasonable values ----
z_test = np.linspace(-200, 0, 11)
rho_test = density_profile(z_test)
assert np.all(rho_test >= 1020.0) and np.all(rho_test <= 1030.0), '[TC01] density_profile out of range FAILED'

# ---- TC02: buoyancy_frequency is positive and stable ----
N_test = buoyancy_frequency(z_test)
assert np.all(N_test > 0), '[TC02] buoyancy_frequency non-positive FAILED'

# ---- TC03: richardson_number handles zero shear safely ----
Ri_test = richardson_number(np.zeros(5), np.zeros(5), 0.01)
assert np.all(np.isfinite(Ri_test)), '[TC03] richardson_number zero shear non-finite FAILED'
assert np.all(Ri_test >= 1e5), '[TC03] richardson_number zero shear value FAILED'

# ---- TC04: internal_wave_dispersion respects bounds [f, N] ----
kh_test = np.linspace(0.001, 0.1, 10)
omega_test = internal_wave_dispersion(kh_test, 2.0 * np.pi / 200.0, 0.01, 1.0e-4)
assert np.all(omega_test >= 1.0e-4 * 0.99), '[TC04] internal_wave_dispersion below f FAILED'
assert np.all(omega_test <= 0.01 * 1.01), '[TC04] internal_wave_dispersion above N FAILED'

# ---- TC05: group_velocity returns finite values ----
cgx_test, cgz_test = group_velocity(kh_test, 2.0 * np.pi / 200.0, 0.01, 1.0e-4)
assert np.all(np.isfinite(cgx_test)), '[TC05] group_velocity cgx non-finite FAILED'
assert np.all(np.isfinite(cgz_test)), '[TC05] group_velocity cgz non-finite FAILED'

# ---- TC06: breaking_criterion detects large amplitude wave ----
is_breaking, steepness, crit = breaking_criterion(amplitude=50.0, wavelength=100.0, N=0.01, depth=200.0)
assert is_breaking == True, '[TC06] breaking_criterion large amp not breaking FAILED'
assert steepness > crit, '[TC06] breaking_criterion steepness <= crit FAILED'

# ---- TC07: NonlinearInternalWave solve produces non-negative energy ----
np.random.seed(42)
wave = NonlinearInternalWave(alpha=1.0, beta=5.0, gamma=8.0, delta=0.02, omega=0.5, N=0.01, f=1.0e-4, depth=200.0)
t_w, xi_w, xi_dot_w, E_w = wave.solve(t_span=(0, 10), dt=0.1)
assert len(t_w) == len(E_w), '[TC07] NonlinearInternalWave time-energy length mismatch FAILED'
assert np.all(E_w >= 0), '[TC07] NonlinearInternalWave energy negative FAILED'
assert np.all(np.isfinite(xi_w)), '[TC07] NonlinearInternalWave xi non-finite FAILED'

# ---- TC08: NonlinearInternalWave wave action is non-negative ----
action = wave.compute_wave_action(t_w, xi_w, xi_dot_w)
assert np.all(action >= 0), '[TC08] wave_action negative FAILED'
assert np.all(np.isfinite(action)), '[TC08] wave_action non-finite FAILED'

# ---- TC09: kdv_internal_wave output shapes consistent ----
x_kdv, t_kdv, eta_kdv = kdv_internal_wave(xi0=2.0, c=1.0, alpha_kdv=0.1, beta_kdv=0.01, t_span=(0, 5), nx=64)
assert eta_kdv.shape[0] == len(t_kdv), '[TC09] kdv eta time dim FAILED'
assert eta_kdv.shape[1] == len(x_kdv), '[TC09] kdv eta space dim FAILED'
assert np.all(np.isfinite(eta_kdv)), '[TC09] kdv eta non-finite FAILED'

# ---- TC10: DG solver produces bounded finite solution ----
solver = DGInternalWaveSolver(N=2, K=5, xmin=0.0, xmax=100.0, wave_speed=1.0, N_buoyancy=0.01)
t_hist, u_hist = solver.solve(t_final=2.0, dt=0.2)
assert np.all(np.isfinite(u_hist)), '[TC10] DG solver non-finite FAILED'
assert np.all(np.abs(u_hist) <= 10.0), '[TC10] DG solver out of bounds FAILED'

# ---- TC11: haar_1d_transform preserves energy (Parseval) ----
np.random.seed(42)
signal = np.sin(2.0 * np.pi * np.arange(64) / 16.0)
coeffs, energies = haar_1d_transform(signal)
time_energy = np.sum(signal[:64]**2)
wav_energy = np.sum(coeffs[0]**2) + np.sum(energies)
assert np.abs(time_energy - wav_energy) < 1e-10, '[TC11] haar energy conservation FAILED'
assert np.all(np.array(energies) >= 0), '[TC11] haar energies negative FAILED'

# ---- TC12: detect_breaking_events on constant signal returns no events ----
const_signal = np.ones(128)
indices, wenergy = detect_breaking_events(const_signal, threshold_factor=3.0)
assert len(indices) == 0, '[TC12] detect_breaking constant signal found events FAILED'

# ---- TC13: multi_scale_spectrum normalizes to unity ----
scales, spectrum = multi_scale_spectrum(signal)
assert np.abs(np.sum(spectrum) - 1.0) < 1e-12, '[TC13] multi_scale_spectrum normalization FAILED'
assert len(scales) == len(spectrum), '[TC13] multi_scale_spectrum length mismatch FAILED'

# ---- TC14: HilbertCurve3D h_to_xyz returns coordinates in valid range ----
hc = HilbertCurve3D(r=2)
for h_val in [0, 7, 15, 63]:
    x_h, y_h, z_h = hc.h_to_xyz(h_val)
    assert 0 <= x_h <= hc.N - 1, '[TC14] Hilbert x out of range FAILED'
    assert 0 <= y_h <= hc.N - 1, '[TC14] Hilbert y out of range FAILED'
    assert 0 <= z_h <= hc.N - 1, '[TC14] Hilbert z out of range FAILED'

# ---- TC15: HilbertCurve3D generate_curve covers all points in range ----
points = hc.generate_curve()
assert len(points) == 64, '[TC15] Hilbert curve point count FAILED'
assert np.all((points >= 0) & (points <= 3)), '[TC15] Hilbert curve coordinate range FAILED'

# ---- TC16: CVT1D generators stay within bounds ----
np.random.seed(42)
cvt = CVT1D(n_generators=5, z_min=-200.0, z_max=0.0, density_type='uniform')
gens, ehist = cvt.lloyd_iteration(n_samples=1000, max_iter=10, tol=1.0e-5)
assert np.all(gens >= -200.0) and np.all(gens <= 0.0), '[TC16] CVT generators out of bounds FAILED'

# ---- TC17: CVT1D energy history is non-negative ----
assert np.all(np.array(ehist) >= 0), '[TC17] CVT energy negative FAILED'

# ---- TC18: triangulate_ocean_domain produces valid triangles ----
nodes, triangles = triangulate_ocean_domain(x_range=(0, 1000), y_range=(0, 1000), n_points=16)
assert len(triangles) > 0, '[TC18] triangulate no triangles FAILED'
assert np.all(triangles >= 0) and np.all(triangles < len(nodes)), '[TC18] triangulate invalid indices FAILED'

# ---- TC19: random_phase_superposition is reproducible with fixed seed ----
np.random.seed(42)
u1, s1, Ri1 = random_phase_superposition(n_modes=5, z=np.linspace(-50, 0, 11), t=0.0, N=0.01)
np.random.seed(42)
u2, s2, Ri2 = random_phase_superposition(n_modes=5, z=np.linspace(-50, 0, 11), t=0.0, N=0.01)
assert np.allclose(u1, u2), '[TC19] random_phase not reproducible FAILED'
assert np.allclose(s1, s2), '[TC19] random_phase shear not reproducible FAILED'

# ---- TC20: monte_carlo_breaking_probability returns valid probability ----
np.random.seed(42)
P_break, P_break_z, z_out = monte_carlo_breaking_probability(n_realizations=50, n_modes=5, n_depths=21, N=0.01)
assert 0.0 <= P_break <= 1.0, '[TC20] breaking probability out of range FAILED'
assert np.all((P_break_z >= 0.0) & (P_break_z <= 1.0)), '[TC20] depth prob out of range FAILED'

# ---- TC21: energy_cascade_simulation energy stays positive ----
np.random.seed(42)
E_hist, breaking_events = energy_cascade_simulation(E0=1.0, n_steps=100, growth_factor=1.03, dissipation_factor=0.98)
assert np.all(E_hist > 0), '[TC21] energy_cascade non-positive energy FAILED'

# ---- TC22: mixing_patch_ifs output in unit range ----
np.random.seed(42)
x_ifs, y_ifs, intensities = mixing_patch_ifs(n_points=100, n_iterations=5)
assert np.all((x_ifs >= 0.0) & (x_ifs <= 1.0)), '[TC22] IFS x out of range FAILED'
assert np.all((y_ifs >= 0.0) & (y_ifs <= 1.0)), '[TC22] IFS y out of range FAILED'
assert np.all((intensities >= 0.0) & (intensities <= 1.0)), '[TC22] IFS intensity out of range FAILED'

# ---- TC23: dijkstra_shortest_path self distance is zero ----
graph = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
distances, previous = dijkstra_shortest_path(graph, 1)
assert distances[1] == 0.0, '[TC23] dijkstra self distance not zero FAILED'
assert distances[0] == 1.0, '[TC23] dijkstra shortest distance FAILED'

# ---- TC24: permutation_cycle_analysis cycle lengths sum to n ----
np.random.seed(42)
cycles, cycle_lengths, success_rate = permutation_cycle_analysis(n_lockers=20, n_tries=10)
assert np.sum(cycle_lengths) == 20, '[TC24] permutation cycle sum FAILED'
assert 0.0 <= success_rate <= 1.0, '[TC24] permutation success rate FAILED'

# ---- TC25: ray_tracing_cycle output arrays have correct length ----
z_ray = np.linspace(-100, 0, 51)
N_ray = 0.01 * np.ones_like(z_ray)
x_path, z_path, theta_path = ray_tracing_cycle(wave_frequency=0.005, N_profile=N_ray, z=z_ray, theta0=np.pi/6, max_steps=50)
assert len(x_path) == 50, '[TC25] ray tracing x length FAILED'
assert len(z_path) == 50, '[TC25] ray tracing z length FAILED'
assert len(theta_path) == 50, '[TC25] ray tracing theta length FAILED'
assert np.all(np.isfinite(x_path)), '[TC25] ray tracing x non-finite FAILED'

# ---- TC26: mixing_efficiency_fixed_point converges for small Ri ----
gamma, history, converged = mixing_efficiency_fixed_point(Ri=0.1, gamma_max=0.2, alpha=5.0, max_iter=100, tol=1.0e-8)
assert converged, '[TC26] mixing_efficiency not converged FAILED'
assert 0.0 <= gamma <= 0.2, '[TC26] mixing_efficiency gamma out of range FAILED'

# ---- TC27: symmetrize_wave_spectrum produces four-fold symmetry ----
kx = np.linspace(-0.1, 0.1, 8)
kz = np.linspace(-0.1, 0.1, 8)
KX, KZ = np.meshgrid(kx, kz)
E_spec = np.exp(-(KX**2 + KZ**2) / 0.001)
E_sym = symmetrize_wave_spectrum(E_spec)
assert np.allclose(E_sym, E_sym[::-1, :]), '[TC27] spectrum symmetry x FAILED'
assert np.allclose(E_sym, E_sym[:, ::-1]), '[TC27] spectrum symmetry z FAILED'

# ---- TC28: ocean_volume_indexing returns positive scales ----
hc_idx, d_scale, lat_scale, lon_scale = ocean_volume_indexing(depth_levels=10, lat_levels=10, lon_levels=10, r=3)
assert d_scale > 0 and lat_scale > 0 and lon_scale > 0, '[TC28] ocean_volume_indexing scales non-positive FAILED'

# ---- TC29: turbulent_dissipation_rate returns non-negative epsilon and bounded Kz ----
Ri_t = np.array([0.1, 0.3, 0.5])
shear_t = np.array([0.01, 0.005, 0.002])
eps, Kz = turbulent_dissipation_rate(Ri_t, shear_t)
assert np.all(eps >= 0), '[TC29] turbulent_dissipation epsilon negative FAILED'
assert np.all(Kz >= 1.0e-7), '[TC29] Kz below lower bound FAILED'
assert np.all(Kz <= 1.0e-1), '[TC29] Kz above upper bound FAILED'

# ---- TC30: thope_internal_wave_spectrum is non-negative ----
kh_test2 = np.linspace(0.001, 0.1, 10)
spec_test = thope_internal_wave_spectrum(kh_test2, N=0.01, f=1.0e-4, E0=6.3e-5)
assert np.all(spec_test >= 0), '[TC30] spectrum negative FAILED'
