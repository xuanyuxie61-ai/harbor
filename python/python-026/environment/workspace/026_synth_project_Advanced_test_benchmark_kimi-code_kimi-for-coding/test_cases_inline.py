# ---- TC01: plasma_frequency zero density returns zero ----
result = plasma_frequency(0.0)
assert abs(result) < 1e-20, '[TC01] plasma_frequency zero density FAILED'

# ---- TC02: critical_density matches analytical formula ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
nc_expected = EPSILON_0 * E_MASS * omega_test**2 / E_CHARGE**2
assert abs(nc_test - nc_expected) / max(abs(nc_expected), 1.0) < 1e-10, '[TC02] critical_density formula FAILED'

# ---- TC03: refractive_index vacuum is 1 and cutoff is 0 ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
eta_vac = refractive_index(0.0, omega_test)
eta_cut = refractive_index(nc_test, omega_test)
assert abs(eta_vac - 1.0) < 1e-12, '[TC03] refractive_index vacuum FAILED'
assert abs(eta_cut) < 1e-12, '[TC03] refractive_index cutoff FAILED'

# ---- TC04: laser intensity E0 round-trip consistency ----
I_test = 1.0e18
E0_test = laser_E0_from_intensity(I_test)
I_back = laser_intensity_from_E0(E0_test)
assert abs(I_back - I_test) / I_test < 1e-12, '[TC04] laser intensity E0 round-trip FAILED'

# ---- TC05: inverse_bremsstrahlung finite for small positive density ----
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
omega_test = 1.0e15
kappa = inverse_bremsstrahlung_absorption(1.0e10, Te_test, omega_test)
assert np.isfinite(kappa), '[TC05] inverse_bremsstrahlung finite FAILED'
assert kappa >= 0.0, '[TC05] inverse_bremsstrahlung non-negative FAILED'

# ---- TC06: rect_grid_2d shape and spacing correctness ----
X, Y, dx, dy = rect_grid_2d(11, 21, (-1e-4, 1e-4), (-2e-4, 2e-4), cx=1, cy=1)
assert X.shape == (11, 21), '[TC06] rect_grid_2d X shape FAILED'
assert Y.shape == (11, 21), '[TC06] rect_grid_2d Y shape FAILED'
assert abs(dx - 2e-5) < 1e-20, '[TC06] rect_grid_2d dx FAILED'
assert abs(dy - 2e-5) < 1e-20, '[TC06] rect_grid_2d dy FAILED'

# ---- TC07: grid_spacing_quality uniform grid ratio is 1 ----
x_uniform = np.linspace(0.0, 1.0, 101)
qual = grid_spacing_quality(x_uniform)
assert abs(qual['ratio'] - 1.0) < 1e-12, '[TC07] grid_spacing_quality ratio FAILED'
assert abs(qual['mean_dx'] - 0.01) < 1e-12, '[TC07] grid_spacing_quality mean_dx FAILED'

# ---- TC08: cell_volumes_2d shape matches grid ----
X, Y, dx, dy = rect_grid_2d(5, 7, (0.0, 1.0), (0.0, 2.0), cx=1, cy=1)
vols = cell_volumes_2d(X, Y)
assert vols.shape == (4, 6), '[TC08] cell_volumes_2d shape FAILED'

# ---- TC09: icf_density_profile center approaches peak density ----
n0_test = 1.0e25
R0_test = 50.0e-6
Ls_test = 5.0e-6
ne_center = icf_density_profile(0.0, 0.0, n0_test, R0_test, Ls_test, f_plateau=0.3, perturbation_amplitude=0.0)
assert abs(ne_center - n0_test) / n0_test < 1e-3, '[TC09] icf_density_profile center FAILED'

# ---- TC10: piecewise_constant_density_2d shape and value ----
xc_test = np.linspace(-1e-4, 1e-4, 6)
yc_test = np.linspace(-1e-4, 1e-4, 6)
def const_density(x, y):
    x_arr = np.asarray(x, dtype=float)
    return np.full_like(x_arr, 1.0e24)
ne_cells, xc_out, yc_out = piecewise_constant_density_2d(xc_test, yc_test, 5, 5, const_density)
assert ne_cells.shape == (5, 5), '[TC10] piecewise_constant_density_2d shape FAILED'
assert np.allclose(ne_cells, 1.0e24), '[TC10] piecewise_constant_density_2d value FAILED'

# ---- TC11: density_gradient_pwc uniform field zero gradient ----
ne_cells = np.ones((5, 5)) * 1.0e24
xc = np.linspace(0.0, 1.0, 6)
yc = np.linspace(0.0, 1.0, 6)
gx, gy = density_gradient_pwc(ne_cells, xc, yc)
assert gx.shape == (4, 5), '[TC11] density_gradient_pwc grad_x shape FAILED'
assert gy.shape == (5, 4), '[TC11] density_gradient_pwc grad_y shape FAILED'
assert np.allclose(gx, 0.0), '[TC11] density_gradient_pwc uniform gx FAILED'
assert np.allclose(gy, 0.0), '[TC11] density_gradient_pwc uniform gy FAILED'

# ---- TC12: total_plasma_mass uniform density exact integral ----
xc_test = np.linspace(0.0, 1.0, 3)
yc_test = np.linspace(0.0, 2.0, 3)
ne_cells = np.ones((2, 2)) * 1.0e24
mass = total_plasma_mass(ne_cells, xc_test, yc_test, ion_mass=1.0)
assert abs(mass - 2.0e24) < 1e-10, '[TC12] total_plasma_mass uniform FAILED'

# ---- TC13: RayTracer vacuum straight ray exits domain ----
tracer = RayTracer(1.0e15, eta_min=1e-4, max_steps=5000)
def vacuum_density(x, y):
    return 0.0
domain = ((-1e-3, 1e-3), (-1e-3, 1e-3))
pos = np.array([[-5e-4, 0.0]])
dir_vec = np.array([[1.0, 0.0]])
results = tracer.trace_beam(pos, dir_vec, vacuum_density, domain)
assert len(results) == 1, '[TC13] RayTracer results length FAILED'
assert results[0]['status'] == 'domain_exit', '[TC13] RayTracer status FAILED'
assert results[0]['path_length'] > 1e-4, '[TC13] RayTracer path length FAILED'

# ---- TC14: integrate_1d_gauss_legendre constant exact ----
result = integrate_1d_gauss_legendre(lambda x: 3.0, 0.0, 2.0, n=5)
assert abs(result - 6.0) < 1e-14, '[TC14] integrate_1d_gauss_legendre constant FAILED'

# ---- TC15: integrate_energy_deposition zero density zero energy ----
s_test = np.linspace(0.0, 1.0, 11)
I_test_arr = np.ones(11) * 1.0e18
ne_zero = np.zeros(11)
Te_test = 1000.0
omega_test = 1.0e15
dep = integrate_energy_deposition_along_ray(s_test, I_test_arr, ne_zero, Te_test, omega_test)
assert abs(dep) < 1e-20, '[TC15] integrate_energy_deposition zero density FAILED'

# ---- TC16: solve_langmuir_wave_dispersion finite positive frequency ----
ne_test = 1.0e25
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
k_test = 1.0e7
omega0_test = 1.0e15
omega_r, gamma, root_sel, all_roots = solve_langmuir_wave_dispersion(ne_test, Te_test, k_test, omega0_test)
assert np.isfinite(omega_r), '[TC16] solve_langmuir omega_r finite FAILED'
assert omega_r > 0, '[TC16] solve_langmuir omega_r positive FAILED'
assert len(all_roots) == 4, '[TC16] solve_langmuir root count FAILED'

# ---- TC17: srs_three_wave_coupling_roots finite output ----
ne_test = 1.0e25
Te_test = 1000.0 * E_CHARGE / K_BOLTZMANN
k_s = 1.0e7
omega0_test = 1.0e15
E0_test = 1.0e10
omega_s_r, gamma_srs, roots = srs_three_wave_coupling_roots(ne_test, Te_test, k_s, omega0_test, E0_test)
assert np.isfinite(omega_s_r), '[TC17] srs omega_s finite FAILED'
assert omega_s_r > 0, '[TC17] srs omega_s positive FAILED'
assert len(roots) == 4, '[TC17] srs root count FAILED'

# ---- TC18: sample_laser_plasma_parameters shape and bounds ----
params, names, bounds = sample_laser_plasma_parameters(10, seed=42)
assert params.shape == (10, 6), '[TC18] sample shape FAILED'
for i, (pmin, pmax) in enumerate(bounds):
    assert np.all(params[:, i] >= pmin), f'[TC18] sample lower bound dim {i} FAILED'
    assert np.all(params[:, i] <= pmax), f'[TC18] sample upper bound dim {i} FAILED'

# ---- TC19: sample_quality_metrics consistency ----
params, names, bounds = sample_laser_plasma_parameters(5, seed=42)
metrics = sample_quality_metrics(params)
assert metrics['min_pairwise_dist'] >= 0, '[TC19] sample_quality min dist FAILED'
assert metrics['max_pairwise_dist'] >= metrics['min_pairwise_dist'], '[TC19] sample_quality max>=min FAILED'

# ---- TC20: spherical_distance quarter circumference exact ----
R_test = 1.0
d = spherical_distance(0.0, 0.0, np.pi / 2.0, 0.0, R_test)
assert abs(d - np.pi / 2.0 * R_test) < 1e-12, '[TC20] spherical_distance quarter FAILED'

# ---- TC21: stereographic_projection known points ----
p_south = np.array([0.0, 0.0, -1.0])
q_south = stereographic_projection_sphere_to_plane(p_south)
assert np.allclose(q_south, [0.0, 0.0]), '[TC21] stereographic south pole FAILED'
p_equator = np.array([1.0, 0.0, 0.0])
q_equator = stereographic_projection_sphere_to_plane(p_equator)
assert np.allclose(q_equator, [2.0, 0.0]), '[TC21] stereographic equator FAILED'

# ---- TC22: icf_target_surface_mesh area converges to sphere ----
R_test = 100.0e-6
vertices, face_areas, total_area = icf_target_surface_mesh(R_test, 41, 81)
expected_area = 4.0 * np.pi * R_test**2
rel_err = abs(total_area - expected_area) / expected_area
assert rel_err < 0.05, '[TC22] icf_target_surface_mesh area FAILED'

# ---- TC23: solid_angle_subtended_by_quad finite non-negative ----
q = np.array([[1.0, 0.0, -1.0, 0.0],
              [0.0, 1.0, 0.0, -1.0],
              [0.0, 0.0, 0.0, 0.0]])
omega = solid_angle_subtended_by_quad(q, np.array([0.0, 0.0, 1.0]))
assert omega >= 0.0, '[TC23] solid_angle non-negative FAILED'
assert np.isfinite(omega), '[TC23] solid_angle finite FAILED'

# ---- TC24: faraday_rotation_integral zero B field ----
z_test = np.linspace(0.0, 1.0, 51)
ne_test_arr = np.ones(51) * 1.0e24
B_zero = np.zeros(51)
theta_F = faraday_rotation_integral(ne_test_arr, B_zero, z_test, 1.0e15)
assert abs(theta_F) < 1e-20, '[TC24] faraday_rotation_integral zero B FAILED'

# ---- TC25: circle_map_matrix_polarization identity aspect ratio ----
x_in, x_out, aspect = circle_map_matrix_polarization(np.eye(2), n_points=100)
assert abs(aspect - 1.0) < 1e-10, '[TC25] circle_map identity aspect FAILED'

# ---- TC26: solve_poisson_2d zero charge zero potential ----
nx_test, ny_test = 5, 5
dx_test, dy_test = 1.0e-6, 1.0e-6
rho_zero = np.zeros((nx_test, ny_test))
phi, residual, info = solve_poisson_2d(rho_zero, nx_test, ny_test, dx_test, dy_test)
assert np.allclose(phi, 0.0, atol=1e-10), '[TC26] solve_poisson zero phi FAILED'
assert residual < 1e-10, '[TC26] solve_poisson zero residual FAILED'

# ---- TC27: compute_electric_field_from_potential linear gradient ----
phi_linear = np.outer(np.arange(5), np.ones(5)).astype(float)
Ex, Ey = compute_electric_field_from_potential(phi_linear, 1.0, 1.0)
assert np.allclose(Ex[1:-1, 1:-1], -1.0, atol=1e-10), '[TC27] electric_field Ex FAILED'
assert np.allclose(Ey[1:-1, 1:-1], 0.0, atol=1e-10), '[TC27] electric_field Ey FAILED'

# ---- TC28: checksum_plasma_state reproducible ----
ne_test_arr = np.ones((3, 3)) * 1.0e24
Te_test_arr = np.ones((3, 3)) * 1000.0
phi_test_arr = np.zeros((3, 3))
cs1 = checksum_plasma_state(ne_test_arr, Te_test_arr, phi_test_arr)
cs2 = checksum_plasma_state(ne_test_arr, Te_test_arr, phi_test_arr)
assert cs1 == cs2, '[TC28] checksum reproducibility FAILED'

# ---- TC29: evolve_polarization_along_ray shape and no-B conservation ----
z_vals = np.linspace(0.0, 1.0e-6, 11)
def ne_func_z(z):
    return 1.0e24
def B_func_z(z):
    return np.array([0.0, 0.0, 0.0])
E0_jones = np.array([1.0, 0.0])
E_hist, S_hist = evolve_polarization_along_ray(1.0e15, ne_func_z, B_func_z, z_vals, E0_jones)
assert E_hist.shape == (11, 2), '[TC29] evolve_polarization E shape FAILED'
assert S_hist.shape == (11, 4), '[TC29] evolve_polarization S shape FAILED'
assert abs(S_hist[0, 0] - S_hist[-1, 0]) < 1e-3 * S_hist[0, 0], '[TC29] evolve_polarization intensity drift FAILED'

# ---- TC30: critical_density and plasma_frequency mutual consistency ----
omega_test = 1.0e15
nc_test = critical_density(omega_test)
omega_p_test = plasma_frequency(nc_test)
assert abs(omega_p_test - omega_test) / omega_test < 1e-10, '[TC30] critical_density plasma_frequency consistency FAILED'
