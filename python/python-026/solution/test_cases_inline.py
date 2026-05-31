# ---- TC01: plasma_frequency returns finite positive for standard density ----
omega_p = plasma_frequency(1e25)
assert np.isfinite(omega_p) and omega_p > 0, '[TC01] plasma_frequency finite positive FAILED'

# ---- TC02: critical_density and plasma_frequency are mutually consistent ----
omega_test = 1.77e15
nc_test = critical_density(omega_test)
omega_back = plasma_frequency(nc_test)
assert np.isclose(omega_back, omega_test, rtol=1e-10), '[TC02] critical_density consistency FAILED'

# ---- TC03: refractive_index is 1.0 in vacuum and 0.0 at critical density ----
nc_test = critical_density(1e15)
eta_vac = refractive_index(0.0, 1e15)
eta_crit = refractive_index(nc_test, 1e15)
assert np.isclose(eta_vac, 1.0) and np.isclose(eta_crit, 0.0), '[TC03] refractive_index bounds FAILED'

# ---- TC04: laser_E0_from_intensity and laser_intensity_from_E0 are inverses ----
I_test = 1e18
E_test = laser_E0_from_intensity(I_test)
I_back = laser_intensity_from_E0(E_test)
assert np.isclose(I_back, I_test, rtol=1e-12), '[TC04] laser E-I inverse FAILED'

# ---- TC05: debye_length returns finite positive for physical parameters ----
from physics_constants import debye_length
lambda_d = debye_length(1e25, 11604.5)
assert np.isfinite(lambda_d) and lambda_d > 0, '[TC05] debye_length finite positive FAILED'

# ---- TC06: line_grid_1d uniform strategy endpoints correct and length matches ----
from plasma_grid import line_grid_1d
x_grid = line_grid_1d(5, 0.0, 1.0, c=1)
assert len(x_grid) == 5 and np.isclose(x_grid[0], 0.0) and np.isclose(x_grid[-1], 1.0), '[TC06] line_grid_1d endpoints FAILED'

# ---- TC07: grid_spacing_quality ratio is 1.0 for uniform grid ----
quality = grid_spacing_quality(x_grid)
assert np.isclose(quality['ratio'], 1.0), '[TC07] grid_spacing_quality uniform FAILED'

# ---- TC08: rect_grid_2d output shapes match requested dimensions ----
X, Y, dx, dy = rect_grid_2d(10, 20, (0.0, 1.0), (0.0, 2.0), cx=1, cy=1)
assert X.shape == (10, 20) and Y.shape == (10, 20), '[TC08] rect_grid_2d shape FAILED'

# ---- TC09: icf_density_profile center value within expected bounds ----
ne_center = icf_density_profile(0.0, 0.0, n0=1e25, R0=100e-6, Ls=10e-6, f_plateau=0.3)
assert 0.3e25 <= ne_center <= 1.1e25, '[TC09] icf_density_profile center bounds FAILED'

# ---- TC10: bilinear_interpolate_density returns non-negative finite values inside grid ----
from density_profile import bilinear_interpolate_density
xc = np.linspace(0, 1e-4, 11)
yc = np.linspace(0, 1e-4, 11)
nxc, nyc = 10, 10
ne_cells = np.ones((nxc, nyc)) * 1e24
ne_q = bilinear_interpolate_density(ne_cells, xc, yc, 5e-5, 5e-5)
assert np.isfinite(ne_q) and ne_q >= 0, '[TC10] bilinear_interpolate_density bounds FAILED'

# ---- TC11: total_plasma_mass matches analytic constant-density solution ----
mass = total_plasma_mass(ne_cells, xc, yc)
dx_total = xc[-1] - xc[0]
dy_total = yc[-1] - yc[0]
mass_expected = np.sum(ne_cells) * (dx_total / nxc) * (dy_total / nyc) * 2.5e-26
assert np.isclose(mass, mass_expected, rtol=1e-12), '[TC11] total_plasma_mass analytic FAILED'

# ---- TC12: density_gradient_pwc of constant field is zero ----
grad_x, grad_y = density_gradient_pwc(ne_cells, xc, yc)
assert np.allclose(grad_x, 0.0) and np.allclose(grad_y, 0.0), '[TC12] density_gradient_pwc constant FAILED'

# ---- TC13: wdk_roots correctly solves quadratic polynomial x^2 - 1 = 0 ----
from dispersion_solver import wdk_roots
roots, conv = wdk_roots(np.array([-1.0 + 0j, 0.0 + 0j, 1.0 + 0j]), tol=1e-12, max_iter=2000)
assert conv and any(np.isclose(r, 1.0) for r in roots) and any(np.isclose(r, -1.0) for r in roots), '[TC13] wdk_roots quadratic FAILED'

# ---- TC14: poly_eval Horner scheme matches direct evaluation ----
from dispersion_solver import poly_eval
val = poly_eval(np.array([1.0 + 0j, 0.0 + 0j, -1.0 + 0j]), 2.0 + 0j)
assert np.isclose(val, -3.0 + 0j), '[TC14] poly_eval direct FAILED'

# ---- TC15: solve_langmuir_wave_dispersion returns finite real frequency ----
omega_r, gamma, root_sel, all_roots = solve_langmuir_wave_dispersion(1e25, 11604.5, 1e6, 1.77e15)
assert np.isfinite(omega_r) and omega_r > 0, '[TC15] langmuir dispersion finite FAILED'

# ---- TC16: latin_edge_sample reproducible with fixed seed ----
from parameter_sampling import latin_edge_sample
np.random.seed(42)
s1 = latin_edge_sample(3, 5, seed=42)
s2 = latin_edge_sample(3, 5, seed=42)
assert np.allclose(s1, s2), '[TC16] latin_edge_sample reproducibility FAILED'

# ---- TC17: sample_quality_metrics distances are positive ----
params, _, _ = sample_laser_plasma_parameters(8, seed=42)
metrics = sample_quality_metrics(params)
assert metrics['min_pairwise_dist'] > 0 and metrics['mean_pairwise_dist'] > 0, '[TC17] sample_quality_metrics positive FAILED'

# ---- TC18: gauss_legendre_rule weights sum to 2 ----
from quadrature_engine import gauss_legendre_rule
x_gl, w_gl = gauss_legendre_rule(5)
assert np.isclose(np.sum(w_gl), 2.0), '[TC18] gauss_legendre_rule weight sum FAILED'

# ---- TC19: integrate_1d_gauss_legendre exact for constant function ----
res = integrate_1d_gauss_legendre(lambda x: 3.0, 0.0, 1.0, n=4)
assert np.isclose(res, 3.0), '[TC19] integrate_1d_gauss_legendre constant FAILED'

# ---- TC20: integrate_energy_deposition_along_ray returns non-negative ----
s_vals = np.linspace(0, 1e-3, 10)
I_vals = np.ones(10) * 1e15
ne_vals = np.ones(10) * 1e24
dep = integrate_energy_deposition_along_ray(s_vals, I_vals, ne_vals, 11604.5, 1.77e15, Z=1)
assert np.isfinite(dep) and dep >= 0, '[TC20] energy_deposition non-negative FAILED'

# ---- TC21: jones_matrix_propagation with zero plasma frequency is near identity ----
from polarization_dynamics import jones_matrix_propagation
T = jones_matrix_propagation(1e15, 0.0, 0.0, np.array([0.0, 0.0, 1.0]), 1e-6)
assert np.allclose(T, np.eye(2), atol=1e-12), '[TC21] jones identity FAILED'

# ---- TC22: polarization_ellipse_parameters linear polarization has S3 near zero ----
from polarization_dynamics import polarization_ellipse_parameters
psi, chi, eps, S = polarization_ellipse_parameters(np.array([1.0, 0.0]))
assert np.isclose(S[3], 0.0, atol=1e-12), '[TC22] linear polarization S3 FAILED'

# ---- TC23: faraday_rotation_integral zero B gives zero rotation ----
z_vals = np.linspace(0, 1e-3, 20)
ne_prof = np.ones(20) * 1e24
B_zero = np.zeros(20)
theta_F = faraday_rotation_integral(ne_prof, B_zero, z_vals, 1e15)
assert np.isclose(theta_F, 0.0, atol=1e-30), '[TC23] faraday zero B FAILED'

# ---- TC24: circle_map_matrix_polarization identity matrix preserves aspect ratio 1 ----
x_in, x_out, aspect = circle_map_matrix_polarization(np.eye(2), n_points=50)
assert np.isclose(aspect, 1.0, atol=1e-10), '[TC24] circle_map identity FAILED'

# ---- TC25: spherical_distance same point is zero ----
d = spherical_distance(0.0, 0.0, 0.0, 0.0, 1.0)
assert np.isclose(d, 0.0), '[TC25] spherical_distance same point FAILED'

# ---- TC26: stereographic_projection_sphere_to_plane and inverse are consistent ----
p = np.array([1.0, 0.0, 0.0])
q = stereographic_projection_sphere_to_plane(p)
from target_geometry import stereographic_projection_plane_to_sphere
p_back = stereographic_projection_plane_to_sphere(q)
assert np.allclose(p, p_back, atol=1e-12), '[TC26] stereographic inverse FAILED'

# ---- TC27: quadrilateral_area_3d unit square area is 1 ----
from target_geometry import quadrilateral_area_3d
sq = np.array([[0.0, 1.0, 1.0, 0.0],
               [0.0, 0.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 0.0]])
area_sq = quadrilateral_area_3d(sq)
assert np.isclose(area_sq, 1.0, atol=1e-10), '[TC27] quadrilateral_area_3d unit FAILED'

# ---- TC28: icf_target_surface_mesh total area close to 4*pi*R^2 ----
vertices, face_areas, total_area = icf_target_surface_mesh(1.0, 21, 41)
assert np.isclose(total_area, 4.0 * np.pi * 1.0**2, rtol=0.05), '[TC28] target_surface_mesh area FAILED'

# ---- TC29: solve_poisson_2d with zero rho yields zero potential ----
from sparse_field_solver import solve_poisson_2d
rho_zero = np.zeros((5, 5))
phi, res, info = solve_poisson_2d(rho_zero, 5, 5, 1e-6, 1e-6)
assert np.allclose(phi, 0.0, atol=1e-8), '[TC29] poisson_zero_rho FAILED'

# ---- TC30: compute_electric_field_from_potential constant phi gives zero E ----
Ex, Ey = compute_electric_field_from_potential(np.ones((5, 5)), 1e-6, 1e-6)
assert np.allclose(Ex, 0.0, atol=1e-12) and np.allclose(Ey, 0.0, atol=1e-12), '[TC30] E_from_const_phi FAILED'

# ---- TC31: checksum_plasma_state identical input produces identical output ----
ne_arr = np.ones((3, 3))
Te_arr = np.ones((3, 3)) * 1000.0
phi_arr = np.zeros((3, 3))
cs1 = checksum_plasma_state(ne_arr, Te_arr, phi_arr)
cs2 = checksum_plasma_state(ne_arr, Te_arr, phi_arr)
assert cs1 == cs2, '[TC31] checksum reproducibility FAILED'
