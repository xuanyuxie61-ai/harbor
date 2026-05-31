from dipole_coupling import dyadic_green_tensor
from wavelet_field import haar_2d_inverse, haar_1d_transform
from volume_integrator import legendre_nodes_weights_1d, integrate_over_box
from resonance_finder import bisection_method
from hotcarrier_transport import hot_electron_escape_probability, effective_diffusion_coefficient
from nanoparticle_layout import cvtp_1d_optimize
from effective_wave import r83s_matvec, conjugate_gradient_r83s

# ---- TC01: drude_permittivity returns complex type ----
eps_drude = drude_permittivity(5.0e15)
assert isinstance(eps_drude, complex) or np.iscomplexobj(eps_drude), '[TC01] drude_permittivity complex type FAILED'

# ---- TC02: drude_permittivity has positive imaginary part (lossy medium absorption) ----
eps_drude2 = drude_permittivity(5.0e15, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert np.imag(eps_drude2) > 0, '[TC02] drude_permittivity positive imaginary part FAILED'

# ---- TC03: drude_permittivity array input yields same-length output ----
omegas_test = np.linspace(2.0e15, 8.0e15, 10)
eps_arr = drude_permittivity(omegas_test)
assert len(eps_arr) == len(omegas_test), '[TC03] drude_permittivity array output length FAILED'

# ---- TC04: drude_permittivity real part converges to eps_inf at high frequency ----
eps_high = drude_permittivity(1.0e20, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert abs(np.real(eps_high) - 9.84) < 1.0, '[TC04] drude_permittivity high-frequency limit FAILED'

# ---- TC05: mie_cross_sections scalar omega returns scalar sigma ----
sigma_ext, sigma_sca = mie_cross_sections(np.array([4.0e15]), 30e-9, eps_medium=1.0)
assert sigma_ext.shape == (1,), '[TC05] mie_cross_sections scalar output FAILED'
assert sigma_ext[0] >= sigma_sca[0] >= 0, '[TC05] mie_cross_sections ext>=sca>=0 FAILED'

# ---- TC06: polarizability_clausius_mossotti is complex ----
alpha_cm = polarizability_clausius_mossotti(complex(-20.0 + 0.5j, 0.5), 1.0, 1.13e-22)
assert np.iscomplexobj(alpha_cm) or isinstance(alpha_cm, complex), '[TC06] polarizability complex type FAILED'

# ---- TC07: dyadic_green_tensor zero distance returns zero matrix ----
G_zero = dyadic_green_tensor(np.array([0.0, 0.0, 0.0]), k=1.0e7)
assert np.allclose(G_zero, 0.0), '[TC07] dyadic_green_tensor zero distance FAILED'

# ---- TC08: dyadic_green_tensor is symmetric G = G^T ----
r_test = np.array([10e-9, 20e-9, 30e-9])
k_test = 1.0e7
G_sym = dyadic_green_tensor(r_test, k_test)
assert np.allclose(G_sym, G_sym.T), '[TC08] dyadic_green_tensor symmetry FAILED'

# ---- TC09: incident_plane_wave output shape matches positions ----
N_test = 5
pos_test = np.random.RandomState(123).rand(N_test, 3) * 100e-9
kvec_test = np.array([0.0, 0.0, 2.0e7])
pol_test = np.array([1.0, 0.0, 0.0])
b_test = incident_plane_wave(pos_test, 1.0e5, kvec_test, pol_test)
assert len(b_test) == 3 * N_test, '[TC09] incident_plane_wave output shape FAILED'

# ---- TC10: power_spectral_density non-negative ----
np.random.seed(42)
ts_test = np.sin(2.0 * np.pi * np.arange(128) / 128.0) + 0.1 * np.random.randn(128)
freqs_test, psd_test = power_spectral_density(ts_test, 1.0)
assert np.all(psd_test >= 0), '[TC10] PSD non-negative FAILED'

# ---- TC11: haar_2d_transform + haar_2d_inverse perfect reconstruction ----
np.random.seed(42)
field_orig = np.random.randn(64, 64)
field_trans = haar_2d_transform(field_orig)
field_recon = haar_2d_inverse(field_trans)
assert np.allclose(field_orig, field_recon, atol=1e-12), '[TC11] Haar 2D round-trip FAILED'

# ---- TC12: haar_1d_transform preserves norm approximately ----
np.random.seed(42)
u_test = np.random.randn(128)
v_test = haar_1d_transform(u_test)
assert abs(np.linalg.norm(v_test) - np.linalg.norm(u_test)) < 1e-10, '[TC12] Haar 1D norm preservation FAILED'

# ---- TC13: find_connected_components_2d basic detection ----
field_cc = np.zeros((8, 8))
field_cc[2:4, 2:4] = 10.0
field_cc[5:7, 5:7] = 10.0
labels_cc, num_cc = find_connected_components_2d(field_cc, 5.0)
assert num_cc == 2, '[TC13] connected_components_2d count FAILED'

# ---- TC14: legendre_nodes_weights_1d sum of weights = 2 ----
x_gl, w_gl = legendre_nodes_weights_1d(5)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC14] legendre sum of weights FAILED'

# ---- TC15: gauss_legendre_3d_set total weight = box volume ----
a_box = np.array([0.0, 0.0, 0.0])
b_box = np.array([1.0, 2.0, 3.0])
_, _, _, w_gl3d = gauss_legendre_3d_set(a_box, b_box, 4, 4, 4)
expected_vol = 1.0 * 2.0 * 3.0
assert abs(np.sum(w_gl3d) - expected_vol) < 1e-12, '[TC15] GL 3D total weight = volume FAILED'

# ---- TC16: test_exactness_monomial small error for low-degree monomials ----
errors_test = test_exactness_monomial(a_box, b_box, max_total_degree=1, nx=6, ny=6, nz=6)
assert max(errors_test) < 1e-10, '[TC16] quadrature exactness FAILED'

# ---- TC17: bisection_method finds known root of f(x)=x ----
f_lin = lambda x: x
root_bis, it_bis, _, _ = bisection_method(f_lin, -1.0, 1.0)
assert abs(root_bis) < 1e-10, '[TC17] bisection root of f(x)=x FAILED'

# ---- TC18: find_single_sphere_resonance returns positive frequency ----
omega_res = find_single_sphere_resonance(1.0, omega_p=1.37e16, gamma=4.05e13, eps_inf=9.84)
assert omega_res > 0, '[TC18] single sphere resonance positive FAILED'

# ---- TC19: find_single_sphere_resonance approximate analytical limit ----
# Drude model resonance: omega_res ≈ omega_p / sqrt(eps_inf + 2*eps_medium)
omega_analytical = 1.37e16 / np.sqrt(9.84 + 2.0)
assert abs(omega_res - omega_analytical) / omega_analytical < 0.1, '[TC19] resonance analytical limit FAILED'

# ---- TC20: effective_diffusion_coefficient positive ----
D_eff_test = effective_diffusion_coefficient(1e-9, 1e-14)
assert D_eff_test > 0, '[TC20] diffusion coefficient positive FAILED'

# ---- TC21: effective_diffusion_coefficient known value ----
D_known = effective_diffusion_coefficient(1e-10, 1e-15)
D_expected = (1e-10)**2 / (3.0 * 1e-15)
assert abs(D_known - D_expected) < 1e-20, '[TC21] diffusion coefficient exact value FAILED'

# ---- TC22: hot_electron_escape_probability above barrier returns 1 ----
prob_escape = hot_electron_escape_probability(energy=2.0, theta=0.0, barrier_height=0.5)
assert prob_escape == 1.0, '[TC22] escape probability above barrier FAILED'

# ---- TC23: hot_electron_escape_probability below barrier returns 0 ----
prob_stay = hot_electron_escape_probability(energy=2.0, theta=np.pi/2, barrier_height=0.5)
assert prob_stay == 0.0, '[TC23] escape probability below barrier FAILED'

# ---- TC24: random_walk_1d MSD approximates analytical value (fixed seed) ----
np.random.seed(42)
step_num_1d = 100
x2_ave, x2_max = random_walk_1d(step_num=step_num_1d, walk_num=1000, step_length=1.0)
expected_msd_1d = step_num_1d * 1.0  # <x²(N)> = N * L² for 1D
assert abs(x2_ave[-1] - expected_msd_1d) / expected_msd_1d < 0.15, '[TC24] 1D random walk MSD analytical FAILED'

# ---- TC25: random_walk_1d reproducibility with fixed seed ----
np.random.seed(42)
x2_a, _ = random_walk_1d(step_num=50, walk_num=100, step_length=1.0)
np.random.seed(42)
x2_b, _ = random_walk_1d(step_num=50, walk_num=100, step_length=1.0)
assert np.array_equal(x2_a, x2_b), '[TC25] 1D random walk reproducibility FAILED'

# ---- TC26: random_walk_3d MSD approximates analytical value (fixed seed) ----
np.random.seed(42)
step_num_3d = 100
r2_ave = random_walk_3d(step_num=step_num_3d, walk_num=1000, step_length=1.0)
expected_msd_3d = step_num_3d * 1.0  # <r²(N)> = N * L² (one axis per step)
assert abs(r2_ave[-1] - expected_msd_3d) / expected_msd_3d < 0.15, '[TC26] 3D random walk MSD analytical FAILED'

# ---- TC27: random_walk_3d reproducibility with fixed seed ----
np.random.seed(42)
r2_a = random_walk_3d(step_num=50, walk_num=100, step_length=1.0)
np.random.seed(42)
r2_b = random_walk_3d(step_num=50, walk_num=100, step_length=1.0)
assert np.array_equal(r2_a, r2_b), '[TC27] 3D random walk reproducibility FAILED'

# ---- TC28: cvtp_1d_optimize generators in [0,1) range ----
np.random.seed(42)
gens, _, _ = cvtp_1d_optimize(g_num=8, it_num=5, s_num=500, seed=42)
assert np.all(gens >= 0) and np.all(gens < 1), '[TC28] CVT generators in [0,1) FAILED'

# ---- TC29: cvtp_1d_optimize returns correct output shapes ----
np.random.seed(42)
gens2, energies2, motions2 = cvtp_1d_optimize(g_num=6, it_num=3, s_num=200, seed=42)
assert len(gens2) == 6, '[TC29] CVT generators count FAILED'
assert len(energies2) == 3 and len(motions2) == 3, '[TC29] CVT energy/motion length FAILED'

# ---- TC30: greedy_partition basic balance ----
weights_test = np.array([5.0, 3.0, 2.0, 8.0, 1.0, 4.0])
assignment, subset_sums, disc = greedy_partition(weights_test, 3)
assert len(assignment) == len(weights_test), '[TC30] greedy partition assignment length FAILED'
assert disc >= 0, '[TC30] greedy partition discrepancy non-negative FAILED'

# ---- TC31: uniform_tensor_grid_3d output shapes ----
Xg, Yg, Zg, dxg, dyg, dzg = uniform_tensor_grid_3d((0, 2), (0, 3), (0, 4), 11, 13, 15)
assert Xg.shape == (11, 13, 15), '[TC31] tensor grid X shape FAILED'
assert Yg.shape == (11, 13, 15), '[TC31] tensor grid Y shape FAILED'
assert Zg.shape == (11, 13, 15), '[TC31] tensor grid Z shape FAILED'

# ---- TC32: cfl_time_step positive ----
dt_cfl_test = cfl_time_step(5e-9, 5e-9, 5e-9)
assert dt_cfl_test > 0, '[TC32] CFL time step positive FAILED'

# ---- TC33: grid_points_in_sphere center is inside ----
np.random.seed(42)
Xgg, Ygg, Zgg, _, _, _ = uniform_tensor_grid_3d((0, 2), (0, 2), (0, 2), 5, 5, 5)
center_test = np.array([1.0, 1.0, 1.0])
mask_sphere = grid_points_in_sphere(Xgg, Ygg, Zgg, center_test, 0.5)
assert np.any(mask_sphere), '[TC33] sphere mask non-empty FAILED'

# ---- TC34: solve_waveguide_modes betas shape correct ----
N_w_test = 64
eps_prof_test = np.ones(N_w_test) * 2.0
k0_test = 1.0e7
h_test = 1e-9
betas_test, modes_test = solve_waveguide_modes(eps_prof_test, k0_test, h_test, num_modes=3, boundary='PEC')
assert len(betas_test) == 3, '[TC34] waveguide modes betas count FAILED'
assert modes_test.shape == (N_w_test, 3), '[TC34] waveguide modes shape FAILED'

# ---- TC35: effective_permittivity_mim_waveguide returns real positive ----
c_test = 2.99792458e8
omega_test_wg = 4.0e15
eps_metal_test = drude_permittivity(omega_test_wg, 1.37e16, 4.05e13, 9.84)
eps_eff_mim = effective_permittivity_mim_waveguide(
    eps_metal_test, 1.0, width_metal=10e-9, width_dielectric=5e-9,
    wavelength=2*np.pi*c_test/omega_test_wg
)
assert np.isreal(eps_eff_mim) and np.real(eps_eff_mim) > 0, '[TC35] effective permittivity real positive FAILED'

# ---- TC36: hot_carrier_generation_rate non-negative ----
field_hc = np.ones((8, 8)) * 1.0e5
G_hc_test = hot_carrier_generation_rate(field_hc, 4.0e15, eps_metal_test, dx=10e-9, dy=10e-9)
assert np.all(G_hc_test >= 0), '[TC36] hot carrier generation non-negative FAILED'

# ---- TC37: extract_multiresolution_hotspots returns expected types ----
np.random.seed(42)
field_mr = np.abs(np.random.randn(32, 32))
scales_mr, coeffs_mr = extract_multiresolution_hotspots(field_mr, threshold_factor=1.5)
assert isinstance(scales_mr, list), '[TC37] multiresolution scales list FAILED'
assert isinstance(coeffs_mr, dict), '[TC37] multiresolution coefficients dict FAILED'

# ---- TC38: mie_cross_sections array omega returns array output ----
omegas_arr = np.linspace(3.0e15, 6.0e15, 5)
sig_ext_arr, sig_sca_arr = mie_cross_sections(omegas_arr, 30e-9)
assert sig_ext_arr.shape == omegas_arr.shape, '[TC38] mie array output shape FAILED'

# ---- TC39: spectral_response_dipole non-negative intensity ----
np.random.seed(42)
dt_spec = 1e-16
N_spec = 128
t_spec = np.arange(N_spec) * dt_spec
p_spec = np.column_stack([np.sin(4e15 * t_spec), np.zeros(N_spec), np.zeros(N_spec)])
freqs_spec, intens_spec = spectral_response_dipole(p_spec, dt_spec)
assert np.all(intens_spec >= 0), '[TC39] spectral response non-negative FAILED'

# ---- TC40: generate_sphere_surface_grid output shapes ----
x_s, y_s, z_s, theta_s, phi_s, area_s = generate_sphere_surface_grid(n_theta=16, n_phi=32, radius=30e-9)
assert x_s.shape == (16, 32), '[TC40] sphere grid x shape FAILED'
assert area_s.shape == (16, 32), '[TC40] sphere grid area shape FAILED'

# ---- TC41: find_connected_components_2d returns integer labels ----
labels_test = find_connected_components_2d(np.abs(np.random.randn(6, 6)), 0.0)[0]
assert labels_test.dtype == int, '[TC41] connected components labels int FAILED'

# ---- TC42: calculate_collection_efficiency in [0,1] with fixed seed ----
np.random.seed(42)
eff_test = calculate_collection_efficiency(
    particle_radius=30e-9, mfp=10e-9, tau=10e-15,
    barrier_height=0.8, plasmon_energy=2.0, num_walkers=100
)
assert 0.0 <= eff_test <= 1.0, '[TC42] collection efficiency in [0,1] FAILED'

# ---- TC43: Legendre nodes in [-1,1] ----
x_nodes, _ = legendre_nodes_weights_1d(10)
assert np.all(x_nodes >= -1.0) and np.all(x_nodes <= 1.0), '[TC43] Legendre nodes in [-1,1] FAILED'

# ---- TC44: expand_bracket finds sign change ----
from resonance_finder import expand_bracket
def f_bracket(x):
    return x - 2.0
a_exp, b_exp = expand_bracket(f_bracket, 0.0, 1.0)
assert f_bracket(a_exp) * f_bracket(b_exp) <= 0, '[TC44] expand bracket sign change FAILED'

# ---- TC45: r83s_matvec correct shape ----
n_r83 = 10
a_r83 = np.array([0.5, 2.0, 0.5])
x_r83 = np.ones(n_r83)
b_r83 = r83s_matvec(n_r83, a_r83, x_r83)
assert len(b_r83) == n_r83, '[TC45] r83s_matvec output shape FAILED'

# ---- TC46: conjugate_gradient_r83s solves known system ----
n_cg = 5
a_cg = np.array([1.0, 2.0, 1.0])
x_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b_cg = r83s_matvec(n_cg, a_cg, x_true)
x0_cg = np.zeros(n_cg)
x_sol = conjugate_gradient_r83s(n_cg, a_cg, b_cg, x0_cg)
assert np.allclose(x_sol, x_true, atol=1e-10), '[TC46] conjugate gradient solves system FAILED'

# ---- TC47: build_coupling_matrix square shape ----
np.random.seed(42)
N_coup = 4
pos_coup = np.random.rand(N_coup, 3) * 50e-9
alpha_coup = np.full(N_coup, complex(1e-30, 0.0))
A_coup = build_coupling_matrix(pos_coup, alpha_coup, 4.0e15, eps_medium=1.0)
assert A_coup.shape == (3*N_coup, 3*N_coup), '[TC47] coupling matrix shape FAILED'

# ---- TC48: build_coupling_graph returns arc list ----
adj, arcs = build_coupling_graph(pos_coup, 4.0e15, eps_medium=1.0, threshold=1.0e20)
assert isinstance(arcs, list), '[TC48] coupling graph arcs list FAILED'

# ---- TC49: spectral_partition_laplacian returns valid assignment ----
from domain_partition import spectral_partition_laplacian
adj_mat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
assign_spec = spectral_partition_laplacian(adj_mat, 2)
assert len(assign_spec) == 3, '[TC49] spectral partition length FAILED'
assert np.min(assign_spec) >= 0 and np.max(assign_spec) < 2, '[TC49] spectral partition range FAILED'

# ---- TC50: integrate_over_box constant function ----
a_int = np.array([0.0, 0.0, 0.0])
b_int = np.array([1.0, 1.0, 1.0])
f_const = lambda x, y, z: np.ones_like(x)
result_const = integrate_over_box(f_const, a_int, b_int, nx=4, ny=4, nz=4)
assert abs(result_const - 1.0) < 1e-10, '[TC50] integrate constant FAILED'
