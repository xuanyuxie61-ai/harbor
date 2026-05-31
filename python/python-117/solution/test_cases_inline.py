# ---- TC01: topology_checksum of valid identifier returns 0 ----
from topology_validator import topology_checksum
cs = topology_checksum("PH-00000")
assert cs == 0, '[TC01] topology_checksum valid string FAILED'

# ---- TC02: topology_check_digit produces digit that makes checksum zero ----
from topology_validator import topology_check_digit
cd = topology_check_digit("PH-0000")
ts = "PH-0000" + str(cd)
assert topology_checksum(ts) == 0, '[TC02] topology_check_digit FAILED'

# ---- TC03: generate_topologies produces all-valid list with correct count ----
bts = ["PH", "GL"]
topos = generate_topologies(bts, count=2)
assert all(validate_topology(t) for t in topos), '[TC03] generate_topologies validity FAILED'
assert len(topos) == 4, '[TC03] generate_topologies count FAILED'

# ---- TC04: TriangulatedMembrane.create_planar_sheet correct dimensions ----
import numpy as np
mem = TriangulatedMembrane.create_planar_sheet(nx=8, ny=8, lx=10.0, ly=10.0)
assert mem.n_v == 64, '[TC04] planar sheet vertex count FAILED'
assert mem.n_e == 2 * 7 * 7, '[TC04] planar sheet element count FAILED'

# ---- TC05: planar membrane bending energy finite and non-negative ----
E_b = mem.bending_energy(kappa=20.0)
assert np.isfinite(E_b), '[TC05] bending energy finite FAILED'
assert E_b >= 0, '[TC05] bending energy non-negative FAILED'

# ---- TC06: membrane element areas all positive, total area = lx*ly ----
areas = mem.compute_element_areas()
assert np.all(areas > 0), '[TC06] element areas positive FAILED'
assert abs(np.sum(areas) - 100.0) < 1e-10, '[TC06] total area FAILED'

# ---- TC07: membrane normals all unit length (planar sheet -> all same dir) ----
normals = mem.compute_normals()
n_norms = np.linalg.norm(normals, axis=1)
assert np.all(np.abs(n_norms - 1.0) < 1e-12), '[TC07] normals unit length FAILED'

# ---- TC08: PoissonBoltzmannSolver debye_length is positive ----
pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0, n_0=0.1, epsilon=80.0*8.854e-12, T=300.0)
assert pb.debye_length() > 0, '[TC08] debye_length positive FAILED'

# ---- TC09: PB Jacobi solver returns finite phi and converges ----
phi_j, r_j, it_j, res_j = pb.solve_jacobi(n_grid=129, it_max=10000)
assert np.all(np.isfinite(phi_j)), '[TC09] Jacobi phi finite FAILED'
assert res_j < 1e-2, '[TC09] Jacobi residual FAILED'

# ---- TC10: PB Jacobi vs BiCG consistency ----
phi_b, r_b, it_b, res_b = pb.solve_bicg(n_grid=129)
diff_pb = np.max(np.abs(phi_j - phi_b))
assert diff_pb < 0.1, '[TC10] Jacobi vs BiCG consistency FAILED'

# ---- TC11: PB electrostatic_force is finite ----
ef = pb.electrostatic_force(n_grid=129)
assert np.isfinite(ef), '[TC11] electrostatic_force finite FAILED'

# ---- TC12: transport initial_condition bounded between 0 and c0 ----
ts_port = AdvectionDiffusionSolver(L=10.0, v=0.05, D=0.1, c0=0.1, nx=101)
c_init = ts_port.initial_condition(depletion_width=2.0)
assert np.all(c_init >= 0), '[TC12] init conc non-negative FAILED'
assert np.all(c_init <= ts_port.c0), '[TC12] init conc bounded FAILED'

# ---- TC13: transport solve preserves non-negative concentration ----
c_final, _ = ts_port.solve(n_steps=100)
assert np.all(c_final >= 0), '[TC13] final conc non-negative FAILED'

# ---- TC14: gauss_legendre_1d weights sum to 2 ----
from potential_energy import gauss_legendre_1d
x_gl, w_gl = gauss_legendre_1d(5)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC14] GL weights sum FAILED'

# ---- TC15: integrate_nd of constant function returns hyper-volume ----
def const_f(x):
    return np.ones(x.shape[0])
a_test = np.array([0.0, 0.0])
b_test = np.array([2.0, 3.0])
val = integrate_nd(const_f, a_test, b_test, n_per_dim=5)
assert abs(val - 6.0) < 1e-10, '[TC15] integrate_nd constant FAILED'

# ---- TC16: membrane_binding_energy_integral numeric approx analytic ----
E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0, sigma=1.0, n_quad=5)
assert abs(E_num - E_exact) / (abs(E_exact) + 1e-12) < 0.01, '[TC16] binding energy integral FAILED'

# ---- TC17: CubicSplineInterpolator recovers values at original nodes (x^2) ----
x_spl = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_spl = x_spl ** 2
spl = CubicSplineInterpolator(x_spl, y_spl)
y_eval = spl.evaluate(x_spl)
assert np.max(np.abs(y_eval - y_spl)) < 1e-12, '[TC17] spline at nodes FAILED'

# ---- TC18: CubicSplineInterpolator derivative of x^2 is positive and monotonic ----
y_der = spl.derivative(np.array([1.0, 2.0, 3.0]))
assert np.all(y_der > 0), '[TC18] spline derivative positive FAILED'
assert y_der[0] < y_der[1] < y_der[2], '[TC18] spline derivative monotonic FAILED'

# ---- TC19: pdf_to_histogram produces non-negative densities ----
def gpdf(x):
    return np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)
b_p, b_l, b_r = pdf_to_histogram(gpdf, n_bins=32, x_min=-4.0, x_max=4.0)
assert np.all(b_p >= 0), '[TC19] histogram non-negative FAILED'

# ---- TC20: histogram_to_cdf is monotonic with correct endpoints ----
c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
assert np.all(np.diff(c_y) >= 0), '[TC20] CDF monotonic FAILED'
assert abs(c_y[0]) < 1e-12, '[TC20] CDF start FAILED'
assert abs(c_y[-1] - 1.0) < 1e-12, '[TC20] CDF end FAILED'

# ---- TC21: cdf_to_sample returns correct count and finite values ----
np.random.seed(42)
samps = cdf_to_sample(c_x, c_y, n_samples=1000)
assert len(samps) == 1000, '[TC21] cdf_to_sample count FAILED'
assert np.all(np.isfinite(samps)), '[TC21] cdf_to_sample finite FAILED'

# ---- TC22: sphere_sample_marsaglia produces unit vectors ----
np.random.seed(42)
pts = sphere_sample_marsaglia(500)
nms = np.linalg.norm(pts, axis=0)
assert np.all(np.abs(nms - 1.0) < 1e-12), '[TC22] sphere norms FAILED'

# ---- TC23: svd_deformation_modes returns valid shapes and non-negative S ----
np.random.seed(42)
D_svd = np.random.randn(3, 20)
U_svd, S_svd, Vt_svd = svd_deformation_modes(D_svd)
assert np.all(S_svd >= 0), '[TC23] SVD singular values non-negative FAILED'
assert U_svd.shape == (3, 3), '[TC23] SVD U shape FAILED'
assert Vt_svd.shape == (3, 20), '[TC23] SVD Vt shape FAILED'

# ---- TC24: boltzmann_acceptance always accepts negative delta_E ----
np.random.seed(42)
acc_neg = boltzmann_acceptance(delta_E=-10.0, T=300.0, k_B=8.314e-3)
assert acc_neg == True, '[TC24] Boltzmann negative energy FAILED'

# ---- TC25: boltzmann_acceptance returns True or False ----
np.random.seed(42)
acc_result = boltzmann_acceptance(delta_E=5.0, T=300.0, k_B=8.314e-3)
assert acc_result in (True, False), '[TC25] Boltzmann return value FAILED'

# ---- TC26: exponential_kernel correct values at tau=0 and monotonic decay ----
from correlated_forces import exponential_kernel
tau_arr = np.array([0.0, 0.1, 0.2])
kern = exponential_kernel(tau_arr, gamma0=1.0, tau_mem=0.1)
assert abs(kern[0] - 1.0) < 1e-10, '[TC26] kernel at zero FAILED'
assert kern[1] < kern[0], '[TC26] kernel monotonic a FAILED'
assert kern[2] < kern[1], '[TC26] kernel monotonic b FAILED'

# ---- TC27: generate_correlated_forces produces finite sequence of correct length ----
np.random.seed(42)
f_corr = generate_correlated_forces(n_steps=1000, dt=0.001, gamma0=1.0, tau_mem=0.05)
assert len(f_corr) == 1000, '[TC27] correlated forces length FAILED'
assert np.all(np.isfinite(f_corr)), '[TC27] correlated forces finite FAILED'

# ---- TC28: correlated forces reproducibility with fixed seed ----
np.random.seed(42)
f1 = generate_correlated_forces(n_steps=100, dt=0.001, gamma0=1.0, tau_mem=0.05)
np.random.seed(42)
f2 = generate_correlated_forces(n_steps=100, dt=0.001, gamma0=1.0, tau_mem=0.05)
assert np.allclose(f1, f2), '[TC28] correlated forces reproducibility FAILED'

# ---- TC29: colored_noise_spectrum PSD non-negative ----
freqs, psd = colored_noise_spectrum(f_corr, dt=0.001)
assert np.all(psd >= 0), '[TC29] PSD non-negative FAILED'

# ---- TC30: chebyshev_proxy_rootfinder finds sin(x)=0 at pi ----
from equilibrium_solver import chebyshev_proxy_rootfinder
roots_sin = chebyshev_proxy_rootfinder(lambda x: np.sin(x), 2.5, 3.5, N=32)
found_pi = any(abs(r - np.pi) < 1e-6 for r in roots_sin)
assert found_pi, '[TC30] CPR sin(pi) root FAILED'

# ---- TC31: chebyshev_proxy_rootfinder finds quadratic roots ----
roots_quad = chebyshev_proxy_rootfinder(lambda x: x**2 - 4.0, -3.0, 3.0, N=32)
found_m2 = any(abs(r - (-2.0)) < 1e-6 for r in roots_quad)
found_p2 = any(abs(r - 2.0) < 1e-6 for r in roots_quad)
assert found_m2 and found_p2, '[TC31] CPR quadratic roots FAILED'

# ---- TC32: generate_training_data correct shapes and finite values ----
X_tr, y_tr = generate_training_data(n_samples=100, seed=42)
assert X_tr.shape == (100, 6), '[TC32] training X shape FAILED'
assert y_tr.shape == (100, 1), '[TC32] training y shape FAILED'
assert np.all(np.isfinite(X_tr)), '[TC32] X finite FAILED'
assert np.all(np.isfinite(y_tr)), '[TC32] y finite FAILED'

# ---- TC33: NanoparticleLangevinDynamics total_force is finite ----
nld = NanoparticleLangevinDynamics(z0=5.0)
F_tot = nld.total_force(3.0, debye_length=1.0)
assert np.isfinite(F_tot), '[TC33] total_force finite FAILED'

# ---- TC34: NanoparticleLangevinDynamics force components all finite ----
F_vdw = nld.force_vdw(3.0)
F_bend = nld.force_bending(3.0)
F_bind = nld.force_binding(3.0)
F_elec = nld.force_electrostatic(3.0, debye_length=1.0)
assert np.isfinite(F_vdw), '[TC34] vdw force finite FAILED'
assert np.isfinite(F_bend), '[TC34] bending force finite FAILED'
assert np.isfinite(F_bind), '[TC34] binding force finite FAILED'
assert np.isfinite(F_elec), '[TC34] electrostatic force finite FAILED'

# ---- TC35: Langevin step_euler_maruyama produces finite z ----
np.random.seed(42)
z_step = nld.step_euler_maruyama(debye_length=1.0)
assert np.isfinite(z_step), '[TC35] Euler-Maruyama step finite FAILED'

# ---- TC36: parallel_map returns correct results ----
data_in = list(range(1, 11))
results = parallel_map(_square_task, data_in, n_workers=2)
assert results == [x**2 for x in data_in], '[TC36] parallel_map FAILED'

# ---- TC37: check_environment returns bool ----
env_ok = check_environment()
assert isinstance(env_ok, bool), '[TC37] check_environment type FAILED'

# ---- TC38: get_platform_info has all required keys ----
info = get_platform_info()
for key in ['python_version', 'platform', 'numpy_mkl', 'float_eps', 'max_threads']:
    assert key in info, f'[TC38] platform_info missing key {key} FAILED'

# ---- TC39: NeuralSurrogate trains and produces finite predictions ----
np.random.seed(42)
Xs, ys = generate_training_data(n_samples=128, seed=42)
nn_test = NeuralSurrogate(input_dim=6, hidden_dims=[16, 8], lr=0.005, lambda_reg=1e-5, seed=42)
losses = nn_test.train(Xs, ys, epochs=30, batch_size=32, verbose=False)
y_pred = nn_test.predict(Xs[:10])
assert np.all(np.isfinite(y_pred)), '[TC39] NN predict finite FAILED'
assert y_pred.shape == (10, 1), '[TC39] NN predict shape FAILED'

# ---- TC40: sample_random_orientation returns valid rotation matrix ----
np.random.seed(42)
R = sample_random_orientation()
assert R.shape == (3, 3), '[TC40] rotation matrix shape FAILED'
assert abs(np.linalg.det(R) - 1.0) < 1e-10, '[TC40] rotation det FAILED'
assert np.all(np.abs(R @ R.T - np.eye(3)) < 1e-10), '[TC40] rotation orthogonality FAILED'

# ---- TC41: transport compute_flux returns finite value ----
flux_val = ts_port.compute_flux(c_final)
assert np.isfinite(flux_val), '[TC41] compute_flux finite FAILED'

# ---- TC42: find_equilibrium_distances finds roots for polynomial force ----
def test_force(z):
    return (z - 3.0) * (z - 7.0)
roots_eq, stab_eq = find_equilibrium_distances(test_force, z_min=1.0, z_max=9.0)
assert len(roots_eq) >= 2, '[TC42] find_equilibrium_distances count FAILED'
found_3 = any(abs(r - 3.0) < 0.1 for r in roots_eq)
found_7 = any(abs(r - 7.0) < 0.1 for r in roots_eq)
assert found_3 and found_7, '[TC42] find_equilibrium_distances roots FAILED'

# ---- TC43: topology_checksum of string without digits returns non-zero ----
cs_empty = topology_checksum("ABC-DEF")
assert cs_empty != 0, '[TC43] topology_checksum no digits FAILED'

# ---- TC44: validate_topology returns False for no-digit string ----
assert validate_topology("NO-DIGITS-HERE") == False, '[TC44] validate no-digit string FAILED'

# ---- TC45: toep_cholesky_lower produces valid lower-triangular Cholesky factor ----
from correlated_forces import toep_cholesky_lower
np.random.seed(42)
t_row = np.array([2.0, 0.5, 0.2])
L45 = toep_cholesky_lower(3, t_row)
A_recon = L45 @ L45.T
assert L45.shape == (3, 3), '[TC45] Cholesky L shape FAILED'
assert abs(A_recon[0, 0] - t_row[0]) < 1e-10, '[TC45] Cholesky diag FAILED'
assert np.all(A_recon >= 0), '[TC45] Cholesky psd FAILED'
