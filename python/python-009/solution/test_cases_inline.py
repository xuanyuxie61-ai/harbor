# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

from sphere_quadrature import integrate_sphere_function, henyey_greenstein_phase_function, compute_scatter_angles, spherical_to_cartesian, cartesian_to_spherical
from inversion_solver import BisectionRootFinder
from data_io import convert_triangle_to_fem


# ---- TC01: AtmosphericProfile gravity returns finite positive values ----
planet = AtmosphericProfile(planet_mass_kg=1.898e27, planet_radius_m=6.9911e7, star_mass_kg=1.98847e30, orbital_distance_m=1.496e11)
g = planet.gravity(np.array([0.0, 1e6, 1e7]))
assert np.all(g > 0) and np.all(np.isfinite(g)), '[TC01] AtmosphericProfile gravity returns finite positive values FAILED'

# ---- TC02: Guillot temperature profile finite and positive ----
P = np.logspace(1, 5, 10)
T = planet.guillot_temperature_profile(P, T_int=150.0, T_irr=1000.0)
assert np.all(T > 0) and np.all(np.isfinite(T)), '[TC02] Guillot temperature profile finite and positive FAILED'

# ---- TC03: Hydrostatic pressure grid monotonic increasing ----
P_grid = planet.hydrostatic_pressure_grid(10, 1e-1, 1e7)
assert np.all(np.diff(P_grid) > 0), '[TC03] Hydrostatic pressure grid monotonic increasing FAILED'

# ---- TC04: Scale height positive and finite ----
H = planet.scale_height(1000.0, 2.3)
assert H > 0 and np.isfinite(H), '[TC04] Scale height positive and finite FAILED'

# ---- TC05: ChemicalEquilibrium H2 abundance equals 0.85 ----
chem = ChemicalEquilibrium(['H2', 'He', 'H2O'])
vmr_h2 = chem.equilibrium_abundance('H2', np.array([1000.0]), np.array([1e5]))
assert abs(vmr_h2[0] - 0.85) < 1e-10, '[TC05] ChemicalEquilibrium H2 abundance equals 0.85 FAILED'

# ---- TC06: Mean molecular weight calculation in expected range ----
abundances = {'H2': np.array([0.85]), 'He': np.array([0.15])}
mu = chem.mean_molecular_weight(abundances)
assert 2.0 < mu[0] < 3.0, '[TC06] Mean molecular weight calculation in expected range FAILED'

# ---- TC07: Cloud optical depth non-negative ----
cloud = CloudModel(P_cloud_top=1e3, P_cloud_base=1e5, cloud_opacity=2.0)
tau_cloud = cloud.cloud_optical_depth(P_grid)
assert np.all(tau_cloud >= 0), '[TC07] Cloud optical depth non-negative FAILED'

# ---- TC08: 2D shell mesh generation shape correctness ----
mesh = AtmosphericMesh(n_layers=10, P_top=1e-1, P_bot=1e7, planet_radius_m=6.9911e7)
nodes, elements = mesh.generate_2d_shell_mesh(n_angular=8)
assert nodes.shape[1] == 2 and elements.shape[1] == 3, '[TC08] 2D shell mesh generation shape correctness FAILED'

# ---- TC09: Distance function sphere shell sign properties ----
points = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0]])
d = distance_function_sphere_shell(points, R_inner=5.0, R_outer=10.0)
assert d[0] > 0 and d[1] <= 0 and d[2] > 0, '[TC09] Distance function sphere shell sign properties FAILED'

# ---- TC10: Mesh size function within bounds ----
pts = np.array([[6.9911e7, 0.0], [7.5e7, 0.0]])
h = mesh_size_function(pts, R_p=6.9911e7, h_min=1e3, h_max=1e5)
assert np.all(h >= 1e3) and np.all(h <= 1e5), '[TC10] Mesh size function within bounds FAILED'

# ---- TC11: Voigt profile positive and finite ----
nu = np.linspace(-1e10, 1e10, 1000)
profile = LineProfile.voigt_profile(nu, nu0=0.0, alpha_d=1e9, gamma_l=1e8)
assert np.all(profile >= 0) and np.all(np.isfinite(profile)), '[TC11] Voigt profile positive and finite FAILED'

# ---- TC12: Voigt profile normalization integral near unity ----
nu_fine = np.linspace(-5e10, 5e10, 20001)
profile_norm = LineProfile.voigt_profile(nu_fine, nu0=0.0, alpha_d=1e9, gamma_l=1e8)
I_voigt = np.trapezoid(profile_norm, nu_fine)
assert abs(I_voigt - 1.0) < 0.1, '[TC12] Voigt profile normalization integral near unity FAILED'

# ---- TC13: Molecular cross section non-negative and finite ----
mol = MolecularCrossSection('H2O')
wn = np.linspace(1000, 7000, 100)
sigma = mol.compute_cross_section(wn, T=1000.0, P=1e5)
assert np.all(sigma >= 0) and np.all(np.isfinite(sigma)), '[TC13] Molecular cross section non-negative and finite FAILED'

# ---- TC14: Rayleigh scattering follows lambda^-4 scaling ----
lam = np.array([1.0, 2.0])
sigma_r = RayleighScattering.cross_section_H2(lam)
assert abs(sigma_r[1] / sigma_r[0] - (1.0/2.0)**4) < 1e-10, '[TC14] Rayleigh scattering follows lambda^-4 scaling FAILED'

# ---- TC15: Radiative transfer optical depth monotonic ----
rt = RadiativeTransferSolver(np.linspace(0.3, 5.0, 10), planet_radius_m=6.9911e7)
P_small = np.logspace(3, 5, 5)
T_small = np.full_like(P_small, 1000.0)
abund = {'H2': np.full_like(P_small, 0.85), 'He': np.full_like(P_small, 0.15)}
cs = {'H2': np.zeros((5, 10)), 'He': np.zeros((5, 10))}
g_small = np.full_like(P_small, 10.0)
tau = rt.compute_optical_depth(P_small, T_small, abund, cs, g_small)
assert np.all(np.diff(tau[:, 0]) >= -1e-15), '[TC15] Radiative transfer optical depth monotonic FAILED'

# ---- TC16: Gauss-Legendre weights sum to 2 ----
mu_gl, w_mu_gl, phi_gl, w_phi_gl = gauss_legendre_angles(n_polar=8, n_azimuth=16)
assert abs(np.sum(w_mu_gl) - 2.0) < 1e-14, '[TC16] Gauss-Legendre weights sum to 2 FAILED'

# ---- TC17: Sphere integral of constant function equals 4pi ----
f_const = np.ones((8, 16))
I_sphere = integrate_sphere_function(f_const, w_mu_gl, w_phi_gl)
assert abs(I_sphere - 4.0 * np.pi) < 1e-10, '[TC17] Sphere integral of constant function equals 4pi FAILED'

# ---- TC18: Henyey-Greenstein isotropic for g=0 ----
cos_theta = np.linspace(-1, 1, 100)
p_hg = henyey_greenstein_phase_function(cos_theta, g=0.0)
assert np.allclose(p_hg, 1.0/(4.0*np.pi)), '[TC18] Henyey-Greenstein isotropic for g=0 FAILED'

# ---- TC19: Delta-Eddington energy conservation R+T bounded ----
tau_test = np.linspace(0.1, 5.0, 10)
omega_test = np.full_like(tau_test, 0.8)
R_de, T_de = delta_eddington_approximation(tau_test, omega_test, g=0.5, mu0=0.5)
assert np.all(R_de >= 0) and np.all(T_de >= 0) and np.all(R_de <= 1.0) and np.all(T_de <= 1.0), '[TC19] Delta-Eddington R and T within [0,1] FAILED'

# ---- TC20: CRS matrix from dense and multiply correctness ----
A_dense = np.array([[2.0, 1.0], [1.0, 2.0]])
crs = CRSMatrix.from_dense(A_dense)
x_vec = np.array([1.0, 2.0])
y_vec = crs.multiply(x_vec)
assert np.allclose(y_vec, A_dense @ x_vec), '[TC20] CRS matrix from dense and multiply correctness FAILED'

# ---- TC21: GMRES solver accuracy for tridiagonal system ----
n_test = 20
A_tridiag = np.diag(np.ones(n_test)*2.0) + np.diag(np.ones(n_test-1)*(-1.0), 1) + np.diag(np.ones(n_test-1)*(-1.0), -1)
crs_A = CRSMatrix.from_dense(A_tridiag)
x_true = np.ones(n_test)
b_vec = crs_A.multiply(x_true)
x_sol, iters_gmres, res_gmres = crs_gmres(crs_A, b_vec, tol=1e-10, max_iter=200)
rel_err = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
assert rel_err < 1e-6, '[TC21] GMRES solver accuracy for tridiagonal system FAILED'

# ---- TC22: Simplex sampler sum to one ----
np.random.seed(42)
samples = SimplexSampler.sample_unit_simplex(m=5, n=1000, seed=42)
sum_samples = np.sum(samples, axis=1)
assert np.allclose(sum_samples, 1.0), '[TC22] Simplex sampler sum to one FAILED'

# ---- TC23: Pink noise zero mean and unit std ----
np.random.seed(42)
png = PinkNoiseGenerator(beta=1.0)
noise = png.generate(n=10000, seed=42)
assert abs(np.mean(noise)) < 0.05 and abs(np.std(noise) - 1.0) < 0.05, '[TC23] Pink noise zero mean and unit std FAILED'

# ---- TC24: Tikhonov first order difference matrix shape ----
L1 = TikhonovRegularization.first_order_difference_matrix(5)
assert L1.shape == (4, 5), '[TC24] Tikhonov first order difference matrix shape FAILED'

# ---- TC25: Bisection root finder solves x^2-4=0 ----
finder = BisectionRootFinder(a=0.0, b=5.0, tol=1e-10)
root = finder.solve(lambda x: x**2 - 4.0)
assert abs(root - 2.0) < 1e-9, '[TC25] Bisection root finder solves x^2-4=0 FAILED'

# ---- TC26: TecDataset variable add and retrieve ----
ds = TecDataset(title="Test", variables=[])
ds.add_variable("X", np.array([1.0, 2.0, 3.0]))
ds.add_variable("Y", np.array([4.0, 5.0, 6.0]))
assert np.allclose(ds.get_variable("X"), np.array([1.0, 2.0, 3.0])), '[TC26] TecDataset variable add and retrieve FAILED'

# ---- TC27: Triangle to FEM 1-based index conversion ----
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_1based = np.array([[1, 2, 3]])
fem_nodes, fem_elems = convert_triangle_to_fem(nodes, elems_1based)
assert np.allclose(fem_elems, np.array([[0, 1, 2]])), '[TC27] Triangle to FEM 1-based index conversion FAILED'

# ---- TC28: Spectrum ASCII write and read roundtrip ----
wavelength = np.array([1.0, 2.0, 3.0])
flux = np.array([10.0, 20.0, 30.0])
error = np.array([0.1, 0.2, 0.3])
write_spectrum_ascii(wavelength, flux, error, "test_spectrum_tmp.dat")
data_back = read_spectrum_ascii("test_spectrum_tmp.dat")
assert np.allclose(data_back['wavelength'], wavelength) and np.allclose(data_back['flux'], flux), '[TC28] Spectrum ASCII write and read roundtrip FAILED'
import os
os.remove("test_spectrum_tmp.dat")

# ---- TC29: Coordinate conversion roundtrip ----
theta = np.array([0.5, 1.0, 1.5])
phi = np.array([0.0, 1.0, 2.0])
x, y, z = spherical_to_cartesian(theta, phi)
theta_back, phi_back = cartesian_to_spherical(x, y, z)
assert np.allclose(theta, theta_back) and np.allclose(phi, phi_back), '[TC29] Coordinate conversion roundtrip FAILED'

# ---- TC30: Scatter angle same direction equals 1 ----
mu_s = np.array([0.5])
phi_s = np.array([1.0])
cos_scatter = compute_scatter_angles(mu_s, phi_s, mu_s, phi_s)
assert np.allclose(cos_scatter, 1.0), '[TC30] Scatter angle same direction equals 1 FAILED'

print('\n全部 30 个测试通过!\n')
