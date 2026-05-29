# ---- TC01: Cosmology.H(1.0) returns finite positive value ----
cosmo = Cosmology()
H1 = cosmo.H(1.0)
assert np.isfinite(H1) and H1 > 0.0, '[TC01] Cosmology.H(1.0) returns finite positive value FAILED'

# ---- TC02: Omega_m_a(1.0) approximately equals Omega_m ----
cosmo = Cosmology()
oma = cosmo.Omega_m_a(1.0)
assert abs(oma - cosmo.Omega_m) < 0.01, '[TC02] Omega_m_a(1.0) approximately equals Omega_m FAILED'

# ---- TC03: age_of_universe(1.0) in reasonable range ----
cosmo = Cosmology()
age = cosmo.age_of_universe(1.0)
assert 10.0 < age < 15.0, '[TC03] age_of_universe(1.0) in reasonable range FAILED'

# ---- TC04: delta_c(0.0) approximately 1.686 ----
cosmo = Cosmology()
dc = cosmo.delta_c(0.0)
assert abs(dc - 1.686) < 0.05, '[TC04] delta_c(0.0) approximately 1.686 FAILED'

# ---- TC05: comoving_distance(0.0) is zero ----
cosmo = Cosmology()
d0 = cosmo.comoving_distance(0.0)
assert d0 == 0.0, '[TC05] comoving_distance(0.0) is zero FAILED'

# ---- TC06: bisect_root_finder finds sqrt(2) ----
cosmo = Cosmology()
a, b, it = cosmo.bisect_root_finder(lambda x: x**2 - 2.0, 1.0, 2.0)
root = (a + b) * 0.5
assert abs(root - np.sqrt(2.0)) < 1e-10, '[TC06] bisect_root_finder finds sqrt(2) FAILED'

# ---- TC07: linear growth factor normalized to 1 at a=1 ----
cosmo = Cosmology()
a_arr, D_arr, _ = cosmo.compute_linear_growth_factor(a_min=1e-4, a_max=1.0, n_steps=100)
assert np.all(np.diff(a_arr) > 0), '[TC07] linear growth factor normalized to 1 at a=1 FAILED'
assert abs(D_arr[-1] - 1.0) < 1e-6, '[TC07] linear growth factor normalized to 1 at a=1 FAILED'

# ---- TC08: PowerSpectrum at k=0 is finite and non-negative ----
cosmo = Cosmology()
ps = PowerSpectrum(cosmo)
Pk0 = ps(np.array([0.0]))[0]
assert np.isfinite(Pk0) and Pk0 >= 0.0, '[TC08] PowerSpectrum at k=0 is finite and non-negative FAILED'

# ---- TC09: Gauss-Hermite weights sum to sqrt(pi) ----
nodes, weights = gauss_hermite_nodes_weights(16)
gh_sum = np.sum(weights)
assert abs(gh_sum - np.sqrt(np.pi)) < 1e-12, '[TC09] Gauss-Hermite weights sum to sqrt(pi) FAILED'

# ---- TC10: latin_edge_sample shape correct ----
np.random.seed(42)
latin = latin_edge_sample(dim_num=3, point_num=8)
assert latin.shape == (3, 8), '[TC10] latin_edge_sample shape correct FAILED'

# ---- TC11: particle_mass_from_cosmology returns positive ----
cosmo = Cosmology()
m_part = particle_mass_from_cosmology(16, 100.0, cosmo)
assert m_part > 0.0, '[TC11] particle_mass_from_cosmology returns positive FAILED'

# ---- TC12: CIC deposit conserves total mass ----
np.random.seed(42)
solver = PMSolver(16, 100.0)
n_part = 100
pos = np.random.rand(n_part, 3) * 100.0
mass = np.ones(n_part) * 1e10
rho = solver.cic_deposit(pos, mass)
total_mass_grid = rho.sum() * solver.dx**3
total_mass_part = mass.sum()
assert abs(total_mass_grid - total_mass_part) / total_mass_part < 1e-10, '[TC12] CIC deposit conserves total mass FAILED'

# ---- TC13: compute_density_contrast with uniform density returns zeros ----
solver = PMSolver(8, 100.0)
rho = np.ones((8, 8, 8)) * 2.0
delta = solver.compute_density_contrast(rho, 2.0)
assert np.allclose(delta, 0.0), '[TC13] compute_density_contrast with uniform density returns zeros FAILED'

# ---- TC14: solve_poisson_fft returns finite real potential ----
np.random.seed(42)
solver = PMSolver(16, 100.0)
delta = np.random.randn(16, 16, 16) * 0.1
phi = solver.solve_poisson_fft(delta, a_scale=1.0)
assert np.all(np.isfinite(phi)), '[TC14] solve_poisson_fft returns finite real potential FAILED'

# ---- TC15: 1D advection mass conservation ----
nx = 101
L = 1.0
dx = L / (nx - 1)
x = np.linspace(0.0, L, nx)
u0 = np.exp(-((x - 0.5)**2) / (2 * 0.02**2))
advect = AdvectionSolver(nx, dx, c=1.0)
t_arr, u_arr = advect.evolve_1d(u0, t_final=0.1, n_steps=100)
mass0 = np.sum(u0) * dx
massf = np.sum(u_arr[-1]) * dx
assert abs(massf - mass0) / mass0 < 1e-6, '[TC15] 1D advection mass conservation FAILED'

# ---- TC16: test_mass_conservation small error ----
err = test_mass_conservation()
assert err < 1e-8, '[TC16] test_mass_conservation small error FAILED'

# ---- TC17: drift_step periodic boundary ----
integrator = NBodyIntegrator(Cosmology(), softening=0.5, eta=0.2, use_adaptive_step=False)
pos = np.array([[99.0, 0.0, 0.0]])
vel = np.array([[2.0, 0.0, 0.0]])
pos_new = integrator.drift_step(pos, vel, 1.0, 100.0)
assert pos_new[0, 0] == 1.0, '[TC17] drift_step periodic boundary FAILED'

# ---- TC18: kick_step updates velocity correctly ----
integrator = NBodyIntegrator(Cosmology(), softening=0.5, eta=0.2, use_adaptive_step=False)
vel = np.array([[1.0, 0.0, 0.0]])
acc = np.array([[2.0, 0.0, 0.0]])
vel_new = integrator.kick_step(vel, acc, 0.5)
assert vel_new[0, 0] == 2.0, '[TC18] kick_step updates velocity correctly FAILED'

# ---- TC19: PowerSpectrumEstimator k_bins monotonic ----
np.random.seed(42)
est = PowerSpectrumEstimator(16, 100.0)
delta = np.random.randn(16, 16, 16) * 0.1
k_bins, Pk, Nm = est.estimate(delta, n_bins=8)
assert np.all(np.diff(k_bins) > 0), '[TC19] PowerSpectrumEstimator k_bins monotonic FAILED'

# ---- TC20: Monte Carlo integral of x^2 y^2 z^2 over unit cube ----
np.random.seed(42)
def integrand(x):
    return np.prod(x**2)
val, err = monte_carlo_nd_integral(integrand, 3, np.array([0, 0, 0]), np.array([1, 1, 1]), 50000)
assert abs(val - 1.0/27.0) < 5e-4, '[TC20] Monte Carlo integral of x^2 y^2 z^2 over unit cube FAILED'

# ---- TC21: Press-Schechter mass function non-negative ----
M = np.logspace(11, 15, 50)
sigma = 2.0 * (M / 1e14)**(-0.2)
nM = press_schechter_mass_function(M, sigma, 2.7e11, delta_c=1.686)
assert np.all(nM >= 0.0), '[TC21] Press-Schechter mass function non-negative FAILED'

# ---- TC22: HaloFinder periodic distance works correctly ----
finder = HaloFinder(L=100.0)
p1 = np.array([1.0, 1.0, 1.0])
p2 = np.array([99.0, 1.0, 1.0])
d = finder._distance_periodic(p1, p2)
assert abs(d - 2.0) < 1e-10, '[TC22] HaloFinder periodic distance works correctly FAILED'

# ---- TC23: level_set_volume_analysis volumes decrease with level ----
np.random.seed(42)
delta = np.random.randn(16, 16, 16)
levels, volumes = level_set_volume_analysis(delta, 100.0, n_levels=10)
assert np.all(np.diff(volumes) <= 0.0), '[TC23] level_set_volume_analysis volumes decrease with level FAILED'

# ---- TC24: sample_sphere_positive_distance returns unit vectors ----
np.random.seed(42)
dirs = sample_sphere_positive_distance(100)
norms = np.linalg.norm(dirs, axis=1)
assert np.allclose(norms, 1.0), '[TC24] sample_sphere_positive_distance returns unit vectors FAILED'

# ---- TC25: halo_mass_function_from_groups empty input returns empty ----
logM, dn, err = halo_mass_function_from_groups(np.array([]), 100.0**3, n_bins=8)
assert len(logM) == 0, '[TC25] halo_mass_function_from_groups empty input returns empty FAILED'

# ---- TC26: alnorm(0.0) equals 0.5 ----
assert abs(alnorm(0.0) - 0.5) < 1e-10, '[TC26] alnorm(0.0) equals 0.5 FAILED'

# ---- TC27: alnorm(1.96) approximately 0.975 ----
assert abs(alnorm(1.96) - 0.975002) < 1e-4, '[TC27] alnorm(1.96) approximately 0.975 FAILED'

# ---- TC28: sample_discrete_cdf range check ----
np.random.seed(42)
pmf = np.array([1, 2, 3, 4, 5, 6]) / 21.0
samples = sample_discrete_cdf(1000, pmf)
assert np.all((samples >= 0) & (samples < len(pmf))), '[TC28] sample_discrete_cdf range check FAILED'

# ---- TC29: tophat_window(0) equals 1 ----
assert abs(tophat_window(np.array([0.0]))[0] - 1.0) < 1e-10, '[TC29] tophat_window(0) equals 1 FAILED'

# ---- TC30: casino_random_walk trajectory length correct ----
np.random.seed(42)
traj, w, l = casino_random_walk(1.0, 100)
assert len(traj) == 101 and w + l == 100, '[TC30] casino_random_walk trajectory length correct FAILED'

# ---- TC31: DenseLU solve accuracy ----
A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
lu = DenseLU(A)
b = np.array([1.0, 2.0, 3.0])
x = lu.solve(b)
resid = np.linalg.norm(A @ x - b)
assert resid < 1e-12, '[TC31] DenseLU solve accuracy FAILED'

# ---- TC32: SparseCRS matvec matches dense multiplication ----
A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
crs = SparseCRS.from_dense(A)
x = np.array([1.0, 2.0, 3.0])
y_crs = crs.matvec(x)
y_dense = A @ x
assert np.allclose(y_crs, y_dense), '[TC32] SparseCRS matvec matches dense multiplication FAILED'

# ---- TC33: solve_tridiagonal accuracy ----
n = 100
a = np.ones(n)
b = -2.0 * np.ones(n)
c = np.ones(n)
d = np.ones(n)
d[0] = d[-1] = 0.0
b[0] = b[-1] = 1.0
a[0] = c[-1] = 0.0
x_tri = solve_tridiagonal(a, b, c, d)
resid = 0.0
for i in range(n):
    ax_i = b[i] * x_tri[i]
    if i > 0:
        ax_i += a[i] * x_tri[i-1]
    if i < n-1:
        ax_i += c[i] * x_tri[i+1]
    resid = max(resid, abs(ax_i - d[i]))
assert resid < 1e-12, '[TC33] solve_tridiagonal accuracy FAILED'

# ---- TC34: utils ch_is_digit and ch_to_low correct ----
from utils import ch_is_digit, ch_to_low, filename_ext_get
assert ch_is_digit('5') == True, '[TC34] utils ch_is_digit and ch_to_low correct FAILED'
assert ch_is_digit('a') == False, '[TC34] utils ch_is_digit and ch_to_low correct FAILED'
assert ch_to_low('A') == 'a', '[TC34] utils ch_is_digit and ch_to_low correct FAILED'

# ---- TC35: filename_ext_get correct ----
assert filename_ext_get('data.txt') == 'txt', '[TC35] filename_ext_get correct FAILED'
assert filename_ext_get('noext') == '', '[TC35] filename_ext_get correct FAILED'

# ---- TC36: integrated PM solver and N-body evolution completes ----
np.random.seed(42)
cosmo = Cosmology()
N = 8
L = 100.0
solver = PMSolver(N, L, G=cosmo.G)
n_part = N**3
pos = np.random.rand(n_part, 3) * L
vel = np.random.randn(n_part, 3) * 1e-3
mass = np.ones(n_part) * 1e10
rho_mean = n_part * 1e10 / (L**3)
def get_acc(p):
    return solver.compute_gravity(p, mass, rho_mean, a_scale=1.0)
integrator = NBodyIntegrator(cosmo, use_adaptive_step=False)
t_arr, pos_arr, vel_arr, acc_arr = integrator.evolve(pos, vel, (0.0, 0.1), L, get_acc, n_steps=5)
assert len(t_arr) == 6 and pos_arr.shape[0] == 6, '[TC36] integrated PM solver and N-body evolution completes FAILED'
