# ---- TC01: linear_cooling - midpoint check ----
t = np.array([3600.0])
T = linear_cooling(t, 350.0, 300.0, 7200.0)
assert abs(T[0] - 325.0) < 1e-6, '[TC01] linear_cooling midpoint FAILED'

# ---- TC02: natural_cooling - initial value equals T0 ----
from cooling_profile import natural_cooling
T = natural_cooling(np.array([0.0]), 350.0, 300.0, 1000.0)
assert abs(T[0] - 350.0) < 1e-6, '[TC02] natural_cooling t=0 FAILED'

# ---- TC03: solubility_vanthoff - positive output ----
c_sat = solubility_vanthoff(np.array([320.0]), 25000.0, 70.0)
assert np.all(c_sat > 0), '[TC03] solubility_vanthoff positive FAILED'

# ---- TC04: supersaturation - zero when c equals c_sat ----
T = np.array([320.0])
c_sat_val = solubility_vanthoff(T, 25000.0, 70.0)
sigma = supersaturation(c_sat_val, T, 25000.0, 70.0)
assert abs(sigma[0]) < 1e-6, '[TC04] supersaturation zero equilibrium FAILED'

# ---- TC05: optimal_cooling_polynomial - monotonic decreasing ----
t = np.linspace(0, 7200, 100)
T_opt = optimal_cooling_polynomial(t, 350.0, 300.0, 7200.0, order=3)
assert np.all(np.diff(T_opt) <= 0), '[TC05] optimal_cooling monotonic FAILED'

# ---- TC06: lambert_w - W(e) = 1.0 ----
w = lambert_w(np.array([np.e]), branch=0)
assert abs(w[0] - 1.0) < 1e-10, '[TC06] lambert_w W(e)=1 FAILED'

# ---- TC07: lambert_w - W(10) is finite and positive ----
w = lambert_w(np.array([10.0]), branch=0)
assert np.isfinite(w[0]) and w[0] > 1.5, '[TC07] lambert_w W(10) FAILED'

# ---- TC08: fresnel_integrals - C(0)=0, S(0)=0 ----
C, S = fresnel_integrals(np.array([0.0]))
assert abs(C[0]) < 1e-12 and abs(S[0]) < 1e-12, '[TC08] fresnel C(0)=S(0)=0 FAILED'

# ---- TC09: fresnel_integrals - large x asymptote near 0.5 ----
C, S = fresnel_integrals(np.array([10.0]))
assert abs(C[0] - 0.5) < 0.06 and abs(S[0] - 0.5) < 0.06, '[TC09] fresnel asymptote FAILED'

# ---- TC10: fraunhofer_diffraction - peak near theta=0 ----
theta = np.linspace(0.001, 0.1, 100)
I = fraunhofer_diffraction_particle_size(25e-6, 632.8e-9, theta)
assert np.argmax(I) == 0, '[TC10] fraunhofer peak at theta~0 FAILED'

# ---- TC11: chen_attractor_rhs - at origin gives zero derivative ----
from chaotic_mixing import chen_attractor_rhs
dstate = chen_attractor_rhs(0.0, np.array([0.0, 0.0, 0.0]))
assert np.allclose(dstate, [0.0, 0.0, 0.0], atol=1e-12), '[TC11] chen attractor origin FAILED'

# ---- TC12: classical_nucleation_rate - increases with supersaturation ----
B1 = classical_nucleation_rate(np.array([0.3]), 320.0, A_prefactor=1e15)
B2 = classical_nucleation_rate(np.array([0.5]), 320.0, A_prefactor=1e15)
assert float(B2) > float(B1), '[TC12] CNT monotonic in sigma FAILED'

# ---- TC13: secondary_nucleation_rate - zero at zero sigma ----
B = secondary_nucleation_rate(np.array([0.0]), 50.0)
assert float(B) == 0.0, '[TC13] secondary nucleation zero FAILED'

# ---- TC14: critical_nucleus_radius - returns positive value ----
r_star = critical_nucleus_radius(0.5, 320.0)
assert float(r_star) > 0, '[TC14] critical radius positive FAILED'

# ---- TC15: power_law_growth - zero at zero sigma ----
G = power_law_growth(0.0, 320.0, 1e-4, 40000.0, 1.5)
assert float(G) == 0.0, '[TC15] power law growth zero FAILED'

# ---- TC16: size_dependent_growth - reduces to power law when alpha=0, beta=0 ----
G_sd = size_dependent_growth(50e-6, 0.5, 320.0, 1e-4, 40000.0, 1.5, 0.0, 0.0)
G_pl = power_law_growth(0.5, 320.0, 1e-4, 40000.0, 1.5)
assert abs(float(G_sd) - float(G_pl)) < 1e-15, '[TC16] size_dependent reduces to power law FAILED'

# ---- TC17: two_step_growth - positive output for valid inputs ----
G = two_step_growth(0.5, 320.0, 1e-5, 1e-3, 35000.0, 2.0)
assert float(G) > 0, '[TC17] two_step positive FAILED'

# ---- TC18: bcf_spiral_growth - positive output for valid inputs ----
G = bcf_spiral_growth(0.5, 320.0, 1e-4, 0.2, 45000.0)
assert float(G) > 0, '[TC18] BCF positive FAILED'

# ---- TC19: newton_cotes_open_weights - returns correct number of nodes/weights ----
from population_balance import newton_cotes_open_weights
x, w = newton_cotes_open_weights(5, -1.0, 1.0)
assert len(x) == 5 and len(w) == 5, '[TC19] NCO weights shape FAILED'
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC19] NCO nodes in interval FAILED'

# ---- TC20: quadrature_integrate - runs and returns float without exception ----
I = quadrature_integrate(lambda x: np.ones_like(x), 0.0, 1.0, n=32)
assert isinstance(I, float), '[TC20] quadrature returns float FAILED'

# ---- TC21: kmeans_1d - two cluster separation ----
rng = np.random.default_rng(42)
data = np.concatenate([rng.normal(0, 0.5, 100), rng.normal(5, 0.5, 100)])
centers, labels, inertia = kmeans_1d(data, 2, rng=rng)
assert len(centers) == 2, '[TC21] kmeans two clusters count FAILED'
assert abs(centers[0]) < 1.0, '[TC21] kmeans center near 0 FAILED'

# ---- TC22: dirichlet_sample_uniform_simplex - reproducibility with fixed seed ----
rng1 = np.random.default_rng(42)
samples1 = dirichlet_sample_uniform_simplex(100, 2, rng1)
rng2 = np.random.default_rng(42)
samples2 = dirichlet_sample_uniform_simplex(100, 2, rng2)
assert np.allclose(samples1, samples2), '[TC22] dirichlet reproducibility FAILED'

# ---- TC23: wedge01_monomial_integral - known value for (0,0,0) = volume ----
I = wedge01_monomial_integral(np.array([0, 0, 0]))
# wedge W: x>=0,y>=0,x+y<=1,-1<=z<=1, volume = 1/2 * 2 = 1
assert abs(I - 1.0) < 1e-12, '[TC23] wedge integral (0,0,0) FAILED'

# ---- TC24: tetrahedron01_monomial_integral - known value for (0,0,0) ----
I = tetrahedron01_monomial_integral(np.array([0, 0, 0]))
assert abs(I - 1.0/6.0) < 1e-12, '[TC24] tetrahedron integral (0,0,0) FAILED'

# ---- TC25: sparse_grid_integrate - 2D exp(-x^2-y^2) accuracy ----
def f_2d(x):
    return np.exp(-x[:, 0]**2 - x[:, 1]**2)
result = sparse_grid_integrate(f_2d, 2, level_max=3)
from scipy.special import erf
exact = (np.sqrt(np.pi) * erf(1.0)) ** 2
assert abs(result - exact) < 0.01, '[TC25] sparse grid 2D FAILED'

# ---- TC26: log_prior_lognormal - valid positive params give finite value ----
lp = log_prior_lognormal(np.array([1e-5, 40000.0, 5e7]),
                          np.array([-9.0, np.log(42000.0), np.log(5e7)]),
                          np.array([0.5, 0.1, 0.5]))
assert np.isfinite(lp), '[TC26] log prior finite FAILED'

# ---- TC27: log_prior_lognormal - negative parameter gives -inf ----
lp = log_prior_lognormal(np.array([-1e-5, 40000.0, 5e7]),
                          np.array([-9.0, np.log(42000.0), np.log(5e7)]),
                          np.array([0.5, 0.1, 0.5]))
assert lp == -np.inf, '[TC27] log prior negative FAILED'

# ---- TC28: gelman_rubin_diagnostic - identical chains give R_hat near 1 ----
from mcmc_inference import gelman_rubin_diagnostic
np.random.seed(42)
chain = np.random.randn(1, 500, 3)
chains = np.repeat(chain, 3, axis=0)
R_hat = gelman_rubin_diagnostic(chains)
# For identical chains, R_hat = sqrt((n-1)/n) ≈ 0.999
assert np.all(np.abs(R_hat - 1.0) < 0.002), '[TC28] Gelman-Rubin identical FAILED'

# ---- TC29: format_moment_vector - output contains expected value string ----
mu = np.array([1.0, 2.0, 3.0])
s = format_moment_vector(mu, ['a', 'b', 'c'])
assert '1.000000e+00' in s, '[TC29] format moment vector FAILED'

# ---- TC30: index_set_to_string - correct formatting ----
s = index_set_to_string([0, 3, 1], name="Test")
assert s == "Test = {0, 1, 3}", '[TC30] index set to string FAILED'

# ---- TC31: csd_statistical_moments - Gaussian distribution mean recovery ----
L = np.linspace(1e-6, 200e-6, 1000)
mu_target = 100e-6
sigma_target = 20e-6
f = np.exp(-0.5 * ((L - mu_target) / sigma_target) ** 2)
stats = csd_statistical_moments(L, f)
assert abs(stats['mean'] - mu_target) < 1e-6, '[TC31] CSD moments mean FAILED'

# ---- TC32: lcg_park_miller - deterministic output ----
from nucleation_model import lcg_park_miller
seed1, u1 = lcg_park_miller(42)
seed2, u2 = lcg_park_miller(42)
assert seed1 == seed2 and abs(u1 - u2) < 1e-15, '[TC32] LCG deterministic FAILED'

# ---- TC33: growth_rate_dispersion - reproducibility with fixed seed ----
rng1 = np.random.default_rng(42)
G1 = growth_rate_dispersion(1e-8, cv=0.1, n_samples=100, rng=rng1)
rng2 = np.random.default_rng(42)
G2 = growth_rate_dispersion(1e-8, cv=0.1, n_samples=100, rng=rng2)
assert np.allclose(G1, G2), '[TC33] GRD reproducibility FAILED'

# ---- TC34: parse_solution_vector - basic parsing ----
from data_io import parse_solution_vector
text = "x0 = 1.5\nx1 = 3.14\nx2 = 2.718"
vals = parse_solution_vector(text)
assert len(vals) == 3 and abs(vals[0] - 1.5) < 1e-10, '[TC34] parse solution FAILED'

# ---- TC35: monomial_value - basic evaluation ----
from simplex_sampling import monomial_value
x = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
v = monomial_value(2, 3, np.array([2, 1]), x)
assert abs(v[0] - 2.0) < 1e-12, '[TC35] monomial value FAILED'

# ---- TC36: tetrahedron01_volume - known value 1/6 ----
from simplex_sampling import tetrahedron01_volume
assert abs(tetrahedron01_volume() - 1.0/6.0) < 1e-12, '[TC36] tetrahedron volume FAILED'

# ---- TC37: sawtooth_cooling - periodic average approximates base average ----
t = np.linspace(0, 6000, 1001)
T_base = linear_cooling(t, 350.0, 300.0, 6000.0)
T_saw = sawtooth_cooling(t, T_base, delta_T=5.0, period=100.0)
assert abs(np.mean(T_saw) - np.mean(T_base)) < 0.1, '[TC37] sawtooth mean FAILED'

# ---- TC38: total_nucleation_rate - positive output ----
B = total_nucleation_rate(0.5, 320.0, 50.0)
assert float(B) > 0, '[TC38] total nucleation positive FAILED'

# ---- TC39: write_simulation_results and read_simulation_results roundtrip ----
tmpf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_roundtrip.json')
data = {'a': np.array([1.0, 2.0, 3.0]), 'b': 42.0}
meta = {'test': 'roundtrip'}
write_simulation_results(tmpf, data, meta)
meta_r, data_r = read_simulation_results(tmpf)
assert abs(data_r['a'][1] - 2.0) < 1e-12, '[TC39] write/read roundtrip FAILED'
os.remove(tmpf)

# ---- TC40: r8vec2_write produces readable output with correct header ----
tmpf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_r8vec2.txt')
x = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 4.0, 5.0])
r8vec2_write(tmpf, x, y)
with open(tmpf, 'r') as f:
    content = f.read()
assert '# Paired vectors, n = 3' in content, '[TC40] r8vec2 write FAILED'
os.remove(tmpf)

# ---- TC41: i4vec_transpose_print - runs without exception ----
i4vec_transpose_print(np.array([1, 2, 3, 4, 5]), title="Test")
assert True, '[TC41] i4vec print no crash FAILED'

# ---- TC42: lambert_w - branch -1 on valid input returns value < -1 ----
w = lambert_w(np.array([-0.3]), branch=-1)
assert np.isfinite(w[0]) and w[0] < -1.0, '[TC42] lambert_w branch -1 FAILED'

# ---- TC43: dirichlet sampling - all samples in simplex ----
rng = np.random.default_rng(100)
samples = dirichlet_sample_uniform_simplex(500, 3, rng)
assert np.all(samples >= 0), '[TC43] dirichlet non-negative FAILED'
assert np.all(np.sum(samples, axis=1) <= 1.0 + 1e-12), '[TC43] dirichlet in simplex FAILED'

# ---- TC44: stochastic_nucleation_events - deterministic with same seed ----
n1, s1 = stochastic_nucleation_events(0.1, 320.0, 1.0, 0.001, seed=12345)
n2, s2 = stochastic_nucleation_events(0.1, 320.0, 1.0, 0.001, seed=12345)
assert n1 == n2, '[TC44] stochastic nucleation deterministic FAILED'

# ---- TC45: analytical_size_dependent_growth_law - beta=-1 produces increasing L ----
from nucleation_model import analytical_size_dependent_growth_law
t = np.linspace(0, 3600, 100)
L_analytical = analytical_size_dependent_growth_law(t, 0.01, -1.0, 1e-7, 0.5)
assert L_analytical[-1] > L_analytical[0], '[TC45] analytical growth increasing FAILED'
assert np.all(np.isfinite(L_analytical)), '[TC45] analytical growth finite FAILED'

# ---- TC46: population balance solver short integration ----
params = {
    'rho_c': 1560.0, 'kv': 0.5236, 'L0': 1e-9, 'k_g0': 5e-5,
    'E_g': 42000.0, 'g_exp': 1.2, 'alpha': 50.0, 'beta': -0.2,
    'A_prefactor': 1e18, 'kb_sec': 5e7, 'b_exp': 1.8, 'j_exp': 0.8,
    'H_diss': 25000.0, 'S_diss': 70.0, 'c0': 0.55, 'T0': 350.0,
    'V': 0.001, 'gamma0': 0.025
}
solver = PopulationBalanceSolver(params)
T_func = lambda tau: float(linear_cooling(tau, 350.0, 300.0, 7200.0))
sol = solver.solve((0.0, 1800.0), T_func, method='RK45')
assert sol.success, '[TC46] PBE solver success FAILED'
mu, c = solver.get_moments_at_time(sol, 1800.0)
assert len(mu) == 6, '[TC46] PBE moments count FAILED'
assert c > 0, '[TC46] PBE concentration positive FAILED'

# ---- TC47: csd_parameters from PBE solver ----
csd_params = solver.get_csd_parameters(sol, 1800.0)
assert csd_params['N'] > 0, '[TC47] CSD params N positive FAILED'
assert csd_params['L_mean'] > 0, '[TC47] CSD params L_mean positive FAILED'

# ---- TC48: generate_chaotic_mixing_trajectory - returns solution with expected fields ----
sol_chaos = generate_chaotic_mixing_trajectory((0.0, 100.0), y0=[-0.1, 0.5, -0.6])
assert sol_chaos.success, '[TC48] chaos trajectory success FAILED'

# ---- TC49: map_chen_to_supersaturation_fluctuation - returns bounded values ----
t_check = np.linspace(0, 100, 200)
sigma_loc = map_chen_to_supersaturation_fluctuation(t_check, sol_chaos, 0.5)
assert np.all(sigma_loc >= 0) and np.all(sigma_loc <= 5.0), '[TC49] chaos fluctuation bounded FAILED'

# ---- TC50: composition_space_integral - returns finite non-negative result ----
def const_func(x):
    return np.ones(x.shape[0])
integral, err = composition_space_integral(const_func, dim=2, n_samples=5000)
assert integral > 0 and np.isfinite(integral), '[TC50] composition integral positive FAILED'
assert err >= 0, '[TC50] composition integral error non-negative FAILED'

# ---- TC51: uncertainty_quantification_crystallization - returns finite output ----
def simple_model(params):
    return params[0] + 2 * params[1] + 3 * params[2]
param_dists = [
    {'mean': 0.0, 'std': 1.0},
    {'mean': 0.0, 'std': 1.0},
    {'mean': 0.0, 'std': 1.0},
]
mean_val, variance, std_val = uncertainty_quantification_crystallization(
    simple_model, param_dists, level_max=2
)
assert np.isfinite(mean_val), '[TC51] UQ mean finite FAILED'
assert variance >= 0, '[TC51] UQ variance non-negative FAILED'

# ---- TC52: discretize_csd_kmeans - returns correct number of classes ----
L_grid = np.linspace(1e-6, 200e-6, 500)
mu_ln = np.log(50e-6)
sigma_ln = 0.3
N = 1e12
f_vals = (N / (L_grid * sigma_ln * np.sqrt(2 * np.pi))) * \
         np.exp(-0.5 * ((np.log(L_grid) - mu_ln) / sigma_ln) ** 2)
f_vals = np.where(np.isnan(f_vals), 0.0, f_vals)
k = 6
class_sizes, class_counts, boundaries = discretize_csd_kmeans(L_grid, f_vals, k)
assert len(class_sizes) == k, '[TC52] discretize CSD class count FAILED'
assert len(boundaries) == k + 1, '[TC52] discretize CSD boundaries FAILED'

# ---- TC53: diffraction_inversion_feret - returns correct output shapes ----
theta = np.linspace(0.001, 0.05, 50)
wavelength = 632.8e-9
radius = 30e-6
I_clean = fraunhofer_diffraction_particle_size(radius, wavelength, theta)
L_bins, n_L = diffraction_inversion_feret(theta, I_clean, wavelength, L_min=1e-6, L_max=200e-6, n_bins=40)
assert len(L_bins) == 40 and len(n_L) == 40, '[TC53] diffraction inversion shape FAILED'

# ---- TC54: index_set_to_string - empty set formatting ----
s = index_set_to_string([], name="Empty")
assert "∅" in s, '[TC54] index set empty FAILED'

# ---- TC55: format_moment_vector - default names ----
mu = np.array([1.0, 2.0])
s = format_moment_vector(mu)
assert "μ_0" in s and "μ_1" in s, '[TC55] format moment default names FAILED'
