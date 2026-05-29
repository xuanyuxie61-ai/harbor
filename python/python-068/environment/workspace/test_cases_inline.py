from epidemic_distributions import incomplete_gamma
from spatial_geometry import minimum_bounding_circle

# ---- TC01: safe_population_density clamps negatives and handles NaN/Inf ----
u_test = np.array([-5.0, 50.0, np.nan, np.inf])
result = safe_population_density(u_test, u_min=0.0, u_max=100.0)
assert result[0] >= 0.0, '[TC01] safe_population_density clamps negatives and handles NaN/Inf FAILED'
assert result[1] == 50.0, '[TC01] safe_population_density clamps negatives and handles NaN/Inf FAILED'
assert np.isfinite(result[2]), '[TC01] safe_population_density clamps negatives and handles NaN/Inf FAILED'
assert np.isfinite(result[3]), '[TC01] safe_population_density clamps negatives and handles NaN/Inf FAILED'

# ---- TC02: critical_threshold_check below above and near threshold ----
assert critical_threshold_check(0.5, threshold=1.0) == -1, '[TC02] critical_threshold_check below above and near threshold FAILED'
assert critical_threshold_check(2.0, threshold=1.0) == 1, '[TC02] critical_threshold_check below above and near threshold FAILED'
assert critical_threshold_check(1.0, threshold=1.0) == 0, '[TC02] critical_threshold_check below above and near threshold FAILED'

# ---- TC03: thin_index_2d correct size and valid bounds ----
idx = thin_index_2d(8, 8, factor=2)
assert len(idx) == 16, '[TC03] thin_index_2d correct size and valid bounds FAILED'
assert np.all(idx >= 0) and np.all(idx < 64), '[TC03] thin_index_2d correct size and valid bounds FAILED'

# ---- TC04: gamma_cdf at zero returns zero ----
assert gamma_cdf(0.0, shape=2.0, scale=2.5) == 0.0, '[TC04] gamma_cdf at zero returns zero FAILED'

# ---- TC05: incomplete_gamma invalid input returns error flag ----
val, err = incomplete_gamma(-1.0, 1.0)
assert err == 1 and val == 0.0, '[TC05] incomplete_gamma invalid input returns error flag FAILED'

# ---- TC06: cumulative_generation_interval monotonic and within bounds ----
t_vals = np.linspace(0, 10, 11)
cdf_vals = cumulative_generation_interval(t_vals, mean=5.2, std=1.7)
assert np.all(np.diff(cdf_vals) >= -1e-12), '[TC06] cumulative_generation_interval monotonic and within bounds FAILED'
assert cdf_vals[0] >= 0.0 and cdf_vals[-1] <= 1.0 + 1e-12, '[TC06] cumulative_generation_interval monotonic and within bounds FAILED'

# ---- TC07: infectious_period_cdf negative input and exact value ----
assert infectious_period_cdf(-1.0, gamma_rate=0.2) == 0.0, '[TC07] infectious_period_cdf negative input and exact value FAILED'
assert abs(infectious_period_cdf(5.0, gamma_rate=0.2) - (1.0 - np.exp(-1.0))) < 1e-10, '[TC07] infectious_period_cdf negative input and exact value FAILED'

# ---- TC08: bernstein_basis sums to one ----
from habitat_surface import bernstein_basis
b = bernstein_basis(0.5)
assert abs(np.sum(b) - 1.0) < 1e-10, '[TC08] bernstein_basis sums to one FAILED'

# ---- TC09: bezier_patch_evaluate corner equals control point ----
from habitat_surface import bezier_patch_evaluate
cp = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)
assert abs(bezier_patch_evaluate(cp, 0.0, 0.0) - 1.0) < 1e-10, '[TC09] bezier_patch_evaluate corner equals control point FAILED'
assert abs(bezier_patch_evaluate(cp, 1.0, 1.0) - 16.0) < 1e-10, '[TC09] bezier_patch_evaluate corner equals control point FAILED'

# ---- TC10: create_habitat_carrying_capacity within declared range ----
K_map = create_habitat_carrying_capacity(nx=16, ny=16, K_base=50.0, K_peak=150.0)
assert K_map.shape == (16, 16), '[TC10] create_habitat_carrying_capacity within declared range FAILED'
assert np.all(K_map >= 50.0) and np.all(K_map <= 150.0), '[TC10] create_habitat_carrying_capacity within declared range FAILED'

# ---- TC11: hexagon_stroud_rule4 weights sum to exact area ----
x_h, y_h, w_h = hexagon_stroud_rule4()
area = 3.0 * np.sqrt(3.0) / 2.0
assert abs(np.sum(w_h) - area) < 1e-10, '[TC11] hexagon_stroud_rule4 weights sum to exact area FAILED'

# ---- TC12: hexagon_monomial_integral odd powers vanish by symmetry ----
from hexagon_quadrature import hexagon_monomial_integral
assert hexagon_monomial_integral(1, 0) == 0.0, '[TC12] hexagon_monomial_integral odd powers vanish by symmetry FAILED'
assert hexagon_monomial_integral(0, 3) == 0.0, '[TC12] hexagon_monomial_integral odd powers vanish by symmetry FAILED'
assert hexagon_monomial_integral(3, 2) == 0.0, '[TC12] hexagon_monomial_integral odd powers vanish by symmetry FAILED'

# ---- TC13: integrate_trait_space returns positive finite value ----
trait_val = integrate_trait_space(lambda t: np.exp(-(t[0]**2 + t[1]**2)), [(0.0, 1.0), (0.0, 2.0)], [4, 4])
assert np.isfinite(trait_val) and trait_val > 0.0, '[TC13] integrate_trait_space returns positive finite value FAILED'

# ---- TC14: equilibrium_analysis disease-free stable when R0 below one ----
params_low = {'beta11': 0.01, 'gamma1': 0.1, 'mu1': 0.02, 'r_mean': 1.0, 'K_mean': 10.0}
eq_low = equilibrium_analysis(params_low)
assert eq_low['disease_free_stable'] == True, '[TC14] equilibrium_analysis disease-free stable when R0 below one FAILED'
params_high = {'beta11': 0.3, 'gamma1': 0.1, 'mu1': 0.02, 'r_mean': 1.0, 'K_mean': 10.0}
eq_high = equilibrium_analysis(params_high)
assert eq_high['disease_free_stable'] == False, '[TC14] equilibrium_analysis disease-free stable when R0 below one FAILED'

# ---- TC15: selkov_infection_rate is non-negative ----
from reaction_kinetics import selkov_infection_rate
rate = selkov_infection_rate(2.0, 3.0)
assert rate >= 0.0, '[TC15] selkov_infection_rate is non-negative FAILED'

# ---- TC16: dense_to_triplet correctly extracts non-zero entries ----
A = np.array([[1.0, 0.0], [2.0, 3.0]])
rows, cols, vals = dense_to_triplet(A)
assert len(rows) == 3, '[TC16] dense_to_triplet correctly extracts non-zero entries FAILED'
assert set(zip(rows.tolist(), cols.tolist())) == {(0, 0), (1, 0), (1, 1)}, '[TC16] dense_to_triplet correctly extracts non-zero entries FAILED'

# ---- TC17: build_coupled_jacobian_sparsity produces expected shape ----
J = build_coupled_jacobian_sparsity(4, 4, stencil='5point')
assert J.shape == (96, 96), '[TC17] build_coupled_jacobian_sparsity produces expected shape FAILED'
assert J.nnz > 0, '[TC17] build_coupled_jacobian_sparsity produces expected shape FAILED'

# ---- TC18: minimum_bounding_circle empty set returns zero ----
c, r = minimum_bounding_circle(np.zeros((0, 2)))
assert np.allclose(c, 0.0) and r == 0.0, '[TC18] minimum_bounding_circle empty set returns zero FAILED'

# ---- TC19: minimum_bounding_circle single point has zero radius ----
pts = np.array([[2.0, 3.0]])
c, r = minimum_bounding_circle(pts)
assert np.allclose(c, [2.0, 3.0]) and r == 0.0, '[TC19] minimum_bounding_circle single point has zero radius FAILED'

# ---- TC20: burgers_exact_solution_2d produces finite values ----
x = np.linspace(0, 2.0 * np.pi, 16)
y = np.linspace(0, 2.0 * np.pi, 16)
bfield = burgers_exact_solution_2d(x, y, 0.5, nu=0.01)
assert np.all(np.isfinite(bfield)), '[TC20] burgers_exact_solution_2d produces finite values FAILED'

# ---- TC21: manufactured_population_field is non-negative ----
x = np.linspace(0, 2.0 * np.pi, 16)
y = np.linspace(0, 2.0 * np.pi, 16)
for fid in range(6):
    mfield = manufactured_population_field(x, y, 0.0, fid)
    assert np.all(mfield >= -1e-10), '[TC21] manufactured_population_field is non-negative FAILED'

# ---- TC22: ETDRK4Solver preserves zero field ----
solver = ETDRK4Solver(nx=8, ny=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, D=[0.01] * 6, dt=0.01, n_fields=6)
u0 = np.zeros((6, 8, 8))
u1 = solver.step(u0, lambda u: np.zeros_like(u))
assert np.allclose(u1, 0.0), '[TC22] ETDRK4Solver preserves zero field FAILED'

# ---- TC23: EcoEpidemicPDE total populations integrate correctly ----
model_t = EcoEpidemicPDE(nx=8, ny=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
u_t = np.ones((6, 8, 8))
totals = model_t.compute_total_populations(u_t)
dx = model_t.x[1] - model_t.x[0]
dy = model_t.y[1] - model_t.y[0]
expected = 8 * 8 * dx * dy
assert abs(totals['S1'] - expected) < 1e-10, '[TC23] EcoEpidemicPDE total populations integrate correctly FAILED'
assert abs(totals['N1'] - 3.0 * expected) < 1e-10, '[TC23] EcoEpidemicPDE total populations integrate correctly FAILED'

# ---- TC24: adaptive_midpoint_solve exact for constant forcing ODE ----
def const_rhs(t, y):
    return np.array([2.0])
res = adaptive_midpoint_solve(const_rhs, np.array([0.0]), t_span=(0.0, 1.0), dt_init=0.2)
assert abs(res['y'][-1, 0] - 2.0) < 1e-8, '[TC24] adaptive_midpoint_solve exact for constant forcing ODE FAILED'

# ---- TC25: mean_field_eco_epi_ode zero state gives zero derivative ----
params_z = EcoEpidemicPDE.default_params()
dy_z = mean_field_eco_epi_ode(0.0, np.zeros(6), params_z)
assert np.allclose(dy_z, 0.0), '[TC25] mean_field_eco_epi_ode zero state gives zero derivative FAILED'

# ---- TC26: compute_infected_patch_geometry empty field has zero patches ----
x_g = np.linspace(0, 2.0 * np.pi, 16)
y_g = np.linspace(0, 2.0 * np.pi, 16)
I_zero = np.zeros((16, 16))
geom_z = compute_infected_patch_geometry(I_zero, x_g, y_g, threshold=0.5)
assert geom_z['num_patches'] == 0, '[TC26] compute_infected_patch_geometry empty field has zero patches FAILED'
assert geom_z['total_infected_area'] == 0.0, '[TC26] compute_infected_patch_geometry empty field has zero patches FAILED'

# ---- TC27: refine_mesh_multiple doubles triangle count correctly ----
nodes_m, tri_m = create_uniform_triangular_mesh(0, 1, 0, 1, 3, 3)
assert len(tri_m) == 8, '[TC27] refine_mesh_multiple doubles triangle count correctly FAILED'
nodes_r, tri_r = refine_mesh_multiple(nodes_m, tri_m, n_refinements=1)
assert len(tri_r) == 32, '[TC27] refine_mesh_multiple doubles triangle count correctly FAILED'

# ---- TC28: compute_reaction_terms zero state returns zero ----
from reaction_kinetics import compute_reaction_terms
K_f = np.ones((4, 4))
r_f = np.ones((4, 4))
state_z = np.zeros((6, 4, 4))
params_c = EcoEpidemicPDE.default_params()
rhs_z = compute_reaction_terms(state_z, K_f, r_f, params_c)
assert np.allclose(rhs_z, 0.0), '[TC28] compute_reaction_terms zero state returns zero FAILED'

# ---- TC29: implicit_midpoint_step exact for linear decay ODE ----
from adaptive_midpoint import implicit_midpoint_step
y_n = implicit_midpoint_step(np.array([1.0]), 0.0, 0.1, lambda t, y: -y)
expected_n = 1.0 * (1.0 - 0.05) / (1.0 + 0.05)
assert abs(y_n[0] - expected_n) < 1e-10, '[TC29] implicit_midpoint_step exact for linear decay ODE FAILED'
