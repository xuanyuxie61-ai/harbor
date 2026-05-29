
# ---- TC01: safe_population_density clips negative and overflow values ----
np.random.seed(42)
result = safe_population_density(np.array([-1.0, 0.0, 5.0, 1e7]), u_min=0.0, u_max=1e6)
assert result[0] >= 0.0, '[TC01] safe_population_density negative clip FAILED'
assert result[1] >= 0.0, '[TC01] safe_population_density zero bound FAILED'
assert result[2] == 5.0, '[TC01] safe_population_density normal value FAILED'
assert result[3] <= 1e6, '[TC01] safe_population_density overflow clip FAILED'

# ---- TC02: safe_population_density handles NaN and Inf safely ----
np.random.seed(42)
result = safe_population_density(np.array([np.nan, np.inf, -np.inf, 5.0]), u_min=0.0, u_max=1e6)
assert not np.isnan(result[0]), '[TC02] NaN replacement FAILED'
assert np.isfinite(result[1]), '[TC02] Inf replacement FAILED'
assert np.isfinite(result[2]), '[TC02] -Inf replacement FAILED'
assert result[3] == 5.0, '[TC02] normal value preservation FAILED'

# ---- TC03: critical_threshold_check returns -1/0/1 correctly ----
np.random.seed(42)
assert critical_threshold_check(0.5, threshold=1.0) == -1, '[TC03] below threshold FAILED'
assert critical_threshold_check(1.0, threshold=1.0) == 0, '[TC03] at threshold FAILED'
assert critical_threshold_check(2.0, threshold=1.0) == 1, '[TC03] above threshold FAILED'

# ---- TC04: thin_index_2d produces correct count and index range ----
np.random.seed(42)
result = thin_index_2d(64, 64, factor=4)
assert len(result) == 256, '[TC04] thin_index count FAILED'
assert result[0] == 0, '[TC04] first index FAILED'
assert result[-1] == 3900, '[TC04] last index FAILED'
assert result.max() < 64 * 64, '[TC04] index out of bounds FAILED'

# ---- TC05: gamma_cdf at t=0 returns 0 ----
np.random.seed(42)
result = gamma_cdf(0.0, shape=2.0, scale=2.5)
assert result == 0.0, '[TC05] gamma_cdf zero boundary FAILED'

# ---- TC06: gamma_cdf normal value accuracy ----
np.random.seed(42)
result = gamma_cdf(5.0, shape=2.0, scale=2.5)
assert abs(result - 0.5939941502901616) < 1e-12, '[TC06] gamma_cdf normal value FAILED'

# ---- TC07: cumulative_generation_interval is monotonically increasing ----
np.random.seed(42)
t = np.linspace(0, 20, 21)
result = cumulative_generation_interval(t, mean=5.2, std=1.7)
assert np.all(np.diff(result) >= 0), '[TC07] monotonicity FAILED'
assert result[-1] > 0.999, '[TC07] tail convergence FAILED'
assert result[0] == 0.0, '[TC07] zero start FAILED'

# ---- TC08: infectious_period_cdf basic correctness ----
np.random.seed(42)
result = infectious_period_cdf(5.0, gamma_rate=1.0 / 5.0)
assert abs(result - 0.6321205588285577) < 1e-12, '[TC08] infectious_period_cdf FAILED'
assert infectious_period_cdf(-1.0, gamma_rate=1.0 / 5.0) == 0.0, '[TC08] negative input FAILED'

# ---- TC09: create_habitat_carrying_capacity shape and range ----
np.random.seed(42)
result = create_habitat_carrying_capacity(16, 16, K_base=100.0, K_peak=200.0)
assert result.shape == (16, 16), '[TC09] shape FAILED'
assert result.min() == 100.0, '[TC09] min base FAILED'
assert result.max() <= 200.0, '[TC09] max peak FAILED'

# ---- TC10: create_growth_rate_map shape and range ----
np.random.seed(42)
result = create_growth_rate_map(16, 16, r_base=0.5, r_peak=1.5)
assert result.shape == (16, 16), '[TC10] shape FAILED'
assert result.min() == 0.5, '[TC10] min base FAILED'
assert result.max() <= 1.5, '[TC10] max peak FAILED'

# ---- TC11: hexagon_stroud_rule4 weights sum to unit hexagon area ----
np.random.seed(42)
x, y, w = hexagon_stroud_rule4()
expected_area = 3.0 * np.sqrt(3.0) / 2.0
assert abs(np.sum(w) - expected_area) < 1e-14, '[TC11] weight sum FAILED'

# ---- TC12: compute_hexagonal_patch_metrics output structure ----
np.random.seed(42)
model = EcoEpidemicPDE(nx=16, ny=16)
state = model.initialize_state(seed=42)
fields = {'S1': state[0], 'I1': state[1]}
hex_centers = [(np.pi, np.pi)]
patch = compute_hexagonal_patch_metrics(fields, model.x, model.y, hex_centers, hex_radius=1.0)
assert list(patch.keys()) == ['S1', 'I1'], '[TC12] keys FAILED'
assert len(patch['S1']) == 1, '[TC12] length FAILED'
assert patch['S1'][0] >= 0.0, '[TC12] non-negative FAILED'

# ---- TC13: adaptive_refinement_indicator on uniform field returns all False ----
np.random.seed(42)
model = EcoEpidemicPDE(nx=16, ny=16)
field_uniform = np.ones((16, 16))
indicator = adaptive_refinement_indicator(field_uniform, model.x, model.y, threshold=0.1)
assert indicator.dtype == bool, '[TC13] dtype FAILED'
assert np.sum(indicator) == 0, '[TC13] uniform field FAILED'

# ---- TC14: compute_infected_patch_geometry on non-empty field ----
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 32)
y = np.linspace(0, 2 * np.pi, 32)
field = np.zeros((32, 32))
field[10:20, 10:20] = 1.0
geom = compute_infected_patch_geometry(field, x, y, threshold=0.5)
assert geom['num_patches'] >= 1, '[TC14] num_patches FAILED'
assert geom['total_infected_area'] > 0.0, '[TC14] total area FAILED'

# ---- TC15: compute_infected_patch_geometry on zero field ----
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 16)
y = np.linspace(0, 2 * np.pi, 16)
field = np.zeros((16, 16))
geom = compute_infected_patch_geometry(field, x, y, threshold=0.5)
assert geom['num_patches'] == 0, '[TC15] num_patches FAILED'
assert geom['total_infected_area'] == 0.0, '[TC15] total area FAILED'

# ---- TC16: refine_mesh_multiple doubles triangles correctly ----
np.random.seed(42)
nodes, tri = create_uniform_triangular_mesh(0, 1, 0, 1, 3, 3)
nodes_r, tri_r = refine_mesh_multiple(nodes, tri, n_refinements=1)
assert len(nodes) == 9, '[TC16] initial nodes FAILED'
assert len(tri) == 8, '[TC16] initial triangles FAILED'
assert len(tri_r) == 32, '[TC16] refined triangles FAILED'
assert len(nodes_r) == 25, '[TC16] refined nodes FAILED'

# ---- TC17: integrate_trait_space 1D quadratic exactness ----
np.random.seed(42)
result = integrate_trait_space(lambda t: t[0] ** 2, trait_ranges=[(0.0, 1.0)], orders=[4])
assert abs(result - 1.0 / 3.0) < 1e-12, '[TC17] 1D quadratic integral FAILED'

# ---- TC18: manufactured_population_field periodic in t with period 2π ----
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 16)
y = np.linspace(0, 2 * np.pi, 16)
f0 = manufactured_population_field(x, y, 0.0, field_id=0)
f2pi = manufactured_population_field(x, y, 2 * np.pi, field_id=0)
assert np.allclose(f0, f2pi), '[TC18] periodicity FAILED'

# ---- TC19: integrate_gauss_hermite_2d of constant 1 equals π ----
np.random.seed(42)
result = integrate_gauss_hermite_2d(lambda x, y: 1.0, n=8, sigma_x=1.0, sigma_y=1.0)
assert abs(result - np.pi) < 1e-12, '[TC19] constant Gauss-Hermite FAILED'

# ---- TC20: dense_to_triplet extracts correct non-zero entries ----
np.random.seed(42)
A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
r, c, v = dense_to_triplet(A)
assert len(v) == 4, '[TC20] non-zero count FAILED'
assert sorted(v.tolist()) == [1.0, 2.0, 3.0, 4.0], '[TC20] values FAILED'

# ---- TC21: build_coupled_jacobian_sparsity dimensions and nnz ----
np.random.seed(42)
J = build_coupled_jacobian_sparsity(4, 4, stencil='5point')
assert J.shape == (96, 96), '[TC21] shape FAILED'
assert J.nnz == 864, '[TC21] nnz FAILED'

# ---- TC22: equilibrium_analysis R0 and disease-free stability ----
np.random.seed(42)
params_eq = {'beta11': 0.3, 'gamma1': 0.1, 'mu1': 0.02, 'r_mean': 1.0, 'K_mean': 100.0}
eq = equilibrium_analysis(params_eq)
assert abs(eq['R0'] - 250.0) < 1e-10, '[TC22] R0 FAILED'
assert eq['disease_free_stable'] == False, '[TC22] stability FAILED'
assert abs(eq['S_star'] - 0.4) < 1e-12, '[TC22] S_star FAILED'

# ---- TC23: manufactured_source_terms output shape ----
np.random.seed(42)
model = EcoEpidemicPDE(nx=16, ny=16)
sources = manufactured_source_terms(model.x, model.y, 0.0)
assert sources.shape == (6, 16, 16), '[TC23] source shape FAILED'
assert np.isfinite(sources).all(), '[TC23] non-finite sources FAILED'

# ---- TC24: ETDRK4Solver step preserves constant field with zero nonlinear ----
np.random.seed(42)
solver = ETDRK4Solver(nx=8, ny=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, D=[0.01, 0.01], dt=0.01, n_fields=2)
u0 = np.ones((2, 8, 8))
u1 = solver.step(u0, lambda u: np.zeros_like(u))
assert u1.shape == (2, 8, 8), '[TC24] output shape FAILED'
assert np.allclose(u1, 1.0, atol=1e-10), '[TC24] conservation FAILED'

# ---- TC25: EcoEpidemicPDE initialize_state reproducible with fixed seed ----
np.random.seed(42)
model = EcoEpidemicPDE(nx=16, ny=16)
state_a = model.initialize_state(seed=123)
state_b = model.initialize_state(seed=123)
assert np.allclose(state_a, state_b), '[TC25] reproducibility FAILED'
assert state_a.shape == (6, 16, 16), '[TC25] shape FAILED'

# ---- TC26: EcoEpidemicPDE compute_total_populations N1 conservation ----
np.random.seed(42)
model = EcoEpidemicPDE(nx=16, ny=16)
state = model.initialize_state(seed=42)
totals = model.compute_total_populations(state)
assert abs(totals['N1'] - (totals['S1'] + totals['I1'] + totals['R1'])) < 1e-10, '[TC26] N1 conservation FAILED'
assert abs(totals['N2'] - (totals['S2'] + totals['I2'] + totals['R2'])) < 1e-10, '[TC26] N2 conservation FAILED'

# ---- TC27: adaptive_midpoint_solve produces correct output shape ----
np.random.seed(42)
params_ode = EcoEpidemicPDE.default_params()
def ode_rhs(t, y):
    return mean_field_eco_epi_ode(t, y, params_ode)
y0 = np.array([50.0, 1.0, 0.0, 30.0, 0.5, 0.0], dtype=float)
res = adaptive_midpoint_solve(ode_rhs, y0, t_span=(0.0, 0.5), dt_init=0.1, abstol=1e-6, reltol=1e-4)
assert res['y'].shape[1] == 6, '[TC27] state dimension FAILED'
assert res['n_steps'] > 0, '[TC27] no steps taken FAILED'
assert res['t'][-1] >= 0.5 - 1e-6, '[TC27] final time FAILED'

# ---- TC28: mean_field_eco_epi_ode output is 6-dimensional ----
np.random.seed(42)
params_ode2 = EcoEpidemicPDE.default_params()
y = np.array([10.0, 1.0, 0.0, 5.0, 0.5, 0.0], dtype=float)
dydt = mean_field_eco_epi_ode(0.0, y, params_ode2)
assert dydt.shape == (6,), '[TC28] output dimension FAILED'
assert np.isfinite(dydt).all(), '[TC28] non-finite values FAILED'
