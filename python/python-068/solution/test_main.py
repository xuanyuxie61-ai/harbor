"""
main.py
Unified entry point for the Spatial Eco-Epidemiological Dynamics project.

This script runs a complete workflow:
  1. Initializes the coupled PDE-ODE eco-epidemiological model
  2. Simulates spatial dynamics using ETDRK4 spectral solver
  3. Simulates mean-field ODE dynamics using adaptive implicit midpoint
  4. Computes ecological metrics via hexagonal patch quadrature
  5. Analyzes spatial geometry of infection spread
  6. Verifies numerical accuracy against manufactured exact solutions
  7. Tests mesh refinement and multi-dimensional trait integration
  8. Computes epidemiological statistics and reproduction numbers

Domain: Ecological Modeling - Population Competition & Infectious Disease Transmission
"""

import numpy as np
import time

# Import all modules
from numerical_robustness import safe_population_density, critical_threshold_check, thin_index_2d
from epidemic_distributions import gamma_cdf, cumulative_generation_interval, infectious_period_cdf
from habitat_surface import create_habitat_carrying_capacity, create_growth_rate_map
from hexagon_quadrature import compute_hexagonal_patch_metrics, hexagon_stroud_rule4
from multi_dim_quadrature import integrate_trait_space
from spatial_geometry import compute_infected_patch_geometry
from mesh_refinement import create_uniform_triangular_mesh, refine_mesh_multiple, adaptive_refinement_indicator
from reaction_kinetics import equilibrium_analysis
from exact_solutions import manufactured_population_field, manufactured_source_terms, burgers_exact_solution_2d, integrate_gauss_hermite_2d
from sparse_matrix_utils import build_coupled_jacobian_sparsity, dense_to_triplet
from etdrk4_solver import ETDRK4Solver
from adaptive_midpoint import adaptive_midpoint_solve, mean_field_eco_epi_ode
from eco_epi_pde import EcoEpidemicPDE


def main():
    print("=" * 70)
    print("SPATIAL ECO-EPIDEMIOLOGICAL DYNAMICS SIMULATION")
    print("Coupled Population Competition & Disease Transmission Model")
    print("=" * 70)

    # ========================================================================
    # 1. Model Setup
    # ========================================================================
    print("\n[1] INITIALIZING MODEL PARAMETERS")
    nx, ny = 64, 64
    Lx, Ly = 2.0 * np.pi, 2.0 * np.pi
    params = EcoEpidemicPDE.default_params()

    print(f"  Grid: {nx} x {ny}")
    print(f"  Domain: [0, {Lx:.4f}] x [0, {Ly:.4f}]")
    print(f"  Species 1: D=({params['D_s1']}, {params['D_i1']}, {params['D_r1']})")
    print(f"  Species 2: D=({params['D_s2']}, {params['D_i2']}, {params['D_r2']})")
    print(f"  Advection: v=({params['vx']}, {params['vy']})")
    print(f"  Cross-transmission: beta12={params['beta12']}, beta21={params['beta21']}")

    # ========================================================================
    # 2. Initialize PDE Model and State
    # ========================================================================
    print("\n[2] INITIALIZING SPATIAL PDE MODEL")
    model = EcoEpidemicPDE(nx=nx, ny=ny, Lx=Lx, Ly=Ly, params=params)
    u0 = model.initialize_state(seed=42)

    # Apply numerical robustness checks
    for i in range(6):
        u0[i] = safe_population_density(u0[i], u_min=0.0, u_max=1e4)

    totals_init = model.compute_total_populations(u0)
    print(f"  Initial totals: N1={totals_init['N1']:.2f}, N2={totals_init['N2']:.2f}")
    print(f"  Initial infected: I1={totals_init['I1']:.4f}, I2={totals_init['I2']:.4f}")

    # ========================================================================
    # 3. ETDRK4 Spatial Simulation
    # ========================================================================
    print("\n[3] RUNNING ETDRK4 SPATIAL SIMULATION")
    dt = 0.05
    n_steps = 40
    D_list = model.D

    solver = ETDRK4Solver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        D=D_list, vx=params['vx'], vy=params['vy'],
        dt=dt, n_fields=6
    )

    # Define nonlinear function closure
    def nonlinear_func(u):
        return model.nonlinear_terms(u)

    t_start = time.time()
    history = solver.solve_with_history(u0, nonlinear_func, n_steps, save_every=10)
    t_elapsed = time.time() - t_start
    print(f"  ETDRK4 simulation completed in {t_elapsed:.3f} seconds")
    print(f"  History shape: {history.shape}")

    u_final = history[-1]
    totals_final = model.compute_total_populations(u_final)
    print(f"  Final totals: N1={totals_final['N1']:.2f}, N2={totals_final['N2']:.2f}")
    print(f"  Final infected: I1={totals_final['I1']:.4f}, I2={totals_final['I2']:.4f}")

    # Reproduction numbers
    R0_stats = model.compute_reproduction_numbers(u_final)
    print(f"  R0_1 (mean, max): ({R0_stats['R0_1_mean']:.4f}, {R0_stats['R0_1_max']:.4f})")
    print(f"  R0_2 (mean, max): ({R0_stats['R0_2_mean']:.4f}, {R0_stats['R0_2_max']:.4f})")

    # Critical threshold checks
    r0_check1 = critical_threshold_check(R0_stats['R0_1_mean'], threshold=1.0)
    r0_check2 = critical_threshold_check(R0_stats['R0_2_mean'], threshold=1.0)
    status = { -1: "BELOW", 0: "NEAR", 1: "ABOVE" }
    print(f"  R0_1 threshold status: {status[r0_check1]} critical threshold")
    print(f"  R0_2 threshold status: {status[r0_check2]} critical threshold")

    # ========================================================================
    # 4. Adaptive Midpoint Mean-Field ODE Simulation
    # ========================================================================
    print("\n[4] RUNNING ADAPTIVE MIDPOINT MEAN-FIELD ODE SIMULATION")
    y0_ode = np.array([
        totals_init['S1'] / (nx * ny),
        totals_init['I1'] / (nx * ny),
        totals_init['R1'] / (nx * ny),
        totals_init['S2'] / (nx * ny),
        totals_init['I2'] / (nx * ny),
        totals_init['R2'] / (nx * ny),
    ])

    # Ensure non-negative initial conditions
    y0_ode = np.maximum(y0_ode, 0.0)

    def ode_rhs(t, y):
        return mean_field_eco_epi_ode(t, y, params)

    t_start = time.time()
    ode_result = adaptive_midpoint_solve(
        ode_rhs, y0_ode, t_span=(0.0, n_steps * dt),
        dt_init=0.1, abstol=1e-8, reltol=1e-6
    )
    t_elapsed = time.time() - t_start
    print(f"  Adaptive ODE simulation completed in {t_elapsed:.3f} seconds")
    print(f"  Steps: {ode_result['n_steps']}, Rejected: {ode_result['n_rejected']}")
    print(f"  fsolve calls: {ode_result['n_fsolve']}")
    print(f"  Final ODE state: S1={ode_result['y'][-1, 0]:.4f}, I1={ode_result['y'][-1, 1]:.4f}")

    # ========================================================================
    # 5. Equilibrium Analysis
    # ========================================================================
    print("\n[5] EQUILIBRIUM ANALYSIS")
    eq = equilibrium_analysis(params)
    print(f"  Disease-free equilibrium stable: {eq['disease_free_stable']}")
    print(f"  Basic reproduction number R0 = {eq['R0']:.4f}")
    print(f"  Endemic equilibrium: S*={eq['S_star']:.4f}, I*={eq['I_star']:.4f}, R*={eq['R_star']:.4f}")

    # ========================================================================
    # 6. Hexagonal Patch Integration
    # ========================================================================
    print("\n[6] HEXAGONAL PATCH ECOLOGICAL METRICS")
    hex_radius = Lx / 6.0
    hex_centers = [
        (Lx * 0.25, Ly * 0.5),
        (Lx * 0.5, Ly * 0.5),
        (Lx * 0.75, Ly * 0.5),
    ]

    fields = {
        'S1': u_final[0],
        'I1': u_final[1],
        'S2': u_final[3],
        'I2': u_final[4],
    }
    patch_metrics = compute_hexagonal_patch_metrics(
        fields, model.x, model.y, hex_centers, hex_radius
    )
    for fname, vals in patch_metrics.items():
        print(f"  {fname} patch totals: {[f'{v:.2f}' for v in vals]}")

    # ========================================================================
    # 7. Spatial Geometry of Infection
    # ========================================================================
    print("\n[7] SPATIAL GEOMETRY OF INFECTION")
    I1_total = u_final[1]
    I2_total = u_final[4]

    geom1 = compute_infected_patch_geometry(I1_total, model.x, model.y, threshold=0.5)
    geom2 = compute_infected_patch_geometry(I2_total, model.x, model.y, threshold=0.3)

    print(f"  Species 1 infected patches: {geom1['num_patches']}")
    print(f"  Species 1 total infected area: {geom1['total_infected_area']:.4f}")
    for i, circ in enumerate(geom1['bounding_circles']):
        print(f"    Patch {i+1}: center=({circ['center'][0]:.2f}, {circ['center'][1]:.2f}), radius={circ['radius']:.2f}")

    print(f"  Species 2 infected patches: {geom2['num_patches']}")
    print(f"  Species 2 total infected area: {geom2['total_infected_area']:.4f}")
    for i, circ in enumerate(geom2['bounding_circles']):
        print(f"    Patch {i+1}: center=({circ['center'][0]:.2f}, {circ['center'][1]:.2f}), radius={circ['radius']:.2f}")

    # ========================================================================
    # 8. Exact Solution Verification
    # ========================================================================
    print("\n[8] EXACT SOLUTION VERIFICATION")
    t_test = 1.0
    u_exact = np.zeros((6, nx, ny))
    for fid in range(6):
        u_exact[fid] = manufactured_population_field(model.x, model.y, t_test, fid)

    # Compute L2 error against a single ETDRK4 step from exact initial data
    u_exact_init = np.zeros((6, nx, ny))
    for fid in range(6):
        u_exact_init[fid] = manufactured_population_field(model.x, model.y, 0.0, fid)

    # One step with zero nonlinear terms (pure advection-diffusion)
    def zero_nonlinear(u):
        return np.zeros_like(u)

    solver_test = ETDRK4Solver(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly,
        D=D_list, vx=params['vx'], vy=params['vy'],
        dt=dt, n_fields=6
    )
    u_after_step = solver_test.step(u_exact_init, zero_nonlinear)

    # Compare with exact solution at t=dt
    u_exact_step = np.zeros((6, nx, ny))
    for fid in range(6):
        u_exact_step[fid] = manufactured_population_field(model.x, model.y, dt, fid)

    errors = []
    for fid in range(6):
        err = np.linalg.norm(u_after_step[fid].ravel() - u_exact_step[fid].ravel()) / nx / ny
        errors.append(err)
    mean_err = np.mean(errors)
    print(f"  Manufactured solution mean L2 error (pure diffusion): {mean_err:.6e}")

    # Burgers-type exact solution test
    burgers_field = burgers_exact_solution_2d(model.x, model.y, t_test, nu=0.01)
    print(f"  Burgers exact solution range: [{burgers_field.min():.4f}, {burgers_field.max():.4f}]")

    # ========================================================================
    # 9. Mesh Refinement Test
    # ========================================================================
    print("\n[9] MESH REFINEMENT TEST")
    nodes, triangles = create_uniform_triangular_mesh(0, Lx, 0, Ly, 5, 5)
    print(f"  Initial mesh: {len(nodes)} nodes, {len(triangles)} triangles")
    nodes_ref, triangles_ref = refine_mesh_multiple(nodes, triangles, n_refinements=2)
    print(f"  Refined mesh: {len(nodes_ref)} nodes, {len(triangles_ref)} triangles")

    # Adaptive refinement indicator on I1 field
    indicator = adaptive_refinement_indicator(I1_total, model.x, model.y, threshold=0.3)
    n_refine = np.sum(indicator)
    print(f"  Adaptive refinement indicator: {n_refine}/{nx*ny} cells need refinement")

    # ========================================================================
    # 10. Multi-Dimensional Trait Integration
    # ========================================================================
    print("\n[10] MULTI-DIMENSIONAL TRAIT SPACE INTEGRATION")

    def susceptibility_trait_func(traits):
        """Example trait-dependent susceptibility function."""
        s, v = traits  # susceptibility, virulence
        return np.exp(-(s - 0.5) ** 2 / 0.1) * np.exp(-(v - 0.3) ** 2 / 0.05)

    trait_integral = integrate_trait_space(
        susceptibility_trait_func,
        trait_ranges=[(0.0, 1.0), (0.0, 0.6)],
        orders=[4, 4]
    )
    print(f"  Trait-space integral (susceptibility-virulence): {trait_integral:.6f}")

    # Gauss-Hermite integration test
    def gauss_test_func(x, y):
        return x ** 2 + y ** 2

    gh_integral = integrate_gauss_hermite_2d(gauss_test_func, n=8, sigma_x=1.0, sigma_y=1.0)
    print(f"  Gauss-Hermite verification integral: {gh_integral:.6f} (expected ~6.28 for this scaling)")

    # ========================================================================
    # 11. Sparse Jacobian Sparsity
    # ========================================================================
    print("\n[11] SPARSE JACOBIAN ANALYSIS")
    J_sparsity = build_coupled_jacobian_sparsity(nx, ny, stencil='5point')
    nnz = J_sparsity.nnz
    total = J_sparsity.shape[0] * J_sparsity.shape[1]
    print(f"  Jacobian size: {J_sparsity.shape}")
    print(f"  Non-zeros: {nnz} / {total} ({100.0 * nnz / total:.4f}%)")

    rows, cols, vals = dense_to_triplet(np.random.rand(10, 10) * (np.random.rand(10, 10) > 0.5))
    print(f"  Dense-to-triplet test: {len(vals)} non-zero entries extracted")

    # ========================================================================
    # 12. Epidemiological Distributions
    # ========================================================================
    print("\n[12] EPIDEMIOLOGICAL DISTRIBUTIONS")
    t_gen = np.linspace(0, 20, 21)
    gen_dist = cumulative_generation_interval(t_gen, mean=5.2, std=1.7)
    print(f"  Generation interval CDF at t=5.2 days: {gen_dist[5]:.4f}")
    print(f"  Generation interval CDF at t=10.0 days: {gen_dist[10]:.4f}")

    inf_period = infectious_period_cdf(5.0, gamma_rate=1.0 / 5.0)
    print(f"  Infectious period CDF at t=5.0 days: {inf_period:.4f}")

    # Gamma CDF test
    gamma_val = gamma_cdf(5.0, shape=2.0, scale=2.5)
    print(f"  Gamma(2, 2.5) CDF at x=5.0: {gamma_val:.4f}")

    # ========================================================================
    # 13. Data Thinning and Numerical Robustness
    # ========================================================================
    print("\n[13] NUMERICAL ROBUSTNESS CHECKS")
    thinned = thin_index_2d(nx, ny, factor=4)
    print(f"  2D thinning factor=4: {len(thinned)} points retained from {nx*ny}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Final spatial populations:")
    print(f"  Species 1: N1={totals_final['N1']:.2f} (S1={totals_final['S1']:.2f}, I1={totals_final['I1']:.4f}, R1={totals_final['R1']:.2f})")
    print(f"  Species 2: N2={totals_final['N2']:.2f} (S2={totals_final['S2']:.2f}, I2={totals_final['I2']:.4f}, R2={totals_final['R2']:.2f})")
    print(f"Mean-field ODE final: S1={ode_result['y'][-1, 0]:.4f}, I1={ode_result['y'][-1, 1]:.4f}")
    print("=" * 70)

    return {
        'spatial_final': u_final,
        'ode_result': ode_result,
        'totals_final': totals_final,
        'R0_stats': R0_stats,
        'patch_metrics': patch_metrics,
        'geom1': geom1,
        'geom2': geom2,
    }


if __name__ == '__main__':
    results = main()

# ================================================================
# 测试用例（29个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 29 个测试通过!\n')
