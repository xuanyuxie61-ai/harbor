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

    # === HOLE 3 START ===
    # 修复要求：正确初始化 ETDRK4Solver 并执行空间模拟。
    # 工程协同要点：
    #   1. ETDRK4Solver 的 n_fields 必须与模型状态变量维度一致（当前为 6，对应 S1,I1,R1,S2,I2,R2）
    #   2. D_list（来自 model.D）的顺序必须与状态变量顺序一致：
    #      [D_s1, D_i1, D_r1, D_s2, D_i2, D_r2]
    #   3. nonlinear_func 闭包必须正确调用 model.nonlinear_terms(u)
    #   4. 调用 solver.solve_with_history(u0, nonlinear_func, n_steps, save_every=10) 获取历史状态
    # 注：此处的 n_fields 和 D_list 与 reaction_kinetics.py 中反应项返回的 6 维场顺序、
    #     eco_epi_pde.py 中 self.D 的定义顺序存在跨文件耦合关系。
    solver = None  # 需正确初始化 ETDRK4Solver
    def nonlinear_func(u):
        raise NotImplementedError("HOLE 3: 请实现 nonlinear_func 闭包")
    history = None  # 需调用 solver.solve_with_history 获取
    t_start = time.time()
    # history = solver.solve_with_history(...)
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
