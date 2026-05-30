
import numpy as np
import time

from particle_dynamics import (
    rucklidge_deriv, arneodo_deriv, integrate_trajectory,
    compute_particle_load_field
)
from mesh_generator import QuadMesh, build_delaunay_triangulation
from fem_solver import FEMSystem
from multigrid_poisson import MultigridPoisson1D, MultigridPoisson2D
from spectral_analysis import (
    laguerre_polynomial, generalized_laguerre_function,
    chebyshev_nodes, divided_differences, newton_interpolate,
    chebyshev_interpolate, radial_distribution_spectrum, chebyshev_spectral_derivative
)
from quadrature_rules import integrate_over_mesh, compute_moment_over_mesh
from load_balancer import LoadBalancer, diffusion_based_load_balance
from fast_summation import (
    toeplitz_mv, toeplitz_embedded_fft_mv,
    sample_unit_ball_positive, sample_unit_sphere_surface,
    compute_prefix_sum_2d, query_region_count,
    multipole_expansion, build_interaction_matrix_toeplitz
)
from utils import (
    safe_divide, check_bounds, compute_triangle_area,
    reference_to_physical_q4, mesh_base_one,
    gauss_seidel_sweep, restrict_coarse_to_fine,
    restrict_fine_to_coarse, is_power_of_two
)


def main():
    print("=" * 78)
    print("  自适应谱-有限元粒子方法负载均衡高性能计算框架")
    print("  Adaptive Spectral-Finite-Element Load Balancing for Particle Methods")
    print("=" * 78)
    print()




    np.random.seed(42)
    n_particles = 800
    n_procs = 8
    domain = (0.0, 1.0, 0.0, 1.0)
    t_final = 5.0
    relerr = 1e-5
    abserr = 1e-8

    print(f"[Config] Particles: {n_particles}")
    print(f"[Config] Processors (simulated): {n_procs}")
    print(f"[Config] Domain: {domain}")
    print(f"[Config] Integration time: {t_final}")
    print()




    print("[Step 1] Particle Initialization via Spherical Sampling")
    print("-" * 50)


    ball_samples = sample_unit_ball_positive(n_particles)

    particles_init = np.zeros((n_particles, 2))
    particles_init[:, 0] = check_bounds(
        0.5 + 0.3 * ball_samples[:, 0] * np.cos(ball_samples[:, 1] * 2 * np.pi),
        domain[0], domain[1], "x_init"
    )
    particles_init[:, 1] = check_bounds(
        0.5 + 0.3 * ball_samples[:, 1] * np.sin(ball_samples[:, 2] * 2 * np.pi),
        domain[2], domain[3], "y_init"
    )
    print(f"  Initial particle centroid: ({np.mean(particles_init[:,0]):.4f}, {np.mean(particles_init[:,1]):.4f})")
    print(f"  Initial particle std: ({np.std(particles_init[:,0]):.4f}, {np.std(particles_init[:,1]):.4f})")
    print()




    print("[Step 2] Chaotic Trajectory Integration (RKF45)")
    print("-" * 50)


    n_demo = min(10, n_particles)
    trajectory_data = []
    for p in range(n_demo):

        xyz0 = np.array([
            particles_init[p, 0] * 10.0 - 5.0,
            particles_init[p, 1] * 10.0 - 5.0,
            0.1 * np.random.randn()
        ])
        t_arr, y_arr = integrate_trajectory(
            rucklidge_deriv, xyz0, (0.0, t_final), relerr, abserr, max_steps=5000
        )
        trajectory_data.append((t_arr, y_arr))
        if p < 3:
            print(f"  Particle {p}: {len(t_arr)} steps, final state=({y_arr[-1,0]:.4f},{y_arr[-1,1]:.4f},{y_arr[-1,2]:.4f})")


    particles = particles_init.copy()
    for p in range(n_demo):
        final = trajectory_data[p][1][-1]

        particles[p, 0] = check_bounds(
            0.5 + 0.2 * final[0], domain[0], domain[1], "x"
        )
        particles[p, 1] = check_bounds(
            0.5 + 0.2 * final[1], domain[2], domain[3], "y"
        )
    print(f"  Updated {n_demo} particle positions from chaotic trajectories.")
    print()




    print("[Step 3] Adaptive Mesh Generation (Q4 + Delaunay)")
    print("-" * 50)

    mesh = QuadMesh(domain, nx=8, ny=8)
    print(f"  Initial mesh: {mesh.nodes.shape[0]} nodes, {len(mesh.elements)} elements")


    mesh.refine_by_load(particles, theta=0.4, max_level=2)
    print(f"  Refined mesh: {mesh.nodes.shape[0]} nodes, {len(mesh.elements)} elements")


    nodes_tri, triangles_tri = mesh.triangulate_elements()
    print(f"  Triangulated: {nodes_tri.shape[0]} nodes, {triangles_tri.shape[0]} triangles")





    raise NotImplementedError("Hole_3: main.py triangle validation 待实现")
    print()




    print("[Step 4] Finite Element Field Solver (Poisson)")
    print("-" * 50)

    fem = FEMSystem(nodes_tri, triangles_tri)
    print(f"  FEM system: {fem.n_nodes} nodes, {fem.n_tri} elements")
    print(f"  Boundary nodes: {len(fem.boundary_nodes)}")


    load_field = compute_particle_load_field(particles, domain, 32, 32)

    from scipy.interpolate import RegularGridInterpolator
    try:
        x_grid = np.linspace(domain[0], domain[1], 32)
        y_grid = np.linspace(domain[2], domain[3], 32)
        interpolator = RegularGridInterpolator(
            (x_grid, y_grid), load_field.T,
            bounds_error=False, fill_value=0.0
        )
        rhs_nodes = interpolator(fem.nodes)
    except Exception as e:
        print(f"  [Fallback] Using nearest-neighbor interpolation due to: {e}")
        rhs_nodes = np.zeros(fem.n_nodes)
        for i in range(fem.n_nodes):
            nx_i = int((fem.nodes[i, 0] - domain[0]) / (domain[1] - domain[0]) * 31)
            ny_i = int((fem.nodes[i, 1] - domain[2]) / (domain[3] - domain[2]) * 31)
            nx_i = max(0, min(31, nx_i))
            ny_i = max(0, min(31, ny_i))
            rhs_nodes[i] = load_field[nx_i, ny_i]

    rhs_nodes = check_bounds(rhs_nodes, -1e6, 1e6, "rhs")


    u_fem = fem.solve_poisson(rhs_nodes)
    print(f"  FEM solution range: [{u_fem.min():.4f}, {u_fem.max():.4f}]")
    print()




    print("[Step 5] Multigrid Poisson Solver")
    print("-" * 50)


    def force_1d(x):
        return np.pi ** 2 * np.sin(np.pi * x)

    mg1d = MultigridPoisson1D(
        n=128, a=0.0, b=1.0, ua=0.0, ub=0.0,
        force_func=force_1d
    )
    u_1d, it_1d = mg1d.solve(tol=1e-8, max_iter=50)
    print(f"  1D MG: converged in {it_1d} iterations, max |u|={np.max(np.abs(u_1d)):.4f}")


    nx_mg = 64
    ny_mg = 64
    mg2d = MultigridPoisson2D(nx_mg, ny_mg, domain[1] - domain[0], domain[3] - domain[2])
    rhs_2d = np.zeros((nx_mg + 1, ny_mg + 1))
    for i in range(nx_mg + 1):
        for j in range(ny_mg + 1):
            x = domain[0] + i * (domain[1] - domain[0]) / nx_mg
            y = domain[2] + j * (domain[3] - domain[2]) / ny_mg
            rhs_2d[i, j] = 100.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)

    u_2d, it_2d = mg2d.solve(rhs_2d, tol=1e-6, max_iter=30)
    print(f"  2D MG: converged in {it_2d} iterations, max |u|={np.max(np.abs(u_2d)):.4f}")
    print()




    print("[Step 6] Spectral Analysis")
    print("-" * 50)


    x_lag = np.linspace(0, 10, 200)
    L_vals = laguerre_polynomial(200, 6, x_lag)
    print(f"  Laguerre L_6(5) = {L_vals[100, 6]:.4f} (expected ~1.0)")


    Lf_vals = generalized_laguerre_function(200, 6, 0.5, x_lag)
    print(f"  Generalized L_6^{'{0.5}'}(5) = {Lf_vals[100, 6]:.4f}")


    def test_func(x):
        return np.sin(3 * x) * np.exp(-x ** 2)

    xp_test = np.linspace(-1, 1, 501)
    yp_interp, maxerr = chebyshev_interpolate(test_func, -1.0, 1.0, 16, xp_test)
    print(f"  Chebyshev interpolation (n=16): max error = {maxerr:.3e}")


    r_bins = np.linspace(0, 0.5, 50)

    distances = []
    for i in range(min(200, n_particles)):
        for j in range(i + 1, min(200, n_particles)):
            d = np.linalg.norm(particles[i] - particles[j])
            if d < 0.5:
                distances.append(d)
    if len(distances) > 5:
        hist, _ = np.histogram(distances, bins=r_bins)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        g_r = safe_divide(hist.astype(float), (2.0 * np.pi * r_centers + 1e-12))
        coeffs = radial_distribution_spectrum(r_centers, g_r, n_modes=5, alpha=0.0, beta=2.0)
        print(f"  Radial distribution spectrum coefficients: {coeffs}")
    else:
        print("  [Skip] Not enough particle pairs for radial distribution.")
    print()




    print("[Step 7] High-Precision Quadrature")
    print("-" * 50)


    if triangles_tri.shape[0] > 0:
        verts = nodes_tri[triangles_tri[0] - 1]
        from quadrature_rules import integrate_over_triangle
        val = integrate_over_triangle(verts, lambda x, y: np.sin(x + y), degree=5)
        print(f"  Integral of sin(x+y) over first triangle: {val:.6f}")


    m00 = compute_moment_over_mesh(nodes_tri, triangles_tri, 0, 0, degree=3)
    m10 = compute_moment_over_mesh(nodes_tri, triangles_tri, 1, 0, degree=3)
    m01 = compute_moment_over_mesh(nodes_tri, triangles_tri, 0, 1, degree=3)
    print(f"  Mesh area (m00): {m00:.6f}")
    print(f"  First moments: m10={m10:.6f}, m01={m01:.6f}")
    print()




    print("[Step 8] Dynamic Load Balancing")
    print("-" * 50)

    balancer = LoadBalancer(n_procs, domain, imbalance_threshold=1.3)


    init_loads = balancer.compute_loads(particles)
    init_imbalance = balancer.imbalance_factor(init_loads)
    init_eff = balancer.evaluate_efficiency(init_loads)
    print(f"  Initial load imbalance: I = {init_imbalance:.4f}")
    print(f"  Initial parallel efficiency: eta = {init_eff['parallel_efficiency']:.4f}")


    result = balancer.rebalance(particles)
    if result['rebalanced']:
        print(f"  Rebalancing triggered!")
        print(f"  Old imbalance: {result['old_imbalance']:.4f}")
        print(f"  New imbalance: {result['new_imbalance']:.4f}")
        print(f"  Estimated migrations: {result['migration_count']}")
        new_eff = balancer.evaluate_efficiency(result['new_loads'])
        print(f"  New parallel efficiency: eta = {new_eff['parallel_efficiency']:.4f}")
    else:
        print("  No rebalancing needed (imbalance within threshold).")
    print()


    loads_demo = np.array([120.0, 80.0, 60.0, 100.0, 90.0, 70.0, 110.0, 50.0])
    conn = np.zeros((8, 8))
    for i in range(8):
        conn[i, (i + 1) % 8] = 1
        conn[i, (i - 1) % 8] = 1
    balanced = diffusion_based_load_balance(loads_demo, conn, n_iterations=200, tolerance=1e-2)
    print(f"  Diffusion balance demo: initial std={np.std(loads_demo):.2f}, final std={np.std(balanced):.2f}")
    print()




    print("[Step 9] Fast Summation (Toeplitz + Multipole)")
    print("-" * 50)

    n_toeplitz = 64
    h_t = 1.0 / n_toeplitz
    a_toep = build_interaction_matrix_toeplitz(
        n_toeplitz, lambda r: 1.0 / max(r, 1e-10), h_t
    )
    x_toep = np.random.rand(n_toeplitz)
    y_toep = toeplitz_mv(n_toeplitz, a_toep, x_toep)
    print(f"  Toeplitz MV: input norm={np.linalg.norm(x_toep):.4f}, output norm={np.linalg.norm(y_toep):.4f}")


    y_toep_fft = toeplitz_embedded_fft_mv(n_toeplitz, a_toep, x_toep)
    rel_diff = np.linalg.norm(y_toep - y_toep_fft) / np.linalg.norm(y_toep)
    print(f"  FFT vs direct relative diff: {rel_diff:.3e}")


    subset = particles[:50]
    charges = np.ones(50)
    center = np.mean(subset, axis=0)
    multipole = multipole_expansion(subset, charges, center, max_order=2)
    print(f"  Multipole: monopole={multipole['monopole']:.2f}, dipole_norm={np.linalg.norm(multipole['dipole']):.4f}")


    prefix = compute_prefix_sum_2d(particles, domain, 16, 16)
    count_center = query_region_count(prefix, 6, 10, 6, 10)
    print(f"  Prefix sum query (center region): {count_center} particles")
    print()




    print("[Step 10] Summary & Performance Metrics")
    print("-" * 50)

    final_loads = balancer.compute_loads(particles)
    final_eff = balancer.evaluate_efficiency(final_loads)
    print(f"  Final load stats:")
    print(f"    Mean load:     {final_eff['mean_load']:.2f}")
    print(f"    Std load:      {final_eff['std_load']:.2f}")
    print(f"    Imbalance I:   {final_eff['imbalance_factor']:.4f}")
    print(f"    Efficiency eta: {final_eff['parallel_efficiency']:.4f}")
    print()


    l2_norm_fem = np.sqrt(np.mean(u_fem ** 2))
    print(f"  FEM solution L2 norm: {l2_norm_fem:.4f}")


    energy_norm = np.sqrt(np.sum(u_2d ** 2) / ((nx_mg + 1) * (ny_mg + 1)))
    print(f"  MG 2D solution energy norm estimate: {energy_norm:.4f}")
    print()

    print("=" * 78)
    print("  Simulation completed successfully.")
    print("=" * 78)


if __name__ == "__main__":
    main()
