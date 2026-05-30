
import numpy as np
import time


from spatial_mesh import generate_simple_tet_mesh
from environment_field import generate_gaussian_bump_field, generate_gradient_field, sample_field_at_positions
from sensor_noise import apply_sensor_noise
from coverage_optimization import cvt_circle_nonuniform_density, coverage_metric
from interaction_matrix import build_sparse_laplacian, fiedler_value
from stochastic_control import feynman_kac_collision_potential, gradient_fk_potential
from swarm_dynamics import SwarmRobot, integrate_swarm
from density_field import density_continuum_1d
from distance_statistics import compute_emergence_index
from spectral_approx import delay_kernel_chebyshev, bernstein_poly_ab_approx


def main():
    np.random.seed(42)
    t_start = time.time()

    print("=" * 72)
    print("SWARM ROBOTICS EMERGENT BEHAVIOR SIMULATION")
    print("Domain: Multi-scale modeling of self-organizing robot collectives")
    print("=" * 72)




    print("\n[1] Generating and refining tetrahedral workspace mesh ...")
    mesh = generate_simple_tet_mesh()
    print(f"    Initial mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} tetrahedra")
    mesh = mesh.refine()
    print(f"    Refined mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} tetrahedra")




    print("\n[2] Constructing environmental scalar fields ...")
    bump_field = generate_gaussian_bump_field(mesh, center=np.array([0.5, 0.5, 0.5]),
                                               sigma=0.25, amplitude=1.0)
    grad_field = generate_gradient_field(mesh, direction=np.array([1.0, 0.5, 0.0]), magnitude=0.5)

    composite_vals = bump_field.nodal_values + grad_field.nodal_values
    from environment_field import EnvironmentField
    env_field = EnvironmentField(mesh, composite_vals)
    print(f"    Field value range: [{composite_vals.min():.4f}, {composite_vals.max():.4f}]")




    print("\n[3] Initializing swarm robots with chaotic internal states ...")
    N_robots = 12
    arena_radius = 1.0
    angles = np.linspace(0, 2 * np.pi, N_robots, endpoint=False)
    init_positions = 0.6 * arena_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    init_positions = np.pad(init_positions, ((0, 0), (0, 1)), mode='constant')

    robots = []
    for i in range(N_robots):
        s0 = np.array([0.2 + 0.01 * i, 0.2 - 0.01 * i, -0.75 + 0.02 * i])
        r = SwarmRobot(position=init_positions[i], velocity=np.zeros(3), internal_state=s0)
        robots.append(r)
    print(f"    Number of robots: {N_robots}")




    print("\n[4] Running CVT coverage optimization ...")
    gens_2d = init_positions[:, :2].copy()
    from coverage_optimization import cvt_lloyd_2d

    def circle_density(x, y):
        r2 = (x ** 2 + y ** 2) / (arena_radius ** 2)
        return 1.0 + 5.0 * np.exp(-5.0 * r2)

    bounds = (-arena_radius, arena_radius, -arena_radius, arena_radius)
    optimized_2d, energy_hist = cvt_lloyd_2d(gens_2d, circle_density, bounds,
                                             n_samples=5000, n_iterations=15)

    for i in range(N_robots):
        robots[i].position[:2] = optimized_2d[i]
        robots[i].position[2] = 0.0
    print(f"    CVT energy initial: {energy_hist[0]:.4f}, final: {energy_hist[-1]:.4f}")


    cov_after = coverage_metric(optimized_2d, circle_density, bounds, n_samples=8000)
    print(f"    Coverage metric after CVT: {cov_after:.6f}")




    print("\n[5] Simulating sensor noise ...")
    robot_positions = np.array([r.position for r in robots])
    field_samples = sample_field_at_positions(env_field, robot_positions)

    fmin, fmax = field_samples.min(), field_samples.max()
    if abs(fmax - fmin) > 1e-10:
        field_norm = (field_samples - fmin) / (fmax - fmin)
    else:
        field_norm = np.zeros_like(field_samples)

    noise_config = {
        "salt_pepper_level": 0.03,
        "uniform_level": 0.02,
        "gaussian_sigma": 0.01
    }
    noisy_norm = apply_sensor_noise(field_norm, noise_config)

    noisy_samples = noisy_norm * (fmax - fmin) + fmin
    print(f"    Clean readings:  mean={field_samples.mean():.4f}, std={field_samples.std():.4f}")
    print(f"    Noisy readings:  mean={noisy_samples.mean():.4f}, std={noisy_samples.std():.4f}")




    print("\n[6] Building sparse interaction graph ...")
    sensing_radius = 0.8
    positions_2d = np.array([r.position[:2] for r in robots])
    L, W = build_sparse_laplacian(positions_2d, sensing_radius)
    lambda2 = fiedler_value(L)
    print(f"    Graph edges (nonzeros in W): {W.nnz}")
    print(f"    Algebraic connectivity (Fiedler value): {lambda2:.6f}")




    print("\n[7] Computing Feynman-Kac collision-avoidance potentials ...")
    obstacles = np.array([[0.0, 0.0]])
    fk_pot = feynman_kac_collision_potential(positions_2d, obstacles,
                                             obstacle_radius=0.3, domain_radius=arena_radius)
    fk_grad = gradient_fk_potential(positions_2d, obstacles, obstacle_radius=0.3,
                                    domain_radius=arena_radius)
    print(f"    FK potential mean: {fk_pot.mean():.4f}, min: {fk_pot.min():.4f}, max: {fk_pot.max():.4f}")




    print("\n[8] Integrating coupled swarm dynamics (BDF3 + Theta) ...")




    control_gains = {}
    consensus_target = None
    t_swarm, traj_swarm = integrate_swarm(robots, (0.0, 2.0), n_steps=25,
                                          control_gains=control_gains,
                                          env_gradient_func=env_grad_wrapper,
                                          consensus_target=consensus_target,
                                          method="rk4")



    from swarm_dynamics import solve_bdf3, solve_theta_method, arneodo_deriv
    y0_arneodo = np.array([0.2, 0.2, -0.75], dtype=float)
    t_bdf, traj_bdf = solve_bdf3(arneodo_deriv, (0.0, 1.0), y0_arneodo, n=20)
    t_theta, traj_theta = solve_theta_method(arneodo_deriv, (0.0, 1.0), y0_arneodo, n=20, theta=0.5)

    final_states = traj_swarm[-1, :]
    final_positions = final_states.reshape(N_robots, -1)[:, :2]
    print(f"    Swarm RK4 integration: t in [{t_swarm[0]:.2f}, {t_swarm[-1]:.2f}], steps={len(t_swarm)-1}")
    print(f"    Arneodo BDF3:          t in [{t_bdf[0]:.2f}, {t_bdf[-1]:.2f}], steps={len(t_bdf)-1}")
    print(f"    Arneodo Theta:         t in [{t_theta[0]:.2f}, {t_theta[-1]:.2f}], steps={len(t_theta)-1}")




    print("\n[9] Solving macroscopic density continuity equation (ETDRK4 spectral) ...")
    x_rho, tt_rho, rho_field = density_continuum_1d(nx=128, tmax=1.0, nu=0.05, D4=1e-4)
    print(f"    Spatial grid: {len(x_rho)} points, temporal snapshots: {len(tt_rho)}")
    print(f"    Density range: [{rho_field.min():.4f}, {rho_field.max():.4f}]")




    print("\n[10] Evaluating emergence metrics ...")

    e_init, dmean_init, dvar_init = compute_emergence_index(positions_2d, arena_radius)

    e_final, dmean_final, dvar_final = compute_emergence_index(final_positions, arena_radius)
    print(f"    Initial distance mean: {dmean_init:.4f}, var: {dvar_init:.4f}, KL: {e_init:.4f}")
    print(f"    Final   distance mean: {dmean_final:.4f}, var: {dvar_final:.4f}, KL: {e_final:.4f}")


    from distance_statistics import multivariate_distance_stats
    mu_mv, var_mv = multivariate_distance_stats(dim=6, n=2000)
    print(f"    High-dim distance stats (dim=6): mu={mu_mv:.4f}, var={var_mv:.4f}")




    print("\n[11] Validating spectral approximation tools ...")
    from spectral_approx import delay_kernel_chebyshev, chebyshev_interpolant
    c = delay_kernel_chebyshev(tau_max=0.5, n=16, kernel_type="exponential")
    tau_test = np.linspace(0, 0.5, 20)
    k_test = chebyshev_interpolant(0.0, 0.5, 16, c, tau_test)

    k_test = np.clip(k_test, 0.0, None)
    integral = np.trapezoid(k_test, tau_test)
    if integral > 1e-6:
        k_test = k_test / integral
    print(f"    Chebyshev delay kernel integral after renormalize: {np.trapezoid(k_test, tau_test):.4f}")


    from spectral_approx import bernstein_poly_ab_approx
    n_bern = 8
    ydata = np.exp(-np.linspace(0, 1, n_bern + 1))
    xval = np.linspace(0, 1, 50)
    yval = bernstein_poly_ab_approx(n_bern, 0.0, 1.0, ydata, xval)
    print(f"    Bernstein approximant range: [{yval.min():.4f}, {yval.max():.4f}]")




    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("SIMULATION SUMMARY")
    print("=" * 72)
    print(f"  Workspace mesh nodes:          {mesh.nodes.shape[0]}")
    print(f"  Number of robots:              {N_robots}")
    print(f"  CVT coverage improvement:      {energy_hist[0]:.4f} -> {energy_hist[-1]:.4f}")
    print(f"  Algebraic connectivity:        {lambda2:.6f}")
    print(f"  Emergence KL divergence:       {e_init:.4f} -> {e_final:.4f}")
    print(f"  Macroscopic density snapshots: {rho_field.shape[1]}")
    print(f"  Total elapsed time:            {elapsed:.2f} s")
    print("  Status: ALL MODULES EXECUTED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
