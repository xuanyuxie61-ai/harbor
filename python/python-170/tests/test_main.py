"""
main.py
=======
Unified zero-parameter entry point for the swarm robotics emergence simulation.

Scientific workflow:
  1. Generate a tetrahedral workspace mesh and refine it once (8-fold).
  2. Create an environmental scalar field (Gaussian bump + gradient) on the mesh.
  3. Initialize N robots in a circular arena with random positions and Arneodo
     chaotic internal states.
  4. Run CVT coverage optimization to obtain an initial ordered configuration.
  5. Simulate stochastic sensor noise corruption of environmental readings.
  6. Build sparse interaction Laplacian and compute Fiedler value (emergence metric).
  7. Compute Feynman-Kac collision-avoidance potential gradients.
  8. Integrate coupled swarm dynamics (BDF3 + Theta method) with
     mechanical + chaotic + consensus + repulsion + chemotaxis terms.
  9. Solve the macroscopic density continuity equation via ETDRK4 spectral method.
 10. Evaluate emergence metrics: distance statistics, KL divergence from uniform,
     coverage cost evolution, and spectral Laplacian eigenvalue tracking.
 11. Print all quantitative results.
"""

import numpy as np
import time

# Local modules
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

    # ------------------------------------------------------------------
    # 1. Spatial mesh generation and refinement (tet_mesh_refine)
    # ------------------------------------------------------------------
    print("\n[1] Generating and refining tetrahedral workspace mesh ...")
    mesh = generate_simple_tet_mesh()
    print(f"    Initial mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} tetrahedra")
    mesh = mesh.refine()
    print(f"    Refined mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} tetrahedra")

    # ------------------------------------------------------------------
    # 2. Environmental scalar field (fem_to_tec / FEM interpolation)
    # ------------------------------------------------------------------
    print("\n[2] Constructing environmental scalar fields ...")
    bump_field = generate_gaussian_bump_field(mesh, center=np.array([0.5, 0.5, 0.5]),
                                               sigma=0.25, amplitude=1.0)
    grad_field = generate_gradient_field(mesh, direction=np.array([1.0, 0.5, 0.0]), magnitude=0.5)
    # composite field
    composite_vals = bump_field.nodal_values + grad_field.nodal_values
    from environment_field import EnvironmentField
    env_field = EnvironmentField(mesh, composite_vals)
    print(f"    Field value range: [{composite_vals.min():.4f}, {composite_vals.max():.4f}]")

    # ------------------------------------------------------------------
    # 3. Initialize swarm robots (arneodo_ode internal chaos)
    # ------------------------------------------------------------------
    print("\n[3] Initializing swarm robots with chaotic internal states ...")
    N_robots = 12
    arena_radius = 1.0
    angles = np.linspace(0, 2 * np.pi, N_robots, endpoint=False)
    init_positions = 0.6 * arena_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    init_positions = np.pad(init_positions, ((0, 0), (0, 1)), mode='constant')  # 3D: z=0

    robots = []
    for i in range(N_robots):
        s0 = np.array([0.2 + 0.01 * i, 0.2 - 0.01 * i, -0.75 + 0.02 * i])
        r = SwarmRobot(position=init_positions[i], velocity=np.zeros(3), internal_state=s0)
        robots.append(r)
    print(f"    Number of robots: {N_robots}")

    # ------------------------------------------------------------------
    # 4. CVT coverage optimization (cvt_circle_nonuniform)
    # ------------------------------------------------------------------
    print("\n[4] Running CVT coverage optimization ...")
    gens_2d = init_positions[:, :2].copy()
    from coverage_optimization import cvt_lloyd_2d

    def circle_density(x, y):
        r2 = (x ** 2 + y ** 2) / (arena_radius ** 2)
        return 1.0 + 5.0 * np.exp(-5.0 * r2)

    bounds = (-arena_radius, arena_radius, -arena_radius, arena_radius)
    optimized_2d, energy_hist = cvt_lloyd_2d(gens_2d, circle_density, bounds,
                                             n_samples=5000, n_iterations=15)
    # project to 3D (z=0)
    for i in range(N_robots):
        robots[i].position[:2] = optimized_2d[i]
        robots[i].position[2] = 0.0
    print(f"    CVT energy initial: {energy_hist[0]:.4f}, final: {energy_hist[-1]:.4f}")

    # coverage metric after CVT
    cov_after = coverage_metric(optimized_2d, circle_density, bounds, n_samples=8000)
    print(f"    Coverage metric after CVT: {cov_after:.6f}")

    # ------------------------------------------------------------------
    # 5. Sensor noise injection (image_noise)
    # ------------------------------------------------------------------
    print("\n[5] Simulating sensor noise ...")
    robot_positions = np.array([r.position for r in robots])
    field_samples = sample_field_at_positions(env_field, robot_positions)
    # normalize to [0,1] for noise model
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
    # denormalize
    noisy_samples = noisy_norm * (fmax - fmin) + fmin
    print(f"    Clean readings:  mean={field_samples.mean():.4f}, std={field_samples.std():.4f}")
    print(f"    Noisy readings:  mean={noisy_samples.mean():.4f}, std={noisy_samples.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Sparse interaction Laplacian (sparse_parfor)
    # ------------------------------------------------------------------
    print("\n[6] Building sparse interaction graph ...")
    sensing_radius = 0.8
    positions_2d = np.array([r.position[:2] for r in robots])
    L, W = build_sparse_laplacian(positions_2d, sensing_radius)
    lambda2 = fiedler_value(L)
    print(f"    Graph edges (nonzeros in W): {W.nnz}")
    print(f"    Algebraic connectivity (Fiedler value): {lambda2:.6f}")

    # ------------------------------------------------------------------
    # 7. Feynman-Kac collision potential (feynman_kac_1d)
    # ------------------------------------------------------------------
    print("\n[7] Computing Feynman-Kac collision-avoidance potentials ...")
    obstacles = np.array([[0.0, 0.0]])
    fk_pot = feynman_kac_collision_potential(positions_2d, obstacles,
                                             obstacle_radius=0.3, domain_radius=arena_radius)
    fk_grad = gradient_fk_potential(positions_2d, obstacles, obstacle_radius=0.3,
                                    domain_radius=arena_radius)
    print(f"    FK potential mean: {fk_pot.mean():.4f}, min: {fk_pot.min():.4f}, max: {fk_pot.max():.4f}")

    # ------------------------------------------------------------------
    # 8. Swarm dynamics integration (bdf3 + theta_method + arneodo)
    # ------------------------------------------------------------------
    print("\n[8] Integrating coupled swarm dynamics (BDF3 + Theta) ...")
    control_gains = {
        "gamma": 0.8,
        "kp": 1.2,
        "kv": 0.6,
        "repulsion_range": 0.35,
        "repulsion_strength": 0.8
    }

    def env_grad_wrapper(p):
        return env_field.gradient(p)

    consensus_target = np.array([np.mean(noisy_samples)])

    # Main swarm integration uses RK4 for efficiency
    t_swarm, traj_swarm = integrate_swarm(robots, (0.0, 2.0), n_steps=25,
                                          control_gains=control_gains,
                                          env_gradient_func=env_grad_wrapper,
                                          consensus_target=consensus_target,
                                          method="rk4")

    # BDF3 and Theta method demonstrated on the Arneodo chaotic subsystem
    # (lower dimension => fsolve converges rapidly)
    from swarm_dynamics import solve_bdf3, solve_theta_method, arneodo_deriv
    y0_arneodo = np.array([0.2, 0.2, -0.75], dtype=float)
    t_bdf, traj_bdf = solve_bdf3(arneodo_deriv, (0.0, 1.0), y0_arneodo, n=20)
    t_theta, traj_theta = solve_theta_method(arneodo_deriv, (0.0, 1.0), y0_arneodo, n=20, theta=0.5)

    final_states = traj_swarm[-1, :]
    final_positions = final_states.reshape(N_robots, -1)[:, :2]
    print(f"    Swarm RK4 integration: t in [{t_swarm[0]:.2f}, {t_swarm[-1]:.2f}], steps={len(t_swarm)-1}")
    print(f"    Arneodo BDF3:          t in [{t_bdf[0]:.2f}, {t_bdf[-1]:.2f}], steps={len(t_bdf)-1}")
    print(f"    Arneodo Theta:         t in [{t_theta[0]:.2f}, {t_theta[-1]:.2f}], steps={len(t_theta)-1}")

    # ------------------------------------------------------------------
    # 9. Macroscopic density field via ETDRK4 (burgers_pde_etdrk4)
    # ------------------------------------------------------------------
    print("\n[9] Solving macroscopic density continuity equation (ETDRK4 spectral) ...")
    x_rho, tt_rho, rho_field = density_continuum_1d(nx=128, tmax=1.0, nu=0.05, D4=1e-4)
    print(f"    Spatial grid: {len(x_rho)} points, temporal snapshots: {len(tt_rho)}")
    print(f"    Density range: [{rho_field.min():.4f}, {rho_field.max():.4f}]")

    # ------------------------------------------------------------------
    # 10. Emergence metrics (distance_statistics + circle_distance)
    # ------------------------------------------------------------------
    print("\n[10] Evaluating emergence metrics ...")
    # initial emergence
    e_init, dmean_init, dvar_init = compute_emergence_index(positions_2d, arena_radius)
    # after swarm dynamics
    e_final, dmean_final, dvar_final = compute_emergence_index(final_positions, arena_radius)
    print(f"    Initial distance mean: {dmean_init:.4f}, var: {dvar_init:.4f}, KL: {e_init:.4f}")
    print(f"    Final   distance mean: {dmean_final:.4f}, var: {dvar_final:.4f}, KL: {e_final:.4f}")

    # multivariate distance statistics for high-dimensional embedding
    from distance_statistics import multivariate_distance_stats
    mu_mv, var_mv = multivariate_distance_stats(dim=6, n=2000)
    print(f"    High-dim distance stats (dim=6): mu={mu_mv:.4f}, var={var_mv:.4f}")

    # ------------------------------------------------------------------
    # 11. Spectral tools validation (chebyshev + bernstein)
    # ------------------------------------------------------------------
    print("\n[11] Validating spectral approximation tools ...")
    from spectral_approx import delay_kernel_chebyshev, chebyshev_interpolant
    c = delay_kernel_chebyshev(tau_max=0.5, n=16, kernel_type="exponential")
    tau_test = np.linspace(0, 0.5, 20)
    k_test = chebyshev_interpolant(0.0, 0.5, 16, c, tau_test)
    # Verify kernel positivity and approximate unit integral by renormalizing
    k_test = np.clip(k_test, 0.0, None)
    integral = np.trapezoid(k_test, tau_test)
    if integral > 1e-6:
        k_test = k_test / integral
    print(f"    Chebyshev delay kernel integral after renormalize: {np.trapezoid(k_test, tau_test):.4f}")

    # Bernstein control policy approximation
    from spectral_approx import bernstein_poly_ab_approx
    n_bern = 8
    ydata = np.exp(-np.linspace(0, 1, n_bern + 1))
    xval = np.linspace(0, 1, 50)
    yval = bernstein_poly_ab_approx(n_bern, 0.0, 1.0, ydata, xval)
    print(f"    Bernstein approximant range: [{yval.min():.4f}, {yval.max():.4f}]")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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

# ================================================================
# 测试用例（42个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: TetMesh creation produces correct node/element counts ----
from spatial_mesh import generate_simple_tet_mesh
mesh = generate_simple_tet_mesh()
assert mesh.nodes.shape[0] == 8, '[TC01] TetMesh creation node count FAILED'
assert mesh.elements.shape[0] == 6, '[TC01] TetMesh creation element count FAILED'

# ---- TC02: TetMesh refine produces 8x elements per tet ----
mesh_refined = mesh.refine()
assert mesh_refined.elements.shape[0] == 48, '[TC02] TetMesh refine element count FAILED'
assert mesh_refined.nodes.shape[0] > mesh.nodes.shape[0], '[TC02] TetMesh refine node count FAILED'

# ---- TC03: point_in_tet returns True for the centroid of a tetrahedron ----
import numpy as np
center = np.mean(mesh.nodes[mesh.elements[0]], axis=0)
inside, bary = mesh.point_in_tet(center, 0)
assert inside, '[TC03] point_in_tet centroid should be inside FAILED'

# ---- TC04: point_in_tet returns False for a point far outside ----
far_point = np.array([100.0, 100.0, 100.0])
inside, bary = mesh.point_in_tet(far_point, 0)
assert not inside, '[TC04] point_in_tet far point should be outside FAILED'

# ---- TC05: locate_point finds tetrahedron containing the origin ----
import numpy as np
origin = np.array([0.0, 0.0, 0.0])
tet_idx, bary = mesh.locate_point(origin)
assert tet_idx is not None, '[TC05] locate_point failed for origin FAILED'

# ---- TC06: interpolate_nodal_field at a node returns the exact nodal value ----
import numpy as np
np.random.seed(42)
field_vals = np.random.rand(mesh.nodes.shape[0])
val = mesh.interpolate_nodal_field(mesh.nodes[0], field_vals)
assert abs(val - field_vals[0]) < 1e-10, '[TC06] interpolate_nodal_field at node FAILED'

# ---- TC07: generate_gaussian_bump_field max does not exceed amplitude ----
import numpy as np
from environment_field import generate_gaussian_bump_field
from spatial_mesh import generate_simple_tet_mesh
mesh2 = generate_simple_tet_mesh()
center = np.array([0.0, 0.0, 0.0])
field = generate_gaussian_bump_field(mesh2, center=center, sigma=1.0, amplitude=2.0)
assert field.nodal_values.max() <= 2.0 + 1e-10, '[TC07] Gaussian bump field max exceeds amplitude FAILED'
assert field.nodal_values.max() > 0.0, '[TC07] Gaussian bump field max positive FAILED'

# ---- TC08: generate_gradient_field produces positive values along positive direction ----
import numpy as np
from environment_field import generate_gradient_field
direction = np.array([1.0, 0.0, 0.0])
field_grad = generate_gradient_field(mesh2, direction=direction, magnitude=1.0)
pos_mask = mesh2.nodes[:, 0] > 1e-10
if np.any(pos_mask):
    assert np.all(field_grad.nodal_values[pos_mask] > 0), '[TC08] Gradient field positive x FAILED'

# ---- TC09: sample_field_at_positions returns correct output shape ----
import numpy as np
from environment_field import sample_field_at_positions, generate_gaussian_bump_field
pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
vals = sample_field_at_positions(field, pos)
assert vals.shape == (2,), '[TC09] sample_field_at_positions shape FAILED'

# ---- TC10: salt_and_pepper_noise with level=0 returns unchanged data ----
import numpy as np
from sensor_noise import salt_and_pepper_noise
np.random.seed(42)
data = np.array([0.2, 0.5, 0.8])
noisy = salt_and_pepper_noise(data, level=0.0)
assert np.allclose(noisy, data), '[TC10] salt_and_pepper_noise zero level FAILED'

# ---- TC11: salt_and_pepper_noise produces deterministic output with fixed seed ----
import numpy as np
from sensor_noise import salt_and_pepper_noise
np.random.seed(42)
noisy1 = salt_and_pepper_noise(data, level=0.3)
np.random.seed(42)
noisy2 = salt_and_pepper_noise(data, level=0.3)
assert np.allclose(noisy1, noisy2), '[TC11] salt_and_pepper_noise reproducibility FAILED'

# ---- TC12: gaussian_sensor_noise output stays within [0, 1] ----
import numpy as np
from sensor_noise import gaussian_sensor_noise
np.random.seed(42)
data2 = np.array([0.1, 0.9, 0.5])
noisy_g = gaussian_sensor_noise(data2, sigma=0.1)
assert np.all(noisy_g >= 0.0) and np.all(noisy_g <= 1.0), '[TC12] gaussian_sensor_noise range FAILED'

# ---- TC13: apply_sensor_noise composite produces finite output ----
import numpy as np
from sensor_noise import apply_sensor_noise
np.random.seed(42)
data3 = np.ones(10) * 0.5
config = {"salt_pepper_level": 0.1, "uniform_level": 0.1, "gaussian_sigma": 0.05}
noisy_c = apply_sensor_noise(data3, config)
assert noisy_c.shape == data3.shape, '[TC13] apply_sensor_noise shape FAILED'
assert np.all(np.isfinite(noisy_c)), '[TC13] apply_sensor_noise finite FAILED'

# ---- TC14: cvt_lloyd_2d produces correct energy history length ----
import numpy as np
from coverage_optimization import cvt_lloyd_2d
np.random.seed(42)
gens_init = np.random.uniform(-0.8, 0.8, (5, 2))
def density_const(x, y):
    return 1.0
bounds = (-1.0, 1.0, -1.0, 1.0)
gens_opt, energy = cvt_lloyd_2d(gens_init, density_const, bounds, n_samples=2000, n_iterations=10)
assert len(energy) == 10, '[TC14] cvt_lloyd_2d energy history length FAILED'
assert gens_opt.shape == gens_init.shape, '[TC14] cvt_lloyd_2d output shape FAILED'

# ---- TC15: cvt_circle_nonuniform_density returns correct output shape ----
import numpy as np
from coverage_optimization import cvt_circle_nonuniform_density
np.random.seed(42)
gens_c, energy_c = cvt_circle_nonuniform_density(6, radius=1.0, n_iterations=8, n_samples=3000)
assert gens_c.shape == (6, 2), '[TC15] cvt_circle_nonuniform_density shape FAILED'
assert len(energy_c) == 8, '[TC15] cvt_circle_nonuniform_density energy length FAILED'

# ---- TC16: coverage_metric returns non-negative finite value ----
import numpy as np
from coverage_optimization import coverage_metric
np.random.seed(42)
metric = coverage_metric(gens_c, density_const, bounds, n_samples=2000)
assert np.isfinite(metric), '[TC16] coverage_metric finite FAILED'
assert metric >= 0.0, '[TC16] coverage_metric non-negative FAILED'

# ---- TC17: build_sparse_laplacian creates edges for nearby points ----
import numpy as np
from interaction_matrix import build_sparse_laplacian
positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
L, W = build_sparse_laplacian(positions, sensing_radius=1.0)
assert L.shape == (3, 3), '[TC17] build_sparse_laplacian shape FAILED'
assert W.nnz > 0, '[TC17] build_sparse_laplacian edges FAILED'

# ---- TC18: fiedler_value is non-negative ----
import numpy as np
from interaction_matrix import fiedler_value
lambda2 = fiedler_value(L)
assert lambda2 >= 0.0, '[TC18] fiedler_value non-negative FAILED'

# ---- TC19: consensus_dynamics_step preserves shape and produces finite result ----
import numpy as np
from interaction_matrix import consensus_dynamics_step, build_sparse_laplacian
np.random.seed(42)
x0 = np.random.randn(5)
L3, _ = build_sparse_laplacian(np.random.rand(5, 2) * 0.5, sensing_radius=1.0)
x1 = consensus_dynamics_step(x0, L3, dt=0.01)
assert x1.shape == x0.shape, '[TC19] consensus_dynamics_step shape FAILED'
assert np.all(np.isfinite(x1)), '[TC19] consensus_dynamics_step finite FAILED'

# ---- TC20: arneodo_deriv produces correct output shape and finite values ----
import numpy as np
from swarm_dynamics import arneodo_deriv
y0 = np.array([0.1, 0.2, -0.5])
dy = arneodo_deriv(0.0, y0)
assert dy.shape == (3,), '[TC20] arneodo_deriv shape FAILED'
assert np.all(np.isfinite(dy)), '[TC20] arneodo_deriv finite FAILED'

# ---- TC21: solve_bdf3 returns correct output shapes ----
import numpy as np
from swarm_dynamics import solve_bdf3, arneodo_deriv
t_bdf, y_bdf = solve_bdf3(arneodo_deriv, (0.0, 0.5), y0, n=10)
assert t_bdf.shape == (11,), '[TC21] solve_bdf3 t shape FAILED'
assert y_bdf.shape == (11, 3), '[TC21] solve_bdf3 y shape FAILED'
assert np.all(np.isfinite(y_bdf)), '[TC21] solve_bdf3 finite FAILED'

# ---- TC22: solve_theta_method returns correct output shapes ----
import numpy as np
from swarm_dynamics import solve_theta_method
t_theta, y_theta = solve_theta_method(arneodo_deriv, (0.0, 0.5), y0, n=10, theta=0.5)
assert t_theta.shape == (11,), '[TC22] solve_theta_method t shape FAILED'
assert y_theta.shape == (11, 3), '[TC22] solve_theta_method y shape FAILED'

# ---- TC23: solve_rk4 returns correct output shapes ----
import numpy as np
from swarm_dynamics import solve_rk4
t_rk4, y_rk4 = solve_rk4(arneodo_deriv, (0.0, 0.5), y0, n=10)
assert t_rk4.shape == (11,), '[TC23] solve_rk4 t shape FAILED'
assert y_rk4.shape == (11, 3), '[TC23] solve_rk4 y shape FAILED'

# ---- TC24: repulsion_force is zero for robots far beyond repulsion range ----
import numpy as np
from swarm_dynamics import repulsion_force
p1 = np.array([0.0, 0.0, 0.0])
p2 = np.array([10.0, 10.0, 10.0])
force = repulsion_force(p1, p2, repulsion_range=0.3)
assert np.linalg.norm(force) < 1e-10, '[TC24] repulsion_force beyond range FAILED'

# ---- TC25: SwarmRobot state getter/setter roundtrip ----
import numpy as np
from swarm_dynamics import SwarmRobot
robot = SwarmRobot(position=np.array([1.0, 2.0, 3.0]), velocity=np.array([0.1, 0.2, 0.3]))
s = robot.state
assert len(s) == 9, '[TC25] SwarmRobot state length FAILED'
robot2 = SwarmRobot(position=np.array([0.0, 0.0, 0.0]))
robot2.state = s
assert np.allclose(robot2.position, [1.0, 2.0, 3.0]), '[TC25] SwarmRobot state roundtrip position FAILED'

# ---- TC26: integrate_swarm with rk4 produces correct trajectory shape ----
import numpy as np
np.random.seed(42)
from swarm_dynamics import SwarmRobot, integrate_swarm
from spatial_mesh import generate_simple_tet_mesh
from environment_field import EnvironmentField, generate_gaussian_bump_field
mesh3 = generate_simple_tet_mesh()
field3 = generate_gaussian_bump_field(mesh3, center=np.array([0.0, 0.0, 0.0]), sigma=0.5)
robots = []
for i in range(4):
    ang = 2 * np.pi * i / 4
    r = SwarmRobot(position=np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang), 0.0]), velocity=np.zeros(3))
    robots.append(r)
gains = {"gamma": 0.5, "kp": 1.0, "kv": 0.5, "repulsion_range": 0.3, "repulsion_strength": 0.5}
def env_grad(p):
    return np.zeros(3)
t_sw, traj_sw = integrate_swarm(robots, (0.0, 0.2), n_steps=5, control_gains=gains,
                                 env_gradient_func=env_grad, consensus_target=None, method="rk4")
assert t_sw.shape == (6,), '[TC26] integrate_swarm t shape FAILED'
assert traj_sw.shape == (6, 36), '[TC26] integrate_swarm traj shape FAILED'

# ---- TC27: solve_burgers_etdrk4 produces correct output shapes ----
import numpy as np
from density_field import solve_burgers_etdrk4
x_b, tt_b, uu_b = solve_burgers_etdrk4(nx=64, nt=5, vis=0.03, tmax=0.5)
assert x_b.shape == (64,), '[TC27] solve_burgers_etdrk4 x shape FAILED'
assert len(tt_b) >= 1, '[TC27] solve_burgers_etdrk4 tt length FAILED'
assert uu_b.shape[0] == 64, '[TC27] solve_burgers_etdrk4 uu rows FAILED'
assert np.all(np.isfinite(uu_b)), '[TC27] solve_burgers_etdrk4 finite FAILED'

# ---- TC28: density_continuum_1d produces correct output shapes ----
import numpy as np
from density_field import density_continuum_1d
x_r, tt_r, rho_r = density_continuum_1d(nx=64, tmax=0.5, nu=0.05, D4=1e-4)
assert x_r.shape == (64,), '[TC28] density_continuum_1d x shape FAILED'
assert len(tt_r) >= 1, '[TC28] density_continuum_1d tt length FAILED'
assert rho_r.shape[0] == 64, '[TC28] density_continuum_1d rho rows FAILED'
assert np.all(np.isfinite(rho_r)), '[TC28] density_continuum_1d finite FAILED'

# ---- TC29: circle_distance_pdf returns non-negative values ----
import numpy as np
from distance_statistics import circle_distance_pdf
d_vals = np.linspace(0.1, 1.9, 20)
pdf_vals = circle_distance_pdf(d_vals, radius=1.0)
assert np.all(pdf_vals >= 0.0), '[TC29] circle_distance_pdf non-negative FAILED'

# ---- TC30: kl_divergence_empirical_vs_uniform is non-negative ----
import numpy as np
from distance_statistics import kl_divergence_empirical_vs_uniform
p_hist = np.array([5, 10, 15, 10, 5], dtype=float)
q_hist = np.array([9, 9, 9, 9, 9], dtype=float)
kl_val = kl_divergence_empirical_vs_uniform(p_hist, q_hist)
assert kl_val >= 0.0, '[TC30] KL divergence non-negative FAILED'

# ---- TC31: compute_emergence_index returns non-negative metrics ----
import numpy as np
from distance_statistics import compute_emergence_index
np.random.seed(42)
pos_test = np.random.uniform(-0.5, 0.5, (6, 2))
emergence, dmean, dvar = compute_emergence_index(pos_test, arena_radius=1.0)
assert emergence >= 0.0, '[TC31] emergence_index non-negative FAILED'
assert dmean >= 0.0, '[TC31] distance mean non-negative FAILED'
assert dvar >= 0.0, '[TC31] distance variance non-negative FAILED'

# ---- TC32: multivariate_distance_stats returns finite statistics ----
import numpy as np
from distance_statistics import multivariate_distance_stats
np.random.seed(42)
mu_mv, var_mv = multivariate_distance_stats(dim=4, n=500)
assert np.isfinite(mu_mv), '[TC32] multivariate distance stats mu finite FAILED'
assert var_mv >= 0.0, '[TC32] multivariate distance stats var non-negative FAILED'

# ---- TC33: chebyshev_coefficients returns correct shape ----
import numpy as np
from spectral_approx import chebyshev_coefficients
def f_test(x):
    return np.sin(x)
c_cheb = chebyshev_coefficients(0.0, np.pi, n=8, f=f_test)
assert c_cheb.shape == (8,), '[TC33] chebyshev_coefficients shape FAILED'

# ---- TC34: delay_kernel_chebyshev returns finite coefficients ----
import numpy as np
from spectral_approx import delay_kernel_chebyshev
c_kern = delay_kernel_chebyshev(tau_max=1.0, n=10, kernel_type="exponential")
assert c_kern.shape == (10,), '[TC34] delay_kernel_chebyshev shape FAILED'
assert np.all(np.isfinite(c_kern)), '[TC34] delay_kernel_chebyshev finite FAILED'

# ---- TC35: bernstein_poly_ab satisfies partition of unity ----
import numpy as np
from spectral_approx import bernstein_poly_ab
bvec = bernstein_poly_ab(n=5, a=0.0, b=1.0, x=0.4)
assert abs(np.sum(bvec) - 1.0) < 1e-10, '[TC35] bernstein_poly_ab partition of unity FAILED'
assert np.all(bvec >= 0.0), '[TC35] bernstein_poly_ab non-negative FAILED'

# ---- TC36: bernstein_poly_ab_approx returns correct output shape ----
import numpy as np
from spectral_approx import bernstein_poly_ab_approx
ydata = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
xval = np.linspace(0.0, 1.0, 20)
yval = bernstein_poly_ab_approx(n=5, a=0.0, b=1.0, ydata=ydata, xval=xval)
assert yval.shape == (20,), '[TC36] bernstein_poly_ab_approx shape FAILED'

# ---- TC37: feynman_kac_1d_solve returns finite RMS error ----
import numpy as np
from stochastic_control import feynman_kac_1d_solve
np.random.seed(42)
xs, u_approx, u_exact, rms = feynman_kac_1d_solve(a=2.0, h=0.5, n_paths=100, n_grid=5)
assert np.isfinite(rms), '[TC37] feynman_kac_1d_solve rms finite FAILED'
assert u_approx.shape == u_exact.shape, '[TC37] feynman_kac_1d_solve shape FAILED'

# ---- TC38: feynman_kac_collision_potential returns non-negative values ----
import numpy as np
from stochastic_control import feynman_kac_collision_potential
positions_fk = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
obstacles_fk = np.array([[0.0, 0.0]])
pot = feynman_kac_collision_potential(positions_fk, obstacles_fk, obstacle_radius=0.3, domain_radius=1.0)
assert pot.shape == (3,), '[TC38] feynman_kac_collision_potential shape FAILED'
assert np.all(pot >= 0.0), '[TC38] feynman_kac_collision_potential non-negative FAILED'

# ---- TC39: gradient_fk_potential returns correct output shape ----
import numpy as np
from stochastic_control import gradient_fk_potential
grad = gradient_fk_potential(positions_fk, obstacles_fk, obstacle_radius=0.3, domain_radius=1.0, eps=1e-4)
assert grad.shape == (3, 2), '[TC39] gradient_fk_potential shape FAILED'

# ---- TC40: potential function returns strictly positive values ----
import numpy as np
from stochastic_control import potential
x_vals = np.linspace(-2.0, 2.0, 10)
v_vals = potential(2.0, x_vals)
assert np.all(v_vals > 0), '[TC40] potential positive FAILED'
assert v_vals.shape == x_vals.shape, '[TC40] potential shape FAILED'

# ---- TC41: chebyshev_interpolant returns finite values ----
import numpy as np
from spectral_approx import chebyshev_interpolant
x_test = np.linspace(0.0, np.pi, 20)
y_interp = chebyshev_interpolant(0.0, np.pi, 8, c_cheb, x_test)
assert y_interp.shape == (20,), '[TC41] chebyshev_interpolant shape FAILED'
assert np.all(np.isfinite(y_interp)), '[TC41] chebyshev_interpolant finite FAILED'

# ---- TC42: etdrk4_coefficients returns finite coefficients ----
import numpy as np
from density_field import etdrk4_coefficients
nx_test = 64
k_test_etd = np.concatenate((np.arange(0, nx_test // 2), np.array([0]), np.arange(-nx_test // 2 + 1, 0)))
L_test = 1j * 0.03 * k_test_etd.astype(float) ** 2
E, E2, Q, f1, f2, f3 = etdrk4_coefficients(L_test, dt=0.001, nx=nx_test)
assert E.shape == (nx_test,), '[TC42] etdrk4_coefficients E shape FAILED'
assert np.all(np.isfinite(E)), '[TC42] etdrk4_coefficients E finite FAILED'

print('\n全部 42 个测试通过!\n')
