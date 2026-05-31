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
