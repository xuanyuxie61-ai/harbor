# ---- TC01: NonlinearAcousticsPhysics water medium parameters ----
physics_water = NonlinearAcousticsPhysics(medium='water', f0=1e6, p0=1e5, geometry='planar')
assert physics_water.c0 == 1500.0, '[TC01] NonlinearAcousticsPhysics water medium parameters FAILED'
assert physics_water.rho0 == 1000.0, '[TC01] NonlinearAcousticsPhysics water medium parameters FAILED'
assert physics_water.beta == 3.5, '[TC01] NonlinearAcousticsPhysics water medium parameters FAILED'

# ---- TC02: NonlinearAcousticsPhysics air medium parameters ----
physics_air = NonlinearAcousticsPhysics(medium='air', f0=1e3, p0=1e3, geometry='spherical')
assert physics_air.c0 == 343.0, '[TC02] NonlinearAcousticsPhysics air medium parameters FAILED'
assert physics_air.rho0 == 1.21, '[TC02] NonlinearAcousticsPhysics air medium parameters FAILED'
assert physics_air.beta == 1.2, '[TC02] NonlinearAcousticsPhysics air medium parameters FAILED'

# ---- TC03: Shock formation distance formula consistency ----
xs_expected = 1.0 / (physics_water.beta * physics_water.k0 * physics_water.M0)
assert abs(physics_water.shock_formation_distance - xs_expected) < 1e-10, '[TC03] Shock formation distance formula consistency FAILED'

# ---- TC04: Goldberg number formula consistency ----
ng_expected = 1.0 / (physics_water.classical_absorption * physics_water.shock_formation_distance)
assert abs(physics_water.Goldberg_number - ng_expected) < 1e-10, '[TC04] Goldberg number formula consistency FAILED'

# ---- TC05: hex_grid_points output shape correctness ----
box = np.array([[0.0, 1.0], [0.0, 1.0]])
pts = hex_grid_points(nodes_per_layer=4, layers=3, box=box)
assert pts.ndim == 2 and pts.shape[1] == 2, '[TC05] hex_grid_points output shape correctness FAILED'
assert pts.shape[0] > 0, '[TC05] hex_grid_points output shape correctness FAILED'

# ---- TC06: hex_grid_approximate_n matches actual count ----
n_est = hex_grid_approximate_n(nodes_per_layer=4, layers=3)
assert n_est == pts.shape[0], '[TC06] hex_grid_approximate_n matches actual count FAILED'

# ---- TC07: triangle_area for right triangle ----
tri_rt = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
area_rt = triangle_area(tri_rt)
assert abs(area_rt - 6.0) < 1e-10, '[TC07] triangle_area for right triangle FAILED'

# ---- TC08: triangle_angles sum to pi ----
tri_gen = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
angles = triangle_angles(tri_gen)
assert abs(np.sum(angles) - np.pi) < 1e-10, '[TC08] triangle_angles sum to pi FAILED'

# ---- TC09: triangle_quality equilateral equals 1 ----
tri_eq = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]])
q_eq = triangle_quality(tri_eq)
assert abs(q_eq - 1.0) < 1e-10, '[TC09] triangle_quality equilateral equals 1 FAILED'

# ---- TC10: circumcircle radius >= incircle radius ----
R, _ = triangle_circumcircle(tri_eq)
r, _ = triangle_incircle(tri_eq)
assert R >= r - 1e-10, '[TC10] circumcircle radius >= incircle radius FAILED'

# ---- TC11: naca4_symmetric zero at leading edge and positive thickness ----
from geometry_utils import naca4_symmetric
y_le = naca4_symmetric(t=0.12, c=1.0, x=0.0)
y_mid = naca4_symmetric(t=0.12, c=1.0, x=0.5)
assert abs(y_le) < 1e-10, '[TC11] naca4_symmetric zero at leading edge and positive thickness FAILED'
assert y_mid > 0.0, '[TC11] naca4_symmetric zero at leading edge and positive thickness FAILED'

# ---- TC12: generate_naca_airfoil_points surface shape ----
surf = generate_naca_airfoil_points(t=0.12, c=0.05, n_points=50)
assert surf.ndim == 2 and surf.shape[1] == 2, '[TC12] generate_naca_airfoil_points surface shape FAILED'
assert surf.shape[0] == 2 * 50 - 1, '[TC12] generate_naca_airfoil_points surface shape FAILED'

# ---- TC13: VandermondeSolver determinant and apply_mv accuracy ----
nodes_v = np.array([1.0, 2.0, 3.0, 4.0])
vand = VandermondeSolver(nodes_v)
V_dense = vand.to_dense()
det_v = vand.determinant()
det_np = np.linalg.det(V_dense)
assert abs(det_v - det_np) < 1e-10, '[TC13] VandermondeSolver determinant and apply_mv accuracy FAILED'
v_poly = np.array([1.0, 0.0, 0.0, 0.0])
y_mv = vand.apply_mv(v_poly)
assert np.allclose(y_mv, np.ones(4)), '[TC13] VandermondeSolver determinant and apply_mv accuracy FAILED'

# ---- TC14: SpectralDifferentiator constant derivative is zero ----
from spectral_solver import SpectralDifferentiator
spec = SpectralDifferentiator(n=8, node_type='chebyshev_gauss_lobatto')
du = spec.differentiate(np.ones(8))
assert np.linalg.norm(du) < 1e-10, '[TC14] SpectralDifferentiator constant derivative is zero FAILED'

# ---- TC15: map_nodes_to_interval range and jacobian ----
from spectral_solver import map_nodes_to_interval
nodes_m = np.array([-1.0, 0.0, 1.0])
xm, jac = map_nodes_to_interval(nodes_m, 0.0, 2.0)
assert np.allclose(xm, np.array([0.0, 1.0, 2.0])), '[TC15] map_nodes_to_interval range and jacobian FAILED'
assert abs(jac - 1.0) < 1e-10, '[TC15] map_nodes_to_interval range and jacobian FAILED'

# ---- TC16: matrix_chain_optimal_order cost positive ----
dims = [10, 20, 30, 40]
cost, s = matrix_chain_optimal_order(dims)
assert cost > 0, '[TC16] matrix_chain_optimal_order cost positive FAILED'

# ---- TC17: apply_optimal_matrix_chain equals naive product ----
from matrix_chain_optimizer import apply_optimal_matrix_chain
np.random.seed(42)
A1 = np.random.randn(10, 20)
A2 = np.random.randn(20, 30)
A3 = np.random.randn(30, 40)
matrices = [A1, A2, A3]
cost, s = matrix_chain_optimal_order([10, 20, 30, 40])
result_opt = apply_optimal_matrix_chain(matrices, s)
result_naive = A1 @ A2 @ A3
assert np.linalg.norm(result_opt - result_naive, 'fro') < 1e-10, '[TC17] apply_optimal_matrix_chain equals naive product FAILED'

# ---- TC18: SVDRomCompressor cumulative energy monotonic ----
np.random.seed(42)
snap = np.random.randn(20, 10)
comp = SVDRomCompressor(snap)
comp.decompose()
cum = comp.cumulative_energy()
assert np.all(np.diff(cum) >= -1e-14), '[TC18] SVDRomCompressor cumulative energy monotonic FAILED'
assert cum[-1] >= 0.999999, '[TC18] SVDRomCompressor cumulative energy monotonic FAILED'

# ---- TC19: SVDRomCompressor error decreases with rank ----
_, err_1, _ = comp.low_rank_approximation(1)
_, err_3, _ = comp.low_rank_approximation(3)
_, err_5, _ = comp.low_rank_approximation(5)
assert err_1 >= err_3 - 1e-14, '[TC19] SVDRomCompressor error decreases with rank FAILED'
assert err_3 >= err_5 - 1e-14, '[TC19] SVDRomCompressor error decreases with rank FAILED'

# ---- TC20: monte_carlo_nd on constant function ----
np.random.seed(42)
def f_const(x):
    return 1.0
res_mc, _, _ = monte_carlo_nd(f_const, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3, 5000)
assert abs(res_mc - 1.0) < 0.05, '[TC20] monte_carlo_nd on constant function FAILED'

# ---- TC21: romberg_nd on constant function ----
def f_const2(x):
    return 2.0
res_rom, ind_rom, _ = romberg_nd(f_const2, [0.0, 0.0], [1.0, 1.0], 2, np.array([2, 2]), it_max=4, tol=1e-2)
assert ind_rom == 1, '[TC21] romberg_nd on constant function FAILED'
assert abs(res_rom - 2.0) < 0.01, '[TC21] romberg_nd on constant function FAILED'

# ---- TC22: CliffGenerator reproducibility with fixed seed ----
cliff1 = CliffGenerator(seed=0.2718281828)
cliff2 = CliffGenerator(seed=0.2718281828)
seq1 = np.array([cliff1.next() for _ in range(5)])
seq2 = np.array([cliff2.next() for _ in range(5)])
assert np.allclose(seq1, seq2), '[TC22] CliffGenerator reproducibility with fixed seed FAILED'

# ---- TC23: latin_hypercube_sampling shape and bounds ----
np.random.seed(42)
lhs = latin_hypercube_sampling(20, 2, a=0.0, b=1.0)
assert lhs.shape == (20, 2), '[TC23] latin_hypercube_sampling shape and bounds FAILED'
assert np.all(lhs >= 0.0) and np.all(lhs <= 1.0), '[TC23] latin_hypercube_sampling shape and bounds FAILED'

# ---- TC24: mesh_topology_checksum reproducibility ----
from luhn_checksum_adapter import mesh_topology_checksum
nodes_ck = np.array([[0, 0], [0.05, 0], [0.05, 0.05], [0, 0.05]], dtype=float)
tris_ck = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
cs1, cd1 = mesh_topology_checksum(tris_ck, nodes_ck)
cs2, cd2 = mesh_topology_checksum(tris_ck, nodes_ck)
assert cs1 == cs2 and cd1 == cd2, '[TC24] mesh_topology_checksum reproducibility FAILED'

# ---- TC25: NumericalIntegrityChecker verify same array passes ----
checker = NumericalIntegrityChecker(scale=1e6)
arr_test = np.array([[1.0, 2.0], [3.0, 4.0]])
checker.checkpoint("test_arr", arr_test)
passed, details = checker.verify("test_arr", arr_test)
assert passed is True, '[TC25] NumericalIntegrityChecker verify same array passes FAILED'
assert details['shape_match'] is True, '[TC25] NumericalIntegrityChecker verify same array passes FAILED'
assert details['checksum_match'] is True, '[TC25] NumericalIntegrityChecker verify same array passes FAILED'
