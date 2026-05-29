
# ---- TC01: FaultRuptureDynamics friction coefficient returns finite scalar ----
fault = FaultRuptureDynamics()
mu_val = fault.friction_coefficient(1e-6, 1.0)
assert np.isfinite(mu_val), '[TC01] Fault friction coefficient finite FAILED'

# ---- TC02: FaultRuptureDynamics solve_rupture_ode returns monotonic time and correct shape ----
fault = FaultRuptureDynamics()
t_rupt, y_rupt = fault.solve_rupture_ode((0.0, 10.0), np.array([1.0, 1.0, 0.0, 1.0]), n_steps=100)
assert np.all(np.diff(t_rupt) > 0), '[TC02] Rupture ODE time monotonic FAILED'
assert y_rupt.shape == (101, 4), '[TC02] Rupture ODE solution shape FAILED'

# ---- TC03: PendulumConservationMonitor energy deviation is non-negative ----
monitor = PendulumConservationMonitor()
theta = np.linspace(0, np.pi / 4, 50)
omega = np.sin(theta)
dev = monitor.check_energy_conservation(theta, omega)
assert dev >= 0.0, '[TC03] Pendulum energy deviation non-negative FAILED'

# ---- TC04: OkadaModel seafloor displacement has correct shape and finite values ----
okada = OkadaModel(strike=0.0, dip=45.0, rake=90.0, slip=1.0, length=100e3, width=50e3, depth=10e3, nu=0.25)
x_g = np.linspace(-200e3, 200e3, 21)
y_g = np.linspace(-200e3, 200e3, 21)
eta = okada.compute_seafloor_displacement(x_g, y_g)
assert eta.shape == (21, 21), '[TC04] Okada displacement shape FAILED'
assert np.all(np.isfinite(eta)), '[TC04] Okada displacement finite FAILED'

# ---- TC05: OkadaModel init raises ValueError for invalid dip ----
try:
    OkadaModel(strike=0.0, dip=-10.0, rake=0.0, slip=1.0, length=100e3, width=50e3, depth=10e3)
    assert False, '[TC05] Okada invalid dip should raise FAILED'
except ValueError:
    pass

# ---- TC06: SphericalGeodesics haversine distance is symmetric ----
geo = SphericalGeodesics()
d1 = geo.haversine_distance(0.0, 0.0, 10.0, 10.0)
d2 = geo.haversine_distance(10.0, 10.0, 0.0, 0.0)
assert abs(d1 - d2) < 1e-6, '[TC06] Haversine symmetry FAILED'

# ---- TC07: SphericalGeodesics haversine distance at same point is near zero ----
geo = SphericalGeodesics()
d = geo.haversine_distance(35.0, 140.0, 35.0, 140.0)
assert abs(d) < 1e-3, '[TC07] Haversine same point FAILED'

# ---- TC08: SphericalGeodesics compute_grid_distances shape and non-negative ----
geo = SphericalGeodesics()
x_g = np.linspace(-100e3, 100e3, 11)
y_g = np.linspace(-100e3, 100e3, 11)
dist = geo.compute_grid_distances(x_g, y_g, 38.0, 142.0)
assert dist.shape == (11, 11), '[TC08] Grid distances shape FAILED'
assert np.all(dist >= 0), '[TC08] Grid distances non-negative FAILED'

# ---- TC09: SphericalGeodesics compute_travel_time finite and non-negative ----
geo = SphericalGeodesics()
dist = np.array([[1e3, 2e3], [3e3, 4e3]])
depth = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])
tt = geo.compute_travel_time(dist, depth)
assert np.all(np.isfinite(tt)), '[TC09] Travel time finite FAILED'
assert np.all(tt >= 0), '[TC09] Travel time non-negative FAILED'

# ---- TC10: BathymetryGenerator random bathymetry returns positive depths ----
np.random.seed(42)
x_g = np.linspace(-100e3, 100e3, 11)
y_g = np.linspace(-100e3, 100e3, 11)
bath = BathymetryGenerator(x_g, y_g)
h = bath.generate_random_bathymetry(depth_mean=3000.0, depth_std=500.0, continental_slope=False)
assert h.shape == (11, 11), '[TC10] Bathymetry shape FAILED'
assert np.all(h > 0), '[TC10] Bathymetry positive FAILED'

# ---- TC11: BathymetryGenerator rcont constraints produce correct shape and positive values ----
np.random.seed(42)
bath = BathymetryGenerator(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
rows = np.array([10, 10, 10, 10, 10])
cols = np.array([10, 10, 10, 10, 10])
h = bath.generate_bathymetry_with_rcont_constraints(rows, cols)
assert h.shape == (5, 5), '[TC11] Rcont shape FAILED'
assert np.all(h > 0), '[TC11] Rcont positive FAILED'

# ---- TC12: MeshInterpolator bilinear interpolation preserves constant field exactly ----
x_c = np.linspace(0, 1, 5)
y_c = np.linspace(0, 1, 5)
z_c = np.ones((5, 5)) * 3.14
interp = MeshInterpolator(x_c, y_c)
x_f = np.linspace(0, 1, 9)
y_f = np.linspace(0, 1, 9)
z_f = interp.bilinear_interpolate(z_c, x_f, y_f)
assert np.allclose(z_f, 3.14, atol=1e-10), '[TC12] Bilinear constant preservation FAILED'

# ---- TC13: MeshInterpolator trigonometric periodic boundary reduces discontinuity ----
np.random.seed(42)
field = np.random.rand(10, 10)
interp = MeshInterpolator(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
fp = interp.trigonometric_periodic_boundary(field, axis=1)
diff_after = np.max(np.abs(fp[:, 0] - fp[:, -1]))
assert diff_after < 1e-10, '[TC13] Periodic boundary discontinuity FAILED'

# ---- TC14: HilbertMeshOrderer h_to_xy and xy_to_h are inverses ----
hmo = HilbertMeshOrderer(order=4)
h = 42
x, y = hmo.h_to_xy(h)
h_back = hmo.xy_to_h(x, y)
assert h == h_back, '[TC14] Hilbert h_to_xy inverse FAILED'

# ---- TC15: HilbertMeshOrderer locality better than row-major ----
hmo = HilbertMeshOrderer(order=4)
loc_h, loc_r = hmo.compare_orderings(8, 8)
assert loc_h <= loc_r, '[TC15] Hilbert locality not better than row-major FAILED'

# ---- TC16: EnergyQuadrature square monomial integral matches analytical formula ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
assert abs(eq.square_monomial_integral((2, 3)) - 1.0 / 12.0) < 1e-12, '[TC16] Square monomial FAILED'

# ---- TC17: EnergyQuadrature Hermite odd degree exact value is zero ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
assert eq.hermite_integral_exact(3) == 0.0, '[TC17] Hermite odd degree FAILED'

# ---- TC18: EnergyQuadrature triangle symmetric quadrature on unit function equals 0.5 ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
val = eq.triangle_symmetric_quadrature_test()
assert abs(val - 0.5) < 1e-10, '[TC18] Triangle quadrature unit function FAILED'

# ---- TC19: EnergyQuadrature total energy is non-negative with zero velocity ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
eta = np.ones((5, 5))
u = np.zeros((5, 5))
v = np.zeros((5, 5))
h_bathy = np.ones((5, 5)) * 10.0
E = eq.compute_total_energy(eta, u, v, h_bathy)
assert E >= 0.0, '[TC19] Total energy non-negative FAILED'

# ---- TC20: AdaptiveMeshCover generate_adaptive_cover returns correct boolean mask ----
amc = AdaptiveMeshCover()
np.random.seed(42)
field = np.random.rand(16, 16)
mask = amc.generate_adaptive_cover(field, threshold=0.1, max_level=2)
assert mask.shape == (16, 16), '[TC20] Adaptive cover shape FAILED'
assert mask.dtype == bool, '[TC20] Adaptive cover dtype FAILED'

# ---- TC21: AdaptiveMeshCover variomino transformations count does not exceed 4 ----
amc = AdaptiveMeshCover()
tile = np.array([[1, 0], [0, 0]])
variants = amc.variomino_transformations(tile)
assert len(variants) <= 4, '[TC21] Variomino transformations count FAILED'

# ---- TC22: MatrixChainOptimizer Catalan numbers match known values ----
mco = MatrixChainOptimizer()
assert mco.catalan_number(0) == 1, '[TC22] Catalan n=0 FAILED'
assert mco.catalan_number(1) == 1, '[TC22] Catalan n=1 FAILED'
assert mco.catalan_number(2) == 2, '[TC22] Catalan n=2 FAILED'
assert mco.catalan_number(3) == 5, '[TC22] Catalan n=3 FAILED'

# ---- TC23: MatrixChainOptimizer optimal chain cost is not worse than naive ----
mco = MatrixChainOptimizer()
dims = [10, 100, 10, 100]
opt_cost, _ = mco.find_optimal_chain(dims)
naive_cost = mco.pivot_sequence_to_cost(3, [1, 0], dims)
assert opt_cost <= naive_cost, '[TC23] Optimal chain cost FAILED'

# ---- TC24: MatrixChainOptimizer optimal_matrix_power M^1 equals M ----
mco = MatrixChainOptimizer()
M = np.array([[2.0, 1.0], [0.0, 1.0]])
Mp, n_mul = mco.optimal_matrix_power(M, 1)
assert np.allclose(Mp, M), '[TC24] Matrix power M^1 FAILED'
assert n_mul == 0, '[TC24] Matrix power M^1 multiply count FAILED'

# ---- TC25: ParallelScheduler divide_tasks covers all tasks exactly ----
sched = ParallelScheduler(n_tasks=100, n_processors=4)
task_map = sched.divide_tasks()
total = sum(end - start + 1 for start, end in task_map.values())
assert total == 100, '[TC25] Task coverage FAILED'

# ---- TC26: ParallelScheduler load balance imbalance is small for uniform division ----
sched = ParallelScheduler(n_tasks=100, n_processors=4)
task_map = sched.divide_tasks()
imbalance, max_load, min_load = sched.compute_load_balance(task_map)
assert imbalance < 0.1, '[TC26] Load balance imbalance FAILED'

# ---- TC27: ShallowWaterSolver set_initial_condition preserves field shape ----
x_g = np.linspace(0, 1e3, 11)
y_g = np.linspace(0, 1e3, 11)
h_bathy = np.ones((11, 11)) * 100.0
solver = ShallowWaterSolver(x_g, y_g, h_bathy, dt=10.0, n_steps=5)
eta0 = np.zeros((11, 11))
solver.set_initial_condition(eta0)
assert solver.eta.shape == (11, 11), '[TC27] Solver IC shape FAILED'

# ---- TC28: ShallowWaterSolver solve returns consistent snapshot lists ----
x_g = np.linspace(0, 1e3, 11)
y_g = np.linspace(0, 1e3, 11)
h_bathy = np.ones((11, 11)) * 100.0
solver = ShallowWaterSolver(x_g, y_g, h_bathy, dt=10.0, n_steps=5)
eta0 = np.zeros((11, 11))
eta0[5, 5] = 1.0
solver.set_initial_condition(eta0)
t_snap, eta_snap, u_snap, v_snap = solver.solve()
assert len(t_snap) == len(eta_snap) == len(u_snap) == len(v_snap), '[TC28] Solver snapshot lengths FAILED'
assert eta_snap[0].shape == (11, 11), '[TC28] Solver snapshot shape FAILED'

# ---- TC29: SphericalGeodesics geographic-to-cartesian roundtrip ----
geo = SphericalGeodesics()
lat, lon = 35.0, 140.0
xyz = geo.geographic_to_cartesian(lat, lon)
lat_back, lon_back = geo.cartesian_to_geographic(xyz[0], xyz[1], xyz[2])
assert abs(lat_back - lat) < 1e-6, '[TC29] Geographic cartesian roundtrip lat FAILED'
assert abs(lon_back - lon) < 1e-6, '[TC29] Geographic cartesian roundtrip lon FAILED'

# ---- TC30: EnergyQuadrature Gauss-Hermite quadrature exact for even polynomial ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
exact = eq.hermite_integral_exact(4, option=1)
numerical = eq.gauss_hermite_quadrature(lambda x: x ** 4, n_points=10)
assert abs(numerical - exact) / abs(exact) < 1e-10, '[TC30] Gauss-Hermite even polynomial FAILED'

# ---- TC31: MeshInterpolator cardinal basis at node equals 1 ----
interp = MeshInterpolator(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
x_nodes = np.linspace(0, 1, 5)
Cj = interp.cardinal_basis(x_nodes, x_nodes[2], 2)
assert abs(Cj - 1.0) < 1e-10, '[TC31] Cardinal basis at node FAILED'

# ---- TC32: FaultRuptureDynamics init raises ValueError for invalid friction parameter a ----
try:
    FaultRuptureDynamics(a=-0.01)
    assert False, '[TC32] Fault invalid a should raise FAILED'
except ValueError:
    pass
