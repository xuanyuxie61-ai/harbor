# ---- TC01: MiddleSquareGenerator random output in [0,1) ----
rng = MiddleSquareGenerator(seed=5678, d=4)
r = rng.random()
assert 0.0 <= r < 1.0, '[TC01] random output in [0,1) FAILED'

# ---- TC02: MiddleSquareGenerator randn finite ----
rng = MiddleSquareGenerator(seed=1234, d=4)
z = rng.randn()
assert np.isfinite(z), '[TC02] randn finite FAILED'

# ---- TC03: MiddleSquareGenerator uniform range ----
rng = MiddleSquareGenerator(seed=4321, d=4)
u = rng.uniform(low=2.0, high=5.0)
assert 2.0 <= u < 5.0, '[TC03] uniform range FAILED'

# ---- TC04: MiddleSquareGenerator lognormal positive ----
rng = MiddleSquareGenerator(seed=1111, d=4)
val = rng.lognormal(mu=0.0, sigma=0.5)
assert val > 0.0, '[TC04] lognormal positive FAILED'

# ---- TC05: FractureNetwork generate_ifs_fractures shape ----
np.random.seed(42)
network = FractureNetwork(domain_size=(10.0, 10.0), nx=10, ny=10, seed=42)
pts = network.generate_ifs_fractures(n_iterations=100, ifs_type="cross")
assert pts.shape == (2, 100), '[TC05] IFS fracture points shape FAILED'

# ---- TC06: FractureNetwork equivalent_permeability non-negative ----
network = FractureNetwork(domain_size=(10.0, 10.0), nx=10, ny=10, seed=42)
network.aperture = np.ones((10, 10)) * 1.0e-4
k_eq = network.equivalent_permeability()
assert k_eq >= 0.0, '[TC06] equivalent_permeability non-negative FAILED'

# ---- TC07: FractureNetwork tortuosity >= 1.0 ----
network = FractureNetwork(domain_size=(10.0, 10.0), nx=10, ny=10, seed=42)
tau = network.tortuosity([(0, 0), (1, 1), (2, 2)])
assert tau >= 1.0, '[TC07] tortuosity >= 1.0 FAILED'

# ---- TC08: OBJGeometryParser parse_string counts correct ----
parser = OBJGeometryParser()
obj_text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"
geo = parser.parse_string(obj_text)
assert geo['n_vertices'] == 3 and geo['n_faces'] == 1, '[TC08] parse_string counts FAILED'

# ---- TC09: OBJGeometryParser total_surface_area positive ----
parser = OBJGeometryParser()
obj_text = parser.generate_sample_fracture_obj(size=1.0, amplitude=0.01, n_segments=5)
parser.parse_string(obj_text)
area = parser.total_surface_area()
assert area > 0.0, '[TC09] total_surface_area positive FAILED'

# ---- TC10: OBJGeometryParser roughness_coefficient non-negative ----
parser = OBJGeometryParser()
obj_text = parser.generate_sample_fracture_obj(size=1.0, amplitude=0.01, n_segments=5)
parser.parse_string(obj_text)
sigma_z = parser.roughness_coefficient()
assert sigma_z >= 0.0, '[TC10] roughness_coefficient non-negative FAILED'

# ---- TC11: MeshGenerator generate_uniform_grid shape ----
mesh = MeshGenerator(domain=(0.0, 10.0, 0.0, 10.0))
pts = mesh.generate_uniform_grid(nx=5, ny=3)
assert pts.shape == (15, 2), '[TC11] uniform grid shape FAILED'

# ---- TC12: MeshGenerator cvt_relaxation reproducible ----
np.random.seed(42)
mesh1 = MeshGenerator(domain=(0.0, 10.0, 0.0, 10.0))
p1 = mesh1.cvt_relaxation(n_points=20, n_iterations=3, n_samples=500, seed=42)
np.random.seed(42)
mesh2 = MeshGenerator(domain=(0.0, 10.0, 0.0, 10.0))
p2 = mesh2.cvt_relaxation(n_points=20, n_iterations=3, n_samples=500, seed=42)
assert np.allclose(p1, p2), '[TC12] CVT relaxation reproducible FAILED'

# ---- TC13: SincInterpolator sincn(0) == 1 ----
val = SincInterpolator.sincn(0.0)
assert abs(val - 1.0) < 1e-12, '[TC13] sincn(0) == 1 FAILED'

# ---- TC14: SincInterpolator sincn_derivative(0) == 0 ----
val = SincInterpolator.sincn_derivative(0.0)
assert abs(val) < 1e-12, '[TC14] sincn_derivative(0) == 0 FAILED'

# ---- TC15: SincInterpolator interpolate_1d Gaussian accuracy ----
x_grid = np.linspace(-3, 3, 31)
f_vals = np.exp(-x_grid ** 2)
x_query = np.array([0.0])
f_interp = SincInterpolator.interpolate_1d(x_grid, f_vals, x_query)
assert abs(f_interp[0] - 1.0) < 1e-6, '[TC15] interpolate_1d Gaussian FAILED'

# ---- TC16: HydraulicSolver solve_gauss_seidel boundary respected ----
T = np.ones((5, 5)) * 1.0e-8
solver = HydraulicSolver(nx=5, ny=5, dx=1.0, dy=1.0)
h = solver.solve_gauss_seidel(T, h_boundary={'left': 5.0, 'right': 1.0, 'top': 3.0, 'bottom': 3.0}, max_iter=100, tol=1e-6, omega=1.0)
assert abs(h[1:-1, 0].mean() - 5.0) < 1e-6 and abs(h[1:-1, -1].mean() - 1.0) < 1e-6, '[TC16] GS boundary respected FAILED'

# ---- TC17: GegenbauerQuadrature integrate constant exact ----
quad = GegenbauerQuadrature(order=8, alpha=0.0, a=0.0, b=2.0)
result = quad.integrate(lambda x: np.ones_like(x) * 3.0)
assert abs(result - 6.0) < 1e-10, '[TC17] integrate constant FAILED'

# ---- TC18: FlowIntegrator breakthrough_curve_moments M0 positive ----
times = np.linspace(0, 10, 50)
C = np.exp(-0.5 * ((times - 5.0) / 1.0) ** 2)
moments = FlowIntegrator.breakthrough_curve_moments(times, C)
assert moments['M0'] > 0.0, '[TC18] breakthrough M0 positive FAILED'

# ---- TC19: FlowIntegrator peclet_number positive ----
Pe = FlowIntegrator.peclet_number(v=0.01, L=10.0, D=1.0e-9)
assert Pe > 0.0, '[TC19] Peclet number positive FAILED'

# ---- TC20: TransportSolver stability_check structure ----
transport = TransportSolver(nx=10, ny=10, dx=1.0, dy=1.0, dt=0.1, R=1.0, lambda_decay=0.0)
vx = np.ones((10, 10)) * 0.01
vy = np.ones((10, 10)) * 0.01
transport.set_velocity_field(vx, vy)
transport.set_dispersivity(alpha_L=0.01, alpha_T=0.001, D_m=1.0e-9)
stab = transport.stability_check()
assert 'stable' in stab and stab['stable'] in (True, False), '[TC20] stability check structure FAILED'

# ---- TC21: TransportSolver solve final mass non-negative ----
transport = TransportSolver(nx=10, ny=10, dx=1.0, dy=1.0, dt=0.01, R=1.0, lambda_decay=0.0)
vx = np.ones((10, 10)) * 0.001
vy = np.ones((10, 10)) * 0.001
transport.set_velocity_field(vx, vy)
transport.set_dispersivity(alpha_L=0.001, alpha_T=0.0001, D_m=1.0e-9)
result = transport.solve(n_steps=10, injection_zone=(4, 6, 1, 3), C_inject=1.0, check_mass=True)
assert result['final_mass'] >= 0.0, '[TC21] solve final mass non-negative FAILED'

# ---- TC22: InverseModel regula_falsi finds root ----
def f_test(x):
    return x - 2.5
root, it = InverseModel.regula_falsi(f_test, 0.0, 5.0)
assert abs(root - 2.5) < 1e-6, '[TC22] regula_falsi root FAILED'

# ---- TC23: InverseModel dirichlet_estimate_moments positive alpha ----
from scipy.stats import dirichlet
np.random.seed(42)
samples = dirichlet.rvs([2.0, 3.0, 5.0], size=200)
est = InverseModel.dirichlet_estimate_moments(samples)
assert np.all(est['alpha'] > 0), '[TC23] Dirichlet alpha positive FAILED'

# ---- TC24: UncertaintyQuantification gamma_sample_stats mean correct ----
stats = UncertaintyQuantification.gamma_sample_stats(alpha=4.0, beta_param=2.0)
assert abs(stats['mean'] - 2.0) < 1e-10, '[TC24] gamma mean FAILED'

# ---- TC25: UncertaintyQuantification noncentral_beta_cdf endpoints ----
uq = UncertaintyQuantification()
cdf0 = uq.noncentral_beta_cdf(0.0, 2.0, 3.0, 1.5)
cdf1 = uq.noncentral_beta_cdf(1.0, 2.0, 3.0, 1.5)
assert abs(cdf0) < 1e-10 and abs(cdf1 - 1.0) < 1e-10, '[TC25] beta_cdf endpoints FAILED'

# ---- TC26: UncertaintyQuantification sensitivity_analysis structure ----
def fm(params):
    return params['a'] * params['b']
sens = UncertaintyQuantification.sensitivity_analysis(fm, {'a': 2.0, 'b': 3.0}, perturbation=0.01)
assert 'a' in sens and 'b' in sens, '[TC26] sensitivity structure FAILED'

# ---- TC27: SincInterpolator sincn symmetric ----
x = np.array([0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
vals = SincInterpolator.sincn(x)
assert np.allclose(vals[0::2], vals[1::2]), '[TC27] sincn symmetric FAILED'

# ---- TC28: MeshGenerator delaunay_triangulation valid triangles ----
np.random.seed(42)
mesh = MeshGenerator(domain=(0.0, 10.0, 0.0, 10.0))
mesh.generate_random_points(n_points=30, seed=42)
tri = mesh.delaunay_triangulation()
assert tri.shape[1] == 3 and tri.min() >= 0 and tri.max() < len(mesh.points), '[TC28] delaunay valid FAILED'

# ---- TC29: FractureNetwork check_percolation returns bool and path ----
network = FractureNetwork(domain_size=(10.0, 10.0), nx=10, ny=10, seed=42)
network.connectivity = np.zeros((10, 10), dtype=bool)
for idx in range(10):
    network.connectivity[idx, 5] = True
percolates, path = network.check_percolation()
assert isinstance(percolates, bool) and len(path) > 0, '[TC29] percolation bool and path FAILED'

# ---- TC30: Integration test main returns 0 ----
result = main()
assert result == 0, '[TC30] main() return code FAILED'

