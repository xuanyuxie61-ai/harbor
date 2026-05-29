# ---- TC01: Material params validate and alpha1 finite ----
params = MultiferroicMaterialParams(temperature=300.0)
params.validate()
assert np.isfinite(params.alpha1), '[TC01] Material params alpha1 finite FAILED'

# ---- TC02: Hermite He_0 equals 1 ----
x = np.array([0.0, 1.0, 2.0])
He0 = hermite_probabilist(0, x)
assert np.allclose(He0, np.ones_like(x)), '[TC02] Hermite He_0 equals 1 FAILED'

# ---- TC03: Hermite He_1 equals x ----
x = np.array([-1.5, 0.0, 3.0])
He1 = hermite_probabilist(1, x)
assert np.allclose(He1, x), '[TC03] Hermite He_1 equals x FAILED'

# ---- TC04: Hermite He_2(0) equals -1 ----
x = np.array([0.0])
He2 = hermite_probabilist(2, x)
assert np.isclose(He2[0], -1.0), '[TC04] Hermite He_2(0) equals -1 FAILED'

# ---- TC05: Landau free energy density finite for zero gradients ----
params = MultiferroicMaterialParams(temperature=300.0)
P = np.array([0.1, 0.0])
M = np.array([0.0, 10.0])
dP = np.array([0.0, 0.0])
f_val = landau_free_energy_density(P, M, dP, dP, dP, dP, params)
assert np.isfinite(f_val), '[TC05] Landau free energy finite FAILED'

# ---- TC06: Thermal fluctuation correction finite ----
params = MultiferroicMaterialParams(temperature=300.0)
P = np.array([0.1, 0.05])
M = np.array([0.02, 10.0])
delta_f = thermal_fluctuation_correction(P, M, params)
assert np.isfinite(delta_f), '[TC06] Thermal fluctuation correction finite FAILED'

# ---- TC07: Mesh node count correct for small grid ----
mesh = MultiferroicMesh(nx=3, ny=3)
expected_nodes = (2*3 - 1) * (2*3 - 1)
assert mesh.node_num == expected_nodes, '[TC07] Mesh node count correct FAILED'

# ---- TC08: Mesh element count correct ----
mesh = MultiferroicMesh(nx=3, ny=3)
expected_elems = 2 * (3 - 1) * (3 - 1)
assert mesh.element_num == expected_elems, '[TC08] Mesh element count correct FAILED'

# ---- TC09: Mesh element area positive ----
mesh = MultiferroicMesh(nx=3, ny=3)
area = mesh.element_area(0)
assert area > 0, '[TC09] Mesh element area positive FAILED'

# ---- TC10: SparseMatrixCOO nnz matches data length ----
coo = SparseMatrixCOO(4, 4, row=np.array([0,1,2]), col=np.array([0,1,2]), data=np.array([1.0,2.0,3.0]))
assert coo.nnz() == 3, '[TC10] SparseMatrixCOO nnz matches FAILED'

# ---- TC11: SparseMatrixCOO write and read round-trip ----
coo = SparseMatrixCOO(3, 3, row=np.array([0,1,2]), col=np.array([0,1,2]), data=np.array([1.0,2.0,3.0]))
coo.write_to_triad_file('test_temp.tri')
coo2 = SparseMatrixCOO.read_from_triad_file('test_temp.tri')
assert coo2.nnz() == 3 and np.isclose(np.sum(coo2.data), 6.0), '[TC11] SparseMatrixCOO write/read round-trip FAILED'
os.remove('test_temp.tri')

# ---- TC12: coo_to_dense_solve identity-like system ----
coo = SparseMatrixCOO(3, 3, row=np.array([0,1,2]), col=np.array([0,1,2]), data=np.array([2.0,2.0,2.0]))
b = np.array([4.0, 6.0, 8.0])
x = coo_to_dense_solve(coo, b)
assert np.allclose(x, np.array([2.0, 3.0, 4.0])), '[TC12] coo_to_dense_solve identity-like FAILED'

# ---- TC13: Hilbert sort preserves all points ----
points = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.2, 0.8]])
order = hilbert_sort_points(points, m=4)
assert len(order) == len(points) and set(order) == set(range(len(points))), '[TC13] Hilbert sort preserves all points FAILED'

# ---- TC14: Hilbert reordering preserves node count ----
mesh = MultiferroicMesh(nx=3, ny=3)
new_xy, new_elem, mapping = apply_hilbert_reordering(mesh.node_xy.copy(), mesh.element_node.copy(), m=3)
assert len(new_xy) == mesh.node_num and len(mapping) == mesh.node_num, '[TC14] Hilbert reordering preserves count FAILED'

# ---- TC15: Pyramid unit volume exact 8/3 ----
vol = pyramid_unit_volume()
assert np.isclose(vol, 8.0/3.0), '[TC15] Pyramid unit volume exact FAILED'

# ---- TC16: Pyramid integrate constant 1 equals volume ----
vol_num = integrate_over_pyramid(lambda x,y,z: 1.0, p=4)
assert np.isclose(vol_num, pyramid_unit_volume(), rtol=1e-10), '[TC16] Pyramid integrate constant FAILED'

# ---- TC17: Pyramid rule weight sum equals volume ----
n_pts, xq, yq, zq, wq = pyramid_jaskowiec_rule(p=3)
assert np.isclose(np.sum(wq), pyramid_unit_volume(), rtol=1e-10), '[TC17] Pyramid weight sum equals volume FAILED'

# ---- TC18: FTCS dt_max positive ----
solver = ReactionDiffusionFTCS(nx=11, ny=11, Lx=1.0, Ly=1.0, D=0.1)
assert solver.dt_max > 0, '[TC18] FTCS dt_max positive FAILED'

# ---- TC19: Fisher-KPP reaction zero at u=0 ----
u = np.zeros((5, 5))
R = fisher_kpp_reaction(u, r=1.0, K=1.0)
assert np.allclose(R, np.zeros_like(u)), '[TC19] Fisher-KPP zero at u=0 FAILED'

# ---- TC20: Fisher-KPP reaction zero at u=K ----
u = np.ones((5, 5)) * 2.0
R = fisher_kpp_reaction(u, r=1.0, K=2.0)
assert np.allclose(R, np.zeros_like(u)), '[TC20] Fisher-KPP zero at u=K FAILED'

# ---- TC21: Allen-Cahn reaction sign at u>1 ----
u = np.array([2.0])
R = allen_cahn_reaction(u, epsilon=0.1)
assert R[0] < 0, '[TC21] Allen-Cahn reaction sign at u>1 FAILED'

# ---- TC22: Adaptive ODE integrator decays to near zero ----
def decay_ode(t, y):
    return np.array([-0.5 * y[0]])
integrator = AdaptiveMidpointIntegrator(reltol=1e-3, abstol=1e-5)
t_arr, y_arr, nstep, n_rej = integrator.integrate(decay_ode, t0=0.0, tmax=2.0, y0=np.array([1.0]), tau0=0.2)
assert len(y_arr) > 0 and abs(y_arr[-1, 0]) < 0.5, '[TC22] Adaptive ODE decay FAILED'

# ---- TC23: Disk distance stats reproducible with fixed seed ----
mu1, var1 = disk_distance_stats(n_samples=500, rng=np.random.default_rng(42))
mu2, var2 = disk_distance_stats(n_samples=500, rng=np.random.default_rng(42))
assert np.isclose(mu1, mu2) and np.isclose(var1, var2), '[TC23] Disk distance stats reproducible FAILED'

# ---- TC24: Metropolis acceptance probability one for negative dE ----
sampler = MetropolisMCSampler(temperature=300.0, rng_seed=42)
p_acc = sampler.acceptance_probability(-1.0)
assert p_acc == 1.0, '[TC24] Metropolis acceptance prob for negative dE FAILED'

# ---- TC25: Correlation function C(0) equals 1 ----
np.random.seed(42)
field = np.random.randn(16, 16)
C = compute_correlation_function(field, max_r=5)
assert np.isclose(C[0], 1.0), '[TC25] Correlation function C(0)=1 FAILED'

# ---- TC26: Hooke-Jeeves reduces quadratic energy ----
def quad(x):
    return x[0]**2 + x[1]**2
iters, endpt = hooke_jeeves(2, np.array([2.0, -3.0]), rho=0.5, eps=1e-6, itermax=1000, f=quad)
assert quad(endpt) < 0.1, '[TC26] Hooke-Jeeves reduces quadratic energy FAILED'

# ---- TC27: TSP-descent returns finite state and energy ----
np.random.seed(42)
state0 = np.random.randn(10)
def quad_energy(s):
    return np.sum(s**2)
state_opt, E_opt = tsp_descent_style_domain_optimization(state0, quad_energy, n_variations=200, step_size=0.1)
assert len(state_opt) == 10 and np.isfinite(E_opt), '[TC27] TSP-descent returns finite FAILED'

# ---- TC28: MultiferroicSimulator initialization creates fields ----
sim = MultiferroicSimulator(nx=8, ny=8, Lx=1.0, Ly=1.0, temperature=300.0)
assert sim.P.shape == (8, 8) and sim.M.shape == (8, 8), '[TC28] Simulator initialization fields shape FAILED'

# ---- TC29: Magnetoelectric coefficient finite ----
sim = MultiferroicSimulator(nx=8, ny=8, Lx=1.0, Ly=1.0, temperature=300.0)
alpha = sim.compute_magnetoelectric_coefficient()
assert np.isfinite(alpha), '[TC29] Magnetoelectric coefficient finite FAILED'

# ---- TC30: FEM stiffness matrix has positive nnz ----
mesh = MultiferroicMesh(nx=3, ny=3)
assembler = FEMAssembler(mesh, nq=3)
diff_coeff = np.ones(mesh.element_num)
K_coo = assembler.assemble_stiffness_diffusion(diff_coeff)
assert K_coo.nnz() > 0, '[TC30] FEM stiffness matrix nnz positive FAILED'

# ---- TC31: FEM mass matrix diagonal positive ----
mesh = MultiferroicMesh(nx=3, ny=3)
assembler = FEMAssembler(mesh, nq=3)
M_coo = assembler.assemble_mass_matrix()
dense = M_coo.to_dense()
diag = np.diag(dense)
assert np.all(diag > 0), '[TC31] FEM mass matrix diagonal positive FAILED'

# ---- TC32: ReactionDiffusionFTCS solve preserves shape ----
solver = ReactionDiffusionFTCS(nx=11, ny=11, Lx=1.0, Ly=1.0, D=0.1)
u0 = np.ones((11, 11))
u_final = solver.solve(u0, lambda u: np.zeros_like(u), nsteps=5)
assert u_final.shape == (11, 11), '[TC32] FTCS solve preserves shape FAILED'
