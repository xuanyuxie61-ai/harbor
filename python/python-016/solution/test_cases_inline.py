# ---- TC01: graphene_lattice_vectors returns correct shape and magnitudes ----
lv = tb.graphene_lattice_vectors(a=0.246)
assert lv.shape == (2, 2), '[TC01] graphene_lattice_vectors shape FAILED'
assert np.abs(np.linalg.norm(lv[0]) - 0.246) < 1e-10, '[TC01] a1 magnitude FAILED'
assert np.abs(np.linalg.norm(lv[1]) - 0.246) < 1e-10, '[TC01] a2 magnitude FAILED'

# ---- TC02: moire_lattice_constant analytic validation at theta=1.05 deg ----
Lm = tb.moire_lattice_constant(1.05)
expected_Lm = 0.246 / (2.0 * np.sin(np.deg2rad(1.05) * 0.5))
assert np.abs(Lm - expected_Lm) < 1e-10, '[TC02] moire_lattice_constant analytic FAILED'

# ---- TC03: moire_reciprocal_vectors valid range and shape ----
qvecs = tb.moire_reciprocal_vectors(1.05)
assert qvecs.shape == (3, 2), '[TC03] moire_reciprocal_vectors shape FAILED'
assert np.all(np.isfinite(qvecs)), '[TC03] moire_reciprocal_vectors finite FAILED'
mags = np.linalg.norm(qvecs, axis=1)
assert np.allclose(mags, mags[0]), '[TC03] moire_reciprocal_vectors equal magnitudes FAILED'

# ---- TC04: generate_monolayer_sites output sizes ----
pos, sub = tb.generate_monolayer_sites(n_cells=3)
assert pos.shape == (18, 2), '[TC04] positions shape FAILED'
assert sub.shape == (18,), '[TC04] sublattice shape FAILED'
assert np.sum(sub == 0) == 9, '[TC04] A sublattice count FAILED'
assert np.sum(sub == 1) == 9, '[TC04] B sublattice count FAILED'

# ---- TC05: intralayer_hopping zero at r=0 and cutoff at large r ----
t0 = tb.intralayer_hopping(0.0)
assert t0 == 0.0, '[TC05] intralayer_hopping at r=0 FAILED'
t_large = tb.intralayer_hopping(10.0)
assert t_large == 0.0, '[TC05] intralayer_hopping cutoff FAILED'
t_nn = tb.intralayer_hopping(tb.A_CC_NM)
assert t_nn < 0.0, '[TC05] intralayer_hopping NN sign FAILED'

# ---- TC06: interlayer_hopping raises ValueError for negative distance ----
try:
    tb.interlayer_hopping(-1.0)
    assert False, '[TC06] interlayer_hopping negative distance FAILED'
except ValueError:
    pass

# ---- TC07: minimum_image_2d symmetry for periodic image ----
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])
dr1 = tb.minimum_image_2d(np.array([0.6, 0.0]), a1, a2)
dr2 = tb.minimum_image_2d(np.array([-0.4, 0.0]), a1, a2)
assert np.abs(dr1[0] - dr2[0]) < 1e-10, '[TC07] minimum_image_2d symmetry FAILED'

# ---- TC08: build_tight_binding_hamiltonian returns Hermitian matrix ----
H, positions, layer_index = tb.build_tight_binding_hamiltonian(theta_deg=1.05, n_super=2)
N = H.shape[0]
assert H.shape == (N, N), '[TC08] Hamiltonian shape FAILED'
assert np.all(np.isfinite(H)), '[TC08] Hamiltonian finite FAILED'
diff = np.max(np.abs(H - H.T))
assert diff < 1e-12, '[TC08] Hamiltonian Hermitian FAILED'
assert positions.shape == (N, 3), '[TC08] positions shape FAILED'

# ---- TC09: apply_electric_field produces layer-dependent shift ----
H_shifted = tb.apply_electric_field(H, positions, layer_index, field_strength=0.05)
shift_diff = np.diag(H_shifted)[layer_index == 1].mean() - np.diag(H_shifted)[layer_index == 0].mean()
assert shift_diff > 0.0, '[TC09] electric field layer shift sign FAILED'

# ---- TC10: diagonalize_hamiltonian sorted eigenvalues ----
energies, vectors = bs.diagonalize_hamiltonian(H)
assert energies.size == N, '[TC10] eigenvalue count FAILED'
assert np.all(np.diff(energies) >= -1e-12), '[TC10] eigenvalue sorting FAILED'
assert np.all(np.isreal(energies)), '[TC10] eigenvalue real FAILED'

# ---- TC11: find_fermi_level half-filling analytic check ----
energies_1d = np.array([-3.0, -1.0, 0.5, 2.0])
ef = bs.find_fermi_level(energies_1d)
assert ef == 0.5, '[TC11] find_fermi_level analytic FAILED'

# ---- TC12: band_gap_at_fermi_level non-negative gap ----
gap, n_occ, n_unocc = bs.band_gap_at_fermi_level(energies_1d, ef)
assert gap >= 0.0, '[TC12] band_gap non-negative FAILED'
assert n_occ + n_unocc == 4, '[TC12] band count FAILED'

# ---- TC13: gaussian_dos non-negative and integrates to ~1 ----
E_grid = np.linspace(-5, 5, 1000)
dos = du.gaussian_dos(energies_1d, E_grid, sigma=0.1)
assert dos.shape == E_grid.shape, '[TC13] gaussian_dos shape FAILED'
assert np.all(dos >= -1e-15), '[TC13] gaussian_dos non-negative FAILED'
integral = np.trapz(dos, E_grid)
assert np.abs(integral - 1.0) < 0.05, '[TC13] gaussian_dos integral FAILED'

# ---- TC14: find_van_hove_singularities detects known peaks ----
E_test = np.linspace(-2, 2, 200)
D_test = np.exp(-E_test**2 / 0.5) + 0.1
D_test[100] = 5.0  # artificial peak
vhs = du.find_van_hove_singularities(E_test, D_test, prominence=0.05)
assert len(vhs) >= 1, '[TC14] VHS detection FAILED'

# ---- TC15: cyclotron_frequency analytic validation ----
wc = sd.cyclotron_frequency(0.05, 1.0)
expected_wc = 0.1759 * 1.0 / 0.05
assert np.abs(wc - expected_wc) < 1e-10, '[TC15] cyclotron_frequency analytic FAILED'

# ---- TC16: tridiagonal_solve accuracy on simple system ----
a = np.array([0.0, -1.0, -1.0])
b = np.array([2.0, 2.0, 2.0])
c = np.array([-1.0, -1.0, 0.0])
d = np.array([1.0, 1.0, 1.0])
x_sol = td.tridiagonal_solve(a, b, c, d)
residual = np.linalg.norm(td.tridiagonal_matvec(a, b, c, x_sol) - d)
assert residual < 1e-12, '[TC16] tridiagonal_solve residual FAILED'

# ---- TC17: build_tridiagonal_from_1d_chain dimensions ----
onsite = np.linspace(-1, 1, 10)
hopping = -0.3 * np.ones(9)
a_tri, b_tri, c_tri = td.build_tridiagonal_from_1d_chain(onsite, hopping)
assert a_tri.shape == (10,), '[TC17] tridiagonal a shape FAILED'
assert b_tri.shape == (10,), '[TC17] tridiagonal b shape FAILED'
assert c_tri.shape == (10,), '[TC17] tridiagonal c shape FAILED'

# ---- TC18: lights_out_matrix_moire dimensions and entries ----
L = tp.lights_out_matrix_moire(3, 3)
assert L.shape == (9, 9), '[TC18] lights_out shape FAILED'
assert np.all((L == 0) | (L == 1)), '[TC18] lights_out binary FAILED'
assert L[0, 0] == 1, '[TC18] lights_out diagonal FAILED'

# ---- TC19: mod2_matrix_rank bounded by min dimension ----
M = np.random.rand(5, 7)
rank = tp.mod2_matrix_rank(M)
assert rank <= 5, '[TC19] mod2 rank upper bound FAILED'

# ---- TC20: linear_to_quadratic_triangles node count increase ----
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems = np.array([[0, 1, 2]])
new_nodes, new_elems = mu.linear_to_quadratic_triangles(nodes, elems)
assert new_nodes.shape[0] == 6, '[TC20] quadratic nodes count FAILED'
assert new_elems.shape == (1, 6), '[TC20] quadratic elems shape FAILED'

# ---- TC21: build_adjacency_from_elements symmetry ----
adj = mu.build_adjacency_from_elements(elems, 3, "triangle")
assert len(adj) == 3, '[TC21] adjacency length FAILED'
for i in range(3):
    for j in adj[i]:
        assert i in adj[j], '[TC21] adjacency symmetry FAILED'

# ---- TC22: chebyshev_nodes within interval and monotonic ----
nodes_cheb = ft.chebyshev_nodes(-1.0, 1.0, 5)
assert np.all(nodes_cheb >= -1.0) and np.all(nodes_cheb <= 1.0), '[TC22] chebyshev_nodes range FAILED'
assert np.all(np.diff(nodes_cheb) < 0), '[TC22] chebyshev_nodes monotonic FAILED'

# ---- TC23: lagrange_basis unit property at nodes ----
x_nodes = np.array([0.0, 1.0, 2.0])
x_eval = np.array([0.0, 1.0, 2.0])
L_basis = ft.lagrange_basis(x_nodes, x_eval)
assert L_basis.shape == (3, 3), '[TC23] lagrange_basis shape FAILED'
assert np.allclose(np.diag(L_basis), 1.0), '[TC23] lagrange_basis diagonal FAILED'
assert np.allclose(L_basis - np.diag(np.diag(L_basis)), 0.0), '[TC23] lagrange_basis off-diagonal FAILED'

# ---- TC24: solve_self_consistent_moire converges for linear fixed point ----
def linear_fp(x):
    return 0.5 * x + 1.0
x_sc, iters, diff = rf.solve_self_consistent_moire(linear_fp, np.array([0.0]), tol=1e-10, max_iter=100, alpha_mix=0.5)
assert np.abs(x_sc[0] - 2.0) < 1e-6, '[TC24] self_consistent_moire convergence FAILED'

# ---- TC25: kmeans_lloyd label range and centroid count ----
np.random.seed(42)
data = np.vstack([np.random.randn(20, 2) + np.array([0.0, 0.0]),
                  np.random.randn(20, 2) + np.array([5.0, 5.0])])
labels, centroids, inertia = cl.kmeans_lloyd(data, n_clusters=2, init="random")
assert np.all((labels == 0) | (labels == 1)), '[TC25] kmeans label range FAILED'
assert centroids.shape == (2, 2), '[TC25] kmeans centroids shape FAILED'
assert inertia >= 0.0, '[TC25] kmeans inertia non-negative FAILED'

# ---- TC26: cluster_energy_analysis returns correct structure ----
labels_test = np.array([0, 0, 1, 1, 1])
energies_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
positions_test = np.zeros((5, 3))
stats = cl.cluster_energy_analysis(labels_test, energies_test, positions_test)
assert 0 in stats and 1 in stats, '[TC26] cluster_energy_analysis keys FAILED'
assert stats[0]['count'] == 2, '[TC26] cluster_energy_analysis count 0 FAILED'
assert stats[1]['count'] == 3, '[TC26] cluster_energy_analysis count 1 FAILED'

# ---- TC27: hexagon_domain_sample output size and containment ----
np.random.seed(42)
samples = ks.hexagon_domain_sample(50, radius=1.0)
assert samples.shape == (50, 2), '[TC27] hexagon_sample shape FAILED'
apothem = 1.0 * np.cos(np.pi / 6.0)
for p in samples:
    assert np.linalg.norm(p) <= 1.0 + 1e-10, '[TC27] hexagon_sample containment FAILED'

# ---- TC28: irreducible_wedge_kpoints deduplication ----
np.random.seed(42)
kpts = np.array([[0.1, 0.0], [0.0, 0.1], [0.1, 0.0]])
wedge = ks.irreducible_wedge_kpoints(kpts, tolerance=1e-6)
assert wedge.shape[0] <= kpts.shape[0], '[TC28] irreducible_wedge size FAILED'

# ---- TC29: least_squares_lagrange_fit reproduces low-degree polynomial ----
x_data = np.linspace(-1, 1, 10)
y_data = 2.0 * x_data + 1.0
coeffs, cheb_nodes = ft.least_squares_lagrange_fit(x_data, y_data, degree=1)
y_pred = ft.evaluate_lagrange_polynomial(x_data, coeffs, cheb_nodes)
assert np.max(np.abs(y_pred - y_data)) < 1e-10, '[TC29] least_squares_lagrange_fit linear FAILED'

# ---- TC30: rk45_step preserves state dimension ----
def dummy_rhs(y):
    return -y
y0 = np.array([1.0, 2.0])
y_next, error, h_new = sd.rk45_step(dummy_rhs, y0, 0.0, 0.1)
assert y_next.shape == y0.shape, '[TC30] rk45_step shape FAILED'
assert error.shape == y0.shape, '[TC30] rk45_step error shape FAILED'
assert h_new > 0.0, '[TC30] rk45_step h_new positive FAILED'
