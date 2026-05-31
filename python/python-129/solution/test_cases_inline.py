np.random.seed(42)

# ---- TC01: bessel_jx_fractional accuracy nu=0.5 x=0.2 ----
result = bessel_jx_fractional(0.5, 0.2)
assert abs(result - 0.3544507442114011) < 1e-6, '[TC01] bessel_jx_fractional accuracy nu=0.5 x=0.2 FAILED'

# ---- TC02: bessel_jx_fractional accuracy nu=0.5 x=1.0 ----
result = bessel_jx_fractional(0.5, 1.0)
assert abs(result - 0.6713967071418031) < 1e-6, '[TC02] bessel_jx_fractional accuracy nu=0.5 x=1.0 FAILED'

# ---- TC03: bessel_jx_fractional array input returns same shape ----
x_arr = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
res_arr = bessel_jx_fractional(0.5, x_arr)
assert res_arr.shape == x_arr.shape, '[TC03] bessel_jx_fractional array input shape FAILED'
assert np.all(np.isfinite(res_arr)), '[TC03] bessel_jx_fractional array input finite FAILED'

# ---- TC04: modified_bessel_clot_profile normalized at r0 ----
r = np.array([0.0, 15.0, 30.0, 50.0])
prof = modified_bessel_clot_profile(r, r0=30.0, D_fibrin=2.5e-2, k_poly=1.2e-3)
assert abs(prof[2] - 1.0) < 1e-6, '[TC04] modified_bessel_clot_profile normalized at r0 FAILED'

# ---- TC05: modified_bessel_clot_profile decays beyond r0 ----
r_test = np.array([30.0, 35.0, 40.0])
prof_test = modified_bessel_clot_profile(r_test, r0=30.0, D_fibrin=2.5e-2, k_poly=1.2e-3)
assert prof_test[1] < prof_test[0], '[TC05] modified_bessel_clot_profile decays beyond r0 FAILED'
assert prof_test[2] < prof_test[1], '[TC05] modified_bessel_clot_profile monotonic beyond r0 FAILED'

# ---- TC06: verify_bessel_accuracy small error ----
err = verify_bessel_accuracy()
assert err < 1e-3, '[TC06] verify_bessel_accuracy small error FAILED'
assert err >= 0.0, '[TC06] verify_bessel_accuracy non-negative FAILED'

# ---- TC07: CoagulationNetwork default species count ----
net = CoagulationNetwork()
assert net.n_species == 12, '[TC07] CoagulationNetwork default species count FAILED'

# ---- TC08: CoagulationNetwork rhs shape and finite output ----
net = CoagulationNetwork()
y0 = np.zeros(net.n_species)
dydt = net.rhs(y0, t=0.0)
assert dydt.shape == (12,), '[TC08] CoagulationNetwork rhs shape FAILED'
assert np.all(np.isfinite(dydt)), '[TC08] CoagulationNetwork rhs finite output FAILED'

# ---- TC09: CoagulationNetwork rhs handles negative inputs ----
net = CoagulationNetwork()
y_neg = -np.ones(net.n_species)
dydt_neg = net.rhs(y_neg, t=0.0)
assert np.all(np.isfinite(dydt_neg)), '[TC09] CoagulationNetwork rhs handles negative inputs FAILED'

# ---- TC10: CoagulationNetwork jacobian shape ----
net = CoagulationNetwork()
J = net.jacobian(np.ones(net.n_species))
assert J.shape == (12, 12), '[TC10] CoagulationNetwork jacobian shape FAILED'
assert np.all(np.isfinite(J)), '[TC10] CoagulationNetwork jacobian finite FAILED'

# ---- TC11: trapezoidal_solve returns correct shapes ----
net = CoagulationNetwork()
y0 = np.zeros(net.n_species)
y0[0] = 5.0
y0[net.SPECIES_NAMES.index("Va")] = 0.5
y0[net.SPECIES_NAMES.index("IIa")] = 0.01
t, y = trapezoidal_solve(net, y0, (0.0, 10.0), n_steps=50)
assert t.shape == (51,), '[TC11] trapezoidal_solve t shape FAILED'
assert y.shape == (51, 12), '[TC11] trapezoidal_solve y shape FAILED'
assert np.all(np.isfinite(y)), '[TC11] trapezoidal_solve y finite FAILED'

# ---- TC12: AnnularDiffusionSolver grid construction ----
solver = AnnularDiffusionSolver(r_inner=1.0, r_outer=10.0, nr=5, ntheta=4, D_coef=1.0)
assert solver.r.shape == (5,), '[TC12] AnnularDiffusionSolver r shape FAILED'
assert solver.theta.shape == (4,), '[TC12] AnnularDiffusionSolver theta shape FAILED'
assert solver.dr > 0, '[TC12] AnnularDiffusionSolver dr positive FAILED'

# ---- TC13: AnnularDiffusionSolver Laplacian shape ----
assert solver.L.shape == (20, 20), '[TC13] AnnularDiffusionSolver Laplacian shape FAILED'

# ---- TC14: integrate_over_annulus approximates ring area ----
solver = AnnularDiffusionSolver(r_inner=1.0, r_outer=10.0, nr=20, ntheta=16, D_coef=1.0)
field = np.ones((solver.nr, solver.ntheta))
total = solver.integrate_over_annulus(field)
expected_area = np.pi * (10.0**2 - 1.0**2)
assert abs(total - expected_area) / expected_area < 0.08, '[TC14] integrate_over_annulus approximates ring area FAILED'

# ---- TC15: wound_source_term peak at wound location ----
val_center = wound_source_term(45.0, 0.0, 0.0, 0.0, r_wound=45.0, theta_wound=0.0)
val_off = wound_source_term(30.0, np.pi, 0.0, 0.0, r_wound=45.0, theta_wound=0.0)
assert val_center > val_off, '[TC15] wound_source_term peak at wound location FAILED'

# ---- TC16: solve_steady_state output shape ----
solver = AnnularDiffusionSolver(r_inner=1.0, r_outer=10.0, nr=5, ntheta=4, D_coef=1.0)
c_steady = solver.solve_steady_state(lambda c: -0.01 * c, max_iter=5)
assert c_steady.shape == (5, 4), '[TC16] solve_steady_state output shape FAILED'
assert np.all(np.isfinite(c_steady)), '[TC16] solve_steady_state finite FAILED'

# ---- TC17: PageRank sums to 1 and non-negative ----
graph = CoagulationNetworkGraph()
pr = graph.pagerank()
assert abs(np.sum(pr) - 1.0) < 1e-10, '[TC17] PageRank sums to 1 FAILED'
assert np.all(pr >= 0), '[TC17] PageRank non-negative FAILED'

# ---- TC18: Jaccard distance symmetry and diagonal zero ----
graph = CoagulationNetworkGraph()
dist = graph.jaccard_distance_matrix()
assert np.allclose(dist, dist.T), '[TC18] Jaccard distance symmetry FAILED'
assert np.all(np.diag(dist) == 0.0), '[TC18] Jaccard distance diagonal zero FAILED'

# ---- TC19: hierarchical_clustering linkage length ----
graph = CoagulationNetworkGraph()
dist = graph.jaccard_distance_matrix()
linkage, clusters = graph.hierarchical_clustering(dist)
assert len(linkage) == graph.n_nodes - 1, '[TC19] hierarchical_clustering linkage length FAILED'

# ---- TC20: PageRank reproducibility ----
graph = CoagulationNetworkGraph()
pr1 = graph.pagerank(alpha=0.85, max_iter=200)
pr2 = graph.pagerank(alpha=0.85, max_iter=200)
assert np.allclose(pr1, pr2), '[TC20] PageRank reproducibility FAILED'

# ---- TC21: PlateletMonteCarlo optimal stopping skip=0 ----
import numpy as np
np.random.seed(42)
mc = PlateletMonteCarlo(n_platelets=10, local_iia=20.0, seed=42)
pots = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
idx, val = mc.optimal_stopping_strategy(pots, skip_num=0)
assert idx == 0, '[TC21] PlateletMonteCarlo optimal stopping skip=0 FAILED'
assert val == pots[0], '[TC21] PlateletMonteCarlo optimal stopping skip=0 value FAILED'

# ---- TC22: PlateletMonteCarlo reproducibility with fixed seed ----
mc1 = PlateletMonteCarlo(n_platelets=100, local_iia=20.0, seed=42)
mc2 = PlateletMonteCarlo(n_platelets=100, local_iia=20.0, seed=42)
p1 = mc1._generate_potentials()
p2 = mc2._generate_potentials()
assert np.allclose(p1, p2), '[TC22] PlateletMonteCarlo reproducibility with fixed seed FAILED'

# ---- TC23: PlateletMonteCarlo find_optimal_skip range ----
np.random.seed(42)
mc = PlateletMonteCarlo(n_platelets=100, local_iia=20.0, seed=42)
best_skip, best_score, _ = mc.find_optimal_skip(n_trials=20)
assert 0 <= best_skip < 100, '[TC23] PlateletMonteCarlo find_optimal_skip range FAILED'

# ---- TC24: PlateletMonteCarlo optimal stopping with valid skip ----
np.random.seed(42)
mc = PlateletMonteCarlo(n_platelets=100, local_iia=20.0, seed=42)
pots = mc._generate_potentials()
idx, val = mc.optimal_stopping_strategy(pots, skip_num=int(round(100 / np.e)))
assert isinstance(idx, (int, np.integer)), '[TC24] PlateletMonteCarlo optimal stopping valid skip FAILED'

# ---- TC25: HexagonQuadrature area matches analytical ----
hex_q = HexagonQuadrature(radius=2.0)
area_num = hex_q.monomial_integral(0, 0)
assert abs(area_num - hex_q.area) < 1e-3, '[TC25] HexagonQuadrature area matches analytical FAILED'

# ---- TC26: HexagonQuadrature odd monomial is zero ----
hex_q = HexagonQuadrature(radius=1.0)
val = hex_q.monomial_integral(1, 0)
assert abs(val) < 1e-12, '[TC26] HexagonQuadrature odd monomial is zero FAILED'

# ---- TC27: HexagonQuadrature porosity in [0,1] ----
hex_q = HexagonQuadrature(radius=2.0)
porosity = hex_q.compute_clot_fiber_volume()
assert 0.0 <= porosity <= 1.0, '[TC27] HexagonQuadrature porosity in [0,1] FAILED'

# ---- TC28: TriangleBarycentricQuadrature exact xy integral ----
tri_q = TriangleBarycentricQuadrature()
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
val = tri_q.integrate_on_triangle(lambda x, y: x * y, verts, order=3)
assert abs(val - 1.0/24.0) < 1e-4, '[TC28] TriangleBarycentricQuadrature exact xy integral FAILED'

# ---- TC29: TriangleBarycentricQuadrature barycentric conversion ----
bary = TriangleBarycentricQuadrature.xy_to_barycentric(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
assert bary.shape == (3, 3), '[TC29] TriangleBarycentricQuadrature barycentric shape FAILED'
assert np.allclose(np.sum(bary, axis=1), 1.0), '[TC29] TriangleBarycentricQuadrature barycentric sum to 1 FAILED'

# ---- TC30: arrhenius_rate_integral analytic close to numerical ----
k_a, k_n = arrhenius_rate_integral(T=310.0, A_pre=1e12, Ea_mean=5e4, Ea_sigma=3e3)
assert abs(k_a - k_n) / k_a < 0.01, '[TC30] arrhenius_rate_integral analytic close to numerical FAILED'

# ---- TC31: arrhenius_rate_integral both positive ----
k_a, k_n = arrhenius_rate_integral(T=300.0, A_pre=1e6, Ea_mean=4e4, Ea_sigma=2e3)
assert k_a > 0 and k_n > 0, '[TC31] arrhenius_rate_integral both positive FAILED'

# ---- TC32: ParameterSensitivitySVD sensitivity matrix shape ----
param_names = ["p1", "p2", "p3", "p4"]
response_names = ["r1", "r2"]
analyzer = ParameterSensitivitySVD(param_names, response_names)
def simple_model(p):
    return np.array([p[0] + p[1], p[2] * p[3]])
p0 = np.array([1.0, 2.0, 3.0, 4.0])
S = analyzer.compute_sensitivity_matrix(simple_model, p0, delta_frac=0.01)
assert S.shape == (2, 4), '[TC32] ParameterSensitivitySVD sensitivity matrix shape FAILED'

# ---- TC33: ParameterSensitivitySVD SVD cum_var monotonic and bounded ----
U, sigma, Vt, cum_var = analyzer.svd_decompose(S)
assert len(cum_var) == min(S.shape), '[TC33] ParameterSensitivitySVD cum_var length FAILED'
assert np.all(np.diff(cum_var) >= -1e-12), '[TC33] ParameterSensitivitySVD cum_var monotonic FAILED'
assert cum_var[-1] > 0.99, '[TC33] ParameterSensitivitySVD cum_var ends near 1 FAILED'

# ---- TC34: ParameterSensitivitySVD principal directions structure ----
dirs = analyzer.principal_parameter_directions(Vt, n_components=2)
assert len(dirs) == 2 and 'top_params' in dirs[0], '[TC34] ParameterSensitivitySVD principal directions structure FAILED'

# ---- TC35: ParameterSensitivitySVD reduced_basis_fit output shape ----
target = np.array([50.0, 40.0])
p0_fit = np.array([1.0, 2.0, 3.0, 4.0])
p_opt, alpha = analyzer.reduced_basis_fit(S, target, p0_fit, n_components=2)
assert p_opt.shape == (4,), '[TC35] ParameterSensitivitySVD reduced_basis_fit shape FAILED'
assert np.all(p_opt >= 0), '[TC35] ParameterSensitivitySVD reduced_basis_fit non-negative FAILED'

# ---- TC36: StabilityAnalyzer trapezoidal stability on negative real ----
stab_analyzer = StabilityAnalyzer()
z = -5.0 + 0.0j
R = stab_analyzer.trapezoidal_stability_function(z)
assert abs(R) <= 1.0 + 1e-12, '[TC36] StabilityAnalyzer trapezoidal stability on negative real FAILED'

# ---- TC37: StabilityAnalyzer analyze_jacobian_stiffness on diagonal ----
J = np.diag([-100.0, -1.0, -0.1])
eigs, stiff_ratio, h_euler, h_trap = stab_analyzer.analyze_jacobian_stiffness(J)
assert abs(stiff_ratio - 1000.0) < 1e-9, '[TC37] StabilityAnalyzer analyze_jacobian_stiffness stiff_ratio FAILED'
assert h_euler > 0 and h_trap > 0, '[TC37] StabilityAnalyzer h_euler and h_trap positive FAILED'

# ---- TC38: StabilityAnalyzer boundary_locus shape ----
bl = stab_analyzer.boundary_locus("trapezoidal", n_points=100)
assert bl.shape == (100,) and np.iscomplexobj(bl), '[TC38] StabilityAnalyzer boundary_locus shape FAILED'

# ---- TC39: StabilityAnalyzer is_stable on left half plane ----
assert stab_analyzer.is_stable("trapezoidal", -10.0 + 0.0j), '[TC39] StabilityAnalyzer is_stable left half plane FAILED'
assert stab_analyzer.is_stable("implicit_euler", -1.0 + 1.0j), '[TC39] StabilityAnalyzer is_stable implicit_euler FAILED'

# ---- TC40: GrayCodeGenetics int_to_gray of 0 returns all zeros ----
bits = GrayCodeGenetics.int_to_gray(0, n_bits=8)
assert np.sum(bits) == 0, '[TC40] GrayCodeGenetics int_to_gray of 0 FAILED'

# ---- TC41: GrayCodeGenetics gray_to_int of all zeros returns 0 ----
val = GrayCodeGenetics.gray_to_int(np.zeros(8, dtype=int))
assert val == 0, '[TC41] GrayCodeGenetics gray_to_int of all zeros FAILED'

# ---- TC42: GrayCodeGenetics hamming_distance correctness ----
a = np.array([0, 1, 0, 1])
b = np.array([1, 1, 0, 0])
d = GrayCodeGenetics.hamming_distance(a, b)
assert d == 2, '[TC42] GrayCodeGenetics hamming_distance correctness FAILED'

# ---- TC43: GrayCodeGenetics encode and risk score ----
bits = GrayCodeGenetics.encode_coagulation_genotype(True, False, True, False)
weights = np.array([3.0, 2.5, 1.0, 2.0] + [0.0] * 12)
risk = GrayCodeGenetics.genetic_risk_score(bits, weights)
assert abs(risk - 4.0) < 1e-10, '[TC43] GrayCodeGenetics encode and risk score FAILED'

# ---- TC44: GrayCodeGenetics binary_distance_matrix symmetry ----
genotypes = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
])
dist_mat = GrayCodeGenetics.binary_distance_matrix(genotypes)
assert np.allclose(dist_mat, dist_mat.T), '[TC44] GrayCodeGenetics binary_distance_matrix symmetry FAILED'
assert dist_mat[0, 0] == 0, '[TC44] GrayCodeGenetics binary_distance_matrix diagonal zero FAILED'

# ---- TC45: VariominoClot condense removes zero borders ----
P = np.zeros((10, 10))
P[3:7, 3:7] = 1.0
clot = VariominoClot(P)
Q = clot.condense()
assert Q.shape == (4, 4), '[TC45] VariominoClot condense removes zero borders FAILED'

# ---- TC46: VariominoClot count_pore_clusters on full matrix ----
P2 = np.ones((5, 5))
clot2 = VariominoClot(P2)
n_pores, labels = clot2.count_pore_clusters(threshold=0.05)
assert n_pores == 0, '[TC46] VariominoClot count_pore_clusters on full matrix FAILED'

# ---- TC47: VariominoClot anisotropy in range [0,1] ----
np.random.seed(42)
P3 = VariominoClot.generate_random_clot(height=20, width=20, n_fibers=5, seed=42)
clot3 = VariominoClot(P3)
ani = clot3.compute_anisotropy()
assert 0.0 <= ani <= 1.0, '[TC47] VariominoClot anisotropy in [0,1] FAILED'

# ---- TC48: VariominoClot embed_in_domain shape ----
embedded = clot3.embed_in_domain((50, 50), center=(25, 25))
assert embedded.shape == (50, 50), '[TC48] VariominoClot embed_in_domain shape FAILED'

# ---- TC49: VariominoClot random clot reproducibility ----
P4 = VariominoClot.generate_random_clot(height=20, width=20, n_fibers=5, seed=123)
P5 = VariominoClot.generate_random_clot(height=20, width=20, n_fibers=5, seed=123)
assert np.allclose(P4, P5), '[TC49] VariominoClot random clot reproducibility FAILED'

# ---- TC50: VariominoClot rotate and flip preserves shape ----
P6 = np.arange(12).reshape(3, 4).astype(float)
clot6 = VariominoClot(P6)
rot = clot6.rotate90(k=1)
assert rot.shape == (4, 3), '[TC50] VariominoClot rotate90 shape FAILED'
fh = clot6.flip_horizontal()
assert fh.shape == P6.shape, '[TC50] VariominoClot flip_horizontal shape FAILED'
fv = clot6.flip_vertical()
assert fv.shape == P6.shape, '[TC50] VariominoClot flip_vertical shape FAILED'

# ---- TC51: SwapClustering fit valid labels and non-negative WCSS ----
X = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1], [10.1, 10.1], [5.0, 5.0]])
clusterer = SwapClustering(n_clusters=2, max_iter=50)
labels_fit, centroids, wcss, n_swaps = clusterer.fit(X)
assert wcss >= 0, '[TC51] SwapClustering WCSS non-negative FAILED'
assert len(labels_fit) == 5, '[TC51] SwapClustering labels length FAILED'
assert len(np.unique(labels_fit)) <= 2, '[TC51] SwapClustering at most 2 clusters FAILED'

# ---- TC52: generate_coagulation_phenotypes shape and 3 classes ----
np.random.seed(42)
X_ph, true_labels_ph = generate_coagulation_phenotypes(n_samples=30, seed=42)
assert X_ph.shape == (30, 5), '[TC52] generate_coagulation_phenotypes shape FAILED'
assert len(np.unique(true_labels_ph)) == 3, '[TC52] generate_coagulation_phenotypes 3 classes FAILED'

# ---- TC53: SwapClustering predict assigns nearest centroid ----
X_pred = np.array([[0.0, 0.0], [10.0, 10.0]])
centroids_pred = np.array([[0.0, 0.0], [10.0, 10.0]])
clusterer_pred = SwapClustering(n_clusters=2)
preds = clusterer_pred.predict(X_pred, centroids_pred)
assert preds[0] == 0, '[TC53] SwapClustering predict assigns nearest centroid FAILED'
assert preds[1] == 1, '[TC53] SwapClustering predict assigns nearest centroid FAILED'

# ---- TC54: generate_coagulation_phenotypes feature ranges ----
np.random.seed(42)
X_range, _ = generate_coagulation_phenotypes(n_samples=60, seed=42)
assert X_range[:, 0].min() > 8.0, '[TC54] generate_coagulation_phenotypes PT range FAILED'
assert X_range[:, 4].min() > 50.0, '[TC54] generate_coagulation_phenotypes Platelet range FAILED'

# ---- TC55: run_special_functions_module returns 1D ndarray ----
profile = run_special_functions_module()
assert isinstance(profile, np.ndarray) and profile.ndim == 1, '[TC55] run_special_functions_module returns 1D ndarray FAILED'

# ---- TC56: run_genetics_module returns GrayCodeGenetics ----
gc_result = run_genetics_module()
assert isinstance(gc_result, GrayCodeGenetics), '[TC56] run_genetics_module returns GrayCodeGenetics FAILED'

# ---- TC57: main() integration returns 0 ----
result = main()
assert result == 0, '[TC57] main() integration returns 0 FAILED'
