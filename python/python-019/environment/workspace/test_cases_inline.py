# ---- TC01: PT-symmetric Hamiltonian returns 2x2 complex matrix ----
H_1d = build_pt_symmetric_hamiltonian_1d(0.5, t=1.0, m=0.5, gamma=0.3)
assert H_1d.shape == (2, 2), '[TC01] PT-symmetric H shape FAILED'
assert np.iscomplexobj(H_1d), '[TC01] PT-symmetric H dtype FAILED'

# ---- TC02: Discriminant vanishes at exceptional point (gamma^2 = m^2 + t^2, k=0) ----
H_ep = build_pt_symmetric_hamiltonian_1d(0.0, t=1.0, m=0.0, gamma=1.0)
delta_ep = discriminant_2x2(H_ep)
assert abs(delta_ep) < 1e-12, '[TC02] Discriminant at EP FAILED'

# ---- TC03: SSH Hamiltonian trace is zero ----
H_ssh = build_nonhermitian_ssh_hamiltonian(0.5, t1=1.0, t2=0.5, gamma=0.2)
assert abs(np.trace(H_ssh)) < 1e-12, '[TC03] SSH trace FAILED'

# ---- TC04: Hofstadter Hamiltonian size equals flux denominator q ----
H_hof = build_nonhermitian_hofstadter_hamiltonian(0.1, 0.2, phi=1.0/4.0, q=4)
assert H_hof.shape == (4, 4), '[TC04] Hofstadter shape FAILED'

# ---- TC05: Biorthogonal eigenvectors satisfy normalization overlap = 1 ----
H_test = np.array([[1.0, 0.5+0.1j], [0.5-0.1j, -1.0+0.2j]], dtype=complex)
E, right, left = compute_biorthogonal_eigenvectors(H_test)
for n in range(2):
    overlap = np.vdot(left[n, :], right[:, n])
    assert abs(overlap - 1.0) < 1e-10, f'[TC05] Biorthogonal overlap band {n} FAILED'

# ---- TC06: Berry connection returns finite scalar ----
bc = berry_connection_1d(
    lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3),
    k=0.5, dk=1e-4
)
assert np.isfinite(bc), '[TC06] Berry connection finite FAILED'

# ---- TC07: Zak phase integration yields finite value ----
zak = zak_phase_1d(
    lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.1),
    k_points=201
)
assert np.isfinite(zak), '[TC07] Zak phase finite FAILED'

# ---- TC08: BZ tetrahedron partition count for n_k=2 is 48 ----
tetras = partition_bz_into_tetrahedra(n_k=2)
assert len(tetras) == 2**3 * 6, '[TC08] Tetrahedra count FAILED'

# ---- TC09: Constant function integral over BZ equals BZ volume ----
def f_const(kx, ky, kz): return 1.0
val = integrate_bz_3d(f_const, n_k=2, degree=3)
expected = (2.0 * np.pi)**3
assert abs(val - expected) < 0.5, '[TC09] BZ constant integral FAILED'

# ---- TC10: RKF45 step advances time and preserves state shape ----
def rhs(t, y): return -1j * y
y0 = np.array([1.0+0j, 0.5-0.2j])
y_new, t_new, h_new = rkf45_step_complex(rhs, 0.0, y0, 0.01, tol=1e-8)
assert t_new > 0.0, '[TC10] RKF45 time advance FAILED'
assert y_new.shape == y0.shape, '[TC10] RKF45 shape preservation FAILED'

# ---- TC11: Non-Hermitian Schrodinger evolution output lengths match ----
H_eff = np.array([[1.0, 0.2], [0.2, -1.0]], dtype=complex) + 1j*np.array([[-0.1,0],[0,0.1]], dtype=complex)
psi0 = np.array([1.0, 0.0], dtype=complex)
t_vals, psi_vals, norms = evolve_nonhermitian_schrodinger(H_eff, psi0, (0.0, 1.0), dt0=1e-3, tol=1e-9)
assert len(t_vals) == len(psi_vals) == len(norms), '[TC11] Evolution output lengths FAILED'
assert norms[-1] >= 0.0, '[TC11] Final norm non-negative FAILED'

# ---- TC12: Lindblad purity remains in physical range [0.5, 1.0] ----
H_l = np.array([[1.0, 0.3], [0.3, -0.5]], dtype=complex)
L1 = np.array([[0.0, 0.2], [0.0, 0.0]], dtype=complex)
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
t_l, rho_l, purity = lindblad_evolve_2level(H_l, [L1], rho0, (0.0, 1.0), dt0=1e-3, tol=1e-9)
assert np.all(purity >= 0.5) and np.all(purity <= 1.0), '[TC12] Lindblad purity range FAILED'

# ---- TC13: Incomplete beta I_0.5(1,1) equals 0.5 ----
val_beta, ifault = incomplete_beta(0.5, 1.0, 1.0)
assert ifault == 0, '[TC13] Incomplete beta ifault FAILED'
assert abs(val_beta - 0.5) < 1e-12, '[TC13] Incomplete beta value FAILED'

# ---- TC14: Ginibre spectrum size equals matrix dimension ----
np.random.seed(42)
eig = generate_ginibre_spectrum(100, seed=42)
assert eig.shape == (100,), '[TC14] Ginibre spectrum size FAILED'

# ---- TC15: Level-spacing ratios lie in [0,1] ----
np.random.seed(42)
eig = generate_ginibre_spectrum(50, seed=42)
spacings, ratios = level_spacing_ratios(eig, sort_by='real')
assert np.all(ratios >= 0.0) and np.all(ratios <= 1.0), '[TC15] Spacing ratios range FAILED'

# ---- TC16: SimpleMesh bounding box contains all nodes ----
pts, tri = triangulate_domain_rectangle((-1.0, 1.0), (-1.0, 1.0), nx=5, ny=5)
mesh = SimpleMesh(pts, tri)
mn, mx = mesh.bounding_box()
assert np.all(mn <= pts.min(axis=0)) and np.all(mx >= pts.max(axis=0)), '[TC16] Mesh bounding box FAILED'

# ---- TC17: Mass matrix is diagonal with positive entries ----
M = build_mass_matrix(mesh)
assert np.allclose(M, np.diag(np.diag(M))), '[TC17] Mass matrix diagonal FAILED'
assert np.all(np.diag(M) > 0.0), '[TC17] Mass matrix positive FAILED'

# ---- TC18: Transfer matrix determinant equals t1/t2 ----
T = transfer_matrix_ssh(0.0+0.0j, t1=1.0, t2=0.5, gamma=0.2)
assert abs(np.linalg.det(T) - (1.0/0.5)) < 1e-12, '[TC18] Transfer matrix determinant FAILED'

# ---- TC19: Lyapunov exponent is finite for disordered SSH ----
lyap = lyapunov_exponent_ssh(0.0, t1=1.0, t2=0.5, gamma=0.2, N=200, seed=42)
assert np.isfinite(lyap), '[TC19] Lyapunov finite FAILED'

# ---- TC20: Markov steady-state distribution sums to 1 ----
L_m = nonhermitian_markov_chain(8, p_forward=0.7, p_backward=0.3, loss_rate=0.1)
pi_ss = steady_state_distribution(L_m)
assert abs(pi_ss.sum() - 1.0) < 1e-10, '[TC20] Steady state sum FAILED'

# ---- TC21: Vandermonde determinant matches product formula ----
nodes = np.array([1.0, 2.0, 3.0, 4.0])
det_vp = vandermonde_determinant(nodes)
det_ref = 1.0 + 0.0j
for j in range(nodes.size):
    for i in range(j + 1, nodes.size):
        det_ref *= (nodes[i] - nodes[j])
assert abs(det_vp - det_ref) < 1e-10, '[TC21] Vandermonde determinant FAILED'

# ---- TC22: Barycentric interpolation recovers node values exactly ----
nodes = np.array([0.0, 0.5, 1.0, 1.5])
vals = np.sin(nodes) + 0.1j * np.cos(nodes)
recovered = barycentric_lagrange_interpolate(nodes, vals, nodes)
assert np.allclose(recovered, vals, atol=1e-12), '[TC22] Barycentric node recovery FAILED'

# ---- TC23: Characteristic polynomial evaluates to zero at given roots ----
roots = np.array([1.0+1j, 2.0-0.5j, -0.5])
coeffs = characteristic_polynomial_from_roots(roots)
for r in roots:
    pval = np.polyval(coeffs, r)
    assert abs(pval) < 1e-10, '[TC23] Characteristic polynomial root FAILED'

# ---- TC24: Poeschl-Teller potential at x=0 is purely real and negative ----
V_pt = complex_poschl_teller(0.0, V0=1.0, W0=0.3, alpha=1.0)
assert abs(V_pt.imag) < 1e-12, '[TC24] PT potential imag at 0 FAILED'
assert V_pt.real < 0.0, '[TC24] PT potential real at 0 FAILED'

# ---- TC25: Kronig-Penney potential is periodic with period a ----
x = np.linspace(0.0, 2.0, 5)
a = 1.0
V1 = nonhermitian_kronig_penney(x, V1=1.0, V2=0.3, a=a)
V2 = nonhermitian_kronig_penney(x + a, V1=1.0, V2=0.3, a=a)
assert np.allclose(V1, V2, atol=1e-12), '[TC25] KP periodicity FAILED'

# ---- TC26: Double-well real part is symmetric V(x)=V(-x) ----
x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
V_dw = double_well_nonhermitian(x_test, V0=1.0, gamma=0.2, a=2.0, b=0.5)
assert np.allclose(V_dw.real, double_well_nonhermitian(-x_test, V0=1.0, gamma=0.2, a=2.0, b=0.5).real), '[TC26] Double-well symmetry FAILED'

# ---- TC27: Circle loop has correct shape and is nearly closed ----
loop = circle_loop((0.5, 0.3), 0.1, n_points=50)
assert loop.shape == (50, 2), '[TC27] Circle loop shape FAILED'
assert np.linalg.norm(loop[0] - loop[-1]) < 0.05, '[TC27] Circle loop closed FAILED'

# ---- TC28: ParameterBox volume and contains are consistent ----
box = ParameterBox([(-np.pi, np.pi), (0.0, 0.5)])
assert abs(box.volume() - (2*np.pi * 0.5)) < 1e-12, '[TC28] Box volume FAILED'
assert box.contains(box.center()), '[TC28] Box contains center FAILED'

# ---- TC29: Rectangle triangulation produces expected triangle count ----
pts, tri = triangulate_domain_rectangle((0.0, 1.0), (0.0, 1.0), nx=3, ny=3)
assert tri.shape[0] == (3-1)*(3-1)*2, '[TC29] Rectangle tri count FAILED'

# ---- TC30: Equilateral triangle quality equals 1 ----
eq_pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
q = triangle_quality(eq_pts, [0, 1, 2])
assert abs(q - 1.0) < 1e-10, '[TC30] Equilateral quality FAILED'

# ---- TC31: String-to-float vector parsing is correct ----
vec = string_to_float_vector("1.5 -2.3 4.0", 3)
expected = np.array([1.5, -2.3, 4.0])
assert np.allclose(vec, expected), '[TC31] String vector parse FAILED'

# ---- TC32: File char count returns -1 for missing file ----
assert file_char_count("nonexistent_file_xyz.txt") == -1, '[TC32] File char count missing FAILED'

# ---- TC33: Config parser reads numeric values correctly ----
import os
config_path = "test_config_tmp.txt"
with open(config_path, "w") as f:
    f.write("N_SITES 8\n")
    f.write("t hopping = 1.5\n")
params = read_parameter_config(config_path)
os.remove(config_path)
assert params.get('N_SITES') == 8, '[TC33] Config int FAILED'
assert params.get('t hopping') == 1.5, '[TC33] Config float FAILED'

# ---- TC34: Random search returns candidates below threshold ----
def dummy_disc(p):
    return abs(p.get('x', 0.0))
bounds = {'x': (-1.0, 1.0)}
np.random.seed(42)
cands = strategy_random_search(dummy_disc, bounds, n_trials=100, threshold=0.5, seed=42)
assert isinstance(cands, list), '[TC34] Random search type FAILED'
assert all(c['abs_delta'] < 0.5 for c in cands), '[TC34] Random search threshold FAILED'

# ---- TC35: Metropolis-Hastings accept rate lies in (0,1] ----
chain, acc_rate = metropolis_hastings_ep_search(dummy_disc, bounds, n_steps=500, beta=10.0, step=0.1, seed=42)
assert 0.0 < acc_rate <= 1.0, '[TC35] MCMC accept rate range FAILED'

# ---- TC36: Wigner-Poisson mixture CDF is monotonic for positive s ----
s_vals = np.linspace(0.0, 5.0, 50)
cdf_vals = wigner_poisson_mixture_cdf(s_vals, 0.5)
assert np.all(np.diff(cdf_vals) >= -1e-12), '[TC36] CDF monotonicity FAILED'

# ---- TC37: Non-degenerate Hamiltonian has EP order 1 ----
H_nep = build_pt_symmetric_hamiltonian_1d(1.0, t=1.0, m=0.5, gamma=0.1)
dH = lambda p: np.zeros((2,2), dtype=complex)
order = local_exceptional_point_order(H_nep, 0.0, dH)
assert order == 1, '[TC37] Non-EP order FAILED'

# ---- TC38: Importance sampling returns a list ----
np.random.seed(42)
cands_imp = strategy_importance_sampling(dummy_disc, bounds, n_trials=100, temperature=1.0, seed=42)
assert isinstance(cands_imp, list), '[TC38] Importance sampling type FAILED'

# ---- TC39: FE Hamiltonian shape matches number of mesh nodes ----
pts_fe, tri_fe = triangulate_domain_rectangle((-1.0, 1.0), (-1.0, 1.0), nx=4, ny=4)
mesh_fe = SimpleMesh(pts_fe, tri_fe)
H_fe, M_fe = assemble_nonhermitian_hamiltonian_fe(mesh_fe, V_func=lambda x,y: 0.0, W_func=lambda x,y: 0.0)
assert H_fe.shape == (mesh_fe.nodes.shape[0], mesh_fe.nodes.shape[0]), '[TC39] FE Hamiltonian shape FAILED'

# ---- TC40: Winding number is finite for gapped 1D PT model ----
W = winding_number_complex_energy(
    lambda k: build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=1.0, gamma=0.1),
    k_points=201
)
assert np.isfinite(W), '[TC40] Winding number finite FAILED'
