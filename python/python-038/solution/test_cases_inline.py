# ---- TC01: alpha_s_1loop returns small positive value at high Q2 ----
val = alpha_s_1loop(1e8)
assert 0.0 < val < 0.2, '[TC01] alpha_s_1loop returns small positive value at high Q2 FAILED'

# ---- TC02: alpha_s_2loop smaller than one-loop at same scale ----
a1 = alpha_s_1loop(100.0)
a2 = alpha_s_2loop(100.0)
assert 0.0 < a2 < a1, '[TC02] alpha_s_2loop smaller than one-loop at same scale FAILED'

# ---- TC03: p_qq_lo finite and positive at mid z ----
val = p_qq_lo(0.5)
assert np.isfinite(val) and val > 0, '[TC03] p_qq_lo finite and positive at mid z FAILED'

# ---- TC04: p_gq_lo and p_qg_lo finite in physical range ----
assert np.isfinite(p_gq_lo(0.3)) and np.isfinite(p_gg_lo(0.3)), '[TC04] p_gq_lo and p_gg_lo finite in physical range FAILED'

# ---- TC05: legendre_poly_vals exact P2 at 0.5 ----
leg = legendre_poly_vals(2, np.array([0.5]))
assert abs(leg[0, 2] - (-0.125)) < 1e-10, '[TC05] legendre_poly_vals exact P2 at 0.5 FAILED'

# ---- TC06: chebyshev_poly_vals exact T3 at 0.5 ----
cheb = chebyshev_poly_vals(3, np.array([0.5]))
assert abs(cheb[0, 3] - (-1.0)) < 1e-10, '[TC06] chebyshev_poly_vals exact T3 at 0.5 FAILED'

# ---- TC07: hermite_poly_vals exact H2 at 1.0 ----
herm = hermite_poly_vals(2, np.array([1.0]))
assert abs(herm[0, 2] - 2.0) < 1e-10, '[TC07] hermite_poly_vals exact H2 at 1.0 FAILED'

# ---- TC08: harmonic_sum H1 equals 1 ----
assert abs(harmonic_sum(1) - 1.0) < 1e-12, '[TC08] harmonic_sum H1 equals 1 FAILED'

# ---- TC09: di_log zero at origin ----
assert abs(di_log(0.0)) < 1e-12, '[TC09] di_log zero at origin FAILED'

# ---- TC10: sudakov_quark returns probability in [0,1] ----
s = sudakov_quark(100.0, 1.0)
assert 0.0 <= s <= 1.0, '[TC10] sudakov_quark returns probability in [0,1] FAILED'

# ---- TC11: r83_cyclic_reduction solves DIF2 exactly for sine ----
N = 64
A = build_dif2_r83(N)
x_ex = np.sin(np.linspace(0, np.pi, N))
b = np.zeros(N)
b[0] = A[1, 0] * x_ex[0] + A[0, 0] * x_ex[1]
for i in range(1, N - 1):
    b[i] = A[2, i] * x_ex[i - 1] + A[1, i] * x_ex[i] + A[0, i] * x_ex[i + 1]
b[-1] = A[2, -1] * x_ex[-2] + A[1, -1] * x_ex[-1]
x_sol = r83_cyclic_reduction(A, b)
assert np.max(np.abs(x_sol - x_ex)) < 1e-10, '[TC11] r83_cyclic_reduction solves DIF2 exactly for sine FAILED'

# ---- TC12: r83_cg_solve converges for DIF2 ----
x_cg, info_cg = r83_cg_solve(A, b)
assert info_cg['converged'] and np.max(np.abs(x_cg - x_ex)) < 1e-8, '[TC12] r83_cg_solve converges for DIF2 FAILED'

# ---- TC13: solve_diffusion_1d preserves non-negativity ----
u0 = np.exp(-5.0 * (np.linspace(0, 1, 32) - 0.5)**2)
u_fin = solve_diffusion_1d(u0, 0.1, 0.001, 1.0 / 31, 10, solver='cyclic')
assert np.all(u_fin >= -1e-12), '[TC13] solve_diffusion_1d preserves non-negativity FAILED'

# ---- TC14: integrate_1d_composite Simpson exact for quadratic ----
val = integrate_1d_composite(lambda x: x**2, 0.0, 1.0, n=100, rule='simpson')
assert abs(val - 1.0 / 3.0) < 1e-10, '[TC14] integrate_1d_composite Simpson exact for quadratic FAILED'

# ---- TC15: integrate_adaptive_1d for sqrt singularity ----
val = integrate_adaptive_1d(lambda x: np.sqrt(x), 0.0, 1.0, tol=1e-6)
assert abs(val - 2.0 / 3.0) < 1e-4, '[TC15] integrate_adaptive_1d for sqrt singularity FAILED'

# ---- TC16: integrate_monte_carlo approximate for 2D product ----
import numpy as np
np.random.seed(42)
val_mc, err_mc = integrate_monte_carlo(lambda x: x[0] * x[1], [(0.0, 1.0), (0.0, 1.0)], n_samples=20000, seed=42)
assert abs(val_mc - 0.25) < 5 * err_mc, '[TC16] integrate_monte_carlo approximate for 2D product FAILED'

# ---- TC17: Hilbert round-trip preserves indices ----
n_bits = 4
h_vals = np.arange(0, 2**(3 * n_bits), 7)
coords = hilbert_h_to_xyz(h_vals, n_bits)
h_back = hilbert_xyz_to_h(coords, n_bits)
assert np.all(h_vals == h_back), '[TC17] Hilbert round-trip preserves indices FAILED'

# ---- TC18: cvt_lloyd_2d returns correct generator shape ----
gens, info = cvt_lloyd_2d(8, lambda x, y: 1.0, n_samples=2000, max_iter=10, seed=42)
assert gens.shape == (8, 2) and info['iterations'] <= 10, '[TC18] cvt_lloyd_2d returns correct generator shape FAILED'

# ---- TC19: mellin_moment_splitting returns all four keys ----
P = mellin_moment_splitting(np.array([2.0 + 0.5j, 3.0 + 0.5j]))
assert set(P.keys()) == {'qq', 'gq', 'qg', 'gg'}, '[TC19] mellin_moment_splitting returns all four keys FAILED'

# ---- TC20: dglap_mellin_evolve produces finite output ----
N_vals = np.array([2.0 + 0.5j, 3.0 + 0.5j])
q0 = np.array([1.0 + 0j, 0.5 + 0j])
g0 = np.array([2.0 + 0j, 1.0 + 0j])
qf, gf = dglap_mellin_evolve(q0, g0, N_vals, 1.0, 100.0)
assert np.all(np.isfinite(qf)) and np.all(np.isfinite(gf)), '[TC20] dglap_mellin_evolve produces finite output FAILED'

# ---- TC21: pdf_initial_model non-negative on grid ----
x_g = np.linspace(0.01, 0.99, 20)
q_pdf, g_pdf = pdf_initial_model(x_g)
assert np.all(q_pdf >= 0) and np.all(g_pdf >= 0), '[TC21] pdf_initial_model non-negative on grid FAILED'

# ---- TC22: generate_hard_process produces two back-to-back partons ----
hard = generate_hard_process(E_cm=14000.0, pt_hard=50.0, seed=42)
assert len(hard) == 2 and abs(hard[0].pt - 50.0) < 1e-6, '[TC22] generate_hard_process produces two back-to-back partons FAILED'

# ---- TC23: sample_z_and_phi returns z within bounds ----
import numpy as np
np.random.seed(42)
rng = np.random.default_rng(42)
z_val, phi_val = sample_z_and_phi(rng, flavor='q', zmin=0.1, zmax=0.9)
assert 0.1 <= z_val <= 0.9 and 0.0 <= phi_val <= 2 * np.pi, '[TC23] sample_z_and_phi returns z within bounds FAILED'

# ---- TC24: shower_multiplicity_ode solution grows ----
sol = shower_multiplicity_ode((0.0, 5.0), 2.0, alpha=0.5, beta=0.05)
assert sol.success and sol.y[0, -1] > 2.0, '[TC24] shower_multiplicity_ode solution grows FAILED'

# ---- TC25: PseudoJet addition conserves 4-momentum ----
p1 = PseudoJet(1.0, 2.0, 3.0, 5.0)
p2 = PseudoJet(2.0, 1.0, 0.0, 3.0)
p_sum = p1 + p2
assert p_sum.px == 3.0 and p_sum.py == 3.0 and p_sum.pz == 3.0 and p_sum.E == 8.0, '[TC25] PseudoJet addition conserves 4-momentum FAILED'

# ---- TC26: cluster_jets returns empty list for empty input ----
empty_jets = cluster_jets([], R=0.4, p=-1, pt_min=5.0)
assert empty_jets == [], '[TC26] cluster_jets returns empty list for empty input FAILED'

# ---- TC27: cluster_jets returns jets above pt_min with anti-kT ----
import numpy as np
np.random.seed(42)
rng = np.random.default_rng(42)
particles = []
for _ in range(15):
    pt = rng.exponential(10.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = 0.0
    E = pt
    particles.append(PseudoJet(px, py, pz, E))
jets = cluster_jets(particles, R=0.4, p=-1, pt_min=5.0)
assert len(jets) >= 1 and all(j.pt >= 5.0 - 1e-6 for j in jets), '[TC27] cluster_jets returns jets above pt_min with anti-kT FAILED'

# ---- TC28: compute_thrust in valid range for collinear particles ----
particles = [PseudoJet(10.0, 0.0, 0.0, 10.0), PseudoJet(5.0, 0.0, 0.0, 5.0), PseudoJet(0.0, 2.0, 0.0, 2.0)]
T, axis = compute_thrust(particles)
assert 0.5 <= T <= 1.0 + 1e-6, '[TC28] compute_thrust in valid range for collinear particles FAILED'

# ---- TC29: compute_sphericity eigenvalues sum to one ----
particles = [PseudoJet(1.0, 0.0, 0.0, 1.0), PseudoJet(0.0, 1.0, 0.0, 1.0), PseudoJet(0.0, 0.0, 1.0, 1.0)]
spher = compute_sphericity(particles)
assert abs(np.sum(spher['eigenvalues']) - 1.0) < 1e-6, '[TC29] compute_sphericity eigenvalues sum to one FAILED'

# ---- TC30: compute_jet_broadening non-negative ----
particles = [PseudoJet(1.0, 0.0, 0.0, 1.0), PseudoJet(0.0, 1.0, 0.0, 1.0)]
T, axis = compute_thrust(particles)
B = compute_jet_broadening(particles, axis)
assert B >= 0.0, '[TC30] compute_jet_broadening non-negative FAILED'

# ---- TC31: LegendrePCE mean exact for quadratic function ----
pce = LegendrePCE(order=5)
pce.fit_projection(lambda xi: xi**2, n_quad=12)
assert abs(pce.mean() - 1.0 / 3.0) < 1e-6, '[TC31] LegendrePCE mean exact for quadratic function FAILED'

# ---- TC32: pce_pdf_uncertainty returns non-negative mean ----
pce_pdf = pce_pdf_uncertainty(lambda x, lam: 0.5 * x**(-lam) * (1.0 - x)**4, x_value=0.1, param_range=(0.25, 0.35), order=4)
assert pce_pdf.mean() >= 0.0, '[TC32] pce_pdf_uncertainty returns non-negative mean FAILED'

# ---- TC33: global_sensitivity_analysis Sobol indices in [0,1] ----
def dummy_model(params):
    return params.get('a', 0.5) * 2 + params.get('b', 0.5) * 3
gs = global_sensitivity_analysis(dummy_model, {'a': (0.0, 1.0), 'b': (0.0, 1.0)}, order=2, n_mc=2000, seed=42)
assert all(0.0 <= v <= 1.0 for v in gs['sobol_indices'].values()), '[TC33] global_sensitivity_analysis Sobol indices in [0,1] FAILED'

# ---- TC34: dglap_spectral_evolve_gluon produces finite non-negative output ----
x_grid = np.logspace(-3, -0.05, 30)
g0_func = lambda xi: np.maximum(2.0 * xi**(-0.3) * (1.0 - xi)**5, 1e-15)
g_ev = dglap_spectral_evolve_gluon(g0_func, x_grid, 1.0, 100.0, nf=5)
assert np.all(np.isfinite(g_ev)) and np.all(g_ev >= 0), '[TC34] dglap_spectral_evolve_gluon produces finite non-negative output FAILED'

# ---- TC35: Hadron properties computed correctly ----
h = Hadron(3.0, 4.0, 0.0, 6.0, pid=211, charge=1)
assert abs(h.pt - 5.0) < 1e-12 and h.mass >= 0, '[TC35] Hadron properties computed correctly FAILED'
