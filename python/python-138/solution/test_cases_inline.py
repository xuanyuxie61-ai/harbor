# ---- TC01: MicroreactorPDESolver construction and basic attributes ----
pde = MicroreactorPDESolver(L=0.05, Nx=120, D_m=2.0e-9, u=0.02, A_arr=5.0e7,
    Ea=48000.0, reaction_order=1.0, rho=980.0, cp=4200.0, lam=0.55, dH=-7.5e4,
    T_wall=360.0, h_wall=600.0, hydraulic_diameter=4.0e-4, C_in=800.0, T_in=310.0)
assert pde.Nx == 120, '[TC01] Nx mismatch FAILED'
assert pde.dx == 0.05 / 119, '[TC01] dx mismatch FAILED'
assert pde.C_in == 800.0, '[TC01] C_in mismatch FAILED'

# ---- TC02: MicroreactorPDESolver reaction_rate finite and non-negative ----
C_test = np.array([500.0, 0.0, 1000.0])
T_test = np.array([350.0, 350.0, 350.0])
r = pde.reaction_rate(C_test, T_test)
assert np.all(np.isfinite(r)), '[TC02] reaction_rate produced non-finite values FAILED'
assert np.all(r >= 0.0), '[TC02] reaction_rate produced negative values FAILED'
assert r[1] == 0.0, '[TC02] reaction_rate for zero concentration should be zero FAILED'

# ---- TC03: MicroreactorPDESolver steady_state solve produces correct shape ----
C_ss_test, T_ss_test = pde.solve_steady_state(max_iter=3000, tol=1.0e-9)
assert len(C_ss_test) == 120, '[TC03] C_ss_test shape mismatch FAILED'
assert len(T_ss_test) == 120, '[TC03] T_ss_test shape mismatch FAILED'
assert C_ss_test[0] == 800.0, '[TC03] inlet concentration boundary FAILED'
assert np.all(np.isfinite(C_ss_test)), '[TC03] C_ss_test contains NaN/Inf FAILED'
assert np.all(np.isfinite(T_ss_test)), '[TC03] T_ss_test contains NaN/Inf FAILED'

# ---- TC04: MicroreactorPDESolver conversion in valid range ----
X_conv_test, C_out_test = pde.compute_conversion_and_yield(C_ss_test)
assert 0.0 <= X_conv_test <= 1.0, '[TC04] Conversion out of [0,1] FAILED'
assert C_out_test >= 0.0, '[TC04] outlet concentration negative FAILED'

# ---- TC05: MicroreactorPDESolver Peclet/Damkohler finite positive ----
Pe_test, Da_test = pde.compute_peclet_damkohler(C_ss_test, T_ss_test)
assert np.isfinite(Pe_test) and Pe_test > 0.0, '[TC05] Peclet number invalid FAILED'
assert np.isfinite(Da_test) and Da_test > 0.0, '[TC05] Damkohler number invalid FAILED'

# ---- TC06: CatalystCVTPlacer construction and initial generators in bounds ----
cvt = CatalystCVTPlacer(dim=2, n_generators=25, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=5000, max_iter=10, tol=1.0e-5)
assert cvt.generators.shape == (25, 2), '[TC06] generator shape mismatch FAILED'
assert np.all(cvt.generators >= 0.0) and np.all(cvt.generators <= 1.0), '[TC06] generators out of bounds FAILED'

# ---- TC07: CatalystCVTPlacer iterate produces finite energy and max_shift >= 0 ----
import numpy as np
np.random.seed(42)
cvt2 = CatalystCVTPlacer(dim=2, n_generators=16, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=4000, max_iter=8, tol=1.0e-5)
gens2, energy2, max_shift2 = cvt2.iterate()
assert np.isfinite(energy2), '[TC07] CVT energy not finite FAILED'
assert max_shift2 >= 0.0, '[TC07] max_shift negative FAILED'
assert gens2.shape == (16, 2), '[TC07] generator shape after iteration mismatch FAILED'

# ---- TC08: CatalystCVTPlacer uniformity index in [0,1] ----
np.random.seed(42)
cvt3 = CatalystCVTPlacer(dim=2, n_generators=16, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=4000, max_iter=8, tol=1.0e-5)
cvt3.iterate()
eta = cvt3.compute_uniformity_index()
assert 0.0 <= eta <= 1.0, '[TC08] Uniformity index out of [0,1] FAILED'

# ---- TC09: KineticsParameterEstimator QR factorization correctness ----
ke = KineticsParameterEstimator(R_gas=8.314)
A_qr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Q, R = ke.qr_factorize(A_qr)
reconstructed = Q @ R
assert np.allclose(reconstructed, A_qr, atol=1e-10), '[TC09] QR factorization reconstruction mismatch FAILED'
I_check = Q.T @ Q
assert np.allclose(I_check, np.eye(3), atol=1e-10), '[TC09] Q not orthogonal FAILED'

# ---- TC10: KineticsParameterEstimator least squares solve ----
A_ls = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b_ls = np.array([1.0, 2.0, 3.0])
x_ls = ke.solve_least_squares(A_ls, b_ls)
assert len(x_ls) == 2, '[TC10] LS solution dimension mismatch FAILED'

# ---- TC11: KineticsParameterEstimator Arrhenius estimation finite outputs ----
np.random.seed(42)
n_est_data = 30
T_est = np.linspace(310.0, 390.0, n_est_data)
C_est = np.linspace(100.0, 500.0, n_est_data)
r_true_est = 5.0e4 * np.exp(-12000.0/(8.314*T_est)) * (C_est**0.85)
noise_est = np.random.normal(0, 0.02*np.median(r_true_est), n_est_data)
r_est = np.maximum(r_true_est + noise_est, 1e-3)
A_est, Ea_est, n_est, res_norm = ke.estimate_arrhenius_parameters(C_est, T_est, r_est)
assert np.isfinite(A_est) and A_est > 0.0, '[TC11] A_est invalid FAILED'
assert np.isfinite(Ea_est) and Ea_est > 0.0, '[TC11] Ea_est invalid FAILED'
assert np.isfinite(n_est) and n_est >= 0.0, '[TC11] n_est invalid FAILED'
assert np.isfinite(res_norm) and res_norm >= 0.0, '[TC11] residual norm invalid FAILED'

# ---- TC12: ReactorStabilityAnalyzer Jacobian shape ----
rsa = ReactorStabilityAnalyzer(Nx=30, L=0.05, D_m=2.0e-9, alpha=0.55/(980.0*4200.0),
    u=0.02, dH=-7.5e4, rho=980.0, cp=4200.0, h_wall=600.0, d_h=4.0e-4)
C_steady_test = np.linspace(800.0, 400.0, 30)
T_steady_test = np.linspace(310.0, 360.0, 30)
J_stab = rsa.compute_jacobian(C_steady_test, T_steady_test, A_arr=5.0e7, Ea=48000.0, n_order=1.0)
assert J_stab.shape == (60, 60), '[TC12] Jacobian shape mismatch FAILED'
assert np.all(np.isfinite(J_stab)), '[TC12] Jacobian contains NaN/Inf FAILED'

# ---- TC13: ReactorStabilityAnalyzer thermal explosion index in [0,1) ----
risk_idx = rsa.compute_thermal_explosion_index(-0.5)
assert 0.0 <= risk_idx <= 1.0, '[TC13] risk index for stable case FAILED'
assert risk_idx == 0.0, '[TC13] risk index should be 0 for negative max_real FAILED'

# ---- TC14: MassEnergyBalanceSolver rate_constant positive finite ----
cstr = MassEnergyBalanceSolver(C0=800.0, T0=310.0, Q=2.0e-6, V=1.0e-5,
    rho=980.0, cp=4200.0, dH=-7.5e4, Ua=0.8, Tc=350.0, A_arr=1.0e8, Ea=50000.0, reaction_order=1.0)
k_rate = cstr.rate_constant(350.0)
assert np.isfinite(k_rate) and k_rate > 0.0, '[TC14] rate constant invalid FAILED'

# ---- TC15: MassEnergyBalanceSolver concentration_from_T in valid range ----
C_from_T = cstr.concentration_from_T(350.0)
assert np.isfinite(C_from_T) and C_from_T >= 0.0, '[TC15] C_from_T invalid FAILED'
assert C_from_T <= cstr.C0, '[TC15] C_from_T exceeds C0 FAILED'

# ---- TC16: MassEnergyBalanceSolver fixed point solve returns valid state ----
np.random.seed(42)
cstr2 = MassEnergyBalanceSolver(C0=800.0, T0=310.0, Q=2.0e-6, V=1.0e-5,
    rho=980.0, cp=4200.0, dH=-7.5e4, Ua=0.8, Tc=350.0,
    A_arr=5.0e7, Ea=48000.0, reaction_order=1.0)
T_cstr_test, C_cstr_test, it_cstr, conv_cstr = cstr2.solve_fixed_point()
assert np.isfinite(T_cstr_test) and T_cstr_test > 200.0, '[TC16] T_cstr_test invalid FAILED'
assert np.isfinite(C_cstr_test) and C_cstr_test >= 0.0, '[TC16] C_cstr_test invalid FAILED'
assert conv_cstr, '[TC16] fixed point did not converge FAILED'

# ---- TC17: MassEnergyBalanceSolver bifurcation indicator finite ----
bif_idx = cstr2.bifurcation_indicator()
assert np.isfinite(bif_idx) and bif_idx >= 0.0, '[TC17] bifurcation indicator invalid FAILED'

# ---- TC18: ReducedOrderBasisBuilder MGS returns correct rank and basis shape ----
np.random.seed(42)
N_dim = 50
m_snap = 15
snapshots_test = np.random.randn(N_dim, m_snap)
rom = ReducedOrderBasisBuilder(tolerance=1.0e-10)
basis_mgs, rank_mgs, err_mgs = rom.modified_gram_schmidt(snapshots_test)
assert basis_mgs.shape[0] == N_dim, '[TC18] basis rows mismatch FAILED'
assert basis_mgs.shape[1] == rank_mgs, '[TC18] basis columns != rank FAILED'
assert rank_mgs <= m_snap, '[TC18] rank exceeds number of snapshots FAILED'
assert err_mgs >= 0.0, '[TC18] orth error negative FAILED'

# ---- TC19: ReducedOrderBasisBuilder POD SVD energy ratio in [0,1] ----
pod_basis, svals, energy_ratio = rom.compute_pod_modes_svd(snapshots_test)
assert len(svals) > 0, '[TC19] no singular values returned FAILED'
assert 0.0 <= energy_ratio <= 1.0, '[TC19] energy ratio out of [0,1] FAILED'
assert pod_basis.shape[0] == N_dim, '[TC19] POD basis rows mismatch FAILED'

# ---- TC20: ReducedOrderBasisBuilder projection-reconstruction cycle ----
if rank_mgs > 0:
    field_test = snapshots_test[:, 0]
    coeffs = rom.project_onto_basis(field_test, basis_mgs)
    reconstructed = rom.reconstruct_from_basis(coeffs, basis_mgs)
    assert len(reconstructed) == N_dim, '[TC20] reconstruction length mismatch FAILED'

# ---- TC21: MixingQualityAnalyzer mixing defect non-negative ----
np.random.seed(42)
mixer = MixingQualityAnalyzer(ideal_mean=500.0, ideal_std=40.0)
samples_test = mixer.sample_concentration_field(n_samples=500, dim=2, mixing_efficiency=0.85)
assert samples_test.shape == (500, 2), '[TC21] sample shape mismatch FAILED'
M_d = mixer.compute_mixing_defect(samples_test)
assert np.isfinite(M_d) and M_d >= 0.0, '[TC21] mixing defect invalid FAILED'

# ---- TC22: MixingQualityAnalyzer efficiency index in [0,1] ----
eta_mix = mixer.compute_mixing_efficiency_index(samples_test)
assert 0.0 <= eta_mix <= 1.0, '[TC22] mixing efficiency index out of [0,1] FAILED'

# ---- TC23: MixingQualityAnalyzer zone distance finite ----
np.random.seed(42)
zone_a = samples_test[:250]
zone_b = samples_test[250:]
D_M, D_B = mixer.statistical_distance_between_zones(zone_a, zone_b)
assert np.isfinite(D_M) and D_M >= 0.0, '[TC23] Mahalanobis distance invalid FAILED'
assert np.isfinite(D_B), '[TC23] Bhattacharyya distance not finite FAILED'

# ---- TC24: MixingQualityAnalyzer multivariate CV non-negative ----
cv_multi = mixer.compute_multivariate_cv(samples_test)
assert np.isfinite(cv_multi) and cv_multi >= 0.0, '[TC24] multivariate CV invalid FAILED'

# ---- TC25: ReactorOptimizer golden section search on quadratic ----
opt = ReactorOptimizer()
def f_quad(x):
    return (x - 3.0)**2
x_opt_gss, f_opt_gss, it_gss, nf_gss = opt.golden_section_search(f_quad, 0.0, 10.0, max_iter=50, x_tol=1e-8)
assert np.abs(x_opt_gss - 3.0) < 1e-4, '[TC25] golden section failed to find minimum of (x-3)^2 FAILED'
assert np.abs(f_opt_gss) < 1e-7, '[TC25] golden section minimum value not near zero FAILED'

# ---- TC26: ReactorOptimizer Rosenbrock objective and gradient consistency ----
np.random.seed(42)
x0_ros = np.array([-1.0, 0.5, 0.3])
f0, g0, H0 = opt.reactor_objective_rosenbrock_like(x0_ros)
assert np.isfinite(f0), '[TC26] Rosenbrock objective not finite FAILED'
assert len(g0) == 3, '[TC26] gradient dimension mismatch FAILED'
assert H0.shape == (3, 3), '[TC26] Hessian shape mismatch FAILED'
assert np.all(np.isfinite(g0)), '[TC26] gradient contains NaN/Inf FAILED'

# ---- TC27: SkylineMatrixOperator build and to_dense symmetry ----
n_sky = 30
lower_s = np.ones(n_sky) * (-1.0); lower_s[0] = 0.0
diag_s = np.ones(n_sky) * 2.01
upper_s = np.ones(n_sky) * (-1.0); upper_s[-1] = 0.0
sky = SkylineMatrixOperator(n_sky)
sky.build_from_tridiagonal(lower_s, diag_s, upper_s)
A_dense_s = sky.to_dense()
assert A_dense_s.shape == (n_sky, n_sky), '[TC27] dense matrix shape mismatch FAILED'
assert np.allclose(A_dense_s, A_dense_s.T, atol=1e-14), '[TC27] matrix not symmetric FAILED'

# ---- TC28: SkylineMatrixOperator multiply correctness ----
x_sky = np.ones(n_sky)
y_sky = sky.multiply(x_sky)
y_dense = A_dense_s @ x_sky
assert np.allclose(y_sky, y_dense, atol=1e-12), '[TC28] skyline multiply does not match dense multiply FAILED'

# ---- TC29: SkylineMatrixOperator solve residual small ----
x_solve_sky = sky.solve_cholesky_skyline(x_sky)
residual = np.linalg.norm(A_dense_s @ x_solve_sky - x_sky)
assert residual < 1e-8, '[TC29] skyline solve residual too large FAILED'

# ---- TC30: SkylineMatrixOperator condition number finite ----
cond_sky = sky.condition_number_estimate()
assert np.isfinite(cond_sky) and cond_sky >= 1.0, '[TC30] condition number invalid FAILED'

# ---- TC31: MicroreactorNetworkTopology degrees and Eulerian ----
net = MicroreactorNetworkTopology(n_nodes=6)
edges_test = [(0,1), (0,2), (1,3), (1,4), (2,5)]
for u, v in edges_test:
    net.add_edge(u, v)
indeg, outdeg = net.compute_degrees()
assert len(indeg) == 6 and len(outdeg) == 6, '[TC31] degree array length mismatch FAILED'

# ---- TC32: MicroreactorNetworkTopology spanning tree count non-negative ----
n_span_test = net.count_spanning_trees()
assert n_span_test >= 0, '[TC32] spanning tree count negative FAILED'

# ---- TC33: MicroreactorNetworkTopology uniform tree edges from pruefer roundtrip ----
tree_edges = net.generate_optimal_distribution_tree(root=0)
pruefer_code = net.tree_to_pruefer(tree_edges)
assert len(pruefer_code) == net.n - 2, '[TC33] Pruefer code length mismatch FAILED'

# ---- TC34: ThermalStressAnalyzer biharmonic residual near zero ----
thermal = ThermalStressAnalyzer(E=210.0e9, nu=0.28, h=0.8e-3, alpha_T=1.1e-5, z_eval=0.4e-3)
nx_t = 20
x_t = np.linspace(-0.01, 0.01, nx_t)
y_t = np.linspace(-0.005, 0.005, nx_t)
Xg_t, Yg_t = np.meshgrid(x_t, y_t)
resid_biharm = thermal.biharmonic_residual(Xg_t, Yg_t, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0)
assert np.all(np.isfinite(resid_biharm)), '[TC34] biharmonic residual contains NaN/Inf FAILED'

# ---- TC35: ThermalStressAnalyzer stresses finite and safety factor positive ----
delta_T_test = 50.0 * np.exp(-(Xg_t**2 + Yg_t**2) / (0.008**2))
sigma_x, sigma_y, tau_xy, sigma_vm = thermal.compute_thermal_stresses(
    Xg_t, Yg_t, delta_T_test, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0)
assert np.all(np.isfinite(sigma_x)), '[TC35] sigma_x contains NaN/Inf FAILED'
assert np.all(np.isfinite(sigma_vm)), '[TC35] sigma_vm contains NaN/Inf FAILED'
sf = thermal.safety_factor(sigma_vm, yield_strength=300.0e6)
assert np.isfinite(sf) and sf > 0.0, '[TC35] safety factor invalid FAILED'

# ---- TC36: ThermalStressAnalyzer thermal shock parameter positive ----
theta_shock = thermal.thermal_shock_parameter(delta_T_max=np.max(delta_T_test))
assert np.isfinite(theta_shock) and theta_shock >= 0.0, '[TC36] thermal shock parameter invalid FAILED'

# ---- TC37: DiscreteCatalystLoadingOptimizer greedy solution satisfies budget ----
a_load = np.array([2.5, 3.0, 4.0, 5.5, 6.0])
budget_load = 50.0
dioph = DiscreteCatalystLoadingOptimizer(a_load, budget_load)
weights_obj = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
x_greedy, obj_greedy = dioph.greedy_heuristic_solution(weights_obj)
total_used = np.dot(a_load, x_greedy)
assert total_used <= budget_load + 1e-8, '[TC37] greedy solution exceeds budget FAILED'
assert np.all(x_greedy >= 0), '[TC37] greedy solution has negative loads FAILED'

# ---- TC38: DiscreteCatalystLoadingOptimizer loading efficiency in [0,1] ----
util_g, unif_g = dioph.compute_loading_efficiency(x_greedy)
assert 0.0 <= util_g <= 1.0, '[TC38] utilization out of [0,1] FAILED'
assert 0.0 <= unif_g <= 1.0, '[TC38] uniformity out of [0,1] FAILED'

# ---- TC39: KineticsParameterEstimator confidence intervals shape ----
A_ci = np.random.randn(20, 4)
b_ci = np.random.randn(20)
np.random.seed(42)
x_ci = ke.solve_least_squares(A_ci, b_ci)
std_dev = ke.compute_confidence_intervals(A_ci, b_ci, x_ci)
assert len(std_dev) == len(x_ci), '[TC39] confidence interval length mismatch FAILED'

# ---- TC40: CatalystCVTPlacer get_catalyst_loading_map shape ----
np.random.seed(42)
cvt_map = CatalystCVTPlacer(dim=2, n_generators=9, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=3000, max_iter=5, tol=1.0e-4)
cvt_map.iterate()
density_map, coords = cvt_map.get_catalyst_loading_map(grid_res=30)
assert density_map.shape == (30, 30), '[TC40] density map shape mismatch FAILED'
assert coords.shape == (30, 30, 2), '[TC40] coords shape mismatch FAILED'
