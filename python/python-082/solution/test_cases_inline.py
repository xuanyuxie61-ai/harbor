# ---- TC01: compute_condition_number of identity matrix equals 1 ----
I3 = np.eye(3)
cond_I3 = compute_condition_number(I3)
assert cond_I3 == 1.0, '[TC01] compute_condition_number of identity matrix FAILED'

# ---- TC02: compute_normalized_residual for exact solution is small ----
A_test = np.array([[2.0, 1.0], [1.0, 3.0]])
x_exact = np.array([1.0, 2.0])
b_test = A_test @ x_exact
norm_res = compute_normalized_residual(A_test, x_exact, b_test)
assert norm_res < 1.0, '[TC02] compute_normalized_residual for exact solution FAILED'

# ---- TC03: file_row_count for non-existent file returns 0 ----
n_rows = file_row_count('nonexistent_file_xyz.txt')
assert n_rows == 0, '[TC03] file_row_count for non-existent file FAILED'

# ---- TC04: file_column_count for non-existent file returns 0 ----
n_cols = file_column_count('nonexistent_file_xyz.txt')
assert n_cols == 0, '[TC04] file_column_count for non-existent file FAILED'

# ---- TC05: RVE fiber volume fraction lies in valid range [0,1] ----
rve_test = generate_hexagonal_fiber_rve(width=100.0, height=100.0, fiber_radius=5.0, n_fibers_x=2, n_fibers_y=2)
V_f_test = rve_test.fiber_volume_fraction()
assert 0.0 <= V_f_test <= 1.0, '[TC05] RVE fiber volume fraction range FAILED'

# ---- TC06: point_in_polygon detects point at fiber center as inside ----
fiber = rve_test.fibers[0]
inside = RVEGeometry._point_in_polygon(fiber.center[0], fiber.center[1], fiber.nodes)
assert inside == True, '[TC06] point_in_polygon interior detection FAILED'

# ---- TC07: point_in_polygon detects distant point as outside ----
outside = RVEGeometry._point_in_polygon(-1000.0, -1000.0, fiber.nodes)
assert outside == False, '[TC07] point_in_polygon exterior detection FAILED'

# ---- TC08: CompositeMaterial E1 equals Voigt mixture rule ----
mat_test = create_carbon_epoxy(V_f=0.5)
E1_expected = 230.0 * 0.5 + 3.5 * 0.5
assert abs(mat_test.E1 - E1_expected) < 1e-6, '[TC08] CompositeMaterial E1 Voigt rule FAILED'

# ---- TC09: transformed stiffness at 0° equals Q matrix ----
Q_0 = mat_test.compute_transformed_stiffness(0.0)
assert np.allclose(Q_0, mat_test.Q), '[TC09] transformed stiffness at 0° FAILED'

# ---- TC10: transformed stiffness at 90° swaps Q11 and Q22 ----
Q_90 = mat_test.compute_transformed_stiffness(90.0)
assert abs(Q_90[0,0] - mat_test.Q[1,1]) < 1e-6 and abs(Q_90[1,1] - mat_test.Q[0,0]) < 1e-6, '[TC10] transformed stiffness at 90° symmetry FAILED'

# ---- TC11: degraded stiffness with zero damage returns original Q ----
Q_deg_zero = mat_test.compute_degraded_stiffness(0.0, 0.0, 0.0)
assert np.allclose(Q_deg_zero, mat_test.Q), '[TC11] degraded stiffness with zero damage FAILED'

# ---- TC12: hashin_failure_criteria for safe stress returns all factors < 1 ----
params_test = DamageParameters()
safe_stress = np.array([100.0, 10.0, 5.0])
criteria_safe = hashin_failure_criteria(safe_stress, params_test)
assert all(v < 1.0 for v in criteria_safe.values()), '[TC12] hashin safe stress criteria FAILED'

# ---- TC13: hashin_failure_criteria for failing stress returns factor >= 1 ----
fail_stress = np.array([3000.0, 100.0, 150.0])
criteria_fail = hashin_failure_criteria(fail_stress, params_test)
assert any(v >= 1.0 for v in criteria_fail.values()), '[TC13] hashin failing stress criteria FAILED'

# ---- TC14: estimate_damage_period returns finite positive value ----
stress_test = np.array([1000.0, 50.0, 30.0])
N_est = estimate_damage_period(stress_test, params_test)
assert N_est > 0 and N_est < np.inf, '[TC14] estimate_damage_period finite FAILED'

# ---- TC15: DamageState with low damage is not failed ----
d_low = DamageState(d_f=0.1, d_m=0.1, d_s=0.1, d_i=0.1)
assert d_low.is_failed() == False, '[TC15] DamageState low damage not failed FAILED'

# ---- TC16: LaminateStiffness total thickness equals sum of ply thicknesses ----
plies_test = [0.0, 45.0, -45.0, 90.0]
thick_test = [0.125, 0.125, 0.125, 0.125]
lam_test = LaminateStiffness(plies_test, thick_test, mat_test)
assert abs(lam_test.get_total_thickness() - 0.5) < 1e-10, '[TC16] LaminateStiffness total thickness FAILED'

# ---- TC17: BandedUpperTriangularSolver solve-then-multiply reconstructs b ----
n_band = 5
mu = 2
A_band = np.zeros((mu + 1, n_band))
for i in range(n_band):
    A_band[mu, i] = 2.0
    if i + 1 < n_band:
        A_band[mu - 1, i + 1] = 0.5
b_band = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
solver = BandedUpperTriangularSolver(n_band, mu)
x_sol = solver.solve(A_band, b_band)
b_recon = solver.multiply(A_band, x_sol)
assert np.allclose(b_band, b_recon), '[TC17] BandedUpperTriangularSolver reconstruction FAILED'

# ---- TC18: solve_equilibrium_dense produces small residual norm ----
np.random.seed(42)
K_dense = np.random.randn(10, 10)
K_dense = K_dense.T @ K_dense + 0.5 * np.eye(10)
F_dense = np.random.randn(10)
U_dense, r_norm, norm_res, cond_K = solve_equilibrium_dense(K_dense, F_dense)
assert r_norm < 1e-8, '[TC18] solve_equilibrium_dense residual FAILED'

# ---- TC19: SparseStiffnessAssembler CSR matrix has correct shape ----
assembler = SparseStiffnessAssembler(n_nodes=4, ndof_per_node=2)
k_e = np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
assembler.add_element_stiffness([0, 1], k_e)
K_csr = assembler.get_csr_matrix()
assert K_csr.shape == (8, 8), '[TC19] SparseStiffnessAssembler CSR shape FAILED'

# ---- TC20: compute_strain_energy_release_rate_quadrature returns scalar ----
U_test = compute_strain_energy_release_rate_quadrature(
    np.array([100.0, 50.0, 0.0]),
    np.array([0.01, 0.005, 0.0]),
    0.1, mat_test, 1.0, n_quad=4)
assert np.isscalar(U_test), '[TC20] strain energy release rate scalar FAILED'

# ---- TC21: Gauss-Legendre integrates x^2 exactly on [0,1] ----
x_gl, w_gl = gauss_legendre_nodes_weights(3, 0.0, 1.0)
integral_x2 = np.sum(w_gl * x_gl ** 2)
assert abs(integral_x2 - 1.0 / 3.0) < 1e-12, '[TC21] gauss_legendre x^2 exactness FAILED'

# ---- TC22: Hermite monomial integral for odd degree equals zero ----
h_odd = hermite_monomial_integral(3, option=1)
assert h_odd == 0.0, '[TC22] hermite_monomial_integral odd degree FAILED'

# ---- TC23: Hermite monomial integral for even degree is positive ----
h_even = hermite_monomial_integral(4, option=1)
assert h_even > 0.0, '[TC23] hermite_monomial_integral even degree FAILED'

# ---- TC24: VCCT energy release rates are positive ----
G_I, G_II = compute_vcct_energy_release_rate(80.0, 0.05, 2.0, n_quad=8)
assert G_I > 0.0 and G_II > 0.0, '[TC24] VCCT energy release rate positive FAILED'

# ---- TC25: probabilistic strength integral returns positive value ----
E_str = probabilistic_strength_integral(mean_strength=2500.0, std_strength=200.0)
assert E_str > 0.0, '[TC25] probabilistic_strength_integral positive FAILED'

# ---- TC26: stress wave reflection coefficient magnitude <= 1 ----
R_ref, T_ref = compute_stress_wave_reflection_coefficient(E1=200e3, E2=150e3, rho1=1600.0, rho2=1400.0)
assert abs(R_ref) <= 1.0, '[TC26] reflection coefficient magnitude FAILED'

# ---- TC27: generate_symmetric_eigenproblem produces correct eigenvalues ----
np.random.seed(42)
A_sym, Q_sym, lambda_sym = generate_symmetric_eigenproblem(5, lambda_mean=2.0, lambda_dev=0.5)
eigvals_sym = np.linalg.eigvalsh(A_sym)
eigvals_sym_sorted = np.sort(eigvals_sym)[::-1]
assert np.allclose(eigvals_sym_sorted, lambda_sym, atol=1e-8), '[TC27] symmetric eigenproblem eigenvalues FAILED'

# ---- TC28: generate_nonsymmetric_eigenproblem preserves trace ----
np.random.seed(42)
A_nsym, Q_nsym, T_nsym = generate_nonsymmetric_eigenproblem(4, lambda_mean=1.0, lambda_dev=0.2)
trace_A = np.trace(A_nsym)
trace_T = np.trace(T_nsym)
assert abs(trace_A - trace_T) < 1e-10, '[TC28] nonsymmetric eigenproblem trace FAILED'

# ---- TC29: rk4_stability_function at z=0 equals 1 ----
R_0 = rk4_stability_function(0.0)
assert R_0 == 1.0, '[TC29] rk4_stability_function at z=0 FAILED'

# ---- TC30: stable eigenvalue lies inside RK4 stability region ----
stable_eig = np.array([-0.5 + 0.0j])
flags = check_eigenvalue_in_stability_region(stable_eig, dt=1.0, R_func=rk4_stability_function)
assert flags[0] == True, '[TC30] stable eigenvalue check FAILED'

# ---- TC31: compute_ply_combinations count is correct ----
n_combos = compute_ply_combinations(4, [0, 45, -45, 90])
assert n_combos == 256, '[TC31] compute_ply_combinations count FAILED'

# ---- TC32: DGSpectralElement1D wave speed is everywhere positive ----
dg_test = DGSpectralElement1D(N=2, K=3, x_bounds=[0.0, 10.0], rho=1.6, E_func=lambda x: 2000.0)
c_vals = dg_test.compute_wave_speed()
assert np.all(c_vals > 0.0), '[TC32] DG wave speed positive FAILED'

# ---- TC33: WavePropagation1D wave speed is everywhere positive ----
wave_test = WavePropagation1D(L=50.0, nx=11, E_func=lambda x: 100e3, rho=1600.0, damping_ratio=0.02, forcing_params=(0.0, 0.0, 25.0))
c_wave = wave_test.compute_wave_speed()
assert np.all(c_wave > 0.0), '[TC33] WavePropagation1D wave speed positive FAILED'

# ---- TC34: WavePropagation1D attenuation coefficient is non-negative ----
alpha_att = wave_test.compute_attenuation_coefficient(50000.0)
assert alpha_att >= 0.0, '[TC34] WavePropagation1D attenuation non-negative FAILED'

# ---- TC35: BucklingAnalysis bending stiffness matrix has correct shape ----
D_test = np.eye(3) * 1e3
buckling_test = BucklingAnalysis(D_test, plate_length=100.0, plate_width=50.0, nx=8, ny=8)
K_b = buckling_test.build_bending_stiffness_matrix()
assert K_b.shape == (64, 64), '[TC35] BucklingAnalysis K_b shape FAILED'

# ---- TC36: integrate_damage_cycles produces monotonically non-decreasing damage ----
np.random.seed(42)
params_int = DamageParameters()
stress_hist = np.tile(np.array([1500.0, 80.0, 100.0]), (50, 1))
d0 = DamageState()
damage_hist = integrate_damage_cycles(d0, stress_hist, params_int, 50)
for col in range(4):
    diffs = np.diff(damage_hist[:, col])
    assert np.all(diffs >= -1e-12), '[TC36] damage integration monotonicity FAILED'

# ---- TC37: compute_damage_dissipation_energy is non-negative ----
W_d_test = compute_damage_dissipation_energy(damage_hist, mat_test, params_int)
assert W_d_test >= 0.0, '[TC37] damage dissipation energy non-negative FAILED'

# ---- TC38: LaminateOptimization objective_function returns scalar ----
np.random.seed(42)
opt_test = LaminateOptimization(material=mat_test, n_plies=4, target_load=100.0)
obj_val = opt_test.objective_function([0.0, 45.0, -45.0, 90.0])
assert np.isscalar(obj_val), '[TC38] optimization objective scalar FAILED'

# ---- TC39: compute_max_stable_timestep is finite for negative eigenvalues ----
test_eig = np.array([-1.0, -10.0, -0.5])
max_dt = compute_max_stable_timestep(test_eig, rk4_stability_function)
assert max_dt > 0.0 and max_dt < np.inf, '[TC39] compute_max_stable_timestep finite FAILED'

# ---- TC40: recommend_time_integrator returns dict with required keys ----
np.random.seed(42)
d_test_rec = DamageState(d_f=0.1, d_m=0.05, d_s=0.05, d_i=0.02)
rec = recommend_time_integrator(d_test_rec, np.array([500.0, 20.0, 10.0]), params_int)
assert 'method' in rec and 'max_dt_rk4' in rec and 'stiffness_ratio' in rec, '[TC40] recommend_time_integrator keys FAILED'
