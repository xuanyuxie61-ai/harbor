np.random.seed(42)

# ---- TC01: RVE empty fiber volume fraction is zero ----
rve_empty = RVEGeometry(100.0, 100.0, [], nx=10, ny=10)
result = rve_empty.fiber_volume_fraction()
assert result == 0.0, '[TC01] RVE empty fiber volume fraction FAILED'

# ---- TC02: generate_hexagonal_fiber_rve returns valid RVE with positive Vf ----
rve = generate_hexagonal_fiber_rve(width=50.0, height=50.0, fiber_radius=5.0, n_fibers_x=2, n_fibers_y=2)
assert isinstance(rve, RVEGeometry), '[TC02] generate_hexagonal_fiber_rve type FAILED'
assert rve.fiber_volume_fraction() > 0.0, '[TC02] generate_hexagonal_fiber_rve Vf positive FAILED'
assert rve.fiber_volume_fraction() < 1.0, '[TC02] generate_hexagonal_fiber_rve Vf less than 1 FAILED'

# ---- TC03: create_carbon_epoxy Vf=0 gives matrix modulus ----
mat_zero = create_carbon_epoxy(V_f=0.0)
assert abs(mat_zero.E1 - mat_zero.E_m) < 1e-6, '[TC03] create_carbon_epoxy Vf=0 FAILED'

# ---- TC04: create_carbon_epoxy Vf=1 gives fiber modulus ----
mat_one = create_carbon_epoxy(V_f=1.0)
assert abs(mat_one.E1 - mat_one.E_f) < 1e-6, '[TC04] create_carbon_epoxy Vf=1 FAILED'

# ---- TC05: compute_transformed_stiffness 0deg equals Q matrix ----
mat = create_carbon_epoxy(V_f=0.6)
Q_0 = mat.compute_transformed_stiffness(0.0)
assert np.allclose(Q_0, mat.Q, atol=1e-6), '[TC05] transformed stiffness 0deg FAILED'

# ---- TC06: compute_transformed_stiffness 90deg swaps Q11 and Q22 ----
Q_90 = mat.compute_transformed_stiffness(90.0)
assert abs(Q_90[0,0] - mat.Q[1,1]) < 1.0, '[TC06] transformed stiffness 90deg Q11 FAILED'
assert abs(Q_90[1,1] - mat.Q[0,0]) < 1.0, '[TC06] transformed stiffness 90deg Q22 FAILED'
assert abs(Q_90[0,1] - mat.Q[0,1]) < 1e-6, '[TC06] transformed stiffness 90deg Q12 FAILED'

# ---- TC07: compute_degraded_stiffness zero damage equals original Q ----
Q_deg = mat.compute_degraded_stiffness(0.0, 0.0, 0.0)
assert np.allclose(Q_deg, mat.Q, atol=1e-6), '[TC07] degraded stiffness zero damage FAILED'

# ---- TC08: compute_degraded_stiffness max damage reduces stiffness ----
Q_deg_max = mat.compute_degraded_stiffness(0.99, 0.99, 0.99)
assert Q_deg_max[0,0] < mat.Q[0,0], '[TC08] degraded stiffness max damage E1 FAILED'
assert Q_deg_max[1,1] < mat.Q[1,1], '[TC08] degraded stiffness max damage E2 FAILED'
assert Q_deg_max[2,2] < mat.Q[2,2], '[TC08] degraded stiffness max damage G12 FAILED'

# ---- TC09: hashin_failure_criteria zero stress all safe ----
params = DamageParameters()
criteria = hashin_failure_criteria(np.array([0.0, 0.0, 0.0]), params)
assert all(v < 1.0 for v in criteria.values()), '[TC09] hashin zero stress FAILED'

# ---- TC10: hashin_failure_criteria high stress triggers failure ----
criteria_high = hashin_failure_criteria(np.array([params.X_T * 1.5, 0.0, 0.0]), params)
assert criteria_high.get('fiber_tension', 0.0) >= 1.0, '[TC10] hashin high stress fiber tension FAILED'

# ---- TC11: DamageState clips out-of-range values ----
ds = DamageState(d_f=1.5, d_m=-0.5, d_s=2.0, d_i=0.5)
assert ds.d_f == 0.99, '[TC11] DamageState clip high d_f FAILED'
assert ds.d_m == 0.0, '[TC11] DamageState clip low d_m FAILED'
assert ds.d_s == 0.99, '[TC11] DamageState clip high d_s FAILED'
assert ds.d_i == 0.5, '[TC11] DamageState normal d_i FAILED'
assert ds.is_failed(), '[TC11] DamageState clipped is failed FAILED'
ds_normal = DamageState(d_f=0.1, d_m=0.2, d_s=0.1, d_i=0.0)
assert not ds_normal.is_failed(), '[TC11] DamageState normal not failed FAILED'

# ---- TC12: estimate_damage_period zero stress returns 1e6 ----
N_est = estimate_damage_period(np.array([0.0, 0.0, 0.0]), params)
assert N_est == 1e6, '[TC12] estimate_damage_period zero stress FAILED'

# ---- TC13: estimate_damage_period high stress returns finite ----
N_est_high = estimate_damage_period(np.array([params.sigma_f0 * 2.0, 0.0, 0.0]), params)
assert N_est_high < 1e6, '[TC13] estimate_damage_period high stress finite FAILED'
assert N_est_high > 0.0, '[TC13] estimate_damage_period high stress positive FAILED'

# ---- TC14: integrate_damage_cycles returns increasing damage ----
initial_damage = DamageState(d_f=0.0, d_m=0.0, d_s=0.0, d_i=0.0)
stress_history = np.tile(np.array([params.X_T * 1.2, params.Y_T * 1.2, params.S * 0.6]), (100, 1))
damage_states = integrate_damage_cycles(initial_damage, stress_history, params, 100)
assert damage_states.shape[0] > 1, '[TC14] integrate damage states length FAILED'
assert np.all(damage_states[-1] >= damage_states[0]), '[TC14] integrate damage increases FAILED'
assert np.all(damage_states[-1] <= 0.99), '[TC14] integrate damage clips to 0.99 FAILED'

# ---- TC15: LaminateStiffness symmetric laminate B matrix zero ----
plies = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
thicknesses = [0.125] * 8
laminate = LaminateStiffness(plies, thicknesses, mat)
assert np.allclose(laminate.B, np.zeros((3,3)), atol=1e-6), '[TC15] symmetric laminate B FAILED'
assert laminate.get_total_thickness() == 1.0, '[TC15] total thickness FAILED'

# ---- TC16: LaminateStiffness A matrix positive definite ----
eig_A = np.linalg.eigvalsh(laminate.A)
assert np.all(eig_A > 0), '[TC16] A matrix positive definite FAILED'

# ---- TC17: BandedUpperTriangularSolver solve multiply consistency ----
solver = BandedUpperTriangularSolver(5, 2)
A_band = np.array([
    [0.0, 0.0, 1.0, 2.0, 3.0],
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [1.0, 1.0, 1.0, 1.0, 1.0]
])
b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x = solver.solve(A_band, b)
b_check = solver.multiply(A_band, x)
assert np.allclose(b, b_check, atol=1e-10), '[TC17] banded solver consistency FAILED'

# ---- TC18: solve_equilibrium_dense exact solution recovery ----
K = np.array([[4.0, 1.0], [1.0, 3.0]])
F = np.array([1.0, 2.0])
U, r_norm, norm_res, cond_K = solve_equilibrium_dense(K, F)
assert r_norm < 1e-10, '[TC18] dense solver residual FAILED'
assert cond_K > 0.0, '[TC18] dense solver condition FAILED'
assert np.allclose(U, np.linalg.solve(K, F), atol=1e-10), '[TC18] dense solver solution FAILED'

# ---- TC19: solve_equilibrium_dense small random system residual small ----
n_dof = 10
K_rand = np.random.randn(n_dof, n_dof)
K_rand = K_rand.T @ K_rand + 0.5 * np.eye(n_dof)
F_rand = np.random.randn(n_dof)
U_rand, r_norm_rand, norm_res_rand, cond_K_rand = solve_equilibrium_dense(K_rand, F_rand)
assert r_norm_rand < 1e-6, '[TC19] dense solver random residual FAILED'

# ---- TC20: SparseStiffnessAssembler shape and values ----
assembler = SparseStiffnessAssembler(n_nodes=3, ndof_per_node=2)
k_e = np.array([[1.0, -1.0], [-1.0, 1.0]])
assembler.add_element_stiffness([0], k_e)
K_csr = assembler.get_csr_matrix()
assert K_csr.shape == (6, 6), '[TC20] sparse assembler shape FAILED'
assert K_csr[0, 0] == 1.0, '[TC20] sparse assembler value FAILED'
assert K_csr[0, 1] == -1.0, '[TC20] sparse assembler value 2 FAILED'

# ---- TC21: gauss_legendre_nodes_weights exact for constant and linear ----
x_gl, w_gl = gauss_legendre_nodes_weights(4, 0.0, 1.0)
assert abs(np.sum(w_gl) - 1.0) < 1e-14, '[TC21] Gauss-Legendre constant FAILED'
integral_linear = np.sum(w_gl * x_gl)
assert abs(integral_linear - 0.5) < 1e-14, '[TC21] Gauss-Legendre linear FAILED'

# ---- TC22: hermite_monomial_integral odd degree is zero ----
for deg in [1, 3, 5, 7]:
    val = hermite_monomial_integral(deg, option=1)
    assert val == 0.0, f'[TC22] Hermite odd degree {deg} FAILED'

# ---- TC23: hermite_monomial_integral even degree positive ----
for deg in [0, 2, 4]:
    val = hermite_monomial_integral(deg, option=1)
    assert val > 0.0, f'[TC23] Hermite even degree {deg} positive FAILED'
assert abs(hermite_monomial_integral(0, option=1) - np.sqrt(np.pi)) < 1e-10, '[TC23] Hermite degree 0 FAILED'

# ---- TC24: compute_strain_energy_release_rate_quadrature positive ----
U_release = compute_strain_energy_release_rate_quadrature(
    stress=np.array([100.0, 50.0, 30.0]),
    strain=np.array([0.01, 0.005, 0.003]),
    damage=0.1,
    material=mat,
    thickness=1.0,
    n_quad=4)
assert U_release >= 0, '[TC24] strain energy release rate FAILED'

# ---- TC25: compute_vcct_energy_release_rate positive ----
G_I, G_II = compute_vcct_energy_release_rate(stress_at_crack_tip=80.0, displacement_jump=0.05, delta_a=2.0, n_quad=8)
assert G_I > 0, '[TC25] VCCT G_I positive FAILED'
assert G_II >= 0, '[TC25] VCCT G_II non-negative FAILED'

# ---- TC26: compute_j_integral positive ----
J_val = compute_j_integral(None, None, [50.0, 0.0], 10.0, n_quad=16)
assert J_val > 0, '[TC26] J-integral positive FAILED'

# ---- TC27: probabilistic_strength_integral deterministic limit ----
E_strength = probabilistic_strength_integral(mean_strength=2500.0, std_strength=1e-6)
assert abs(E_strength - 2500.0) < 1.0, '[TC27] probabilistic strength deterministic FAILED'

# ---- TC28: generate_symmetric_eigenproblem eigenvalue reconstruction ----
np.random.seed(42)
A_sym, Q_sym, lambda_sym = generate_symmetric_eigenproblem(4, lambda_mean=5.0, lambda_dev=0.0)
eigvals = np.linalg.eigvalsh(A_sym)
assert np.allclose(np.sort(eigvals), np.sort(lambda_sym), atol=1e-10), '[TC28] symmetric eigenproblem FAILED'

# ---- TC29: generate_nonsymmetric_eigenproblem shape correct ----
np.random.seed(42)
A_nsym, Q_nsym, T_nsym = generate_nonsymmetric_eigenproblem(4, lambda_mean=-0.5, lambda_dev=0.2)
assert A_nsym.shape == (4, 4), '[TC29] nonsymmetric eigenproblem shape FAILED'
assert T_nsym.shape == (4, 4), '[TC29] nonsymmetric T shape FAILED'

# ---- TC30: BucklingAnalysis bending stiffness matrix shape ----
buckling = BucklingAnalysis(np.eye(3)*1e3, plate_length=100.0, plate_width=100.0, nx=6, ny=6)
K_b = buckling.build_bending_stiffness_matrix()
assert K_b.shape == (36, 36), '[TC30] buckling stiffness shape FAILED'

# ---- TC31: BucklingAnalysis solve returns positive eigenvalues ----
lambdas, modes = buckling.solve_buckling_loads(N_x=1.0, n_modes=3)
assert len(lambdas) > 0, '[TC31] buckling eigenvalues count FAILED'
assert np.all(lambdas > 0), '[TC31] buckling eigenvalues positive FAILED'

# ---- TC32: compute_stress_wave_reflection_coefficient same material ----
R, T = compute_stress_wave_reflection_coefficient(E1=100.0, E2=100.0, rho1=1000.0, rho2=1000.0)
assert abs(R) < 1e-10, '[TC32] reflection same material FAILED'
assert abs(T - 1.0) < 1e-10, '[TC32] transmission same material FAILED'

# ---- TC33: compute_stress_wave_reflection_coefficient impedance contrast ----
R2, T2 = compute_stress_wave_reflection_coefficient(E1=100.0, E2=400.0, rho1=1000.0, rho2=1000.0)
assert R2 > 0, '[TC33] reflection contrast positive FAILED'
assert T2 > 0, '[TC33] transmission contrast positive FAILED'
assert abs(R2) + abs(T2) > 1.0, '[TC33] reflection transmission sum FAILED'

# ---- TC34: rk4_stability_function at origin equals 1 ----
val = rk4_stability_function(0.0)
assert abs(val - 1.0) < 1e-14, '[TC34] RK4 stability at origin FAILED'

# ---- TC35: check_eigenvalue_in_stability_region zero eigenvalue stable ----
stable = check_eigenvalue_in_stability_region(np.array([0.0, 0.0]), 1.0, rk4_stability_function)
assert np.all(stable), '[TC35] zero eigenvalue stable FAILED'

# ---- TC36: analyze_damage_jacobian_eigenvalues shape ----
d_test = DamageState(d_f=0.1, d_m=0.1, d_s=0.1, d_i=0.1)
eig_dam = analyze_damage_jacobian_eigenvalues(d_test, np.array([1200.0, 60.0, 80.0]), params)
assert len(eig_dam) == 4, '[TC36] damage jacobian eigenvalues length FAILED'

# ---- TC37: recommend_time_integrator returns dict with keys ----
rec = recommend_time_integrator(d_test, np.array([1200.0, 60.0, 80.0]), params)
assert 'method' in rec, '[TC37] recommend integrator method key FAILED'
assert 'max_dt_rk4' in rec, '[TC37] recommend integrator max_dt_rk4 key FAILED'
assert 'max_dt_rk54' in rec, '[TC37] recommend integrator max_dt_rk54 key FAILED'
assert 'stiffness_ratio' in rec, '[TC37] recommend integrator stiffness_ratio key FAILED'

# ---- TC38: compute_ply_combinations simple case ----
n = compute_ply_combinations(3, [0, 45, 90])
assert n == 27, '[TC38] ply combinations FAILED'

# ---- TC39: DGSpectralElement1D wave speed shape and positive ----
def E_const(x):
    return 2000.0
dg = DGSpectralElement1D(N=2, K=4, x_bounds=[0.0, 10.0], rho=1.6, E_func=E_const)
c = dg.compute_wave_speed()
assert c.shape == dg.x.shape, '[TC39] DG wave speed shape FAILED'
assert np.all(c > 0), '[TC39] DG wave speed positive FAILED'

# ---- TC40: DGSpectralElement1D solve preserves shape ----
sigma0 = np.zeros((dg.Np, dg.K))
v0 = np.zeros((dg.Np, dg.K))
v0[dg.Np//2, dg.K//2] = 1.0
sigma_final, v_final = dg.solve(sigma0, v0, FinalTime=0.01)
assert sigma_final.shape == sigma0.shape, '[TC40] DG solve sigma shape FAILED'
assert v_final.shape == v0.shape, '[TC40] DG solve v shape FAILED'

# ---- TC41: WavePropagation1D wave speed shape and positive ----
def E_wave(x):
    return 100e9
wave = WavePropagation1D(L=10.0, nx=11, E_func=E_wave, rho=1600.0, damping_ratio=0.05, forcing_params=(0.0, 0.0, 5.0))
c_wave = wave.compute_wave_speed()
assert len(c_wave) == 11, '[TC41] wave propagation speed shape FAILED'
assert np.all(c_wave > 0), '[TC41] wave propagation speed positive FAILED'

# ---- TC42: WavePropagation1D attenuation coefficient positive ----
alpha = wave.compute_attenuation_coefficient(50000.0)
assert alpha >= 0, '[TC42] attenuation coefficient non-negative FAILED'

# ---- TC43: legendre_polynomial P0 equals 1 ----
p0 = legendre_polynomial(0, np.array([-1.0, 0.0, 1.0]))
assert np.allclose(p0, np.ones(3), atol=1e-14), '[TC43] Legendre P0 FAILED'
p1 = legendre_polynomial(1, np.array([-1.0, 0.0, 1.0]))
assert np.allclose(p1, np.array([-1.0, 0.0, 1.0]), atol=1e-14), '[TC43] Legendre P1 FAILED'

# ---- TC44: jacobi_gauss_lobatto_points shape correct ----
nodes_gll = jacobi_gauss_lobatto_points(4)
assert len(nodes_gll) == 5, '[TC44] GLL nodes count FAILED'
assert abs(nodes_gll[0] + 1.0) < 1e-14, '[TC44] GLL left endpoint FAILED'
assert abs(nodes_gll[-1] - 1.0) < 1e-14, '[TC44] GLL right endpoint FAILED'

# ---- TC45: vandermonde_matrix_2d_total_degree shape correct ----
points_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V2d = vandermonde_matrix_2d_total_degree(2, points_2d)
expected_cols = (2 + 1) * (2 + 2) // 2
assert V2d.shape == (4, expected_cols), '[TC45] 2D Vandermonde shape FAILED'

# ---- TC46: Mesh1D uniform nodes monotonic ----
mesh = Mesh1D(x_min=0.0, x_max=1.0, num_elements=10, refine_strength=0.0)
assert len(mesh.nodes) == 11, '[TC46] Mesh1D node count FAILED'
assert np.all(np.diff(mesh.nodes) > 0), '[TC46] Mesh1D nodes monotonic FAILED'
assert mesh.nodes[0] == 0.0, '[TC46] Mesh1D left boundary FAILED'
assert mesh.nodes[-1] == 1.0, '[TC46] Mesh1D right boundary FAILED'

# ---- TC47: Mesh1D locate_point correct ----
idx = mesh.locate_point(0.5)
assert 0 <= idx < mesh.num_elements, '[TC47] Mesh1D locate_point range FAILED'
assert mesh.nodes[idx] <= 0.5 < mesh.nodes[idx + 1], '[TC47] Mesh1D locate_point correct FAILED'

# ---- TC48: file_row_count nonexistent file returns 0 ----
rows = file_row_count('/nonexistent/path/to/file.txt')
assert rows == 0, '[TC48] file_row_count nonexistent FAILED'

# ---- TC49: safe_inverse identity returns identity ----
I3 = np.eye(3)
I3_inv = safe_inverse(I3)
assert np.allclose(I3_inv, I3, atol=1e-10), '[TC49] safe inverse identity FAILED'

# ---- TC50: compute_condition_number identity equals 1 ----
cond_I = compute_condition_number(np.eye(5))
assert abs(cond_I - 1.0) < 1e-10, '[TC50] condition number identity FAILED'
