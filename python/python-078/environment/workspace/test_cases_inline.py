# ---- TC01: pi_high_precision returns value close to numpy.pi ----
pi_val = pi_high_precision()
assert abs(pi_val - np.pi) < 1e-14, '[TC01] pi_high_precision FAILED'

# ---- TC02: is_prime correctly identifies primes and non-primes ----
assert is_prime(17) == True, '[TC02] is_prime(17) FAILED'
assert is_prime(18) == False, '[TC02] is_prime(18) FAILED'

# ---- TC03: prime_sieve returns correct primes up to 20 ----
primes = prime_sieve(20)
expected = np.array([2, 3, 5, 7, 11, 13, 17, 19])
assert np.array_equal(primes, expected), '[TC03] prime_sieve FAILED'

# ---- TC04: FEMMesh computes correct triangle area ----
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
elements = np.array([[0, 1, 2]])
mesh = FEMMesh(nodes, elements)
assert abs(mesh.element_area(0) - 0.5) < 1e-12, '[TC04] FEMMesh element_area FAILED'

# ---- TC05: womersley_number positive and finite for arterial params ----
alpha = womersley_number(0.005, 3.3e-6, 2.0 * np.pi * 72.0 / 60.0)
assert alpha > 0.0 and np.isfinite(alpha), '[TC05] womersley_number FAILED'

# ---- TC06: reynolds_number in physiological range ----
Re = reynolds_number(0.3, 0.01, 3.3e-6)
assert 500.0 < Re < 2000.0, '[TC06] reynolds_number FAILED'

# ---- TC07: murray_law_radius symmetric bifurcation matches formula ----
r_child = murray_law_radius(0.006, 2)
expected = 0.006 / (2.0 ** (1.0 / 3.0))
assert abs(r_child - expected) < 1e-12, '[TC07] murray_law_radius FAILED'

# ---- TC08: safe_sqrt handles negative small value gracefully ----
assert safe_sqrt(-1e-13) == 0.0, '[TC08] safe_sqrt small negative FAILED'
assert abs(safe_sqrt(4.0) - 2.0) < 1e-12, '[TC08] safe_sqrt positive FAILED'

# ---- TC09: safe_divide avoids division by zero ----
assert safe_divide(5.0, 0.0, default=99.0) == 99.0, '[TC09] safe_divide FAILED'

# ---- TC10: hexagon01_area matches theoretical value ----
area = hexagon01_area()
assert abs(area - 3.0 * np.sqrt(3.0) / 2.0) < 1e-12, '[TC10] hexagon01_area FAILED'

# ---- TC11: integrate_on_hexagon integrates constant to area ----
f_const = lambda x, y: np.ones_like(x)
int_const = integrate_on_hexagon(f_const, rule_id=3)
assert abs(int_const - area) < 1e-10, '[TC11] integrate_on_hexagon constant FAILED'

# ---- TC12: legendre_gauss_nodes_weights sum equals 2 and nodes in range ----
x_nodes, w_nodes = legendre_gauss_nodes_weights(8)
assert abs(np.sum(w_nodes) - 2.0) < 1e-12, '[TC12] legendre_gauss_nodes_weights sum FAILED'
assert np.all(np.abs(x_nodes) <= 1.0), '[TC12] legendre nodes out of range FAILED'

# ---- TC13: gauss_legendre_quadrature integrates polynomial exactly ----
f_pow2 = lambda x: x ** 2
int_pow2 = gauss_legendre_quadrature(f_pow2, -1.0, 1.0, n=4)
assert abs(int_pow2 - 2.0 / 3.0) < 1e-12, '[TC13] gauss_legendre_quadrature FAILED'

# ---- TC14: runge_fun at origin equals 1 ----
assert abs(runge_fun(np.array([0.0]))[0] - 1.0) < 1e-12, '[TC14] runge_fun FAILED'

# ---- TC15: R83TMatrix DIF2 eigenvalue formula matches theory for n=10 ----
A10 = R83TMatrix.dif2(10)
assert abs(A10.eigenvalue_dif2(1) - 4.0 * np.sin(np.pi / 22.0) ** 2) < 1e-12, '[TC15] eigenvalue_dif2 FAILED'

# ---- TC16: r83t_mv computes correct matrix-vector product ----
x_test = np.array([1.0, 2.0, 3.0, 4.0])
A4 = R83TMatrix.dif2(4)
y = r83t_mv(A4, x_test)
y_dense = A4.to_dense() @ x_test
assert np.linalg.norm(y - y_dense) < 1e-12, '[TC16] r83t_mv FAILED'

# ---- TC17: thomas_algorithm solves small tridiagonal system exactly ----
sub = np.array([0.0, -1.0, -1.0])
main = np.array([2.0, 2.0, 2.0])
super = np.array([-1.0, -1.0, 0.0])
A3 = R83TMatrix.from_diagonals(sub, main, super)
b3 = np.array([1.0, 0.0, 0.0])
x3 = thomas_algorithm(A3, b3)
residual = np.linalg.norm(r83t_mv(A3, x3) - b3)
assert residual < 1e-12, '[TC17] thomas_algorithm FAILED'

# ---- TC18: r83t_cg_solve converges for DIF2 with n=20 ----
A20 = R83TMatrix.dif2(20)
x_exact = np.ones(20)
b20 = r83t_mv(A20, x_exact)
x_cg, it_cg, res_cg = r83t_cg_solve(A20, b20, max_iter=25, tol=1e-12)
assert np.linalg.norm(x_cg - x_exact, ord=np.inf) < 1e-10, '[TC18] r83t_cg_solve FAILED'

# ---- TC19: effective_diffusion_plasma returns positive finite value ----
D = effective_diffusion_plasma(temperature_kelvin=310.15, particle_radius_nm=100.0, plasma_viscosity_pa_s=0.0012)
assert D > 0.0 and np.isfinite(D), '[TC19] effective_diffusion_plasma FAILED'

# ---- TC20: einstein_viscosity_correction increases with hematocrit ----
mu1 = einstein_viscosity_correction(hematocrit=0.40)
mu2 = einstein_viscosity_correction(hematocrit=0.45)
assert mu2 > mu1 > 1.0, '[TC20] einstein_viscosity_correction FAILED'

# ---- TC21: peclet_number positive for physiological parameters ----
Pe = peclet_number(shear_rate=1000.0, particle_radius_nm=100.0, diffusion_coeff=D)
assert Pe > 0.0 and np.isfinite(Pe), '[TC21] peclet_number FAILED'

# ---- TC22: brownian_motion_simulation reproducible with fixed seed ----
np.random.seed(42)
traj1 = brownian_motion_simulation(m_dim=3, n_steps=10, diffusion_coeff=1e-9, total_time=1.0, seed=42)
np.random.seed(42)
traj2 = brownian_motion_simulation(m_dim=3, n_steps=10, diffusion_coeff=1e-9, total_time=1.0, seed=42)
assert np.allclose(traj1, traj2), '[TC22] brownian_motion_simulation reproducibility FAILED'

# ---- TC23: verify_einstein_relation relative error small ----
np.random.seed(42)
dsq = brownian_displacement_simulation(k_trials=100, n_steps=20, m_dim=3, diffusion_coeff=1e-9, total_time=1.0, seed=42)
stats = verify_einstein_relation(dsq, 3, 1e-9, 1.0)
assert stats['relative_error'] < 0.5, '[TC23] verify_einstein_relation FAILED'

# ---- TC24: vessel_wall_density positive everywhere ----
x_test = np.array([0.0, 0.5, 1.0])
y_test = np.array([0.0, 0.5, 1.0])
rho = vessel_wall_density(x_test, y_test)
assert np.all(rho > 0.0), '[TC24] vessel_wall_density FAILED'

# ---- TC25: map_cvt_to_annulus radial range correct ----
gen = np.array([[0.0, 0.0], [1.0, 1.0]])
pts = map_cvt_to_annulus(gen, inner_radius=0.004, outer_radius=0.006)
r_pts = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
assert np.all(r_pts >= 0.004 - 1e-12) and np.all(r_pts <= 0.006 + 1e-12), '[TC25] map_cvt_to_annulus FAILED'

# ---- TC26: VascularCVTMesh radial coordinates within annulus ----
np.random.seed(42)
gens = np.random.rand(10, 2)
vasc = VascularCVTMesh(gens, 0.004, 0.006)
rads = vasc.radial_coordinates()
assert np.all(rads >= 0.004 - 1e-12) and np.all(rads <= 0.006 + 1e-12), '[TC26] VascularCVTMesh FAILED'

# ---- TC27: VesselElasticPendulum g_eff and period positive ----
pend = VesselElasticPendulum(equilibrium_radius=0.005, elastic_modulus_pa=1.0e6, wall_thickness_m=1.0e-3, wall_density_kg_m3=1050.0)
assert pend.g_eff > 0.0, '[TC27] VesselElasticPendulum g_eff FAILED'
T0 = pend.period(0.01)
assert T0 > 0.0 and np.isfinite(T0), '[TC27] VesselElasticPendulum period FAILED'

# ---- TC28: simulate_vessel_oscillation returns expected keys ----
t_span = np.linspace(0, 0.1, 20)
result = simulate_vessel_oscillation(pend, t_span, xi0=0.01)
assert 'displacement' in result and 'velocity' in result and 'energy' in result and 'radius' in result, '[TC28] simulate_vessel_oscillation FAILED'

# ---- TC29: rbc_interaction_force antisymmetric for two particles ----
pos = np.array([[0.0, 0.0], [1e-6, 0.0]])
forces = rbc_interaction_force(pos)
assert np.linalg.norm(forces[0] + forces[1]) < 1e-20, '[TC29] rbc_interaction_force FAILED'

# ---- TC30: apparent_viscosity_from_rbc increases with RBC count ----
mu_10 = apparent_viscosity_from_rbc(n_rbc=10, domain_volume=1e-12)
mu_100 = apparent_viscosity_from_rbc(n_rbc=100, domain_volume=1e-12)
assert mu_100 > mu_10, '[TC30] apparent_viscosity_from_rbc FAILED'

# ---- TC31: pressure_wave_speed physically reasonable ----
c = pressure_wave_speed(elastic_modulus_pa=1.0e6, wall_thickness_m=1.0e-3, vessel_radius_m=0.005)
assert 1.0 < c < 20.0, '[TC31] pressure_wave_speed FAILED'

# ---- TC32: shallow_water_lax_wendroff conserves total area approximately ----
A0 = np.ones(20) * 2.0e-5
Q0 = np.zeros(20)
A_final, Q_final = shallow_water_lax_wendroff(A0, Q0, dx=0.01, dt=0.001, g_eff=100.0, n_steps=5, boundary_type="reflecting")
assert abs(np.sum(A_final) - np.sum(A0)) / np.sum(A0) < 0.1, '[TC32] shallow_water_lax_wendroff FAILED'

# ---- TC33: NLSEPressurePulse mass conservation positive ----
nlse = NLSEPressurePulse(nx=32, z_min=-5.0, z_max=5.0, gamma=0.5)
psi0 = nlse.initial_double_soliton(nlse.z, amplitude=0.5, c1=1.0, c2=0.1, delta=5.0)
m0 = nlse.mass_conservation(psi0)
assert m0 > 0.0 and np.isfinite(m0), '[TC33] NLSEPressurePulse mass_conservation FAILED'

# ---- TC34: incidence_to_transition produces column-stochastic matrix ----
adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
T = incidence_to_transition(adj)
col_sums = np.sum(T, axis=0)
assert np.allclose(col_sums, np.ones(3)), '[TC34] incidence_to_transition FAILED'

# ---- TC35: page_rank_with_damping returns normalized vector ----
pr = page_rank_with_damping(adj, damping=0.85, max_iter=200, tol=1e-10)
assert abs(np.sum(pr) - 1.0) < 1e-8 and np.all(pr >= 0.0), '[TC35] page_rank_with_damping FAILED'

# ---- TC36: ArterialNetwork root flow equals total and all flows non-negative ----
network = ArterialNetwork()
flows = network.compute_flow_distribution(total_flow=5.0e-5)
assert abs(flows["Ascending_Aorta"] - 5.0e-5) < 1e-15, '[TC36] ArterialNetwork root flow FAILED'
assert all(v >= 0.0 for v in flows.values()), '[TC36] ArterialNetwork negative flow FAILED'

# ---- TC37: bifurcation_flow_split ratios sum to 1 ----
q1, q2 = bifurcation_flow_split(0.006, 0.004, 0.004)
assert abs(q1 + q2 - 1.0) < 1e-7, '[TC37] bifurcation_flow_split FAILED'

# ---- TC38: wss_physiological_score in valid range ----
assert wss_physiological_score(2.5) == 1.0, '[TC38] wss_physiological_score FAILED'
assert wss_physiological_score(0.3) == 0.0, '[TC38] wss_physiological_score FAILED'
assert 0.0 <= wss_physiological_score(8.0) <= 1.0, '[TC38] wss_physiological_score FAILED'

# ---- TC39: compute_control_cost non-negative ----
wss_traj = np.array([2.0, 2.5, 3.0])
ctrl_traj = np.array([0.1, 0.0, 0.2])
J = compute_control_cost(wss_traj, 2.5, ctrl_traj, B=0.1)
assert J >= 0.0, '[TC39] compute_control_cost FAILED'

# ---- TC40: WSSOptimalControl wss_from_radius decreases with larger radius ----
controller = WSSOptimalControl(equilibrium_radius=0.005, target_wss_pa=2.5)
wss_small = controller.wss_from_radius(0.004)
wss_large = controller.wss_from_radius(0.006)
assert wss_small > wss_large, '[TC40] WSSOptimalControl wss_from_radius FAILED'

# ---- TC41: relative_resistance_index in [0,1] ----
rri = relative_resistance_index(5.0, 1.0)
assert 0.0 <= rri <= 1.0, '[TC41] relative_resistance_index FAILED'

# ---- TC42: compute_osi in valid range [0, 0.5] ----
t_hist = np.linspace(0, 1.0, 50)
wss_hist = np.sin(2.0 * np.pi * t_hist) + 1.0
osi = compute_osi(wss_hist, t_hist)
assert 0.0 <= osi <= 0.5, '[TC42] compute_osi FAILED'

# ---- TC43: compute_tawss non-negative ----
t_hist = np.linspace(0, 1.0, 50)
wss_hist = np.abs(np.sin(2.0 * np.pi * t_hist))
tawss = compute_tawss(wss_hist, t_hist)
assert tawss >= 0.0, '[TC43] compute_tawss FAILED'

# ---- TC44: WomersleySolver alpha positive ----
solver = WomersleySolver(radius=0.005, kinematic_viscosity=3.3e-6, heart_rate_bpm=72.0)
assert solver.alpha > 0.0 and np.isfinite(solver.alpha), '[TC44] WomersleySolver alpha FAILED'

# ---- TC45: generate_wss_report contains expected keys ----
solver_small = WomersleySolver(radius=0.005, kinematic_viscosity=3.3e-6, heart_rate_bpm=72.0, n_radial=20)
result_small = solver_small.solve_pulsatile(n_cardiac_cycles=0.5, n_steps_per_cycle=20)
report = generate_wss_report(solver_small, result_small)
expected_keys = ["TAWSS_Pa", "OSI", "WSSG_Pa_s", "WSS_max_Pa", "WSS_min_Pa", "RRI", "physiological_score"]
for k in expected_keys:
    assert k in report, f'[TC45] generate_wss_report missing key {k} FAILED'
