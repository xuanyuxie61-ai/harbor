
# ================================================================
# 测试用例（27个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Dirac Hamiltonian eigenvalues at k=0 equal ±Delta ----
H_test = DiracSurfaceHamiltonian(v_F=4.5e5, Delta=0.025, lambda_w=0.0)
E_p, E_m = H_test.eigenvalues(0.0, 0.0)
assert abs(E_p - H_test.Delta) < 1e-30, '[TC01] Eigenvalue at k=0 FAILED'
assert abs(E_m + H_test.Delta) < 1e-30, '[TC01] Eigenvalue at k=0 FAILED'

# ---- TC02: Dirac Hamiltonian hermiticity ----
H_mat = H_test.hamiltonian(1e9, 2e9)
assert np.allclose(H_mat, H_mat.T.conj()), '[TC02] Hamiltonian hermiticity FAILED'

# ---- TC03: effective_mass_tensor inverse near k=0 ----
m_inv = effective_mass_tensor(0.0, 0.0, v_F=4.5e5, Delta=0.025)
Delta_J = 0.025 * 1.602176634e-19
expected = (4.5e5 ** 2) / Delta_J
assert abs(m_inv[0, 0] - expected) < 1e-6 * expected, '[TC03] Effective mass tensor FAILED'
assert abs(m_inv[1, 1] - expected) < 1e-6 * expected, '[TC03] Effective mass tensor FAILED'

# ---- TC04: Berry curvature sign upper vs lower band ----
berry = BerryCurvatureCalculator(H_test)
omega_up = berry.berry_curvature_analytical(1e9, 0.0, band='upper')
omega_low = berry.berry_curvature_analytical(1e9, 0.0, band='lower')
assert omega_up < 0, '[TC04] Berry curvature upper band sign FAILED'
assert omega_low > 0, '[TC04] Berry curvature lower band sign FAILED'
assert abs(omega_up + omega_low) < 1e-30, '[TC04] Berry curvature symmetry FAILED'

# ---- TC05: Chern number lower band near +0.5 (Delta>0 gives positive integral) ----
C = berry.chern_number(k_max=5e9, n_k=200, method='analytical')
assert abs(C - 0.5) < 0.05, '[TC05] Chern number FAILED'

# ---- TC06: Berry phase for closed loop enclosing Dirac point near pi ----
theta_loop = np.linspace(0.0, 2.0 * np.pi, 100)
k_path = np.column_stack((1e9 * np.cos(theta_loop), 1e9 * np.sin(theta_loop)))
gamma = berry.berry_phase_1d(k_path)
assert abs(gamma - np.pi) < 0.5, '[TC06] Berry phase enclosing Dirac point FAILED'

# ---- TC07: Fermi wavevector zero when E_F inside gap ----
fs = FermiSurface(hamiltonian=H_test, E_F=0.01)
k_F = fs.fermi_wavevector()
assert k_F == 0.0, '[TC07] Fermi wavevector inside gap FAILED'

# ---- TC08: Fermi surface carrier density non-negative ----
fs2 = FermiSurface(hamiltonian=H_test, E_F=0.12)
n_c = fs2.carrier_density()
assert n_c >= 0.0, '[TC08] Carrier density FAILED'

# ---- TC09: Cyclotron frequency positive for B>0 ----
omega_c = fs2.cyclotron_frequency(B=1.0)
assert omega_c > 0.0, '[TC09] Cyclotron frequency FAILED'

# ---- TC10: DOS zero inside gap ----
disorder = DisorderScattering(hamiltonian=H_test, n_imp=5e14, V0=0.3, disorder_type='delta')
dos = disorder.density_of_states(0.5 * H_test.Delta)
assert dos == 0.0, '[TC10] DOS inside gap FAILED'

# ---- TC11: Spin overlap factor in [0,1] ----
overlap = disorder.spin_overlap_factor(1e9, 0.0, 2e9, 1e9)
assert 0.0 <= overlap <= 1.0, '[TC11] Spin overlap factor range FAILED'

# ---- TC12: Transport scattering time positive ----
tau_tr = disorder.transport_scattering_time(0.12 * 1.602176634e-19)
assert tau_tr > 0.0, '[TC12] Transport scattering time FAILED'

# ---- TC13: Kubo longitudinal conductivity positive ----
kubo = KuboConductivity(H_test, disorder)
sigma_xx = kubo.dc_conductivity_semicalassical(0.12)
assert sigma_xx > 0.0, '[TC13] DC conductivity FAILED'

# ---- TC14: Thermoelectric coefficients finite ----
S, L = kubo.thermoelectric_coefficients(0.12, T=10.0)
assert np.isfinite(S), '[TC14] Seebeck coefficient FAILED'
assert np.isfinite(L), '[TC14] Lorenz number FAILED'

# ---- TC15: Tight-binding diagonalize returns correct number of states ----
tb = TightBindingSurface(Nx=6, Ny=6, a=2.0, v_F=4.5e5, Delta=0.025, boundary='open')
energies_tb, evecs_tb = tb.diagonalize()
assert len(energies_tb) == tb.N_states, '[TC15] TB eigenvalue count FAILED'
assert evecs_tb.shape == (tb.N_states, tb.N_states), '[TC15] TB eigenvector shape FAILED'

# ---- TC16: Edge state probability weight in [0,1] ----
profile, edge_weight = tb.edge_state_probability(evecs_tb, band_index=0)
assert 0.0 <= edge_weight <= 1.0, '[TC16] Edge state weight FAILED'

# ---- TC17: Fibonacci sequence correctness ----
lat = LatticeIntegrator(dim=2)
assert lat.fibonacci(1) == 1, '[TC17] Fibonacci F(1) FAILED'
assert lat.fibonacci(5) == 5, '[TC17] Fibonacci F(5) FAILED'
assert lat.fibonacci(10) == 55, '[TC17] Fibonacci F(10) FAILED'

# ---- TC18: Jacobi quadrature integrates constant ----
jq = JacobiQuadrature()
result = jq.integrate(lambda x: 2.0, n=8, alpha=0.0, beta=0.0, a=-1.0, b=1.0)
assert abs(result - 4.0) < 1e-10, '[TC18] Jacobi constant integral FAILED'

# ---- TC19: Monte Carlo circle area with fixed seed reproducibility ----
np.random.seed(42)
mc = MonteCarloIntegrator(seed=42)
q1, _ = mc.integrate_circle(lambda x, y: 1.0, radius=2.0, n_samples=5000)
np.random.seed(42)
mc2 = MonteCarloIntegrator(seed=42)
q2, _ = mc2.integrate_circle(lambda x, y: 1.0, radius=2.0, n_samples=5000)
assert abs(q1 - q2) < 1e-10, '[TC19] MC reproducibility FAILED'
assert abs(q1 - 4.0 * np.pi) < 0.5, '[TC19] MC circle area FAILED'

# ---- TC20: Snyder method finds sqrt(2) ----
solver = NonlinearSolver(max_iter=100, tol=1e-10)
root_snyder, it_snyder = solver.snyder_method(lambda x: x ** 2 - 2.0, 1.0, 2.0)
assert abs(root_snyder - np.sqrt(2.0)) < 1e-8, '[TC20] Snyder sqrt(2) FAILED'

# ---- TC21: Trigamma at x=1 equals pi^2/6 ----
tg = TrigammaFunction()
val_tg, err_tg = tg.evaluate(1.0)
assert err_tg == 0, '[TC21] Trigamma error flag FAILED'
assert abs(val_tg - np.pi ** 2 / 6.0) < 1e-8, '[TC21] Trigamma(1) value FAILED'

# ---- TC22: Carlson RF symmetric case ----
carlson = CarlsonEllipticIntegrals(errtol=1e-6)
rf_val, rf_err = carlson.rf(1.0, 1.0, 1.0)
assert rf_err == 0, '[TC22] Carlson RF error flag FAILED'
assert abs(rf_val - 1.0) < 1e-6, '[TC22] Carlson RF(1,1,1) FAILED'

# ---- TC23: Interpolation exact match at data point ----
np.random.seed(42)
pts = np.random.rand(20, 2) * 10
vals = np.sin(pts[:, 0]) * np.cos(pts[:, 1])
interp = Interpolation2D(pts, vals)
val_idw = interp.inverse_distance_weighting(pts[0, 0], pts[0, 1])
assert abs(float(val_idw.flat[0]) - vals[0]) < 1e-6, '[TC23] IDW exact match FAILED'

# ---- TC24: Hexagon area formula ----
geom_hex = SampleGeometry(size=10.0, shape='hexagon')
area_hex = geom_hex.area()
expected_area = 3.0 * np.sqrt(3.0) / 2.0 * 100.0
assert abs(area_hex - expected_area) < 1e-8, '[TC24] Hexagon area FAILED'

# ---- TC25: Point in polygon center test ----
vertices = geom_hex.hexagon_vertices()
inside = geom_hex.point_in_polygon((0.0, 0.0), vertices)
assert inside == True, '[TC25] Point in polygon center FAILED'

# ---- TC26: IOManager write simulation header and verify file exists ----
import tempfile, os
tmpdir = tempfile.mkdtemp()
io_tmp = IOManager(tmpdir)
test_params = {'v_F': 4.5e5, 'Delta': 0.025}
io_tmp.write_simulation_header('test_params.dat', test_params)
assert os.path.exists(os.path.join(tmpdir, 'test_params.dat')), '[TC26] IOManager header file FAILED'

# Test parse_variable_line on correctly formatted line
vars_parsed = io_tmp.parse_variable_line('VARIABLES = "X" "Y" "Z"')
assert vars_parsed == ['X', 'Y', 'Z'], '[TC26] IOManager parse variables FAILED'

# ---- TC27: run_simulation returns dict with expected keys ----
results = run_simulation()
assert isinstance(results, dict), '[TC27] Simulation result type FAILED'
assert 'Chern_number' in results, '[TC27] Simulation result keys FAILED'
assert 'Total_Hall_S' in results, '[TC27] Simulation result keys FAILED'
assert np.isfinite(results['Chern_number']), '[TC27] Chern number finite FAILED'
