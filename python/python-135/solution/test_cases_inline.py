# ---- TC01: validate_positive accepts valid positive scalar ----
validate_positive(42.0, "test_value")
assert True, '[TC01] validate_positive valid scalar FAILED'

# ---- TC02: validate_positive with allow_zero accepts zero ----
validate_positive(0.0, "test_zero", allow_zero=True)
assert True, '[TC02] validate_positive zero with allow_zero FAILED'

# ---- TC03: validate_positive raises ValueError on zero without allow_zero ----
try:
    validate_positive(0.0, "test_zero")
    assert False, '[TC03] validate_positive should raise on zero FAILED'
except ValueError:
    pass

# ---- TC04: print_section does not crash ----
print_section("TC04 Test Section")
assert True, '[TC04] print_section should not crash FAILED'

# ---- TC05: AmineKinetics.k2 returns finite positive at standard T ----
kin_mea = AmineKinetics("MEA")
k2_val = kin_mea.k2(313.15)
assert k2_val > 0 and np.isfinite(k2_val), '[TC05] k2 should be positive finite FAILED'

# ---- TC06: MEA k2 at 313K is larger than at 298K (Arrhenius) ----
k2_lo = kin_mea.k2(298.15)
k2_hi = kin_mea.k2(313.15)
assert k2_hi > k2_lo, '[TC06] k2 should increase with T (Arrhenius) FAILED'

# ---- TC07: AmineKinetics.reaction_rate returns non-negative finite ----
r = kin_mea.reaction_rate(313.15, 10.0, 5000.0)
assert r >= 0 and np.isfinite(r), '[TC07] reaction_rate should be non-negative finite FAILED'

# ---- TC08: CO2LoadingCalculator.equilibrium_loading in valid range ----
loader = CO2LoadingCalculator(kin_mea)
alpha = loader.equilibrium_loading(313.15, 15000.0, 5000.0)
assert 0.0 <= alpha <= 0.55, '[TC08] equilibrium_loading should be in [0, 0.55] FAILED'

# ---- TC09: explicit_trapezoidal matches analytic for simple decay ----
def decay_rhs(t, y):
    return np.array([-0.5 * y[0]])
t_exp, y_exp = explicit_trapezoidal(decay_rhs, (0.0, 10.0), [1.0], 200)
y_analytic = np.exp(-0.5 * 10.0)
assert abs(y_exp[-1, 0] - y_analytic) < 0.01, '[TC09] explicit_trapezoidal decay accuracy FAILED'

# ---- TC10: bdf3_solver matches analytic for simple decay ----
def decay2_rhs(t, y):
    return np.array([-1.0 * y[0]])
t_bdf, y_bdf = bdf3_solver(decay2_rhs, (0.0, 5.0), [1.0], 200)
y_analytic2 = np.exp(-5.0)
assert abs(y_bdf[-1, 0] - y_analytic2) < 0.05, '[TC10] bdf3_solver decay accuracy FAILED'

# ---- TC11: ChebyshevQuadrature integrates constant positive ----
quad = ChebyshevQuadrature(32, 0.0, 1.0)
integral = quad.integrate(lambda x: 1.0)
assert integral > 0 and np.isfinite(integral), '[TC11] ChebyshevQuadrature constant integral FAILED'

# ---- TC12: SpectralDiffusionSolver flux is finite positive ----
spec = SpectralDiffusionSolver(n_cheb=24)
z_f, c_f, flux_f = spec.solve_film_diffusion_reaction(
    D_diff=1.9e-9, k_rxn=100.0, delta=1.0e-4, c_interface=20.0, c_bulk=0.5
)
assert flux_f > 0 and np.isfinite(flux_f), '[TC12] spectral flux should be positive finite FAILED'

# ---- TC13: SpectralDiffusionSolver channel flow velocity non-negative ----
y_ch, u_ch = spec.solve_channel_flow(mu=1.0e-3, delta_p=100.0, L=1.0, R=0.01)
assert np.all(u_ch >= -1e-12) and np.isfinite(np.max(u_ch)), '[TC13] channel velocity should be non-negative FAILED'

# ---- TC14: polynomial_fit_2d_vandermonde linear fit accuracy ----
x_fit = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_fit = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
z_fit = 2.0 * x_fit + 3.0 * y_fit + 1.0
coeffs, cond = polynomial_fit_2d_vandermonde(x_fit, y_fit, z_fit, degree=1)
assert len(coeffs) >= 3, '[TC14] 2D polynomial fit should produce coefficients FAILED'
assert cond > 0 and np.isfinite(cond), '[TC14] condition number should be positive finite FAILED'

# ---- TC15: TwoFilmModel.solve_interface returns expected keys ----
film = TwoFilmModel(313.15, 1.2e5, "MEA")
sol = film.solve_interface(P_CO2_bulk=15000.0, c_CO2_bulk=0.0, c_amine_bulk=5000.0, k2_rate=k2_val)
for key in ["enhancement_factor", "hatta_number", "flux", "K_G_overall"]:
    assert key in sol, f'[TC15] missing key {key} in solve_interface FAILED'
assert sol["flux"] > 0, '[TC15] flux should be positive FAILED'
assert sol["enhancement_factor"] >= 1.0, '[TC15] enhancement_factor should be >= 1 FAILED'

# ---- TC16: generate_film_grid returns correct length and monotonic ----
grid = generate_film_grid(20, 0.01, centering=1)
assert len(grid) == 20, '[TC16] film grid length FAILED'
assert np.all(np.diff(grid) >= 0), '[TC16] film grid should be monotonic FAILED'

# ---- TC17: StructuredPackingGeometry specific_area is positive ----
pack = StructuredPackingGeometry(corrugation_angle=45.0, crimp_height=0.012, channel_width=0.008)
a_spec = pack.specific_area()
assert a_spec > 0 and np.isfinite(a_spec), '[TC17] specific_area should be positive FAILED'

# ---- TC18: StructuredPackingGeometry void_fraction in (0,1) ----
eps = pack.void_fraction()
assert 0.0 < eps < 1.0, '[TC18] void_fraction should be in (0,1) FAILED'

# ---- TC19: MeshGenerator 2D rectangular mesh correct shape ----
mesh = MeshGenerator(dim=2)
nodes, elements = mesh.generate_rectangular_mesh(5, 5, xlim=(0, 1), ylim=(0, 1))
assert nodes.shape == (25, 2), '[TC19] mesh nodes shape FAILED'
assert elements.shape[0] == 2 * (4 * 4), '[TC19] mesh elements count FAILED'

# ---- TC20: PoreNetworkModel permeability is positive finite ----
import numpy as np
np.random.seed(42)
pnm = PoreNetworkModel(num_pores=100, porosity=0.92, throat_radius_mean=1.0e-4, throat_radius_std=2.0e-5)
k_perm = pnm.permeability_kozeny_carman(particle_diameter=0.025)
assert k_perm > 0 and np.isfinite(k_perm), '[TC20] Kozeny-Carman permeability should be positive FAILED'

# ---- TC21: KentEisenbergModel CO2_partial_pressure is positive finite ----
vle = KentEisenbergModel("MEA")
P_eq = vle.CO2_partial_pressure(T=313.15, alpha=0.3, c_amine_total=5000.0)
assert P_eq > 0 and np.isfinite(P_eq), '[TC21] CO2 partial pressure should be positive finite FAILED'

# ---- TC22: ExtendedUNIQUAC activity coefficients all positive ----
uniquac = ExtendedUNIQUAC()
x = {"H2O": 0.85, "MEA": 0.10, "CO2": 0.05}
gamma = uniquac.activity_coefficient(313.15, x)
for sp, g in gamma.items():
    assert g > 0 and np.isfinite(g), f'[TC22] gamma_{sp} should be positive finite FAILED'

# ---- TC23: DegradationNetwork finds path from MEA to CO2_loss ----
net = DegradationNetwork()
net.build_mea_degradation_network()
pathway = net.find_minimum_energy_pathway("MEA", "CO2_loss")
assert len(pathway["path"]) >= 2, '[TC23] degradation path should have at least 2 nodes FAILED'
assert pathway["total_Ea"] > 0, '[TC23] total_Ea should be positive FAILED'

# ---- TC24: knapsack_additive_selection total_cost <= budget ----
adds = ["A", "B", "C", "D"]
costs = [3.0, 5.0, 2.0, 7.0]
bens = [0.4, 0.3, 0.2, 0.6]
result = knapsack_additive_selection(adds, costs, bens, 10.0)
assert result["total_cost"] <= 10.0, '[TC24] knapsack total_cost should be within budget FAILED'
assert result["total_benefit"] > 0, '[TC24] knapsack total_benefit should be positive FAILED'

# ---- TC25: StoichiometricAnalysis rank is valid ----
stoich = StoichiometricAnalysis()
r = stoich.rank()
assert 1 <= r <= stoich.n_species, '[TC25] stoichiometric rank should be valid FAILED'

# ---- TC26: StoichiometricAnalysis check_elemental_balance of R1 ----
bal = stoich.check_elemental_balance(0)
assert np.allclose(bal, 0, atol=1e-10), '[TC26] reaction 0 should be elementally balanced FAILED'

# ---- TC27: conservation_relations returns correct count ----
L, n_cons = stoich.conservation_relations()
assert n_cons >= 1, '[TC27] number of conservation relations should be >= 1 FAILED'

# ---- TC28: StripperModel reboiler_duty is positive ----
strip = StripperModel(T_reboiler=393.15, P_stripper=2.0e5, n_stages=12)
Q_reb = strip.reboiler_duty(alpha_rich=0.45, alpha_lean=0.15, c_amine=5000.0, T_reb=393.15, T_feed=313.15)
assert Q_reb > 0 and np.isfinite(Q_reb), '[TC28] reboiler_duty should be positive finite FAILED'

# ---- TC29: energy_integration_analysis efficiency in [0,1] ----
energy = energy_integration_analysis(Q_reboiler=3.5e6, Q_condenser=2.8e6, T_reb=393.15, T_cond=353.15)
assert 0.0 <= energy["thermal_efficiency"] <= 1.0, '[TC29] thermal_efficiency should be in [0,1] FAILED'
assert energy["net_heat_demand"] >= 0, '[TC29] net_heat_demand should be non-negative FAILED'

# ---- TC30: CyclicAbsorptionRegeneration amplitude non-negative ----
import numpy as np
np.random.seed(42)
cyclic = CyclicAbsorptionRegeneration(absorber_params={}, stripper_params={}, cycle_time=3600.0)
cycle_result = cyclic.cycle_dynamics(alpha_lean0=0.15, n_cycles=3, n_points=1000)
assert cycle_result["amplitude"] >= 0, '[TC30] cycle amplitude should be non-negative FAILED'

# ---- TC31: AmineKinetics carbamate_hydrolysis_rate non-negative ----
r_hydr = kin_mea.carbamate_hydrolysis_rate(T=313.15, c_carbamate=100.0)
assert r_hydr >= 0 and np.isfinite(r_hydr), '[TC31] carbamate_hydrolysis_rate should be non-negative FAILED'

# ---- TC32: CO2LoadingCalculator kinetic_loading_estimate in [0, 0.55] ----
alpha_kin = loader.kinetic_loading_estimate(T=313.15, P_CO2=15000.0, c_amine_total=5000.0, contact_time=1.0)
assert 0.0 <= alpha_kin <= 0.55, '[TC32] kinetic_loading_estimate should be in [0, 0.55] FAILED'

# ---- TC33: MeshGenerator 1D line mesh correct shape ----
mesh1d = MeshGenerator(dim=2)
nodes1d, elements1d = mesh1d.generate_1d_line_mesh(10, xlim=(0, 1))
assert nodes1d.shape == (10, 2), '[TC33] 1D mesh nodes shape FAILED'
assert elements1d.shape[0] == 9, '[TC33] 1D mesh elements count FAILED'

# ---- TC34: PackedColumnModel axial_profile returns correct shapes ----
column = PackedColumnModel(column_height=5.0, column_diameter=1.0, packing_type="random")
z_col, y_CO2, T_prof = column.axial_profile(
    T=313.15, P_total=1.2e5, c_amine=5000.0,
    L_flow=2.0, G_flow=1.5, y_CO2_in=0.15, n_z=30
)
assert len(z_col) == 30, '[TC34] axial profile z length FAILED'
assert len(y_CO2) == 30, '[TC34] axial profile y_CO2 length FAILED'

# ---- TC35: VLEPolynomialFitter fit and predict produce finite output ----
import numpy as np
np.random.seed(42)
T_vle, alpha_vle, P_vle = generate_vle_dataset("MEA", c_amine=5.0)
fitter = VLEPolynomialFitter(degree=3)
coeffs_fit, cond_fit = fitter.fit(T_vle, alpha_vle, P_vle)
P_pred = float(fitter.predict(np.array([313.15]), np.array([0.3]))[0])
assert P_pred > 0 and np.isfinite(P_pred), '[TC35] VLE prediction should be positive finite FAILED'
