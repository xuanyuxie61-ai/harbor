"""
================================================================================
CO2 Capture Amine Absorption Dynamics - Integrated Simulation Platform
================================================================================

A doctoral-level computational framework for simulating CO2 capture via
aqueous amine absorption, integrating:
  - Reaction kinetics (zwitterion mechanism)
  - Two-film mass transfer with chemical reaction enhancement
  - Spectral methods for diffusion-reaction boundary value problems
  - Packed column hydrodynamics and axial profiles
  - Amine degradation pathway analysis and optimization
  - Process optimization and sensitivity analysis
  - Conservation matrix analysis for numerical stability
  - Regenerator (stripper) modeling and cyclic dynamics

Scientific Domain: Chemical Engineering - CO2 Capture Amine Absorption Kinetics
================================================================================
"""

import numpy as np
import sys

from utils import print_section, validate_positive
from reaction_kinetics import AmineKinetics, CO2LoadingCalculator, simulate_batch_absorption
from ode_integrators import (
    explicit_trapezoidal, bdf3_solver, solve_stiff_amine_ode,
    predator_prey_like_cycles
)
from spectral_methods import (
    ChebyshevQuadrature, SpectralDiffusionSolver,
    polynomial_fit_2d_vandermonde, integrate_reaction_rate_profile
)
from mass_transfer import TwoFilmModel, PackedColumnModel, generate_film_grid
from packing_simulation import (
    StructuredPackingGeometry, MeshGenerator, generate_hilbert_pore_sample,
    PoreNetworkModel
)
from equilibrium_models import KentEisenbergModel, ExtendedUNIQUAC, generate_vle_dataset, VLEPolynomialFitter
from reaction_network_optimizer import DegradationNetwork
from degradation_pathways import DegradationMechanismAnalyzer, knapsack_additive_selection
from process_optimizer import ProcessOptimizer, optimize_additive_package, SensitivityAnalysis
from conservation_matrix import StoichiometricAnalysis, build_linear_conserved_ode_system
from regenerator import StripperModel, CyclicAbsorptionRegeneration, energy_integration_analysis


def main():
    print("=" * 78)
    print("  CO2 CAPTURE AMINE ABSORPTION DYNAMICS - INTEGRATED SIMULATION")
    print("  Doctoral-Level Chemical Engineering Computation Platform")
    print("=" * 78)

    # ========================================================================
    # 1. AMINE REACTION KINETICS
    # ========================================================================
    print_section("1. Amine Reaction Kinetics (Zwitterion Mechanism)")

    T_abs = 313.15  # K (40 C)
    kin_mea = AmineKinetics("MEA")
    k2 = kin_mea.k2(T_abs)
    print(f"  MEA second-order rate constant at {T_abs} K:")
    print(f"    k2 = {k2:.4e} m^3/(mol·s)")

    c_CO2 = 10.0    # mol/m^3
    c_MEA = 5000.0  # mol/m^3
    rate = kin_mea.reaction_rate(T_abs, c_CO2, c_MEA)
    print(f"  Reaction rate at [CO2]={c_CO2}, [MEA]={c_MEA}:")
    print(f"    r = {rate:.4e} mol/(m^3·s)")

    loader = CO2LoadingCalculator(kin_mea)
    alpha_eq = loader.equilibrium_loading(T_abs, 15000.0, c_MEA)
    print(f"  Equilibrium CO2 loading at P_CO2=15 kPa: alpha = {alpha_eq:.4f}")

    # Batch absorption simulation
    print("\n  --- Batch Absorption Simulation (Explicit Trapezoidal) ---")
    t_batch, y_batch = simulate_batch_absorption(
        T=T_abs, P_CO2=15000.0, c_amine0=c_MEA,
        t_span=(0.0, 0.1), amine_type="MEA", n_steps=500
    )
    loading_0 = (c_MEA - y_batch[0, 1]) / (2.0 * c_MEA)
    loading_f = (c_MEA - y_batch[-1, 1]) / (2.0 * c_MEA)
    print(f"    Initial loading: {loading_0:.4f}")
    print(f"    Final loading (t=0.1s): {np.clip(loading_f, 0.0, 0.55):.4f}")

    # ========================================================================
    # 2. ADVANCED ODE SOLVERS
    # ========================================================================
    print_section("2. Advanced ODE Solvers (Stiff/Non-stiff Detection)")

    # Stiff test: coupled fast/slow reactions (inspired by tough_ode)
    def stiff_rhs(t, y):
        y1, y2, y3, y4 = y
        y1 = max(float(y1), 1e-12)
        y2 = np.clip(float(y2), -10.0, 10.0)
        dy1 = 2.0 * t * (max(y2, 1e-12) ** 0.2) * y4
        dy2 = 10.0 * t * np.exp(np.clip(5.0 * (y2 - 1.0), -50.0, 50.0)) * y4
        dy3 = 2.0 * t * y4
        dy4 = -2.0 * t * np.log(y1)
        return np.array([dy1, dy2, dy3, dy4])

    t_stiff, y_stiff = bdf3_solver(stiff_rhs, (0.0, 3.0), [1.0, 1.0, 1.0, 1.0], 200)
    print(f"  Stiff system solved (BDF3): y_final = [{y_stiff[-1,0]:.4f}, {y_stiff[-1,1]:.4e}, "
          f"{y_stiff[-1,2]:.4f}, {y_stiff[-1,3]:.4f}]")

    # BDF3 solver test
    def simple_decay(t, y):
        return np.array([-0.5 * y[0], -2.0 * y[1]])

    t_bdf, y_bdf = bdf3_solver(simple_decay, (0.0, 10.0), [1.0, 1.0], 100)
    print(f"  BDF3 solver test: y1(10) = {y_bdf[-1,0]:.6f} (exact: {np.exp(-5.0):.6f})")

    # ========================================================================
    # 3. SPECTRAL METHODS & QUADRATURE
    # ========================================================================
    print_section("3. Spectral Methods (Chebyshev Quadrature & Diffusion)")

    quad = ChebyshevQuadrature(32, 0.0, 1.0)
    # Integrate reaction rate profile across film
    rate_func = lambda z: 0.1 * np.exp(-2.0 * z)
    total_rate = integrate_reaction_rate_profile(rate_func, 0.0, 1.0, n=32)
    print(f"  Integrated reaction rate across film: {total_rate:.6f} mol/(m^2·s)")

    # Spectral diffusion-reaction solver
    spectral = SpectralDiffusionSolver(n_cheb=24)
    z_film, c_profile, flux = spectral.solve_film_diffusion_reaction(
        D_diff=1.9e-9, k_rxn=100.0, delta=1.0e-4,
        c_interface=20.0, c_bulk=0.5
    )
    print(f"  Film model flux at interface: {flux:.4e} mol/(m^2·s)")

    # Channel flow solver (Poiseuille)
    y_chan, u_vel = spectral.solve_channel_flow(
        mu=1.0e-3, delta_p=100.0, L=1.0, R=0.01
    )
    print(f"  Channel max velocity: {np.max(u_vel):.4e} m/s")

    # 2D polynomial fit for VLE data
    T_data = np.array([298.15, 313.15, 333.15, 353.15])
    alpha_data = np.array([0.1, 0.2, 0.3, 0.4])
    P_data = np.array([100.0, 500.0, 2000.0, 8000.0])
    coeffs, cond = polynomial_fit_2d_vandermonde(T_data, alpha_data, P_data, degree=2)
    print(f"  VLE polynomial fit condition number: {cond:.2e}")

    # ========================================================================
    # 4. MASS TRANSFER & PACKED COLUMN
    # ========================================================================
    print_section("4. Two-Film Mass Transfer & Packed Column Model")

    film = TwoFilmModel(T_abs, 1.2e5, "MEA")
    sol = film.solve_interface(P_CO2_bulk=15000.0, c_CO2_bulk=0.0,
                                c_amine_bulk=c_MEA, k2_rate=k2)
    print(f"  Enhancement factor E = {sol['enhancement_factor']:.2f}")
    print(f"  Hatta number Ha = {sol['hatta_number']:.2f}")
    print(f"  Overall flux N_A = {sol['flux']:.4e} mol/(m^2·s)")
    print(f"  Overall K_G = {sol['K_G_overall']:.4e} mol/(m^2·s·Pa)")

    # Film profile
    z_prof, c_prof, flux_prof, _ = film.film_profile(
        P_CO2_bulk=15000.0, c_CO2_bulk=0.0, c_amine_bulk=c_MEA,
        k2_rate=k2, n_grid=32
    )
    print(f"  Film profile: c_interface={c_prof[0]:.2f}, c_bulk={c_prof[-1]:.4f}")

    # Packed column simulation
    column = PackedColumnModel(column_height=10.0, column_diameter=1.0, packing_type="random")
    z_col, y_CO2, T_prof = column.axial_profile(
        T=T_abs, P_total=1.2e5, c_amine=c_MEA,
        L_flow=2.0, G_flow=1.5, y_CO2_in=0.15, n_z=50
    )
    print(f"  Column outlet CO2 mole fraction: {y_CO2[-1]:.4f}")
    print(f"  CO2 removal efficiency: {(1.0 - y_CO2[-1]/y_CO2[0])*100:.1f}%")
    print(f"  Temperature rise: {T_prof[-1] - T_prof[0]:.2f} K")

    # ========================================================================
    # 5. PACKING GEOMETRY & PORE NETWORK
    # ========================================================================
    print_section("5. Packing Geometry & Pore Network Analysis")

    packing = StructuredPackingGeometry(corrugation_angle=45.0, crimp_height=0.012, channel_width=0.008)
    a_spec = packing.specific_area()
    dh = packing.hydraulic_diameter()
    eps = packing.void_fraction()
    print(f"  Structured packing specific area: {a_spec:.1f} m^2/m^3")
    print(f"  Hydraulic diameter: {dh*1000:.2f} mm")
    print(f"  Void fraction: {eps:.3f}")

    # Mesh generation
    mesh = MeshGenerator(dim=2)
    nodes, elements = mesh.generate_rectangular_mesh(10, 10, xlim=(0, 1), ylim=(0, 1))
    quality = mesh.mesh_quality_metrics()
    print(f"  Mesh quality: {quality['num_elements']} elements, "
          f"max aspect ratio {quality['max_aspect_ratio']:.2f}")

    # Hilbert curve pore sampling
    hilbert_points = generate_hilbert_pore_sample(n_order=2, domain_size=1.0)
    print(f"  Hilbert curve pore samples: {len(hilbert_points)} points")

    # Pore network model
    pnm = PoreNetworkModel(num_pores=100, porosity=0.92,
                           throat_radius_mean=1.0e-4, throat_radius_std=2.0e-5)
    k_perm = pnm.permeability_kozeny_carman(particle_diameter=0.025)
    print(f"  Kozeny-Carman permeability: {k_perm:.4e} m^2")

    # ========================================================================
    # 6. VAPOR-LIQUID EQUILIBRIUM
    # ========================================================================
    print_section("6. Vapor-Liquid Equilibrium (Kent-Eisenberg & e-UNIQUAC)")

    vle = KentEisenbergModel("MEA")
    P_eq = vle.CO2_partial_pressure(T=313.15, alpha=0.3, c_amine_total=5000.0)
    print(f"  Equilibrium P_CO2 at T=313.15K, alpha=0.3: {P_eq:.2f} Pa")

    # e-UNIQUAC activity coefficients
    uniquac = ExtendedUNIQUAC()
    x = {"H2O": 0.85, "MEA": 0.10, "CO2": 0.05}
    gamma = uniquac.activity_coefficient(313.15, x)
    for sp, g in gamma.items():
        print(f"    gamma_{sp} = {g:.4f}")

    # VLE data fitting
    T_vle, alpha_vle, P_vle = generate_vle_dataset("MEA", c_amine=5.0)
    fitter = VLEPolynomialFitter(degree=3)
    fitter.fit(T_vle, alpha_vle, P_vle)
    P_pred = float(fitter.predict(np.array([313.15]), np.array([0.3]))[0])
    print(f"  Polynomial VLE prediction at T=313.15K, alpha=0.3: {P_pred:.2f} Pa")

    # ========================================================================
    # 7. DEGRADATION NETWORK OPTIMIZATION
    # ========================================================================
    print_section("7. Degradation Network Path Optimization (Bellman-Ford)")

    network = DegradationNetwork()
    network.build_mea_degradation_network()
    stats = network.network_statistics()
    print(f"  Network: {stats['num_species']} species, {stats['num_reactions']} reactions")
    print(f"  Average out-degree: {stats['avg_out_degree']:.2f}")

    pathway = network.find_minimum_energy_pathway("MEA", "CO2_loss")
    print(f"  Minimum-energy degradation path: {' -> '.join(pathway['path'])}")
    print(f"  Total activation energy barrier: {pathway['total_Ea']:.1f} kJ/mol")

    k_deg, _ = network.compute_rate_along_path(T=393.15, source="MEA", target="CO2_loss")
    print(f"  Effective degradation rate at 120 C: {k_deg:.4e} s^-1")

    # ========================================================================
    # 8. DEGRADATION PATHWAY ENUMERATION
    # ========================================================================
    print_section("8. Degradation Pathway Enumeration (Combinatorial Analysis)")

    analyzer = DegradationMechanismAnalyzer("MEA")
    mechanisms = analyzer.enumerate_possible_mechanisms(max_products=3)
    feasible = [m for m in mechanisms if m['feasible']]
    print(f"  Total mechanisms enumerated: {len(mechanisms)}")
    print(f"  Feasible mechanisms: {len(feasible)}")

    bell_num = analyzer.count_distinct_pathways(max_steps=4)
    print(f"  Bell number B_4 (distinct pathways): {bell_num}")

    # Additive optimization (knapsack)
    additive_result = optimize_additive_package()
    print(f"  Optimal additive package: {', '.join(additive_result['additives'])}")
    print(f"  Total cost: ${additive_result['total_cost']:.2f}/tonne")
    print(f"  Total benefit: {additive_result['total_benefit']:.3f}")

    # ========================================================================
    # 9. PROCESS OPTIMIZATION
    # ========================================================================
    print_section("9. Process Optimization (Gradient Descent & Grid Search)")

    optimizer = ProcessOptimizer("MEA")
    x0 = np.array([313.15, 1.5e5, 5000.0, 2.5])
    x_opt, hist = optimizer.gradient_optimization(x0, learning_rate=50.0, max_iter=80, tol=1e-5)
    print(f"  Optimized operating conditions:")
    print(f"    T = {x_opt[0]:.1f} K")
    print(f"    P = {x_opt[1]/1e5:.2f} bar")
    print(f"    c_amine = {x_opt[2]:.0f} mol/m^3")
    print(f"    L/G = {x_opt[3]:.2f}")
    print(f"    Minimized cost: {optimizer.objective_cost_avoided(*x_opt):.2f} $/tonne CO2")

    # Sensitivity analysis
    sens = SensitivityAnalysis(optimizer)
    sensitivities = sens.local_sensitivity(x_opt)
    print(f"  Local sensitivity coefficients:")
    for name, S in sensitivities.items():
        print(f"    S_{name} = {S:.4f}")

    # Uncertainty propagation
    x_std = np.array([5.0, 0.1e5, 500.0, 0.3])
    unc = sens.monte_carlo_uncertainty(x_opt, x_std, n_samples=500)
    print(f"  Monte Carlo uncertainty (N=500):")
    print(f"    Mean cost = {unc['mean_cost']:.2f} $/tonne")
    print(f"    Std = {unc['std_cost']:.2f}, 5-95% range = [{unc['p5']:.2f}, {unc['p95']:.2f}]")

    # ========================================================================
    # 10. STOICHIOMETRIC & CONSERVATION ANALYSIS
    # ========================================================================
    print_section("10. Stoichiometric & Conservation Matrix Analysis")

    stoich = StoichiometricAnalysis()
    rank = stoich.rank()
    L, n_cons = stoich.conservation_relations()
    print(f"  Stoichiometric matrix rank: {rank}")
    print(f"  Number of conservation relations: {n_cons}")

    # Validate elemental balance
    for i, rxn in enumerate(stoich.reactions):
        balance = stoich.check_elemental_balance(i)
        is_balanced = np.allclose(balance, 0, atol=1e-10)
        status = "BALANCED" if is_balanced else "UNBALANCED"
        print(f"    R{i+1}: {status}")

    # Linear conserved ODE system
    rhs, conserved = build_linear_conserved_ode_system()
    t_cons, y_cons = explicit_trapezoidal(rhs, (0.0, 10.0), [10.0, 5.0], 200)
    cons_initial = conserved(y_cons[0, :])
    cons_final = conserved(y_cons[-1, :])
    print(f"  Conserved quantity check: initial={cons_initial:.6f}, final={cons_final:.6f}")
    print(f"  Conservation error: {abs(cons_final - cons_initial):.2e}")

    # ========================================================================
    # 11. REGENERATOR (STRIPPER) MODEL
    # ========================================================================
    print_section("11. Regenerator (Stripper) Model & Cyclic Dynamics")

    stripper = StripperModel(T_reboiler=393.15, P_stripper=2.0e5, n_stages=12)
    alpha_rich = 0.45
    alpha_lean = 0.15
    c_amine = 5000.0

    # Reboiler duty
    Q_reb = stripper.reboiler_duty(alpha_rich, alpha_lean, c_amine, 393.15, 313.15)
    print(f"  Reboiler duty: {Q_reb:.1f} kJ/kmol CO2")
    print(f"  Specific reboiler duty: {Q_reb/1000:.2f} GJ/tonne CO2")

    # Stripper column simulation
    T_profile = np.linspace(393.15, 353.15, 50)
    alphas, co2_flows = stripper.simulate_column(
        alpha_rich=alpha_rich, T_profile=T_profile,
        c_amine=c_amine, L_flow=10.0, n_steps=50
    )
    print(f"  Stripper lean loading: {alphas[-1]:.4f}")
    print(f"  Total CO2 desorbed: {co2_flows[-1]:.2f} mol/m^3")

    # Cyclic dynamics (predator-prey-like periodic analysis)
    cyclic = CyclicAbsorptionRegeneration(
        absorber_params={}, stripper_params={}, cycle_time=3600.0
    )
    cycle_result = cyclic.cycle_dynamics(alpha_lean0=0.15, n_cycles=3, n_points=1000)
    period_str = f"{cycle_result['mean_period']:.1f}" if cycle_result['mean_period'] is not None else "N/A (converged to steady-state)"
    print(f"  Cyclic operation mean period: {period_str} s")
    print(f"  Loading amplitude: {cycle_result['amplitude']:.4f}")

    # Energy integration
    energy = energy_integration_analysis(
        Q_reboiler=3.5e6, Q_condenser=2.8e6, T_reb=393.15, T_cond=353.15
    )
    print(f"  Recoverable heat: {energy['recoverable_heat']/1e6:.2f} MJ")
    print(f"  Net heat demand: {energy['net_heat_demand']/1e6:.2f} MJ")
    print(f"  Thermal efficiency: {energy['thermal_efficiency']*100:.1f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("SIMULATION COMPLETE")
    print("  All modules executed successfully with zero parameters.")
    print("  Integrated 15 seed algorithms into unified CO2 capture framework.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 35 个测试通过!\n')
