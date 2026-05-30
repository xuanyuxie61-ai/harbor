
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




    print_section("1. Amine Reaction Kinetics (Zwitterion Mechanism)")

    T_abs = 313.15
    kin_mea = AmineKinetics("MEA")
    k2 = kin_mea.k2(T_abs)
    print(f"  MEA second-order rate constant at {T_abs} K:")
    print(f"    k2 = {k2:.4e} m^3/(mol·s)")

    c_CO2 = 10.0
    c_MEA = 5000.0
    rate = kin_mea.reaction_rate(T_abs, c_CO2, c_MEA)
    print(f"  Reaction rate at [CO2]={c_CO2}, [MEA]={c_MEA}:")
    print(f"    r = {rate:.4e} mol/(m^3·s)")

    loader = CO2LoadingCalculator(kin_mea)
    alpha_eq = loader.equilibrium_loading(T_abs, 15000.0, c_MEA)
    print(f"  Equilibrium CO2 loading at P_CO2=15 kPa: alpha = {alpha_eq:.4f}")


    print("\n  --- Batch Absorption Simulation (Explicit Trapezoidal) ---")
    t_batch, y_batch = simulate_batch_absorption(
        T=T_abs, P_CO2=15000.0, c_amine0=c_MEA,
        t_span=(0.0, 0.1), amine_type="MEA", n_steps=500
    )
    loading_0 = (c_MEA - y_batch[0, 1]) / (2.0 * c_MEA)
    loading_f = (c_MEA - y_batch[-1, 1]) / (2.0 * c_MEA)
    print(f"    Initial loading: {loading_0:.4f}")
    print(f"    Final loading (t=0.1s): {np.clip(loading_f, 0.0, 0.55):.4f}")




    print_section("2. Advanced ODE Solvers (Stiff/Non-stiff Detection)")


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


    def simple_decay(t, y):
        return np.array([-0.5 * y[0], -2.0 * y[1]])

    t_bdf, y_bdf = bdf3_solver(simple_decay, (0.0, 10.0), [1.0, 1.0], 100)
    print(f"  BDF3 solver test: y1(10) = {y_bdf[-1,0]:.6f} (exact: {np.exp(-5.0):.6f})")




    print_section("3. Spectral Methods (Chebyshev Quadrature & Diffusion)")

    quad = ChebyshevQuadrature(32, 0.0, 1.0)

    rate_func = lambda z: 0.1 * np.exp(-2.0 * z)
    total_rate = integrate_reaction_rate_profile(rate_func, 0.0, 1.0, n=32)
    print(f"  Integrated reaction rate across film: {total_rate:.6f} mol/(m^2·s)")


    spectral = SpectralDiffusionSolver(n_cheb=24)
    z_film, c_profile, flux = spectral.solve_film_diffusion_reaction(
        D_diff=1.9e-9, k_rxn=100.0, delta=1.0e-4,
        c_interface=20.0, c_bulk=0.5
    )
    print(f"  Film model flux at interface: {flux:.4e} mol/(m^2·s)")


    y_chan, u_vel = spectral.solve_channel_flow(
        mu=1.0e-3, delta_p=100.0, L=1.0, R=0.01
    )
    print(f"  Channel max velocity: {np.max(u_vel):.4e} m/s")


    T_data = np.array([298.15, 313.15, 333.15, 353.15])
    alpha_data = np.array([0.1, 0.2, 0.3, 0.4])
    P_data = np.array([100.0, 500.0, 2000.0, 8000.0])
    coeffs, cond = polynomial_fit_2d_vandermonde(T_data, alpha_data, P_data, degree=2)
    print(f"  VLE polynomial fit condition number: {cond:.2e}")




    print_section("4. Two-Film Mass Transfer & Packed Column Model")

    film = TwoFilmModel(T_abs, 1.2e5, "MEA")
    sol = film.solve_interface(P_CO2_bulk=15000.0, c_CO2_bulk=0.0,
                                c_amine_bulk=c_MEA, k2_rate=k2)
    print(f"  Enhancement factor E = {sol['enhancement_factor']:.2f}")
    print(f"  Hatta number Ha = {sol['hatta_number']:.2f}")
    print(f"  Overall flux N_A = {sol['flux']:.4e} mol/(m^2·s)")
    print(f"  Overall K_G = {sol['K_G_overall']:.4e} mol/(m^2·s·Pa)")


    z_prof, c_prof, flux_prof, _ = film.film_profile(
        P_CO2_bulk=15000.0, c_CO2_bulk=0.0, c_amine_bulk=c_MEA,
        k2_rate=k2, n_grid=32
    )
    print(f"  Film profile: c_interface={c_prof[0]:.2f}, c_bulk={c_prof[-1]:.4f}")


    column = PackedColumnModel(column_height=10.0, column_diameter=1.0, packing_type="random")
    z_col, y_CO2, T_prof = column.axial_profile(
        T=T_abs, P_total=1.2e5, c_amine=c_MEA,
        L_flow=2.0, G_flow=1.5, y_CO2_in=0.15, n_z=50
    )
    print(f"  Column outlet CO2 mole fraction: {y_CO2[-1]:.4f}")
    print(f"  CO2 removal efficiency: {(1.0 - y_CO2[-1]/y_CO2[0])*100:.1f}%")
    print(f"  Temperature rise: {T_prof[-1] - T_prof[0]:.2f} K")




    print_section("5. Packing Geometry & Pore Network Analysis")

    packing = StructuredPackingGeometry(corrugation_angle=45.0, crimp_height=0.012, channel_width=0.008)
    a_spec = packing.specific_area()
    dh = packing.hydraulic_diameter()
    eps = packing.void_fraction()
    print(f"  Structured packing specific area: {a_spec:.1f} m^2/m^3")
    print(f"  Hydraulic diameter: {dh*1000:.2f} mm")
    print(f"  Void fraction: {eps:.3f}")


    mesh = MeshGenerator(dim=2)
    nodes, elements = mesh.generate_rectangular_mesh(10, 10, xlim=(0, 1), ylim=(0, 1))
    quality = mesh.mesh_quality_metrics()
    print(f"  Mesh quality: {quality['num_elements']} elements, "
          f"max aspect ratio {quality['max_aspect_ratio']:.2f}")


    hilbert_points = generate_hilbert_pore_sample(n_order=2, domain_size=1.0)
    print(f"  Hilbert curve pore samples: {len(hilbert_points)} points")


    pnm = PoreNetworkModel(num_pores=100, porosity=0.92,
                           throat_radius_mean=1.0e-4, throat_radius_std=2.0e-5)
    k_perm = pnm.permeability_kozeny_carman(particle_diameter=0.025)
    print(f"  Kozeny-Carman permeability: {k_perm:.4e} m^2")




    print_section("6. Vapor-Liquid Equilibrium (Kent-Eisenberg & e-UNIQUAC)")

    vle = KentEisenbergModel("MEA")
    P_eq = vle.CO2_partial_pressure(T=313.15, alpha=0.3, c_amine_total=5000.0)
    print(f"  Equilibrium P_CO2 at T=313.15K, alpha=0.3: {P_eq:.2f} Pa")


    uniquac = ExtendedUNIQUAC()
    x = {"H2O": 0.85, "MEA": 0.10, "CO2": 0.05}
    gamma = uniquac.activity_coefficient(313.15, x)
    for sp, g in gamma.items():
        print(f"    gamma_{sp} = {g:.4f}")


    T_vle, alpha_vle, P_vle = generate_vle_dataset("MEA", c_amine=5.0)
    fitter = VLEPolynomialFitter(degree=3)
    fitter.fit(T_vle, alpha_vle, P_vle)
    P_pred = float(fitter.predict(np.array([313.15]), np.array([0.3]))[0])
    print(f"  Polynomial VLE prediction at T=313.15K, alpha=0.3: {P_pred:.2f} Pa")




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




    print_section("8. Degradation Pathway Enumeration (Combinatorial Analysis)")

    analyzer = DegradationMechanismAnalyzer("MEA")
    mechanisms = analyzer.enumerate_possible_mechanisms(max_products=3)
    feasible = [m for m in mechanisms if m['feasible']]
    print(f"  Total mechanisms enumerated: {len(mechanisms)}")
    print(f"  Feasible mechanisms: {len(feasible)}")

    bell_num = analyzer.count_distinct_pathways(max_steps=4)
    print(f"  Bell number B_4 (distinct pathways): {bell_num}")


    additive_result = optimize_additive_package()
    print(f"  Optimal additive package: {', '.join(additive_result['additives'])}")
    print(f"  Total cost: ${additive_result['total_cost']:.2f}/tonne")
    print(f"  Total benefit: {additive_result['total_benefit']:.3f}")




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


    sens = SensitivityAnalysis(optimizer)
    sensitivities = sens.local_sensitivity(x_opt)
    print(f"  Local sensitivity coefficients:")
    for name, S in sensitivities.items():
        print(f"    S_{name} = {S:.4f}")


    x_std = np.array([5.0, 0.1e5, 500.0, 0.3])
    unc = sens.monte_carlo_uncertainty(x_opt, x_std, n_samples=500)
    print(f"  Monte Carlo uncertainty (N=500):")
    print(f"    Mean cost = {unc['mean_cost']:.2f} $/tonne")
    print(f"    Std = {unc['std_cost']:.2f}, 5-95% range = [{unc['p5']:.2f}, {unc['p95']:.2f}]")




    print_section("10. Stoichiometric & Conservation Matrix Analysis")

    stoich = StoichiometricAnalysis()
    rank = stoich.rank()
    L, n_cons = stoich.conservation_relations()
    print(f"  Stoichiometric matrix rank: {rank}")
    print(f"  Number of conservation relations: {n_cons}")


    for i, rxn in enumerate(stoich.reactions):
        balance = stoich.check_elemental_balance(i)
        is_balanced = np.allclose(balance, 0, atol=1e-10)
        status = "BALANCED" if is_balanced else "UNBALANCED"
        print(f"    R{i+1}: {status}")


    rhs, conserved = build_linear_conserved_ode_system()
    t_cons, y_cons = explicit_trapezoidal(rhs, (0.0, 10.0), [10.0, 5.0], 200)
    cons_initial = conserved(y_cons[0, :])
    cons_final = conserved(y_cons[-1, :])
    print(f"  Conserved quantity check: initial={cons_initial:.6f}, final={cons_final:.6f}")
    print(f"  Conservation error: {abs(cons_final - cons_initial):.2e}")




    print_section("11. Regenerator (Stripper) Model & Cyclic Dynamics")

    stripper = StripperModel(T_reboiler=393.15, P_stripper=2.0e5, n_stages=12)
    alpha_rich = 0.45
    alpha_lean = 0.15
    c_amine = 5000.0


    Q_reb = stripper.reboiler_duty(alpha_rich, alpha_lean, c_amine, 393.15, 313.15)
    print(f"  Reboiler duty: {Q_reb:.1f} kJ/kmol CO2")
    print(f"  Specific reboiler duty: {Q_reb/1000:.2f} GJ/tonne CO2")


    T_profile = np.linspace(393.15, 353.15, 50)
    alphas, co2_flows = stripper.simulate_column(
        alpha_rich=alpha_rich, T_profile=T_profile,
        c_amine=c_amine, L_flow=10.0, n_steps=50
    )
    print(f"  Stripper lean loading: {alphas[-1]:.4f}")
    print(f"  Total CO2 desorbed: {co2_flows[-1]:.2f} mol/m^3")


    cyclic = CyclicAbsorptionRegeneration(
        absorber_params={}, stripper_params={}, cycle_time=3600.0
    )
    cycle_result = cyclic.cycle_dynamics(alpha_lean0=0.15, n_cycles=3, n_points=1000)
    period_str = f"{cycle_result['mean_period']:.1f}" if cycle_result['mean_period'] is not None else "N/A (converged to steady-state)"
    print(f"  Cyclic operation mean period: {period_str} s")
    print(f"  Loading amplitude: {cycle_result['amplitude']:.4f}")


    energy = energy_integration_analysis(
        Q_reboiler=3.5e6, Q_condenser=2.8e6, T_reb=393.15, T_cond=353.15
    )
    print(f"  Recoverable heat: {energy['recoverable_heat']/1e6:.2f} MJ")
    print(f"  Net heat demand: {energy['net_heat_demand']/1e6:.2f} MJ")
    print(f"  Thermal efficiency: {energy['thermal_efficiency']*100:.1f}%")




    print_section("SIMULATION COMPLETE")
    print("  All modules executed successfully with zero parameters.")
    print("  Integrated 15 seed algorithms into unified CO2 capture framework.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
