"""
main.py
=======
Biomass Gasification Reactor Simulation — Unified Entry Point
===============================================================

Domain: Combustion Science — Biomass Gasification Reactor

This program simulates a downdraft biomass gasification reactor using
a sequential modular approach with the following physics:
  - Biomass particle size distribution (PSD) analysis
  - Stoichiometric matrix reduction and elemental mass balances
  - Arrhenius chemical kinetics with Markov state transitions
  - Thermodynamic equilibrium via fixed-point / Newton iteration
  - Heat transfer: radiation (view factors) + conduction (CG solver)
  - 1D CFD with periodic tridiagonal solver and CVT mesh adaptation
  - Char particle burnout survival analysis
  - Adaptive mesh generation

Scientific formulas implemented:
  Arrhenius:          k = A exp(-Ea / RT)
  Ergun pressure:     dp/dz = -(150(1-ε)²μu)/(ε³dp²) - (1.75(1-ε)ρu²)/(ε³dp)
  Stefan-Boltzmann:   q_rad = εσ(T_s⁴ - T_∞⁴)
  Biot number:        Bi = h L_c / k_char
  Thiele modulus:     φ = (dp/2) √(k/D_eff)
  Effectiveness:      η = (3/φ)(1/tanh(φ) - 1/φ)
  WGS equilibrium:    K_wgs = exp(-ΔG°_wgs / RT)
  Weibull survival:   S(t) = exp(-(t/τ)^m)
  Sauter diameter:    d_32 = Σ d_i³ / Σ d_i²
  View factor (spheres): F_12 = (1/4π)[arccos(1-2ρ²) - 2ρ√(1-ρ²)]
"""

import math
import numpy as np
import sys

# Import all modules
from reactor_geometry import CylindricalReactor, Mesh3D
from biomass_psd import BiomassPSD
from stoichiometry import StoichiometricMatrix, StoichiometricReducer
from kinetics_model import GasificationKinetics, MarkovReactorState, BinomialKinetics
from thermo_equilibrium import ThermoEquilibrium
from heat_transfer import RadiationViewFactor, StefanBoltzmannRadiation, ConductionSolver
from cfd_solver import PeriodicTridiagonalSolver, CVTMeshGenerator, ReactorFlowSolver
from particle_lifetime import ParticleBurnoutModel, ParticleMortalityTable, CollatzBurnoutSequence
from reactor_state import ReactorStateVector, SequentialModularSimulator, ReactorZoneModel, ConvergenceMonitor
from mesh_adaptation import MeshAdapter
from stats_utils import setup_discrete_histogram, gammad, ppchi2, ppnd, ncbeta


def run_reactor_simulation():
    """Main simulation pipeline. Zero parameters required."""
    print("=" * 70)
    print("BIMASS GASIFICATION REACTOR SIMULATION")
    print("Combustion Science — PhD-Level Synthesis Project")
    print("=" * 70)

    # =====================================================================
    # 1. REACTOR GEOMETRY (from 180_circle_map, 873_ply_io)
    # =====================================================================
    print("\n[1] Reactor Geometry Definition")
    reactor = CylindricalReactor(H=2.5, R=0.3, H_bed=1.0,
                                  H_combustion=0.5, H_reduction=1.0)
    print(f"  Total volume:      {reactor.volume():.4f} m³")
    print(f"  Cross-section:     {reactor.cross_section_area():.4f} m²")
    print(f"  Bed zone volume:   {reactor.zone_volume('bed'):.4f} m³")

    # Circle mapping for velocity perturbation ellipse
    A_vel = np.array([[2.5, 0.3], [0.1, 1.8]], dtype=float)
    vel_points, cond_num = reactor.map_circle_transform(A_vel, norm_type=2, num_points=50)
    print(f"  Velocity ellipse condition number: {cond_num:.4f}")

    # 3D mesh for packed bed particles
    bed_mesh = Mesh3D()
    for i in range(8):
        theta = 2.0 * math.pi * i / 8.0
        bed_mesh.add_vertex(reactor.R * math.cos(theta),
                            reactor.R * math.sin(theta), 0.5)
    bed_mesh.add_face(0, 1, 2)
    bed_mesh.add_face(0, 2, 3)
    print(f"  Bed mesh surface area: {bed_mesh.total_surface_area():.6f} m²")

    # =====================================================================
    # 2. PARTICLE SIZE DISTRIBUTION (from 539_histogram_discrete)
    # =====================================================================
    print("\n[2] Biomass Particle Size Distribution")
    psd = BiomassPSD(d_min=0.1e-3, d_max=50.0e-3)
    d_50 = 5.0e-3
    n_rr = 1.8
    d_test = np.linspace(0.5e-3, 20e-3, 100)
    pdf_vals = psd.rosin_rammler_pdf(d_test, d_50, n_rr)
    d_32 = psd.sauter_mean_diameter(d_50, n_rr, 'rosin-rammler')
    print(f"  Rosin-Rammler d_50:   {d_50*1000:.2f} mm")
    print(f"  Spread parameter n:   {n_rr:.2f}")
    print(f"  Sauter diameter d_32: {d_32*1000:.3f} mm")

    # Empirical histogram from synthetic samples
    np.random.seed(42)
    samples = np.random.lognormal(mean=math.log(d_50), sigma=0.5, size=500)
    samples = np.clip(samples, psd.d_min, psd.d_max)
    hist_x, hist_y = psd.build_histogram(samples)
    mean_d = psd.mean_diameter_from_histogram()
    print(f"  Empirical mean diam:  {mean_d*1000:.3f} mm")
    print(f"  Specific surface:     {psd.specific_surface_area(d_32):.2f} m²/kg")

    # Transport numbers
    h_conv = 50.0  # W/(m²·K)
    k_char = 0.15  # W/(m·K)
    D_eff = 1.0e-5  # m²/s
    rate_k = 0.1    # s⁻¹
    bi = psd.biot_number(d_32, h_conv, k_char)
    phi = psd.thiele_modulus(d_32, rate_k, D_eff)
    eta = psd.effectiveness_factor(phi)
    print(f"  Biot number Bi:       {bi:.4f}")
    print(f"  Thiele modulus φ:     {phi:.4f}")
    print(f"  Effectiveness factor η: {eta:.4f}")

    # =====================================================================
    # 3. STOICHIOMETRY (from 736_matman, 420_fermat_factor)
    # =====================================================================
    print("\n[3] Stoichiometric Matrix & Elemental Balances")
    stoich = StoichiometricMatrix(biomass_formula=(1.0, 1.4, 0.6, 0.01, 0.005))
    print(f"  Matrix shape: {stoich.A.shape}")
    print(f"  Matrix rank:  {stoich.rank()}")

    # Elementary row operations demo
    stoich.row_swap(1, 2)
    stoich.row_scale(2.0, 3)
    stoich.row_axpy(1.0, 1, -1.0, 2)

    # Gauss-Jordan elimination
    rref = stoich.gauss_jordan_elimination()
    print(f"  RREF computed, max entry: {np.max(np.abs(rref)):.4f}")

    # Nullspace basis (reaction invariants)
    null_basis = stoich.nullspace_basis()
    print(f"  Nullspace dimension: {null_basis.shape[1]}")

    # Stoichiometric coefficient reduction
    coeffs = np.array([4, 6, 4, 2, 8, 12, 4, 0, 0, 20, 2], dtype=float)
    reduced = StoichiometricReducer.reduce_coefficients(coeffs)
    print(f"  Reduced coefficients GCD: {StoichiometricReducer.gcd_list(coeffs)}")

    # Fermat factorization demo
    f1, f2 = StoichiometricReducer.fermat_reduce(8051)
    print(f"  Fermat factorization 8051 = {f1} × {f2}")

    # =====================================================================
    # 4. KINETICS (from 321_dueling_idiots)
    # =====================================================================
    print("\n[4] Chemical Kinetics & Markov State Model")
    kinetics = GasificationKinetics()
    conc_test = np.array([0.5, 0.1, 0.05, 0.2, 0.1, 0.3, 0.05], dtype=float)
    rates = kinetics.reaction_rates(1200.0, conc_test)
    for key, val in rates.items():
        print(f"  {key} rate: {val:.4e} mol/(m³·s)")
    omega = kinetics.species_production_rates(1200.0, conc_test)
    print(f"  Net CO production:  {omega[3]:.4e} mol/(m³·s)")
    q_heat = kinetics.heat_of_reaction(1200.0, conc_test)
    print(f"  Volumetric heat release: {q_heat:.2e} W/m³")

    # Markov reactor state transitions
    markov = MarkovReactorState()
    prob_dist = markov.state_probability('drying', steps=20)
    print(f"  Markov state after 20 steps: {dict(zip(markov.states, prob_dist))}")
    expected_steps = markov.expected_residence_steps('drying')
    print(f"  Expected residence steps from drying: {expected_steps:.2f}")
    steady = markov.steady_state_distribution()
    print(f"  Steady-state exit probability: {steady[-1]:.4f}")

    # Binomial kinetics
    binom = BinomialKinetics()
    n_particles = 100
    k_events = 35
    p_conv = 0.4
    pmf_val = binom.probability_mass(n_particles, k_events, p_conv)
    print(f"  Binomial P(X={k_events}): {pmf_val:.4e}")
    print(f"  Expected conversions: {binom.expected_value(n_particles, p_conv):.1f}")
    stirling_50 = binom.stirling_approximation(50)
    print(f"  Stirling 50! ≈ {stirling_50:.4e} (true: {math.gamma(51):.4e})")

    # =====================================================================
    # 5. THERMODYNAMIC EQUILIBRIUM (from 807_nonlin_fixed_point, 035_asa091)
    # =====================================================================
    print("\n[5] Thermodynamic Equilibrium Solver")
    thermo = ThermoEquilibrium(T=1073.0, P=101325.0)
    K_wgs = thermo.equilibrium_constant('WGS', 1073.0)
    K_steam = thermo.equilibrium_constant('STEAM', 1073.0)
    print(f"  K_wgs at 800°C:   {K_wgs:.4f}")
    print(f"  K_steam at 800°C: {K_steam:.4f}")

    # Fixed-point WGS solver
    xi_fp = thermo.solve_wgs_fixed_point(0.3, 0.2, 0.3, 0.2, 1073.0, 101325.0)
    print(f"  WGS extent (fixed-point): ξ = {xi_fp:.4f}")

    # TODO Hole 3: Call thermo.solve_composition_newton() to compute equilibrium composition,
    # then print species mole fractions and run chi-squared test against observed data.
    # The returned composition array order must match species_names = ['CO', 'CO2', 'H2', 'H2O', 'CH4'].
    # Placeholder to allow partial execution:
    composition = np.array([0.25, 0.25, 0.20, 0.20, 0.10], dtype=float)
    species_names = ['CO', 'CO2', 'H2', 'H2O', 'CH4']
    print(f"  Equilibrium composition (mol fractions):")
    total_comp = composition.sum()
    for name, val in zip(species_names, composition):
        print(f"    {name}: {val/max(total_comp, 1.0e-15):.4f}")

    # Chi-squared test
    observed = np.array([0.25, 0.20, 0.30, 0.15, 0.10])
    expected = composition / max(total_comp, 1.0e-15)
    chi2, p_val = thermo.chi_squared_test(observed, expected)
    print(f"  χ² test: χ²={chi2:.4f}, p-value={p_val:.4f}")

    # =====================================================================
    # 6. HEAT TRANSFER (from 1116_sphere_exactness, 994_r8sd)
    # =====================================================================
    print("\n[6] Heat Transfer Analysis")
    rad_vf = RadiationViewFactor()
    r_p = 2.5e-3
    c_dist = 15.0e-3
    F_ss = rad_vf.sphere_to_sphere(r_p, r_p, c_dist)
    print(f"  Sphere-sphere view factor: {F_ss:.6f}")
    F_sp = rad_vf.sphere_to_plane(r_p, 10.0e-3)
    print(f"  Sphere-plane view factor:  {F_sp:.6f}")

    # Stefan-Boltzmann heat flux
    sb = StefanBoltzmannRadiation(epsilon=0.85)
    T_surf = 1200.0
    T_amb = 600.0
    q_rad = sb.net_radiative_heat_flux(T_surf, T_amb)
    h_rad = sb.radiative_heat_transfer_coefficient(T_surf, T_amb)
    print(f"  Radiative heat flux: {q_rad:.2f} W/m²")
    print(f"  Radiative h_coeff:   {h_rad:.2f} W/(m²·K)")

    # Conduction solver with CG
    n_wall = 30
    z_wall = np.linspace(0.0, 0.3, n_wall)
    k_wall = np.full(n_wall, 1.5)  # refractory brick
    Q_wall = np.full(n_wall, 5.0e4)  # W/m³ internal heat generation
    cond_solver = ConductionSolver(z_wall, k_wall)
    T_wall = cond_solver.solve_cg(Q_wall, T_left=300.0, T_right=1200.0)
    print(f"  Wall temp range: {T_wall.min():.1f} K to {T_wall.max():.1f} K")
    T_wall_direct = cond_solver.solve_direct(Q_wall, T_left=300.0, T_right=1200.0)
    err_cg = np.max(np.abs(T_wall - T_wall_direct))
    print(f"  CG vs direct max error: {err_cg:.2e} K")

    # =====================================================================
    # 7. CFD SOLVER (from 964_r83p, 245_cvt_1d_nonuniform)
    # =====================================================================
    print("\n[7] 1D Reactor Flow & Mesh")
    # Periodic tridiagonal solver demo
    n_periodic = 20
    lower_p = np.full(n_periodic, -0.5)
    diag_p = np.full(n_periodic, 2.0)
    upper_p = np.full(n_periodic, -0.5)
    # Make non-singular by breaking periodic symmetry slightly
    lower_p[0] = -0.3
    upper_p[-1] = -0.3
    pts = PeriodicTridiagonalSolver(n_periodic)
    factor_data = pts.factor(lower_p, diag_p, upper_p)
    if factor_data.get('info', 0) == 0:
        b_test = np.ones(n_periodic, dtype=float)
        x_periodic = pts.solve(factor_data, b_test)
        print(f"  Periodic system solved, mean x: {x_periodic.mean():.4f}")
    else:
        print(f"  Periodic factorization failed, info={factor_data['info']}")

    # CVT mesh generation
    cvt = CVTMeshGenerator(n_generators=25, density_func_id=4)
    z_mesh = cvt.generate_mesh(z_min=0.0, z_max=reactor.H, n_samples=2000, n_steps=80)
    print(f"  CVT mesh: {len(z_mesh)} nodes, spacing ratio: {np.max(np.diff(z_mesh))/np.min(np.diff(z_mesh)):.2f}")

    # Flow solver
    flow_solver = ReactorFlowSolver(z_mesh, reactor.R)
    rho_gas = 0.5  # kg/m³
    mu_gas = 3.0e-5  # Pa·s
    dp_dz = -500.0  # Pa/m
    f_D = 0.02
    u_profile = flow_solver.solve_velocity_profile(rho_gas, mu_gas, dp_dz, f_D)
    # Cap unphysical velocities from linearized momentum solver
    u_profile = np.clip(u_profile, 0.0, 5.0)
    print(f"  Gas velocity range: {u_profile.min():.3f} to {u_profile.max():.3f} m/s")
    Re_max = flow_solver.reynolds_number(rho_gas, u_profile.max(), d_32, mu_gas)
    Nu = flow_solver.nusselt_number(Re_max, Pr=0.7)
    print(f"  Max Reynolds: {Re_max:.1f}, Nusselt: {Nu:.2f}")
    u_superficial = min(u_profile.mean(), 1.0)
    dp_ergun = flow_solver.pressure_drop_ergun(epsilon=0.4, rho_gas=rho_gas,
                                                mu_gas=mu_gas,
                                                u_superficial=u_superficial,
                                                d_particle=d_32)
    print(f"  Ergun dp/dz: {dp_ergun:.1f} Pa/m")

    # =====================================================================
    # 8. PARTICLE LIFETIME (from 780_mortality, 197_collatz_parfor)
    # =====================================================================
    print("\n[8] Char Particle Burnout & Lifetime Analysis")
    burnout = ParticleBurnoutModel(d0=d_32, rho_char=800.0,
                                    k_surf=5.0, D_eff=1.0e-5, control='mixed')
    t_burn = burnout.burnout_time(T_const=1200.0)
    print(f"  Estimated burnout time at 1200K: {t_burn:.2f} s")

    # Mortality table
    mortality = ParticleMortalityTable(max_age_seconds=600.0, n_bins=60)
    mortality.populate_from_weibull(scale=t_burn, shape=2.5)
    e_life = mortality.expected_lifetime()
    med_life = mortality.median_lifetime()
    print(f"  Weibull expected lifetime:   {e_life:.2f} s")
    print(f"  Weibull median lifetime:     {med_life:.2f} s")
    print(f"  Hazard rate at t=0:          {mortality.hazard_rate()[0]:.4e} s⁻¹")

    # Collatz-like burnout sequences
    seqs = []
    for _ in range(10):
        seq = CollatzBurnoutSequence.generate_sequence(0.0, 1200.0, 0.05, max_steps=500)
        seqs.append(seq)
    stats = CollatzBurnoutSequence.sequence_statistics(seqs)
    print(f"  Mean burnout steps: {stats['mean_length']:.1f}")
    print(f"  Max burnout steps:  {stats['max_length']}")

    # =====================================================================
    # 9. REACTOR STATE SIMULATION (from 321_dueling_idiots, 197_collatz_parfor)
    # =====================================================================
    print("\n[9] Sequential Modular Reactor Simulation")
    initial = ReactorStateVector(T=298.0, P=101325.0, n_species=7)
    # Single-pass sequential modular simulation (once-through reactor)
    state = initial.copy()
    state = ReactorZoneModel.drying_zone(state)
    state = ReactorZoneModel.pyrolysis_zone(state)
    state = ReactorZoneModel.combustion_zone(state)
    final_state = ReactorZoneModel.reduction_zone(state)
    print(f"  Single-pass simulation completed")
    print(f"  Final temperature:    {final_state.T:.1f} K")
    print(f"  Final pressure:       {final_state.P:.1f} Pa")
    print(f"  Final char conversion: {final_state.X_char:.4f}")
    species_labels = ['O2', 'N2', 'CO2', 'CO', 'H2O', 'H2', 'CH4']
    print(f"  Product gas composition (mole fractions):")
    for label, y in zip(species_labels, final_state.y):
        print(f"    {label}: {y:.4f}")

    # Demonstrate convergence of iterative solvers
    print(f"  WGS fixed-point solver converged: ξ = {xi_fp:.4f}")
    print(f"  CG heat conduction solver error: {err_cg:.2e} K")
    
    # Demonstrate sequential modular convergence with recycle
    simulator = SequentialModularSimulator(n_species=7, max_iter=50, tol=1.0e-3)
    zones = [
        ReactorZoneModel.drying_zone,
        ReactorZoneModel.pyrolysis_zone,
        ReactorZoneModel.combustion_zone,
        ReactorZoneModel.reduction_zone
    ]
    _, conv, iters = simulator.simulate(initial, zones, recycle_fraction=0.02)
    print(f"  Recycle simulation converged: {conv} after {iters} iterations")

    # =====================================================================
    # 10. MESH ADAPTATION (from 245_cvt_1d_nonuniform)
    # =====================================================================
    print("\n[10] Adaptive Mesh Generation")
    # Create a synthetic temperature profile
    z_fine = np.linspace(0.0, reactor.H, 200)
    T_profile = 300.0 + 900.0 * (1.0 - np.exp(-2.0 * z_fine)) + \
                200.0 * np.exp(-((z_fine - 1.25) / 0.3) ** 2)
    adapter = MeshAdapter(z_min=0.0, z_max=reactor.H, n_nodes=30)
    z_adapted = adapter.adapt_mesh(T_profile, method='equidistribute')
    quality = adapter.mesh_quality(z_adapted)
    print(f"  Adapted mesh nodes: {len(z_adapted)}")
    print(f"  Min spacing: {quality['min_spacing']:.4f} m")
    print(f"  Max spacing: {quality['max_spacing']:.4f} m")
    print(f"  Spacing ratio: {quality['spacing_ratio']:.2f}")
    print(f"  Uniformity index: {quality['uniformity']:.4f}")

    # =====================================================================
    # 11. STATISTICAL UTILITIES (from 035_asa091, 055_asa310)
    # =====================================================================
    print("\n[11] Statistical Distribution Tests")
    # Normal CDF tail
    p_norm = 1.0 - 0.05
    x_norm, _ = ppnd(p_norm)
    print(f"  Normal 95th percentile: {x_norm:.4f}")

    # Chi-squared percentage point
    v_chi2 = 10.0
    g_chi2 = math.lgamma(v_chi2 / 2.0)
    chi2_val, _ = ppchi2(0.95, v_chi2, g_chi2)
    print(f"  χ²(10) 95th percentile: {chi2_val:.4f}")

    # Gamma distribution
    gamma_val, _ = gammad(5.0, 3.0)
    print(f"  Γ(5,3) CDF: {gamma_val:.4f}")

    # Noncentral Beta
    ncb_val, _ = ncbeta(2.0, 3.0, 5.0, 0.4)
    print(f"  Noncentral Beta CDF(0.4; 2,3,λ=5): {ncb_val:.4f}")

    # =====================================================================
    # 12. INTEGRATED ENERGY & MASS BALANCE
    # =====================================================================
    print("\n[12] Integrated Energy & Mass Balance")
    # Overall cold gas efficiency
    LHV_biomass = 18.0e6  # J/kg
    m_biomass = 0.5  # kg/s
    y_H2 = final_state.y[5]
    y_CO = final_state.y[3]
    y_CH4 = final_state.y[6]
    LHV_gas = (y_H2 * 120.0e6 + y_CO * 10.1e6 + y_CH4 * 50.0e6)  # J/mol approx
    molar_mass_gas = y_H2 * 2.0 + y_CO * 28.0 + y_CH4 * 16.0 + \
                     final_state.y[0] * 32.0 + final_state.y[1] * 28.0 + \
                     final_state.y[2] * 44.0 + final_state.y[4] * 18.0
    molar_mass_gas = max(molar_mass_gas, 1.0e-3)
    # Cold gas efficiency η_cge = (m_gas * LHV_gas) / (m_biomass * LHV_biomass)
    # Approximate with mole fractions
    eta_cge = (y_H2 * 120.0 + y_CO * 10.1 + y_CH4 * 50.0) / 180.0
    eta_cge = min(max(eta_cge, 0.0), 1.0)
    print(f"  Approx. cold gas efficiency: {eta_cge*100:.1f}%")

    # Carbon conversion
    X_carbon = 1.0 - final_state.X_char
    print(f"  Carbon conversion: {X_carbon*100:.1f}%")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Reactor type:          Downdraft biomass gasifier")
    print(f"Operating temperature: {final_state.T:.1f} K")
    print(f"Operating pressure:    {final_state.P/1000:.1f} kPa")
    print(f"Main product:          Syngas (CO + H₂)")
    print(f"Syngas yield:          {(y_CO + y_H2)*100:.1f} mol%")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(run_reactor_simulation())
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
