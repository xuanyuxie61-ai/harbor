
import numpy as np
import sys
import time


sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')

from particle_geometry import CoalParticle, compute_surface_area, compute_volume_revolution
from particle_diffusion import (
    compute_j0_zeros, thiele_modulus, effectiveness_factor,
    effective_diffusivity, fuel_n_release_rate, char_oxidation_rate,
    concentration_profile_spectral
)
from reaction_mechanism import (
    SPECIES_NAMES, NSPEC, get_pathway_contributions,
    compute_production_rates
)
from reaction_kinetics import (
    normal_ode_exact, verify_integrator, integrate_backward_euler,
    ReactorODE, simulate_batch_reactor
)
from reactor_fem1d import simulate_1d_burner
from thermophysical_props import (
    cp_mixture, thermal_conductivity, dynamic_viscosity,
    mass_diffusivity_NO, mixture_density, prandtl_number, lewis_number
)
from particle_population import (
    cvt_lloyd_3d, density_burner_profile, cluster_statistics,
    simulate_smoluchowski_aggregation
)
from parameter_search import optimize_combustion_parameters, nox_objective
from stiffness_analysis import (
    analyze_stiffness, generate_test_jordan_spectrum,
    recommend_timestep
)
from data_statistics import (
    group_statistics, fit_response_surface, monte_carlo_uncertainty
)
from utils import (
    safe_exp, gauss_legendre_nodes_weights, newton_raphson_scalar,
    condition_estimate
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_particle_geometry():
    print_section("STEP 1: Particle Geometry Modeling")
    
    particles = []
    for shape in ["egg", "pyriform"]:
        p = CoalParticle(L=120e-6, B=85e-6, w=8e-6, shape_type=shape)
        desc = p.shape_descriptor()
        particles.append(p)
        print(f"\n  Shape: {shape}")
        print(f"    Aspect ratio L/B = {desc['aspect_ratio']:.3f}")
        print(f"    Sphericity Psi   = {desc['sphericity']:.4f}")
        print(f"    Surface area     = {desc['surface_area']:.3e} m^2")
        print(f"    Volume           = {desc['volume']:.3e} m^3")
        print(f"    S/V ratio        = {desc['Sv_ratio']:.1f} m^{-1}")
        print(f"    Boundary word    = {desc['boundary_word'][:40]}...")
    
    return particles


def run_particle_diffusion(particles):
    print_section("STEP 2: Intra-Particle Diffusion-Reaction")
    
    T_p = 1600.0
    P = 101325.0
    D_bulk = mass_diffusivity_NO(T_p, P)
    D_eff = effective_diffusivity(D_bulk, porosity=0.5, tortuosity=3.0,
                                   knudsen=True, pore_radius=5e-9, T=T_p)
    print(f"\n  Bulk diffusivity D_bulk = {D_bulk:.3e} m^2/s")
    print(f"  Effective diffusivity D_eff = {D_eff:.3e} m^2/s")
    
    for p in particles:
        R_p = p.equivalent_diameter / 2.0
        k_char = 1.0e2
        phi = thiele_modulus(R_p, k_char, D_eff)
        eta = effectiveness_factor(phi)
        print(f"\n  Particle (eq. dia = {2*R_p:.1e} m):")
        print(f"    Thiele modulus phi = {phi:.3f}")
        print(f"    Effectiveness eta  = {eta:.4f}")
        

        r = np.linspace(0.0, R_p, 51)
        C_prof = concentration_profile_spectral(r, R_p, C_surf=0.05,
                                                 D_eff=D_eff, k=k_char)
        print(f"    C(0)/C_surf = {C_prof[0]/max(C_prof[-1],1e-30):.4f}")
        

        Y_N = 0.015
        r_n = fuel_n_release_rate(R_p, T_p, Y_N, D_eff)
        print(f"    Fuel-N release rate = {r_n:.3e} kg_N/(m^3*s)")
        

        P_O2 = 0.05 * P
        r_char = char_oxidation_rate(T_p, P_O2)
        print(f"    Char oxidation rate = {r_char:.3e} kg_C/(m^2*s)")
    

    zeros = compute_j0_zeros(10)
    print(f"\n  First 10 j_0 zeros: {np.round(zeros, 6)}")


def run_reaction_kinetics():
    print_section("STEP 3: NOx Reaction Kinetics (Stiff ODE)")
    

    print("\n  --- Normal ODE verification ---")
    def integrator_wrapper(f, y0, t_vals):
        y = y0
        ys = [y]
        for i in range(1, len(t_vals)):
            dt = t_vals[i] - t_vals[i-1]
            y = y + dt * f(t_vals[i-1], y)
        return np.array(ys)
    

    def be_integrator(f, y0, t_vals):
        y = y0
        ys = [y]
        for i in range(1, len(t_vals)):
            dt = t_vals[i] - t_vals[i-1]
            y_new = y + dt * f(t_vals[i], y_new)

            for _ in range(20):
                y_new_old = y_new
                y_new = y + dt * f(t_vals[i], y_new)
                if abs(y_new - y_new_old) < 1e-12:
                    break
            y = y_new
            ys.append(y)
        return np.array(ys)
    
    t0, tf = -5.0, 5.0
    n_steps = 200
    t_vals = np.linspace(t0, tf, n_steps + 1)
    y0 = normal_ode_exact(t0)
    

    y_exp = [y0]
    y = y0
    for i in range(1, len(t_vals)):
        dt = t_vals[i] - t_vals[i-1]
        y = y + dt * (-t_vals[i-1] * y)
        y_exp.append(y)
    y_exp = np.array(y_exp)
    y_exact = np.array([normal_ode_exact(t) for t in t_vals])
    err = np.max(np.abs(y_exp - y_exact))
    print(f"  Explicit Euler max error = {err:.3e}")
    

    print("\n  --- Batch reactor NOx kinetics ---")
    Y0 = np.zeros(NSPEC)
    Y0[SPECIES_NAMES.index("N2")] = 0.70
    Y0[SPECIES_NAMES.index("O2")] = 0.15
    Y0[SPECIES_NAMES.index("CH4")] = 0.08
    Y0[SPECIES_NAMES.index("HCN")] = 0.005
    Y0[SPECIES_NAMES.index("O")] = 1e-4
    Y0[SPECIES_NAMES.index("OH")] = 5e-5
    Y0[SPECIES_NAMES.index("CH")] = 1e-6
    Y0 /= np.sum(Y0)
    
    T_reactor = 2000.0
    t_end = 0.01
    result = simulate_batch_reactor(Y0, T_reactor, t_end)
    
    print(f"    Integration steps: {result['n_steps']}")
    print(f"    Rejected steps: {result['n_rejected']}")
    print(f"    Final NO  = {result['NO_ppm']:.3f} ppm")
    print(f"    Final NO2 = {result['NO2_ppm']:.3f} ppm")
    print(f"    Final N2O = {result['N2O_ppm']:.3f} ppm")
    
    pathways = result['pathways']
    print(f"    Pathway contributions [kg/(m^3*s)]:")
    for k, v in pathways.items():
        print(f"      {k:10s}: {v:+.3e}")


def run_reactor_fem1d():
    print_section("STEP 4: 1D Burner FEM Simulation")
    
    result = simulate_1d_burner(
        L=5.0, n_nodes=81,
        u_inlet=2.0, T_inlet=400.0,
        T_wall=900.0, P=101325.0
    )
    
    print(f"\n  Burner length: 5.0 m")
    print(f"  Peak temperature: {result['max_T']:.1f} K")
    print(f"  Max NO concentration: {result['max_NO_ppm']:.3f} ppm")
    print(f"  Outlet NO concentration: {result['outlet_NO_ppm']:.3f} ppm")
    

    T_peak = result['max_T']
    print(f"\n  Properties at T_peak = {T_peak:.0f} K:")
    print(f"    k_therm = {thermal_conductivity(T_peak):.4f} W/(m*K)")
    print(f"    mu      = {dynamic_viscosity(T_peak):.3e} Pa*s")
    print(f"    D_NO    = {mass_diffusivity_NO(T_peak):.3e} m^2/s")
    print(f"    Pr      = {prandtl_number(T_peak):.3f}")
    print(f"    Le      = {lewis_number(T_peak):.3f}")


def run_particle_population():
    print_section("STEP 5: Particle Population CVT Distribution")
    
    cvt_result = cvt_lloyd_3d(
        n_generators=32,
        density_func=density_burner_profile,
        n_samples=24,
        max_iter=30,
        domain=((-0.5, 0.5), (-0.5, 0.5), (0.0, 2.0))
    )
    
    print(f"\n  CVT converged in {cvt_result['n_iter']} iterations")
    print(f"  Final energy: {cvt_result['energies'][-1]:.3e}")
    print(f"  Final displacement: {cvt_result['displacements'][-1]:.3e}")
    
    stats = cluster_statistics(cvt_result['generators'],
                                domain=((-0.5, 0.5), (-0.5, 0.5), (0.0, 2.0)))
    print(f"\n  Cluster statistics:")
    print(f"    Mean spacing: {stats['mean_spacing']:.3f} m")
    print(f"    Min spacing:  {stats['min_spacing']:.3f} m")
    print(f"    Max spacing:  {stats['max_spacing']:.3f} m")
    

    volumes = np.logspace(-15, -11, 10)
    N_final = simulate_smoluchowski_aggregation(volumes, n_steps=50)
    print(f"\n  Aggregation: final number concentrations (first 5 bins):")
    for i in range(min(5, len(N_final))):
        print(f"    Bin {i}: N = {N_final[i]:.3f}")


def run_parameter_search():
    print_section("STEP 6: Parameter Space Optimization")
    
    opt_result = optimize_combustion_parameters(n_evals=128, use_hilbert=True)
    
    bp = opt_result['best_params']
    print(f"\n  Best operating conditions (Hilbert sampling):")
    print(f"    Excess air ratio    = {bp[0]:.3f}")
    print(f"    Particle diameter   = {bp[1]:.1f} um")
    print(f"    Peak temperature    = {bp[2]:.0f} K")
    print(f"    Residence time      = {bp[3]:.0f} ms")
    print(f"    Objective cost      = {opt_result['best_cost']:.3e}")
    print(f"  Mean cost = {opt_result['mean_cost']:.3e}, "
          f"Std = {opt_result['std_cost']:.3e}")


def run_stiffness_analysis():
    print_section("STEP 7: Stiffness Analysis")
    

    jordan_test = generate_test_jordan_spectrum(n=16)
    print(f"\n  Synthetic Jordan matrix (n=16):")
    print(f"    Block sizes: {jordan_test['block_sizes']}")
    print(f"    Max eigenvalue sensitivity: {jordan_test['max_sensitivity']:.3e}")
    

    Y_test = np.zeros(NSPEC)
    Y_test[SPECIES_NAMES.index("N2")] = 0.70
    Y_test[SPECIES_NAMES.index("O2")] = 0.15
    Y_test[SPECIES_NAMES.index("CH4")] = 0.10
    Y_test[SPECIES_NAMES.index("HCN")] = 0.005
    Y_test /= np.sum(Y_test)
    
    T_test = 1800.0
    rho_test = mixture_density(T_test, 101325.0)
    stiff = analyze_stiffness(Y_test, T_test, rho_test)
    
    print(f"\n  Combustion Jacobian at T={T_test:.0f} K:")
    print(f"    Stiffness ratio     = {stiff['stiffness_ratio']:.3e}")
    print(f"    Fastest time scale  = {stiff['fastest_time_scale']:.3e} s")
    print(f"    Slowest time scale  = {stiff['slowest_time_scale']:.3e} s")
    print(f"    Condition number    = {stiff['condition_number']:.3e}")
    print(f"    Negative eigenvals  = {stiff['n_negative_eigenvalues']}")
    print(f"    Spectral abscissa   = {stiff['spectral_abscissa']:.3e}")
    
    rec = recommend_timestep(stiff)
    print(f"\n  Recommendation:")
    print(f"    Max explicit dt = {rec['dt_max_explicit']:.3e} s")
    print(f"    Method: {rec['recommended_method']}")


def run_data_statistics():
    print_section("STEP 8: Statistical Post-Processing")
    

    rng = np.random.default_rng(99)
    n_cases = 100
    excess_air = rng.uniform(0.9, 1.3, n_cases)
    T_peak = rng.uniform(1400.0, 1900.0, n_cases)
    

    NOx = 200.0 * np.exp(-319e3 / (8.314 * T_peak)) * 1e6 + \
          50.0 * (1.0 / excess_air) + rng.normal(0, 5.0, n_cases)
    burnout = 1.0 / (1.0 + np.exp(-0.01 * (T_peak - 1500.0))) * \
              (1.0 - 0.1 * (excess_air - 1.0)) + rng.normal(0, 0.02, n_cases)
    burnout = np.clip(burnout, 0.0, 1.0)
    

    grouped = group_statistics(excess_air, NOx, burnout, n_bins=5)
    print(f"\n  Grouped by excess air ratio:")
    for g in grouped['groups']:
        print(f"    Bin {g['bin_index']} [{g['bin_range'][0]:.2f}, {g['bin_range'][1]:.2f}): "
              f"NOx={g['NOx_mean']:.1f}±{g['NOx_std']:.1f} ppm, "
              f"burnout={g['burnout_mean']:.3f}")
    

    rs = fit_response_surface(excess_air, T_peak, NOx, degree=3)
    print(f"\n  Response surface (NOx vs excess_air, T_peak):")
    print(f"    R^2 = {rs['R2']:.4f}")
    print(f"    RMSE = {rs['RMSE']:.3f} ppm")
    

    def dummy_model(params):
        ea, tp = params[0], params[1]
        return 200.0 * np.exp(-319e3 / (8.314 * tp)) * 1e6 + 50.0 * (1.0 / ea)
    
    uq = monte_carlo_uncertainty(
        dummy_model,
        param_means=np.array([1.1, 1650.0]),
        param_stds=np.array([0.05, 50.0]),
        n_samples=500
    )
    print(f"\n  Uncertainty quantification:")
    print(f"    Mean predicted NOx = {uq['mean']:.1f} ppm")
    print(f"    Std deviation = {uq['std']:.1f} ppm")
    print(f"    95% CI = ({uq['ci_95'][0]:.1f}, {uq['ci_95'][1]:.1f}) ppm")


def main():
    print("\n" + "#" * 70)
    print("#  Pulverized Coal Combustion NOx Formation")
    print("#  Multi-Scale Coupled Simulation System")
    print("#  Synthesis Project 158 | Python | Doctoral-level")
    print("#" * 70)
    
    t_start = time.time()
    

    particles = run_particle_geometry()
    run_particle_diffusion(particles)
    run_reaction_kinetics()
    run_reactor_fem1d()
    run_particle_population()
    run_parameter_search()
    run_stiffness_analysis()
    run_data_statistics()
    
    t_elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"#  Simulation completed in {t_elapsed:.3f} seconds")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
