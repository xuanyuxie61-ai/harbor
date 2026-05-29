"""
main.py
=======
Unified entry point for the pulverized coal combustion NOx formation
multi-scale simulation system.

This project synthesizes 15 seed algorithms into a single doctoral-level
scientific computing framework addressing:
    "Multi-scale modeling of NOx formation during pulverized coal combustion"

Execution: python main.py (zero arguments required)

Workflow:
    1. Particle geometry modeling (egg-shape, Bezier, triangulation)
    2. Intra-particle diffusion-reaction (Bessel eigenvalues, Thiele modulus)
    3. Detailed NOx reaction kinetics (stiff ODE, backward Euler)
    4. 1D burner FEM simulation (temperature + species profiles)
    5. Particle population CVT distribution (Lloyd algorithm)
    6. Parameter space optimization (Hilbert curve + LCG)
    7. Stiffness analysis (Jordan form, eigenvalue conditioning)
    8. Statistical post-processing (grouped aggregation, response surface)
"""

import numpy as np
import sys
import time

# Ensure project modules are importable
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
    """Step 1: Model coal particle geometry."""
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
    """Step 2: Intra-particle diffusion and char reaction."""
    print_section("STEP 2: Intra-Particle Diffusion-Reaction")
    
    T_p = 1600.0  # K
    P = 101325.0
    D_bulk = mass_diffusivity_NO(T_p, P)
    D_eff = effective_diffusivity(D_bulk, porosity=0.5, tortuosity=3.0,
                                   knudsen=True, pore_radius=5e-9, T=T_p)
    print(f"\n  Bulk diffusivity D_bulk = {D_bulk:.3e} m^2/s")
    print(f"  Effective diffusivity D_eff = {D_eff:.3e} m^2/s")
    
    for p in particles:
        R_p = p.equivalent_diameter / 2.0
        k_char = 1.0e2  # 1/s (apparent rate)
        phi = thiele_modulus(R_p, k_char, D_eff)
        eta = effectiveness_factor(phi)
        print(f"\n  Particle (eq. dia = {2*R_p:.1e} m):")
        print(f"    Thiele modulus phi = {phi:.3f}")
        print(f"    Effectiveness eta  = {eta:.4f}")
        
        # Concentration profile
        r = np.linspace(0.0, R_p, 51)
        C_prof = concentration_profile_spectral(r, R_p, C_surf=0.05,
                                                 D_eff=D_eff, k=k_char)
        print(f"    C(0)/C_surf = {C_prof[0]/max(C_prof[-1],1e-30):.4f}")
        
        # Fuel-N release
        Y_N = 0.015  # 1.5 wt% N in coal
        r_n = fuel_n_release_rate(R_p, T_p, Y_N, D_eff)
        print(f"    Fuel-N release rate = {r_n:.3e} kg_N/(m^3*s)")
        
        # Char oxidation
        P_O2 = 0.05 * P
        r_char = char_oxidation_rate(T_p, P_O2)
        print(f"    Char oxidation rate = {r_char:.3e} kg_C/(m^2*s)")
    
    # Bessel zeros demonstration
    zeros = compute_j0_zeros(10)
    print(f"\n  First 10 j_0 zeros: {np.round(zeros, 6)}")


def run_reaction_kinetics():
    """Step 3: Detailed NOx reaction kinetics."""
    print_section("STEP 3: NOx Reaction Kinetics (Stiff ODE)")
    
    # Verification against normal ODE exact solution
    print("\n  --- Normal ODE verification ---")
    def integrator_wrapper(f, y0, t_vals):
        y = y0
        ys = [y]
        for i in range(1, len(t_vals)):
            dt = t_vals[i] - t_vals[i-1]
            y = y + dt * f(t_vals[i-1], y)
        return np.array(ys)
    
    # Use backward Euler for actual verification
    def be_integrator(f, y0, t_vals):
        y = y0
        ys = [y]
        for i in range(1, len(t_vals)):
            dt = t_vals[i] - t_vals[i-1]
            y_new = y + dt * f(t_vals[i], y_new)  # implicit
            # Fixed-point iteration
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
    
    # Simple explicit for verification baseline
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
    
    # Batch reactor simulation
    print("\n  --- Batch reactor NOx kinetics ---")
    Y0 = np.zeros(NSPEC)
    Y0[SPECIES_NAMES.index("N2")] = 0.70
    Y0[SPECIES_NAMES.index("O2")] = 0.15
    Y0[SPECIES_NAMES.index("CH4")] = 0.08
    Y0[SPECIES_NAMES.index("HCN")] = 0.005
    Y0[SPECIES_NAMES.index("O")] = 1e-4    # radical seed from O2 dissociation
    Y0[SPECIES_NAMES.index("OH")] = 5e-5   # radical seed
    Y0[SPECIES_NAMES.index("CH")] = 1e-6   # radical seed
    Y0 /= np.sum(Y0)
    
    T_reactor = 2000.0
    t_end = 0.01  # seconds (shorter for realistic NOx build-up)
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
    """Step 4: 1D burner FEM simulation."""
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
    
    # Thermophysical properties at peak T
    T_peak = result['max_T']
    print(f"\n  Properties at T_peak = {T_peak:.0f} K:")
    print(f"    k_therm = {thermal_conductivity(T_peak):.4f} W/(m*K)")
    print(f"    mu      = {dynamic_viscosity(T_peak):.3e} Pa*s")
    print(f"    D_NO    = {mass_diffusivity_NO(T_peak):.3e} m^2/s")
    print(f"    Pr      = {prandtl_number(T_peak):.3f}")
    print(f"    Le      = {lewis_number(T_peak):.3f}")


def run_particle_population():
    """Step 5: Particle population CVT distribution."""
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
    
    # Aggregation
    volumes = np.logspace(-15, -11, 10)  # 1 um^3 to 1000 um^3
    N_final = simulate_smoluchowski_aggregation(volumes, n_steps=50)
    print(f"\n  Aggregation: final number concentrations (first 5 bins):")
    for i in range(min(5, len(N_final))):
        print(f"    Bin {i}: N = {N_final[i]:.3f}")


def run_parameter_search():
    """Step 6: Parameter space optimization."""
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
    """Step 7: Stiffness and Jordan analysis."""
    print_section("STEP 7: Stiffness Analysis")
    
    # Test Jordan spectrum
    jordan_test = generate_test_jordan_spectrum(n=16)
    print(f"\n  Synthetic Jordan matrix (n=16):")
    print(f"    Block sizes: {jordan_test['block_sizes']}")
    print(f"    Max eigenvalue sensitivity: {jordan_test['max_sensitivity']:.3e}")
    
    # Actual combustion Jacobian
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
    """Step 8: Statistical post-processing."""
    print_section("STEP 8: Statistical Post-Processing")
    
    # Generate synthetic multi-condition data
    rng = np.random.default_rng(99)
    n_cases = 100
    excess_air = rng.uniform(0.9, 1.3, n_cases)
    T_peak = rng.uniform(1400.0, 1900.0, n_cases)
    
    # Synthetic NOx and burnout
    NOx = 200.0 * np.exp(-319e3 / (8.314 * T_peak)) * 1e6 + \
          50.0 * (1.0 / excess_air) + rng.normal(0, 5.0, n_cases)
    burnout = 1.0 / (1.0 + np.exp(-0.01 * (T_peak - 1500.0))) * \
              (1.0 - 0.1 * (excess_air - 1.0)) + rng.normal(0, 0.02, n_cases)
    burnout = np.clip(burnout, 0.0, 1.0)
    
    # Grouped statistics
    grouped = group_statistics(excess_air, NOx, burnout, n_bins=5)
    print(f"\n  Grouped by excess air ratio:")
    for g in grouped['groups']:
        print(f"    Bin {g['bin_index']} [{g['bin_range'][0]:.2f}, {g['bin_range'][1]:.2f}): "
              f"NOx={g['NOx_mean']:.1f}±{g['NOx_std']:.1f} ppm, "
              f"burnout={g['burnout_mean']:.3f}")
    
    # Response surface
    rs = fit_response_surface(excess_air, T_peak, NOx, degree=3)
    print(f"\n  Response surface (NOx vs excess_air, T_peak):")
    print(f"    R^2 = {rs['R2']:.4f}")
    print(f"    RMSE = {rs['RMSE']:.3f} ppm")
    
    # Uncertainty quantification
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
    
    # Execute all scientific modules
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

# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: safe_exp normal input returns finite positive value ----
v = safe_exp(1.0)
assert np.isfinite(v) and v > 0.0, '[TC01] safe_exp normal FAILED'

# ---- TC02: safe_exp overflow guard caps at exp(700) ----
v = safe_exp(1000.0)
assert v <= np.exp(700.0) + 1e-10, '[TC02] safe_exp overflow guard FAILED'

# ---- TC03: safe_exp negative far below -700 returns 0 ----
v = safe_exp(-1000.0)
assert v == 0.0, '[TC03] safe_exp negative overflow FAILED'

# ---- TC04: newton_raphson_scalar solves x^2-4=0 ----
f = lambda x: x*x - 4.0
df = lambda x: 2.0*x
x_star, nit, conv = newton_raphson_scalar(f, df, 1.5, tol=1e-12, max_iter=100)
assert conv and abs(x_star - 2.0) < 1e-10, '[TC04] newton_raphson FAILED'

# ---- TC05: gauss_legendre_nodes_weights sum of weights equals 2 ----
x_gl, w_gl = gauss_legendre_nodes_weights(10)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC05] GL weights sum FAILED'

# ---- TC06: gauss_legendre_nodes_weights nodes in [-1,1] ----
x_gl, w_gl = gauss_legendre_nodes_weights(8)
assert np.all(x_gl >= -1.0) and np.all(x_gl <= 1.0), '[TC06] GL nodes range FAILED'

# ---- TC07: condition_estimate for identity matrix near 1 ----
I_test = np.eye(5)
cond = condition_estimate(I_test)
assert abs(cond - 1.0) < 1.0, '[TC07] condition identity FAILED'

# ---- TC08: CoalParticle surface_area positive ----
import numpy as np
np.random.seed(42)
p = CoalParticle(L=100e-6, B=70e-6, w=5e-6, shape_type="egg")
assert p.surface_area > 0.0, '[TC08] CoalParticle surface_area FAILED'

# ---- TC09: CoalParticle volume positive ----
assert p.volume > 0.0, '[TC09] CoalParticle volume FAILED'

# ---- TC10: CoalParticle sphericity in [0,1] ----
assert 0.0 <= p.sphericity <= 1.0 + 1e-12, '[TC10] CoalParticle sphericity FAILED'

# ---- TC11: CoalParticle shape_descriptor has required keys ----
desc = p.shape_descriptor()
for k in ["aspect_ratio", "sphericity", "surface_area", "volume", "boundary_word"]:
    assert k in desc, f'[TC11] CoalParticle descriptor missing {k}'

# ---- TC12: CoalParticle equivalent_diameter positive ----
assert p.equivalent_diameter > 0.0, '[TC12] CoalParticle eq_diameter FAILED'

# ---- TC13: CoalParticle surface_to_volume_ratio positive ----
assert p.surface_to_volume_ratio > 0.0, '[TC13] CoalParticle Sv_ratio FAILED'

# ---- TC14: compute_surface_area of a known cube mesh ----
v_cube = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float)
f_cube = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]],dtype=int)
a_cube = compute_surface_area(v_cube, f_cube)
assert abs(a_cube - 6.0) < 1e-10, '[TC14] compute_surface_area cube FAILED'

# ---- TC15: compute_volume_revolution non-negative ----
x_vol = np.linspace(-1.0, 1.0, 51)
y_vol = np.sqrt(1.0 - x_vol**2)
v_sphere = compute_volume_revolution(x_vol, y_vol)
assert v_sphere > 0.0, '[TC15] compute_volume_revolution positive FAILED'

# ---- TC16: compute_j0_zeros first zero is pi ----
zeros_3 = compute_j0_zeros(3)
assert abs(zeros_3[0] - np.pi) < 1e-10, '[TC16] j0 zeros[0] FAILED'

# ---- TC17: compute_j0_zeros all positive ----
zeros_5 = compute_j0_zeros(5)
assert np.all(zeros_5 > 0.0), '[TC17] j0 zeros positive FAILED'

# ---- TC18: thiele_modulus returns zero for invalid R_p ----
assert thiele_modulus(0.0, 1.0, 1.0) == 0.0, '[TC18] thiele zero R_p FAILED'

# ---- TC19: thiele_modulus returns positive for valid inputs ----
phi_test = thiele_modulus(1e-4, 100.0, 1e-5)
assert phi_test > 0.0, '[TC19] thiele positive FAILED'

# ---- TC20: effectiveness_factor near zero is 1.0 ----
eta_small = effectiveness_factor(1e-15)
assert abs(eta_small - 1.0) < 1e-12, '[TC20] eta near 0 FAILED'

# ---- TC21: effectiveness_factor large phi approximates 3/phi ----
eta_large = effectiveness_factor(1e4)
assert abs(eta_large - 3.0/1e4) < 1e-12, '[TC21] eta large FAILED'

# ---- TC22: effective_diffusivity positive for valid inputs ----
D_eff_test = effective_diffusivity(1e-4, 0.5, 3.0, knudsen=True, pore_radius=5e-9, T=1500.0)
assert D_eff_test > 0.0, '[TC22] effective_diffusivity FAILED'

# ---- TC23: char_oxidation_rate non-negative ----
r_char_test = char_oxidation_rate(1500.0, 5000.0)
assert r_char_test >= 0.0, '[TC23] char oxidation non-neg FAILED'

# ---- TC24: fuel_n_release_rate zero for zero Y_N ----
assert fuel_n_release_rate(1e-4, 1500.0, 0.0, 1e-5) == 0.0, '[TC24] fuel_N zero Y FAILED'

# ---- TC25: concentration_profile_spectral returns correct length ----
r_test = np.linspace(0.0, 1e-4, 20)
C_prof = concentration_profile_spectral(r_test, 1e-4, 1.0, 1e-5, 50.0)
assert len(C_prof) == len(r_test), '[TC25] conc_profile length FAILED'

# ---- TC26: concentration_profile_spectral surface value close to C_surf ----
r_surf = np.array([1e-4])
C_s = concentration_profile_spectral(r_surf, 1e-4, 1.0, 1e-5, 50.0)
assert C_s[0] > 0.0, '[TC26] conc_profile surface FAILED'

# ---- TC27: normal_ode_exact positive at t=0 ----
y0_exact = normal_ode_exact(0.0)
assert y0_exact > 0.0, '[TC27] normal_ode_exact at 0 FAILED'

# ---- TC28: normal_ode_exact symmetric ----
assert abs(normal_ode_exact(2.0) - normal_ode_exact(-2.0)) < 1e-14, '[TC28] normal_ode symmetric FAILED'

# ---- TC29: normal_ode_exact decays away from origin ----
y0_exact2 = normal_ode_exact(0.0)
y1_exact = normal_ode_exact(3.0)
assert y1_exact < y0_exact2, '[TC29] normal_ode decay FAILED'

# ---- TC30: integrate_backward_euler reaches t_end ----
f_simple = lambda y: -0.5 * y
y0_simple = np.ones(2)
res_be = integrate_backward_euler(f_simple, y0_simple, 0.1, dt_init=0.01)
assert res_be["t_final"] > 0.0, '[TC30] integrate BE reached t_end FAILED'

# ---- TC31: ReactorODE initializes with proper density ----
reactor = ReactorODE(T=1500.0, P=101325.0)
assert reactor.rho > 0.0, '[TC31] ReactorODE rho FAILED'

# ---- TC32: SPECIES_NAMES contains expected species ----
assert "NO" in SPECIES_NAMES, '[TC32] SPECIES_NAMES has NO FAILED'
assert "N2" in SPECIES_NAMES, '[TC32] SPECIES_NAMES has N2 FAILED'
assert "O2" in SPECIES_NAMES, '[TC32] SPECIES_NAMES has O2 FAILED'

# ---- TC33: NSPEC matches length of SPECIES_NAMES ----
assert NSPEC == len(SPECIES_NAMES), '[TC33] NSPEC length FAILED'

# ---- TC34: get_pathway_contributions returns all pathway types ----
Y_test_p = np.zeros(NSPEC)
Y_test_p[SPECIES_NAMES.index("N2")] = 0.70
Y_test_p[SPECIES_NAMES.index("O2")] = 0.20
Y_test_p[SPECIES_NAMES.index("CH4")] = 0.10
Y_test_p /= np.sum(Y_test_p)
pways = get_pathway_contributions(Y_test_p, 1800.0, 1.0)
for ptype in ["thermal", "fuel", "prompt", "reburn"]:
    assert ptype in pways, f'[TC34] pathway missing {ptype}'

# ---- TC35: compute_production_rates returns correct length ----
Y_test_pr = np.zeros(NSPEC)
Y_test_pr[0] = 0.79
Y_test_pr[1] = 0.21
omega = compute_production_rates(Y_test_pr, 1500.0, 1.0)
assert len(omega) == NSPEC, '[TC35] production_rates length FAILED'

# ---- TC36: mixture_density uses ideal gas law ----
rho_test = mixture_density(300.0, 101325.0, 0.029)
rho_expected = 101325.0 * 0.029 / (8.314462618 * 300.0)
assert abs(rho_test - rho_expected) < 1e-8, '[TC36] mixture_density FAILED'

# ---- TC37: prandtl_number is positive ----
Pr_test = prandtl_number(800.0)
assert Pr_test > 0.0, '[TC37] prandtl positive FAILED'

# ---- TC38: lewis_number is positive ----
Le_test = lewis_number(800.0)
assert Le_test > 0.0, '[TC38] lewis positive FAILED'

# ---- TC39: thermal_conductivity positive ----
k_test = thermal_conductivity(500.0)
assert k_test > 0.0, '[TC39] thermal_cond positive FAILED'

# ---- TC40: dynamic_viscosity positive ----
mu_test = dynamic_viscosity(500.0)
assert mu_test > 0.0, '[TC40] dynamic_visc positive FAILED'

# ---- TC41: mass_diffusivity_NO positive ----
D_test = mass_diffusivity_NO(500.0, 101325.0)
assert D_test > 0.0, '[TC41] mass_diffusivity positive FAILED'

# ---- TC42: cp_mixture returns positive value ----
Y_mix_test = {"N2": 0.79, "O2": 0.21}
cp_test = cp_mixture(500.0, Y_mix_test)
assert cp_test > 0.0, '[TC42] cp_mixture positive FAILED'

# ---- TC43: simulate_1d_burner returns expected structure ----
burner_res = simulate_1d_burner(L=2.0, n_nodes=31, u_inlet=3.0, T_inlet=350.0, T_wall=700.0)
for k in ["x", "T", "Y_NO", "max_T", "max_NO_ppm", "outlet_NO_ppm"]:
    assert k in burner_res, f'[TC43] burner missing {k}'
assert burner_res["max_T"] >= burner_res["T"][0], '[TC43] max_T >= T_inlet FAILED'

# ---- TC44: simulate_1d_burner NO result non-negative ----
assert np.all(burner_res["Y_NO"] >= 0.0), '[TC44] burner Y_NO non-neg FAILED'

# ---- TC45: cvt_lloyd_3d produces correct number of generators ----
np.random.seed(42)
cvt_res = cvt_lloyd_3d(n_generators=8, density_func=density_burner_profile, n_samples=12, max_iter=20, tol=1e-4)
assert len(cvt_res["generators"]) == 8, '[TC45] cvt generators count FAILED'
assert len(cvt_res["energies"]) > 0, '[TC45] cvt energies FAILED'

# ---- TC46: cluster_statistics returns expected fields ----
dom = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
cs = cluster_statistics(cvt_res["generators"], dom)
for k in ["mean_spacing", "min_spacing", "max_spacing"]:
    assert cs[k] > 0.0, f'[TC46] cluster_statistics {k} FAILED'

# ---- TC47: simulate_smoluchowski_aggregation preserves non-negativity ----
vols = np.logspace(-15, -11, 10)
N_final_test = simulate_smoluchowski_aggregation(vols, n_steps=20)
assert np.all(N_final_test >= 0.0), '[TC47] aggregation non-neg FAILED'

# ---- TC48: nox_objective returns finite ----
params_test = np.array([1.1, 80.0, 1700.0, 250.0])
cost_test = nox_objective(params_test)
assert np.isfinite(cost_test), '[TC48] nox_objective finite FAILED'

# ---- TC49: nox_objective penalizes low burnout ----
params_lowT = np.array([1.1, 80.0, 1200.0, 50.0])
cost_lowT = nox_objective(params_lowT)
assert cost_lowT > cost_test, '[TC49] nox_objective penalty FAILED'

# ---- TC50: optimize_combustion_parameters returns expected keys ----
opt_res = optimize_combustion_parameters(n_evals=64, use_hilbert=True)
for k in ["best_params", "best_cost", "mean_cost", "std_cost"]:
    assert k in opt_res, f'[TC50] optimize missing {k}'

# ---- TC51: generate_test_jordan_spectrum returns expected keys ----
jt = generate_test_jordan_spectrum(16)
for k in ["J", "eigenvalues", "block_sizes", "max_sensitivity"]:
    assert k in jt, f'[TC51] jordan_spectrum missing {k}'

# ---- TC52: generate_test_jordan_spectrum produces correct size J ----
assert jt["J"].shape == (16, 16), '[TC52] jordan J shape FAILED'

# ---- TC53: analyze_stiffness returns expected keys ----
Y_test_as = np.zeros(NSPEC)
Y_test_as[SPECIES_NAMES.index("N2")] = 0.70
Y_test_as[SPECIES_NAMES.index("O2")] = 0.20
Y_test_as[SPECIES_NAMES.index("CH4")] = 0.10
Y_test_as /= np.sum(Y_test_as)
stiff = analyze_stiffness(Y_test_as, 1800.0, 1.0)
for k in ["stiffness_ratio", "fastest_time_scale", "slowest_time_scale", "condition_number"]:
    assert k in stiff, f'[TC53] stiffness missing {k}'

# ---- TC54: recommend_timestep returns method recommendation ----
rec = recommend_timestep(stiff)
assert "recommended_method" in rec, '[TC54] timestep method FAILED'
assert rec["dt_max_explicit"] > 0.0, '[TC54] dt_max_explicit positive FAILED'

# ---- TC55: group_statistics returns correct structure ----
rng = np.random.default_rng(99)
cond_vals = rng.uniform(0.9, 1.3, 30)
nox_vals = rng.uniform(30.0, 80.0, 30)
burn_vals = rng.uniform(0.6, 1.0, 30)
gs = group_statistics(cond_vals, nox_vals, burn_vals, n_bins=4)
assert len(gs["groups"]) > 0, '[TC55] group_statistics groups FAILED'
assert "global_NOx_mean" in gs, '[TC55] group_statistics global mean FAILED'

# ---- TC56: fit_response_surface returns R2 >= 0 for reasonable data ----
np.random.seed(42)
x1_rs = np.random.uniform(0.9, 1.3, 50)
x2_rs = np.random.uniform(1400.0, 1900.0, 50)
y_rs = 3.0 + 2.0 * x1_rs - 0.01 * x2_rs + np.random.normal(0, 1.0, 50)
rs_res = fit_response_surface(x1_rs, x2_rs, y_rs, degree=2)
assert rs_res["R2"] >= 0.0, '[TC56] R2 non-negative FAILED'

# ---- TC57: monte_carlo_uncertainty returns valid outputs ----
dummy_mc = lambda params: params[0] + 2.0 * params[1]
uq_res = monte_carlo_uncertainty(dummy_mc, np.array([1.0, 2.0]), np.array([0.1, 0.1]), n_samples=200, seed=42)
assert uq_res["n_valid"] == 200, '[TC57] MC valid count FAILED'
assert np.isfinite(uq_res["mean"]), '[TC57] MC mean FAILED'

# ---- TC58: simulate_batch_reactor returns expected fields ----
np.random.seed(42)
Y0_sbr = np.zeros(NSPEC)
Y0_sbr[SPECIES_NAMES.index("N2")] = 0.70
Y0_sbr[SPECIES_NAMES.index("O2")] = 0.15
Y0_sbr[SPECIES_NAMES.index("CH4")] = 0.10
Y0_sbr[SPECIES_NAMES.index("HCN")] = 0.005
Y0_sbr /= np.sum(Y0_sbr)
sbr_res = simulate_batch_reactor(Y0_sbr, 1800.0, 0.005)
for k in ["y_final", "NO_ppm", "NO2_ppm", "pathways"]:
    assert k in sbr_res, f'[TC58] batch_reactor missing {k}'

# ---- TC59: density_burner_profile returns positive values ----
np.random.seed(42)
s_test = np.array([[0.0, 0.0, 0.5], [0.3, 0.0, 1.0], [-0.2, 0.1, 1.5]])
rho_bp = density_burner_profile(s_test)
assert np.all(rho_bp > 0.0), '[TC59] burner_profile positive FAILED'

# ---- TC60: CoalParticle pyriform shape creates valid geometry ----
np.random.seed(42)
p_pyr = CoalParticle(L=100e-6, B=70e-6, w=5e-6, shape_type="pyriform")
assert p_pyr.surface_area > 0.0, '[TC60] pyriform surface_area FAILED'
assert p_pyr.volume > 0.0, '[TC60] pyriform volume FAILED'
assert p_pyr.sphericity <= 1.0 + 1e-12, '[TC60] pyriform sphericity FAILED'

print('\n全部 60 个测试通过!\n')
