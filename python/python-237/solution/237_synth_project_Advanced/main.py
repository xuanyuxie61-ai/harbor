"""
main.py - Unified entry point for the Lattice QCD project.

Project: Lattice QCD - Quark Propagator and Gauge Field Sampling
         High-Order Finite Differences and Stability Analysis
         (Small-Scale Reproducible Experiment)

This script performs a complete small-scale lattice QCD experiment on
a 4^4 Euclidean lattice with SU(2) gauge group.

All results are printed as text. No visualization.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import LatticeParams, GAMMA, su2_trace
from lattice_geometry import (LatticeGeometry, SiteIndex, compute_site_quality_metric,
                                global_plaquette_quality)
from gauge_field import (GaugeField, covariant_derivative, covariant_laplacian,
                           symmetric_derivative)
from wilson_dirac import (WilsonDiracOperator, SpinorField, make_point_source,
                            estimate_condition_number)
from quark_propagator import solve_propagator
from gauge_sampler import MetropolisSampler, SpinalTrajectorySampler, BioInspiredSampler
from spectral_analysis import (arnoldi_eigenvalues, spectral_density_histogram,
                                 banks_casher_estimate, newton_interp_spectral_function)
from gauge_orbit import (gauge_field_frobenius_distance, find_closest_gauge_transform_brute,
                           landau_gauge_fix, permutation_distance_between_configs)
from plaquette_topology import (plaquette_quality_field, topological_charge_clover,
                                   fractal_dimension_topological, topological_susceptibility)
from canalization_topology import (enumerate_topological_sectors,
                                     expected_topological_distribution,
                                     B_star, count_canalizing_functions,
                                     canalization_entropy)
from interpolation_utils import (multilinear_interpolate_scalar,
                                    gauge_covariant_interpolate,
                                    newton_polynomial_interpolation,
                                    continuum_extrapolation)
from codeword_gauge_fix import (gauge_fix_via_codeword, build_codeword_matrix,
                                   overlap_objective, birregular_code)
from coupled_eigenvalue import (CoupledEigenvalueEstimator, TemporalSpectralPredictor,
                                   coupled_spectral_analysis)


def print_header(title: str):
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def run_experiment():
    print_header("Lattice QCD: Quark Propagator & Gauge Field Sampling")
    print("High-Order Finite Differences and Stability Analysis")
    print("(Small-Scale Reproducible Experiment)")

    # ================================================================
    # Step 1: Lattice initialization
    # ================================================================
    print_header("Step 1: Lattice Initialization")
    params = LatticeParams(Ns=4, Nt=4, a=0.1, beta=2.5, m0=0.1)
    print(f"  Lattice: {params.Ns}^3 x {params.Nt} = {params.V} sites")
    print(f"  Lattice spacing: a = {params.a} fm")
    print(f"  Gauge coupling: beta = {params.beta}, g^2 = {params.g_sq:.4f}")
    print(f"  Bare quark mass: m0 = {params.m0}")
    print(f"  Hopping parameter: kappa = {params.kappa:.6f}")
    print(f"  Critical kappa: kappa_c = {params.critical_kappa():.6f}")
    print(f"  Physical extent: L = {params.Ns * params.a:.2f} fm, T = {params.Nt * params.a:.2f} fm")

    geo = LatticeGeometry(params)
    print(f"  Total plaquettes: {geo.n_plaquettes()}")

    rng = np.random.default_rng(237)

    gf = GaugeField(params, rng)
    gf.initialize_cold()
    S0 = gf.action_density()
    print(f"  Cold start action: S = {S0:.4f}")

    # ================================================================
    # Step 2: Gauge field thermalization
    # ================================================================
    print_header("Step 2: Gauge Field Thermalization")

    print_subheader("2a: Metropolis-Hastings Sampling")
    gf_therm = gf.copy()
    metropolis = MetropolisSampler(gf_therm, proposal_eps=0.5, seed=237)
    n_sweeps_therm = 5
    result = metropolis.thermalize(n_sweeps_therm)
    print(f"  {n_sweeps_therm} Metropolis sweeps completed")
    print(f"  Mean acceptance rate: {result['mean_acceptance']:.4f}")
    print(f"  Final action: {result['final_action']:.4f}")
    print(f"  Action change: {S0:.4f} -> {result['final_action']:.4f}")

    print_subheader("2b: Spinal Trajectory (Langevin) Sampling")
    gf_langevin = gf.copy()
    langevin = SpinalTrajectorySampler(gf_langevin, dtau=0.01, seed=237)
    n_langevin = 10
    result_lang = langevin.run(n_langevin)
    print(f"  {n_langevin} Langevin steps completed")
    print(f"  Final action: {result_lang['final_action']:.4f}")

    print_subheader("2c: Bio-Inspired Levy-Flight Sampling")
    gf_bio = gf.copy()
    bio_sampler = BioInspiredSampler(gf_bio, base_eps=0.3, levy_alpha=1.5, seed=237)
    n_bio = 3
    result_bio = bio_sampler.run(n_bio)
    print(f"  {n_bio} bio-inspired sweeps completed")
    print(f"  Mean acceptance rate: {result_bio['mean_acceptance']:.4f}")
    print(f"  Final action: {result_bio['final_action']:.4f}")
    print(f"  Adapted eps: {result_bio['final_eps']:.4f}")

    gf = gf_therm

    # ================================================================
    # Step 3: Plaquette quality and topology
    # ================================================================
    print_header("Step 3: Plaquette Quality and Topological Analysis")

    print_subheader("3a: Plaquette Quality Metrics")
    plaq_info = plaquette_quality_field(gf)
    print(f"  Mean plaquette <P>: {plaq_info['mean']:.6f}")
    print(f"  Std(P): {plaq_info['std']:.6f}")
    print(f"  Skewness: {plaq_info['skewness']:.4f}")
    print(f"  Kurtosis: {plaq_info['kurtosis']:.4f}")
    print(f"  Quality fraction (P > 0.9): {plaq_info['quality_fraction']:.4f}")
    print(f"  Hot fraction (P < 0.5): {plaq_info['hot_fraction']:.4f}")
    print(f"  Range: [{plaq_info['min']:.4f}, {plaq_info['max']:.4f}]")

    print_subheader("3b: Topological Charge (Clover)")
    top_info = topological_charge_clover(gf)
    print(f"  Topological charge Q = {top_info['Q']:.6f}")
    print(f"  Rounded Q = {top_info['Q_round']}")
    print(f"  |Q| = {top_info['abs_Q']:.6f}")

    print_subheader("3c: Fractal Dimension of Topological Support")
    fractal_info = fractal_dimension_topological(
        top_info['q_density'], params.shape, threshold=1e-6, n_boxes=3)
    print(f"  Box-counting dimension D_box = {fractal_info['D_box']:.4f}")
    print(f"  Threshold: {fractal_info['threshold']:.2e}")
    print(f"  Box sizes: {fractal_info['box_sizes']}")
    print(f"  Box counts: {fractal_info['N_boxes']}")

    # ================================================================
    # Step 4: High-order finite differences
    # ================================================================
    print_header("Step 4: High-Order Covariant Finite Differences")

    psi = SpinorField(params, zero=False)
    psi.data = rng.standard_normal(psi.data.shape) + 1j * rng.standard_normal(psi.data.shape)

    def psi_at(s):
        return psi.at(s)

    test_site = SiteIndex(0, 0, 0, 0)

    print_subheader("4a: Covariant Derivative (Order 2)")
    D2_mu0 = covariant_derivative(gf, psi_at, test_site, mu=0, order=2)
    print(f"  Shape: {D2_mu0.shape}")
    print(f"  Norm: {np.linalg.norm(D2_mu0):.6e}")

    print_subheader("4b: Covariant Derivative (Order 4)")
    D4_mu0 = covariant_derivative(gf, psi_at, test_site, mu=0, order=4)
    print(f"  Shape: {D4_mu0.shape}")
    print(f"  Norm: {np.linalg.norm(D4_mu0):.6e}")

    print_subheader("4c: Covariant Derivative (Order 6)")
    D6_mu0 = covariant_derivative(gf, psi_at, test_site, mu=0, order=6)
    print(f"  Shape: {D6_mu0.shape}")
    print(f"  Norm: {np.linalg.norm(D6_mu0):.6e}")

    print_subheader("4d: Finite-Difference Convergence")
    diff_24 = np.linalg.norm(D2_mu0 - D4_mu0)
    diff_46 = np.linalg.norm(D4_mu0 - D6_mu0)
    print(f"  ||D^(2) - D^(4)|| = {diff_24:.6e}")
    print(f"  ||D^(4) - D^(6)|| = {diff_46:.6e}")
    print(f"  Higher-order difference smaller: {diff_46 < diff_24}")

    print_subheader("4e: Covariant Laplacian (Order 2 and 4)")
    lap2 = covariant_laplacian(gf, psi_at, test_site, order=2)
    lap4 = covariant_laplacian(gf, psi_at, test_site, order=4)
    print(f"  ||Lap^(2)|| = {np.linalg.norm(lap2):.6e}")
    print(f"  ||Lap^(4)|| = {np.linalg.norm(lap4):.6e}")
    print(f"  ||Lap^(2) - Lap^(4)|| = {np.linalg.norm(lap2 - lap4):.6e}")

    # ================================================================
    # Step 5: Wilson-Dirac operator and propagator
    # ================================================================
    print_header("Step 5: Wilson-Dirac Operator and Quark Propagator")

    print_subheader("5a: Wilson-Dirac Operator")
    D = WilsonDiracOperator(gf)
    print(f"  kappa = {D.kappa:.6f}")
    print(f"  kappa_c = {params.critical_kappa():.6f}")
    print(f"  kappa / kappa_c = {D.kappa / params.critical_kappa():.4f}")

    test_spinor = make_point_source(params, SiteIndex(0, 0, 0, 0), spin_idx=0, color_idx=0)
    D_test = D.apply(test_spinor)
    print(f"  ||D psi_point|| = {np.sqrt(D_test.norm_sq()):.6e}")

    print_subheader("5b: Condition Number Estimation")
    cond_info = estimate_condition_number(gf, n_iter=10, seed=237)
    print(f"  lambda_max(D^dag D) = {cond_info['lambda_max']:.6e}")
    print(f"  lambda_min(D^dag D) = {cond_info['lambda_min']:.6e}")
    print(f"  Condition number = {cond_info['condition_number']:.4e}")

    print_subheader("5c: Quark Propagator (CGNE solver)")
    source_site = SiteIndex(0, 0, 0, 0)
    prop_result = solve_propagator(gf, source_site, method='cgne',
                                     max_iter=100, tol=1e-4,
                                     spin_idx=0, color_idx=0)
    prop = prop_result['propagator']
    info = prop_result['solver_info']
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    if info['residuals']:
        print(f"  Initial residual: {info['residuals'][0]:.6e}")
        print(f"  Final residual: {info['residuals'][-1]:.6e}")
    print(f"  ||S(x; 0)|| = {np.sqrt(prop.norm_sq()):.6e}")

    print_subheader("5d: Jacobi Solver Comparison")
    jacobi_result = solve_propagator(gf, source_site, method='jacobi',
                                       max_iter=50, tol=1e-3)
    jac_info = jacobi_result['solver_info']
    print(f"  Jacobi converged: {jac_info['converged']}")
    print(f"  Jacobi iterations: {jac_info['iterations']}")
    if jac_info['residuals']:
        print(f"  Jacobi final residual: {jac_info['residuals'][-1]:.6e}")

    # ================================================================
    # Step 6: Spectral Analysis
    # ================================================================
    print_header("Step 6: Spectral Analysis")

    print_subheader("6a: Arnoldi Eigenvalue Estimation")
    eig_info = arnoldi_eigenvalues(D, n_eig=6, seed=237, which='SM')
    eigs = eig_info['eigenvalues']
    print(f"  Computed {len(eigs)} eigenvalues")
    print(f"  Arnoldi iterations: {eig_info['n_iterations']}")
    for i, ev in enumerate(eigs):
        print(f"    lambda_{i} = {ev.real:+.6f} {ev.imag:+.6f}i  (|lambda| = {abs(ev):.6f})")

    print_subheader("6b: Spectral Density Histogram")
    if len(eigs) > 1:
        hist_re = spectral_density_histogram(eigs, n_bins=5, mode='real')
        hist_abs = spectral_density_histogram(eigs, n_bins=5, mode='abs')
        print(f"  Re(lambda): mean = {hist_re['mean']:.6f}, std = {hist_re['std']:.6f}")
        print(f"  |lambda|:   mean = {hist_abs['mean']:.6f}, std = {hist_abs['std']:.6f}")

    print_subheader("6c: Banks-Casher Estimate")
    bc = banks_casher_estimate(eigs, V=params.V, lambda_window=0.5)
    print(f"  rho(0) = {bc['rho_0']:.6e}")
    print(f"  Chiral condensate = {bc['condensate']:.6e}")
    print(f"  N eigenvalues in window: {bc['n_in_window']}")

    print_subheader("6d: Newton Interpolation of Spectral Function")
    if len(eigs) >= 2:
        x_eval = np.linspace(-1, 1, 5)
        resolvent = newton_interp_spectral_function(eigs, x_eval, func='resolvent')
        print(f"  Resolvent G(x) at 5 test points:")
        for i, (x, g) in enumerate(zip(x_eval, resolvent)):
            print(f"    G({x:+.2f}) = {g.real:+.6f} {g.imag:+.6f}i")

    # ================================================================
    # Step 7: Gauge Orbit Analysis
    # ================================================================
    print_header("Step 7: Gauge Orbit Analysis")

    print_subheader("7a: Frobenius Distance Between Configurations")
    d_cold_hot = gauge_field_frobenius_distance(gf, GaugeField(params, rng).initialize_hot())
    d_cold_cold = gauge_field_frobenius_distance(gf, GaugeField(params, rng).initialize_cold())
    print(f"  d(thermalized, hot) = {d_cold_hot:.6f}")
    print(f"  d(thermalized, cold) = {d_cold_cold:.6f}")

    print_subheader("7b: Brute-Force Closest Gauge Transform")
    gf_target = GaugeField(params, rng).initialize_cold()
    orbit_info = find_closest_gauge_transform_brute(gf, gf_target, n_random=10, seed=237)
    print(f"  Min distance to cold config: {orbit_info['min_distance']:.6f}")

    print_subheader("7c: Permutation Distance")
    gf2 = GaugeField(params, rng).initialize_approx_cold(noise_level=0.05)
    perm_info = permutation_distance_between_configs(gf, gf2)
    print(f"  Ulam distance: {perm_info['ulam_distance']}")
    print(f"  Kendall tau: {perm_info['kendall_tau']}")
    print(f"  Spearman rho: {perm_info['spearman_rho']:.6f}")

    print_subheader("7d: Landau Gauge Fixing")
    landau_info = landau_gauge_fix(gf, n_iter=5, omega_relax=1.5)
    print(f"  Converged: {landau_info['converged']}")
    print(f"  Iterations: {landau_info['n_iterations']}")
    print(f"  Functional: {landau_info['functional']:.6f}")
    print(f"  Theta (gauge violation): {landau_info['theta']:.6e}")

    # ================================================================
    # Step 8: Codeword Gauge Fixing
    # ================================================================
    print_header("Step 8: Codeword-Based Gauge Optimization")

    print_subheader("8a: Codeword Matrix Construction")
    M = build_codeword_matrix(gf)
    print(f"  Codeword matrix shape: {M.shape}")
    print(f"  Initial overlap objective: {overlap_objective(M)}")

    print_subheader("8b: Overlap Reduction")
    cw_result = gauge_fix_via_codeword(gf, n_iters=100, seed=237)
    print(f"  Initial objective: {cw_result['initial_objective']}")
    print(f"  Final objective: {cw_result['final_objective']}")
    print(f"  Reduction factor: {cw_result['reduction_factor']:.4f}")

    print_subheader("8c: Birregular Code Construction")
    birreg = birregular_code(n_features=8, n_neurons=16, K=4, seed=237)
    print(f"  Birregular code shape: {birreg.shape}")
    print(f"  Row sums (feature degrees): {birreg.sum(axis=1)[:4]}...")
    print(f"  Column sums (neuron degrees): {birreg.sum(axis=0)[:4]}...")

    # ================================================================
    # Step 9: Coupled Eigenvalue Estimation
    # ================================================================
    print_header("Step 9: Coupled Eigenvalue Estimation (MTLR)")

    print_subheader("9a: Coupled Spectral Analysis")
    coupled = coupled_spectral_analysis(gf, n_eig=5, seed=237)
    print(f"  Eigenvalues of D_W:")
    for i, ev in enumerate(coupled['eigs_D']):
        print(f"    lambda_{i} = {ev.real:+.6f} {ev.imag:+.6f}i")
    print(f"  Spectral gap: {coupled['spectral_gap']:.6e}")
    print(f"  Condition estimate: {coupled['condition_estimate']:.4e}")
    print(f"  Coupling error: {coupled['coupling_error']:.6e}")

    print_subheader("9b: Temporal Spectral Prediction")
    predictor = TemporalSpectralPredictor(tau=3.0, memory=3)
    if metropolis.stats['action_history']:
        action_ts = np.array(metropolis.stats['action_history'])
        print(f"  Action time series: {action_ts}")
        if len(action_ts) >= 2:
            pred_next = predictor.predict_next(action_ts)
            print(f"  Predicted next action: {pred_next:.6f}")

    # ================================================================
    # Step 10: Topological Sector Enumeration & Interpolation
    # ================================================================
    print_header("Step 10: Topological Sectors and Interpolation")

    print_subheader("10a: Boolean Canalization Counts")
    for n in range(1, 5):
        b = B_star(n)
        c = count_canalizing_functions(n)
        print(f"  n={n}: B*(n) = {b}, canalizing = {c}")

    print_subheader("10b: Topological Sector Enumeration")
    sectors = enumerate_topological_sectors(n_plaquettes_per_cube=6, max_Q=2)
    print(f"  Number of cubes: {sectors['n_cubes']}")
    for Q, count in sectors['sectors'].items():
        print(f"    Q = {Q:+d}: {count} configurations")

    print_subheader("10c: Expected Topological Distribution")
    topo_dist = expected_topological_distribution(beta=params.beta, V=params.V)
    print(f"  Mean Q = {topo_dist['mean']:.4f}")
    print(f"  sigma(Q) = {topo_dist['sigma']:.4f}")
    print(f"  Variance = {topo_dist['variance']:.4f}")
    print(f"  P(Q) distribution:")
    for Q, P in zip(topo_dist['Q_values'], topo_dist['P_Q']):
        if P > 1e-6:
            print(f"    Q = {Q:+d}: P = {P:.6f}")

    print_subheader("10d: Canalization Entropy")
    H = canalization_entropy(n_vars=2)
    print(f"  Canalization entropy H = {H:.6f} nats")

    print_subheader("10e: Topological Susceptibility")
    Q_ts = topo_dist['Q_values']
    P_ts = topo_dist['P_Q']
    Q2_mean = float(np.sum(Q_ts ** 2 * P_ts))
    chi_t = topological_susceptibility(np.array([Q2_mean ** 0.5]), V=params.V)
    print(f"  chi_t = {chi_t['chi_t']:.6e}")

    print_subheader("10f: Multilinear Interpolation")
    scalar_field = rng.standard_normal(params.shape)
    test_coords = np.array([0.5, 1.5, 2.5, 3.5])
    interp_val = multilinear_interpolate_scalar(scalar_field, test_coords, params.shape)
    print(f"  Interpolated value at {test_coords}: {interp_val:.6f}")

    print_subheader("10g: Gauge-Covariant Interpolation")
    coords_interp = np.array([1.3, 2.7, 0.5, 3.1])
    psi_interp = gauge_covariant_interpolate(gf, prop, coords_interp)
    print(f"  Interpolated propagator shape: {psi_interp.shape}")
    print(f"  ||psi_interp|| = {np.linalg.norm(psi_interp):.6e}")

    print_subheader("10h: Newton Polynomial Interpolation")
    x_nodes = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    f_vals = np.sin(x_nodes) + 0.1j * x_nodes ** 2
    x_eval = np.array([0.15, 0.25, 0.35])
    f_interp = newton_polynomial_interpolation(x_nodes, f_vals, x_eval)
    f_exact = np.sin(x_eval) + 0.1j * x_eval ** 2
    err = np.max(np.abs(f_interp - f_exact))
    print(f"  Newton interpolation max error: {err:.6e}")

    print_subheader("10i: Continuum Extrapolation")
    a_values = np.array([0.12, 0.10, 0.08, 0.06])
    observables = 1.0 + 0.5 * a_values ** 2 + 0.1 * a_values ** 4 + 0.01 * rng.standard_normal(4)
    cont = continuum_extrapolation(observables, a_values, a_cont=0.0, order=2)
    print(f"  Continuum-extrapolated value: {cont['O_cont']:.6f}")
    print(f"  Fit coefficients: {cont['coefficients']}")
    print(f"  chi^2 = {cont['chi2']:.6e}")
    print(f"  True value (a=0): 1.0")

    # ================================================================
    # Final summary
    # ================================================================
    print_header("Experiment Summary")
    print(f"  Lattice: {params.Ns}^3 x {params.Nt}, a = {params.a} fm")
    print(f"  beta = {params.beta}, kappa = {params.kappa:.6f}")
    print(f"  Thermalized action: {result['final_action']:.4f}")
    print(f"  Mean plaquette: {plaq_info['mean']:.6f}")
    print(f"  Topological charge: {top_info['Q']:.4f}")
    print(f"  Spectral gap: {coupled['spectral_gap']:.6e}")
    print(f"  Condition number: {coupled['condition_estimate']:.4e}")
    print(f"  Propagator solver converged: {info['converged']}")
    print(f"  Codeword objective reduction: {cw_result['reduction_factor']:.2f}x")
    print()
    print("  All steps completed successfully.")
    print()

    return True


if __name__ == '__main__':
    try:
        success = run_experiment()
        if success:
            print("Lattice QCD experiment finished without errors.")
            sys.exit(0)
        else:
            print("Lattice QCD experiment encountered issues.")
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
