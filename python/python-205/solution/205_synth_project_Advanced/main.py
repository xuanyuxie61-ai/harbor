"""
main.py - Unified Entry Point for Multi-Physics Surrogate-Based UQ

Project: Adaptive Polynomial Chaos Surrogate Modeling for Uncertainty
         Quantification in Coupled Laser-Reaction-Diffusion Systems

This script executes the complete UQ pipeline:
  1. Define uncertain parameters and their distributions
  2. Generate experimental design (DOE) for training data
  3. Evaluate expensive truth models at design points
  4. Build polynomial chaos surrogates via regression
  5. Adaptive refinement with pseudo-labeling
  6. Uncertainty propagation via Monte Carlo
  7. Sobol sensitivity analysis
  8. Stability analysis under uncertainty
  9. Robust optimal control using surrogate

All outputs are numerical (no visualization).
Zero parameters required - runs end-to-end automatically.

Scientific Problem
------------------
We consider a multi-physics system where:
  - A reaction-diffusion predator-prey system forms Turing patterns
  - The reaction rates depend on uncertain parameters (xi_1, ..., xi_d)
  - A bad-cavity laser modulates the energy input
  - The domain geometry is parameterized by uncertain shape parameters

The goal is to quantify how parameter uncertainty affects:
  - Pattern characteristics (amplitude, wavelength)
  - System stability (Turing conditions)
  - Laser output (power, linewidth)
  - Energy costs

And to find robust optimal controls that perform well despite uncertainty.

Usage
-----
  python main.py
"""

import numpy as np
import time
import sys
import os

# Import all modules
from numerical_utils import (legendre_ek_compute, total_order_multi_index,
                              hyperbolic_cross_multi_index, gamma_ln,
                              beta_function, condition_number_estimate)
from geometry_domain import DomainGeometry, sdf_circle, sdf_union
from io_utils import (StructuredDataWriter, StructuredDataReader,
                      SimulationBatchManager, save_surrogate_metadata)
from truth_model import (ReactionDiffusionParams, LaserParams,
                          run_reaction_diffusion_1d, solve_laser_steady_state,
                          compute_laser_linewidth, compute_network_energy,
                          evaluate_truth_model)
from design_of_experiments import (halton_sequence, latin_hypercube_sample,
                                    generate_doe, tensor_product_quadrature)
from polynomial_chaos import PolynomialChaosSurrogate
from surrogate_fitting import (MultiResponseSurrogate, select_optimal_degree,
                                compute_validation_metrics, preprocess_noisy_data)
from adaptive_refinement import AdaptiveRefinement
from uncertainty_propagation import (monte_carlo_integration,
                                      kernel_density_estimate,
                                      compute_confidence_intervals,
                                      run_uncertainty_propagation)
from stability_analysis import (eigenvalue_classification,
                                 turing_stability_analysis,
                                 stability_probability, compute_stability_map)
from optimal_control_uq import (forward_backward_sweep, RobustOptimalControl,
                                 evaluate_chance_constraint)


def print_section(title: str, char: str = '='):
    """Print a section header."""
    width = 72
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_metrics(metrics: dict, indent: int = 2):
    """Print metrics dictionary."""
    prefix = ' ' * indent
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.6e}")
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 4)
        elif isinstance(value, np.ndarray):
            if value.size <= 10:
                print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: ndarray({value.shape})")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    """Execute the complete UQ pipeline."""
    total_start = time.time()

    print("=" * 72)
    print("  ADAPTIVE POLYNOMIAL CHAOS SURROGATE MODELING")
    print("  for Uncertainty Quantification in")
    print("  Coupled Laser-Reaction-Diffusion Systems")
    print("=" * 72)

    # ===================================================================
    # STEP 1: Define uncertain parameters and distributions
    # ===================================================================
    print_section("STEP 1: Parameter Definition and Distributions")

    # Uncertain parameters for the reaction-diffusion system
    param_names = ['D_u', 'D_v', 'alpha', 'beta']
    d = len(param_names)

    # Parameter bounds (uniform distributions on these intervals)
    param_bounds = {
        'D_u': (0.005, 0.02),     # prey diffusion
        'D_v': (0.3, 0.8),        # predator diffusion
        'alpha': (1.5, 2.5),      # prey growth rate
        'beta': (0.5, 1.5),       # predation rate
    }

    print(f"  Dimension d = {d}")
    print(f"  Parameters: {param_names}")
    for name in param_names:
        lo, hi = param_bounds[name]
        print(f"    {name} ~ U({lo}, {hi})")

    # ===================================================================
    # STEP 2: Domain geometry setup
    # ===================================================================
    print_section("STEP 2: Domain Geometry")

    domain = DomainGeometry('rectangle', {
        'x_min': 0.0, 'x_max': 1.0,
        'y_min': 0.0, 'y_max': 1.0
    })
    area = domain.domain_area_estimate(100, 100)
    print(f"  Domain type: {domain.domain_type}")
    print(f"  Domain area (estimate): {area:.4f}")

    # Also test annulus domain
    domain_annulus = DomainGeometry('annulus', {
        'center': (0.5, 0.5),
        'r_inner': 0.1,
        'r_outer': 0.4
    })
    area_ann = domain_annulus.domain_area_estimate(100, 100)
    print(f"  Annulus domain area (estimate): {area_ann:.4f}")

    # ===================================================================
    # STEP 3: Generate experimental design
    # ===================================================================
    print_section("STEP 3: Experimental Design (DOE)")

    # Generate Halton sequence DOE
    n_train = 80  # Training points
    X_doe, w_doe = generate_doe(d, n_train, method='halton',
                                param_bounds=param_bounds, seed=42)
    print(f"  Halton DOE: {X_doe.shape[0]} points in {d}D")
    print(f"  X range: [{X_doe.min():.4f}, {X_doe.max():.4f}]")

    # Also generate LHS for comparison
    X_lhs, w_lhs = generate_doe(d, n_train, method='lhs',
                                param_bounds=param_bounds, seed=42)
    print(f"  LHS DOE: {X_lhs.shape[0]} points in {d}D")

    # Generate candidate pool for adaptive refinement
    n_candidates = 200
    X_candidates, _ = generate_doe(d, n_candidates, method='halton',
                                   param_bounds=param_bounds, seed=123)
    print(f"  Candidate pool: {X_candidates.shape[0]} points")

    # ===================================================================
    # STEP 4: Evaluate truth model at design points
    # ===================================================================
    print_section("STEP 4: Truth Model Evaluation")

    # Batch manager for organizing simulations
    batch_mgr = SimulationBatchManager('.')

    # Map DOE points to truth model parameters and evaluate
    Y_rd_amplitude = np.zeros(n_train)
    Y_rd_wavelength = np.zeros(n_train)
    Y_rd_max_u = np.zeros(n_train)
    Y_laser_power = np.zeros(n_train)
    Y_network_energy = np.zeros(n_train)

    print(f"  Evaluating {n_train} design points...")
    t_eval_start = time.time()

    for i in range(n_train):
        # Map uncertain parameters to laser model too
        # w depends on D_u, N_atom depends on D_v, Omega depends on alpha
        w_laser = 1.5 + 0.5 * (X_doe[i, 2] - 1.5)  # alpha -> pump rate
        N_atom = 5000.0 + 10000.0 * (X_doe[i, 1] - 0.3) / 0.5  # D_v -> atom number
        omega_laser = 0.8 + 0.4 * (X_doe[i, 0] - 0.005) / 0.015  # D_u -> coupling

        params_dict = {
            'D_u': float(X_doe[i, 0]),
            'D_v': float(X_doe[i, 1]),
            'alpha': float(X_doe[i, 2]),
            'beta': float(X_doe[i, 3]),
            'gamma_rd': 1.0,
            'delta': 0.5,
            'epsilon': 0.05,
            'w': float(w_laser),
            'N_atom': float(N_atom),
            'Omega': float(omega_laser),
            'kappa': 1e3,
            'N_nodes': 1000,
            'k_clusters': max(1, int(10 * X_doe[i, 0] / 0.02)),
            'd_to_BS': 50.0 + 100.0 * (X_doe[i, 3] - 0.5),
            'M_network': 50.0 + 100.0 * X_doe[i, 2] / 2.5,
        }

        # Register scenario
        sid = batch_mgr.add_scenario(params_dict)

        # Evaluate combined truth model
        result = evaluate_truth_model(params_dict, physics='combined')

        Y_rd_amplitude[i] = result.get('pattern_amplitude', 0.0)
        Y_rd_wavelength[i] = result.get('pattern_wavelength', 1.0)
        Y_rd_max_u[i] = result.get('max_u', 0.0)
        Y_laser_power[i] = result.get('laser_power', 0.0)
        Y_network_energy[i] = result.get('network_energy', 0.0)

        batch_mgr.store_result(sid, result)

    t_eval = time.time() - t_eval_start
    print(f"  Evaluation time: {t_eval:.2f} seconds")
    print(f"  Pattern amplitude range: [{Y_rd_amplitude.min():.4e}, {Y_rd_amplitude.max():.4e}]")
    print(f"  Pattern wavelength range: [{Y_rd_wavelength.min():.4e}, {Y_rd_wavelength.max():.4e}]")
    print(f"  Laser power range: [{Y_laser_power.min():.4e}, {Y_laser_power.max():.4e}]")
    print(f"  Network energy range: [{Y_network_energy.min():.4e}, {Y_network_energy.max():.4e}]")

    # ===================================================================
    # STEP 5: Build polynomial chaos surrogates
    # ===================================================================
    print_section("STEP 5: Polynomial Chaos Surrogate Construction")

    # Select optimal degree
    opt_degree = select_optimal_degree(X_doe, Y_rd_amplitude,
                                       max_degree_test=4, d=d)
    print(f"  Optimal degree for pattern amplitude: {opt_degree}")

    # Multi-response surrogate
    multi_surrogate = MultiResponseSurrogate(d, max_degree=opt_degree,
                                             index_set_type='total_order')
    multi_surrogate.add_qoi('pattern_amplitude')
    multi_surrogate.add_qoi('pattern_wavelength')
    multi_surrogate.add_qoi('max_u')
    multi_surrogate.add_qoi('laser_power')
    multi_surrogate.add_qoi('network_energy')

    Y_dict = {
        'pattern_amplitude': Y_rd_amplitude,
        'pattern_wavelength': Y_rd_wavelength,
        'max_u': Y_rd_max_u,
        'laser_power': Y_laser_power,
        'network_energy': Y_network_energy
    }

    # Fit all surrogates
    errors = multi_surrogate.fit_all(X_doe, Y_dict, method='ridge',
                                     lambda_reg=1e-4)
    print(f"  Surrogate fitting results:")
    for qoi_name, err in errors.items():
        print(f"    {qoi_name}:")
        print(f"      R2 = {err['r2']:.6f}, LOO Q2 = {err['loo_q2']:.6f}")
        print(f"      RMSE = {err['rmse']:.6e}, N_terms = {err['n_terms']}")

    # ===================================================================
    # STEP 6: Adaptive refinement
    # ===================================================================
    print_section("STEP 6: Adaptive Refinement with Pseudo-Labeling")

    # Set up adaptive refinement for pattern amplitude
    def simple_truth(params_dict):
        """Simplified truth model for refinement."""
        result = evaluate_truth_model(params_dict, physics='rd')
        return {'pattern_amplitude': result.get('pattern_amplitude', 0.0)}

    adaptive = AdaptiveRefinement(
        d=d, max_degree=opt_degree,
        index_set_type='total_order',
        truth_model=simple_truth
    )

    # Initialize with existing data
    adaptive.initialize(X_doe, Y_rd_amplitude)
    print(f"  Initial training: {adaptive.X_train.shape[0]} points")
    print(f"  Initial LOO Q2: {adaptive.history[-1].get('loo_loo_q2', 'N/A')}")

    # Run adaptive refinement
    refinement_result = adaptive.run_adaptive_loop(
        X_candidates, max_iterations=5, n_per_iter=10,
        q2_threshold=0.90, criterion='ucb',
        pseudo_label_ratio=0.3
    )

    print(f"  Refinement converged: {refinement_result['converged']}")
    print(f"  Iterations: {refinement_result['n_iterations']}")
    print(f"  Final training size: {refinement_result['n_final_train']}")

    # ===================================================================
    # STEP 7: Uncertainty propagation
    # ===================================================================
    print_section("STEP 7: Uncertainty Propagation via Monte Carlo")

    # Get the best surrogate (pattern amplitude)
    surrogate_amp = multi_surrogate.surrogates['pattern_amplitude']

    # Run full UQ pipeline
    uq_results = run_uncertainty_propagation(
        surrogate_amp, param_names, param_bounds,
        n_mc=5000, seed=42
    )

    print("  Monte Carlo Results:")
    print_metrics(uq_results['mc_moments'])
    print("\n  PCE-based Moments:")
    print_metrics(uq_results['pce_moments'])
    print("\n  Confidence Intervals:")
    print(f"    95% CI: [{uq_results['confidence_intervals']['95pct']['lower']:.6e}, "
          f"{uq_results['confidence_intervals']['95pct']['upper']:.6e}]")
    print(f"    99% CI: [{uq_results['confidence_intervals']['99pct']['lower']:.6e}, "
          f"{uq_results['confidence_intervals']['99pct']['upper']:.6e}]")
    print("\n  Percentiles:")
    print_metrics(uq_results['percentiles'])

    # PDF estimation
    mc_for_pdf = monte_carlo_integration(
        surrogate_amp, 3000,
        np.array([param_bounds[n][0] for n in param_names]),
        np.array([param_bounds[n][1] for n in param_names]),
        seed=99)
    y_grid, pdf_vals = kernel_density_estimate(mc_for_pdf['y_samples'])
    print(f"\n  PDF estimation: {len(y_grid)} grid points")
    print(f"  PDF max: {pdf_vals.max():.6e} at y = {y_grid[np.argmax(pdf_vals)]:.6e}")

    # ===================================================================
    # STEP 8: Sobol sensitivity analysis
    # ===================================================================
    print_section("STEP 8: Sobol Sensitivity Analysis")

    for qoi_name in multi_surrogate.qoi_names:
        surrogate = multi_surrogate.surrogates[qoi_name]
        if surrogate.is_fitted:
            sobol = surrogate.sobol_indices()
            print(f"\n  {qoi_name}:")
            print(f"    First-order indices:")
            for j, name in enumerate(param_names):
                print(f"      S_{name} = {sobol['first_order'][j]:.6f}")
            print(f"    Total-order indices:")
            for j, name in enumerate(param_names):
                print(f"      ST_{name} = {sobol['total_order'][j]:.6f}")
            print(f"    Sum S_i = {np.sum(sobol['first_order']):.6f}")
            print(f"    Sum ST_i = {np.sum(sobol['total_order']):.6f}")

    # ===================================================================
    # STEP 9: Stability analysis under uncertainty
    # ===================================================================
    print_section("STEP 9: Stability Analysis Under Uncertainty")

    # Compute stability for each training point
    n_stability = min(n_train, 50)  # Limit for speed
    n_stable = 0
    n_turing = 0
    n_unstable = 0

    for i in range(n_stability):
        rd_params = ReactionDiffusionParams(
            alpha=float(X_doe[i, 2]),
            beta=float(X_doe[i, 3]),
            D_u=float(X_doe[i, 0]),
            D_v=float(X_doe[i, 1])
        )
        turing = rd_params.turing_condition()
        if turing['turing_possible']:
            n_turing += 1
        elif turing['trace'] < 0 and turing['determinant'] > 0:
            n_stable += 1
        else:
            n_unstable += 1

    total = float(n_stability)
    print(f"  Stability analysis over {n_stability} parameter samples:")
    print(f"    P(stable) = {n_stable / total:.4f}")
    print(f"    P(Turing) = {n_turing / total:.4f}")
    print(f"    P(unstable) = {n_unstable / total:.4f}")

    # Eigenvalue classification demo
    rd_demo = ReactionDiffusionParams(
        alpha=2.0, beta=1.0, D_u=0.01, D_v=0.5
    )
    J = rd_demo.jacobian_at_steady_state()
    eig_info = eigenvalue_classification(J[0, 0] + J[1, 1],
                                         J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    print(f"\n  Nominal eigenvalue analysis:")
    print(f"    Type: {eig_info['type']}")
    print(f"    Stable: {eig_info['stable']}")
    print(f"    Eigenvalues: {eig_info['eigenvalues']}")

    # Turing analysis at nominal point
    turing_demo = turing_stability_analysis(
        J[0, 0], J[0, 1], J[1, 0], J[1, 1],
        rd_demo.D_u, rd_demo.D_v
    )
    print(f"\n  Turing analysis at nominal point:")
    print(f"    Turing possible: {turing_demo['turing_possible']}")
    print(f"    Critical wavenumber: {turing_demo['critical_wavenumber']:.6f}")
    print(f"    Critical wavelength: {turing_demo['critical_wavelength']:.6f}")
    print(f"    Max growth rate: {turing_demo['max_growth_rate']:.6e}")

    # Stability map
    stab_map = compute_stability_map((-3.0, 1.0), (-1.0, 3.0), 20, 20)
    print(f"\n  Stability map computed: {stab_map['trace_grid'].shape}")

    # ===================================================================
    # STEP 10: Robust optimal control
    # ===================================================================
    print_section("STEP 10: Robust Optimal Control")

    # Simple ODE-constrained optimal control problem
    def state_rhs(t, x, u):
        """State equation for predator-prey with harvesting control."""
        x1, x2 = max(x[0], 1e-10), max(x[1], 1e-10)
        alpha_val = 2.0
        beta_val = 1.0
        eps_val = 0.05
        gamma_val = 1.0
        dx1 = alpha_val * x1 - beta_val * x1 * x2 - u * x1
        dx2 = eps_val * (beta_val * x1 * x2 - gamma_val * x2)
        return np.array([dx1, dx2])

    def adjoint_rhs(t, x, u, lam):
        """Adjoint equation."""
        x1, x2 = max(x[0], 1e-10), max(x[1], 1e-10)
        alpha_val = 2.0
        beta_val = 1.0
        eps_val = 0.05
        gamma_val = 1.0
        df1_dx1 = alpha_val - beta_val * x2 - u
        df1_dx2 = -beta_val * x1
        df2_dx1 = eps_val * beta_val * x2
        df2_dx2 = eps_val * (beta_val * x1 - gamma_val)
        dlam1 = -(lam[0] * df1_dx1 + lam[1] * df2_dx1)
        dlam2 = -(lam[0] * df1_dx2 + lam[1] * df2_dx2)
        return np.array([dlam1, dlam2])

    def control_update(t, x, lam):
        """Optimal control from dH/du = 0."""
        x1 = max(x[0], 1e-10)
        u_opt = lam[0] * x1 / 2.0
        return u_opt

    x0 = np.array([1.0, 0.5])
    ocp_result = forward_backward_sweep(
        state_rhs, adjoint_rhs, control_update,
        x0, t_final=10.0, n_steps=200,
        u_min=0.0, u_max=0.5,
        relaxation=0.3, tol=1e-3, max_iter=30
    )

    print(f"  Optimal control problem solved:")
    print(f"    Converged: {ocp_result['converged']}")
    print(f"    Iterations: {ocp_result['n_iterations']}")
    print(f"    Cost: {ocp_result['cost']:.6e}")
    print(f"    Final state: prey = {ocp_result['x'][-1, 0]:.4f}, "
          f"predator = {ocp_result['x'][-1, 1]:.4f}")
    print(f"    Mean control: {np.mean(ocp_result['u']):.4f}")

    # Robust control with uncertainty
    print("\n  Robust optimal control (surrogate-accelerated):")

    def surrogate_factory(u_ctrl, xi_samp):
        """Build PCE surrogate for objective at given control."""
        n_xi = xi_samp.shape[0]
        Y_obj = np.zeros(n_xi)
        for k in range(n_xi):
            alpha_k = 1.5 + xi_samp[k, 0] * 0.5
            beta_k = 0.8 + xi_samp[k, 1] * 0.3
            x = x0.copy()
            dt_oc = 10.0 / 100
            cost_k = 0.0
            for step in range(100):
                u_k = np.clip(u_ctrl[0], 0.0, 0.5)
                dx1 = alpha_k * x[0] - beta_k * x[0] * x[1] - u_k * x[0]
                dx2 = 0.05 * (beta_k * x[0] * x[1] - x[1])
                x[0] = max(x[0] + dt_oc * dx1, 1e-10)
                x[1] = max(x[1] + dt_oc * dx2, 1e-10)
                cost_k += dt_oc * (u_k ** 2 + (x[0] - 1.0) ** 2)
            Y_obj[k] = cost_k

        surr = PolynomialChaosSurrogate(d, max_degree=2)
        if len(Y_obj) > surr.n_terms:
            surr.fit_regression(xi_samp, Y_obj, 'ridge', 1e-4)
        return surr

    rng_ctrl = np.random.RandomState(42)
    xi_ctrl = rng_ctrl.uniform(-1, 1, (100, d))

    robust_ctrl = RobustOptimalControl(control_dim=1, uncertainty_dim=d, beta=0.1)
    robust_result = robust_ctrl.optimize(
        u0=np.array([0.1]),
        surrogate_factory=surrogate_factory,
        xi_samples=xi_ctrl,
        u_bounds=[(0.0, 0.5)],
        max_iter=20,
        learning_rate=0.05
    )

    print(f"    Robust optimal control: u* = {robust_result['optimal_control'][0]:.6f}")
    print(f"    Robust objective: {robust_result['optimal_objective']:.6e}")
    print(f"    Iterations: {robust_result['n_iterations']}")

    # Chance constraint evaluation
    if surrogate_amp.is_fitted:
        cc_result = evaluate_chance_constraint(
            surrogate_amp,
            threshold=float(np.mean(Y_rd_amplitude) + 2 * np.std(Y_rd_amplitude)),
            n_samples=3000,
            lower=np.array([param_bounds[n][0] for n in param_names]),
            upper=np.array([param_bounds[n][1] for n in param_names])
        )
        print(f"\n  Chance constraint evaluation:")
        print(f"    P(Y <= threshold) = {cc_result['probability']:.4f}")
        print(f"    Constraint satisfied: {cc_result['satisfied']}")
        print(f"    Violation probability: {cc_result['violation_probability']:.4f}")

    # ===================================================================
    # STEP 11: Save results
    # ===================================================================
    print_section("STEP 11: Results Archival")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(output_dir, 'uq_results.dat')
    writer = StructuredDataWriter(data_file)
    writer.add_metadata('project', '205_synth_project')
    writer.add_metadata('dimension', d)
    writer.add_metadata('n_training', n_train)
    writer.add_metadata('max_degree', opt_degree)
    writer.add_data_block('training_inputs', X_doe)
    writer.add_data_block('pattern_amplitude', X_doe, Y_rd_amplitude.reshape(-1, 1))
    writer.write()
    print(f"  Results saved to: {data_file}")

    # Save surrogate metadata
    meta_file = os.path.join(output_dir, 'surrogate_metadata.json')
    meta = {
        'dimension': d,
        'max_degree': opt_degree,
        'n_training': n_train,
        'param_names': param_names,
        'param_bounds': {k: list(v) for k, v in param_bounds.items()},
        'qoi_names': multi_surrogate.qoi_names,
        'validation_errors': {},
        'sobol_summary': {}
    }
    for k_name, v_err in errors.items():
        meta['validation_errors'][k_name] = {}
        for kk, vv in v_err.items():
            if isinstance(vv, (int, float, np.floating, np.integer)):
                meta['validation_errors'][k_name][kk] = float(vv)
            else:
                meta['validation_errors'][k_name][kk] = str(vv)

    for qoi_name in multi_surrogate.qoi_names:
        if multi_surrogate.surrogates[qoi_name].is_fitted:
            sob = multi_surrogate.surrogates[qoi_name].sobol_indices()
            meta['sobol_summary'][qoi_name] = {
                'first_order': sob['first_order'].tolist(),
                'total_order': sob['total_order'].tolist()
            }
    save_surrogate_metadata(meta_file, meta)
    print(f"  Metadata saved to: {meta_file}")

    # Verify data round-trip
    reader = StructuredDataReader(data_file)
    blocks = reader.list_blocks()
    print(f"  Data file contains {len(blocks)} blocks: {blocks}")

    # ===================================================================
    # Final Summary
    # ===================================================================
    total_time = time.time() - total_start

    print_section("PIPELINE COMPLETE - SUMMARY")
    print(f"  Total execution time: {total_time:.2f} seconds")
    print(f"  Input dimension: {d}")
    print(f"  Training points: {n_train}")
    print(f"  Polynomial degree: {opt_degree}")
    print(f"  PCE terms: {multi_surrogate.surrogates['pattern_amplitude'].n_terms}")
    print(f"  Adaptive refinement iterations: {refinement_result['n_iterations']}")
    print(f"  Final training size: {refinement_result['n_final_train']}")
    print()
    print("  QoI Surrogate Quality:")
    for qoi_name, err in errors.items():
        print(f"    {qoi_name}: R2={err['r2']:.4f}, Q2={err['loo_q2']:.4f}")
    print()
    print("  Key Sobol Indices (pattern_amplitude):")
    sobol_amp = multi_surrogate.surrogates['pattern_amplitude'].sobol_indices()
    for j, name in enumerate(param_names):
        print(f"    S_{name} = {sobol_amp['first_order'][j]:.4f}, "
              f"ST_{name} = {sobol_amp['total_order'][j]:.4f}")
    print()
    print("  Uncertainty Quantification:")
    print(f"    Mean(pattern_amplitude) = {uq_results['mc_moments']['mean']:.6e}")
    print(f"    Std(pattern_amplitude) = {uq_results['mc_moments']['std']:.6e}")
    print(f"    95% CI: [{uq_results['confidence_intervals']['95pct']['lower']:.6e}, "
          f"{uq_results['confidence_intervals']['95pct']['upper']:.6e}]")
    print()
    print("  Stability:")
    print(f"    P(stable) = {n_stable / total:.4f}")
    print(f"    P(Turing) = {n_turing / total:.4f}")
    print()
    print("  Optimal Control:")
    print(f"    Deterministic cost: {ocp_result['cost']:.6e}")
    print(f"    Robust optimal control: {robust_result['optimal_control'][0]:.6f}")
    print()
    print("=" * 72)
    print("  All computations completed successfully.")
    print("=" * 72)


if __name__ == '__main__':
    main()
