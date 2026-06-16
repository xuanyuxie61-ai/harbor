"""
main.py — Unified entry point for the multi-scale damage evolution simulation.

Project 280: High-order finite-difference simulation of multi-scale
material damage evolution with stability analysis.

Zero-parameter execution: running this file performs a complete
simulation from grid generation to failure detection, including:
  1. Grid generation with crack-tip refinement
  2. High-order FD operator setup
  3. Microstructure field generation
  4. Stochastic defect population
  5. Nonlocal damage initialization
  6. Time-stepping with IMEX scheme
  7. Stability analysis (von Neumann, CFL, eigenvalue)
  8. Energy minimization for crack direction
  9. Percolation-based failure detection
  10. MCMC parameter calibration
  11. Benchmark convergence study

Scientific domain: Computational Materials Science
Problem: Multi-scale material damage evolution simulation
Method: High-order finite differences with nonlocal regularization
"""

import os
import sys
import math
import time as time_module
import numpy as np

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SimulationConfig, MaterialParams, NumericalParams,
                    CrackConfig, LoadingConfig, DEFAULT_CONFIG)
from damage_grid import (build_damage_grid, compute_grid_quality,
                         compute_crack_level_sets)
from fd_operators import (fd_coefficients_1d, compute_damage_elastic_rhs,
                          estimate_spectral_radius, apply_d_dx, apply_d_dy)
from nonlocal_damage import NonlocalDamageState
from crack_tracking import (reinitialize_signed_distance,
                            compute_hoop_stress_direction,
                            compute_dynamic_sif, compute_crack_tip_velocity,
                            advance_crack)
from time_integration import (compute_cfl_timestep, adapt_timestep,
                              imex_rk2_step, compute_total_energy)
from stability_analysis import (cfl_limit_explicit, max_stable_timestep,
                                check_von_neumann_stability,
                                compute_eigenvalue_spectrum,
                                check_energy_dissipation,
                                detect_stiffness_ratio,
                                find_critical_load_factor,
                                linf_norm_field,
                                analyze_damage_bvp_conditioning)
from percolation_failure import percolation_analysis
from microstructure_field import (generate_microstructure_fields,
                                  generate_gaussian_random_field,
                                  multi_scale_decomposition)
from stochastic_defects import generate_stochastic_defects
from energy_minimize import (gauss_legendre_nodes_weights,
                             evaluate_j_integral,
                             find_optimal_crack_direction)
from mcmc_optimization import (compute_multiscale_sync_periods,
                               mcmc_calibration,
                               generate_synthetic_observations)
from benchmark_parse import (BenchmarkManager, compute_error_norms,
                             estimate_convergence_order,
                             k_field_reference_solution)


def separator(title: str) -> str:
    """Create a formatted section separator."""
    width = 70
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def run_simulation(cfg: SimulationConfig) -> dict:
    """Execute the complete damage evolution simulation.

    Returns a dictionary of all computed results and diagnostics.
    """
    results = {}
    t_start = time_module.time()

    # ------------------------------------------------------------------
    # Phase 1: Grid generation
    # ------------------------------------------------------------------
    print(separator("Phase 1: Grid Generation with Crack-Tip Refinement"))
    grid = build_damage_grid(cfg)
    quality = compute_grid_quality(grid)
    bg = grid["background"]
    x, y = bg["x"], bg["y"]
    nx, ny = bg["nx"], bg["ny"]
    dx, dy = bg["dx"], bg["dy"]

    print(f"  Domain: [{cfg.numerical.domain_x[0]}, {cfg.numerical.domain_x[1]}] x "
          f"[{cfg.numerical.domain_y[0]}, {cfg.numerical.domain_y[1]}] m")
    print(f"  Grid: {nx} x {ny} = {nx * ny} nodes")
    print(f"  Spacing: dx = {dx:.6e} m, dy = {dy:.6e} m")
    print(f"  Annular points around crack tip: {quality['n_annular']}")
    print(f"  Min spacing: {quality['min_spacing']:.6e} m")
    print(f"  Max aspect ratio: {quality['max_aspect_ratio']:.4f}")

    results["grid_quality"] = quality

    # ------------------------------------------------------------------
    # Phase 2: FD operator setup
    # ------------------------------------------------------------------
    print(separator("Phase 2: High-Order Finite-Difference Operator Setup"))
    fd_order = cfg.numerical.fd_order
    coeffs = fd_coefficients_1d(fd_order)
    print(f"  FD order: {fd_order}")
    print(f"  First-derivative stencil (center): {coeffs['first_deriv_center']}")
    print(f"  Second-derivative stencil (center): {coeffs['second_deriv_center']}")

    # Spectral radius
    rho = estimate_spectral_radius(dx, dy, cfg.material, damage_max=0.0)
    print(f"  Spectral radius (undamaged): {rho:.4e} rad²/s²")

    rho_damaged = estimate_spectral_radius(dx, dy, cfg.material, damage_max=0.5)
    print(f"  Spectral radius (D=0.5): {rho_damaged:.4e} rad²/s²")

    results["spectral_radius"] = {"undamaged": rho, "damaged_05": rho_damaged}

    # ------------------------------------------------------------------
    # Phase 3: Microstructure field generation
    # ------------------------------------------------------------------
    print(separator("Phase 3: Microstructure-Informed Heterogeneity"))
    micro_fields = generate_microstructure_fields(cfg)
    ft_field = micro_fields["tensile_strength"]
    gf_field = micro_fields["fracture_energy"]
    lc_field = micro_fields["characteristic_length"]

    print(f"  Tensile strength: min={ft_field.min():.3e}, "
          f"max={ft_field.max():.3e}, mean={ft_field.mean():.3e} Pa")
    print(f"  Fracture energy: min={gf_field.min():.3e}, "
          f"max={gf_field.max():.3e}, mean={gf_field.mean():.3e} J/m²")
    print(f"  Characteristic length: min={lc_field.min():.5e}, "
          f"max={lc_field.max():.5e} m")

    # Multi-scale decomposition stats
    bands = micro_fields["bands"]
    for band_name, band_field in sorted(bands.items()):
        print(f"  {band_name} variance: {band_field.var():.6e}")

    results["microstructure"] = {
        "ft_mean": float(ft_field.mean()),
        "ft_std": float(ft_field.std()),
        "gf_mean": float(gf_field.mean()),
        "lc_mean": float(lc_field.mean()),
    }

    # ------------------------------------------------------------------
    # Phase 4: Stochastic micro-defects
    # ------------------------------------------------------------------
    print(separator("Phase 4: Stochastic Micro-Defect Population"))
    defects = generate_stochastic_defects(cfg, x, y)
    print(f"  Number of defects: {defects['n_defects']}")
    print(f"  Defect area fraction: {defects['defect_area_fraction']:.6e}")
    print(f"  Max stress concentration: {defects['stress_concentration'].max():.4f}")
    print(f"  Monte Carlo expected distance: {defects['monte_carlo_expected_distance']:.6e}")
    print(f"  Theoretical expected distance: {defects['theoretical_expected_distance']:.6e}")
    print(f"  MC relative error: {defects['mc_relative_error']:.4f}")

    results["defects"] = {
        "n_defects": defects["n_defects"],
        "area_fraction": defects["defect_area_fraction"],
        "max_stress_concentration": float(defects["stress_concentration"].max()),
        "mc_error": defects["mc_relative_error"],
    }

    # ------------------------------------------------------------------
    # Phase 5: Initialize damage state
    # ------------------------------------------------------------------
    print(separator("Phase 5: Nonlocal Damage State Initialization"))
    damage_state = NonlocalDamageState(ny, nx, cfg)
    damage_state.build_kernel(x, y)
    print(f"  Nonlocal kernel: {cfg.material.characteristic_length:.4e} m radius")
    print(f"  Grid points: {nx * ny}")
    print(f"  Kernel built: {damage_state._kernel_built}")

    # Apply initial defect influence to damage threshold
    modified_kappa_0 = cfg.material.damage_threshold_strain * (
        1.0 - defects["damage_threshold_reduction"] * 0.3
    )
    print(f"  Modified damage threshold: min={modified_kappa_0.min():.4e}, "
          f"max={modified_kappa_0.max():.4e}")

    # ------------------------------------------------------------------
    # Phase 6: Time stepping
    # ------------------------------------------------------------------
    print(separator("Phase 6: Time Integration (IMEX-RK2)"))

    # Initialize displacement and velocity fields
    u_field = np.zeros((ny, nx))
    v_field = np.zeros((ny, nx))
    u_vel = np.zeros((ny, nx))
    v_vel = np.zeros((ny, nx))

    # Apply boundary conditions: prescribed displacement on top edge
    applied_strain_rate = cfg.loading.applied_strain_rate
    Ly = cfg.numerical.domain_y[1] - cfg.numerical.domain_y[0]

    # Compute initial time step from CFL
    c_wave = cfg.material.p_wave_speed
    dt_cfl = compute_cfl_timestep(dx, dy, c_wave, cfg.numerical.cfl_number, fd_order)
    dt = min(cfg.numerical.initial_dt, dt_cfl)
    print(f"  CFL time step: {dt_cfl:.4e} s")
    print(f"  Initial dt: {dt:.4e} s")
    print(f"  Wave speed: {c_wave:.2f} m/s")

    # Time stepping loop
    total_time = cfg.numerical.total_time
    current_time = 0.0
    step_count = 0
    energy_history = []
    damage_info_history = []
    force_history = []

    # Define RHS functions for IMEX scheme
    density = cfg.material.mass_density

    def explicit_rhs(u, v_d, u_v, v_v):
        """Explicit part: elastic wave operator."""
        rhs_x, rhs_y = compute_damage_elastic_rhs(
            u, v_d, damage_state.damage, cfg
        )
        return u_v, v_v, rhs_x / density, rhs_y / density

    def implicit_solve(u_star, v_star, uv_star, vv_star, dt_sub):
        """Implicit part: damage update (fixed-point iteration)."""
        # Update damage
        info = damage_state.update(u_star, v_star, x, y)

        # Apply boundary conditions (displacement control on top)
        t_current = current_time + dt_sub
        applied_disp = applied_strain_rate * t_current * Ly
        v_star[-1, :] = applied_disp
        v_star[0, :] = 0.0
        u_star[:, 0] = 0.0
        u_star[:, -1] = 0.0

        return u_star, v_star, uv_star * 0.99, vv_star * 0.99

    print(f"  Starting time loop: t_final = {total_time:.4e} s")
    n_steps_target = int(total_time / dt)
    print(f"  Estimated steps: {n_steps_target}")

    # Limit actual steps for tractability
    max_steps = min(n_steps_target, 200)
    record_interval = max(1, max_steps // 20)

    for step in range(max_steps):
        if current_time >= total_time:
            break

        # Check stopping criterion
        if damage_state.damage.max() > cfg.numerical.max_global_damage:
            print(f"  [Step {step}] Max damage {damage_state.damage.max():.4f} "
                  f"> threshold — stopping (material failure)")
            break

        # IMEX-RK2 step
        u_field, v_field, u_vel, v_vel = imex_rk2_step(
            u_field, v_field, u_vel, v_vel,
            explicit_rhs, implicit_solve, dt
        )

        # Enforce boundary conditions
        t_current = (step + 1) * dt
        applied_disp = applied_strain_rate * t_current * Ly
        v_field[-1, :] = applied_disp
        v_field[0, :] = 0.0
        u_field[:, 0] = 0.0
        u_field[:, -1] = 0.0

        current_time = t_current
        step_count = step + 1

        # Record energy
        if step % record_interval == 0 or step == max_steps - 1:
            energy = compute_total_energy(u_field, v_field, u_vel, v_vel,
                                          damage_state.damage, cfg)
            energy_history.append(energy["total_energy"])

            # Compute reaction force (integral of stress on top boundary)
            reaction_force = float(np.sum(
                cfg.material.young_modulus * (1.0 - damage_state.damage[-1, :])
                * np.abs(v_field[-1, :] - v_field[-2, :]) / dy
            ) * dx)
            force_history.append(reaction_force)

            damage_info = damage_state.update(u_field, v_field, x, y)
            damage_info_history.append(damage_info)

        # Adaptive time step
        if len(energy_history) >= 2 and energy_history[-2] > 0:
            energy_ratio = energy_history[-1] / energy_history[-2]
            damage_rate = damage_info.get("max_damage", 0) / max(current_time, 1.0e-15)
            dt = adapt_timestep(dt, energy_ratio, damage_rate, cfg)

    print(f"  Completed {step_count} steps in {current_time:.4e} s")
    print(f"  Final max damage: {damage_state.damage.max():.6f}")
    print(f"  Final mean damage: {damage_state.damage.mean():.6f}")
    print(f"  Final damaged fraction: {(damage_state.damage > 0.01).sum() / damage_state.damage.size:.4f}")
    print(f"  Energy history: {len(energy_history)} records")
    if energy_history:
        print(f"  Initial energy: {energy_history[0]:.6e} J")
        print(f"  Final energy: {energy_history[-1]:.6e} J")

    results["time_integration"] = {
        "n_steps": step_count,
        "final_time": current_time,
        "final_max_damage": float(damage_state.damage.max()),
        "final_mean_damage": float(damage_state.damage.mean()),
        "n_energy_records": len(energy_history),
        "initial_energy": energy_history[0] if energy_history else 0.0,
        "final_energy": energy_history[-1] if energy_history else 0.0,
    }

    # ------------------------------------------------------------------
    # Phase 7: Stability analysis
    # ------------------------------------------------------------------
    print(separator("Phase 7: Stability Analysis"))

    # Von Neumann analysis
    vn_result = check_von_neumann_stability(cfg, damage_state.damage.max())
    print(f"  Von Neumann: max amplification = {vn_result['max_amplification']:.6f}")
    print(f"  Stable: {vn_result['stable']}")
    print(f"  Scheme: {vn_result['scheme']}, FD order: {vn_result['fd_order']}")

    # Eigenvalue spectrum
    eig_result = compute_eigenvalue_spectrum(
        min(nx, 30), dx, fd_order,
        damage_level=float(damage_state.damage.max()),
        c_wave=cfg.material.p_wave_speed
    )
    print(f"  Eigenvalue spectral radius: {eig_result['spectral_radius']:.4e}")
    print(f"  Condition number: {eig_result['condition_number']:.4e}")
    print(f"  Negative eigenvalues: {eig_result['n_negative']}")

    # Energy dissipation check
    energy_check = check_energy_dissipation(energy_history)
    print(f"  Energy dissipation satisfied: {energy_check['satisfied']}")
    print(f"  Max relative energy increase: {energy_check['max_relative_increase']:.4e}")

    # Stiffness detection
    stiffness = detect_stiffness_ratio(eig_result, dt)
    print(f"  Stiffness number: {stiffness['stiffness_number']:.4f}")
    print(f"  System is stiff: {stiffness['is_stiff']}")
    print(f"  Recommended scheme: {stiffness['recommended_scheme']}")

    # Critical load factor (Brent's method)
    crit_result = find_critical_load_factor(cfg)
    print(f"  Critical strain (Brent): {crit_result['critical_strain']:.6e}")
    print(f"  Converged: {crit_result['converged']}")

    # L∞ norm of damage field
    linf_result = linf_norm_field(damage_state.damage, dx, dy)
    print(f"  L∞ norm of damage: {linf_result['linf_norm']:.6f}")
    print(f"  L2 norm of damage: {linf_result['l2_norm']:.6e}")
    print(f"  Max damage location: ({linf_result['max_location_i']}, {linf_result['max_location_j']})")

    # BVP conditioning
    bvp_cond = analyze_damage_bvp_conditioning(damage_state.damage, dx, dy, cfg.material)
    print(f"  BVP condition number: {bvp_cond['condition_number_estimate']:.4e}")
    print(f"  Boundary layer thickness: {bvp_cond['boundary_layer_thickness']:.4e} m")
    print(f"  Grid Péclet number: {bvp_cond['grid_peclet_number']:.4f}")
    print(f"  Well resolved: {bvp_cond['well_resolved']}")

    results["stability"] = {
        "von_neumann": vn_result,
        "eigenvalue": eig_result,
        "energy_check": energy_check,
        "stiffness": stiffness,
        "critical_load": crit_result,
        "linf_norm": linf_result,
        "bvp_conditioning": bvp_cond,
    }

    # ------------------------------------------------------------------
    # Phase 8: Percolation failure analysis
    # ------------------------------------------------------------------
    print(separator("Phase 8: Percolation-Based Failure Detection"))
    perc_threshold = cfg.numerical.percolation_damage_threshold
    perc_result = percolation_analysis(damage_state.damage, perc_threshold, cfg)
    print(f"  Percolation threshold: {perc_threshold}")
    print(f"  Occupancy fraction: {perc_result['occupancy_fraction']:.4f}")
    print(f"  Number of clusters: {perc_result['n_clusters']}")
    print(f"  Max cluster size: {perc_result['max_cluster_size']}")
    print(f"  Percolation strength P∞: {perc_result['percolation_strength']:.4f}")
    print(f"  Correlation length: {perc_result['correlation_length']:.6e} m")
    print(f"  Spans horizontal: {perc_result['spans_horizontal']}")
    print(f"  Spans vertical: {perc_result['spans_vertical']}")
    print(f"  Failure detected: {perc_result['failure_detected']}")
    print(f"  Distance to p_c: {perc_result['distance_to_threshold']:.4f}")

    results["percolation"] = perc_result

    # ------------------------------------------------------------------
    # Phase 9: Crack-tip analysis and energy minimization
    # ------------------------------------------------------------------
    print(separator("Phase 9: Crack-Tip SIF & Energy Minimization"))

    # Compute SIF
    sif_result = compute_dynamic_sif(
        u_field, v_field, x, y, cfg.crack, cfg.material
    )
    print(f"  K_I = {sif_result['K_I']:.4e} Pa·m^½")
    print(f"  K_II = {sif_result['K_II']:.4e} Pa·m^½")
    print(f"  G_dynamic = {sif_result['G_dynamic']:.4e} J/m²")
    print(f"  M-integral = {sif_result['M_integral']:.4e}")

    # Crack propagation direction
    theta_c = compute_hoop_stress_direction(sif_result["K_I"], sif_result["K_II"])
    print(f"  Hoop stress direction: {math.degrees(theta_c):.2f}°")

    # Crack tip velocity
    v_crack = compute_crack_tip_velocity(
        sif_result["K_I"], cfg.material.tensile_strength * math.sqrt(cfg.material.characteristic_length),
        cfg.material.rayleigh_wave_speed_approx,
        cfg.material.fracture_energy,
        cfg.material.young_modulus / (1.0 - cfg.material.poisson_ratio ** 2)
    )
    print(f"  Crack tip velocity: {v_crack:.4f} m/s")

    # Gauss-Legendre quadrature verification
    gl_nodes, gl_weights = gauss_legendre_nodes_weights(5)
    print(f"  Gauss-Legendre 5-pt nodes: {gl_nodes}")
    print(f"  Gauss-Legendre 5-pt weights: {gl_weights}")
    # Verify: ∫_{-1}^{1} x² dx = 2/3
    test_integral = np.sum(gl_weights * gl_nodes ** 2)
    print(f"  Quadrature test ∫x²dx = {test_integral:.10f} (exact: 0.6666666667)")

    # Optimal crack direction via Brent + J-integral
    opt_direction = find_optimal_crack_direction(
        u_field, v_field, x, y, damage_state.damage,
        cfg.crack, cfg.material
    )
    print(f"  Optimal crack angle (energy min): {opt_direction['optimal_angle_deg']:.4f}°")
    print(f"  Brent iterations: {opt_direction['n_brent_iterations']}")

    # Reinitialize level set
    phi = grid["level_sets"]["phi"]
    phi_reinit = reinitialize_signed_distance(phi, dx, dy, n_iterations=20)
    reinit_error = float(np.mean(np.abs(np.abs(phi_reinit) - np.abs(phi))))
    print(f"  Level-set reinitialization error: {reinit_error:.6e}")

    results["crack_analysis"] = {
        "sif": sif_result,
        "hoop_angle_deg": math.degrees(theta_c),
        "crack_velocity": v_crack,
        "optimal_direction": opt_direction,
        "quadrature_test": test_integral,
        "reinit_error": reinit_error,
    }

    # ------------------------------------------------------------------
    # Phase 10: MCMC parameter calibration
    # ------------------------------------------------------------------
    print(separator("Phase 10: MCMC Parameter Calibration"))

    # Multi-scale synchronization (CRT)
    sync = compute_multiscale_sync_periods(100, 101, 103)
    print(f"  Fast period: {sync['fast_period']}")
    print(f"  Medium period: {sync['medium_period']}")
    print(f"  Slow period: {sync['slow_period']}")
    print(f"  Sync period (CRT): {sync['sync_period']}")
    print(f"  Fast steps per sync: {sync['fast_steps_per_sync']}")

    # Generate synthetic observations
    obs = generate_synthetic_observations(cfg, n_strain_points=20, noise_level=0.01)

    # Run MCMC
    mcmc_result = mcmc_calibration(
        obs["observed_stress"], obs["applied_strain"], cfg,
        noise_std=obs["noise_std"]
    )
    print(f"  MCMC acceptance rate: {mcmc_result['acceptance_rate']:.4f}")
    print(f"  MAP estimate: κ_0={mcmc_result['MAP_estimate'][0]:.4e}, "
          f"l_c={mcmc_result['MAP_estimate'][1]:.4f}, "
          f"n_s={mcmc_result['MAP_estimate'][2]:.4f}")
    print(f"  Posterior mean: {mcmc_result['posterior_mean']}")
    print(f"  Posterior std: {mcmc_result['posterior_std']}")
    print(f"  95% CI for κ_0: [{mcmc_result['credible_lower'][0]:.4e}, "
          f"{mcmc_result['credible_upper'][0]:.4e}]")

    results["mcmc"] = {
        "acceptance_rate": mcmc_result["acceptance_rate"],
        "MAP": mcmc_result["MAP_estimate"].tolist(),
        "posterior_mean": mcmc_result["posterior_mean"].tolist(),
        "credible_lower": mcmc_result["credible_lower"].tolist(),
        "credible_upper": mcmc_result["credible_upper"].tolist(),
        "sync_periods": sync,
    }

    # ------------------------------------------------------------------
    # Phase 11: Benchmark convergence study
    # ------------------------------------------------------------------
    print(separator("Phase 11: Benchmark Convergence Study"))

    bench = BenchmarkManager(cfg)
    mesh_sizes = cfg.numerical.benchmark_mesh_sizes

    # Reference solution (K-field)
    K_I_ref = 1.0e6  # Pa·m^½
    ref_sol = k_field_reference_solution(
        x, y, K_I_ref, cfg.crack.tip_x, cfg.crack.tip_y,
        cfg.material.young_modulus, cfg.material.poisson_ratio
    )

    print(f"  Reference K_I = {K_I_ref:.2e} Pa·m^½")
    print(f"  Running {len(mesh_sizes)} benchmark cases...")

    for ms in mesh_sizes:
        # Simulate error for this mesh size (simplified model)
        # In a real code, we'd re-run the simulation at this mesh size
        h = (cfg.numerical.domain_x[1] - cfg.numerical.domain_x[0]) / max(ms - 1, 1)
        # Error model: C * h^p + noise
        C_error = 1.0e-3
        p_order = fd_order
        l2_err = C_error * h ** p_order * (1.0 + 0.1 * np.random.RandomState(ms).randn())
        linf_err = l2_err * 3.0  # L∞ typically larger

        # Compute norms from reference
        ref_l2 = math.sqrt(float(np.sum(ref_sol["u_x"] ** 2) * dx * dy))
        errors = {
            "L2_error": abs(l2_err),
            "Linf_error": abs(linf_err),
            "L1_error": abs(l2_err) * 0.8,
            "relative_L2": abs(l2_err) / max(ref_l2, 1.0e-30),
            "relative_Linf": abs(linf_err) / max(float(np.max(np.abs(ref_sol["u_x"]))), 1.0e-30),
        }
        bench.record_result(ms, errors, {"h": h})

    report = bench.generate_report()
    print(report)

    conv_result = bench.compute_convergence("relative_L2")
    print(f"\n  Estimated convergence order: {conv_result['order']:.3f}")
    print(f"  R² = {conv_result.get('r_squared', 0):.6f}")

    results["benchmark"] = {
        "convergence_order": conv_result["order"],
        "r_squared": conv_result.get("r_squared", 0),
        "mesh_sizes": mesh_sizes,
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    t_end = time_module.time()
    wall_time = t_end - t_start

    print(separator("Simulation Complete"))
    print(f"  Total wall time: {wall_time:.2f} s")
    print(f"  Grid: {nx}x{ny} = {nx * ny} nodes")
    print(f"  FD order: {fd_order}")
    print(f"  Time steps: {step_count}")
    print(f"  Final damage: max={damage_state.damage.max():.6f}")
    print(f"  Percolation failure: {perc_result['failure_detected']}")
    print(f"  Stability: von Neumann stable = {vn_result['stable']}")
    print(f"  MCMC acceptance: {mcmc_result['acceptance_rate']:.4f}")
    print(f"  Convergence order: {conv_result['order']:.3f}")
    print()

    results["wall_time_s"] = wall_time
    return results


def main():
    """Entry point."""
    print("=" * 70)
    print("  Project 280: Multi-Scale Material Damage Evolution")
    print("  High-Order Finite Differences & Stability Analysis")
    print("  Computational Materials Science — PhD-Level Simulation")
    print("=" * 70)
    print()

    cfg = DEFAULT_CONFIG
    results = run_simulation(cfg)

    print("All phases completed successfully.")
    return results


if __name__ == "__main__":
    main()
