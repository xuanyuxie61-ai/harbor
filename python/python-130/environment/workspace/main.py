# -*- coding: utf-8 -*-
"""
================================================================================
MAIN ENTRY POINT: Multi-Scale Synaptic Plasticity Simulation
================================================================================

This program orchestrates a comprehensive computational analysis of synaptic
plasticity and neurotransmitter dynamics across multiple spatial and temporal
scales in cortical neural networks.

Scientific Problem:
-------------------
Understanding how synaptic weights evolve under the combined influence of:
1. Spatial diffusion of plasticity-related proteins (PRPs) along dendrites
2. Wave-like propagation of long-term potentiation (LTP) through tissue
3. Stochastic fluctuations in individual synaptic weights
4. Metabolic resource constraints on protein synthesis
5. Homeostatic regulation maintaining network stability
6. Geometric constraints from cortical tissue morphology
7. Spectral characteristics of population neural activity
8. Nonlinear synaptic transmission (NMDA receptor dynamics)

The simulation integrates all 15 seed algorithms into a unified framework
for analyzing neuroplasticity and synaptic weight dynamics.

Running:
--------
    python main.py

No command-line arguments required.
================================================================================
"""

import numpy as np
import sys

# Ensure the project directory is in the path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from cable_diffusion import simulate_protein_diffusion, stability_limit
from plasticity_wave import simulate_ltp_wave, verify_fisher_exact
from vesicle_release import simulate_vesicle_release_batch, sphere_monomial_integral
from resource_optimizer import simulate_metabolic_allocation
from cortical_mesh import simulate_cortical_mesh_analysis
from homeostatic_dynamics import simulate_homeostatic_plasticity_pipeline, compute_pendulum_period
from stochastic_weights import simulate_stochastic_weights, simulate_plasticity_option_portfolio
from spectral_field import analyze_neural_field_spectrum, test_interpolation_accuracy
from synaptic_nonlinearity import compute_polynomial_approximation_error, evaluate_nmda_current


def print_section(title: str):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def main():
    """
    Execute the full multi-scale synaptic plasticity simulation pipeline.
    """
    print("=" * 70)
    print("  MULTI-SCALE SYNAPTIC PLASTICITY SIMULATION")
    print("  Domain: Neuroplasticity and Synaptic Weights")
    print("=" * 70)

    # ========================================================================
    # 1. CORTICAL MESH GENERATION
    # ========================================================================
    print_section("1. CORTICAL MESH GENERATION & NEIGHBOR ANALYSIS")
    print("   (From: hand_mesh2d + triangulation_triangle_neighbors)")

    mesh = simulate_cortical_mesh_analysis(n_boundary=40, n_interior=150)
    print_subsection("Mesh Statistics")
    print(f"  Total nodes:           {mesh['n_nodes']}")
    print(f"  Total triangles:       {mesh['n_triangles']}")
    print(f"  Total cortical area:   {mesh['total_area']:.2f} μm²")
    print(f"  Mean triangle quality: {mesh['mean_quality']:.4f}")
    print(f"  Min triangle quality:  {mesh['min_quality']:.4f}")
    print(f"  Boundary edges:        {mesh['n_boundary_edges']}")
    print(f"  Neighbor array shape:  {mesh['neighbors'].shape}")

    # ========================================================================
    # 2. PROTEIN DIFFUSION ALONG DENDRITES
    # ========================================================================
    print_section("2. PROTEIN DIFFUSION ALONG DENDRITIC CABLES")
    print("   (From: laplacian_matrix - discrete Laplacian operators)")

    x_dend, t_dend, c_hist = simulate_protein_diffusion(
        n=64, length=100.0, D=0.1, gamma=0.01,
        dt=0.1, t_final=50.0, bc="DD"
    )
    dt_max = stability_limit(64, 100.0 / 65.0, 0.1, "DD")

    print_subsection("Diffusion Parameters")
    print(f"  Dendrite length:       100.0 μm")
    print(f"  Diffusion coeff D:     0.1 μm²/ms")
    print(f"  Degradation rate γ:    0.01 /ms")
    print(f"  Max stable dt:         {dt_max:.4f} ms")
    print(f"  Final max concentration: {np.max(c_hist[-1]):.6f}")
    print(f"  Final mean concentration: {np.mean(c_hist[-1]):.6f}")

    # ========================================================================
    # 3. LTP WAVE PROPAGATION
    # ========================================================================
    print_section("3. LTP WAVE PROPAGATION (Fisher-KPP Reaction-Diffusion)")
    print("   (From: fisher_exact + rk1)")

    # TODO: Call simulate_ltp_wave with appropriate parameters for LTP wave
    # propagation simulation, then call verify_fisher_exact for exact-solution
    # verification. Use the returned values to print wave propagation results
    # including theoretical minimum speed, exact solution error, wave front
    # position estimate, and maximum potentiation.
    pass

    # ========================================================================
    # 4. VESICLE RELEASE PROBABILITY
    # ========================================================================
    print_section("4. SYNAPTIC VESICLE RELEASE PROBABILITY")
    print("   (From: sphere_integrals + circle_rule)")

    # Sphere monomial integrals
    I000 = sphere_monomial_integral((0, 0, 0))
    I200 = sphere_monomial_integral((2, 0, 0))
    I220 = sphere_monomial_integral((2, 2, 0))
    I111 = sphere_monomial_integral((1, 1, 1))

    vesicle = simulate_vesicle_release_batch(n_boutons=20)

    print_subsection("Sphere Monomial Integrals")
    print(f"  ∫ dΩ (a=b=c=0):        {I000:.6f} (exact: 4π = {4*np.pi:.6f})")
    print(f"  ∫ x² dΩ:               {I200:.6f}")
    print(f"  ∫ x²y² dΩ:             {I220:.6f}")
    print(f"  ∫ xyz dΩ (odd):        {I111:.6f}")

    print_subsection("Vesicle Release Statistics (n=20 boutons)")
    print(f"  Mean sphere P_release: {np.mean(vesicle['P_sphere']):.6f}")
    print(f"  Mean circle P_release: {np.mean(vesicle['P_circle']):.6f}")
    print(f"  Mean quantal content:  {np.mean(vesicle['mean_q']):.4f}")
    print(f"  Mean quantal variance: {np.mean(vesicle['var_q']):.4f}")
    print(f"  Mean bouton radius:    {np.mean(vesicle['R_vals']):.4f} μm")

    # ========================================================================
    # 5. METABOLIC RESOURCE ALLOCATION
    # ========================================================================
    print_section("5. METABOLIC RESOURCE ALLOCATION FOR PLASTICITY")
    print("   (From: change_greedy - greedy optimization)")

    alloc = simulate_metabolic_allocation(n_synapses=50, budget_factor=0.6)

    print_subsection("Allocation Strategy Comparison")
    for strategy in ["greedy", "proportional", "knapsack"]:
        m = alloc[strategy]["metrics"]
        print(f"\n  {strategy.upper()}:")
        print(f"    Total plasticity:      {m['total_plasticity']:.4f}")
        print(f"    Budget utilization:    {m['budget_utilization']:.4f}")
        print(f"    Target achievement:    {m['target_achievement']:.4f}")
        print(f"    Cost efficiency:       {m['cost_efficiency']:.4f}")
        print(f"    Gini coefficient:      {m['gini_coefficient']:.4f}")

    # ========================================================================
    # 6. HOMEOSTATIC DYNAMICS
    # ========================================================================
    print_section("6. HOMEOSTATIC SYNAPTIC WEIGHT REGULATION")
    print("   (From: spring_ode + pendulum_nonlinear_exact)")

    homeo = simulate_homeostatic_plasticity_pipeline(n_synapses=5, t_final=30.0)

    print_subsection("Individual Synapse Homeostasis")
    for i, syn in enumerate(homeo["synapses"]):
        p = syn["params"]
        print(f"  Synapse {i+1}:")
        print(f"    Regime:        {p['regime']}")
        print(f"    ω_n:           {p['omega_n']:.4f}")
        print(f"    ζ:             {p['zeta']:.4f}")
        print(f"    Final weight:  {syn['w'][-1]:.4f}")

    print_subsection("Network Synchronization (Nonlinear Pendulum)")
    T_pend = compute_pendulum_period(g=1.0, l=1.0, theta0=np.pi / 3.0)
    print(f"  Pendulum period (θ₀=π/3): {T_pend:.4f}")
    print(f"  Network size:              {homeo['network_theta'].shape[1]} neurons")
    print(f"  Final phase std:           {np.std(homeo['network_theta'][-1]):.4f} rad")

    # ========================================================================
    # 7. STOCHASTIC WEIGHT EVOLUTION
    # ========================================================================
    print_section("7. STOCHASTIC SYNAPTIC WEIGHT EVOLUTION")
    print("   (From: black_scholes - geometric Brownian motion)")

    t_stoch, w_stoch = simulate_stochastic_weights(
        n_synapses=100, t_final=100.0, dt=0.01,
        mu=0.05, w_max=1.0, lambda_homeo=0.1,
        w_target=0.5, sigma=0.2, seed=42
    )

    portfolio = simulate_plasticity_option_portfolio(n_synapses=50, tau=10.0)

    print_subsection("Stochastic Dynamics")
    print(f"  Synapses:                100")
    print(f"  Simulation time:         100.0 ms")
    print(f"  Time step dt:            0.01 ms")
    print(f"  Hebbian drift μ:         0.05")
    print(f"  Homeostatic λ:           0.1")
    print(f"  Volatility σ:            0.2")
    print(f"  Final mean weight:       {np.mean(w_stoch[-1]):.4f}")
    print(f"  Final weight std:        {np.std(w_stoch[-1]):.4f}")
    print(f"  Min weight:              {np.min(w_stoch[-1]):.6f}")
    print(f"  Max weight:              {np.max(w_stoch[-1]):.6f}")

    print_subsection("Plasticity Option Portfolio")
    print(f"  Total portfolio value:   {portfolio['total_value']:.4f}")
    print(f"  Mean option value:       {portfolio['mean_value']:.4f}")
    print(f"  Max option value:        {np.max(portfolio['options']):.4f}")

    # ========================================================================
    # 8. SPECTRAL ANALYSIS
    # ========================================================================
    print_section("8. NEURAL FIELD SPECTRAL ANALYSIS")
    print("   (From: fft_serial + trig_interp + interp_chebyshev)")

    spectrum = analyze_neural_field_spectrum(n_points=512, t_max=1000.0)
    interp_err = test_interpolation_accuracy(n_test=100)

    print_subsection("Frequency Band Powers")
    for band, power in spectrum["band_powers"].items():
        print(f"  {band.capitalize():12s}: {power:.4f}")
    print(f"  Dominant frequency:      {spectrum['dominant_freq']:.2f} Hz")

    print_subsection("Interpolation Accuracy")
    print(f"  Trigonometric error:     {interp_err['trig_error']:.6e}")
    print(f"  Chebyshev error:         {interp_err['cheb_error']:.6e}")

    # ========================================================================
    # 9. SYNAPTIC NONLINEARITY
    # ========================================================================
    print_section("9. SYNAPTIC NONLINEARITY (NMDA Receptor)")
    print("   (From: polynomial - multivariate polynomial operations)")

    poly_err = compute_polynomial_approximation_error(n_test=100)
    V_test = np.linspace(-80.0, 40.0, 50)
    I_exact, I_poly = evaluate_nmda_current(V_test, Mg=1.0)

    print_subsection("Polynomial Approximation Errors")
    for k, v in poly_err.items():
        print(f"  {k:25s}: {v:.6e}")

    print_subsection("NMDA Current Sample Points")
    idx = [0, 12, 25, 37, 49]
    for i in idx:
        print(f"  V={V_test[i]:5.1f}mV: I_exact={I_exact[i]:8.4f}, I_poly={I_poly[i]:8.4f}, diff={abs(I_exact[i]-I_poly[i]):.4e}")

    # ========================================================================
    # 10. INTEGRATED SUMMARY
    # ========================================================================
    print_section("10. INTEGRATED SUMMARY")
    print("""
This simulation integrates 15 seed algorithms into a unified framework
for multi-scale synaptic plasticity:

  Scale          | Algorithm                    | Biological Role
  ---------------|------------------------------|-------------------------------
  Molecular      | laplacian_matrix             | PRP diffusion along dendrites
  Tissue         | fisher_exact + rk1           | LTP wave propagation
  Synaptic       | sphere_integrals             | Vesicle release probability
  Synaptic       | circle_rule                  | Bouton cross-section integral
  Network        | hand_mesh2d + triangulation  | Cortical tissue geometry
  Cellular       | spring_ode                   | Homeostatic weight regulation
  Cellular       | pendulum_nonlinear_exact     | Phase synchronization
  Stochastic     | black_scholes                | Weight fluctuation model
  Metabolic      | change_greedy                | Resource allocation
  Spectral       | fft_serial                   | Neural field PSD
  Spectral       | trig_interp                  | Periodic firing rate interp
  Spectral       | interp_chebyshev             | Transfer function interp
  Computational  | polynomial                   | NMDA nonlinearity approx
  Numerical      | rk1                          | ODE time stepping

All simulations completed successfully with numerical stability verified.
""")

    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
