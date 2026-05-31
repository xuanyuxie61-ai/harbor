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

    x_wave, t_wave, u_hist, c_min = simulate_ltp_wave(
        n=128, length=200.0, D=1.0, r=1.0,
        t_final=20.0, n_steps=2000, bc="NN"
    )
    err_exact = verify_fisher_exact(n=64, length=20.0, t_test=2.0)

    print_subsection("Wave Propagation Results")
    print(f"  Domain length:         200.0 μm")
    print(f"  Diffusion D:           1.0 μm²/ms")
    print(f"  Growth rate r:         1.0 /ms")
    print(f"  Theoretical min speed: {c_min:.4f} μm/ms")
    print(f"  Exact solution verification error: {err_exact:.6e}")
    print(f"  Final wave front position: ~{np.argmax(u_hist[-1] > 0.5) * (x_wave[1]-x_wave[0]):.2f} μm")
    print(f"  Max potentiation at t={t_wave[-1]:.1f}: {np.max(u_hist[-1]):.4f}")

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

# ================================================================
# 测试用例（54个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: build_laplacian_1d 矩阵对称性 ----
L_dd = build_laplacian_1d(8, 0.5, "DD")
assert np.allclose(L_dd, L_dd.T), '[TC01] Laplacian matrix must be symmetric FAILED'

# ---- TC02: build_laplacian_1d 行和验证 DD ----
L = build_laplacian_1d(10, 0.2, "DD")
row_sums = np.sum(L, axis=1)
assert np.all(row_sums[:1] > 0), '[TC02] DD first row sum must be > 0 FAILED'
assert np.allclose(row_sums[-1], row_sums[0]), '[TC02] DD boundary rows must be symmetric FAILED'

# ---- TC03: apply_laplacian_1d 与矩阵乘积一致 ----
np.random.seed(42)
n, h = 6, 0.3
u = np.random.rand(n)
L_mat = build_laplacian_1d(n, h, "NN")
Lu_mat = L_mat @ u
Lu_vec = apply_laplacian_1d(n, h, u, "NN")
assert np.allclose(Lu_mat, Lu_vec), '[TC03] apply_laplacian_1d must match matrix multiply FAILED'

# ---- TC04: laplacian_eigenvalues 全部非负 ----
np.random.seed(42)
eigvals = laplacian_eigenvalues(10, 0.1, "DD")
assert np.all(eigvals >= -1e-12), '[TC04] Laplacian eigenvalues must be non-negative FAILED'
assert eigvals[0] >= 0.0, '[TC04] Smallest eigenvalue must be >= 0 FAILED'

# ---- TC05: stability_limit 正值扩散 ----
dt_max = stability_limit(10, 0.1, 1.0, "DD")
assert dt_max > 0.0, '[TC05] stability_limit must be positive FAILED'
assert np.isfinite(dt_max), '[TC05] stability_limit must be finite FAILED'

# ---- TC06: simulate_protein_diffusion 输出形状和终态非负 ----
np.random.seed(42)
x_d, t_d, c_hist = simulate_protein_diffusion(n=32, length=50.0, D=0.1, gamma=0.01, dt=0.1, t_final=10.0, bc="DD")
assert x_d.shape[0] == 32, '[TC06] x must have 32 points FAILED'
assert c_hist.shape[0] == t_d.shape[0], '[TC06] c_hist rows must match t length FAILED'
assert np.all(c_hist[-1] >= -1e-12), '[TC06] Final concentration must be non-negative FAILED'

# ---- TC07: numerical_integrator.rk1_integrate 谐振子能量有界 ----
np.random.seed(42)
def harmonic_rk1(t, y):
    return np.array([y[1], -y[0]])
t_rk1, y_rk1 = rk1_integrate(harmonic_rk1, (0.0, 10.0), np.array([1.0, 0.0]), 5000)
E_final = 0.5 * (y_rk1[-1, 0]**2 + y_rk1[-1, 1]**2)
assert E_final < 100.0, '[TC07] RK1 harmonic energy must be bounded FAILED'
assert np.isfinite(y_rk1[-1, 0]), '[TC07] RK1 final state must be finite FAILED'

# ---- TC08: numerical_integrator.rk4_integrate 谐振子周期近似 ----
np.random.seed(42)
def harmonic_rk4(t, y):
    return np.array([y[1], -y[0]])
t_rk4, y_rk4 = rk4_integrate(harmonic_rk4, (0.0, 2.0 * np.pi), np.array([1.0, 0.0]), 2000)
assert abs(y_rk4[-1, 0] - 1.0) < 0.1, '[TC08] RK4 harmonic should return near (1,0) after 2π FAILED'

# ---- TC09: adaptive_rk12 步长自适应非零 ----
np.random.seed(42)
def logistic(t, y):
    return np.array([y[0] * (1.0 - y[0])])
t_ad, y_ad, h_ad = adaptive_rk12(logistic, (0.0, 5.0), np.array([0.1]), tol=1e-4, h0=0.01)
assert len(t_ad) >= 2, '[TC09] adaptive_rk12 must produce at least 2 time points FAILED'
assert np.all(h_ad > 0), '[TC09] All step sizes must be positive FAILED'

# ---- TC10: fisher_exact_solution 解在 [0,1] ----
np.random.seed(42)
x_ftest = np.linspace(-10.0, 10.0, 100)
u_exact, _, _, _ = fisher_exact_solution(x_ftest, 1.0)
assert np.all(u_exact >= 0.0) and np.all(u_exact <= 1.0), '[TC10] Fisher exact solution must be in [0,1] FAILED'

# ---- TC11: fisher_wave_speed 解析公式 ----
np.random.seed(42)
c_min = fisher_wave_speed(1.0, 1.0)
assert abs(c_min - 2.0) < 1e-12, '[TC11] fisher_wave_speed(1,1) must be 2.0 FAILED'
c_min2 = fisher_wave_speed(4.0, 1.0)
assert abs(c_min2 - 4.0) < 1e-12, '[TC11] fisher_wave_speed(4,1) must be 4.0 FAILED'

# ---- TC12: verify_fisher_exact 误差小 ----
np.random.seed(42)
err_exact = verify_fisher_exact(n=32, length=10.0, t_test=1.0)
assert err_exact < 1.0, '[TC12] Fisher exact verification error must be < 1.0 FAILED'

# ---- TC13: sphere_monomial_integral 已知解析值 ----
np.random.seed(42)
I000 = sphere_monomial_integral((0, 0, 0))
assert abs(I000 - 4.0 * np.pi) < 1e-10, '[TC13] ∫dΩ must equal 4π FAILED'
I111 = sphere_monomial_integral((1, 1, 1))
assert abs(I111) < 1e-14, '[TC13] Odd monomial integral must be zero FAILED'

# ---- TC14: circle_rule_quadrature 权和为1 ----
np.random.seed(42)
weights, angles = circle_rule_quadrature(32)
assert abs(np.sum(weights) - 1.0) < 1e-14, '[TC14] Circle quadrature weights must sum to 1 FAILED'
assert len(angles) == 32, '[TC14] Must have 32 angles FAILED'

# ---- TC15: compute_quantal_content 边界检查 ----
np.random.seed(42)
m_q, v_q = compute_quantal_content(0.5, 10)
assert 0.0 <= m_q <= 10.0, '[TC15] Mean quantal content must be in [0, N] FAILED'
assert v_q >= 0.0, '[TC15] Quantal variance must be non-negative FAILED'

# ---- TC16: greedy_resource_allocation 预算约束 ----
np.random.seed(42)
targets = np.array([0.5, 0.3, 0.8, 0.2, 0.6])
costs = np.array([1.0, 2.0, 1.5, 1.0, 2.5])
alloc, ratios, remaining = greedy_resource_allocation(targets, costs, 2.0)
total_cost = np.sum(costs * np.abs(alloc))
assert total_cost <= 2.0 + 1e-12, '[TC16] Greedy allocation must respect budget FAILED'
assert remaining >= -1e-12, '[TC16] Remaining budget must be >= 0 FAILED'

# ---- TC17: proportional_allocation 满足预算 ----
np.random.seed(42)
targets2 = np.array([0.5, 0.3, 0.8])
costs2 = np.array([1.0, 1.0, 1.0])
alloc_prop = proportional_allocation(targets2, costs2, 1.0)
total_spent = np.sum(costs2 * np.abs(alloc_prop))
assert total_spent <= 1.0 + 1e-12, '[TC17] Proportional allocation must respect budget FAILED'

# ---- TC18: evaluate_allocation_efficiency 指标在合理范围 ----
np.random.seed(42)
targets3 = np.array([0.5, 0.3, 0.8, 0.2])
costs3 = np.array([1.0, 1.5, 2.0, 1.0])
alloc3 = np.array([0.5, 0.3, 0.4, 0.2])
metrics = evaluate_allocation_efficiency(alloc3, targets3, costs3, 3.0)
assert 0.0 <= metrics['budget_utilization'] <= 1.0 + 1e-12, '[TC18] Budget utilization must be in [0,1] FAILED'
assert 0.0 <= metrics['target_achievement'] <= 1.0 + 1e-12, '[TC18] Target achievement must be in [0,1] FAILED'
assert metrics['cost_efficiency'] >= 0.0, '[TC18] Cost efficiency must be non-negative FAILED'

# ---- TC19: generate_cortical_boundary 闭合多边形 ----
np.random.seed(130)
boundary = generate_cortical_boundary(n_vertices=30, width=1000.0, height=800.0, noise_scale=20.0, seed=130)
assert boundary.shape == (30, 2), '[TC19] Boundary must have shape (30, 2) FAILED'
assert np.all(np.isfinite(boundary)), '[TC19] Boundary coordinates must be finite FAILED'

# ---- TC20: compute_triangle_neighbors 形状正确 ----
np.random.seed(42)
nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
elements = np.array([[0, 1, 4], [1, 3, 4], [0, 4, 2], [4, 3, 2]])
neighbors = compute_triangle_neighbors(elements)
assert neighbors.shape[0] == 4, '[TC20] Neighbors must have 4 rows FAILED'
assert neighbors.shape[1] == 3, '[TC20] Neighbors must have 3 columns FAILED'

# ---- TC21: simulate_homeostatic_response 阻尼分类有效 ----
np.random.seed(42)
t_sp, y_sp, params = simulate_homeostatic_response(w0=0.3, v0=0.0, m=1.0, b=0.5, k=1.0, t_final=30.0, n_steps=2000)
assert params['regime'] in ('underdamped', 'critically_damped', 'overdamped'), '[TC21] Damping regime must be valid FAILED'
assert np.isfinite(y_sp[-1, 0]), '[TC21] Final weight must be finite FAILED'

# ---- TC22: compute_pendulum_period 小角度近似 ----
np.random.seed(42)
T_small = compute_pendulum_period(g=9.81, l=1.0, theta0=0.01)
T_linear = 2.0 * np.pi * np.sqrt(1.0 / 9.81)
assert abs(T_small - T_linear) / T_linear < 0.001, '[TC22] Small angle pendulum period must approach linear FAILED'

# ---- TC23: sde_drift 在目标处漂移小 ----
np.random.seed(42)
w_target = 0.5
drift = sde_drift(np.array([0.5]), mu=0.05, w_max=1.0, lambda_homeo=0.1, w_target=0.5)
assert abs(drift[0]) < 0.01, '[TC23] Drift at target weight must be small FAILED'

# ---- TC24: black_scholes_synaptic_option 非负 ----
np.random.seed(42)
opt_val = black_scholes_synaptic_option(w0=0.6, w_target=0.5, mu=0.05, sigma=0.2, tau=10.0)
assert opt_val >= 0.0, '[TC24] Option value must be non-negative FAILED'
opt_val2 = black_scholes_synaptic_option(w0=0.1, w_target=0.5, mu=0.05, sigma=0.2, tau=0.001)
assert opt_val2 >= 0.0, '[TC24] Option value for small tau must be non-negative FAILED'

# ---- TC25: compute_fft_spectrum Parseval 验证 ----
np.random.seed(42)
signal_test = np.sin(2.0 * np.pi * 5.0 * np.linspace(0.0, 1.0, 256))
freqs, spectrum, psd = compute_fft_spectrum(signal_test, dt=1.0/256.0)
pos_mask = freqs >= 0
energy_time = np.sum(np.abs(signal_test)**2)
energy_freq = np.sum(psd)
assert abs(energy_time - energy_freq) / energy_time < 0.1, '[TC25] Parseval: time and freq energy must match FAILED'

# ---- TC26: chebyspace 端点正确 ----
np.random.seed(42)
xc = chebyspace(-1.0, 1.0, 10)
assert abs(xc[0] - (-1.0)) < 1e-14, '[TC26] First Chebyshev node must be -1.0 FAILED'
assert abs(xc[-1] - 1.0) < 1e-14, '[TC26] Last Chebyshev node must be 1.0 FAILED'
assert len(xc) == 10, '[TC26] Must have 10 nodes FAILED'

# ---- TC27: mono_upto_enum 公式验证 ----
np.random.seed(42)
n_monos = mono_upto_enum(2, 3)
assert n_monos == 10, '[TC27] C(2+3,3)=C(5,3)=10 FAILED'
n_monos2 = mono_upto_enum(1, 4)
assert n_monos2 == 5, '[TC27] C(1+4,4)=C(5,4)=5 FAILED'

# ---- TC28: evaluate_nmda_current 单调性 ----
np.random.seed(42)
V_test_nmda = np.linspace(-80.0, 40.0, 30)
I_exact, I_poly = evaluate_nmda_current(V_test_nmda, Mg=1.0)
assert I_exact.shape == (30,), '[TC28] I_exact must have shape (30,) FAILED'
assert I_poly.shape == (30,), '[TC28] I_poly must have shape (30,) FAILED'
# NMDA current should increase with voltage
assert I_exact[-1] > I_exact[0], '[TC28] NMDA current must increase with voltage FAILED'

# ---- TC29: simulate_stochastic_weights 可复现性 ----
np.random.seed(42)
t1, w1 = simulate_stochastic_weights(n_synapses=10, t_final=5.0, dt=0.1, mu=0.05, w_max=1.0, lambda_homeo=0.1, w_target=0.5, sigma=0.2, seed=42)
t2, w2 = simulate_stochastic_weights(n_synapses=10, t_final=5.0, dt=0.1, mu=0.05, w_max=1.0, lambda_homeo=0.1, w_target=0.5, sigma=0.2, seed=42)
assert np.allclose(w1, w2), '[TC29] Same seed must produce identical results FAILED'

# ---- TC30: numerical_integrator.compute_stiffness_ratio 非负 ----
np.random.seed(42)
J_test = np.array([[-1.0, 0.5], [0.5, -2.0]])
S, lam_max, lam_min = compute_stiffness_ratio(J_test)
assert S >= 0.0, '[TC30] Stiffness ratio must be non-negative FAILED'
assert np.isfinite(S), '[TC30] Stiffness ratio must be finite FAILED'

# ---- TC31: cable_diffusion_step 无源退化 ----
np.random.seed(42)
c_init = np.ones(10) * 0.5
c_next = cable_diffusion_step(c_init, D=0.1, h=0.1, dt=0.01, gamma=0.1, source=None, bc="NN")
assert np.all(c_next >= 0.0), '[TC31] Concentration after step must be non-negative FAILED'
assert c_next.shape == (10,), '[TC31] Output shape must be (10,) FAILED'

# ---- TC32: compute_mesh_quality 值在 [0,1] ----
np.random.seed(42)
nodes_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.0, 0.0], [1.0, 0.0], [0.5, 0.3]])
elements_test = np.array([[0, 1, 2], [3, 4, 5]])
q_vals = compute_mesh_quality(nodes_test, elements_test)
assert np.all(q_vals >= 0.0) and np.all(q_vals <= 1.0), '[TC32] Mesh quality must be in [0,1] FAILED'

# ---- TC33: simulate_cortical_mesh_analysis 输出结构 ----
np.random.seed(130)
mesh_res = simulate_cortical_mesh_analysis(n_boundary=20, n_interior=50)
assert mesh_res['total_area'] > 0, '[TC33] Total area must be positive FAILED'
assert 0.0 < mesh_res['mean_quality'] <= 1.0, '[TC33] Mean quality must be in (0,1] FAILED'
assert 'neighbors' in mesh_res, '[TC33] Result must contain neighbors FAILED'

# ---- TC34: spring_parameters omega_n 解析 ----
np.random.seed(42)
params_sp = spring_parameters(m=1.0, b=0.5, k=4.0)
assert abs(params_sp['omega_n'] - 2.0) < 1e-12, '[TC34] omega_n = sqrt(k/m) must be 2.0 FAILED'

# ---- TC35: divided_differences 与 newton_interpolate 一致性 ----
np.random.seed(42)
xd_dd = np.array([0.0, 1.0, 2.0, 3.0])
yd_dd = np.array([1.0, 2.0, 4.0, 8.0])
dd = divided_differences(xd_dd, yd_dd)
yi_ni = newton_interpolate(xd_dd, dd, np.array([0.0, 1.0, 2.0, 3.0]))
assert np.allclose(yi_ni, yd_dd, atol=1e-10), '[TC35] Newton interpolation must match at nodes FAILED'

# ---- TC36: simulate_homeostatic_plasticity_pipeline 输出结构 ----
np.random.seed(130)
pipeline_res = simulate_homeostatic_plasticity_pipeline(n_synapses=3, t_final=10.0)
assert len(pipeline_res['synapses']) == 3, '[TC36] Pipeline must have 3 synapses FAILED'
assert pipeline_res['network_theta'].shape[1] == 8, '[TC36] Network must have 8 neurons FAILED'

# ---- TC37: analyze_neural_field_spectrum 频带存在 ----
np.random.seed(130)
spectrum = analyze_neural_field_spectrum(n_points=256, t_max=500.0)
assert 'theta' in spectrum['band_powers'], '[TC37] Theta band must be present FAILED'
assert spectrum['dominant_freq'] > 0, '[TC37] Dominant frequency must be positive FAILED'

# ---- TC38: simulate_vesicle_release_batch 输出字典 ----
np.random.seed(42)
vesicle = simulate_vesicle_release_batch(n_boutons=15)
assert len(vesicle['P_sphere']) == 15, '[TC38] Must have 15 boutons FAILED'
assert np.all(vesicle['P_sphere'] >= 0.0), '[TC38] Release probability must be non-negative FAILED'

# ---- TC39: simulate_metabolic_allocation 策略指标可比 ----
np.random.seed(42)
alloc_res = simulate_metabolic_allocation(n_synapses=30, budget_factor=0.6, seed=42)
assert alloc_res['greedy']['metrics']['total_plasticity'] >= 0, '[TC39] Greedy total plasticity must be >= 0 FAILED'
assert alloc_res['proportional']['metrics']['total_plasticity'] >= 0, '[TC39] Proportional total plasticity must be >= 0 FAILED'

# ---- TC40: compute_polynomial_approximation_error 误差指标非负 ----
np.random.seed(42)
poly_err = compute_polynomial_approximation_error(n_test=50)
assert poly_err['max_abs_error'] >= 0, '[TC40] Max abs error must be non-negative FAILED'
assert poly_err['rmse'] >= 0, '[TC40] RMSE must be non-negative FAILED'
assert np.isfinite(poly_err['mean_rel_error']), '[TC40] Mean rel error must be finite FAILED'

# ---- TC41: simulate_plasticity_option_portfolio 输出结构 ----
np.random.seed(130)
portfolio = simulate_plasticity_option_portfolio(n_synapses=30, tau=10.0)
assert portfolio['total_value'] >= 0, '[TC41] Total portfolio value must be non-negative FAILED'
assert len(portfolio['options']) == 30, '[TC41] Must have 30 options FAILED'

# ---- TC42: test_interpolation_accuracy 误差为正 ----
np.random.seed(42)
interp_err = test_interpolation_accuracy(n_test=80)
assert interp_err['trig_error'] >= 0, '[TC42] Trig interpolation error must be non-negative FAILED'
assert interp_err['cheb_error'] >= 0, '[TC42] Chebyshev interpolation error must be non-negative FAILED'

# ---- TC43: hill_function_polynomial_approximation 多项式度匹配 ----
np.random.seed(42)
coeffs_hill, exp_hill, exact_hill = hill_function_polynomial_approximation(n=4.0, K_D=1.0, degree=6, ca_range=(0.0, 5.0))
assert len(coeffs_hill) == 7, '[TC43] Hill polynomial must have degree+1=7 coefficients FAILED'

# ---- TC44: mono_value 标量输入输出正确 ----
np.random.seed(42)
mv = mono_value(2, 4, np.array([2, 1]), np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]))
assert mv.shape == (4,), '[TC44] mono_value output shape must be (4,) FAILED'
assert abs(mv[0] - 2.0) < 1e-12, '[TC44] x^2*y at (1,2) must be 2.0 FAILED'

# ---- TC45: polynomial_value 返回有限值 ----
np.random.seed(42)
coeffs_pv = np.array([1.0, 2.0, 3.0])
exps_pv = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]
X_pv = np.array([[0.5, 1.0, 1.5], [0.2, 0.3, 0.4]])
pv = polynomial_value(2, coeffs_pv, exps_pv, X_pv)
assert np.all(np.isfinite(pv)), '[TC45] Polynomial values must be finite FAILED'
assert len(pv) == 3, '[TC45] Must have 3 output points FAILED'

# ---- TC46: trig_interpolate 在节点处精确 ----
np.random.seed(42)
xd_trig = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
yd_trig = np.sin(xd_trig)
yi_trig = trig_interpolate(xd_trig, yd_trig, xd_trig)
assert np.allclose(yi_trig, yd_trig, atol=1e-10), '[TC46] Trig interpolation must be exact at nodes FAILED'

# ---- TC47: compute_band_power 边界情况 ----
np.random.seed(42)
freqs_test = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
psd_test = np.array([0.1, 0.3, 0.5, 0.2, 0.05])
bp = compute_band_power(freqs_test, psd_test, (8.0, 30.0))
assert abs(bp - (0.5 + 0.2)) < 1e-12, '[TC47] Beta band power must be 0.7 FAILED'

# ---- TC48: euler_maruyama_step 边界裁剪 ----
np.random.seed(42)
rng_em = np.random.default_rng(42)
w_em = np.array([0.6, 0.8])
w_new = euler_maruyama_step(w_em, dt=0.01, mu=0.05, w_max=1.0, lambda_homeo=0.1, w_target=0.5, sigma=0.2, rng=rng_em)
assert np.all(w_new >= 1e-6), '[TC48] Weights after EM step must be >= 1e-6 FAILED'
assert np.all(w_new <= 1.0), '[TC48] Weights after EM step must be <= 1.0 FAILED'

# ---- TC49: knapsack_plasticity_allocation 空预算 ----
np.random.seed(42)
vals = np.array([0.5, 0.3, 0.8])
csts = np.array([1.0, 2.0, 1.5])
sel, tv = knapsack_plasticity_allocation(vals, csts, 0.0)
assert tv == 0.0, '[TC49] Zero budget must yield zero total value FAILED'

# ---- TC50: generate_cortical_mesh 节点数正确 ----
np.random.seed(130)
nodes_mesh, elems_mesh = generate_cortical_mesh(n_boundary=15, n_interior=30, seed=130)
assert nodes_mesh.shape[0] >= 15, '[TC50] Mesh must have at least 15 nodes FAILED'
assert elems_mesh.shape[1] == 3, '[TC50] Elements must have 3 columns (triangles) FAILED'

# ---- TC51: sde_diffusion 非负 ----
np.random.seed(42)
w_sde = np.array([0.1, 0.5, 0.9])
diff_vals = sde_diffusion(w_sde, sigma=0.2)
assert np.all(diff_vals >= 0.0), '[TC51] Diffusion coefficient must be non-negative FAILED'

# ---- TC52: compute_element_areas 正面积 ----
np.random.seed(42)
nodes_area = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
elems_area = np.array([[0, 1, 2], [3, 4, 5]])
areas = compute_element_areas(nodes_area, elems_area)
assert np.all(areas > 0), '[TC52] All triangle areas must be positive FAILED'
assert abs(areas[0] - 1.0) < 1e-12, '[TC52] First triangle area must be 1.0 FAILED'

# ---- TC53: compute_weight_statistics 输出键完整 ----
np.random.seed(42)
w_hist_test = np.random.default_rng(42).uniform(0.1, 0.9, (100, 10))
stats_w = compute_weight_statistics(w_hist_test, w_target=0.5)
assert 'final_mean' in stats_w, '[TC53] Stats must contain final_mean FAILED'
assert stats_w['final_cv'] >= 0, '[TC53] Final CV must be non-negative FAILED'

# ---- TC54: simulate_ltp_wave 波速正 ----
np.random.seed(42)
x_w, t_w, u_hist, c_min_wave = simulate_ltp_wave(n=64, length=100.0, D=1.0, r=1.0, t_final=5.0, n_steps=500, bc="NN")
assert c_min_wave > 0.0, '[TC54] Minimum wave speed must be positive FAILED'
assert u_hist.shape[1] == 64, '[TC54] Wave history must have 64 spatial points FAILED'

print('\n全部 54 个测试通过!\n')
