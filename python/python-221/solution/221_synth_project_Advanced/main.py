"""
main.py
=======

Unified entry point for the computational high-energy physics simulation:
Monte Carlo event generation, phase-space integration, high-order finite
difference evolution, and stability analysis for a 2->3 scattering process.

Scientific problem:
-------------------
We simulate the process pp -> 3 jets at leading order (LO) and
next-to-leading order (NLO) accuracy in a simplified QCD model:

1. Phase-space generation with fractal hadronization boundary
2. Importance sampling in bounded annular momentum regions
3. High-order quadrature over triangular detector acceptance regions
4. DGLAP evolution via high-order finite difference methods
5. GLR-MQ saturation equation via nonlinear Newton solver
6. Running coupling evolution via implicit midpoint ODE integration
7. Parton shower as a Markov process with semi-Markov waiting times
8. K-factor estimation via neural tangent kernel regression
9. Hadronic cascade modeled as a compartmental system
10. Moment-space resummation via Hankel Cholesky factorization
11. Kinematic threshold finding via Laguerre root-finding
12. Stability analysis of all numerical methods

All computations are fully deterministic given the random seed.

Usage:
    python main.py
    (no arguments needed)

Author: PROJECT_221 synthesis
"""

import math
import random
import time
from typing import Dict, List, Tuple

# Import all modules
from physics_constants import (
    PI, alpha_s_1loop, alpha_s_2loop, mandelstam_s, threshold_s_2to3,
    flux_factor, kallen_function, splitting_pqq, splitting_pgg,
    MP_PROTON, MH_HIGGS, BETA0_QCD
)
from phase_space_geometry import (
    fractal_hadronization_boundary, fractal_dimension_estimate,
    annulus_sample, annulus_area,
    cube01_monomial_integral, cube01_sample,
    phase_space_volume_ndim, feynman_x_moment_integral
)
from quadrature_engines import (
    lyness_integrate, kallen_threshold_root,
    dalitz_boundary_s34, zero_laguerre
)
from fd_poisson_solver import (
    fd1d_poisson, dglap_moment_evolution,
    burgers_steady_viscous, glr_mq_saturation
)
from ode_integrators import (
    midpoint_fixed, doughnut_exact, DoughnutParameters,
    evolve_coupling, doughnut_rhs
)
from markov_transitions import (
    MarkovChain, generate_branching_times,
    BranchingTimeDistribution, sudakov_form_factor
)
from kernel_methods import (
    ntk_relu_kernel, kernel_regression_predict,
    k_factor_from_matrix_element
)
from cascade_compartments import (
    CascadeSystem, build_default_event_cuts, levels_extract
)
from hankel_moments import (
    hankel_spd_cholesky_lower, hankel_from_cholesky,
    check_hankel_property, moment_hankel_matrix,
    resummation_exponent
)
from event_generator import (
    generate_events, compute_scale_uncertainty,
    pdf_gluon_x, pdf_quark_x
)
from stability_analysis import (
    anomalous_dimension_matrix, matrix_eigenvalues_2x2,
    check_stability, l2_norm, convergence_order,
    cfl_condition, fixed_point_contraction_rate,
    hankel_condition
)


def section_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subsection_header(title: str) -> None:
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")


# ============================================================================
# Section A: Phase-space geometry and fractal boundary
# ============================================================================
def run_phase_space_geometry() -> Dict:
    """Run phase-space geometry computations."""
    section_header("A. PHASE-SPACE GEOMETRY AND FRACTAL BOUNDARY")

    # A1: Fractal hadronization boundary
    subsection_header("A1. Fractal Hadronization Boundary")
    print("Computing fractal boundary in (pT, rapidity) space...")
    print("  Method: Midpoint displacement with perturbation strength mu")
    print("  Physical meaning: Boundary between perturbative and non-perturbative regions")
    print("  Formula: Q[2k+1] = 0.5*(P[k]+P[k+1]) + w*(P[k]+P[k+1]) - w*(P[k-1]+P[k+2])")

    boundary_0 = fractal_hadronization_boundary(mu=0.05, iterations=0, seed=42)
    boundary_1 = fractal_hadronization_boundary(mu=0.10, iterations=1, seed=42)
    boundary_2 = fractal_hadronization_boundary(mu=0.10, iterations=2, seed=42)
    boundary_3 = fractal_hadronization_boundary(mu=0.15, iterations=3, seed=42)

    print(f"  Iteration 0: {len(boundary_0)} points")
    print(f"  Iteration 1: {len(boundary_1)} points")
    print(f"  Iteration 2: {len(boundary_2)} points")
    print(f"  Iteration 3: {len(boundary_3)} points")

    d0 = fractal_dimension_estimate(boundary_0)
    d1 = fractal_dimension_estimate(boundary_1)
    d2 = fractal_dimension_estimate(boundary_2)
    d3 = fractal_dimension_estimate(boundary_3)

    print(f"  Fractal dimension (iter 0): {d0:.4f}")
    print(f"  Fractal dimension (iter 1): {d1:.4f}")
    print(f"  Fractal dimension (iter 2): {d2:.4f}")
    print(f"  Fractal dimension (iter 3): {d3:.4f}")
    print(f"  Expected: D increases with iteration (smooth -> rough)")

    # A2: Annulus sampling for transverse momenta
    subsection_header("A2. Annular Phase-Space Sampling")
    rng = random.Random(42)
    pt_min, pt_max = 20.0, 500.0
    n_sample = 100
    pts = annulus_sample(pt_min, pt_max, n_sample, rng)
    area = annulus_area(pt_min, pt_max)

    print(f"  Annular region: {pt_min} < pT < {pt_max} GeV")
    print(f"  Area = pi*(r2^2 - r1^2) = {area:.2f} GeV^2")
    print(f"  Sampled {n_sample} points uniformly")
    # Compute mean pT
    mean_pt = sum(math.sqrt(p[0] ** 2 + p[1] ** 2) for p in pts) / n_sample
    print(f"  Mean pT = {mean_pt:.2f} GeV")

    # A3: Hypercube monomial integrals
    subsection_header("A3. Feynman-x Moment Integrals")
    for exps in [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]]:
        val = cube01_monomial_integral(exps)
        print(f"  I(x^{exps[0]} y^{exps[1]} z^{exps[2]}) = {val:.8f}")

    # A4: Phase space volume
    sqrts = 13000.0  # LHC 13 TeV
    ps_vol = phase_space_volume_ndim(3, sqrts, [0.0, 0.0, 0.0])
    print(f"\n  3-body phase space volume at sqrt(s) = {sqrts} GeV:")
    print(f"  Phi_3 = {ps_vol:.6e}")

    # A5: Feynman-x simplex integral
    moment_val = feynman_x_moment_integral([2, 1, 1], 3)
    print(f"  Feynman-x moment M(2,1,1) for n=3: {moment_val:.8f}")

    return {
        'fractal_dim': d3,
        'annulus_area': area,
        'ps_volume': ps_vol,
    }


# ============================================================================
# Section B: Quadrature and root-finding
# ============================================================================
def run_quadrature() -> Dict:
    """Run quadrature and root-finding computations."""
    section_header("B. HIGH-ORDER QUADRATURE AND KINEMATIC ROOT-FINDING")

    # B1: Lyness quadrature over triangular detector region
    subsection_header("B1. Lyness-Jespersen Symmetric Quadrature")
    print("Integrating over triangular detector acceptance region...")
    print("  Triangle vertices in (eta, phi) space:")

    v1 = (-2.0, 0.0)
    v2 = (2.0, 0.0)
    v3 = (0.0, 2.0 * PI / 3.0)
    print(f"    v1 = {v1}, v2 = {v2}, v3 = {v3}")

    # Integrate a test function: the matrix-element-like function
    def test_func(x: float, y: float) -> float:
        r2 = x * x + y * y
        return math.exp(-r2 / 10.0) * (1.0 + 0.1 * math.cos(3.0 * y))

    for rule in [1, 2, 3, 4]:
        val = lyness_integrate(test_func, rule, v1, v2, v3)
        print(f"  Rule {rule}: integral = {val:.8f}")

    # B2: Kinematic threshold via Laguerre root-finding
    subsection_header("B2. Kinematic Threshold via Laguerre Method")
    sqrts = 13000.0
    s = sqrts ** 2
    m1 = 5.0  # jet mass
    m2 = 5.0
    m3 = 5.0
    s_th = threshold_s_2to3(m1, m2, m3)
    print(f"  Threshold s_th = (m3+m4+m5)^2 = {s_th:.2f} GeV^2")
    print(f"  Collision s = {s:.2e} GeV^2")
    print(f"  Well above threshold: {s > s_th}")

    # Find the s45 threshold using Laguerre
    s45_th = kallen_threshold_root(s, m1 ** 2, m2 ** 2)
    print(f"  s45 threshold from Laguerre: {s45_th:.4f} GeV^2")
    # Analytical value
    s45_analytic = (sqrts - m1) ** 2
    print(f"  s45 threshold analytical: {s45_analytic:.4f} GeV^2")
    print(f"  Relative error: {abs(s45_th - s45_analytic) / (s45_analytic + 1e-30):.2e}")

    # B3: Dalitz plot boundaries
    subsection_header("B3. Dalitz Plot Boundaries")
    s34_min, s34_max = dalitz_boundary_s34(s, m1, m2, m3, m2, m3)
    print(f"  s34 range: [{s34_min:.2f}, {s34_max:.2f}] GeV^2")

    return {
        'quadrature_val': val,
        's45_threshold': s45_th,
    }


# ============================================================================
# Section C: Finite difference evolution
# ============================================================================
def run_fd_evolution() -> Dict:
    """Run finite-difference evolution computations."""
    section_header("C. HIGH-ORDER FINITE DIFFERENCE EVOLUTION")

    # C1: DGLAP-like Poisson solver
    subsection_header("C1. DGLAP Moment Evolution (Poisson FD)")
    print("Solving -d^2 q(N) / d(ln Q^2)^2 = source(N)...")
    print("  Physical meaning: PDF evolution in moment space")

    def dglap_source(lnq2: float) -> float:
        # Source term from splitting function convolution
        return 0.1 * math.exp(-lnq2 / 5.0) * math.sin(lnq2)

    def dglap_bc(lnq2: float) -> float:
        # Boundary conditions: PDF value at initial/final scale
        return math.exp(-lnq2 / 10.0)

    nx = 50
    lnq2_min = math.log(2.0)   # Q^2 = 2 GeV^2
    lnq2_max = math.log(10000.0)  # Q^2 = 10000 GeV^2
    u, x = fd1d_poisson(nx, lnq2_min, lnq2_max, dglap_source, dglap_bc)

    print(f"  Grid: {nx} points in ln(Q^2) from {lnq2_min:.3f} to {lnq2_max:.3f}")
    print(f"  Solution at Q^2_min: u[0] = {u[0]:.6f}")
    print(f"  Solution at Q^2_max: u[-1] = {u[-1]:.6f}")
    print(f"  Solution at midpoint: u[{nx//2}] = {u[nx//2]:.6f}")

    # C2: DGLAP moment evolution (direct)
    subsection_header("C2. Direct DGLAP Moment Evolution")
    q0 = [1.0, 0.5, 0.25, 0.125, 0.0625]  # initial moments N=1,2,3,4,5
    ln_grid = [lnq2_min + i * (lnq2_max - lnq2_min) / 20 for i in range(21)]

    def anomalous_dim(t: float) -> float:
        # Simplified anomalous dimension
        return -0.5 * alpha_s_1loop(math.exp(t))

    q_evolved = dglap_moment_evolution(q0, ln_grid, anomalous_dim)
    print(f"  Initial moments: {[f'{q:.4f}' for q in q0]}")
    print(f"  Evolved moments: {[f'{q:.4f}' for q in q_evolved]}")

    # C3: GLR-MQ saturation via Burgers solver
    subsection_header("C3. GLR-MQ Saturation (Burgers Newton)")
    print("Solving steady viscous Burgers equation for gluon saturation...")
    print("  u * du/dx = nu * d^2u/dx^2")
    print("  Physical meaning: gluon density saturation at small x")

    xbj = [math.exp(-i * 0.5) for i in range(20)]  # x from 1 to small
    G, t_grid = glr_mq_saturation(xbj, q2=100.0, g0=1.0, nu_eff=0.1, n=30)
    print(f"  Grid: {len(G)} points in ln(1/x)")
    print(f"  G(x_max) = {G[0]:.6f}")
    print(f"  G(x_min) = {G[-1]:.6f}")
    print(f"  Saturation ratio G_min/G_max = {G[-1]/(G[0]+1e-30):.4f}")

    # C4: Convergence study
    subsection_header("C4. FD Convergence Order")
    errors = []
    spacings = []
    # Get reference solution at finest grid
    u_ref, x_ref = fd1d_poisson(161, lnq2_min, lnq2_max, dglap_source, dglap_bc)
    for nx_test in [11, 21, 41, 81, 161]:
        u_test, x_test = fd1d_poisson(nx_test, lnq2_min, lnq2_max,
                                       dglap_source, dglap_bc)
        dx = (lnq2_max - lnq2_min) / (nx_test - 1)
        spacings.append(dx)
        if nx_test < 161:
            # Interpolate reference solution onto coarser grid
            u_ref_interp = []
            for xi in x_test:
                # Find index in reference grid
                frac = (xi - lnq2_min) / (lnq2_max - lnq2_min) * 160
                idx = int(frac)
                idx = max(0, min(159, idx))
                t_frac = frac - idx
                u_interp = u_ref[idx] * (1.0 - t_frac) + u_ref[idx + 1] * t_frac
                u_ref_interp.append(u_interp)
            err = l2_norm(u_test, u_ref_interp)
            errors.append(err)
        else:
            errors.append(1e-15)  # reference (essentially zero error)

    if len(errors) >= 3:
        p = convergence_order(errors[:-1], spacings[:-1])
        print(f"  Observed convergence order: p = {p:.2f}")
        print(f"  Expected: p = 2 (second-order FD)")

    return {
        'pdf_moment_evolved': q_evolved,
        'gluon_saturation': G[-1],
    }


# ============================================================================
# Section D: ODE integration and running coupling
# ============================================================================
def run_ode_integration() -> Dict:
    """Run ODE integration computations."""
    section_header("D. IMPLICIT ODE INTEGRATION AND RUNNING COUPLING")

    # D1: Running coupling evolution
    subsection_header("D1. Running Coupling via Implicit Midpoint")
    a0 = alpha_s_1loop(100.0)  # alpha_s at Q^2 = 100 GeV^2
    ln_q2_start = math.log(100.0)
    ln_q2_end = math.log(10000.0)
    n_steps = 100

    t_grid, alpha_grid = evolve_coupling(
        a0, ln_q2_start, ln_q2_end, n_steps,
        BETA0_QCD, 102.0 - 38.0 * 5 / 3.0
    )
    print(f"  alpha_s(Q^2=100 GeV^2) = {alpha_grid[0]:.6f}")
    print(f"  alpha_s(Q^2=10000 GeV^2) = {alpha_grid[-1]:.6f}")
    print(f"  Asymptotic freedom: coupling decreases with Q^2")
    print(f"  Ratio alpha_high/alpha_low = {alpha_grid[-1]/alpha_grid[0]:.4f}")

    # D2: Compare with analytical 1-loop
    subsection_header("D2. Comparison with Analytical Formula")
    for q2_test in [100.0, 1000.0, 5000.0, 10000.0]:
        a_numerical = alpha_grid[min(len(alpha_grid) - 1,
                                     int((math.log(q2_test) - ln_q2_start)
                                         / (ln_q2_end - ln_q2_start) * n_steps))]
        a_analytical = alpha_s_1loop(q2_test)
        err = abs(a_numerical - a_analytical) / (a_analytical + 1e-30)
        print(f"  Q^2 = {q2_test:8.1f}: numerical = {a_numerical:.6f}, "
              f"analytical = {a_analytical:.6f}, rel.err = {err:.2e}")

    # D3: Doughnut manifold for color flow
    subsection_header("D3. Color-Flow Manifold (Doughnut ODE)")
    params = DoughnutParameters(m=3.0, n=5.0, y0=(1.0, 1.0, 3.0))
    t_doughnut = [i * 0.1 for i in range(50)]
    y_doughnut = doughnut_exact(t_doughnut, params)
    print(f"  Color-flow trajectory: {len(y_doughnut)} points")
    print(f"  y1 range: [{min(y[0] for y in y_doughnut):.4f}, "
          f"{max(y[0] for y in y_doughnut):.4f}]")
    print(f"  y2 range: [{min(y[1] for y in y_doughnut):.4f}, "
          f"{max(y[1] for y in y_doughnut):.4f}]")
    print(f"  y3 range: [{min(y[2] for y in y_doughnut):.4f}, "
          f"{max(y[2] for y in y_doughnut):.4f}]")

    # D4: Numerical integration of doughnut ODE
    subsection_header("D4. Numerical ODE Integration")
    y0_num = [params.y0[0], params.y0[1], params.y0[2]]
    t_num, y_num = midpoint_fixed(
        lambda t, y: doughnut_rhs(t, y, params),
        (0.0, 2.0), y0_num, 50, it_max=30
    )
    print(f"  Integrated doughnut ODE: {len(y_num)} steps")
    print(f"  Final state (numerical): ({y_num[-1][0]:.4f}, "
          f"{y_num[-1][1]:.4f}, {y_num[-1][2]:.4f})")
    y_exact = doughnut_exact([2.0], params)[0]
    print(f"  Final state (exact):     ({y_exact[0]:.4f}, "
          f"{y_exact[1]:.4f}, {y_exact[2]:.4f})")

    return {
        'alpha_s_initial': alpha_grid[0],
        'alpha_s_final': alpha_grid[-1],
    }


# ============================================================================
# Section E: Markov shower and branching distributions
# ============================================================================
def run_markov_shower() -> Dict:
    """Run Markov chain parton shower computations."""
    section_header("E. MARKOV PARTON SHOWER AND BRANCHING DISTRIBUTIONS")

    # E1: Build parton shower Markov chain
    subsection_header("E1. Parton Shower Markov Chain")
    alpha_s = 0.118
    chain = MarkovChain()
    chain.build_parton_shower_chain(alpha_s)
    print(f"  Built parton shower chain with alpha_s = {alpha_s}")
    print(f"  States: {list(chain.states.keys())}")
    state_order = ['q', 'g', 'qg', 'gg', 'qqbar']
    mat = chain.transition_matrix(state_order)
    print(f"  Transition matrix ({len(state_order)} x {len(state_order)}):")
    print(f"    {'':>8}", end="")
    for s in state_order:
        print(f"{s:>8}", end="")
    print()
    for i, s in enumerate(state_order):
        print(f"    {s:>6}", end="")
        for j in range(len(state_order)):
            print(f"{mat[i][j]:8.4f}", end="")
        print()

    # E2: Sample shower trajectories
    subsection_header("E2. Shower Trajectories")
    rng = random.Random(42)
    n_trajectories = 5
    for traj_i in range(n_trajectories):
        traj = chain.sample('q', 10, rng)
        print(f"  Trajectory {traj_i + 1}: {' -> '.join(traj)}")

    # E3: Branching time distribution
    subsection_header("E3. Branching Time Distribution (Semi-Markov)")
    data, dist = generate_branching_times(200, seed=42)
    print(f"  Generated {len(data)} branching times")
    print(f"  Fitted log-normal: mu = {dist.mu:.4f}, sigma = {dist.sigma:.4f}")
    # Sample new branching times
    samples = [dist.sample(rng) for _ in range(5)]
    print(f"  New samples: {[f'{s:.4f}' for s in samples]}")

    # E4: Sudakov form factor
    subsection_header("E4. Sudakov Form Factor")
    for gamma in [0.1, 0.5, 1.0]:
        for dt in [0.5, 1.0, 2.0, 5.0]:
            delta = sudakov_form_factor(0.0, dt, gamma)
            print(f"  gamma={gamma:.1f}, dt={dt:.1f}: Delta = {delta:.6f}")

    return {
        'n_states': len(chain.states),
        'fitted_mu': dist.mu,
    }


# ============================================================================
# Section F: K-factor estimation via NTK kernel
# ============================================================================
def run_kernel_regression() -> Dict:
    """Run kernel regression for K-factor estimation."""
    section_header("F. K-FACTOR ESTIMATION VIA NEURAL TANGENT KERNEL")

    # F1: Generate training data (phase-space points and K-factors)
    subsection_header("F1. Training Data Generation")
    rng = random.Random(123)
    n_train = 20
    n_test = 5

    x_train = []
    y_train = []
    for _ in range(n_train):
        # 3D phase-space point: (pT, eta, phi) of the leading jet
        pt = rng.uniform(20.0, 500.0)
        eta = rng.uniform(-4.0, 4.0)
        phi = rng.uniform(0.0, 2.0 * PI)
        x_train.append([pt / 100.0, eta / 4.0, phi / PI])
        # K-factor: approximate NLO/LO ratio
        k = 1.0 + 0.3 * alpha_s_1loop(pt * pt) * (1.0 + 0.1 * abs(eta))
        y_train.append(k)

    x_test = []
    for _ in range(n_test):
        pt = rng.uniform(20.0, 500.0)
        eta = rng.uniform(-4.0, 4.0)
        phi = rng.uniform(0.0, 2.0 * PI)
        x_test.append([pt / 100.0, eta / 4.0, phi / PI])

    print(f"  Training points: {n_train}")
    print(f"  Test points: {n_test}")

    # F2: Kernel regression
    subsection_header("F2. NTK Kernel Regression")
    y_pred = kernel_regression_predict(x_train, y_train, x_test,
                                        depth=2, reg=1e-3)
    print(f"  NTK depth = 2, regularization = 1e-3")
    for i in range(n_test):
        print(f"  Test point {i + 1}: predicted K = {y_pred[i]:.6f}")

    # F3: Single kernel evaluation
    subsection_header("F3. NTK Kernel Values")
    x_a = [1.0, 0.5, 0.3]
    x_b = [0.8, 0.6, 0.4]
    N, S = ntk_relu_kernel(x_a, x_b, depth=2)
    print(f"  NTK(x_a, x_b) = {N:.6f}")
    print(f"  NNGP(x_a, x_b) = {S:.6f}")

    return {
        'k_factor_mean': sum(y_pred) / len(y_pred) if y_pred else 1.0,
    }


# ============================================================================
# Section G: Cascade and viability
# ============================================================================
def run_cascade() -> Dict:
    """Run hadronic cascade and viability computations."""
    section_header("G. HADRONIC CASCADE AND EVENT VIABILITY")

    # G1: Cascade system
    subsection_header("G1. Rapidity Cascade System")
    n_comp = 12
    cascade = CascadeSystem(n_comp, y_min=-3.0, y_max=3.0)
    print(f"  {n_comp} compartments in rapidity [-3, 3]")

    # Initialize with some partons
    rng = random.Random(42)
    rapidities = [rng.uniform(-2.0, 2.0) for _ in range(20)]
    energies = [rng.uniform(1.0, 10.0) for _ in range(20)]
    cascade.initialize_from_parton_shower(rapidities, energies)
    print(f"  Initialized with {len(rapidities)} partons")
    print(f"  Initial multiplicity: {cascade.total_multiplicity():.2f}")
    print(f"  Initial entropy: {cascade.entropy():.4f}")

    # Evolve
    dt = 0.1
    n_steps = 20
    for step in range(n_steps):
        cascade.step(dt)
    print(f"  After {n_steps} steps (dt={dt}):")
    print(f"  Final multiplicity: {cascade.total_multiplicity():.2f}")
    print(f"  Final entropy: {cascade.entropy():.4f}")

    # G2: Viability scoring
    subsection_header("G2. Event Viability Scoring")
    cuts = build_default_event_cuts(pt_min=20.0, eta_max=4.5, mjj_min=100.0)
    test_events = [
        {'pt_jet1': 150.0, 'pt_jet2': 100.0, 'eta_jet1': 1.0,
         'eta_jet2': -1.5, 'mjj': 500.0},
        {'pt_jet1': 15.0, 'pt_jet2': 100.0, 'eta_jet1': 1.0,
         'eta_jet2': -1.5, 'mjj': 500.0},  # pt_jet1 too low
        {'pt_jet1': 150.0, 'pt_jet2': 100.0, 'eta_jet1': 5.0,
         'eta_jet2': -1.5, 'mjj': 500.0},  # eta_jet1 too high
    ]
    for i, ev_vals in enumerate(test_events):
        v = cuts.compute_viability(ev_vals)
        print(f"  Event {i + 1}: viability = {v:.4f} "
              f"({'ACCEPTED' if v > 0.5 else 'REJECTED'})")

    # G3: Cross-section contour levels
    subsection_header("G3. Cross-Section Contour Levels")

    def xs_function(mu_r: float, mu_f: float) -> float:
        a_r = alpha_s_1loop(mu_r * mu_r)
        a_f = alpha_s_1loop(mu_f * mu_f)
        return a_r ** 3 * a_f ** 2 * 1000.0

    levels = levels_extract(xs_function, level_num=10,
                            x_range=(50.0, 500.0),
                            y_range=(50.0, 500.0),
                            data_num=30, seed=42)
    print(f"  Extracted {len(levels)} contour levels from sigma(mu_R, mu_F)")
    for i, lev in enumerate(levels[:5]):
        print(f"    Level {i + 1}: {lev:.4f}")

    return {
        'final_multiplicity': cascade.total_multiplicity(),
        'final_entropy': cascade.entropy(),
    }


# ============================================================================
# Section H: Hankel moment resummation
# ============================================================================
def run_hankel_resummation() -> Dict:
    """Run Hankel moment-space resummation computations."""
    section_header("H. MOMENT-SPACE RESUMMATION VIA HANKEL CHOLESKY")

    # H1: Build moment sequence
    subsection_header("H1. Mellin Moment Sequence")
    n_moments = 10
    moments = []
    for n in range(2 * n_moments):
        # Moments of a simple PDF: q(x) = 6*x*(1-x)
        # q(N) = 6 * B(N+1, 2) = 6 * Gamma(N+1)*Gamma(2) / Gamma(N+3)
        val = 6.0 * math.gamma(n + 2.0) * math.gamma(2.0) / math.gamma(n + 4.0)
        moments.append(val)
    print(f"  Computed {len(moments)} Mellin moments")
    print(f"  First 5 moments: {[f'{m:.6f}' for m in moments[:5]]}")

    # H2: Hankel matrix
    subsection_header("H2. Hankel Moment Matrix")
    H = moment_hankel_matrix(moments, n_moments)
    is_hankel = check_hankel_property(H)
    print(f"  Matrix size: {n_moments} x {n_moments}")
    print(f"  Is Hankel (constant anti-diagonals): {is_hankel}")
    # Print first few elements
    print(f"  H[0,0] = {H[0][0]:.6f}, H[0,1] = {H[0][1]:.6f}, H[1,0] = {H[1][0]:.6f}")
    print(f"  Check H[0,1] == H[1,0]: {abs(H[0][1] - H[1][0]) < 1e-12}")

    # H3: Hankel Cholesky
    subsection_header("H3. Hankel SPD Cholesky Factorization")
    # Construct lii and liim1 for the Hankel Cholesky
    lii = [math.sqrt(moments[2 * i]) if moments[2 * i] > 0 else 1e-6
           for i in range(n_moments)]
    liim1 = [moments[2 * i + 1] / (lii[i] + 1e-30) * 0.5
             for i in range(n_moments - 1)]
    L = hankel_spd_cholesky_lower(n_moments, lii, liim1)
    print(f"  Cholesky factor L computed ({n_moments} x {n_moments})")
    print(f"  L[0,0] = {L[0][0]:.6f}")
    print(f"  L[1,0] = {L[1][0]:.6f}, L[1,1] = {L[1][1]:.6f}")

    # H4: Reconstruct H = L*L^T
    H_recon = hankel_from_cholesky(L)
    print(f"  Reconstructed H from L*L^T")
    print(f"  H_recon[0,0] = {H_recon[0][0]:.6f} (should match moment[0] = {moments[0]:.6f})")

    # H5: Condition number
    subsection_header("H5. Hankel Matrix Conditioning")
    cond_info = hankel_condition(moments, n_moments)
    print(f"  Condition number: {cond_info['condition_number']:.2e}")
    print(f"  Well-conditioned: {cond_info['is_well_conditioned']}")

    # H6: Resummation exponent
    subsection_header("H6. Threshold Resummation Exponent")
    alpha_s = 0.118
    for N in [2, 5, 10, 50, 100]:
        g = resummation_exponent(float(N), alpha_s, a_coeff=1.0, b_coeff=0.5)
        print(f"  g(N={N}, alpha_s={alpha_s}) = {g:.6f}")

    return {
        'condition_number': cond_info['condition_number'],
    }


# ============================================================================
# Section I: Event generation and cross-section
# ============================================================================
def run_event_generation() -> Dict:
    """Run the full event generation chain."""
    section_header("I. MONTE CARLO EVENT GENERATION")

    # I1: Generate events
    subsection_header("I1. Event Generation (2->3 process)")
    sqrts = 13000.0  # 13 TeV LHC
    n_events = 200
    pt_min = 20.0
    pt_max = 500.0

    events, stats = generate_events(n_events, sqrts, pt_min, pt_max, seed=42)
    print(f"  Center-of-mass energy: sqrt(s) = {sqrts} GeV")
    print(f"  Generated {stats['n_generated']} events")
    print(f"  Accepted: {stats['n_accepted']} "
          f"({stats['acceptance_rate']*100:.1f}%)")
    print(f"  Mean weight: {stats['mean_weight']:.4e}")
    print(f"  Cross-section estimate: {stats['cross_section_pb']:.4e} pb")

    # I2: Scale uncertainty
    subsection_header("I2. Scale Uncertainty")
    scale_info = compute_scale_uncertainty(events, mu_r_factor=2.0)
    print(f"  Central scale: sigma = {scale_info['sigma_central']:.4e}")
    print(f"  mu_R x 2:      sigma = {scale_info['sigma_up']:.4e}")
    print(f"  mu_R / 2:      sigma = {scale_info['sigma_down']:.4e}")
    print(f"  Scale variation: {scale_info['scale_variation_pct']:.2f}%")

    # I3: PDF evaluation
    subsection_header("I3. Parton Distribution Functions")
    for x in [0.001, 0.01, 0.1, 0.3, 0.5]:
        g = pdf_gluon_x(x, 100.0)
        u = pdf_quark_x(x, 100.0)
        print(f"  x = {x:.3f}: g(x, 100 GeV^2) = {g:.4f}, "
              f"u_v(x, 100 GeV^2) = {u:.4f}")

    return stats


# ============================================================================
# Section J: Stability analysis
# ============================================================================
def run_stability_analysis() -> Dict:
    """Run stability analysis of all numerical methods."""
    section_header("J. STABILITY ANALYSIS AND CONVERGENCE")

    # J1: Anomalous dimension eigenvalues
    subsection_header("J1. Anomalous Dimension Matrix Stability")
    alpha_s = 0.118
    n_mom = 4
    gamma_mat = anomalous_dimension_matrix(n_mom, alpha_s)
    print(f"  Anomalous dimension matrix ({n_mom}x{n_mom}):")
    for i in range(n_mom):
        row_str = "    [" + ", ".join(f"{gamma_mat[i][j]:10.6f}"
                                       for j in range(n_mom)) + "]"
        print(row_str)

    # Check 2x2 sub-block eigenvalues
    if n_mom >= 2:
        sub = [[gamma_mat[0][0], gamma_mat[0][1]],
               [gamma_mat[1][0], gamma_mat[1][1]]]
        eigs = matrix_eigenvalues_2x2(sub)
        stab = check_stability(eigs)
        print(f"  2x2 sub-block eigenvalues: {eigs}")
        print(f"  Max real part: {stab['max_real_part']:.6f}")
        print(f"  Stable: {stab['is_stable']}")
        print(f"  Stiffness ratio: {stab['stiffness_ratio']:.2e}")

    # J2: CFL condition
    subsection_header("J2. CFL Condition Check")
    dx_vals = [0.1, 0.05, 0.01, 0.005]
    D = 0.1  # diffusion coefficient
    for dx in dx_vals:
        dt_max = cfl_condition(dx, D)
        print(f"  dx = {dx:.3f}: dt_max = {dt_max:.4f}")

    # J3: Fixed-point contraction
    subsection_header("J3. Fixed-Point Contraction Rate")
    L_lipschitz = 2.0  # Lipschitz constant
    for h in [0.1, 0.5, 0.8, 1.0]:
        rate = fixed_point_contraction_rate(L_lipschitz, h)
        conv = "CONVERGES" if rate < 1.0 else "DIVERGES"
        print(f"  h = {h:.1f}: contraction rate = {rate:.3f} ({conv})")

    # J4: Splitting function integrability
    subsection_header("J4. Splitting Function Behavior")
    for z in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        pqq = splitting_pqq(z)
        pgg = splitting_pgg(z)
        print(f"  z = {z:.2f}: P_qq = {pqq:.4f}, P_gg = {pgg:.4f}")

    return {'stable': stab['is_stable'] if n_mom >= 2 else True}


# ============================================================================
# Main execution
# ============================================================================
def main():
    """Main entry point: run all sections of the simulation."""
    print("=" * 72)
    print("  PROJECT 221: Computational High-Energy Physics")
    print("  Monte Carlo Event Generation & Phase-Space Integration:")
    print("  High-Order Finite Differences & Stability Analysis")
    print("=" * 72)
    print()
    print("  Scientific problem: 2->3 scattering in simplified QCD")
    print("  Center-of-mass energy: 13 TeV (LHC Run 2)")
    print("  Process: pp -> 3 jets at LO + NLO K-factor")
    print()
    print("  Modules:")
    print("    A. Phase-space geometry and fractal hadronization boundary")
    print("    B. High-order quadrature and kinematic root-finding")
    print("    C. Finite difference DGLAP/GLR-MQ evolution")
    print("    D. Implicit ODE integration and running coupling")
    print("    E. Markov parton shower and branching distributions")
    print("    F. Neural tangent kernel for K-factor estimation")
    print("    G. Hadronic cascade and event viability")
    print("    H. Moment-space resummation via Hankel Cholesky")
    print("    I. Monte Carlo event generation")
    print("    J. Stability analysis and convergence verification")

    t_start = time.time()

    results = {}

    # Run all sections
    results['A'] = run_phase_space_geometry()
    results['B'] = run_quadrature()
    results['C'] = run_fd_evolution()
    results['D'] = run_ode_integration()
    results['E'] = run_markov_shower()
    results['F'] = run_kernel_regression()
    results['G'] = run_cascade()
    results['H'] = run_hankel_resummation()
    results['I'] = run_event_generation()
    results['J'] = run_stability_analysis()

    # Summary
    t_end = time.time()

    section_header("SUMMARY")
    print()
    print("  All 10 sections completed successfully.")
    print()
    print("  Key results:")
    print(f"    Fractal boundary dimension:  {results['A']['fractal_dim']:.4f}")
    print(f"    Phase-space volume:          {results['A']['ps_volume']:.4e}")
    print(f"    s45 kinematic threshold:     {results['B']['s45_threshold']:.4f} GeV^2")
    print(f"    alpha_s(MZ) -> alpha_s(10 TeV): {results['D']['alpha_s_initial']:.4f} -> "
          f"{results['D']['alpha_s_final']:.4f}")
    print(f"    Mean K-factor (NTK):         {results['F']['k_factor_mean']:.4f}")
    print(f"    Cascade final multiplicity:  {results['G']['final_multiplicity']:.2f}")
    print(f"    Hankel condition number:     {results['H']['condition_number']:.2e}")
    print(f"    Event acceptance rate:       {results['I']['acceptance_rate']*100:.1f}%")
    print(f"    Numerical stability:         {'PASS' if results['J']['stable'] else 'FAIL'}")
    print()
    print(f"  Total wall time: {t_end - t_start:.3f} seconds")
    print()
    print("=" * 72)
    print("  Simulation complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
