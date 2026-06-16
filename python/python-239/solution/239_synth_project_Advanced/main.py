"""
main.py  —  Unified Entry Point for the QGP Hydrodynamics Project
==================================================================
Drives the complete computational pipeline for a small-scale reproducible
experiment on heavy-ion collision quark-gluon plasma hydrodynamics:

  1. Initialise the QGP fireball (Glauber + hot-spot model).
  2. Add stochastic noise to the initial state (fluctuations).
  3. Run Israel-Stewart viscous hydrodynamic evolution.
  4. Monitor conservation laws and entropy production.
  5. Compute spatial eccentricities and momentum anisotropies (v_n).
  6. Perform a von Neumann stability analysis of the numerical scheme.
  7. Analyse the differentiation matrix structure (Toeplitz, symmetry, ...).
  8. Run a Bayesian (DREAM MCMC) inference of the specific shear viscosity.
  9. Extract latent features via the field encoder + manifold mixup.
  10. Simulate a hard jet traversing the medium and depositing energy.
  11. Perform Cooper-Frye freeze-out and hadron sampling.
  12. Print a comprehensive summary report.

Zero arguments required; simply run:  python main.py

Derived from the synthesis of 15 seed projects:
  - 1102 (RepresentationLearningWaves): field_encoder (attention + residual)
  - 1104 (manifold_mixup): latent-space data augmentation
  - 208 (conservation_ode): conserved quantities monitoring
  - 319 (dream): DREAM MCMC parameter estimation
  - 179 (circle_integrals): flow-harmonic circle integrals
  - 1167 (UniversalPolicies): diffusion-based noise refinement
  - 350 (fd_predator_prey): finite-difference core
  - 115 (box_games): lattice grid initialisation
  - 898 (polynomials): EoS polynomial representation
  - 971 (r8bto): block-Toeplitz matrix analysis
  - 1178 (medin/kind): hadron sampling via residue-level selection
  - 1039 (replay-simulations): jet-medium graph network
  - 654 (lattice_rule): Fibonacci lattice for hot-spot placement
  - 737 (matrix_analyze): structural matrix analysis
  - 018 (arenstorf_ode): long-time conservation tracking
"""

import math
import sys
import time
import numpy as np

from constants_and_units import (
    PI, HBAR_C, TAU_0_FM, T_CROSSOVER, G_GLUEON, G_QUARK_LIGHT,
    G_QUARK_STRANGE, ideal_entropy_density_sb, ideal_energy_density_sb,
    debye_mass_sq, knudsen_number,
)
from eos_qgp import (
    pressure as p_eos, energy_density as eps_eos,
    entropy_density as s_eos, speed_of_sound_sq as cs2_eos,
    temperature_from_energy_density as T_of_eps,
)
from high_order_fd import (
    first_derivative_1d, weno5_first_derivative_1d, laplacian_2d,
)
from initial_conditions import initial_state, initial_dissipative_fields
from hydro_solver import (
    QGPParams, evolve, primitive_from_conserved_simple,
)
from stability_analysis import (
    von_neumann_scan, spectral_analysis, differentiation_matrix,
    matrix_properties_report, block_toeplitz_symbol,
    euler_jacobian_1d_eigenvalues,
)
from flow_harmonics import (
    spatial_eccentricity, momentum_anisotropy, flow_harmonics_full,
    circle_integral_monomial,
)
from mcmc_transport import infer_eta_over_s
from field_encoder import FieldEncoder, manifold_mixup
from stochastic_noise import correlated_noise_field, multi_scale_noise
from hadronization import sample_hadrons, compute_particle_ratios
from network_response import (
    q_hat_local, collisional_dEdx, radiative_dEdx_BDMPS,
    MediumGraph, simulate_jet_path,
)
from conservation_laws import (
    total_quantities, conservation_report,
    check_positive_energy, check_subluminal_velocity,
    entropy_production,
)


def banner(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run():
    t0 = time.time()
    np.random.seed(42)

    banner("1. Constants, Equation of State, and Pre-Equilibrium Scales")
    g_eff = G_GLUEON + (7.0/8.0) * (G_QUARK_LIGHT + G_QUARK_STRANGE)
    for T_test in [0.150, 0.200, 0.300, 0.500]:
        p = p_eos(T_test)
        e = eps_eos(T_test)
        s = s_eos(T_test)
        cs2 = cs2_eos(T_test)
        m2D = debye_mass_sq(T_test, 0.0)
        print(f"  T = {T_test:.3f} GeV: p = {p:8.4f}, eps = {e:8.4f}, "
              f"s = {s:8.3f}, cs^2 = {cs2:.4f}, m_D^2 = {m2D:.4f} GeV^2")
    print(f"  T_crossover  = {T_CROSSOVER*1e3:.1f} MeV")
    print(f"  tau_0        = {TAU_0_FM:.2f} fm/c")
    print(f"  Knudsen(T=0.3 GeV, L=10 fm) = {knudsen_number(0.3, 10.0):.4f}")

    banner("2. High-Order FD Convergence Test")
    for N in [32, 64, 128]:
        L = 2.0 * PI
        dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        f = np.sin(3.0 * x)
        exact = 3.0 * np.cos(3.0 * x)
        for order in [2, 4, 6]:
            dfdx = first_derivative_1d(f, dx, order=order)
            err = np.linalg.norm(dfdx - exact) / np.linalg.norm(exact)
            print(f"  N={N:4d} order={order}: rel_L2_err = {err:.3e}")
        dfdx_w = weno5_first_derivative_1d(f, dx)
        err = np.linalg.norm(dfdx_w - exact) / np.linalg.norm(exact)
        print(f"  N={N:4d} WENO5     : rel_L2_err = {err:.3e}")

    banner("3. Initial Conditions: Glauber + Hot-Spot + Noise")
    nx, ny = 20, 20
    Lx_fm = 20.0
    x_grid, y_grid, U_bg = initial_state(nx, ny, Lx_fm,
                                          model="glauber+hotspot", seed=42)
    # Add stochastic noise
    noise = correlated_noise_field(nx, ny, Lx_fm, amplitude=0.5,
                                   correlation_length=0.7, seed=123)
    eps_noise = U_bg[:, :, 1] * (1.0 + 0.2 * noise)
    U_bg[:, :, 1] = np.maximum(eps_noise, 1e-6)
    eps = U_bg[:, :, 1]
    print(f"  Grid: {nx} x {ny}, L = {Lx_fm} fm, dx = {Lx_fm/nx:.3f} fm")
    print(f"  eps_max = {np.max(eps):.3f} GeV/fm^3")
    print(f"  eps_mean = {np.mean(eps):.3f} GeV/fm^3")
    eps2, Phi2 = spatial_eccentricity(x_grid, y_grid, eps, n_harmonic=2, r_power=2)
    eps3, Phi3 = spatial_eccentricity(x_grid, y_grid, eps, n_harmonic=3, r_power=2)
    print(f"  epsilon_2 = {eps2:.4f}  (Phi_2 = {Phi2*180/PI:.1f} deg)")
    print(f"  epsilon_3 = {eps3:.4f}  (Phi_3 = {Phi3*180/PI:.1f} deg)")

    banner("4. Hydrodynamic Evolution (Israel-Stewart, RK2)")
    Pi_init, pi_init = initial_dissipative_fields(nx, ny)
    params = QGPParams(eta_over_s=0.16, zeta_over_s_max=0.04,
                        fd_order=4, cfl=0.25, diffusion_coeff=0.5,
                        use_weno=False)
    tau_start = TAU_0_FM
    tau_end = 8.0
    snapshots = evolve(U_bg, Pi_init, pi_init, tau_start, tau_end,
                        nx, ny, Lx_fm, params, output_every=5, max_steps=80)
    print(f"  Evolved from tau = {tau_start:.2f} to tau = {tau_end:.2f} fm/c")
    print(f"  Number of snapshots: {len(snapshots)}")
    tau_final, U_final, Pi_final, pi_final = snapshots[-1]
    print(f"  Final tau = {tau_final:.3f} fm/c")
    eps_final = U_final[:, :, 1]
    print(f"  eps_final: max = {np.max(eps_final):.3f}, "
          f"mean = {np.mean(eps_final):.3f} GeV/fm^3")

    banner("5. Conservation Law Verification")
    dx = Lx_fm / nx; dy = Lx_fm / ny
    ok_e, emin = check_positive_energy(U_final)
    ok_v, vmax = check_subluminal_velocity(U_final)
    print(f"  Positive energy: {ok_e}  (min eps = {emin:.3e})")
    print(f"  Subluminal v:    {ok_v}  (max |v| = {vmax:.4f})")
    rep = conservation_report(snapshots, dx, dy)
    print(f"  Relative drifts over evolution:")
    print(f"    B_drift  = {max(abs(x) for x in rep['B_drift']):.3e}")
    print(f"    E_drift  = {max(abs(x) for x in rep['E_drift']):.3e}")
    print(f"    Px_drift = {max(abs(x) for x in rep['Px_drift']):.3e}")
    print(f"    Py_drift = {max(abs(x) for x in rep['Py_drift']):.3e}")
    taus_ent, S_ent = entropy_production(snapshots, dx, dy)
    dS = S_ent[-1] - S_ent[0]
    print(f"  Entropy production Delta S = {dS:+.4f} (>0 => second law OK)")

    banner("6. Flow Harmonics v_n at Final Time")
    vn_result = flow_harmonics_full(x_grid, y_grid, U_final, n_max=5)
    for n in range(1, 6):
        vn, Psi_n = vn_result[n]
        print(f"  v_{n} = {vn:.5f}   Psi_{n} = {Psi_n*180/PI:.1f} deg")

    banner("7. Von Neumann Stability Analysis")
    orders = [2, 4, 6]
    cfl_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    vn_scan = von_neumann_scan(orders, cfl_values)
    print(f"  {'Order':>6} {'CFL':>6} {'max|g|':>10}  Status")
    for (order, cfl), (mg, th) in sorted(vn_scan.items()):
        status = "STABLE" if mg <= 1.0 + 1e-3 else "UNSTABLE"
        print(f"  {order:6d} {cfl:6.2f} {mg:10.6f}  {status}")

    banner("8. Spectral Analysis of Differentiation Matrix")
    N = 24; dx_sp = 2.0 * PI / N
    for order in [2, 4, 6]:
        D = differentiation_matrix(N, dx_sp, order)
        rep_m = matrix_properties_report(D, label=f"FD_{order}")
        print(f"  Order {order}:")
        print(f"    symmetric={rep_m['is_symmetric']}, "
              f"diag_dom={rep_m['is_diagonally_dominant']}, "
              f"toeplitz={rep_m['is_toeplitz']}")
        print(f"    spectral_radius={rep_m['spectral_radius']:.4f}, "
              f"condition={rep_m['condition_estimate']:.3e}")

    banner("9. Block-Toeplitz Symbol (QGP Flux Jacobian)")
    B0 = np.array([[0.0, 1.0], [cs2_eos(0.3), 0.0]])
    B1 = np.array([[0.0, 0.5], [0.5 * cs2_eos(0.3), 0.0]])
    thetas, eigvals = block_toeplitz_symbol([B0, B1])
    print(f"  Eigenvalue range over theta in [0, 2pi]:")
    print(f"    Re: [{np.min(np.real(eigvals)):+.4f}, "
          f"{np.max(np.real(eigvals)):+.4f}]")
    print(f"    Im: [{np.min(np.imag(eigvals)):+.4f}, "
          f"{np.max(np.imag(eigvals)):+.4f}]")

    banner("10. Relativistic Euler Eigenstructure")
    for vx_test in [0.0, 0.3, 0.6]:
        eigs = euler_jacobian_1d_eigenvalues(vx_test, cs2_eos(0.3))
        print(f"  v = {vx_test:.2f}: eigenvalues = "
              f"[{eigs[0]:+.4f}, {eigs[1]:+.4f}, {eigs[2]:+.4f}]")

    banner("11. Bayesian Inference of eta/s (DREAM MCMC)")
    chains, log_prob, R_hat, summary = infer_eta_over_s(seed=42)
    print(f"  eta/s  = {summary['eta_over_s_mean']:.4f} +/- "
          f"{summary['eta_over_s_std']:.4f}")
    print(f"  zeta/s_max = {summary['zeta_over_s_max_mean']:.4f} +/- "
          f"{summary['zeta_over_s_max_std']:.4f}")
    print(f"  Gelman-Rubin R_hat = "
          f"[{summary['R_hat'][0]:.4f}, {summary['R_hat'][1]:.4f}]")

    banner("12. Field Encoder + Manifold Mixup")
    encoder = FieldEncoder(C_in=4, C_hidden=16, n_blocks=2, d_latent=8, seed=0)
    z_bg = encoder.encode_grid(U_bg)
    z_final = encoder.encode_grid(U_final)
    print(f"  latent(z_initial) = {z_bg}")
    print(f"  latent(z_final)   = {z_final}")
    z_mix = manifold_mixup(z_bg, z_final, alpha=1.0, seed=1)
    print(f"  manifold_mixup    = {z_mix}")

    banner("13. Jet-Medium Interaction")
    qhat = q_hat_local(0.3)
    dEc = collisional_dEdx(100.0, 0.3)
    dEr = radiative_dEdx_BDMPS(100.0, 0.3, 5.0)
    print(f"  q_hat(T=0.3 GeV) = {qhat:.3f} GeV^2/fm")
    print(f"  dE/dx_coll (E=100, T=0.3) = {dEc:.4f} GeV/fm")
    print(f"  dE/dx_rad  (E=100, T=0.3, L=5 fm) = {dEr:.4f} GeV/fm")
    # Simulate a jet path through the evolving medium
    path_xy, E_out, deposit = simulate_jet_path(
        x_grid, y_grid, U_final,
        x_start=-8.0, y_start=0.0, phi_jet=0.0,
        E_jet=100.0, n_steps=25, path_length=16.0, seed=42)
    print(f"  Jet entered with E=100.0 GeV, exits with E={E_out:.2f} GeV")
    print(f"  Total deposited = {np.sum(deposit) * dx * dy:.3f} GeV")

    banner("14. Cooper-Frye Freeze-Out and Hadron Sampling")
    samples = sample_hadrons(U_final, T_fo=0.150, n_samples=15, seed=42)
    counts, ratios = compute_particle_ratios(samples)
    print(f"  Total sampled hadrons: {len(samples)}")
    print(f"  Species fractions:")
    for k, v in sorted(ratios.items(), key=lambda x: -x[1])[:6]:
        print(f"    {k:4s}: {v:.4f}")

    banner("15. Circle Integral Validation (Monomial Formula)")
    for e1, e2 in [(2, 0), (0, 2), (2, 2), (4, 0), (4, 2), (1, 1)]:
        val = circle_integral_monomial(e1, e2)
        print(f"  int_0^{{2pi}} cos^{e1} sin^{e2} dphi = {val:.6f}")

    banner("16. Summary")
    elapsed = time.time() - t0
    print(f"  Total wall-clock time: {elapsed:.2f} s")
    print(f"  Pipeline completed successfully.")
    print(f"  All 15 seed projects integrated:")
    seeds = [
        "1102 (RepresentationLearningWaves -> field_encoder)",
        "1104 (manifold_mixup             -> latent mixup)",
        "208  (conservation_ode           -> conservation_laws)",
        "319  (dream                      -> mcmc_transport)",
        "179  (circle_integrals           -> flow_harmonics)",
        "1167 (UniversalPolicies          -> stochastic_noise)",
        "350  (fd_predator_prey           -> high_order_fd)",
        "115  (box_games                  -> initial_conditions)",
        "898  (polynomials                -> eos_qgp)",
        "971  (r8bto                      -> stability_analysis)",
        "1178 (medin/kind                 -> hadronization)",
        "1039 (replay-simulations         -> network_response)",
        "654  (lattice_rule               -> initial_conditions)",
        "737  (matrix_analyze             -> stability_analysis)",
        "018  (arenstorf_ode              -> conservation_laws)",
    ]
    for s in seeds:
        print(f"    - {s}")
    print()
    print("  Science problem solved:")
    print("    Small-scale reproducible 2+1D viscous relativistic")
    print("    hydrodynamic simulation of a QGP fireball with:")
    print("    - Glauber + hot-spot initial conditions")
    print("    - Israel-Stewart causal dissipative evolution")
    print("    - High-order finite differences (order 2/4/6, WENO5)")
    print("    - Von Neumann + block-Toeplitz stability analysis")
    print("    - Bayesian inference of eta/s from v_n")
    print("    - Cooper-Frye freeze-out and hadron sampling")
    print("    - Jet-medium interaction via graph message passing")


if __name__ == "__main__":
    run()
