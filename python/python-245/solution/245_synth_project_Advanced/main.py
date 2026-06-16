"""
main.py
========
Unified entry point for the nuclear fission simulation project.

PROJECT_245: Nuclear Fission Simulation - Fragment Mass Distribution
and Energy Release Modelling with High-Order Finite Differences
and Stability Analysis (Small-Scale Reproducible Experiments)

This script runs the complete fission simulation pipeline:
1. PES computation and critical point analysis
2. Nuclear mesh generation
3. High-order finite difference and stability analysis
4. Langevin dynamics for fission trajectories
5. FTCS PDE solver for probability evolution
6. FEM energy diffusion
7. Fragment mass distribution and energy release
8. Semi-supervised fission mode classification
9. Legendre angular distribution
10. Levenshtein pattern classification
11. 2D FEM thermal diffusion
12. QAOA fission channel optimisation

All modules are executed in sequence with output printed to stdout.
No visualisation is performed.  No input arguments required.
"""

import math
import time
import sys
import os

# Ensure project directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def separator(title: str) -> None:
    """Print a section separator."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    """
    Main entry point: runs all fission simulation components.
    """
    start_time = time.time()

    print("=" * 72)
    print("  PROJECT_245: Nuclear Fission Simulation")
    print("  Fragment Mass Distribution & Energy Release Modelling")
    print("  High-Order Finite Differences & Stability Analysis")
    print("=" * 72)

    # ===================================================================
    # SECTION 1: Nuclear Constants and Binding Energy
    # ===================================================================
    separator("1. Nuclear Constants and Binding Energy [nuclear_constants.py]")

    from nuclear_constants import (
        binding_energy_ld, atomic_mass_ld, get_atomic_mass,
        q_value_fission, woods_saxon_density, coulomb_barrier_energy,
        level_density_parameter, fermi_gas_level_density,
        prompt_neutron_multiplicity, LD_VOLUME, LD_SURFACE,
        ATOMIC_MASS_UNIT_MEV, NEUTRON_MASS_MEV
    )

    # Binding energy per nucleon for key nuclei
    test_nuclei = [
        (4, 2, "He-4"),
        (56, 26, "Fe-56"),
        (132, 50, "Sn-132"),
        (208, 82, "Pb-208"),
        (236, 92, "U-236"),
    ]

    print("\n  Binding energy per nucleon B/A (Liquid Drop Model):")
    print(f"  {'Nucleus':<10} {'A':>5} {'Z':>5} {'B(A,Z) [MeV]':>14} {'B/A [MeV]':>12}")
    print("  " + "-" * 52)
    for a, z, name in test_nuclei:
        b_val = binding_energy_ld(a, z)
        print(f"  {name:<10} {a:>5} {z:>5} {b_val:>14.4f} {b_val/a:>12.4f}")

    # Woods-Saxon density profile
    print("\n  Woods-Saxon density for U-236:")
    for r in [0.0, 3.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]:
        rho = woods_saxon_density(r, 236, 92)
        print(f"    r = {r:>5.1f} fm: rho/rho_0 = {rho:.6f}")

    # Q-value for U-236 fission
    q_val = q_value_fission(236, 92, 95, 38, 141, 54, 0)
    print(f"\n  Q-value for U-236 -> Sr-95 + Xe-141: {q_val:.4f} MeV")

    # Level density
    a_param = level_density_parameter(236)
    log_rho = fermi_gas_level_density(10.0, 236, -3.0)
    print(f"  Level density parameter a(236) = {a_param:.4f} MeV^-1")
    print(f"  ln(rho) at U=10 MeV: {log_rho:.4f}")

    # ===================================================================
    # SECTION 2: Nuclear Mesh Generation
    # ===================================================================
    separator("2. Nuclear Mesh Generation [nuclear_mesh.py]")

    from nuclear_mesh import NuclearMesh, barycentric_coordinates

    # Generate mesh for U-236 at different deformations
    for c, h, alpha, label in [
        (1.0, 1.0, 0.0, "spherical"),
        (1.3, 0.9, 0.0, "elongated"),
        (1.8, 0.5, 0.1, "pre-scission"),
    ]:
        mesh = NuclearMesh(236, 15, 12)
        mesh.generate_shape(c, h, alpha)
        print(f"\n  Shape ({label}): c={c}, h={h}, alpha={alpha}")
        print(f"  {mesh.summary()}")

    # Test barycentric coordinates
    v0 = (0.0, 0.0, 0.0)
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 0.0)
    p = (0.25, 0.25, 0.0)
    lam = barycentric_coordinates(p, v0, v1, v2)
    print(f"\n  Barycentric test: point {p} in triangle {v0},{v1},{v2}")
    print(f"    lambda = ({lam[0]:.4f}, {lam[1]:.4f}, {lam[2]:.4f})")

    # ===================================================================
    # SECTION 3: Potential Energy Surface
    # ===================================================================
    separator("3. Potential Energy Surface [potential_energy_surface.py]")

    from potential_energy_surface import FissionPES

    pes = FissionPES(92, 236)

    # PES along elongation
    print("\n  PES V(c, h=1.0, alpha=0.0) along elongation:")
    print(f"  {'c':>6} {'V_LD [MeV]':>14} {'V_shell [MeV]':>14} {'V_total [MeV]':>14}")
    print("  " + "-" * 52)
    for c_val in [1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.7, 1.75, 1.8, 2.0, 2.2]:
        v_total = pes.total_potential(c_val, 1.0, 0.0)
        v_ld = pes.liquid_drop_energy(c_val, 1.0, 0.0)
        v_shell = pes.shell_correction(c_val, 1.0, 0.0)
        print(f"  {c_val:>6.2f} {v_ld:>14.4f} {v_shell:>14.4f} {v_total:>14.4f}")

    # Find saddle points
    saddles = pes.find_saddle_points(0.0, 1.0)
    print(f"\n  Critical points found: {len(saddles)}")
    for sp in saddles:
        print(f"    {sp['type']:>10} at c={sp['c']:.4f}, E={sp['energy']:.4f} MeV")

    # Gradient test
    grad = pes.gradient(1.35, 1.0, 0.0)
    print(f"\n  Gradient at inner barrier (1.35, 1.0, 0.0):")
    print(f"    dV/dc = {grad[0]:.6f}, dV/dh = {grad[1]:.6f}, dV/da = {grad[2]:.6f}")

    # ===================================================================
    # SECTION 4: High-Order Finite Difference
    # ===================================================================
    separator("4. High-Order Finite Difference [high_order_finite_difference.py]")

    from high_order_finite_difference import (
        apply_fd1, apply_fd2, apply_fd4, build_laplacian_matrix,
        tdse_step_fd
    )
    import numpy as np

    # Test 6th-order derivative on sin(x)
    n_test = 65
    dx_test = 2.0 * math.pi / (n_test - 1)
    x_test = np.linspace(0, 2 * math.pi, n_test)
    f_test = np.sin(x_test)

    # 1st derivative (should be cos(x))
    df1 = apply_fd1(f_test, dx_test, 'periodic')
    err1 = np.max(np.abs(df1 - np.cos(x_test)))

    # 2nd derivative (should be -sin(x))
    df2 = apply_fd2(f_test, dx_test, 'periodic')
    err2 = np.max(np.abs(df2 + np.sin(x_test)))

    print(f"\n  6th-order FD accuracy test on sin(x), N={n_test}:")
    print(f"    max|f'(x) - cos(x)| = {err1:.2e}")
    print(f"    max|f''(x) + sin(x)| = {err2:.2e}")

    # Build Laplacian matrix
    L = build_laplacian_matrix(21, 0.1, 'dirichlet')
    print(f"\n  Laplacian matrix: {L.shape}, spectral range check:")
    eigs = np.linalg.eigvalsh(L)
    print(f"    eigenvalues: min={eigs[0]:.4f}, max={eigs[-1]:.4f}")

    # ===================================================================
    # SECTION 5: Stability Analysis
    # ===================================================================
    separator("5. Von Neumann Stability Analysis [stability_analysis.py]")

    from stability_analysis import (
        von_neumann_amplification, cfl_limit_fd6, cfl_limit_fd2_standard,
        stability_check_collective, dispersion_analysis_fd6,
        compute_fem_mass_matrix_1d, compute_fem_stiffness_matrix_1d,
        quadrature_gauss_legendre
    )

    # CFL limits
    D_test = 0.1
    dx_test = 0.05
    dt_fd6 = cfl_limit_fd6(D_test, dx_test)
    dt_fd2 = cfl_limit_fd2_standard(D_test, dx_test)
    print(f"\n  CFL stability limits (D={D_test}, dx={dx_test}):")
    print(f"    6th-order: dt_max = {dt_fd6:.6e}")
    print(f"    2nd-order: dt_max = {dt_fd2:.6e}")
    print(f"    Improvement factor: {dt_fd6/dt_fd2:.4f}" if dt_fd2 > 0 else "")

    # Amplification factor
    kh = np.linspace(0, math.pi, 100)
    G = von_neumann_amplification(0.4, kh)
    print(f"\n  Amplification factor |G| max: {np.max(np.abs(G)):.6f}")
    print(f"  (Stable if |G| <= 1.0)")

    # Dispersion analysis
    disp = dispersion_analysis_fd6(100)
    max_err = np.max(np.abs(disp['relative_error'][:-1]))
    print(f"\n  6th-order stencil dispersion error: max relative = {max_err:.6e}")

    # Collective Schrodinger stability
    stab = stability_check_collective(15.0, 0.1, 5.0)
    print(f"\n  Collective Schrodinger stability:")
    print(f"    spectral radius: {stab['spectral_radius']:.4f}")
    print(f"    dt_Euler max: {stab['dt_euler_max']:.6e}")
    print(f"    dt recommended: {stab['dt_recommended']:.6e}")

    # Gauss-Legendre quadrature test
    pts, wts = quadrature_gauss_legendre(5)
    print(f"\n  5-point Gauss-Legendre on [-1,1]:")
    print(f"    points: {[f'{p:.6f}' for p in pts]}")
    print(f"    weights sum: {sum(wts):.10f} (should be 2.0)")

    # ===================================================================
    # SECTION 6: Langevin Dynamics
    # ===================================================================
    separator("6. Langevin Dynamics for Fission [langevin_fission.py]")

    from langevin_fission import FissionLangevin, compute_fission_timescale

    import random
    random.seed(42)

    lang = FissionLangevin(92, 236, temperature=1.5)
    lang.set_initial_conditions(1.0, 1.0, 0.0)

    # Run multiple trajectories
    n_traj = 5
    scission_results = []
    for i_traj in range(n_traj):
        lang.set_initial_conditions(1.0, 1.0, 0.0)
        result = lang.run_trajectory(dt=0.005, n_steps=2000)
        scission_results.append(result)
        status = "SCISSION" if result['scissioned'] else "NO_FISSION"
        print(f"  Trajectory {i_traj+1}: {status}, "
              f"c_f={result['final_c']:.3f}, h_f={result['final_h']:.3f}, "
              f"A_H={result['a_heavy']}, A_L={result['a_light']}")

    # MCMC sampling at saddle point
    print(f"\n  Metropolis-Hastings sampling at inner saddle point:")
    mh_samples = lang.metropolis_sample_saddle(n_samples=300, temp_sample=1.5)
    alphas = [s['alpha'] for s in mh_samples]
    mean_alpha = sum(alphas) / len(alphas)
    std_alpha = math.sqrt(sum((a - mean_alpha)**2 for a in alphas) / len(alphas))
    print(f"    N samples: {len(mh_samples)}")
    print(f"    <alpha> = {mean_alpha:.4f}, std = {std_alpha:.4f}")
    print(f"    <A_heavy> = {sum(s['a_heavy'] for s in mh_samples)/len(mh_samples):.1f}")

    # ===================================================================
    # SECTION 7: FTCS Fission PDE
    # ===================================================================
    separator("7. FTCS Fission PDE Solver [ftcs_fission_pde.py]")

    from ftcs_fission_pde import FTCSFissionPDE, kramers_escape_rate

    ftcs = FTCSFissionPDE(92, 236, n_grid=61, temperature=1.5)
    ftcs.initialize_gaussian(1.0, 0.05)

    print(f"\n  FTCS parameters:")
    print(f"    Grid: {ftcs.n} points, dc = {ftcs.dc:.4f}")
    print(f"    dt = {ftcs.dt:.6e}, CFL = {ftcs.D * ftcs.dt / ftcs.dc**2:.4f}")

    pde_result = ftcs.run(n_steps=200)
    print(f"\n  After {pde_result['n_steps']} steps (t={pde_result['total_time']:.6e}):")
    print(f"    Total probability: {pde_result['total_probability']:.6f}")
    print(f"    Mean elongation: {pde_result['mean_elongation']:.4f}")
    print(f"    Flux at scission: {pde_result['flux_at_scission']:.6e}")
    print(f"    Prob fissioned: {pde_result['prob_fissioned']:.6f}")

    # Kramers rate
    kr = kramers_escape_rate(5.8, 1.5, 3.0, 2.0, 5.0e21)
    print(f"\n  Kramers escape rate: {kr:.6e} s^-1")

    # ===================================================================
    # SECTION 8: FEM 1D Energy Diffusion
    # ===================================================================
    separator("8. FEM 1D Energy Diffusion [fem1d_energy_diffusion.py]")

    from fem1d_energy_diffusion import FEM1DEnergyDiffusion

    fem1d = FEM1DEnergyDiffusion(n_elements=40, length=1.6,
                                  c_min=0.9, diffusivity=0.05)
    fem1d.initialize_gaussian(1.2, 0.15, 5.0)

    fem1d_result = fem1d.run(n_steps=100)
    print(f"\n  FEM 1D results after {fem1d_result['n_steps']} steps:")
    print(f"    Total energy: {fem1d_result['total_energy']:.4f}")
    print(f"    Max energy: {fem1d_result['max_energy']:.4f}")
    print(f"    Mean position: {fem1d_result['mean_position']:.4f}")
    print(f"    CFL number: {fem1d_result['cfl_number']:.4f}")

    # ===================================================================
    # SECTION 9: Fragment Mass Distribution
    # ===================================================================
    separator("9. Fragment Mass Distribution [fragment_mass_distribution.py]")

    from fragment_mass_distribution import FragmentMassDistribution

    fmd = FragmentMassDistribution(92, 236, temperature=1.5)
    masses, yields = fmd.compute_yield_curve(75, 165)

    # Find peaks
    peaks = fmd.peak_positions(masses, yields)
    print(f"\n  Mass yield curve for U-236 fission:")
    print(f"  Peaks found: {len(peaks)}")
    for pk in peaks:
        label = "HEAVY" if pk['is_heavy'] else "LIGHT"
        print(f"    {label}: A={pk['mass']}, Y={pk['yield']:.6f}")

    # Spline interpolation
    test_mass = 133.5
    y_interp = fmd.interpolated_yield(test_mass, masses, yields)
    print(f"\n  Spline interpolation at A={test_mass}: Y = {y_interp:.6f}")

    # ===================================================================
    # SECTION 10: Fragment Energy Release
    # ===================================================================
    separator("10. Fragment Energy Release [fragment_energy_release.py]")

    from fragment_energy_release import FragmentEnergyRelease

    fer = FragmentEnergyRelease(92, 236)

    # Energy balance for typical split
    e_balance = fer.compute_energy_balance(95, 38, 141, 54, nu=3)
    print(f"\n  Energy balance for U-236 -> Sr-95 + Xe-141 + 3n:")
    for key, val in e_balance.items():
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")
        else:
            print(f"    {key}: {val}")

    # Gauss-Jacobi quadrature test
    gj_test = fer.jacobi_quadrature_test(degree_max=8, n_quad=5, alpha=0.5, beta=0.5)
    print(f"\n  Gauss-Jacobi quadrature test (n=5, alpha=0.5, beta=0.5):")
    print(f"    {'deg':>4} {'exact':>14} {'quadrature':>14} {'error':>12} {'exact?':>8}")
    for entry in gj_test:
        tag = "YES" if entry['is_exact'] else "no"
        print(f"    {entry['degree']:>4} {entry['exact']:>14.8f} "
              f"{entry['quadrature']:>14.8f} {entry['abs_error']:>12.2e} {tag:>8}")

    # ===================================================================
    # SECTION 11: Semi-supervised Fission Mode Classification
    # ===================================================================
    separator("11. Semi-supervised Fission Mode Classification [semisupervised_fission.py]")

    from semisupervised_fission import (
        FissionModeClassifier, generate_synthetic_fission_data
    )

    labelled, unlabelled = generate_synthetic_fission_data(200, noise=0.15)
    print(f"\n  Synthetic data: {len(labelled)} labelled, {len(unlabelled)} unlabelled")

    classifier = FissionModeClassifier()
    st_result = classifier.run_self_training(labelled, unlabelled,
                                              max_iterations=4, threshold=2.0)
    print(f"\n  Self-training results:")
    print(f"    Initial CV accuracy: {st_result['initial_cv_accuracy']:.4f}")
    print(f"    Self-training iterations: {st_result['n_iterations']}")
    print(f"    Final labelled count: {st_result['final_labelled']}")

    # ===================================================================
    # SECTION 12: Polynomial Resultant
    # ===================================================================
    separator("12. Polynomial Resultant for Critical Points [polynomial_resultant.py]")

    from polynomial_resultant import (
        test_resultant_exactness, find_critical_points,
        resultant_sylvester, resultant_roots
    )

    # Test resultant
    res_test = test_resultant_exactness()
    print(f"\n  Resultant tests:")
    t1 = res_test['test_no_common_root']
    print(f"    p1=x^2-1, p2=x^2-4: Res={t1['resultant_sylvester']:.4f} "
          f"(expected {t1['expected']}, error={t1['error']:.2e})")
    t2 = res_test['test_common_root']
    print(f"    p1=x^2-1, p2=x-1: Res={t2['resultant_sylvester']:.6e} "
          f"(should be ~0, near_zero={t2['is_near_zero']})")

    # Find critical points of PES
    cp_result = find_critical_points(pes, 1.0, 0.0)
    print(f"\n  PES critical points (h=1.0, alpha=0.0):")
    print(f"    Resultant(V',V''): {cp_result['resultant_sylvester']:.6e}")
    print(f"    Degenerate: {cp_result['is_degenerate']}")
    for cp in cp_result['critical_points']:
        print(f"    c={cp['c']:.4f}, V={cp['energy']:.4f} MeV, "
              f"type={cp['type']}, V''={cp['d2v']:.4f}")

    # ===================================================================
    # SECTION 13: Legendre Angular Distribution
    # ===================================================================
    separator("13. Legendre Angular Distribution [legendre_angular.py]")

    from legendre_angular import (
        FissionAngularDistribution, shifted_legendre_value,
        convergence_test_shifted_legendre
    )

    # Shifted Legendre test
    conv_test = convergence_test_shifted_legended = convergence_test_shifted_legendre(8, 0.5)
    print(f"\n  Shifted Legendre P01(n, 0.5) convergence:")
    for ct in conv_test:
        print(f"    n={ct['n']:>2}: P01={ct['shifted_value']:>12.8f}, "
              f"Pn(0)={ct['standard_at_0']:>12.8f}, diff={ct['difference']:.2e}")

    # Angular distribution
    ang_dist = FissionAngularDistribution(spin_i=3.0, k_variance=6.0, max_legendre=6)
    coeffs = ang_dist.compute_coefficients()
    anisotropy = ang_dist.anisotropy()
    print(f"\n  Angular distribution (I=3, K0^2=6):")
    print(f"    Coefficients: {[f'{c:.4f}' for c in coeffs]}")
    print(f"    Anisotropy A = W(0)/W(pi/2) = {anisotropy:.4f}")

    # Tabulate at selected angles
    ang_table = ang_dist.angular_distribution_table(7)
    print(f"\n  W(theta) at selected angles:")
    for row in ang_table:
        print(f"    theta={row['theta_deg']:>6.1f} deg: W={row['W_standard']:.6f}")

    # ===================================================================
    # SECTION 14: Levenshtein Pattern Classification
    # ===================================================================
    separator("14. Levenshtein Pattern Classification [levenshtein_pattern.py]")

    from levenshtein_pattern import (
        FissionPatternClassifier, generate_reference_patterns,
        levenshtein_distance, normalised_levenshtein
    )

    # Generate reference patterns
    ref_patterns = generate_reference_patterns()
    print(f"\n  Reference yield patterns generated: {len(ref_patterns)} systems")

    classifier_pat = FissionPatternClassifier()
    for name, yields_ref in ref_patterns.items():
        classifier_pat.add_reference(name, yields_ref,
                                      metadata={'system': name})

    # Classify a test pattern
    fmd_test = FragmentMassDistribution(92, 236, 1.5)
    _, test_yields = fmd_test.compute_yield_curve(70, 170)
    matches = classifier_pat.classify(test_yields, top_k=3)
    print(f"\n  Pattern classification for test yields:")
    for m in matches:
        print(f"    {m['system']}: d={m['edit_distance']}, "
              f"d_norm={m['normalised_distance']:.4f}")

    # Distance matrix
    dist_mat = classifier_pat.build_distance_matrix()
    print(f"\n  Pairwise distance matrix ({len(dist_mat)}x{len(dist_mat)}):")
    names = list(ref_patterns.keys())
    for i, name in enumerate(names[:4]):
        row_str = " ".join(f"{dist_mat[i][j]:.3f}" for j in range(min(4, len(names))))
        print(f"    {name:<12}: [{row_str}]")

    # ===================================================================
    # SECTION 15: 2D FEM Thermal Diffusion
    # ===================================================================
    separator("15. 2D FEM Thermal Diffusion [fem2d_thermal_diffusion.py]")

    from fem2d_thermal_diffusion import FEM2DThermalDiffusion

    fem2d = FEM2DThermalDiffusion(nx=9, nz=9, lr=8.0, lz=16.0,
                                    k_light=0.5, k_heavy=0.3)
    fem2d.initialize_temperature(3.0, 2.0)

    fem2d_result = fem2d.run(n_steps=50, dt=0.01)
    print(f"\n  2D FEM thermal diffusion results:")
    print(f"    Mesh: {fem2d.nx}x{fem2d.nz} = {fem2d.nn} nodes, {fem2d.ne} elements")
    print(f"    CFL dt_max: {fem2d_result['dt_max_cfl']:.6e}")
    print(f"    T range: [{fem2d_result['T_min']:.4f}, {fem2d_result['T_max']:.4f}]")
    print(f"    T mean: {fem2d_result['T_mean']:.4f}")
    print(f"    T_light region: {fem2d_result['T_light_region']:.4f}")
    print(f"    T_heavy region: {fem2d_result['T_heavy_region']:.4f}")

    # ===================================================================
    # SECTION 16: QAOA Fission Channel Optimisation
    # ===================================================================
    separator("16. QAOA Fission Channel Optimisation [qaoa_fission_channel.py]")

    from qaoa_fission_channel import FissionChannelOptimizer

    opt = FissionChannelOptimizer(92, 236)
    opt_result = opt.solve_and_analyse()
    optim = opt_result['optimization']

    print(f"\n  QAOA-inspired optimisation results:")
    print(f"    Candidate channels: {opt_result['n_candidate_channels']}")
    print(f"    Best QUBO energy: {optim['best_energy']:.4f}")
    print(f"    Selected channels: {optim['n_selected']}")
    constr = optim['constraints']
    print(f"    Mass conservation: A_sum={constr['a_total']}, error={constr['a_error']:.2f}")
    print(f"    Charge conservation: Z_sum={constr['z_total']}, error={constr['z_error']:.2f}")
    print(f"    Constraints satisfied: {constr['satisfied']}")

    qaoa_angles = opt_result['qaoa_best_angles']
    print(f"    QAOA optimal angles: gamma={qaoa_angles['gamma']:.4f}, "
          f"beta={qaoa_angles['beta']:.4f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    separator("SIMULATION COMPLETE")

    elapsed = time.time() - start_time
    print(f"\n  Total execution time: {elapsed:.3f} seconds")
    print(f"\n  Modules executed: 16")
    print(f"  All computations completed successfully.")
    print(f"\n  Project: PROJECT_245 Advanced")
    print(f"  Domain: Nuclear Fission Simulation")
    print(f"  Focus: Fragment Mass Distribution & Energy Release Modelling")
    print(f"  Methods: High-Order Finite Differences & Stability Analysis")
    print()


if __name__ == '__main__':
    main()
