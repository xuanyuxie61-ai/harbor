
import numpy as np
import time




from special_functions_qcd import (
    alpha_s_1loop, alpha_s_2loop, CF, CA, N_F,
    p_qq_lo, p_gq_lo, p_gg_lo, sudakov_quark,
    legendre_poly_vals, chebyshev_poly_vals, hermite_poly_vals,
    harmonic_sum, di_log, anomalous_dim_gamma_0, anomalous_dim_gamma_1,
    validate_special_functions
)
from tridiagonal_solver import (
    r83_cg_solve, r83_cyclic_reduction,
    build_dif2_r83, solve_diffusion_1d, test_tridiagonal_solvers
)
from cubature_integrator import (
    integrate_1d_composite, integrate_pyramid,
    integrate_monte_carlo, integrate_adaptive_1d, test_cubature
)
from adaptive_sampling import (
    hilbert_h_to_xyz, hilbert_xyz_to_h,
    HilbertSpatialIndex, cvt_lloyd_2d, test_adaptive_sampling
)
from dglap_pdf import (
    mellin_moment_splitting, dglap_mellin_evolve,
    pdf_initial_model, pdf_shooting_solve,
    dglap_spectral_evolve_gluon, test_dglap
)
from parton_shower import (
    Parton, generate_hard_process, run_parton_shower,
    sample_z_and_phi, shower_multiplicity_ode, test_parton_shower
)
from hadronization_ca import (
    Hadron, run_cellular_automaton_hadronization,
    boundary_word_from_jet, test_hadronization
)
from jet_reconstruction import (
    PseudoJet, cluster_jets, knapsack_optimal_subjets,
    compute_thrust, compute_sphericity, compute_jet_broadening,
    test_jet_reconstruction
)
from uncertainty_pce import (
    LegendrePCE, pce_jet_mass_uncertainty,
    pce_pdf_uncertainty, global_sensitivity_analysis, test_pce
)


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    print(f"\n--- {title} ---")


def run_full_simulation():
    
    t_start = time.time()
    rng = np.random.default_rng(2024)
    



    print_header("STEP 0: Core Library Validation")
    
    err_sf = validate_special_functions()
    print(f"  Special functions validation max error: {err_sf:.2e}")
    
    tri_test = test_tridiagonal_solvers()
    print(f"  Tridiagonal CG error: {tri_test['cg_error']:.2e}")
    print(f"  Tridiagonal CR error: {tri_test['cr_error']:.2e}")
    
    test_cubature()
    print("  Cubature integrator tests: PASSED")
    
    test_adaptive_sampling()
    print("  Adaptive sampling tests: PASSED")
    
    test_dglap()
    print("  DGLAP PDF tests: PASSED")
    
    test_parton_shower()
    print("  Parton shower tests: PASSED")
    
    test_hadronization()
    print("  Hadronization CA tests: PASSED")
    
    test_jet_reconstruction()
    print("  Jet reconstruction tests: PASSED")
    
    test_pce()
    print("  PCE uncertainty tests: PASSED")
    



    print_header("STEP 1: DGLAP Parton Distribution Evolution")
    
    x_grid = np.logspace(-3, -0.05, 40)
    Q20 = 1.69
    Q2_final = 10000.0
    
    q_pdf, g_pdf, pdf_info = pdf_shooting_solve(
        x_grid, Q20=Q20, Q2_final=Q2_final, target_momentum=0.95, nf=5
    )
    
    print_subheader("Shooting Method Results")
    print(f"  Optimal normalization A = {pdf_info['A_opt']:.4f}")
    print(f"  Iterations = {pdf_info['iterations']}")
    print(f"  Final momentum sum rule residual = {pdf_info['final_residual']:.2e}")
    
    print_subheader("PDF Values at Q^2 = 10^4 GeV^2")
    for x_test in [1e-3, 1e-2, 0.1, 0.5]:
        print(f"  x = {x_test:.0e}:  x·q(x) = {q_pdf(x_test):.4f},  x·g(x) = {g_pdf(x_test):.4f}")
    

    g0_func = lambda x: np.maximum(2.0 * x**(-0.3) * (1.0 - x)**5, 1e-15)
    g_evolved = dglap_spectral_evolve_gluon(g0_func, x_grid, Q20, Q2_final, nf=5)
    print_subheader("Spectral ETDRK4 Gluon Evolution")
    print(f"  Initial gluon integral:  {np.trapezoid(g0_func(x_grid), x_grid):.4f}")
    print(f"  Evolved gluon integral:  {np.trapezoid(g_evolved, x_grid):.4f}")
    



    print_header("STEP 2: Hard Scattering Process Generation")
    
    E_cm = 14000.0
    pt_hard = 80.0
    
    hard_partons = generate_hard_process(E_cm=E_cm, pt_hard=pt_hard, seed=42)
    print_subheader("Initial Hard Partons")
    for p in hard_partons:
        print(f"  PID={p.pid}, flavor={p.flavor}, "
              f"pT={p.pt:.2f} GeV, η={p.eta:.3f}, φ={p.phi:.3f}, E={p.p[3]:.2f} GeV")
    



    print_header("STEP 3: Angular-Ordered Parton Shower (Markov Chain MC)")
    
    Q_cut = 1.0
    z_cut = 0.05
    
    shower_partons, shower_history = run_parton_shower(
        hard_partons, Q_cut=Q_cut, z_cut=z_cut,
        max_multiplicity=150, seed=42
    )
    
    print_subheader("Shower Statistics")
    print(f"  Initial multiplicity: {len(hard_partons)}")
    print(f"  Final parton multiplicity: {len(shower_partons)}")
    print(f"  Number of emissions: {len(shower_history)}")
    
    total_E_partons = sum(p.p[3] for p in shower_partons)
    total_px = sum(p.p[0] for p in shower_partons)
    total_py = sum(p.p[1] for p in shower_partons)
    print(f"  Total parton energy: {total_E_partons:.2f} GeV")
    print(f"  Total parton p_x:    {total_px:.3f} GeV")
    print(f"  Total parton p_y:    {total_py:.3f} GeV")
    

    sol_mult = shower_multiplicity_ode((0.0, np.log(100.0)), len(hard_partons),
                                       alpha=0.4, beta=0.02)
    print_subheader("ODE Multiplicity Prediction")
    print(f"  Predicted final multiplicity: {sol_mult.y[0, -1]:.1f}")
    



    print_header("STEP 4: CVT Adaptive Sampling of Parton Phase Space")
    
    def parton_density(x, y):
        return np.exp(-2.0 * (x**2 + y**2)) + 0.05
    
    cvt_gens, cvt_info = cvt_lloyd_2d(
        n_generators=32, density_func=parton_density,
        n_samples=10000, max_iter=30, tol=1e-4, seed=42,
        domain=[(-1.0, 1.0), (-1.0, 1.0)]
    )
    print(f"  CVT generators: {cvt_gens.shape[0]}")
    print(f"  Lloyd iterations: {cvt_info['iterations']}")
    print(f"  Final displacement: {cvt_info['final_displacement']:.2e}")
    



    print_header("STEP 5: Cellular Automaton Hadronization")
    
    hadrons, clusters = run_cellular_automaton_hadronization(
        shower_partons, R_cone=0.4, pt_min=0.3, sigma_kT=0.35, seed=42
    )
    
    print_subheader("Hadronization Output")
    print(f"  Number of string clusters: {len(clusters)}")
    print(f"  Total hadrons produced: {len(hadrons)}")
    print(f"  Charged hadron multiplicity: {sum(1 for h in hadrons if abs(h.charge) > 0)}")
    
    total_E_hadrons = sum(h.p[3] for h in hadrons)
    print(f"  Total hadron energy: {total_E_hadrons:.2f} GeV")
    print(f"  Energy conservation: {(total_E_hadrons/total_E_partons - 1)*100:.2f}%")
    

    if len(hadrons) > 5:
        word, c_eta, c_phi = boundary_word_from_jet(hadrons[:20])
        print_subheader("Leading Jet Boundary Word")
        print(f"  Centroid (η, φ) = ({c_eta:.3f}, {c_phi:.3f})")
        print(f"  Boundary word length: {len(word)}")
    



    print_header("STEP 6: Jet Reconstruction (Anti-kT Algorithm)")
    
    pseudo_jets = [PseudoJet(h.p[0], h.p[1], h.p[2], h.p[3], index=i)
                   for i, h in enumerate(hadrons)]
    
    R_jet = 0.4
    pT_min_jet = 5.0
    
    jets_akt = cluster_jets(pseudo_jets, R=R_jet, p=-1, pt_min=pT_min_jet)
    jets_ca = cluster_jets(pseudo_jets, R=R_jet, p=0, pt_min=pT_min_jet)
    jets_kt = cluster_jets(pseudo_jets, R=R_jet, p=1, pt_min=pT_min_jet)
    
    print_subheader("Clustering Results")
    print(f"  Anti-kT jets found:  {len(jets_akt)}")
    print(f"  C/A jets found:      {len(jets_ca)}")
    print(f"  kT jets found:       {len(jets_kt)}")
    
    if len(jets_akt) > 0:
        print_subheader("Leading Anti-kT Jet Properties")
        lead = jets_akt[0]
        print(f"    pT  = {lead.pt:.2f} GeV")
        print(f"    η   = {lead.eta:.3f}")
        print(f"    φ   = {lead.phi:.3f}")
        print(f"    Mass = {lead.mass:.3f} GeV")
        print(f"    E   = {lead.E:.2f} GeV")
        

        best_mass, best_subset = knapsack_optimal_subjets(lead, n_subjets_target=2)
        print_subheader("Knapsack Subjet Reconstruction")
        print(f"    Target Z-boson mass: 91.1876 GeV")
        print(f"    Best reconstructed mass: {best_mass:.3f} GeV")
        print(f"    Number of constituents in optimal subset: {len(best_subset)}")
    



    print_header("STEP 7: Jet Shape and Event Shape Analysis")
    
    if len(pseudo_jets) > 0:
        thrust_val, thrust_axis = compute_thrust(pseudo_jets)
        spher = compute_sphericity(pseudo_jets)
        broadening = compute_jet_broadening(pseudo_jets, thrust_axis)
        
        print_subheader("Event Shapes")
        print(f"  Thrust T = {thrust_val:.4f}")
        print(f"  Sphericity S = {spher['S']:.4f}")
        print(f"  Aplanarity A = {spher['A']:.4f}")
        print(f"  Jet Broadening B = {broadening:.4f}")
        print(f"  Eigenvalues: {spher['eigenvalues']}")
    



    print_header("STEP 8: Polynomial Chaos Expansion Uncertainty Quantification")
    

    def shower_for_alpha(alpha_s_val):


        n_had_eff = int(len(hadrons) * (0.5 + 0.5 * alpha_s_val / 0.12))
        if n_had_eff < 5:
            return 0.0

        E_avg = total_E_hadrons / max(len(hadrons), 1)
        m_eff = E_avg * np.sqrt(n_had_eff) * 0.05
        return float(m_eff)
    
    pce_mass, samples_mass = pce_jet_mass_uncertainty(
        shower_for_alpha, alpha_s_range=(0.10, 0.14), order=5, n_samples=50, seed=42
    )
    
    print_subheader("Jet Mass vs α_s (PCE)")
    print(f"  Mean jet mass:     {pce_mass.mean():.3f} GeV")
    print(f"  Std deviation:     {pce_mass.std():.3f} GeV")
    print(f"  Variance:          {pce_mass.variance():.4f} GeV^2")
    

    def pdf_model(x, lam):
        return max(0.5 * x**(-lam) * (1.0 - x)**4, 1e-15)
    
    pce_pdf = pce_pdf_uncertainty(pdf_model, x_value=0.1, param_range=(0.25, 0.35), order=5)
    print_subheader("PDF Uncertainty at x=0.1")
    print(f"  Mean x·q(x):       {pce_pdf.mean():.4f}")
    print(f"  Std deviation:     {pce_pdf.std():.4f}")
    

    def jet_mass_model(params):
        alpha = params.get('alpha_s', 0.12)
        qcut = params.get('Q_cut', 1.0)
        zcut = params.get('z_cut', 0.05)

        return 5.0 + 30.0 * alpha / 0.12 + 2.0 * (qcut - 1.0) + 1.0 * (zcut - 0.05) / 0.01
    
    gs_results = global_sensitivity_analysis(
        jet_mass_model,
        param_ranges={'alpha_s': (0.10, 0.14), 'Q_cut': (0.5, 2.0), 'z_cut': (0.01, 0.10)},
        order=3, n_mc=3000, seed=42
    )
    
    print_subheader("Global Sensitivity Analysis (Sobol-like)")
    print(f"  Mean predicted mass: {gs_results['mean']:.3f} GeV")
    print(f"  Std deviation:       {gs_results['std']:.3f} GeV")
    for name, idx in gs_results['sobol_indices'].items():
        print(f"  Sensitivity index [{name}]: {idx:.4f}")
    



    print_header("STEP 9: High-Dimensional Momentum Space Integrals")
    

    def energy_profile(x, y, z):
        r2 = x**2 + y**2
        return np.exp(-3.0 * r2) * z * (1.0 - z)
    
    E_pyramid = integrate_pyramid(energy_profile, precision=3)
    print(f"  Pyramid energy integral: {E_pyramid:.6f}")
    

    def phase_space_element(p):
        E, px, py, pz = p
        if E < 0:
            return 0.0
        m2 = E**2 - px**2 - py**2 - pz**2
        return max(m2, 0.0) * np.exp(-E / 10.0)
    
    ps_integral, ps_error = integrate_monte_carlo(
        phase_space_element,
        domain=[(0.0, 50.0), (-20.0, 20.0), (-20.0, 20.0), (-50.0, 50.0)],
        n_samples=20000, seed=42
    )
    print(f"  4D phase-space MC integral: {ps_integral:.3f} ± {ps_error:.3f}")
    

    def splitting_integral(z):
        return (p_qq_lo(z) + p_gq_lo(z)) / (z * (1.0 - z))
    
    split_int = integrate_adaptive_1d(splitting_integral, 1e-4, 1.0 - 1e-4, tol=1e-4)
    print(f"  Splitting function integral: {split_int:.3f}")
    



    print_header("STEP 10: Jet Quenching Diffusion in QGP Medium")
    

    nx_diff = 128
    x_diff = np.linspace(0.0, 5.0, nx_diff)
    dx_diff = x_diff[1] - x_diff[0]
    

    u0 = np.exp(-2.0 * (x_diff - 2.5)**2)
    D_medium = 0.5
    dt_diff = 0.01
    n_steps_diff = 100
    
    u_final = solve_diffusion_1d(u0, D_medium, dt_diff, dx_diff, n_steps_diff, solver='cyclic')
    
    E_initial = np.trapezoid(u0, x_diff)
    E_final = np.trapezoid(u_final, x_diff)
    print(f"  Initial energy profile integral: {E_initial:.4f}")
    print(f"  Final energy profile integral:   {E_final:.4f}")
    print(f"  Energy loss fraction:            {(1.0 - E_final/E_initial)*100:.2f}%")
    



    print_header("STEP 11: Hilbert Curve Spatial Index for Momentum Space")
    
    if len(hadrons) > 10:
        hilbert = HilbertSpatialIndex(
            n_bits=8,
            bbox=[(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)]
        )
        for i, h in enumerate(hadrons):
            hilbert.add_point(h.p[:3], i)
        hilbert.build_index()
        
        center = np.array([0.0, 0.0, 0.0])
        radius = 20.0
        candidates = hilbert.range_query(center, radius)
        print(f"  Total hadrons indexed: {len(hadrons)}")
        print(f"  Candidates within R={radius} of origin: {len(candidates)}")
    



    print_header("SIMULATION COMPLETE")
    t_elapsed = time.time() - t_start
    print(f"  Total execution time: {t_elapsed:.2f} seconds")
    print(f"  Final parton multiplicity: {len(shower_partons)}")
    print(f"  Final hadron multiplicity: {len(hadrons)}")
    print(f"  Reconstructed anti-kT jets: {len(jets_akt)}")
    print(f"  Leading jet pT: {jets_akt[0].pt:.2f} GeV" if len(jets_akt) > 0 else "  No jets reconstructed")
    print("\n  All steps completed successfully. ✓\n")


if __name__ == "__main__":
    run_full_simulation()
