"""
main.py
=======
PROJECT_224 :: Computational HEP - Higgs Decay Signal Strength Fitting
High-Order Finite Differences & Stability Analysis (Small-scale reproducible experiment)

This is the unified entry point. It runs the full analysis pipeline:

    1. SM constants & Higgs effective potential V(phi)
    2. Multi-channel signal strength fit (mu_i for 5 Higgs channels)
    3. High-order FD Hessian of profile likelihood
    4. Stability analysis (Hessian + vacuum)
    5. Fisher information matrix & uncertainties
    6. Monte Carlo coverage test over confidence ellipse
    7. RG flow of lambda(mu) with ETDRK4 (m_t -> 10^10 GeV)
    8. Phase-space integrals for Higgs partial widths
    9. Channel selection & referral decision pipeline
   10. Topology edit distances between channels
   11. Benchmark of FD convergence + Feynman-Kac cross-check
   12. Signal migration via advection-diffusion

Run with zero arguments:
    $ python main.py
"""
from __future__ import annotations
import sys
import time
import numpy as np

from sm_constants import SMConstants
from higgs_potential import HiggsPotential
from phase_space import PhaseSpaceIntegrator
from signal_strength import SignalStrengthFitter, SignalMigration
from finite_difference import HighOrderFD
from stability_analysis import StabilityAnalyzer
from fisher_information import FisherMatrix
from monte_carlo_confidence import MonteCarloConfidence
from rg_flow import RGFlowSolver
from channel_selector import ChannelSelector
from topology_distance import TopologyDistance
from benchmark import BenchmarkRunner


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title.center(72))
    print("=" * 72)


def _sub(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 66 - len(title)))


def main() -> int:
    t_start = time.time()
    np.random.seed(224)

    _banner("PROJECT_224 :: Higgs Decay Signal Strength Fitting")
    print("   High-Order Finite Differences & Stability Analysis")
    print("   Computational High-Energy Physics (small-scale reproducible experiment)")

    # ================================================================== #
    # Step 1: SM Constants and Higgs potential                           #
    # ================================================================== #
    _sub("1/12  SM Constants & Higgs effective potential")
    sm = SMConstants()
    print(sm.summary())

    pot = HiggsPotential(sm)
    v_rec = pot.find_vev_bisection()
    curv = pot.curvature_at_minimum()
    lam_v = float(pot.lambda_eff(np.array([sm.v]))[0])
    barrier = pot.barrier_height()
    print(f"  Recovered VEV          : v = {v_rec:.6f} GeV  (input {sm.v:.4f})")
    print(f"  V''(v)                 : {curv:.4f} GeV^2   (m_h^2 = {sm.m_h**2:.2f})")
    print(f"  lambda_eff(v)          : {lam_v:.6f}")
    print(f"  EW phase-transition    : barrier height = {barrier:.4e} GeV^4")

    # ================================================================== #
    # Step 2: Signal-strength best fit                                   #
    # ================================================================== #
    _sub("2/12  Multi-channel signal-strength fit")
    channels = ["ggH_bb", "ggH_tautau", "VBF_hww", "VH_zz", "ttH_gammagamma"]
    fitter = SignalStrengthFitter(channels, sm=sm)
    fitter.generate_asimov_data()
    result = fitter.profile_likelihood_fit()
    print(f"  Channels fitted : {len(channels)}")
    print(f"  Best-fit mu_hat : {np.array2string(result.mu_hat, precision=4)}")
    print(f"  -ln L_min       : {result.nll_min:.4f}")

    # ================================================================== #
    # Step 3: High-order FD Hessian                                      #
    # ================================================================== #
    _sub("3/12  High-order FD Hessian of profile likelihood")
    fd = HighOrderFD(max_order=8)

    def nll_profile(mu_vec):
        return -fitter.profile_over_theta(np.asarray(mu_vec))

    H = fd.hessian(nll_profile, result.mu_hat, h=5e-3, order=4)
    print(f"  Hessian shape: {H.shape}")
    print(f"  Hessian diagonal: {np.array2string(np.diag(H), precision=3)}")

    # ================================================================== #
    # Step 4: Stability analysis                                         #
    # ================================================================== #
    _sub("4/12  Stability analysis")
    analyzer = StabilityAnalyzer()
    stability = analyzer.analyze(H, pot)
    print(stability)
    print(f"  VEV is minimum         : {stability.vacuum_details['vev_is_minimum']}")
    print(f"  Bounded below          : {stability.vacuum_details['bounded_below']}")
    print(f"  EW vacuum deeper       : {stability.vacuum_details['ew_vacuum_deeper']}")
    print(f"  Hessian pos. definite  : {stability.hessian_pos_def}")

    # ================================================================== #
    # Step 5: Fisher information                                         #
    # ================================================================== #
    _sub("5/12  Fisher information matrix & uncertainties")
    fisher = FisherMatrix(fitter)
    F = fisher.asimov_fisher(result.mu_hat)
    sigma_mu = fisher.uncertainties(F)
    rho = fisher.correlation_matrix(F)
    print(f"  Fisher matrix F (diag): {np.array2string(np.diag(F), precision=3)}")
    print(f"  sigma(mu_i)           : {np.array2string(sigma_mu, precision=4)}")
    print(f"  Correlation matrix diag: {np.array2string(np.diag(rho), precision=3)}")

    # Hyperspherical parameterization
    hyper = FisherMatrix.cartesian_to_hyperspherical(result.mu_hat)
    R, *angles = hyper
    solid_angle = FisherMatrix.solid_angle_density(hyper[1:])
    print(f"  Hyperspherical R = |mu|: {R:.4f}")
    print(f"  Solid angle density dOmega: {solid_angle:.4f}")

    # ================================================================== #
    # Step 6: Monte Carlo coverage                                       #
    # ================================================================== #
    _sub("6/12  Monte Carlo coverage test")
    mc = MonteCarloConfidence(F, result.mu_hat, seed=224)
    for cl in (0.68, 0.95):
        cov = mc.estimate_coverage(cl=cl, n_samples=5000)
        print(f"  CL = {cl:.2f}  empirical coverage = {cov:.4f}  (target {cl:.2f})")

    area_2d = mc.ellipse_area_2d(axes=(0, 1), alpha=0.95)
    print(f"  2D 95% CL ellipse area (ggH_bb vs ggH_tautau): {area_2d:.3f}")

    # Diffusion smearing of 1D likelihood
    mu_grid = np.linspace(0.5, 1.5, 60)
    L_smeared = mc.diffuse_likelihood(mu_grid, sigma_diff=0.05)
    print(f"  Diffusion-smeared likelihood max: {L_smeared.max():.4f}")

    # ================================================================== #
    # Step 7: RG flow (ETDRK4)                                           #
    # ================================================================== #
    _sub("7/12  Renormalization group flow with ETDRK4")
    rg = RGFlowSolver(
        g_prime=sm.gp, g=sm.g, gs=sm.gs, y_t=sm.y_t, lam=sm.lam_tree
    )
    rg_result = rg.solve(mu_low=sm.m_t, mu_high=1e10, n_steps=400)
    print(f"  mu range             : [{sm.m_t:.1f}, 1e10] GeV")
    print(f"  lambda(m_t)          : {sm.lam_tree:.6f}")
    print(f"  lambda(10^10 GeV)    : {rg_result['lambda_at_high']:.6f}")
    print(f"  lambda turns neg.?   : {rg_result['lambda_turns_negative']}")
    if rg_result["lambda_turns_negative"] is not None:
        print(f"    at mu ~ {rg_result['lambda_turns_negative']:.3e} GeV")
    print(f"  y_t(10^10 GeV)       : {rg_result['y_t_at_high']:.6f}")

    # Lyapunov exponent of the RG trajectory
    lyap = StabilityAnalyzer.lyapunov_from_trajectory(rg_result["trajectory"], dt=0.1)
    print(f"  Lyapunov exponent    : {lyap:.4e}")

    # ================================================================== #
    # Step 8: Phase-space integrals                                      #
    # ================================================================== #
    _sub("8/12  Higgs partial widths & branching ratios")
    ps = PhaseSpaceIntegrator(sm)
    br = ps.branching_ratios()
    print(f"  Gamma(H->bb)         : {ps.gamma_bb():.4e} GeV")
    print(f"  Gamma(H->tautau)     : {ps.gamma_tautau():.4e} GeV")
    print(f"  Gamma(H->gammagamma) : {ps.gamma_gammagamma():.4e} GeV")
    print(f"  Gamma(H->gg)         : {ps.gamma_gg():.4e} GeV")
    print(f"  Gamma(H->ZZ*)        : {ps.gamma_ZZ_star():.4e} GeV")
    print(f"  Total width          : {ps.total_width():.4e} GeV")
    for k, v in br.items():
        print(f"    BR(H->{k:10s}) = {v:.4e}")

    # Dalitz polygon area (H -> Z l+ l- nu nu, treating as 3-body)
    dalitz = ps.dalitz_polygon_area(m1=0.0, m2=0.0, m3=sm.m_Z)
    print(f"  Dalitz polygon area (massless 3-body proxy): {dalitz:.4e} GeV^4")

    # ================================================================== #
    # Step 9: Channel selector / referral                                #
    # ================================================================== #
    _sub("9/12  Channel decision pipeline & referral")
    selector = ChannelSelector(channels)
    ranking = selector.rank_by_sensitivity(result, sigma_mu)
    for r in ranking:
        print(
            f"  {r['channel']:20s}  mu_hat = {r['mu_hat']:.3f} +/- {r['sigma_mu']:.3f}"
            f"  S = {r['significance']:6.2f}  eps = {r['relative_precision']:.3f}"
            f"  -> {r['decision'].upper()}"
        )
    S_arr = np.array([r["significance"] for r in ranking])
    S_comb = selector.combined_significance(S_arr, mode="serial")
    print(f"  Combined significance (serial) = {S_comb:.2f}")

    # ================================================================== #
    # Step 10: Topology edit distance                                    #
    # ================================================================== #
    _sub("10/12  Topology edit distances")
    topo = TopologyDistance()
    pairwise = topo.pairwise_matrix(channels, sigma=0.3)
    D = pairwise["distance"]
    K = pairwise["similarity"]
    print("  Pairwise edit distances (normalized):")
    for i, chi in enumerate(channels):
        row = "  " + chi.ljust(20) + " "
        row += " ".join(f"{D[i,j]:.2f}" for j in range(len(channels)))
        print(row)
    print(f"  Mean similarity kernel value: {K.mean():.4f}")

    # ================================================================== #
    # Step 11: Benchmark                                                 #
    # ================================================================== #
    _sub("11/12  FD convergence benchmark & Feynman-Kac cross-check")
    bench = BenchmarkRunner(fd, fitter)
    rates = bench.convergence_rates()
    print("  FD order  -> observed convergence order")
    for k, v in rates.items():
        if k.startswith("order_"):
            print(f"    {k:10s}  -> {v:.3f}")
    print(f"  Average observed order : {rates['observed_order']:.3f}")

    fk = bench.feynman_kac_likelihood_check(
        mu_center=1.0, sigma=0.2, n_paths=500, n_steps=100, T=0.1, mu_eval=1.0
    )
    print(f"  Feynman-Kac mean     : {fk['fk_mean']:.4f}")
    print(f"  FK analytical (small-T): {fk['analytical_small_T']:.4f}")
    print(f"  FK ratio             : {fk['ratio']:.4f}")

    fisher_cond = bench.fisher_condition_vs_channels((2, 3, 4, 5))
    print("  Fisher cond # vs n_channels:")
    for n, d in fisher_cond.items():
        print(f"    n = {n}  kappa = {d['condition_number']:.3e}")

    # ================================================================== #
    # Step 12: Signal migration                                          #
    # ================================================================== #
    _sub("12/12  Signal migration via advection-diffusion")
    migration = SignalMigration(n_grid=80, D=0.01, v_adv=0.05)
    p = migration.solve_steady()
    frac = migration.migration_fraction(mu_window=(0.9, 1.1))
    print(f"  Peak of steady profile    : {p.max():.4f}")
    if hasattr(np, "trapezoid"):
        _integral = float(np.trapezoid(p, migration.mu_grid))
    else:
        _integral = float(np.trapz(p, migration.mu_grid))
    print(f"  Integral of profile       : {_integral:.4f}")
    print(f"  Fraction in [0.9, 1.1]    : {frac:.4f}")

    # ================================================================== #
    # Summary                                                            #
    # ================================================================== #
    _banner("SUMMARY")
    print(f"  Best-fit signal strengths  : {np.array2string(result.mu_hat, precision=4)}")
    print(f"  Fisher uncertainties       : {np.array2string(sigma_mu, precision=4)}")
    print(f"  Hessian condition number   : {stability.condition_number:.3e}")
    print(f"  Vacuum stable (tree+1L)    : {stability.vacuum_stable}")
    print(f"  lambda(10^10 GeV)          : {rg_result['lambda_at_high']:.6f}")
    print(f"  Combined S (serial)        : {S_comb:.2f}")
    print(f"  FD observed order          : {rates['observed_order']:.3f}")
    print(f"  MC 95% CL coverage         : {mc.estimate_coverage(cl=0.95, n_samples=5000):.4f}")
    print(f"  Wall-clock time            : {time.time()-t_start:.2f} s")
    _banner("END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
