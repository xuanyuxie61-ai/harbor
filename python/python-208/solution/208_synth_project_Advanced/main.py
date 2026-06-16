"""
main.py
=======

Unified entry point for the multi-fidelity uncertainty-quantification
project.  Zero-argument invocation:

    $ python main.py

Performs the full pipeline:
  1. Initialize the protoplanetary-disk physical system and fidelity hierarchy.
  2. Build the Lagrange tensor-product surrogate (LEVEL_1) on a coarse grid.
  3. Verify the circle quadrature, ODE chemistry, and banded solvers.
  4. Run the adaptive multi-fidelity sampling loop.
  5. Fit the NAR-GP predictor with denoised training labels.
  6. Construct CLP-style confidence intervals on a validation grid.
  7. Compute azimuthal integrals of the QoI over the disk.
  8. Evolve the molecular-dynamics potential with metadynamics bias.
  9. Project all samples into the 4-D latent state space.
  10. Summarize the experiment log via the filename-sequence tracker.
  11. Print a comprehensive diagnostic report.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Tuple

# ---- project modules ------------------------------------------------
from physical_system import (
    AU_CGS,
    ChemistryParams,
    DiskParams,
    DiskState,
    PlanetParams,
    build_disk_state,
    column_co_abundance,
    evaluate_high_fidelity,
)
from fidelity_manager import FidelityManager
from lagrange_surrogate import (
    build_lagrange_calibration,
    chebyshev_nodes,
    lagrange_value_1d,
    tensor_lebesgue,
)
from circle_quadrature import (
    angular_integral,
    circle_monomial_integral,
    circle_rule,
)
from ode_chemistry import (
    DEFAULT_REACTIONS,
    exp_ode_unit_test,
    integrate_chemistry,
)
from banded_covariance import (
    banded_cg,
    banded_fa,
    banded_mv,
    banded_sl,
    gp_covariance_banded,
    matern32,
    _cholesky_band_textbook,
)
from adaptive_fidelity_sampler import (
    FidelitySample,
    icd_batch_selection,
    run_adaptive_sampling,
)
from coordinate_transform import (
    build_whitening,
    encode,
    decode,
    ge_to_ccs,
    log_prior_pdf,
)
from confidence_bounds import (
    fit_bias_correction,
    gaussian_pi,
    mf_prediction_interval,
)
from denoising_filter import (
    denoise_training_labels,
    median_filter_1d,
)
from temporal_scheduler import (
    DEFAULT_CADENCE,
    assign_fidelity_level,
    schedule_fidelity_batch,
    weekday_diagnostic,
)
from experiment_tracker import (
    ExperimentTracker,
    filename_inc,
    sample_log_filename,
)
from latent_state import (
    LatentCloud,
    LatentPoint,
    latent_distance,
    latent_summary,
    project_to_latent,
)
from molecular_dynamics import (
    MetaDConfig,
    MetaDState,
    bias_potential,
    deepmd_local_potential,
    run_md_relaxation,
)
from multi_fidelity_gp import (
    NARGPPredictor,
    build_nargp_data,
    MultiFidelityPredictor,
)


# ----------------------------------------------------------------------
# Helper: formatted section header.
# ----------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ----------------------------------------------------------------------
# Main pipeline.
# ----------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("MULTI-FIDELITY UNCERTAINTY QUANTIFICATION PIPELINE")
    print("Protoplanetary-disk chemistry with autoregressive GP fusion")
    print(f"Python {sys.version.split()[0]}")

    # ==================================================================
    section("1. Physical-system initialization")
    # ==================================================================
    disk = DiskParams()
    planet = PlanetParams()
    chem = ChemistryParams()
    t_sim = 0.5e6 * 3.15576e7   # 0.5 Myr in seconds
    state_ref = build_disk_state(disk, planet, chem, t_sim, n_r=32)
    print(f"  Reference disk state assembled at t = 0.5 Myr")
    print(f"  n_r = {len(state_ref.r_grid_cm)}")
    print(f"  a_p = {planet.a_p0 / AU_CGS:.2f} AU")
    print(f"  M_p = {planet.M_p:.3e} g")
    print(f"  QoI (column CO abundance) = {column_co_abundance(state_ref):.3e}")

    # ==================================================================
    section("2. Fidelity hierarchy initialization")
    # ==================================================================
    fm = FidelityManager(planet, chem, t_sim)
    for l in range(fm.n_levels):
        tag = fm.describe(l)
        print(f"  Level {l}: {tag.fidelity_class}, tags={tag.physics_tags}")
        print(f"           cost_ratio={fm.cost(l):.1f}, "
              f"rho(l,3)={fm.correlation(l, 3):.2f}")

    # Reference xi: fiducial parameter vector.
    xi_fid = [-2.0, -1.523, -2.0, 0.6]
    y_ref = fm.evaluate(xi_fid, 3)
    print(f"  Reference QoI at xi_fid = {xi_fid}: F_3 = {y_ref:.4f}")

    # ==================================================================
    section("3. Lagrange surrogate calibration (LEVEL_1)")
    # ==================================================================
    bounds = [(-3.0, -0.5), (-3.0, -1.0), (-3.0, -0.5), (0.2, 1.0)]
    n_per_dim = 3
    calib = build_lagrange_calibration(
        lambda xi: fm.evaluate(xi, 2), bounds, n_per_dim
    )
    print(f"  Tensor grid: {len(calib['nodes'])} nodes, n_per_dim={n_per_dim}")
    print(f"  Lebesgue constant (tensor, d=4, n={n_per_dim}): "
          f"{tensor_lebesgue(4, n_per_dim):.3f}")
    # Inject into fidelity manager.
    fm.surrogate_calibration = calib
    y_l1 = fm.evaluate(xi_fid, 1)
    print(f"  LEVEL_1 surrogate at xi_fid: {y_l1:.4f} (ref {y_ref:.4f})")

    # ==================================================================
    section("4. Circle quadrature verification")
    # ==================================================================
    for p, q in [(0, 0), (2, 0), (4, 0), (2, 2), (4, 2)]:
        exact = circle_monomial_integral(p, q)
        def f(theta, p=p, q=q):
            return (math.cos(theta) ** p) * (math.sin(theta) ** q)
        num = angular_integral(f, "trapezoidal", n=16)
        print(f"  I(cos^{p} sin^{q}): exact={exact:.6f}, numerical={num:.6f}, "
              f"err={abs(num - exact):.2e}")

    # ==================================================================
    section("5. Chemical ODE integrator verification")
    # ==================================================================
    res = exp_ode_unit_test(alpha=-0.5, y0=1.0, tstop=1.0)
    print(f"  exp_ode test (alpha=-0.5): numerical={res['numerical']:.6f}, "
          f"exact={res['exact']:.6f}, rel_err={res['rel_error']:.2e}")
    # Short chemical network integration.
    n0 = [1.0e6, 1.0e-4 * 1.0e6, 1.0e-6 * 1.0e6,
          1.0e-7 * 1.0e6, 1.0e-8 * 1.0e6, 1.0e-9 * 1.0e6]
    n_out, times = integrate_chemistry(
        n0, T=100.0, t_total=1.0e5, method="rk4", dt_init=1.0e2, rtol=1.0e-3
    )
    print(f"  Chemical network integration (T=100 K, t=1e5 s): {len(times)} steps")
    print(f"  Final densities (cm^-3): {[f'{n:.2e}' for n in n_out]}")

    # ==================================================================
    section("6. Banded covariance solver verification")
    # ==================================================================
    n_grid = 32
    x_grid = [i * 0.1 for i in range(n_grid)]
    AB = gp_covariance_banded(x_grid, ell=0.5, sigma_f=1.0, sigma_n=0.01, mu=8)
    # Manufacture a solution.
    x_true = [math.sin(2.0 * math.pi * xi / (n_grid * 0.1)) for xi in x_grid]
    b = banded_mv(n_grid, 8, AB, x_true)
    x_cg, iters_cg, res_cg = banded_cg(n_grid, 8, AB, b, tol=1.0e-10, maxit=200)
    err_cg = math.sqrt(sum((x_cg[i] - x_true[i]) ** 2 for i in range(n_grid)))
    print(f"  CG solve: {iters_cg} iterations, residual={res_cg:.2e}, "
          f"err={err_cg:.2e}")
    # Dense Cholesky (for small n, using standard formula).
    def dense_cholesky(A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for j in range(n):
            s = A[j][j]
            for k in range(j):
                s -= L[j][k] ** 2
            if s <= 0.0:
                s = max(s, 1.0e-12)
            L[j][j] = math.sqrt(s)
            for i in range(j + 1, n):
                s = A[i][j]
                for k in range(j):
                    s -= L[i][k] * L[j][k]
                L[i][j] = s / L[j][j]
        return L
    # Build full dense matrix for small test.
    A_full = [[0.0] * n_grid for _ in range(n_grid)]
    for i in range(n_grid):
        for j in range(i, n_grid):
            v = matern32(x_grid[j] - x_grid[i], 0.5, 1.0)
            A_full[i][j] = v
            A_full[j][i] = v
            if i == j:
                A_full[i][j] += 0.01 ** 2
    L = dense_cholesky(A_full)
    # Manufacture a new RHS consistent with A_full.
    b_full = [sum(A_full[i][j] * x_true[j] for j in range(n_grid))
              for i in range(n_grid)]
    # Solve Ly = b then L^T x = y.
    y = [0.0] * n_grid
    for i in range(n_grid):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i]
    x_dense = [0.0] * n_grid
    for i in range(n_grid - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n_grid):
            s -= L[k][i] * x_dense[k]
        x_dense[i] = s / L[i][i]
    err_dense = math.sqrt(sum((x_dense[i] - x_true[i]) ** 2 for i in range(n_grid)))
    print(f"  Dense Cholesky solve: err={err_dense:.2e}")

    # ==================================================================
    section("7. Adaptive multi-fidelity sampling")
    # ==================================================================
    tracker = ExperimentTracker("mf_run_000")
    # Generate initial samples for whitening.
    import random
    rng = random.Random(42)
    xi_init = [[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(16)]
    W = build_whitening(xi_init)
    print(f"  Whitening transform built from {len(xi_init)} samples")
    print(f"  Prior log-pdf at xi_fid: {log_prior_pdf(xi_fid, W):.3f}")
    # Run adaptive sampling.
    mf_pred_init = lambda xi: (0.0, 1.0)  # placeholder predictor
    budget = 500.0
    samples = run_adaptive_sampling(
        fm, mf_pred_init, budget=budget,
        n_init=8, n_iter=6, seed=42, bounds=bounds,
    )
    print(f"  Adaptive sampling: {len(samples)} samples, "
          f"budget used = {sum(s.cost for s in samples):.1f}")
    for l in range(fm.n_levels):
        n_l = sum(1 for s in samples if s.level == l)
        print(f"    Level {l}: {n_l} samples")
    # Record in experiment tracker.
    run_name = tracker.next({"n_samples": len(samples)})
    print(f"  Experiment '{run_name}' recorded")

    # ==================================================================
    section("8. NAR-GP predictor fitting")
    # ==================================================================
    nargp_data = build_nargp_data(samples, fm.n_levels)
    nargp = NARGPPredictor(
        nargp_data, W, fm.correlation_matrix,
        ell=1.5, sigma_f=1.0, sigma_n=0.05, mu_band=12,
        denoise_method="median",
    )
    mf_pred = MultiFidelityPredictor(nargp)
    m_pred, s_pred = mf_pred(xi_fid)
    print(f"  NAR-GP at xi_fid: mean={m_pred:.4f}, std={s_pred:.4f}")
    print(f"  Truth (LEVEL_3) at xi_fid: {y_ref:.4f}")
    # Batch validation on a small grid.
    xi_val = [[rng.uniform(b[0], b[1]) for b in bounds] for _ in range(8)]
    means, stds = mf_pred.batch_predict(xi_val)
    print(f"  Batch validation: {len(means)} predictions, "
          f"mean std = {sum(stds) / len(stds):.4f}")

    # ==================================================================
    section("9. CLP confidence intervals")
    # ==================================================================
    # Compute residuals between NAR-GP and LEVEL_3 truth on training xi.
    residuals: List[float] = []
    for s in samples:
        if s.level >= 2:
            m, _ = mf_pred(s.xi)
            residuals.append(s.y - m)
    if not residuals:
        residuals = [0.0]
    bc = fit_bias_correction(
        [s.xi for s in samples if s.level >= 2] or [xi_fid],
        residuals,
        method="ridge",
        lam=0.5,
    )
    print(f"  Bias correction: regularizer={bc.regularizer}, "
          f"n_features={len(bc.beta)}")
    pi = mf_prediction_interval(xi_fid, m_pred, s_pred ** 2, bc, sigma_n_sq=0.01)
    print(f"  95% PI at xi_fid: [{pi.lower:.4f}, {pi.upper:.4f}]")
    print(f"    center={pi.center:.4f}, SE={pi.se:.4f}, contains truth: "
          f"{pi.lower <= y_ref <= pi.upper}")

    # ==================================================================
    section("10. Temporal scheduling and coordinate transforms")
    # ==================================================================
    print(f"  {weekday_diagnostic()}")
    times_myr = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    sched = schedule_fidelity_batch(times_myr, DEFAULT_CADENCE)
    for t, l, jed in sched:
        print(f"    t = {t:.2f} Myr -> level {l} (JED = {jed:.2f})")
    # CCS conversion test.
    A_test = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
    nz, colptr, rowind, Accs = ge_to_ccs(A_test)
    print(f"  ge_to_ccs test: nz={nz}, colptr={colptr}")

    # ==================================================================
    section("11. Molecular dynamics with metadynamics bias")
    # ==================================================================
    metad = MetaDState(config=MetaDConfig(sigma=0.1, W=0.02, tau=3, n_max=50))
    r_traj, v_traj = run_md_relaxation(
        r0=1.3, v0=0.0, mass=1.0, dt=0.005, n_steps=60,
        epsilon=1.0, sigma_lj=1.0, cutoff=2.5, metad_state=metad,
    )
    print(f"  MD relaxation: {len(r_traj)} steps, "
          f"final r = {r_traj[-1]:.4f}")
    print(f"  Metadynamics: {metad.n_deposited} Gaussians deposited")
    print(f"  V_bias(r=1.3) = {bias_potential(1.3, metad):.6f}")

    # ==================================================================
    section("12. Latent-space projection and diagnostics")
    # ==================================================================
    cloud = LatentCloud()
    for s in samples:
        p = project_to_latent(s.xi, s.y, s.level, s.cost, W, fm.n_levels)
        cloud.add_point(p)
    # Connect points at the same fidelity level in chronological order.
    by_level: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        by_level.setdefault(s.level, []).append(i)
    for l, idxs in by_level.items():
        if len(idxs) > 1:
            cloud.add_line(idxs)
    summary = latent_summary(cloud)
    print(f"  Latent cloud: {summary['n_points']} points, "
          f"{summary['n_lines']} lines")
    print(f"  Bounding-box diagonal: {summary['bbox_diag']:.4f}")

    # ==================================================================
    section("13. Denoising filter verification")
    # ==================================================================
    noisy = [1.0, 1.2, 5.0, 0.9, 1.1, 0.8, 3.0, 1.0]  # with outliers
    den = median_filter_1d(noisy, kernel=3)
    print(f"  Median filter: {noisy} -> {den}")

    # ==================================================================
    section("14. Filename sequence and experiment log")
    # ==================================================================
    names = [filename_inc(f"run_{i:03d}") for i in range(5)]
    print(f"  Filename sequence: {names}")
    # Add more experiments to the tracker.
    for _ in range(3):
        tracker.next({"phase": "validation"})
    print(f"  {tracker.summary()}")
    log_name = sample_log_filename(run_name, 3)
    print(f"  Sample-log filename: {log_name}")

    # ==================================================================
    section("15. Summary and timing")
    # ==================================================================
    t_end = time.time()
    print(f"  Total wall-clock time: {t_end - t_start:.3f} s")
    print(f"  Samples collected: {len(samples)}")
    print(f"  NAR-GP posterior at xi_fid: {m_pred:.4f} +/- {s_pred:.4f}")
    print(f"  95% PI: [{pi.lower:.4f}, {pi.upper:.4f}]")
    print(f"  Truth: {y_ref:.4f}")
    coverage = 1.0 if (pi.lower <= y_ref <= pi.upper) else 0.0
    print(f"  PI coverage at xi_fid: {coverage}")
    print()
    print("PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
