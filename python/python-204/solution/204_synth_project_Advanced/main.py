"""
main.py
=======

Unified entry point for the Sobol sensitivity analysis of the
disk-shaped geophysical reactor.  Zero arguments:

    python main.py

runs the complete pipeline:

    1. Smoke test (``robustness.run_smoke_tests``).
    2. Forward-model evaluation at the nominal point.
    3. Sobol first-order / total-order / second-order indices via
       Saltelli sampling (``sobol_indices``).
    4. Cross-check with polynomial chaos expansion (``quadrature_pce``).
    5. Surrogate construction on a 2-D response-field snapshot
       (``shearlet_surrogate``).
    6. Bayesian refinement of the Sobol indices via high-order
       Langevin Monte Carlo (``langevin_inversion``).
    7. Sensitivity graph assembly (``spatial_ops.build_sensitivity_graph``).
    8. Summary printout.

All output is textual (no plotting).
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from forward_model import forward_model, response_field, _set_defaults
from sobol_indices import sobol_bootstrap
from quadrature_pce import (smolyak_sparse_grid, pce_fit,
                            pce_sobol_from_coefficients)
from shearlet_surrogate import ShearletSurrogate, gsm_fit
from langevin_inversion import sobol_posterior_sampler
from spatial_ops import (param_to_disk, build_sensitivity_graph,
                         disk_mesh_polar, mesh_stats)
from elliptic_green import elliptic_fk, elliptic_ek, disk_green_value
from chaotic_mixing import effective_velocity, chirikov_ftle
from reaction_kinetics import (LVParams, lotka_volterra_integrate,
                               steady_state, logistic_exact)
from gauss_seidel_coupled import (gs_solve, build_polar_laplacian,
                                  gs_spectral_radius)
from r8col_utils import r8col_duplicates, dedupe_sample_matrix
from matrix_kernels import gram_centered, mxm_timed
from robustness import run_smoke_tests, ConditionMonitor


# =====================================================================
# Pretty printing helpers
# =====================================================================
def banner(title: str) -> None:
    bar = "=" * 72
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def report(name: str, value, fmt: str = ".6f") -> None:
    if isinstance(value, np.ndarray):
        body = "  ".join(f"{v:{fmt}}" for v in value)
    else:
        body = f"{value:{fmt}}" if isinstance(value, float) else str(value)
    print(f"  {name:<28s} {body}")


# =====================================================================
# Pipeline stages
# =====================================================================
def stage_smoke() -> dict:
    banner("Stage 0 / 7 : module smoke tests")
    t0 = time.perf_counter()
    res = run_smoke_tests(verbose=True)
    dt = time.perf_counter() - t0
    print(f"  elapsed: {dt:.2f} s")
    return res


def stage_nominal() -> dict:
    banner("Stage 1 / 7 : forward model at nominal theta = (0.5, ..., 0.5)")
    theta = np.full(8, 0.5)
    phys = param_to_disk(theta)
    report("K_chir   (Chirikov K)", phys[0])
    report("r_log    (logistic r)", phys[1])
    report("k_cap    (carrying K)", phys[2])
    report("D_eff    (diffusivity)", phys[3], fmt=".4e")
    report("R_disk   (radius)", phys[4])
    report("alpha_k  (competition)", phys[5])
    report("beta_k   (mutualism)", phys[6])
    report("gamma_k  (mortality)", phys[7])
    Y = forward_model(theta, verbose=False)
    report("Y (QoI)  at nominal", Y)
    # Chirikov FTLE diagnostic
    L0 = chirikov_ftle(1.0, 0.5, phys[0], n_iter=64)
    report("FTLE @ (1, 0.5)", L0, fmt=".4f")
    v_eff = effective_velocity(phys[4], phys[0], n_iter=32, n_quad=5, seed=1)
    report("v_eff (mixing velocity)", v_eff)
    # LV kinetics diagnostic
    p = LVParams(r_a=phys[1], K_a=phys[2], r_b=0.7 * phys[1],
                 K_b=0.8 * phys[2],
                 alpha=phys[5], beta=phys[6], gamma=phys[7])
    t_arr, y_arr = lotka_volterra_integrate(p, n_steps=80)
    report("<A> over LV traj", float(y_arr[:, 0].mean()))
    report("<B> over LV traj", float(y_arr[:, 1].mean()))
    y_star = steady_state(p)
    report("LV steady state", y_star)
    # Green's function diagnostic
    G0 = disk_green_value(phys[4], 0.3 * phys[4], 0.4 * phys[4], 0.5)
    report("G(r=0.3R, r0=0.4R, dth=0.5)", G0)
    return dict(theta=theta, phys=phys, Y=Y, p=p)


def stage_sobol(nominal: dict, N: int = 200, n_boot: int = 40) -> dict:
    banner(f"Stage 2 / 7 : Sobol indices (Saltelli N = {N}, boot = {n_boot})")

    def f(theta):
        return forward_model(theta)

    rng = np.random.default_rng(2024)
    t0 = time.perf_counter()
    out = sobol_bootstrap(f, d=8, N=N, n_boot=n_boot, rng=rng)
    dt = time.perf_counter() - t0
    print(f"  forward-model evaluations: {out['n_evals']}")
    print(f"  wall-clock: {dt:.2f} s")
    report("E[Y] = f0", out['f0'])
    report("Var(Y) = V", out['V'])
    param_names = ["K_chir", "r_log", "k_cap", "D_eff",
                   "R_disk", "alpha", "beta", "gamma"]
    print()
    print(f"  {'param':<8s}  {'S1':>8s}  {'S1_lo':>8s}  {'S1_hi':>8s}  "
          f"{'ST':>8s}  {'ST_lo':>8s}  {'ST_hi':>8s}")
    for i, nm in enumerate(param_names):
        print(f"  {nm:<8s}  {out['S1'][i]:>8.4f}  {out['S1_ci'][0][i]:>8.4f}  "
              f"{out['S1_ci'][1][i]:>8.4f}  "
              f"{out['ST'][i]:>8.4f}  {out['ST_ci'][0][i]:>8.4f}  "
              f"{out['ST_ci'][1][i]:>8.4f}")
    # Second-order summary
    S2 = out['S2']
    print()
    print("  Top 2nd-order interactions |S_{ij}|:")
    pairs = []
    d = S2.shape[0]
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((abs(S2[i, j]), i, j))
    pairs.sort(reverse=True)
    for val, i, j in pairs[:5]:
        print(f"    |S_{{{param_names[i]},{param_names[j]}}}| = {val:.4f}")
    return dict(sobol=out, param_names=param_names)


def stage_pce(nominal: dict) -> dict:
    banner("Stage 3 / 7 : PCE surrogate (sparse-grid validation)")
    d = 8
    rng = np.random.default_rng(7)
    # Sample a Latin-ish design for the PCE
    N_pce = 200
    X = rng.random((d, N_pce))
    y = np.array([forward_model(X[:, j]) for j in range(N_pce)])
    c, alphas, info = pce_fit(X, y, p=2, tol=1.0e-10)
    si = pce_sobol_from_coefficients(c, alphas, d)
    param_names = ["K_chir", "r_log", "k_cap", "D_eff",
                   "R_disk", "alpha", "beta", "gamma"]
    print(f"  PCE rank of design matrix: {info['dq']['rank']}")
    print(f"  PCE basis count (|alpha|<=2) : {info['P']}")
    print(f"  PCE-estimated Var(Y):          {si['var_y']:.4e}")
    print()
    print(f"  {'param':<8s}  {'S1_PCE':>8s}  {'ST_PCE':>8s}")
    for i, nm in enumerate(param_names):
        print(f"  {nm:<8s}  {si['S1'][i]:>8.4f}  {si['ST'][i]:>8.4f}")
    # Cross-check Smolyak mean integral
    f_mean = lambda z: forward_model(z)
    try:
        I_smol, n_smol = smolyak_sparse_grid(d, level=2)
        vals = np.array([f_mean(I_smol[:, j]) for j in range(I_smol.shape[1])])
        W = np.ones(I_smol.shape[1]) / I_smol.shape[1]  # simplified
        smol_mean = float(np.dot(W, vals))
        print(f"  Smolyak-style mean (level=2, {I_smol.shape[1]} pts): {smol_mean:.4f}")
    except ValueError as exc:
        print(f"  Smolyak integral skipped: {exc}")
    return dict(pce_info=info, pce_sobol=si)


def stage_surrogate(nominal: dict) -> dict:
    banner("Stage 4 / 7 : shearlet surrogate of the response field")
    img = response_field(nominal['theta'], N_field=10)
    print(f"  response-field snapshot : shape {img.shape}, "
          f"range [{img.min():.4f}, {img.max():.4f}]")
    surr = ShearletSurrogate(img, keep_ratio=0.5, nu=3.0, smooth_before=True)
    rec = surr()
    err = float(np.max(np.abs(rec - img)))
    print(f"  surrogate max-abs error : {err:.4e}")
    print(f"  GSM fit on LH band      : alpha={surr.gsm_LH['alpha']:.2f}  "
          f"beta={surr.gsm_LH['beta']:.3e}  "
          f"kurtosis={surr.gsm_LH['kurtosis']:.3f}")
    return dict(img=img, rec=rec, err=err)


def stage_bayesian(sobol_res: dict) -> dict:
    banner("Stage 5 / 7 : Bayesian refinement via Picard-Lagrange LMC")
    out = sobol_res['sobol']
    d = 8
    s_hat = np.concatenate([out['S1'], out['ST']])
    cov = out['cov'] + 1.0e-8 * np.eye(2 * d)
    rng = np.random.default_rng(3)
    t0 = time.perf_counter()
    post = sobol_posterior_sampler(s_hat, cov, d=d,
                                   n_samples=80, burn_in=20,
                                   K=3, h=1.0e-3, rng=rng)
    dt = time.perf_counter() - t0
    print(f"  Langevin wall-clock: {dt:.2f} s")
    print(f"  accept ratio:        {post['accept_ratio']:.3f}")
    param_names = sobol_res['param_names']
    print()
    print(f"  Posterior mean (S1, ST):")
    print(f"  {'param':<8s}  {'S1_MC':>8s}  {'S1_post':>8s}  "
          f"{'ST_MC':>8s}  {'ST_post':>8s}")
    for i, nm in enumerate(param_names):
        print(f"  {nm:<8s}  {out['S1'][i]:>8.4f}  {post['mean'][i]:>8.4f}  "
              f"{out['ST'][i]:>8.4f}  {post['mean'][d + i]:>8.4f}")
    return dict(posterior=post)


def stage_graph(sobol_res: dict) -> dict:
    banner("Stage 6 / 7 : sensitivity adjacency graph")
    S2 = sobol_res['sobol']['S2']
    g = build_sensitivity_graph(S2, threshold=0.02)
    print(f"  number of nodes      : {g['n_nodes']}")
    print(f"  number of edges      : {len(g['edges'])}")
    print(f"  per-node degree      : {g['degree']}")
    print(f"  total degree         : {g['degree'].sum()}")
    print()
    print("  GRF file content (first 20 lines):")
    for line in g['grf_text'].splitlines()[:20]:
        print(f"    {line}")
    return dict(graph=g)


def stage_summary(stage_times: dict) -> None:
    banner("Stage 7 / 7 : summary")
    print()
    print("  All 7 pipeline stages completed successfully.")
    print()
    print("  Stage wall-clock breakdown:")
    total = 0.0
    for name, dt in stage_times.items():
        print(f"    {name:<32s}  {dt:>7.2f} s")
        total += dt
    print(f"    {'TOTAL':<32s}  {total:>7.2f} s")
    print()
    print("  Project modules:")
    mods = ["r8col_utils", "matrix_kernels", "elliptic_green",
            "chaotic_mixing", "reaction_kinetics",
            "gauss_seidel_coupled", "quadrature_pce",
            "shearlet_surrogate", "langevin_inversion",
            "spatial_ops", "sobol_indices", "forward_model",
            "robustness", "main"]
    for m in mods:
        print(f"    - {m}.py")
    print()
    print("  Scientific problem solved:")
    print("    Sobol global sensitivity analysis of a coupled")
    print("    chaotic-advection reactive-transport model in a")
    print("    disk-shaped geophysical reactor with autocatalytic")
    print("    Lotka-Volterra population kinetics.")


# =====================================================================
# Main entry
# =====================================================================
def main() -> int:
    print("=" * 72)
    print("  PROJECT 204 : Sobol sensitivity analysis of a disk-shaped")
    print("                geophysical reactor with chaotic mixing")
    print("                and autocatalytic Lotka-Volterra kinetics.")
    print("=" * 72)
    # Use small numerical budgets so the demo finishes in under a minute
    _set_defaults(Nr=3, Ntheta=4, n_chirikov_iter=8,
                  n_chirikov_quad=3, lv_n_steps=30, gs_max_iter=20,
                  gs_tol=1.0e-5)
    stage_times = {}
    t0 = time.perf_counter()
    stage_smoke()
    stage_times['0_smoke_tests'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    nominal = stage_nominal()
    stage_times['1_nominal_evaluation'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sobol_res = stage_sobol(nominal, N=4, n_boot=20)
    stage_times['2_sobol_saltelli'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stage_pce(nominal)
    stage_times['3_pce_surrogate'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stage_surrogate(nominal)
    stage_times['4_shearlet_surrogate'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stage_bayesian(sobol_res)
    stage_times['5_langevin_inversion'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stage_graph(sobol_res)
    stage_times['6_sensitivity_graph'] = time.perf_counter() - t0
    stage_summary(stage_times)
    return 0


if __name__ == "__main__":
    sys.exit(main())
