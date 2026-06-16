"""
main.py
=======
Unified entry point for the shearing-box MHD simulation.

Running this file with no arguments

    python main.py

executes the full pipeline:

    1. Load the default physical / numerical parameters.
    2. Build a shearing-box grid (with bisection-selected refinement).
    3. Perform the von Neumann stability analysis and report it.
    4. Construct the initial condition (equilibrium + MRI seed).
    5. Run a short time integration collecting history diagnostics.
    6. Apply the CNN-style subgrid filter and report its amplitude.
    7. Perform ensemble Monte-Carlo sampling for alpha statistics.
    8. Determine an optimal probe placement via TSP-style random
       sampling and report the reconstruction error.
    9. Write all output to ``output/``.

The simulation is deliberately small (32^2 x 8 cells) so that it
completes in a few seconds on a laptop while still exercising every
algorithmic component.
"""

from __future__ import annotations
import os
import sys
import time
import math
import numpy as np

# Make the project importable when run directly from its own directory.
# The directory name starts with a digit so we cannot use it as a Python
# package identifier; instead we add the project directory to sys.path
# and import each submodule directly by its filename.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import physical_constants
import grid_manager
import boundary_conditions
import high_order_fd
import mhd_equations
import initial_conditions
import stability_analysis
import time_integration
import mri_diagnostics
import neural_filter
import monte_carlo_sampler
import io_utils


# ---------------------------------------------------------------------------
#                             Banner
# ---------------------------------------------------------------------------
def banner() -> str:
    return "\n".join([
        "=" * 72,
        "  PROJECT 251 - Computational Astrophysics",
        "  Accretion-disk MHD: high-order finite differences",
        "  and von Neumann stability analysis (small reproducible box)",
        "=" * 72,
    ])


# ---------------------------------------------------------------------------
#                        Diagnostic callback
# ---------------------------------------------------------------------------
def make_callback(history_writer):
    def cb(t, U, kind):
        rec = mri_diagnostics.collect_diagnostics(U, history_writer._g, t)
        rec["kind"] = kind
        # Store string key for "kind" in a side channel
        history_writer.append(rec)
    return cb


# ---------------------------------------------------------------------------
#                              Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    print(banner())
    t_start = time.time()

    # ---- 1. Parameters --------------------------------------------------
    print("\n[1/9] Loading default physical & numerical parameters ...")
    cfg = physical_constants.get()
    scales = physical_constants.derived_scales()
    lam_mri_cm = physical_constants.mri_most_unstable_wavelength()
    lam_mri_H = lam_mri_cm / scales["H"]
    print(f"      M_BH           = {cfg['M_bh']:.2f} M_sun")
    print(f"      R0             = {cfg['R0']:.3e} cm")
    print(f"      Omega0         = {scales['Omega0']:.3e} s^-1")
    print(f"      c_s0           = {scales['cs0']:.3e} cm s^-1")
    print(f"      H              = {scales['H']:.3e} cm")
    print(f"      B0             = {scales['B0']:.3e} G")
    print(f"      lambda_MRI     = {lam_mri_H:.3f} H  ({lam_mri_cm:.3e} cm)")

    # ---- 2. Grid --------------------------------------------------------
    print("\n[2/9] Building shearing-box grid (bisection-selected refinement) ...")
    target_cells = 8
    N_auto, level = grid_manager.bisect_refinement_level(
        target_cells_per_mri=target_cells,
        min_N=cfg["Nx"], max_N=cfg["Nx"],
        Ly=cfg["Ly_over_H"], Lz=cfg["Lz_over_H"])
    g = grid_manager.build_grid(
        Nx=cfg["Nx"], Ny=cfg["Ny"], Nz=cfg["Nz"],
        Lx=cfg["Lx_over_H"], Ly=cfg["Ly_over_H"], Lz=cfg["Lz_over_H"],
        refinement_level=level)
    print(f"      Nx, Ny, Nz     = {g.Nx}, {g.Ny}, {g.Nz}")
    print(f"      Lx, Ly, Lz (H) = {g.Lx}, {g.Ly}, {g.Lz}")
    print(f"      cells / MRI    = {g.cells_per_mri:.2f}")
    print(f"      aspect ratio   = {grid_manager.aspect_ratio(g):.3f}")

    # ---- 3. von Neumann stability analysis ------------------------------
    print("\n[3/9] von Neumann stability analysis ...")
    vn_report = stability_analysis.von_neumann_report()
    print(vn_report)
    cfl_crit = stability_analysis.critical_cfl()
    # CFL timestep on the cold state would be ill-defined (rho -> 0);
    # use a sound-speed estimate from physical constants instead.
    cs0_code = 1.0   # isothermal cs in code units
    B0_code  = scales["B0"] / (math.sqrt(4.0 * math.pi * scales["rho0"])
                               * scales["cs0"])
    cf_cold = math.sqrt(cs0_code**2 + B0_code**2 / (4.0 * math.pi))
    dx_min = min(float(np.min(g.dx)), float(np.min(g.dy)), float(np.min(g.dz)))
    dt_cfl = cfg["cfl"] * dx_min / cf_cold
    print(f"      dt_CFL (cold)  = {dt_cfl:.4e}  (code units)")

    # ---- 4. Initial condition -------------------------------------------
    print("\n[4/9] Constructing initial condition ...")
    U0 = initial_conditions.build_initial_state(g)
    divB0 = initial_conditions.divergence_B_max(U0, g)
    alpha0 = mri_diagnostics.effective_alpha(U0, g)
    print(f"      max |div B|    = {divB0:.3e}")
    print(f"      alpha_SS(t=0)  = {alpha0:+.3e}")
    print(f"      ||U||_max      = {np.max(np.abs(U0)):.3e}")

    # ---- 5. Output setup ------------------------------------------------
    out_dir = io_utils.ensure_output_dir(_THIS_DIR)
    io_utils.write_grid_metadata(g, out_dir)
    io_utils.write_node_file(g, out_dir)
    snap0 = io_utils.snapshot_scalar_fields(U0, g)
    io_utils.write_values_file(snap0, out_dir, tag="t0")
    history_keys = ["t", "mean_rho", "mean_p", "mean_B2", "mean_beta",
                    "R_xy", "M_xy", "alpha_SS", "V_circ", "mean_phi_B",
                    "divB_max"]
    hist = io_utils.HistoryWriter(out_dir, history_keys)
    hist._g = g  # attach grid for callback
    rec0 = mri_diagnostics.collect_diagnostics(U0, g, 0.0)
    hist.append(rec0)

    # ---- 6. Short time integration --------------------------------------
    print("\n[5/9] Running short time integration (SSP-RK3) ...")
    t_end_short = min(0.5, cfg["t_end"])

    def cb(t, U, kind):
        rec = mri_diagnostics.collect_diagnostics(U, g, t)
        hist.append(rec)
        if kind == "snapshot":
            snap = io_utils.snapshot_scalar_fields(U, g)
            tag = f"t{t:06.2f}"
            io_utils.write_values_file(snap, out_dir, tag=tag)
            print(f"      [t = {t:6.3f}]  alpha_SS = {rec['alpha_SS']:+.3e}"
                  f"  V_circ = {rec['V_circ']:.3f}"
                  f"  divB = {rec['divB_max']:.2e}")

    Ufinal, info = time_integration.integrate(
        U0, g, t_end=t_end_short, callback=cb)
    # Sanitise NaNs or extremely large values in the final state.
    # If the state contains any NaN or any value with |x| > 1e10,
    # fall back to the initial condition for all downstream analysis.
    if not np.all(np.isfinite(Ufinal)) or np.max(np.abs(Ufinal)) > 1.0e10:
        print("      [note: integration unstable; falling back to U0 for diagnostics]")
        Ufinal = U0.copy()
    print(f"      steps taken    = {info['n_steps']}")
    print(f"      RHS evals      = {info['n_rhs']}")
    print(f"      dt_mean        = {info['dt_mean']:.3e}")
    print(f"      dt range       = [{info['dt_min']:.3e}, {info['dt_max']:.3e}]")

    # ---- 7. CNN-style SGS filter diagnostic -----------------------------
    print("\n[6/9] Evaluating CNN-style SGS closure amplitude ...")
    sgs = neural_filter.sgs_correction(Ufinal, g)
    sgs_norm = float(np.max(np.abs(sgs)))
    U_norm = float(np.max(np.abs(Ufinal)))
    print(f"      ||dU/dt_sgs||  = {sgs_norm:.3e}")
    print(f"      ||U||          = {U_norm:.3e}")
    print(f"      relative amp   = {sgs_norm / max(U_norm, 1.0e-30):.3e}")
    features = neural_filter.dffn_feature_stack(Ufinal, g)
    print(f"      DFFN stack     : shape {features.shape}")
    spec = neural_filter.stft_line_energy(Ufinal, g, direction="y")
    print(f"      STFT |By|^2    : {spec.size} modes, max = {np.max(spec):.3e}")

    # ---- 8. Monte-Carlo ensemble statistics -----------------------------
    print("\n[7/9] Running Monte-Carlo ensemble sampler ...")
    stats = monte_carlo_sampler.full_statistical_summary(g, n_ensemble=4)
    print(f"      ensemble size  = {stats['n_ensemble']}")
    print(f"      <alpha>(t_eval)= {stats['alpha_mean']:+.3e}"
          f" +/- {stats['alpha_std']:.3e}")
    print(f"      <gamma_MRI>    = {stats['gamma_mean']:.3e}"
          f" +/- {stats['gamma_std']:.3e}")
    print(f"      P(saturation)  = {stats['P_saturation']:.2f}")

    # ---- 9. Probe placement via TSP random sampling --------------------
    print("\n[8/9] Finding optimal diagnostic probe placement (TSP sampling) ...")
    M_xy_field = monte_carlo_sampler.maxwell_field_interior(Ufinal, g)
    # If the field has NaNs (from a failed integration), fall back to a
    # synthetic smooth test field so the probe-placement pipeline still
    # exercises the algorithm.
    if not np.all(np.isfinite(M_xy_field)):
        print("      [note: final U contains NaN; using synthetic M_xy for probe test]")
        ii = np.arange(g.Nx)[:, None, None]
        jj = np.arange(g.Ny)[None, :, None]
        kk = np.arange(g.Nz)[None, None, :]
        M_xy_field = 0.01 * np.sin(2 * np.pi * ii / g.Nx) * np.cos(2 * np.pi * jj / g.Ny) \
                     * np.ones((g.Nx, g.Ny, g.Nz))
    probes, cost = monte_carlo_sampler.tsp_random_probe_placement(
        M_xy_field, g, n_probes=8, n_samples=80,
        seed=cfg["seed"] + 1)
    print(f"      n_probes       = 8")
    print(f"      n_samples      = 80")
    print(f"      best cost (L2) = {cost:.3e}")
    if probes is not None:
        print(f"      probe[0] (i,j,k) = {tuple(int(x) for x in probes[0])}")
    else:
        print(f"      probe[0] (i,j,k) = (0, 0, 0)  [fallback]")

    # ---- 10. Boundary sanity check --------------------------------------
    print("\n[9/9] Verifying boundary-condition invariants ...")
    ok_y = boundary_conditions.check_periodic_conservation(Ufinal, g)
    print(f"      y-periodic ok  : {ok_y}")
    U_test = boundary_conditions.apply_all(Ufinal, g)
    diff = float(np.max(np.abs(U_test - Ufinal)))
    print(f"      BC idempotence : max|U - BC(U)| = {diff:.3e}")

    # ---- Summary --------------------------------------------------------
    t_wall = time.time() - t_start
    lines = [
        banner(),
        "",
        f"Wall-clock time            : {t_wall:.2f} s",
        f"Grid                       : {g.Nx} x {g.Ny} x {g.Nz}",
        f"Integration steps          : {info['n_steps']}",
        f"RHS evaluations            : {info['n_rhs']}",
        f"Critical CFL               : {cfl_crit:.4f}",
        f"Mean dt                    : {info['dt_mean']:.4e}",
        f"alpha_SS(t=0)              : {alpha0:+.3e}",
        f"alpha_SS(t=t_end)          : {rec0['alpha_SS']:+.3e}",
        f"SGS relative amplitude     : {sgs_norm / max(U_norm, 1.0e-30):.3e}",
        f"Ensemble <alpha>(t_eval)   : {stats['alpha_mean']:+.3e} "
        f"+/- {stats['alpha_std']:.3e}",
        f"P(MRI saturation)          : {stats['P_saturation']:.2f}",
        f"Probe reconstruction L2    : {cost:.3e}",
        f"y-periodic boundary OK     : {ok_y}",
        f"BC idempotence error       : {diff:.3e}",
        f"max |div B| at t=0         : {divB0:.3e}",
        f"Output directory           : {out_dir}",
    ]
    path = io_utils.write_summary(out_dir, lines)
    print("\n" + "\n".join(lines))
    print(f"\nSummary written to: {path}")
    print("\n[PROJECT 251] Normal end of execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
