"""
main.py
=======
Unified entry point for the PROJECT-283 perovskite solar-cell defect-state
calculation code. Runs the complete pipeline:

  [1] Sanity checks on physical constants and high-order FD stencils
  [2] Mesh generation, quality check, and CVT optimization
  [3] Self-consistent Poisson + SRH (Gummel) solve
  [4] Defect kinetics (logistic generation, bond breaking, reaction pathway)
  [5] Carrier transport (DG advection + drift-diffusion)
  [6] ML prediction of defect transition levels
  [7] Higher-order Langevin defect-configuration sampling
  [8] FEM scalar-field analysis (L2/H1 norms, extrema)
  [9] Interpolation accuracy (nearest vs linear)
  [10] Stability analysis (von Neumann, Gummel, DG CFL)
  [11] Benchmark aggregation across solvers/materials/methods

All output is textual (no visualization), as required.

Running
-------
    python main.py

No command-line arguments are required. A summary report is printed at the
end; intermediate results are kept in memory.
"""

from __future__ import annotations
import time
import sys
import math
from typing import Dict, Any

import numpy as np

# --- Module imports (each corresponds to a synthesis seed project) ---------
import perovskite_constants as pc
import high_order_fd as hfd
import mesh_generator as mg
import poisson_defect_solver as pds
import defect_rate_equations as dre
import carrier_transport_dg as dg
import ml_defect_predictor as mldp
import langevin_defect_sampler as lds
import fem_scalar_field as fsf
import defect_field_interpolation as dfi
import stability_analysis as sa
import benchmark_analyzer as ba


# ============================================================================
# Utility: pretty-print a nested dict
# ============================================================================
def pretty(d: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(d, dict):
        lines = []
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{pad}{k}:")
                lines.append(pretty(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)
    elif isinstance(d, (list, tuple)):
        if len(d) == 0:
            return f"{pad}[]"
        if len(d) <= 6 and all(isinstance(x, (int, float, str)) for x in d):
            return f"{pad}{list(d)}"
        lines = []
        for i, x in enumerate(d):
            if i > 5:
                lines.append(f"{pad}  ... ({len(d) - 6} more)")
                break
            lines.append(pretty(x, indent + 1))
        return "\n".join(lines)
    else:
        return f"{pad}{d}"


def section(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ============================================================================
# Pipeline stages
# ============================================================================
def stage_constants() -> Dict[str, Any]:
    section("1. Physical constants sanity check")
    san = pc.sanity_check()
    for k, v in san.items():
        print(f"  {k:22s} = {v}")
    return san


def stage_fd_stencil() -> Dict[str, Any]:
    section("2. High-order FD stencil validation")
    p_target = pc.FD_ORDER
    val = hfd.validate_stencil(p_target)
    print(f"  Stencil half-width p = {p_target}")
    print(f"  err(x^2)       = {val['err_x2']:.3e}")
    print(f"  err(x^4)       = {val['err_x4']:.3e}")
    r_max = hfd.max_stable_r(p_target)
    print(f"  max stable r   = {r_max:.6f}")
    # Sweep orders
    for p in [1, 2, 3, 4, 5, 6]:
        r = hfd.max_stable_r(p)
        s = pc.fd_stability_factor(p)
        print(f"    p={p}: r_max = {r:.6f}, S_p = {s:.6f}")
    # Quick 1D Poisson test
    nx = 64
    L = pc.DEVICE_LENGTH_M
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    xc = 0.5 * L
    sigma_x = 0.1 * L
    rho = pc.E_CHARGE * pc.DEFECT_DENSITY_DEFAULT * np.exp(
        -((x - xc) ** 2) / (2.0 * sigma_x ** 2))
    x_grid, phi = hfd.compute_defect_potential(rho, nx=nx, p=p_target)
    print(f"  1D Poisson: phi_max = {np.max(phi):.4f} V, "
          f"phi_min = {np.min(phi):.4f} V")
    return {"r_max": r_max, "phi_max": float(np.max(phi))}


def stage_mesh() -> Dict[str, Any]:
    section("3. Mesh generation + quality + Lloyd CVT + Delaunay check")
    result = mg.build_defect_mesh(nx=12, ny=6,
                                  defect_centers=[(0.3, 0.5), (0.7, 0.5)])
    q = result["quality"]
    print(f"  Q4 mesh: {q['n_elements']} elements, "
          f"total area = {q['total_area']:.4f}")
    print(f"  detJ_min = {q['detJ_min']:.3e}, "
          f"detJ_max = {q['detJ_max']:.3e}")
    print(f"  min angle = {q['angle_min_deg']:.2f} deg")
    print(f"  max aspect = {q['aspect_max']:.3f}")
    print(f"  Delaunay discrepancy = {result['delaunay_disc']:.3e} deg")
    print(f"  CVT generators (x): {np.round(result['defect_cvt_x'][:6], 4)}...")
    return {k: v for k, v in result.items()
            if k not in ("node_xy", "elem_node", "defect_cvt_x")}


def stage_poisson_srh() -> Dict[str, Any]:
    section("4. Self-consistent Poisson + SRH (Gummel)")
    result = pds.gummel_poisson_srh(nx=64, p_fd=pc.FD_ORDER,
                                    max_iter=30, tol=1e-6, omega=0.3)
    print(f"  Converged: {result['converged']} "
          f"after {result['iterations']} iterations")
    print(f"  Final residual: {result['residual_history'][-1]:.3e}")
    J_rec = pds.recombination_current(result)
    print(f"  Recombination current density J_rec = {J_rec:.3e} A/m^2")
    # Debye length
    n_avg = float(np.mean(result["n"]))
    L_D = pds.debye_length(n_avg)
    print(f"  Debye length L_D = {L_D:.3e} m")
    # Electric field
    E = pds.electric_field(result["phi"],
                           result["x"][1] - result["x"][0],
                           p=pc.FD_ORDER)
    print(f"  Max |E| = {np.max(np.abs(E)):.3e} V/m")
    return {
        "converged": result["converged"],
        "iterations": result["iterations"],
        "J_rec_A_m2": J_rec,
        "Debye_length_m": L_D,
        "max_E_V_m": float(np.max(np.abs(E))),
        "phi_max_V": float(np.max(result["phi"])),
    }


def stage_kinetics() -> Dict[str, Any]:
    section("5. Defect kinetics (logistic, bond, network, pathway)")
    result = dre.run_defect_kinetics(t_end_s=1e4, n_time_steps=200,
                                     n_sites=32)
    print(f"  Logistic rate r = {result['r_form']:.3e} 1/s")
    print(f"  Final N_t / N_max = "
          f"{result['N_t_log'][-1] / result['N_max']:.4f}")
    net_last = result['network_history'][-1]
    print(f"  Network final state: {net_last}")
    print(f"  PbI2 formed (pyrolysis, final): "
          f"{result['pbi2_formed'][-1]:.4f}")
    print(f"  Reaction-pathway cost (4-species) = "
          f"{result['reaction_pathway_cost']} ops")
    print(f"  Reaction order: {result['reaction_order']}")
    print(f"  Catalan C_4 = {result['catalan_n4']}")
    return {
        "r_form_1_s": result["r_form"],
        "Nt_over_Nmax_final": result["N_t_log"][-1] / result["N_max"],
        "network_final": net_last,
        "reaction_cost": result["reaction_pathway_cost"],
        "catalan_4": result["catalan_n4"],
    }


def stage_dg_transport() -> Dict[str, Any]:
    section("6. Carrier transport (DG advection + drift-diffusion)")
    # DG advection on a normalized domain [0, 1] with moderate parameters.
    # Use a small number of elements and moderate speed to keep the test
    # within a stable regime for this demonstration.
    def u0(x):
        return np.sin(2.0 * math.pi * x)
    result = dg.dg_advec_1d(u0, FinalTime=0.05, a=1.0, K=16, N=3,
                            xmin=0.0, xmax=1.0)
    print(f"  DG advection: K = {16}, N = {3}, a = 1.0")
    print(f"  dt = {result['dt']:.3e}, N_steps = {result['N_steps']}")
    # Drift-diffusion in the defect channel
    nx = 64
    L = pc.DEVICE_LENGTH_M
    x = np.linspace(0.0, L, nx)
    xc = 0.5 * L
    sigma_x = 0.1 * L
    N_t = pc.DEFECT_DENSITY_DEFAULT * np.exp(
        -((x - xc) ** 2) / (2.0 * sigma_x ** 2))
    phi = np.linspace(pc.V_NET, 0.0, nx)
    trans = dg.solve_carrier_transport(N_t, x, phi, n_steps=30)
    print(f"  Steady-state n_e range: "
          f"[{np.min(trans['n_ss']):.2e}, {np.max(trans['n_ss']):.2e}] m^-3")
    print(f"  Current density J = {trans['current_density_A_m2']:.3e} A/m^2")
    return {
        "DG_dt": result["dt"],
        "DG_N_steps": result["N_steps"],
        "J_drift_diffusion_A_m2": trans["current_density_A_m2"],
    }


def stage_ml() -> Dict[str, Any]:
    section("7. ML prediction of defect transition levels")
    result = mldp.run_ml_defect_pipeline()
    print(f"  Dataset: {result['dataset_shape'][0]} samples, "
          f"{result['dataset_shape'][1]} features")
    print(f"  PCA variance ratio (3 comp.): "
          f"{[round(v, 3) for v in result['pca_variance_ratio']]}")
    print(f"  RF CV MAE  = {result['rf_cv']['mae_mean']:.4f} "
          f"+/- {result['rf_cv']['mae_std']:.4f} eV")
    print(f"  KNN CV MAE = {result['knn_cv']['mae_mean']:.4f} "
          f"+/- {result['knn_cv']['mae_std']:.4f} eV")
    print(f"  RF-PCA CV  = {result['rf_pca_cv']['mae_mean']:.4f} "
          f"+/- {result['rf_pca_cv']['mae_std']:.4f} eV")
    print(f"  Train MAE  = {result['train_mae']:.4f} eV")
    print(f"  Feature importances (permutation):")
    for f, imp in result['feature_importances'].items():
        print(f"    {f:25s}: {imp:+.4f}")
    print(f"  Benchmark summary:")
    print(pretty(result['benchmark'], indent=2))
    return {
        "rf_mae": result["rf_cv"]["mae_mean"],
        "knn_mae": result["knn_cv"]["mae_mean"],
        "rf_pca_mae": result["rf_pca_cv"]["mae_mean"],
    }


def stage_langevin() -> Dict[str, Any]:
    section("8. Higher-order Langevin defect sampling")
    result = lds.run_langevin_defect_sampling(
        n_defects=3, slab_length_nm=50.0, n_steps=100, K=3)
    print(f"  K = {result['K']}, n_defects = {result['n_defects']}")
    print(f"  n_steps = {result['n_steps']}, "
          f"n_samples (thinned) = {len(result['samples'])}")
    print(f"  Final positions (m): {result['final_positions']}")
    print(f"  Energy trajectory length: {len(result['energies'])}")
    if len(result['energies']) > 1:
        print(f"  |grad U| range: "
              f"[{np.min(result['energies']):.3e}, "
              f"{np.max(result['energies']):.3e}]")
    return {
        "K": result["K"],
        "n_samples": len(result["samples"]),
        "final_positions": result["final_positions"].tolist(),
    }


def stage_fem() -> Dict[str, Any]:
    section("9. FEM scalar-field analysis (P1 on triangles)")
    result = fsf.analyze_2d_potential(nx=16, ny=16)
    print(f"  Triangular mesh: {result['n_nodes']} nodes, "
          f"{result['n_elements']} elements")
    print(f"  Field stats: {result['field_stats']}")
    print(f"  |grad phi| stats: {result['grad_mag_stats']}")
    print(f"  L2 norm = {result['L2_norm']:.4e}")
    print(f"  H1 seminorm = {result['H1_seminorm']:.4e}")
    print(f"  Contour coverage (phi >= V/2): "
          f"{result['contour_coverage_half_Vbi']:.3f}")
    print(f"  Local maxima: {result['n_local_maxima']}, "
          f"minima: {result['n_local_minima']}")
    print(f"  Triangle quality: "
          f"min = {result['triangle_quality_min']:.3f}, "
          f"mean = {result['triangle_quality_mean']:.3f}")
    return result


def stage_interpolation() -> Dict[str, Any]:
    section("10. Defect-field interpolation (nearest vs linear)")
    result = dfi.test_defect_interpolation()
    print(f"  L2 err (nearest) = {result['err_nearest']:.3e}")
    print(f"  L2 err (linear)  = {result['err_linear']:.3e}")
    print(f"  ratio (nearest / linear) = {result['ratio']:.2f}x")
    print(f"  L2 norm (nearest) = {result['L2_nearest']:.3e}")
    print(f"  L2 norm (linear)  = {result['L2_linear']:.3e}")
    return result


def stage_stability() -> Dict[str, Any]:
    section("11. Stability analysis (von Neumann, Gummel, DG CFL)")
    result = sa.run_full_stability_analysis()
    print("  von Neumann (max stable r):")
    for p, v in result['von_neumann'].items():
        print(f"    p = {p}: r_max = {v['r_max']:.6f}, "
              f"S_p = {v['theoretical_S_p']:.6f}")
    gm = result['gummel']
    print(f"  Gummel spectral radius: {gm['spectral_radius']:.4f}")
    print(f"  Gummel converges: {gm['converges']}")
    print(f"  Gummel iters to 1e-10: {gm['n_iters']}")
    print("  DG CFL limits:")
    for N, v in result['dg_cfl'].items():
        print(f"    N={N}: CFL_a={v['cfl_advective']:.4f}, "
              f"CFL_d={v['cfl_diffusive']:.6f}")
    for p in [1, 3, pc.FD_ORDER]:
        key = f"dispersion_p{p}"
        if key in result:
            print(f"  Dispersion p={p}: "
                  f"max omega^2 = {result[key]['max_omega_sq']:.3e}")
    return result


def stage_benchmark() -> Dict[str, Any]:
    section("12. Benchmark aggregation across solvers/materials")
    result = ba.run_benchmark_analysis()
    print(f"  n entries = {result['n_entries']}")
    print("  Solver ranking (by MAE):")
    for solver, mean, std in result['solver_ranking']:
        print(f"    {solver:15s}: MAE = {mean:.4f} +/- {std:.4f} eV")
    print("  Best solver per material:")
    for mat, (solver, mean) in result['best_per_material'].items():
        print(f"    {mat:10s}: {solver:15s} (MAE = {mean:.4f} eV)")
    tt = result['ttest_top_two']
    print(f"  Paired t-test (top 2 solvers): "
          f"t = {tt['t_stat']:.3f}, p = {tt['p_value']:.3f}")
    return result


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    t0 = time.time()
    print("=" * 72)
    print("  PROJECT 283: Perovskite Solar-Cell Defect-State Calculator")
    print("  High-Order Finite Differences + Stability Analysis")
    print("=" * 72)
    print(f"  Device length: {pc.DEVICE_LENGTH_M * 1e9:.1f} nm")
    print(f"  Built-in voltage: {pc.V_BI:.2f} V")
    print(f"  FD order (2p): {2 * pc.FD_ORDER}")
    print(f"  Defect density (default): "
          f"{pc.DEFECT_DENSITY_DEFAULT:.2e} m^-3")

    results: Dict[str, Any] = {}
    stages = [
        ("1_constants", stage_constants),
        ("2_fd_stencil", stage_fd_stencil),
        ("3_mesh", stage_mesh),
        ("4_poisson_srh", stage_poisson_srh),
        ("5_kinetics", stage_kinetics),
        ("6_dg_transport", stage_dg_transport),
        ("7_ml", stage_ml),
        ("8_langevin", stage_langevin),
        ("9_fem", stage_fem),
        ("10_interpolation", stage_interpolation),
        ("11_stability", stage_stability),
        ("12_benchmark", stage_benchmark),
    ]
    failures = []
    for name, fn in stages:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [STAGE FAILED] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failures.append((name, str(e)))

    elapsed = time.time() - t0
    section("SUMMARY")
    print(f"  Total wall time: {elapsed:.2f} s")
    print(f"  Stages completed: {len(stages) - len(failures)}/{len(stages)}")
    if failures:
        print("  FAILURES:")
        for name, err in failures:
            print(f"    {name}: {err}")
        return 1

    # Final sanity: check all key outputs are finite
    finite_check = []
    if not math.isfinite(results['2_fd_stencil']['phi_max']):
        finite_check.append("phi_max")
    if not math.isfinite(results['4_poisson_srh']['J_rec_A_m2']):
        finite_check.append("J_rec")
    if not math.isfinite(results['7_ml']['rf_mae']):
        finite_check.append("rf_mae")
    if finite_check:
        print(f"  WARNING: non-finite values detected: {finite_check}")
        return 1

    print("\n  All stages completed successfully.")
    print("  Key physical results:")
    print(f"    - Max electrostatic potential: "
          f"{results['4_poisson_srh']['phi_max_V']:.4f} V")
    print(f"    - Recombination current density: "
          f"{results['4_poisson_srh']['J_rec_A_m2']:.3e} A/m^2")
    print(f"    - Debye length: "
          f"{results['4_poisson_srh']['Debye_length_m']:.3e} m")
    print(f"    - ML (RF) MAE for defect transition levels: "
          f"{results['7_ml']['rf_mae']:.4f} eV")
    print(f"    - von Neumann stable r (p={pc.FD_ORDER}): "
          f"{results['11_stability']['von_neumann'][pc.FD_ORDER]['r_max']:.6f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
