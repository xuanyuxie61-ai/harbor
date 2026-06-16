# -*- coding: utf-8 -*-
"""
main.py  --  unified entry point
================================
Computational condensed-matter study of a simple superconductor:

    "Electron-phonon coupling and superconducting transition temperature
     prediction via high-order finite differences and stability analysis
     (small-scale reproducible experiment)."

Run this file with no arguments:

    python main.py

The script executes an 11-stage pipeline that covers

  1. adaptive k-point mesh (CVT on the half-BZ)
  2. Brillouin-zone construction (Minkowski sum of reciprocal vectors)
  3. phonon-shell degeneracies (Diophantine enumeration)
  4. Matsubara-axis Eliashberg kernel (cosine integral)
  5. spectral expansion of the gap (Chebyshev basis)
  6. Tc search by eigenvalue bracketing (EMA-accelerated)
  7. gap quantisation (k-means on the Matsubara grid)
  8. Monte-Carlo uncertainty quantification
  9. optimal phonon-mode subset (submodular maximisation)
 10. coupled kinetic steady-state check
 11. numerical-stability report (von Neumann / energy / CFL)

Scientific-origin mapping (15 seed projects)
--------------------------------------------
* 837_opt_sample                   -> sampling strategy in stage 8 (MC)
* 1323_triangle_twb_rule           -> BZ quadrature in stage 1
* 894_polynomial_conversion        -> gap basis conversion in stage 5
* 742_mcnuggets                    -> phonon-shell degeneracy in stage 3
* 1106_ocular-surface-ion-transport -> kinetic rate equations in stage 10
* 583_image_quantization           -> gap quantisation in stage 7
* 1174_MIMUW-RL_time-series        -> EMA optimiser in stage 6
* 245_cvt_1d_nonuniform            -> adaptive k-mesh in stage 1
* 1057_Bio-Inspired-Navigation     -> pipeline orchestration
* 1355_tridiagonal_solver          -> compact FD solver in stage 4/6
* 277_dice_simulation              -> MC sampling engine in stage 8
* 221_cosine_integral              -> special functions in stage 4
* 1224_GraphComBO-Blind            -> graph optimisation in stage 9
* 887_polygon_minkowski            -> BZ as polygon in stage 2
* 1077_cisc662 (QAOA / HPC)        -> reproducible-seed HPC ethos throughout
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Make the current directory importable when invoked as "python main.py"
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Helpers for textual reporting (no plotting)
# ---------------------------------------------------------------------------
def header(title: str, char: str = "=") -> str:
    line = char * 72
    return f"\n{line}\n  {title}\n{line}"


def report_kpoint_mesh(res):
    k = res["k_points"]
    w = res["weights"]
    print(f"    # k-points     : {k.size}")
    print(f"    k in [0, pi]   : min = {k.min():.6f}, max = {k.max():.6f}")
    print(f"    sum(w_k)       : {w.sum():.6f}  (should be pi = {math.pi:.6f})")
    print(f"    first 5 k      : {k[:5].tolist()}")


def report_bz(res):
    bz = res["bz"]
    print(f"    BZ area        : {bz.area:.6f}")
    print(f"    # vertices     : {bz.vertices.shape[0]}")
    print(f"    # generators   : {bz.reciprocal_vectors.shape[0]}")


def report_phonon_shells(res):
    spec = res["spectrum"]
    print(f"    # branches     : {spec.branch_multiplicities.size}")
    print(f"    max shell      : {spec.max_shell}")
    print(f"    Frobenius #    : {spec.frobenius_number}")
    print(f"    lambda_eff     : {res['lambda_eff']:.6f}")
    print(f"    first 5 W(N)   : {spec.degeneracy[:5].tolist()}")


def report_kernel(res):
    ker = res["kernel"]
    print(f"    # Matsubara    : {ker.omega_n.size}")
    print(f"    trace(K_ret)   : {ker.lambda_ret:.6f}")
    print(f"    omega_0        : {ker.omega_n[0]:.6e}")


def report_gap_expansion(res):
    exp = res["expansion"]
    print(f"    basis          : {exp.basis}")
    print(f"    order          : {exp.n_order}")
    print(f"    cond(M)        : {exp.cond_number:.3e}")
    print(f"    first 5 coeffs : {exp.coeffs[:5].tolist()}")


def report_tc_search(res):
    tc_res = res["tc_result"]
    print(f"    Tc estimate    : {tc_res.Tc_estimate:.4f} K")
    print(f"    # search temps : {len(tc_res.search_temperatures)}")
    print(f"    eigenvalue at Tc : {tc_res.lambda_at_Tc:.4f}")


def report_gap_quantization(res):
    gq = res["gap_quantization"]
    print(f"    # levels       : {gq.centres.size}")
    print(f"    quant. error   : {gq.quantisation_error:.6e}")
    print(f"    compression    : {gq.compression_ratio:.2f}x")
    print(f"    centres        : {gq.centres.tolist()}")


def report_monte_carlo(res):
    mc = res["monte_carlo"]
    print(f"    # samples      : {mc.tc_samples.size}")
    print(f"    # rejected     : {mc.n_rejected}")
    print(f"    Tc mean        : {mc.tc_mean:.4f} K")
    print(f"    Tc std         : {mc.tc_std:.4f} K")
    print(f"    Tc median      : {mc.tc_median:.4f} K")
    print(f"    Tc [5%, 95%]   : [{mc.tc_5pct:.4f}, {mc.tc_95pct:.4f}]")


def report_optimal_subset(res):
    opt = res["optimal_subset"]
    print(f"    selected modes : {opt.selected_indices}")
    print(f"    frequencies    : {opt.selected_frequencies.tolist()}")
    print(f"    lambda_eff     : {opt.effective_lambda:.4f}")
    print(f"    omega_log      : {opt.omega_log:.6f} eV")
    print(f"    Tc (McMillan)  : {opt.tc_mcmillan:.4f} K")


def report_kinetic(res):
    sw = res["kinetic_sweep"]
    print(f"    # temperatures : {sw.temperatures.size}")
    valid = ~np.isnan(sw.n_e)
    if valid.any():
        print(f"    n_e range      : [{np.nanmin(sw.n_e):.3e}, {np.nanmax(sw.n_e):.3e}]")
        print(f"    n_p range      : [{np.nanmin(sw.n_p):.3e}, {np.nanmax(sw.n_p):.3e}]")
        print(f"    pairing range  : [{np.nanmin(sw.pairing_rates):.3e}, {np.nanmax(sw.pairing_rates):.3e}]")


def report_stability(res):
    vn = res["von_neumann"]
    em = res["energy_method"]
    cfl = res["cfl"]
    ms = res["matrix_stability"]
    print(f"    Von Neumann stable : {vn.stable}")
    print(f"    max |G|            : {vn.max_amplification:.6f}")
    print(f"    critical dt        : {vn.critical_dt:.6e}")
    print(f"    Energy-method rho  : {em.spectral_radius:.6f}")
    print(f"    stable dt bound    : {em.stable_dt_bound:.6e}")
    print(f"    CFL number         : {cfl.cfl_number:.4f}")
    print(f"    Eliashberg rho     : {ms.spectral_radius:.6f}")
    print(f"    Eliashberg conv.   : {ms.converged}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(header("PROJECT 274 -- Electron-Phonon Coupling and Tc Prediction", "="))
    print("  High-order finite differences, stability analysis,")
    print("  small-scale reproducible experiment.")
    print(f"  seed = 274, numpy version = {np.__version__}")

    from pipeline import PipelineConfig, TcPredictionPipeline

    cfg = PipelineConfig(
        n_kpoints=32,
        cvt_density_index=2,
        seed=274,
        n_einstein_branches=6,
        branch_multiplicities=(1, 2, 3, 4, 5, 6),
        omega_E=0.025,
        max_phonon_shell=50,
        V_ph=0.30,
        mu_star=0.10,
        n_matsubara=64,
        fd_order=4,
        T_search_low=1.0,
        T_search_high=20.0,
        n_T_search=24,
        ema_decay=0.95,
        n_gap_levels=8,
        n_mc_samples=4000,
        sigma_lambda=0.05,
        sigma_mu=0.02,
        n_selected_modes=3,
    )
    pipeline = TcPredictionPipeline(cfg)

    print(header("Pipeline execution", "-"))
    results = pipeline.run()

    print(header("Results summary", "-"))
    print("\n[Stage 1] Adaptive k-point mesh")
    report_kpoint_mesh(results["kpoint_mesh"])
    print("\n[Stage 2] Brillouin zone")
    report_bz(results["brillouin_zone"])
    print("\n[Stage 3] Phonon-shell spectrum")
    report_phonon_shells(results["phonon_shells"])
    print("\n[Stage 4] Eliashberg kernel")
    report_kernel(results["eliashberg_kernel"])
    print("\n[Stage 5] Spectral gap expansion")
    report_gap_expansion(results["gap_expansion"])
    print("\n[Stage 6] Tc search")
    report_tc_search(results["tc_search"])
    print("\n[Stage 7] Gap quantisation")
    report_gap_quantization(results["gap_quantization"])
    print("\n[Stage 8] Monte-Carlo uncertainty")
    report_monte_carlo(results["monte_carlo"])
    print("\n[Stage 9] Optimal phonon subset")
    report_optimal_subset(results["optimal_subset"])
    print("\n[Stage 10] Kinetic steady-state check")
    report_kinetic(results["kinetic_check"])
    print("\n[Stage 11] Numerical stability")
    report_stability(results["stability"])

    print(header("Pipeline timings", "-"))
    for k, v in pipeline._timings.items():
        print(f"    {k:15s}: {v:.3f} s")

    print(header("Done.", "="))
    return 0


if __name__ == "__main__":
    sys.exit(main())
