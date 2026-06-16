"""
main.py
=======
Unified entry point for PROJECT_247 -- a small-scale, reproducible
博士级 computation of dark matter halo formation, merger tree
construction and high-order finite-difference stability analysis.

Running
-------
    $ python main.py

No command-line arguments are required.  The script executes the
full pipeline:

    1. Seed subhalo positions  (gauss_prime_seeding)
    2. Build a spectral halo potential  (halo_gegenbauer_potential)
    3. Interpolate a halo orbit  (trigonometric_orbit_interpolator)
    4. Enumerate, simplify and assess resilience of merger trees
       (merger_tree_enumerator)
    5. Find the optimal merger path  (minimal_merger_path)
    6. Coarse-grain the halo phase-space distribution
       (phase_space_flow_coarse_grain)
    7. Integrate the density-oscillation and halo-precession ODEs
       (halo_ode_systems)
    8. Advance the 4-zone accretion ring  (multizone_accretion)
    9. Advance the convection-driven vortex  (convection_vortex_halo)
   10. Estimate the halo mass by least-squares quadrature
       (least_squares_quadrature)
   11. Analyse all trees in a parallel pool  (analysis_pool)
   12. Run von Neumann stability analysis of the finite-difference
       schemes  (stability_von_neumann)

A compact summary is printed to stdout at the end.
"""

from __future__ import annotations
import math
import random
import time
from typing import Dict, Any

import numpy as np

# Local modules
import halo_gegenbauer_potential as hp
import trigonometric_orbit_interpolator as toi
import merger_tree_enumerator as mte
import minimal_merger_path as mmp
import phase_space_flow_coarse_grain as psg
import halo_ode_systems as hodes
import gauss_prime_seeding as gps
import multizone_accretion as mza
import convection_vortex_halo as cvh
import least_squares_quadrature as lsq
import analysis_pool as ap
import stability_von_neumann as svn


# ---------- Pipeline stages ---------------------------------------------------

def stage_seeding() -> Dict[str, Any]:
    pts = gps.seed_subhalo_positions(n_sub=24,
                                     c0=8 + 2j, d0=1 + 0j,
                                     scale=0.05, seed=2024)
    return {"subhalo_positions": pts,
            "shape": list(pts.shape)}


def stage_halo_potential() -> Dict[str, Any]:
    data = hp.build_halo_potential(n_order=16, alpha=0.5,
                                   rs=0.25, rvir=1.0, rho_s=1.0)
    r = np.linspace(0.0, 1.0, 25)
    phi = hp.halo_potential_at(r, data["coeffs"],
                               data["rs"], data["rvir"], data["rho_s"])
    return {"n_order": data["n_order"],
            "n_coeffs": data["coeffs"].size,
            "phi_mid": float(phi[len(phi) // 2])}


def stage_orbit_interpolation() -> Dict[str, Any]:
    t, r, phi = toi.make_keplerian_orbit(a=1.0, ecc=0.12,
                                         period=2.0 * math.pi,
                                         n_samples=17)
    t_fine = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    out = toi.interpolate_orbit(t, r, phi, t_fine)
    return {"r_max": float(out["r"].max()),
            "dr_dt_rms": float(np.sqrt(np.mean(out["dr_dt"] ** 2)))}


def stage_merger_tree() -> Dict[str, Any]:
    rng = random.Random(2024)
    seq = mte.bal_seq_random(8, rng=rng)
    root = mte.dyck_to_tree(seq, rng=rng)
    mte.mark_main_branch(root)
    removed = mte.simplify_tree(root, eta_min=0.02)
    R = mte.resilience_index(root, removal_fraction=0.3, rng=rng)
    return {"catalan_C8": mte.bal_seq_enum(8),
            "n_removed": removed,
            "resilience": R}


def stage_minimal_merger_path() -> Dict[str, Any]:
    rng = random.Random(7)
    prog = [(10.5 + 0.3 * rng.gauss(0, 1), 0.15 * k + rng.gauss(0, 0.05))
            for k in range(8)]
    res = mmp.optimal_merger_path(prog, n_samples=500, seed=7)
    return {"L_random": res["length_random"],
            "L_opt": res["length_opt"]}


def stage_phase_space_flow() -> Dict[str, Any]:
    flow = psg.build_halo_coarse_graining(dim=4, seed=42)
    diag = psg.liouville_deviation(flow, n_samples=100, seed=1)
    sc = psg.self_check()
    return {"mean_liouville_deviation": diag["mean_abs_deviation"],
            "roundtrip_err": sc["roundtrip_err"]}


def stage_ode_systems() -> Dict[str, Any]:
    osc = hodes.DensityOscillator()
    y0 = np.array([0.8, 0.3, 0.5])
    ts, ys = hodes.integrate_trajectory(osc.rhs, y0, (0.0, 2.0), 200)
    gyro = hodes.HaloPrecession()
    y0g = np.array([0.1, 1.0, 0.0, 0.5, 0.3, 1.0])
    tsg, ysg = hodes.integrate_trajectory(gyro.rhs, y0g, (0.0, 2.0), 200)
    return {"osc_final": ys[-1].tolist(),
            "gyro_final": ysg[-1].tolist()}


def stage_multizone_accretion() -> Dict[str, Any]:
    ring = mza.FourZoneAccretionRing(n_cells_per_column=16)
    hist = ring.advance(dt=0.02, n_steps=50, species="baryon")
    return {name: {"final": float(arr[-1]),
                   "mean": float(arr.mean())}
            for name, arr in hist.items()}


def stage_convection_vortex() -> Dict[str, Any]:
    grid = cvh.VortexGrid(nR=17, nz=17)
    params = cvh.VortexParams()
    R, Z = np.meshgrid(grid.R, grid.z, indexing="ij")
    omega0 = 0.1 * np.exp(-(R - 0.5) ** 2 / 0.1 - Z ** 2 / 0.2)
    omega1 = cvh.advance_vortex(grid, params, omega0, dt=1e-3, n_steps=15)
    return {"mean_omega_initial": float(omega0.mean()),
            "mean_omega_final": float(omega1.mean()),
            "max_omega_final": float(omega1.max())}


def stage_halo_mass_quadrature() -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    # NFW-like sampling
    r = np.sort(rng.uniform(0.05, 1.0, size=40))
    rs = 0.25
    rho = (1.0 / (r / rs)) / (1.0 + r / rs) ** 2
    mass, W = lsq.halo_mass_integral(r, rho, degree=4)
    sc = lsq.self_check()
    return {"mass_estimate": mass,
            "quadrature_test_err": sc["abs_error"]}


def stage_analysis_pool() -> Dict[str, Any]:
    pool = ap.AnalysisPool(n_workers=2)

    def work(n: int) -> Dict[str, float]:
        rng = random.Random(n)
        seq = mte.bal_seq_random(n, rng=rng)
        root = mte.dyck_to_tree(seq, rng=rng)
        mte.mark_main_branch(root)
        removed = mte.simplify_tree(root, eta_min=0.02)
        R = mte.resilience_index(root, 0.3, rng=rng)
        return {"n": n, "removed": removed, "resilience": R}

    results = pool.run([4, 5, 6, 7, 8, -1], work)
    return {"n_jobs": len(results),
            "summary": results}


def stage_stability() -> Dict[str, Any]:
    return svn.stability_report()


# ---------- Runner ------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("PROJECT_247 : Dark matter halo formation and merger tree")
    print("             High-order finite-difference stability analysis")
    print("=" * 68)
    t0 = time.time()
    stages = [
        ("1. Gaussian-prime seeding",           stage_seeding),
        ("2. Gegenbauer halo potential",        stage_halo_potential),
        ("3. Trigonometric orbit interpolation",stage_orbit_interpolation),
        ("4. Merger-tree enumeration",          stage_merger_tree),
        ("5. Optimal merger path",              stage_minimal_merger_path),
        ("6. Phase-space coarse-graining",      stage_phase_space_flow),
        ("7. Halo ODE systems",                 stage_ode_systems),
        ("8. Four-zone accretion ring",         stage_multizone_accretion),
        ("9. Convection-driven vortex",         stage_convection_vortex),
        ("10. Least-squares halo mass",         stage_halo_mass_quadrature),
        ("11. Parallel analysis pool",          stage_analysis_pool),
        ("12. von Neumann stability analysis",  stage_stability),
    ]
    summary = {}
    for label, fn in stages:
        print(f"\n[{label}]")
        try:
            res = fn()
            summary[label] = res
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"    {k:40s} = {v:.6g}")
                elif isinstance(v, int):
                    print(f"    {k:40s} = {v}")
                elif isinstance(v, list) and len(v) <= 8:
                    # print small lists of numbers compactly
                    disp = [f"{x:.4g}" if isinstance(x, float) else str(x)
                            for x in v]
                    print(f"    {k:40s} = [{', '.join(disp)}]")
                elif isinstance(v, np.ndarray):
                    if v.size <= 8:
                        print(f"    {k:40s} = {v.tolist()}")
                    else:
                        print(f"    {k:40s} = <ndarray shape={v.shape}>")
                elif isinstance(v, dict):
                    print(f"    {k:40s} = <dict with {len(v)} entries>")
                else:
                    print(f"    {k:40s} = <{type(v).__name__}>")
        except Exception as exc:          # noqa: BLE001
            print(f"    FAILED: {exc}")
            summary[label] = {"error": str(exc)}
    dt = time.time() - t0
    print("\n" + "=" * 68)
    print(f"Pipeline completed in {dt:.3f} s")
    print("=" * 68)


if __name__ == "__main__":
    main()
