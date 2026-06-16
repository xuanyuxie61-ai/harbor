# -*- coding: utf-8 -*-
"""
PROJECT 271
===========

Computational condensed-matter study of the 1D Transverse-Field Ising
Model quantum phase transition.

Scope
-----
This unified entry point performs, in sequence:

  (1) Benchmark-kernel verification   (NAS-style sanity checks)
  (2) High-order FD stencil stability analysis
  (3) Exact diagonalisation of TFIM for several L
  (4) Finite-size scaling analysis of E_0, gap, Binder cumulant
  (5) Real-space RG flow and critical exponent extraction
  (6) Adaptive ESN/RLS estimation of lambda_c
  (7) L-BFGS-B fit of an effective Hamiltonian
  (8) Path-integral Monte Carlo of the Binder cumulant
  (9) Elliptic-integral based exact free-energy density
  (10) CVT sampling of the Brillouin zone
  (11) Boundary-mesh construction for the Ginzburg-Landau FEM
  (12) Spectral-function calibration of S(q, omega)
  (13) NACA dispersion fit and MC quadrature of thermodynamics
  (14) Monte-Carlo quadrature of the free-energy density

All output is textual; no visualisation is produced.  Every step is
designed to be *small-scale and reproducible* (runtime on the order
of seconds on a single CPU) while exercising the full method stack.

Usage
-----
    python main.py

No command-line arguments are required.
"""

from __future__ import annotations
import time
import sys
import numpy as np

# Ensure package imports work when the directory is run directly.
import os as _os
from importlib import import_module
_dir = _os.path.dirname(_os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)


def _import(name: str):
    """Import a sibling module by absolute name (the directory has been
    added to sys.path)."""
    return import_module(name)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
def banner(msg: str) -> None:
    bar = "=" * 72
    print()
    print(bar)
    print(f"  {msg}")
    print(bar)


# ---------------------------------------------------------------------------
# Stage 1  -- benchmark kernels
# ---------------------------------------------------------------------------
def stage_benchmarks() -> dict:
    banner("Stage 1 : NAS-style benchmark kernels")
    bk = _import("benchmark_kernels")
    res = bk.run_all_kernels(verbose=False)
    for k, v in res.items():
        if k == "all_passed":
            continue
        print(f"  {k:22s} residual = {v:.3e}")
    print(f"  all_passed = {res['all_passed']}")
    return res


# ---------------------------------------------------------------------------
# Stage 2  -- high-order FD stencil stability
# ---------------------------------------------------------------------------
def stage_fd_stability() -> dict:
    banner("Stage 2 : high-order FD stability analysis")
    fd = _import("high_order_fd")
    sa = _import("stability_analysis")

    for order in (2, 4, 6, 8):
        rep = sa.analyse_stencil(order)
        w = fd.central_fd_weights(order, 2)
        leb = fd.stencil_lebesgue_constant(w)
        print(f"  order={order:2d}  sigma_max={rep.sigma_max:+.4e}  "
              f"c_trunc={rep.c_trunc:+.4e}  dt_expl={rep.dt_expl:.4e}  "
              f"dt_leap={rep.dt_leap:.4e}  Lebesgue={leb:.3e}  "
              f"well_posed={rep.well_posed}")
    return {"orders": [2, 4, 6, 8]}


# ---------------------------------------------------------------------------
# Stage 3  -- exact diagonalisation
# ---------------------------------------------------------------------------
def stage_exact_diag() -> dict:
    banner("Stage 3 : exact diagonalisation of TFIM")
    tfim = _import("tfim_hamiltonian")
    ed = _import("elliptic_determinants")

    Ls = [4, 6, 8, 10]
    lams = [0.5, 0.9, 1.0, 1.1, 1.5]
    out = {}
    for L in Ls:
        e0_list = []
        gap_list = []
        for lam in lams:
            e0 = tfim.ground_state_energy(L, 1.0, lam)
            g = tfim.gap(L, 1.0, lam)
            e0_exact = ed.ground_state_energy_exact(L, 1.0, lam)
            e0_list.append(e0)
            gap_list.append(g)
            print(f"  L={L:2d}  lam={lam:.2f}  "
                  f"E_0(exact)={e0:.6f}  E_0(elliptic)={e0_exact:.6f}  "
                  f"gap={g:.4e}")
        out[L] = {"e0": e0_list, "gap": gap_list}
    return out


# ---------------------------------------------------------------------------
# Stage 4  -- finite-size scaling
# ---------------------------------------------------------------------------
def stage_fss() -> dict:
    banner("Stage 4 : finite-size scaling analysis")
    tfim = _import("tfim_hamiltonian")
    fss = _import("finite_size_scaling")
    fd = _import("high_order_fd")

    Ls = [4, 6, 8]
    lams = np.linspace(0.8, 1.2, 9)
    # Compute d^2 E_0 / d lam^2 at each L, lam  (specific heat analogue)
    C_by_L = []
    for L in Ls:
        def e0_of_lam(lam):
            return tfim.ground_state_energy(L, 1.0, float(lam))
        c_vals = []
        for lam in lams:
            try:
                d2 = fd.derivative_at_lambda(e0_of_lam, float(lam),
                                               deriv=2, order=4,
                                               n_richardson=2, h0=5.0e-3)
            except Exception:
                d2 = 0.0
            c_vals.append(d2)
        C_by_L.append(np.array(c_vals))
        pk = fss.peak_location(lams, np.array(c_vals))
        print(f"  L={L:2d}  peak(lambda) = {pk:.4f}")

    # Peak-shift fit
    peak_lams = np.array([fss.peak_location(lams, c) for c in C_by_L])
    lam_c, a = fss.peak_shift_fit(np.array(Ls), peak_lams)
    print(f"  FSS lambda_c (peak shift) = {lam_c:.5f}  (exact: 1.00000)")
    print(f"  fit amplitude a = {a:+.4e}")

    # Binder cumulant crossing
    from mc_path_integral import run_simulation
    Ls_b = [6, 8, 10]
    binder_curves = {}
    for L in Ls_b:
        bs = []
        for lam in [0.95, 1.0, 1.05]:
            out = run_simulation(L, M=8, J=1.0, h=lam,
                                   n_sweeps=20, n_warmup=10, seed=L)
            bs.append(out["binder"])
        binder_curves[L] = np.array(bs)
        print(f"  L={L}  U(0.95,1.0,1.05) = {binder_curves[L]}")
    if len(Ls_b) >= 2:
        lams_b = np.array([0.95, 1.0, 1.05])
        lc_cross = fss.crossing_point(lams_b, binder_curves[Ls_b[0]],
                                         binder_curves[Ls_b[1]])
        print(f"  Binder crossing (L={Ls_b[0]}/{Ls_b[1]}) = {lc_cross:.4f}")

    return {"Ls": Ls, "lams": lams, "lam_c": lam_c}


# ---------------------------------------------------------------------------
# Stage 5  -- real-space RG flow
# ---------------------------------------------------------------------------
def stage_rg_flow() -> dict:
    banner("Stage 5 : real-space RG flow")
    rg = _import("rg_flow")

    for which in ["perturbative", "two_loop"]:
        fp = rg.find_fixed_point(
            (lambda g: rg.beta_perturbative(g)) if which == "perturbative"
            else rg.beta_two_loop,
            g_guess=1.0)
        nu = rg.critical_exponent_nu(which)
        print(f"  beta ({which:13s})  fixed point g* = {fp:.5f}  "
              f"nu = {nu:.5f}")

    # Vector flow
    ln_b, traj = rg.integrate_flow_vector(
        np.array([1.0, 0.8, 0.8]), ln_b_max=4.0, n_steps=200)
    print(f"  3-coupling flow: (J, h, g) at ln b = {ln_b[-1]:.2f} -> "
          f"({traj[-1, 0]:.4f}, {traj[-1, 1]:.4f}, {traj[-1, 2]:.4f})")
    return {"nu_perturbative": rg.critical_exponent_nu("perturbative"),
             "nu_two_loop": rg.critical_exponent_nu("two_loop")}


# ---------------------------------------------------------------------------
# Stage 6  -- ESN adaptive estimator
# ---------------------------------------------------------------------------
def stage_esn() -> dict:
    banner("Stage 6 : ESN/RLS adaptive estimator of lambda_c")
    re = _import("reservoir_estimator")
    tfim = _import("tfim_hamiltonian")
    fd = _import("high_order_fd")

    Ls = [4, 6, 8]
    lams = np.linspace(0.85, 1.15, 13)
    est = re.QCPAdaptiveEstimator(reservoir_size=80, spectral_radius=0.85,
                                    leakage=0.3, lam=0.995, seed=7)
    data_lam, data_L, data_e0, data_target = [], [], [], []
    for L in Ls:
        for lam in lams:
            e0 = tfim.ground_state_energy(L, 1.0, float(lam))
            def f_(ll):
                return tfim.ground_state_energy(L, 1.0, float(ll))
            try:
                d2 = fd.derivative_at_lambda(f_, float(lam), deriv=2,
                                               order=4, n_richardson=2,
                                               h0=5.0e-3)
            except Exception:
                d2 = 0.0
            data_lam.append(float(lam))
            data_L.append(float(L))
            data_e0.append(float(e0))
            data_target.append(float(d2))

    preds = re.train_estimator(est, np.array(data_lam), np.array(data_L),
                                 np.array(data_e0), np.array(data_target),
                                 warmup=20)
    lam_c_est = est.estimate_qcp()
    print(f"  ESN estimate of lambda_c = {lam_c_est:.5f}  (exact 1.00000)")
    return {"lambda_c_esn": lam_c_est}


# ---------------------------------------------------------------------------
# Stage 7  -- L-BFGS-B effective Hamiltonian fit
# ---------------------------------------------------------------------------
def stage_lbfgs() -> dict:
    banner("Stage 7 : L-BFGS-B effective Hamiltonian fit")
    lb = _import("lbfgs_hamiltonian_fitter")
    tfim = _import("tfim_hamiltonian")

    Ls = np.array([4, 6, 8, 10])
    lams_arr = np.linspace(0.7, 1.3, 20)
    L_grid, lam_grid = np.meshgrid(Ls, lams_arr, indexing="ij")
    L_grid = L_grid.ravel()
    lam_grid = lam_grid.ravel()
    E_target = np.array([tfim.ground_state_energy(int(L), 1.0, float(lam))
                          for L, lam in zip(L_grid, lam_grid)])
    fit = lb.fit_effective_hamiltonian(L_grid, lam_grid, E_target,
                                         max_iter=120)
    print(f"  fitted J = {fit['theta_dict']['J']:.5f}")
    print(f"  fitted V = {fit['theta_dict']['V']:+.5f}")
    print(f"  fitted K = {fit['theta_dict']['K']:+.5f}")
    print(f"  residual = {fit['fun']:.4e}   success = {fit['success']}")
    cv = lb.kfold_score(L_grid, lam_grid, E_target, k_folds=3)
    print(f"  3-fold CV MSE = {cv:.4e}")
    return fit


# ---------------------------------------------------------------------------
# Stage 8  -- path-integral MC
# ---------------------------------------------------------------------------
def stage_mc() -> dict:
    banner("Stage 8 : path-integral Monte Carlo")
    mc = _import("mc_path_integral")
    for L in [6, 8, 10]:
        for lam in [0.9, 1.0, 1.1]:
            out = mc.run_simulation(L, M=8, J=1.0, h=lam,
                                      n_sweeps=30, n_warmup=15, seed=L + 100)
            print(f"  L={L}  lam={lam:.1f}  |m|={out['mag_mean']:.4f}  "
                  f"binder={out['binder']:.4f}  e={out['energy_mean']:.4f}")
    return {"done": True}


# ---------------------------------------------------------------------------
# Stage 9  -- elliptic integrals / free energy
# ---------------------------------------------------------------------------
def stage_elliptic() -> dict:
    banner("Stage 9 : elliptic-integral free energy + fidelity suscept.")
    ed = _import("elliptic_determinants")
    mc_q = _import("mc_quadrature")

    for lam in [0.5, 0.9, 1.0, 1.1, 1.5]:
        e0 = ed.ground_state_energy_exact(L=10, J=1.0, h=lam)
        chi_f = ed.fidelity_susceptibility(L=10, J=1.0, h=lam)
        print(f"  lam={lam:.1f}  E_0(10,elliptic)={e0:.6f}  "
              f"chi_F={chi_f:.4e}")

    # MC quadrature of the free-energy density
    f, se = mc_q.free_energy_density(L=10, J=1.0, h=1.0, beta=4.0,
                                        n_samples=4000, seed=0)
    print(f"  f(L=10, beta=4, lam=1) = {f:.6f} +/- {se:.4e}")

    # Ellipsoid-area analogue for anisotropic couplings
    S = ed.ellipsoid_area_analogue(a=1.2, b=1.0, c=0.8)
    print(f"  ellipsoid-area surrogate (1.2,1.0,0.8) = {S:.4f}")
    return {"f": f, "se": se}


# ---------------------------------------------------------------------------
# Stage 10  -- CVT Brillouin-zone sampler
# ---------------------------------------------------------------------------
def stage_cvt() -> dict:
    banner("Stage 10 : CVT sampling of the Brillouin zone")
    cvt = _import("cvt_sampler")
    for n_pts in [8, 16, 32]:
        ps = cvt.cvt_1d(n_pts, n_samples=2048, n_iter=30, seed=0)
        E = cvt.cvt_energy(ps, n_samples=2048)
        print(f"  n_pts={n_pts:2d}  CVT energy = {E:.4e}  "
              f"first/last generator = {ps[0]:+.4f} / {ps[-1]:+.4f}")
    # Fermi circle surrogate
    kx, ky = cvt.fermi_circle(J=1.0, h=0.9, n_points=16)
    print(f"  Fermi circle (lam=0.9): 16 points, "
          f"radius ~ {np.sqrt(kx[0]**2 + ky[0]**2):.4f}")
    return {"done": True}


# ---------------------------------------------------------------------------
# Stage 11  -- boundary mesh
# ---------------------------------------------------------------------------
def stage_boundary() -> dict:
    banner("Stage 11 : boundary mesh for Ginzburg-Landau FEM")
    bm = _import("boundary_mesh")

    nx, ny = 8, 8
    Lx, Ly = 1.0, 1.0
    nodes = bm.rectangle_nodes(nx, ny, Lx, Ly)
    flags = bm.boundary_flags(nodes, Lx, Ly)
    edges = bm.rectangle_edges(nx, ny)
    print(f"  mesh: {nx}x{ny} = {len(nodes)} nodes, {len(edges)} edges")
    print(f"  boundary nodes (any flag): {int((flags != 0).sum())}")

    # Region predicates
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside = bm.polygon_contains_point(poly, np.array([0.5, 0.5]))
    outside = bm.polygon_contains_point(poly, np.array([1.5, 0.5]))
    print(f"  point-in-polygon (inside/outside): {inside}, {outside}")

    # Neumann test function
    phi = bm.neumann_test_function(nodes, Lx, Ly)
    print(f"  Neumann test function: max = {phi.max():.4f}, "
          f"min = {phi.min():.4f}")
    print(f"  boundary measure = {bm.boundary_measure(Lx, Ly):.4f}")
    return {"n_nodes": len(nodes)}


# ---------------------------------------------------------------------------
# Stage 12  -- spectral function calibration
# ---------------------------------------------------------------------------
def stage_spectral() -> dict:
    banner("Stage 12 : spectral-function calibration of S(q, omega)")
    sc = _import("spectral_calibration")

    L = 8
    for lam in [0.9, 1.0, 1.1]:
        omega, S = sc.calibrated_dynamical_sf(L, J=1.0, h=lam,
                                                 n_q=4, n_omega=80,
                                                 omega_max=6.0, eta=0.1)
        peak = omega[int(np.argmax(S))]
        print(f"  L={L} lam={lam:.1f}  S_peak at omega = {peak:.3f}  "
              f"max S = {np.max(S):.4e}")

    # Static susceptibility
    chi = sc.static_susceptibility(L=8, J=1.0, h=0.9, n_q=4)
    print(f"  static chi(L=8, lam=0.9) = {chi:.4e}")
    return {"done": True}


# ---------------------------------------------------------------------------
# Stage 13  -- NACA dispersion fit
# ---------------------------------------------------------------------------
def stage_naca() -> dict:
    banner("Stage 13 : NACA dispersion fit")
    nd = _import("naca_dispersion")
    fit = nd.fit_naca_to_tfim(L=10, J=1.0, h=1.0)
    print(f"  NACA fit: t={fit['t']:.4f}  m={fit['m']:.4f}  p={fit['p']:.4f}  "
          f"residual={fit['residual']:.4e}")
    dos = nd.naca_dos_at_zero(t=fit["t"], J=1.0)
    print(f"  DOS coefficient at omega=0: {dos:.4e}")
    return fit


# ---------------------------------------------------------------------------
# Stage 14  -- MC quadrature of thermodynamic integrals
# ---------------------------------------------------------------------------
def stage_mc_quad() -> dict:
    banner("Stage 14 : MC quadrature of thermodynamic integrals")
    mc_q = _import("mc_quadrature")

    # 1-D test: Gaussian integral  int_{-3}^{3} exp(-x^2) dx = sqrt(pi) erf(3)
    import math
    est, se = mc_q.mc_quadrature_1d(lambda x: np.exp(-x * x),
                                       -3.0, 3.0,
                                       n_samples=5000, seed=0)
    exact = math.sqrt(math.pi) * math.erf(3.0)
    print(f"  Gaussian integral: est = {est:.6f}  "
          f"exact = {exact:.6f}  err = {abs(est - exact):.3e}")

    # Control-variate
    est_cv, se_cv = mc_q.control_variate_estimate(
        f=lambda x: np.exp(-x * x) * (1.0 + 0.1 * x),
        g=lambda x: np.exp(-x * x),
        a=-3.0, b=3.0,
        g_exact=exact,
        n_samples=5000, seed=1)
    print(f"  control-variate est = {est_cv:.6f}  se = {se_cv:.4e}")

    # Bootstrap
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=1.0, scale=0.1, size=200)
    lo, lb, ub = mc_q.bootstrap_statistic(samples, n_boot=300, seed=2)
    print(f"  bootstrap mean: {lo:.4f}  "
          f"95% CI = [{lb:.4f}, {ub:.4f}]")
    return {"done": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    print("PROJECT 271 : 1D TFIM quantum phase transition, "
          "high-order FD, finite-size scaling, stability analysis")
    print("  (small-scale reproducible experiment, Python)")

    results = {}
    results["benchmarks"] = stage_benchmarks()
    results["fd_stability"] = stage_fd_stability()
    results["exact_diag"] = stage_exact_diag()
    results["fss"] = stage_fss()
    results["rg_flow"] = stage_rg_flow()
    results["esn"] = stage_esn()
    results["lbfgs"] = stage_lbfgs()
    results["mc"] = stage_mc()
    results["elliptic"] = stage_elliptic()
    results["cvt"] = stage_cvt()
    results["boundary"] = stage_boundary()
    results["spectral"] = stage_spectral()
    results["naca"] = stage_naca()
    results["mc_quad"] = stage_mc_quad()

    banner("Summary")
    if results["benchmarks"]["all_passed"]:
        print("  [OK] all benchmark kernels PASSED")
    else:
        print("  [WARN] one or more benchmark kernels failed")
    lam_c_fss = results["fss"].get("lam_c", float("nan"))
    print(f"  FSS lambda_c (peak-shift) = {lam_c_fss:.5f}")
    lam_c_esn = results["esn"].get("lambda_c_esn", float("nan"))
    print(f"  ESN lambda_c (adaptive)   = {lam_c_esn:.5f}")
    nu_p = results["rg_flow"]["nu_perturbative"]
    nu_t = results["rg_flow"]["nu_two_loop"]
    print(f"  RG nu (perturbative) = {nu_p:.5f}")
    print(f"  RG nu (two-loop)     = {nu_t:.5f}")
    print(f"  exact TFIM nu = 1.00000, lambda_c = 1.00000")
    print(f"  wall time = {time.time() - t0:.2f} s")
    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
