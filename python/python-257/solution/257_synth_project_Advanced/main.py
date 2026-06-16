"""
main.py
=======
Unified entry point for the CMB power-spectrum estimation pipeline.

Scientific goal:
  Estimate the angular power spectrum C_l of the CMB temperature field
  using a discretised Laplace-Beltrami operator on a cubed-sphere mesh,
  validate the quadrature rules, analyse stability, fit cosmological
  parameters, and classify modes into E/B/systematics.

Pipeline stages:
  1. Build mesh                   (spherical_mesh.py)
  2. Assemble discrete Laplacian  (spherical_laplacian.py)
  3. Stability analysis            (stability_analysis.py)
  4. Convert to sparse CCS         (sparse_operator.py)
  5. Validate quadrature rules     (quadrature_validator.py)
  6. Monte-Carlo C_l estimation    (monte_carlo_spectrum.py)
  7. Beam simulation               (beam_simulator.py)
  8. Foreground model              (foreground_model.py)
  9. Mode classification           (mode_classifier.py)
 10. Parameter fitting             (parameter_optimizer.py)
 11. Gradient/contour analysis      (gradient_contour.py)
"""

from __future__ import annotations
import math
import sys
import os
import time
from typing import List, Dict, Tuple

# Ensure local directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# All project modules
import cosmology_params as cp
import spherical_mesh as sm
import fd_coefficients as fdc
import spherical_laplacian as sl
import stability_analysis as sa
import sparse_operator as so
import monte_carlo_spectrum as mc
import parameter_optimizer as po
import mode_classifier as mcmod
import foreground_model as fg
import beam_simulator as bs
import quadrature_validator as qv
import gradient_contour as gc


# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
def stage_1_mesh() -> sm.CubedSphere:
    banner("Stage 1:  Build cubed-sphere mesh")
    nside = 4
    mesh = sm.CubedSphere(nside=nside)
    print(f"  nside          = {nside}")
    print(f"  n_nodes        = {mesh.n_nodes}")
    print(f"  n_faces        = {mesh.n_faces}")
    print(f"  mean spacing   = {mesh.mean_pixel_spacing() * 180 / math.pi:.2f} deg")
    print(f"  solid angle    = {4.0 * math.pi / mesh.n_nodes:.4e} sr")
    return mesh


# ---------------------------------------------------------------------------
def stage_2_laplacian(mesh: sm.CubedSphere) -> List[List[float]]:
    banner("Stage 2:  Assemble discrete Laplace-Beltrami operator")
    L = sl.assemble_fd_laplacian(mesh)
    n = mesh.n_nodes
    diag_sum = sum(L[i][i] for i in range(n))
    off_diag = sum(abs(L[i][j]) for i in range(n) for j in range(n) if i != j)
    print(f"  Matrix size    = {n} x {n}")
    print(f"  Sum of diag    = {diag_sum:+.6e}")
    print(f"  Sum |off-diag| = {off_diag:+.6e}")
    return L


# ---------------------------------------------------------------------------
def stage_3_stability(L: List[List[float]]) -> Dict[str, float]:
    banner("Stage 3:  Stability analysis of FD operator")
    rho, _ = sa.spectral_radius(L)
    lam_min, lam_max, kappa = sa.eigenvalue_spread(L)
    dt_euler = sa.cfl_explicit_euler(L)
    dt_rk4 = sa.cfl_rk4(L)
    stable_be = sa.backward_euler_stable(L)
    # von Neumann check on a 5-point 2nd-derivative stencil
    w5 = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
    stable_vn, max_mag, max_k = sa.von_neumann_stability_check(w5, 0.1)

    results = {
        "spectral_radius": rho,
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "condition_number": kappa,
        "dt_euler": dt_euler,
        "dt_rk4": dt_rk4,
        "backward_euler_stable": stable_be,
        "von_neumann_stable": stable_vn,
        "von_neumann_max|G|": max_mag,
    }
    for k, v in results.items():
        print(f"  {k:25s} = {v}")
    return results


# ---------------------------------------------------------------------------
def stage_4_sparse(L: List[List[float]]) -> Tuple[int, List[int], List[int], List[float]]:
    banner("Stage 4:  Convert to sparse CCS format")
    n = len(L)
    nnz, colptr, rowind, values = so.ge_to_ccs(L)
    stats = so.ccs_stats(n, n, nnz)
    lb, ub = so.detect_bandwidth(colptr, rowind, n)
    is_sym = so.ccs_is_symmetric(n, n, colptr, rowind, values)
    frob = so.ccs_frobenius_norm(colptr, values)
    print(f"  nnz              = {nnz}")
    print(f"  total entries    = {stats['total']}")
    print(f"  sparsity         = {stats['sparsity']:.4f}")
    print(f"  compression      = {stats['compression_ratio']:.2f}x")
    print(f"  lower bandwidth  = {lb}")
    print(f"  upper bandwidth  = {ub}")
    print(f"  symmetric?       = {is_sym}")
    print(f"  Frobenius norm   = {frob:.6e}")
    return nnz, colptr, rowind, values


# ---------------------------------------------------------------------------
def stage_5_quadrature() -> List[Tuple[int, float, float]]:
    banner("Stage 5:  Validate quadrature rules")
    # Gauss-Legendre 8-point
    nodes, weights = qv.gauss_legendre_nodes_weights(8)
    print(f"  Gauss-Legendre 8-point nodes: {[f'{v:+.6f}' for v in nodes]}")
    # Chebyshev exactness
    results = qv.chebyshev1_exactness(nodes, weights, 20)
    print("  Monomial exactness (Chebyshev weight 1/sqrt(1-x^2)):")
    max_exact = 0
    for n, exact, err in results:
        if err < 1.0e-8:
            max_exact = n
        if n <= 10 or err > 1.0e-8:
            print(f"    degree {n:2d}: exact={exact:+.6e}, err={err:.4e}")
    print(f"  Exactness degree = {max_exact}")
    # Spherical quadrature
    val_res = qv.validate_spherical_quadrature(l_max=3, n_theta=8, n_phi=16)
    max_err = max(err for _, err in val_res)
    print(f"  Spherical |Y_lm|^2 normalisation max error = {max_err:.4e}")
    return results


# ---------------------------------------------------------------------------
def stage_6_power_spectrum(mesh: sm.CubedSphere) -> Dict[str, List[float]]:
    banner("Stage 6:  Monte-Carlo estimation of C_l")
    # Build synthetic CMB map: sum of l=2, m=0 and l=3, m=1
    alm_true = {(2, 0): 1.0e-4, (3, 1): 0.5e-4, (4, 2): 0.2e-4}
    T = mc.synthetic_cmb(alm_true)
    l_max = 6
    n_samples = 3000
    print(f"  l_max     = {l_max}")
    print(f"  n_samples = {n_samples}")
    alm_est = mc.mc_alm(T, l_max, n_samples, seed=7)
    cl_est = mc.cl_from_alm(alm_est, l_max)
    print("  Estimated C_l:")
    for ell in range(l_max + 1):
        print(f"    l = {ell:2d}:  C_l = {cl_est[ell]:.6e}")
    return {"cl": cl_est, "alm_true": alm_true}


# ---------------------------------------------------------------------------
def stage_7_beam() -> Dict[str, float]:
    banner("Stage 7:  Beam simulation (MD-style)")
    result = bs.beam_simulation(n_photons=30, n_steps=30, dt=0.05)
    bl = bs.beam_window_bl(50, result["sigma_beam"])
    print(f"  FWHM              = {result['fwhm']:.4e}")
    print(f"  sigma_beam        = {result['sigma_beam']:.4e}")
    print(f"  energy drift max  = {result['energy_drift_max']:.4e}")
    print(f"  B_0 = {bl[0]:.4f}, B_10 = {bl[10]:.4f}, B_30 = {bl[30]:.4f}")
    return result


# ---------------------------------------------------------------------------
def stage_8_foreground() -> Dict[str, object]:
    banner("Stage 8:  Foreground model (EEIO-style)")
    freqs = [30.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0]
    amplitudes = [100.0, 50.0, 30.0, 200.0, 10.0]
    result = fg.foreground_pipeline(freqs, amplitudes, recycling_price=0.1)
    print(f"  Frequencies     = {freqs}")
    print(f"  Components      = {fg.COMPONENTS}")
    print(f"  Baseline foregrounds  = {[f'{v:.2f}' for v in result['foreground_base']]}")
    print(f"  Recycled foregrounds  = {[f'{v:.2f}' for v in result['foreground_recycled']]}")
    print(f"  Total reduction = {result['scenario']['total_reduction']:.3f}")
    print(f"  GDP loss        = {result['scenario']['gdp_loss']:.3f}")
    return result


# ---------------------------------------------------------------------------
def stage_9_classification() -> Dict[str, List[Tuple[int, int]]]:
    banner("Stage 9:  Mode classification (E/B/systematic)")
    import random
    rng = random.Random(42)
    # Synthetic features for 90 modes (l=2..10, -l<=m<=l)
    X = []
    keys = []
    for l in range(2, 11):
        for m in range(-l, l + 1):
            # E-mode features (low parity)
            if (l + m) % 3 == 0:
                X.append([rng.gauss(1.0, 0.2), 0.1, 0.2, 1.0])
            # B-mode features (medium parity)
            elif (l + m) % 3 == 1:
                X.append([rng.gauss(1.0, 0.2), 0.5, 0.8, 1.0])
            # Systematic
            else:
                X.append([rng.gauss(5.0, 0.5), 0.0, 3.0, 10.0])
            keys.append((l, m))
    X_std, _, _ = mcmod.standardise(X)
    labels = mcmod.agglomerative_ward(X_std, n_clusters=3)
    classified = mcmod.classify_clusters(X_std, labels, keys)
    print(f"  Total modes      = {len(X)}")
    print(f"  E-mode count     = {len(classified['E'])}")
    print(f"  B-mode count     = {len(classified['B'])}")
    print(f"  Systematic count = {len(classified['systematic'])}")
    return classified


# ---------------------------------------------------------------------------
def stage_10_parameter_fit(cl_obs: List[float]) -> Dict[str, float]:
    banner("Stage 10:  Cosmological parameter fitting")
    l_max = len(cl_obs) - 1
    sigma_l = [abs(cl_obs[l]) * 0.1 + 1e-20 for l in range(l_max + 1)]
    init = cp.PLANCK_2018.copy()
    init["ns"] = 1.0
    init["H0"] = 70.0
    result = po.fit_cosmology(cl_obs, sigma_l, l_min=2, l_max=min(20, l_max),
                                 method="nelder-mead", initial_params=init)
    print(f"  Fitted parameters:")
    for k in ["H0", "ombh2", "omch2", "ns", "ln10As"]:
        print(f"    {k:8s} = {result[k]:.4f}")
    print(f"  chi^2 = {result['chi2']:.4e}")
    print(f"  nfev  = {result['nfev']}")
    return result


# ---------------------------------------------------------------------------
def stage_11_gradient_contour() -> Dict[str, object]:
    banner("Stage 11:  Gradient and contour analysis of CMB map")
    f = lambda th, ph: mc.spherical_harmonic_real(3, 1, th, ph)
    gt, gp = gc.spherical_gradient(f, math.pi / 4, 0.0)
    L = gc.spherical_laplacian(f, math.pi / 4, 0.0)
    print(f"  Y_31 gradient at (pi/4, 0):  ({gt:+.4e}, {gp:+.4e})")
    print(f"  Y_31 Laplacian at (pi/4, 0): {L:+.4e}  (theory = -3*4 = -12)")
    # Grid for contour
    grid, th, ph = gc.gaussian_field_on_grid(3, 1, n_theta=16, n_phi=32)
    contour_pts = gc.compute_contour_levels(grid, th, ph, level=0.0)
    genus = gc.genus_of_contour(grid, th, ph, level=0.0)
    print(f"  Contour points at level 0 = {len(contour_pts)}")
    print(f"  Genus of zero-contour     = {genus}")
    return {"gradient": (gt, gp), "laplacian": L, "contour_points": len(contour_pts)}


# ---------------------------------------------------------------------------
def run_all() -> None:
    print("\n" + "#" * 72)
    print("#" + " " * 70 + "#")
    print("#   CMB POWER SPECTRUM ESTIMATION PIPELINE                            #")
    print("#   High-Order Finite Differences & Stability Analysis                #")
    print("#   (Small-Scale Reproducible Experiment)                             #")
    print("#" + " " * 70 + "#")
    print("#" * 72)
    t0 = time.time()

    # Derived cosmological parameters
    banner("Preliminary:  Planck 2018 derived parameters")
    p = cp.PLANCK_2018.copy()
    h_val = cp.h_from_H0(p["H0"])
    om_b = cp.omega_b(p["ombh2"], p["H0"])
    om_c = cp.omega_c(p["omch2"], p["H0"])
    om_g = cp.omega_gamma(p["T_cmb"], p["H0"])
    om_L = cp.omega_lambda(p["ombh2"], p["omch2"], p["T_cmb"], p["NEFF"], p["m_nu"], p["H0"])
    r_s = cp.sound_horizon_rs(p["ombh2"], p["omch2"], om_g)
    th_s = cp.theta_star(p["ombh2"], p["omch2"], p["T_cmb"], p["NEFF"], p["m_nu"], p["H0"])
    print(f"  h          = {h_val:.6f}")
    print(f"  Omega_b    = {om_b:.6f}")
    print(f"  Omega_c    = {om_c:.6f}")
    print(f"  Omega_g    = {om_g:.6e}")
    print(f"  Omega_L    = {om_L:.6f}")
    print(f"  r_s        = {r_s:.3f} Mpc")
    print(f"  theta_*    = {th_s:.6e}")
    # Theoretical C_l reference
    print("  C_l reference:")
    for ell in [2, 10, 100, 500, 1000, 2000]:
        print(f"    l = {ell:4d}:  C_l = {cp.cmb_cl_theory(ell, p):.4e}")

    # Stage 1
    mesh = stage_1_mesh()

    # Stage 2
    L = stage_2_laplacian(mesh)

    # Stage 3
    stability = stage_3_stability(L)

    # Stage 4
    sparse = stage_4_sparse(L)

    # Stage 5
    quad = stage_5_quadrature()

    # Stage 6
    ps = stage_6_power_spectrum(mesh)
    cl_est = ps["cl"]

    # Stage 7
    beam = stage_7_beam()

    # Stage 8
    foreground = stage_8_foreground()

    # Stage 9
    classified = stage_9_classification()

    # Stage 10
    fit = stage_10_parameter_fit(cl_est)

    # Stage 11
    gc_results = stage_11_gradient_contour()

    # Also test Fornberg FD coefficients
    banner("Supplementary:  Fornberg FD weights")
    w = fdc.fornberg_weights(0.0, [-2.0, -1.0, 0.0, 1.0, 2.0], 4)
    print("  5-point 2nd-derivative weights:", w[2])
    print("  5-point 4th-derivative weights:", w[4])
    cw = fdc.central_fd_weights(2, 2)
    print("  Central 4th-order 2nd-deriv:", cw)

    # Also test FEM T4 basis
    banner("Supplementary:  T4 cubic-bubble FEM basis")
    tri_coords = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)]
    centroid = (0.5, math.sqrt(3.0) / 6.0)
    t4_coords = tri_coords + [centroid]
    for i in [1, 2, 3, 4]:
        phi, _, _ = sl.basis_t4_physical(t4_coords, i, centroid)
        print(f"  phi_{i}(centroid) = {phi:.6f}")

    # Also test Icosahedral mesh
    banner("Supplementary:  Icosahedral mesh")
    ico = sm.IcosahedralMesh(level=2)
    print(f"  level = 2: {ico.n_nodes} nodes, {ico.n_faces} faces")
    print(f"  node 0 has {len(ico.neighbours[0])} neighbours")

    # Summary
    elapsed = time.time() - t0
    banner("Summary")
    print(f"  Pipeline stages           : 11 + 3 supplementary")
    print(f"  Python modules            : 12")
    print(f"  Mesh nodes (nside=4)      : {mesh.n_nodes}")
    print(f"  FD operator size          : {mesh.n_nodes} x {mesh.n_nodes}")
    print(f"  Sparse nnz                : {sparse[0]}")
    print(f"  Quadrature exactness deg  : {max(n for n, _, e in quad if e < 1.0e-8)}")
    print(f"  C_l computed for l in     : [0, {len(cl_est) - 1}]")
    print(f"  E-mode count              : {len(classified['E'])}")
    print(f"  B-mode count              : {len(classified['B'])}")
    print(f"  Systematic count          : {len(classified['systematic'])}")
    print(f"  Best-fit ns               : {fit['ns']:.4f}")
    print(f"  Best-fit H0               : {fit['H0']:.2f}")
    print(f"  Best-fit chi^2            : {fit['chi2']:.4e}")
    print(f"  Beam FWHM (normalised)    : {beam['fwhm']:.4e}")
    print(f"  Wall-clock time           : {elapsed:.2f} s")

    banner("Done")
    print("  CMB power spectrum pipeline completed successfully.")
    print()


if __name__ == "__main__":
    run_all()
