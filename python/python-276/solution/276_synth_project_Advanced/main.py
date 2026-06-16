"""
main.py — Unified entry point for the crystal-defect formation energy project
=============================================================================

Run this file with no arguments to perform the full computation:

    python main.py

The pipeline proceeds in 9 stages:

  Stage 1. Validate the numerical kernels against analytical benchmarks
           (high-order FD stencils, minimal-surface PDEs, Gaussian decay,
           Eshelby circular inclusion, LU solver residual).
  Stage 2. Build the host 2-D hexagonal crystal and its migration graph.
  Stage 3. Construct the high-order finite-difference Laplacian in the
           row-indexed sparse format and verify its dispersion relation.
  Stage 4. Run the von Neumann stability analysis and report Δτ_max.
  Stage 5. Build the defect supercell and compute the bulk / defect total
           energies using the orbital-free DFT functional.
  Stage 6. Solve the defect Green's function via the Dyson equation on a
           cluster around the defect and compute the band-energy shift.
  Stage 7. Add the Eshelby elastic correction and the Makov–Payne
           correction for charged defects.
  Stage 8. Build a refined triangular mesh around the defect and project
           the strain energy density onto the mesh nodes.
  Stage 9. Run the statistical ensemble for vacancy & interstitial and
           report the 95 % confidence interval and Welch's t-test.

Each stage prints a short summary to stdout. No output files are written;
no visualisation is performed (in compliance with the synthesis rules).

Seed-project mapping summary (15 → 1):
  464_gen_hermite_exactness → Fermi–Dirac quadrature (defect_formation_energy)
  426_fft_serial            → FFT (stability_analysis, fft_poisson)
  1221_WillemWybo_SGF       → sparse Green's function (sparse_green_defect)
  414_fem2d_scalar_display  → FEM scalar field on mesh (mesh_defect)
  576_image_denoise         → 3×3 median filter (denoise_filter)
  342_euclid                → GCD / Miller reduction (crystal_lattice)
  286_digraph_arc           → migration graph (crystal_lattice)
  687_linpack_bench         → LU solver (linear_solver)
  335_elliptic_integral     → Eshelby inclusion (eshelby_strain, fft_poisson)
  1100_clinical_trial       → Welch's t-test (statistical_qc)
  548_human_mesh2d          → 2-D mesh generation (mesh_defect)
  768_minimal_surface_exact → analytical benchmarks (analytical_benchmarks)
  918_prob                  → distribution samplers (statistical_qc)
  992_r8ri                  → RI sparse storage (high_order_fd)
  1331_triangulation_boundary → mesh boundary (crystal_lattice, mesh_defect)
"""

from __future__ import annotations
import math
import sys
import time
import numpy as np

# Ensure the package is importable when run as a script
sys.path.insert(0, ".")

from config import default_config, HARTREE_TO_EV


def _header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _subheader(title: str) -> None:
    print()
    print(f"--- {title} ---")


def stage1_benchmarks() -> None:
    """Stage 1 — analytical benchmarks."""
    _header("Stage 1 — Analytical benchmarks")
    from analytical_benchmarks import run_all_benchmarks
    results = run_all_benchmarks()

    _subheader("FD-stencil exactness")
    for order in (2, 4, 6, 8):
        key = f"stencil_exactness_order{order}"
        print(f"  order {order}: {'PASS' if results[key]['pass'] else 'FAIL'}")

    _subheader("Laplacian residual on cos(x) cos(y) (periodic, exact −2 cos(x) cos(y))")
    for order in (2, 4, 6, 8):
        key = f"catenoid_residual_order{order}"
        print(f"  order {order}: max |∇²_fd − ∇²_exact| = {results[key]['max_residual']:.3e}")

    _subheader("Minimal-surface PDE residuals")
    for name, res in results["minimal_surfaces"].items():
        print(f"  {name:10s}: max |R| = {res:.3e}")

    _subheader("Gaussian wave-packet validation")
    gv = results["gaussian_validation"]
    print(f"  numerical E = {gv['numerical_energy']:.6f}")
    print(f"  exact E     = {gv['exact_energy']:.6f}")
    print(f"  rel. error  = {gv['rel_error']:.3e}")

    _subheader("Eshelby circular-inclusion validation")
    ev = results["eshelby_circular"]
    print(f"  numerical E = {ev['numerical']:.6f}")
    print(f"  exact E     = {ev['exact']:.6f}")
    print(f"  rel. error  = {ev['rel_error']:.3e}")

    _subheader("Elliptic integrals K and E")
    ee = results["elliptic_K_E"]
    print(f"  K(0)   = {ee['K(0)']:.6f}   (exact π/2 = {ee['K(0)_exact']:.6f})")
    print(f"  E(0)   = {ee['E(0)']:.6f}   (exact π/2 = {ee['E(0)_exact']:.6f})")
    print(f"  K(1/2) = {ee['K(1/2)']:.6f} (ref 1.8541)")
    print(f"  E(1/2) = {ee['E(1/2)']:.6f} (ref 1.3506)")

    _subheader("LINPACK LU solver residual")
    lu = results["lu_solver"]
    print(f"  system size          = {lu['n']}")
    print(f"  ||A x − b||_∞       = {lu['residual_inf_norm']:.3e}")
    print(f"  ||x_computed − x||_∞ = {lu['solution_error']:.3e}")


def stage2_lattice() -> dict:
    """Stage 2 — build crystal lattice and migration graph."""
    _header("Stage 2 — Crystal lattice and migration graph")
    from crystal_lattice import build_default_lattice, gcd_euclid_fast
    cfg = default_config()
    lat, graph = build_default_lattice(cfg.crystal.lattice_constant,
                                       cfg.crystal.n_cells)
    print(f"  lattice constant a = {cfg.crystal.lattice_constant:.3f} Bohr")
    print(f"  N×N cells          = {cfg.crystal.n_cells} × {cfg.crystal.n_cells}")
    print(f"  total sites        = {lat.n_sites}")
    print(f"  supercell area     = {lat.supercell_area():.3f} Bohr²")
    print(f"  migration edges    = {graph.edge_num // 2}")
    print(f"  Eulerian property  = {graph.is_eulerian()}")
    # a few Miller-index reductions
    samples = [(6, 3), (12, 8), (15, 10), (0, 7)]
    for u, v in samples:
        up, vp = u // gcd_euclid_fast(u, v) if gcd_euclid_fast(u, v) else 0, v // gcd_euclid_fast(u, v) if gcd_euclid_fast(u, v) else 0
        g = gcd_euclid_fast(u, v)
        print(f"  [{u} {v}] / gcd {g} = [{up} {vp}]")
    return {"lattice": lat, "graph": graph}


def stage3_fd_operator() -> None:
    """Stage 3 — high-order FD Laplacian in RI sparse format."""
    _header("Stage 3 — High-order FD Laplacian (RI sparse)")
    from high_order_fd import RISparseLaplacian, dispersion_error
    cfg = default_config()
    N = cfg.grid.n_grid
    h = cfg.crystal.supercell_length / N

    for order in (2, 4, 6, 8):
        t0 = time.perf_counter()
        A = RISparseLaplacian(N, N, h, order)
        t1 = time.perf_counter()
        print(f"  order {order}: N = {N}² = {N * N}, "
              f"nz_off = {A.nz_off}, build time = {(t1 - t0) * 1e3:.1f} ms")
        # quick sanity check: apply to a constant field ⇒ 0
        ones = np.ones(N * N)
        y = A.mv(ones)
        print(f"    ||L·1||_∞ = {np.max(np.abs(y)):.3e}")

    _subheader("Dispersion error at k h = π/4")
    kh = math.pi / 4
    for order in (2, 4, 6, 8):
        err = dispersion_error(kh / h, h, order)
        print(f"  order {order}: k_fd² − k² = {err:.3e}")


def stage4_stability() -> None:
    """Stage 4 — von Neumann stability analysis."""
    _header("Stage 4 — Von Neumann stability analysis")
    from stability_analysis import stability_table, max_stable_dt
    cfg = default_config()
    N = cfg.grid.n_grid
    h = cfg.crystal.supercell_length / N
    print(f"  grid N = {N}, h = {h:.4f} Bohr")
    table = stability_table(N, h)
    for order, dt in table:
        print(f"  order {order}: Δτ_max = {dt:.4e} Hartree⁻¹")


def stage5_formation_energy() -> dict:
    """Stage 5 — formation energy of vacancy and interstitial."""
    _header("Stage 5 — Formation energy (neutral charge state)")
    from defect_formation_energy import run_single_defect
    from denoise_filter import denoising_report
    cfg = default_config()

    _subheader("Vacancy (q = 0)")
    cfg_vac = default_config()
    res_vac = run_single_defect(cfg_vac)
    for k, v in res_vac.items():
        print(f"  {k:25s} = {v}")

    _subheader("Interstitial (q = 0)")
    cfg_int = default_config()
    object.__setattr__(cfg_int.defect, "kind", "interstitial")
    res_int = run_single_defect(cfg_int)
    for k, v in res_int.items():
        print(f"  {k:25s} = {v}")

    _subheader("Denoising report (vacancy charge density)")
    # build a small noisy field to demonstrate the 3x3 median filter
    N = cfg.grid.n_grid
    L = cfg.crystal.supercell_length
    h = L / N
    rng = np.random.default_rng(276)
    n0 = 0.5
    field = np.full((N, N), n0) + 0.05 * rng.standard_normal((N, N))
    r_bef, r_aft, q_err = denoising_report(field, h, target_charge=n0 * L * L)
    print(f"  roughness before  = {r_bef:.4e}")
    print(f"  roughness after   = {r_aft:.4e}")
    print(f"  charge error      = {q_err:.4e}")

    return {"vacancy": res_vac, "interstitial": res_int}


def stage6_green_function() -> None:
    """Stage 6 — sparse Green's function and Dyson equation."""
    _header("Stage 6 — Sparse Green's function (Dyson equation)")
    from sparse_green_defect import (HostHamiltonian, select_cluster,
                                     build_defect_potential, DysonSolver,
                                     partial_fraction_fit)
    from crystal_lattice import CrystalLattice
    cfg = default_config()
    lat = CrystalLattice(cfg.crystal.lattice_constant,
                         cfg.crystal.n_cells, "hexagonal")
    N = cfg.grid.n_grid
    L = cfg.crystal.supercell_length
    h = L / N
    H = HostHamiltonian(N, N, h, cfg.grid.fd_order,
                        lat.positions, cfg.crystal.z_eff, cfg.crystal.r_core)
    # select cluster around the centre
    defect_idx = lat.n_sites // 2
    cluster_idx = select_cluster(lat.positions, defect_idx, R_cut=2.0 * cfg.crystal.lattice_constant)
    print(f"  cluster size = {cluster_idx.size} sites")
    dV = build_defect_potential(N, N, h, (N // 2, N // 2),
                                cfg.crystal.z_eff, cfg.crystal.r_core,
                                kind="vacancy")
    solver = DysonSolver(H, cluster_idx, dV)
    # band-energy shift
    dE = solver.band_energy_shift(E_min=-2.0, E_max=2.0, n_quad=16,
                                  eta=0.1, kT=cfg.crystal.kt_hartree)
    print(f"  ΔE_band (cluster) = {dE:.6f} Hartree")
    print(f"  ΔE_band (eV)      = {dE * HARTREE_TO_EV:.4f}")
    # partial-fraction test
    zs = -1.0 + 1j * np.logspace(-1, 1, 8)
    G_diag = np.array([solver.green_cluster(z)[0, 0] for z in zs])
    alpha, poles = partial_fraction_fit(G_diag, zs, n_poles=3)
    print(f"  partial-fraction residues: {[f'{a:.3f}' for a in alpha]}")


def stage7_corrections() -> dict:
    """Stage 7 — Makov–Payne and Eshelby corrections."""
    _header("Stage 7 — Finite-size corrections")
    from fft_poisson import charged_defect_hartree
    from eshelby_strain import eshelby_strain_energy, validate_eshelby_circular
    cfg = default_config()
    N = cfg.grid.n_grid
    L = cfg.crystal.supercell_length
    h = L / N

    _subheader("Charged-defect Hartree (q = +1, Gaussian σ = 0.5 Bohr)")
    EH, EMP, Eew = charged_defect_hartree(N, h, q=+1, sigma=0.5)
    print(f"  E_Hartree      = {EH:.6f} Ha  ({EH * HARTREE_TO_EV:.4f} eV)")
    print(f"  E_Makov-Payne  = {EMP:.6f} Ha  ({EMP * HARTREE_TO_EV:.4f} eV)")
    print(f"  E_Ewald        = {Eew:.6f} Ha  ({Eew * HARTREE_TO_EV:.4f} eV)")

    _subheader("Eshelby strain energy (circular & elliptical)")
    E_circ, E_exact = validate_eshelby_circular()
    print(f"  circular  E = {E_circ:.6f}  (exact {E_exact:.6f})")
    for aspect in (1.0, 0.7, 0.5):
        E = eshelby_strain_energy(1.0, 1.0, 0.3, aspect=aspect, misfit_strain=0.1)
        print(f"  aspect {aspect:.2f}: E = {E:.6f}")

    return {"E_Hartree": EH, "E_MP": EMP, "E_Ewald": Eew}


def stage8_mesh() -> dict:
    """Stage 8 — refined mesh and strain-energy projection."""
    _header("Stage 8 — 2-D defect-centred mesh")
    from mesh_defect import build_defect_mesh, strain_energy_on_mesh
    cfg = default_config()
    L = cfg.crystal.supercell_length
    defect_pt = np.array([L / 2, L / 2])
    info = build_defect_mesh(L, L, defect_pt, R_ref=1.5, max_levels=3)
    print(f"  n_nodes          = {info['n_nodes']}")
    print(f"  n_triangles      = {info['n_triangles']}")
    print(f"  n_boundary_edges = {info['n_boundary_edges']}")
    print(f"  boundary nodes   = {info['n_boundary_nodes']}")
    # strain energy density on mesh
    w = strain_energy_on_mesh(info["nodes"], info["triangles"], defect_pt,
                              G=cfg.eshelby.shear_modulus * HARTREE_TO_EV,
                              nu=cfg.eshelby.poisson_ratio, misfit=0.05)
    print(f"  max strain energy density = {w.max():.3e}")
    print(f"  mean strain energy density = {w.mean():.3e}")
    return info


def stage9_statistics() -> dict:
    """Stage 9 — statistical ensemble and Welch's t-test."""
    _header("Stage 9 — Statistical ensemble & Welch's t-test")
    from statistical_qc import statistical_summary
    from config import HARTREE_TO_EV
    # take the formation energies from stage 5 as base values
    cfg = default_config()
    from defect_formation_energy import run_single_defect
    res_v = run_single_defect(cfg)
    cfg2 = default_config()
    object.__setattr__(cfg2.defect, "kind", "interstitial")
    res_i = run_single_defect(cfg2)
    base_vac = res_v["E_formation_eV"]
    base_int = res_i["E_formation_eV"]
    kT_eV = cfg.crystal.kT_ev
    summary = statistical_summary(base_vac, base_int,
                                  n_samples=cfg.stats.n_samples,
                                  kT_eV=kT_eV,
                                  seed=cfg.stats.seed)
    for k, v in summary.items():
        print(f"  {k:20s} = {v}")
    return summary


def main() -> None:
    """Run the full 9-stage pipeline."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Crystal-Defect Formation Energy (PhD-level synthesis project) ║")
    print("║  2-D hexagonal crystal · high-order FD · sparse Green function ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    t_start = time.perf_counter()

    stage1_benchmarks()
    stage2_lattice()
    stage3_fd_operator()
    stage4_stability()
    stage5_formation_energy()
    stage6_green_function()
    stage7_corrections()
    stage8_mesh()
    stage9_statistics()

    t_end = time.perf_counter()
    _header("Pipeline complete")
    print(f"  total wall-clock time = {t_end - t_start:.2f} s")
    print("  ✓ All 9 stages executed successfully, zero parameters required.")
    print()


if __name__ == "__main__":
    main()
