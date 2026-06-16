"""
PROJECT_286 — main.py
=====================
Tokamak Grad-Shafranov equilibrium with high-order finite differences,
MHD stability analysis, and multi-physics diagnostics.

This is the unified entry point; no command-line arguments are required.
Running `python main.py` performs:

    1. Build the (R,Z) grid and seed the poloidal flux ψ.
    2. Assemble the Grad-Shafranov operator Δ* in CRS (sparse) form,
       including a 4th-order compact-finite-difference variant.
    3. Solve the nonlinear GS equation Δ* ψ = S(ψ) by Picard iteration
       with a pseudo-time inner solve (implicit FD style).
    4. Detect the magnetic axis and X-point (lightning-pose keypoint
       detection analog).
    5. Reconstruct j_φ and plasma current via a PIC-like deposition.
    6. Compute flux-surface integrals and QMC volume averages.
    7. Run stability analysis (Mercier, tearing Δ', MHD eigenvalues).
    8. Sample turbulent χ⊥(ψ) from a truncated log-normal PDF.
    9. Trace 3-D field lines and write them in xyz/xyzl format.
   10. Export the FEM-like mesh and emit a multi-panel diagnostic summary.

All outputs are plain text — no visualisation.
"""

from __future__ import annotations
import math
import os
import sys
import time


# ---------------------------------------------------------------------------
# Make the project importable when run as `python main.py`
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# Now import project modules (package-style relative imports when used as a
# package; absolute imports when run as a script).
from tokamak_geometry import TokamakGeometry, find_keypoints
from plasma_profiles import (GSProfiles, SpectroscopicDiagnostic,
                            cosine_integral, sine_integral, incomplete_gamma_upper)
from sparse_operators import (gs_operator_crs, gs_operator_compact_crs,
                              write_crs, jacobi_preconditioner)
from pic_reconstruction import (compute_jtor, sample_particles, deposit_particles,
                                bootstrap_fraction, PICGrid)
from field_line_tracer import (trace_field_line, safety_factor, write_xyzl,
                               interpolate_b)
from flux_surface_integrals import (volume_integral, plasma_volume, plasma_current,
                                   monomial_torus_integral, flux_surface_integral)
from stability_analysis import (mercier_criterion, tearing_delta_prime,
                                build_mhd_matrix, symmetric_eigenvalues,
                                mhd_growth_rates, troyon_beta_limit,
                                sauter_bootstrap)
from transport_pdf import (sample_chi_profile, tl_mean, tl_variance, tl_pdf,
                           log_normal_mean, log_normal_variance)
from quasi_mc import (niederreiter2_generate, qmc_integrate_2d,
                      qmc_volume_average)
from boundary_reconstruction import reconstruct_geometry, refine_keypoint
from diagnostics import (compute_diagnostics, multi_panel_summary,
                         write_fem_mesh, magic_index_matrix)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MU0 = 4.0 * math.pi * 1e-7
KB = 1.380649e-23
E_CHARGE = 1.602176634e-19


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def build_p_field(psi: list[list[float]], profiles: GSProfiles,
                  psi_axis: float) -> list[list[float]]:
    """Evaluate p(ψ_n) on the grid given normalised ψ."""
    Nr = len(psi); Nz = len(psi[0])
    P = [[0.0] * Nz for _ in range(Nr)]
    dpsi = psi_axis - 0.0
    if abs(dpsi) < 1e-12:
        dpsi = 1e-12
    for i in range(Nr):
        for j in range(Nz):
            psin = (psi[i][j] - 0.0) / dpsi
            psin = max(0.0, min(1.0, psin))
            P[i][j] = profiles.pressure(psin) if psin > 0.0 else 0.0
    return P


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    section("PROJECT_286: Tokamak Grad-Shafranov Equilibrium")
    print("  High-order finite differences + MHD stability analysis")
    print("  (small-scale reproducible experiment)")

    # ------------------------------------------------------------------
    # 1. Geometry and profiles
    # ------------------------------------------------------------------
    section("1. Geometry and plasma profiles")
    geom = TokamakGeometry(
        R0=1.0, a=0.33, kappa=1.7, delta=0.33,
        R_min=0.5, R_max=1.6, Z_min=-0.6, Z_max=0.6,
        Nr=41, Nz=41,
    )
    print(f"   R_0 = {geom.R0} m,  a = {geom.a} m,  ε = {geom.eps:.3f}")
    print(f"   κ = {geom.kappa},  δ = {geom.delta}")
    print(f"   grid: Nr={geom.Nr}, Nz={geom.Nz},  dR={geom.dR:.4f}, dZ={geom.dZ:.4f}")

    profiles = GSProfiles(c1=4.0e4, alpha=1.5, d1=1.5, beta=1.0, d2=0.0)
    print(f"   pressure amp  c_1 = {profiles.c1} Pa")
    print(f"   FF'  amp      d_1 = {profiles.d1} T²")

    # Sanity check on special functions
    print("   special-function check:")
    print(f"      Ci(1.0) = {cosine_integral(1.0):.6f}   (ref ≈ 0.337404)")
    print(f"      Si(1.0) = {sine_integral(1.0):.6f}   (ref ≈ 0.946083)")
    print(f"      Γ(5/2, 1) = {incomplete_gamma_upper(2.5, 1.0):.6f}")

    # LIF-like diagnostic sanity check
    spec = SpectroscopicDiagnostic()
    spectrum = spec.synthetic_spectrum(T_e=50.0, n_e=2e20)
    T_inv, n_inv = spec.invert(spectrum)
    print(f"   LIF diagnostic inversion: T_e = {T_inv:.1f} eV,  n_e = {n_inv:.2e} m^-3")

    # ------------------------------------------------------------------
    # 2. Assemble the GS operator (CRS)
    # ------------------------------------------------------------------
    section("2. Grad-Shafranov operator assembly")
    A2 = gs_operator_crs(geom.Nr, geom.Nz, geom.dR, geom.dZ, geom.R_min)
    A4 = gs_operator_compact_crs(geom.Nr, geom.Nz, geom.dR, geom.dZ, geom.R_min)
    print(f"   2nd-order CRS:  n={A2.n}, nnz={A2.nnz}")
    print(f"   4th-order compact CRS: n={A4.n}, nnz={A4.nnz}")

    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    paths = write_crs(os.path.join(out_dir, "gs_op"), A2)
    print(f"   wrote: {paths['row_path']} etc.")

    Minv = jacobi_preconditioner(A2)
    print(f"   Jacobi precond max |1/A_ii| = {max(abs(v) for v in Minv):.3e}")

    # ------------------------------------------------------------------
    # 3. Nonlinear solve (Picard + pseudo-time inner)
    # ------------------------------------------------------------------
    section("3. Nonlinear Grad-Shafranov solve")
    from grad_shafranov import solve_grad_shafranov
    res = solve_grad_shafranov(geom, profiles, method="picard",
                               compact=False, tol=1e-6, maxiter=80)
    psi = res["psi"]
    print(f"   solver   : {res['state'].method}")
    print(f"   iters    : {res['state'].iterations}")
    print(f"   residual : {res['state'].residual:.3e}")
    print(f"   converged: {res['state'].converged}")
    psi_max = max(max(row) for row in psi)
    psi_min = min(min(row) for row in psi)
    print(f"   ψ range  : [{psi_min:.4f}, {psi_max:.4f}]")

    # ------------------------------------------------------------------
    # 4. Keypoint detection (magnetic axis / X-point)
    # ------------------------------------------------------------------
    section("4. Magnetic-axis / X-point keypoint detection")
    keys = find_keypoints(psi, geom)
    if keys["axis"] is not None:
        R_ax, Z_ax, psi_ax = refine_keypoint(psi, geom.dR, geom.dZ,
                                             geom.R_grid(), geom.Z_grid(),
                                             int((keys["axis"][0] - geom.R_min) / geom.dR),
                                             int((keys["axis"][1] - geom.Z_min) / geom.dZ))
        print(f"   axis    : R = {R_ax:.4f} m,  Z = {Z_ax:.4f} m")
    else:
        R_ax, Z_ax, psi_ax = geom.R0, 0.0, psi_max
        print(f"   axis (fallback) : R = {R_ax:.4f} m,  Z = {Z_ax:.4f} m")
    if keys["xpoints"]:
        x = keys["xpoints"][0]
        print(f"   X-point : R = {x[0]:.4f} m,  Z = {x[1]:.4f} m  (ψ = {x[2]:.4f})")
    else:
        print("   X-point : none (limiter configuration)")

    # ------------------------------------------------------------------
    # 5. j_φ reconstruction + plasma current (PIC analog)
    # ------------------------------------------------------------------
    section("5. Toroidal current j_φ and plasma current")
    Rg = geom.R_grid(); Zg = geom.Z_grid()
    jt = compute_jtor(psi, Rg, geom.dR, geom.dZ)
    jt_max = max(abs(jt[i][j]) for i in range(geom.Nr) for j in range(geom.Nz))
    print(f"   max |j_φ| = {jt_max:.3e} A/m²")

    # PIC deposition
    particles = sample_particles(jt, Rg, Zg, n_particles=1500)
    print(f"   PIC particles sampled: {len(particles)}")
    pic_grid = PICGrid(R_min=geom.R_min, R_max=geom.R_max,
                       Z_min=geom.Z_min, Z_max=geom.Z_max,
                       Nr=geom.Nr, Nz=geom.Nz)
    rho = deposit_particles(particles, pic_grid)
    rho_max = max(abs(rho[i][j]) for i in range(geom.Nr) for j in range(geom.Nz))
    print(f"   deposited ρ max = {rho_max:.3e}")

    Ip = plasma_current(jt, Rg, Zg)
    print(f"   I_p (from GS) = {Ip:.3e} A  = {Ip/1e6:.3f} MA")
    f_bs = bootstrap_fraction(jt, Rg, Zg, f_bs=0.55)
    print(f"   f_bs (model)  = {f_bs:.3f}")

    # ------------------------------------------------------------------
    # 6. Flux-surface integrals & volume averages
    # ------------------------------------------------------------------
    section("6. Flux-surface integrals and QMC volume averages")
    V_plasma = plasma_volume(psi, Rg, Zg)
    print(f"   plasma volume V = {V_plasma:.4f} m³")

    # Build pressure field
    p_field = build_p_field(psi, profiles, psi_ax)
    p_max = max(max(row) for row in p_field)
    print(f"   max p on axis ≈ {p_max:.3e} Pa")

    # Volume integral of p
    Vp_int = volume_integral(psi, p_field, Rg, Zg)
    print(f"   ∫∫ p 2π R dR dZ = {Vp_int:.3e}")

    # Monomial benchmark (cube01 analog)
    I_mono = monomial_torus_integral(2, 1, 3)
    print(f"   monomial_torus_integral(2,1,3) = {I_mono:.6f}  (ref = 1/72)")

    # QMC 2-D integral of a smooth test function over the (R,Z) box
    def test_fn(x, y):
        return math.exp(-((x - geom.R0) ** 2 + y ** 2) / (0.2 ** 2))
    qmc_val = qmc_integrate_2d(test_fn,
                               (geom.R_min, geom.Z_min),
                               (geom.R_max, geom.Z_max),
                               n_points=512)
    print(f"   QMC 2-D integral of test fn = {qmc_val:.6f}")

    Q_ones = [[1.0] * geom.Nz for _ in range(geom.Nr)]
    qmc_vol_avg = qmc_volume_average(psi, Q_ones, Rg, Zg)
    print(f"   QMC <1>_plasma = {qmc_vol_avg:.4f}  (≈ 1)")

    # ------------------------------------------------------------------
    # 7. Stability analysis
    # ------------------------------------------------------------------
    section("7. MHD stability analysis")
    B0 = 2.5     # T
    F0 = geom.R0 * B0

    merc = mercier_criterion(psi, jt, Rg, geom.Nz // 2, geom.R0, B0)
    print(f"   Mercier D_M (axis) = {merc['D_M_axis']:.4e}")
    print(f"   Mercier stable     : {merc['stable']}")

    # q profile along Z=0 for tearing
    q_profile = [0.0] * geom.Nr
    for i in range(2, geom.Nr - 2):
        B_theta = abs(psi[i][geom.Nz // 2 + 1] - psi[i][geom.Nz // 2 - 1]) \
                  / (2.0 * geom.dR) / max(Rg[i], 1e-6)
        B_phi = B0 * geom.R0 / max(Rg[i], 1e-6)
        q_profile[i] = (Rg[i] * B_theta / max(B_phi, 1e-12)) if B_phi > 1e-12 else 1.0

    dp = tearing_delta_prime(psi, q_profile, Rg, m=2, n_tor=1)
    print(f"   tearing Δ' (2,1) = {dp:.4f} m^-1")

    # MHD eigenvalues (small subgrid)
    nsub = 12
    i0 = geom.Nr // 2 - nsub // 2
    j0 = geom.Nz // 2 - nsub // 2
    psi_sub = [[psi[i0 + ii][j0 + jj] for jj in range(nsub)] for ii in range(nsub)]
    jt_sub = [[jt[i0 + ii][j0 + jj] for jj in range(nsub)] for ii in range(nsub)]
    M = build_mhd_matrix(psi_sub, jt_sub, geom.dR, geom.dZ,
                         geom.R_min + i0 * geom.dR, nsub, nsub)
    evals = symmetric_eigenvalues(M, n_iter=25)
    rates = mhd_growth_rates(evals)
    max_rate = rates[0] if rates else 0.0
    print(f"   # eigenvalues computed = {len(evals)}")
    print(f"   max growth rate γ_max = {max_rate:.3e} s^-1")

    beta_lim = troyon_beta_limit(Ip_MA=Ip / 1e6, a_m=geom.a, B0_T=B0)
    print(f"   Troyon β_N,lim = {beta_lim:.3f}")

    f_bs_sauter = sauter_bootstrap(eps=geom.eps, nu_star=0.3, q=1.5)
    print(f"   Sauter f_bs (ε={geom.eps:.2f}, ν*=0.3, q=1.5) = {f_bs_sauter:.3f}")

    # ------------------------------------------------------------------
    # 8. Turbulent transport PDF (truncated log-normal χ⊥)
    # ------------------------------------------------------------------
    section("8. Turbulent transport coefficient PDF")
    chi = sample_chi_profile(n_psi=24, seed=42,
                             chi_neo=0.05, chi_bohm=5.0,
                             mu=0.0, sigma=0.7)
    print(f"   χ⊥(ψ_n) samples (24 values):")
    print(f"      min = {min(chi):.3f},  max = {max(chi):.3f},  mean = {sum(chi)/len(chi):.3f}")
    mu_chi = tl_mean(0.05, 5.0, 0.0, 0.7)
    var_chi = tl_variance(0.05, 5.0, 0.0, 0.7)
    print(f"   truncated log-normal theory:  <χ> = {mu_chi:.3f},  Var(χ) = {var_chi:.3f}")

    # ------------------------------------------------------------------
    # 9. Field-line tracing (Verlet) + 3-D xyz/xyzl output
    # ------------------------------------------------------------------
    section("9. Magnetic field line tracing")
    lines = []
    # Three start points: on-axis vicinity, mid-radius, edge
    for r_frac in (0.05, 0.3, 0.6):
        R0 = R_ax + r_frac * geom.a
        Z0 = Z_ax
        line = trace_field_line(psi, F0, Rg, Zg, R0, Z0,
                                n_steps=1500, ds=0.008)
        lines.append(line)
        print(f"   r/a = {r_frac}:  length = {line.length:.3f} m,  "
              f"#points = {len(line.R)}")

    q0 = safety_factor(psi, F0, Rg, Zg, R_ax, Z_ax, psi_ax)
    print(f"   q_0 (axis) ≈ {q0:.3f}")

    xyz_paths = write_xyzl(os.path.join(out_dir, "fieldlines"), lines)
    print(f"   wrote {xyz_paths['xyz_path']}")
    print(f"   wrote {xyz_paths['xyzl_path']}")

    # ------------------------------------------------------------------
    # 10. FEM mesh export + multi-panel summary
    # ------------------------------------------------------------------
    section("10. FEM mesh export and multi-panel diagnostics")
    fem_paths = write_fem_mesh(os.path.join(out_dir, "equilibrium"),
                               Rg, Zg, psi, p_field, jt)
    print(f"   wrote {fem_paths['nodes']}")
    print(f"   wrote {fem_paths['elements']}")
    print(f"   wrote {fem_paths['values']}")

    magic = magic_index_matrix(min(9, geom.Nr), min(9, geom.Nz))
    print(f"   magic index (9×9 sample) — first row: {magic[0][:5]}")

    # Build full geometry object
    eq_geom = reconstruct_geometry(psi, jt, Rg, Zg, geom.R0, B0, Ip)
    print(f"   refined axis : R = {eq_geom.R_axis:.4f}, Z = {eq_geom.Z_axis:.4f}")
    print(f"   a_geo = {eq_geom.a_geo:.4f} m,  κ = {eq_geom.kappa_geo:.3f},  "
          f"δ = {eq_geom.delta_geo:.3f}")

    diag = compute_diagnostics(psi, jt, p_field, Rg, Zg, B0, Ip)
    stability_summary = {
        "D_M_axis": merc["D_M_axis"],
        "stable": merc["stable"],
        "delta_prime": dp,
        "max_growth": max_rate,
    }
    summary = multi_panel_summary(diag, eq_geom, stability_summary)
    print(summary)

    # Save the summary
    with open(os.path.join(out_dir, "diagnostic_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n   summary written to {out_dir}/diagnostic_summary.txt")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    dt = time.time() - t0
    section("PROJECT_286 completed")
    print(f"   elapsed wall time: {dt:.2f} s")
    print(f"   outputs directory: {out_dir}")


if __name__ == "__main__":
    main()
