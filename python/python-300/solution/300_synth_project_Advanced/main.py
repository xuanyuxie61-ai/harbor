"""
main.py - Tritium Breeding Blanket Neutron Transport Solver
===========================================================

Unified entry point for the high-order finite-difference discrete-ordinates
(S_N) solver of the multigroup neutron transport equation in a fusion
breeding blanket.  Zero-argument execution.

Scientific problem
------------------
In a D-T fusion reactor the 14.1 MeV neutrons born in the plasma must be
captured in a lithium-containing breeding blanket to produce tritium fuel
via the reactions  6Li(n,t)4He  and  7Li(n,n't)4He.  The design goal is
TBR >= 1.05 for tritium self-sufficiency.  This code solves the steady-
state multigroup SN transport equation in 1-D slab geometry using a
fourth-order compact finite-difference scheme, performs von Neumann and
spectral stability analysis, and verifies the solution against a
Feynman-Kac stochastic estimator.

Modules
-------
physics_constants    nuclear data, group structure, material specs
cross_sections       ENDF-like multigroup XS with PWL 2-D interpolation
energy_spectra       Dirichlet-multigroup D-T source spectrum
blanket_geometry     1-D Voronoi pebble-bed + vertex-to-element map
angular_quadrature   Legendre S_N quadrature + triangle integrals
fd_transport         compact FD SN solver + Thomas algorithm
stability_analysis   von Neumann + spectral radius + Chladni eigenmodes
monte_carlo_fk       Feynman-Kac verification + Monty-Hall scattering
tbr_calculator       tritium breeding ratio
latent_decomposer    VAE-inspired XS library compression
"""

from __future__ import annotations
import math
import time
import os
import sys

# Make the project directory importable so we can run `python main.py` directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physics_constants as pc
import cross_sections as xsm
import energy_spectra as espec
import blanket_geometry as bgeom
import angular_quadrature as aq
import fd_transport as fdt
import stability_analysis as stab
import monte_carlo_fk as mcfk
import tbr_calculator as tbr
import latent_decomposer as latd


# ---------------------------------------------------------------------------
# Utility: formatted section printing (no visualization)
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def kv(key: str, value, width: int = 32) -> None:
    if isinstance(value, float):
        print(f"  {key:<{width}} : {value:+.6e}")
    else:
        print(f"  {key:<{width}} : {value}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print()
    print("#" * 72)
    print("#  Tritium Breeding Blanket Neutron Transport Solver")
    print("#  High-order compact finite-difference SN method")
    print("#  with von Neumann / spectral stability analysis")
    print("#  and Feynman-Kac stochastic verification")
    print("#" * 72)

    # ===================================================================
    # 1. Physical setup: blanket geometry
    # ===================================================================
    section("1. Blanket geometry (1-D Voronoi pebble bed)")
    geo = bgeom.BlanketRegion(
        L_total_cm=pc.BLANKET_THICKNESS_CM,
        fw_cm=pc.FIRST_WALL_THICKNESS_CM,
        bw_cm=pc.BACK_WALL_THICKNESS_CM,
        n_cells=pc.DEFAULT_NX,
        pebble_diameter_cm=0.06,
        packing_fraction=0.62,
        seed=12345,
    )
    summary = geo.summary()
    for k, v in summary.items():
        kv(k, v)

    # ===================================================================
    # 2. Cross-section library with PWL 2-D interpolation
    # ===================================================================
    section("2. Multigroup cross-section library (PWL 2-D interp)")
    T_grid = [300.0, 600.0, 900.0, 1200.0, 1500.0]
    T_operating = 800.0     # K  (demo blanket temperature)
    lib_li6 = xsm.MultigroupCrossSection('li6', T_grid)
    lib_li7 = xsm.MultigroupCrossSection('li7', T_grid)
    lib_li2o = xsm.MultigroupCrossSection('li2o', T_grid)
    lib_eurofer = xsm.MultigroupCrossSection('eurofer', T_grid)

    print(f"  Operating temperature : {T_operating:.1f} K")
    print(f"  Number of groups      : {pc.N_GROUPS} (fast groups, 0-14.9 MeV)")
    print(f"  Temperature grid      : {T_grid}")

    # Build per-cell macroscopic total cross section
    Sigma_t = [[0.0] * geo.n_cells for _ in range(pc.N_GROUPS)]
    for g in range(pc.N_GROUPS):
        sigma_li6_T = lib_li6.sigma(g, T_operating)
        sigma_li7_T = lib_li7.sigma(g, T_operating)
        sigma_eurofer_T = lib_eurofer.sigma(g, T_operating)
        for i in range(geo.n_cells):
            f_p = geo.pebble_fraction[i]
            tag = geo.material_tag(i)
            if tag in ('fw', 'bw'):
                Sigma_t[g][i] = sigma_eurofer_T * pc.ATOM_DENSITY_EUROFER
            elif tag == 'li2o_pebble':
                sigma_mix = (pc.FRAC_LI6 * sigma_li6_T
                             + pc.FRAC_LI7 * sigma_li7_T)
                Sigma_t[g][i] = (f_p * pc.ATOM_DENSITY_LI * sigma_mix
                                  + (1.0 - f_p) * sigma_eurofer_T
                                  * pc.ATOM_DENSITY_EUROFER)
            else:
                Sigma_t[g][i] = sigma_eurofer_T * pc.ATOM_DENSITY_EUROFER

    # Sanity: check positivity and ordering
    for g in range(pc.N_GROUPS):
        for i in range(geo.n_cells):
            if Sigma_t[g][i] < 0.0:
                Sigma_t[g][i] = pc.EPS_NUMERICAL

    sigma_t_mean = sum(
        Sigma_t[g][i] for g in range(pc.N_GROUPS) for i in range(geo.n_cells)
    ) / (pc.N_GROUPS * geo.n_cells)
    kv("mean Sigma_t (cm^-1)", sigma_t_mean)
    kv("min Sigma_t (cm^-1)",
       min(Sigma_t[g][i] for g in range(pc.N_GROUPS) for i in range(geo.n_cells)))
    kv("max Sigma_t (cm^-1)",
       max(Sigma_t[g][i] for g in range(pc.N_GROUPS) for i in range(geo.n_cells)))

    # Stratified uncertainty sampling for one cross-section value
    mean_xs, std_xs = xsm.stratified_sample_cross_section(
        lib_li6, group=0, T_center_K=T_operating, T_sigma_K=50.0,
        n_samples=20, seed=1729,
    )
    kv("Li-6 sigma_g=0 mean (stratified)", mean_xs)
    kv("Li-6 sigma_g=0 std  (stratified)", std_xs)

    # ===================================================================
    # 3. Scattering matrix (isotropic, down-scatter only)
    # ===================================================================
    section("3. Scattering matrix (isotropic, downscatter)")
    # Build a simple downscatter model:  Sigma_s(g'->g) = c * Sigma_t(g)
    # for g > g' (downscatter), with c = 0.3 (moderating ratio).
    c_scatter = 0.30
    Sigma_s = [[[0.0] * pc.N_GROUPS for _ in range(pc.N_GROUPS)]
               for _ in range(geo.n_cells)]
    for i in range(geo.n_cells):
        for gp in range(pc.N_GROUPS):
            for g in range(gp + 1, pc.N_GROUPS):
                Sigma_s[i][g][gp] = c_scatter * Sigma_t[gp][i] / max(
                    pc.N_GROUPS - gp - 1, 1)

    # Effective scattering ratio for stability analysis
    c_mean = c_scatter * (pc.N_GROUPS - 1) / (2.0 * pc.N_GROUPS)
    kv("effective scattering ratio c", c_mean)

    # ===================================================================
    # 4. Source spectrum (Dirichlet-multigroup D-T)
    # ===================================================================
    section("4. D-T source spectrum (Dirichlet-multigroup)")
    T_i_keV = 20.0
    alpha_dir = espec.dirichlet_source_weights(T_i_keV, concentration=50.0)
    mean_dir = espec.dirichlet_mean(alpha_dir)
    var_dir = espec.dirichlet_variance(alpha_dir)
    print(f"  Ion temperature         : {T_i_keV:.1f} keV")
    print(f"  Dirichlet concentration : 50.0")
    for g in range(min(5, pc.N_GROUPS)):
        kv(f"  <p_g={g}>", mean_dir[g], width=24)
        kv(f"  Var[p_g={g}]", var_dir[g], width=24)

    # Draw one sample and normalise
    p_sample = espec.sample_group_fraction(T_i_keV, concentration=50.0,
                                           seed=314159)
    # Source strength: 1e14 neutrons / cm^2 / s at the plasma face
    S0 = 1.0e14
    source = [[0.0] * geo.n_cells for _ in range(pc.N_GROUPS)]
    for g in range(pc.N_GROUPS):
        # uniform source in the first cell (plasma-facing)
        source[g][0] = S0 * p_sample[g] / max(geo.dx, pc.EPS_NUMERICAL)
    kv("total source S0 (n/cm^2/s)", S0)

    # ===================================================================
    # 5. Angular quadrature (S_N)
    # ===================================================================
    section("5. S_N angular quadrature")
    sn_order = pc.DEFAULT_SN_ORDER
    sn = aq.SNQuadrature(sn_order)
    print(f"  S_N order : {sn_order}")
    print(f"  Number of directions : {sn.n_dirs}")
    for m in range(sn.n_dirs):
        print(f"    mu[{m:2d}] = {sn.mu[m]:+.6f}    w[{m:2d}] = {sn.w[m]:.6f}")
    kv("sum w_m", sum(sn.w))

    # Reference triangle integration (for angular moment cross-checks)
    tri_area = aq.triangle01_area((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    tri_int = aq.triangle01_monomial_integral(
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), e1=1, e2=1,
    )
    kv("reference triangle area", tri_area)
    kv("integral of x*y over ref tri", tri_int)

    # Local basis test
    test_x = [0.0, 0.5, 1.0]
    test_v = [1.0, 2.0, 1.5]
    test_eval = aq.local_basis_1d(test_x, test_v, order=3, sample_x=0.25)
    kv("local_basis_1d at x=0.25", test_eval)

    # ===================================================================
    # 6. Transport solve
    # ===================================================================
    section("6. Multigroup SN transport solve")
    solver = fdt.MultigroupTransportSolver(
        geometry=geo,
        xs_total=Sigma_t,
        xs_scatter=Sigma_s,
        source=source,
        sn_order=sn_order,
        max_outer=50,
        tol=1.0e-6,
    )
    t_solve = time.time()
    result = solver.solve()
    t_solve = time.time() - t_solve
    kv("outer iterations", result["outer_iters"])
    kv("final residual", result["final_residual"])
    kv("converged", result["converged"])
    kv("solve time (s)", t_solve)

    phi = result["phi"]
    # Report scalar flux in a few groups / locations
    for g in [0, 3, 7, 13]:
        i_mid = geo.n_cells // 2
        kv(f"phi_g={g} at mid-cell", phi[g][i_mid])
    # Total flux at mid-plane
    phi_total_mid = sum(phi[g][geo.n_cells // 2] for g in range(pc.N_GROUPS))
    kv("total flux at mid-plane", phi_total_mid)

    # ===================================================================
    # 7. Stability analysis
    # ===================================================================
    section("7. Stability analysis")
    stab_report = stab.full_stability_report(
        n_cells=geo.n_cells, dx=geo.dx,
        mu_values=sn.mu, sigma_t_mean=sigma_t_mean, c_mean=c_mean,
    )
    for k, v in stab_report.items():
        kv(k, v)
    # Detailed von Neumann factors for a few directions
    for m in [0, sn.n_dirs // 2, sn.n_dirs - 1]:
        mu = sn.mu[m]
        factors = stab.von_neumann_factor(c_mean, mu, sigma_t_mean, geo.dx,
                                           n_modes=16)
        max_g = max(abs(g) for _, g in factors)
        kv(f"max |g(k)| for mu[{m}]={mu:+.4f}", max_g)
    # Chladni eigenmodes
    modes = stab.laplacian_eigenmodes_1d(geo.n_cells, geo.dx, n_modes=5)
    for idx, (lam, vec) in enumerate(modes):
        kv(f"eigenmode {idx+1}: lambda", lam)
        kv(f"  max |v|", max(abs(v) for v in vec))

    # ===================================================================
    # 8. Feynman-Kac verification
    # ===================================================================
    section("8. Feynman-Kac stochastic verification")
    # Use a simple 1-group approximation for the FK estimator
    def sigma_t_1g(x: float) -> float:
        # approximate Sigma_t at group 0
        return sigma_t_mean
    def source_1g(x: float) -> float:
        # source in the first cell only
        if x < geo.dx:
            return S0 / max(geo.dx, pc.EPS_NUMERICAL)
        return 0.0
    x_eval = 0.5 * geo.dx
    mu_eval = sn.mu[-1]     # most forward direction
    phi_det = phi[0][0]     # deterministic group-0 flux at first cell
    fk_rep = mcfk.fk_verification_report(
        x_eval, mu_eval, sigma_t_1g, source_1g, phi_det,
        n_walks=200, domain_L=geo.L_total, seed=141421,
    )
    for k, v in fk_rep.items():
        kv(k, v)

    # Monty-Hall scattering angle test
    mu_test = 0.8
    mu_out = mcfk.monty_hall_scattering_angle(mu_test, n_doors=3, seed=628318)
    kv("Monty-Hall scattering: mu_in", mu_test)
    kv("Monty-Hall scattering: mu_out", mu_out)
    # Conditional scattering kernel
    mu_mean, mu_std = mcfk.conditional_scatter_kernel(
        mu_in=0.5, A_mass=6.0, n_samples=100, seed=161803,
    )
    kv("conditional scatter mean(Li-6)", mu_mean)
    kv("conditional scatter std (Li-6)", mu_std)

    # ===================================================================
    # 9. TBR calculation
    # ===================================================================
    section("9. Tritium Breeding Ratio")
    tbr_res = tbr.compute_tbr(phi, geo.dx, geo.pebble_fraction,
                               temperature_K=T_operating)
    for k, v in tbr_res.items():
        kv(k, v)
    # Sensitivity to enrichment
    enrichments = [0.3, 0.5, 0.6, 0.75, 0.9]
    enr_results = tbr.tbr_vs_enrichment(
        phi, geo.dx, geo.pebble_fraction, enrichments, T_operating,
    )
    print()
    print("  TBR vs Li-6 enrichment:")
    for r in enr_results:
        print(f"    f_Li6 = {r['enrichment']:.2f}  ->  TBR = {r['tbr_total']:.4f}")
    # Flow continuity check
    balance = tbr.tritium_production_balance(
        phi, geo.dx, geo.pebble_fraction, T_operating,
    )
    print()
    print("  Tritium production balance:")
    for k, v in balance.items():
        kv(k, v)

    # ===================================================================
    # 10. Latent-space decomposition of XS library
    # ===================================================================
    section("10. VAE-inspired XS library decomposition")
    decomposer = latd.LatentDecomposer(
        material='li2o', T_grid_K=[300.0, 600.0, 900.0, 1200.0, 1500.0],
        n_modes=3,
    )
    for T_test in [450.0, 750.0, 1100.0]:
        loss = decomposer.reconstruction_quality(T_test)
        print(f"  T = {T_test:.0f} K:")
        for k, v in loss.items():
            kv(f"    {k}", v, width=28)
    # Latent-space interpolation
    z_interp_spec = decomposer.interpolate_temperatures(600.0, 1200.0, alpha=0.5)
    kv("interpolated sigma[0] (T=900 K via latent)", z_interp_spec[0])
    kv("direct sigma[0] at T=900 K", lib_li2o.sigma(0, 900.0))

    # ===================================================================
    # 11. Current balance / conservation check
    # ===================================================================
    section("11. Particle conservation check")
    balance_rep = fdt.current_balance_check(geo, phi, Sigma_t, sn)
    for k, v in balance_rep.items():
        kv(k, v)

    # ===================================================================
    # 12. Dirichlet-multigroup source statistics
    # ===================================================================
    section("12. Source statistics (Dirichlet-multigroup)")
    # Multiple Dirichlet draws with lower concentration for visible variation
    n_draws = 10
    conc_low = 5.0    # lower concentration -> broader Dirichlet
    draws = [espec.sample_group_fraction(T_i_keV, conc_low, seed=100 + d)
             for d in range(n_draws)]
    # per-group mean and std over draws
    p_mean = [sum(draws[d][g] for d in range(n_draws)) / n_draws
              for g in range(pc.N_GROUPS)]
    p_std = [
        math.sqrt(sum((draws[d][g] - p_mean[g]) ** 2 for d in range(n_draws))
                  / max(n_draws - 1, 1))
        for g in range(pc.N_GROUPS)
    ]
    print(f"  {n_draws} Dirichlet draws at T_i = {T_i_keV} keV, concentration = {conc_low}:")
    for g in range(min(7, pc.N_GROUPS)):
        kv(f"  p_mean[g={g}]", p_mean[g], width=24)
        kv(f"  p_std [g={g}]", p_std[g], width=24)

    # Slowing-down tail sample
    E_tail = espec.sample_slowing_down_tail(
        group_lo_mev=pc.GROUP_BOUNDS_MEV[-2],
        xi_mev=0.5, u=0.3,
    )
    kv("slowing-down tail sample energy (MeV)", E_tail)

    # ===================================================================
    # 13. Gauss-Legendre cross-check
    # ===================================================================
    section("13. Gauss-Legendre quadrature cross-check")
    nodes, weights = espec.gauss_legendre(8)
    # integrate x^2 over [-1,1] -> 2/3
    integral = sum(w * n * n for n, w in zip(nodes, weights))
    kv("integral of x^2 on [-1,1] (GL-8)", integral)
    kv("exact value", 2.0 / 3.0)
    kv("error", abs(integral - 2.0 / 3.0))

    # ===================================================================
    # Final summary
    # ===================================================================
    section("FINAL SUMMARY")
    t_total = time.time() - t0
    kv("total runtime (s)", t_total)
    kv("blanket thickness (cm)", pc.BLANKET_THICKNESS_CM)
    kv("number of spatial cells", geo.n_cells)
    kv("number of energy groups", pc.N_GROUPS)
    kv("SN order", sn_order)
    kv("TBR", tbr_res["tbr_total"])
    kv("TBR self-sufficient (>=1.05)", tbr_res["self_sufficient"])
    kv("stability: von Neumann rho", stab_report["von_neumann_rho"])
    kv("stability: matrix spectral rho", stab_report["matrix_spectral_radius"])
    kv("transport converged", result["converged"])
    kv("FK agreement", fk_rep["agreement"])

    print()
    print("#" * 72)
    print("#  Computation complete.")
    print("#" * 72)
    print()


if __name__ == "__main__":
    main()
