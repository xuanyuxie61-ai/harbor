"""
main.py — Unified entry point for PROJECT_253.

Scientific domain:
    Computational astrophysics: gravitational-wave template numerical computation
    via high-order finite differences and stability analysis.

Workflow
--------
 1. Define a standard binary-black-hole system (physics_constants).
 2. Build the Schwarzschild background and Regge-Wheeler potential on a
    uniform tortoise-coordinate grid (spacetime_geometry).
 3. Construct high-order finite-difference stencils and verify their
    truncation-error coefficients (high_order_fd).
 4. Perform von Neumann stability analysis; report CFL limits for orders
    p = 2, 4, 6, 8, 10 and select the safe time step (stability_analyzer).
 5. Time-evolve the Regge-Wheeler equation with a Gaussian-pulse initial
    data and extract the waveform at a finite radius (regge_wheeler_solver).
 6. Compute the post-Newtonian inspiral waveform and matched-filter SNR
    using Gauss-Chebyshev / Gauss-Hermite quadrature (waveform_template,
    spectral_quadrature).
 7. Extract the quasi-normal-mode spectrum via a companion-matrix
    eigenvalue problem in the Chebyshev basis (qnm_companion).
 8. Demonstrate domain decomposition with CUDA-style grid/thread indexing
    and adaptive topology reconfiguration (domain_decomposition,
    adaptive_topology).
 9. Perform a Monte Carlo coverage study of a template bank and a
    clock-solitaire-style parameter sweep (monte_carlo_explorer).
10. Apply boundary treatments (Sommerfeld, hyperboloidal, sponge layer),
    integrate a Duffing-type perturbed-ringdown oscillator, and evaluate
    logistic-amplitude and capped-spread envelopes (boundary_engine).
11. Print a comprehensive text report of all results.
"""

from __future__ import annotations
import math
import numpy as np

# project modules
import physics_constants as pc
import spacetime_geometry as sg
import high_order_fd as fd
import stability_analyzer as sa
import regge_wheeler_solver as rw
import waveform_template as wt
import spectral_quadrature as sq
import qnm_companion as qnm
import domain_decomposition as dd
import adaptive_topology as at
import monte_carlo_explorer as mce
import boundary_engine as be


def section(title: str) -> None:
    bar = "=" * 72
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def main() -> None:
    print()
    print("PROJECT 253 — Gravitational-wave template computation")
    print("  via high-order finite differences and stability analysis")
    print("  (small-scale reproducible experiment)")

    # ------------------------------------------------------------------
    #  1. Physical parameters
    # ------------------------------------------------------------------
    section("1. Binary black-hole parameters")
    params = pc.equal_mass_light_bbh()
    print(f"  m1 = {params.m1_si} Msun,  m2 = {params.m2_si} Msun")
    print(f"  chirp mass   M_c   = {params.chirp_mass:.4f} Msun")
    print(f"  total mass   M     = {params.M_tot:.4f} Msun")
    print(f"  sym. ratio   nu    = {params.symmetric_ratio:.4f}")
    print(f"  f_ISCO             = {params.f_isco_hz:.2f} Hz")
    print(f"  distance     D_L   = {params.distance_mpc} Mpc")
    print(f"  ell                = {params.ell}")
    print(f"  parameter set valid: {params.validate()}")

    # ------------------------------------------------------------------
    #  2. Spacetime geometry
    # ------------------------------------------------------------------
    section("2. Schwarzschild background and Regge-Wheeler potential")
    M_geom = params.M_tot            # total mass in geometric units
    Nx = 1024
    rstar, r_geo, dr = sg.build_tortoise_grid(M_geom, Nx,
                                               r_min_factor=1.0 + 1.0e-6,
                                               r_max_factor=60.0)
    V = sg.sample_potential_on_grid(r_geo, M_geom, ell=params.ell,
                                    potential_kind="regge_wheeler")
    V_zer = sg.sample_potential_on_grid(r_geo, M_geom, ell=params.ell,
                                        potential_kind="zerilli")
    print(f"  grid: N = {Nx},  dr* = {dr:.4e}")
    print(f"  r*_min = {rstar[0]:.4e},  r*_max = {rstar[-1]:.4e}")
    print(f"  r_min  = {r_geo[0]:.4e}  (horizon at {2.0 * M_geom:.4e})")
    print(f"  max(V_RW) = {np.max(V):.4e}  at r = {r_geo[np.argmax(V)]:.4e}")
    print(f"  max(V_Ze) = {np.max(V_zer):.4e}")
    print(f"  photon-sphere radius = {sg.photon_sphere_radius(M_geom):.4e}")
    # tortoise round-trip test
    rs_test = 10.0 * M_geom
    r_test = sg.inverse_tortoise(rs_test, M_geom)
    rs_back = sg.tortoise_coordinate(r_test, M_geom)
    print(f"  tortoise round-trip |r* - r*'| = {abs(rs_test - rs_back):.2e}")

    # ------------------------------------------------------------------
    #  3. High-order finite-difference stencils
    # ------------------------------------------------------------------
    section("3. High-order finite-difference stencils")
    for order in (2, 4, 6, 8):
        c = fd.STENCIL_D2[order]
        coeff = fd.truncation_error_coeff(order)
        print(f"  order {order}:  support = {len(c)},  "
              f"C_0 = {c[len(c)//2]:+.6f},  "
              f"leading trunc. coeff = {coeff:+.4e}")

    # test D2 on sin(x):  d^2 sin / dx^2 = -sin
    x_test = np.linspace(0.0, 2.0 * np.pi, 201, endpoint=False)
    dx_test = x_test[1] - x_test[0]
    u_test = np.sin(x_test)
    for order in (2, 4, 6, 8):
        D2u = fd.apply_d2(u_test, dx_test, order, bc="periodic")
        err = np.max(np.abs(D2u + u_test))
        print(f"  FD order {order} on sin(x):  max|D2 sin + sin| = {err:.3e}")

    # ------------------------------------------------------------------
    #  4. Von Neumann stability analysis
    # ------------------------------------------------------------------
    section("4. Von Neumann stability analysis and CFL limits")
    cfl_table = sa.cfl_table()
    for order, cmax in cfl_table.items():
        print(f"  order {order}:  C_max = dt/dr* = {cmax:.6f}")
    # report amplification at a typical Courant number
    C_used = 0.5
    for order in (2, 4, 6, 8):
        gmax = sa.max_amplification(order, C_used)
        print(f"  order {order}, C = {C_used}:  max|g| = {gmax:.6f}")
    # spectral scan (print a few samples)
    theta, g_theta = sa.spectral_scan(4, 0.5, n_sample=9)
    print(f"  spectral scan (order 4, C=0.5):")
    for th, gg in zip(theta[::1], g_theta[::1]):
        print(f"    theta = {th:.3f}, |g| = {gg:.6f}")

    # ------------------------------------------------------------------
    #  5. Regge-Wheeler time evolution
    # ------------------------------------------------------------------
    section("5. Time-domain Regge-Wheeler evolution")
    # Gaussian pulse centred near the potential peak
    rstar_peak = rstar[np.argmax(V)]
    width = 3.0 * dr
    Psi0, dPsi0 = rw.gaussian_pulse(rstar, rstar_peak, width, 1.0)
    t_final = 200.0 * M_geom
    result = rw.solve_regge_wheeler(
        rstar=rstar, V=V, dr=dr,
        t_final=t_final,
        cfl_factor=0.5,
        fd_order=4,
        initial_data=(Psi0, dPsi0),
        source=None,
        verbose=False,
    )
    print(f"  status   : {result['status']}")
    print(f"  Nt       : {len(result['t'])}")
    print(f"  dt       : {result['dt']:.4e}")
    print(f"  max|Psi| : {np.max(np.abs(result['Psi'])):.4e}")
    # extract waveform close to the peak (outgoing wave)
    rstar_ext = rstar_peak + 20.0 * dr
    # clip to grid range
    rstar_ext = min(rstar_ext, rstar[-1])
    t_ext, h_ext = rw.extract_waveform(result, rstar_ext)
    print(f"  extraction radius r*_ext = {rstar_ext:.4e}  (peak at {rstar_peak:.4e})")
    print(f"  max|h| at extraction = {np.max(np.abs(h_ext)):.4e}")

    # ------------------------------------------------------------------
    #  6. PN waveform and matched-filter SNR
    # ------------------------------------------------------------------
    section("6. Post-Newtonian inspiral waveform and matched filtering")
    t_td, h_td, f_td = wt.generate_td_waveform(params, n_points=1024)
    print(f"  leading-order chirp:  {len(t_td)} points")
    print(f"  duration  T = {t_td[-1]:.4e} s")
    print(f"  f_low     = {params.f_low_hz:.2f} Hz")
    print(f"  f_max     = {f_td[-1]:.2f} Hz")
    print(f"  max|h|    = {np.max(np.abs(h_td)):.4e}")
    # SPA waveform on a frequency grid
    f_arr = np.linspace(params.f_low_hz, 0.9 * params.f_isco_hz, 256)
    h_tilde = wt.generate_spa_waveform(params, f_arr)
    print(f"  SPA waveform: {len(f_arr)} freq bins, "
          f"max|h_tilde| = {np.max(np.abs(h_tilde)):.4e}")
    # matched-filter SNR
    rho_raw = sq.matched_filter_snr(
        lambda f: np.interp(f, f_arr, h_tilde, left=0.0, right=0.0),
        lambda f: np.interp(f, f_arr, h_tilde, left=0.0, right=0.0),
        sq.aligo_psd_design,
        f_min=params.f_low_hz,
        f_max=0.9 * params.f_isco_hz,
        n_quad=64,
    )
    rho = float(rho_raw)
    print(f"  matched-filter SNR (aLIGO design) = {rho:.4f}")

    # ------------------------------------------------------------------
    #  7. Quasi-normal mode extraction
    # ------------------------------------------------------------------
    section("7. Quasi-normal-mode spectrum via companion matrices")
    modes_mono = qnm.find_qnm_frequencies(ell=2, n_max=6, basis="monomial")
    modes_cheb = qnm.find_qnm_frequencies(ell=2, n_max=6, basis="chebyshev")
    modes_herm = qnm.find_qnm_frequencies(ell=2, n_max=6, basis="hermite")
    modes_lege = qnm.find_qnm_frequencies(ell=2, n_max=6, basis="legendre")
    print(f"  monomial basis : {len(modes_mono)} modes")
    for i, w in enumerate(modes_mono[:3]):
        print(f"    omega_{i} = {w.real:+.5f} {w.imag:+.5f} i")
    print(f"  chebyshev basis: {len(modes_cheb)} modes")
    for i, w in enumerate(modes_cheb[:3]):
        print(f"    omega_{i} = {w.real:+.5f} {w.imag:+.5f} i")
    print(f"  hermite basis  : {len(modes_herm)} modes")
    print(f"  legendre basis : {len(modes_lege)} modes")
    # ringdown test
    if modes_mono:
        t_ring = np.linspace(0.0, 50.0, 500)
        h_ring = qnm.ringdown_signal(t_ring, modes_mono[:3])
        print(f"  ringdown test: max|h| = {np.max(np.abs(h_ring)):.4e}")

    # ------------------------------------------------------------------
    #  8. Domain decomposition and adaptive topology
    # ------------------------------------------------------------------
    section("8. Domain decomposition and adaptive topology")
    layout = dd.CudaLayout(blocks=(4, 1, 1), threads=(128, 1, 1), n_tasks=Nx)
    stats = dd.load_balance_stats(layout)
    print(f"  CUDA layout: blocks={layout.blocks}, threads={layout.threads}")
    print(f"  total_threads={stats['total_threads']}, active={stats['active_threads']}, "
          f"idle_fraction={stats['idle_fraction']:.3f}")
    print(f"  tasks/thread: min={stats['min_tasks']}, max={stats['max_tasks']}, "
          f"mean={stats['mean_tasks']:.2f}")
    # 1D partitioning
    parts = dd.partition_with_halo(Nx, n_subdomains=4, halo=2)
    for p in parts:
        print(f"    subdomain {p['id']}: interior=[{p['i_start']},{p['i_end']}), "
              f"total=[{p['halo_lo']},{p['halo_hi']})")
    # FD operator assembly
    entries = dd.assemble_fd_operator(Nx, list(fd.STENCIL_D2[4]), n_subdomains=4)
    print(f"  FD operator non-zeros: {len(entries)}")
    # adaptive topology
    if len(h_ext) >= Nx:
        u_field = np.abs(h_ext[:Nx])
    else:
        u_field = np.pad(np.abs(h_ext), (0, Nx - len(h_ext)))
    g0 = at.MeshGraph(rstar, u_field)
    q0 = at.topology_quality(g0)
    print(f"  initial mesh: N={q0['n_nodes']:.0f}, min_dx={q0['min_dx']:.3e}, "
          f"aspect={q0['aspect_ratio']:.3f}")
    g_simplified = at.simplify_topology(g0, tol=1.0e-3)
    q1 = at.topology_quality(g_simplified)
    print(f"  after simplify: N={q1['n_nodes']:.0f}")
    g_refined = at.resilient_topology(g0, threshold=1.0e-4, max_nodes=2000)
    q2 = at.topology_quality(g_refined)
    print(f"  after refine  : N={q2['n_nodes']:.0f}, aspect={q2['aspect_ratio']:.3f}")
    sub_parts = at.decompose_topology(g0, n_parts=4, halo=2)
    print(f"  decomposed into {len(sub_parts)} sub-topologies")

    # ------------------------------------------------------------------
    #  9. Monte Carlo template-bank coverage
    # ------------------------------------------------------------------
    section("9. Monte Carlo template-bank coverage study")
    sweep = mce.clock_sweep(n_rounds=20, seed=42)
    print(f"  clock-solitaire sweep: {sweep['steps_taken']} steps, "
          f"mean M_c={sweep['mean_mc']:.3f}, std M_c={sweep['std_mc']:.3f}")
    coverage = mce.coverage_fraction(n_templates=16, n_signals=32,
                                     mismatch_threshold=0.03, seed=7)
    print(f"  coverage: {coverage['covered']}/{coverage['n_signals']} "
          f"= {coverage['coverage_fraction']:.3f} "
          f"(threshold={coverage['threshold']})")

    # ------------------------------------------------------------------
    #  10. Boundary treatments and auxiliary models
    # ------------------------------------------------------------------
    section("10. Boundary treatments, Duffing ringdown, logistic envelope")
    # sponge layer
    sigma = be.sponge_profile(rstar, rstar[0], rstar[-1],
                              width=10.0 * dr, amplitude=3.0)
    print(f"  sponge: max sigma = {np.max(sigma):.4e}")
    # hyperboloidal map
    rho_hyp = be.hyperboloidal_map(rstar, R=20.0 * M_geom)
    print(f"  hyperboloidal rho range: [{rho_hyp[0]:.4f}, {rho_hyp[-1]:.4f}]")
    # Duffing oscillator
    t_duf, y_duf = be.integrate_duffing((0.0, 40.0), np.array([1.0, 0.0]),
                                        n_steps=2000,
                                        alpha=1.0, beta=0.2, gamma=0.3,
                                        delta=0.1, omega=1.0)
    print(f"  Duffing oscillator: max|x| = {np.max(np.abs(y_duf[:, 0])):.4e}")
    # logistic amplitude
    t_log = np.linspace(0.0, 20.0, 200)
    A_log = be.logistic_amplitude(t_log, r=0.5, K=1.0, t0=0.0, A0=0.01)
    print(f"  logistic envelope: max A = {np.max(A_log):.4e}")
    # capped spread
    A_cap = be.capped_spread(t_log, t0=5.0, L=10.0, tau=0.3)
    print(f"  capped spread    : max A = {np.max(A_cap):.4e}")
    # area under the extracted waveform
    area = be.area_under_curve(np.abs(h_ext), t_ext)
    print(f"  area under |h(t)| = {area:.4e}")
    # interpolation
    x_fine = np.linspace(t_ext[0], t_ext[-1], 5 * len(t_ext))
    h_fine = be.piecewise_linear_interp(t_ext, h_ext, x_fine)
    print(f"  interpolated h: {len(h_fine)} points, max|h| = {np.max(np.abs(h_fine)):.4e}")

    # ------------------------------------------------------------------
    #  11. Quadrature exactness tests
    # ------------------------------------------------------------------
    section("11. Quadrature exactness tests (Chebyshev / Hermite)")
    cheb_err = sq.chebyshev1_exactness(n=8, degree_max=20)
    print(f"  Gauss-Chebyshev1 (n=8) exactness errors:")
    for k in (0, 2, 4, 6, 8, 10, 14, 16, 20):
        if k in cheb_err:
            print(f"    degree {k:2d}: {cheb_err[k]:.3e}")
    herm_err = sq.hermite_exactness(n=8, degree_max=20)
    print(f"  Gauss-Hermite (n=8) exactness errors:")
    for k in (0, 2, 4, 6, 8, 10, 14, 16, 20):
        if k in herm_err:
            print(f"    degree {k:2d}: {herm_err[k]:.3e}")

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    print(f"  Binary system: {params.m1_si} + {params.m2_si} Msun")
    print(f"  Grid: {Nx} points in r*, dr* = {dr:.4e}")
    print(f"  RW evolution status: {result['status']}")
    print(f"  Peak strain at extraction: {np.max(np.abs(h_ext)):.4e}")
    print(f"  Matched-filter SNR: {rho:.4f}")
    print(f"  Number of QNMs (monomial): {len(modes_mono)}")
    print(f"  Template-bank coverage: {coverage['coverage_fraction']:.3f}")
    print()
    print("  Project 253 completed successfully.")
    print()


if __name__ == "__main__":
    main()
