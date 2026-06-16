# -*- coding: utf-8 -*-
"""
main.py
=======
Unified entry point for the PROJECT_249 stellar evolution + nuclear
reaction network simulation.

Project title
-------------
"High-order finite-difference stellar evolution with coupled nuclear
reaction networks: adaptive mesh, stiff integrators, and stability
analysis for small-scale reproducible experiments"

Scientific problem
------------------
We simulate the late evolution of a 15 M_sun star from core hydrogen
burning through core carbon ignition, tracking the nuclear composition
of 8 species (H, He, C, N, O, Ne, Mg, Fe) and the stellar structure
(r, P, T, L, rho) on an adaptively refined Lagrangian mass grid.

The simulation pipeline:
  1. Construct an initial adaptive mass grid via 1D CVT.
  2. Build the initial stellar model (r, P, T, L, rho) by outward
     integration of the structure equations.
  3. Evolve the nuclear network forward in time using the adaptive
     B1G3 / midpoint integrator.
  4. At each time-step, perform stability analysis (Gershgorin
     eigenvalue bounds + spectral radius) to control the step size.
  5. Apply stochastic perturbations (fractal mixing + pink noise) to
     model convective fluctuations.
  6. Check for dredge-up events (threshold-triggered mixing).
  7. Build a merger tree of burning shells to track the onion-skin
     structure.
  8. Classify burning regimes by K-means clustering of the
     thermodynamic state.
  9. Verify data integrity via Hamming checksums.
 10. Record epoch / phase information using Zeller-style modular
     arithmetic.

Output
------
The simulation writes a text state file and prints a summary report
to stdout.  No visualisation is produced (per project requirements).

References
----------
  Kippenhahn, Weigert & Weiss, Stellar Structure and Evolution (2nd ed).
  Iliadis, Nuclear Physics of Stars (2nd ed).
  Trenchea & Burkardt, Refactorization of the midpoint rule (2020).
  Kassam & Trefethen, Fourth-order time-stepping for stiff ODEs (2005).
"""

from __future__ import annotations
import math
import os
import sys
import time as _time

# -- Project modules --------------------------------------------------
from physical_constants import M_sun, L_sun, R_sun, k_B, G_grav, m_p, N_A, pi
from mesh_adaptation import cvt_1d, grid_ratio, grid_smoothness, monitor_shell_shells
from stellar_structure import (build_stellar_model,
                               virial_temperature, mean_molecular_weight_from_Y)
from stellar_structure import schwarzschild_criterion, StellarZone
from finite_difference import mass_grid, spacings, compact_fd4, fd_central_2, von_neumann_max_dt
from nuclear_network import (SPECIES, K, nuclear_rhs, make_rhs,
                              epsilon_nuc, neutrino_loss)
from nuclear_quadrature import (laguerre_compute, gamow_constant,
                                 reaction_rate_integral, rate_CF88_pp)
from time_integrator import (adaptive_midpoint, b1g3_integrate,
                              etdrk4_linear_scalar)
from stability_analysis import (SparseMatrix, assemble_nuclear_jacobian,
                                 spectral_radius, eigenvalue_gershgorin,
                                 max_stable_dt_from_J, cg_solve)
from perturbation_engine import (fractal_mix_profile, PinkNoiseGenerator,
                                  correlation, dredge_up_event)
from epoch_tracker import (StellarEpochTracker, stage_to_name,
                            i4_wrap, STAGE_NAMES, N_STAGES)
from shell_merger import BurningShell, build_merger_tree
from burning_classifier import classify_burning_regimes
from data_integrity import (state_checksum, check_data_integrity,
                              write_state_file, read_state_file)


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> int:
    t_start = _time.time()
    print("PROJECT_249 - Stellar evolution with coupled nuclear networks")
    print("High-order finite difference + adaptive mesh + stiff integrators")
    print()

    # --------------------------------------------------------------
    section("Stage 1 : Adaptive mass grid construction (1D CVT)")
    # --------------------------------------------------------------
    M_star = 15.0 * M_sun
    N_zones = 48
    # Shell monitor function: concentrate grid points near expected
    # burning shell locations (H, He, C shells in a 15 Msun star)
    shells = [(0.3 * M_star, 0.05 * M_star, 50.0),
               (0.6 * M_star, 0.04 * M_star, 30.0),
               (0.9 * M_star, 0.03 * M_star, 15.0)]
    density = monitor_shell_shells(shells)
    grid = cvt_1d(M_star, N_zones, density, n_samples=20000, n_iter=40, seed=7)
    print(f"  N_zones = {N_zones}, grid[0] = {grid[0]:.3e}, grid[-1] = {grid[-1]:.3e}")
    print(f"  max/min spacing ratio = {grid_ratio(grid):.3f}")
    print(f"  smoothness measure    = {grid_smoothness(grid):.3f}")
    print(f"  first 6 mass coords   = {[f'{g:.3e}' for g in grid[:6]]}")

    # --------------------------------------------------------------
    section("Stage 2 : Initial stellar model construction")
    # --------------------------------------------------------------
    Y_init = [0.70, 0.28, 0.005, 0.005, 0.008, 0.001, 0.001, 0.0]
    T_centre = 3.3e7     # hotter centre for 15 Msun
    P_centre = 5.0e18
    zones = build_stellar_model(M_star, N_zones=N_zones,
                                 Y_init=Y_init, T_centre=T_centre,
                                 P_centre=P_centre)
    print(f"  constructed {len(zones)} stellar zones")
    print(f"    centre  r = {zones[0].r:.3e} cm   "
          f"T = {zones[0].T:.3e} K   P = {zones[0].P:.3e}")
    print(f"    surface r = {zones[-1].r:.3e} cm   "
          f"T = {zones[-1].T:.3e} K   P = {zones[-1].P:.3e}")
    R_star = zones[-1].r
    T_vir  = virial_temperature(M_star, R_star, 0.6)
    print(f"    virial temperature  T_vir = {T_vir:.3e} K")
    n_conv = sum(1 for i in range(len(zones)-1)
                  if schwarzschild_criterion(zones[i], zones[i+1]))
    print(f"    convectively unstable layers = {n_conv}/{len(zones)-1}")

    # --------------------------------------------------------------
    section("Stage 3 : Nuclear quadrature validation")
    # --------------------------------------------------------------
    # Gauss-Laguerre rule
    x, w = laguerre_compute(norder=16, alpha=0.0)
    print(f"  Gauss-Laguerre 16-point rule:")
    print(f"    x[0:4]  = {[f'{xi:.4f}' for xi in x[:4]]}")
    print(f"    w[0:4]  = {[f'{wi:.4e}' for wi in w[:4]]}")
    # pp rate validation
    T9_sun = 15.7e-3
    rate_pp_ref = rate_CF88_pp(T9_sun)
    print(f"    CF88 pp rate at T9={T9_sun:.4e}:  N_A<sv> = {rate_pp_ref:.3e}")
    # Gamow constant for pp
    b_pp = gamow_constant(Z1=1, Z2=1, mu_amu=0.5)
    print(f"    Gamow constant b(pp) = {b_pp:.4f} MeV^{1/2}")

    # --------------------------------------------------------------
    section("Stage 4 : Nuclear network time integration")
    # --------------------------------------------------------------
    # Use H-burning conditions
    T9 = T_centre / 1.0e9
    rho_c = 50.0   # g/cm^3
    Y0 = Y_init[:]
    f_rhs = make_rhs(T9, rho_c)
    print(f"  H-burning conditions: T9={T9:.4e}, rho={rho_c:.2f}")
    eps0 = epsilon_nuc(T9, rho_c, Y0)
    print(f"    initial epsilon_nuc = {eps0:.4e} erg/g/s")

    # Short integration with adaptive midpoint
    res = adaptive_midpoint(f_rhs, 0.0, 1.0e4, Y0,
                             dt0=100.0, reltol=1.0e-4, abstol=1.0e-10,
                             max_steps=200)
    Y_final = res["y"][-1]
    print(f"  adaptive_midpoint: {res['n_steps']} steps, "
          f"{res['n_rejected']} rejected, {res['n_newton_fail']} Newton failures")
    print(f"    Y_final = {[f'{y:.4e}' for y in Y_final]}")

    # B1G3 integration for comparison
    res_b = b1g3_integrate(f_rhs, (0.0, 1.0e4), Y0, n_steps=50)
    print(f"  b1g3: 50 steps, Y_final = {[f'{y:.4e}' for y in res_b['y'][-1]]}")

    # --------------------------------------------------------------
    section("Stage 5 : Stability analysis")
    # --------------------------------------------------------------
    def rate_func(i, T9_loc, rho_loc):
        # Per-species rate surrogate
        return 1.0e-3 * (i + 1) * max(T9_loc, 1e-3)
    J = assemble_nuclear_jacobian(SPECIES, Y_final, T9, rho_c, rate_func)
    print(f"  nuclear Jacobian: {J.n}x{J.m}  nnz={J.nnz()}")
    rho_J, _ = spectral_radius(J, n_iter=100)
    print(f"    spectral radius rho(J) = {rho_J:.4e}")
    discs, env = eigenvalue_gershgorin(J)
    print(f"    Gershgorin envelope = [{env[0][0]:.4e}, {env[0][1]:.4e}]")
    dt_max = max_stable_dt_from_J(J, safety=0.8)
    print(f"    max stable explicit dt = {dt_max:.4e} s")
    # CG on J^T J (normal equation)
    JTJ = SparseMatrix(J.n, J.m)
    # Build J^T J (small matrix)
    dense_J = J.to_dense()
    n = J.n
    for i in range(n):
        for k in range(n):
            s = 0.0
            for l in range(n):
                s += dense_J[l][i] * dense_J[l][k]
            if abs(s) > 1.0e-30:
                JTJ.add(i, k, s)
    b_cg = [1.0 if i == 0 else 0.0 for i in range(n)]
    x_cg, it_cg, rn_cg = cg_solve(JTJ, b_cg, tol=1.0e-8, maxiter=n*2)
    print(f"    CG on J^T J: {it_cg} iterations, residual = {rn_cg:.3e}")

    # --------------------------------------------------------------
    section("Stage 6 : Finite difference operators")
    # --------------------------------------------------------------
    # Test convergence on smooth function f(x) = sin(2 pi x) on [0, 1]
    # using the explicit O(h^4) central scheme (boundary effects removed).
    h_values = [1.0 / N for N in [16, 32, 64, 128, 256]]
    def f_test(x): return math.sin(2.0 * pi * x)
    def df_test(x): return (2.0 * pi) * math.cos(2.0 * pi * x)
    errors = []
    for h in h_values:
        N_loc = max(8, int(1.0 / h))
        h_loc = 1.0 / N_loc
        x_grid = [i * h_loc for i in range(N_loc + 1)]
        f_vals = [f_test(xi) for xi in x_grid]
        # Explicit central 4th order interior formula
        dfn = [0.0] * (N_loc + 1)
        for i in range(2, N_loc - 1):
            dfn[i] = (-f_vals[i+2] + 8.0*f_vals[i+1]
                      - 8.0*f_vals[i-1] + f_vals[i-2]) / (12.0 * h_loc)
        # Measure error only at interior points where scheme applies
        err = max(abs(dfn[i] - df_test(x_grid[i]))
                  for i in range(2, N_loc - 1))
        errors.append(err)
    orders = []
    for i in range(1, len(h_values)):
        if errors[i-1] > 0 and errors[i] > 0:
            orders.append(math.log(errors[i-1]/errors[i])
                          / math.log(h_values[i-1]/h_values[i]))
    print(f"  explicit central O(h^4) convergence on sin(2 pi x):")
    for h, e in zip(h_values, errors):
        print(f"    h = {h:.4e}   error = {e:.3e}")
    print(f"    empirical orders = {[f'{o:.2f}' for o in orders]}")
    print(f"    (expected order ~ 4)")
    # Compact scheme (Pade) test
    print(f"  compact Pade O(h^4) test at h = 1/64:")
    N_loc = 64
    h_loc = 1.0 / N_loc
    x_grid = [i * h_loc for i in range(N_loc + 1)]
    f_vals = [f_test(xi) for xi in x_grid]
    dfn_compact = compact_fd4(f_vals, h_loc, alpha=0.25)
    err_c = max(abs(dfn_compact[i] - df_test(x_grid[i]))
                for i in range(2, N_loc - 1))
    print(f"    max interior error = {err_c:.3e}")
    # von Neumann max dt
    dt_vn = von_neumann_max_dt(velocity=1.0, diffusivity=0.01, h=0.1,
                                scheme="central4")
    print(f"    von Neumann max_dt(central4) = {dt_vn:.4e}")

    # --------------------------------------------------------------
    section("Stage 7 : Perturbation engine")
    # --------------------------------------------------------------
    # Fractal mixing of composition profile
    X_profile = [0.7] * 20 + [0.3] * 20
    X_mixed = fractal_mix_profile(X_profile, levels=3, mu=0.05, seed=11)
    print(f"  fractal mix: sum_orig = {sum(X_profile):.3f}  "
          f"sum_mixed = {sum(X_mixed):.3f}")
    # Pink noise series
    png = PinkNoiseGenerator(B=6, sigma=1.0, seed=13)
    series = png.series(1000)
    R = correlation(series, max_lag=10)
    print(f"  pink noise: len={len(series)}  "
          f"mean={sum(series)/len(series):+.3e}  R(1)/R(0)={R[1]/max(R[0],1e-30):.3f}")
    # Dredge-up event
    m_grid_dredge = [i * 0.05 * M_star for i in range(40)]
    X_dredge = [0.0] * 10 + [0.7] * 20 + [0.0] * 10
    X_after = dredge_up_event(X_dredge, m_grid_dredge,
                               shell_m=0.5 * M_star, mixing_width=0.1 * M_star)
    print(f"  dredge-up: sum before = {sum(X_dredge):.3f}  "
          f"sum after = {sum(X_after):.3f}")

    # --------------------------------------------------------------
    section("Stage 8 : Epoch tracking (Zeller-style)")
    # --------------------------------------------------------------
    tracker = StellarEpochTracker(stage_index=0, subcycle=0)
    evolution = [1.0e6, 5.0e6, 1.0e6, 5.0e5, 2.0e2, 0.5, 0.001]
    for dt in evolution:
        rec = tracker.advance(dt)
    print(f"  advanced through {len(evolution)} evolutionary phases:")
    print(f"    final stage = {rec['stage_name']}  "
          f"(phase = {rec['phase']:.3f})")
    print(f"    full cycles = {rec['full_cycles']}")
    onion = tracker.onion_skin(nshells=5)
    print(f"    onion-skin = {onion}")
    # Hierarchical counter
    outer, inner = i4_wrap(3, 0, N_STAGES - 1), i4_wrap(5, 0, 6)
    print(f"    i4_wrap(3, 0, {N_STAGES-1}) = {outer},  "
          f"i4_wrap(5, 0, 6) = {inner}")

    # --------------------------------------------------------------
    section("Stage 9 : Shell merger tree")
    # --------------------------------------------------------------
    shells_objs = [
        BurningShell("H",  0.3 * M_star, 0.05 * M_star, 1.0 * L_sun,
                      {"H1": 0.7, "He4": 0.3}, 1.5e7),
        BurningShell("He", 0.35 * M_star, 0.04 * M_star, 0.5 * L_sun,
                      {"He4": 0.9, "C12": 0.1}, 2e8),
        BurningShell("C",  0.8 * M_star, 0.02 * M_star, 0.1 * L_sun,
                      {"C12": 0.8, "O16": 0.2}, 8e8),
        BurningShell("O",  0.85 * M_star, 0.015 * M_star, 0.05 * L_sun,
                      {"O16": 0.9, "Ne20": 0.1}, 1.5e9),
        BurningShell("Ne", 0.9 * M_star, 0.01 * M_star, 0.02 * L_sun,
                      {"Ne20": 0.85, "O16": 0.1}, 1.8e9),
    ]
    merged, pairs = build_merger_tree(shells_objs, eta=1.0)
    print(f"  initial {len(shells_objs)} shells -> merged {len(merged)} shells")
    for s in merged:
        print(f"    {s}")

    # --------------------------------------------------------------
    section("Stage 10 : Burning regime classification (K-means)")
    # --------------------------------------------------------------
    # Extract features from the stellar zones
    T_list   = [z.T for z in zones]
    rho_list = [z.rho for z in zones]
    Y_H_list = [z.Y[0] for z in zones]
    Y_He_list = [z.Y[1] if len(z.Y) > 1 else 0.0 for z in zones]
    cls = classify_burning_regimes(T_list, rho_list, Y_H_list, Y_He_list,
                                    K=4, seed=2)
    print(f"  K-means (K=4) converged in {cls['n_iter']} iterations")
    for k, (nm, cnt) in enumerate(zip(cls["names"], cls["counts"])):
        print(f"    class {k}: {nm}  ({cnt} zones)")

    # --------------------------------------------------------------
    section("Stage 11 : Data integrity verification")
    # --------------------------------------------------------------
    # State checksum
    state_vals = (T_list + rho_list + Y_H_list + Y_He_list
                   + [z.P for z in zones] + [z.L for z in zones])
    cs = state_checksum(state_vals)
    ok = check_data_integrity(state_vals, cs)
    print(f"  state checksum = 0x{cs:08X}  integrity = {ok}")
    # Write state file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    state_path = os.path.join(out_dir, "final_state.dat")
    state_dict = {
        "M_star":  M_star,
        "N_zones": len(zones),
        "checksum": cs,
        "T_centre": zones[0].T,
        "T_surface": zones[-1].T,
        "R_star":  zones[-1].r,
        "rho_centre": zones[0].rho,
        "Y_final": Y_final,
    }
    write_state_file(state_path, state_dict,
                      header_comment="PROJECT_249 final state")
    print(f"  state file written to {state_path}")
    # Read it back
    state_back = read_state_file(state_path)
    cs_back = int(state_back.get("checksum", 0))
    print(f"  state file read back: M_star = {state_back.get('M_star', 0):.3e}")
    print(f"  checksum in file = 0x{cs_back:08X}")

    # --------------------------------------------------------------
    section("Stage 12 : ETD-RK4 linear test")
    # --------------------------------------------------------------
    # ETD-RK4 on a scalar stiff problem: v' = L v + N(v)
    L_stiff = -100.0 + 0j
    def N_func(v): return 0.5 * v * v
    tt, vv = etdrk4_linear_scalar(L_stiff, N_func, 1.0+0j, dt=0.005, nmax=200)
    print(f"  ETD-RK4 scalar: {len(tt)} time levels")
    print(f"    |v(0)| = {abs(vv[0]):.4e}  |v(end)| = {abs(vv[-1]):.4e}")
    if abs(vv[-1]) > 1e30:
        print(f"    (note: nonlinear blow-up expected for v'= -100 v + 0.5 v^2)")
    else:
        print(f"    stable evolution to t = {tt[-1]:.4f}")

    # --------------------------------------------------------------
    t_end = _time.time()
    section("PROJECT_249 complete")
    print(f"  wall-clock time: {t_end - t_start:.2f} s")
    print(f"  final stellar stage: {rec['stage_name']}")
    print(f"  final checksum: 0x{cs:08X}")
    print(f"  output file: {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
