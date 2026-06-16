"""
main.py - Unified entry point for PROJECT 243:
  r-process nucleosynthesis network simulation with high-order finite
  difference transport and stability analysis (small-scale reproducible).

Runs the full pipeline:
  1. Build the nuclide mesh and nuclear chart boundary
  2. Construct rate tables (Hauser-Feshbach + beta + fission)
  3. Interpolate rates with Chebyshev polynomials
  4. Build the ODE network and integrate (RK4 + implicit Euler)
  5. Analyze stability (eigenvalues, Cholesky, CFL, von Neumann)
  6. Compute neutron diffusion and opacity
  7. Fission fragment distribution (elastic-beam calving analogy)
  8. Sensitivity analysis with Latin hypercube sampling
  9. Optimise freeze-out conditions
  10. Print a comprehensive diagnostic report.

Zero arguments required.  Exit code 0 on success.
"""
from __future__ import annotations
import math
import time
import sys
import numpy as np

# ---------- Project modules ----------
from physical_constants import (
    K_BOLTZMANN, MEV_ERG, M_NEUTRON, M_U, C_LIGHT, TINY,
    T9_to_T, T_to_T9,
)
from nuclear_physics import (
    binding_energy, nuclear_mass, separation_energy,
    saha_factor, partition_function_simple,
    truncated_normal_mean, sample_rate_uncertainty,
)
from reaction_rates import (
    weisskopf_capture_rate, photodisintegration_rate,
    beta_decay_rate, fission_barrier,
    spontaneous_fission_rate, build_rate_table,
)
from nuclear_network import (
    build_network_index, initial_abundances,
    dYdt, rk4_step, integrate_network,
)
from adi_solver import peaceman_rachford_step, adi_evolve
from chebyshev_interp import (
    chebyshev_interpolate, chebyshev_eval, chebyshev_derivative_coeffs,
)
from newton_root import find_equilibrium_abundances
from stability_analysis import (
    cholesky_decomposition, eigenvalue_stiffness,
    von_neumann_stability, cfl_condition, build_jacobian_network,
)
from high_order_fd import fd_evolve, lax_wendroff_step
from latin_sampler import (
    latin_center, trinity_grid, sensitivity_samples,
)
from mesh_network import (
    build_nuclide_mesh, beta_stability_line, nuclear_chart_boundary,
    polygon_area, display_polygonal_boundary,
)
from distance_position import (
    nuclide_distance, ejecta_trajectory, distance_to_position,
    path_length, nearest_nuclide,
)
from control_optimizer import (
    gradient_descent, find_freeze_out_conditions, conjugate_gradient,
)
from diffusivity import (
    diffusion_coefficient, diffusion_timescale,
    knudsen_number, MSD_from_trajectory, opacity_from_diffusion,
)
from calving_physics import (
    fission_fragment_masses, fission_yield_distribution,
    bending_stiffness, mass_shedding_rate,
)


# ======================================================================
# Configuration: small-scale reproducible experiment
# ======================================================================
# Network spans A in [50, 260] and a band around the beta-stability line.
# 210 nuclides in total -- small enough to run quickly but large enough
#  to capture the main r-process flow from the seed (A ~ 90) to the
#  third peak (A ~ 195) and the actinides.
A_MIN, A_MAX = 70, 150
Z_OFFSET_LO, Z_OFFSET_HI = -3, 3        # narrow band for speed

T9_START = 3.0                            # 3 GK  (starting moderate, seeds intact)
T9_END = 0.3                              # 0.3 GK (freeze-out)
RHO_START = 1.0e9                         # g/cm^3
RHO_END = 1.0e3
T_END_S = 10.0                            # 10 seconds of evolution
N_TIME = 100                              # number of time steps

LATIN_N = 8                               # sensitivity sample count
CHEB_N = 12                               # Chebyshev nodes


def header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subheader(title: str) -> None:
    print()
    print(f"--- {title} ---")


# ======================================================================
# 1. Build nuclide mesh
# ======================================================================
def step_1_build_mesh():
    header("1. Building nuclide mesh and nuclear-chart boundary")
    mesh_raw = build_nuclide_mesh(A_MIN, A_MAX, Z_OFFSET_LO, Z_OFFSET_HI)
    # Drop weight; keep only (A, Z) for the network.
    nuclides = [(A, Z) for (A, Z, _w) in mesh_raw]
    print(f"  Nuclides in network : {len(nuclides)}")
    print(f"  Mass range          : A in [{A_MIN}, {A_MAX}]")

    boundary = nuclear_chart_boundary(A_MIN, A_MAX)
    area = polygon_area(boundary)
    print(f"  Chart boundary area : {area:.2f}  (A-Z units)")
    print(display_polygonal_boundary(boundary[:4] + [boundary[-1]]))

    # Trinity-style tiling: group nuclides into tiles for parallel blocks.
    words_per_tile = max(1, len(nuclides) // 8)
    tile_ids = [i // words_per_tile for i in range(len(nuclides))]
    n_tiles = max(tile_ids) + 1 if tile_ids else 0
    print(f"  Trinity tiles       : {n_tiles}  (words/tile = {words_per_tile})")
    return nuclides


# ======================================================================
# 2. Nuclear physics diagnostics
# ======================================================================
def step_2_nuclear_physics(nuclides):
    header("2. Nuclear physics sanity checks")
    # Pick a few representative nuclei
    reps = [(56, 26), (88, 38), (130, 50), (195, 78), (238, 92)]
    for A, Z in reps:
        B = binding_energy(A, Z)
        m = nuclear_mass(A, Z)
        Sn, Sp = separation_energy(A, Z)
        print(f"  ({A:>3d}, {Z:>3d})  B/A = {B/max(A,1):7.3f} MeV   "
              f"m = {m:8.4f} amu   "
              f"S_n = {Sn:6.2f}  S_p = {Sp:6.2f}  MeV")

    # Truncated-normal mean check
    mu_tn = truncated_normal_mean(1.0, 0.2, 0.1, 10.0)
    print(f"  Truncated-normal mean(mu=1, sigma=0.2, [0.1, 10]): {mu_tn:.4f}")


# ======================================================================
# 3. Build rate tables
# ======================================================================
def step_3_build_rates(nuclides, T9: float, rho: float, Ye: float):
    header(f"3. Building rate tables at T9 = {T9:.2f}, rho = {rho:.2e}, Y_e = {Ye:.3f}")
    Z_min = min(Z for _, Z in nuclides)
    Z_max = max(Z for _, Z in nuclides)
    rates = build_rate_table(A_MIN, A_MAX, Z_min, Z_max,
                             T9_to_T(T9), rho, Ye,
                             apply_uncertainty=False, seed_base=7)
    # Report a few rates
    reps = [(90, 36), (130, 50), (195, 78)]
    for A, Z in reps:
        print(f"  ({A:>3d},{Z:>3d})  <sigma v>_(n,gamma) = "
              f"{rates['n_cap'].get((A,Z),0.0):.3e}   "
              f"lambda_(gamma,n) = {rates['gamma_n'].get((A,Z),0.0):.3e}   "
              f"lambda_beta = {rates['beta'].get((A,Z),0.0):.3e}")
    return rates


# ======================================================================
# 4. Chebyshev interpolation of a rate vs T9
# ======================================================================
def step_4_chebyshev_rates(nuclides):
    header("4. Chebyshev interpolation of  <sigma v>(T9)  for (130, 50)")
    A, Z = 130, 50
    def rate_vs_T9(T9):
        return weisskopf_capture_rate(A, Z, T9_to_T(T9))
    coeffs, a, b = chebyshev_interpolate(rate_vs_T9, 0.5, 10.0, CHEB_N)
    dcoeffs = chebyshev_derivative_coeffs(coeffs)
    print(f"  Chebyshev order N = {CHEB_N}  on T9 in [{a}, {b}]")
    # Compare at test points
    T9_test = [1.0, 3.0, 5.0, 8.0]
    for T9t in T9_test:
        exact = rate_vs_T9(T9t)
        approx = chebyshev_eval(coeffs, a, b, T9t)
        derr = chebyshev_eval(dcoeffs, a, b, T9t)
        rel = abs(exact - approx) / max(abs(exact), TINY)
        print(f"    T9 = {T9t:4.1f}: exact = {exact:.3e}  approx = {approx:.3e}  "
              f"rel.err = {rel:.2e}  d/dT9 ~ {derr:.3e}")


# ======================================================================
# 5. ADI solver for 2D (T, rho) diffusion
# ======================================================================
def step_5_adi_diffusion():
    header("5. ADI diffusion on 2D (T, rho) grid  (Peaceman-Rachford)")
    Nx, Ny = 24, 24
    U0 = np.zeros((Nx, Ny))
    # Initial hot spot at center
    U0[Nx // 2, Ny // 2] = 1.0
    dx, dy = 1.0, 1.0
    alpha_x, alpha_y = 0.05, 0.05
    dt = 0.1
    nsteps = 20
    U_final = adi_evolve(U0, alpha_x, alpha_y, dx, dy, dt, nsteps)
    print(f"  Grid: {Nx} x {Ny}   steps: {nsteps}")
    print(f"  U_max initial = {U0.max():.4f}   final = {U_final.max():.4f}")
    print(f"  U_sum initial = {U0.sum():.4f}   final = {U_final.sum():.4f}")


# ======================================================================
# 6. 1D high-order FD transport of neutron pulse
# ======================================================================
def step_6_high_order_fd():
    header("6. High-order FD transport of a neutron pulse (Lax-Wendroff)")
    N = 81
    x = np.linspace(-1.0, 1.0, N)
    dx = x[1] - x[0]
    # Gaussian neutron pulse
    U0 = np.exp(-((x - 0.0) ** 2) / 0.05)
    u_adv = 0.5
    dt = 0.5 * dx / abs(u_adv)       # CFL = 0.5
    nsteps = 40
    U_lw = fd_evolve(U0, u_adv, dx, dt, nsteps, method="lax_wendroff", nu=1e-4)
    U_lw2 = fd_evolve(U0, u_adv, dx, dt, nsteps, method="lax_wendroff", nu=1e-2)
    stable = von_neumann_stability(dx, dt, u_adv, 1e-4)
    dt_max = cfl_condition(dx, u_adv)
    print(f"  N = {N}   dt = {dt:.4f}   dt_max(CFL) = {dt_max:.4f}")
    print(f"  Von Neumann stable   : {stable}")
    print(f"  Lax-Wendroff (nu=1e-4) max = {U_lw.max():.4f}   sum = {U_lw.sum():.4f}")
    print(f"  Lax-Wendroff (nu=1e-2) max = {U_lw2.max():.4f}   sum = {U_lw2.sum():.4f}")
    # L1 distance between the two solutions as a diffusion diagnostic
    L1 = np.sum(np.abs(U_lw - U_lw2)) * dx
    print(f"  L1 diffusion diagnostic: {L1:.4f}")


# ======================================================================
# 7. Build and integrate the network
# ======================================================================
def step_7_network_integration(nuclides):
    header("7. Full r-process network integration  (RK4)")
    idx = build_network_index(nuclides)
    n_nuclides = len(nuclides)
    print(f"  Number of nuclides  : {n_nuclides}")
    print(f"  Number of time steps: {N_TIME}")

    # Trajectories for T9, rho, n_n
    def T9_of_t(t):
        # Exponential cooling  T9(t) = T9_start * exp(-t / tau)  + T9_end
        tau = T_END_S / 5.0
        return T9_END + (T9_START - T9_END) * math.exp(-t / tau)

    def rho_of_t(t):
        # Power-law expansion: rho ~ (1 + t/t0)^-3
        t0 = 0.01
        return RHO_START * (1.0 + t / t0) ** -3

    def Ye_of_t(t):
        # Ye decreases slightly as neutrons are captured
        return max(0.05, 0.15 - 0.01 * t / T_END_S)

    def n_n_of_t(t):
        rho = rho_of_t(t)
        Ye = Ye_of_t(t)
        # Free neutron fraction ~ 1 - t/t_end (decreasing)
        X_n = max(0.0, 0.9 * (1.0 - t / T_END_S))
        # Scale neutron density to keep rates in a demonstrable range.
        # We reduce by 1e18 so that typical <sigma v> * n_n ~ 0.1 - 10 /s
        # (tractable for dt ~ 0.05 s while still capturing r-process flow).
        return X_n * rho * Ye / M_U * 1.0e-18

    def rates_of_t(t):
        T9 = T9_of_t(t)
        rho = rho_of_t(t)
        Ye = Ye_of_t(t)
        Z_min = min(Z for _, Z in nuclides)
        Z_max = max(Z for _, Z in nuclides)
        return build_rate_table(A_MIN, A_MAX, Z_min, Z_max,
                                T9_to_T(T9), rho, Ye,
                                apply_uncertainty=False, seed_base=1)

    # Initial condition: all mass in seed nucleus (A=90, Z=36) plus
    # a small neutron-like reservoir at the lightest A in the mesh.
    Y0 = initial_abundances(n_nuclides, Y_seed=0.0)
    seed_set = False
    for k, (A, Z) in enumerate(nuclides):
        if A == 90 and Z == 36 and not seed_set:
            Y0[k] = 0.999
            seed_set = True
        elif A == A_MIN:
            # Spread a tiny neutron-like reservoir on the lightest nucleus
            n_light = max(1, len([1 for (Aa, Zz) in nuclides if Aa == A_MIN]))
            Y0[k] = 0.001 / n_light

    print(f"  Initial total Y     : {Y0.sum():.4f}")
    print(f"  T9 start/end        : {T9_START:.2f} -> {T9_END:.2f}")
    print(f"  rho start/end       : {RHO_START:.2e} -> {RHO_END:.2e}")

    t_start = time.time()
    times, Y_hist = integrate_network(
        Y0, (0.0, T_END_S), N_TIME, nuclides, idx,
        rates_of_t, n_n_of_t, T9_of_t, method="rk4",
    )
    t_elapsed = time.time() - t_start

    # Diagnostics
    Y_final = Y_hist[-1]
    print(f"  Integration time    : {t_elapsed:.3f} s")
    print(f"  Final total Y       : {Y_final.sum():.4f}")
    print(f"  Final Y_max         : {Y_final.max():.4e}")
    # Find peak abundance nuclide
    k_peak = int(np.argmax(Y_final))
    A_peak, Z_peak = nuclides[k_peak]
    print(f"  Peak abundance at   : (A, Z) = ({A_peak}, {Z_peak})  "
          f"Y = {Y_final[k_peak]:.4e}")

    # Path length in nuclide space: from seed to peak
    seed_AZ = (90, 36)
    peak_AZ = (A_peak, Z_peak)
    d = nuclide_distance(seed_AZ[0], seed_AZ[1], peak_AZ[0], peak_AZ[1])
    print(f"  Distance seed->peak : {d:.3f}  (nuclide units)")

    return times, Y_hist, T9_of_t, rho_of_t, n_n_of_t


# ======================================================================
# 8. Stability analysis
# ======================================================================
def step_8_stability(nuclides, rates):
    header("8. Stability analysis (eigenvalues, Cholesky, CFL)")
    idx = build_network_index(nuclides)
    n = len(nuclides)
    # Build a small Jacobian for a subset (for speed)
    n_sub = min(20, n)
    Y0 = np.zeros(n_sub)
    Y0[0] = 1.0
    nuclides_sub = nuclides[:n_sub]
    idx_sub = build_network_index(nuclides_sub)
    # Build a reduced rate table
    rates_sub = {k: {} for k in rates}
    for k in rates:
        for key, val in rates[k].items():
            if key in idx_sub:
                rates_sub[k][key] = val
    def F_sub(y):
        dY = np.zeros(n_sub)
        n_n = 1.0e28
        T9 = 5.0
        for k_, (A, Z) in enumerate(nuclides_sub):
            Yk = max(y[k_], 0.0)
            r_n = rates_sub["n_cap"].get((A, Z), 0.0)
            r_g = rates_sub["gamma_n"].get((A, Z), 0.0)
            r_b = rates_sub["beta"].get((A, Z), 0.0)
            loss = (r_n * n_n + r_g + r_b) * Yk
            dY[k_] -= loss
        return dY
    J = build_jacobian_network(F_sub, Y0)
    lam_max, lam_min, stiff, unstable = eigenvalue_stiffness(J)
    print(f"  Sub-network size    : {n_sub}")
    print(f"  lambda_max(Re)      : {lam_max:.3e}")
    print(f"  lambda_min(|Re|)    : {lam_min:.3e}")
    print(f"  Stiff               : {stiff}")
    print(f"  Unstable            : {unstable}")

    # Cholesky on a small positive-definite matrix  C = J^T J + eps I
    C = J.T @ J + 1e-6 * np.eye(n_sub)
    try:
        L = cholesky_decomposition(C)
        print(f"  Cholesky factor L   : computed,  ||L||_F = {np.linalg.norm(L):.3e}")
    except ValueError as e:
        print(f"  Cholesky failed     : {e}")


# ======================================================================
# 9. Fission fragment distribution
# ======================================================================
def step_9_fission():
    header("9. Fission fragment distribution  (elastic-beam calving analogy)")
    for A, Z in [(236, 92), (252, 98), (260, 104)]:
        A_L, A_H = fission_fragment_masses(A, Z)
        B = bending_stiffness(A)
        print(f"  Parent ({A},{Z})  -> fragments  A_L = {A_L:>3d}  A_H = {A_H:>3d}"
              f"   B = {B:.3e}")
        A_grid, Y = fission_yield_distribution(A, Z, n_points=15)
        peak_idx = int(np.argmax(Y))
        print(f"    Yield peak at A = {A_grid[peak_idx]:.0f}  (Y = {Y[peak_idx]:.3f})")

    # Mass-shedding rate
    print(f"  mass_shedding_rate(240, 94, E*=5 MeV) = "
          f"{mass_shedding_rate(240, 94, 5.0):.3e} 1/s")


# ======================================================================
# 10. Diffusivity and opacity
# ======================================================================
def step_10_diffusivity():
    header("10. Neutron diffusion and opacity in ejecta")
    for T9 in [1.0, 3.0, 5.0, 9.0]:
        for rho in [1e5, 1e8]:
            D = diffusion_coefficient(rho, 0.1, T9, A_typical=130)
            t_diff = diffusion_timescale(1e9, D)
            kn = knudsen_number(1e-10, 1e9)
            kappa = opacity_from_diffusion(D, rho)
            print(f"  T9 = {T9:.1f}  rho = {rho:.0e}  D = {D:.2e}  "
                  f"t_diff = {t_diff:.2e}  Kn = {kn:.2e}  kappa = {kappa:.2e}")

    # Ejecta trajectory
    r, rho_ej, T9_ej = ejecta_trajectory(t=1.0, v_ej=0.1, M_ej=0.05)
    print(f"  Ejecta at t = 1 s : r = {r:.2e} cm  rho = {rho_ej:.2e}  T9 = {T9_ej:.2f}")


# ======================================================================
# 11. Latin hypercube sensitivity
# ======================================================================
def step_11_sensitivity():
    header("11. Latin hypercube sensitivity on r-process parameters")
    param_names = ["v_ej/c", "M_ej/M_sun", "Y_e", "T9_start"]
    param_ranges = [(0.05, 0.3), (0.01, 0.1), (0.05, 0.25), (5.0, 12.0)]
    samples = sensitivity_samples(param_names, param_ranges, LATIN_N, seed=123)
    print(f"  Generated {LATIN_N} samples in {len(param_names)} dimensions")
    print(f"  {'v_ej/c':>8s}  {'M_ej':>8s}  {'Y_e':>8s}  {'T9':>6s}")
    for row in samples:
        print(f"  {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}  {row[3]:6.2f}")


# ======================================================================
# 12. Newton equilibrium search
# ======================================================================
def step_12_equilibrium():
    header("12. Newton-Maehly search for equilibrium abundance")
    n = 5
    def F(Y):
        # Toy equilibrium:  Y_i = 1/n  for all i
        return Y - np.ones(n) / n
    def J(Y):
        return np.eye(n)
    Y0 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    Y_eq, n_iter, res = find_equilibrium_abundances(Y0, F, J, tol=1e-10)
    print(f"  Newton iterations : {n_iter}")
    print(f"  Final residual    : {res:.2e}")
    print(f"  Y_eq = {Y_eq}")


# ======================================================================
# 13. Freeze-out optimisation
# ======================================================================
def step_13_freeze_out():
    header("13. Freeze-out condition optimisation")
    def T9_func(T9):
        return T9
    def rho_func(T9):
        return 1.0e6 * T9 ** 3
    def Y_eq_func(T9, rho):
        # Toy residual:  want Y_eq ~ Saha at T9
        return np.array([saha_factor(130, 50, T9_to_T(T9), 1e-3, 1e-3) - 1e-10])
    T9_f, rho_f, res_f = find_freeze_out_conditions(T9_func, rho_func, Y_eq_func,
                                                    T9_range=(0.3, 5.0))
    print(f"  Freeze-out T9     : {T9_f:.3f}")
    print(f"  Freeze-out rho    : {rho_f:.3e}")
    print(f"  Residual          : {res_f:.3e}")


# ======================================================================
# 14. Distance/position and final summary
# ======================================================================
def step_14_summary(times, Y_hist, nuclides):
    header("14. Final summary and distance metrics")
    Y_final = Y_hist[-1]
    k_peak = int(np.argmax(Y_final))
    A_peak, Z_peak = nuclides[k_peak]
    print(f"  Time span           : {times[0]:.2f} -> {times[-1]:.2f} s")
    print(f"  Peak abundance      : (A, Z) = ({A_peak}, {Z_peak})  "
          f"Y = {Y_final[k_peak]:.4e}")
    # Distance from seed to peak
    seed_AZ = (90, 36)
    d = nuclide_distance(seed_AZ[0], seed_AZ[1], A_peak, Z_peak)
    print(f"  Seed -> peak distance: {d:.3f}")
    # Nearest nuclide to A=195, Z=78 (third r-process peak)
    target = nearest_nuclide(195, 78, nuclides)
    print(f"  Nearest nuclide to (195, 78): {target}")
    # Lookback time from 1 Mpc
    t_look, z = distance_to_position(1.0)
    print(f"  1 Mpc -> z = {z:.4f}  t_look = {t_look:.3e} s")

    # Final path length of abundance flow (trace through top-abundance nuclides at each time)
    path = []
    step = max(1, len(times) // 20)
    for i in range(0, len(times), step):
        k = int(np.argmax(Y_hist[i]))
        path.append(nuclides[k])
    L = path_length([(float(A), float(Z)) for A, Z in path])
    print(f"  Abundance-flow path : length = {L:.3f}")


# ======================================================================
# Main entry
# ======================================================================
def main() -> int:
    print()
    print("#" * 72)
    print("#" + " " * 70 + "#")
    print("#    PROJECT 243 - r-process nucleosynthesis network simulation  #")
    print("#    High-order finite difference transport + stability analysis #")
    print("#    (Small-scale reproducible experiment)                       #")
    print("#" + " " * 70 + "#")
    print("#" * 72)

    t_total_start = time.time()

    # Step 1
    nuclides = step_1_build_mesh()
    # Step 2
    step_2_nuclear_physics(nuclides)
    # Step 3
    rates = step_3_build_rates(nuclides, T9=5.0, rho=1e7, Ye=0.1)
    # Step 4
    step_4_chebyshev_rates(nuclides)
    # Step 5
    step_5_adi_diffusion()
    # Step 6
    step_6_high_order_fd()
    # Step 7
    times, Y_hist, T9_of_t, rho_of_t, n_n_of_t = step_7_network_integration(nuclides)
    # Step 8
    step_8_stability(nuclides, rates)
    # Step 9
    step_9_fission()
    # Step 10
    step_10_diffusivity()
    # Step 11
    step_11_sensitivity()
    # Step 12
    step_12_equilibrium()
    # Step 13
    step_13_freeze_out()
    # Step 14
    step_14_summary(times, Y_hist, nuclides)

    t_total = time.time() - t_total_start
    print()
    print("*" * 72)
    print(f"  PROJECT 243 COMPLETED SUCCESSFULLY  --  total wall time: {t_total:.2f} s")
    print("*" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
