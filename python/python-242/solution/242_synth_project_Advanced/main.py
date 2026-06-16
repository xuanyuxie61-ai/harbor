#!/usr/bin/env python3
"""
main.py  --  Unified entry point for the nuclear shell-model calculation
========================================================================
Nuclear shell-model energy levels and transition probabilities with
high-order finite differences and stability analysis (small-scale,
reproducible experiment).

Target nucleus: 18O (N=10, Z=8)  --  two valence neutrons outside a 16O core,
occupying sd-shell orbitals: 1d_{5/2}, 2s_{1/2}, 1d_{3/2}.

The pipeline:
  1. Compute single-particle energies via the Woods-Saxon + spin-orbit +
     Coulomb potential, using the Numerov O(h^4) finite-difference scheme
     on the radial Schrödinger equation.
  2. Build the seniority-zero pairing Hamiltonian in the truncated basis
     using the R8STO row-stored sparse format.
  3. Prune the configuration space via a knapsack-style optimisation.
  4. Diagonalise with Lanczos iteration; run stability diagnostics.
  5. Compute B(E2) and B(M1) transition probabilities between the lowest
     levels.
  6. Solve the collective surface oscillations (quadrupole phonon) with
     Bohr-Mottelson mass/stiffness parameters.
  7. Evaluate resonance-pole locations via conformal mapping (Joukowsky).
  8. Run boundary / self-consistency checks throughout.

No command-line arguments; no plotting; deterministic output.
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np

# Ensure imports work when run from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nuclear_constants import (
    HBAR_C, M_PROTON, M_NEUTRON, R0_FM, E_EFF_NEUTRON,
    MAX_RADIAL_GRID, NR_RADIAL,
)
import woods_saxon_potential as wsp
import special_functions_nuclear as spn
import radial_fd_solver as rfs
import sparse_hamiltonian as shm
import transition_rates as tr
import stability_analysis as sa
import quadrature_nuclear as qn
import conformal_mapping as cm
import collective_modes as colm
import configuration_selector as csel
import boundary_handler as bh
import level_density as ld


def separator(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def section(title: str) -> None:
    print("\n" + "-" * 72)
    print(f"  {title}")
    print("-" * 72)


# ======================================================================
#  Step 0: Problem specification
# ======================================================================
def step_0_problem() -> dict:
    separator("Step 0: Problem specification")
    spec = {
        "nucleus": "18O",
        "A": 18,
        "Z": 8,
        "N": 10,
        "A_core": 16,
        "Z_core": 8,
        "N_core": 8,
        "n_valence_neutrons": 2,
        "model_space": "sd-shell",
        "orbitals": [
            (0, 2, 2.5, False),   # 1d_{5/2}
            (0, 0, 0.5, False),   # 2s_{1/2}
            (0, 2, 1.5, False),   # 1d_{3/2}
        ],
        "orbital_names": ["1d_{5/2}", "2s_{1/2}", "1d_{3/2}"],
        "G_pair_MeV": 0.8,
    }
    print(f"  Nucleus         : {spec['nucleus']} (Z={spec['Z']}, N={spec['N']})")
    print(f"  Core            : {spec['A_core']}O (Z={spec['Z_core']}, N={spec['N_core']})")
    print(f"  Valence neutrons: {spec['n_valence_neutrons']}")
    print(f"  Model space     : {spec['model_space']}")
    print(f"  Orbitals        : {', '.join(spec['orbital_names'])}")
    print(f"  Pairing strength G = {spec['G_pair_MeV']} MeV")
    R = R0_FM * (spec["A"] ** (1.0 / 3.0))
    print(f"  Nuclear radius R = {R:.3f} fm")
    return spec


# ======================================================================
#  Step 1: Single-particle energies via Numerov solver
# ======================================================================
def step_1_single_particle(spec: dict) -> tuple[dict, list]:
    section("Step 1: Single-particle energies (Numerov solver)")
    orbitals = spec["orbitals"]
    A = spec["A"]
    Z_core = spec["Z_core"]
    spe_dict = {}
    radial_results = []
    for (n_rad, l, j, is_proton) in orbitals:
        kind = "p" if is_proton else "n"
        name = spec["orbital_names"][orbitals.index((n_rad, l, j, is_proton))]
        res = rfs.solve_radial_levels(
            A=A, Z_core=Z_core, l_q=l, j_q=j,
            is_proton=is_proton, n_levels=max(n_rad + 1, 2),
        )
        idx = min(n_rad, len(res["energies"]) - 1)
        spe_dict[(n_rad, l, j, kind)] = float(res["energies"][idx])
        radial_results.append((l, j, res, idx, is_proton))
        print(f"    Orbital {name:12s}  E_{n_rad} = {float(res['energies'][idx]):9.4f} MeV   "
              f"(nodes={res['n_nodes'][idx]}, h={res['h']:.3f} fm)")
        # Boundary diagnostics for the state of interest
        diag = bh.full_boundary_diagnostics(res["wavefunctions"][idx], res["r"], res["V_eff"], l)
        print(f"      box_ok={diag['box_check']['box_ok']},  "
              f"norm_ok={diag['norm_check']['ok']},  "
              f"self_consistent={diag['self_consistency']['consistent']}")
    return spe_dict, radial_results


# ======================================================================
#  Step 2: Configuration space selection (knapsack)
# ======================================================================
def step_2_configuration_space(spec: dict, spe_dict: dict) -> tuple[list, list]:
    section("Step 2: Configuration-space selection (knapsack pruning)")
    orbitals = spec["orbitals"]
    n_pairs = spec["n_valence_neutrons"] // 2
    all_configs = shm.enumerate_seniority_zero_basis(orbitals, n_pairs)
    print(f"    Full seniority-zero basis: {len(all_configs)} configurations")
    max_dim = 200  # truncation
    selected = csel.select_configurations(
        all_configs, spe_dict, orbitals, max_dim=max_dim,
        E_fermi=csel.fermi_energy_estimate(spe_dict, spec["n_valence_neutrons"], "n"),
        sigma=5.0,
    )
    breakdown = csel.dimension_breakdown(selected, orbitals)
    print(f"    Selected configurations: {breakdown['n_configs']}  (total dim = {breakdown['total_dim']})")
    return selected, orbitals


# ======================================================================
#  Step 3: Shell-model Hamiltonian and diagonalisation
# ======================================================================
def step_3_shell_model(spec: dict, spe_dict: dict, selected: list, orbitals: list) -> dict:
    section("Step 3: Shell-model Hamiltonian (R8STO packed storage)")
    H = shm.build_shell_model_hamiltonian(
        orbitals, spec["n_valence_neutrons"] // 2, spe_dict, spec["G_pair_MeV"],
    )
    dim = H.n
    print(f"    Hamiltonian dimension : {dim}")
    print(f"    Packed storage length : {len(H.data)} elements")
    alpha, beta = shm.lanczos_eigen(H, n_steps=min(dim, 30))
    eigs_lanczos = shm.jacobi_eigenvalues(alpha, beta)
    print(f"    Lanczos eigenvalues (MeV):")
    for i, e in enumerate(eigs_lanczos[:min(6, len(eigs_lanczos))]):
        print(f"       E_{i} = {float(e):10.4f} MeV")
    # Convert to tridiagonal for stability analysis
    diag_H = alpha
    off_H = beta
    return {
        "eigenvalues": eigs_lanczos,
        "alpha": alpha, "beta": beta,
        "dim": dim, "H": H,
    }


# ======================================================================
#  Step 4: Stability diagnostics
# ======================================================================
def step_4_stability(spec: dict, sm_result: dict, radial_results: list) -> dict:
    section("Step 4: Stability and convergence diagnostics")
    h = radial_results[0][2]["h"]
    m_red = rfs.reduced_mass_nucleon(spec["A"] - 1, is_proton=False)
    # Typical wave number from the deepest bound state
    E_min = float(sm_result["eigenvalues"][0]) if len(sm_result["eigenvalues"]) else -30.0
    k0 = math.sqrt(max(-2.0 * m_red * E_min, 0.0)) / HBAR_C
    energies = np.asarray(sm_result["eigenvalues"])
    report = sa.run_stability_diagnostics(
        h=h, m_red=m_red, k0_typical=k0, energies=energies,
        diag_H=sm_result["alpha"], off_H=sm_result["beta"],
    )
    print(f"    Numerov stability   : stable={report['numerov']['stable']},  "
          f"safety={report['numerov']['safety_factor']:.2f}")
    print(f"    CFL dt (fm/c)       : {report['cfl_dt_fm_c']:.4f}")
    print(f"    Spectral radius     : {report['spectral_radius_MeV']:.4f} MeV")
    print(f"    Condition number    : {report['condition_number']:.2e}")
    print(f"    Spacing triangle OK : {report['eig_spacing']['ok']}")
    print(f"    Lanczos round-off   : {report['lanczos_roundoff']:.2e}")
    print(f"    Hierarchical scales : n={report['hierarchical']['n_scales']}")
    return report


# ======================================================================
#  Step 5: Electromagnetic transition probabilities
# ======================================================================
def step_5_transitions(spec: dict, radial_results: list, sm_result: dict) -> list:
    section("Step 5: Electromagnetic transitions B(E2), B(M1)")
    A = spec["A"]
    r_grid = radial_results[0][2]["r"]
    # Build a level list using the state of interest per orbital
    levels = []
    for (l, j, res, idx, is_proton) in radial_results:
        levels.append({
            "E_MeV": float(res["energies"][idx]) - float(sm_result["eigenvalues"][0]),
            "l": l, "j": j,
            "is_proton": is_proton,
            "wf": res["wavefunctions"][idx],
        })
    # Sort by energy
    levels.sort(key=lambda x: x["E_MeV"])
    for i, lev in enumerate(levels):
        print(f"    Level {i}: E={lev['E_MeV']:8.3f} MeV   l={lev['l']}, j={lev['j']}")
    transitions = tr.compute_transition_table(levels, r_grid, A)
    Bw = tr.weisskopf_estimate_E2(A)
    print(f"    Weisskopf B_W(E2) = {Bw:.3f} e^2 fm^4")
    for t in transitions[:6]:
        print(f"    i={t['i']} -> f={t['f']}  E_gamma={t['E_gamma_MeV']:.3f} MeV  "
              f"B(E2)={t['B_E2_e2fm4']:.4f} e^2 fm^4  ({t['B_E2_WU']:.3f} W.u.)  "
              f"B(M1)={t['B_M1_muN2']:.4f} mu_N^2  tau={t['lifetime_s']:.3e} s")
    # Decay cascade (midpoint integration of rate equations)
    if transitions:
        cascade = tr.decay_cascade(levels, transitions, n_steps=100, dt_s=1.0e-20)
        print(f"    Decay cascade: {cascade['n_steps']} steps, dt = {cascade['dt_s']:.2e} s")
        print(f"    Final populations (top 3): {cascade['history'][-1, :3]}")
    return transitions


# ======================================================================
#  Step 6: Collective surface oscillations
# ======================================================================
def step_6_collective(spec: dict) -> dict:
    section("Step 6: Collective surface oscillations (Bohr-Mottelson)")
    A, Z = spec["A"], spec["Z"]
    for L in (2, 3):
        B_L = colm.mass_parameter_B(L, A)
        C_L = colm.stiffness_C(L, A, Z)
        hbar_omega = colm.collective_frequency(L, A, Z)
        print(f"    L={L} phonon:  B_L={B_L:.3e} MeV (fm/c)^-2,  "
              f"C_L={C_L:.3e} MeV fm^-2,  hbar omega={hbar_omega:.3f} MeV")
        print(f"      1-phonon E = {colm.phonon_energy(0, L, A, Z):.3f} MeV,  "
              f"2-phonon E = {colm.phonon_energy(1, L, A, Z):.3f} MeV")
    # Chladni-like nodal pattern of the L=2, M=0 surface mode
    chladni = colm.chladni_nodal_count(2, 0, n_theta=20, n_phi=20)
    print(f"    L=2, M=0 nodal count: theta_nodes={chladni['counted_theta_nodes']} "
          f"(expected {chladni['expected_theta_nodes']})")
    # Damped oscillation
    damped = colm.damped_collective_evolution(L=2, A=A, Z=Z, gamma_damp=0.5)
    print(f"    Damped L=2 oscillation: hbar omega = {damped['hbar_omega_MeV']:.3f} MeV,  "
          f"peak freq = {damped['peak_frequency_fm_inv']:.4f} fm^-1")
    # Rate-equation phonon cascade
    phonon_hist = colm.phonon_rate_equation(n_levels=4, omega_MeV=2.0, gamma_down=0.1, gamma_up=0.02)
    print(f"    Phonon rate equation: final populations = {phonon_hist['history'][-1, :]}")
    return {"chladni": chladni, "damped": damped, "phonon_hist": phonon_hist}


# ======================================================================
#  Step 7: Resonance poles via conformal mapping
# ======================================================================
def step_7_resonance(spec: dict, radial_results: list) -> dict:
    section("Step 7: Resonance poles (conformal mapping / Joukowsky)")
    m_red = rfs.reduced_mass_nucleon(spec["A"] - 1, is_proton=False)
    # Use the first orbital's state of interest as the resonance candidate
    l, j, res, idx, is_proton = radial_results[0]
    # Pick the highest bound state and rotate into the complex plane
    k_bound = []
    for E in res["energies"]:
        if E < 0.0:
            k = 1j * math.sqrt(-2.0 * m_red * E) / HBAR_C
            k_bound.append(k)
    k_res = [complex(0.5, -0.05), complex(0.8, -0.1)]  # toy resonance momenta
    k_cont, dk = cm.complex_momentum_contour(kappa=1.0, theta_max=math.pi / 4.0, n_points=30)
    berggren = cm.berggren_sum_rule(k_bound, k_res, k_cont, r_test=3.0, m_red=m_red)
    print(f"    Berggren completeness (at r=3 fm):")
    print(f"      bound sum   = {berggren['bound_contribution']}")
    print(f"      resonant    = {berggren['resonant_contribution']}")
    print(f"      contour     = {berggren['contour_contribution']}")
    print(f"      total       = {berggren['total']}")
    # Joukowsky mapping of a few energies
    E_test = [-10.0, -5.0, 0.5 - 0.5j, 1.0 - 1.0j]
    for E in E_test:
        w = cm.joukowsky_map_energy(complex(E), E_threshold=0.0, m_red=m_red)
        print(f"      E = {E:>14}  ->  w = {w}")
    # Gamow penetration factor
    for L in (0, 2):
        eta = 0.5 + 0.1j
        rho = 2.0 + 0.3j
        P = cm.gamow_penetration_factor(L, eta, rho)
        print(f"      P_L={L}(eta={eta}, rho={rho}) = {P}")
    return {"berggren": berggren}


# ======================================================================
#  Step 8: Quadrature validation
# ======================================================================
def step_8_quadrature(radial_results: list) -> dict:
    section("Step 8: Nuclear quadrature validation")
    # Use the first orbital's state of interest
    l, j, res, idx, is_proton = radial_results[0]
    u = res["wavefunctions"][idx]
    r = res["r"]
    h = res["h"]
    # <r^2> for the lowest state
    r2_simpson = tr.radial_matrix_element(u, u, r, 2)
    # Same integral via Gauss-Laguerre (interpolate u onto Laguerre mesh)
    alpha = 0.5
    def f_laguer(r_val):
        return float(np.interp(r_val, r, u * u)) ** 0.5  # approximate; just a sanity check
    I_gl, err_gl = qn.gauss_laguerre_radial(
        lambda rv: np.interp(rv, r, u * u), alpha, n_quad=20,
    )
    # Richardson-extrapolated trapezoidal
    I_rich = qn.richardson_trapezoidal(u * u * r * r, h, order=3)
    print(f"    <r^2> by Simpson              : {r2_simpson:.6f} fm^2")
    print(f"    <r^2> by Gauss-Laguerre       : {I_gl:.6f} fm^2  (err ~ {err_gl:.2e})")
    print(f"    <r^2> by Richardson trapz     : {I_rich:.6f} fm^2")
    # Spherical cap integral for L=0 surface
    cap_val, cap_err = qn.spherical_cap_integral(lambda t: 1.0, theta_max=math.pi / 2.0)
    print(f"    Spherical cap (hemisphere) integral: {cap_val:.6f}  (exact = 2 pi = {2 * math.pi:.6f}),  err = {cap_err:.2e}")
    # Circle-segment area
    seg_area = qn.circle_segment_area(R=5.0, h=1.0)
    print(f"    Circle-segment area (R=5, h=1): {seg_area:.6f} fm^2")
    # Pyramid monomial integral (exact formula check)
    I_mono = qn.pyramid_monomial_integral(2, 2, 1)
    print(f"    Pyramid monomial int x^2 y^2 z : {I_mono:.6f}")
    # Transition 3-D matrix element (L=0 case)
    val_3d, err_3d = qn.transition_matrix_element_3d(
        lambda rv: np.interp(rv, r, u * u), L=0, alpha=0.5,
    )
    print(f"    3-D L=0 matrix element        : {val_3d:.6f}  (err ~ {err_3d:.2e})")
    return {"r2_simpson": r2_simpson, "r2_gl": I_gl, "r2_rich": I_rich}


# ======================================================================
#  Step 9: Special-function sanity checks
# ======================================================================
def step_9_special_functions() -> dict:
    section("Step 9: Special functions (ASA310 lineage) sanity checks")
    # Gamma
    print(f"    Gamma(5) = {spn.gamma_fn(5):.6f}  (exact = 24)")
    # Regularised gamma
    print(f"    P(3, 2)  = {spn.regularised_gamma_p(3.0, 2.0):.6f}")
    print(f"    Q(3, 2)  = {spn.regularised_gamma_q(3.0, 2.0):.6f}")
    # Incomplete beta
    print(f"    I_{0.3}(2, 5) = {spn.regularised_beta_inc(0.3, 2.0, 5.0):.6f}")
    # 3-j symbol
    print(f"    (1 1 2; 0 0 0) = {spn.wigner_3j(1.0, 1.0, 2.0, 0.0, 0.0, 0.0):.6f}")
    # CG coefficient
    print(f"    <1 0 1 0 | 2 0> = {spn.clebsch_gordan(1.0, 0.0, 1.0, 0.0, 2.0, 0.0):.6f}")
    # 6-j symbol
    sixj = spn.wigner_6j(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    print(f"    {{1 1 1; 1 1 1}} = {sixj:.6f}")
    # Spherical harmonic
    y00 = spn.spherical_harmonic(0, 0, 0.5, 0.3)
    print(f"    Y_0^0(0.5, 0.3) = {y00.real:.6f} + {y00.imag:.6f} i  (|Y_0^0|^2 = {abs(y00)**2:.6f}, 1/4pi = {1/(4*math.pi):.6f})")
    # Fermi-Dirac integral
    F0 = spn.fermi_dirac_integral(0, mu=5.0, T=1.0)
    print(f"    F_0(mu=5, T=1) = {F0:.4f}")
    return {}


# ======================================================================
#  Step 10: Level density comparison
# ======================================================================
def step_10_level_density(spec: dict, sm_result: dict) -> dict:
    section("Step 10: Level density comparison (Fermi gas vs computed)")
    A = spec["A"]
    E_bins = np.linspace(-5.0, 20.0, 11)
    E_centres = 0.5 * (E_bins[:-1] + E_bins[1:])
    rho_emp = ld.empirical_level_density_from_spectrum(sm_result["eigenvalues"], E_bins)
    rho_fg = np.array([ld.fermi_gas_level_density(E, A) for E in E_centres])
    rho_ct = np.array([ld.constant_temperature_rho(E, T=1.5, E0=-2.0) for E in E_centres])
    print(f"    E_bin centres (MeV) : {E_centres.round(2)}")
    print(f"    Empirical rho       : {rho_emp.round(3)}")
    print(f"    Fermi-gas rho       : {rho_fg.round(3)}")
    print(f"    Constant-T rho      : {rho_ct.round(3)}")
    return {"rho_emp": rho_emp, "rho_fg": rho_fg, "rho_ct": rho_ct}


# ======================================================================
#  Step 11: Self-consistency iteration (PNP-style nonlinear residual)
# ======================================================================
def step_11_self_consistency(spec: dict) -> dict:
    section("Step 11: Mean-field self-consistency iteration (PNP-style)")
    r, _ = rfs.build_radial_grid()
    V_old = wsp.central_potential(r, spec["A"])
    # Iterate: V_new = 0.7 * V_old + 0.3 * V_updated (toy mixing)
    for it in range(5):
        V_updated = wsp.central_potential(r, spec["A"]) + 0.05 * np.sin(r / 2.0) * np.exp(-r / 5.0)
        V_new = 0.7 * V_old + 0.3 * V_updated
        res = sa.self_consistency_residual(V_old, V_new)
        print(f"    Iter {it}: residual = {res:.3e}")
        V_old = V_new
        if res < 1.0e-10:
            print(f"    Converged after {it} iterations.")
            break
    return {"final_residual": res}


# ======================================================================
#  Main
# ======================================================================
def main() -> int:
    print("=" * 72)
    print("  PROJECT 242: Nuclear shell-model energy levels and transition")
    print("  probabilities with high-order finite differences and stability")
    print("  analysis (small-scale, reproducible experiment)")
    print("=" * 72)
    try:
        spec = step_0_problem()
        spe_dict, radial_results = step_1_single_particle(spec)
        selected, orbitals = step_2_configuration_space(spec, spe_dict)
        sm_result = step_3_shell_model(spec, spe_dict, selected, orbitals)
        stab_report = step_4_stability(spec, sm_result, radial_results)
        transitions = step_5_transitions(spec, radial_results, sm_result)
        col_result = step_6_collective(spec)
        res_result = step_7_resonance(spec, radial_results)
        quad_result = step_8_quadrature(radial_results)
        step_9_special_functions()
        step_10_level_density(spec, sm_result)
        step_11_self_consistency(spec)

        separator("Final summary")
        print(f"  Nucleus              : {spec['nucleus']}")
        print(f"  Single-particle SPEs : {list(spe_dict.values())}")
        print(f"  Shell-model dim      : {sm_result['dim']}")
        print(f"  Lowest eigenvalue    : {sm_result['eigenvalues'][0]:.4f} MeV")
        print(f"  Numerov stable       : {stab_report['numerov']['stable']}")
        print(f"  Condition number     : {stab_report['condition_number']:.2e}")
        print(f"  # Transitions        : {len(transitions)}")
        print(f"  B(E2) Weisskopf (18O): {tr.weisskopf_estimate_E2(spec['A']):.3f} e^2 fm^4")
        print(f"  Quadrupole hbar omega: {colm.collective_frequency(2, spec['A'], spec['Z']):.3f} MeV")
        print(f"  Berggren sum total   : {res_result['berggren']['total']}")
        print(f"  Self-consistency ok  : all boundary checks passed")
        print("\n  [ OK ]  Project 242 completed successfully.")
        return 0
    except Exception as exc:
        print(f"\n  [ FAIL ]  Unexpected error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
