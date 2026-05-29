"""
damage_repair.py
DNA damage detection and repair kinetics orchestration module.

This module integrates all sub-models (electrostatics, diffusion,
reaction-diffusion, potential surfaces, Markov models) into a
unified framework for simulating the full DNA damage repair cascade:

  1. DSB induction and chromatin remodeling
  2. Electrostatic recruitment of sensor proteins (e.g., MRN complex)
  3. 1D sliding and 3D diffusion search for damage sites
  4. Focus formation via reaction-diffusion of repair factors
  5. Repair completion and dissociation

Key biophysical parameters:
  - DNA persistence length: l_p ~ 50 nm (~150 bp)
  - 1D sliding diffusion: D_1D ~ 10^5 - 10^7 bp^2/s
  - 3D diffusion: D_3D ~ 1-10 um^2/s for small proteins
  - Debye length at physiological ionic strength (~150 mM): ~0.8 nm
  - Typical DSB repair time: minutes to hours
"""

import numpy as np
from electrostatics import (
    debye_length,
    solve_nonlinear_pb,
    setup_dna_charge_density,
    electrostatic_free_energy,
)
from protein_diffusion import (
    smoluchowski_1d_sliding_step,
    porous_medium_step_1d,
    sliding_search_time,
)
from reaction_diffusion import simulate_repair_focus, estimate_repair_time
from potential_surface import build_dna_repair_energy_surface, compute_activation_barrier
from markov_model import build_msm_transition_matrix, msm_mfpt
from tet_mesh_core import generate_tet_mesh_box, integrate_over_tet_mesh
from matrix_utils import banded_solve_tridiagonal


def simulate_dsb_repair_cascade(
    n_grid=32,
    dna_length_bp=1000,
    ionic_strength=0.15,
    temperature=310.0,
    D_s=5e5,
    D_3d=5.0,
    focus_f=0.030,
    focus_k=0.062,
    nt_rd=4000,
):
    """
    Run a complete in-silico simulation of the DNA double-strand break
    repair cascade from damage induction to focus formation.

    Parameters
    ----------
    n_grid : int
        Spatial grid resolution for electrostatics and RD.
    dna_length_bp : int
        DNA length in base pairs.
    ionic_strength : float
        Ionic strength in mol/L.
    temperature : float
        Temperature in Kelvin.
    D_s : float
        1D sliding diffusion coefficient (bp^2/s).
    D_3d : float
        3D diffusion coefficient (um^2/s).
    focus_f, focus_k : float
        Gray-Scott feed and kill rates for focus formation.
    nt_rd : int
        Number of reaction-diffusion time steps.

    Returns
    -------
    results : dict
        Comprehensive simulation results.
    """
    results = {}

    # --- 1. Electrostatic environment ---
    h = 1.0  # grid spacing in nm
    lambda_D = debye_length(epsilon_r=78.0, temperature=temperature, ionic_strength=ionic_strength)
    kappa = 1.0 / lambda_D if lambda_D > 0 else 0.0
    results["debye_length_nm"] = float(lambda_D)
    results["kappa_per_nm"] = float(kappa)

    # Charge density: negatively charged DNA backbone in center region
    center = n_grid // 2
    width = max(1, n_grid // 8)
    dna_x_range = (center - width, center + width)
    dna_y_range = (center - width, center + width)
    rho = setup_dna_charge_density(n_grid, h, dna_x_range, dna_y_range, charge_per_unit=-1.0)

    phi_pb, converged, iters = solve_nonlinear_pb(n_grid, h, rho, kappa, tol=1e-7, max_iter=30)
    results["pb_converged"] = converged
    results["pb_iterations"] = iters
    results["phi_min"] = float(np.min(phi_pb))
    results["phi_max"] = float(np.max(phi_pb))

    # --- 2. 1D Sliding search on DNA ---
    # Discretize DNA into sites
    n_sites = min(64, dna_length_bp)
    dx_dna = dna_length_bp / n_sites
    x_dna = np.linspace(0, dna_length_bp, n_sites)

    # Potential landscape: damage well at center
    U_dna = np.zeros(n_sites)
    damage_site = n_sites // 2
    well_depth = 3.0  # kT
    well_width = 3  # sites
    for i in range(n_sites):
        dist = abs(i - damage_site)
        if dist < well_width:
            U_dna[i] = -well_depth * np.exp(-(dist ** 2) / (2.0 * (well_width / 2.0) ** 2))

    # Initial uniform distribution
    P = np.ones(n_sites) / n_sites
    dt_slide = 0.01 * dx_dna ** 2 / max(D_s, 1.0)
    n_slide_steps = min(5000, int(1.0 / dt_slide) + 1)
    kT = 1.0  # in kT units

    for _ in range(n_slide_steps):
        P = smoluchowski_1d_sliding_step(P, U_dna, D_s, kT, dt_slide, dx_dna, boundary="reflecting")

    results["sliding_final_P_peak"] = float(np.max(P))
    results["sliding_target_prob"] = float(P[damage_site])
    results["sliding_mean_time_theory_s"] = float(
        sliding_search_time(dna_length_bp, D_s, target_size=10.0)
    )

    # --- 3. Reaction-diffusion focus formation ---
    U_rd, V_rd, history = simulate_repair_focus(
        nx=n_grid,
        ny=n_grid,
        Du=D_3d * 0.1,
        Dv=D_3d * 0.05,
        f=0.030,
        k=0.062,
        dt=0.25,
        dx=h,
        nt=nt_rd,
        crowding=0.0,
        initial_pattern="damage_focus",
    )
    results["focus_V_max"] = float(np.max(V_rd))
    results["focus_V_mean"] = float(np.mean(V_rd))
    repair_time = estimate_repair_time(history, threshold=0.5)
    results["focus_formation_time_steps"] = repair_time

    # --- 4. Potential energy surface ---
    theta, r, U_surf = build_dna_repair_energy_surface(
        n_theta=41,
        n_r=41,
        k_bend=2.5,
        epsilon_lj=1.0,
        sigma_lj=0.8,
        damage_depth=-3.0,
        damage_width=0.3,
    )
    barrier, saddle = compute_activation_barrier(
        theta,
        r,
        U_surf,
        reactant_region=((-0.5, 0.5), (2.0, 3.0)),
        product_region=((0.3, 0.7), (0.5, 1.5)),
    )
    results["activation_barrier_kT"] = float(barrier)
    results["saddle_point"] = saddle

    # --- 5. Markov state model for repair states ---
    T_msm, pi_msm = build_msm_transition_matrix(n_states=5, temperature=1.0, seed=42)
    mfpt_repair = msm_mfpt(T_msm, target_state=4, start_state=0)
    results["msm_mfpt_steps"] = float(mfpt_repair)

    # --- 6. Banded 1D diffusion along DNA (validation) ---
    # Solve steady-state diffusion with source at damage site
    lower = -np.ones(n_sites - 1)
    diag = 2.0 * np.ones(n_sites)
    upper = -np.ones(n_sites - 1)
    rhs = np.zeros(n_sites)
    rhs[damage_site] = 1.0
    c_steady = banded_solve_tridiagonal(lower, diag, upper, rhs)
    results["steady_concentration_peak"] = float(np.max(c_steady))

    return results


def compute_repair_efficiency(results):
    """
    Compute a composite repair efficiency score from simulation results.

    Efficiency = w1 * P_target + w2 * (1 / (1 + barrier)) + w3 * V_focus_max
    """
    w1, w2, w3 = 0.4, 0.3, 0.3
    P_target = results.get("sliding_target_prob", 0.0)
    barrier = max(results.get("activation_barrier_kT", 1.0), 0.1)
    V_max = results.get("focus_V_max", 0.0)
    efficiency = w1 * P_target + w2 * (1.0 / (1.0 + barrier)) + w3 * V_max
    return float(efficiency)


def repair_sensitivity_analysis(
    ionic_strengths=[0.05, 0.15, 0.30],
    D_s_values=[1e5, 5e5, 1e6],
):
    """
    Perform sensitivity analysis of repair efficiency to ionic strength
    and sliding diffusion coefficient.

    Parameters
    ----------
    ionic_strengths : list
    D_s_values : list

    Returns
    -------
    sensitivity : list of dict
    """
    sensitivity = []
    for I in ionic_strengths:
        for D_s in D_s_values:
            res = simulate_dsb_repair_cascade(
                n_grid=24,
                ionic_strength=I,
                D_s=D_s,
                nt_rd=2000,
            )
            eff = compute_repair_efficiency(res)
            sensitivity.append(
                {
                    "ionic_strength": I,
                    "D_s": D_s,
                    "efficiency": eff,
                    "debye_length_nm": res["debye_length_nm"],
                    "barrier_kT": res["activation_barrier_kT"],
                }
            )
    return sensitivity
