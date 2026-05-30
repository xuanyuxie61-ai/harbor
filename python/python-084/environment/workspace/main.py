# -*- coding: utf-8 -*-

import numpy as np
import time


from structure_model import ShearBuildingModel
from modal_analysis import ModalAnalysis
from isolation_bearing import IsolationSystem
from optimization import IsolationOptimizer
from seismic_wave import SeismicWaveGenerator
from time_integrator import NewmarkBetaIntegrator
from linear_solver import solve_linear_system
from response_analysis import ResponseAnalyzer
from quadrature_rules import bearing_contact_force, pyramid_consistent_mass
from verification import VerificationSuite


def iso_force_closure(iso_system):
    def force(u, v):

        F = np.zeros_like(u)
        F[0] = iso_system.total_force(float(u[0]), float(v[0]))
        return F
    return force


def main():
    print("=" * 70)
    print("  SEISMIC TIME-HISTORY ANALYSIS OF BASE-ISOLATED BUILDING")
    print("  Domain: Structural Mechanics — Earthquake Response & Isolation")
    print("=" * 70)
    t_start = time.time()




    print("\n[Step 1] Initializing structural model...")
    n_story = 10
    story_heights = np.full(n_story, 3.5)
    story_masses = np.linspace(1.2e6, 8.0e5, n_story)
    story_stiffness = np.linspace(8.0e8, 2.0e8, n_story)

    model = ShearBuildingModel(
        n_story=n_story,
        story_heights=story_heights,
        story_masses=story_masses,
        story_stiffness=story_stiffness,
        damping_ratio=0.05,
    )
    M, C, K = model.get_matrices()
    Gamma = model.get_influence_vector()
    coords = model.get_node_coordinates()
    n_dof = model.n_dof

    print(f"  Building: {n_story} stories, {n_dof} DOFs")
    print(f"  Total height: {model.total_height:.2f} m")
    print(f"  Total mass: {np.sum(np.diag(M)):,.0f} kg")




    print("\n[Step 2] Performing modal analysis...")
    modal = ModalAnalysis(M, K, n_modes=5)
    periods = modal.get_periods()
    freqs = modal.get_natural_frequencies()
    print(f"  Mode 1 period: {periods[0]:.3f} s  (freq = {freqs[0]/(2*np.pi):.3f} Hz)")
    print(f"  Mode 2 period: {periods[1]:.3f} s  (freq = {freqs[1]/(2*np.pi):.3f} Hz)")
    print(f"  Mode 3 period: {periods[2]:.3f} s")
    print(f"  Cumulative mass ratio (3 modes): {modal.meff_ratio[2]:.3f}")
    n_needed = modal.truncation_error(target_ratio=0.90)
    print(f"  Modes needed for 90% mass participation: {n_needed}")




    print("\n[Step 3] Optimizing isolation bearing parameters...")
    M_total = float(np.sum(np.diag(M)))
    W_total = M_total * 9.81
    optimizer = IsolationOptimizer(
        M_total=M_total,
        W_total=W_total,
        T_iso_target=2.5,
        d_y_max=0.15,
        n_b_min=8,
        n_b_max=30,
    )
    best_design = optimizer.optimize(n_samples=300)
    print(f"  Optimal bearings: {best_design['n_bearings']} units")
    print(f"  Q_d per bearing:  {best_design['Q_d_per']:,.0f} N")
    print(f"  k_d per bearing:  {best_design['k_d_per']:,.0f} N/m")
    print(f"  Estimated period: {best_design['period_est']:.3f} s")
    print(f"  Objective value:  {best_design['objective']:.4f}")


    iso_system = IsolationSystem(
        n_bearings=best_design["n_bearings"],
        Q_d_per=best_design["Q_d_per"],
        k_d_per=best_design["k_d_per"],
    )
    iso_system.reset()


    k_iso_elastic = iso_system.k_e_total
    k_iso_post = iso_system.k_d_total

    model.update_isolation_stiffness(k_iso_post)
    M, C, K = model.get_matrices()




    print("\n[Step 4] Generating synthetic ground motion...")
    dt = 0.01
    t_max = 20.0
    wave_gen = SeismicWaveGenerator(dt=dt, t_max=t_max, seed=84)
    a_g_fused = wave_gen.get_fused_record()
    t = wave_gen.t
    print(f"  Duration: {t_max:.1f} s, dt = {dt:.3f} s, steps = {len(t)}")
    print(f"  PGA (fused): {np.max(np.abs(a_g_fused)):.3f} m/s^2")




    print("\n[Step 5] Running nonlinear time-history analysis...")
    integrator = NewmarkBetaIntegrator(
        M=M,
        C=C,
        K=K,
        gamma=0.5,
        beta=0.25,
        dt=dt,
        max_iter=10,
        tol=1e-8,
    )


    u0 = np.zeros(n_dof, dtype=float)
    v0 = np.zeros(n_dof, dtype=float)
    a0 = np.zeros(n_dof, dtype=float)


    iso_force_func = iso_force_closure(iso_system)


    iso_force_history = np.zeros(len(t), dtype=float)


    n_time = len(t)
    U = np.zeros((n_time, n_dof), dtype=float)
    V = np.zeros((n_time, n_dof), dtype=float)
    A = np.zeros((n_time, n_dof), dtype=float)

    U[0, :] = u0
    V[0, :] = v0
    A[0, :] = a0
    iso_force_history[0] = iso_system.total_force(u0[0], v0[0])

    u_n = u0.copy()
    v_n = v0.copy()
    a_n = a0.copy()

    for i in range(1, n_time):
        u_n, v_n, a_n = integrator.step(
            u_n, v_n, a_n, a_g_fused[i], iso_force_func, solve_linear_system, Gamma
        )
        U[i, :] = u_n
        V[i, :] = v_n
        A[i, :] = a_n
        iso_force_history[i] = iso_system.total_force(u_n[0], v_n[0])

    print("  Time-history integration complete.")




    print("\n[Step 6] Computing response quantities...")
    analyzer = ResponseAnalyzer(
        U=U,
        V=V,
        A=A,
        t=t,
        M=M,
        C=C,
        K=K,
        Gamma=Gamma,
        a_g=a_g_fused,
        story_heights=story_heights,
        iso_force_history=iso_force_history,
    )
    summary = analyzer.summary()

    print(f"  Max isolation displacement: {summary['max_isolation_displacement_m']:.4f} m")
    print(f"  Max roof displacement:      {summary['max_roof_displacement_m']:.4f} m")
    print(f"  Max inter-story drift:      {summary['max_drift_ratio']:.4f}")
    print(f"  Max base shear:             {summary['max_base_shear_kN']:.1f} kN")
    print(f"  Max floor acceleration:     {summary['max_floor_accel_g']:.3f} g")
    print(f"  Energy balance error:       {summary['energy_balance_error']:.3e}")




    print("\n[Step 7] Performing quadrature-based structural checks...")

    bearing_radius = 0.25
    p_uniform = 5.0e6
    def pressure_func(x, y):
        return p_uniform * np.ones_like(x)

    F_contact = bearing_contact_force(pressure_func, bearing_radius=bearing_radius, nr=12, nt=24)
    F_theoretical = p_uniform * np.pi * bearing_radius ** 2
    print(f"  Bearing contact force (quadrature): {F_contact/1e3:.1f} kN")
    print(f"  Theoretical contact force:          {F_theoretical/1e3:.1f} kN")
    print(f"  Quadrature relative error:          {abs(F_contact - F_theoretical)/F_theoretical:.3e}")


    rho_concrete = 2500.0
    base_area = 20.0 * 20.0
    height = 3.5
    M_pyramid = pyramid_consistent_mass(rho_concrete, base_area, height)
    m_total_pyramid = float(np.sum(M_pyramid))
    m_lumped = rho_concrete * base_area * height / 3.0
    print(f"  Pyramid element total mass:         {m_total_pyramid:,.0f} kg")
    print(f"  Lumped mass (reference):            {m_lumped:,.0f} kg")
    print(f"  Mass matrix relative error:         {abs(m_total_pyramid - m_lumped)/m_lumped:.3e}")




    print("\n[Step 8] Running numerical verification suite...")
    verifier = VerificationSuite(M, K, C)
    phi = modal.get_modal_matrix()
    verifier.run_all(phi=phi)
    verifier.print_results()

    all_pass = all(ok for ok, _ in verifier.results.values())
    if all_pass:
        print("\n  >>> ALL VERIFICATION TESTS PASSED <<<")
    else:
        print("\n  >>> SOME VERIFICATION TESTS FAILED (see details above) <<<")




    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Total elapsed time: {elapsed:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
