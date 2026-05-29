# -*- coding: utf-8 -*-
"""
main.py
=======
Unified entry point for the base-isolated building seismic time-history
analysis and optimization system.

This script executes the full workflow without requiring any command-line
arguments:
  1. Structural model initialization (10-story shear building)
  2. Modal analysis and period verification
  3. Isolation bearing parameter optimization
  4. Synthetic ground-motion generation (3-component, fused)
  5. Nonlinear time-history analysis (Newmark-beta implicit integration)
  6. Response post-processing (drift, shear, energy)
  7. Numerical verification suite
  8. Summary output

Scientific domain:  Structural Mechanics — Seismic Response Time-History
Analysis and Base Isolation Design.
"""

import numpy as np
import time

# Project modules
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

# Additional imports for test cases
from isolation_bearing import LeadRubberBearing
from optimization import diophantine_nd_nonnegative, random_constrained_sample
from seismic_wave import envelope_ari, kanai_tajimi_psd, chirikov_map_perturbation, constrained_random_phases, synthesize_ground_motion, fuse_components
from time_integrator import backward_euler_step
from linear_solver import cholesky_factorize, solve_upper_triangular, solve_lower_triangular, cholesky_solve, cgs_squared
from response_analysis import trapz_integral, cumtrapz
from quadrature_rules import legendre_ek_compute, integrate_over_disk, pyramid01_monomial_integral, pyramid_volume, bearing_contact_force, pyramid_consistent_mass
from verification import exact_harmonic_response_sdof


def iso_force_closure(iso_system):
    """Closure returning isolation force as a function of (u, v)."""
    def force(u, v):
        # Only the base DOF (index 0) interacts with isolation
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

    # ================================================================== #
    # Step 1: Structural model
    # ================================================================== #
    print("\n[Step 1] Initializing structural model...")
    n_story = 10
    story_heights = np.full(n_story, 3.5)   # 3.5 m typical story
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

    # ================================================================== #
    # Step 2: Modal analysis
    # ================================================================== #
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

    # ================================================================== #
    # Step 3: Isolation bearing optimization
    # ================================================================== #
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

    # Initialize isolation system with optimized parameters
    iso_system = IsolationSystem(
        n_bearings=best_design["n_bearings"],
        Q_d_per=best_design["Q_d_per"],
        k_d_per=best_design["k_d_per"],
    )
    iso_system.reset()

    # Update structural stiffness with isolation contribution
    k_iso_elastic = iso_system.k_e_total
    k_iso_post = iso_system.k_d_total
    # We use an effective initial stiffness for the linear K matrix
    model.update_isolation_stiffness(k_iso_post)
    M, C, K = model.get_matrices()

    # ================================================================== #
    # Step 4: Ground motion generation
    # ================================================================== #
    print("\n[Step 4] Generating synthetic ground motion...")
    dt = 0.01
    t_max = 20.0
    wave_gen = SeismicWaveGenerator(dt=dt, t_max=t_max, seed=84)
    a_g_fused = wave_gen.get_fused_record()
    t = wave_gen.t
    print(f"  Duration: {t_max:.1f} s, dt = {dt:.3f} s, steps = {len(t)}")
    print(f"  PGA (fused): {np.max(np.abs(a_g_fused)):.3f} m/s^2")

    # ================================================================== #
    # Step 5: Time-history analysis
    # ================================================================== #
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

    # Initial conditions
    u0 = np.zeros(n_dof, dtype=float)
    v0 = np.zeros(n_dof, dtype=float)
    a0 = np.zeros(n_dof, dtype=float)

    # Isolation force function
    iso_force_func = iso_force_closure(iso_system)

    # Pre-allocate isolation force history
    iso_force_history = np.zeros(len(t), dtype=float)

    # Manual integration loop (to record iso force at each step)
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

    # ================================================================== #
    # Step 6: Response analysis
    # ================================================================== #
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

    # ================================================================== #
    # Step 7: Quadrature-based structural checks
    # ================================================================== #
    print("\n[Step 7] Performing quadrature-based structural checks...")
    # Bearing contact pressure integration
    bearing_radius = 0.25   # m
    p_uniform = 5.0e6       # Pa uniform pressure
    def pressure_func(x, y):
        return p_uniform * np.ones_like(x)

    F_contact = bearing_contact_force(pressure_func, bearing_radius=bearing_radius, nr=12, nt=24)
    F_theoretical = p_uniform * np.pi * bearing_radius ** 2
    print(f"  Bearing contact force (quadrature): {F_contact/1e3:.1f} kN")
    print(f"  Theoretical contact force:          {F_theoretical/1e3:.1f} kN")
    print(f"  Quadrature relative error:          {abs(F_contact - F_theoretical)/F_theoretical:.3e}")

    # Pyramid consistent mass matrix
    rho_concrete = 2500.0   # kg/m^3
    base_area = 20.0 * 20.0
    height = 3.5
    M_pyramid = pyramid_consistent_mass(rho_concrete, base_area, height)
    m_total_pyramid = float(np.sum(M_pyramid))
    m_lumped = rho_concrete * base_area * height / 3.0
    print(f"  Pyramid element total mass:         {m_total_pyramid:,.0f} kg")
    print(f"  Lumped mass (reference):            {m_lumped:,.0f} kg")
    print(f"  Mass matrix relative error:         {abs(m_total_pyramid - m_lumped)/m_lumped:.3e}")

    # ================================================================== #
    # Step 8: Numerical verification
    # ================================================================== #
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

    # ================================================================== #
    # Step 9: Final summary
    # ================================================================== #
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Total elapsed time: {elapsed:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: ShearBuildingModel default construction and matrix shapes ----
model = ShearBuildingModel(n_story=3)
M, C, K = model.get_matrices()
assert M.shape == (4, 4), '[TC01] ShearBuildingModel M shape FAILED'
assert K.shape == (4, 4), '[TC01] ShearBuildingModel K shape FAILED'
assert C.shape == (4, 4), '[TC01] ShearBuildingModel C shape FAILED'

# ---- TC02: ShearBuildingModel total height consistency ----
model2 = ShearBuildingModel(n_story=5, story_heights=np.full(5, 4.0))
assert abs(model2.total_height - 20.0) < 1e-6, '[TC02] Total height consistency FAILED'

# ---- TC03: ShearBuildingModel influence vector is all ones ----
Gamma_test = model.get_influence_vector()
assert np.allclose(Gamma_test, 1.0), '[TC03] Influence vector not all ones FAILED'

# ---- TC04: ShearBuildingModel update isolation stiffness raises on non-positive ----
try:
    model.update_isolation_stiffness(-1.0)
    assert False, '[TC04] Negative stiffness should raise FAILED'
except ValueError:
    pass

# ---- TC05: ModalAnalysis periods and frequencies are positive ----
np.random.seed(42)
M_test = np.diag([2.0, 1.0, 1.0])
K_test = np.array([[3.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
modal = ModalAnalysis(M_test, K_test, n_modes=3)
periods = modal.get_periods()
freqs = modal.get_natural_frequencies()
assert np.all(periods > 0), '[TC05] Modal periods not positive FAILED'
assert np.all(freqs > 0), '[TC05] Modal frequencies not positive FAILED'

# ---- TC06: ModalAnalysis truncation error returns integer within bounds ----
n_needed = modal.truncation_error(target_ratio=0.5)
assert isinstance(n_needed, int), '[TC06] Truncation error not int FAILED'
assert 1 <= n_needed <= modal.n_modes, '[TC06] Truncation error out of bounds FAILED'

# ---- TC07: ModalAnalysis modal matrix orthogonality (mass-normalized) ----
phi = modal.get_modal_matrix()
ident = phi.T @ M_test @ phi
assert np.allclose(ident, np.eye(3), atol=1e-7), '[TC07] Modal orthogonality FAILED'

# ---- TC08: LeadRubberBearing elastic branch force correctness ----
bearing = LeadRubberBearing(Q_d=1.0e5, k_d=1.0e7)
bearing.reset_state()
F_small = bearing.force(0.001, 0.0)
assert abs(F_small - bearing.k_e * 0.001) < 1.0, '[TC08] Elastic branch force FAILED'

# ---- TC09: LeadRubberBearing energy dissipation is zero below yield ----
E_d = bearing.energy_dissipation(d_max=bearing.d_y * 0.5)
assert E_d == 0.0, '[TC09] Energy dissipation below yield not zero FAILED'

# ---- TC10: LeadRubberBearing effective stiffness at zero displacement ----
k_eff0 = bearing.effective_stiffness(0.0)
assert k_eff0 == bearing.k_e, '[TC10] Effective stiffness at zero FAILED'

# ---- TC11: IsolationSystem total force scales with number of bearings ----
iso1 = IsolationSystem(n_bearings=10, Q_d_per=1.0e4, k_d_per=1.0e6)
iso2 = IsolationSystem(n_bearings=20, Q_d_per=1.0e4, k_d_per=1.0e6)
iso1.reset()
iso2.reset()
F1 = iso1.total_force(0.01, 0.0)
F2 = iso2.total_force(0.01, 0.0)
assert abs(F2 - 2.0 * F1) < 1.0, '[TC11] IsolationSystem force scaling FAILED'

# ---- TC12: IsolationSystem effective period is positive ----
T_eff = iso1.effective_period(M_base=1.0e6)
assert T_eff > 0, '[TC12] Effective period not positive FAILED'

# ---- TC13: diophantine_nd_nonnegative basic correctness ----
sols = diophantine_nd_nonnegative(np.array([2, 3]), 6)
assert sols.shape[0] >= 1, '[TC13] Diophantine no solutions FAILED'
assert np.all(sols @ np.array([2, 3]) == 6), '[TC13] Diophantine equation not satisfied FAILED'

# ---- TC14: diophantine_nd_nonnegative empty for negative b ----
sols_empty = diophantine_nd_nonnegative(np.array([1, 2]), -1)
assert sols_empty.shape[0] == 0, '[TC14] Diophantine negative b should be empty FAILED'

# ---- TC15: IsolationOptimizer returns valid design dict ----
optimizer = IsolationOptimizer(M_total=1.0e7, W_total=9.81e7, T_iso_target=2.5)
design = optimizer.optimize(n_samples=50)
assert 'n_bearings' in design, '[TC15] Optimizer design missing n_bearings FAILED'
assert 'Q_d_per' in design, '[TC15] Optimizer design missing Q_d_per FAILED'
assert 'k_d_per' in design, '[TC15] Optimizer design missing k_d_per FAILED'
assert design['n_bearings'] >= optimizer.n_b_min, '[TC15] n_bearings below minimum FAILED'

# ---- TC16: envelope_ari shape and boundary conditions ----
t_env = np.linspace(-1.0, 20.0, 100)
env = envelope_ari(t_env, t_rise=2.0, t_flat=8.0, t_decay=4.0)
assert env.shape == t_env.shape, '[TC16] Envelope shape mismatch FAILED'
assert np.all(env[t_env < 0] == 0.0), '[TC16] Envelope not zero for negative t FAILED'
assert np.all((env >= 0) & (env <= 1.0)), '[TC16] Envelope out of [0,1] FAILED'

# ---- TC17: kanai_tajimi_psd is non-negative ----
omega_psd = np.linspace(0.1, 50.0, 100)
psd = kanai_tajimi_psd(omega_psd, omega_g=15.0, zeta_g=0.6, S0=0.01)
assert np.all(psd >= 0), '[TC17] PSD has negative values FAILED'

# ---- TC18: chirikov_map_perturbation preserves array shape ----
phases_in = np.array([0.0, 1.0, 2.0, 3.0])
phases_out = chirikov_map_perturbation(phases_in, K=0.8, iterations=3)
assert phases_out.shape == phases_in.shape, '[TC18] Chirikov shape changed FAILED'

# ---- TC19: constrained_random_phases output in [0, 2pi] ----
np.random.seed(42)
phases_rand = constrained_random_phases(64, seed=123)
assert np.all(phases_rand >= 0) and np.all(phases_rand <= 2.0 * np.pi), '[TC19] Phases out of range FAILED'

# ---- TC20: synthesize_ground_motion output shapes and zero-mean ----
np.random.seed(42)
t_gm, a_gm = synthesize_ground_motion(dt=0.02, t_max=10.0, seed=42)
assert t_gm.shape == a_gm.shape, '[TC20] Ground motion shape mismatch FAILED'
assert len(t_gm) > 0, '[TC20] Ground motion empty FAILED'
assert abs(np.mean(a_gm)) < 0.1, '[TC20] Ground motion not near zero mean FAILED'

# ---- TC21: fuse_components default weights sum to 1 ----
a_x = np.ones(10)
a_y = np.ones(10) * 2.0
a_z = np.ones(10) * 3.0
fused = fuse_components(a_x, a_y, a_z)
expected = 0.7 * 1.0 + 0.2 * 2.0 + 0.1 * 3.0
assert np.allclose(fused, np.full(10, expected)), '[TC21] Fuse components default weights FAILED'

# ---- TC22: SeismicWaveGenerator fused record shape ----
wave_gen = SeismicWaveGenerator(dt=0.02, t_max=5.0, seed=77)
rec = wave_gen.get_fused_record()
assert rec.shape == wave_gen.t.shape, '[TC22] Fused record shape mismatch FAILED'

# ---- TC23: NewmarkBetaIntegrator step advances state ----
M_int = np.diag([1.0, 1.0])
C_int = np.diag([0.1, 0.1])
K_int = np.array([[2.0, -1.0], [-1.0, 2.0]])
integrator = NewmarkBetaIntegrator(M_int, C_int, K_int, gamma=0.5, beta=0.25, dt=0.01)
def dummy_iso(u, v):
    return np.zeros_like(u)
u0 = np.zeros(2)
v0 = np.zeros(2)
a0 = np.zeros(2)
u1, v1, a1 = integrator.step(u0, v0, a0, 0.0, dummy_iso, solve_linear_system, np.ones(2))
assert u1.shape == u0.shape, '[TC23] Step output shape mismatch FAILED'
assert np.all(np.isfinite(u1)), '[TC23] Step output non-finite FAILED'

# ---- TC24: backward_euler_step converges for linear ODE ----
def linear_ode(y):
    return -2.0 * y
y0 = np.array([1.0])
y1 = backward_euler_step(y0, linear_ode, dt=0.1, max_inner_iter=20)
assert abs(y1[0] - 1.0 / (1.0 + 0.2)) < 1e-6, '[TC24] Backward Euler linear ODE FAILED'

# ---- TC25: cholesky_factorize on identity matrix ----
R, info = cholesky_factorize(np.eye(4))
assert info == 0, '[TC25] Cholesky factorize identity info FAILED'
assert np.allclose(R, np.eye(4)), '[TC25] Cholesky factorize identity R FAILED'

# ---- TC26: solve_upper_triangular correctness ----
U_tri = np.array([[2.0, 1.0], [0.0, 3.0]])
b_tri = np.array([5.0, 6.0])
x_tri = solve_upper_triangular(U_tri, b_tri)
assert np.allclose(U_tri @ x_tri, b_tri), '[TC26] Upper triangular solve FAILED'

# ---- TC27: solve_lower_triangular correctness ----
L_tri = np.array([[2.0, 0.0], [1.0, 3.0]])
b_tri2 = np.array([4.0, 7.0])
x_tri2 = solve_lower_triangular(L_tri, b_tri2)
assert np.allclose(L_tri @ x_tri2, b_tri2), '[TC27] Lower triangular solve FAILED'

# ---- TC28: cholesky_solve vs numpy solve ----
A_spd = np.array([[4.0, 1.0], [1.0, 3.0]])
b_spd = np.array([5.0, 4.0])
x_chol = cholesky_solve(A_spd, b_spd)
x_np = np.linalg.solve(A_spd, b_spd)
assert np.allclose(x_chol, x_np), '[TC28] Cholesky solve vs numpy FAILED'

# ---- TC29: cgs_squared solves linear system ----
A_cgs = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b_cgs = np.array([1.0, 2.0, 3.0])
x_cgs = cgs_squared(A_cgs, b_cgs, tol=1e-10)
assert np.allclose(A_cgs @ x_cgs, b_cgs, atol=1e-8), '[TC29] CGS solver FAILED'

# ---- TC30: solve_linear_system auto method returns correct solution ----
A_slv = np.diag([1.0, 2.0, 3.0])
b_slv = np.array([1.0, 2.0, 3.0])
x_slv = solve_linear_system(A_slv, b_slv, method="auto")
assert np.allclose(x_slv, np.array([1.0, 1.0, 1.0])), '[TC30] Auto solver FAILED'

# ---- TC31: trapz_integral of linear function ----
x_trap = np.linspace(0.0, 1.0, 101)
y_trap = 2.0 * x_trap
I_trap = trapz_integral(x_trap, y_trap)
assert abs(I_trap - 1.0) < 1e-3, '[TC31] Trapezoidal integral linear FAILED'

# ---- TC32: cumtrapz monotonic increase for positive integrand ----
x_cum = np.linspace(0.0, 1.0, 51)
y_cum = np.ones(51)
I_cum = cumtrapz(x_cum, y_cum)
assert I_cum[0] == 0.0, '[TC32] cumtrapz start not zero FAILED'
assert np.all(np.diff(I_cum) >= 0), '[TC32] cumtrapz not monotonic FAILED'
assert abs(I_cum[-1] - 1.0) < 1e-6, '[TC32] cumtrapz final value FAILED'

# ---- TC33: ResponseAnalyzer summary keys completeness ----
U_ra = np.zeros((20, 4))
V_ra = np.zeros((20, 4))
A_ra = np.zeros((20, 4))
t_ra = np.linspace(0.0, 0.2, 20)
M_ra = np.diag([1.0, 1.0, 1.0, 1.0])
C_ra = np.diag([0.1, 0.1, 0.1, 0.1])
K_ra = np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 1.0]])
Gamma_ra = np.ones(4)
a_g_ra = np.zeros(20)
story_heights_ra = np.array([3.5, 3.5, 3.5])
analyzer = ResponseAnalyzer(U_ra, V_ra, A_ra, t_ra, M_ra, C_ra, K_ra, Gamma_ra, a_g_ra, story_heights_ra)
summary = analyzer.summary()
expected_keys = {"max_isolation_displacement_m", "max_roof_displacement_m", "max_drift_ratio", "max_base_shear_kN", "max_floor_accel_g", "energy_balance_error"}
assert expected_keys.issubset(set(summary.keys())), '[TC33] Response summary keys missing FAILED'

# ---- TC34: ResponseAnalyzer interstory_drift shape ----
drift = analyzer.interstory_drift()
assert drift.shape == (20, 3), '[TC34] Interstory drift shape FAILED'

# ---- TC35: ResponseAnalyzer displaced coordinates shape ----
base_coords = np.zeros((4, 3))
coords_t = analyzer.displaced_coordinates(base_coords)
assert coords_t.shape == (20, 4, 3), '[TC35] Displaced coordinates shape FAILED'

# ---- TC36: legendre_ek_compute weights sum to 2 ----
x_leg, w_leg = legendre_ek_compute(5)
assert abs(np.sum(w_leg) - 2.0) < 1e-12, '[TC36] Legendre weights sum FAILED'
assert len(x_leg) == 5, '[TC36] Legendre node count FAILED'

# ---- TC37: integrate_over_disk unit constant function ----
F_disk = integrate_over_disk(lambda x, y: np.ones_like(x), nr=8, nt=16, radius=1.0)
assert abs(F_disk - np.pi) < 0.05, '[TC37] Disk integral of constant FAILED'

# ---- TC38: pyramid01_monomial_integral odd x exponent is zero ----
I_pyramid = pyramid01_monomial_integral((1, 0, 0))
assert I_pyramid == 0.0, '[TC38] Pyramid odd exponent integral FAILED'

# ---- TC39: pyramid_volume is 4/3 ----
assert abs(pyramid_volume() - 4.0 / 3.0) < 1e-12, '[TC39] Pyramid volume FAILED'

# ---- TC40: bearing_contact_force with uniform pressure ----
F_bearing = bearing_contact_force(lambda x, y: np.ones_like(x) * 1.0e6, bearing_radius=0.5, nr=12, nt=24)
F_theory = 1.0e6 * np.pi * 0.5 ** 2
assert abs(F_bearing - F_theory) / F_theory < 0.01, '[TC40] Bearing contact force FAILED'

# ---- TC41: pyramid_consistent_mass matrix properties ----
M_pyr = pyramid_consistent_mass(rho=2500.0, base_area=100.0, height=3.0)
assert M_pyr.shape == (5, 5), '[TC41] Pyramid mass matrix shape FAILED'
assert np.allclose(M_pyr, M_pyr.T), '[TC41] Pyramid mass matrix not symmetric FAILED'
m_total_pyr = float(np.sum(M_pyr))
m_expected = 2500.0 * 100.0 * 3.0 / 3.0
assert abs(m_total_pyr - m_expected) / m_expected < 1e-10, '[TC41] Pyramid mass total FAILED'

# ---- TC42: exact_harmonic_response_sdof off-resonance amplitude ----
t_harm = np.linspace(0.0, 10.0, 1000)
u_harm = exact_harmonic_response_sdof(m=1.0, c=0.2, k=100.0, f0=10.0, omega=8.0, t=t_harm)
assert np.all(np.isfinite(u_harm)), '[TC42] Harmonic response non-finite FAILED'
U_expected = 10.0 / np.sqrt((100.0 - 64.0) ** 2 + (1.6) ** 2)
assert abs(np.max(np.abs(u_harm)) - U_expected) < 1e-6, '[TC42] Harmonic response amplitude mismatch FAILED'

# ---- TC43: VerificationSuite mass matrix SPD on valid M ----
M_v = np.diag([1.0, 2.0, 3.0])
K_v = np.eye(3)
verifier = VerificationSuite(M_v, K_v)
ok_spd = verifier.test_mass_matrix_spd()
assert ok_spd, '[TC43] VerificationSuite mass SPD FAILED'

# ---- TC44: VerificationSuite static equilibrium on well-conditioned K ----
K_v2 = np.array([[3.0, -1.0], [-1.0, 2.0]])
verifier2 = VerificationSuite(np.eye(2), K_v2)
ok_static = verifier2.test_static_equilibrium()
assert ok_static, '[TC44] VerificationSuite static equilibrium FAILED'

# ---- TC45: VerificationSuite energy conservation ----
verifier3 = VerificationSuite(np.eye(2), np.eye(2) * 4.0)
ok_energy = verifier3.test_energy_conservation(dt=0.01, n_steps=200)
assert ok_energy, '[TC45] VerificationSuite energy conservation FAILED'

# ---- TC46: random_constrained_sample returns requested count ----
np.random.seed(42)
samples = random_constrained_sample(
    n_samples=5,
    n_b_min=10,
    n_b_max=15,
    Q_d_grid=np.array([1.0e5, 2.0e5]),
    k_d_grid=np.array([1.0e6]),
    M_total=2.0e6,
    T_iso_target=2.5,
    W_total=2.0e7,
    seed=42,
)
assert len(samples) == 5, '[TC46] Random constrained sample count FAILED'
for s in samples:
    assert 10 <= s['n_bearings'] <= 15, '[TC46] Sample n_bearings out of range FAILED'

# ---- TC47: SeismicWaveGenerator generate shape ----
wave_gen2 = SeismicWaveGenerator(dt=0.02, t_max=4.0, seed=99)
acc_multi = wave_gen2.generate(n_components=3)
assert acc_multi.shape == (len(wave_gen2.t), 3), '[TC47] Generate multi-component shape FAILED'

# ---- TC48: NewmarkBetaIntegrator integrate full history ----
M_int2 = np.diag([1.0])
C_int2 = np.diag([0.05])
K_int2 = np.diag([10.0])
integrator2 = NewmarkBetaIntegrator(M_int2, C_int2, K_int2, dt=0.01)
a_g_short = np.zeros(50)
U_full, V_full, A_full = integrator2.integrate(
    np.zeros(1), np.zeros(1), np.zeros(1), a_g_short, dummy_iso, solve_linear_system, np.ones(1)
)
assert U_full.shape == (50, 1), '[TC48] Integrate U shape FAILED'
assert V_full.shape == (50, 1), '[TC48] Integrate V shape FAILED'
assert A_full.shape == (50, 1), '[TC48] Integrate A shape FAILED'

# ---- TC49: solve_linear_system method cholesky on small SPD ----
A_small = np.array([[5.0, 1.0], [1.0, 3.0]])
b_small = np.array([7.0, 5.0])
x_chol_direct = solve_linear_system(A_small, b_small, method="cholesky")
assert np.allclose(A_small @ x_chol_direct, b_small), '[TC49] Cholesky method solve FAILED'

# ---- TC50: solve_linear_system method cgs on small system ----
x_cgs_direct = solve_linear_system(A_small, b_small, method="cgs")
assert np.allclose(A_small @ x_cgs_direct, b_small), '[TC50] CGS method solve FAILED'

print('\n全部 50 个测试通过!\n')
