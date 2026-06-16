# -*- coding: utf-8 -*-
"""
main.py -- unified entry point for PROJECT 289
===============================================

1D slab gyrokinetic turbulence simulator with high-order finite
differences and linear / nonlinear stability analysis.

The pipeline:

    1. Construct the slab equilibrium (physics_constants)
    2. Build the radial FEM mesh and the high-order FD operators
       (radial_mesh, finite_difference)
    3. Assemble the velocity-space quadrature and FLR weights
       (velocity_space, gyroaverage)
    4. Set up the collision operator (collision_operator)
    5. Solve the slab ITG eigenvalue problem for several k_y
       (linear_eigenvalue) -- linear stability
    6. Bisect in R/L_Ti to locate the ITG threshold
    7. Initialise a zonal-flow residual test and relax it
       (zonal_flow)
    8. Generate a synthetic turbulent heat-flux time series by
       integrating a reduced low-order model with IMEX time advance
       (time_integrator)
    9. Apply the full diagnostic suite -- SG filtering, synthetic
       diagnostic noise, flux PDF, intermittency exponent
       (turbulence_diagnostics)
   10. Save a summary JSON and print a human-readable report.

Run
---
    python main.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict

import numpy as np

# Make the project importable whether invoked from the parent directory
# or from inside the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from physics_constants import (          # noqa: E402
    EquilibriumProfiles,
    ion_sound_speed,
    ion_cyclotron_freq,
    gyroradius,
    debye_length,
    beta_plasma,
    coulomb_logarithm,
    PI,
)
from velocity_space import (             # noqa: E402
    jn_eval,
    jn_zeros,
    patterson_rule_ab,
    cvt_1d_nonuniform,
    velocity_moments,
)
from radial_mesh import (                # noqa: E402
    RadialMesh1D,
    TriangularPoloidalMesh,
    PhaseSpaceIO,
    basic_hat,
)
from finite_difference import (          # noqa: E402
    centred_diff1,
    centred_diff2,
    compact_diff1,
    BandedMatrixR8GB,
    assemble_radial_d2_operator,
)
from gyroaverage import (                # noqa: E402
    gamma0,
    gamma1,
    gamma0_pade,
    gyrokinetic_poisson_1d,
    apply_j0_to_field,
    flr_operator_1d,
)
from collision_operator import (         # noqa: E402
    LorentzOperator,
    CollisionMatrix,
    collision_frequency,
)
from time_integrator import (            # noqa: E402
    ode_trapezoidal,
    ssp_rk3_step,
    imex_trapezoidal_step,
    IMEXTrapezoidalState,
    stability_region_boundary,
)
from linear_eigenvalue import (          # noqa: E402
    SlabITGConfig,
    solve_slab_itg,
    find_itg_threshold,
)
from zonal_flow import (                 # noqa: E402
    ZonalFlowParams,
    compute_zonal_flow_residual,
    relax_zonal_flow,
)
from turbulence_diagnostics import (     # noqa: E402
    sgolay1d,
    sgolay2d,
    SyntheticDiagnostic,
    regional_mean_2d,
    FluxPDF,
    run_diagnostics,
)


# ============================================================================
# Helper utilities
# ============================================================================
def _section(title: str) -> None:
    line = "=" * 72
    print()
    print(line)
    print("  " + title)
    print(line)


def _save_summary(path: str, summary: dict) -> None:
    # Convert numpy scalars to plain python for JSON serialisation.
    def _normalise(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, complex):
            return {"re": obj.real, "im": obj.imag}
        if isinstance(obj, dict):
            return {k: _normalise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_normalise(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_normalise(summary), fh, indent=2, ensure_ascii=False)


# ============================================================================
# Pipeline steps
# ============================================================================
def step_0_equilibrium() -> EquilibriumProfiles:
    _section("Step 0 -- Equilibrium construction")
    eq = EquilibriumProfiles(
        Lx=10.0,
        n_ref=1.0,
        T_ref=1.0e3,          # eV
        B0=2.5,
        m_i_amu=2.0,
        Z_i=1.0,
        delta_n=0.1,
        eta_i=3.0,
        eta_e=3.0,
        k_y_rho=0.3,
        L_s=20.0,
    )
    x = np.linspace(0.0, eq.Lx, 5)
    print(f"  ion sound speed  c_s      = {eq.c_s:.4e} m/s")
    print(f"  ion cyclotron    omega_ci = {eq.omega_ci:.4e} rad/s")
    print(f"  gyroradius       rho_s    = {eq.rho_s:.4e} m")
    print(f"  L_n = {eq.L_n():.3f},  L_Ti = {eq.L_Ti():.3f},  R/L_Ti = {eq.R_LTi():.3f}")
    print(f"  ITG threshold (rough)  (R/L_Ti)_c ~ {eq.R_LTc():.3f}")
    print(f"  n0 profile (5 pts)  = {eq.n0(x)}")
    print(f"  T_i profile (5 pts) = {eq.T_i(x)}")
    return eq


def step_1_mesh_and_fd() -> RadialMesh1D:
    _section("Step 1 -- Radial FEM mesh and high-order FD operators")
    mesh = RadialMesh1D.uniform(0.0, 1.0, 61)
    # FEM steady heat solve (sanity)
    u_fem = mesh.solve_dirichlet(lambda x: np.full_like(x, 1.0), kappa=1.0, ua=0.0, ub=0.0)
    exact = 0.5 * mesh.nodes * (1.0 - mesh.nodes)
    print(f"  FEM steady-heat max error = {np.max(np.abs(u_fem - exact)):.3e}")
    # high-order FD test
    h = mesh.nodes[1] - mesh.nodes[0]
    f = np.sin(PI * mesh.nodes)
    df_num = compact_diff1(f, h)
    df_ex = PI * np.cos(PI * mesh.nodes)
    err = np.max(np.abs(df_num[2:-2] - df_ex[2:-2]))
    print(f"  compact D1 error on sin(pi x) = {err:.3e}")
    # banded LU
    N = 50
    A = BandedMatrixR8GB(N, 2, 2)
    for i in range(N):
        for j in range(max(0, i - 2), min(N, i + 3)):
            A[i, j] = 5.0 if i == j else -1.0
    x = np.random.RandomState(0).randn(N)
    b = A.mv(x)
    xh = A.solve(b)
    print(f"  banded solve rel. error = {np.max(np.abs(x - xh)) / np.max(np.abs(x)):.3e}")
    # radial D2 operator
    op = assemble_radial_d2_operator(61, h, kappa=1.0, kperp2_rho2=0.1, Gamma0=0.9)
    print(f"  radial D2 operator bandwidth = ({op.ml}, {op.mu})")
    # triangulation (poloidal)
    tri = TriangularPoloidalMesh.annular(R0=3.0, a=1.0, nr=3, nt=8)
    print(f"  poloidal triangulation: {tri.triangles.shape[0]} triangles, "
          f"{(tri.neighbours >= 0).sum()} neighbour pairs")
    return mesh


def step_2_velocity_space() -> dict:
    _section("Step 2 -- Velocity-space quadrature, Bessel, CVT grid")
    # J_n and zeros
    print(f"  J_0(0) = {jn_eval(0, 0.0):.6f}")
    print(f"  J_1(3.8) = {jn_eval(1, 3.8):.6f}")
    zeros0 = jn_zeros(0, 5)
    print(f"  first 5 zeros of J_0 = {zeros0}")
    zeros1 = jn_zeros(1, 3)
    print(f"  first 3 zeros of J_1 = {zeros1}")
    # Gauss-Patterson
    nodes, weights = patterson_rule_ab(7, -6.0, 6.0)
    print(f"  Gauss-Patterson 7-pt on [-6,6], sum w = {weights.sum():.6f} (exact = 12)")
    # moment integration of a Maxwellian
    def gauss(vp, mu):
        return np.exp(-(vp * vp + mu))
    mom = velocity_moments(gauss, v_par_max=6.0, mu_max=9.0,
                            order_par=7, order_mu=7)
    print(f"  velocity moments of Gaussian:  n1 = {mom['n1']:.4e},  upar = {mom['upar']:.4e}")
    # CVT
    z = cvt_1d_nonuniform(n_generators=15, n_steps=40, density="maxwellian")
    print(f"  CVT maxwellian generators = {np.round(z, 3)}")
    return {"j0_zeros": zeros0, "cvt": z}


def step_3_gyroaverage_and_collisions() -> None:
    _section("Step 3 -- FLR operators and collision matrix")
    b = np.linspace(0.0, 3.0, 7)
    print(f"  b           = {np.round(b, 3)}")
    print(f"  Gamma0      = {np.round(gamma0(b), 5)}")
    print(f"  Gamma0 pade = {np.round(gamma0_pade(b), 5)}")
    print(f"  Gamma1      = {np.round(gamma1(b), 5)}")
    # apply J0 to a 2-D field
    v_par = np.linspace(-3, 3, 11)
    mu = np.linspace(0, 3, 7)
    f = np.exp(-(v_par[:, None] ** 2 + mu[None, :]))
    J0f = apply_j0_to_field(f, v_par, mu, kperp_rho=0.5)
    print(f"  max |J0 f - f| at mu=0 = {np.max(np.abs(J0f[:, 0] - f[:, 0])):.3e}")
    # collision frequency
    nu = collision_frequency(
        n_b=1.0e20, T_a_eV=10.0e3, m_a_amu=2.0, Z_a=1.0, Z_b=1.0,
        n_e_m3=1.0e20, T_e_eV=10.0e3,
    )
    print(f"  nu_ii (ITER-like) = {nu:.3e} s^-1")
    # Toeplitz collision matrix
    C = CollisionMatrix(N_xi=21, Lmax=6, nu_D=1.0)
    g = np.exp(-C.xi * C.xi)
    g_new = C.solve_shifted(0.01, g)
    print(f"  (I - 0.01 C) solve: max |C x - rhs| = {np.max(np.abs(C.mv(g_new) - g + 0.01 * C.mv(g))):.3e}")
    # Lorentz eigenvalues
    L = LorentzOperator(Lmax=6)
    print(f"  Lorentz eigenvalues (unscaled) = {L.eigenvalues_unscaled}")


def step_4_time_integrator() -> dict:
    _section("Step 4 -- Time integrators and stability regions")
    # Trapezoidal on stiff test problem y' = -10 y
    t, y = ode_trapezoidal(lambda t, y: -10.0 * y, 0.0, 1.0, np.array([1.0]), 100)
    exact = np.exp(-10.0 * t)
    print(f"  trapezoidal on y' = -10 y, max error = {np.max(np.abs(y[:, 0] - exact)):.3e}")
    # SSP-RK3 on a mild problem
    y0 = np.array([1.0])
    rhs = lambda t, y: -y
    y_rk = y0.copy()
    for _ in range(200):
        y_rk = ssp_rk3_step(rhs, 0.0, y_rk, 0.01)
    print(f"  SSP-RK3 on y' = -y, t=2.0, solution = {y_rk[0]:.6f}, "
          f"exact = {math.exp(-2.0):.6f}")
    # stability region boundaries
    xs_trap, ys_trap = stability_region_boundary("trapezoidal")
    print(f"  trapezoidal stability Re range = [{xs_trap.min():.3f}, {xs_trap.max():.3f}]")
    xs_rk3, ys_rk3 = stability_region_boundary("ssp_rk3")
    print(f"  SSP-RK3       stability Re range = [{xs_rk3.min():.3f}, {xs_rk3.max():.3f}]")
    # IMEX on a mildly stiff test system
    state = IMEXTrapezoidalState(t=0.0, y=np.array([1.0, 0.0]))
    for _ in range(50):
        state = imex_trapezoidal_step(
            rhs_explicit=lambda t, y: np.array([-0.5 * y[0] + y[1], -y[1]]),
            solve_implicit=lambda y0, rhs: np.array([rhs[0] / (1.0 + 0.5 * 0.05),
                                                    rhs[1] / (1.0 + 0.05)]),
            state=state,
            dt=0.05,
        )
    print(f"  IMEX final state = {state.y}, steps = {state.step}")
    return {"final_state": state.y}


def step_5_linear_itg() -> dict:
    _section("Step 5 -- Slab ITG eigenvalue problem")
    cfg = SlabITGConfig(
        N_radial=32, N_vpar=16, N_balloon=2,
        k_y_rho=0.3, k_z_qR=0.3,
        L_Ti_over_L_n=5.0, tau_e=1.0, q_safety=2.0,
        epsilon=0.3, L_s=20.0,
    )
    omega, growth = solve_slab_itg(cfg)
    print("  top 5 eigenvalues (Re, Im):")
    for w in omega[:5]:
        print(f"    omega = {w.real:+.5f} + {w.imag:+.5f} j")
    print(f"  max growth rate = {growth.max():+.5f}")
    return {"top_omega": omega[:5].tolist(), "max_growth": float(growth.max())}


def step_6_itg_threshold() -> dict:
    _section("Step 6 -- ITG threshold search in R/L_Ti")
    cfg = SlabITGConfig(
        N_radial=24, N_vpar=12, N_balloon=2,
        k_y_rho=0.3, k_z_qR=0.3,
        L_Ti_over_L_n=3.0, tau_e=1.0,
    )
    thr, info = find_itg_threshold(cfg, rl_ti_lo=1.0, rl_ti_hi=10.0, tol=0.2, max_iter=8)
    print(f"  approximate ITG threshold (R/L_Ti)_c ~ {thr:.3f}")
    print(f"  gamma(lo) = {info['gamma_lo']:+.4f},  gamma(hi) = {info['gamma_hi']:+.4f}")
    return {"threshold_rl_ti": thr, "info": info}


def step_7_zonal_flow() -> dict:
    _section("Step 7 -- Rosenbluth-Hinton zonal-flow residual")
    params = ZonalFlowParams(tau_e=1.0, q_safety=2.0, epsilon=0.3, k_x_rho=0.15)
    print(f"  Gamma0(params) = {params.Gamma0:.4f},  b = {params.b:.4f}")
    print(f"  analytic residual (circular limit) = {params.analytic_residual:.4f}")
    mesh = RadialMesh1D.uniform(0.0, 1.0, 61)
    phi0 = np.sin(PI * mesh.nodes) + 0.3 * np.sin(3.0 * PI * mesh.nodes)
    phi_res, rhs, ratio = compute_zonal_flow_residual(mesh, phi0, params)
    print(f"  residual ratio |phi_res|/|phi_0| = {ratio:.4f}")
    t, hist = relax_zonal_flow(mesh, phi0, params, t_max=20.0, n_steps=100)
    amp = np.max(np.abs(hist), axis=1)
    print(f"  amplitude t=0: {amp[0]:.4f}   t=t_max: {amp[-1]:.4f}")
    return {
        "analytic_residual": params.analytic_residual,
        "measured_residual": ratio,
        "phi0_amp": float(amp[0]),
        "phi_res_amp": float(amp[-1]),
    }


def step_8_synthetic_turbulence() -> dict:
    _section("Step 8 -- Synthetic turbulence time series and diagnostics")
    rng = np.random.default_rng(123)
    Nt = 2000
    Nx = 60
    t = np.linspace(0.0, 50.0, Nt)
    # build a simple "turbulent" signal as a sum of damped oscillators + noise
    Q = np.zeros(Nt)
    for (f0, amp, tau) in [(0.3, 1.0, 30.0), (0.7, 0.4, 20.0), (1.2, 0.2, 15.0)]:
        Q += amp * np.exp(-t / tau) * np.sin(2.0 * PI * f0 * t)
    # inject intermittent bursts (heavy tail)
    n_bursts = 20
    burst_idx = rng.integers(100, Nt - 100, size=n_bursts)
    for idx in burst_idx:
        burst = 3.0 * rng.standard_normal()
        width = 8
        Q[idx:idx + width] += burst * np.exp(-np.arange(width) / 3.0)
    Q += 0.05 * rng.standard_normal(Nt)

    phi = np.outer(np.sin(2.0 * PI * t / 10.0),
                   np.exp(-np.linspace(0, 3, Nx) ** 2)) + 0.1 * rng.standard_normal((Nt, Nx))

    d = run_diagnostics(phi, Q, Lx=1.0, smooth_window=7, smooth_order=3)
    stats = d["heat_flux_stats"]
    alpha, b = d["stretched_exp_fit"]
    print(f"  heat-flux mean     = {stats['mean']:+.4f}")
    print(f"  heat-flux std      = {stats['std']:.4f}")
    print(f"  heat-flux skewness = {stats['skewness']:.4f}")
    print(f"  heat-flux kurtosis = {stats['kurtosis']:.4f}")
    print(f"  stretched-exp (alpha, b) = ({alpha:.3f}, {b:.3f})")
    print(f"  noisy diagnostic shape = {d['noisy_Q'].shape}")
    print(f"  zonal phi shape = {d['zonal_phi'].shape}")
    return {
        "stats": stats,
        "stretched_exp_fit": {"alpha": alpha, "b": b},
        "n_t": Nt,
        "n_bursts": n_bursts,
    }


def step_9_phase_space_io(tmpdir: str) -> dict:
    _section("Step 9 -- 5-D phase-space I/O check")
    data = np.random.RandomState(0).randn(3, 3, 3, 7, 5).astype(np.float64) * 1.0e-3
    path = os.path.join(tmpdir, "phase_space.bin")
    PhaseSpaceIO.save(path, data, meta={"test": True, "units": "normalised"})
    data2, meta = PhaseSpaceIO.load(path)
    err = np.max(np.abs(data - data2))
    print(f"  phase-space round-trip max error = {err:.3e}")
    print(f"  meta = {meta}")
    return {"io_roundtrip_error": float(err)}


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    t0 = time.time()
    print("=" * 72)
    print("  PROJECT 289 -- 1D slab gyrokinetic turbulence simulator")
    print("  Domain:  计算等离子体 / 湍流输运 / gyrokinetic 模拟")
    print("           高阶有限差分 + 稳定性分析 (小规模可复现实验)")
    print("=" * 72)

    tmpdir = os.path.join(_HERE, "_run_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    summary: Dict[str, object] = {}
    summary["equilibrium"] = {}
    try:
        eq = step_0_equilibrium()
        summary["equilibrium"] = {
            "c_s": eq.c_s,
            "omega_ci": eq.omega_ci,
            "rho_s": eq.rho_s,
            "L_n": eq.L_n(),
            "L_Ti": eq.L_Ti(),
            "R_LTi": eq.R_LTi(),
            "R_LTc": eq.R_LTc(),
        }
    except Exception as e:
        print(f"  !! step 0 failed: {e!r}")
        summary["equilibrium"]["error"] = repr(e)

    try:
        step_1_mesh_and_fd()
    except Exception as e:
        print(f"  !! step 1 failed: {e!r}")
        summary["mesh_fd"] = {"error": repr(e)}

    try:
        step_2_velocity_space()
    except Exception as e:
        print(f"  !! step 2 failed: {e!r}")
        summary["velocity_space"] = {"error": repr(e)}

    try:
        step_3_gyroaverage_and_collisions()
    except Exception as e:
        print(f"  !! step 3 failed: {e!r}")
        summary["gyroaverage_collisions"] = {"error": repr(e)}

    try:
        ti = step_4_time_integrator()
        summary["time_integrator"] = ti
    except Exception as e:
        print(f"  !! step 4 failed: {e!r}")
        summary["time_integrator"] = {"error": repr(e)}

    try:
        li = step_5_linear_itg()
        summary["linear_itg"] = li
    except Exception as e:
        print(f"  !! step 5 failed: {e!r}")
        summary["linear_itg"] = {"error": repr(e)}

    try:
        th = step_6_itg_threshold()
        summary["itg_threshold"] = th
    except Exception as e:
        print(f"  !! step 6 failed: {e!r}")
        summary["itg_threshold"] = {"error": repr(e)}

    try:
        zf = step_7_zonal_flow()
        summary["zonal_flow"] = zf
    except Exception as e:
        print(f"  !! step 7 failed: {e!r}")
        summary["zonal_flow"] = {"error": repr(e)}

    try:
        turb = step_8_synthetic_turbulence()
        summary["turbulence"] = turb
    except Exception as e:
        print(f"  !! step 8 failed: {e!r}")
        summary["turbulence"] = {"error": repr(e)}

    try:
        io = step_9_phase_space_io(tmpdir)
        summary["phase_space_io"] = io
    except Exception as e:
        print(f"  !! step 9 failed: {e!r}")
        summary["phase_space_io"] = {"error": repr(e)}

    summary["wall_time_seconds"] = time.time() - t0

    # write summary
    summary_path = os.path.join(_HERE, "results_summary.json")
    try:
        _save_summary(summary_path, summary)
        print()
        print(f"  summary written to  {summary_path}")
    except Exception as e:
        print(f"  !! could not write summary: {e!r}")

    # clean-up temp files
    try:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
    except Exception:
        pass

    _section("Pipeline complete")
    print(f"  wall-clock time: {summary['wall_time_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
