# -*- coding: utf-8 -*-
"""
main.py  -- PROJECT 265
=======================
Unified, zero-argument driver for the heliospheric cosmic-ray
transport project.  It exercises every module in turn and prints
a compact diagnostic report.  No visualisation.

Usage::

    python main.py
"""
from __future__ import annotations
import math
import os
import sys
import time
import tempfile
import warnings
import numpy as np

# suppress benign numerical warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure package imports work both as a package and as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from cosmic_ray_physics import (          # noqa: E402
    AU, c_light, q_e, m_p,
    parker_imf, kinetic_to_rigidity, velocity_from_Ek, lorentz_factor,
    quasilinear_kappa_parallel, lis_proton_vladimir2015,
    force_field_modulation, rigidity_larmor, year_seconds,
)
import heliocentric_mesh as hm            # noqa: E402
import high_order_stencils as hos         # noqa: E402
import stability_analysis as sa           # noqa: E402
import focused_transport_eq as fte        # noqa: E402
import multigrid_fas as mg                # noqa: E402
import analytical_benchmarks as ab        # noqa: E402
import stochastic_parker as sp            # noqa: E402
import inverse_kappa as ik                # noqa: E402
import dispersion_roots as dr             # noqa: E402
import turbulence_mcmc as tm              # noqa: E402
import species_composition as sc          # noqa: E402
import grid_data_io as gdio               # noqa: E402


# =====================================================================
# Banner
# =====================================================================
def banner(msg: str) -> None:
    line = "=" * 64
    print("\n" + line)
    print(msg)
    print(line)


# =====================================================================
# Experiment 1: mesh + Parker IMF
# =====================================================================
def experiment_mesh_and_imf() -> dict:
    banner("EXPERIMENT 1 -- heliospheric mesh & Parker IMF")
    r = hm.logarithmic_radial_mesh(Nr=48)
    mu = hm.pitch_angle_mesh(Nmu=16)
    print(f"  radial grid : Nr = {r.size}, "
          f"r_min = {r[0] / AU:.3f} AU, r_max = {r[-1] / AU:.1f} AU")
    print(f"  mu grid     : Nmu = {mu.size}, "
          f"min = {mu[0]:+.4f}, max = {mu[-1]:+.4f}")
    for rau in (0.1, 1.0, 10.0, 50.0):
        Br, BT, Bmag, psi = parker_imf(rau * AU)
        print(f"  r = {rau:5.1f} AU: |B| = {Bmag * 1e9:7.3f} nT, "
              f"psi = {math.degrees(psi):5.1f} deg")
    return {"r": r, "mu": mu}


# =====================================================================
# Experiment 2: high-order stencil benchmark
# =====================================================================
def experiment_stencils() -> dict:
    banner("EXPERIMENT 2 -- high-order finite-difference stencils")
    # smooth test function on a non-uniform radial grid
    r = hm.logarithmic_radial_mesh(Nr=64)
    f = np.sin(2.0 * math.pi * (r - r[0]) / (r[-1] - r[0]))
    exact = (2.0 * math.pi / (r[-1] - r[0])) * np.cos(
        2.0 * math.pi * (r - r[0]) / (r[-1] - r[0]))
    dx = np.diff(r)
    df_up1 = hos.upwind_first_derivative(f, dx, speed=1.0)
    df_up2 = hos.upwind_second_order(f, dx, speed=1.0)
    df_c2 = hos.central_second_derivative(
        np.zeros_like(r), dx)  # placeholder: used for d2f only
    df_cp4 = hos.compact4_first_derivative(f, dx)
    df_w5 = hos.weno5_first_derivative(f, dx)
    err_up1 = float(np.max(np.abs(df_up1[2:-2] - exact[2:-2])))
    err_up2 = float(np.max(np.abs(df_up2[2:-2] - exact[2:-2])))
    err_cp4 = float(np.max(np.abs(df_cp4[2:-2] - exact[2:-2])))
    err_w5 = float(np.max(np.abs(df_w5[2:-2] - exact[2:-2])))
    print(f"  upwind1 max error = {err_up1:.3e}")
    print(f"  upwind2 max error = {err_up2:.3e}")
    print(f"  compact4 max error = {err_cp4:.3e}")
    print(f"  WENO5 max error    = {err_w5:.3e}")
    return {"err_up1": err_up1, "err_cp4": err_cp4, "err_w5": err_w5}


# =====================================================================
# Experiment 3: stability analysis
# =====================================================================
def experiment_stability(r: np.ndarray, mu: np.ndarray) -> dict:
    banner("EXPERIMENT 3 -- von Neumann & CFL stability analysis")
    Ek = 1.0e9 * q_e
    R = kinetic_to_rigidity(Ek)
    v = velocity_from_Ek(Ek)
    B = parker_imf(AU)[2]
    kpar = quasilinear_kappa_parallel(R, B, 0.3, 1.0e8, v)
    # von Neumann at CFL = 0.8
    k_wave = np.linspace(0.0, math.pi, 64)
    dx = float(np.mean(np.diff(r)))
    dt = 0.8 * dx / (abs(v) + 4.0e5)
    for sch in ("upwind1", "central2", "compact4", "weno5"):
        G = sa.von_neumann_amplification(k_wave, dx, v, kpar, dt,
                                         scheme=sch)
        print(f"  scheme={sch:9s}  max|G| = {np.max(np.abs(G)):.4f}")
    # matrix stability
    Krr = np.zeros((r.size, mu.size))
    for j, muj in enumerate(mu):
        Krr[:, j] = kpar * muj * muj + 0.02 * kpar * math.sin(
            parker_imf(AU)[3]) ** 2
    Dmu = 1.0e-4 * np.ones((r.size, mu.size))
    info = sa.matrix_stability_test(r, mu, Krr, Dmu, v, scheme="upwind1")
    print(f"  spectral radius  = {info['spectral_radius']:.3e}")
    print(f"  max dt (euler)   = {info['max_dt_euler']:.3e} s")
    dt_cfl = sa.explicit_cfl_timestep(r, mu, Krr, Dmu, v + 4.0e5)
    print(f"  CFL dt           = {dt_cfl:.3e} s")
    # critical rigidity
    R_c = sa.critical_rigidity(dx, 4.0e5, v, B)
    print(f"  critical R       = {R_c / 1e9:.3e} GV")
    return {"rho": info["spectral_radius"], "dt_cfl": dt_cfl}


# =====================================================================
# Experiment 4: focused transport forward integration
# =====================================================================
def experiment_focused_transport(r: np.ndarray,
                                 mu: np.ndarray) -> dict:
    banner("EXPERIMENT 4 -- forward integration (focused transport)")
    Ek = 1.0e9 * q_e
    R = kinetic_to_rigidity(Ek)
    op = fte.FocusedTransportOperator(
        r, mu, Ek, stencil_order="upwind2")
    f = np.ones((r.size, mu.size)) * 1.0
    hm.apply_all_boundaries(f, np.ones(mu.size) * 1.0,
                            np.zeros(mu.size))
    dt = op.cfl_timestep() * 0.5
    n_steps = min(80, max(int(1.0 / max(dt, 1.0)), 20))
    t0 = time.time()
    for step in range(n_steps):
        dfdt = op(f)
        f = f + dt * dfdt
        hm.apply_all_boundaries(f, np.ones(mu.size),
                                np.zeros(mu.size))
    wall = time.time() - t0
    omni = np.sum(f * np.diff(np.concatenate(
        [[mu[0] - 0.5 * (mu[1] - mu[0])],
         0.5 * (mu[:-1] + mu[1:]),
         [mu[-1] + 0.5 * (mu[-1] - mu[-2])]])), axis=1)
    print(f"  integration: {n_steps} steps in {wall:.2f} s")
    print(f"  f at 1 AU (central mu): {f[r.size // 3, mu.size // 2]:.3e}")
    print(f"  omnidirectional flux max: {omni.max():.3e}")
    return {"f": f, "omni": omni, "dt": dt, "n_steps": n_steps}


# =====================================================================
# Experiment 5: multigrid steady-state
# =====================================================================
def experiment_multigrid(r: np.ndarray, mu: np.ndarray) -> dict:
    banner("EXPERIMENT 5 -- FAS multigrid steady-state solver")
    kappa = 1.0e22 * (r / AU) ** 0.3
    Q = np.zeros((r.size, mu.size))
    src = np.exp(-((r - AU) ** 2) / (0.3 * AU) ** 2)
    for j in range(mu.size):
        Q[:, j] += src * 1.0e-10
    Q[0, :] = 1.0
    Q[-1, :] = 0.0
    solver = mg.FASMultigrid(r, mu, kappa, Q, nu1=2, nu2=2,
                             max_cycles=15, tol=1.0e-4)
    f, info = solver.solve()
    print(f"  converged: {info['converged']} in {info['cycles']} cycles")
    print(f"  residual history (first/last 3): "
          f"{info['residual_history'][:3]} ... "
          f"{info['residual_history'][-3:]}")
    return {"f_mg": f, "info": info}


# =====================================================================
# Experiment 6: analytical benchmarks
# =====================================================================
def experiment_analytical() -> dict:
    banner("EXPERIMENT 6 -- analytical benchmarks")
    # force-field modulation
    Ek_GeV = np.logspace(-1, 2, 16)
    J_mod = ab.force_field_model(Ek_GeV, phi_MV=500.0)
    print(f"  force-field J(Ek) max: {J_mod.max():.3e}")
    # steady convection-diffusion
    r = np.linspace(0.0, 1.0, 64)
    f_steady = ab.convection_diffusion_steady(r, 4.0e5, 1.0e22, 1.0, 1.0)
    print(f"  conv-diff f(0) = {f_steady[0]:.3e}, "
          f"f(L/2) = {f_steady[32]:.3e}")
    # Barenblatt residual
    b = ab.BarenblattCosmicRay(m=2.0, D0=1.0)
    rb = np.linspace(0.0, 1.0, 32)
    res = b.residual(rb, 1.0)
    print(f"  Barenblatt residual max: {np.max(np.abs(res)):.3e}")
    # Green's function
    gf = ab.greens_function_constant(rb, 0.5, 0.1, 0.01, 0.5)
    print(f"  Green's function peak: {gf.max():.3e}")
    return {"J_mod": J_mod}


# =====================================================================
# Experiment 7: stochastic propagation
# =====================================================================
def experiment_stochastic(r: np.ndarray, mu: np.ndarray) -> dict:
    banner("EXPERIMENT 7 -- stochastic Parker propagator")
    Ek = 1.0e9 * q_e
    prop = sp.StochasticParkerPropagator(
        r, mu, Ek, N_particles=60, dt=3.0e4, max_steps=100, seed=1)
    t0 = time.time()
    f = prop.solve_at(AU, mu[::2])
    wall = time.time() - t0
    print(f"  f(1 AU, mu) for {mu[::2].size} mu bins in {wall:.2f} s:")
    for j, muj in enumerate(mu[::2]):
        print(f"    mu = {muj:+.3f}: f = {f[j]:.3e}")
    return {"f_stoch": f}


# =====================================================================
# Experiment 8: inverse problem
# =====================================================================
def experiment_inverse(r: np.ndarray, mu: np.ndarray) -> dict:
    banner("EXPERIMENT 8 -- inverse diffusion problem (Adam+Polyak)")
    Ek = 1.0e9 * q_e
    kappa_true = lambda rr: 1.0e22 * (rr / AU) ** 0.3
    f_obs = ik.synthetic_observations(r, kappa_true, mu_idx=4,
                                      noise_rel=0.02, seed=3)
    theta0 = np.full(r.size, np.log(5.0e21))
    theta, theta_bar, hist = ik.solve_inverse(
        r, mu, Ek, f_obs, theta0=theta0,
        n_iter=3, n_steps=50, alpha=1.0e-2, verbose=True)
    kappa_inv = np.exp(theta_bar)
    print(f"  loss history: {[f'{h:.3e}' for h in hist]}")
    err = float(np.mean(
        (kappa_inv - np.array([kappa_true(ri) for ri in r])) ** 2
        / np.array([kappa_true(ri) ** 2 for ri in r])))
    print(f"  relative MSE on log kappa: {err:.3e}")
    return {"theta_bar": theta_bar, "loss_history": hist}


# =====================================================================
# Experiment 9: Lambert W, Brent, dispersion roots
# =====================================================================
def test_roots() -> dict:
    banner("EXPERIMENT 9 -- Lambert W & plasma dispersion roots")
    # Lambert W tests
    w0_1 = dr.lambert_w0(1.0)
    w0_neg = dr.lambert_w0(-0.1)
    wm1_neg = dr.lambert_wm1(-0.1)
    print(f"  W0(1)     = {w0_1:.8f}  (ref 0.56714329)")
    print(f"  W0(-0.1)  = {w0_neg:.8f}")
    print(f"  Wm1(-0.1) = {wm1_neg:.8f}")
    # Brent root
    x_pi = dr.brent_root(math.sin, 3.0, 3.5)
    print(f"  brent_root(sin, 3, 3.5) = {x_pi:.8f}  (ref pi = {math.pi:.8f})")
    # resonant k
    k_res = dr.find_resonant_k(1.0e9, 0.5, 5.0e-9)
    print(f"  k_res(1 GV, mu=0.5, B=5nT) = {k_res:.3e} 1/m")
    # escape-time inversion
    t_esc = dr.escape_time_lambert(1.0, 10.0)
    print(f"  escape_time(tau_d=1, tau_a=10) = {t_esc:.4f}")
    return {"w0_1": w0_1, "x_pi": x_pi}


# =====================================================================
# Experiment 10: MCMC turbulence
# =====================================================================
def experiment_mcmc(r: np.ndarray, mu: np.ndarray) -> dict:
    banner("EXPERIMENT 10 -- MCMC sampling of turbulence parameters")
    Ek = 1.0e9 * q_e
    theta_true = (1.7, 1.7, 1.0e8, 0.3)
    f_obs = tm.synthetic_obs_turbulence(r, mu, Ek, theta_true, seed=5)
    theta0 = np.array([1.6, 1.8, 5.0e7, 0.28])
    chain, ll, acc = tm.run_mcmc(theta0, r, mu, Ek, f_obs,
                                 n_steps=5, proposal_scale=0.05, seed=1)
    print(f"  acceptance rate: {acc:.2f}")
    print(f"  final log-likelihood: {ll[-1]:.3e}")
    print(f"  chain mean: {chain.mean(axis=0)}")
    return {"acceptance": acc}


# =====================================================================
# Experiment 11: species composition
# =====================================================================
def experiment_species() -> dict:
    banner("EXPERIMENT 11 -- CR species composition (urn + leaky box)")
    names = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]
    K = sc.source_abundance_vector(names)
    print("  source abundances (normalised):")
    for n, k in zip(names, K):
        print(f"    {n:>3s}: {k * 100:6.2f}%")
    obs = sc.monte_carlo_composition(names, N_total=200_000,
                                     N_escape=40_000,
                                     escape_length_g_cm2=10.0, seed=3)
    print("  observed fractions (after escape + spallation):")
    for n in names:
        print(f"    {n:>3s}: {obs[n] * 100:6.2f}%")
    lb = sc.leaky_box_composition(names, escape_length_g_cm2=10.0)
    print("  leaky-box equilibrium fractions:")
    for n, v in zip(names, lb):
        print(f"    {n:>3s}: {v * 100:6.2f}%")
    return {"obs": obs}


# =====================================================================
# Experiment 12: grid I/O and diagnostics
# =====================================================================
def experiment_io(f: np.ndarray, r: np.ndarray,
                  mu: np.ndarray) -> dict:
    banner("EXPERIMENT 12 -- grid data I/O & diagnostics")
    tmpdir = tempfile.mkdtemp(prefix="cr265_")
    prefix = os.path.join(tmpdir, "fd_run")
    node_p, val_p = gdio.write_fd_data(prefix, r, mu, f, t=1.0)
    out_p = gdio.fd_to_structured(prefix, prefix + "_struct",
                                   variable_names=["r", "mu", "f"])
    d = gdio.read_fd_data(prefix)
    print(f"  wrote {node_p} ({os.path.getsize(node_p)} bytes)")
    print(f"  wrote {val_p} ({os.path.getsize(val_p)} bytes)")
    print(f"  wrote {out_p} ({os.path.getsize(out_p)} bytes)")
    print(f"  read back {d['values'].shape[0]} rows")
    diag = gdio.summary_diagnostics(r, mu, f, Ek_GeV=1.0)
    return {"diag": diag}


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    t0 = time.time()
    print("==========================================================")
    print("PROJECT 265 -- Cosmic-Ray Transport in the Heliosphere")
    print("High-order finite differences & stability analysis")
    print("==========================================================")
    out = experiment_mesh_and_imf()
    r, mu = out["r"], out["mu"]
    experiment_stencils()
    experiment_stability(r, mu)
    ft_out = experiment_focused_transport(r, mu)
    mg_out = experiment_multigrid(r, mu)
    experiment_analytical()
    experiment_stochastic(r, mu)
    experiment_inverse(r, mu)
    test_roots()
    experiment_mcmc(r, mu)
    experiment_species()
    experiment_io(ft_out["f"], r, mu)
    print("\n" + "=" * 64)
    print(f"ALL EXPERIMENTS COMPLETED in {time.time() - t0:.2f} s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
