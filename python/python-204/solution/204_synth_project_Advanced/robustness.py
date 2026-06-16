"""
robustness.py
=============

Boundary checks, numerical robustness helpers, and unit-system
guard-rails shared across the Sobol pipeline.  In a global
sensitivity study the forward model is queried at thousands of
parameter combinations — many of them at the edges of the unit
hypercube where individual sub-routines are most likely to blow up.
This module collects defensive wrappers so that each call site does
not have to reinvent them.

Provided:

* ``safe_finite`` — replace non-finite values with a fallback.
* ``unit_clamp`` — clamp ``x`` into ``[lo, hi]`` with numerical
  slack.
* ``relative_error`` — relative error with floor.
* ``condition_monitor`` — track spectral condition numbers and emit
  warnings when a linear solve becomes ill-conditioned.
* ``mass_conservation_check`` — verify that the LV integration
  preserves the mass balance up to a tolerance.
* ``run_smoke_tests`` — self-contained test that exercises every
  module in the project; used both as a sanity check and as a
  demonstration of boundary behaviour.

All functions are pure and side-effect free (except for the optional
logging inside ``run_smoke_tests``).
"""

from __future__ import annotations

import math

import numpy as np


# =====================================================================
# Basic safe primitives
# =====================================================================
def safe_finite(x, fallback: float = 0.0) -> float:
    """Return ``x`` if finite, else ``fallback``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v):
        return fallback
    return v


def unit_clamp(x: float, lo: float = 0.0, hi: float = 1.0,
               slack: float = 1.0e-12) -> float:
    """Clamp ``x`` into ``[lo + slack, hi - slack]``."""
    if lo >= hi:
        raise ValueError("unit_clamp: lo must be < hi")
    return float(max(lo + slack, min(x, hi - slack)))


def relative_error(a: float, b: float, floor: float = 1.0e-14) -> float:
    """Relative error ``|a - b| / max(|b|, floor)``."""
    return abs(a - b) / max(abs(b), floor)


# =====================================================================
# Condition monitor
# =====================================================================
class ConditionMonitor:
    """Track condition numbers across linear solves.

    Usage::

        mon = ConditionMonitor()
        mon.log("polar_laplacian", A)
        mon.report()
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, name: str, M: np.ndarray) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            rcond = 0.0
        else:
            try:
                s = np.linalg.svd(M, compute_uv=False)
                if s.size == 0 or s[0] == 0.0:
                    rcond = 0.0
                else:
                    rcond = float(s[-1] / s[0])
            except np.linalg.LinAlgError:
                rcond = 0.0
        self.records.append(dict(name=name, rcond=rcond))

    def report(self) -> str:
        lines = ["Condition-number report:"]
        for rec in self.records:
            status = "OK" if rec['rcond'] > 1.0e-10 else "ILL"
            lines.append(f"  {rec['name']:<28s}  rcond = {rec['rcond']:.3e}  "
                         f"[{status}]")
        return "\n".join(lines)


# =====================================================================
# Mass conservation check (LV system)
# =====================================================================
def mass_conservation_check(A_traj: np.ndarray, B_traj: np.ndarray,
                            K_a: float, K_b: float,
                            tol: float = 0.5) -> dict:
    r"""Check the mass balance of the LV trajectories.

    The total "biomass" ``M(t) = A(t)/K_a + B(t)/K_b`` should not
    exceed ``2`` (twice the steady-state value) by more than ``tol``.
    Returns ``{'ok': bool, 'max_M': float, 'max_excess': float}``.
    """
    A_traj = np.atleast_1d(A_traj)
    B_traj = np.atleast_1d(B_traj)
    if A_traj.size != B_traj.size:
        raise ValueError("mass_conservation_check: trajectory length mismatch")
    M = A_traj / max(K_a, 1.0e-12) + B_traj / max(K_b, 1.0e-12)
    max_M = float(np.max(M))
    max_excess = max(0.0, max_M - 2.0)
    return dict(ok=max_excess <= tol, max_M=max_M, max_excess=max_excess)


# =====================================================================
# Smoke test (runs every module once)
# =====================================================================
def run_smoke_tests(verbose: bool = True) -> dict:
    """Run a minimal forward pass through every module.

    Returns a dict summarising the status of each step.  No
    exceptions are raised on individual step failures; instead the
    dict records ``{'status': 'ok' | 'fail', 'msg': str}``.
    """
    from r8col_utils import r8col_duplicates, dedupe_sample_matrix
    from matrix_kernels import mxm_tiled, gram_centered
    from elliptic_green import elliptic_fk, elliptic_ek, disk_green_value
    from chaotic_mixing import (chirikov_ftle, disk_sample,
                                effective_velocity, disk_monomial_integral)
    from reaction_kinetics import (LVParams, lotka_volterra_integrate,
                                   logistic_exact, steady_state)
    from gauss_seidel_coupled import gs_solve, build_polar_laplacian
    from quadrature_pce import (patterson_rule_1d, smolyak_sparse_grid,
                                dqrlss, pce_fit, pce_sobol_from_coefficients)
    from shearlet_surrogate import (shearlet_decompose_2d,
                                    ShearletSurrogate, gsm_fit)
    from langevin_inversion import sobol_posterior_sampler
    from spatial_ops import (triangulation_histogram, disk_mesh_polar,
                             param_to_disk, build_sensitivity_graph)
    from sobol_indices import sobol_full_analysis
    from forward_model import forward_model

    results = {}

    def step(name, fn):
        try:
            fn()
            results[name] = dict(status="ok", msg="")
            if verbose:
                print(f"  [ok]   {name}")
        except Exception as exc:
            results[name] = dict(status="fail", msg=str(exc))
            if verbose:
                print(f"  [fail] {name}: {exc}")

    rng = np.random.default_rng(0)

    step("r8col_utils", lambda: dedupe_sample_matrix(
        r8col_duplicates(3, 20, 7, rng=rng)))
    step("matrix_kernels", lambda: gram_centered(rng.standard_normal((30, 5))))
    step("elliptic_green", lambda: disk_green_value(1.0, 0.5, 0.3, 0.7))
    step("chaotic_mixing", lambda: effective_velocity(1.0, 1.2))
    step("disk_monomial_integral",
         lambda: disk_monomial_integral(2, 0, 1.0))
    p = LVParams()
    step("reaction_kinetics", lambda: lotka_volterra_integrate(p, n_steps=40))
    step("logistic_exact",
         lambda: logistic_exact(np.linspace(0, 1, 5), r=1.0, k=10.0, y0=2.0))
    step("steady_state", lambda: steady_state(p))
    step("gauss_seidel",
         lambda: gs_solve(2.0 * np.eye(8) - np.diag(np.ones(7), 1)
                          - np.diag(np.ones(7), -1), np.ones(8),
                          tol=1.0e-6, max_iter=400))
    step("patterson_rule", lambda: patterson_rule_1d(3))
    step("smolyak_sparse_grid", lambda: smolyak_sparse_grid(2, 2))
    step("dqrlss",
         lambda: dqrlss(rng.standard_normal((15, 4)), rng.standard_normal(15)))

    def pce_step():
        X = rng.standard_normal((2, 80))
        y = X[0] + 0.5 * X[1] + 0.1 * X[0] * X[1]
        c, alphas, _ = pce_fit(X, y, p=2)
        pce_sobol_from_coefficients(c, alphas, 2)

    step("pce_fit", pce_step)
    step("shearlet_decompose",
         lambda: shearlet_decompose_2d(rng.standard_normal((16, 16))))
    step("ShearletSurrogate",
         lambda: ShearletSurrogate(rng.standard_normal((16, 16)),
                                   keep_ratio=0.4))
    step("gsm_fit", lambda: gsm_fit(rng.standard_normal(400)))

    def langevin_step():
        s_hat = np.array([0.3, 0.2, 0.1, 0.5, 0.4, 0.3])
        cov = np.eye(6) * 0.01
        sobol_posterior_sampler(s_hat, cov, d=3, n_samples=50,
                                burn_in=20, h=1.0e-3, rng=rng)

    step("langevin_inversion", langevin_step)

    step("disk_mesh_polar", lambda: disk_mesh_polar(1.0, 4, 6))
    step("param_to_disk", lambda: param_to_disk(np.full(8, 0.5)))
    step("build_sensitivity_graph",
         lambda: build_sensitivity_graph(rng.random((4, 4)), 0.3))

    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    elems = np.array([[0, 1, 2], [0, 2, 3]])
    samples = rng.random((20, 2))
    step("triangulation_histogram",
         lambda: triangulation_histogram(nodes, elems, samples))

    def sobol_step():
        def f_simple(x):
            return float(x[0] + 0.5 * x[1] + 0.2 * x[2])
        sobol_full_analysis(f_simple, d=3, N=60, rng=rng)

    step("sobol_full_analysis", sobol_step)

    def fm_step():
        forward_model(np.full(8, 0.5))

    step("forward_model", fm_step)

    n_ok = sum(1 for r in results.values() if r['status'] == "ok")
    results["_summary"] = f"{n_ok}/{len(results)} steps passed"
    if verbose:
        print(results["_summary"])
    return results


# =====================================================================
if __name__ == "__main__":
    run_smoke_tests(verbose=True)
