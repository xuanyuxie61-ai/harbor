#!/usr/bin/env python3
"""
main.py
=======

Zero-parameter entry point for the galaxy-formation hydrodynamics
simulation test problem.  Running this file executes the full pipeline:

  1. Pyramid-AMR grid construction
  2. Equation of state + characteristic-ellipse diagnostics
  3. Turbulent spectrum initialisation with Laguerre quadrature
  4. Von Neumann stability analysis of the FD+RK3 scheme
  5. Initial-condition builder (density profile + turbulence)
  6. Rational-knapsack AMR tagging
  7. Voronoi point-mass gravity diagnostic
  8. SSP-RK3 time integration (a few steps for demonstration)
  9. Shock quantization + bagged causal analysis

Usage
-----
    python main.py

All outputs are printed to stdout.  No parameters, no external data,
no plotting.  The simulation is intentionally kept small (16^3 periodic
grid, a handful of time steps) so that it runs in seconds on a laptop
while still exercising every component of the numerical machinery.

Author: PROJECT_248 synthesis
Domain: Computational Astrophysics - Galaxy Formation Hydrodynamics:
        High-Order Finite Differences and Stability Analysis
        (small-scale reproducible experiment)
"""

from __future__ import annotations
import sys
import os
import math
import time as _timer

# ensure the project root is importable when invoked from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _banner() -> None:
    print("=" * 76)
    print("  PROJECT_248 :: Galaxy Formation Hydrodynamics Test Problem")
    print("  Domain: high-order finite differences + stability analysis")
    print("  (small-scale reproducible experiment)")
    print("=" * 76)


def _import_chain() -> None:
    """Sanity check: import every module so syntax errors surface early."""
    import astro_constants          # noqa: F401
    import grid_geometry            # noqa: F401
    import eos_gas                  # noqa: F401
    import high_order_fd            # noqa: F401
    import weno_reconstruction      # noqa: F401
    import rk_integrator            # noqa: F401
    import cfl_stability            # noqa: F401
    import poisson_cholesky         # noqa: F401
    import von_neumann_analysis     # noqa: F401
    import turbulent_spectrum       # noqa: F401
    import duration_feedback        # noqa: F401
    import shock_quantization       # noqa: F401
    import voronoi_potential        # noqa: F401
    import adaptive_refinement      # noqa: F401
    import causal_analysis          # noqa: F401
    import simulation_core          # noqa: F401
    print("[ok] all 16 Python modules imported successfully")


def _fd_convergence_check() -> None:
    """Demonstrate the design-order convergence of the FD operators."""
    import numpy as np
    from high_order_fd import apply_fd1
    print("\n" + "-" * 72)
    print("  High-order FD convergence check: f(x) = sin(2 pi x) on [0, 1]")
    print("-" * 72)
    print("  N     dx           L_inf (order 4)   L_inf (order 6)   rate_4  rate_6")

    prev_err4 = None
    prev_err6 = None
    prev_dx = None
    for N in [32, 64, 128, 256]:
        x = np.linspace(0, 1, N, endpoint=False)
        f = np.sin(2.0 * math.pi * x)
        df_exact = 2.0 * math.pi * np.cos(2.0 * math.pi * x)
        dx = 1.0 / N
        df4 = apply_fd1(f, dx, order=4)
        df6 = apply_fd1(f, dx, order=6)
        err4 = float(np.max(np.abs(df4 - df_exact)))
        err6 = float(np.max(np.abs(df6 - df_exact)))
        rate4 = (math.log(prev_err4 / err4) / math.log(prev_dx / dx)
                 if prev_err4 and prev_err4 > 0 and prev_dx else 0.0)
        rate6 = (math.log(prev_err6 / err6) / math.log(prev_dx / dx)
                 if prev_err6 and prev_err6 > 0 and prev_dx else 0.0)
        print(f"  {N:4d}  {dx:10.3e}   {err4:15.3e}    {err6:15.3e}    "
              f"{rate4:6.2f}  {rate6:6.2f}")
        prev_err4, prev_err6, prev_dx = err4, err6, dx


def _weno_shock_test() -> None:
    """Demonstrate WENO5 reconstruction on a step function."""
    import numpy as np
    from weno_reconstruction import weno5_flux_array
    print("\n" + "-" * 72)
    print("  WENO5 reconstruction: step function f(x) = (x > 0.5)")
    print("-" * 72)
    N = 64
    x = np.linspace(0, 1, N, endpoint=False)
    f = np.where(x > 0.5, 1.0, 0.0)
    for method in ("j", "z", "hybrid"):
        f_rec = weno5_flux_array(f, method=method)
        overshoot = max(0.0, float(np.max(f_rec)) - 1.0)
        undershoot = max(0.0, -float(np.min(f_rec)))
        print(f"  method={method:7s}: max overshoot = {overshoot:.4f}, "
              f"max undershoot = {undershoot:.4f}")


def _main() -> int:
    t0 = _timer.perf_counter()
    _banner()
    _import_chain()
    _fd_convergence_check()
    _weno_shock_test()

    print("\n" + "=" * 76)
    print("  Running full simulation pipeline...")
    print("=" * 76)

    from simulation_core import GalaxyFormationSimulation
    sim = GalaxyFormationSimulation(
        N=8,
        box_kpc=10.0,
        max_level=1,
        total_time_myr=2.0,
        seed=42,
    )
    result = sim.run()

    t_final = _timer.perf_counter()
    print("\n" + "=" * 76)
    print(f"  Total wall time: {t_final - t0:.2f} s")
    print("  Pipeline exit: SUCCESS")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
