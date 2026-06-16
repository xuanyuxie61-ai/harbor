"""
analytical_benchmarks.py — Analytical benchmarks for the numerical pipeline
===========================================================================

Before trusting the numerical formation energy we validate every major
numerical kernel against an analytical solution. This module collects
those benchmarks:

  1. **Minimal-surface exact solutions** (from 768_minimal_surface_exact):
     the catenoid, helicoid, linear ramp, and Scherk surface all satisfy
     the minimal-surface PDE
         (1 + U_x²) U_yy − 2 U_x U_y U_xy + (1 + U_y²) U_xx = 0.
     We evaluate each exact solution and its analytical Laplacian on a
     uniform grid and compare against the high-order FD Laplacian.

  2. **Gaussian wave-packet imaginary-time decay** — analytical energy α
     vs numerical energy after propagation.

  3. **Eshelby circular inclusion** — analytical strain energy vs computed.

  4. **Quadrature exactness** — Gauss–Hermite on monomials.

  5. **LU solve residual** — from the Linpack-style solver.

Seed project integration:
  * 768_minimal_surface_exact/{catenoid,helicoid,scherk,linear}_*.m:
    closed-form solutions and their partial derivatives.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple


# -------------------------------------------------------------------------
# (1) Minimal-surface exact solutions (from 768_minimal_surface_exact)
# -------------------------------------------------------------------------
def catenoid_exact(X: np.ndarray, Y: np.ndarray, a: float = 1.0
                   ) -> Dict[str, np.ndarray]:
    """Catenoid minimal surface U(x, y) = (1/a) acosh(a r), r = √(x² + y²).

    Returns U, U_x, U_y, U_xx, U_xy, U_yy analytically.
    Port of `minimal_surface_catenoid_exact.m`.

    The catenoid is only defined for a r ≥ 1 (the "neck" radius is 1/a).
    We clamp r to avoid evaluating arccosh for a r < 1, and the residual
    will only be reliable away from the neck.
    """
    R = np.sqrt(X * X + Y * Y + 1e-20)
    aR = np.maximum(a * R, 1.0 + 1e-8)    # enforce a r ≥ 1
    U = np.arccosh(aR) / a
    dU_dR = 1.0 / (a * np.sqrt(aR ** 2 - 1.0))
    Rx = X / R
    Ry = Y / R
    Ux = dU_dR * Rx
    Uy = dU_dR * Ry
    d2U_dR2 = -aR / (a * (aR ** 2 - 1.0) ** 1.5)
    Uxx = d2U_dR2 * Rx ** 2 + dU_dR * (1.0 / R - X ** 2 / R ** 3)
    Uyy = d2U_dR2 * Ry ** 2 + dU_dR * (1.0 / R - Y ** 2 / R ** 3)
    Uxy = d2U_dR2 * Rx * Ry + dU_dR * (-X * Y / R ** 3)
    return dict(U=U, Ux=Ux, Uy=Uy, Uxx=Uxx, Uxy=Uxy, Uyy=Uyy)


def helicoid_exact(X: np.ndarray, Y: np.ndarray
                   ) -> Dict[str, np.ndarray]:
    """Helicoid minimal surface U(x, y) = atan2(y, x).

    Port of `minimal_surface_helicoid_exact.m`. Note that ∇² U = 0 away
    from the origin (harmonic function), so the minimal-surface PDE is
    satisfied trivially.
    """
    U = np.arctan2(Y, X)
    R2 = X * X + Y * Y + 1e-20
    Ux = -Y / R2
    Uy = X / R2
    Uxx = 2.0 * X * Y / R2 ** 2
    Uyy = -2.0 * X * Y / R2 ** 2
    Uxy = (Y * Y - X * X) / R2 ** 2
    return dict(U=U, Ux=Ux, Uy=Uy, Uxx=Uxx, Uxy=Uxy, Uyy=Uyy)


def scherk_exact(X: np.ndarray, Y: np.ndarray
                 ) -> Dict[str, np.ndarray]:
    """Scherk's first surface U(x, y) = ln(cos(y) / cos(x)).

    Port of `minimal_surface_scherk_exact.m`. Defined for |x|, |y| < π/2.
    """
    cx = np.cos(np.clip(X, -1.5, 1.5))
    cy = np.cos(np.clip(Y, -1.5, 1.5))
    U = np.log(np.maximum(cy, 1e-12) / np.maximum(cx, 1e-12))
    tx = np.tan(np.clip(X, -1.5, 1.5))
    ty = np.tan(np.clip(Y, -1.5, 1.5))
    Ux = tx
    Uy = -ty
    Uxx = 1.0 / np.cos(np.clip(X, -1.5, 1.5)) ** 2
    Uyy = -1.0 / np.cos(np.clip(Y, -1.5, 1.5)) ** 2
    Uxy = np.zeros_like(X)
    return dict(U=U, Ux=Ux, Uy=Uy, Uxx=Uxx, Uxy=Uxy, Uyy=Uyy)


def linear_exact(X: np.ndarray, Y: np.ndarray,
                 ax: float = 1.0, by: float = 0.5, c: float = 0.0
                 ) -> Dict[str, np.ndarray]:
    """Linear surface U(x, y) = ax + by + c (trivially minimal).

    Port of `minimal_surface_linear_exact.m`.
    """
    U = ax * X + by * Y + c
    return dict(U=U, Ux=np.full_like(X, ax), Uy=np.full_like(X, by),
                Uxx=np.zeros_like(X), Uxy=np.zeros_like(X),
                Uyy=np.zeros_like(X))


# -------------------------------------------------------------------------
# (2) Minimal-surface PDE residual
# -------------------------------------------------------------------------
def minimal_surface_residual(U_dict: Dict[str, np.ndarray],
                             mask: np.ndarray = None) -> float:
    """Compute the residual of the minimal-surface PDE:
        R = (1 + U_x²) U_yy − 2 U_x U_y U_xy + (1 + U_y²) U_xx
    The maximum of |R| should be ≈ 0 for exact solutions.
    If `mask` is provided, only points where mask is True are considered.
    """
    Ux, Uy = U_dict["Ux"], U_dict["Uy"]
    Uxx, Uxy, Uyy = U_dict["Uxx"], U_dict["Uxy"], U_dict["Uyy"]
    R = ((1.0 + Ux ** 2) * Uyy - 2.0 * Ux * Uy * Uxy
         + (1.0 + Uy ** 2) * Uxx)
    if mask is not None:
        R = R[mask]
    return float(np.max(np.abs(R)))


def all_minimal_surface_residuals(N: int = 32, L: float = 2.0
                                  ) -> Dict[str, float]:
    """Evaluate PDE residuals for all four minimal surfaces on a grid."""
    xs = np.linspace(-L / 2, L / 2, N)
    ys = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X * X + Y * Y)
    res = {}
    # catenoid: only evaluate where a*R > 1.5 to stay clear of the neck
    a_cat = 1.2
    d = catenoid_exact(X, Y, a=a_cat)
    mask = (a_cat * R) > 1.5
    res["catenoid"] = minimal_surface_residual(d, mask=mask)
    # others: defined everywhere
    for (name, fn, kw) in [
        ("helicoid", helicoid_exact, {}),
        ("scherk", scherk_exact, {}),
        ("linear", linear_exact, {}),
    ]:
        d = fn(X, Y, **kw)
        res[name] = minimal_surface_residual(d)
    return res


# -------------------------------------------------------------------------
# (3) Composite benchmark summary
# -------------------------------------------------------------------------
def run_all_benchmarks() -> Dict[str, dict]:
    """Run every analytical benchmark and return a nested result dict."""
    import high_order_fd as hfd
    import stability_analysis as sa
    import eshelby_strain as esh
    import linear_solver as ls

    results = {}

    # stencil exactness
    for order in (2, 4, 6, 8):
        ok = hfd.verify_stencil_exactness(order)
        results[f"stencil_exactness_order{order}"] = {"pass": ok}

    # catenoid residual vs FD order
    for order in (2, 4, 6, 8):
        res = hfd.catenoid_laplacian_residual(N=64, a=1.2, order=order)
        results[f"catenoid_residual_order{order}"] = {"max_residual": res}

    # minimal-surface PDE residuals
    results["minimal_surfaces"] = all_minimal_surface_residuals()

    # Gaussian validation
    E_num, E_exact = sa.validate_gaussian()
    results["gaussian_validation"] = {
        "numerical_energy": E_num, "exact_energy": E_exact,
        "rel_error": abs(E_num - E_exact) / max(abs(E_exact), 1e-15),
    }

    # Eshelby validation
    E_num_e, E_ex_e = esh.validate_eshelby_circular()
    results["eshelby_circular"] = {
        "numerical": E_num_e, "exact": E_ex_e,
        "rel_error": abs(E_num_e - E_ex_e) / max(abs(E_ex_e), 1e-15),
    }
    K0, E0, Kh, Eh = esh.validate_elliptic_K_E()
    results["elliptic_K_E"] = {
        "K(0)": K0, "E(0)": E0, "K(1/2)": Kh, "E(1/2)": Eh,
        "K(0)_exact": math.pi / 2, "E(0)_exact": math.pi / 2,
    }

    # Linpack LU solver residual
    n_test = 24
    A, x_sol, b = ls.make_test_system(n_test, seed=276)
    x_computed = ls.solve_lu(A, b)
    res_norm = ls.residual_norm(A, b, x_computed)
    results["lu_solver"] = {
        "n": n_test,
        "residual_inf_norm": res_norm,
        "solution_error": float(np.max(np.abs(x_computed - x_sol))),
    }

    return results
