"""
grad_shafranov.py — Top-level Grad-Shafranov solver that couples the sparse
operator, the plasma profiles and the nonlinear iteration.

The equilibrium ψ(R,Z) satisfies

    Δ* ψ = -μ₀ R² p'(ψ) - F(ψ) F'(ψ)                     (GS)

with boundary conditions ψ = 0 on ∂Ω (we take ∂Ω outside the LCFS so this is
an excellent approximation) and a normalisation ψ_axis = 1 after the solve.
"""

from __future__ import annotations
import math
try:
    from .tokamak_geometry import TokamakGeometry
    from .plasma_profiles import GSProfiles
    from .sparse_operators import gs_operator_crs, gs_operator_compact_crs, CRSMatrix
    from .nonlinear_solver import picard_solve, pseudo_time_solve, broyden_solve, SolverState
except ImportError:
    from tokamak_geometry import TokamakGeometry
    from plasma_profiles import GSProfiles
    from sparse_operators import gs_operator_crs, gs_operator_compact_crs, CRSMatrix
    from nonlinear_solver import picard_solve, pseudo_time_solve, broyden_solve, SolverState


MU0 = 4.0 * math.pi * 1e-7   # vacuum permeability [H/m]


def build_source(psi_flat: list[float], geom: TokamakGeometry,
                 profiles: GSProfiles, psi_axis: float, psi_bnd: float) -> list[float]:
    """Build the right-hand side S(ψ) = -μ₀ R² p'(ψ_n) - F F'(ψ_n)
    where ψ_n = (ψ - ψ_bnd)/(ψ_axis - ψ_bnd) ∈ [0,1]."""
    Nr, Nz = geom.Nr, geom.Nz
    Rg = geom.R_grid()
    dpsi = psi_axis - psi_bnd
    if abs(dpsi) < 1e-12:
        dpsi = 1e-12
    S = [0.0] * (Nr * Nz)
    for i in range(Nr):
        R = Rg[i]
        for j in range(Nz):
            k = i * Nz + j
            if i == 0 or i == Nr - 1 or j == 0 or j == Nz - 1:
                S[k] = 0.0
                continue
            psin = (psi_flat[k] - psi_bnd) / dpsi
            psin = max(0.0, min(1.0, psin))
            # Clip sources outside the plasma (ψ_n < 0) to avoid unphysical values
            if psin <= 0.0:
                S[k] = 0.0
                continue
            src = -MU0 * R * R * profiles.pp(psin) - profiles.ffp(psin)
            S[k] = src
    return S


def solve_grad_shafranov(geom: TokamakGeometry, profiles: GSProfiles,
                         method: str = "picard",
                         compact: bool = False,
                         tol: float = 1e-8,
                         maxiter: int = 300) -> dict:
    """Solve GS and return a dict with 'psi' (2-D list), 'axis', 'state'."""
    if compact:
        A = gs_operator_compact_crs(geom.Nr, geom.Nz, geom.dR, geom.dZ, geom.R_min)
    else:
        A = gs_operator_crs(geom.Nr, geom.Nz, geom.dR, geom.dZ, geom.R_min)

    # Seed ψ: start from the geometric seed and find the initial axis value.
    psi2d = geom.seed_psi_field()
    Nr, Nz = geom.Nr, geom.Nz
    psi_flat = [psi2d[i][j] for i in range(Nr) for j in range(Nz)]

    # Initial estimates for ψ_axis, ψ_bnd
    psi_axis = max(psi_flat)
    psi_bnd = 0.0

    # Source closure
    def source_fun(psi_vec):
        return build_source(psi_vec, geom, profiles, psi_axis, psi_bnd)

    # Iterate with normalisation update every few steps (modified Picard)
    for outer in range(6):
        if method == "picard":
            state = picard_solve(A, source_fun, psi_flat, tol=tol, maxiter=maxiter)
        elif method == "pseudo-time":
            tau = 0.5 * min(geom.dR, geom.dZ) ** 2
            state = pseudo_time_solve(A, source_fun, psi_flat, tau=tau, tol=tol, maxiter=maxiter)
        elif method == "broyden":
            state = broyden_solve(A, source_fun, psi_flat, tol=tol, maxiter=maxiter // 3)
        else:
            raise ValueError(f"Unknown method {method}")
        psi_flat = state.psi
        # Update normalisation
        psi_axis = max(psi_flat)
        psi_bnd = 0.0
        # Rebuild source closure with new normalisation
        def source_fun(psi_vec, pa=psi_axis, pb=psi_bnd):
            return build_source(psi_vec, geom, profiles, pa, pb)
        if state.converged:
            break

    # Normalise so ψ_axis = 1
    psi_axis = max(psi_flat)
    if abs(psi_axis) > 1e-30:
        psi_flat = [v / psi_axis for v in psi_flat]

    psi2d = [[psi_flat[i * Nz + j] for j in range(Nz)] for i in range(Nr)]
    return {"psi": psi2d, "axis_value": 1.0, "state": state, "operator": A}
