"""
nonlinear_solver.py — Picard + Newton-Raphson + Broyden iteration for the
nonlinear Grad-Shafranov equation, built on the implicit time-stepping idea
from the 1-D heat-equation project (361_fd1d_heat_implicit).

Scientific background
---------------------
We view the GS equation as a nonlinear elliptic problem

    L ψ = S(ψ)                                           (1)

where L = Δ* is the linear GS operator and S(ψ) = -μ₀ R² p'(ψ) - F(ψ)F'(ψ)
is the plasma current source.  Three iterative strategies are supplied:

Picard (fixed-point)
    L ψ^{k+1} = S(ψ^k)                                  (2)

This is unconditionally stable for mild profiles but only linearly convergent.

Pseudo-time relaxation (inspired by the heat-equation implicit FD)
    (I - τ L) ψ^{k+1} = ψ^k + τ S(ψ^k)                  (3)
Large τ → Picard; small τ → explicit.  We use τ ≈ 1/||L|| as default.

Newton-Raphson
    [L - J(ψ^k)] δψ = S(ψ^k) - L ψ^k,   ψ^{k+1} = ψ^k + δψ    (4)
with Jacobian J_{mn} = ∂S_m/∂ψ_n.  Quadratic convergence near the solution.

Broyden (Jacobian-free quasi-Newton)
    δψ_k = -H_k F_k,   H_{k+1} = (I - ρ_k s_k H_k) with ρ_k = 1/(y_k·s_k)   (5)

Convergence is assessed by the discrete residual

    ||F_k||_∞ = ||L ψ^k - S(ψ^k)||_∞ / max(1, ||ψ^k||_∞)

with tolerance 1e-8 by default.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
try:
    from .sparse_operators import CRSMatrix, jacobi_preconditioner
except ImportError:
    from sparse_operators import CRSMatrix, jacobi_preconditioner


@dataclass
class SolverState:
    psi: list[float]
    residual: float
    iterations: int
    converged: bool
    method: str


def residual(A: CRSMatrix, psi: list[float], source: list[float]) -> list[float]:
    """F(ψ) = A ψ - S(ψ)."""
    Ap = A.matvec(psi)
    return [Ap[i] - source[i] for i in range(A.n)]


def inf_norm(v: list[float]) -> float:
    return max((abs(x) for x in v), default=0.0)


# ---------------------------------------------------------------------------
# CG solver for the symmetric part of L (or as a preconditioned inner solver)
# ---------------------------------------------------------------------------

def cg_solve(A: CRSMatrix, b: list[float], x0: list[float] | None = None,
             tol: float = 1e-10, maxiter: int = 2000) -> tuple[list[float], int]:
    """Conjugate-gradient solve for A x = b, using Jacobi preconditioning.
    The GS operator is not quite SPD because of the (1/R) ψ_R term, but for
    well-resolved grids the symmetric part dominates and CG converges."""
    n = A.n
    x = list(x0) if x0 is not None else [0.0] * n
    Minv = jacobi_preconditioner(A)
    r = [b[i] - A.matvec(x)[i] for i in range(n)]
    z = [Minv[i] * r[i] for i in range(n)]
    p = list(z)
    rz = sum(r[i] * z[i] for i in range(n))
    it = 0
    for it in range(1, maxiter + 1):
        Ap = A.matvec(p)
        pAp = sum(p[i] * Ap[i] for i in range(n))
        if abs(pAp) < 1e-30:
            break
        alpha = rz / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        if inf_norm(r) < tol * (inf_norm(b) + 1e-15):
            break
        z = [Minv[i] * r[i] for i in range(n)]
        rz_new = sum(r[i] * z[i] for i in range(n))
        beta = rz_new / (rz + 1e-30)
        p = [z[i] + beta * p[i] for i in range(n)]
        rz = rz_new
    return x, it


# ---------------------------------------------------------------------------
# Picard iteration
# ---------------------------------------------------------------------------

def picard_solve(A: CRSMatrix, source_fun, psi0: list[float],
                 tol: float = 1e-8, maxiter: int = 200) -> SolverState:
    psi = list(psi0)
    for k in range(maxiter):
        S = source_fun(psi)
        F = residual(A, psi, S)
        res = inf_norm(F) / max(1.0, inf_norm(psi))
        if res < tol:
            return SolverState(psi, res, k, True, "picard")
        psi_new, _ = cg_solve(A, S, psi, tol=tol * 0.1)
        psi = psi_new
    return SolverState(psi, res, maxiter, False, "picard")


# ---------------------------------------------------------------------------
# Pseudo-time relaxation (implicit FD analog)
# ---------------------------------------------------------------------------

def pseudo_time_solve(A: CRSMatrix, source_fun, psi0: list[float],
                      tau: float = 0.05, tol: float = 1e-8,
                      maxiter: int = 2000) -> SolverState:
    """Pseudo-time: ψ^{k+1} = ψ^k + τ (S(ψ^k) - A ψ^k)
    rewritten as (I + τ A) ψ^{k+1} = ψ^k + τ S(ψ^k) for implicit variant."""
    psi = list(psi0)
    n = A.n
    # Build (I + τ A) operator (same sparsity as A)
    B = CRSMatrix(n=n, row_ptr=list(A.row_ptr),
                  col_idx=list(A.col_idx), values=[tau * v for v in A.values])
    # Add identity: add 1 to diagonal entries
    for i in range(n):
        for k in range(B.row_ptr[i], B.row_ptr[i + 1]):
            if B.col_idx[k] == i:
                B.values[k] += 1.0
                break
    res = 0.0
    for it in range(maxiter):
        S = source_fun(psi)
        rhs = [psi[i] + tau * S[i] for i in range(n)]
        psi_new, _ = cg_solve(B, rhs, psi, tol=tol * 0.01, maxiter=500)
        F = residual(A, psi_new, S)
        res = inf_norm(F) / max(1.0, inf_norm(psi_new))
        psi = psi_new
        if res < tol:
            return SolverState(psi, res, it, True, "pseudo-time")
    return SolverState(psi, res, maxiter, False, "pseudo-time")


# ---------------------------------------------------------------------------
# Broyden (Jacobian-free quasi-Newton)
# ---------------------------------------------------------------------------

def broyden_solve(A: CRSMatrix, source_fun, psi0: list[float],
                  tol: float = 1e-8, maxiter: int = 60) -> SolverState:
    """Limited-memory Broyden with direct application of the initial
    preconditioner H_0 = diag(A)^{-1}."""
    n = A.n
    psi = list(psi0)
    Minv = jacobi_preconditioner(A)
    S = source_fun(psi)
    Fk = residual(A, psi, S)

    # H is applied as a dense linear map (we keep it dense for small n ~ 4000)
    if n > 2500:
        # Fall back to Picard for large problems to avoid O(n²) memory
        return picard_solve(A, source_fun, psi0, tol, maxiter * 3)

    H = [[Minv[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    res = inf_norm(Fk) / max(1.0, inf_norm(psi))
    for it in range(maxiter):
        # Step δψ = -H F
        dpsi = [-sum(H[i][j] * Fk[j] for j in range(n)) for i in range(n)]
        # Line-search with backtracking
        alpha = 1.0
        psi_new = [psi[i] + alpha * dpsi[i] for i in range(n)]
        S_new = source_fun(psi_new)
        F_new = residual(A, psi_new, S_new)
        res_new = inf_norm(F_new) / max(1.0, inf_norm(psi_new))
        if res_new > 2.0 * res and alpha > 0.1:
            alpha = 0.5
            psi_new = [psi[i] + alpha * dpsi[i] for i in range(n)]
            S_new = source_fun(psi_new)
            F_new = residual(A, psi_new, S_new)
            res_new = inf_norm(F_new) / max(1.0, inf_norm(psi_new))
        # Broyden update: s = ψ_new - ψ, y = F_new - F
        s = [psi_new[i] - psi[i] for i in range(n)]
        y = [F_new[i] - Fk[i] for i in range(n)]
        yHs = sum(y[i] * sum(H[i][j] * s[j] for j in range(n)) for i in range(n))
        # Sherman-Morrison update of H (only if y·H·s is safe)
        if abs(yHs) > 1e-25:
            Hs = [sum(H[i][j] * s[j] for j in range(n)) for i in range(n)]
            for i in range(n):
                for j in range(n):
                    H[i][j] -= Hs[i] * y[j] * (1.0 / yHs) * sum(H[j][m] * s[m] for m in range(n))
                    # Simpler rank-1 approximation: keep H well-conditioned
            # Re-regularise H towards Minv
            for i in range(n):
                H[i][i] = 0.95 * H[i][i] + 0.05 * Minv[i]
        psi = psi_new
        Fk = F_new
        res = res_new
        if res < tol:
            return SolverState(psi, res, it, True, "broyden")
    return SolverState(psi, res, maxiter, False, "broyden")
