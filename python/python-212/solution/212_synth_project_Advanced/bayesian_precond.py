"""
bayesian_precond.py
===================
Bayesian / variational preconditioning of the KKT linear system.

Mathematical background
-----------------------
The reduced KKT system arising from a PDE-constrained optimization with
Gaussian observation noise and Gaussian prior has the form

    (H + R_obs^{-1} + B^{-1}) d = -g

where
    H     : Gauss-Newton Hessian approximation of the data-misfit term
    R_obs : observation-error covariance (precision R_obs^{-1})
    B     : prior covariance of the control (precision B^{-1})
    g     : gradient of the Lagrangian

In a fully Bayesian setting, the posterior covariance of the control is

    Sigma_post = (H + R_obs^{-1} + B^{-1})^{-1}

and the posterior mean is  m_post = Sigma_post (H u + R_obs^{-1} y + B^{-1} u_b).

The posterior precision  P = Sigma_post^{-1}  is a natural preconditioner
for the KKT system because:
  1. It captures the dominant curvature (data + prior).
  2. It is symmetric positive definite.
  3. Its condition number is bounded by the ratio of extreme posterior
     eigenvalues, which is typically much smaller than that of the bare
     KKT matrix.

Hardy-type preconditioning (HTP)
--------------------------------
In the BayRad3D framework (vcasasmo), the Hardy transform is used to
precondition ill-conditioned Bayesian inverse problems. The key idea is
to approximate the posterior precision by a low-rank-plus-diagonal form:

    P ≈ D + U C U^T

where D is diagonal (cheap to invert), U has orthonormal columns spanning
the dominant subspace, and C is a small dense matrix.

Applying P^{-1} via the Woodbury identity:

    P^{-1} = D^{-1} - D^{-1} U (C^{-1} + U^T D^{-1} U)^{-1} U^T D^{-1}

This gives an O(n k^2) preconditioner application where k << n is the rank.

In the KKT context, we use the HTP preconditioner in a Preconditioned
Conjugate Gradient (PCG) inner iteration for the KKT system. The outer
active-set loop is unchanged.

KKT role
--------
The active-set iteration solves a sequence of KKT systems. As the active
set stabilizes, the KKT matrix becomes nearly constant, and the HTP
preconditioner becomes increasingly effective. PCG with HTP converges
in far fewer iterations than unpreconditioned CG.

References
----------
  - Casas, V., "BayRad3D: Bayesian Solver with Hardy Transform
    Preconditioning", GitHub repository.
  - Tarantola, A., "Inverse Problem Theory", SIAM, 2005.
  - Bui-Thanh, T., et al., "A computational framework for infinite-dimensional
    Bayesian inverse problems", SIAM J. Sci. Comput., 2013.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Low-rank eigendecomposition (Lanczos / power iteration hybrid)
# ---------------------------------------------------------------------------
def low_rank_eigen(
    A: np.ndarray,
    k: int,
    max_iter: int = 50,
    tol: float = 1.0e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the k largest-magnitude eigenpairs of a symmetric matrix A.

    Uses subspace (block) power iteration with Rayleigh-Ritz extraction.
    For the HTP preconditioner we only need the dominant modes.

    Parameters
    ----------
    A       : (n, n) symmetric matrix
    k       : int, number of eigenpairs (k <= n)
    max_iter: int, maximum iterations
    tol     : float, convergence tolerance on eigenvalue relative change

    Returns
    -------
    W : (k,) eigenvalues in descending order of |lambda|
    V : (n, k) orthonormal eigenvectors (columns)
    """
    n = A.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(Q)

    eigs_old = np.zeros(k)
    for it in range(max_iter):
        AQ = A @ Q
        # Rayleigh-Ritz
        H = Q.T @ AQ
        H = 0.5 * (H + H.T)  # symmetrize
        w, Z = np.linalg.eigh(H)
        # Sort by descending |lambda|
        order = np.argsort(-np.abs(w))
        w = w[order]
        Z = Z[:, order]
        Q = Q @ Z
        eigs = w[:k]
        if np.max(np.abs(eigs - eigs_old)) < tol * max(1.0, np.max(np.abs(eigs))):
            break
        eigs_old = eigs

    return eigs[:k], Q[:, :k]


# ---------------------------------------------------------------------------
# HTP preconditioner (low-rank + diagonal)
# ---------------------------------------------------------------------------
class HTPPreconditioner:
    """Hardy-Transform Preconditioner in low-rank-plus-diagonal form.

    Represents P ≈ D + U C U^T where D is diagonal, U is (n, k) orthonormal,
    and C is (k, k) SPD. Application of P^{-1} uses the Woodbury identity.

    Usage:
        P = HTPPreconditioner.build_from_matrix(M, rank=k)
        x = P.apply(b)   # x ≈ M^{-1} b
    """

    def __init__(self, D: np.ndarray, U: np.ndarray, C: np.ndarray):
        self.D = np.asarray(D, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.n = self.D.size
        self.k = self.U.shape[1] if self.U.ndim == 2 else 0
        # Precompute Woodbury factors
        if self.k > 0:
            self._Dinv = 1.0 / np.maximum(self.D, 1.0e-14)
            self._UtDinvU = self.U.T @ (self._Dinv[:, None] * self.U)
            self._M = self.C + self._UtDinvU
            self._Minv = np.linalg.inv(self._M)
        else:
            self._Dinv = 1.0 / np.maximum(self.D, 1.0e-14)

    @staticmethod
    def build_from_matrix(
        M: np.ndarray,
        rank: int = 20,
        diag_floor: float = 1.0e-10,
    ) -> "HTPPreconditioner":
        """Build HTP preconditioner from a dense symmetric matrix M.

        Extracts the dominant `rank` eigenpairs of M. If rank is too large
        relative to n, we fall back to exact factorization.
        """
        n = M.shape[0]
        rank = min(rank, n)

        if rank >= n // 2:
            # Exact factorization
            w, V = np.linalg.eigh(M)
            order = np.argsort(-np.abs(w))
            w = w[order]
            V = V[:, order]
            D = np.full(n, np.median(np.abs(w)) + diag_floor)
            return HTPPreconditioner(D, V, np.diag(w))

        # Low-rank extraction
        eigs, U = low_rank_eigen(M, rank)
        # Diagonal: use the trace-matching diagonal
        diag_approx = np.abs(np.diag(M))
        diag_approx = np.maximum(diag_approx, diag_floor)
        # Rescale C so that trace(D + U C U^T) ≈ trace(M)
        trace_M = np.trace(M)
        trace_D = np.sum(diag_approx)
        trace_UCUT = np.sum(eigs)  # approximate
        scale = max(1.0, (trace_M - trace_D) / max(trace_UCUT, 1.0e-14))
        C = scale * np.diag(np.abs(eigs) + diag_floor)
        return HTPPreconditioner(diag_approx, U, C)

    def apply(self, b: np.ndarray) -> np.ndarray:
        """Compute x ≈ P^{-1} b via the Woodbury identity."""
        b = np.asarray(b, dtype=float)
        if self.k == 0:
            return self._Dinv * b
        # x = D^{-1} b - D^{-1} U (C^{-1} + U^T D^{-1} U)^{-1} U^T D^{-1} b
        Dinv_b = self._Dinv * b
        Ut_Dinv_b = self.U.T @ Dinv_b
        correction = self._Minv @ Ut_Dinv_b
        x = Dinv_b - self._Dinv * (self.U @ correction)
        return x

    def condition_estimate(self) -> float:
        """Rough estimate of the condition number of P."""
        if self.k == 0:
            return np.max(self.D) / max(np.min(self.D), 1.0e-14)
        # Dominant eigenvalues: approx from C; others: from D
        eigs_C = np.linalg.eigvalsh(self.C)
        all_eigs = np.concatenate([np.abs(eigs_C), np.abs(self.D)])
        return float(np.max(all_eigs) / max(np.min(all_eigs), 1.0e-14))


# ---------------------------------------------------------------------------
# Preconditioned CG for the KKT reduced system
# ---------------------------------------------------------------------------
def pcg_kkt_solve(
    A: np.ndarray,
    b: np.ndarray,
    precond: HTPPreconditioner,
    tol: float = 1.0e-8,
    max_iter: int = 500,
) -> Tuple[np.ndarray, dict]:
    """Solve A x = b using Preconditioned Conjugate Gradient.

    A is assumed SPD. The preconditioner P^{-1} is applied via precond.apply().

    Returns
    -------
    x     : (n,) solution
    info  : dict with keys {converged, iterations, residual_norm, history}
    """
    n = b.size
    x = np.zeros(n)
    r = b - A @ x
    z = precond.apply(r)
    p = z.copy()
    rz = np.dot(r, z)

    history = []
    for k in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        if pAp <= 0.0:
            # Matrix not SPD along p; fall back to unpreconditioned
            alpha = rz / max(np.dot(p, p) * np.linalg.norm(A, 1), 1.0e-14)
        else:
            alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = np.linalg.norm(r)
        history.append(r_norm)
        if r_norm < tol * np.linalg.norm(b):
            return x, {"converged": True, "iterations": k + 1,
                       "residual_norm": r_norm, "history": history}
        z = precond.apply(r)
        rz_new = np.dot(r, z)
        beta = rz_new / max(rz, 1.0e-14)
        p = z + beta * p
        rz = rz_new

    return x, {"converged": False, "iterations": max_iter,
               "residual_norm": np.linalg.norm(r), "history": history}


# ---------------------------------------------------------------------------
# Variational posterior mean / covariance diagnostic
# ---------------------------------------------------------------------------
def variational_posterior_diagnostic(
    H: np.ndarray,
    B_inv: np.ndarray,
    R_inv: np.ndarray,
    u_b: np.ndarray,
    y_obs: np.ndarray,
    H_obs: np.ndarray,
) -> dict:
    """Compute Bayesian posterior diagnostics for the inverse problem.

    The posterior precision is P = H + B^{-1} + H_obs^T R^{-1} H_obs.
    The posterior mean solves P m = H u + B^{-1} u_b + H_obs^T R^{-1} y_obs.

    Returns
    -------
    dict with keys:
      - posterior_precision_trace : trace(P)
      - posterior_cov_trace       : trace(P^{-1})  (uncertainty)
      - effective_rank            : sum of eigenvalues of P / max eigenvalue
      - information_gain          : log det(P) - log det(B^{-1})
      - posterior_mean_norm       : ||m||
    """
    P = H + B_inv + H_obs.T @ R_inv @ H_obs
    P = 0.5 * (P + P.T)

    try:
        L = np.linalg.cholesky(P)
        logdet_P = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # fallback via eigh
        eigs_P = np.linalg.eigvalsh(P)
        eigs_P = np.maximum(eigs_P, 1.0e-14)
        logdet_P = np.sum(np.log(eigs_P))

    try:
        L_b = np.linalg.cholesky(B_inv)
        logdet_Binv = 2.0 * np.sum(np.log(np.diag(L_b)))
    except np.linalg.LinAlgError:
        eigs_B = np.linalg.eigvalsh(B_inv)
        eigs_B = np.maximum(eigs_B, 1.0e-14)
        logdet_Binv = np.sum(np.log(eigs_B))

    eigs_P = np.linalg.eigvalsh(P)
    eigs_P = np.maximum(eigs_P, 1.0e-14)
    eff_rank = np.sum(eigs_P) / np.max(eigs_P)

    # Posterior mean (assuming data term = 0 for simplicity)
    rhs = B_inv @ u_b + H_obs.T @ R_inv @ y_obs
    try:
        m = np.linalg.solve(P, rhs)
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(P, rhs, rcond=None)[0]

    return {
        "posterior_precision_trace": float(np.trace(P)),
        "posterior_cov_trace": float(np.sum(1.0 / eigs_P)),
        "effective_rank": float(eff_rank),
        "information_gain": float(logdet_P - logdet_Binv),
        "posterior_mean_norm": float(np.linalg.norm(m)),
        "condition_number": float(np.max(eigs_P) / np.min(eigs_P)),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[bayesian_precond] Self-test")
    print("-" * 60)

    # Build a random SPD matrix
    rng = np.random.default_rng(0)
    n = 60
    A = rng.standard_normal((n, n))
    M = A.T @ A + 10.0 * np.eye(n)

    # Build HTP preconditioner
    P = HTPPreconditioner.build_from_matrix(M, rank=15)
    print(f"  HTP preconditioner: n={n}, rank=15")
    print(f"    condition_estimate(P) = {P.condition_estimate():.2e}")
    print(f"    condition_estimate(M) = {np.linalg.cond(M):.2e}")

    # Solve M x = b with PCG
    b = rng.standard_normal(n)
    x_pcg, info = pcg_kkt_solve(M, b, P, tol=1.0e-10, max_iter=200)
    x_exact = np.linalg.solve(M, b)
    err = np.linalg.norm(x_pcg - x_exact)
    print(f"  PCG solve:")
    print(f"    converged       : {info['converged']}")
    print(f"    iterations      : {info['iterations']}")
    print(f"    residual        : {info['residual_norm']:.2e}")
    print(f"    ||x_pcg - x_ex||: {err:.2e}")

    # Variational posterior diagnostic
    H = A.T @ A
    B_inv = 0.1 * np.eye(n)
    R_inv = 1.0 * np.eye(10)
    H_obs = rng.standard_normal((10, n))
    u_b = rng.standard_normal(n)
    y_obs = rng.standard_normal(10)
    diag = variational_posterior_diagnostic(H, B_inv, R_inv, u_b, y_obs, H_obs)
    print(f"  Variational posterior diagnostic:")
    for k, v in diag.items():
        print(f"    {k:30s} = {v:.4e}")
