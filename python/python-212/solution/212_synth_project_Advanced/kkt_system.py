"""
kkt_system.py
=============
Assembly and solution of the Karush-Kuhn-Tucker (KKT) saddle-point system
for the PDE-constrained optimal control problem.

The KKT conditions
------------------
For the problem

    min  J(y, u) = 0.5 ||y - y_d||_M^2 + 0.5 alpha ||u||_M^2 + 0.5 (u-u_b)^T B^{-1} (u-u_b)
    s.t. A y = u + f                 (PDE equality)
         u_a <= u <= u_b             (box inequality)
         c^T u <= E_max              (integral inequality)

the KKT conditions are:

    (1)  A y = u + f                              (state equation)
    (2)  A^T p = M (y_d - y)                      (adjoint equation)
    (3)  (alpha M + B^{-1}) u - p + B^{-1} u_b    (stationarity)
              + eta - theta + lambda c = 0
    (4)  eta >= 0,  u >= u_a,  eta^T (u - u_a) = 0   (lower complementarity)
    (5)  theta >= 0,  u_b >= u,  theta^T (u_b - u) = 0  (upper complementarity)
    (6)  lambda >= 0,  E_max >= c^T u,  lambda (c^T u - E_max) = 0  (integral)

On the inactive set (box constraints not active), we solve the full KKT
system as a (3 N^2 + 1) x (3 N^2 + 1) saddle-point problem.
"""

from __future__ import annotations
import numpy as np

from linear_algebra import plu_factor, plu_solve, plu_solve_safe, iterative_refinement
from physics_models import PhysicalParameters


# ---------------------------------------------------------------------------
# Solve the full KKT system (saddle-point approach)
# ---------------------------------------------------------------------------

def solve_kkt_system(
    A: np.ndarray,
    M_diag: np.ndarray,
    H_reg: np.ndarray,
    c_vec: np.ndarray,
    f_vec: np.ndarray,
    y_d: np.ndarray,
    u_background: np.ndarray,
    B_cov_inv: np.ndarray,
    E_max: float,
    inactive: np.ndarray,
    active_lower: np.ndarray,
    active_upper: np.ndarray,
    params: PhysicalParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve the KKT system with the current active-set partition.

    Strategy:
    1. First solve WITHOUT the integral constraint (lam = 0).
    2. Check if c^T u <= E_max. If yes, done.
    3. If no, solve WITH the integral constraint active (c^T u = E_max, lam >= 0).

    This implements the standard active-set approach for inequality constraints.
    """
    n2 = A.shape[0]
    inactive_idx = np.where(inactive)[0]
    active_idx = np.where(~inactive)[0]
    n_inactive = inactive_idx.size

    # Active control values
    u_active_full = np.zeros(n2, dtype=np.float64)
    u_active_full[active_lower] = params.u_lower
    u_active_full[active_upper] = params.u_upper

    # Factor A once
    LU_A, pivot_A, info_A = plu_factor(A.copy())
    if info_A != 0:
        raise ValueError("solve_kkt_system: singular PDE operator A")

    def Ainv(v):
        return plu_solve(LU_A, pivot_A, v)

    def AinvT(v):
        return plu_solve(LU_A, pivot_A, v)  # A is symmetric here

    if n_inactive == 0:
        # All controls are active; compute y, p, lambda
        u = u_active_full.copy()
        y = Ainv(u + f_vec)
        p = AinvT(M_diag * (y_d - y))
        # Compute lam from stationarity
        grad_u_full = H_reg @ u - B_cov_inv @ u_background - p
        c_norm2 = float(np.dot(c_vec, c_vec))
        lam = -np.dot(c_vec, grad_u_full) / max(c_norm2, 1.0e-30) if c_norm2 > 1.0e-30 else 0.0
        lam = max(lam, 0.0)
        return y, p, u, float(lam)

    # ---- Step 1: Solve without integral constraint (lam = 0) ----
    # Build Schur complement
    Z = np.zeros((n2, n_inactive), dtype=np.float64)
    for j, jj in enumerate(inactive_idx):
        ej = np.zeros(n2, dtype=np.float64)
        ej[jj] = 1.0
        Z[:, j] = plu_solve(LU_A, pivot_A, ej)

    Z_I = Z[inactive_idx, :]
    MZ = M_diag[:, None] * Z
    MZ_I = MZ[inactive_idx, :]
    W_II = Z_I.T @ MZ_I

    z_A = Ainv(u_active_full)
    Mz_A = M_diag * z_A
    W_uA_full = AinvT(Mz_A)
    W_uA_I = W_uA_full[inactive_idx]

    z_f = Ainv(f_vec)
    Mz_f = M_diag * z_f
    Wf_full = AinvT(Mz_f)
    Wf_I = Wf_full[inactive_idx]

    AtM_yd = AinvT(M_diag * y_d)
    AtM_yd_I = AtM_yd[inactive_idx]

    # System without integral constraint: just (n_inactive x n_inactive)
    H_reg_II = H_reg[np.ix_(inactive_idx, inactive_idx)]
    K_noint = H_reg_II + W_II
    rhs_noint = np.zeros(n_inactive, dtype=np.float64)
    rhs_noint = AtM_yd_I - W_uA_I - Wf_I + (B_cov_inv @ u_background)[inactive_idx]
    if active_idx.size > 0:
        H_reg_IA = H_reg[np.ix_(inactive_idx, active_idx)]
        rhs_noint -= H_reg_IA @ u_active_full[active_idx]

    u_I_noint = iterative_refinement(K_noint, rhs_noint)

    # Reconstruct u
    u_noint = u_active_full.copy()
    u_noint[inactive_idx] = u_I_noint
    u_noint = np.clip(u_noint, params.u_lower, params.u_upper)

    # Check integral constraint
    ctu_noint = float(np.dot(c_vec, u_noint))

    if ctu_noint <= E_max:
        # Constraint inactive; use this solution
        u = u_noint
        lam = 0.0
    else:
        # ---- Step 2: Solve with integral constraint active ----
        n_sys = n_inactive + 1
        K_sys = np.zeros((n_sys, n_sys), dtype=np.float64)
        rhs_sys = np.zeros(n_sys, dtype=np.float64)

        K_sys[:n_inactive, :n_inactive] = K_noint
        c_I = c_vec[inactive_idx]
        K_sys[:n_inactive, -1] = c_I
        K_sys[-1, :n_inactive] = c_I

        rhs_sys[:n_inactive] = rhs_noint
        rhs_sys[-1] = E_max - np.dot(c_vec[active_idx], u_active_full[active_idx]) if active_idx.size > 0 else E_max

        sol = iterative_refinement(K_sys, rhs_sys)
        u_I = sol[:n_inactive]
        lam = float(sol[-1])

        # Enforce lam >= 0 (if lam < 0, the constraint should be inactive)
        if lam < 0:
            # Degenerate: use the unconstrained solution
            u = u_noint
            lam = 0.0
        else:
            u = u_active_full.copy()
            u[inactive_idx] = u_I
            u = np.clip(u, params.u_lower, params.u_upper)

    # Compute y and p
    y = Ainv(u + f_vec)
    p = AinvT(M_diag * (y_d - y))

    return y, p, u, lam


# ---------------------------------------------------------------------------
# KKT residual computation
# ---------------------------------------------------------------------------

def compute_kkt_residual(
    y: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    lam: float,
    A: np.ndarray,
    M_diag: np.ndarray,
    H_reg: np.ndarray,
    c_vec: np.ndarray,
    f_vec: np.ndarray,
    y_d: np.ndarray,
    u_background: np.ndarray,
    B_cov_inv: np.ndarray,
    E_max: float,
    params: PhysicalParameters,
    eta: np.ndarray,
    theta: np.ndarray,
) -> dict:
    """Compute the KKT residual (a measure of how well the KKT conditions are satisfied).

    Returns a dict with:
        'primal_pde'    : ||A y - u - f||
        'primal_adj'    : ||A^T p - M(y_d - y)||
        'stationarity'  : ||H_reg u - p + B^{-1}(u - u_b) + eta - theta + lam c||
        'complementarity': max(eta_i (u_i - u_a), theta_i (u_b - u_i))
        'integral_viol' : max(0, c^T u - E_max)
        'dual_feas'     : max(0, -min(eta), -min(theta), -lam)
        'total'         : combined residual (RMS)
    """
    n2 = A.shape[0]

    # Primal PDE residual
    res_pde = float(np.linalg.norm(A @ y - u - f_vec))

    # Adjoint residual
    res_adj = float(np.linalg.norm(A.T @ p - M_diag * (y_d - y)))

    # Stationarity residual
    # dL/du = H_reg u - B^{-1} u_b - p + lam c - eta + theta = 0
    grad_L = H_reg @ u - B_cov_inv @ u_background - p + lam * c_vec - eta + theta
    res_stat = float(np.linalg.norm(grad_L))

    # Complementarity
    comp_low = eta * (u - params.u_lower)
    comp_up = theta * (params.u_upper - u)
    res_comp = 0.0
    if eta.size > 0:
        res_comp = max(res_comp, float(np.max(np.abs(comp_low))))
    if theta.size > 0:
        res_comp = max(res_comp, float(np.max(np.abs(comp_up))))

    # Integral violation
    int_viol = max(0.0, float(np.dot(c_vec, u)) - E_max)

    # Dual feasibility
    dual_viol = 0.0
    if eta.size > 0:
        dual_viol = max(dual_viol, float(np.max(-eta)))
    if theta.size > 0:
        dual_viol = max(dual_viol, float(np.max(-theta)))
    dual_viol = max(dual_viol, -lam)

    # Total residual
    residuals = np.array([res_pde, res_adj, res_stat, res_comp, int_viol, dual_viol])
    res_total = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "primal_pde": res_pde,
        "primal_adj": res_adj,
        "stationarity": res_stat,
        "complementarity": res_comp,
        "integral_viol": int_viol,
        "dual_feas": dual_viol,
        "total": res_total,
    }
