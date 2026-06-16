"""
cost_functional.py
==================
Objective functional for the KKT-constrained optimal control problem.

The cost functional combines several terms from different scientific
disciplines:

  J(y, u) = J_track(y)  +  J_tikh(u)  +  J_bg(u)  +  J_sparse(u)

  J_track(y) = 0.5 * (y - y_d)^T M (y - y_d)         [tracking / data misfit]
  J_tikh(u)  = 0.5 * alpha * u^T M u                   [Tikhonov regularisation]
  J_bg(u)    = 0.5 * (u - u_b)^T B^{-1} (u - u_b)    [4D-Var background term]
  J_sparse(u)= beta * sum_i |u_i| * h^2                [L1 sparsity promotion]

Here M = h^2 * I is the lumped mass matrix (approximating the L2 inner
product on [0,1]^2), B is the background-error covariance, and u_b is
the background control.

The gradient is

    nabla_y J = M (y - y_d)
    nabla_u J = alpha M u + B^{-1} (u - u_b) + beta * sign(u) * h^2

The Hessian blocks (for Newton / quasi-Newton) are

    nabla^2_{yy} J = M
    nabla^2_{uu} J = alpha M + B^{-1}

KKT role
--------
The cost functional defines the Lagrangian of the constrained problem;
its gradient enters the stationarity conditions, and its Hessian defines
the (3,3) block of the KKT system.
"""

from __future__ import annotations
import numpy as np


class CostFunctional:
    """Encapsulates the objective functional and its derivatives."""

    def __init__(
        self,
        N: int,
        alpha: float,
        beta: float,
        B_cov_inv: np.ndarray,
        u_background: np.ndarray,
        y_d: np.ndarray,
    ):
        """
        Parameters
        ----------
        N           : grid size per dimension
        alpha       : Tikhonov weight
        beta        : L1 sparsity weight
        B_cov_inv   : inverse of background-error covariance (N^2, N^2)
        u_background: background control (N^2,)
        y_d         : target state (N^2,)
        """
        self.N = N
        self.n2 = N * N
        self.alpha = alpha
        self.beta = beta
        self.B_cov_inv = B_cov_inv
        self.u_background = np.asarray(u_background, dtype=np.float64).ravel()
        self.y_d = np.asarray(y_d, dtype=np.float64).ravel()

        if self.u_background.size != self.n2:
            raise ValueError(f"CostFunctional: u_background size mismatch")
        if self.y_d.size != self.n2:
            raise ValueError(f"CostFunctional: y_d size mismatch")

        h = 1.0 / (N + 1)
        self.h = h
        self.h2 = h * h
        self.M_diag = np.full(self.n2, self.h2, dtype=np.float64)

    # ------------------------------------------------------------------
    def tracking_term(self, y: np.ndarray) -> float:
        """J_track = 0.5 * (y - y_d)^T M (y - y_d).

        This is the L2(Omega) data misfit, weighted by the mass matrix.
        In the 4D-Var interpretation, this is the observation term J_o.
        """
        diff = np.asarray(y, dtype=np.float64).ravel() - self.y_d
        return 0.5 * float(np.dot(diff, self.M_diag * diff))

    def tikhonov_term(self, u: np.ndarray) -> float:
        """J_tikh = 0.5 * alpha * u^T M u.

        Standard Tikhonov regularisation penalising control energy.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        return 0.5 * self.alpha * float(np.dot(u, self.M_diag * u))

    def background_term(self, u: np.ndarray) -> float:
        """J_bg = 0.5 * (u - u_b)^T B^{-1} (u - u_b).

        The 4D-Var background term penalises deviation from a prior
        estimate u_b, weighted by the inverse background-error covariance.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        diff = u - self.u_background
        return 0.5 * float(np.dot(diff, self.B_cov_inv @ diff))

    def sparse_term(self, u: np.ndarray) -> float:
        """J_sparse = beta * sum_i |u_i| * h^2.

        L1 sparsity promotion on the control.  This is non-differentiable
        at u = 0; we use the subgradient  sign(u) in the gradient.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        return self.beta * float(np.sum(np.abs(u))) * self.h2

    # ------------------------------------------------------------------
    def evaluate(self, y: np.ndarray, u: np.ndarray) -> float:
        """Total cost  J(y, u) = J_track + J_tikh + J_bg + J_sparse."""
        return (
            self.tracking_term(y)
            + self.tikhonov_term(u)
            + self.background_term(u)
            + self.sparse_term(u)
        )

    # ------------------------------------------------------------------
    def gradient_y(self, y: np.ndarray) -> np.ndarray:
        """Gradient of J w.r.t. y:  nabla_y J = M (y - y_d)."""
        y = np.asarray(y, dtype=np.float64).ravel()
        return self.M_diag * (y - self.y_d)

    def gradient_u(self, u: np.ndarray) -> np.ndarray:
        """Gradient of J w.r.t. u (excluding L1 subgradient):

        nabla_u J = alpha M u + B^{-1} (u - u_b)

        The L1 subgradient  beta * sign(u) * h^2 is handled separately
        via proximal / active-set methods.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        grad = self.alpha * self.M_diag * u + self.B_cov_inv @ (u - self.u_background)
        return grad

    def l1_subgradient(self, u: np.ndarray) -> np.ndarray:
        """Subgradient of J_sparse:  beta * sign(u) * h^2.

        At u = 0, we return 0 (consistent with the proximal operator).
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        return self.beta * np.sign(u) * self.h2

    # ------------------------------------------------------------------
    def hessian_u_block(self) -> np.ndarray:
        """The (u, u) block of the Hessian:  alpha M + B^{-1}.

        This is SPD and enters the (3,3) block of the KKT system.
        """
        return self.alpha * np.diag(self.M_diag) + self.B_cov_inv

    # ------------------------------------------------------------------
    def reduced_cost(self, u: np.ndarray, A: np.ndarray, f_vec: np.ndarray) -> float:
        """Evaluate the reduced cost  J_hat(u) = J(A^{-1}(u + f), u).

        This eliminates y via the PDE constraint and gives a function
        of u alone (still subject to box and integral constraints).
        """
        from linear_algebra import plu_solve_safe
        y = plu_solve_safe(A, u + f_vec)
        return self.evaluate(y, u)

    def reduced_gradient(
        self, u: np.ndarray, A: np.ndarray, f_vec: np.ndarray
    ) -> np.ndarray:
        """Gradient of the reduced cost:

        nabla J_hat = alpha M u + B^{-1}(u - u_b) - p + beta sign(u) h^2

        where p = A^{-T} M (y_d - y) is the adjoint state.
        """
        from linear_algebra import plu_solve_safe
        y = plu_solve_safe(A, u + f_vec)
        p_rhs = self.M_diag * (self.y_d - y)
        p = plu_solve_safe(A.T, p_rhs)
        grad = self.gradient_u(u) - p + self.l1_subgradient(u)
        return grad
