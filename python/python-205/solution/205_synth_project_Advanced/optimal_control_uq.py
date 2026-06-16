"""
optimal_control_uq.py - Robust Optimal Control Under Uncertainty

This module implements optimal control using surrogate-accelerated
uncertainty quantification, integrating:
  - Forward-backward sweep method (from control_bio)
  - Pontryagin's Maximum Principle for ODE-constrained optimization
  - Robust formulation with chance constraints via PCE surrogate

Mathematical Framework
----------------------
### Deterministic Optimal Control ###

Minimize: J(u) = integral_0^T L(x(t), u(t), t) dt + Phi(x(T))
Subject to: dx/dt = f(x(t), u(t), t), x(0) = x_0
            u_min <= u(t) <= u_max

### Pontryagin's Maximum Principle ###

Hamiltonian: H(x, u, lambda, t) = L(x, u, t) + lambda^T f(x, u, t)

Necessary conditions:
  State equation:     dx/dt = dH/d_lambda = f(x, u, t)
  Adjoint equation:  d_lambda/dt = -dH/d_x
  Optimality:         dH/d_u = 0 (for interior controls)
  Transversality:     lambda(T) = dPhi/dx(T)

### Forward-Backward Sweep ###

1. Initialize u(t) = u_0(t)
2. Forward sweep: integrate state equation with current u
3. Backward sweep: integrate adjoint equation backward from lambda(T)
4. Update control: u_new = projection of -dH/du onto [u_min, u_max]
5. Relaxation: u = (1 - omega)*u_old + omega*u_new
6. Check convergence: ||u_new - u_old|| / ||u_old|| < tol
7. Repeat 2-6 until convergence

### Robust Optimal Control via PCE ###

When parameters xi are uncertain:
  min_u E_xi[J(u, xi)] + beta * Var_xi[J(u, xi)]
  s.t. P_xi(g(x, u, xi) <= 0) >= 1 - alpha  (chance constraint)

Using PCE surrogate for J(u, xi):
  E[J] = c_0(u)  (zeroth PCE coefficient)
  Var[J] = sum_{j>0} c_j(u)^2

The robust objective is directly computed from PCE coefficients
without additional sampling, making the optimization efficient.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Callable, Any
import math
from numerical_utils import convergence_ratio


# ---------------------------------------------------------------------------
# Forward-Backward Sweep Solver
# ---------------------------------------------------------------------------

def forward_backward_sweep(
        state_rhs: Callable,
        adjoint_rhs: Callable,
        control_update: Callable,
        x0: NDArray,
        t_final: float,
        n_steps: int = 500,
        u_init: Optional[NDArray] = None,
        u_min: float = 0.0,
        u_max: float = 1.0,
        relaxation: float = 0.5,
        tol: float = 1e-4,
        max_iter: int = 50) -> Dict[str, Any]:
    """
    Solve optimal control problem via forward-backward sweep.

    Parameters
    ----------
    state_rhs : callable(t, x, u) -> dx/dt
        State equation RHS.
    adjoint_rhs : callable(t, x, u, lambda_) -> d_lambda/dt
        Adjoint equation RHS.
    control_update : callable(t, x, lambda_) -> u_optimal
        Optimal control from optimality condition.
    x0 : ndarray(nx,)
        Initial state.
    t_final : float
        Final time.
    n_steps : int
        Number of time steps.
    u_init : ndarray(n_steps+1,), optional
        Initial control guess.
    u_min, u_max : float
        Control bounds.
    relaxation : float
        Relaxation parameter in (0, 1].
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict with:
        't': time array
        'x': state trajectory
        'u': control history
        'lambda_': adjoint trajectory
        'cost': objective value
        'converged': bool
        'n_iterations': int
    """
    dt = t_final / n_steps
    t = np.linspace(0, t_final, n_steps + 1)
    nx = len(x0)

    # Initialize control
    if u_init is not None:
        u = u_init.copy()
    else:
        u = np.full(n_steps + 1, (u_min + u_max) / 2.0)

    x = np.zeros((n_steps + 1, nx))
    x[0] = x0.copy()
    lam = np.zeros((n_steps + 1, nx))
    cost = 0.0
    converged = False
    n_iter = 0
    conv_ratio = 1.0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # --- Forward sweep: integrate state equation ---
        for i in range(n_steps):
            dxdt = state_rhs(t[i], x[i], u[i])
            x[i + 1] = x[i] + dt * dxdt
            # Cost accumulation (running cost = u^2 for regularization)
            cost += dt * (u[i] ** 2)

        # Terminal cost
        cost += np.sum(x[-1] ** 2)

        # --- Backward sweep: integrate adjoint equation ---
        lam[-1] = 2.0 * x[-1]  # Transversality: lambda(T) = dPhi/dx

        for i in range(n_steps - 1, -1, -1):
            dlamdt = adjoint_rhs(t[i], x[i], u[i], lam[i + 1])
            lam[i] = lam[i + 1] - dt * dlamdt

        # --- Update control ---
        u_new = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            u_opt = control_update(t[i], x[i], lam[i])
            # Project onto bounds
            u_new[i] = np.clip(u_opt, u_min, u_max)

        # --- Relaxation ---
        u_old = u.copy()
        u = (1.0 - relaxation) * u + relaxation * u_new

        # --- Convergence check ---
        conv_ratio = convergence_ratio(u, u_old, conv_ratio)
        if conv_ratio < tol:
            converged = True
            break

    # Compute final cost
    cost_final = 0.0
    for i in range(n_steps):
        cost_final += dt * (u[i] ** 2)
    cost_final += np.sum(x[-1] ** 2)

    return {
        't': t,
        'x': x,
        'u': u,
        'lambda': lam,
        'cost': cost_final,
        'converged': converged,
        'n_iterations': n_iter,
        'convergence_ratio': conv_ratio
    }


# ---------------------------------------------------------------------------
# Surrogate-Accelerated Robust Optimal Control
# ---------------------------------------------------------------------------

class RobustOptimalControl:
    """
    Robust optimal control using PCE surrogate for uncertainty propagation.

    The robust objective:
      min_u J_robust(u) = E_xi[J(u, xi)] + beta * Var_xi[J(u, xi)]

    Using PCE surrogate for J(u, xi) at fixed u:
      E[J] = c_0
      Var[J] = sum_{j>0} c_j^2

    So J_robust = c_0 + beta * sum_{j>0} c_j^2

    The gradient w.r.t. u is computed by finite differences on the
    PCE coefficients, which is much cheaper than MC sampling.

    Attributes
    ----------
    control_dim : int
        Dimension of control variable.
    uncertainty_dim : int
        Dimension of uncertain parameters.
    beta : float
        Risk aversion parameter (0 = risk-neutral, large = risk-averse).
    """

    def __init__(self, control_dim: int, uncertainty_dim: int,
                 beta: float = 0.1):
        self.control_dim = control_dim
        self.uncertainty_dim = uncertainty_dim
        self.beta = beta

    def evaluate_robust_objective(
            self, u: NDArray,
            surrogate_factory: Callable,
            xi_samples: NDArray) -> Tuple[float, NDArray]:
        """
        Evaluate robust objective and its gradient.

        Parameters
        ----------
        u : ndarray(control_dim,)
            Control variable.
        surrogate_factory : callable(u, xi_samples) -> surrogate
            Builds PCE surrogate for J(u, xi) at given u.
        xi_samples : ndarray(n_xi, uncertainty_dim)
            Uncertain parameter samples.

        Returns
        -------
        J_robust : float
            Robust objective value.
        grad : ndarray(control_dim,)
            Gradient of robust objective.
        """
        # Build surrogate at current u
        surrogate = surrogate_factory(u, xi_samples)

        # PCE-based moments
        moments = surrogate.statistical_moments()
        J_robust = moments['mean'] + self.beta * moments['variance']

        # Gradient via finite differences
        eps = 1e-5
        grad = np.zeros(self.control_dim)
        for k in range(self.control_dim):
            u_plus = u.copy()
            u_plus[k] += eps
            surrogate_plus = surrogate_factory(u_plus, xi_samples)
            moments_plus = surrogate_plus.statistical_moments()
            J_plus = moments_plus['mean'] + self.beta * moments_plus['variance']
            grad[k] = (J_plus - J_robust) / eps

        return J_robust, grad

    def optimize(self, u0: NDArray,
                 surrogate_factory: Callable,
                 xi_samples: NDArray,
                 u_bounds: Optional[List[Tuple[float, float]]] = None,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 tol: float = 1e-6) -> Dict[str, Any]:
        """
        Optimize robust objective via projected gradient descent.

        Parameters
        ----------
        u0 : ndarray(control_dim,)
            Initial control.
        surrogate_factory : callable
            Builds PCE surrogate at given control.
        xi_samples : ndarray
            Uncertainty samples.
        u_bounds : list of (lo, hi), optional
            Control bounds.
        max_iter : int
            Maximum iterations.
        learning_rate : float
            Step size for gradient descent.
        tol : float
            Convergence tolerance on gradient norm.

        Returns
        -------
        dict with optimization results.
        """
        u = u0.copy()
        history = []

        for iteration in range(max_iter):
            J_robust, grad = self.evaluate_robust_objective(
                u, surrogate_factory, xi_samples)

            grad_norm = float(np.linalg.norm(grad))
            history.append({
                'iteration': iteration,
                'objective': J_robust,
                'grad_norm': grad_norm,
                'control': u.copy()
            })

            if grad_norm < tol:
                break

            # Gradient step
            u_new = u - learning_rate * grad

            # Project onto bounds
            if u_bounds is not None:
                for k in range(self.control_dim):
                    lo, hi = u_bounds[k]
                    u_new[k] = np.clip(u_new[k], lo, hi)

            # Line search (simple backtracking)
            alpha = learning_rate
            for _ in range(10):
                J_new, _ = self.evaluate_robust_objective(
                    u_new, surrogate_factory, xi_samples)
                if J_new < J_robust:
                    break
                alpha *= 0.5
                u_new = u - alpha * grad
                if u_bounds is not None:
                    for k in range(self.control_dim):
                        lo, hi = u_bounds[k]
                        u_new[k] = np.clip(u_new[k], lo, hi)

            u = u_new

        return {
            'optimal_control': u,
            'optimal_objective': history[-1]['objective'] if history else 0.0,
            'n_iterations': len(history),
            'history': history,
            'converged': history[-1]['grad_norm'] < tol if history else False
        }


# ---------------------------------------------------------------------------
# Chance Constraint Evaluation
# ---------------------------------------------------------------------------

def evaluate_chance_constraint(surrogate: 'PolynomialChaosSurrogate',
                               threshold: float,
                               n_samples: int = 5000,
                               lower: Optional[NDArray] = None,
                               upper: Optional[NDArray] = None,
                               seed: int = 42) -> Dict[str, float]:
    """
    Evaluate chance constraint: P(Y <= threshold) >= 1 - alpha.

    Uses MC sampling from the surrogate to estimate the probability.

    Parameters
    ----------
    surrogate : PolynomialChaosSurrogate
        PCE surrogate for the constraint function.
    threshold : float
        Constraint threshold.
    n_samples : int
        MC samples.
    lower, upper : ndarray, optional
        Parameter bounds.
    seed : int
        Random seed.

    Returns
    -------
    dict with:
        'probability': estimated P(Y <= threshold)
        'satisfied': bool (P >= 1 - alpha for alpha = 0.05)
        'violation_probability': 1 - P
        'margin': P - (1 - alpha)
    """
    d = surrogate.d
    if lower is None:
        lower = -np.ones(d)
    if upper is None:
        upper = np.ones(d)

    rng = np.random.RandomState(seed)
    X = lower + rng.random((n_samples, d)) * (upper - lower)
    Y = surrogate.predict(X)

    prob = float(np.mean(Y <= threshold))
    alpha = 0.05  # Default risk level

    return {
        'probability': prob,
        'satisfied': prob >= 1.0 - alpha,
        'violation_probability': 1.0 - prob,
        'margin': prob - (1.0 - alpha),
        'threshold': threshold
    }
