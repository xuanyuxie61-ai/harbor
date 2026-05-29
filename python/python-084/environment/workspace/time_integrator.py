# -*- coding: utf-8 -*-
"""
time_integrator.py
==================
Implicit time integration for nonlinear structural dynamics.

Incorporates ideas from two seed projects:
  - 826_ode_euler_backward:  Backward Euler implicit ODE solver
  - 377_fem_neumann:         Mass-matrix formulation of FEM ODE system

Governing equation of motion:
    M * a(t) + C * v(t) + K * u(t) + F_iso(u, v) = -M * Gamma * a_g(t)

where:
    u(t) = displacement vector
    v(t) = velocity vector    = du/dt
    a(t) = acceleration vector = d^2u/dt^2

We use the Newmark-beta method (unconditionally stable for beta >= 0.25,
gamma >= 0.5).  The constant-average-acceleration variant is:
    beta  = 0.25
    gamma = 0.50

Newmark update formulas:
    a_{n+1} = (1 / (beta * dt^2)) * (u_{n+1} - u_n) - (1 / (beta * dt)) * v_n - (1/(2*beta) - 1) * a_n
    v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})

Effective stiffness:
    K_eff = K + (gamma / (beta * dt)) * C + (1 / (beta * dt^2)) * M

Effective residual:
    R_eff = -M * Gamma * a_g(t_{n+1}) - M * a_n^* - C * v_n^* - F_iso(u_{n+1}^{(k)}, v_{n+1}^{(k)})

where the predictors are:
    a_n^* = (1 / (2*beta) - 1) * a_n + (1 / (beta * dt)) * v_n + (1 / (beta * dt^2)) * u_n
    v_n^* = v_n + dt * (1 - gamma) * a_n
"""

import numpy as np
from typing import Tuple, Optional, Callable


class NewmarkBetaIntegrator:
    """
    Newmark-beta implicit integrator for nonlinear MDOF structural dynamics.
    
    Parameters
    ----------
    M, C, K : np.ndarray
        Mass, damping, and stiffness matrices.
    gamma : float
        Newmark gamma parameter (default 0.5 for constant avg acceleration).
    beta : float
        Newmark beta parameter (default 0.25 for constant avg acceleration).
    dt : float
        Time step [s].
    max_iter : int
        Maximum Newton-Raphson iterations per step (default 10).
    tol : float
        Convergence tolerance for displacement increment (default 1e-8).
    """

    def __init__(
        self,
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        gamma: float = 0.5,
        beta: float = 0.25,
        dt: float = 0.01,
        max_iter: int = 10,
        tol: float = 1e-8,
    ):
        self.M = np.asarray(M, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.dt = float(dt)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_dof = self.M.shape[0]

        # TODO: Hole 1 - Implement effective stiffness precomputation for Newmark-beta
        # K_eff = K + (gamma/(beta*dt)) * C + (1/(beta*dt^2)) * M
        pass

    # ------------------------------------------------------------------ #
    # Single time step (Newton-Raphson for nonlinearity)
    # ------------------------------------------------------------------ #
    def step(
        self,
        u_n: np.ndarray,
        v_n: np.ndarray,
        a_n: np.ndarray,
        a_g: float,
        iso_force_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        solver_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Gamma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance one time step from (u_n, v_n, a_n) to (u_{n+1}, v_{n+1}, a_{n+1}).
        
        Parameters
        ----------
        u_n, v_n, a_n : np.ndarray
            Current displacement, velocity, acceleration.
        a_g : float
            Ground acceleration at t_{n+1}.
        iso_force_func : callable
            Function iso_force(u, v) returning isolation restoring force.
        solver_func : callable
            Function solver(A, b) returning x = A^{-1} * b.
        Gamma : np.ndarray
            Influence vector.
        
        Returns
        -------
        u_{n+1}, v_{n+1}, a_{n+1}
        """
        dt = self.dt
        beta = self.beta
        gamma = self.gamma

        # Predictors
        a_star = (
            (1.0 / (2.0 * beta) - 1.0) * a_n
            + (1.0 / (beta * dt)) * v_n
            + (1.0 / (beta * dt ** 2)) * u_n
        )
        v_star = v_n + dt * (1.0 - gamma) * a_n

        # External force from ground acceleration
        F_ext = -self.M @ Gamma * a_g

        # Initial guess: u_{n+1} = u_n (zero incremental displacement)
        u_new = u_n.copy()

        # Newton-Raphson iteration for nonlinear isolation force
        for _iter in range(self.max_iter):
            # Velocity at n+1 from displacement guess
            a_new = (1.0 / (beta * dt ** 2)) * (u_new - u_n) - (1.0 / (beta * dt)) * v_n - (1.0 / (2.0 * beta) - 1.0) * a_n
            v_new = v_star + gamma * dt * a_new

            # Isolation force
            F_iso = iso_force_func(u_new, v_new)

            # Residual
            R = F_ext - self.M @ a_new - self.C @ v_new - self.K @ u_new - F_iso

            # Effective stiffness may need to account for tangent stiffness of isolation
            # For simplicity, we use the precomputed K_eff and add isolation tangent
            # if needed.  Here we assume iso_force is handled explicitly (modified NR).
            K_eff = self._K_eff.copy()

            # Solve for displacement increment
            du = solver_func(K_eff, R)

            # Update
            u_new = u_new + du

            # Check convergence
            norm_du = float(np.linalg.norm(du))
            norm_u = float(np.linalg.norm(u_new))
            if norm_u > 1e-12:
                rel_err = norm_du / norm_u
            else:
                rel_err = norm_du

            if rel_err < self.tol:
                break
        else:
            # Did not converge within max_iter; accept last value but warn
            pass

        # Final state
        a_new = (
            (1.0 / (beta * dt ** 2)) * (u_new - u_n)
            - (1.0 / (beta * dt)) * v_n
            - (1.0 / (2.0 * beta) - 1.0) * a_n
        )
        v_new = v_star + gamma * dt * a_new

        return u_new, v_new, a_new

    # ------------------------------------------------------------------ #
    # Full time-history integration
    # ------------------------------------------------------------------ #
    def integrate(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        a_g: np.ndarray,
        iso_force_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        solver_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        Gamma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate over the full ground-motion record.
        
        Parameters
        ----------
        u0, v0, a0 : np.ndarray
            Initial displacement, velocity, acceleration.
        a_g : np.ndarray
            Ground acceleration time history.
        
        Returns
        -------
        U, V, A : np.ndarray, shape (n_time, n_dof)
            Displacement, velocity, acceleration histories.
        """
        n_time = len(a_g)
        n_dof = self.n_dof

        U = np.zeros((n_time, n_dof), dtype=float)
        V = np.zeros((n_time, n_dof), dtype=float)
        A = np.zeros((n_time, n_dof), dtype=float)

        U[0, :] = u0
        V[0, :] = v0
        A[0, :] = a0

        u_n = u0.copy()
        v_n = v0.copy()
        a_n = a0.copy()

        for i in range(1, n_time):
            u_n, v_n, a_n = self.step(
                u_n, v_n, a_n, a_g[i], iso_force_func, solver_func, Gamma
            )
            U[i, :] = u_n
            V[i, :] = v_n
            A[i, :] = a_n

        return U, V, A


# ====================================================================== #
# Backward Euler verification solver (from 826_ode_euler_backward seed)
# ====================================================================== #
def backward_euler_step(
    y_n: np.ndarray,
    f_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
    max_inner_iter: int = 10,
) -> np.ndarray:
    """
    Single step of backward Euler for a first-order ODE system:
        dy/dt = f(y)
        y_{n+1} = y_n + dt * f(y_{n+1})
    
    Solved by fixed-point iteration (Picard iteration):
        y^{(k+1)} = y_n + dt * f(y^{(k)})
    
    This is used as a verification tool for low-order accuracy checks
    on simplified single-DOF subsystems.
    """
    y = y_n.copy()
    for _ in range(max_inner_iter):
        y_new = y_n + dt * f_func(y)
        if np.linalg.norm(y_new - y) < 1e-12:
            break
        y = y_new
    return y
