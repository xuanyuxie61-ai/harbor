"""
stokes_manufactured.py
======================
Manufactured exact solutions for the 2D incompressible Stokes equations.
These are used to verify the convergence of the domain-decomposition
spectral-element solver.

Integrates concepts from:
  * stokes_2d_exact (exact solutions and residuals)

Mathematical background
-----------------------
The steady incompressible Stokes equations on domain Omega are:
    -nu * nabla^2 u + dp/dx = f_x     (momentum x)
    -nu * nabla^2 v + dp/dy = f_y     (momentum y)
    du/dx + dv/dy = 0                  (incompressibility)

For transient Stokes, we add:
    du/dt = -nu * nabla^2 u + dp/dx - f_x
    dv/dt = -nu * nabla^2 v + dp/dy - f_y

Manufactured solution: prescribe smooth u(x,y,t), v(x,y,t), p(x,y,t)
and compute the forcing (f_x, f_y) that makes them exact solutions.

Solution 1 (polynomial, divergence-free):
    u(x,y) =  2 * x^2 * (x-1)^2 * y * (2*y-1) * (y-1)
    v(x,y) = -2 * x * (2*x-1) * (x-1) * y^2 * (y-1)^2
    p(x,y) = x * (1-x) * y * (1-y)

Solution 2 (trigonometric):
    u(x,y) =  sin(pi*x) * cos(pi*y)
    v(x,y) = -cos(pi*x) * sin(pi*y)
    p(x,y) =  sin(pi*x) * sin(pi*y)

Solution 3 (Kovasznay flow, modified):
    Based on a recirculating flow with analytic expressions.
"""

import numpy as np
from typing import Tuple


def stokes_solution_polynomial(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Polynomial manufactured solution that is exactly divergence-free
    and vanishes on the boundary of [0,1]^2.
    """
    u = 2.0 * x ** 2 * (x - 1.0) ** 2 * y * (2.0 * y - 1.0) * (y - 1.0)
    v = -2.0 * x * (2.0 * x - 1.0) * (x - 1.0) * y ** 2 * (y - 1.0) ** 2
    p = x * (1.0 - x) * y * (1.0 - y)
    return u, v, p


def stokes_rhs_polynomial(x: np.ndarray, y: np.ndarray, nu: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the forcing (fx, fy, h) for the polynomial manufactured solution.
    h should be zero because the velocity is divergence-free.
    """
    # TODO: Implement the analytical derivation of forcing terms for the
    # polynomial manufactured solution. Must compute:
    #   - Second derivatives d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
    #   - Pressure gradients dpdx, dpdy
    #   - Momentum forcing: fx = -nu*(d2u_dx2 + d2u_dy2) + dpdx
    #   - Momentum forcing: fy = -nu*(d2v_dx2 + d2v_dy2) + dpdy
    #   - Divergence h = du/dx + dv/dy (should be zero)
    # This requires symbolic/analytical differentiation of the polynomial
    # velocity and pressure fields defined in stokes_solution_polynomial.
    raise NotImplementedError("Hole 1: stokes_rhs_polynomial 需要补全制造解的解析推导")


def stokes_solution_trigonometric(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trigonometric manufactured solution.
    """
    u = np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y)
    p = np.sin(np.pi * x) * np.sin(np.pi * y)
    return u, v, p


def stokes_rhs_trigonometric(x: np.ndarray, y: np.ndarray, nu: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forcing for trigonometric solution.
    u = sin(pi x) cos(pi y)
    v = -cos(pi x) sin(pi y)
    p = sin(pi x) sin(pi y)

    nabla^2 u = -2 pi^2 sin(pi x) cos(pi y)
    nabla^2 v = 2 pi^2 cos(pi x) sin(pi y)
    dp/dx = pi cos(pi x) sin(pi y)
    dp/dy = pi sin(pi x) cos(pi y)
    """
    pi2 = np.pi ** 2
    d2u = -2.0 * pi2 * np.sin(np.pi * x) * np.cos(np.pi * y)
    d2v = 2.0 * pi2 * np.cos(np.pi * x) * np.sin(np.pi * y)
    dpdx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dpdy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    fx = -nu * d2u + dpdx
    fy = -nu * d2v + dpdy
    h = np.zeros_like(x)
    return fx, fy, h


def stokes_solution_kovasznay(x: np.ndarray, y: np.ndarray, Re: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modified Kovasznay flow (steady, nonlinear term small at low Re).
    This is a classic benchmark for Navier-Stokes solvers.
    """
    if Re <= 0:
        Re = 40.0
    lambda_k = Re / 2.0 - np.sqrt(Re ** 2 / 4.0 + 4.0 * np.pi ** 2)
    u = 1.0 - np.exp(lambda_k * x) * np.cos(2.0 * np.pi * y)
    v = lambda_k / (2.0 * np.pi) * np.exp(lambda_k * x) * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lambda_k * x))
    return u, v, p


def evaluate_solution(
    x: np.ndarray, y: np.ndarray,
    sol_type: str = "polynomial",
    nu: float = 1.0, Re: float = 40.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate manufactured solution and its forcing.

    Returns
    -------
    u, v, p, fx, fy, h
    """
    if sol_type == "polynomial":
        u, v, p = stokes_solution_polynomial(x, y)
        fx, fy, h = stokes_rhs_polynomial(x, y, nu)
    elif sol_type == "trigonometric":
        u, v, p = stokes_solution_trigonometric(x, y)
        fx, fy, h = stokes_rhs_trigonometric(x, y, nu)
    elif sol_type == "kovasznay":
        u, v, p = stokes_solution_kovasznay(x, y, Re)
        # For Stokes, we approximate by using only viscous and pressure terms
        fx = np.zeros_like(x)
        fy = np.zeros_like(x)
        h = np.zeros_like(x)
    else:
        raise ValueError(f"Unknown solution type: {sol_type}")
    return u, v, p, fx, fy, h


def compute_discrete_residual(
    u_h: np.ndarray, v_h: np.ndarray, p_h: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    nu: float = 1.0,
    sol_type: str = "polynomial"
) -> float:
    """
    Compute L2-like discrete residual against manufactured solution.
    """
    u_ex, v_ex, p_ex, _, _, _ = evaluate_solution(x, y, sol_type, nu)
    err_u = np.linalg.norm(u_h - u_ex) / max(1.0, np.linalg.norm(u_ex))
    err_v = np.linalg.norm(v_h - v_ex) / max(1.0, np.linalg.norm(v_ex))
    err_p = np.linalg.norm(p_h - p_ex) / max(1.0, np.linalg.norm(p_ex))
    return float(np.sqrt(err_u ** 2 + err_v ** 2 + err_p ** 2))
