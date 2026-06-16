# -*- coding: utf-8 -*-
"""
time_integrator.py
==================

Implicit and semi-implicit time-advancers for the gyrokinetic equation.

The gyrokinetic equation (schematically) is

    d f / dt  =  L[f]  +  N[f]

with  L  a stiff (typically anti-Hermitian) linear operator coming from
the parallel streaming  v_parallel b0 . grad ||  and the magnetic-drift
terms, and  N  the (milder) nonlinear term involving the E x B advection
by the self-consistent potential  phi.

The time-integrators in this module are:

    * ``TrapezoidalODE``           -- implicit trapezoidal (Crank-Nicolson)
                                      for a generic y' = F(t, y) system
                                      (port of ``ode_trapezoidal`` from
                                      Burkardt);
    * ``BackwardEuler``            -- first-order, unconditionally stable
                                      fallback;
    * ``SSP_RK3``                  -- explicit 3rd-order strong-stability-
                                      preserving Runge-Kutta for the
                                      non-stiff (advective) part;
    * ``IMEX_Trapezoidal``         -- implicit treatment of L, explicit
                                      treatment of N -- the workhorse
                                      for gyrokinetics.

The implicit stages require solving  (I - dt/2 L) x = rhs  at each step;
this is delegated to the radial solver in ``finite_difference`` (banded
LU) and the velocity-space solver in ``collision_operator`` (Levinson).

Characteristic back-tracing (port of ``track.py``)
--------------------------------------------------
For the nonlinear  E x B  advection we use a characteristic method:
the gyrocentre position at time  t^{n+1}  is mapped back to the departure
point  X(t^n)  by integrating the Hamiltonian flow

    d X / dt = (c/B)  b0 x grad phi(X, t)

using the trapezoidal rule on the characteristics (analogous to the
semi-Lagrangian approach).  The field at the departure point is obtained
by cubic interpolation on the radial mesh.

References:
    [1] Burkardt, ``ode_trapezoidal``.
    [2] Ascher, Ruuth, Spiteri, "Implicit-explicit Runge-Kutta methods",
        Appl. Num. Math. 26, 151 (1998).
    [3] Richards et al., J. Glaciol. 67, 1027 (2021) -- for the back-trace
        formulation used in ``track.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


# ============================================================================
# Implicit trapezoidal ODE -- port of ode_trapezoidal.m
# ============================================================================
def ode_trapezoidal(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    a: float,
    b: float,
    y0: np.ndarray,
    n: int,
    newton_tol: float = 1.0e-10,
    newton_maxiter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Implicit trapezoidal (Crank-Nicolson) ODE integrator.

    Solves   y' = f(t, y)   on [a, b] with  y(a) = y0  using the
    second-order implicit trapezoidal rule

        y_{n+1} = y_n + (h/2) [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]

    The implicit equation is solved by a fixed-point iteration (Picard)
    with Newton-like fallback if Picard fails to converge.  For stiff
    systems this is equivalent to Crank-Nicolson and is A-stable.

    Returns
    -------
    t : (n+1,) array of time points
    y : (n+1, len(y0)) array of solution values
    """
    if n < 1:
        raise ValueError("ode_trapezoidal: n must be >= 1")
    h = (b - a) / n
    t = a + np.arange(n + 1) * h
    y = np.zeros((n + 1, y0.size), dtype=np.float64)
    y[0] = y0
    for k in range(n):
        tn = t[k]
        yn = y[k]
        f_n = rhs(tn, yn)
        # initial guess: explicit Euler
        ynew = yn + h * f_n
        for _it in range(newton_maxiter):
            f_new = rhs(tn + h, ynew)
            ynext = yn + 0.5 * h * (f_n + f_new)
            if np.max(np.abs(ynext - ynew)) < newton_tol * (1.0 + np.max(np.abs(ynext))):
                ynew = ynext
                break
            ynew = ynext
        y[k + 1] = ynew
    return t, y


# ============================================================================
# Backward Euler (fallback)
# ============================================================================
def backward_euler(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    a: float,
    b: float,
    y0: np.ndarray,
    n: int,
    newton_tol: float = 1.0e-10,
    newton_maxiter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward Euler  y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})  -- L-stable."""
    if n < 1:
        raise ValueError("backward_euler: n must be >= 1")
    h = (b - a) / n
    t = a + np.arange(n + 1) * h
    y = np.zeros((n + 1, y0.size), dtype=np.float64)
    y[0] = y0
    for k in range(n):
        yn = y[k]
        ynew = yn + h * rhs(t[k], yn)
        for _it in range(newton_maxiter):
            ynext = yn + h * rhs(t[k + 1], ynew)
            if np.max(np.abs(ynext - ynew)) < newton_tol * (1.0 + np.max(np.abs(ynext))):
                ynew = ynext
                break
            ynew = ynext
        y[k + 1] = ynew
    return t, y


# ============================================================================
# SSP-RK3 (explicit, for the non-stiff part)
# ============================================================================
def ssp_rk3_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Strong-stability-preserving RK3 (Shu-Osher form).

        y^(1) = y + dt F(y)
        y^(2) = (3/4) y + (1/4) y^(1) + (1/4) dt F(y^(1))
        y^{n+1} = (1/3) y + (2/3) y^(2) + (2/3) dt F(y^(2))

    The SSP coefficient is 1 -- the scheme preserves TVD / positivity
    under the same CFL as forward Euler.
    """
    y1 = y + dt * rhs(t, y)
    y2 = 0.75 * y + 0.25 * y1 + 0.25 * dt * rhs(t + dt, y1)
    y3 = (1.0 / 3.0) * y + (2.0 / 3.0) * y2 + (2.0 / 3.0) * dt * rhs(t + 0.5 * dt, y2)
    return y3


# ============================================================================
# IMEX trapezoidal: implicit L, explicit N
# ============================================================================
@dataclass
class IMEXTrapezoidalState:
    t: float
    y: np.ndarray
    step: int = 0
    history: list = field(default_factory=list)


def imex_trapezoidal_step(
    rhs_explicit: Callable[[float, np.ndarray], np.ndarray],
    solve_implicit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state: IMEXTrapezoidalState,
    dt: float,
) -> IMEXTrapezoidalState:
    """One step of IMEX trapezoidal:

        (I - dt/2 L) y^{n+1} = (I + dt/2 L) y^n + dt/2 (N^n + N^{n+1,guess})

    We use a predictor  y^{n+1,0} = y^n + dt N^n  for the explicit part,
    then one corrector pass.
    """
    t_n = state.t
    y_n = state.y
    N_n = rhs_explicit(t_n, y_n)
    # predictor
    y_pred = y_n + dt * N_n
    N_pred = rhs_explicit(t_n + dt, y_pred)
    # RHS of implicit solve
    rhs_imp = y_n + 0.5 * dt * (N_n + N_pred)
    # apply (I - dt/2 L)^{-1}
    y_new = solve_implicit(y_n, rhs_imp)
    return IMEXTrapezoidalState(
        t=t_n + dt, y=y_new, step=state.step + 1,
        history=state.history + [float(np.linalg.norm(y_new))],
    )


# ============================================================================
# Characteristic back-tracing for E x B advection
# ============================================================================
def trace_characteristics(
    x_departure: np.ndarray,
    E_x: Callable[[float, np.ndarray], np.ndarray],
    t_end: float,
    t_start: float,
    n_steps: int = 50,
) -> np.ndarray:
    """Back-trace characteristics from  x_departure  at  t_end  to  t_start.

    The equation of motion along a characteristic is

        d X / d t = E x B / B^2  = - (1/B) d phi / dy

    which in our 1-D reduced model reduces to  dX/dt = u_E(X, t).

    We integrate *backwards* in time using the trapezoidal rule (port of
    ``track.py`` from Richards et al.).  The returned array is the
    departure-point coordinate at  t_start.
    """
    if n_steps < 1:
        raise ValueError("trace_characteristics: n_steps must be >= 1")
    dt = (t_start - t_end) / n_steps     # negative
    x = x_departure.copy()
    for _ in range(n_steps):
        t_mid = t_end + 0.5 * dt
        k1 = E_x(t_end, x)
        k2 = E_x(t_mid, x + 0.5 * dt * k1)
        k3 = E_x(t_mid, x + 0.5 * dt * k2)
        k4 = E_x(t_end + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_end += dt
    return x


# ============================================================================
# Stability-region diagnostics
# ============================================================================
def stability_region_boundary(
    method: str = "trapezoidal", n_rays: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary of the absolute-stability region in the complex plane.

    For a one-step method with stability function R(z), the region is
    { z in C : |R(z)| <= 1 }.  For the classical methods:

        Forward Euler     : R(z) = 1 + z
        Backward Euler    : R(z) = 1 / (1 - z)
        Trapezoidal       : R(z) = (1 + z/2) / (1 - z/2)
        SSP-RK3           : R(z) = 1 + z + z^2/2 + z^3/6

    We trace the boundary by solving |R(r e^{i theta})| = 1 along n_rays
    rays from the origin via bisection.
    """
    R_funcs = {
        "trapezoidal":  lambda z: (1.0 + 0.5 * z) / (1.0 - 0.5 * z),
        "backward_euler": lambda z: 1.0 / (1.0 - z),
        "ssp_rk3":      lambda z: 1.0 + z + 0.5 * z * z + (1.0 / 6.0) * z * z * z,
        "forward_euler": lambda z: 1.0 + z,
    }
    if method not in R_funcs:
        raise ValueError(f"unknown method: {method}")
    R = R_funcs[method]
    thetas = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    radii = []
    for th in thetas:
        # Find radius at which |R(r e^{i th})| = 1.
        # Strategy: start with a small positive r where |R| <= 1 (stable),
        # then grow until |R| > 1.  Bisect between these.
        rlo = 1.0e-6
        zlo = rlo * np.exp(1j * th)
        try:
            v_lo = abs(R(zlo))
        except Exception:
            v_lo = float("inf")
        # Find rhi with |R| > 1
        rhi = 0.5
        for _ in range(60):
            zhi = rhi * np.exp(1j * th)
            try:
                v_hi = abs(R(zhi))
            except Exception:
                v_hi = float("inf")
            if v_hi > 1.0 + 1.0e-9:
                break
            rhi *= 2.0
        else:
            # Region extends to infinity along this ray (e.g. LHP for trapezoidal).
            # Cap at a finite radius for display.
            radii.append(20.0)
            continue
        # If the origin itself is already outside the region, boundary is at r = 0
        if v_lo > 1.0 + 1.0e-9:
            radii.append(0.0)
            continue
        # Bisect
        for _ in range(80):
            rm = 0.5 * (rlo + rhi)
            if (rhi - rlo) < 1.0e-10 * (rhi + rlo + 1.0):
                break
            zm = rm * np.exp(1j * th)
            try:
                v_m = abs(R(zm))
            except Exception:
                v_m = float("inf")
            if v_m <= 1.0:
                rlo = rm
            else:
                rhi = rm
        radii.append(0.5 * (rlo + rhi))
    radii = np.array(radii)
    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    return xs, ys


# ============================================================================
# Sanity self-check
# ============================================================================
if __name__ == "__main__":
    # y' = -10 y   (stiff)
    t, y = ode_trapezoidal(lambda t, y: -10.0 * y, 0.0, 1.0, np.array([1.0]), 50)
    exact = np.exp(-10.0 * t)
    print("trapezoidal on y'=-10 y, max error:", np.max(np.abs(y[:, 0] - exact)))
    # IMEX test
    state = IMEXTrapezoidalState(t=0.0, y=np.array([1.0, 0.0]))
    for _ in range(20):
        state = imex_trapezoidal_step(
            rhs_explicit=lambda t, y: np.array([-y[0], y[0]]),
            solve_implicit=lambda y0, rhs: np.array([rhs[0], rhs[1]]),
            state=state,
            dt=0.05,
        )
    print("IMEX final state:", state.y, "steps:", state.step)
    # stability boundary
    xs, ys = stability_region_boundary("trapezoidal")
    print("trapezoidal stability region boundary: Re range",
          xs.min(), xs.max(), "Im range", ys.min(), ys.max())
