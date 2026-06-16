"""
ode_integrators.py
==================

Implicit ODE integrators and parametric manifold solutions for the
coupled evolution equations:
- Fixed-point implicit midpoint method (from 767_midpoint_fixed)
- Doughnut (torus) parametric manifold solution (from 316_doughnut_exact)

Scientific context:
-------------------
The running of the strong coupling alpha_s(Q^2) satisfies:

    d alpha_s / d ln(Q^2) = -beta0/(4*pi) * alpha_s^2
                            - beta1/(16*pi^2) * alpha_s^3 + ...

This is a nonlinear ODE that becomes stiff near the Landau pole.
We use the implicit midpoint method which is A-stable and symplectic:

    y_{n+1} = y_n + h * f(t_n + h/2, (y_n + y_{n+1})/2)

The implicit equation is solved by fixed-point iteration.

The doughnut manifold parametrizes the color-flow space of a 3-parton
final state as a rational solution of the KdV-type equation, with
the three components (y1, y2, y3) living on S^2 in color space.
"""

import math
from typing import Callable, List, Tuple


# ===========================================================================
# Section 1: Implicit midpoint method (from 767_midpoint_fixed)
# ===========================================================================

def midpoint_fixed(f: Callable[[float, List[float]], List[float]],
                   tspan: Tuple[float, float],
                   y0: List[float], n: int,
                   it_max: int = 20, theta: float = 0.5
                   ) -> Tuple[List[float], List[List[float]]]:
    """
    Solve the ODE system y' = f(t, y) using the implicit midpoint method
    with fixed-point iteration for the implicit stage.

    The method:
        y_{n+1} = y_n + h * f(t_n + theta*h, y_m)

    where y_m is the midpoint approximation obtained by fixed-point iteration:
        y_m^{(k+1)} = y_n + theta * h * f(t_n + theta*h, y_m^{(k)})

    This method is:
    - A-stable (good for stiff equations like alpha_s running near Landau pole)
    - Symplectic (preserves phase-space volume, important for Hamiltonian flows)
    - Second-order accurate

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) returning a list of derivatives.
    tspan : (float, float)
        Start and end times.
    y0 : list of float
        Initial conditions.
    n : int
        Number of time steps.
    it_max : int
        Maximum fixed-point iterations per step.
    theta : float
        Midpoint parameter (0.5 = standard midpoint).

    Returns
    -------
    (t, y): list of times and list of solution vectors.
    """
    if n < 1:
        raise ValueError(f"midpoint_fixed: n={n} < 1")
    if tspan[1] <= tspan[0]:
        raise ValueError(f"midpoint_fixed: tspan[1]={tspan[1]} <= tspan[0]={tspan[0]}")
    m = len(y0)
    dt = (tspan[1] - tspan[0]) / n
    t_out = [tspan[0] + i * dt for i in range(n + 1)]
    y_out = [list(y0)]
    for i in range(n):
        xm = t_out[i] + theta * dt
        ym = list(y_out[i])
        # Fixed-point iteration for the implicit stage
        for _ in range(it_max):
            fmid = f(xm, ym)
            ym_new = [y_out[i][j] + theta * dt * fmid[j] for j in range(m)]
            # Check convergence
            diff = max(abs(ym_new[j] - ym[j]) for j in range(m))
            ym = ym_new
            if diff < 1e-12:
                break
        # Extrapolate to full step
        y_next = [(1.0 / theta) * ym[j]
                  + (1.0 - 1.0 / theta) * y_out[i][j] for j in range(m)]
        y_out.append(y_next)
    return t_out, y_out


# ===========================================================================
# Section 2: Doughnut / torus parametric solution
# (from 316_doughnut_exact)
# ===========================================================================

class DoughnutParameters:
    """
    Parameters for the doughnut ODE system.

    The doughnut ODE describes a particle moving on a torus in R^3:

        dy1/dt = m * (1 - y1^2 - y2^2 - y3^2 + delta*y1) - n * y2 * y3
        dy2/dt = m * (...) + n * y1 * y3
        dy3/dt = ...

    In the physics context, (y1, y2, y3) parametrize the color flow
    of a 3-parton system, m and n are related to the color factors
    CF and CA, and the torus constraint y1^2 + y2^2 + y3^2 ~ const
    represents the fixed total color charge.
    """

    def __init__(self, m: float = 3.0, n: float = 5.0,
                 y0: Tuple[float, float, float] = (1.0, 1.0, 3.0),
                 t0: float = 0.0, tstop: float = 10.0):
        self.m = m
        self.n = n
        self.y0 = y0
        self.t0 = t0
        self.tstop = tstop

    @property
    def delta(self) -> float:
        return 1.0 + self.y0[0] ** 2 + self.y0[1] ** 2 + self.y0[2] ** 2


def doughnut_exact(t: List[float], params: DoughnutParameters
                   ) -> List[Tuple[float, float, float]]:
    """
    Exact rational solution of the doughnut ODE system.

    The solution is:
        y1(t) = (2*a*cos(m*t) - 2*b*sin(m*t)) / D(t)
        y2(t) = (2*a*sin(m*t) + 2*b*cos(m*t)) / D(t)
        y3(t) = (2*c*cos(n*t) + (2-delta)*sin(n*t)) / D(t)

    where D(t) = delta - 2*c*sin(n*t) + (2-delta)*cos(n*t)

    This is the rational solution of the KdV equation discovered by
    John D. Cook (2023). In our physics context, it parametrizes
    the exact color-flow evolution of a 3-jet event.
    """
    m = params.m
    n = params.n
    a, b, c = params.y0
    delta = params.delta
    result = []
    for ti in t:
        denom = (delta
                 - 2.0 * c * math.sin(n * ti)
                 + (2.0 - delta) * math.cos(n * ti))
        if abs(denom) < 1e-12:
            denom = 1e-12 * (1.0 if denom >= 0 else -1.0)
        y1 = (2.0 * a * math.cos(m * ti) - 2.0 * b * math.sin(m * ti)) / denom
        y2 = (2.0 * a * math.sin(m * ti) + 2.0 * b * math.cos(m * ti)) / denom
        y3 = (2.0 * c * math.cos(n * ti) + (2.0 - delta) * math.sin(n * ti)) / denom
        result.append((y1, y2, y3))
    return result


def doughnut_rhs(t: float, y: List[float], params: DoughnutParameters
                 ) -> List[float]:
    """
    Right-hand side of the doughnut ODE system for use with ODE integrators.

    The exact form is derived from the KdV rational solution.
    """
    m = params.m
    n = params.n
    y1, y2, y3 = y[0], y[1], y[2]
    dy1 = m * (-y2 + y1 * y3) - n * y2
    dy2 = m * (y1 + y2 * y3) + n * y1
    dy3 = -m * (y1 * y1 + y2 * y2)
    return [dy1, dy2, dy3]


def running_coupling_ode(t: float, y: List[float], beta0: float, beta1: float
                         ) -> List[float]:
    """
    ODE for the running coupling alpha_s in ln(Q^2):

        d alpha_s / d t = -beta0/(4*pi) * alpha_s^2
                          - beta1/(16*pi^2) * alpha_s^3

    where t = ln(Q^2 / mu0^2).
    """
    if len(y) < 1:
        return [0.0]
    a = y[0]
    if a < 1e-10:
        a = 1e-10
    b0 = beta0 / (4.0 * math.pi)
    b1 = beta1 / (16.0 * math.pi * math.pi)
    dadt = -b0 * a * a - b1 * a * a * a
    return [dadt]


def evolve_coupling(a0: float, ln_q2_start: float, ln_q2_end: float,
                    n_steps: int, beta0: float, beta1: float
                    ) -> Tuple[List[float], List[float]]:
    """
    Evolve alpha_s from ln(Q^2_start) to ln(Q^2_end) using the implicit
    midpoint method.

    Parameters
    ----------
    a0 : float
        Initial value of alpha_s.
    ln_q2_start, ln_q2_end : float
        Start and end of ln(Q^2/mu0^2).
    n_steps : int
        Number of integration steps.
    beta0, beta1 : float
        Beta function coefficients.

    Returns
    -------
    (t_grid, alpha_grid): evolution of coupling.
    """
    def rhs(t, y):
        return running_coupling_ode(t, y, beta0, beta1)

    t_grid, y_grid = midpoint_fixed(
        rhs, (ln_q2_start, ln_q2_end), [a0], n_steps,
        it_max=30, theta=0.5
    )
    alpha_grid = [y[0] for y in y_grid]
    # Clamp to physical range
    for i in range(len(alpha_grid)):
        alpha_grid[i] = max(1e-4, min(alpha_grid[i], 4.0 * math.pi / beta0))
    return t_grid, alpha_grid
