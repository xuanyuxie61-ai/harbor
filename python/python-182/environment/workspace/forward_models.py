"""
forward_models.py

Forward physical models for the Bayesian inference engine:
  - FitzHugh-Nagumo reaction kinetics (ODE + Euler time-stepping)
  - Helmholtz wave equation exact solution (Bessel separation of variables)

Reimplemented from seed projects:
  - fitzhugh_nagumo_ode + euler: excitable media dynamics
  - helmholtz_exact + besselzero: vibrating membrane wave fields
"""
import math
import numpy as np
from scipy.special import jv, yv


def fitzhugh_nagumo_deriv(t: float, y: np.ndarray, a: float, b: float, c: float, d: float):
    """
    Right-hand side of the FitzHugh-Nagumo system:

        dv/dt = v - v^3/3 - w + d
        dw/dt = (v + a - b*w) / c

    Parameters:
        t: time (unused, autonomous system)
        y: array [v, w]
        a, b, c, d: model parameters

    Returns:
        dydt: array [dv/dt, dw/dt]
    """
    v = y[0]
    w = y[1]
    dvdt = v - (v ** 3) / 3.0 - w + d
    dwdt = (v + a - b * w) / c
    return np.array([dvdt, dwdt], dtype=float)


def euler_integrate(dydt, tspan, y0: np.ndarray, n: int, **kwargs):
    """
    Classical forward Euler explicit integration.

        y_{i+1} = y_i + dt * f(t_i, y_i)
        dt = (tstop - t0) / n

    Parameters:
        dydt: callable (t, y, **kwargs) -> dy/dt
        tspan: (t0, tstop)
        y0: initial condition vector
        n: number of steps
        **kwargs: passed to dydt

    Returns:
        t: array of length n+1
        y: array of shape (n+1, m)
    """
    t0, tstop = float(tspan[0]), float(tspan[1])
    if n < 1:
        raise ValueError("euler_integrate: n must be >= 1")
    dt = (tstop - t0) / n
    m = len(y0)
    t = np.empty(n + 1, dtype=float)
    y = np.empty((n + 1, m), dtype=float)
    t[0] = t0
    y[0, :] = np.asarray(y0, dtype=float)
    for i in range(n):
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * np.asarray(dydt(t[i], y[i, :], **kwargs), dtype=float)
    return t, y


def _findzero(n: int, kind: int, x0: float):
    """
    Halley's method for finding a zero of J_n (kind=1) or Y_n (kind=2).
    Reimplemented from besselzero.m.
    """
    ITERATIONS_MAX = 100
    TOLERANCE_RELATIVE = 1e4
    error = 1.0
    loop_count = 0
    x = x0
    while math.fabs(error) > math.fabs(x) * np.finfo(float).eps * TOLERANCE_RELATIVE and loop_count < ITERATIONS_MAX:
        if kind == 1:
            a = jv(n, x)
            b = jv(n + 1, x)
        else:
            a = yv(n, x)
            b = yv(n + 1, x)
        x2 = x * x
        denom = (2.0 * b * b * x2 - a * b * x * (4.0 * n + 1.0)
                 + (n * (n + 1.0) + x2) * a * a)
        if abs(denom) < 1e-30:
            break
        error = 2.0 * a * x * (n * a - b * x) / denom
        x = x - error
        loop_count += 1
    return x


def besselzero(n: int, k: int, kind: int = 1):
    """
    Calculate the first k positive zeros of the n-th order Bessel function
    using least-squares initial guesses and Halley's method.

    Parameters:
        n: order (non-negative real)
        k: number of zeros
        kind: 1 for J_n, 2 for Y_n

    Returns:
        zeros: array of length k
    """
    if k < 1:
        raise ValueError("besselzero: k must be >= 1")
    n = abs(n)
    zeros = np.empty(k, dtype=float)

    if kind == 1:
        coeff1 = np.array([0.411557013144507, 0.999986723293410,
                           0.698028985524484, 1.06977507291468])
        exp1 = np.array([0.335300369843979, 0.339671493811664])
        guess = (coeff1[0] + coeff1[1] * n
                 + coeff1[2] * (n + 1.0) ** exp1[0]
                 + coeff1[3] * (n + 1.0) ** exp1[1])
        zeros[0] = _findzero(n, kind, guess)

        if k >= 2:
            coeff2 = np.array([1.93395115137444, 1.00007656297072,
                               -0.805720018377132, 3.38764629174694])
            exp2 = np.array([0.456215294517928, 0.388380341189200])
            guess = (coeff2[0] + coeff2[1] * n
                     + coeff2[2] * (n + 1.0) ** exp2[0]
                     + coeff2[3] * (n + 1.0) ** exp2[1])
            zeros[1] = _findzero(n, kind, guess)

        if k >= 3:
            coeff3 = np.array([5.40770803992613, 1.00093850589418,
                               2.66926179799040, -0.174925559314932])
            exp3 = np.array([0.429702214054531, 0.633480051735955])
            guess = (coeff3[0] + coeff3[1] * n
                     + coeff3[2] * (n + 1.0) ** exp3[0]
                     + coeff3[3] * (n + 1.0) ** exp3[1])
            zeros[2] = _findzero(n, kind, guess)
    else:
        coeff1 = np.array([0.0795046982450635, 0.999998378297752,
                           0.890380645613825, 0.0270604048106402])
        exp1 = np.array([0.335377217953294, 0.308720059086699])
        guess = (coeff1[0] + coeff1[1] * n
                 + coeff1[2] * (n + 1.0) ** exp1[0]
                 + coeff1[3] * (n + 1.0) ** exp1[1])
        zeros[0] = _findzero(n, kind, guess)

        if k >= 2:
            coeff2 = np.array([1.04502538172394, 1.00002054874161,
                               -0.437921325402985, 2.70113114990400])
            exp2 = np.array([0.434823025111322, 0.366245194174671])
            guess = (coeff2[0] + coeff2[1] * n
                     + coeff2[2] * (n + 1.0) ** exp2[0]
                     + coeff2[3] * (n + 1.0) ** exp2[1])
            zeros[1] = _findzero(n, kind, guess)

        if k >= 3:
            coeff3 = np.array([3.72777931751914, 1.00035294977757,
                               2.68566718444899, -0.112980454967090])
            exp3 = np.array([0.398247585896959, 0.604770035236606])
            guess = (coeff3[0] + coeff3[1] * n
                     + coeff3[2] * (n + 1.0) ** exp3[0]
                     + coeff3[3] * (n + 1.0) ** exp3[1])
            zeros[2] = _findzero(n, kind, guess)

    # Higher roots via linear extrapolation of spacing
    for iroot in range(3, k):
        guess = 2.0 * zeros[iroot - 1] - zeros[iroot - 2]
        zeros[iroot] = _findzero(n, kind, guess)

    return zeros


def helmholtz_exact(a: float, m: int, n: int, alpha: float, beta: float,
                    gamma: float, x: np.ndarray, y: np.ndarray):
    """
    Exact solution of the Helmholtz equation on a disk of radius a:

        Del^2 Z = -k^2 Z,   Z(a, theta) = 0

    via separation of variables:
        Z(r,theta) = gamma * J_n(k*r) * (alpha*cos(n*theta) + beta*sin(n*theta))
    where k = rho(m,n) / a and rho(m,n) is the m-th positive zero of J_n.

    Parameters:
        a: disk radius
        m: index of Bessel zero (m>=1 for n=0, m>=0 otherwise)
        n: order of Bessel function
        alpha, beta, gamma: angular and radial coefficients
        x, y: Cartesian coordinates (arrays)

    Returns:
        Z: solution values at (x,y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    T = alpha * np.cos(n * theta) + beta * np.sin(n * theta)
    if m == 0:
        if n == 0:
            raise ValueError("helmholtz_exact: m=0 is illegal when n=0")
        rho = 0.0
    else:
        rho_vec = besselzero(n, m, 1)
        rho = rho_vec[m - 1] * r / a
    R = gamma * jv(n, rho)
    return R * T


def fhn_stationary_voltage(a: float, b: float, c: float, d: float,
                           tspan=(0.0, 20.0), n_steps: int = 30):
    """
    Integrate FitzHugh-Nagumo from rest (v=0, w=0) and return the
    membrane voltage v at the final time.
    """
    y0 = np.array([0.0, 0.0], dtype=float)
    t, y = euler_integrate(fitzhugh_nagumo_deriv, tspan, y0, n_steps,
                           a=a, b=b, c=c, d=d)
    return float(y[-1, 0])
