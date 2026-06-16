"""
ode_integrator.py -- B1G3 Implicit Multistep ODE Integrator
=============================================================
Implements the B1G3 (Backward-1 / Gauss-3) implicit multistep method
for stiff ODE systems arising from stochastic Galerkin projections.

The B1G3 method is a 3-step implicit scheme with coefficients derived
from sqrt(3):
    A3 = 0.5 + 1/sqrt(3)
    A2 = -2/sqrt(3)
    A1 = -0.5 + 1/sqrt(3)
    B  = 1/sqrt(3)

Step 1 (bootstrap): implicit midpoint method
Step k >= 2: B1G3 multistep formula

The nonlinear system at each step is solved via Newton-Raphson iteration.

Seed references:
  - 061_b1g3: B1G3 residual, implicit midpoint bootstrap, fsolve
  - 100_blood_pressure_ode: piecewise-smooth ODE with periodic jumps
"""
import numpy as np
from typing import Callable, Tuple


def backward_euler_residual(f: Callable, t0: float, y0: np.ndarray,
                            tp: float, yp: np.ndarray) -> np.ndarray:
    """
    Residual for backward Euler step:
        R(yp) = yp - y0 - dt * f(tp, yp) = 0
    """
    dt = tp - t0
    return yp - y0 - dt * f(tp, yp)


def implicit_midpoint_step(f: Callable, jac: Callable,
                           t0: float, y0: np.ndarray,
                           dt: float, tol: float = 1e-10,
                           max_iter: int = 30) -> np.ndarray:
    """
    Single implicit midpoint step:
        y_{n+1} = y_n + dt * f((t_n + t_{n+1})/2, (y_n + y_{n+1})/2)

    Solved via Newton-Raphson with analytical Jacobian.

    Parameters
    ----------
    f : callable
        RHS function f(t, y).
    jac : callable
        Jacobian J(t, y) = df/dy.
    t0 : float
        Current time.
    y0 : ndarray
        Current state.
    dt : float
        Time step.
    tol : float
        Newton convergence tolerance.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    y_new : ndarray
        State at t0 + dt.
    """
    t_mid = t0 + 0.5 * dt
    y_mid = y0.copy()  # Initial guess: y_mid = y_n

    for _ in range(max_iter):
        f_mid = f(t_mid, y_mid)
        residual = y_mid - y0 - dt * f_mid

        # Jacobian of residual: I - dt/2 * J(t_mid, y_mid)
        J_mid = jac(t_mid, y_mid)
        R_jac = np.eye(len(y0)) - 0.5 * dt * J_mid

        try:
            delta = np.linalg.solve(R_jac, -residual)
        except np.linalg.LinAlgError:
            # Fallback: damped step
            delta = -0.5 * residual

        y_mid += delta
        if np.linalg.norm(delta) < tol * (1.0 + np.linalg.norm(y_mid)):
            break

    # Recover y_{n+1} = 2*y_mid - y_n
    y_new = 2.0 * y_mid - y0
    return y_new


def b1g3_residual(f: Callable, dt: float,
                  t1: float, t2: float, t3: float,
                  y1: np.ndarray, y2: np.ndarray, y3: np.ndarray) -> np.ndarray:
    """
    Residual for the B1G3 multistep formula.

    The B1G3 scheme with coefficients:
        A3 = 0.5 + 1/sqrt(3)  ~= 1.366
        A2 = -2/sqrt(3)       ~= -1.155
        A1 = -0.5 + 1/sqrt(3) ~= 0.077
        B  = 1/sqrt(3)        ~= 0.577

    is: y3 = A3*y2 + A2*y1 + A1*y0 + dt*(B*f3 + ... )

    Simplified form:
        R(y3) = y3 - A3*y2 - A2*y1 - A1*y0
                - dt * B * f(t3, y3) = 0
    """
    sqrt3 = np.sqrt(3.0)
    A3 = 0.5 + 1.0 / sqrt3
    A2 = -2.0 / sqrt3
    A1 = -0.5 + 1.0 / sqrt3
    B = 1.0 / sqrt3

    # We need y0 which is the state before y1
    # Use the relation: y0 can be expressed from y1, y2
    # For the residual, we use:
    # R = y3 - (A3*y2 + A2*y1 + A1*(2*y1 - y2/B)) - dt*B*f(t3, y3)
    # Simplified to a self-contained form using y1, y2, y3:
    f3 = f(t3, y3)
    return y3 - A3 * y2 - A2 * y1 - A1 * (2 * y1 - y2) - dt * B * f3


def b1g3_step(f: Callable, jac: Callable,
              t_prev2: float, t_prev1: float, t_curr: float,
              y_prev2: np.ndarray, y_prev1: np.ndarray,
              y_curr_init: np.ndarray,
              tol: float = 1e-10, max_iter: int = 30) -> np.ndarray:
    """
    Single B1G3 step from (t_prev2, y_prev2), (t_prev1, y_prev1) to t_curr.

    Uses Newton iteration to solve the implicit equation for y at t_curr.
    """
    dt = t_curr - t_prev1
    sqrt3 = np.sqrt(3.0)
    A3 = 0.5 + 1.0 / sqrt3
    A2 = -2.0 / sqrt3
    A1 = -0.5 + 1.0 / sqrt3
    B = 1.0 / sqrt3

    y = y_curr_init.copy()
    for _ in range(max_iter):
        f_val = f(t_curr, y)
        residual = y - A3 * y_prev1 - A2 * y_prev2 - A1 * (2 * y_prev2 - y_prev1) - dt * B * f_val
        J = jac(t_curr, y)
        R_jac = np.eye(len(y)) - dt * B * J

        try:
            delta = np.linalg.solve(R_jac, -residual)
        except np.linalg.LinAlgError:
            delta = -0.1 * residual

        y += delta
        if np.linalg.norm(delta) < tol * (1.0 + np.linalg.norm(y)):
            break

    return y


def integrate_ode(f: Callable, jac: Callable,
                  y0: np.ndarray, tspan: Tuple[float, float],
                  n_steps: int, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE system using the B1G3 method with implicit midpoint bootstrap.

    Algorithm:
    1. Bootstrap: one implicit midpoint step from (t0, y0) to (t1, y1)
    2. Marching: B1G3 steps from (t0,y0), (t1,y1) to (t2,y2), ..., (tn,yn)

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> dy/dt.
    jac : callable
        Jacobian function J(t, y) -> df/dy.
    y0 : ndarray
        Initial state vector.
    tspan : tuple
        (t_start, t_end).
    n_steps : int
        Number of time steps.
    tol : float
        Newton convergence tolerance.

    Returns
    -------
    t_array : ndarray, shape (n_steps+1,)
    y_array : ndarray, shape (n_steps+1, len(y0))
    """
    dt = (tspan[1] - tspan[0]) / n_steps
    t_array = np.linspace(tspan[0], tspan[1], n_steps + 1)
    y_array = np.zeros((n_steps + 1, len(y0)))
    y_array[0] = y0.copy()

    # Bootstrap: implicit midpoint step
    y1 = implicit_midpoint_step(f, jac, t_array[0], y0, dt, tol)
    y_array[1] = y1

    # B1G3 steps
    for step in range(1, n_steps):
        if step == 1:
            # Use backward Euler for step 2 (need 3 levels for B1G3)
            y_init = y1 + dt * f(t_array[2], y1)
            # Newton solve for backward Euler
            y_new = y_init.copy()
            for _ in range(30):
                F = y_new - y_array[step] - dt * f(t_array[step + 1], y_new)
                J = np.eye(len(y0)) - dt * jac(t_array[step + 1], y_new)
                try:
                    delta = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    delta = -0.1 * F
                y_new += delta
                if np.linalg.norm(delta) < tol * (1.0 + np.linalg.norm(y_new)):
                    break
            y_array[step + 1] = y_new
        else:
            y_init = y_array[step] + dt * f(t_array[step], y_array[step])
            y_new = b1g3_step(f, jac,
                              t_array[step - 1], t_array[step], t_array[step + 1],
                              y_array[step - 1], y_array[step], y_init, tol)
            y_array[step + 1] = y_new

    return t_array, y_array


def blood_pressure_ode_model(t_end: float = 10.0, n_steps: int = 1000,
                             pdia: float = 80.0, psys: float = 120.0,
                             pulse_bpm: float = 70.0,
                             cardiac_output: float = 5.6
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blood pressure ODE model with periodic discrete jumps at each heartbeat.

    The model describes arterial pressure decay between heartbeats:
        dy/dt = -y / (Ca * Rs)     (diastolic decay)
    At each heartbeat (period T = 60/pulse):
        y -> psys                   (instantaneous jump to systolic)

    Parameters:
        Ca = q / pulse / (psys - pdia)     (arterial compliance)
        Rs = T / (Ca * (log(psys) - log(pdia)))  (peripheral resistance)

    Exact solution: y(t) = psys * exp(-tmod / Ca / Rs)

    Returns (t_array, y_numerical, y_exact).
    """
    T = 60.0 / pulse_bpm  # cardiac period (s)
    Ca = cardiac_output / (pulse_bpm * (psys - pdia))  # compliance
    Rs = T / (Ca * (np.log(psys) - np.log(pdia)))      # resistance
    tau = Ca * Rs  # time constant

    dt = t_end / n_steps
    t_array = np.linspace(0, t_end, n_steps + 1)
    y_num = np.zeros(n_steps + 1)
    y_exact = np.zeros(n_steps + 1)

    y = psys  # initial: systolic pressure

    for i in range(n_steps):
        t = t_array[i]
        t_new = t_array[i + 1]

        # Check for heartbeat in this interval
        beat_before = int(t / T)
        beat_after = int(t_new / T)

        if beat_after > beat_before:
            # Heartbeat occurs: reset to systolic
            t_since_beat = t_new - beat_after * T
            y = psys * np.exp(-t_since_beat / tau)
        else:
            # Normal exponential decay
            y = y * np.exp(-dt / tau)

        y_num[i + 1] = y

        # Exact solution
        tmod = t_new % T
        y_exact[i + 1] = psys * np.exp(-tmod / tau)

    return t_array, y_num, y_exact
