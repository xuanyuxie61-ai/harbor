"""
swarm_dynamics.py
=================
Multi-scale swarm robot dynamics: microscopic ODEs + macroscopic PDE coupling.

Incorporates:
  - bdf3 (from 075_bdf3)
  - theta_method (from 1259_theta_method)
  - arneodo_parameters / arneodo_deriv (from 019_arneodo_ode)

Scientific role:
  Each robot is modeled by a hybrid dynamical system coupling mechanical
  states (position, velocity) with an internal chaotic neuronal state
  (Arneodo attractor). The mechanical dynamics are integrated with high-order
  BDF3 for stiffness and Theta-method for the coupled PDE-ODE system.

  Individual robot state:
      z_i = [p_i, v_i, s_i]  in R^9
  where p_i in R^3, v_i in R^3, s_i in R^3 (Arneodo internal state).

  Mechanical equations:
      dp_i/dt = v_i
      dv_i/dt = u_i + F_env(p_i) + F_rep(p_i, {p_j}) - gamma * v_i

  Internal chaos (Arneodo system):
      ds1/dt = s2
      ds2/dt = s3
      ds3/dt = -alpha*s1 - beta*s2 - s3 + delta*s1^3

  The coupling between robots appears through the repulsion force F_rep
  and the consensus term in the control u_i.
"""

import numpy as np
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Arneodo chaotic system
# ---------------------------------------------------------------------------

ARNEODO_DEFAULTS = {
    "alpha": -5.5,
    "beta": 3.5,
    "delta": -1.0,
}


def arneodo_deriv(t: float, xyz: np.ndarray, alpha: float = None, beta: float = None, delta: float = None):
    """
    Right-hand side of the Arneodo chaotic ODE.

        dx/dt = y
        dy/dt = z
        dz/dt = -alpha*x - beta*y - z + delta*x^3

    Parameters
    ----------
    t : float
        Time (unused, system is autonomous).
    xyz : ndarray, shape (3,)
        State vector [x, y, z].
    alpha, beta, delta : float or None
        Parameters; defaults used if None.

    Returns
    -------
    dxyzdt : ndarray, shape (3,)
    """
    if alpha is None:
        alpha = ARNEODO_DEFAULTS["alpha"]
    if beta is None:
        beta = ARNEODO_DEFAULTS["beta"]
    if delta is None:
        delta = ARNEODO_DEFAULTS["delta"]

    x, y, z = xyz
    dxdt = y
    dydt = z
    dzdt = -alpha * x - beta * y - z + delta * x ** 3
    return np.array([dxdt, dydt, dzdt], dtype=float)


# ---------------------------------------------------------------------------
# ODE solvers: BDF3 and Theta method
# ---------------------------------------------------------------------------

def bdf3_residual(f, dt: float, t4: float, y1: np.ndarray, y2: np.ndarray,
                  y3: np.ndarray, y4: np.ndarray):
    """
    BDF3 residual:
        11*y4 - 18*y3 + 9*y2 - 2*y1 - 6*dt*f(t4, y4) = 0
    """
    return 11.0 * y4 - 18.0 * y3 + 9.0 * y2 - 2.0 * y1 - 6.0 * dt * f(t4, y4)


def solve_bdf3(f, tspan: tuple, y0: np.ndarray, n: int):
    """
    Solve ODE y' = f(t,y) using BDF3 with RK3 startup.

    First two steps use explicit RK3:
        k1 = dt*f(t, y)
        k2 = dt*f(t+dt, y+k1)
        k3 = dt*f(t+0.5*dt, y+0.25*k1+0.25*k2)
        y_next = y + (k1 + k2 + 4*k3)/6

    Subsequent steps use BDF3 solved by fsolve with limited iterations.

    Parameters
    ----------
    f : callable
        RHS function f(t, y) -> ndarray.
    tspan : tuple
        (t0, tstop).
    y0 : ndarray
        Initial condition.
    n : int
        Number of steps.

    Returns
    -------
    t : ndarray, shape (n+1,)
    y : ndarray, shape (n+1, m)
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n + 1)
    y = np.zeros((n + 1, m), dtype=float)
    dt = (tspan[1] - tspan[0]) / n
    y[0, :] = y0

    for i in range(n):
        if i < 2:
            # RK3 explicit
            k1 = dt * f(t[i], y[i, :])
            k2 = dt * f(t[i] + dt, y[i, :] + k1)
            k3 = dt * f(t[i] + 0.5 * dt, y[i, :] + 0.25 * k1 + 0.25 * k2)
            y[i + 1, :] = y[i, :] + (k1 + k2 + 4.0 * k3) / 6.0
        else:
            # BDF3 implicit
            y3 = y[i, :]
            y2 = y[i - 1, :]
            y1 = y[i - 2, :]
            t4 = t[i + 1]
            y4_guess = y3 + dt * f(t[i], y3)

            def residual(y4):
                return bdf3_residual(f, dt, t4, y1, y2, y3, y4)

            sol = fsolve(residual, y4_guess, full_output=False, xtol=1e-7, maxfev=100 * m)
            y[i + 1, :] = sol
    return t, y


def theta_residual(f, to: float, yo: np.ndarray, tn: float, yn: np.ndarray, theta: float):
    """
    Theta-method residual:
        yn - yo - dt * [ theta*f(to,yo) + (1-theta)*f(tn,yn) ] = 0
    """
    dt = tn - to
    return yn - yo - dt * (theta * f(to, yo) + (1.0 - theta) * f(tn, yn))


def solve_theta_method(f, tspan: tuple, y0: np.ndarray, n: int, theta: float = 0.5):
    """
    Solve ODE with the Theta method (theta=0.5 gives Crank-Nicolson, 2nd order).

    Parameters
    ----------
    f : callable
    tspan : tuple
    y0 : ndarray
    n : int
    theta : float
        0 <= theta <= 1. theta=1 is backward Euler.

    Returns
    -------
    t : ndarray, shape (n+1,)
    y : ndarray, shape (n+1, m)
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError("theta must be in [0, 1].")
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n + 1)
    y = np.zeros((n + 1, m), dtype=float)
    dt = (tspan[1] - tspan[0]) / n
    y[0, :] = y0

    for i in range(n):
        to = t[i]
        yo = y[i, :]
        tn = to + dt
        yn_guess = yo + dt * f(to, yo)

        def residual(yn):
            return theta_residual(f, to, yo, tn, yn, theta)

        sol = fsolve(residual, yn_guess, full_output=False, xtol=1e-7, maxfev=100 * m)
        y[i + 1, :] = sol
    return t, y


def solve_rk4(f, tspan: tuple, y0: np.ndarray, n: int):
    """
    Explicit Runge-Kutta 4 for non-stiff or moderately stiff systems.

    Parameters
    ----------
    f : callable
    tspan : tuple
    y0 : ndarray
    n : int

    Returns
    -------
    t : ndarray, shape (n+1,)
    y : ndarray, shape (n+1, m)
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.shape[0]
    t = np.linspace(tspan[0], tspan[1], n + 1)
    y = np.zeros((n + 1, m), dtype=float)
    dt = (tspan[1] - tspan[0]) / n
    y[0, :] = y0
    for i in range(n):
        ti = t[i]
        yi = y[i, :]
        k1 = f(ti, yi)
        k2 = f(ti + 0.5 * dt, yi + 0.5 * dt * k1)
        k3 = f(ti + 0.5 * dt, yi + 0.5 * dt * k2)
        k4 = f(ti + dt, yi + dt * k3)
        y[i + 1, :] = yi + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, y


# ---------------------------------------------------------------------------
# Swarm robot dynamics
# ---------------------------------------------------------------------------

class SwarmRobot:
    """
    Single robot state and dynamics.
    """

    def __init__(self, position: np.ndarray, velocity: np.ndarray = None,
                 internal_state: np.ndarray = None):
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float) if velocity is not None else np.zeros_like(self.position)
        if internal_state is None:
            self.internal = np.array([0.2, 0.2, -0.75], dtype=float)
        else:
            self.internal = np.asarray(internal_state, dtype=float)

    @property
    def state(self):
        """Full state vector [p, v, s]."""
        return np.concatenate([self.position, self.velocity, self.internal])

    @state.setter
    def state(self, z: np.ndarray):
        z = np.asarray(z, dtype=float)
        d = self.position.shape[0]
        self.position = z[0:d].copy()
        self.velocity = z[d:2 * d].copy()
        self.internal = z[2 * d:2 * d + 3].copy()


def repulsion_force(pi: np.ndarray, pj: np.ndarray, repulsion_range: float = 0.3,
                    repulsion_strength: float = 1.0):
    """
    Lennard-Jones-like repulsion force between two robots.

        F = strength * (1/r^2 - 1/r^4) * (pi - pj)/r   for r < R
        F = 0                                          for r >= R

    Parameters
    ----------
    pi, pj : ndarray
        Positions.
    repulsion_range : float
    repulsion_strength : float

    Returns
    -------
    force : ndarray
    """
    diff = pi - pj
    r = np.linalg.norm(diff)
    if r < 1e-10 or r >= repulsion_range:
        return np.zeros_like(pi)
    scale = repulsion_strength * (1.0 / (r ** 2 + 1e-4) - 1.0 / (repulsion_range ** 2))
    scale = max(scale, 0.0)
    return scale * diff / r


def swarm_rhs(t: float, z: np.ndarray, robots: list, control_gains: dict,
              env_gradient_func=None, consensus_target: np.ndarray = None):
    """
    Compute time derivative of the full swarm state vector.

    Parameters
    ----------
    t : float
    z : ndarray
        Flattened swarm state.
    robots : list of SwarmRobot
        Template list (used for dimension info).
    control_gains : dict
        Keys: 'gamma', 'kp', 'kv', 'repulsion_range', 'repulsion_strength'.
    env_gradient_func : callable or None
        Gradient of environmental field at a position.
    consensus_target : ndarray or None
        Target scalar for consensus (e.g., average sensor reading).

    Returns
    -------
    dzdt : ndarray
    """
    N = len(robots)
    d = robots[0].position.shape[0]
    dzdt = np.zeros_like(z)

    positions = np.array([z[i * (2 * d + 3): i * (2 * d + 3) + d] for i in range(N)])

    for i in range(N):
        base = i * (2 * d + 3)
        p = z[base: base + d]
        v = z[base + d: base + 2 * d]
        s = z[base + 2 * d: base + 2 * d + 3]

        # TODO HOLE 1: Implement the hybrid robot dynamics control input
        # and state derivatives. The mechanical subsystem requires:
        #   - consensus control: u += kp * (consensus_target - mean(p)) * ones(d) / d
        #   - damping: u -= kv * v
        #   - chemotaxis: u += 0.3 * grad / (|grad| + eps)  from env_gradient_func
        #   - Lennard-Jones repulsion force accumulation over all j != i
        #   - state derivatives: dpdt = v, dvdt = u + F_rep - gamma * v,
        #                        dsdt = arneodo_deriv(t, s)
        # Assign results to dzdt[base:base+d], dzdt[base+d:base+2d],
        # and dzdt[base+2d:base+2d+3] respectively.
        raise NotImplementedError("HOLE 1: swarm_rhs control and dynamics not implemented")

    return dzdt


def integrate_swarm(robots: list, tspan: tuple, n_steps: int,
                    control_gains: dict, env_gradient_func=None,
                    consensus_target: np.ndarray = None, method: str = "rk4"):
    """
    Integrate swarm dynamics.

    Parameters
    ----------
    robots : list of SwarmRobot
    tspan : tuple
    n_steps : int
    control_gains : dict
    env_gradient_func : callable or None
    consensus_target : ndarray or None
    method : str
        "rk4", "bdf3", or "theta".

    Returns
    -------
    t : ndarray
    trajectory : ndarray, shape (n_steps+1, N*(2d+3))
    """
    z0 = np.concatenate([r.state for r in robots])

    def f(t, z):
        return swarm_rhs(t, z, robots, control_gains, env_gradient_func, consensus_target)

    if method == "rk4":
        t, traj = solve_rk4(f, tspan, z0, n_steps)
    elif method == "bdf3":
        t, traj = solve_bdf3(f, tspan, z0, n_steps)
    elif method == "theta":
        t, traj = solve_theta_method(f, tspan, z0, n_steps, theta=0.5)
    else:
        raise ValueError("method must be 'rk4', 'bdf3', or 'theta'.")

    return t, traj
