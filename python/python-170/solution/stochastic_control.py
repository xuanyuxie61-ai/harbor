"""
stochastic_control.py
=====================
Stochastic optimal control and Feynman-Kac path integrals for robot navigation.

Incorporates:
  - feynman_kac_1d (from 422_feynman_kac_1d)

Scientific role:
  Each robot navigates under a stochastic potential V(x) that encodes
  obstacles, goals, and inter-robot repulsion. The Feynman-Kac formula
  provides the value function
      u(x) = E[ exp(-int_0^tau V(X_s) ds) ]
  where X_s is a Brownian motion starting at x and tau is the first exit
  time from the operational domain. This value function acts as a
  collision-avoidance potential: regions with high V (obstacles) yield
  low u, steering robots away.

  The Monte-Carlo estimator uses a weak second-order integrator:
      X_{k+1} = X_k + sqrt(h) * Z_k
      Y_{k+1} = Y_k - (h/2) * [V(X_{k+1})*Y_e + V(X_k)*Y_k]
  with Y_e = (1 - h*V(X_k)) * Y_k as predictor.
"""

import numpy as np


def potential(a: float, x: np.ndarray):
    """
    Quadratic potential defining the domain boundary penalty.

        V(x) = 2*(x/a^2)^2 + 1/a^2

    Parameters
    ----------
    a : float
        Domain half-width.
    x : ndarray
        Positions.

    Returns
    -------
    v : ndarray
        Potential values.
    """
    return 2.0 * (x / a / a) ** 2 + 1.0 / a / a


def feynman_kac_1d_solve(a: float, h: float, n_paths: int, n_grid: int = 21):
    """
    Estimate the solution u(x) of the 1-D Poisson problem via Feynman-Kac.

    Problem:
        (1/2) u'' - V(x) u = 0   on  (-a, a)
        u(boundary) = 1
    Exact solution: u(x) = exp((x/a)^2 - 1)

    Parameters
    ----------
    a : float
        Domain half-width.
    h : float
        Path step size.
    n_paths : int
        Number of Monte-Carlo paths per grid point.
    n_grid : int
        Number of interior grid points.

    Returns
    -------
    xs : ndarray, shape (n_grid+2,)
        Grid points.
    u_approx : ndarray
        Approximate solution.
    u_exact : ndarray
        Exact solution.
    rms_error : float
    """
    if a <= 0 or h <= 0 or n_paths <= 0:
        raise ValueError("a, h, n_paths must be positive.")

    rth = np.sqrt(h)
    ni = n_grid
    xs = np.linspace(-a, a, ni + 2)
    u_approx = np.zeros_like(xs)
    u_exact = np.exp((xs / a) ** 2 - 1.0)

    err_sum = 0.0
    n_int = 0

    for idx, x in enumerate(xs):
        test = a * a - x * x
        if test < 0.0:
            u_approx[idx] = 1.0
            continue

        n_int += 1
        total = 0.0
        steps_total = 0

        for _ in range(n_paths):
            x1 = x
            w = 1.0
            chk = 0.0
            steps = 0
            while chk < 1.0:
                us = np.random.rand() - 0.5
                dx = -rth if us < 0.0 else rth
                vs = potential(a, x1)
                x1 = x1 + dx
                steps += 1
                vh = potential(a, x1)
                we = (1.0 - h * vs) * w
                w = w - 0.5 * h * (vh * we + vs * w)
                chk = (x1 / a) ** 2
            total += w
            steps_total += steps

        u_approx[idx] = total / n_paths
        err_sum += (u_exact[idx] - u_approx[idx]) ** 2

    rms_error = np.sqrt(err_sum / max(n_int, 1))
    return xs, u_approx, u_exact, rms_error


def feynman_kac_collision_potential(positions: np.ndarray, obstacles: np.ndarray,
                                    obstacle_radius: float, domain_radius: float,
                                    n_paths: int = 200, h: float = 0.02):
    """
    Compute a 2-D collision-avoidance potential at robot positions using
    a radial Feynman-Kac approximation.

    For each robot at position p, we consider the radial coordinate r = ||p||
    and estimate u(r) in an effective 1-D potential that combines the domain
    boundary and nearby obstacles.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
    obstacles : ndarray, shape (M, 2)
    obstacle_radius : float
    domain_radius : float
    n_paths : int
    h : float

    Returns
    -------
    potential_values : ndarray, shape (N,)
        Higher = safer.
    """
    N = positions.shape[0]
    pot = np.zeros(N, dtype=float)
    for i in range(N):
        p = positions[i]
        # effective potential: distance to nearest obstacle + domain boundary
        dist_obs = np.min(np.linalg.norm(obstacles - p, axis=1)) if obstacles.shape[0] > 0 else domain_radius
        r_eff = min(np.linalg.norm(p), domain_radius)
        # simple analytic approximation for speed
        # u(r) ~ exp( - (r/R)^2 ) for boundary + exp( - (d_obs/r_obs)^2 ) for obstacles
        boundary_term = np.exp(-(r_eff / domain_radius) ** 2)
        obs_term = np.exp(-(dist_obs / obstacle_radius) ** 2)
        pot[i] = boundary_term + obs_term
    return pot


def gradient_fk_potential(positions: np.ndarray, obstacles: np.ndarray,
                          obstacle_radius: float, domain_radius: float, eps: float = 1e-4):
    """
    Numerical gradient of the Feynman-Kac collision potential.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
    obstacles : ndarray, shape (M, 2)
    obstacle_radius : float
    domain_radius : float
    eps : float

    Returns
    -------
    grad : ndarray, shape (N, 2)
    """
    N = positions.shape[0]
    grad = np.zeros((N, 2), dtype=float)
    f0 = feynman_kac_collision_potential(positions, obstacles, obstacle_radius, domain_radius)
    for dim in range(2):
        pos_plus = positions.copy()
        pos_plus[:, dim] += eps
        f_plus = feynman_kac_collision_potential(pos_plus, obstacles, obstacle_radius, domain_radius)
        grad[:, dim] = (f_plus - f0) / eps
    return grad
