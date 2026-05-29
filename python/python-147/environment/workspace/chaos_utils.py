"""
chaos_utils.py
==============
Chaotic dynamics utilities for generating initial conditions, testing
nonlinear stability, and enriching the PINN training data.

This module integrates three seed projects:
  1. squircle_ode (1152): Generalized trigonometric functions for periodic
     boundary parameterization.
  2. cross_chaos (227): Iterated Function System (IFS) for fractal attractor
     point generation.
  3. cellular_automaton (148): Rule-30 cellular automaton for binary pattern
     generation used as spatial mask functions.

The KS equation exhibits spatiotemporal chaos; these tools help generate
rich, non-trivial initial conditions and test the PINN's ability to capture
chaotic transients.
"""

import numpy as np


def squircle_trajectory(s=4.0, t0=0.0, y0=None, tstop=20.0, n_points=1000):
    """
    Solve the squircle ODE:
        du/dt =  v^{s-1}
        dv/dt = -u^{s-1}

    This generalization of harmonic motion traces a super-elliptic curve.
    For s=2, it reduces to standard circular motion (sine/cosine).
    For s=4, the trajectory is a "squircle".

    We integrate using RK4 and return the trajectory.

    Parameters
    ----------
    s : float
        Exponent (s > 1).
    t0 : float
        Initial time.
    y0 : array-like, optional
        Initial state [u0, v0]. Defaults to [0, 1].
    tstop : float
        Final time.
    n_points : int
        Number of output points.

    Returns
    -------
    t : ndarray
        Time points.
    y : ndarray, shape (n_points, 2)
        State trajectory [u(t), v(t)].
    """
    if s <= 1.0:
        raise ValueError("s must be > 1")
    if y0 is None:
        y0 = np.array([0.0, 1.0])
    else:
        y0 = np.array(y0, dtype=float)
    if y0.shape != (2,):
        raise ValueError("y0 must have shape (2,)")

    def rhs(t, y):
        u, v = y
        dudt = np.sign(v) * (np.abs(v) ** (s - 1.0))
        dvdt = -np.sign(u) * (np.abs(u) ** (s - 1.0))
        return np.array([dudt, dvdt])

    t = np.linspace(t0, tstop, n_points)
    dt = t[1] - t[0]
    y = np.zeros((n_points, 2))
    y[0] = y0

    for i in range(n_points - 1):
        k1 = rhs(t[i], y[i])
        k2 = rhs(t[i] + dt / 2, y[i] + dt * k1 / 2)
        k3 = rhs(t[i] + dt / 2, y[i] + dt * k2 / 2)
        k4 = rhs(t[i] + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def squircle_activation_basis(x, s=4.0, n_modes=8):
    """
    Generate a basis of squircle-inspired periodic functions for x in [0, L].

    Uses the squircle trajectory to define a parametric periodic basis:
        \psi_k(x) = u_k( x * tstop / L )

    where (u_k, v_k) are time-shifted squircle trajectories.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    tstop = 2.0 * np.pi
    t_base, y_base = squircle_trajectory(s=s, tstop=tstop, n_points=2048)
    u_base = y_base[:, 0]

    L = x.max() - x.min()
    if L <= 0:
        L = 1.0

    basis = np.zeros((len(x), n_modes))
    for k in range(n_modes):
        phase = k * np.pi / n_modes
        # Map x to time index with phase shift
        t_mapped = ((x - x.min()) / L * tstop + phase) % tstop
        indices = (t_mapped / tstop * (len(t_base) - 1)).astype(int)
        indices = np.clip(indices, 0, len(t_base) - 1)
        basis[:, k] = u_base[indices]

    return basis


def cross_chaos_ifs(n_points=5000, seed=42):
    """
    Generate points from the Cross IFS (Iterated Function System).

    The IFS consists of 5 affine transformations:
        x_{k+1} = A x_k + b_j
    where A = (1/3) I and b_j are translations forming a cross pattern.

    The attractor is a fractal cross in the unit square [0,1]^2.

    Adapted from seed project 227_cross_chaos.

    Parameters
    ----------
    n_points : int
        Number of iterations (after a burn-in period).
    seed : int
        Random seed.

    Returns
    -------
    points : ndarray, shape (n_points, 2)
        Points on the fractal cross attractor.
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    rng = np.random.default_rng(seed)

    A = np.array([[1.0 / 3.0, 0.0],
                  [0.0, 1.0 / 3.0]])
    b = np.array([
        [1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
        [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
    ])

    # Random start in unit square
    x = rng.random(2)
    burn_in = min(100, n_points)
    total = burn_in + n_points

    points = np.zeros((n_points, 2))
    count = 0
    for _ in range(total):
        x = A @ x
        j = rng.integers(0, 5)
        x = x + b[:, j]
        if _ >= burn_in:
            points[count] = x
            count += 1

    return points


def cellular_automaton_rule30(cell_num=256, step_num=128, seed_center=True):
    """
    Evolve Rule-30 cellular automaton.

    Rule 30 is defined by:
        111 110 101 100 011 010 001 000
          0   0   0   1   1   1   1   0

    which is binary 00011110 = decimal 30.

    This generates pseudo-random spatial patterns useful for testing the
    PINN on non-smooth initial data.

    Adapted from seed project 148_cellular_automaton.

    Parameters
    ----------
    cell_num : int
        Number of cells.
    step_num : int
        Number of time steps.
    seed_center : bool
        If True, initialize a single 1 in the center; otherwise random.

    Returns
    -------
    states : ndarray, shape (step_num, cell_num)
        Binary state matrix.
    """
    if cell_num < 3:
        raise ValueError("cell_num must be >= 3")
    if step_num < 1:
        raise ValueError("step_num must be >= 1")

    states = np.zeros((step_num, cell_num), dtype=int)
    if seed_center:
        mid = cell_num // 2
        states[0, mid] = 1
    else:
        rng = np.random.default_rng(42)
        states[0] = rng.integers(0, 2, size=cell_num)

    for i in range(1, step_num):
        for j in range(1, cell_num - 1):
            left = states[i - 1, j - 1]
            center = states[i - 1, j]
            right = states[i - 1, j + 1]
            # Rule 30: 00011110
            pattern = (left << 2) | (center << 1) | right
            if pattern in [1, 2, 3, 4]:
                states[i, j] = 1
            else:
                states[i, j] = 0

    return states


def generate_chaotic_initial_condition(L_domain, nx, chaos_type='squircle',
                                       amplitude=1.0):
    """
    Generate a spatial initial condition u(0, x) using chaotic generators.

    Parameters
    ----------
    L_domain : float
        Domain length.
    nx : int
        Number of spatial points.
    chaos_type : str
        'squircle', 'cross_ifs', or 'ca_rule30'.
    amplitude : float
        Scaling amplitude.

    Returns
    -------
    u0 : ndarray, shape (nx,)
        Initial condition.
    x : ndarray, shape (nx,)
        Spatial grid.
    """
    x = np.linspace(0.0, L_domain, nx, endpoint=False)

    if chaos_type == 'squircle':
        basis = squircle_activation_basis(x, s=4.0, n_modes=4)
        coeffs = np.array([1.0, 0.5, -0.3, 0.2])
        u0 = basis @ coeffs
    elif chaos_type == 'cross_ifs':
        points = cross_chaos_ifs(n_points=nx * 10, seed=42)
        # Project 2D IFS points to 1D signal via radial projection
        angles = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)
        hist, bin_edges = np.histogram(angles, bins=nx, range=(-np.pi, np.pi))
        u0 = hist.astype(float)
        u0 = u0 / (np.max(np.abs(u0)) + 1e-12)
    elif chaos_type == 'ca_rule30':
        states = cellular_automaton_rule30(cell_num=nx, step_num=1)
        u0 = states[0].astype(float)
        u0 = u0 * 2.0 - 1.0  # Map {0,1} to {-1,1}
    else:
        raise ValueError(f"Unknown chaos_type: {chaos_type}")

    u0 = u0 * amplitude
    return u0, x
