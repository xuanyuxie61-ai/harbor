"""
time_integrator.py
==================
Time integration schemes for the resistive MHD equations with
high-order spatial discretization.

Implements:
  - Forward Euler (1st order, for reference)
  - RK2 (Midpoint / Heun's method)
  - RK4 (Classic 4th-order Runge-Kutta)
  - SSP-RK3 (Strong Stability Preserving, 3rd order)
  - Semi-implicit treatment for stiff diffusion terms

Maps seed projects:
  - 127_burgers_time_viscous: explicit time stepping with conservation
    form and multiple BC types
  - 875_poisson_1d: Gauss-Seidel relaxation for implicit solves

Key equations:
  - RK4: k1 = f(y_n), k2 = f(y_n + dt/2 k1), k3 = f(y_n + dt/2 k2),
          k4 = f(y_n + dt k3), y_{n+1} = y_n + dt/6 (k1 + 2k2 + 2k3 + k4)
  - SSP-RK3: u1 = u + dt L(u), u2 = 3/4 u + 1/4 (u1 + dt L(u1)),
              u3 = 1/3 u + 2/3 (u2 + dt L(u2))
  - Semi-implicit Crank-Nicolson for diffusion:
    (I - dt/2 eta nabla^2) B^{n+1} = (I + dt/2 eta nabla^2) B^n + dt * RHS_adv
"""

import numpy as np


# ============================================================
# Explicit Runge-Kutta Schemes
# ============================================================

def forward_euler(rhs_func, state, dt):
    """
    Forward Euler (1st order):
        y_{n+1} = y_n + dt * f(y_n)

    Stable only for very small dt. Included for comparison.
    """
    k1 = rhs_func(state)
    state_new = {}
    for key in state:
        state_new[key] = state[key] + dt * k1[key]
    return state_new


def rk2_midpoint(rhs_func, state, dt):
    """
    RK2 midpoint method (2nd order):
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        y_{n+1} = y_n + dt * k2
    """
    k1 = rhs_func(state)
    state_mid = {}
    for key in state:
        state_mid[key] = state[key] + 0.5 * dt * k1[key]

    k2 = rhs_func(state_mid)
    state_new = {}
    for key in state:
        state_new[key] = state[key] + dt * k2[key]
    return state_new


def rk4_classic(rhs_func, state, dt):
    """
    Classic 4th-order Runge-Kutta:
        k1 = f(y_n)
        k2 = f(y_n + dt/2 k1)
        k3 = f(y_n + dt/2 k2)
        k4 = f(y_n + dt k3)
        y_{n+1} = y_n + dt/6 (k1 + 2 k2 + 2 k3 + k4)

    This is the workhorse integrator for high-order MHD.
    Local truncation error: O(dt^5).
    """
    k1 = rhs_func(state)
    s2 = {key: state[key] + 0.5 * dt * k1[key] for key in state}

    k2 = rhs_func(s2)
    s3 = {key: state[key] + 0.5 * dt * k2[key] for key in state}

    k3 = rhs_func(s3)
    s4 = {key: state[key] + dt * k3[key] for key in state}

    k4 = rhs_func(s4)

    state_new = {}
    for key in state:
        state_new[key] = (state[key]
                          + dt / 6.0 * (k1[key] + 2.0 * k2[key]
                                        + 2.0 * k3[key] + k4[key]))
    return state_new


def ssprk3(rhs_func, state, dt):
    """
    Strong Stability Preserving RK3 (Shu-Osher form):
        u1 = L(dt) u_n
        u2 = 3/4 u_n + 1/4 (L(dt) u1)
        u3 = 1/3 u_n + 2/3 (L(dt) u2)

    where L(dt) u = u + dt * f(u) is the Forward Euler operator.

    SSP property: preserves TVD/positivity under the same CFL
    condition as Forward Euler. Important for shock-capturing
    in reconnection outflows.
    """
    def euler_step(s, h):
        k = rhs_func(s)
        return {key: s[key] + h * k[key] for key in s}

    # Stage 1
    u1 = euler_step(state, dt)

    # Stage 2
    u2_euler = euler_step(u1, dt)
    u2 = {}
    for key in state:
        u2[key] = 0.75 * state[key] + 0.25 * u2_euler[key]

    # Stage 3
    u3_euler = euler_step(u2, dt)
    u3 = {}
    for key in state:
        u3[key] = (1.0 / 3.0 * state[key]
                   + 2.0 / 3.0 * u3_euler[key])

    return u3


# ============================================================
# Semi-Implicit Diffusion Solver
# ============================================================

def crank_nicolson_1d(field_1d, eta, dx, dt):
    """
    1D Crank-Nicolson solve for the diffusion equation:
        du/dt = eta d^2u/dx^2

    (I - dt/2 eta D2) u^{n+1} = (I + dt/2 eta D2) u^n

    where D2 is the tridiagonal 2nd-derivative matrix.
    Uses Thomas algorithm for tridiagonal solve.

    Maps to the Gauss-Seidel solver in poisson_1d (875_poisson_1d)
    but adapted for time-dependent diffusion.
    """
    n = len(field_1d)
    alpha = eta * dt / (2.0 * dx ** 2)

    # Tridiagonal coefficients
    # Lower diagonal: -alpha (indices 1..n-1)
    # Main diagonal: 1 + 2*alpha (indices 0..n-1)
    # Upper diagonal: -alpha (indices 0..n-2)
    lower = -alpha * np.ones(n)
    main = (1.0 + 2.0 * alpha) * np.ones(n)
    upper = -alpha * np.ones(n)

    # RHS: (I + dt/2 eta D2) u^n
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        rhs[i] = (alpha * field_1d[i - 1]
                  + (1.0 - 2.0 * alpha) * field_1d[i]
                  + alpha * field_1d[i + 1])
    rhs[0] = field_1d[0]
    rhs[-1] = field_1d[-1]

    # Thomas algorithm
    return thomas_solve(lower, main, upper, rhs)


def thomas_solve(lower, main, upper, rhs):
    """
    Thomas algorithm for tridiagonal system Ax = d.
    O(n) direct solve for tridiagonal matrices.

    a[i] = lower diagonal (a[0] unused)
    b[i] = main diagonal
    c[i] = upper diagonal (c[n-1] unused)
    d[i] = right-hand side
    """
    n = len(rhs)
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    # Forward sweep
    c_star[0] = upper[0] / main[0] if abs(main[0]) > 1e-30 else 0.0
    d_star[0] = rhs[0] / main[0] if abs(main[0]) > 1e-30 else 0.0

    for i in range(1, n):
        denom = main[i] - lower[i] * c_star[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        if i < n - 1:
            c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i] * d_star[i - 1]) / denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


# ============================================================
# Gauss-Seidel Relaxation (maps to 875_poisson_1d)
# ============================================================

def gauss_seidel_poisson_2d(rhs, grid, tol=1e-6, max_iter=5000):
    """
    2D Gauss-Seidel iterative solver for the Poisson equation:
        nabla^2 phi = rhs

    Used for:
      - Pressure projection (divergence cleaning)
      - Flux function inversion
      - Implicit magnetic field update

    Maps to the 1D Gauss-Seidel solver in poisson_1d (875_poisson_1d),
    extended to 2D.

    Returns:
        phi: solution array
        n_iter: number of iterations
        residual: final L2 residual
    """
    nx, nz = rhs.shape
    phi = np.zeros((nx, nz))
    dx2 = grid.dx ** 2
    dz2 = grid.dz ** 2

    for iteration in range(max_iter):
        phi_old = phi.copy()

        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                phi[i, j] = 0.5 * (dx2 * dz2) / (dx2 + dz2) * (
                    (phi[i + 1, j] + phi[i - 1, j]) / dx2
                    + (phi[i, j + 1] + phi[i, j - 1]) / dz2
                    - rhs[i, j]
                )

        # Boundary conditions: phi = 0 on boundaries
        phi[0, :] = 0.0
        phi[-1, :] = 0.0
        phi[:, 0] = 0.0
        phi[:, -1] = 0.0

        # Check convergence
        diff = np.max(np.abs(phi - phi_old))
        if diff < tol:
            break

    # Compute residual: r = nabla^2 phi - rhs
    residual = np.zeros((nx, nz))
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            residual[i, j] = (
                (phi[i + 1, j] - 2.0 * phi[i, j] + phi[i - 1, j]) / dx2
                + (phi[i, j + 1] - 2.0 * phi[i, j] + phi[i, j - 1]) / dz2
                - rhs[i, j]
            )

    n_iter = iteration + 1
    l2_residual = np.sqrt(np.mean(residual ** 2))

    return phi, n_iter, l2_residual


# ============================================================
# Adaptive Time Stepping
# ============================================================

def adaptive_dt(state, grid, plasma, cfl_target=0.4, eta=None,
                safety_factor=1.1):
    """
    Compute the adaptive time step based on the CFL condition.

    dt = CFL * min(dx, dz) / max(|v| + v_f)

    If resistivity is included, also check the diffusion CFL:
    dt_eta = dx^2 / (2 * dim * eta)

    Returns the minimum of all constraints.
    """
    from stability_analysis import cfl_condition, diffusion_cfl

    vx = state.get('vx', np.zeros_like(grid.X))
    vz = state.get('vz', np.zeros_like(grid.X))

    vx_max = np.max(np.abs(vx))
    vz_max = np.max(np.abs(vz))

    V_A = plasma.V_A
    cs = plasma.c_s

    dt_cfl, v_f, v_max = cfl_condition(
        vx_max, vz_max, V_A, cs, grid.dx, grid.dz, cfl_target)

    dt_min = dt_cfl

    if eta is not None and eta > 0:
        dt_eta = diffusion_cfl(eta, grid.dx, grid.dz)
        dt_min = min(dt_min, dt_eta)

    return dt_min, v_max, v_f


# ============================================================
# Time Integration Driver
# ============================================================

def time_integrate(rhs_func, state0, grid, plasma, t_max,
                   scheme='rk4', cfl=0.4, eta=None,
                   output_interval=10, verbose=True):
    """
    Main time integration loop.

    Input:
        rhs_func: callable(state) -> dict of time derivatives
        state0: initial state dictionary
        grid: ReconnectionGrid
        plasma: SolarCoronaPlasma
        t_max: maximum simulation time
        scheme: 'euler', 'rk2', 'rk4', 'ssprk3'
        cfl: CFL number for adaptive time stepping
        eta: resistivity (for diffusion CFL check)
        output_interval: output every N steps
        verbose: print progress

    Output:
        history: list of (time, state) snapshots
        info: dict with timing and iteration info
    """
    if scheme == 'euler':
        step_func = forward_euler
    elif scheme == 'rk2':
        step_func = rk2_midpoint
    elif scheme == 'rk4':
        step_func = rk4_classic
    elif scheme == 'ssprk3':
        step_func = ssprk3
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    state = {k: v.copy() for k, v in state0.items()}
    t = 0.0
    step = 0
    history = [(t, {k: v.copy() for k, v in state.items()})]

    while t < t_max:
        dt, v_max, v_f = adaptive_dt(state, grid, plasma, cfl, eta)
        # Don't overshoot t_max
        dt = min(dt, t_max - t)
        if dt < 1e-15:
            break

        state = step_func(rhs_func, state, dt)
        t += dt
        step += 1

        if verbose and step % output_interval == 0:
            print(f"  Step {step:6d} | t = {t:.6f} | dt = {dt:.4e} | "
                  f"v_max = {v_max:.4e}")

        if step % output_interval == 0:
            history.append((t, {k: v.copy() for k, v in state.items()}))

    info = {
        'n_steps': step,
        't_final': t,
        'v_max_final': v_max,
    }
    return history, info
