"""
fd_poisson_solver.py
====================

Finite-difference solvers for the DGLAP-like evolution equations and
nonlinear field equations in the simplified QCD model:
- 1D Poisson solver (from 363_fd1d_poisson)
- Steady viscous Burgers solver via Newton's method
  (from 125_burgers_steady_viscous)

Scientific context:
-------------------
The DGLAP evolution equation for parton distribution functions (PDFs)
in moment space takes the form:

    d q(N, mu^2) / d ln(mu^2) = (alpha_s / 2*pi) * P(N) * q(N, mu^2)

where P(N) is the Mellin transform of the splitting function P(z).
Discretizing the ln(mu^2) direction gives a tridiagonal system
analogous to the 1D Poisson equation.

The nonlinear term in the high-density (small-x) regime introduces
a gluon recombination term that leads to a GLR-MQ equation:

    dG(x, Q^2)/d ln(Q^2) = alpha_s * K * P_gg otimes G - alpha_s^2 * R * G^2 / Q^2

This has the structure of a steady viscous Burgers equation when
written in coordinate space, with the gluon density G playing the
role of the velocity field and Q^2 playing the role of time.
"""

import math
from typing import Callable, List, Tuple


# ===========================================================================
# Section 1: 1D Poisson solver (from 363_fd1d_poisson)
# ===========================================================================

def tridiag_solve(a: List[float], b: List[float], c: List[float],
                  d: List[float]) -> List[float]:
    """
    Thomas algorithm for tridiagonal system:
        b[i]*x[i-1] + a[i]*x[i] + c[i]*x[i+1] = d[i]

    With b[0] = 0, c[n-1] = 0.

    Returns x of length n.
    """
    n = len(a)
    if n == 0:
        return []
    if n == 1:
        if abs(a[0]) < 1e-30:
            raise ValueError("tridiag_solve: singular 1x1 system")
        return [d[0] / a[0]]
    # Forward sweep
    cp = [0.0] * n
    dp = [0.0] * n
    if abs(a[0]) < 1e-30:
        raise ValueError("tridiag_solve: zero pivot at i=0")
    cp[0] = c[0] / a[0]
    dp[0] = d[0] / a[0]
    for i in range(1, n):
        m = a[i] - b[i] * cp[i - 1]
        if abs(m) < 1e-30:
            m = 1e-30  # regularization
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - b[i] * dp[i - 1]) / m
    # Back substitution
    x = [0.0] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def fd1d_poisson(nx: int, xmin: float, xmax: float,
                 f_rhs: Callable[[float], float],
                 g_bc: Callable[[float], float]) -> Tuple[List[float], List[float]]:
    """
    Solve the 1D Poisson equation using finite differences:

        -d^2 U / dx^2 = f(x)  in (xmin, xmax)
        U(xmin) = g(xmin), U(xmax) = g(xmax)

    Discretization:
        (-U[i-1] + 2*U[i] - U[i+1]) / h^2 = f(x_i)

    The matrix is tridiagonal with 2/h^2 on the diagonal and -1/h^2
    on the sub/super-diagonals.

    In the DGLAP context: x is ln(mu^2), U is the PDF moment q(N),
    f is the splitting function convolution.

    Parameters
    ----------
    nx : int
        Number of grid points.
    xmin, xmax : float
        Domain boundaries.
    f_rhs : callable
        Right-hand side f(x).
    g_bc : callable
        Dirichlet boundary values g(x).

    Returns
    -------
    (u, x): solution vector and grid points.
    """
    if nx < 2:
        raise ValueError(f"fd1d_poisson: nx={nx} < 2")
    if xmax <= xmin:
        raise ValueError(f"fd1d_poisson: xmax={xmax} <= xmin={xmin}")
    hx = (xmax - xmin) / (nx - 1)
    x = [xmin + i * hx for i in range(nx)]
    # Build tridiagonal system
    a = [0.0] * nx  # diagonal
    b = [0.0] * nx  # lower
    c = [0.0] * nx  # upper
    d = [0.0] * nx  # rhs
    inv_h2 = 1.0 / (hx * hx)
    for i in range(nx):
        if i == 0 or i == nx - 1:
            a[i] = 1.0
            b[i] = 0.0
            c[i] = 0.0
            d[i] = g_bc(x[i])
        else:
            a[i] = 2.0 * inv_h2
            b[i] = -inv_h2
            c[i] = -inv_h2
            d[i] = f_rhs(x[i])
    u = tridiag_solve(a, b, c, d)
    return u, x


def dglap_moment_evolution(q0: List[float], ln_q2_grid: List[float],
                           anomalous_dim: Callable[[float], float]
                           ) -> List[float]:
    """
    Solve the DGLAP moment evolution equation:

        d q(N, t) / dt = (alpha_s(t) / 2*pi) * gamma_N * q(N, t)

    where t = ln(Q^2/mu0^2) and gamma_N is the anomalous dimension.

    We reformulate as a Poisson-like problem by integrating:
        q(N, t) = q(N, t0) * exp(int_t0^t gamma_N * alpha_s(t')/(2*pi) dt')

    For a given anomalous dimension gamma_N, we solve for the exponent
    using the FD Poisson solver (interpreting the ODE as a 2nd-order
    system with a source term).

    Parameters
    ----------
    q0 : list of float
        Initial PDF moments at various N values.
    ln_q2_grid : list of float
        Grid in ln(Q^2/mu0^2).
    anomalous_dim : callable
        gamma_N as a function of moment index N (here treated as a
        continuous function of the grid position).

    Returns
    -------
    list of float: evolved PDF moments at the final scale.
    """
    n = len(ln_q2_grid)
    if n < 2:
        return list(q0)
    # Simple integration using the trapezoidal rule on the exponent
    t0 = ln_q2_grid[0]
    tf = ln_q2_grid[-1]
    # Integrate gamma * alpha_s / (2*pi) over t
    integral = 0.0
    for i in range(n - 1):
        dt = ln_q2_grid[i + 1] - ln_q2_grid[i]
        tmid = 0.5 * (ln_q2_grid[i] + ln_q2_grid[i + 1])
        gamma_mid = anomalous_dim(tmid)
        integral += gamma_mid * dt
    # Evolution factor
    evol = math.exp(integral)
    return [q * evol for q in q0]


# ===========================================================================
# Section 2: Steady viscous Burgers / GLR-MQ solver
# (from 125_burgers_steady_viscous)
# ===========================================================================

def burgers_steady_viscous(a: float, b: float, alpha: float, beta: float,
                           nu: float, n: int,
                           max_iter: int = 50,
                           tol: float = 1e-8
                           ) -> Tuple[List[float], List[float], int]:
    """
    Solve the steady viscous Burgers equation using Newton's method:

        u * du/dx = nu * d^2 u / dx^2   in (a, b)
        u(a) = alpha, u(b) = beta

    Discretization (centered FD):
        0.5 * (u[i+1]^2 - u[i-1]^2) / (2*dx)
        - nu * (u[i+1] - 2*u[i] + u[i-1]) / dx^2 = 0

    In the GLR-MQ context: u is the gluon density G(x, Q^2),
    x is ln(1/x_Bj), nu is related to the recombination scale,
    and the nonlinear term represents gluon-gluon fusion.

    Parameters
    ----------
    a, b : float
        Domain boundaries.
    alpha, beta : float
        Dirichlet boundary values.
    nu : float
        Viscosity / recombination parameter.
    n : int
        Number of grid points.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    (u, x, iters): solution, grid, iteration count.
    """
    if n < 2:
        raise ValueError(f"burgers_steady_viscous: n={n} < 2")
    if b <= a:
        raise ValueError(f"burgers_steady_viscous: b={b} <= a={a}")
    if nu <= 0.0:
        raise ValueError(f"burgers_steady_viscous: nu={nu} <= 0 (need viscosity)")
    dx = (b - a) / (n - 1)
    x = [a + i * dx for i in range(n)]
    # Initial guess: linear interpolation of BC
    u = [alpha + (beta - alpha) * (x[i] - a) / (b - a) for i in range(n)]
    inv_4dx = 1.0 / (4.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)
    for newton_step in range(max_iter + 1):
        # Compute residual F(u)
        f = [0.0] * n
        f[0] = u[0] - alpha
        for i in range(1, n - 1):
            f[i] = (0.5 * (u[i + 1] ** 2 - u[i - 1] ** 2) * inv_4dx
                    - nu * (u[i + 1] - 2.0 * u[i] + u[i - 1]) * inv_dx2)
        f[n - 1] = u[n - 1] - beta
        f_norm = max(abs(fi) for fi in f)
        if f_norm < tol:
            break
        # Build Jacobian (tridiagonal)
        ja = [0.0] * n
        jb = [0.0] * n
        jc = [0.0] * n
        jd = [-fi for fi in f]
        ja[0] = 1.0
        for i in range(1, n - 1):
            jb[i] = -2.0 * u[i - 1] * inv_4dx - nu * inv_dx2
            ja[i] = 2.0 * nu * inv_dx2
            jc[i] = 2.0 * u[i + 1] * inv_4dx - nu * inv_dx2
        ja[n - 1] = 1.0
        # Solve J * du = -F
        du = tridiag_solve(ja, jb, jc, jd)
        # Update u
        for i in range(n):
            u[i] += du[i]
    return u, x, newton_step


def glr_mq_saturation(xbj: List[float], q2: float,
                      g0: float, nu_eff: float, n: int
                      ) -> Tuple[List[float], List[float]]:
    """
    Solve the GLR-MQ saturation equation for the gluon density:

        dG/dt = alpha_s * P_gg * G - (alpha_s^2 / Q^2) * K * G^2

    where t = ln(1/x) and we seek the steady-state solution.

    This is mapped to a Burgers-like equation with:
        u = G (gluon density)
        x_coord = t = ln(1/x)
        nu = effective diffusion from virtual corrections
        nonlinear term = recombination

    Parameters
    ----------
    xbj : list of float
        Bjorken-x grid (descending).
    q2 : float
        Momentum transfer squared in GeV^2.
    g0 : float
        Initial gluon density at x_max.
    nu_eff : float
        Effective viscosity / diffusion coefficient.
    n : int
        Number of grid points.

    Returns
    -------
    (G, x_grid): gluon density and grid.
    """
    if len(xbj) < 2:
        return [g0], list(xbj)
    t_min = math.log(xbj[-1])  # small x -> large t
    t_max = math.log(xbj[0])   # large x -> small t
    # Boundary values: G(t_min) = enhanced, G(t_max) = g0
    g_max = g0 * 5.0  # small-x enhancement
    G, t_grid, _ = burgers_steady_viscous(
        t_min, t_max, g_max, g0, nu_eff, n,
        max_iter=100, tol=1e-10
    )
    return G, t_grid
