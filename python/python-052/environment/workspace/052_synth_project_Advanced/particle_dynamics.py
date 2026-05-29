"""
Lagrangian Particle Seeding and Dynamics in Spiral Eddies
=========================================================
Derived from seed project 427_fibonacci_spiral (golden-angle spiral
point distribution).

Lagrangian particles are seeded within mesoscale eddies using a
Fibonacci spiral pattern, which provides uniform angular coverage
and avoids clustering at the poles.

Golden angle:  φ = 2π · (1 − 1/Φ) = 2π · (Φ − 1)/Φ
where Φ = (1 + √5)/2 ≈ 1.618 is the golden ratio.

For N particles in a disk of radius R, polar coordinates are:
    r_k = R · √(k / N)
    θ_k = k · φ_golden,   k = 0,…,N−1

Particle advection follows the kinematic equation:
    dx/dt = u(x,t) = −∂ψ/∂y
    dy/dt = v(x,t) =  ∂ψ/∂x

with a 4th-order Runge-Kutta integration.

The spiral seeding ensures isotropic coverage of the eddy interior,
making it ideal for computing Lagrangian coherent structures (LCS)
and finite-time Lyapunov exponents (FTLE).
"""

import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = 2.0 * np.pi * (1.0 - 1.0 / PHI)

def fibonacci_spiral_points(N, R, center=(0.0, 0.0)):
    """
    Generate N points uniformly distributed in a disk using the
    Fibonacci/golden-angle spiral.

    Parameters
    ----------
    N : int
        Number of points.
    R : float
        Disk radius.
    center : tuple
        (x0, y0) center coordinates.

    Returns
    -------
    x, y : ndarray
        Particle coordinates.
    """
    if N < 1:
        return np.array([]), np.array([])
    k = np.arange(N, dtype=np.float64)
    r = R * np.sqrt(k / N)
    theta = k * GOLDEN_ANGLE
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return x, y

def seed_particles_in_eddy(N, x0, y0, sigma, Lx, Ly):
    """
    Seed Lagrangian particles within a Gaussian eddy of radius σ.
    Uses Fibonacci spiral distribution truncated at r = 3σ.
    """
    R = 3.0 * sigma
    x, y = fibonacci_spiral_points(N, R, center=(x0, y0))
    # Periodic wrap
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)
    return x, y

def advect_particles_rk4(x, y, u_field, v_field, x_grid, y_grid, dt):
    """
    Advect particles using 4th-order Runge-Kutta with bilinear
    velocity interpolation.

    Parameters
    ----------
    x, y : ndarray
        Particle positions.
    u_field, v_field : ndarray
        Velocity fields on uniform grid.
    x_grid, y_grid : ndarray
        Grid coordinates.
    dt : float
        Time step.
    """
    def interpolate_velocity(px, py):
        # Bilinear interpolation
        ix = np.floor((px - x_grid[0]) / (x_grid[1] - x_grid[0])).astype(int)
        iy = np.floor((py - y_grid[0]) / (y_grid[1] - y_grid[0])).astype(int)
        ix = np.clip(ix, 0, len(x_grid) - 2)
        iy = np.clip(iy, 0, len(y_grid) - 2)
        dx = px - x_grid[ix]
        dy = py - y_grid[iy]
        hx = x_grid[1] - x_grid[0]
        hy = y_grid[1] - y_grid[0]
        tx = dx / hx
        ty = dy / hy

        u = ((1-tx)*(1-ty) * u_field[iy, ix] +
             tx*(1-ty) * u_field[iy, ix+1] +
             (1-tx)*ty * u_field[iy+1, ix] +
             tx*ty * u_field[iy+1, ix+1])
        v = ((1-tx)*(1-ty) * v_field[iy, ix] +
             tx*(1-ty) * v_field[iy, ix+1] +
             (1-tx)*ty * v_field[iy+1, ix] +
             tx*ty * v_field[iy+1, ix+1])
        return u, v

    # RK4 stages
    k1u, k1v = interpolate_velocity(x, y)
    k2u, k2v = interpolate_velocity(x + 0.5*dt*k1u, y + 0.5*dt*k1v)
    k3u, k3v = interpolate_velocity(x + 0.5*dt*k2u, y + 0.5*dt*k2v)
    k4u, k4v = interpolate_velocity(x + dt*k3u, y + dt*k3v)

    x_new = x + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    y_new = y + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return x_new, y_new


def compute_ftle_field(x_grid, y_grid, u_fields, v_fields, dt, T_integration):
    """
    Compute Finite-Time Lyapunov Exponent (FTLE) field from a sequence
    of velocity snapshots.

    FTLE(x₀) = (1/|T|) · ln √[ λ_max( (∇F_T)^T · (∇F_T) ) ]

    where F_T is the flow map after time T and λ_max is the maximum
    eigenvalue of the Cauchy-Green tensor.
    """
    Nx, Ny = len(x_grid), len(y_grid)
    ftle = np.zeros((Nx, Ny), dtype=np.float64)

    # Seed particles on grid
    X0, Y0 = np.meshgrid(x_grid, y_grid, indexing='ij')
    x_p = X0.ravel().copy()
    y_p = Y0.ravel().copy()

    n_steps = int(T_integration / dt)
    for step in range(n_steps):
        t_idx = min(step, len(u_fields) - 1)
        x_p, y_p = advect_particles_rk4(x_p, y_p, u_fields[t_idx], v_fields[t_idx],
                                        x_grid, y_grid, dt)

    # Reshape to grid
    X_T = x_p.reshape((Nx, Ny))
    Y_T = y_p.reshape((Nx, Ny))

    # Compute spatial gradients of flow map
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    dFdx = np.gradient(X_T, dx, axis=0)
    dFdy = np.gradient(X_T, dy, axis=1)
    dGdx = np.gradient(Y_T, dx, axis=0)
    dGdy = np.gradient(Y_T, dy, axis=1)

    for i in range(Nx):
        for j in range(Ny):
            J = np.array([[dFdx[i,j], dFdy[i,j]],
                          [dGdx[i,j], dGdy[i,j]]])
            CG = J.T @ J
            eigvals = np.linalg.eigvalsh(CG)
            lambda_max = np.max(eigvals)
            if lambda_max > 0 and T_integration > 0:
                ftle[i, j] = np.log(np.sqrt(lambda_max)) / abs(T_integration)
            else:
                ftle[i, j] = 0.0

    return ftle
