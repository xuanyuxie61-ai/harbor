"""
potential_surface.py
2D piecewise linear interpolation of molecular potential energy surfaces.

Derived from: 927_pwl_interp_2d

In DNA damage repair, proteins navigate complex energy landscapes defined
by reaction coordinates such as:
  - DNA bending angle (theta)
  - Protein-DNA separation distance (r)
  - Base-pair opening coordinate (x_bp)

This module provides robust barycentric interpolation on triangular
subdivisions of rectangular grids, essential for evaluating free-energy
surfaces at arbitrary conformational coordinates.

Mathematical basis:
  Given a rectangle [x_i, x_{i+1}] x [y_j, y_{j+1}] with values z_{ij},
  split into two triangles:
    Lower triangle: (i,j), (i+1,j), (i,j+1)
    Upper triangle: (i+1,j+1), (i,j+1), (i+1,j)
  Barycentric interpolation inside a triangle with vertices A,B,C:
      P = alpha*A + beta*B + gamma*C,  alpha+beta+gamma=1
      z(P) = alpha*z_A + beta*z_B + gamma*z_C
"""

import numpy as np


def bracket_index(n, x_sorted, xq):
    """
    Find the interval index i such that x_sorted[i] <= xq <= x_sorted[i+1].

    Returns
    -------
    i : int
        Interval index, or -1 if out of bounds.
    """
    if xq < x_sorted[0] or xq > x_sorted[-1]:
        return -1
    # Binary search for robustness with large grids
    lo = 0
    hi = n - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if x_sorted[mid] <= xq <= x_sorted[mid + 1]:
            return mid
        elif xq < x_sorted[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1


def pwl_interp_2d_scalar(xd, yd, Zd, xi, yi):
    """
    Piecewise linear interpolation of scalar data Zd on a 2D grid.

    Parameters
    ----------
    xd : ndarray, shape (nxd,)
        Sorted x-coordinates.
    yd : ndarray, shape (nyd,)
        Sorted y-coordinates.
    Zd : ndarray, shape (nxd, nyd)
        Grid values Zd[i,j] = f(xd[i], yd[j]).
    xi, yi : float
        Interpolation point.

    Returns
    -------
    zi : float
        Interpolated value, or np.inf if out of bounds.
    """
    nxd = len(xd)
    nyd = len(yd)

    i = bracket_index(nxd, xd, xi)
    if i == -1:
        return np.inf
    j = bracket_index(nyd, yd, yi)
    if j == -1:
        return np.inf

    # Diagonal splitting of rectangle
    # Lower-left to upper-right diagonal
    x0, x1 = xd[i], xd[i + 1]
    y0, y1 = yd[j], yd[j + 1]
    z00 = Zd[i, j]
    z10 = Zd[i + 1, j]
    z01 = Zd[i, j + 1]
    z11 = Zd[i + 1, j + 1]

    # Determine which triangle the point lies in
    # Diagonal line: y - y0 = (y1 - y0)/(x1 - x0) * (x - x0)
    diag_y = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)

    if yi <= diag_y:
        # Lower triangle: (x0,y0), (x1,y0), (x0,y1)
        dxa = x1 - x0
        dya = 0.0
        dxb = 0.0
        dyb = y1 - y0
        dxi = xi - x0
        dyi = yi - y0
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return z00
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        zi = alpha * z10 + beta * z01 + gamma * z00
    else:
        # Upper triangle: (x1,y1), (x0,y1), (x1,y0)
        dxa = 0.0
        dya = y0 - y1
        dxb = x0 - x1
        dyb = 0.0
        dxi = xi - x1
        dyi = yi - y1
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return z11
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        zi = alpha * z01 + beta * z10 + gamma * z11

    return zi


def pwl_interp_2d_grid(xd, yd, Zd, xi_grid, yi_grid):
    """
    Vectorized piecewise linear interpolation over a grid of query points.

    Parameters
    ----------
    xd, yd : ndarray
        Grid coordinates.
    Zd : ndarray, shape (nxd, nyd)
    xi_grid, yi_grid : ndarray
        Query coordinates (1D arrays).

    Returns
    -------
    Zi : ndarray, shape (len(xi_grid), len(yi_grid))
    """
    Zi = np.zeros((len(xi_grid), len(yi_grid)))
    for ii, xq in enumerate(xi_grid):
        for jj, yq in enumerate(yi_grid):
            Zi[ii, jj] = pwl_interp_2d_scalar(xd, yd, Zd, xq, yq)
    return Zi


def build_dna_repair_energy_surface(
    theta_range=(-np.pi / 4, np.pi / 4),
    r_range=(0.5, 3.0),
    n_theta=41,
    n_r=41,
    k_bend=2.5,
    epsilon_lj=1.0,
    sigma_lj=0.8,
    damage_depth=-3.0,
    damage_width=0.3,
):
    """
    Construct a model 2D potential energy surface for a repair protein
    approaching a bent DNA segment with a local damage well.

    Coordinates:
      theta: DNA bending angle (rad)
      r: protein-DNA separation distance (nm)

    Energy components:
      U_bend = 0.5 * k_bend * theta^2           (elastic DNA bending)
      U_LJ   = 4*eps * [(sigma/r)^12 - (sigma/r)^6]   (Lennard-Jones)
      U_dam  = damage_depth * exp(-(r-r0)^2/(2*w^2))  (damage recognition well)

    Parameters
    ----------
    theta_range, r_range : tuple
        Ranges for coordinates.
    n_theta, n_r : int
        Grid resolution.
    k_bend : float
        DNA bending force constant (kT/rad^2).
    epsilon_lj, sigma_lj : float
        Lennard-Jones parameters.
    damage_depth : float
        Depth of damage recognition well (negative = attractive).
    damage_width : float
        Width of damage well.

    Returns
    -------
    theta_grid, r_grid, U_surface : ndarray
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_theta)
    r = np.linspace(r_range[0], r_range[1], n_r)

    U = np.zeros((n_theta, n_r))
    r0 = 1.2  # optimal binding distance

    for i, th in enumerate(theta):
        for j, rv in enumerate(r):
            U_bend = 0.5 * k_bend * th ** 2
            if rv < 0.1:
                rv = 0.1  # Avoid singularity
            sr6 = (sigma_lj / rv) ** 6
            U_lj = 4.0 * epsilon_lj * sr6 * (sr6 - 1.0)
            U_dam = damage_depth * np.exp(-((rv - r0) ** 2) / (2.0 * damage_width ** 2))
            U[i, j] = U_bend + U_lj + U_dam

    # Add an explicit conformational activation barrier hill to model
    # the protein conformational change required for damage recognition.
    # The hill is centered at intermediate r and nonzero theta.
    barrier_height = 3.0
    barrier_center_theta = 0.25
    barrier_center_r = 1.8
    barrier_sigma_theta = 0.25
    barrier_sigma_r = 0.35
    for i, th in enumerate(theta):
        for j, rv in enumerate(r):
            hill = barrier_height * np.exp(
                -((th - barrier_center_theta) ** 2) / (2.0 * barrier_sigma_theta ** 2)
                - ((rv - barrier_center_r) ** 2) / (2.0 * barrier_sigma_r ** 2)
            )
            U[i, j] += hill

    return theta, r, U


def compute_activation_barrier(theta, r, U, reactant_region, product_region):
    """
    Compute the activation barrier between reactant and product basins
    on a 2D potential energy surface.

    Method:
      1. Identify reactant minimum E_react and product minimum E_prod.
      2. Construct a linear path between the two minima in grid index space.
      3. Find the maximum energy along this path (saddle approximation).
      4. Barrier = E_saddle - E_react.

    Parameters
    ----------
    theta, r : ndarray
        Coordinate vectors.
    U : ndarray, shape (n_theta, n_r)
        Energy surface.
    reactant_region : tuple
        ((theta_min, theta_max), (r_min, r_max)) for reactant basin.
    product_region : tuple
        Same for product basin.

    Returns
    -------
    barrier : float
        Activation barrier in kT.
    saddle_point : tuple
        (theta_s, r_s) coordinates.
    """
    # Find reactant minimum
    t_min1, t_max1 = reactant_region[0]
    r_min1, r_max1 = reactant_region[1]
    mask1 = (theta >= t_min1) & (theta <= t_max1)
    r_mask1 = (r >= r_min1) & (r <= r_max1)
    if np.any(mask1) and np.any(r_mask1):
        U_react = U[np.ix_(mask1, r_mask1)]
        E_react = np.min(U_react)
        # Find index of reactant minimum
        local_idx = np.unravel_index(np.argmin(U_react), U_react.shape)
        idx_t_react = np.where(mask1)[0][local_idx[0]]
        idx_r_react = np.where(r_mask1)[0][local_idx[1]]
    else:
        E_react = np.min(U)
        idx_t_react = 0
        idx_r_react = 0

    # Find product minimum
    t_min2, t_max2 = product_region[0]
    r_min2, r_max2 = product_region[1]
    mask2 = (theta >= t_min2) & (theta <= t_max2)
    r_mask2 = (r >= r_min2) & (r <= r_max2)
    if np.any(mask2) and np.any(r_mask2):
        U_prod = U[np.ix_(mask2, r_mask2)]
        E_prod = np.min(U_prod)
        local_idx = np.unravel_index(np.argmin(U_prod), U_prod.shape)
        idx_t_prod = np.where(mask2)[0][local_idx[0]]
        idx_r_prod = np.where(r_mask2)[0][local_idx[1]]
    else:
        E_prod = np.min(U)
        idx_t_prod = U.shape[0] - 1
        idx_r_prod = U.shape[1] - 1

    # Linear path in index space between minima
    n_points = max(abs(idx_t_prod - idx_t_react), abs(idx_r_prod - idx_r_react)) + 1
    t_path = np.linspace(idx_t_react, idx_t_prod, n_points)
    r_path = np.linspace(idx_r_react, idx_r_prod, n_points)

    # Sample energy along path using bilinear interpolation
    energies = []
    for tp, rp in zip(t_path, r_path):
        it = int(np.floor(tp))
        ir = int(np.floor(rp))
        it = np.clip(it, 0, U.shape[0] - 2)
        ir = np.clip(ir, 0, U.shape[1] - 2)
        dt = tp - it
        dr = rp - ir
        val = (
            (1 - dt) * (1 - dr) * U[it, ir]
            + dt * (1 - dr) * U[it + 1, ir]
            + (1 - dt) * dr * U[it, ir + 1]
            + dt * dr * U[it + 1, ir + 1]
        )
        energies.append(val)

    energies = np.array(energies)
    E_saddle = np.max(energies)
    saddle_idx = np.argmax(energies)
    saddle_t = theta[int(np.clip(np.round(t_path[saddle_idx]), 0, len(theta) - 1))]
    saddle_r = r[int(np.clip(np.round(r_path[saddle_idx]), 0, len(r) - 1))]
    saddle_point = (float(saddle_t), float(saddle_r))

    barrier = E_saddle - E_react
    # Numerical robustness: barrier cannot be negative
    if barrier < 0:
        barrier = 0.0
    return barrier, saddle_point
