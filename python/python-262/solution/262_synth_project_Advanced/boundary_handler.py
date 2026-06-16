"""
boundary_handler.py
===================
Boundary condition implementations for the resistive MHD reconnection
simulation on a 2D Cartesian domain.

Physical boundary conditions for reconnection:
  - x-direction (outflow): periodic or line-tied (outflow BCs)
  - z-direction (inflow): line-tied (ideal walls) or open boundaries

Line-tied boundary conditions are essential for modeling the solar corona,
where magnetic field lines are anchored in the dense photosphere:
    v = 0 at z-boundaries (no-slip for velocity)
    dB/dt = 0 at z-boundaries (frozen-in for ideal walls)
    dp/dn = 0 at z-boundaries (zero normal pressure gradient)

Maps seed projects:
  - 127_burgers_time_viscous: 5 boundary condition types (Dirichlet,
    Neumann, periodic, mixed)
  - 875_poisson_1d: Dirichlet BCs for Poisson solve

Key equations:
  - Line-tied: v_par = 0, v_perp = 0 at wall; B_n = const
  - Periodic: f(x + Lx) = f(x)
  - Open/outflow: df/dn = 0 at exit
  - Absorbing: f = f_eq + (f - f_eq) * exp(-(x - x_boundary)^2 / sigma^2)
"""

import numpy as np


# ============================================================
# Boundary Condition Types
# ============================================================

BC_PERIODIC = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_LINE_TIED = 3
BC_OPEN = 4
BC_ABSORBING = 5


class BoundaryConfig:
    """Configuration for boundary conditions on each domain edge."""

    def __init__(self, bc_x='periodic', bc_z='line_tied'):
        self.bc_x = bc_x
        self.bc_z = bc_z

    def __repr__(self):
        return f"BoundaryConfig(bc_x={self.bc_x}, bc_z={self.bc_z})"


# ============================================================
# Apply Boundary Conditions
# ============================================================

def apply_bc(field, grid, bc_config, bc_type='field'):
    """
    Apply boundary conditions to a 2D field.

    Input:
        field: 2D array of shape (nx, nz)
        grid: ReconnectionGrid
        bc_config: BoundaryConfig object
        bc_type: 'field' (generic), 'velocity', 'magnetic', 'pressure'

    The bc_type determines the specific BC values:
      - 'velocity': v = 0 at line-tied boundaries
      - 'magnetic': B_normal fixed, B_tangential extrapolated
      - 'pressure': dp/dn = 0 at walls
      - 'field': generic (Dirichlet=0 or periodic)
    """
    f = field.copy()

    # X-direction BCs
    if bc_config.bc_x == 'periodic':
        f = apply_periodic_x(f, grid)
    elif bc_config.bc_x == 'open':
        f = apply_open_x(f, grid)
    elif bc_config.bc_x == 'absorbing':
        f = apply_absorbing_x(f, grid)

    # Z-direction BCs
    if bc_config.bc_z == 'line_tied':
        f = apply_line_tied_z(f, grid, bc_type)
    elif bc_config.bc_z == 'dirichlet':
        f = apply_dirichlet_z(f, grid)
    elif bc_config.bc_z == 'neumann':
        f = apply_neumann_z(f, grid)
    elif bc_config.bc_z == 'periodic':
        f = apply_periodic_z(f, grid)
    elif bc_config.bc_z == 'open':
        f = apply_open_z(f, grid)

    return f


# ============================================================
# Individual BC Implementations
# ============================================================

def apply_periodic_x(f, grid):
    """
    Periodic in x: f(0, z) = f(Lx, z), f(dx, z) = f(Lx+dx, z)
    Implemented via ghost cells (2 on each side for 4th-order FD).
    """
    f_out = f.copy()
    # For centered FD with periodic, just wrap
    # The interior derivative operators handle this via np.roll
    return f_out


def apply_periodic_z(f, grid):
    """Periodic in z."""
    return f.copy()


def apply_line_tied_z(f, grid, bc_type='field'):
    """
    Line-tied boundary conditions in z:
      - For velocity: v = 0 at z-boundaries (field lines frozen in photosphere)
      - For magnetic field: normal B fixed, tangential extrapolated
      - For pressure: zero normal gradient
    """
    f_out = f.copy()

    if bc_type == 'velocity':
        # No-slip: v = 0 at walls, and ghost cells for derivative BCs
        f_out[:, 0] = 0.0
        f_out[:, -1] = 0.0
        if grid.nz > 2:
            f_out[:, 1] = 0.0
            f_out[:, -2] = 0.0
    elif bc_type == 'magnetic':
        # Extrapolate tangential B (zero normal derivative)
        if grid.nz > 2:
            f_out[:, 0] = f_out[:, 1]
            f_out[:, -1] = f_out[:, -2]
    elif bc_type == 'pressure':
        # Zero normal gradient: dp/dz = 0
        if grid.nz > 1:
            f_out[:, 0] = f_out[:, 1]
            f_out[:, -1] = f_out[:, -2]
    else:
        # Generic: zero-gradient extrapolation
        if grid.nz > 1:
            f_out[:, 0] = f_out[:, 1]
            f_out[:, -1] = f_out[:, -2]

    return f_out


def apply_dirichlet_z(f, grid, value=0.0):
    """Dirichlet BC: f = value at z-boundaries."""
    f_out = f.copy()
    f_out[:, 0] = value
    f_out[:, -1] = value
    return f_out


def apply_neumann_z(f, grid):
    """Neumann BC: df/dz = 0 at z-boundaries."""
    f_out = f.copy()
    if grid.nz > 1:
        f_out[:, 0] = f_out[:, 1]
        f_out[:, -1] = f_out[:, -2]
    return f_out


def apply_open_x(f, grid):
    """
    Open/outflow BC in x: zero-gradient (df/dx = 0) at x-boundaries.
    Allows waves to exit without reflection.
    """
    f_out = f.copy()
    if grid.nx > 1:
        f_out[0, :] = f_out[1, :]
        f_out[-1, :] = f_out[-2, :]
    return f_out


def apply_open_z(f, grid):
    """Open BC in z."""
    f_out = f.copy()
    if grid.nz > 1:
        f_out[:, 0] = f_out[:, 1]
        f_out[:, -1] = f_out[:, -2]
    return f_out


def apply_absorbing_x(f, grid, sigma=3.0, f_eq=None):
    """
    Absorbing (sponge) layer BC in x:
        f = f_eq + (f - f_eq) * exp(-((x - x_bnd) / sigma)^2)

    This gradually damps perturbations to the equilibrium near boundaries,
    preventing wave reflections.

    sigma: width of the absorbing layer (in grid cells)
    f_eq: equilibrium value (default: mean of interior)
    """
    f_out = f.copy()
    nx = grid.nx

    if f_eq is None:
        f_eq_val = np.mean(f)
    else:
        f_eq_val = f_eq

    # Left absorbing layer
    for i in range(min(int(3 * sigma), nx)):
        weight = np.exp(-((i) / max(sigma, 0.1)) ** 2)
        f_out[i, :] = f_eq_val + (f_out[i, :] - f_eq_val) * weight

    # Right absorbing layer
    for i in range(min(int(3 * sigma), nx)):
        idx = nx - 1 - i
        weight = np.exp(-((i) / max(sigma, 0.1)) ** 2)
        f_out[idx, :] = f_eq_val + (f_out[idx, :] - f_eq_val) * weight

    return f_out


# ============================================================
# Full State Boundary Application
# ============================================================

def apply_all_bc(state, grid, bc_config):
    """
    Apply appropriate BCs to all fields in the state dictionary.

    State dict has keys: 'Bx', 'Bz', 'vx', 'vz', 'rho', 'p'
    """
    state_bc = {}
    for key, field in state.items():
        if key in ('vx', 'vz'):
            bc_type = 'velocity'
        elif key in ('Bx', 'Bz'):
            bc_type = 'magnetic'
        elif key == 'p':
            bc_type = 'pressure'
        else:
            bc_type = 'field'
        state_bc[key] = apply_bc(field, grid, bc_config, bc_type)

    return state_bc


# ============================================================
# Ghost Cell Management
# ============================================================

def extend_with_ghosts(field, n_ghost=2, bc_x='periodic', bc_z='line_tied'):
    """
    Extend a 2D field with ghost cells for high-order FD stencils.

    For 4th-order FD, we need 2 ghost cells on each side.

    Returns:
        field_ext: extended array of shape (nx + 2*n_ghost, nz + 2*n_ghost)
    """
    nx, nz = field.shape
    ext = np.zeros((nx + 2 * n_ghost, nz + 2 * n_ghost))
    ext[n_ghost:n_ghost + nx, n_ghost:n_ghost + nz] = field

    # X-direction
    if bc_x == 'periodic':
        ext[:n_ghost, n_ghost:n_ghost + nz] = field[-n_ghost:, :]
        ext[n_ghost + nx:, n_ghost:n_ghost + nz] = field[:n_ghost, :]
    else:
        for g in range(n_ghost):
            ext[g, n_ghost:n_ghost + nz] = field[0, :]
            ext[n_ghost + nx + g, n_ghost:n_ghost + nz] = field[-1, :]

    # Z-direction
    if bc_z == 'periodic':
        ext[:, :n_ghost] = ext[:, -2 * n_ghost:-n_ghost]
        ext[:, n_ghost + nz:] = ext[:, n_ghost:2 * n_ghost]
    elif bc_z == 'line_tied':
        for g in range(n_ghost):
            ext[:, g] = ext[:, n_ghost]
            ext[:, n_ghost + nz + g] = ext[:, n_ghost + nz - 1]
    else:
        for g in range(n_ghost):
            ext[:, g] = ext[:, n_ghost]
            ext[:, n_ghost + nz + g] = ext[:, n_ghost + nz - 1]

    return ext
