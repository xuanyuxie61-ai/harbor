"""
Reconnection Automaton Module
=============================
Based on seed project 671_life:
- life_update.m  →  cellular automaton update rules

Physics:
--------
Magnetic reconnection in GRB jets is a primary mechanism for
dissipating magnetic energy and accelerating particles.
The sites of reconnection form a complex, dynamic pattern that
can be modelled as a cellular automaton on a 2D lattice
representing the jet cross-section.

Each cell represents a flux tube with binary state:
    0 = quiescent (closed field lines)
    1 = active (reconnecting/current-sheet region)

The update rules encode the physical condition for reconnection:
- A quiescent cell becomes active if exactly 3 neighbors are active
  (triple-point reconnection instability).
- An active cell remains active only if 2 or 3 neighbors are active
  (stable current sheet); otherwise it quenches.

This is isomorphic to Conway's Game of Life, but with the
interpretation that "life" represents sustained magnetic
dissipation.  The total dissipated power is:

    P_rec(t) = N_active(t) · (B² / 8π) · V_cell / τ_rec

where τ_rec ≈ L/cs is the reconnection timescale (L = cell size,
cs = sound speed).

The reconnection electric field is:

    E_rec = (v_in / c) B_in ≈ 0.1 B_in

for fast reconnection with inflow velocity v_in ≈ 0.1 c.
"""

import numpy as np


def life_update(m, n, grid):
    """
    Update a Life grid with m×n interior cells, surrounded by a
    boundary layer of zeros.

    Parameters
    ----------
    m, n : int
        Interior grid dimensions.
    grid : ndarray, shape (m+2, n+2)
        Grid with zero boundary padding.

    Returns
    -------
    grid : ndarray
        Updated grid.
    """
    s = np.zeros((m, n), dtype=int)

    for j in range(n):
        for i in range(m):
            s[i, j] = (grid[i, j] + grid[i, j + 1] + grid[i, j + 2]
                       + grid[i + 1, j] + grid[i + 1, j + 2]
                       + grid[i + 2, j] + grid[i + 2, j + 1] + grid[i + 2, j + 2])

    for j in range(n):
        for i in range(m):
            if grid[i + 1, j + 1] == 0:
                if s[i, j] == 3:
                    grid[i + 1, j + 1] = 1
            elif grid[i + 1, j + 1] == 1:
                if s[i, j] < 2 or s[i, j] > 3:
                    grid[i + 1, j + 1] = 0

    return grid


def initialize_reconnection_sites(m, n, seed_density=0.1, seed_type='random'):
    """
    Initialize the reconnection automaton grid.

    Parameters
    ----------
    m, n : int
        Grid dimensions.
    seed_density : float
        Fraction of initially active cells.
    seed_type : str
        'random' or 'central'.

    Returns
    -------
    grid : ndarray
        Initialized grid with boundary padding.
    """
    grid = np.zeros((m + 2, n + 2), dtype=int)

    if seed_type == 'random':
        interior = np.random.rand(m, n) < seed_density
        grid[1:m + 1, 1:n + 1] = interior.astype(int)
    elif seed_type == 'central':
        cx, cy = m // 2, n // 2
        r = min(m, n) // 4
        for i in range(m):
            for j in range(n):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                    grid[i + 1, j + 1] = 1

    return grid


def evolve_reconnection(m, n, n_steps, seed_density=0.1, B=10.0, cell_size=1e10):
    """
    Evolve the magnetic reconnection automaton and compute
dissipated power.

    Parameters
    ----------
    m, n : int
        Grid dimensions.
    n_steps : int
        Number of time steps.
    seed_density : float
        Initial active fraction.
    B : float
        Magnetic field strength (Gauss).
    cell_size : float
        Physical size of a grid cell (cm).

    Returns
    -------
    history : ndarray, shape (n_steps,)
        Number of active cells per step.
    power : ndarray, shape (n_steps,)
        Dissipated power in erg/s.
    """
    grid = initialize_reconnection_sites(m, n, seed_density)
    history = np.zeros(n_steps, dtype=int)
    power = np.zeros(n_steps, dtype=float)

    c = 2.99792458e10
    tau_rec = cell_size / c
    V_cell = cell_size ** 3
    energy_density = B ** 2 / (8.0 * np.pi)

    for t in range(n_steps):
        grid = life_update(m, n, grid)
        n_active = int(np.sum(grid[1:m + 1, 1:n + 1]))
        history[t] = n_active
        power[t] = n_active * energy_density * V_cell / tau_rec

    return history, power
