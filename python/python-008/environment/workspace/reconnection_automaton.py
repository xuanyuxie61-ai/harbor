
import numpy as np


def life_update(m, n, grid):
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
