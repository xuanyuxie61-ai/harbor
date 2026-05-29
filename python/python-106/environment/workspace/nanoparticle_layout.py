"""
nanoparticle_layout.py
======================
Centroidal Voronoi Tessellation (CVT) optimization for nanoparticle array
placement in plasmonic metasurfaces.

Given a set of generator points {g_k} on the periodic unit interval [0,1],
the CVT minimizes the energy functional:

    E({g}) = Σ_{k=1}^{K} ∫_{V_k} ρ(x) |x − g_k|² dx

where V_k is the Voronoi cell of generator g_k and ρ(x) is a density
function.  In the uniform-density case, the optimal 1D CVT is the equally
spaced lattice, but for non-uniform ρ(x) the generators cluster where
ρ(x) is large.

For plasmonic metasurfaces, we want nanoparticles placed at positions
that maximize the local field enhancement.  This can be encoded as a
density function:

    ρ(x) = |E_inc(x)|² / ∫ |E_inc|² dx

and the CVT generators become the optimized nanoparticle positions.

The Lloyd algorithm updates generators to the centroids of their Voronoi
cells.  In 1D with periodic boundary conditions, the update is:

    g_k^{(new)} = (1/W_k) Σ_{x ∈ V_k} x ρ(x) Δx

where W_k = Σ_{x ∈ V_k} ρ(x) Δx.
"""

import numpy as np


def cvtp_1d_optimize(g_num, it_num, s_num, density_func=None, seed=42):
    """
    Optimize generator positions on the periodic unit interval [0,1]
    using the Lloyd algorithm with sampling.

    Parameters
    ----------
    g_num : int
        Number of generators (nanoparticles).
    it_num : int
        Number of Lloyd iterations.
    s_num : int
        Number of sample points per iteration.
    density_func : callable or None
        Function ρ(x) returning density in [0,1].  If None, uniform.
    seed : int

    Returns
    -------
    generators : ndarray, shape (g_num,)
        Optimized positions on [0,1).
    energies : ndarray, shape (it_num,)
        Energy per iteration.
    motions : ndarray, shape (it_num,)
        Average generator motion per iteration.
    """
    if g_num < 1 or it_num < 1 or s_num < 1:
        raise ValueError("g_num, it_num, and s_num must be positive.")

    np.random.seed(seed)
    g = np.sort(np.random.rand(g_num))

    if density_func is None:
        density_func = lambda x: np.ones_like(x)

    energies = np.zeros(it_num)
    motions = np.zeros(it_num)

    for it in range(it_num):
        samples = np.random.rand(s_num)
        rho = density_func(samples)
        if np.any(rho < 0):
            rho = np.maximum(rho, 0.0)

        g_new = np.zeros(g_num)
        w_new = np.zeros(g_num)
        e_new = np.zeros(g_num)

        for i in range(s_num):
            s = samples[i]
            # Find nearest generator with periodic images
            d = np.abs(s - g)
            d = np.minimum(d, np.minimum(np.abs(s + 1.0 - g), np.abs(s - 1.0 - g)))
            k = np.argmin(d)
            # Use periodic image closest to generator
            s_eff = s
            if abs(s + 1.0 - g[k]) < abs(s_eff - g[k]):
                s_eff = s + 1.0
            if abs(s - 1.0 - g[k]) < abs(s_eff - g[k]):
                s_eff = s - 1.0

            g_new[k] += rho[i] * s_eff
            w_new[k] += rho[i]
            e_new[k] += rho[i] * (s_eff - g[k]) ** 2

        # Avoid division by zero
        for k in range(g_num):
            if w_new[k] == 0:
                g_new[k] = g[k]
            else:
                g_new[k] /= w_new[k]

        g_new = np.mod(g_new, 1.0)
        g_new = np.sort(g_new)

        energies[it] = np.sum(e_new) / s_num

        # Compute motion
        t2 = 0.0
        for k in range(g_num):
            t = abs(g_new[k] - g[k])
            t = min(t, abs(g_new[k] - g[k] + 1.0))
            t = min(t, abs(g_new[k] - g[k] - 1.0))
            t2 += t ** 2
        motions[it] = t2 / g_num

        g = g_new.copy()

    return g, energies, motions


def place_nanoparticles_2d_cvt(num_particles, region, it_num=20, s_num=5000,
                                density_func=None, seed=42):
    """
    Place nanoparticles in a 2D rectangular region using independent
    1D CVT along x and y, then combine into a 2D grid with optional
    density-based perturbation.

    Parameters
    ----------
    num_particles : int
        Approximate number of particles.
    region : tuple
        (xmin, xmax, ymin, ymax).
    it_num : int
    s_num : int
    density_func : callable or None
        ρ(x,y) for 2D density.
    seed : int

    Returns
    -------
    positions : ndarray, shape (N, 2)
    """
    nx = int(np.round(np.sqrt(num_particles)))
    ny = nx
    g_x, _, _ = cvtp_1d_optimize(nx, it_num, s_num,
                                  density_func=None, seed=seed)
    g_y, _, _ = cvtp_1d_optimize(ny, it_num, s_num,
                                  density_func=None, seed=seed + 1)

    xmin, xmax, ymin, ymax = region
    xs = xmin + g_x * (xmax - xmin)
    ys = ymin + g_y * (ymax - ymin)

    positions = np.zeros((nx * ny, 2))
    idx = 0
    for i in range(nx):
        for j in range(ny):
            positions[idx, 0] = xs[i]
            positions[idx, 1] = ys[j]
            idx += 1

    if density_func is not None:
        # Acceptance-rejection to thin particles where density is low
        rho_vals = density_func(positions[:, 0], positions[:, 1])
        rho_max = np.max(rho_vals)
        if rho_max > 0:
            keep = np.random.rand(len(rho_vals)) < (rho_vals / rho_max)
            positions = positions[keep]

    return positions
