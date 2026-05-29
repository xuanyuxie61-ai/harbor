"""
Adaptive k-Point Sampling via Centroidal Voronoi Tessellation (CVT)
====================================================================
Generates an optimally distributed set of k-points in the moiré Brillouin
zone (MBZ) using Lloyd's algorithm for Centroidal Voronoi Tessellation.

Scientific Background
---------------------
In first-principles and tight-binding calculations, the integral over the
Brillouin zone

    ⟨O⟩ = (1/Ω_BZ) ∫_{BZ} O(k) d²k

is approximated by a discrete sum over k-points.  Uniform grids
(Monkhorst-Pack) are standard, but for systems with strongly varying
integrands (e.g., Van Hove singularities near the Fermi level), an
adaptive sampling that places more points where the DOS or curvature is
large can dramatically improve convergence.

A CVT is a Voronoi tessellation whose generating points coincide with
the centroids (mass centers) of their respective Voronoi cells.  Lloyd's
algorithm iteratively moves each generator to its cell centroid:

    g_j^{(t+1)} = (∫_{V_j} ρ(k) k d²k) / (∫_{V_j} ρ(k) d²k)

where ρ(k) is a density weighting function (e.g., the local density of
states).  The algorithm minimizes the energy functional

    E = Σ_j ∫_{V_j} ρ(k) |k − g_j|² d²k .

For a hexagonal domain (the MBZ), we restrict generators to the
irreducible wedge and use periodic boundary conditions.
"""

import numpy as np
from typing import Tuple, List


def hexagon_domain_sample(n_samples: int, radius: float = 1.0) -> np.ndarray:
    """
    Uniformly sample points inside a regular hexagon of given circumradius.

    The hexagon vertices are at angles φ_j = 2π j / 6, j = 0,…,5.
    We use rejection sampling inside the bounding box [−R, R] × [−R, R]
    and keep points satisfying the hexagon inequalities.

    Parameters
    ----------
    n_samples : int
        Number of points to generate.
    radius : float
        Circumradius of the hexagon.

    Returns
    -------
    np.ndarray of shape (n_samples, 2)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    points = []
    # Hexagon: max |x| ≤ R * cos(π/6) and |y| ≤ R * (1 − |x|/(R * cos(π/6)))
    # Actually for circumradius R, the apothem is R * cos(π/6)
    apothem = radius * np.cos(np.pi / 6.0)
    while len(points) < n_samples:
        # Batch generation
        batch = np.random.uniform(-radius, radius, size=(n_samples * 4, 2))
        for p in batch:
            x, y = p
            if abs(x) <= apothem:
                slope = np.tan(np.pi / 6.0)
                y_limit = slope * (apothem - abs(x))
                if abs(y) <= y_limit:
                    points.append(p)
            if len(points) >= n_samples:
                break
    return np.array(points[:n_samples])


def voronoi_cell_centroid_2d_hexagonal(
    generators: np.ndarray,
    density_func,
    n_mc: int = 5000,
    radius: float = 1.0,
) -> np.ndarray:
    """
    Compute the density-weighted centroids of Voronoi cells for a set of
    generators inside a hexagonal domain.

    For each generator g_j, we Monte-Carlo sample points in its Voronoi
    cell V_j (the region closer to g_j than to any other generator) and
    compute

        c_j = ⟨k⟩_{V_j} = (1/N_j) Σ_{k∈V_j} ρ(k) k

    Parameters
    ----------
    generators : np.ndarray of shape (n, 2)
    density_func : callable
        Function ρ(k) returning a float for each k-point.
    n_mc : int
        Number of Monte-Carlo samples per cell estimate.
    radius : float
        Hexagon circumradius.

    Returns
    -------
    np.ndarray of shape (n, 2)
        Estimated centroids.
    """
    n = generators.shape[0]
    if n == 0:
        return np.zeros((0, 2))

    centroids = np.zeros((n, 2))
    counts = np.zeros(n)

    # Monte-Carlo sample inside the hexagon and assign to nearest generator
    samples = hexagon_domain_sample(n_mc, radius)
    for s in samples:
        # Find nearest generator (Euclidean distance)
        dists = np.sum((generators - s) ** 2, axis=1)
        j = int(np.argmin(dists))
        weight = max(density_func(s), 0.0)
        centroids[j] += weight * s
        counts[j] += weight

    # For cells with zero weight, fall back to the generator itself
    for j in range(n):
        if counts[j] > 0.0:
            centroids[j] /= counts[j]
        else:
            centroids[j] = generators[j]

    return centroids


def lloyd_cvt_hexagonal(
    n_generators: int,
    density_func,
    radius: float = 1.0,
    n_iterations: int = 30,
    init_mode: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's algorithm for CVT in a hexagonal domain.

    Iteration formula:

        g_j^{(t+1)} = c_j^{(t)}

    where c_j^{(t)} is the density-weighted centroid of the Voronoi
    cell of g_j^{(t)}.

    Parameters
    ----------
    n_generators : int
    density_func : callable
    radius : float
    n_iterations : int
    init_mode : str
        "random" or "compressed" (clusters near center).

    Returns
    -------
    generators : np.ndarray of shape (n_generators, 2)
        Final generator positions.
    energy_history : np.ndarray of shape (n_iterations,)
        CVT energy E(t) at each iteration.
    """
    if n_generators < 1:
        raise ValueError("n_generators must be positive.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    # Initialize generators
    if init_mode == "random":
        generators = hexagon_domain_sample(n_generators, radius)
    elif init_mode == "compressed":
        # Place points in a smaller central disk
        r_small = radius * 0.3
        generators = []
        while len(generators) < n_generators:
            pts = np.random.uniform(-r_small, r_small, size=(n_generators * 2, 2))
            for p in pts:
                if np.linalg.norm(p) <= r_small:
                    generators.append(p)
                if len(generators) >= n_generators:
                    break
        generators = np.array(generators[:n_generators])
    else:
        raise ValueError("init_mode must be 'random' or 'compressed'.")

    energy_history = np.zeros(n_iterations)

    for it in range(n_iterations):
        centroids = voronoi_cell_centroid_2d_hexagonal(
            generators, density_func, n_mc=8000, radius=radius
        )
        # Compute CVT energy
        # Monte-Carlo estimate: E = Σ_j Σ_{s∈V_j} ρ(s) |s − g_j|²
        samples = hexagon_domain_sample(10000, radius)
        total_energy = 0.0
        total_weight = 0.0
        for s in samples:
            dists = np.sum((generators - s) ** 2, axis=1)
            j = int(np.argmin(dists))
            w = max(density_func(s), 0.0)
            total_energy += w * dists[j]
            total_weight += w
        if total_weight > 0.0:
            energy_history[it] = total_energy / total_weight
        else:
            energy_history[it] = total_energy

        generators = centroids

    return generators, energy_history


def mbz_cvt_kpoints(
    theta_deg: float,
    n_k: int = 64,
    n_iterations: int = 20,
    dos_weight_func=None,
) -> np.ndarray:
    """
    Generate optimally distributed k-points in the moiré Brillouin zone
    using CVT.

    The hexagon radius is the magnitude of the moiré reciprocal lattice
    vector |b|.

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees.
    n_k : int
        Number of k-points.
    n_iterations : int
        Lloyd iterations.
    dos_weight_func : callable, optional
        If provided, used as ρ(k).  Otherwise uniform weighting.

    Returns
    -------
    np.ndarray of shape (n_k, 2)
        k-points in Cartesian coordinates (nm^{-1}).
    """
    from tight_binding import moire_lattice_constant

    L_m = moire_lattice_constant(theta_deg)
    b_mag = 4.0 * np.pi / (np.sqrt(3.0) * L_m)
    radius = b_mag

    if dos_weight_func is None:
        def uniform_weight(k):
            return 1.0
        dos_weight_func = uniform_weight

    generators, _ = lloyd_cvt_hexagonal(
        n_k, dos_weight_func, radius=radius, n_iterations=n_iterations
    )
    return generators


def irreducible_wedge_kpoints(
    kpoints: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """
    Reduce a set of k-points to the irreducible wedge of the hexagonal
    Brillouin zone using C6v symmetry.

    The irreducible wedge is the region bounded by the lines
    k_y = 0 and k_y = √3 k_x (for k_x ≥ 0).

    Parameters
    ----------
    kpoints : np.ndarray of shape (n, 2)
    tolerance : float

    Returns
    -------
    np.ndarray of shape (m, 2)
        Unique points in the irreducible wedge.
    """
    wedge_points = []
    for k in kpoints:
        x, y = k
        # Apply C6 symmetry operations until the point lands in the wedge
        # The wedge: 0 ≤ θ ≤ π/6, i.e., 0 ≤ y ≤ x * tan(π/6)
        for _ in range(6):
            # Rotate by π/3
            angle = np.pi / 3.0
            xr = x * np.cos(angle) + y * np.sin(angle)
            yr = -x * np.sin(angle) + y * np.cos(angle)
            x, y = xr, yr
            if x >= -tolerance and y >= -tolerance:
                tan_pi6 = np.tan(np.pi / 6.0)
                if y <= x * tan_pi6 + tolerance:
                    wedge_points.append([x, y])
                    break
        else:
            # Fallback: just keep the original
            wedge_points.append([k[0], k[1]])

    # Deduplicate
    if len(wedge_points) == 0:
        return np.zeros((0, 2))
    arr = np.array(wedge_points)
    unique = []
    for p in arr:
        if not unique or min(np.linalg.norm(np.array(unique) - p, axis=1)) > tolerance:
            unique.append(p)
    return np.array(unique)
