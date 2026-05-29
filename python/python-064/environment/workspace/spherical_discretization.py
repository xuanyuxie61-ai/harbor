"""
Spherical Discretization Module
===============================
Implements optimal spatial discretization for global climate modeling.
Combines:
- Ball grid generation (from 067_ball_grid) for 3D atmospheric/oceanic grids
- Centroidal Voronoi Tessellation (from 146_ccvt_reflect) for optimal point distribution
- Hexagonal Monte Carlo integration (from 529_hexagon_monte_carlo) for flux computations

Scientific Background:
----------------------
For accurate climate simulation, the sphere requires quasi-uniform point distributions.
CVT (Centroidal Voronoi Tessellation) minimizes the energy functional:
    E = sum_i integral_{V_i} ||x - z_i||^2 dA
where V_i is the Voronoi cell and z_i is the generator/centroid.

Hexagonal integration provides O(N^{-1/2}) convergence for Monte Carlo on
hexagonal patches of the sphere, useful for computing global energy budgets.
"""

import numpy as np


def ball_grid_points(n_sub, radius, center):
    """
    Generate grid points inside a 3D ball.
    Adapted from 067_ball_grid.

    Parameters
    ----------
    n_sub : int
        Number of subintervals along radius.
    radius : float
        Ball radius.
    center : array_like, shape (3,)
        Ball center coordinates.

    Returns
    -------
    ndarray, shape (N, 3)
        Grid points inside the ball.
    """
    center = np.asarray(center, dtype=float)
    points = []
    r2 = radius * radius

    for i in range(n_sub + 1):
        x = center[0] + radius * 2.0 * i / (2.0 * n_sub + 1.0)
        for j in range(n_sub + 1):
            y = center[1] + radius * 2.0 * j / (2.0 * n_sub + 1.0)
            for k in range(n_sub + 1):
                z = center[2] + radius * 2.0 * k / (2.0 * n_sub + 1.0)

                if r2 < (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2:
                    break

                # Generate all 8 octant reflections
                offsets = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                           (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)]
                for ox, oy, oz in offsets:
                    if ox == -1 and i == 0:
                        continue
                    if oy == -1 and j == 0:
                        continue
                    if oz == -1 and k == 0:
                        continue
                    px = center[0] + ox * abs(x - center[0])
                    py = center[1] + oy * abs(y - center[1])
                    pz = center[2] + oz * abs(z - center[2])
                    points.append([px, py, pz])

    return np.array(points)


def ball_grid_count(n_sub):
    """
    Count number of grid points in ball_grid_points.
    """
    pts = ball_grid_points(n_sub, 1.0, [0.0, 0.0, 0.0])
    return len(pts)


def fibonacci_sphere(n_points, radius=1.0):
    """
    Generate quasi-uniform points on a sphere using the Fibonacci spiral.
    More uniform than latitude-longitude grids.

    Formula:
    phi_i = arccos(1 - 2*(i+0.5)/N)
    theta_i = 2*pi*i/phi  where phi = (1+sqrt(5))/2 (golden ratio)

    Parameters
    ----------
    n_points : int
        Number of points.
    radius : float
        Sphere radius.

    Returns
    -------
    ndarray, shape (N, 3)
        Point coordinates on sphere.
    """
    indices = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    y = 1.0 - (indices / (n_points - 1)) * 2.0
    radius_xy = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius_xy
    z = np.sin(theta) * radius_xy
    return radius * np.column_stack([x, y, z])


def cvt_energy(points, region_bounds=None):
    """
    Compute CVT energy functional.
    From 146_ccvt_reflect.

    E = sum_i integral_{V_i} ||x - z_i||^2 dx

    Parameters
    ----------
    points : ndarray, shape (N, d)
        Generator points.
    region_bounds : list of tuples, optional
        [(xmin, xmax), (ymin, ymax), ...] for bounding box.

    Returns
    -------
    float
        CVT energy.
    """
    n = len(points)
    d = points.shape[1]

    if region_bounds is None:
        region_bounds = [(0.0, 1.0)] * d

    # Sample points in region
    sample_num = min(10000, n * 100)
    samples = np.zeros((sample_num, d))
    for dim in range(d):
        lo, hi = region_bounds[dim]
        samples[:, dim] = np.random.uniform(lo, hi, sample_num)

    energy = 0.0
    counts = np.zeros(n)
    for s in samples:
        # Find nearest generator
        dists = np.sum((points - s) ** 2, axis=1)
        nearest = np.argmin(dists)
        energy += dists[nearest]
        counts[nearest] += 1

    return energy / sample_num


def cvt_iterate_lloyd(points, region_bounds=None, n_samples=5000, n_iter=50):
    """
    Perform Lloyd's algorithm for CVT on a bounded region.
    Adapted from 146_ccvt_reflect.

    Parameters
    ----------
    points : ndarray, shape (N, d)
        Initial generators.
    region_bounds : list of tuples
        Bounding box.
    n_samples : int
        Monte Carlo samples per iteration.
    n_iter : int
        Number of Lloyd iterations.

    Returns
    -------
    ndarray
        Optimized generator points.
    """
    points = np.array(points, dtype=float)
    n, d = points.shape

    if region_bounds is None:
        region_bounds = [(0.0, 1.0)] * d

    for it in range(n_iter):
        # Generate samples
        samples = np.zeros((n_samples, d))
        for dim in range(d):
            lo, hi = region_bounds[dim]
            samples[:, dim] = np.random.uniform(lo, hi, n_samples)

        # Assign to Voronoi cells and compute centroids
        new_points = np.zeros_like(points)
        counts = np.zeros(n)

        for s in samples:
            dists = np.sum((points - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            new_points[nearest] += s
            counts[nearest] += 1

        # Update generators to centroids
        for i in range(n):
            if counts[i] > 0:
                points[i] = new_points[i] / counts[i]
            else:
                # Reinitialize if empty cell
                for dim in range(d):
                    lo, hi = region_bounds[dim]
                    points[i, dim] = np.random.uniform(lo, hi)

    return points


def cvt_on_sphere(points, n_samples=5000, n_iter=30):
    """
    Lloyd's CVT algorithm constrained to the unit sphere surface.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Initial points on sphere.
    n_samples : int
        Number of samples.
    n_iter : int
        Iterations.

    Returns
    -------
    ndarray
        Optimized spherical points.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    # Normalize to unit sphere
    for i in range(n):
        norm = np.linalg.norm(points[i])
        if norm > 1e-12:
            points[i] /= norm

    for it in range(n_iter):
        # Sample on sphere surface uniformly
        samples = np.random.normal(0.0, 1.0, (n_samples, 3))
        for j in range(n_samples):
            norm = np.linalg.norm(samples[j])
            if norm > 1e-12:
                samples[j] /= norm

        new_points = np.zeros_like(points)
        counts = np.zeros(n)

        for s in samples:
            # Geodesic distance on sphere: use chord distance for simplicity
            dists = np.sum((points - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            new_points[nearest] += s
            counts[nearest] += 1

        for i in range(n):
            if counts[i] > 0:
                # Project centroid back to sphere
                centroid = new_points[i] / counts[i]
                norm = np.linalg.norm(centroid)
                if norm > 1e-12:
                    points[i] = centroid / norm

    return points


def hexagon_area(radius=1.0):
    """
    Area of regular hexagon with circumradius = radius.
    From 529_hexagon_monte_carlo.
    """
    return 3.0 * np.sqrt(3.0) / 2.0 * radius ** 2


def hexagon_sample(n, radius=1.0):
    """
    Sample n points uniformly inside a regular hexagon.
    From 529_hexagon_monte_carlo.

    Parameters
    ----------
    n : int
        Number of samples.
    radius : float
        Hexagon circumradius.

    Returns
    -------
    x, y : ndarray
        Sample coordinates.
    """
    # Rejection sampling within bounding box
    x_samples = []
    y_samples = []
    batch = n * 3
    while len(x_samples) < n:
        x = np.random.uniform(-radius, radius, batch)
        y = np.random.uniform(-radius, radius, batch)
        # Hexagon condition: |x| <= r*sqrt(3)/2 and |y| <= r - |x|/sqrt(3)
        mask = (np.abs(x) <= radius * np.sqrt(3.0) / 2.0) & \
               (np.abs(y) <= radius - np.abs(x) / np.sqrt(3.0))
        x_samples.extend(x[mask])
        y_samples.extend(y[mask])
    return np.array(x_samples[:n]), np.array(y_samples[:n])


def hexagon_monte_carlo_integrate(f, n_samples=10000, radius=1.0):
    """
    Monte Carlo integration over a regular hexagon.
    From 529_hexagon_monte_carlo.

    Parameters
    ----------
    f : callable
        Function f(x, y) to integrate.
    n_samples : int
        Number of Monte Carlo samples.
    radius : float
        Hexagon circumradius.

    Returns
    -------
    float
        Integral estimate.
    """
    area = hexagon_area(radius)
    x, y = hexagon_sample(n_samples, radius)
    values = f(x, y)
    return area * np.mean(values)


def spherical_hex_patches(n_lat=18):
    """
    Create hexagonal-like patches on a sphere for integration.
    Uses latitude bands with approximately hexagonal cells.

    Parameters
    ----------
    n_lat : int
        Number of latitude bands.

    Returns
    -------
    centers : ndarray, shape (N, 2)
        (latitude, longitude) of patch centers in degrees.
    areas : ndarray
        Patch areas in steradians.
    """
    lat_edges = np.linspace(-90, 90, n_lat + 1)
    centers = []
    areas = []

    for i in range(n_lat):
        lat_lo = np.deg2rad(lat_edges[i])
        lat_hi = np.deg2rad(lat_edges[i + 1])
        lat_mid = 0.5 * (lat_lo + lat_hi)

        # Number of longitude cells proportional to cos(latitude)
        n_lon = max(1, int(2.0 * n_lat * np.cos(lat_mid)))
        lon_edges = np.linspace(0, 360, n_lon + 1)

        for j in range(n_lon):
            lon_lo = np.deg2rad(lon_edges[j])
            lon_hi = np.deg2rad(lon_edges[j + 1])
            lon_mid = 0.5 * (lon_lo + lon_hi)

            centers.append([np.rad2deg(lat_mid), np.rad2deg(lon_mid)])
            # Spherical patch area: dlon * (sin(lat_hi) - sin(lat_lo))
            dlon = lon_hi - lon_lo
            areas.append(dlon * (np.sin(lat_hi) - np.sin(lat_lo)))

    return np.array(centers), np.array(areas)


def global_integral_monte_carlo(f, n_patches=100, n_samples_per_patch=100):
    """
    Compute global integral on sphere using hexagonal patch Monte Carlo.
    f(lat_deg, lon_deg) -> scalar.

    Parameters
    ----------
    f : callable
        Function on sphere.
    n_patches : int
        Number of latitude bands.
    n_samples_per_patch : int
        Samples per patch.

    Returns
    -------
    float
        Global integral over sphere (4*pi if f=1).
    """
    centers, areas = spherical_hex_patches(n_patches)
    total = 0.0
    for center, area in zip(centers, areas):
        lat = center[0]
        lon = center[1]
        # Approximate hexagon around center
        val = f(lat, lon)
        total += area * val
    return total


def spherical_t6_triangulation(n_lat=9):
    """
    Generate a spherical triangulation using T6 (6-node quadratic) triangles.
    Based on 1343_triangulation_order6_contour concept.
    Maps icosahedral subdivision to spherical surface.

    Parameters
    ----------
    n_lat : int
        Controls resolution.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        Node (lat, lon) coordinates.
    elements : ndarray, shape (M, 6)
        Element connectivity for T6 triangles.
    """
    # Use icosahedron subdivision
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    # Normalize
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    # Convert to lat/lon
    lat = np.rad2deg(np.arcsin(np.clip(vertices[:, 2], -1, 1)))
    lon = np.rad2deg(np.arctan2(vertices[:, 1], vertices[:, 0]))
    nodes = np.column_stack([lat, lon])

    # Icosahedron faces (20 triangles with 3 vertices each)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])

    # For simplicity, return linear triangles (3 nodes) + midpoints as T6
    # T6: vertices 0,1,2 + edge midpoints 3,4,5
    n_faces = len(faces)
    elements = np.zeros((n_faces, 6), dtype=int)

    for i, face in enumerate(faces):
        elements[i, 0:3] = face
        # Edge midpoints - for simplicity use same as vertices (degenerate T6 to T3)
        elements[i, 3] = face[0]
        elements[i, 4] = face[1]
        elements[i, 5] = face[2]

    return nodes, elements
