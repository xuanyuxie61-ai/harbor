
import numpy as np


def ball_grid_points(n_sub, radius, center):
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
    pts = ball_grid_points(n_sub, 1.0, [0.0, 0.0, 0.0])
    return len(pts)


def fibonacci_sphere(n_points, radius=1.0):
    indices = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (indices / (n_points - 1)) * 2.0
    radius_xy = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius_xy
    z = np.sin(theta) * radius_xy
    return radius * np.column_stack([x, y, z])


def cvt_energy(points, region_bounds=None):
    n = len(points)
    d = points.shape[1]

    if region_bounds is None:
        region_bounds = [(0.0, 1.0)] * d


    sample_num = min(10000, n * 100)
    samples = np.zeros((sample_num, d))
    for dim in range(d):
        lo, hi = region_bounds[dim]
        samples[:, dim] = np.random.uniform(lo, hi, sample_num)

    energy = 0.0
    counts = np.zeros(n)
    for s in samples:

        dists = np.sum((points - s) ** 2, axis=1)
        nearest = np.argmin(dists)
        energy += dists[nearest]
        counts[nearest] += 1

    return energy / sample_num


def cvt_iterate_lloyd(points, region_bounds=None, n_samples=5000, n_iter=50):
    points = np.array(points, dtype=float)
    n, d = points.shape

    if region_bounds is None:
        region_bounds = [(0.0, 1.0)] * d

    for it in range(n_iter):

        samples = np.zeros((n_samples, d))
        for dim in range(d):
            lo, hi = region_bounds[dim]
            samples[:, dim] = np.random.uniform(lo, hi, n_samples)


        new_points = np.zeros_like(points)
        counts = np.zeros(n)

        for s in samples:
            dists = np.sum((points - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            new_points[nearest] += s
            counts[nearest] += 1


        for i in range(n):
            if counts[i] > 0:
                points[i] = new_points[i] / counts[i]
            else:

                for dim in range(d):
                    lo, hi = region_bounds[dim]
                    points[i, dim] = np.random.uniform(lo, hi)

    return points


def cvt_on_sphere(points, n_samples=5000, n_iter=30):
    points = np.array(points, dtype=float)
    n = len(points)

    for i in range(n):
        norm = np.linalg.norm(points[i])
        if norm > 1e-12:
            points[i] /= norm

    for it in range(n_iter):

        samples = np.random.normal(0.0, 1.0, (n_samples, 3))
        for j in range(n_samples):
            norm = np.linalg.norm(samples[j])
            if norm > 1e-12:
                samples[j] /= norm

        new_points = np.zeros_like(points)
        counts = np.zeros(n)

        for s in samples:

            dists = np.sum((points - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            new_points[nearest] += s
            counts[nearest] += 1

        for i in range(n):
            if counts[i] > 0:

                centroid = new_points[i] / counts[i]
                norm = np.linalg.norm(centroid)
                if norm > 1e-12:
                    points[i] = centroid / norm

    return points


def hexagon_area(radius=1.0):
    return 3.0 * np.sqrt(3.0) / 2.0 * radius ** 2


def hexagon_sample(n, radius=1.0):

    x_samples = []
    y_samples = []
    batch = n * 3
    while len(x_samples) < n:
        x = np.random.uniform(-radius, radius, batch)
        y = np.random.uniform(-radius, radius, batch)

        mask = (np.abs(x) <= radius * np.sqrt(3.0) / 2.0) & \
               (np.abs(y) <= radius - np.abs(x) / np.sqrt(3.0))
        x_samples.extend(x[mask])
        y_samples.extend(y[mask])
    return np.array(x_samples[:n]), np.array(y_samples[:n])


def hexagon_monte_carlo_integrate(f, n_samples=10000, radius=1.0):
    area = hexagon_area(radius)
    x, y = hexagon_sample(n_samples, radius)
    values = f(x, y)
    return area * np.mean(values)


def spherical_hex_patches(n_lat=18):
    lat_edges = np.linspace(-90, 90, n_lat + 1)
    centers = []
    areas = []

    for i in range(n_lat):
        lat_lo = np.deg2rad(lat_edges[i])
        lat_hi = np.deg2rad(lat_edges[i + 1])
        lat_mid = 0.5 * (lat_lo + lat_hi)


        n_lon = max(1, int(2.0 * n_lat * np.cos(lat_mid)))
        lon_edges = np.linspace(0, 360, n_lon + 1)

        for j in range(n_lon):
            lon_lo = np.deg2rad(lon_edges[j])
            lon_hi = np.deg2rad(lon_edges[j + 1])
            lon_mid = 0.5 * (lon_lo + lon_hi)

            centers.append([np.rad2deg(lat_mid), np.rad2deg(lon_mid)])

            dlon = lon_hi - lon_lo
            areas.append(dlon * (np.sin(lat_hi) - np.sin(lat_lo)))

    return np.array(centers), np.array(areas)


def global_integral_monte_carlo(f, n_patches=100, n_samples_per_patch=100):
    centers, areas = spherical_hex_patches(n_patches)
    total = 0.0
    for center, area in zip(centers, areas):
        lat = center[0]
        lon = center[1]

        val = f(lat, lon)
        total += area * val
    return total


def spherical_t6_triangulation(n_lat=9):

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)

    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)


    lat = np.rad2deg(np.arcsin(np.clip(vertices[:, 2], -1, 1)))
    lon = np.rad2deg(np.arctan2(vertices[:, 1], vertices[:, 0]))
    nodes = np.column_stack([lat, lon])


    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])



    n_faces = len(faces)
    elements = np.zeros((n_faces, 6), dtype=int)

    for i, face in enumerate(faces):
        elements[i, 0:3] = face

        elements[i, 3] = face[0]
        elements[i, 4] = face[1]
        elements[i, 5] = face[2]

    return nodes, elements
