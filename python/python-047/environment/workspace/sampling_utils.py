
import numpy as np


def radical_inverse(n, base):
    result = 0.0
    inv_base = 1.0 / base
    f = inv_base
    while n > 0:
        d = n % base
        result += d * f
        f *= inv_base
        n //= base
    return result


def halton_sequence(dim, n_points, offset=0):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError("dim {} exceeds available primes".format(dim))
    
    seq = np.zeros((n_points, dim))
    for i in range(n_points):
        idx = i + offset
        for j in range(dim):
            seq[i, j] = radical_inverse(idx, primes[j])
    
    return seq


def hammersley_sequence_nd(dim, n_points, offset=0):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes) + 1:
        raise ValueError("dim too large")
    
    seq = np.zeros((n_points, dim))
    for i in range(n_points):
        idx = i + offset
        if n_points > 0:
            seq[i, 0] = (idx % (n_points + 1)) / n_points
        else:
            seq[i, 0] = 0.0
        for j in range(1, dim):
            seq[i, j] = radical_inverse(idx, primes[j - 1])
    
    return seq


def sphere_surface_uniform(n_points, method='fibonacci'):
    if method == 'fibonacci':
        phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
        indices = np.arange(n_points, dtype=float)
        theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_points)
        phi = 2.0 * np.pi * indices / phi_golden
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.column_stack([x, y, z])
    
    elif method == 'hammersley':
        seq = hammersley_sequence_nd(2, n_points)
        theta = np.arccos(1.0 - 2.0 * seq[:, 0])
        phi = 2.0 * np.pi * seq[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    elif method == 'random':
        phi = 2.0 * np.pi * np.random.rand(n_points)
        cos_theta = 2.0 * np.random.rand(n_points) - 1.0
        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    else:
        raise ValueError("Unknown method: {}".format(method))


def great_circle_distance(p1, p2, radius=6371e3):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    
    if p1.ndim == 1 and p2.ndim == 1:
        lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
        lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        a = min(1.0, max(0.0, a))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return radius * c
    else:

        lat1 = np.radians(p1[:, 0])
        lon1 = np.radians(p1[:, 1])
        lat2 = np.radians(p2[:, 0])
        lon2 = np.radians(p2[:, 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return radius * c


def sphere_distance_statistics(points, n_pairs=None):
    points = np.asarray(points, dtype=float)
    
    if points.shape[1] == 2:

        lat, lon = np.radians(points[:, 0]), np.radians(points[:, 1])
        theta = np.pi / 2.0 - lat
        x = np.sin(theta) * np.cos(lon)
        y = np.sin(theta) * np.sin(lon)
        z = np.cos(theta)
        cart = np.column_stack([x, y, z])
    else:
        cart = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    N = cart.shape[0]
    
    if n_pairs is None or n_pairs >= N * (N - 1) // 2:

        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(cart[i] - cart[j])
                dists.append(d)
        dists = np.array(dists)
    else:

        dists = np.zeros(n_pairs)
        for k in range(n_pairs):
            i, j = np.random.choice(N, 2, replace=False)
            dists[k] = np.linalg.norm(cart[i] - cart[j])
    
    if len(dists) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return np.mean(dists), np.std(dists), np.min(dists), np.max(dists)


def generate_gravity_station_network(n_stations, method='fibonacci',
                                      lat_range=(-60, 60), lon_range=(0, 360)):
    if method == 'uniform_grid':
        n_lat = int(np.sqrt(n_stations))
        n_lon = int(np.ceil(n_stations / n_lat))
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        lons = np.linspace(lon_range[0], lon_range[1], n_lon)
        LAT, LON = np.meshgrid(lats, lons)
        stations = np.column_stack([LAT.flatten(), LON.flatten()])[:n_stations]
    elif method == 'random':
        lats = np.random.uniform(lat_range[0], lat_range[1], n_stations)
        lons = np.random.uniform(lon_range[0], lon_range[1], n_stations)
        stations = np.column_stack([lats, lons])
    else:

        sphere_pts = sphere_surface_uniform(n_stations, method='fibonacci')

        x, y, z = sphere_pts[:, 0], sphere_pts[:, 1], sphere_pts[:, 2]
        lat = 90.0 - np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        lon = np.mod(lon, 360.0)

        mask = (lat >= lat_range[0]) & (lat <= lat_range[1]) & \
               (lon >= lon_range[0]) & (lon <= lon_range[1])
        stations = np.column_stack([lat, lon])
        if not np.all(mask):

            n_missing = n_stations - np.sum(mask)
            extra_lats = np.random.uniform(lat_range[0], lat_range[1], n_missing)
            extra_lons = np.random.uniform(lon_range[0], lon_range[1], n_missing)
            stations = np.vstack([stations[mask], np.column_stack([extra_lats, extra_lons])])
    
    return stations[:n_stations]
