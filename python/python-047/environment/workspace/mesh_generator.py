
import numpy as np


def voronoi_nearest_neighbor(points, query_points):
    points = np.asarray(points, dtype=float)
    query_points = np.asarray(query_points, dtype=float)
    
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if query_points.ndim == 1:
        query_points = query_points.reshape(-1, 1)
    
    N, d = points.shape
    M = query_points.shape[0]
    
    distances = np.zeros(M)
    indices = np.zeros(M, dtype=int)
    
    for m in range(M):
        q = query_points[m]
        diff = points - q
        dists = np.sum(diff**2, axis=1)
        idx = np.argmin(dists)
        distances[m] = np.sqrt(dists[idx])
        indices[m] = idx
    
    return distances, indices


def voronoi_region_area_2d(points, n_samples=10000):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    N = points.shape[0]
    

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins

    mins -= 0.1 * ranges
    maxs += 0.1 * ranges
    ranges = maxs - mins
    
    d = points.shape[1]
    total_area = np.prod(ranges)
    

    samples = np.random.rand(n_samples, d) * ranges + mins
    _, indices = voronoi_nearest_neighbor(points, samples)
    
    areas = np.zeros(N)
    for i in range(N):
        count = np.sum(indices == i)
        areas[i] = count / n_samples * total_area
    
    return areas


def cvt_lloyd_iteration(points, metric_func, n_samples=5000, n_iter=20):
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    N, d = points.shape
    energies = []
    

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    mins -= 0.2 * ranges
    maxs += 0.2 * ranges
    ranges = maxs - mins
    
    for it in range(n_iter):

        samples = np.random.rand(n_samples, d) * ranges + mins
        

        nearest = np.zeros(n_samples, dtype=int)
        min_dists = np.full(n_samples, np.inf)
        
        for g in range(N):
            pg = points[g]
            A = metric_func(pg)

            A = _ensure_spd(A)
            
            diff = samples - pg

            dists = np.sum(diff @ A * diff, axis=1)
            mask = dists < min_dists
            min_dists[mask] = dists[mask]
            nearest[mask] = g
        

        energy = np.mean(min_dists)
        energies.append(energy)
        

        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        for g in range(N):
            mask = nearest == g
            count = np.sum(mask)
            counts[g] = count
            if count > 0:
                new_points[g] = np.mean(samples[mask], axis=0)
            else:
                new_points[g] = points[g]
        
        points = new_points.copy()
    
    return points, energies


def _ensure_spd(A):
    A = np.asarray(A, dtype=float)

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-8)
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A


def adaptive_metric_from_gravity(gravity_grad, base_scale=1.0):
    grad = np.asarray(gravity_grad, dtype=float)
    g_norm = np.linalg.norm(grad)
    factor = (1.0 + g_norm / 1e-3) * base_scale
    d = len(grad)
    A = np.eye(d) * factor
    return A


def spherical_triangle_histogram(lat, lon, n_divisions=6):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    N = len(lat)
    

    theta = np.radians(90.0 - lat)
    phi = np.radians(lon)
    

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    






    
    abs_coords = np.column_stack([np.abs(x), np.abs(y), np.abs(z)])
    max_dim = np.argmax(abs_coords, axis=1)
    sign = np.sign([x, y, z]).T
    

    face_id = np.zeros(N, dtype=int)
    for i in range(N):
        sx = 0 if sign[i, 0] > 0 else 1
        sy = 0 if sign[i, 1] > 0 else 1
        sz = 0 if sign[i, 2] > 0 else 1
        face_id[i] = sx * 4 + sy * 2 + sz
    

    n_tri = n_divisions * n_divisions
    histo = np.zeros(8 * n_tri, dtype=int)
    
    for i in range(N):
        fid = face_id[i]


        md = max_dim[i]
        if md == 0:
            u = abs(y[i]) / (abs(y[i]) + abs(z[i]) + 1e-15)
            v = abs(z[i]) / (abs(y[i]) + abs(z[i]) + 1e-15)
        elif md == 1:
            u = abs(x[i]) / (abs(x[i]) + abs(z[i]) + 1e-15)
            v = abs(z[i]) / (abs(x[i]) + abs(z[i]) + 1e-15)
        else:
            u = abs(x[i]) / (abs(x[i]) + abs(y[i]) + 1e-15)
            v = abs(y[i]) / (abs(x[i]) + abs(y[i]) + 1e-15)
        
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        

        iu = min(int(u * n_divisions), n_divisions - 1)
        iv = min(int(v * n_divisions), n_divisions - 1)
        tri_idx = iu * n_divisions + iv
        global_idx = fid * n_tri + tri_idx
        if global_idx < len(histo):
            histo[global_idx] += 1
    
    expected_count = N / (8.0 * n_tri)
    if expected_count > 0:
        variance = np.var(histo)
        max_dev = np.max(np.abs(histo - expected_count))
        uniformity_index = max_dev / (N + 1e-15)
    else:
        uniformity_index = 0.0
    
    return histo, uniformity_index, expected_count


def adaptive_gravity_mesh(obs_lat, obs_lon, obs_gravity,
                           base_nx=20, base_ny=20,
                           cvt_samples=2000, cvt_iter=10):

    lat_min, lat_max = np.min(obs_lat), np.max(obs_lat)
    lon_min, lon_max = np.min(obs_lon), np.max(obs_lon)
    
    lat_grid = np.linspace(lat_min, lat_max, base_nx)
    lon_grid = np.linspace(lon_min, lon_max, base_ny)
    LAT, LON = np.meshgrid(lat_grid, lon_grid)
    points = np.column_stack([LAT.flatten(), LON.flatten()])
    

    def metric_func(p):

        diffs = obs_lat - p[0] + obs_lon - p[1]
        idx = np.argmin(np.abs(diffs))

        grad_lat = 0.0
        grad_lon = 0.0
        if idx > 0 and idx < len(obs_gravity) - 1:
            grad_lat = (obs_gravity[idx + 1] - obs_gravity[idx - 1]) / (obs_lat[idx + 1] - obs_lat[idx - 1] + 1e-10)
            grad_lon = (obs_gravity[idx + 1] - obs_gravity[idx - 1]) / (obs_lon[idx + 1] - obs_lon[idx - 1] + 1e-10)
        grad = np.array([grad_lat, grad_lon])
        return adaptive_metric_from_gravity(grad, base_scale=1.0)
    

    opt_points, energies = cvt_lloyd_iteration(points, metric_func,
                                                n_samples=cvt_samples,
                                                n_iter=cvt_iter)
    

    _, uniformity, _ = spherical_triangle_histogram(opt_points[:, 0], opt_points[:, 1], n_divisions=4)
    
    return opt_points, uniformity, energies
