
import numpy as np






def disk_sample_uniform(n, r=1.0):
    u = np.random.rand(n)
    v = np.random.rand(n)
    radius = r * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def disk_sample_nonuniform(n, r=1.0, density_power=1.0):
    u = np.random.rand(n)
    v = np.random.rand(n)

    radius = r * u**(1.0 / (2.0 + density_power))
    theta = 2.0 * np.pi * v
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def find_closest_samples(samples, generators):
    n_s = samples.shape[0]
    labels = np.zeros(n_s, dtype=int)
    for i in range(n_s):
        dists = np.sum((generators - samples[i])**2, axis=1)
        labels[i] = np.argmin(dists)
    return labels


def cvt_on_disk(n_generators, r=1.0, n_samples=10000, n_iterations=50,
                boundary_generators=None, density_power=0.0):
    if boundary_generators is None:
        boundary_generators = max(3, n_generators // 3)
    
    n_total = n_generators + boundary_generators
    

    generators = np.zeros((n_total, 2))
    types = np.zeros(n_total, dtype=int)
    

    generators[:n_generators, :] = disk_sample_uniform(n_generators, r)
    

    angles = np.linspace(0, 2*np.pi, boundary_generators, endpoint=False)
    generators[n_generators:, 0] = r * np.cos(angles)
    generators[n_generators:, 1] = r * np.sin(angles)
    types[n_generators:] = 1
    
    for it in range(n_iterations):

        if density_power > 0:
            samples = disk_sample_nonuniform(n_samples, r, density_power)
        else:
            samples = disk_sample_uniform(n_samples, r)
        
        labels = find_closest_samples(samples, generators)
        

        for g in range(n_generators):
            mask = (labels == g)
            if np.sum(mask) > 0:
                generators[g, :] = np.mean(samples[mask, :], axis=0)
        

        for g in range(n_generators, n_total):
            dx, dy = generators[g, 0], generators[g, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-10:
                generators[g, 0] = r * dx / dist
                generators[g, 1] = r * dy / dist
    
    return generators, types


def cvt_on_rectangle(n_generators, Lx, Ly, n_samples=10000, n_iterations=50):
    n_boundary = max(4, 2 * (n_generators // 5))
    n_total = n_generators + n_boundary
    
    generators = np.zeros((n_total, 2))
    types = np.zeros(n_total, dtype=int)
    

    generators[:n_generators, 0] = np.random.rand(n_generators) * Lx
    generators[:n_generators, 1] = np.random.rand(n_generators) * Ly
    

    nb_per_side = n_boundary // 4
    idx = n_generators
    for side in range(4):
        for k in range(nb_per_side):
            t = k / max(1, nb_per_side - 1)
            if side == 0:
                x, y = t * Lx, 0.0
            elif side == 1:
                x, y = Lx, t * Ly
            elif side == 2:
                x, y = (1-t) * Lx, Ly
            else:
                x, y = 0.0, (1-t) * Ly
            if idx < n_total:
                generators[idx, :] = [x, y]
                types[idx] = 1
                idx += 1
    
    for it in range(n_iterations):
        samples = np.column_stack([
            np.random.rand(n_samples) * Lx,
            np.random.rand(n_samples) * Ly
        ])
        
        labels = find_closest_samples(samples, generators)
        
        for g in range(n_generators):
            mask = (labels == g)
            if np.sum(mask) > 0:
                generators[g, :] = np.mean(samples[mask, :], axis=0)
        

        for g in range(n_generators, n_total):
            x, y = generators[g, :]

            d_left = x
            d_right = Lx - x
            d_bottom = y
            d_top = Ly - y
            d_min = min(d_left, d_right, d_bottom, d_top)
            if d_min == d_left:
                generators[g, 0] = 0.0
            elif d_min == d_right:
                generators[g, 0] = Lx
            elif d_min == d_bottom:
                generators[g, 1] = 0.0
            else:
                generators[g, 1] = Ly
    
    return generators, types






def compute_distance_matrix(coords):
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def path_length(p, D):
    n = len(p)
    total = 0.0
    for i in range(n - 1):
        total += D[p[i], p[i+1]]
    return total


def solve_tsp_moler(D, n_iterations=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = D.shape[0]
    if n <= 1:
        return 0.0, np.array([0, 0])
    

    p = np.random.permutation(n)
    best_path = p.copy()
    best_len = path_length(np.append(p, p[0]), D)
    
    for it in range(n_iterations):

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if i == j:
            continue
        
        p_new = p.copy()
        p_new[i:j+1] = p_new[i:j+1][::-1]
        len_new = path_length(np.append(p_new, p_new[0]), D)
        if len_new < best_len:
            best_len = len_new
            best_path = p_new.copy()
            p = p_new.copy()
            continue
        

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        
        p_new = p.copy()
        node = p_new[i]
        p_new = np.delete(p_new, i)
        p_new = np.insert(p_new, j, node)
        len_new = path_length(np.append(p_new, p_new[0]), D)
        if len_new < best_len:
            best_len = len_new
            best_path = p_new.copy()
            p = p_new.copy()
    
    return best_len, np.append(best_path, best_path[0])


def plan_ocean_sampling_route(station_coords, seed=None):
    D = compute_distance_matrix(station_coords)
    best_len, best_path = solve_tsp_moler(D, n_iterations=20000, seed=seed)
    
    return {
        'total_distance': best_len,
        'path': best_path,
        'distance_matrix': D,
    }






def deploy_sensor_network(domain_type, domain_params, n_sensors, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if domain_type == 'disk':
        r = domain_params.get('r', 1.0)
        generators, types = cvt_on_disk(n_sensors, r=r, n_samples=15000,
                                         n_iterations=60, density_power=0.5)
    elif domain_type == 'rectangle':
        Lx = domain_params.get('Lx', 1.0)
        Ly = domain_params.get('Ly', 1.0)
        generators, types = cvt_on_rectangle(n_sensors, Lx, Ly,
                                              n_samples=15000, n_iterations=60)
    else:
        raise ValueError(f"不支持的域类型: {domain_type}")
    

    interior_mask = (types == 0)
    interior_coords = generators[interior_mask]
    
    route = plan_ocean_sampling_route(interior_coords, seed=seed)
    
    return {
        'all_coords': generators,
        'sensor_types': types,
        'interior_coords': interior_coords,
        'route': route,
    }
