
import numpy as np
from math import sqrt






def polygon_contains_point(polygon, q):
    polygon = np.asarray(polygon)
    q = np.asarray(q)
    
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    x1, y1 = polygon[n - 1]
    
    for i in range(n):
        x2, y2 = polygon[i]
        

        if ((y1 < q[1] <= y2) or (q[1] <= y1 and y2 < q[1])):

            if y2 != y1:
                x_intersect = x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1)
                if q[0] <= x_intersect:
                    inside = not inside
        
        x1, y1 = x2, y2
    

    x1, y1 = polygon[n - 1]
    for i in range(n):
        x2, y2 = polygon[i]
        

        cross = (q[1] - y1) * (x2 - x1) - (y2 - y1) * (q[0] - x1)
        if abs(cross) < 1e-10:

            dot = (q[0] - x1) * (x2 - x1) + (q[1] - y1) * (y2 - y1)
            sq_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if 0 <= dot <= sq_len:
                return True
        
        x1, y1 = x2, y2
    
    return inside


def compute_polygon_bbox(polygon):
    polygon = np.asarray(polygon)
    if len(polygon) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    xmin = np.min(polygon[:, 0])
    xmax = np.max(polygon[:, 0])
    ymin = np.min(polygon[:, 1])
    ymax = np.max(polygon[:, 1])
    return (xmin, xmax, ymin, ymax)






def find_closest(ndim, n_generators, n_samples, samples, generators):
    nearest = np.zeros(n_samples, dtype=int)
    for j in range(n_samples):
        min_dist = float('inf')
        min_idx = 0
        for i in range(n_generators):
            d = 0.0
            for k in range(ndim):
                diff = generators[k, i] - samples[k, j]
                d += diff * diff
            if d < min_dist:
                min_dist = d
                min_idx = i
        nearest[j] = min_idx
    return nearest


def cvt_iterate(n, r, ratio):
    ndim = 2
    sample_num = ratio * n
    

    s = np.random.rand(ndim, sample_num)
    

    nearest = find_closest(ndim, n, sample_num, s, r)
    

    r2 = np.zeros((ndim, n))
    energy = 0.0
    count = np.zeros(n)
    
    for j in range(sample_num):
        idx = nearest[j]
        r2[:, idx] += s[:, j]
        dx = r[0, idx] - s[0, j]
        dy = r[1, idx] - s[1, j]
        energy += dx * dx + dy * dy
        count[idx] += 1
    
    energy = energy / sample_num
    

    for j in range(n):
        if count[j] > 0:
            r2[:, j] /= count[j]
    

    diff = 0.0
    for j in range(n):
        dx = r2[0, j] - r[0, j]
        dy = r2[1, j] - r[1, j]
        diff += sqrt(dx * dx + dy * dy)
    

    r[:, :] = r2[:, :]
    
    return r, diff, energy


def generate_cvt_mesh(n_generators, n_iterations=50, ratio=1000, domain=None):
    if domain is None:
        domain = ((0.0, 1.0), (0.0, 1.0))
    
    (xmin, xmax), (ymin, ymax) = domain
    

    r = np.zeros((2, n_generators))
    r[0, :] = np.random.uniform(xmin, xmax, n_generators)
    r[1, :] = np.random.uniform(ymin, ymax, n_generators)
    
    diff_history = []
    energy_history = []
    
    for _ in range(n_iterations):
        r, diff, energy = cvt_iterate(n_generators, r, ratio)
        diff_history.append(diff)
        energy_history.append(energy)
    
    return r, diff_history, energy_history






def define_cardiac_boundary(model='ventricle'):
    if model == 'ventricle':

        t = np.linspace(0, 2 * np.pi, 100)

        a, b = 1.0, 1.4
        x = a * np.cos(t)
        y = b * np.sin(t) * (1.0 + 0.1 * np.cos(3 * t))

        y = y - 0.3 * np.sin(t) ** 2
        polygon = np.column_stack((x, y))
        return polygon
    elif model == 'atrium':

        t = np.linspace(0, 2 * np.pi, 80)
        x = 1.2 * np.cos(t) + 0.2 * np.cos(3 * t)
        y = 0.8 * np.sin(t) + 0.1 * np.sin(4 * t)
        polygon = np.column_stack((x, y))
        return polygon
    else:

        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def filter_points_in_polygon(points, polygon):
    points = np.asarray(points)
    n = len(points)
    mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        mask[i] = polygon_contains_point(polygon, points[i])
    
    return points[mask], mask


def generate_cardiac_mesh(n_points, model='ventricle', n_cvt_iter=30):
    polygon = define_cardiac_boundary(model)
    bbox = compute_polygon_bbox(polygon)
    

    n_generators = int(n_points / 0.6)
    

    domain = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    generators, _, _ = generate_cvt_mesh(n_generators, n_cvt_iter, ratio=500,
                                          domain=domain)
    

    nodes, mask = filter_points_in_polygon(generators.T, polygon)
    

    while len(nodes) < n_points:
        extra = np.zeros((n_points - len(nodes), 2))
        extra[:, 0] = np.random.uniform(bbox[0], bbox[1], len(extra))
        extra[:, 1] = np.random.uniform(bbox[2], bbox[3], len(extra))
        extra_inside, _ = filter_points_in_polygon(extra, polygon)
        if len(extra_inside) > 0:
            nodes = np.vstack([nodes, extra_inside])
        else:
            break
    

    if len(nodes) > n_points:
        nodes = nodes[:n_points]
    
    return nodes, polygon






def scattered_interpolation_2d(data_points, data_values, query_points):
    data_points = np.asarray(data_points)
    data_values = np.asarray(data_values)
    query_points = np.asarray(query_points)
    
    n_query = len(query_points)
    n_data = len(data_points)
    interpolated = np.zeros(n_query)
    
    p = 2.0
    
    for j in range(n_query):
        qx, qy = query_points[j]
        weights = np.zeros(n_data)
        
        for i in range(n_data):
            dx = qx - data_points[i, 0]
            dy = qy - data_points[i, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                interpolated[j] = data_values[i]
                weights = None
                break
            weights[i] = 1.0 / (dist_sq ** (p / 2.0))
        
        if weights is not None:
            w_sum = np.sum(weights)
            if w_sum > 0:
                interpolated[j] = np.sum(weights * data_values) / w_sum
            else:
                interpolated[j] = np.mean(data_values)
    
    return interpolated
