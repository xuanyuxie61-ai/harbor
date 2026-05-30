
import numpy as np


def barycentric_coordinates_tetrahedron(p, p0, p1, p2, p3):
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p3 - p0
    vp = p - p0

    mat = np.column_stack((v0, v1, v2))
    try:
        coords = np.linalg.solve(mat, vp)
    except np.linalg.LinAlgError:
        return None

    lam1, lam2, lam3 = coords
    lam0 = 1.0 - lam1 - lam2 - lam3
    return np.array([lam0, lam1, lam2, lam3])


def interpolate_in_tetrahedron(p, p0, p1, p2, p3, v0, v1, v2, v3):
    lams = barycentric_coordinates_tetrahedron(p, p0, p1, p2, p3)
    if lams is None:
        return None
    if np.any(lams < -0.01) or np.any(lams > 1.01):
        return None
    return lams[0] * v0 + lams[1] * v1 + lams[2] * v2 + lams[3] * v3


def find_containing_tetrahedron(point, nodes, elements, search_radius=0.5):

    candidates = []
    for idx, elem in enumerate(elements):
        pts = nodes[elem]
        center = np.mean(pts, axis=0)
        if np.linalg.norm(center - point) < search_radius:
            candidates.append(idx)

    for idx in candidates:
        elem = elements[idx]
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        lams = barycentric_coordinates_tetrahedron(point, p0, p1, p2, p3)
        if lams is not None and np.all(lams >= -1e-6) and np.all(lams <= 1 + 1e-6):
            return idx, lams


    if len(candidates) > 0:
        return candidates[0], None
    return None, None


def scalar_field_interpolator(nodes, elements, values):
    def interp(point):
        idx, lams = find_containing_tetrahedron(point, nodes, elements)
        if idx is None:

            dists = np.linalg.norm(nodes - point, axis=1)
            nearest = np.argmin(dists)
            return values[nearest]

        if lams is not None:
            elem = elements[idx]
            return lams[0] * values[elem[0]] + lams[1] * values[elem[1]] + \
                   lams[2] * values[elem[2]] + lams[3] * values[elem[3]]

        elem = elements[idx]
        pts = nodes[elem]
        dists = np.linalg.norm(pts - point, axis=1)
        weights = 1.0 / (dists + 1e-15)
        weights /= np.sum(weights)
        return np.dot(weights, values[elem])

    return interp


def compute_volume_integral(nodes, elements, values):
    if elements.size == 0:
        return np.mean(values) * len(nodes) * 0.001

    total = 0.0
    total_vol = 0.0
    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p3 - p0
        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        f_avg = np.mean(values[elem])
        total += f_avg * vol
        total_vol += vol

    return total, total_vol


def compute_surface_integral_sphere(nodes, elements_tri, values, r_target=1.0):
    if elements_tri.size == 0:
        return 0.0, 0.0

    total = 0.0
    total_area = 0.0
    for tri in elements_tri:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        f_avg = np.mean(values[tri])
        total += f_avg * area
        total_area += area

    return total, total_area


def extract_surface_triangles(elements):
    if elements.size == 0:
        return np.array([])

    face_count = {}
    face_to_tet = {}

    for ei, elem in enumerate(elements):
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]
        for face in faces:
            face_count[face] = face_count.get(face, 0) + 1

    surface_faces = [list(face) for face, count in face_count.items() if count == 1]
    return np.array(surface_faces, dtype=int)


def gradient_recovery_superconvergent(nodes, elements, values):
    n_nodes = len(nodes)
    grad_sum = np.zeros((n_nodes, 3))
    weight_sum = np.zeros(n_nodes)

    if elements.size == 0:
        return grad_sum

    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        vals = values[elem]


        mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
        rhs = np.array([vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]])
        try:
            grad_elem = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            grad_elem = np.zeros(3)


        vol = abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
        for idx in elem:
            grad_sum[idx] += grad_elem * vol
            weight_sum[idx] += vol

    for i in range(n_nodes):
        if weight_sum[i] > 0:
            grad_sum[i] /= weight_sum[i]

    return grad_sum


def find_critical_points(nodes, elements, values):
    n_nodes = len(nodes)
    critical = []

    for i in range(n_nodes):

        neighbors = set()
        if elements.size > 0:
            for elem in elements:
                if i in elem:
                    for idx in elem:
                        if idx != i:
                            neighbors.add(idx)

        if len(neighbors) == 0:
            continue

        vals_neighbor = [values[j] for j in neighbors]
        val_i = values[i]

        if val_i > max(vals_neighbor):
            critical.append((i, 'maximum', val_i))
        elif val_i < min(vals_neighbor):
            critical.append((i, 'minimum', val_i))
        else:

            has_higher = any(val_i > v for v in vals_neighbor)
            has_lower = any(val_i < v for v in vals_neighbor)
            if has_higher and has_lower:
                critical.append((i, 'saddle', val_i))

    return critical
