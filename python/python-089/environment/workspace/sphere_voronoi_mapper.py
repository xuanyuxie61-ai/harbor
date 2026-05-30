
import numpy as np


def spherical_delaunay_triangulation(points):

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        faces = hull.simplices

        oriented_faces = []
        for f in faces:
            v1 = points[f[1]] - points[f[0]]
            v2 = points[f[2]] - points[f[0]]
            normal = np.cross(v1, v2)
            centroid = np.mean(points[f], axis=0)
            if np.dot(normal, centroid) < 0:
                f = [f[0], f[2], f[1]]
            oriented_faces.append(f)
        return np.array(oriented_faces)
    except ImportError:

        n = points.shape[0]
        if n < 4:
            return np.array([[0, 1, 2]]) if n == 3 else np.array([])

        faces = []
        for i in range(n):
            dists = np.sum((points - points[i]) ** 2, axis=1)
            nearest = np.argsort(dists)[1:4]
            faces.append([i, nearest[0], nearest[1]])
        return np.array(faces)


def voronoi_vertices_on_sphere(points, faces):
    v_vertices = np.zeros((faces.shape[0], 3))
    for i, f in enumerate(faces):
        p1, p2, p3 = points[f]

        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n = n / norm
        else:
            n = p1

        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
        v_vertices[i] = n
    return v_vertices


def spherical_triangle_area(p1, p2, p3):

    a = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(p1, p3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    
    s = (a + b + c) / 2.0


    if s >= np.pi:
        return 0.0
    
    tan_s2 = np.tan(s / 2.0)
    tan_sa2 = np.tan(max(0.0, (s - a)) / 2.0)
    tan_sb2 = np.tan(max(0.0, (s - b)) / 2.0)
    tan_sc2 = np.tan(max(0.0, (s - c)) / 2.0)
    
    if tan_s2 <= 0 or tan_sa2 < 0 or tan_sb2 < 0 or tan_sc2 < 0:


        n1 = np.cross(p1, p2)
        n2 = np.cross(p2, p3)
        n3 = np.cross(p3, p1)
        n1 = n1 / (np.linalg.norm(n1) + 1e-15)
        n2 = n2 / (np.linalg.norm(n2) + 1e-15)
        n3 = n3 / (np.linalg.norm(n3) + 1e-15)
        alpha = np.arccos(np.clip(-np.dot(n1, n2), -1.0, 1.0))
        beta = np.arccos(np.clip(-np.dot(n2, n3), -1.0, 1.0))
        gamma = np.arccos(np.clip(-np.dot(n3, n1), -1.0, 1.0))
        return alpha + beta + gamma - np.pi
    
    tan_E4 = np.sqrt(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2)
    E = 4.0 * np.arctan(tan_E4)
    return E


def voronoi_areas_on_sphere(points, faces, v_vertices):
    n = points.shape[0]
    areas = np.zeros(n)
    

    point_to_faces = [[] for _ in range(n)]
    for fi, f in enumerate(faces):
        for vi in f:
            point_to_faces[vi].append(fi)
    

    for i in range(n):
        incident = point_to_faces[i]
        if len(incident) < 3:
            continue
        



        for fi in incident:


            pass
        






        


        n_inc = len(incident)
        for k in range(n_inc):
            v1 = v_vertices[incident[k]]
            v2 = v_vertices[incident[(k + 1) % n_inc]]
            areas[i] += spherical_triangle_area(points[i], v1, v2)
    
    return areas


def map_stochastic_to_sphere(xi_samples, radius=1.0):
    norms = np.linalg.norm(xi_samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    directions = xi_samples / norms * radius
    radii = norms.flatten()
    return directions, radii


def directional_reliability_sensitivity(directions, response_values, 
                                         n_directional_bins=20):
    if directions.shape[1] > 3:

        from numpy.linalg import svd
        U, s, Vt = svd(directions, full_matrices=False)
        directions_3d = directions @ Vt[:3, :].T
        directions = directions_3d
    

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    directions = directions / norms
    

    if directions.shape[1] == 2:
        directions = np.column_stack([directions, np.zeros(len(directions))])
    

    theta = np.arctan2(directions[:, 1], directions[:, 0])
    phi = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    

    theta_bins = np.linspace(-np.pi, np.pi, n_directional_bins + 1)
    phi_bins = np.linspace(0, np.pi, n_directional_bins // 2 + 1)
    
    max_response_per_sector = np.zeros((n_directional_bins, n_directional_bins // 2))
    count_per_sector = np.zeros_like(max_response_per_sector)
    
    for i in range(n_directional_bins):
        for j in range(n_directional_bins // 2):
            mask = ((theta >= theta_bins[i]) & (theta < theta_bins[i + 1]) &
                    (phi >= phi_bins[j]) & (phi < phi_bins[j + 1]))
            if np.any(mask):
                max_response_per_sector[i, j] = np.max(response_values[mask])
                count_per_sector[i, j] = np.sum(mask)
    
    return {
        'max_response': max_response_per_sector,
        'counts': count_per_sector,
        'theta_bins': theta_bins,
        'phi_bins': phi_bins
    }
