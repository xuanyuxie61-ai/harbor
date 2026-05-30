
import numpy as np


PHI = (1.0 + np.sqrt(5.0)) / 2.0


def normalize_to_sphere(v):
    v = np.array(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return v / norms[:, np.newaxis]


def icosahedron_vertices():
    verts = np.array([
        [-1.0,  PHI,  0.0],
        [ 1.0,  PHI,  0.0],
        [-1.0, -PHI,  0.0],
        [ 1.0, -PHI,  0.0],
        [ 0.0, -1.0,  PHI],
        [ 0.0,  1.0,  PHI],
        [ 0.0, -1.0, -PHI],
        [ 0.0,  1.0, -PHI],
        [ PHI,  0.0, -1.0],
        [ PHI,  0.0,  1.0],
        [-PHI,  0.0, -1.0],
        [-PHI,  0.0,  1.0]
    ], dtype=float)
    return normalize_to_sphere(verts)


def icosahedron_faces():
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)
    return faces


def subdivide_sphere_mesh(vertices, faces):
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)
    
    new_vertices = vertices.tolist()
    new_faces = []
    

    edge_map = {}
    
    def get_midpoint_index(v1, v2):
        key = tuple(sorted([v1, v2]))
        if key in edge_map:
            return edge_map[key]
        
        mid = (vertices[v1] + vertices[v2]) / 2.0
        mid = normalize_to_sphere(mid)
        idx = len(new_vertices)
        new_vertices.append(mid)
        edge_map[key] = idx
        return idx
    
    for face in faces:
        v0, v1, v2 = face
        

        a = get_midpoint_index(v0, v1)
        b = get_midpoint_index(v1, v2)
        c = get_midpoint_index(v2, v0)
        

        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])
    
    return np.array(new_vertices, dtype=float), np.array(new_faces, dtype=int)


def generate_geodesic_grid(refinement_levels=3):
    vertices = icosahedron_vertices()
    faces = icosahedron_faces()
    
    for _ in range(refinement_levels):
        vertices, faces = subdivide_sphere_mesh(vertices, faces)
    
    return vertices, faces


def spherical_triangle_area(v1, v2, v3, radius=6.371e6):
    v1 = normalize_to_sphere(v1)
    v2 = normalize_to_sphere(v2)
    v3 = normalize_to_sphere(v3)
    

    a = np.arctan2(np.linalg.norm(np.cross(v2, v3)), np.dot(v2, v3))
    b = np.arctan2(np.linalg.norm(np.cross(v1, v3)), np.dot(v1, v3))
    c = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    
    s = 0.5 * (a + b + c)
    

    tan_s2 = np.tan(s / 2.0)
    tan_sa2 = np.tan(max(s - a, 0.0) / 2.0)
    tan_sb2 = np.tan(max(s - b, 0.0) / 2.0)
    tan_sc2 = np.tan(max(s - c, 0.0) / 2.0)
    
    product = tan_s2 * tan_sa2 * tan_sb2 * tan_sc2
    product = max(product, 0.0)
    
    E = 4.0 * np.arctan(np.sqrt(product))
    area = E * radius**2
    
    return area


def compute_mesh_statistics(vertices, faces, radius=6.371e6):
    n_v = len(vertices)
    n_f = len(faces)
    
    areas = []
    for face in faces:
        area = spherical_triangle_area(vertices[face[0]], vertices[face[1]],
                                       vertices[face[2]], radius)
        areas.append(area)
    
    areas = np.array(areas)
    
    stats = {
        'n_vertices': n_v,
        'n_faces': n_f,
        'total_area': np.sum(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'area_uniformity': np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0.0
    }
    
    return stats


def spherical_voronoi_centroids(vertices, faces):
    centroids = []
    for face in faces:
        c = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]]) / 3.0
        c = normalize_to_sphere(c)
        centroids.append(c)
    return np.array(centroids)
