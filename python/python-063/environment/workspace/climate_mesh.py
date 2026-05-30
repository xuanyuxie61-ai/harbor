
import numpy as np


def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Cannot normalize zero vector")
    return v / norm


def generate_icosahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0,  phi, 0.0], [ 1.0,  phi, 0.0],
        [-1.0, -phi, 0.0], [ 1.0, -phi, 0.0],
        [0.0, -1.0,  phi], [0.0,  1.0,  phi],
        [0.0, -1.0, -phi], [0.0,  1.0, -phi],
        [ phi, 0.0, -1.0], [ phi, 0.0,  1.0],
        [-phi, 0.0, -1.0], [-phi, 0.0,  1.0]
    ], dtype=np.float64)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)
    return vertices, faces


def subdivide_spherical_mesh(vertices, faces, n_subdiv=2):
    verts = {i: tuple(v) for i, v in enumerate(vertices)}
    face_list = [list(f) for f in faces]
    next_idx = len(verts)

    for _ in range(n_subdiv):
        edge_dict = {}
        new_faces = []

        def get_midpoint_index(i, j):
            nonlocal next_idx
            key = tuple(sorted((i, j)))
            if key not in edge_dict:
                vi = np.array(verts[i])
                vj = np.array(verts[j])
                mid = _normalize((vi + vj) * 0.5)
                edge_dict[key] = next_idx
                verts[next_idx] = tuple(mid)
                next_idx += 1
            return edge_dict[key]

        for tri in face_list:
            a, b, c = tri
            ab = get_midpoint_index(a, b)
            bc = get_midpoint_index(b, c)
            ca = get_midpoint_index(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        face_list = new_faces

    max_idx = max(verts.keys())
    verts_array = np.zeros((max_idx + 1, 3), dtype=np.float64)
    for idx, v in verts.items():
        verts_array[idx] = v
    return verts_array, np.array(face_list, dtype=int)


def compute_spherical_triangle_area(v1, v2, v3):
    def spherical_distance(a, b):
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        return np.arccos(dot)

    a = spherical_distance(v2, v3)
    b = spherical_distance(v1, v3)
    c = spherical_distance(v1, v2)
    s = (a + b + c) * 0.5

    if s <= 0 or a <= 0 or b <= 0 or c <= 0:
        return 0.0

    tan_s2 = np.tan(s * 0.5)
    tan_sa2 = np.tan(max((s - a) * 0.5, 1e-15))
    tan_sb2 = np.tan(max((s - b) * 0.5, 1e-15))
    tan_sc2 = np.tan(max((s - c) * 0.5, 1e-15))

    tan_E4 = np.sqrt(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2)
    E = 4.0 * np.arctan(tan_E4)
    return E


def compute_mesh_areas(vertices, faces):
    areas = np.zeros(len(faces), dtype=np.float64)
    for i, tri in enumerate(faces):
        areas[i] = compute_spherical_triangle_area(
            vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        )
    return areas


def compute_dual_voronoi_areas(vertices, faces):
    n_nodes = len(vertices)
    areas = np.zeros(n_nodes, dtype=np.float64)
    for tri in faces:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        area = compute_spherical_triangle_area(v1, v2, v3)
        areas[tri[0]] += area / 3.0
        areas[tri[1]] += area / 3.0
        areas[tri[2]] += area / 3.0
    return areas


def mesh_info(vertices, faces):
    areas = compute_mesh_areas(vertices, faces)
    return {
        'n_nodes': len(vertices),
        'n_faces': len(faces),
        'total_area': float(np.sum(areas)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas)),
        'mean_area': float(np.mean(areas))
    }
