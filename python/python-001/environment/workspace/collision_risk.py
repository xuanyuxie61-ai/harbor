
import numpy as np
from typing import Tuple, Optional


class CollisionRiskError(Exception):
    pass


def ball_unit_sample(n: int = 1, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    points = np.zeros((n, 3))
    for i in range(n):
        u = np.random.rand(3)
        r = u[0] ** (1.0 / 3.0)
        theta = np.arccos(2.0 * u[1] - 1.0)
        phi = 2.0 * np.pi * u[2]
        points[i, 0] = r * np.sin(theta) * np.cos(phi)
        points[i, 1] = r * np.sin(theta) * np.sin(phi)
        points[i, 2] = r * np.cos(theta)
    return points


def ball_distance_stats(n_samples: int = 10000, seed: Optional[int] = None) -> Tuple[float, float]:
    if seed is not None:
        np.random.seed(seed)
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        p = ball_unit_sample(1)
        q = ball_unit_sample(1)
        distances[i] = np.linalg.norm(p - q)

    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def build_surface_adjacency_matrix(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    n_faces = faces.shape[0]
    adj = np.zeros((n_faces, n_faces), dtype=int)


    edge_to_faces = {}
    for fi in range(n_faces):
        for e in range(3):
            v1 = faces[fi, e]
            v2 = faces[fi, (e + 1) % 3]
            edge = tuple(sorted((int(v1), int(v2))))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)


    for edge, face_list in edge_to_faces.items():
        if len(face_list) >= 2:
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    f1 = face_list[i]
                    f2 = face_list[j]
                    adj[f1, f2] = 1
                    adj[f2, f1] = 1

    return adj


def compute_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n_faces = faces.shape[0]
    centroids = np.zeros((n_faces, 3))
    for i in range(n_faces):
        centroids[i] = np.mean(vertices[faces[i]], axis=0)
    return centroids


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n_faces = faces.shape[0]
    areas = np.zeros(n_faces)
    for i in range(n_faces):
        v1 = vertices[faces[i, 0]]
        v2 = vertices[faces[i, 1]]
        v3 = vertices[faces[i, 2]]
        areas[i] = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    return areas


def collision_probability_surface(
    pos: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    safe_distance: float = 0.5,
    position_uncertainty: float = 0.1
) -> float:
    centroids = compute_face_centroids(vertices, faces)
    areas = compute_face_areas(vertices, faces)
    total_area = np.sum(areas)
    if total_area < 1e-14:
        return 0.0

    p_total = 0.0
    for i in range(faces.shape[0]):
        d = np.linalg.norm(pos - centroids[i])

        z = (safe_distance - d) / max(position_uncertainty, 1e-12)
        p_face = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0) * 0.8))
        p_total += (areas[i] / total_area) * p_face

    return min(p_total, 1.0)


def find_safe_hover_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_altitude: float = 1.0,
    n_samples: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    centroids = compute_face_centroids(vertices, faces)
    normals = np.zeros_like(centroids)
    for i in range(faces.shape[0]):
        v1 = vertices[faces[i, 0]]
        v2 = vertices[faces[i, 1]]
        v3 = vertices[faces[i, 2]]
        n_vec = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n_vec)
        if norm > 1e-14:
            normals[i] = n_vec / norm


    candidates = centroids + min_altitude * normals

    safe_points = []
    safe_probs = []
    for i in range(min(n_samples, candidates.shape[0])):
        idx = i
        p_coll = collision_probability_surface(
            candidates[idx], vertices, faces,
            safe_distance=min_altitude * 0.5,
            position_uncertainty=min_altitude * 0.1
        )
        if p_coll < 0.1:
            safe_points.append(candidates[idx])
            safe_probs.append(p_coll)

    if len(safe_points) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(safe_points), np.array(safe_probs)


def region_connectivity_analysis(adj: np.ndarray) -> dict:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    component_sizes = []

    def bfs(start: int) -> Tuple[int, np.ndarray]:
        dist = -np.ones(n, dtype=int)
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if adj[u, v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return int(np.max(dist[dist >= 0])), dist

    max_diameter = 0
    for i in range(n):
        if not visited[i]:
            _, dist = bfs(i)
            component = np.where(dist >= 0)[0]
            component_sizes.append(len(component))
            visited[component] = True

            far_node = int(np.argmax(dist))
            d_far, _ = bfs(far_node)
            max_diameter = max(max_diameter, d_far)

    return {
        "n_components": len(component_sizes),
        "component_sizes": component_sizes,
        "diameter_est": max_diameter,
        "total_nodes": n
    }
