
import numpy as np
from typing import Tuple, List


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ], dtype=float)


def triangle_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    v1 = p2 - p1
    v2 = p3 - p1
    n = cross_product(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def compute_face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n_faces = faces.shape[0]
    normals = np.zeros((n_faces, 3))
    for f in range(n_faces):
        p1 = points[faces[f, 0]]
        p2 = points[faces[f, 1]]
        p3 = points[faces[f, 2]]
        normals[f] = triangle_normal(p1, p2, p3)
    return normals


def compute_vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n_nodes = points.shape[0]
    vnormals = np.zeros((n_nodes, 3))
    areas = np.zeros(n_nodes)
    for f in range(faces.shape[0]):
        p1 = points[faces[f, 0]]
        p2 = points[faces[f, 1]]
        p3 = points[faces[f, 2]]
        n = cross_product(p2 - p1, p3 - p1)
        area = 0.5 * np.linalg.norm(n)
        for v in faces[f]:
            vnormals[v] += n
            areas[v] += area
    for v in range(n_nodes):
        if areas[v] > 0.0:
            vnormals[v] = vnormals[v] / np.linalg.norm(vnormals[v])
        else:
            vnormals[v] = np.array([0.0, 0.0, 1.0])
    return vnormals


def mesh_edge_list(faces: np.ndarray) -> List[Tuple[int, int]]:
    edge_set = set()
    for f in range(faces.shape[0]):
        a, b, c = faces[f]
        edges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
        for e in edges:
            edge_set.add(e)
    return sorted(list(edge_set))


def vertex_degree(faces: np.ndarray, n_nodes: int) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=int)
    for f in range(faces.shape[0]):
        for v in faces[f]:
            deg[v] += 2

    adj = [set() for _ in range(n_nodes)]
    for f in range(faces.shape[0]):
        a, b, c = faces[f]
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    deg = np.array([len(s) for s in adj])
    return deg


def stla_string(points: np.ndarray, faces: np.ndarray) -> str:
    normals = compute_face_normals(points, faces)
    lines = ["solid CausalMesh"]
    for f in range(faces.shape[0]):
        n = normals[f]
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for v in faces[f]:
            p = points[v]
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid CausalMesh")
    return "\n".join(lines)


def generate_icosphere_nodes(radius: float = 1.0, subdivisions: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    points = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    points = radius * points / np.linalg.norm(points, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=int)


    if subdivisions > 0:
        for _ in range(subdivisions):
            new_faces = []
            edge_mid = {}
            def get_mid(a, b):
                nonlocal points
                key = tuple(sorted([a, b]))
                if key not in edge_mid:
                    mid = (points[a] + points[b]) / 2.0
                    mid = radius * mid / np.linalg.norm(mid)
                    edge_mid[key] = len(points)
                    points = np.vstack([points, mid])
                return edge_mid[key]

            for f in faces:
                a, b, c = f
                ab = get_mid(a, b)
                bc = get_mid(b, c)
                ca = get_mid(c, a)
                new_faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
            faces = np.array(new_faces, dtype=int)

    return points, faces


def demo():
    points, faces = generate_icosphere_nodes(radius=1.0, subdivisions=1)
    normals = compute_face_normals(points, faces)
    vnormals = compute_vertex_normals(points, faces)
    edges = mesh_edge_list(faces)
    deg = vertex_degree(faces, points.shape[0])
    print(f"[geometry_utils] 二十面体球: 节点={points.shape[0]}, 面片={faces.shape[0]}, 边={len(edges)}")
    print(f"[geometry_utils] 顶点度数范围: [{deg.min()}, {deg.max()}]")
    print(f"[geometry_utils] 面法向量示例: {normals[0].round(4)}")
    print(f"[geometry_utils] 顶点法向量示例: {vnormals[0].round(4)}")
    stl_str = stla_string(points[:3], faces[:1])
    print(f"[geometry_utils] STL 字符串长度: {len(stl_str)}")
    return points, faces


if __name__ == "__main__":
    demo()
