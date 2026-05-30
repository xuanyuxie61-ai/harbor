
import numpy as np
from typing import Tuple, List, Optional


def sphere_llt_grid_points(r: float, pc: np.ndarray, lat_num: int, long_num: int) -> np.ndarray:
    pc = np.asarray(pc)
    n_points = 2 + lat_num * long_num
    xyz = np.zeros((n_points, 3), dtype=float)


    xyz[0, :] = pc + np.array([0.0, 0.0, r])
    idx = 1


    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        for lon in range(long_num):
            theta = 2.0 * np.pi * lon / long_num
            xyz[idx, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
            xyz[idx, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
            xyz[idx, 2] = pc[2] + r * np.cos(phi)
            idx += 1


    xyz[idx, :] = pc + np.array([0.0, 0.0, -r])
    return xyz


def spherical_triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, r: float = 1.0) -> float:
    v1, v2, v3 = np.asarray(v1), np.asarray(v2), np.asarray(v3)

    v1 = v1 / (np.linalg.norm(v1) + 1e-15)
    v2 = v2 / (np.linalg.norm(v2) + 1e-15)
    v3 = v3 / (np.linalg.norm(v3) + 1e-15)


    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


    s = (a + b + c) / 2.0

    tan_E_4 = np.sqrt(
        np.maximum(0.0,
            np.tan(s / 2.0) *
            np.tan((s - a) / 2.0) *
            np.tan((s - b) / 2.0) *
            np.tan((s - c) / 2.0)
        )
    )
    E = 4.0 * np.arctan(tan_E_4)
    return E * r * r


def spherical_voronoi_areas(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import ConvexHull
    except ImportError:

        return _approximate_voronoi_areas(xyz)

    n = xyz.shape[0]
    hull = ConvexHull(xyz)
    faces = hull.simplices

    areas = np.zeros(n, dtype=float)
    for face in faces:
        i, j, k = face
        area = spherical_triangle_area(xyz[i], xyz[j], xyz[k])
        areas[i] += area / 3.0
        areas[j] += area / 3.0
        areas[k] += area / 3.0

    return areas, faces


def _approximate_voronoi_areas(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    areas = np.zeros(n, dtype=float)

    for i in range(n):

        dots = xyz @ xyz[i]

        dots[i] = -2.0

        neighbors = np.argsort(-dots)[:6]

        ni = xyz[i] / (np.linalg.norm(xyz[i]) + 1e-15)

        if abs(ni[2]) < 0.9:
            u = np.cross(ni, np.array([0.0, 0.0, 1.0]))
        else:
            u = np.cross(ni, np.array([0.0, 1.0, 0.0]))
        u = u / (np.linalg.norm(u) + 1e-15)
        v = np.cross(ni, u)
        v = v / (np.linalg.norm(v) + 1e-15)

        angles = []
        for j in neighbors:
            pj = xyz[j] - np.dot(xyz[j], ni) * ni
            pj = pj / (np.linalg.norm(pj) + 1e-15)
            angle = np.arctan2(np.dot(pj, v), np.dot(pj, u))
            angles.append(angle)
        order = np.argsort(angles)

        poly_area = 0.0
        for k in range(len(order)):
            k1 = order[k]
            k2 = order[(k + 1) % len(order)]

            a = np.arccos(np.clip(dots[neighbors[k1]], -1.0, 1.0))
            b = np.arccos(np.clip(dots[neighbors[k2]], -1.0, 1.0))

            p1 = xyz[neighbors[k1]]
            p2 = xyz[neighbors[k2]]
            gamma = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
            poly_area += spherical_triangle_area(ni, p1, p2)
        areas[i] = poly_area / 2.0

    faces = np.array([])
    return areas, faces


def voronoi_neighbor_adjacency(xyz: np.ndarray, faces: Optional[np.ndarray] = None) -> np.ndarray:
    n = xyz.shape[0]
    adj = np.zeros((n, n), dtype=bool)

    if faces is None or len(faces) == 0:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xyz)
            faces = hull.simplices
        except ImportError:

            for i in range(n):
                dots = xyz @ xyz[i]
                dots[i] = -2.0
                neighbors = np.argsort(-dots)[:6]
                adj[i, neighbors] = True
            return adj

    for face in faces:
        i, j, k = face
        adj[i, j] = True
        adj[j, i] = True
        adj[i, k] = True
        adj[k, i] = True
        adj[j, k] = True
        adj[k, j] = True

    return adj


def build_regional_default_correlation(
    n_regions: int = 20,
    base_correlation: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:



    lat_num = max(2, int(np.sqrt(n_regions)))
    long_num = max(3, int((n_regions - 2) / lat_num))

    xyz = sphere_llt_grid_points(1.0, np.zeros(3), lat_num, long_num)
    n = xyz.shape[0]

    areas, faces = spherical_voronoi_areas(xyz)

    areas = areas / (areas.sum() + 1e-15)

    adj = voronoi_neighbor_adjacency(xyz, faces if len(faces) > 0 else None)










    corr = np.eye(n, dtype=float)


    from utils import nearest_correlation_matrix
    corr = nearest_correlation_matrix(corr)

    return xyz, areas, adj, corr


def test_spherical_grid():
    xyz = sphere_llt_grid_points(1.0, np.zeros(3), 3, 4)
    assert xyz.shape[0] == 2 + 3 * 4, "点数计算错误"

    areas, faces = spherical_voronoi_areas(xyz)
    assert np.all(areas >= 0), "面积存在负值"
    total_area = areas.sum()
    assert abs(total_area - 4 * np.pi) < 0.5, f"总面积偏离 4pi: {total_area}"

    adj = voronoi_neighbor_adjacency(xyz, faces if len(faces) > 0 else None)
    assert adj.shape == (xyz.shape[0], xyz.shape[0]), "邻接矩阵维度错误"

    xyz2, areas2, adj2, corr = build_regional_default_correlation(n_regions=12)
    assert corr.shape[0] == xyz2.shape[0], "相关性矩阵维度不匹配"
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5), "对角线不为 1"
    print(f"spherical_correlation_grid test passed. n_regions={xyz2.shape[0]}")


if __name__ == "__main__":
    test_spherical_grid()
