
import numpy as np
from typing import Tuple, List


def bernstein_poly_01(n: int, x: float) -> np.ndarray:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"bernstein_poly_01: x={x} 超出 [0,1] 区间")
    if n < 0:
        raise ValueError("bernstein_poly_01: n 必须非负")

    bern = np.zeros(n + 1)
    if n == 0:
        bern[0] = 1.0
        return bern


    bern[0] = 1.0 - x
    bern[1] = x







    for i in range(2, n + 1):
        bern[i] = x * bern[i - 1]
        for j in range(i - 1, 0, -1):
            bern[j] = x * bern[j - 1] + (1.0 - x) * bern[j]
        bern[0] = (1.0 - x) * bern[0]

    return bern


def bernstein_tumor_boundary(
    control_points: np.ndarray, num_samples: int = 256
) -> np.ndarray:
    n = control_points.shape[0] - 1
    if n < 1:
        raise ValueError("bernstein_tumor_boundary: 至少需要 2 个控制点")
    if control_points.shape[1] != 2:
        raise ValueError("bernstein_tumor_boundary: 控制点必须是二维坐标")

    t_vals = np.linspace(0.0, 1.0, num_samples)
    boundary_points = np.zeros((num_samples, 2))

    for idx, t in enumerate(t_vals):

        t_clamped = float(np.clip(t, 0.0, 1.0))
        bern = bernstein_poly_01(n, t_clamped)
        boundary_points[idx, :] = np.dot(bern, control_points)

    return boundary_points


def generate_delaunay_triangulation(
    boundary_points: np.ndarray, interior_points: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("generate_delaunay_triangulation: 边界点必须为 (B,2)")

    if interior_points is None:
        interior_points = np.zeros((0, 2))

    nodes = np.vstack([boundary_points, interior_points])
    n_nodes = nodes.shape[0]
    n_boundary = boundary_points.shape[0]



    min_xy = np.min(nodes, axis=0)
    max_xy = np.max(nodes, axis=0)
    dx, dy = max_xy - min_xy
    margin = max(dx, dy) * 2.0
    if margin <= 0:
        margin = 1.0

    super_tri = np.array([
        [min_xy[0] - margin, min_xy[1] - margin],
        [max_xy[0] + margin, min_xy[1] - margin],
        [min_xy[0] + 0.5 * dx, max_xy[1] + margin],
    ])


    tri_list = [(n_nodes, n_nodes + 1, n_nodes + 2)]
    nodes_extended = np.vstack([nodes, super_tri])

    def _circumcircle_contains(tri, p):
        a, b, c = tri
        pts = nodes_extended[[a, b, c], :]
        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]

        d = 2.0 * ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))
        if abs(d) < 1e-14:
            return False

        ux = (((ax - cx) ** 2 + (ay - cy) ** 2) * (by - cy) -
              ((bx - cx) ** 2 + (by - cy) ** 2) * (ay - cy)) / d
        uy = (((bx - cx) ** 2 + (by - cy) ** 2) * (ax - cx) -
              ((ax - cx) ** 2 + (ay - cy) ** 2) * (bx - cx)) / d
        cx_c = cx + ux
        cy_c = cy + uy
        r2 = (cx_c - cx) ** 2 + (cy_c - cy) ** 2
        dist2 = (p[0] - cx_c) ** 2 + (p[1] - cy_c) ** 2
        return dist2 < r2 - 1e-12

    for p_idx in range(n_nodes):
        p = nodes_extended[p_idx]
        bad_tris = []
        for tri in tri_list:
            if _circumcircle_contains(tri, p):
                bad_tris.append(tri)


        edges = []
        for tri in bad_tris:
            e = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in e:

                is_shared = False
                for other in bad_tris:
                    if other is tri:
                        continue
                    if (edge[1], edge[0]) in [(other[0], other[1]),
                                               (other[1], other[2]),
                                               (other[2], other[0])]:
                        is_shared = True
                        break
                if not is_shared:
                    edges.append(edge)


        for bt in bad_tris:
            tri_list.remove(bt)


        for edge in edges:
            tri_list.append((edge[0], edge[1], p_idx))


    filtered = []
    for tri in tri_list:
        if n_nodes not in tri and n_nodes + 1 not in tri and n_nodes + 2 not in tri:
            filtered.append(tri)

    triangles = np.array(filtered, dtype=int)
    return nodes, triangles


def detect_boundary_nodes(
    node_num: int, triangle_num: int, triangle_node: np.ndarray
) -> np.ndarray:
    if triangle_node.size == 0:
        return np.zeros(node_num, dtype=bool)


    if triangle_node.shape[0] == 3 and triangle_node.shape[1] == triangle_num:
        tri = triangle_node.T.copy()
    else:
        tri = triangle_node.copy()

    node_min = tri.min()
    node_max = tri.max()


    if node_min == 0 and node_max == node_num - 1:
        pass
    elif node_min == 1 and node_max == node_num:
        tri = tri - 1
    else:
        raise ValueError(
            f"detect_boundary_nodes: 节点索引范围 [{node_min}, {node_max}] "
            f"与节点总数 {node_num} 不匹配"
        )

    edge_count = {}
    for t in range(triangle_num):
        verts = tri[t, :]
        edges = [(verts[0], verts[1]),
                 (verts[1], verts[2]),
                 (verts[2], verts[0])]
        for e in edges:
            a, b = e
            if a > b:
                a, b = b, a
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    node_boundary = np.zeros(node_num, dtype=bool)
    for (a, b), count in edge_count.items():
        if count == 1:
            node_boundary[a] = True
            node_boundary[b] = True

    return node_boundary


def tumor_surface_triangulation(
    boundary_points: np.ndarray, interior_density: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boundary_points.shape[0] < 3:
        raise ValueError("tumor_surface_triangulation: 至少需要 3 个边界点")


    min_xy = np.min(boundary_points, axis=0)
    max_xy = np.max(boundary_points, axis=0)
    x_vals = np.linspace(min_xy[0], max_xy[0], interior_density)
    y_vals = np.linspace(min_xy[1], max_xy[1], interior_density)
    xx, yy = np.meshgrid(x_vals, y_vals)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])


    def _point_in_polygon(pt, poly):
        x, y = pt
        n = poly.shape[0]
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i
        return inside

    interior_points = []
    for pt in candidates:
        if _point_in_polygon(pt, boundary_points):

            dists = np.linalg.norm(boundary_points - pt, axis=1)
            if np.min(dists) > 1e-3 * np.linalg.norm(max_xy - min_xy):
                interior_points.append(pt)

    interior = np.array(interior_points) if interior_points else np.zeros((0, 2))

    nodes, triangles = generate_delaunay_triangulation(boundary_points, interior)
    is_boundary = detect_boundary_nodes(nodes.shape[0], triangles.shape[0], triangles)
    return nodes, triangles, is_boundary


def compute_tumor_area(nodes: np.ndarray, triangles: np.ndarray) -> float:
    area = 0.0
    for t in range(triangles.shape[0]):
        a, b, c = triangles[t, :]
        pa, pb, pc = nodes[a], nodes[b], nodes[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        area += 0.5 * abs(cross)
    return area


def compute_tumor_perimeter(boundary_points: np.ndarray) -> float:
    n = boundary_points.shape[0]
    if n < 2:
        return 0.0
    perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        perim += np.linalg.norm(boundary_points[j] - boundary_points[i])
    return perim
