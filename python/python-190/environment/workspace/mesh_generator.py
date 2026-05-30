
import numpy as np


def circumcircle_center(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    d = 2.0 * ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))
    if abs(d) < 1e-14:

        center = (p1 + p2 + p3) / 3.0
        return center, 1e12

    ux = ((p1[0] ** 2 + p1[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p2[1] - p3[1])
          - (p2[0] ** 2 + p2[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p1[1] - p3[1]))
    uy = ((p2[0] ** 2 + p2[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p1[0] - p3[0])
          - (p1[0] ** 2 + p1[1] ** 2 - p3[0] ** 2 - p3[1] ** 2) * (p2[0] - p3[0]))
    center = np.array([ux / d, uy / d])
    radius = np.linalg.norm(center - p1)
    return center, radius


def point_in_circumcircle(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray) -> bool:
    center, radius = circumcircle_center(p1, p2, p3)
    return np.linalg.norm(pt - center) <= radius + 1e-12


def bowyer_watson(points: np.ndarray) -> list:
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("至少需要 3 个点才能进行三角剖分。")


    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    dmax = max(dx, dy)
    xmid = (xmin + xmax) * 0.5
    ymid = (ymin + ymax) * 0.5


    p_super = np.array([
        [xmid - 20.0 * dmax, ymid - 10.0 * dmax],
        [xmid, ymid + 20.0 * dmax],
        [xmid + 20.0 * dmax, ymid - 10.0 * dmax],
    ])
    pts_all = np.vstack([pts, p_super])
    super_idx = [n, n + 1, n + 2]

    triangles = [[super_idx[0], super_idx[1], super_idx[2]]]

    for i in range(n):
        pt = pts_all[i]
        bad_triangles = []
        for tri in triangles:
            if point_in_circumcircle(pt, pts_all[tri[0]], pts_all[tri[1]], pts_all[tri[2]]):
                bad_triangles.append(tri)


        polygon = []
        for tri in bad_triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                shared = False
                for other in bad_triangles:
                    if other is tri:
                        continue
                    oe = [(other[0], other[1]), (other[1], other[2]), (other[2], other[0])]

                    if (e[0], e[1]) in oe or (e[1], e[0]) in oe:
                        shared = True
                        break
                if not shared:
                    polygon.append(e)


        for tri in bad_triangles:
            triangles.remove(tri)


        for e in polygon:
            triangles.append([e[0], e[1], i])


    final_triangles = []
    for tri in triangles:
        if super_idx[0] not in tri and super_idx[1] not in tri and super_idx[2] not in tri:
            final_triangles.append(tuple(tri))

    return final_triangles


def generate_mesh_from_boundary(boundary: np.ndarray, hmax: float = 0.25) -> tuple:
    boundary = np.asarray(boundary, dtype=float)

    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)

    nx = max(3, int(np.ceil((xmax - xmin) / hmax)) + 1)
    ny = max(3, int(np.ceil((ymax - ymin) / hmax)) + 1)
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    candidates = np.column_stack([Xg.ravel(), Yg.ravel()])


    def point_in_polygon(pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]

            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if xinters > x:
                    inside = not inside
        return inside

    interior_points = []
    for pt in candidates:
        if point_in_polygon(pt, boundary):
            interior_points.append(pt)


    all_points = np.vstack([boundary, np.array(interior_points)])

    all_points = np.unique(np.round(all_points, 12), axis=0)
    triangles = bowyer_watson(all_points)
    return all_points, triangles


def human_outline_boundary(scale: float = 1.0) -> np.ndarray:
    pts = []

    for theta in np.linspace(np.pi * 0.5, -np.pi * 0.5, 15):
        x = 0.5 * np.cos(theta)
        y = 2.5 + 0.5 * np.sin(theta)
        pts.append([x, y])

    pts.append([0.6, 2.0])
    pts.append([0.55, 1.2])
    pts.append([0.5, 0.5])

    pts.append([0.45, 0.0])
    pts.append([0.3, -0.8])
    pts.append([0.2, -1.5])
    pts.append([0.15, -2.0])

    pts.append([0.0, -2.2])

    pts.append([-0.15, -2.0])
    pts.append([-0.2, -1.5])
    pts.append([-0.3, -0.8])
    pts.append([-0.45, 0.0])
    pts.append([-0.5, 0.5])
    pts.append([-0.55, 1.2])
    pts.append([-0.6, 2.0])
    boundary = np.array(pts, dtype=float) * scale
    return boundary


def get_triangle_vertices(nodes: np.ndarray, triangles: list) -> list:
    verts = []
    for tri in triangles:
        verts.append((nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]))
    return verts


def mesh_quality_stats(nodes: np.ndarray, triangles: list) -> dict:
    angles = []
    areas = []
    for tri in triangles:
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        def angle_from_sides(x, y, z):
            num = x * x + y * y - z * z
            den = 2.0 * x * y
            if den < 1e-14:
                return 0.0
            return np.arccos(np.clip(num / den, -1.0, 1.0))
        angles.append(angle_from_sides(b, c, a))
        angles.append(angle_from_sides(a, c, b))
        angles.append(angle_from_sides(a, b, c))

        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        area_sq = max(area_sq, 0.0)
        areas.append(np.sqrt(area_sq))

    angles_deg = np.degrees(np.array(angles))
    areas_arr = np.array(areas)
    return {
        "min_angle_deg": float(np.min(angles_deg)),
        "max_area": float(np.max(areas_arr)),
        "min_area": float(np.min(areas_arr)),
        "area_ratio": float(np.max(areas_arr) / (np.min(areas_arr) + 1e-15)),
        "num_triangles": len(triangles),
        "num_nodes": nodes.shape[0],
    }
