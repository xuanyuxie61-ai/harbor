
import numpy as np
from typing import Tuple, List, Optional


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def barycentric_coordinates(x: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    A = triangle_area(p1, p2, p3)
    if abs(A) < 1e-14:
        return -1.0, -1.0, -1.0
    A1 = triangle_area(x, p2, p3)
    A2 = triangle_area(p1, x, p3)
    A3 = triangle_area(p1, p2, x)
    l1 = A1 / A
    l2 = A2 / A
    l3 = A3 / A
    return l1, l2, l3


def point_in_triangle(x: np.ndarray,
                      p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                      tol: float = 1e-10) -> bool:
    l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
    return (l1 >= -tol) and (l2 >= -tol) and (l3 >= -tol) and abs(l1 + l2 + l3 - 1.0) < 1e-8


def interpolate_on_triangle(x: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                            u1: float, u2: float, u3: float) -> float:
    l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
    if l1 < -1e-8 or l2 < -1e-8 or l3 < -1e-8:
        return np.nan
    return l1 * u1 + l2 * u2 + l3 * u3


def find_containing_triangle(x: np.ndarray,
                              points: np.ndarray,
                              triangles: np.ndarray) -> Tuple[int, np.ndarray]:
    for tri_idx in range(triangles.shape[0]):
        nodes = triangles[tri_idx]
        p1 = points[nodes[0]]
        p2 = points[nodes[1]]
        p3 = points[nodes[2]]
        if point_in_triangle(x, p1, p2, p3):
            l1, l2, l3 = barycentric_coordinates(x, p1, p2, p3)
            return tri_idx, np.array([l1, l2, l3])
    return -1, np.zeros(3)


def interpolate_mesh_field(points: np.ndarray,
                            triangles: np.ndarray,
                            field_values: np.ndarray,
                            query_points: np.ndarray) -> np.ndarray:
    nq = query_points.shape[0]
    interp = np.full(nq, np.nan)
    for iq in range(nq):
        tri_idx, bary = find_containing_triangle(query_points[iq], points, triangles)
        if tri_idx >= 0:
            nodes = triangles[tri_idx]
            interp[iq] = bary[0] * field_values[nodes[0]] + \
                         bary[1] * field_values[nodes[1]] + \
                         bary[2] * field_values[nodes[2]]
    return interp


def polygon_contains_point(poly: np.ndarray, q: np.ndarray) -> bool:
    n = poly.shape[0]
    inside = False
    x1, y1 = poly[n - 1]
    for i in range(n):
        x2, y2 = poly[i]

        cross = (q[1] - y1) * (x2 - x1) - (y2 - y1) * (q[0] - x1)
        if abs(cross) < 1e-12:

            if min(x1, x2) - 1e-12 <= q[0] <= max(x1, x2) + 1e-12 and \
               min(y1, y2) - 1e-12 <= q[1] <= max(y1, y2) + 1e-12:
                return True

        if ((y1 > q[1]) != (y2 > q[1])):
            xinters = (q[1] - y1) * (x2 - x1) / (y2 - y1) + x1
            if xinters > q[0]:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def integrate_field_over_mesh(points: np.ndarray,
                               triangles: np.ndarray,
                               field_values: np.ndarray) -> float:
    total = 0.0
    for tri in triangles:
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        A = abs(triangle_area(p1, p2, p3))
        avg_val = (field_values[tri[0]] + field_values[tri[1]] + field_values[tri[2]]) / 3.0
        total += A * avg_val
    return total


def demo():

    points = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
    ])
    triangles = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
    ])

    field = np.array([0.0, 1.0, 2.0, 1.0, 1.5])

    query = np.array([[0.6, 0.4], [0.2, 0.8], [1.5, 0.5]])
    interp = interpolate_mesh_field(points, triangles, field, query)
    print(f"[causal_mesh_interpolator] 插值结果: {interp}")

    total = integrate_field_over_mesh(points, triangles, field)
    print(f"[causal_mesh_interpolator] 网格上因果场积分: {total:.4f}")


    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    print(f"[causal_mesh_interpolator] 点(0.5,0.5)在多边形内? {polygon_contains_point(poly, np.array([0.5, 0.5]))}")
    return interp, total


if __name__ == "__main__":
    demo()
