
import numpy as np


def circumcircle(points):
    A, B, C = points
    D = 2.0 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-15:
        return (A + B + C) / 3.0, 1e15

    ux = ((A[0] ** 2 + A[1] ** 2) * (B[1] - C[1]) +
          (B[0] ** 2 + B[1] ** 2) * (C[1] - A[1]) +
          (C[0] ** 2 + C[1] ** 2) * (A[1] - B[1])) / D

    uy = ((A[0] ** 2 + A[1] ** 2) * (C[0] - B[0]) +
          (B[0] ** 2 + B[1] ** 2) * (A[0] - C[0]) +
          (C[0] ** 2 + C[1] ** 2) * (B[0] - A[0])) / D

    center = np.array([ux, uy])
    radius = np.linalg.norm(center - A)
    return center, radius


def bowyer_watson(points):
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    if N < 3:
        return []


    minxy = points.min(axis=0)
    maxxy = points.max(axis=0)
    dx = maxxy[0] - minxy[0]
    dy = maxxy[1] - minxy[1]
    dmax = max(dx, dy)
    midx = (minxy[0] + maxxy[0]) / 2.0
    midy = (minxy[1] + maxxy[1]) / 2.0

    super_tri = np.array([
        [midx - 20 * dmax, midy - dmax],
        [midx + 20 * dmax, midy - dmax],
        [midx, midy + 20 * dmax],
    ])

    tri_points = np.vstack([points, super_tri])
    triangles = [(N, N + 1, N + 2)]

    for i in range(N):
        p = tri_points[i]
        bad_triangles = []
        for tri in triangles:
            center, radius = circumcircle(tri_points[list(tri)])
            if np.linalg.norm(p - center) < radius:
                bad_triangles.append(tri)

        polygon = []
        for tri in bad_triangles:
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                shared = False
                for other in bad_triangles:
                    if other is tri:
                        continue
                    if edge[0] in other and edge[1] in other:
                        shared = True
                        break
                if not shared:
                    polygon.append(edge)

        for tri in bad_triangles:
            triangles.remove(tri)

        for edge in polygon:
            triangles.append((edge[0], edge[1], i))


    final = []
    for tri in triangles:
        if N not in tri and (N + 1) not in tri and (N + 2) not in tri:
            final.append(tri)

    return final


def triangulate_domain_rectangle(xlim, ylim, nx=21, ny=21):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    triangles = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            i00 = iy * nx + ix
            i10 = i00 + 1
            i01 = i00 + nx
            i11 = i01 + 1
            triangles.append((i00, i10, i11))
            triangles.append((i00, i11, i01))

    return points, np.array(triangles)


def triangulate_domain_delaunay(points):
    tri_list = bowyer_watson(points)
    return points, np.array(tri_list)


def triangle_quality(points, tri):
    p = points[tri]
    a = np.linalg.norm(p[1] - p[0])
    b = np.linalg.norm(p[2] - p[1])
    c = np.linalg.norm(p[0] - p[2])

    area = 0.5 * abs(np.cross(p[1] - p[0], p[2] - p[0]))
    denom = a ** 2 + b ** 2 + c ** 2
    if denom < 1e-15:
        return 0.0
    return 4.0 * np.sqrt(3.0) * area / denom
