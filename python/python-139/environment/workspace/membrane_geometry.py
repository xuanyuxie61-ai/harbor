
import numpy as np


def triangle_area_signed(xa, ya, xb, yb, xc, yc):
    return 0.5 * ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))


def polygon_area(n, x, y):
    area = 0.0
    im1 = n - 1
    for i in range(n):
        area += x[im1] * y[i] - x[i] * y[im1]
        im1 = i
    return 0.5 * area


def is_collinear(xa, ya, xb, yb, xc, yc, eps=1e-12):
    area = abs(triangle_area_signed(xa, ya, xb, yb, xc, yc))
    side_ab_sq = (xa - xb) ** 2 + (ya - yb) ** 2
    side_bc_sq = (xb - xc) ** 2 + (yb - yc) ** 2
    side_ca_sq = (xc - xa) ** 2 + (yc - ya) ** 2
    side_max_sq = max(side_ab_sq, side_bc_sq, side_ca_sq)
    if side_max_sq <= eps:
        return True
    return 2.0 * area <= eps * side_max_sq


def is_between(xa, ya, xb, yb, xc, yc):
    if not is_collinear(xa, ya, xb, yb, xc, yc):
        return False
    if abs(ya - yb) < abs(xa - xb):
        return min(xa, xb) <= xc <= max(xa, xb)
    else:
        return min(ya, yb) <= yc <= max(ya, yb)


def intersect_prop(xa, ya, xb, yb, xc, yc, xd, yd):
    if is_collinear(xa, ya, xb, yb, xc, yc):
        return False
    if is_collinear(xa, ya, xb, yb, xd, yd):
        return False
    if is_collinear(xc, yc, xd, yd, xa, ya):
        return False
    if is_collinear(xc, yc, xd, yd, xb, yb):
        return False
    t1 = triangle_area_signed(xa, ya, xb, yb, xc, yc)
    t2 = triangle_area_signed(xa, ya, xb, yb, xd, yd)
    t3 = triangle_area_signed(xc, yc, xd, yd, xa, ya)
    t4 = triangle_area_signed(xc, yc, xd, yd, xb, yb)
    return (t1 > 0) != (t2 > 0) and (t3 > 0) != (t4 > 0)


def intersect(xa, ya, xb, yb, xc, yc, xd, yd):
    if intersect_prop(xa, ya, xb, yb, xc, yc, xd, yd):
        return True
    if is_between(xa, ya, xb, yb, xc, yc):
        return True
    if is_between(xa, ya, xb, yb, xd, yd):
        return True
    if is_between(xc, yc, xd, yd, xa, ya):
        return True
    if is_between(xc, yc, xd, yd, xb, yb):
        return True
    return False


def in_cone(im1, ip1, n, prev_node, next_node, x, y):
    im2 = prev_node[im1]
    i = next_node[im1]
    t1 = triangle_area_signed(x[im1], y[im1], x[i], y[i], x[im2], y[im2])
    if t1 >= 0.0:
        t2 = triangle_area_signed(x[im1], y[im1], x[ip1], y[ip1], x[im2], y[im2])
        t3 = triangle_area_signed(x[ip1], y[ip1], x[im1], y[im1], x[i], y[i])
        return t2 > 0.0 and t3 > 0.0
    else:
        t4 = triangle_area_signed(x[im1], y[im1], x[ip1], y[ip1], x[i], y[i])
        t5 = triangle_area_signed(x[ip1], y[ip1], x[im1], y[im1], x[im2], y[im2])
        return not (t4 >= 0.0 and t5 >= 0.0)


def diagonalie(im1, ip1, n, next_node, x, y):
    first = im1
    j = first
    jp1 = next_node[first]
    while True:
        if j != im1 and j != ip1 and jp1 != im1 and jp1 != ip1:
            if intersect(x[im1], y[im1], x[ip1], y[ip1], x[j], y[j], x[jp1], y[jp1]):
                return False
        j = jp1
        jp1 = next_node[j]
        if j == first:
            break
    return True


def is_diagonal(im1, ip1, n, prev_node, next_node, x, y):
    return (in_cone(im1, ip1, n, prev_node, next_node, x, y) and
            in_cone(ip1, im1, n, prev_node, next_node, x, y) and
            diagonalie(im1, ip1, n, next_node, x, y))


def polygon_triangulate(n, x, y):
    angle_tol = 5.7e-05
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices.")

    for i in range(n):
        im1 = (i - 1) % n
        if x[im1] == x[i] and y[im1] == y[i]:
            raise ValueError("Two consecutive nodes are identical.")

    node1 = n - 1
    for node2 in range(n):
        node3 = (node2 + 1) % n
        dx1 = x[node1] - x[node2]
        dy1 = y[node1] - y[node2]
        dx2 = x[node3] - x[node2]
        dy2 = y[node3] - y[node2]
        ang = np.degrees(np.arctan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2))
        if abs(ang) <= angle_tol:
            raise ValueError(f"Polygon has an angle smaller than tolerance at node {node2}.")
        node1 = node2
    area = polygon_area(n, x, y)
    if area <= 0.0:
        raise ValueError("Polygon has zero or negative area; vertices must be CCW.")

    prev_node = np.zeros(n, dtype=int)
    next_node = np.zeros(n, dtype=int)
    prev_node[0] = n - 1
    next_node[0] = 1
    for i in range(1, n - 1):
        prev_node[i] = i - 1
        next_node[i] = i + 1
    prev_node[n - 1] = n - 2
    next_node[n - 1] = 0

    ear = np.zeros(n, dtype=bool)
    for i in range(n):
        ear[i] = is_diagonal(prev_node[i], next_node[i], n, prev_node, next_node, x, y)

    triangles = np.zeros((n - 2, 3), dtype=int)
    triangle_num = 0
    i2 = 0
    while triangle_num < n - 3:
        if ear[i2]:
            i3 = next_node[i2]
            i4 = next_node[i3]
            i1 = prev_node[i2]
            i0 = prev_node[i1]
            next_node[i1] = i3
            prev_node[i3] = i1
            ear[i1] = is_diagonal(i0, i3, n, prev_node, next_node, x, y)
            ear[i3] = is_diagonal(i1, i4, n, prev_node, next_node, x, y)
            triangles[triangle_num, 0] = i3
            triangles[triangle_num, 1] = i1
            triangles[triangle_num, 2] = i2
            triangle_num += 1
        i2 = next_node[i2]

    i3 = next_node[i2]
    i1 = prev_node[i2]
    triangles[triangle_num, 0] = i3
    triangles[triangle_num, 1] = i1
    triangles[triangle_num, 2] = i2
    return triangles


def generate_hollow_fiber_cross_section(n_vertices, inner_r, outer_r):
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    x = outer_r * np.cos(theta)
    y = outer_r * np.sin(theta)
    return x, y


def integrate_flux_over_triangles(triangles, x, y, flux_values):
    total = 0.0
    for tri in triangles:
        i, j, k = tri
        area = abs(triangle_area_signed(x[i], y[i], x[j], y[j], x[k], y[k]))
        avg_flux = (flux_values[i] + flux_values[j] + flux_values[k]) / 3.0
        total += area * avg_flux
    return total
