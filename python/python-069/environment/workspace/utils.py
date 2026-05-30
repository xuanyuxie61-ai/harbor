import numpy as np
from scipy.linalg import lu_factor, lu_solve


def safe_divide(a, b, fill_value=0.0):
    b = np.asarray(b, dtype=float)
    result = np.full_like(np.asarray(a, dtype=float), fill_value, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = np.asarray(a, dtype=float)[mask] / b[mask]
    return result


from scipy.linalg import solve_banded

def banded_solve(a, f, ib):
    n = f.shape[0]
    ab = np.zeros((2 * ib + 1, n), dtype=float)
    for j in range(n):
        for i in range(max(0, j - ib), min(n, j + ib + 1)):
            ab[ib + i - j, j] = a[i - j + 2 * ib, j]

    x = solve_banded((ib, ib), ab, f)
    return x


def check_finite(arr, name="array"):
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def accumulation_index(assign_to, values, weights=None):
    n = int(np.max(assign_to)) + 1 if len(assign_to) > 0 else 0
    if weights is None:
        weights = np.ones_like(values, dtype=float)
    result = np.zeros(n, dtype=float)
    np.add.at(result, assign_to, weights * values)
    return result


def triangle_area_2d(t):
    x1, y1 = t[0, 0], t[1, 0]
    x2, y2 = t[0, 1], t[1, 1]
    x3, y3 = t[0, 2], t[1, 2]
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return max(area, 1e-14)


def reference_to_physical_t3(t3, quad_num, quad_xy):
    xy = np.zeros((2, quad_num), dtype=float)
    for q in range(quad_num):
        xi = quad_xy[0, q]
        eta = quad_xy[1, q]
        xy[0, q] = t3[0, 0] + (t3[0, 1] - t3[0, 0]) * xi + (t3[0, 2] - t3[0, 0]) * eta
        xy[1, q] = t3[1, 0] + (t3[1, 1] - t3[1, 0]) * xi + (t3[1, 2] - t3[1, 0]) * eta
    return xy


def basis_11_t6(t6, i, p):
    if i < 0 or i > 5:
        raise ValueError("Basis index i must be in [0,5]")
    if i <= 2:
        j1 = (i + 1) % 3
        j2 = (i + 2) % 3
        k1 = i + 3
        k2 = (i + 5) % 3 + 3
    else:
        j1 = i - 3
        j2 = (i - 3 + 2) % 3
        k1 = (i - 3 + 1) % 3
        k2 = (i - 3 + 2) % 3

    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    gf = cross(p[0] - t6[0, j1], p[1] - t6[1, j1],
               t6[0, j2] - t6[0, j1], t6[1, j2] - t6[1, j1])
    gn = cross(t6[0, i] - t6[0, j1], t6[1, i] - t6[1, j1],
               t6[0, j2] - t6[0, j1], t6[1, j2] - t6[1, j1])
    hf = cross(p[0] - t6[0, k1], p[1] - t6[1, k1],
               t6[0, k2] - t6[0, k1], t6[1, k2] - t6[1, k1])
    hn = cross(t6[0, i] - t6[0, k1], t6[1, i] - t6[1, k1],
               t6[0, k2] - t6[0, k1], t6[1, k2] - t6[1, k1])

    gn = max(abs(gn), 1e-14) * np.sign(gn) if gn != 0 else 1e-14
    hn = max(abs(hn), 1e-14) * np.sign(hn) if hn != 0 else 1e-14

    bi = (gf / gn) * (hf / hn)
    dbidx = ((t6[1, j2] - t6[1, j1]) / gn) * (hf / hn) + (gf / gn) * ((t6[1, k2] - t6[1, k1]) / hn)
    dbidy = -((t6[0, j2] - t6[0, j1]) / gn) * (hf / hn) - (gf / gn) * ((t6[0, k2] - t6[0, k1]) / hn)
    return bi, dbidx, dbidy


def bandwidth(element_order, element_num, element_node):
    nhba = 0
    for element in range(element_num):
        for local_i in range(element_order):
            global_i = element_node[local_i, element]
            for local_j in range(element_order):
                global_j = element_node[local_j, element]
                nhba = max(nhba, abs(global_j - global_i))
    return nhba
