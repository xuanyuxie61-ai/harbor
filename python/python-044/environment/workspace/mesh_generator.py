
import numpy as np


def fibonacci_spiral_points(n, radius=1.0, center=(0.0, 0.0)):
    if n < 1:
        return np.zeros((0, 2))
    phi_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    da = 2.0 * np.pi * (phi_ratio - 1.0) / phi_ratio
    dr = radius / max(n - 1, 1)

    pts = np.zeros((n, 2))
    a = 0.0
    r = 0.0
    for i in range(n):
        pts[i, 0] = center[0] + r * np.cos(a)
        pts[i, 1] = center[1] + r * np.sin(a)
        a = (a + da) % (2.0 * np.pi)
        r += dr
    return pts


def chebyshev_zero_nodes(n, a=-1.0, b=1.0):
    if n < 1:
        return np.array([])
    k = np.arange(1, n + 1)
    nodes = np.cos(np.pi * (2.0 * k - 1.0) / (2.0 * n))

    nodes = ((1.0 - nodes) * a + (1.0 + nodes) * b) / 2.0
    return nodes


def r8vec_linspace2(n, a, b):
    if n < 1:
        return np.array([])
    x = np.zeros(n)
    for i in range(1, n + 1):
        x[i - 1] = ((n - i + 1) * a + i * b) / (n + 1)
    return x


def triangle_area(p):
    area = 0.5 * (
        p[0, 0] * (p[1, 1] - p[2, 1])
        + p[1, 0] * (p[2, 1] - p[0, 1])
        + p[2, 0] * (p[0, 1] - p[1, 1])
    )
    return area


def triangle_centroid(p):
    return np.mean(p, axis=0)


def triangle_diameter(p):
    d = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            dd = np.sum((p[i] - p[j]) ** 2)
            if dd > d:
                d = dd
    return np.sqrt(d)


def triangle_quality(p):
    a = np.linalg.norm(p[1] - p[0])
    b = np.linalg.norm(p[2] - p[1])
    c = np.linalg.norm(p[0] - p[2])
    s = 0.5 * (a + b + c)
    area = triangle_area(p)
    if abs(area) < 1e-14:
        return 0.0
    r_in = area / s
    r_circ = a * b * c / (4.0 * area)
    if r_circ < 1e-14:
        return 0.0
    return 2.0 * r_in / r_circ


def generate_structured_triangle_mesh(xmin, xmax, ymin, ymax, nx, ny):
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2.")

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    n_nodes = nx * ny
    nodes = np.zeros((n_nodes, 2))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            nodes[idx, 0] = x[i]
            nodes[idx, 1] = y[j]

    n_elements = 2 * (nx - 1) * (ny - 1)
    elements = np.zeros((n_elements, 3), dtype=int)
    e = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + i
            n3 = (j + 1) * nx + (i + 1)

            elements[e, :] = [n0, n1, n2]
            e += 1

            elements[e, :] = [n1, n3, n2]
            e += 1

    return nodes, elements


def generate_quadratic_nodes(nodes, elements):
    n_elements = elements.shape[0]

    edge_map = {}
    midpoints = []

    def edge_key(na, nb):
        return (min(na, nb), max(na, nb))

    for e in range(n_elements):
        for edge in [(0, 1), (1, 2), (2, 0)]:
            ek = edge_key(elements[e, edge[0]], elements[e, edge[1]])
            if ek not in edge_map:
                mid = 0.5 * (nodes[ek[0]] + nodes[ek[1]])
                edge_map[ek] = len(midpoints)
                midpoints.append(mid)

    n_mid = len(midpoints)
    n_nodes6 = nodes.shape[0] + n_mid
    nodes6 = np.vstack([nodes, np.array(midpoints)])

    elements6 = np.zeros((n_elements, 6), dtype=int)
    for e in range(n_elements):
        v = elements[e, :3]
        elements6[e, 0] = v[0]
        elements6[e, 1] = v[1]
        elements6[e, 2] = v[2]
        elements6[e, 3] = edge_map[edge_key(v[0], v[1])] + nodes.shape[0]
        elements6[e, 4] = edge_map[edge_key(v[1], v[2])] + nodes.shape[0]
        elements6[e, 5] = edge_map[edge_key(v[2], v[0])] + nodes.shape[0]

    return nodes6, elements6


def mesh_quality_metrics(nodes, elements):
    n_elements = elements.shape[0]
    areas = np.zeros(n_elements)
    qualities = np.zeros(n_elements)
    diameters = np.zeros(n_elements)

    for e in range(n_elements):
        p = nodes[elements[e, :3]]
        areas[e] = abs(triangle_area(p))
        qualities[e] = triangle_quality(p)
        diameters[e] = triangle_diameter(p)

    return {
        "n_elements": n_elements,
        "n_nodes": nodes.shape[0],
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
        "area_mean": float(np.mean(areas)),
        "quality_min": float(np.min(qualities)),
        "quality_max": float(np.max(qualities)),
        "quality_mean": float(np.mean(qualities)),
        "diameter_max": float(np.max(diameters)),
    }


def cvt_lloyd_1d(n, it_num, s_num, density_func, init=2):
    if n < 2:
        raise ValueError("n must be at least 2.")


    gc = chebyshev_zero_nodes(n, -1.0, 1.0)

    if init == 1:
        g = np.sort(2.0 * np.random.rand(n) - 1.0)
    elif init == 2:
        g = gc.copy()
    else:
        g = r8vec_linspace2(n, -1.0, 1.0)

    s = np.linspace(-1.0 + 1e-12, 1.0 - 1e-12, s_num)
    mu = density_func(s)
    mu = np.maximum(mu, 1e-14)
    r = mu ** 3

    energy = np.zeros(it_num)
    motion = np.zeros(it_num)
    g_new = np.zeros(n)

    for it in range(it_num):

        gb = np.zeros(n + 1)
        gb[0] = -1.0
        for j in range(1, n):
            gb[j] = 0.5 * (g[j - 1] + g[j])
        gb[n] = 1.0


        left = np.searchsorted(s, gb, side="right") - 1
        left = np.clip(left, 0, s_num - 1)

        e_it = 0.0
        for j in range(n):
            k1 = left[j]
            k2 = left[j + 1]
            if k2 <= k1:
                g_new[j] = g[j]
            else:
                r_sum = np.sum(r[k1:k2])
                if r_sum > 1e-30:
                    g_new[j] = np.sum(r[k1:k2] * s[k1:k2]) / r_sum
                else:
                    g_new[j] = g[j]
                for k in range(k1, k2):
                    e_it += r[k] * (s[k] - g[j]) ** 2

        if n % 2 == 1:
            g_new[n // 2] = 0.0

        energy[it] = e_it / s_num
        motion[it] = np.sum((g_new - g) ** 2) / n
        g = g_new.copy()

    return g, energy, motion


def identify_boundary_nodes(nodes, xmin, xmax, ymin, ymax, tol=1e-9):
    bc_dict = {
        "left": np.where(np.abs(nodes[:, 0] - xmin) < tol)[0],
        "right": np.where(np.abs(nodes[:, 0] - xmax) < tol)[0],
        "bottom": np.where(np.abs(nodes[:, 1] - ymin) < tol)[0],
        "top": np.where(np.abs(nodes[:, 1] - ymax) < tol)[0],
    }
    all_bc = np.unique(np.concatenate([
        bc_dict["left"], bc_dict["right"],
        bc_dict["bottom"], bc_dict["top"]
    ]))
    bc_dict["all"] = all_bc
    return bc_dict
