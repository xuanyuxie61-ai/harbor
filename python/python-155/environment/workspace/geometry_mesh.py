import numpy as np
from typing import List, Tuple, Optional





def generate_2d_mesh(boundary: np.ndarray, hmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if boundary.shape[0] < 3:
        raise ValueError("Boundary must have at least 3 vertices")

    if not np.allclose(boundary[0], boundary[-1]):
        boundary = np.vstack([boundary, boundary[0]])
    n = boundary.shape[0] - 1

    cx = np.mean(boundary[:n, 0])
    cy = np.mean(boundary[:n, 1])
    centroid = np.array([[cx, cy]])
    nodes = np.vstack([boundary[:n], centroid])
    centroid_idx = n
    elems = []
    for i in range(n):
        j = (i + 1) % n
        elems.append([i, j, centroid_idx])
    elems = np.array(elems, dtype=int)


    if hmax is not None and hmax > 0:
        nodes, elems = refine_mesh(nodes, elems, hmax)
    return nodes, elems


def refine_mesh(nodes: np.ndarray, elems: np.ndarray, hmax: float,
                max_levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(max_levels):

        max_edge_len = 0.0
        for tri in elems:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edge_len = np.linalg.norm(nodes[a] - nodes[b])
                if edge_len > max_edge_len:
                    max_edge_len = edge_len
        if max_edge_len <= hmax:
            break

        new_nodes = list(nodes)
        new_elems = []
        edge_midpoint = {}

        def get_midpoint(i: int, j: int) -> int:
            key = tuple(sorted((i, j)))
            if key not in edge_midpoint:
                mp = 0.5 * (nodes[i] + nodes[j])
                edge_midpoint[key] = len(new_nodes)
                new_nodes.append(mp)
            return edge_midpoint[key]

        for tri in elems:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_elems.append([a, ab, ca])
            new_elems.append([ab, b, bc])
            new_elems.append([ca, bc, c])
            new_elems.append([ab, bc, ca])
        nodes = np.array(new_nodes)
        elems = np.array(new_elems, dtype=int)
    return nodes, elems


def mesh_adjacency(nodes: np.ndarray, elems: np.ndarray) -> List[List[int]]:
    n = nodes.shape[0]
    adj = [set() for _ in range(n)]
    for tri in elems:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            if 0 <= a < n and 0 <= b < n:
                adj[a].add(b)
                adj[b].add(a)
    return [list(s) for s in adj]





def sphere_cubed_grid_point_count(n: int) -> int:
    if n < 1:
        return 8
    return (n + 1) ** 3 - (n - 1) ** 3


def sphere_cubed_grid_line_count(n: int) -> int:
    if n < 1:
        return 12
    return 24 + 12 * (n - 2) + 12 * n * (n - 1)


def cubed_grid_ijk_to_xyz(i: int, j: int, k: int, n: int) -> np.ndarray:
    def coord(idx: int) -> float:
        if idx == 0:
            return -1.0
        elif idx == n:
            return 1.0
        else:
            return np.tan((2.0 * idx - n) * 0.25 * np.pi / n)
    xc = coord(i)
    yc = coord(j)
    zc = coord(k)
    norm = np.sqrt(xc * xc + yc * yc + zc * zc)
    if np.isclose(norm, 0.0):
        return np.array([0.0, 0.0, 1.0])
    return np.array([xc, yc, zc]) / norm


def generate_cubed_sphere_grid(n: int) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    index_map = {}

    def add_point(i: int, j: int, k: int):
        key = (i, j, k)
        if key not in index_map:
            idx = len(points)
            index_map[key] = idx
            points.append(cubed_grid_ijk_to_xyz(i, j, k, n))
        return index_map[key]

    lines = []


    for i in range(n + 1):
        for j in range(n + 1):
            add_point(i, j, 0)

    for i in range(n + 1):
        for j in range(n + 1):
            add_point(i, j, n)

    for k in range(1, n):
        for i in range(n + 1):
            add_point(i, 0, k)
            add_point(i, n, k)
        for j in range(1, n):
            add_point(0, j, k)
            add_point(n, j, k)


    def add_line(idx1: int, idx2: int):
        p1 = points[idx1]
        p2 = points[idx2]
        lines.append(np.stack([p1, p2]))


    for i in range(n):
        for j in range(n + 1):
            add_line(index_map[(i, j, 0)], index_map[(i + 1, j, 0)])
    for j in range(n):
        for i in range(n + 1):
            add_line(index_map[(i, j, 0)], index_map[(i, j + 1, 0)])

    for i in range(n):
        for j in range(n + 1):
            add_line(index_map[(i, j, n)], index_map[(i + 1, j, n)])
    for j in range(n):
        for i in range(n + 1):
            add_line(index_map[(i, j, n)], index_map[(i, j + 1, n)])

    for k in range(n):
        for i in range(n + 1):
            add_line(index_map[(i, 0, k)], index_map[(i, 0, k + 1)])
            add_line(index_map[(i, n, k)], index_map[(i, n, k + 1)])
        for j in range(1, n):
            add_line(index_map[(0, j, k)], index_map[(0, j, k + 1)])
            add_line(index_map[(n, j, k)], index_map[(n, j, k + 1)])

    points = np.array(points)
    lines = np.array(lines) if lines else np.zeros((0, 2, 3))
    return points, lines


def cubed_sphere_adjacency(points: np.ndarray, lines: np.ndarray) -> List[List[int]]:
    n = points.shape[0]
    adj = [set() for _ in range(n)]
    for seg in lines:

        p1, p2 = seg[0], seg[1]
        d1 = np.linalg.norm(points - p1, axis=1)
        d2 = np.linalg.norm(points - p2, axis=1)
        i1 = int(np.argmin(d1))
        i2 = int(np.argmin(d2))
        if i1 != i2:
            adj[i1].add(i2)
            adj[i2].add(i1)
    return [list(s) for s in adj]





def hexagon_unit_vertices() -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6.0
    return np.column_stack([np.cos(angles), np.sin(angles)])


def hexagon_area() -> float:
    return 3.0 * np.sqrt(3.0) / 2.0


def generate_hexagonal_lattice(n_rings: int, spacing: float = 1.0) -> np.ndarray:
    points = []

    for q in range(-n_rings, n_rings + 1):
        r1 = max(-n_rings, -q - n_rings)
        r2 = min(n_rings, -q + n_rings)
        for r in range(r1, r2 + 1):
            x = spacing * (np.sqrt(3.0) * q + np.sqrt(3.0) / 2.0 * r)
            y = spacing * (3.0 / 2.0 * r)
            points.append([x, y])
    return np.array(points)


def hexagonal_adjacency(points: np.ndarray, spacing: float = 1.0,
                        tol: float = 1e-6) -> List[List[int]]:
    n = points.shape[0]
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if np.isclose(d, spacing, atol=tol) or np.isclose(d, spacing * np.sqrt(3.0), atol=tol):

                if np.isclose(d, spacing, atol=tol):
                    adj[i].add(j)
                    adj[j].add(i)
    return [list(s) for s in adj]


def hexagon_stroud_rule1() -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0])
    y = np.array([0.0])
    w = np.array([hexagon_area()])
    return np.column_stack([x, y]), w


def hexagon_stroud_rule2() -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(3.0) / 3.0
    x = np.array([r, -r, -r, r])
    y = np.array([r, r, -r, -r])
    w = np.full(4, hexagon_area() / 4.0)
    return np.column_stack([x, y]), w


def hexagon_stroud_rule3() -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(6.0) / 3.0
    x = np.array([0.0, r, -r / 2.0, -r / 2.0, r / 2.0, r / 2.0, -r])
    y = np.array([0.0, 0.0, r * np.sqrt(3.0) / 2.0, -r * np.sqrt(3.0) / 2.0,
                  r * np.sqrt(3.0) / 2.0, -r * np.sqrt(3.0) / 2.0, 0.0])
    w = np.full(7, hexagon_area() / 7.0)
    return np.column_stack([x, y]), w


def hexagon_stroud_rule4() -> Tuple[np.ndarray, np.ndarray]:
    r1 = np.sqrt(14.0) / 5.0
    r2 = np.sqrt(42.0) / 10.0
    x = np.array([0.0, r1, -r1 / 2.0, -r1 / 2.0, r2, -r2 / 2.0, -r2 / 2.0])
    y = np.array([0.0, 0.0, r1 * np.sqrt(3.0) / 2.0, -r1 * np.sqrt(3.0) / 2.0,
                  0.0, r2 * np.sqrt(3.0) / 2.0, -r2 * np.sqrt(3.0) / 2.0])
    w = np.array([0.5, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0]) * hexagon_area()

    w[5] = w[6] = (hexagon_area() - w[0] - w[1] - w[2] - w[3] - w[4]) / 2.0
    return np.column_stack([x, y]), w


def hexagon_monomial_integral(p: int, q: int) -> float:
    if p < 0 or q < 0:
        return 0.0
    if (p % 2) == 1 or (q % 2) == 1:
        return 0.0



    return _hexagon_moment_steger(p, q)


def _hexagon_moment_steger(p: int, q: int) -> float:
    verts = hexagon_unit_vertices()
    m = verts.shape[0]
    nu = 0.0
    for i in range(m):
        j = (i + 1) % m
        xi, yi = verts[i]
        xj, yj = verts[j]
        cross = xi * yj - xj * yi
        if np.isclose(cross, 0.0):
            continue
        s = 0.0
        for k in range(p + 1):
            for l in range(q + 1):
                s += (comb(p, k) * comb(q, l) *
                      (xi ** k) * (yi ** l) *
                      (xj ** (p - k)) * (yj ** (q - l)))
        nu += cross * s / ((p + q + 2) * (p + q + 1) * comb(p + q, p))
    return nu


def comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result
