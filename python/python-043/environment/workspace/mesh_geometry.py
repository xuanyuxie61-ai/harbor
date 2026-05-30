
import numpy as np
from typing import List, Tuple







def radial_mesh(r_icb: float, r_cmb: float, n: int,
                stretching: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("n must be >= 2")
    if r_icb <= 0.0 or r_cmb <= r_icb:
        raise ValueError("Invalid radial bounds")

    xi = np.linspace(0.0, 1.0, n)
    if stretching == 1.0:
        r = r_icb * (r_cmb / r_icb) ** xi
    else:

        s = stretching
        num = xi + s * xi * (1.0 - xi)
        den = 1.0 + s * 0.25
        r = r_icb + (r_cmb - r_icb) * num / den
    dr = np.diff(r)
    dr = np.append(dr, dr[-1])
    return r, dr






def disk_uniform_samples(n_samples: int, radius: float = 1.0,
                         seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def cvt_disk_uniform(n_generators: int, n_samples: int = 20000,
                     n_iterations: int = 20, radius: float = 1.0,
                     seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    generators = disk_uniform_samples(n_generators, radius, seed=seed)

    for it in range(n_iterations):
        samples = disk_uniform_samples(n_samples, radius, seed=seed + it + 1)


        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)

        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=int)
        for i in range(n_samples):
            gid = nearest[i]
            new_generators[gid] += samples[i]
            counts[gid] += 1

        for g in range(n_generators):
            if counts[g] > 0:
                new_generators[g] /= counts[g]
            else:

                new_generators[g] = disk_uniform_samples(1, radius, seed=seed + 1000 + g)[0]
        generators = new_generators

    return generators






def build_node_adjacency(elements: np.ndarray, n_nodes: int) -> List[List[int]]:
    adj = [set() for _ in range(n_nodes)]
    nelem = elements.shape[0]
    for e in range(nelem):
        n1, n2, n3 = elements[e]
        adj[n1].add(n2)
        adj[n1].add(n3)
        adj[n2].add(n1)
        adj[n2].add(n3)
        adj[n3].add(n1)
        adj[n3].add(n2)
    return [list(s) for s in adj]


def pseudo_peripheral_node(adj: List[List[int]], start: int = 0) -> int:
    n = len(adj)
    visited = [False] * n
    queue = [start]
    visited[start] = True
    last_level = [start]
    while queue:
        next_level = []
        for node in queue:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    next_level.append(nb)
        if not next_level:
            break
        last_level = next_level
        queue = next_level


    min_deg = float('inf')
    best = last_level[0]
    for node in last_level:
        deg = len(adj[node])
        if deg < min_deg:
            min_deg = deg
            best = node
    return best


def reverse_cuthill_mckee(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    visited = [False] * n
    ordering = []

    for start in range(n):
        if visited[start]:
            continue
        root = pseudo_peripheral_node(adj, start)

        cm_order = []
        queue = [root]
        visited[root] = True
        while queue:
            level = queue[:]
            queue = []

            level.sort(key=lambda node: len(adj[node]))
            for node in level:
                cm_order.append(node)
                neighbors = [nb for nb in adj[node] if not visited[nb]]
                neighbors.sort(key=lambda nb: len(adj[nb]))
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
        ordering.extend(cm_order)


    rcm_order = ordering[::-1]
    return np.array(rcm_order, dtype=int)


def compute_bandwidth(elements: np.ndarray) -> int:
    bw = 0
    nelem = elements.shape[0]
    for e in range(nelem):
        n1, n2, n3 = elements[e]
        bw = max(bw, abs(n1 - n2), abs(n1 - n3), abs(n2 - n3))
    return bw + 1


def spherical_surface_nodes(n_theta: int, n_phi: int) -> np.ndarray:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    nodes = []
    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            nodes.append((x, y, z))
    return np.array(nodes, dtype=float)


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices.astype(int)
    except Exception:


        n = points.shape[0]

        nelem = n // 3
        elements = np.zeros((nelem, 3), dtype=int)
        for e in range(nelem):
            elements[e] = [3 * e, 3 * e + 1, 3 * e + 2]
        return elements





def generate_core_mesh(r_icb: float, r_cmb: float,
                       n_radial: int, n_theta: int, n_phi: int) -> dict:
    r, dr = radial_mesh(r_icb, r_cmb, n_radial, stretching=1.5)
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)


    nodes_3d = []
    for ri in r:
        for t in theta:
            for p in phi:
                x = ri * np.sin(t) * np.cos(p)
                y = ri * np.sin(t) * np.sin(p)
                z = ri * np.cos(t)
                nodes_3d.append((x, y, z))
    nodes_3d = np.array(nodes_3d, dtype=float)

    return {
        "r": r,
        "theta": theta,
        "phi": phi,
        "nodes_3d": nodes_3d,
        "dr": dr,
        "n_radial": n_radial,
        "n_theta": n_theta,
        "n_phi": n_phi,
    }





def write_triangle_nodes(filename: str, nodes: np.ndarray):
    n_nodes = nodes.shape[0]
    dim = nodes.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} 0 0\n")
        for i in range(n_nodes):
            line = f"{i+1} " + " ".join(f"{nodes[i, d]:.18e}" for d in range(dim))
            f.write(line + "\n")


def write_triangle_elements(filename: str, elements: np.ndarray):
    n_elem = elements.shape[0]
    order = elements.shape[1]
    with open(filename, 'w') as f:
        f.write(f"{n_elem} {order} 0\n")
        for e in range(n_elem):
            line = f"{e+1} " + " ".join(str(elements[e, j] + 1) for j in range(order))
            f.write(line + "\n")


def read_triangle_nodes(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_nodes = int(header[0])
    dim = int(header[1])
    nodes = np.zeros((n_nodes, dim), dtype=float)
    for i in range(n_nodes):
        parts = lines[i + 1].strip().split()
        for d in range(dim):
            nodes[i, d] = float(parts[d + 1])
    return nodes


def read_triangle_elements(filename: str) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split()
    n_elem = int(header[0])
    order = int(header[1])
    elements = np.zeros((n_elem, order), dtype=int)
    for e in range(n_elem):
        parts = lines[e + 1].strip().split()
        for j in range(order):
            elements[e, j] = int(parts[j + 1]) - 1
    return elements





def _self_test():

    r, dr = radial_mesh(1221e3, 3480e3, 16)
    assert len(r) == 16
    assert r[0] == 1221e3
    assert r[-1] == 3480e3


    pts = cvt_disk_uniform(10, n_samples=5000, n_iterations=5)
    assert pts.shape == (10, 2)
    assert np.all(np.linalg.norm(pts, axis=1) <= 1.0 + 1e-10)


    elements = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=int)
    adj = build_node_adjacency(elements, 5)
    rcm = reverse_cuthill_mckee(adj)
    assert len(rcm) == 5
    bw_before = compute_bandwidth(elements)

    reordered = np.zeros_like(elements)
    inv = np.argsort(rcm)
    for e in range(elements.shape[0]):
        for j in range(3):
            reordered[e, j] = inv[elements[e, j]]
    bw_after = compute_bandwidth(reordered)
    assert bw_after <= bw_before


    mesh = generate_core_mesh(1221e3, 3480e3, 8, 8, 8)
    assert mesh["nodes_3d"].shape[0] == 8 * 8 * 8

    print("mesh_geometry: self-test passed.")


if __name__ == "__main__":
    _self_test()
