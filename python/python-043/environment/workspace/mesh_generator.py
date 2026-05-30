
import numpy as np
from scipy.spatial import Delaunay


def cvt_sphere_uniform(n_points, n_samples=10000, n_iter=20):

    generators = _random_sphere_points(n_points)

    for it in range(n_iter):

        samples = _random_sphere_points(n_samples)


        closest = _nearest_neighbor(samples, generators)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_points)
        for i in range(n_points):
            mask = (closest == i)
            if np.any(mask):
                pts = samples[mask]
                centroid = np.mean(pts, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-15:
                    centroid /= norm
                new_generators[i] = centroid
                counts[i] = np.sum(mask)
            else:
                new_generators[i] = generators[i]
                counts[i] = 1

        generators = new_generators

    return generators


def _random_sphere_points(n):
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def _nearest_neighbor(points, generators):
    dists = np.linalg.norm(points[:, None, :] - generators[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def spherical_shell_mesh(n_radial, n_theta, n_phi,
                          r_icb=0.35, r_cmb=1.0,
                          use_cvt=False, cvt_samples=5000, cvt_iter=15):

    s = np.linspace(0.0, 1.0, n_radial)

    r_levels = r_icb + (r_cmb - r_icb) * (s ** 2)

    theta_levels = np.linspace(0.0, np.pi, n_theta)
    phi_levels = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    nodes_list = []
    for r in r_levels:
        if use_cvt and r > r_icb + 1e-6:

            n_surf = max(50, n_theta * n_phi // 4)
            surf_pts = cvt_sphere_uniform(n_surf, cvt_samples, cvt_iter)
            surf_pts *= r
            nodes_list.append(surf_pts)
        else:

            layer_pts = []
            for theta in theta_levels:
                for phi in phi_levels:
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    layer_pts.append([x, y, z])
            nodes_list.append(np.array(layer_pts))

    nodes = np.vstack(nodes_list)


    if len(nodes) <= 5000:
        elements = Delaunay(nodes).simplices
    else:
        elements = np.array([])

    return nodes, elements, r_levels, theta_levels, phi_levels


def build_adjacency_matrix(nodes, elements):
    n_nodes = len(nodes)
    adjacency = [set() for _ in range(n_nodes)]

    if elements.size == 0:

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(nodes[i] - nodes[j])
                if dist < 0.3:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return adjacency

    for elem in elements:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                a, b = elem[i], elem[j]
                adjacency[a].add(b)
                adjacency[b].add(a)

    return adjacency


def reverse_cuthill_mckee(adjacency):
    n = len(adjacency)
    visited = [False] * n
    permutation = []


    degrees = [len(adjacency[i]) for i in range(n)]

    while len(permutation) < n:

        unvisited_degrees = [(degrees[i], i) for i in range(n) if not visited[i]]
        if not unvisited_degrees:
            break
        _, start = min(unvisited_degrees)

        queue = [start]
        visited[start] = True

        while queue:

            current = queue.pop(0)
            permutation.append(current)

            neighbors = sorted([v for v in adjacency[current] if not visited[v]],
                               key=lambda v: degrees[v])
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)


    permutation = permutation[::-1]
    return permutation


def apply_rcm_permutation(nodes, elements, permutation):
    n = len(nodes)
    inv_perm = [0] * n
    for i, p in enumerate(permutation):
        inv_perm[p] = i

    new_nodes = nodes[permutation]
    new_elements = np.copy(elements)
    for i in range(elements.shape[0]):
        for j in range(elements.shape[1]):
            new_elements[i, j] = inv_perm[elements[i, j]]

    return new_nodes, new_elements, permutation, inv_perm


def write_triangle_node_file(filename, nodes, attributes=None, markers=None):
    n_nodes = len(nodes)
    dim = 3
    n_att = 0 if attributes is None else attributes.shape[1]
    n_marker = 0 if markers is None else 1

    with open(filename, 'w') as f:
        f.write(f"{n_nodes} {dim} {n_att} {n_marker}\n")
        for i, node in enumerate(nodes):
            line = f"{i + 1} {node[0]:.15e} {node[1]:.15e} {node[2]:.15e}"
            if attributes is not None:
                for att in attributes[i]:
                    line += f" {att:.15e}"
            if markers is not None:
                line += f" {int(markers[i])}"
            f.write(line + "\n")


def write_triangle_element_file(filename, elements, attributes=None):
    n_elem = len(elements)
    n_nodes_per_elem = elements.shape[1]
    n_att = 0 if attributes is None else attributes.shape[1]

    with open(filename, 'w') as f:
        f.write(f"{n_elem} {n_nodes_per_elem} {n_att}\n")
        for i, elem in enumerate(elements):
            line = f"{i + 1}"
            for idx in elem:
                line += f" {int(idx) + 1}"
            if attributes is not None:
                for att in attributes[i]:
                    line += f" {att:.15e}"
            f.write(line + "\n")


def read_triangle_node_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()


    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            clean_lines.append(stripped)

    header = clean_lines[0].split()
    n_nodes = int(header[0])
    dim = int(header[1])
    n_att = int(header[2])
    n_marker = int(header[3])

    nodes = np.zeros((n_nodes, dim))
    attributes = np.zeros((n_nodes, n_att)) if n_att > 0 else None
    markers = np.zeros(n_nodes, dtype=int) if n_marker > 0 else None

    for i, line in enumerate(clean_lines[1:1 + n_nodes]):
        parts = line.split()
        for d in range(dim):
            nodes[i, d] = float(parts[1 + d])
        offset = 1 + dim
        if n_att > 0:
            for a in range(n_att):
                attributes[i, a] = float(parts[offset + a])
            offset += n_att
        if n_marker > 0:
            markers[i] = int(parts[offset])

    return nodes, attributes, markers


def read_triangle_element_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            clean_lines.append(stripped)

    header = clean_lines[0].split()
    n_elem = int(header[0])
    n_nodes_per_elem = int(header[1])
    n_att = int(header[2])

    elements = np.zeros((n_elem, n_nodes_per_elem), dtype=int)
    attributes = np.zeros((n_elem, n_att)) if n_att > 0 else None

    for i, line in enumerate(clean_lines[1:1 + n_elem]):
        parts = line.split()
        for j in range(n_nodes_per_elem):
            elements[i, j] = int(parts[1 + j]) - 1
        offset = 1 + n_nodes_per_elem
        if n_att > 0:
            for a in range(n_att):
                attributes[i, a] = float(parts[offset + a])

    return elements, attributes


def estimate_mesh_quality(nodes, elements):
    if elements.size == 0:
        return {"min_dihedral": 0.0, "max_aspect_ratio": 1.0}

    quality_metrics = []
    for elem in elements:
        pts = nodes[elem]
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(pts[i] - pts[j]))
        edges = np.array(edges)
        min_edge = np.min(edges)
        max_edge = np.max(edges)
        aspect = max_edge / (min_edge + 1e-30)
        quality_metrics.append(aspect)

    return {
        "max_aspect_ratio": float(np.max(quality_metrics)),
        "mean_aspect_ratio": float(np.mean(quality_metrics)),
    }
