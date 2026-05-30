
import numpy as np


def generate_collocation_grid(tmax, L_domain, nt, nx):
    if tmax <= 0 or L_domain <= 0:
        raise ValueError("tmax and L_domain must be positive")
    if nt < 2 or nx < 2:
        raise ValueError("nt and nx must be >= 2")

    t_grid = np.linspace(0.0, tmax, nt)
    x_grid = np.linspace(0.0, L_domain, nx, endpoint=False)
    T_grid, X_grid = np.meshgrid(t_grid, x_grid, indexing='ij')
    X = np.column_stack([T_grid.ravel(), X_grid.ravel()])
    return X, t_grid, x_grid


def generate_boundary_points(tmax, L_domain, n_bc):
    if n_bc < 2:
        raise ValueError("n_bc must be >= 2")
    t_bc = np.linspace(0.0, tmax, n_bc)
    x0 = np.zeros_like(t_bc)
    xL = np.full_like(t_bc, L_domain)
    X_bc_0 = np.column_stack([t_bc, x0])
    X_bc_L = np.column_stack([t_bc, xL])
    return X_bc_0, X_bc_L


def generate_initial_condition_points(L_domain, nx_ic):
    if nx_ic < 2:
        raise ValueError("nx_ic must be >= 2")
    x_ic = np.linspace(0.0, L_domain, nx_ic, endpoint=False)
    t_ic = np.zeros_like(x_ic)
    return np.column_stack([t_ic, x_ic])


def triangulation_boundary_edges(triangle_nodes):
    if triangle_nodes.ndim != 2 or triangle_nodes.shape[1] != 3:
        raise ValueError("triangle_nodes must have shape (n_tri, 3)")

    n_tri = triangle_nodes.shape[0]
    n_edges = 3 * n_tri


    edges = np.zeros((n_edges, 2), dtype=int)
    edges[0:n_tri, 0] = triangle_nodes[:, 0]
    edges[0:n_tri, 1] = triangle_nodes[:, 1]
    edges[n_tri:2*n_tri, 0] = triangle_nodes[:, 1]
    edges[n_tri:2*n_tri, 1] = triangle_nodes[:, 2]
    edges[2*n_tri:3*n_tri, 0] = triangle_nodes[:, 2]
    edges[2*n_tri:3*n_tri, 1] = triangle_nodes[:, 0]


    e_min = np.minimum(edges[:, 0], edges[:, 1])
    e_max = np.maximum(edges[:, 0], edges[:, 1])
    edges_canonical = np.column_stack([e_min, e_max])


    sort_idx = np.lexsort((edges_canonical[:, 1], edges_canonical[:, 0]))
    edges_sorted = edges_canonical[sort_idx]
    edges_original = edges[sort_idx]


    boundary_edges = []
    i = 0
    while i < n_edges:
        if i == n_edges - 1:
            boundary_edges.append(edges_original[i])
            i += 1
        else:
            if np.array_equal(edges_sorted[i], edges_sorted[i + 1]):
                i += 2
            else:
                boundary_edges.append(edges_original[i])
                i += 1

    return np.array(boundary_edges, dtype=int)


def boundary_edge_to_path(boundary_edges):
    if boundary_edges.ndim != 2 or boundary_edges.shape[1] != 2:
        raise ValueError("boundary_edges must have shape (n, 2)")

    n = boundary_edges.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    path = np.zeros(n, dtype=int)
    edges = boundary_edges.copy()
    path[0] = edges[0, 0]
    current_node = edges[0, 1]
    path[1] = current_node
    used = np.zeros(n, dtype=bool)
    used[0] = True

    for k in range(2, n):
        found = False
        for e in range(n):
            if used[e]:
                continue
            if edges[e, 0] == current_node:
                current_node = edges[e, 1]
                found = True
            elif edges[e, 1] == current_node:
                current_node = edges[e, 0]
                found = True
            if found:
                path[k] = current_node
                used[e] = True
                break
        if not found:
            raise RuntimeError("Boundary edges do not form a closed path")

    if current_node != path[0]:
        raise RuntimeError("Boundary path does not close")

    return path


def find_nearest_neighbors(points_ref, points_query):
    if points_ref.ndim != 2 or points_query.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")
    if points_ref.shape[1] != points_query.shape[1]:
        raise ValueError("Reference and query points must have same dimension")

    nq = points_query.shape[0]
    nearest_idx = np.zeros(nq, dtype=int)
    min_distances = np.zeros(nq)

    for i in range(nq):
        diff = points_ref - points_query[i]
        dists = np.sum(diff ** 2, axis=1)
        nearest_idx[i] = np.argmin(dists)
        min_distances[i] = np.sqrt(dists[nearest_idx[i]])

    return nearest_idx, min_distances


def cluster_points_by_distance(points, threshold):
    if points.ndim != 2:
        raise ValueError("points must be 2D")
    n = points.shape[0]
    assigned = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster_mask = np.zeros(n, dtype=bool)
        cluster_mask[i] = True
        assigned[i] = True
        changed = True
        while changed:
            changed = False
            cluster_indices = np.where(cluster_mask)[0]
            for ci in cluster_indices:
                if not assigned[ci]:
                    continue
                dists = np.sqrt(np.sum((points - points[ci]) ** 2, axis=1))
                new_members = (dists < threshold) & ~assigned
                if np.any(new_members):
                    cluster_mask[new_members] = True
                    assigned[new_members] = True
                    changed = True
        clusters.append(np.where(cluster_mask)[0])

    centers = np.array([points[clist].mean(axis=0) for clist in clusters])
    return clusters, centers
