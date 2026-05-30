
import numpy as np
from collections import deque


def build_mesh_adjacency(n_nodes, triangles):
    adj_set = [set() for _ in range(n_nodes)]

    for tri in triangles:
        i, j, k = tri
        adj_set[i].add(j)
        adj_set[i].add(k)
        adj_set[j].add(i)
        adj_set[j].add(k)
        adj_set[k].add(i)
        adj_set[k].add(j)


    for i in range(n_nodes):
        adj_set[i].add(i)

    adj_list = [sorted(list(s)) for s in adj_set]


    adj_row = np.zeros(n_nodes + 1, dtype=int)
    for i in range(n_nodes):
        adj_row[i + 1] = adj_row[i] + len(adj_list[i])

    adj = np.zeros(adj_row[-1], dtype=int)
    idx = 0
    for i in range(n_nodes):
        for j in adj_list[i]:
            adj[idx] = j
            idx += 1

    return adj_list, adj_row, adj


def adj_bandwidth(n_nodes, adj_row, adj):
    band_lo = 0
    band_hi = 0
    for i in range(n_nodes):
        for j_idx in range(adj_row[i], adj_row[i + 1]):
            col = adj[j_idx]
            band_lo = max(band_lo, i - col)
            band_hi = max(band_hi, col - i)
    return band_lo + 1 + band_hi


def rcm_reordering(n_nodes, adj_row, adj):

    degree = np.zeros(n_nodes, dtype=int)
    for i in range(n_nodes):
        degree[i] = adj_row[i + 1] - adj_row[i] - 1

    mask = np.ones(n_nodes, dtype=bool)
    perm = np.zeros(n_nodes, dtype=int)
    perm_inv = np.zeros(n_nodes, dtype=int)
    perm_idx = n_nodes - 1

    while perm_idx >= 0:

        min_deg = n_nodes + 1
        root = -1
        for i in range(n_nodes):
            if mask[i] and degree[i] < min_deg:
                min_deg = degree[i]
                root = i

        if root == -1:
            break


        queue = deque([root])
        mask[root] = False
        bfs_order = [root]

        while queue:
            node = queue.popleft()
            neighbors = []
            for j_idx in range(adj_row[node], adj_row[node + 1]):
                neighbor = adj[j_idx]
                if neighbor != node and mask[neighbor]:
                    neighbors.append(neighbor)


            neighbors.sort(key=lambda x: degree[x])
            for neighbor in neighbors:
                if mask[neighbor]:
                    mask[neighbor] = False
                    queue.append(neighbor)
                    bfs_order.append(neighbor)


        for node in reversed(bfs_order):
            perm[node] = perm_idx
            perm_inv[perm_idx] = node
            perm_idx -= 1

    return perm, perm_inv


def apply_rcm_to_mesh(nodes, triangles, adj_row, adj):
    n_nodes = len(nodes)
    perm, perm_inv = rcm_reordering(n_nodes, adj_row, adj)

    bandwidth_before = adj_bandwidth(n_nodes, adj_row, adj)


    reordered_nodes = nodes[perm_inv]


    reordered_triangles = perm[triangles]


    _, new_adj_row, new_adj = build_mesh_adjacency(n_nodes, reordered_triangles)
    bandwidth_after = adj_bandwidth(n_nodes, new_adj_row, new_adj)

    return reordered_nodes, reordered_triangles, perm, bandwidth_before, bandwidth_after


def graph_distance_from_node(adj_list, source):
    n = len(adj_list)
    distance = np.full(n, -1, dtype=int)
    distance[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor != node and distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance


def graph_is_connected(adj_list):
    n = len(adj_list)
    if n == 0:
        return True

    visited = np.zeros(n, dtype=bool)
    queue = deque([0])
    visited[0] = True
    count = 1

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor != node and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                count += 1

    return count == n


def hits_ranking(n_nodes, adj_row, adj, max_iter=50, tol=1e-6):
    a = np.ones(n_nodes) / np.sqrt(n_nodes)
    h = np.ones(n_nodes) / np.sqrt(n_nodes)


    def adj_multiply(v, transpose=False):
        result = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if i != j:
                    if transpose:
                        result[i] += v[j]
                    else:
                        result[i] += v[j]
        return result

    for _ in range(max_iter):

        a_new = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if j != i:
                    a_new[j] += h[i]

        norm_a = np.linalg.norm(a_new)
        if norm_a > 1e-14:
            a_new /= norm_a


        h_new = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j_idx in range(adj_row[i], adj_row[i + 1]):
                j = adj[j_idx]
                if j != i:
                    h_new[i] += a_new[j]

        norm_h = np.linalg.norm(h_new)
        if norm_h > 1e-14:
            h_new /= norm_h

        if np.linalg.norm(a_new - a) < tol and np.linalg.norm(h_new - h) < tol:
            a, h = a_new, h_new
            break

        a, h = a_new, h_new

    return a, h


def compute_element_adjacency(n_triangles, triangles):
    edge_to_tri = {}
    for t in range(n_triangles):
        for e in range(3):
            i = triangles[t, e]
            j = triangles[t, (e + 1) % 3]
            edge = tuple(sorted([i, j]))
            if edge not in edge_to_tri:
                edge_to_tri[edge] = []
            edge_to_tri[edge].append(t)

    element_neighbors = [[] for _ in range(n_triangles)]
    for edge, tri_list in edge_to_tri.items():
        if len(tri_list) == 2:
            t1, t2 = tri_list
            element_neighbors[t1].append(t2)
            element_neighbors[t2].append(t1)

    return element_neighbors
