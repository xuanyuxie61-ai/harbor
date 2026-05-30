
import numpy as np


def generate_disk_triangulation(n_r=8, n_theta=16):
    if n_r < 2 or n_theta < 3:
        raise ValueError("n_r must be >=2 and n_theta >=3")
    
    nodes = []
    boundary_mask = []
    

    nodes.append([0.0, 0.0])
    boundary_mask.append(0)
    
    for i in range(1, n_r + 1):
        r = i / n_r
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])
            boundary_mask.append(1 if i == n_r else 0)
    
    nodes = np.array(nodes, dtype=float)
    boundary_mask = np.array(boundary_mask, dtype=int)
    
    elements = []

    for j in range(n_theta):
        j1 = j
        j2 = (j + 1) % n_theta
        elements.append([0, 1 + j1, 1 + j2])
    

    for i in range(1, n_r):
        base_prev = 1 + (i - 1) * n_theta
        base_curr = 1 + i * n_theta
        for j in range(n_theta):
            j1 = j
            j2 = (j + 1) % n_theta

            elements.append([base_prev + j1, base_curr + j1, base_curr + j2])
            elements.append([base_prev + j1, base_curr + j2, base_prev + j2])
    
    elements = np.array(elements, dtype=int)
    return nodes, elements, boundary_mask


def triangle_area(nodes, elements):
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    area = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return area


def compute_element_quality(nodes, elements):
    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    
    a2 = np.sum((p2 - p3) ** 2, axis=1)
    b2 = np.sum((p1 - p3) ** 2, axis=1)
    c2 = np.sum((p1 - p2) ** 2, axis=1)
    
    area = triangle_area(nodes, elements)
    quality = 4.0 * np.sqrt(3.0) * area / (a2 + b2 + c2 + 1e-15)
    quality = np.clip(quality, 0.0, 1.0)
    return quality


def extract_boundary_edges(elements):
    edge_count = {}
    for tri in elements:
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1
    

    boundary = [e for e, c in edge_count.items() if c == 1]
    if not boundary:
        return np.zeros((0, 2), dtype=int)
    

    boundary = list(boundary)
    ordered = [boundary.pop(0)]
    
    while boundary:
        last = ordered[-1][1]
        found = False
        for i, e in enumerate(boundary):
            if e[0] == last:
                ordered.append(e)
                boundary.pop(i)
                found = True
                break
            elif e[1] == last:
                ordered.append((e[1], e[0]))
                boundary.pop(i)
                found = True
                break
        if not found:

            return np.array(ordered, dtype=int)
    
    return np.array(ordered, dtype=int)


def build_node_adjacency(elements, n_nodes):
    adj = [set() for _ in range(n_nodes)]
    for tri in elements:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = tri[i], tri[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj


def domain_decomposition(nodes, elements, n_parts):
    n_nodes = nodes.shape[0]
    partition = np.zeros(n_nodes, dtype=int)
    
    def rcb(idx, part_id, remaining_parts):
        if remaining_parts <= 1 or len(idx) <= 1:
            partition[idx] = part_id
            return part_id + 1
        

        coords = nodes[idx]
        ranges = np.max(coords, axis=0) - np.min(coords, axis=0)
        split_dim = int(np.argmax(ranges))
        median = np.median(coords[:, split_dim])
        
        left = idx[coords[:, split_dim] < median]
        right = idx[coords[:, split_dim] >= median]
        
        if len(left) == 0 or len(right) == 0:
            partition[idx] = part_id
            return part_id + 1
        
        next_id = rcb(left, part_id, remaining_parts // 2)
        next_id = rcb(right, next_id, remaining_parts - remaining_parts // 2)
        return next_id
    
    rcb(np.arange(n_nodes), 0, n_parts)
    return partition


def compute_interface_nodes(elements, partition):
    n_nodes = len(partition)
    interface = np.zeros(n_nodes, dtype=int)
    
    for tri in elements:
        parts = {partition[tri[i]] for i in range(3)}
        if len(parts) > 1:
            for nid in tri:
                interface[nid] = 1
    return interface
