#!/usr/bin/env python3

import numpy as np


def generate_pemfc_mesh():

    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.3],
        [1.0, 0.0, 0.3],
        [1.0, 1.0, 0.3],
        [0.0, 1.0, 0.3],
    ], dtype=float)


    elements = np.array([
        [0, 1, 3, 4],
        [1, 3, 4, 5],
        [1, 2, 3, 5],
        [2, 3, 5, 6],
        [3, 4, 5, 7],
        [3, 5, 6, 7],
        [4, 5, 7, 6],
        [0, 1, 2, 4],
    ], dtype=int)

    return nodes, elements


def tetrahedron_volume(nodes, tet):
    p0 = nodes[tet[0]]
    p1 = nodes[tet[1]]
    p2 = nodes[tet[2]]
    p3 = nodes[tet[3]]

    M = np.array([
        [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
        [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
        [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]],
    ], dtype=float)

    vol = abs(np.linalg.det(M)) / 6.0
    return vol


def refine_mesh(nodes, elements):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_tets = elements.shape[0]


    edges = []
    for t in range(n_tets):
        tet = elements[t]
        edge_list = [
            (tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
            (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3]),
        ]
        for e in edge_list:
            edges.append(tuple(sorted(e)))


    edges_unique = sorted(list(set(edges)))
    n_edges = len(edges_unique)


    new_nodes = np.zeros((n_nodes + n_edges, 3), dtype=float)
    new_nodes[:n_nodes] = nodes

    edge_to_node = {}
    for k, (i, j) in enumerate(edges_unique):
        new_nodes[n_nodes + k] = 0.5 * (nodes[i] + nodes[j])
        edge_to_node[(i, j)] = n_nodes + k



    new_elements = np.zeros((n_tets * 8, 4), dtype=int)

    for t in range(n_tets):
        tet = elements[t]
        v0, v1, v2, v3 = tet


        n01 = edge_to_node[tuple(sorted((v0, v1)))]
        n02 = edge_to_node[tuple(sorted((v0, v2)))]
        n03 = edge_to_node[tuple(sorted((v0, v3)))]
        n12 = edge_to_node[tuple(sorted((v1, v2)))]
        n13 = edge_to_node[tuple(sorted((v1, v3)))]
        n23 = edge_to_node[tuple(sorted((v2, v3)))]


        sub_tets = [
            [v0, n01, n02, n03],
            [n01, v1, n12, n13],
            [n02, n12, v2, n23],
            [n03, n13, n23, v3],
            [n01, n02, n03, n13],
            [n01, n02, n12, n13],
            [n02, n03, n13, n23],
            [n02, n12, n13, n23],
        ]

        for s in range(8):
            new_elements[8 * t + s] = sub_tets[s]


    valid = []
    for e in range(new_elements.shape[0]):
        vol = tetrahedron_volume(new_nodes, new_elements[e])
        if vol > 1e-14:
            valid.append(e)
    new_elements = new_elements[valid]

    return new_nodes, new_elements


def compute_mesh_quality(nodes, elements):
    vols = np.array([tetrahedron_volume(nodes, elements[e])
                     for e in range(elements.shape[0])])
    return {
        'n_nodes': nodes.shape[0],
        'n_elements': elements.shape[0],
        'min_volume': float(np.min(vols)),
        'max_volume': float(np.max(vols)),
        'mean_volume': float(np.mean(vols)),
    }


if __name__ == '__main__':
    nodes, elements = generate_pemfc_mesh()
    print("Initial:", compute_mesh_quality(nodes, elements))
    nodes_r, elements_r = refine_mesh(nodes, elements)
    print("Refined:", compute_mesh_quality(nodes_r, elements_r))
