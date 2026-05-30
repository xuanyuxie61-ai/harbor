
import numpy as np
from typing import Tuple, List, Optional


def generate_cube_tetrahedral_mesh(nx: int = 4, ny: int = 4, nz: int = 4,
                                    xlim: Tuple[float, float] = (0.0, 1.0),
                                    ylim: Tuple[float, float] = (0.0, 1.0),
                                    zlim: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("网格层数必须至少为1")

    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    dz = (zlim[1] - zlim[0]) / nz


    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[idx, 0] = xlim[0] + i * dx
                nodes[idx, 1] = ylim[0] + j * dy
                nodes[idx, 2] = zlim[0] + k * dz
                idx += 1

    def node_id(i: int, j: int, k: int) -> int:
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i




    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = node_id(i, j, k)
                n1 = node_id(i + 1, j, k)
                n2 = node_id(i + 1, j + 1, k)
                n3 = node_id(i, j + 1, k)
                n4 = node_id(i, j, k + 1)
                n5 = node_id(i + 1, j, k + 1)
                n6 = node_id(i + 1, j + 1, k + 1)
                n7 = node_id(i, j + 1, k + 1)


                tets = [
                    [n0, n1, n2, n4],
                    [n1, n2, n4, n5],
                    [n2, n4, n5, n6],

                    [n0, n2, n3, n4],
                    [n2, n3, n4, n6],
                    [n3, n4, n6, n7],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=np.int32)
    return nodes, elements


def tetrahedron_volume(nodes: np.ndarray, element: np.ndarray) -> float:
    x1, x2, x3, x4 = nodes[element[0]], nodes[element[1]], nodes[element[2]], nodes[element[3]]
    mat = np.array([x2 - x1, x3 - x1, x4 - x1], dtype=np.float64)
    vol = np.linalg.det(mat) / 6.0
    return vol


def check_mesh_quality(nodes: np.ndarray, elements: np.ndarray) -> dict:
    vols = []
    negative_count = 0
    for e in elements:
        v = tetrahedron_volume(nodes, e)
        vols.append(v)
        if v <= 0:
            negative_count += 1
    vols = np.array(vols)
    return {
        "min_volume": float(np.min(vols)),
        "max_volume": float(np.max(vols)),
        "mean_volume": float(np.mean(vols)),
        "negative_count": negative_count,
        "total_elements": len(elements),
    }


def refine_tetrahedral_mesh(nodes: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes 必须是 (N, 3) 数组")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements 必须是 (E, 4) 数组")

    n_nodes_old = nodes.shape[0]


    edges = []
    for e in elements:
        edges.append(tuple(sorted((e[0], e[1]))))
        edges.append(tuple(sorted((e[0], e[2]))))
        edges.append(tuple(sorted((e[0], e[3]))))
        edges.append(tuple(sorted((e[1], e[2]))))
        edges.append(tuple(sorted((e[1], e[3]))))
        edges.append(tuple(sorted((e[2], e[3]))))

    unique_edges = []
    edge_map = {}
    for ed in edges:
        if ed not in edge_map:
            edge_map[ed] = len(unique_edges)
            unique_edges.append(ed)

    n_edges = len(unique_edges)
    n_nodes_new = n_nodes_old + n_edges
    n_elements_new = 8 * elements.shape[0]


    new_nodes = np.zeros((n_nodes_new, 3), dtype=np.float64)
    new_nodes[:n_nodes_old, :] = nodes


    for idx, (i, j) in enumerate(unique_edges):
        new_nodes[n_nodes_old + idx, :] = 0.5 * (nodes[i] + nodes[j])




    new_elements = np.zeros((n_elements_new, 4), dtype=np.int32)
    e_count = 0
    for e in elements:
        n0, n1, n2, n3 = e
        m01 = n_nodes_old + edge_map[tuple(sorted((n0, n1)))]
        m02 = n_nodes_old + edge_map[tuple(sorted((n0, n2)))]
        m03 = n_nodes_old + edge_map[tuple(sorted((n0, n3)))]
        m12 = n_nodes_old + edge_map[tuple(sorted((n1, n2)))]
        m13 = n_nodes_old + edge_map[tuple(sorted((n1, n3)))]
        m23 = n_nodes_old + edge_map[tuple(sorted((n2, n3)))]


        subtets = [
            [n0, m01, m02, m03],
            [n1, m01, m12, m13],
            [n2, m02, m12, m23],
            [n3, m03, m13, m23],
            [m01, m02, m03, m13],
            [m01, m02, m12, m13],
            [m02, m03, m13, m23],
            [m02, m12, m13, m23],
        ]
        for st in subtets:
            new_elements[e_count, :] = st
            e_count += 1

    return new_nodes, new_elements


def get_surface_triangles(elements: np.ndarray) -> np.ndarray:
    face_count = {}
    for e in elements:
        faces = [
            tuple(sorted((e[0], e[1], e[2]))),
            tuple(sorted((e[0], e[1], e[3]))),
            tuple(sorted((e[0], e[2], e[3]))),
            tuple(sorted((e[1], e[2], e[3]))),
        ]
        for f in faces:
            face_count[f] = face_count.get(f, 0) + 1

    surface_faces = [f for f, c in face_count.items() if c == 1]
    return np.array(surface_faces, dtype=np.int32)
