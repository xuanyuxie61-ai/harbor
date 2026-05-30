
import numpy as np


def build_mesh_graph(element_nodes, n_nodes):
    adjacency = {i: set() for i in range(n_nodes)}

    for elem in element_nodes:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                ni, nj = elem[i], elem[j]
                if 0 <= ni < n_nodes and 0 <= nj < n_nodes:
                    adjacency[ni].add(nj)
                    adjacency[nj].add(ni)

    degrees = np.array([len(adjacency[i]) for i in range(n_nodes)], dtype=int)
    return adjacency, degrees


def element_neighbor_tets(element_nodes):
    n_elem = element_nodes.shape[0]
    neighbors = np.full((n_elem, 4), -1, dtype=int)


    face_to_elem = {}
    for e in range(n_elem):
        elem = element_nodes[e]
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f in faces:
            if f not in face_to_elem:
                face_to_elem[f] = []
            face_to_elem[f].append(e)


    for e in range(n_elem):
        elem = element_nodes[e]
        faces = [
            tuple(sorted((elem[0], elem[1], elem[2]))),
            tuple(sorted((elem[0], elem[1], elem[3]))),
            tuple(sorted((elem[0], elem[2], elem[3]))),
            tuple(sorted((elem[1], elem[2], elem[3]))),
        ]
        for f_idx, f in enumerate(faces):
            elems = face_to_elem[f]
            if len(elems) == 2:
                neighbors[e, f_idx] = elems[0] if elems[1] == e else elems[1]

    return neighbors


def mesh_quality_metrics(nodes, element_nodes):
    from fem_basis import tetrahedron_volume

    volumes = []
    for e in range(element_nodes.shape[0]):
        en = element_nodes[e]
        try:
            vol = tetrahedron_volume(nodes[en])
            volumes.append(vol)
        except ValueError:
            volumes.append(0.0)

    volumes = np.array(volumes)
    valid = volumes > 1e-15

    if not np.any(valid):
        raise ValueError("mesh_quality_metrics: 所有单元体积退化")

    metrics = {
        'min_volume': np.min(volumes[valid]),
        'max_volume': np.max(volumes[valid]),
        'mean_volume': np.mean(volumes[valid]),
        'volume_ratio': np.max(volumes[valid]) / np.min(volumes[valid]),
        'n_degenerate': np.sum(~valid)
    }
    return metrics
