
import numpy as np
from typing import List, Tuple, Optional


def read_fem_nodes(filepath: str) -> np.ndarray:
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    coords = [float(p) for p in parts[:3]] if len(parts) >= 3 else [float(parts[0]), float(parts[1]), 0.0]
                    rows.append(coords)
    except FileNotFoundError:

        return np.zeros((0, 3), dtype=np.float64)

    return np.array(rows, dtype=np.float64)


def read_fem_elements(filepath: str) -> np.ndarray:
    rows = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    nodes = [int(p) for p in parts[:3]]
                    rows.append(nodes)
    except FileNotFoundError:
        return np.zeros((0, 3), dtype=np.int64)

    return np.array(rows, dtype=np.int64)


def write_medit_mesh(nodes: np.ndarray,
                     elements: np.ndarray,
                     boundary_nodes: Optional[np.ndarray] = None,
                     filepath: str = "ice_mesh.mesh") -> None:
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if elements.size > 0 and np.min(elements) == 0:
        elements = elements + 1

    n_nodes = len(nodes)
    n_elems = len(elements)

    refs = np.ones(n_nodes, dtype=np.int64)
    if boundary_nodes is not None:
        refs = np.asarray(boundary_nodes, dtype=np.int64)

    with open(filepath, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n")
        f.write("Vertices\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{nodes[i, 0]:.6e} {nodes[i, 1]:.6e} {nodes[i, 2]:.6e} {refs[i]}\n")

        if n_elems > 0:
            f.write("Triangles\n")
            f.write(f"{n_elems}\n")
            for e in elements:
                f.write(f"{e[0]} {e[1]} {e[2]} 1\n")

        f.write("End\n")


def read_medit_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    nodes = []
    elements = []
    mode = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                lower = line.lower()

                if lower == 'vertices':
                    mode = 'vertices_count'
                    continue
                elif lower == 'triangles':
                    mode = 'triangles_count'
                    continue
                elif lower == 'end':
                    mode = None
                    continue

                if mode == 'vertices_count':
                    mode = 'vertices'
                    continue
                elif mode == 'triangles_count':
                    mode = 'triangles'
                    continue

                parts = line.split()
                if mode == 'vertices' and len(parts) >= 3:
                    nodes.append([float(parts[0]), float(parts[1]), float(parts[2])])
                elif mode == 'triangles' and len(parts) >= 3:
                    elements.append([int(parts[0]), int(parts[1]), int(parts[2])])
    except FileNotFoundError:
        pass

    nodes_arr = np.array(nodes, dtype=np.float64)
    elems_arr = np.array(elements, dtype=np.int64)
    if elems_arr.size > 0:
        elems_arr = elems_arr - 1
    return nodes_arr, elems_arr


def coo_to_csc(values: np.ndarray, row_indices: np.ndarray,
               col_indices: np.ndarray, n: int, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nnz = len(values)
    if nnz == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64), np.zeros(m + 1, dtype=np.int64)


    order = np.lexsort((row_indices, col_indices))
    data = values[order]
    rows = row_indices[order]
    cols = col_indices[order]

    col_ptr = np.zeros(m + 1, dtype=np.int64)
    for j in range(m):
        col_ptr[j] = np.searchsorted(cols, j, side='left')
    col_ptr[m] = nnz

    return data, rows, col_ptr


def csc_to_dense(data: np.ndarray, row_ind: np.ndarray,
                 col_ptr: np.ndarray, n: int, m: int) -> np.ndarray:
    A = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        for idx in range(col_ptr[j], col_ptr[j + 1]):
            i = row_ind[idx]
            A[i, j] = data[idx]
    return A


def assemble_ice_stiffness_matrix_2d(nodes: np.ndarray,
                                      elements: np.ndarray,
                                      diffusivity: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]

    n_nodes = len(nodes)
    vals = []
    rows = []
    cols = []

    for e in elements:
        n1, n2, n3 = e
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        x3, y3 = nodes[n3]


        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area < 1e-15:
            continue


        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * area)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * area)


        for i_local in range(3):
            for j_local in range(3):
                gi = e[i_local]
                gj = e[j_local]
                ke_ij = diffusivity * area * (b[i_local] * b[j_local] + c[i_local] * c[j_local])
                vals.append(ke_ij)
                rows.append(gi)
                cols.append(gj)

    return np.array(vals, dtype=np.float64), np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), n_nodes, n_nodes


def write_harwell_boeing_csc(data: np.ndarray, row_ind: np.ndarray,
                              col_ptr: np.ndarray, n: int, m: int,
                              filepath: str = "ice_matrix.hb") -> None:
    nnz = len(data)
    with open(filepath, 'w') as f:
        f.write(f"{n} {m} {nnz}\n")
        f.write(" ".join(str(v) for v in col_ptr) + "\n")
        f.write(" ".join(str(v) for v in row_ind) + "\n")
        f.write(" ".join(f"{v:.12e}" for v in data) + "\n")


def read_harwell_boeing_csc(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return (np.zeros(0), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64), 0, 0)

    header = lines[0].strip().split()
    n, m, nnz = int(header[0]), int(header[1]), int(header[2])
    col_ptr = np.array([int(x) for x in lines[1].strip().split()], dtype=np.int64)
    row_ind = np.array([int(x) for x in lines[2].strip().split()], dtype=np.int64)
    data = np.array([float(x) for x in lines[3].strip().split()], dtype=np.float64)
    return data, row_ind, col_ptr, n, m
