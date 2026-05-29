"""
Sparse Matrix and Unstructured Mesh I/O
=======================================
Derived from seed projects 508_hb_to_mm (Harwell-Boeing to Matrix Market
sparse format conversion) and 570_ice_io (3D unstructured mesh I/O).

Ocean models generate large sparse linear systems for implicit solvers
(e.g., barotropic Poisson equation, streamfunction inversion).
This module provides I/O for:

1. Sparse matrices in Matrix Market coordinate format:
       %%MatrixMarket matrix coordinate real general
       M N NNZ
       i j value

2. Unstructured mesh topology with labeled vertices and elements.

For the barotropic streamfunction ψ, the Poisson equation is:
       ∇·(H ∇ψ) = ζ_b
where H is ocean depth and ζ_b is barotropic vorticity.
Discretized on an unstructured mesh, this yields:
       A_{ij} ψ_j = b_i
with A being a sparse SPD matrix.
"""

import numpy as np
from scipy.sparse import coo_matrix

def write_matrix_market(filename, A, comment="Ocean QG sparse operator"):
    """
    Write a sparse matrix to Matrix Market coordinate format.

    Parameters
    ----------
    filename : str
    A : scipy.sparse matrix
    comment : str
    """
    A_coo = A.tocoo()
    rows, cols, data = A_coo.row + 1, A_coo.col + 1, A_coo.data  # 1-indexed
    with open(filename, 'w') as f:
        f.write(f"%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% {comment}\n")
        f.write(f"{A.shape[0]} {A.shape[1]} {len(data)}\n")
        for i, j, v in zip(rows, cols, data):
            f.write(f"{i} {j} {v:.12e}\n")

def read_matrix_market(filename):
    """
    Read a Matrix Market coordinate file into a COO sparse matrix.
    """
    rows, cols, data = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) == 3:
                rows.append(int(parts[0]) - 1)
                cols.append(int(parts[1]) - 1)
                data.append(float(parts[2]))
    return coo_matrix((data, (rows, cols)))


def write_unstructured_mesh(filename, vertices, triangles, vertex_labels=None):
    """
    Write a 2D triangular unstructured mesh in a simple labelled format.

    Parameters
    ----------
    filename : str
    vertices : ndarray, shape (N_v, 2)
        (x, y) coordinates.
    triangles : ndarray, shape (N_t, 3)
        Vertex indices (0-based).
    vertex_labels : ndarray, shape (N_v,), optional
        Integer labels (e.g., 0=interior, 1=boundary, 2=coast).
    """
    Nv = len(vertices)
    Nt = len(triangles)
    with open(filename, 'w') as f:
        f.write(f"# Unstructured Ocean Mesh\n")
        f.write(f"# N_vertices = {Nv}, N_triangles = {Nt}\n")
        f.write(f"VERTICES {Nv}\n")
        for i in range(Nv):
            line = f"{vertices[i,0]:.12e} {vertices[i,1]:.12e}"
            if vertex_labels is not None:
                line += f" {int(vertex_labels[i])}"
            f.write(line + "\n")
        f.write(f"TRIANGLES {Nt}\n")
        for i in range(Nt):
            f.write(f"{triangles[i,0]} {triangles[i,1]} {triangles[i,2]}\n")

def read_unstructured_mesh(filename):
    """
    Read an unstructured mesh file.

    Returns
    -------
    vertices, triangles, labels
    """
    vertices = []
    triangles = []
    labels = []
    mode = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('VERTICES'):
                mode = 'vertices'
                continue
            elif line.startswith('TRIANGLES'):
                mode = 'triangles'
                continue
            parts = line.split()
            if mode == 'vertices':
                vertices.append([float(parts[0]), float(parts[1])])
                if len(parts) > 2:
                    labels.append(int(parts[2]))
            elif mode == 'triangles':
                triangles.append([int(parts[0]), int(parts[1]), int(parts[2])])
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    labels = np.array(labels) if labels else None
    return vertices, triangles, labels


def build_sparse_laplacian_unstructured(vertices, triangles, areas=None):
    """
    Build the sparse finite-element Laplacian matrix for an unstructured
    triangular mesh using linear (P1) elements.

    Element stiffness matrix for triangle T with vertices (x₁,y₁),(x₂,y₂),(x₃,y₃):
        K_T = (1/(4·A_T)) · [ (y₂−y₃)² + (x₃−x₂)²    ...                ...
                              ...                      ...                ...
                              ...                      ...    (y₁−y₂)² + (x₂−x₁)² ]

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Sparse Laplacian (negative semidefinite).
    """
    Nv = len(vertices)
    row_ind, col_ind, data = [], [], []

    for tri in triangles:
        i, j, k = tri
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        xk, yk = vertices[k]

        A_T = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        if A_T < 1e-14:
            continue

        # Gradients of shape functions (constant per element)
        # ∇N_i = ((y_j - y_k), (x_k - x_j)) / (2*A_T)
        dNix = (yj - yk) / (2.0 * A_T)
        dNiy = (xk - xj) / (2.0 * A_T)
        dNjx = (yk - yi) / (2.0 * A_T)
        dNjy = (xi - xk) / (2.0 * A_T)
        dNkx = (yi - yj) / (2.0 * A_T)
        dNky = (xj - xi) / (2.0 * A_T)

        grads = [(dNix, dNiy), (dNjx, dNjy), (dNkx, dNky)]
        local_idx = [i, j, k]

        for a in range(3):
            for b in range(3):
                val = A_T * (grads[a][0] * grads[b][0] + grads[a][1] * grads[b][1])
                row_ind.append(local_idx[a])
                col_ind.append(local_idx[b])
                data.append(val)

    L = coo_matrix((data, (row_ind, col_ind)), shape=(Nv, Nv)).tocsr()
    return L
