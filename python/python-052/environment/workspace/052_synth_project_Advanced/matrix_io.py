
import numpy as np
from scipy.sparse import coo_matrix

def write_matrix_market(filename, A, comment="Ocean QG sparse operator"):
    A_coo = A.tocoo()
    rows, cols, data = A_coo.row + 1, A_coo.col + 1, A_coo.data
    with open(filename, 'w') as f:
        f.write(f"%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% {comment}\n")
        f.write(f"{A.shape[0]} {A.shape[1]} {len(data)}\n")
        for i, j, v in zip(rows, cols, data):
            f.write(f"{i} {j} {v:.12e}\n")

def read_matrix_market(filename):
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
