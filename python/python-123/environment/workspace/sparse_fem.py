
import numpy as np
from typing import Tuple


def st_to_ge(nst: int, ist: np.ndarray, jst: np.ndarray,
             ast: np.ndarray) -> np.ndarray:
    if nst < 0:
        raise ValueError("st_to_ge: nst >= 0")
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    ast = np.asarray(ast, dtype=float)

    if ist.shape[0] < nst or jst.shape[0] < nst or ast.shape[0] < nst:
        raise ValueError("st_to_ge: 输入数组长度不足")

    m = int(np.max(ist)) if nst > 0 else 0
    n = int(np.max(jst)) if nst > 0 else 0
    Age = np.zeros((m, n))

    for k in range(nst):
        i = ist[k] - 1
        j = jst[k] - 1
        if 0 <= i < m and 0 <= j < n:
            Age[i, j] += ast[k]

    return Age


def assemble_fem_stiffness_2d(
    nodes: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    N = nodes.shape[0]
    T = triangles.shape[0]

    ist_list = []
    jst_list = []
    ast_list = []

    for t in range(T):
        a, b, c = triangles[t, :]
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        xc, yc = nodes[c]


        area = 0.5 * ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
        area_abs = abs(area)
        if area_abs < 1e-15:
            continue




        vx = np.array([yb - yc, yc - ya, ya - yb])
        vy = np.array([xc - xb, xa - xc, xb - xa])

        for i_loc in range(3):
            for j_loc in range(3):
                val = (vx[i_loc] * vx[j_loc] + vy[i_loc] * vy[j_loc]) / (4.0 * area_abs)
                ist_list.append(int([a, b, c][i_loc]) + 1)
                jst_list.append(int([a, b, c][j_loc]) + 1)
                ast_list.append(val)

    ist = np.array(ist_list, dtype=int)
    jst = np.array(jst_list, dtype=int)
    ast = np.array(ast_list, dtype=float)
    nst = len(ast_list)

    return ist, jst, ast, nst


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, bc_nodes: np.ndarray, bc_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    A_mod = A.copy()
    b_mod = b.copy()

    for idx, node in enumerate(bc_nodes):
        if 0 <= node < A_mod.shape[0]:
            A_mod[node, :] = 0.0
            A_mod[:, node] = 0.0
            A_mod[node, node] = 1.0
            b_mod[node] = bc_values[idx]

    return A_mod, b_mod


def sparse_matrix_vector_product(
    ist: np.ndarray, jst: np.ndarray, ast: np.ndarray,
    nst: int, x: np.ndarray, n_rows: int
) -> np.ndarray:
    y = np.zeros(n_rows)
    for k in range(nst):
        i = ist[k] - 1
        j = jst[k] - 1
        if 0 <= i < n_rows and 0 <= j < x.shape[0]:
            y[i] += ast[k] * x[j]
    return y


def compute_fem_l2_error(
    u_h: np.ndarray, u_exact: np.ndarray,
    nodes: np.ndarray, triangles: np.ndarray
) -> float:
    error_sq = 0.0
    for t in range(triangles.shape[0]):
        a, b, c = triangles[t, :]
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        xc, yc = nodes[c]
        area = 0.5 * abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
        if area < 1e-15:
            continue


        u_centroid = (u_h[a] + u_h[b] + u_h[c]) / 3.0
        x_c = (xa + xb + xc) / 3.0
        y_c = (ya + yb + yc) / 3.0
        u_ex = u_exact(x_c, y_c)
        error_sq += area * (u_centroid - u_ex) ** 2

    return np.sqrt(error_sq)
