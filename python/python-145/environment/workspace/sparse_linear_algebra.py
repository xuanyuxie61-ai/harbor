
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, splu


def st_to_coo(ist, jst, ast, shape):
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    ast = np.asarray(ast, dtype=float)


    if ist.min() == 1 or jst.min() == 1:
        ist = ist - 1
        jst = jst - 1


    ist = np.clip(ist, 0, shape[0] - 1)
    jst = np.clip(jst, 0, shape[1] - 1)

    return sp.coo_matrix((ast, (ist, jst)), shape=shape)


def coo_to_st(mat):
    mat = mat.tocoo()
    return mat.row, mat.col, mat.data


def write_st_file(filename, m, n, nst, ist, jst, ast):
    with open(filename, 'w') as f:
        f.write(f"# ST sparse matrix: {m} x {n}, nnz={nst}\n")
        for k in range(nst):
            f.write(f"{ist[k]+1:8d}  {jst[k]+1:8d}  {ast[k]:20.12e}\n")


def read_st_file(filename):
    rows = []
    with open(filename, 'r') as f:
        header_m, header_n, header_nnz = None, None, None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):

                if 'x' in line and 'nnz' in line:
                    parts = line.replace('#', '').split(',')
                    for part in parts:
                        if 'x' in part and 'nnz' not in part:
                            dim_part = part.strip().split('x')
                            if len(dim_part) == 2:
                                try:
                                    header_m = int(dim_part[0].strip().split()[-1])
                                    header_n = int(dim_part[1].strip().split()[0])
                                except ValueError:
                                    pass
                        elif 'nnz' in part:
                            try:
                                header_nnz = int(part.split('=')[1].strip())
                            except (ValueError, IndexError):
                                pass
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    v = float(parts[2])
                    rows.append((i, j, v))
                except ValueError:
                    continue

    if len(rows) == 0:
        raise ValueError(f"read_st_file: 文件 {filename} 中未找到有效数据")

    ist = np.array([r[0] for r in rows], dtype=int)
    jst = np.array([r[1] for r in rows], dtype=int)
    ast = np.array([r[2] for r in rows], dtype=float)


    if ist.min() == 1 or jst.min() == 1:
        ist = ist - 1
        jst = jst - 1

    nst = len(rows)
    if header_m is not None and header_n is not None:
        m, n = header_m, header_n
    else:
        m = ist.max() + 1
        n = jst.max() + 1

    return m, n, nst, ist, jst, ast


def estimate_bandwidth(A):
    A = A.tocoo()
    if A.nnz == 0:
        return 0
    return int(np.max(np.abs(A.row - A.col)))


def sparse_matvec(A, x):
    x = np.asarray(x, dtype=float)
    if x.shape[0] != A.shape[1]:
        raise ValueError("sparse_matvec: 维度不匹配")
    return A @ x


def solve_sparse_system(A, b, use_lu=False):
    b = np.asarray(b, dtype=float)
    if b.shape[0] != A.shape[1]:
        raise ValueError("solve_sparse_system: 维度不匹配")

    info = {'method': 'spsolve', 'nnz': A.nnz, 'shape': A.shape}

    if use_lu:
        lu = splu(A.tocsc())
        x = lu.solve(b)
        info['method'] = 'splu'
        info['fill_factor'] = lu.nnz / max(1, A.nnz)
    else:
        x = spsolve(A, b)


    if x is not None:
        residual = np.linalg.norm(A @ x - b)
        info['residual'] = residual
    else:
        info['residual'] = np.inf

    return x, info
