
import numpy as np






def r8utt_det(n, a):
    if n < 1:
        raise ValueError("矩阵阶数必须 ≥ 1")
    a = np.asarray(a)
    if len(a) < n:
        raise ValueError("数组长度不足")
    return float(a[0] ** n)


def r8utt_solve(n, a, b):
    if n < 1:
        raise ValueError("矩阵阶数必须 ≥ 1")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < n or len(b) < n:
        raise ValueError("数组长度不足")
    if abs(a[0]) < 1e-15:
        raise ValueError("对角元 a[0] 为零，矩阵奇异")

    x = b.copy()[:n]
    for j in range(n - 1, -1, -1):
        x[j] /= a[0]
        for i in range(j):
            x[i] -= a[j - i] * x[j]
    return x






def read_matrix_market(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()


    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith('%'):
        idx += 1

    if idx >= len(lines):
        raise ValueError("文件为空或格式错误")

    header = lines[idx].strip().split()
    if len(header) < 3:
        raise ValueError("Matrix Market 头格式错误")

    M, N, nnz = int(header[0]), int(header[1]), int(header[2])
    A = np.zeros((M, N), dtype=float)

    for line in lines[idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        if 0 <= i < M and 0 <= j < N:
            A[i, j] = val

    info = {"rows": M, "cols": N, "nnz": nnz, "format": "coordinate"}
    return A, info


def write_matrix_market(filename, A, title="Matrix"):
    A = np.asarray(A, dtype=float)
    M, N = A.shape
    nnz = np.count_nonzero(A)
    with open(filename, 'w') as f:
        f.write(f"%%MatrixMarket matrix coordinate real general\n")
        f.write(f"%{title}\n")
        f.write(f"{M} {N} {nnz}\n")
        for i in range(M):
            for j in range(N):
                if abs(A[i, j]) > 1e-15:
                    f.write(f"{i + 1} {j + 1} {A[i, j]:.16e}\n")






def matrix_to_hb_format(A, title="Title", key="Key", ifmt=8, job=2):
    A = np.asarray(A, dtype=float)
    nrow, ncol = A.shape


    col_ptr = [1]
    row_ind = []
    values = []
    for j in range(ncol):
        for i in range(nrow):
            if abs(A[i, j]) > 1e-15:
                row_ind.append(i + 1)
                values.append(A[i, j])
        col_ptr.append(len(row_ind) + 1)

    nnzero = len(values)
    neltvl = 0


    ptr_per_line = 8
    ind_per_line = 8
    val_per_line = 4
    ptrcrd = (len(col_ptr) + ptr_per_line - 1) // ptr_per_line
    indcrd = (len(row_ind) + ind_per_line - 1) // ind_per_line
    valcrd = (len(values) + val_per_line - 1) // val_per_line
    rhscrd = 0
    totcrd = ptrcrd + indcrd + valcrd + rhscrd

    lines = []
    lines.append(f"{title[:72]:72}{key[:8]:8}")
    lines.append(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}")
    lines.append(f"{'RUA':3}{nrow:14d}{ncol:14d}{nnzero:14d}{neltvl:14d}")
    lines.append(f"{'(8I10)':16}{'(8I10)':16}{'(4E20.13)':20}{'(4E20.13)':20}")


    for k in range(0, len(col_ptr), ptr_per_line):
        chunk = col_ptr[k:k + ptr_per_line]
        lines.append("".join(f"{v:10d}" for v in chunk))


    for k in range(0, len(row_ind), ind_per_line):
        chunk = row_ind[k:k + ind_per_line]
        lines.append("".join(f"{v:10d}" for v in chunk))


    for k in range(0, len(values), val_per_line):
        chunk = values[k:k + val_per_line]
        lines.append("".join(f"{v:20.13e}" for v in chunk))

    return "\n".join(lines) + "\n"


def write_hb_file(filename, A, title="Title", key="Key"):
    hb_text = matrix_to_hb_format(A, title=title, key=key)
    with open(filename, 'w') as f:
        f.write(hb_text)






def assemble_global_stiffness(vertices, triangles):
    from quadrature_engine import assemble_stiffness_triangle

    n_v = vertices.shape[0]
    K = np.zeros((n_v, n_v))

    for tri in triangles:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        K_loc = assemble_stiffness_triangle(v1, v2, v3)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += K_loc[i, j]

    return K


def solve_stiffness_system(K, b, tol=1e-12):
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    r = b - K @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    max_iter = min(n * 10, 5000)
    for it in range(max_iter):
        Ap = K @ p
        alpha = rs_old / (np.dot(p, Ap) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    info = {"iterations": it + 1, "residual": float(np.sqrt(rs_new))}
    return x, info


def condition_number_estimate(A):
    A = np.asarray(A)

    x = np.random.randn(A.shape[1])
    for _ in range(20):
        x = A.T @ (A @ x)
        x /= np.linalg.norm(x)
    sigma_max = np.sqrt(np.linalg.norm(A @ x))


    try:
        A_inv = np.linalg.inv(A + 1e-12 * np.eye(A.shape[0]))
        y = np.random.randn(A.shape[0])
        for _ in range(20):
            y = A_inv.T @ (A_inv @ y)
            y /= np.linalg.norm(y)
        sigma_min = np.sqrt(np.linalg.norm(A_inv @ y))
        if sigma_min < 1e-15:
            sigma_min = 1e-15
        return float(sigma_max / sigma_min)
    except np.linalg.LinAlgError:
        return float('inf')
