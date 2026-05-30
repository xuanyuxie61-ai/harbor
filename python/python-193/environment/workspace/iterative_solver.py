
import numpy as np
from utils import i4_div_rounded


def task_division(task_number, proc_first=0, proc_last=None):
    if proc_last is None:
        proc_last = task_number - 1
    nproc = proc_last - proc_first + 1
    if nproc <= 0 or task_number <= 0:
        return []

    divisions = []
    task_remain = task_number
    proc_remain = nproc
    task_start = 0

    for p in range(proc_first, proc_last + 1):
        tasks = i4_div_rounded(task_remain, proc_remain)
        task_end = task_start + tasks - 1
        divisions.append((p, task_start, task_end))
        task_start = task_end + 1
        task_remain -= tasks
        proc_remain -= 1

    return divisions


def parallel_matvec(A, x, nproc=4):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    n = A.shape[0]
    y = np.zeros(n)
    divisions = task_division(n, 0, nproc - 1)
    for p, start, end in divisions:
        if start <= end and end < n:
            y[start:end + 1] = A[start:end + 1, :] @ x
    return y


def pcg_solve(A, b, x0=None, M_inv=None, tol=1e-10, max_iter=None, nproc=1):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        b_norm = 1.0


    if nproc > 1:
        r = b - parallel_matvec(A, x, nproc)
    else:
        r = b - A @ x

    if M_inv is not None:
        z = M_inv(r)
    else:
        z = r.copy()

    p = z.copy()
    rz_old = np.dot(r, z)

    residual_history = []
    res_norm = np.linalg.norm(r) / b_norm
    residual_history.append(res_norm)

    if res_norm <= tol:
        return x, {'iterations': 0, 'residual': res_norm, 'converged': True,
                   'history': residual_history}










    raise NotImplementedError("HOLE_2: Implement PCG iteration loop.")


def gmres_solve(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    if max_iter is None:
        max_iter = n
    if restart is None:
        restart = min(20, n)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        b_norm = 1.0

    total_iter = 0
    for _outer in range(max_iter // restart + 1):
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta / b_norm <= tol:
            break

        V = [r / beta]
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta

        for j in range(restart):
            w = A @ V[j]

            for i in range(j + 1):
                H[i, j] = np.dot(w, V[i])
                w = w - H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-15:
                break
            V.append(w / H[j + 1, j])


            H_sub = H[:j + 2, :j + 1]
            y, _, _, _ = np.linalg.lstsq(H_sub, g[:j + 2], rcond=None)
            res = np.linalg.norm(g[:j + 2] - H_sub @ y)
            total_iter += 1
            if res / b_norm <= tol:
                for i in range(j + 1):
                    x = x + y[i] * V[i]
                return x, {'iterations': total_iter, 'residual': res / b_norm,
                           'converged': True}


        j_max = min(j + 1, restart)
        H_sub = H[:j_max + 1, :j_max]
        y, _, _, _ = np.linalg.lstsq(H_sub, g[:j_max + 1], rcond=None)
        for i in range(j_max):
            x = x + y[i] * V[i]

        if total_iter >= max_iter:
            break

    final_res = np.linalg.norm(b - A @ x) / b_norm
    return x, {'iterations': total_iter, 'residual': final_res, 'converged': False}
