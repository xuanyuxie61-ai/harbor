"""
Iterative Sparse Linear Solvers with Parallel Task Division.

Integrates:
  - 1196_task_division: load balancing for parallel sparse operations

Scientific formulas:
  Preconditioned Conjugate Gradient (PCG):
    Given SPD matrix A, preconditioner M, RHS b:
    r_0 = b - A*x_0
    z_0 = M^{-1} * r_0
    p_0 = z_0
    For k = 0, 1, 2, ... until convergence:
      alpha_k = (r_k^T * z_k) / (p_k^T * A * p_k)
      x_{k+1} = x_k + alpha_k * p_k
      r_{k+1} = r_k - alpha_k * A * p_k
      z_{k+1} = M^{-1} * r_{k+1}
      beta_k  = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)
      p_{k+1} = z_{k+1} + beta_k * p_k

  Convergence criterion:
      ||r_k||_2 / ||b||_2 <= tol

  Task division for parallel matvec:
      Partition rows [0, n-1] among nproc processors using:
          tasks_per_proc = round(remaining_tasks / remaining_procs)
"""

import numpy as np
from utils import i4_div_rounded


def task_division(task_number, proc_first=0, proc_last=None):
    """
    Divide task_number tasks among processors [proc_first, proc_last].
    Based on seed 1196_task_division.

    Returns list of (proc_id, task_start, task_end) tuples.
    Each processor gets a contiguous block.
    """
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
    """
    Compute y = A*x with simulated parallel task division.
    Each "processor" gets a contiguous row block.
    """
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
    """
    Preconditioned Conjugate Gradient solver.

    Parameters:
      A      : (n, n) SPD matrix (dense or sparse-like)
      b      : (n,) RHS vector
      x0     : initial guess
      M_inv  : preconditioner function M_inv(r) -> z, or None (no preconditioning)
      tol    : relative residual tolerance
      max_iter: maximum iterations (default n)
      nproc  : number of processors for parallel matvec simulation

    Returns:
      x      : solution vector
      info   : dict with {'iterations': k, 'residual': ..., 'converged': bool}
    """
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

    # Initial residual
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

    # HOLE_2: PCG iteration loop is missing.
    # Implement the preconditioned conjugate gradient iterations.
    # Key scientific formulas:
    #   - alpha_k = (r_k^T * z_k) / (p_k^T * A * p_k)
    #   - x_{k+1} = x_k + alpha_k * p_k
    #   - r_{k+1} = r_k - alpha_k * A * p_k
    #   - beta_k  = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)
    #   - p_{k+1} = z_{k+1} + beta_k * p_k
    #   - Convergence: ||r||_2 / ||b||_2 <= tol
    raise NotImplementedError("HOLE_2: Implement PCG iteration loop.")


def gmres_solve(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    """
    GMRES(m) solver for non-symmetric sparse systems.

    Scientific basis:
      Minimize ||b - A*x||_2 over Krylov subspace K_k(A, r0).
      Uses Arnoldi iteration to build orthonormal basis V_k and
      upper Hessenberg matrix H_k.
    """
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
            # Gram-Schmidt orthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[i])
                w = w - H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-15:
                break
            V.append(w / H[j + 1, j])

            # Solve least-squares problem min ||g - H*y||
            H_sub = H[:j + 2, :j + 1]
            y, _, _, _ = np.linalg.lstsq(H_sub, g[:j + 2], rcond=None)
            res = np.linalg.norm(g[:j + 2] - H_sub @ y)
            total_iter += 1
            if res / b_norm <= tol:
                for i in range(j + 1):
                    x = x + y[i] * V[i]
                return x, {'iterations': total_iter, 'residual': res / b_norm,
                           'converged': True}

        # Update solution at restart
        j_max = min(j + 1, restart)
        H_sub = H[:j_max + 1, :j_max]
        y, _, _, _ = np.linalg.lstsq(H_sub, g[:j_max + 1], rcond=None)
        for i in range(j_max):
            x = x + y[i] * V[i]

        if total_iter >= max_iter:
            break

    final_res = np.linalg.norm(b - A @ x) / b_norm
    return x, {'iterations': total_iter, 'residual': final_res, 'converged': False}
