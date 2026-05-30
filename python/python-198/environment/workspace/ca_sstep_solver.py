
import numpy as np


def power_basis_arnoldi(A, v0, m, tol=1e-10):
    n = len(v0)
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    beta = np.linalg.norm(v0)
    if beta < 1e-15:
        return V, H
    V[:, 0] = v0 / beta
    
    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < tol:
            break
        V[:, j + 1] = w / H[j + 1, j]
    
    return V, H


def ca_sstep_arnoldi(A, v0, s, m_total, tol=1e-10):
    n = len(v0)
    V = np.zeros((n, m_total + 1))
    H = np.zeros((m_total + 1, m_total))
    
    beta = np.linalg.norm(v0)
    if beta < 1e-15:
        return V, H
    V[:, 0] = v0 / beta
    
    num_blocks = (m_total + s - 1) // s
    current_col = 0
    
    for block in range(num_blocks):

        s_local = min(s, m_total - current_col)
        if s_local <= 0:
            break
        

        block_vecs = np.zeros((n, s_local))
        vec = V[:, current_col].copy()
        for k in range(s_local):
            block_vecs[:, k] = vec
            if k < s_local - 1:
                vec = A @ vec
        

        for k in range(s_local):

            for i in range(current_col):
                ip = np.dot(V[:, i], block_vecs[:, k])
                H[i, current_col] = ip
                block_vecs[:, k] -= ip * V[:, i]
            

            norm = np.linalg.norm(block_vecs[:, k])
            if norm < tol:
                break
            H[current_col + 1, current_col] = norm
            V[:, current_col + 1] = block_vecs[:, k] / norm
            current_col += 1
            
            if current_col >= m_total:
                break
    
    return V, H


def gmres_solve(A, b, restart=20, max_iter=100, tol=1e-8, s_step=1):
    n = len(b)
    x = np.zeros(n)
    
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return x, [0.0], 0
    
    res_history = []
    
    for outer in range(max_iter):
        r = b - A @ x
        r_norm = np.linalg.norm(r)
        res_history.append(r_norm / b_norm)
        
        if r_norm / b_norm < tol:
            break
        
        if s_step == 1:
            V, H = power_basis_arnoldi(A, r, restart)
        else:
            V, H = ca_sstep_arnoldi(A, r, s_step, restart)
        

        beta = r_norm
        e1 = np.zeros(H.shape[0])
        e1[0] = beta
        

        y, residuals, rank, s_vals = np.linalg.lstsq(H, e1, rcond=1e-14)
        

        x += V[:, :len(y)] @ y
    
    return x, res_history, len(res_history)


def apply_pce_ca_gmres(A_pce_blocks, b, n_pce, s_step=4, restart=20, tol=1e-8):
    n = len(b)
    
    def matvec(v):
        return A_pce_blocks @ v
    

    class LinOp:
        def __matmul__(self, v):
            return matvec(v)
    

    A_dense = np.zeros((n, n))
    for (i, j), block in A_pce_blocks.items():
        pass
    

    x, res_hist, iters = gmres_solve(A_pce_blocks, b, restart=restart,
                                       max_iter=50, tol=tol, s_step=s_step)
    return x, res_hist, iters
