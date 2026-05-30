
import numpy as np


def lu_decomposition_pivot(A, eps=1e-12):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("输入矩阵必须是方阵")
    

    M = A.copy()
    P = np.arange(n)
    
    for k in range(n - 1):

        pivot_idx = k + np.argmax(np.abs(M[k:, k]))
        
        if abs(M[pivot_idx, k]) < eps:

            continue
        

        if pivot_idx != k:
            M[[k, pivot_idx], :] = M[[pivot_idx, k], :]
            P[[k, pivot_idx]] = P[[pivot_idx, k]]
        

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k] = factor
            M[i, k + 1:] -= factor * M[k, k + 1:]
    

    L = np.tril(M, -1) + np.eye(n)
    U = np.triu(M)
    
    return L, U, P, True


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            x[i] = 0.0
        else:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x


def solve_linear_system(A, b, eps=1e-12):
    L, U, P, success = lu_decomposition_pivot(A, eps)
    

    b_perm = b[P]
    

    y = forward_substitution(L, b_perm)
    x = backward_substitution(U, y)
    

    residual = np.dot(A, x) - b
    residual_norm = np.linalg.norm(residual)
    
    return x, residual_norm


def sparse_matrix_vector_product(A_data, A_row, A_col, x):
    n = len(A_row) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(A_row[i], A_row[i + 1]):
            y[i] += A_data[j] * x[A_col[j]]
    
    return y


def iterative_refinement(A, b, x0, max_iter=5, eps=1e-12):
    x = x0.copy()
    history = []
    
    L, U, P, _ = lu_decomposition_pivot(A, eps)
    
    for _ in range(max_iter):
        residual = b - np.dot(A, x)
        res_norm = np.linalg.norm(residual)
        history.append(res_norm)
        
        if res_norm < 1e-12:
            break
        

        b_perm = residual[P]
        y = forward_substitution(L, b_perm)
        dx = backward_substitution(U, y)
        
        x = x + dx
    
    return x, history
