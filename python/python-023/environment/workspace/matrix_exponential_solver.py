#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def matrix_exponential_pade(A):
    n = A.shape[0]
    

    inf_norm = np.linalg.norm(A, ord=np.inf)
    if inf_norm < 1e-30:
        return np.eye(n)
    

    s = max(0, int(np.log2(inf_norm)) + 1)
    

    A_scaled = A / (2.0 ** s)
    

    q = 6
    

    I = np.eye(n)
    
    X = A_scaled.copy()
    c = 0.5
    E = I + c * A_scaled
    D = I - c * A_scaled
    
    p = True
    
    for k in range(2, q + 1):

        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        X = A_scaled @ X
        cX = c * X
        E = E + cX
        if p:
            D = D + cX
        else:
            D = D - cX
        p = not p
    

    try:
        E_result = np.linalg.solve(D, E)
    except np.linalg.LinAlgError:

        E_result = np.linalg.lstsq(D, E, rcond=None)[0]
    

    for _ in range(s):
        E_result = E_result @ E_result
    
    return E_result


def arnoldi_iteration(A, b, m):
    n = len(b)
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    V[:, 0] = b / np.linalg.norm(b)
    
    for j in range(m):
        w = A @ V[:, j]
        
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        
        H[j + 1, j] = np.linalg.norm(w)
        
        if H[j + 1, j] < 1e-30:
            H[j + 1, j] = 1e-30
        
        V[:, j + 1] = w / H[j + 1, j]
    
    return V, H


def expm_krylov(A, v, m=20):
    n = len(v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-30:
        return np.zeros(n)
    
    V, H = arnoldi_iteration(A, v, m)
    

    e1 = np.zeros(m)
    e1[0] = 1.0
    

    exp_H = matrix_exponential_pade(H[:m, :m])
    
    result = v_norm * V[:, :m] @ (exp_H @ e1)
    
    return result


def evolve_diffusion_operator(A, f0, dt, n_steps, use_krylov=False):









    
    n = len(f0)
    f = f0.copy()
    
    return f
