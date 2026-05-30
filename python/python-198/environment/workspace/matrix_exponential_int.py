
import numpy as np


def matrix_exponential_pade(A, q=6):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    

    inf_norm = np.linalg.norm(A, ord=np.inf)
    if inf_norm < 1e-15:
        return np.eye(n)
    
    s = max(0, int(np.ceil(np.log2(inf_norm))))
    s = max(0, s + 1)
    A_scaled = A / (2.0 ** s)
    

    I = np.eye(n)
    c = 0.5
    E = I + c * A_scaled
    D = I - c * A_scaled
    X = A_scaled.copy()
    
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
    

    E = np.linalg.solve(D, E)
    

    for _ in range(s):
        E = E @ E
    
    return E


def pce_matrix_exponential_step(A_pce, u, dt):






    raise NotImplementedError("HOLE 2: pce_matrix_exponential_step 待修复")


def pce_matrix_exponential_integrate(A_pce, u0, tf, nt):
    u = np.asarray(u0, dtype=float).copy()
    dt = tf / nt
    n_pce = len(u)
    U = np.zeros((nt + 1, n_pce))
    U[0] = u
    
    for i in range(nt):
        u = pce_matrix_exponential_step(A_pce, u, dt)
        U[i + 1] = u
    
    t = np.linspace(0, tf, nt + 1)
    return t, U
