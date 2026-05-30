
import numpy as np


def r83v_mv(n, a, b, c, x):
    x = np.asarray(x).flatten()
    ax = np.zeros(n)
    if n > 1:
        ax[1:] += a * x[:-1]
    ax += b * x
    if n > 1:
        ax[:-1] += c * x[1:]
    return ax


def r83v_fs(n, a, b, c, rhs):

    cp = np.zeros(n)
    cp[1:] = a.copy()
    dp = np.zeros(n)
    if n > 1:
        dp[:-1] = c.copy()
    dp[-1] = 0.0
    bp = b.copy()
    x = rhs.copy()
    

    bp[0] = b[0]
    if n >= 2:
        dp[0] = c[0] if n > 1 else 0.0
        ep = np.zeros(n)
        ep[0] = 0.0
        ep[-1] = 0.0
        
        for k in range(1, n):

            if abs(bp[k - 1]) <= abs(cp[k]):

                bp[k - 1], cp[k] = cp[k], bp[k - 1]
                dp[k - 1], ep[k] = ep[k], dp[k - 1]
                if k < n - 1:
                    ep[k - 1], dp[k] = dp[k], ep[k - 1]
                x[k - 1], x[k] = x[k], x[k - 1]
            
            if abs(bp[k - 1]) < 1e-30:
                raise ValueError(f"Zero pivot at step k={k}")
            
            t = -cp[k] / bp[k - 1]
            bp[k] = bp[k] + t * dp[k - 1]
            if k < n - 1:
                cp[k + 1] = cp[k + 1] + t * ep[k - 1]
            x[k] = x[k] + t * x[k - 1]
    
    if abs(bp[-1]) < 1e-30:
        raise ValueError("Zero pivot at final step")
    

    x[-1] = x[-1] / bp[-1]
    if n > 1:
        x[-2] = (x[-2] - dp[-2] * x[-1]) / bp[-2]
        for k in range(n - 3, -1, -1):
            x[k] = (x[k] - dp[k] * x[k + 1] - (ep[k] if k < n - 2 else 0.0) * x[k + 2]) / bp[k]
    
    return x


def r83v_cg(n, a, b, c, rhs, x0=None, tol=1e-10, max_iter=None):
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = rhs - r83v_mv(n, a, b, c, x)
    p = r.copy()
    rsold = np.dot(r, r)
    
    for it in range(max_iter):
        Ap = r83v_mv(n, a, b, c, p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            return x, it + 1, np.sqrt(rsnew)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, max_iter, np.sqrt(rsold)


def r83v_jac_sl(n, a, b, c, rhs, x0=None, it_max=100, tol=1e-10):
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    if np.any(np.abs(b) < 1e-30):
        raise ValueError("Zero diagonal entries detected")
    
    x_new = np.zeros(n)
    for it in range(it_max):
        x_new[0] = (rhs[0] - c[0] * x[1]) / b[0]
        if n > 2:
            x_new[1:-1] = (rhs[1:-1] - a[:-1] * x[:-2] - c[1:] * x[2:]) / b[1:-1]
        if n > 1:
            x_new[-1] = (rhs[-1] - a[-1] * x[-2]) / b[-1]
        
        diff = np.linalg.norm(x_new - x)
        x[:] = x_new
        if diff < tol:
            residual = np.linalg.norm(rhs - r83v_mv(n, a, b, c, x))
            return x, it + 1, residual
    
    residual = np.linalg.norm(rhs - r83v_mv(n, a, b, c, x))
    return x, it_max, residual


def build_beam_tridiagonal(n, EI, L, load_type='uniform'):
    h = L / (n + 1)
    


    factor = EI / (h ** 3)
    a = -factor * np.ones(n - 1)
    b = 2.0 * factor * np.ones(n)
    c = -factor * np.ones(n - 1)
    
    if load_type == 'uniform':
        q0 = 1.0
        rhs = q0 * np.ones(n) * h
    elif load_type == 'point_center':
        rhs = np.zeros(n)
        rhs[n // 2] = 1.0
    else:
        rhs = np.zeros(n)
    
    return a, b, c, rhs, h


def modal_analysis_tridiagonal(n, a, b, c, n_modes=3, max_iter=100, tol=1e-8):
    eigenvalues = np.zeros(n_modes)
    eigenvectors = np.zeros((n, n_modes))
    

    sigma = 0.0
    
    for mode in range(n_modes):

        phi = np.sin(np.linspace(0, np.pi, n) * (mode + 1))
        phi = phi / np.linalg.norm(phi)
        
        for it in range(max_iter):

            b_shifted = b.copy() - sigma
            try:
                phi_new = r83v_fs(n, a.copy(), b_shifted, c.copy(), phi)
            except ValueError:

                phi_new, _, _ = r83v_cg(n, a, b_shifted, c, phi, tol=tol * 0.1)
            

            Aphi = r83v_mv(n, a, b, c, phi_new)
            rq = np.dot(phi_new, Aphi) / np.dot(phi_new, phi_new)
            
            phi_new = phi_new / np.linalg.norm(phi_new)
            

            diff = np.linalg.norm(phi_new - phi)
            phi = phi_new
            if diff < tol:
                break
        
        eigenvalues[mode] = rq
        eigenvectors[:, mode] = phi
        

        sigma = rq * 1.1
    
    return eigenvalues, eigenvectors
