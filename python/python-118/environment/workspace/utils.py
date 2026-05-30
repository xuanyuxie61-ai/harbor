
import numpy as np
from scipy.special import gamma, factorial, spherical_jn, eval_legendre
from scipy.special import roots_legendre, roots_laguerre, roots_jacobi






def bessel_zero_j(n, k, kind=1, tol=1e-12, max_iter=100):
    from scipy.special import jv, yv
    
    n = abs(n)
    

    if k == 1:

        if kind == 1:
            x0 = 0.4116 + 0.99999 * n + 0.6980 * (n + 1) ** 0.3353 + 1.0698 * (n + 1) ** 0.3397
        else:
            x0 = 0.0795 + 0.999998 * n + 0.8904 * (n + 1) ** 0.3354 + 0.0271 * (n + 1) ** 0.3087
    elif k == 2:
        if kind == 1:
            x0 = 1.93395 + 1.00008 * n - 0.80572 * (n + 1) ** 0.4562 + 3.38765 * (n + 1) ** 0.3884
        else:
            x0 = 1.04503 + 1.00002 * n - 0.43792 * (n + 1) ** 0.4348 + 2.70113 * (n + 1) ** 0.3662
    elif k == 3:
        if kind == 1:
            x0 = 5.40771 + 1.00094 * n + 2.66926 * (n + 1) ** 0.4297 - 0.17493 * (n + 1) ** 0.6335
        else:
            x0 = 3.72778 + 1.00035 * n + 2.68567 * (n + 1) ** 0.3982 - 0.11298 * (n + 1) ** 0.6048
    else:


        x2 = bessel_zero_j(n, 2, kind, tol, max_iter)
        x3 = bessel_zero_j(n, 3, kind, tol, max_iter)
        spacing = x3 - x2
        x0 = x3 + (k - 3) * spacing
    

    x = x0
    for _ in range(max_iter):
        if kind == 1:
            fx = jv(n, x)
            fpx = jv(n - 1, x) - n / x * jv(n, x) if x > 1e-10 else 0.0

            if x > 1e-10:
                fpx = 0.5 * (jv(n - 1, x) - jv(n + 1, x))
            else:
                fpx = 0.0
        else:
            fx = yv(n, x)
            if x > 1e-10:
                fpx = 0.5 * (yv(n - 1, x) - yv(n + 1, x))
            else:
                fpx = 0.0
        
        if abs(fpx) < 1e-30:
            break
            
        dx = fx / fpx
        x_new = x - dx
        
        if abs(dx) < tol * max(abs(x), 1.0):
            return x_new
        x = x_new
    
    return x






def laguerre_polynomial(m, n_max, x):
    x = np.asarray(x).reshape(-1)
    m = len(x)
    
    if n_max < 0:
        return np.zeros((m, 0))
    
    v = np.zeros((m, n_max + 1))
    v[:, 0] = 1.0
    
    if n_max == 0:
        return v
    
    v[:, 1] = 1.0 - x
    
    for j in range(2, n_max + 1):
        v[:, j] = ((2.0 * j - 1.0 - x) * v[:, j - 1] - (j - 1.0) * v[:, j - 2]) / j
    
    return v


def generalized_laguerre_integral(expon, alpha):
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1 for generalized Laguerre")
    return gamma(alpha + expon + 1)






def gegenbauer_rule(order, alpha, a, b):
    if alpha <= -1.0:
        raise ValueError("alpha must be > -1")
    if a >= b:
        raise ValueError("require a < b")
    if order < 1:
        raise ValueError("order must be >= 1")
    



    jacobi_alpha = alpha
    jacobi_beta = alpha
    

    xi, wi = roots_jacobi(order, jacobi_alpha, jacobi_beta)
    

    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    w = wi * ((b - a) / 2.0) ** (2.0 * alpha + 1.0)
    
    return x, w






def lagrange_basis_1d(x_nodes, x_eval):
    x_nodes = np.asarray(x_nodes)
    x_eval = np.atleast_1d(x_eval)
    n = len(x_nodes)
    
    if len(np.unique(x_nodes)) < n:
        raise ValueError("Nodes must be distinct")
    
    L = np.ones((len(x_eval), n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = x_nodes[i] - x_nodes[j]
                if abs(denom) < 1e-14:
                    denom = 1e-14
                L[:, i] *= (x_eval - x_nodes[j]) / denom
    
    return L


def lagrange_interp_nd(grid_nodes, values, x_eval):
    d = len(grid_nodes)
    values = np.asarray(values)
    x_eval = np.atleast_2d(x_eval)
    m = x_eval.shape[0]
    
    if x_eval.shape[1] != d:
        raise ValueError("x_eval must have d columns")
    

    basis_1d = []
    for dim in range(d):
        L = lagrange_basis_1d(grid_nodes[dim], x_eval[:, dim])
        basis_1d.append(L)
    

    result = np.zeros(m)
    


    if d == 1:
        result = basis_1d[0] @ values.flatten()
    elif d == 2:
        n1, n2 = values.shape
        for i in range(n1):
            for j in range(n2):
                result += values[i, j] * basis_1d[0][:, i] * basis_1d[1][:, j]
    elif d == 3:
        n1, n2, n3 = values.shape
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    result += values[i, j, k] * basis_1d[0][:, i] * basis_1d[1][:, j] * basis_1d[2][:, k]
    else:

        flat_vals = values.flatten()
        indices = np.array(np.unravel_index(np.arange(len(flat_vals)), values.shape)).T
        for idx, val in enumerate(flat_vals):
            contrib = np.ones(m)
            for dim in range(d):
                contrib *= basis_1d[dim][:, indices[idx, dim]]
            result += val * contrib
    
    return result






def r8vec_uniform_01(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(n)


def r8vec_normal_01(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(n)






def comp_next(n, k, a, more, h, t):
    a = np.asarray(a, dtype=int)
    
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n
        return a, more, h, t
    
    if 1 < t:
        h = 0
    
    h = h + 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1
    
    more = (a[k - 1] != n)
    
    return a, more, h, t


def level_to_order_open(dim_num, level_1d):
    level_1d = np.asarray(level_1d, dtype=int)
    order_1d = np.zeros(dim_num, dtype=int)
    
    for i in range(dim_num):
        if level_1d[i] < 0:
            order_1d[i] = 1
        else:
            order_1d[i] = 2 ** (level_1d[i] + 1) - 1
    
    return order_1d
