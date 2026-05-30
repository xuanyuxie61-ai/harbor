import numpy as np
import math
from constants import TINY




def imtqlx(n, d, e, z):
    d = np.asarray(d, dtype=float).copy()
    e = np.asarray(e, dtype=float)
    z = np.asarray(z, dtype=float)
    
    if n == 1:
        return d, z
    

    T = np.diag(d) + np.diag(e[:n-1], 1) + np.diag(e[:n-1], -1)
    eigvals, eigvecs = np.linalg.eigh(T)
    

    z_out = eigvecs[0, :]
    return eigvals, z_out


def legendre_gauss_rule(n):
    if n < 1:
        return np.array([]), np.array([])
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def jacobi_gauss_rule(n, alpha, beta):
    if n < 1:
        return np.array([]), np.array([])
    ab = alpha + beta
    d = np.zeros(n)
    e = np.zeros(n)
    
    d[0] = (beta - alpha) / (ab + 2.0) if ab + 2.0 != 0.0 else 0.0
    if n > 1:
        e[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta) / ((ab + 3.0) * (ab + 2.0) ** 2))
    
    for i in range(2, n + 1):
        denom1 = 2.0 * i + ab
        d[i - 1] = (beta ** 2 - alpha ** 2) / (denom1 * (denom1 + 2.0)) if denom1 != 0.0 else 0.0
        if i < n:
            num = i * (i + alpha) * (i + beta) * (i + ab)
            den = (denom1 - 1.0) * (denom1 + 1.0) * denom1 ** 2
            e[i - 1] = np.sqrt(max(num / den, 0.0)) if den > 0.0 else 0.0
    

    norm = 2.0 ** (ab + 1.0) * math.gamma(alpha + 1.0) * math.gamma(beta + 1.0) / math.gamma(ab + 2.0)
    z = np.zeros(n)
    z[0] = 1.0
    nodes, weights_vec = imtqlx(n, d, e, z)
    weights = norm * weights_vec ** 2
    return nodes, weights





def composite_trapezoidal(f, a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return h * total


def adaptive_trapezoidal(f, a, b, tol=1.0e-8, max_level=20):
    def recurse(lo, hi, fl, fh, whole, level):
        if level >= max_level:
            return whole
        m = (lo + hi) / 2.0
        fm = f(m)
        left = (m - lo) / 2.0 * (fl + fm)
        right = (hi - m) / 2.0 * (fm + fh)
        delta = left + right - whole
        if abs(delta) < 3.0 * tol * (hi - lo) / (b - a):
            return left + right + delta / 3.0
        return recurse(lo, m, fl, fm, left, level + 1) + recurse(m, hi, fm, fh, right, level + 1)
    
    fa = f(a)
    fb = f(b)
    whole = (b - a) / 2.0 * (fa + fb)
    return recurse(a, b, fa, fb, whole, 0)





def tensor_product_2d(rule_1d_a, rule_1d_b, func, rect_a, rect_b):
    nodes_a, weights_a = rule_1d_a
    nodes_b, weights_b = rule_1d_b
    a1, b1 = rect_a
    a2, b2 = rect_b
    

    scale_a = (b1 - a1) / 2.0
    shift_a = (a1 + b1) / 2.0
    scale_b = (b2 - a2) / 2.0
    shift_b = (a2 + b2) / 2.0
    
    total = 0.0
    for i in range(len(nodes_a)):
        x = scale_a * nodes_a[i] + shift_a
        for j in range(len(nodes_b)):
            y = scale_b * nodes_b[j] + shift_b
            total += weights_a[i] * weights_b[j] * func(x, y)
    
    return total * scale_a * scale_b


def gauss_legendre_2d(n1, n2, func, rect_a, rect_b):
    nodes1, weights1 = legendre_gauss_rule(n1)
    nodes2, weights2 = legendre_gauss_rule(n2)
    return tensor_product_2d((nodes1, weights1), (nodes2, weights2), func, rect_a, rect_b)





def integrate_nested(rules_1d, func, ranges):
    dim = len(rules_1d)
    scales = [(ranges[i][1] - ranges[i][0]) / 2.0 for i in range(dim)]
    shifts = [(ranges[i][0] + ranges[i][1]) / 2.0 for i in range(dim)]
    
    nodes_list = [rules_1d[i][0] for i in range(dim)]
    weights_list = [rules_1d[i][1] for i in range(dim)]
    n_list = [len(nodes_list[i]) for i in range(dim)]
    
    total = 0.0
    
    def recurse(d, point, w_prod):
        nonlocal total
        if d == dim:
            total += w_prod * func(*point)
            return
        for i in range(n_list[d]):
            x = scales[d] * nodes_list[d][i] + shifts[d]
            point[d] = x
            recurse(d + 1, point, w_prod * weights_list[d][i])
    
    point = [0.0] * dim
    recurse(0, point, 1.0)
    
    jacobian = 1.0
    for s in scales:
        jacobian *= s
    return total * jacobian





def breit_wigner(m, m0, gamma):
    denom = (m ** 2 - m0 ** 2) ** 2 + (m0 * gamma) ** 2
    if denom < TINY:
        return 0.0
    return (1.0 / np.pi) * (m0 * gamma) / denom


def integrate_dsigma_dm1dm2(matrix_element_sq, m_higgs, m_z, gamma_z, n_points=16):
    m_ll_min = 0.001
    









    raise NotImplementedError("HOLE 2: 请实现 integrate_dsigma_dm1dm2")






def composite_simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            total += 4.0 * f(x)
        else:
            total += 2.0 * f(x)
    return h * total / 3.0
