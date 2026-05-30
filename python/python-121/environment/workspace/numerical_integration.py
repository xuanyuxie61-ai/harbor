
import numpy as np
from math import sqrt






def quadrilateral_witherden_rule(p):
    if p < 0:
        p = 0
    if p > 21:
        p = 21
    
    n = _rule_order(p)
    
    if p <= 1:
        x, y, w = _rule01()
    elif p <= 3:
        x, y, w = _rule03()
    elif p <= 5:
        x, y, w = _rule05()
    elif p <= 7:
        x, y, w = _rule07()
    elif p <= 9:
        x, y, w = _rule09()
    else:

        x, y, w = _rule09()
    
    return n, np.array(x), np.array(y), np.array(w)


def _rule_order(p):
    orders = [1, 1, 3, 3, 6, 6, 7, 7, 12, 12,
              16, 16, 20, 20, 24, 24, 28, 28, 33, 33, 37, 37]
    return orders[min(p, 21)]


def _rule01():
    x = [0.5]
    y = [0.5]
    w = [1.0]
    return x, y, w


def _rule03():
    x = [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]
    y = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
    w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return x, y, w


def _rule05():
    a = 0.816847572980459
    b = 0.091576213509771
    c = 0.108103018168070
    d = 0.445948490915965
    v = 0.109951743655322
    ww = 0.223381589678011
    
    x = [a, b, b, c, d, d]
    y = [b, a, b, d, c, d]
    w = [v, v, v, ww, ww, ww]
    return x, y, w


def _rule07():
    a = 1.0 / 3.0
    b = (9.0 + 2.0 * sqrt(15.0)) / 21.0
    c = (6.0 - sqrt(15.0)) / 21.0
    d = (9.0 - 2.0 * sqrt(15.0)) / 21.0
    e = (6.0 + sqrt(15.0)) / 21.0
    u = 0.225
    v = (155.0 - sqrt(15.0)) / 1200.0
    ww = (155.0 + sqrt(15.0)) / 1200.0
    
    x = [a, b, c, c, d, e, e]
    y = [a, c, b, c, e, d, e]
    w = [u, v, v, v, ww, ww, ww]
    return x, y, w


def _rule09():
    a = 0.124949503233232
    b = 0.437525248383384
    c = 0.797112651860071
    d = 0.165409927389841
    e = 0.037477420750088
    u = 0.205950504760887
    v = 0.063691414286223
    
    x = [a, b, b, c, c, d, d, e, e, e, e, e]
    y = [b, a, b, d, e, c, e, c, d, e, e, e]

    x = [a, b, b, c, d, d, c, e, e, d, e, e]
    y = [b, a, b, d, c, e, e, c, d, e, d, e]
    w = [u, u, u, v, v, v, v, v, v, v, v, v]

    w = np.array(w)
    w = w / np.sum(w)
    return x, y, w.tolist()


def integrate_2d_quadrilateral(f, precision=7):
    n, x, y, w = quadrilateral_witherden_rule(precision)
    result = 0.0
    for i in range(n):
        result += w[i] * f(x[i], y[i])
    return result


def integrate_2d_mapped(f, vertices, precision=7):
    v = np.asarray(vertices)
    if v.shape != (4, 2):
        return 0.0
    
    n, xi, eta, w = quadrilateral_witherden_rule(precision)
    
    result = 0.0
    for i in range(n):

        xi_i = xi[i]
        eta_i = eta[i]
        x = (1 - xi_i) * (1 - eta_i) * v[0, 0] \
            + xi_i * (1 - eta_i) * v[1, 0] \
            + xi_i * eta_i * v[2, 0] \
            + (1 - xi_i) * eta_i * v[3, 0]
        y = (1 - xi_i) * (1 - eta_i) * v[0, 1] \
            + xi_i * (1 - eta_i) * v[1, 1] \
            + xi_i * eta_i * v[2, 1] \
            + (1 - xi_i) * eta_i * v[3, 1]
        

        dx_dxi = (1 - eta_i) * (v[1, 0] - v[0, 0]) + eta_i * (v[2, 0] - v[3, 0])
        dx_deta = (1 - xi_i) * (v[3, 0] - v[0, 0]) + xi_i * (v[2, 0] - v[1, 0])
        dy_dxi = (1 - eta_i) * (v[1, 1] - v[0, 1]) + eta_i * (v[2, 1] - v[3, 1])
        dy_deta = (1 - xi_i) * (v[3, 1] - v[0, 1]) + xi_i * (v[2, 1] - v[1, 1])
        
        J = abs(dx_dxi * dy_deta - dx_deta * dy_dxi)
        result += w[i] * f(x, y) * J
    
    return result






def monte_carlo_integral_1d(f, a, b, n_samples):
    if n_samples <= 0 or a >= b:
        return 0.0, 0.0
    
    x = np.random.uniform(a, b, n_samples)
    fx = np.array([f(xi) for xi in x])
    
    estimate = (b - a) * np.mean(fx)
    std_error = (b - a) * np.std(fx, ddof=1) / sqrt(n_samples)
    
    return estimate, std_error


def monte_carlo_integral_2d(f, x_range, y_range, n_samples):
    x0, x1 = x_range
    y0, y1 = y_range
    
    if n_samples <= 0 or x0 >= x1 or y0 >= y1:
        return 0.0, 0.0
    
    x = np.random.uniform(x0, x1, n_samples)
    y = np.random.uniform(y0, y1, n_samples)
    
    fx = np.array([f(x[i], y[i]) for i in range(n_samples)])
    
    area = (x1 - x0) * (y1 - y0)
    estimate = area * np.mean(fx)
    std_error = area * np.std(fx, ddof=1) / sqrt(n_samples)
    
    return estimate, std_error


def compute_monomial_integral_moments(f, domain, max_order, n_samples=10000):
    x0, x1 = domain[0]
    y0, y1 = domain[1]
    area = (x1 - x0) * (y1 - y0)
    
    x = np.random.uniform(x0, x1, n_samples)
    y = np.random.uniform(y0, y1, n_samples)
    
    moments = np.zeros((max_order + 1, max_order + 1))
    
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            vals = np.array([f(x[k], y[k]) * (x[k] ** i) * (y[k] ** j)
                             for k in range(n_samples)])
            moments[i, j] = area * np.mean(vals)
    
    return moments
