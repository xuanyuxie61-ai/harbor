
import numpy as np



_LINE_RULES = {
    1: (np.array([0.0]), np.array([2.0])),
    2: (np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]),
        np.array([1.0, 1.0])),
    3: (np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]),
        np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])),
    4: (np.array([-0.8611363115940526, -0.3399810435848563,
                   0.3399810435848563, 0.8611363115940526]),
        np.array([0.3478548451374538, 0.6521451548625461,
                  0.6521451548625461, 0.3478548451374538])),
    5: (np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640]),
        np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                  0.4786286704993665, 0.2369268850561891])),
}


def line_unit_rule(order):
    if order not in _LINE_RULES:
        raise ValueError(f"Order {order} not supported. Use 1-5.")
    x, w = _LINE_RULES[order]
    return x.copy(), w.copy()


def cube_rule(a, b, order_1d):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    order_1d = np.asarray(order_1d, dtype=np.int64)

    if a.size != 3 or b.size != 3 or order_1d.size != 3:
        raise ValueError("a, b, and order_1d must have length 3.")
    if np.any(order_1d < 1) or np.any(order_1d > 5):
        raise ValueError("Orders must be in [1, 5].")
    if np.any(b <= a):
        raise ValueError("Upper bounds must exceed lower bounds.")


    coords = []
    weights = []
    for dim in range(3):
        o = int(order_1d[dim])
        xi, wi = line_unit_rule(o)

        x_mapped = ((1.0 - xi) * a[dim] + (1.0 + xi) * b[dim]) / 2.0
        w_mapped = wi * (b[dim] - a[dim]) / 2.0
        coords.append(x_mapped)
        weights.append(w_mapped)

    x_vals, y_vals, z_vals = coords
    wx, wy, wz = weights

    n_total = len(x_vals) * len(y_vals) * len(z_vals)
    xyz = np.zeros((n_total, 3))
    w = np.zeros(n_total)

    idx = 0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            for k in range(len(z_vals)):
                xyz[idx] = [x_vals[i], y_vals[j], z_vals[k]]
                w[idx] = wx[i] * wy[j] * wz[k]
                idx += 1

    return w, xyz


def hexahedron_witherden_rule(precision):
    if precision < 0 or precision > 11:
        raise ValueError("Precision must be in [0, 11].")

    if precision <= 1:
        return _rule01()
    elif precision <= 3:
        return _rule03()
    elif precision <= 5:
        return _rule05()
    elif precision <= 7:
        return _rule07()
    elif precision <= 9:
        return _rule09()
    else:
        return _rule11()


def _rule01():
    n = 1
    x = np.array([0.5])
    y = np.array([0.5])
    z = np.array([0.5])
    w = np.array([1.0])
    return n, x, y, z, w


def _rule03():
    xi = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    wi = np.array([1.0, 1.0])

    xi = (xi + 1.0) / 2.0
    wi = wi / 8.0
    n = 8
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x[idx] = xi[i]
                y[idx] = xi[j]
                z[idx] = xi[k]
                w[idx] = wi[i] * wi[j] * wi[k] * 8.0
                idx += 1
    return n, x, y, z, w


def _rule05():
    xi = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    wi = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    xi = (xi + 1.0) / 2.0
    wi = wi / 2.0
    n = 27
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x[idx] = xi[i]
                y[idx] = xi[j]
                z[idx] = xi[k]
                w[idx] = wi[i] * wi[j] * wi[k]
                idx += 1
    return n, x, y, z, w


def _rule07():
    xi = np.array([-0.8611363115940526, -0.3399810435848563,
                    0.3399810435848563, 0.8611363115940526])
    wi = np.array([0.3478548451374538, 0.6521451548625461,
                   0.6521451548625461, 0.3478548451374538])
    xi = (xi + 1.0) / 2.0
    wi = wi / 2.0
    n = 64
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x[idx] = xi[i]
                y[idx] = xi[j]
                z[idx] = xi[k]
                w[idx] = wi[i] * wi[j] * wi[k]
                idx += 1
    return n, x, y, z, w


def _rule09():
    xi = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                    0.5384693101056831, 0.9061798459386640])
    wi = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891])
    xi = (xi + 1.0) / 2.0
    wi = wi / 2.0
    n = 125
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x[idx] = xi[i]
                y[idx] = xi[j]
                z[idx] = xi[k]
                w[idx] = wi[i] * wi[j] * wi[k]
                idx += 1
    return n, x, y, z, w


def _rule11():

    xi = np.array([-0.9324695142031521, -0.6612093864662645,
                   -0.2386191860831969, 0.2386191860831969,
                   0.6612093864662645, 0.9324695142031521])
    wi = np.array([0.1713244923791704, 0.3607615730481386,
                   0.4679139345726910, 0.4679139345726910,
                   0.3607615730481386, 0.1713244923791704])
    xi = (xi + 1.0) / 2.0
    wi = wi / 2.0
    n = 216
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x[idx] = xi[i]
                y[idx] = xi[j]
                z[idx] = xi[k]
                w[idx] = wi[i] * wi[j] * wi[k]
                idx += 1
    return n, x, y, z, w


def integrate_scalar_field_hexahedron(f_func, a, b, order_1d):
    w, xyz = cube_rule(a, b, order_1d)
    integral = 0.0
    for i in range(w.size):
        integral += w[i] * f_func(xyz[i, 0], xyz[i, 1], xyz[i, 2])
    return integral


def integrate_vector_field_hexahedron(f_vec_func, a, b, order_1d):
    w, xyz = cube_rule(a, b, order_1d)
    integral = None
    for i in range(w.size):
        val = np.asarray(f_vec_func(xyz[i, 0], xyz[i, 1], xyz[i, 2]))
        if integral is None:
            integral = np.zeros_like(val, dtype=np.float64)
        integral += w[i] * val
    return integral
