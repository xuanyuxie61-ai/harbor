
import numpy as np


def trapezoid_1d(f, a, b, n):
    if n < 1:
        raise ValueError("trapezoid_1d: n 必须 ≥ 1")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = np.atleast_1d(f(x))
    val = 0.5 * fx[0] + 0.5 * fx[-1] + np.sum(fx[1:-1])
    return h * val


def romberg_1d(f, a, b, max_k=6):
    T = np.zeros((max_k + 1, max_k + 1), dtype=float)
    n = 1
    h = b - a
    x = np.array([a, b])
    fx = np.atleast_1d(f(x))
    T[0, 0] = 0.5 * h * (fx[0] + fx[1])

    for k in range(1, max_k + 1):
        n *= 2
        h *= 0.5

        x_new = np.linspace(a + h, b - h, n // 2)
        fx_new = np.atleast_1d(f(x_new))
        T[k, 0] = 0.5 * T[k - 1, 0] + h * np.sum(fx_new)
        for m in range(1, k + 1):
            T[k, m] = (4.0 ** m * T[k, m - 1] - T[k - 1, m - 1]) / (4.0 ** m - 1.0)

    return T[max_k, max_k], T


def triangle_symmetric_rule(degree):
    if degree <= 1:

        n = 1
        w = np.array([0.5])
        xi = np.array([1.0 / 3.0])
        eta = np.array([1.0 / 3.0])
    elif degree == 2:

        n = 3
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        xi = np.array([0.5, 0.5, 0.0])
        eta = np.array([0.5, 0.0, 0.5])
    elif degree == 3:

        n = 4
        w = np.array([-9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
        xi = np.array([1.0 / 3.0, 3.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0])
        eta = np.array([1.0 / 3.0, 1.0 / 5.0, 3.0 / 5.0, 1.0 / 5.0])
    elif degree == 4:

        n = 6
        a1 = 0.445948490915965
        b1 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        w = np.array([w1, w1, w1, w2, w2, w2])
        xi = np.array([a1, 1.0 - 2.0 * a1, a1, b1, 1.0 - 2.0 * b1, b1])
        eta = np.array([a1, a1, 1.0 - 2.0 * a1, b1, b1, 1.0 - 2.0 * b1])
    elif degree >= 5:

        n = 7
        a1 = 0.470142064105115
        b1 = 0.101286507323456
        w1 = 0.066197076394253
        w2 = 0.062969590272413
        w0 = 0.1125
        w = np.array([w0, w1, w1, w1, w2, w2, w2])
        xi = np.array([1.0 / 3.0, a1, 1.0 - 2.0 * a1, a1, b1, 1.0 - 2.0 * b1, b1])
        eta = np.array([1.0 / 3.0, a1, a1, 1.0 - 2.0 * a1, b1, b1, 1.0 - 2.0 * b1])
    else:
        raise ValueError(f"triangle_symmetric_rule: 不支持的精度 degree={degree}")

    return n, w, xi, eta


def integrate_over_triangle(points, f, degree=3):
    p1, p2, p3 = points
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    if area < 1.0e-15:
        return 0.0

    n, w, xi, eta = triangle_symmetric_rule(degree)



    x_phys = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
    y_phys = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

    f_vals = np.atleast_1d(f(x_phys, y_phys))
    return area * np.sum(w * f_vals)


def hexahedron_jaskowiec_rule(precision):
    if precision <= 1:
        n = 1
        x = np.array([0.5])
        y = np.array([0.5])
        z = np.array([0.5])
        w = np.array([1.0])
    elif precision == 3:

        n = 6
        a = 0.5
        b = (5.0 - np.sqrt(5.0)) / 10.0
        c = (5.0 + np.sqrt(5.0)) / 10.0
        ww = 1.0 / 6.0
        x = np.array([a, a, b, c, a, a])
        y = np.array([b, c, a, a, a, a])
        z = np.array([a, a, a, a, b, c])
        w = np.full(n, ww)
    elif precision >= 5:

        n = 14
        a = 0.5
        b = 0.25
        c = 0.75

        x = np.array([b, c, c, b, b, c, c, b, a, a, a, a, a, a])
        y = np.array([b, b, c, c, b, b, c, c, a, a, b, c, a, a])
        z = np.array([b, b, b, b, c, c, c, c, b, c, a, a, a, a])
        w = np.array([0.05] * 8 + [0.1] * 6)
        w = w / np.sum(w)
    else:
        raise ValueError(f"hexahedron_jaskowiec_rule: 不支持的精度 {precision}")

    return n, x, y, z, w


def monte_carlo_nd(f, dim, box, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.array([b[0] for b in box], dtype=float)
    b = np.array([b[1] for b in box], dtype=float)
    volume = np.prod(b - a)

    samples = rng.random((n_samples, dim)) * (b - a) + a
    vals = np.array([f(samples[i]) for i in range(n_samples)])
    estimate = volume * np.mean(vals)
    std_err = volume * np.std(vals, ddof=1) / np.sqrt(n_samples)
    return estimate, std_err


def p5_nd_rule(f, dim, box):
    a = np.array([b[0] for b in box], dtype=float)
    b_arr = np.array([b[1] for b in box], dtype=float)
    scale = 0.5 * (b_arr - a)
    shift = 0.5 * (a + b_arr)
    volume = np.prod(b_arr - a)


    gl_nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gl_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])



    pts = np.zeros((1, 0), dtype=float)
    wts = np.array([1.0], dtype=float)

    for d in range(dim):
        new_pts = []
        new_wts = []
        for i in range(len(wts)):
            for j in range(3):
                pt = np.append(pts[i], gl_nodes[j])
                new_pts.append(pt)
                new_wts.append(wts[i] * gl_weights[j])
        pts = np.array(new_pts)
        wts = np.array(new_wts)



    jac = np.prod(scale)
    total = 0.0
    for pt, wt in zip(pts, wts):
        phys_pt = pt * scale + shift
        total += wt * jac * f(phys_pt)

    return total
