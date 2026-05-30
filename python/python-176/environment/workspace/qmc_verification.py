
import numpy as np



_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
], dtype=int)


def radical_inverse(i, base):
    result = 0.0
    f = 1.0 / base
    while i > 0:
        digit = i % base
        result += f * digit
        i //= base
        f /= base
    return result


def hammersley_sequence(dim, n_points, offset=0):
    if dim < 1:
        raise ValueError("hammersley_sequence: dim 必须 ≥ 1")
    if dim > len(_PRIMES) + 1:
        raise ValueError("hammersley_sequence: 维数过大，超出预计算素数表")

    points = np.zeros((n_points, dim), dtype=float)
    for i in range(n_points):
        idx = i + offset

        points[i, 0] = (idx % n_points) / n_points if n_points > 0 else 0.0

        for j in range(1, dim):
            points[i, j] = radical_inverse(idx, _PRIMES[j - 1])

    return points


def hammersley_ellipse_sample(a, b, n_points, offset=0):
    qmc = hammersley_sequence(2, n_points, offset)
    u = qmc[:, 0]
    v = qmc[:, 1]
    r = np.sqrt(u)
    theta = 2.0 * np.pi * v
    x = a * r * np.cos(theta)
    y = b * r * np.sin(theta)
    return np.column_stack((x, y))


def qmc_integrate_ellipse(f, a, b, n_points):
    samples = hammersley_ellipse_sample(a, b, n_points)
    vals = np.array([f(samples[i, 0], samples[i, 1]) for i in range(n_points)])
    area = np.pi * a * b
    estimate = area * np.mean(vals)
    return estimate


def qmc_integrate_box(f, box, n_points, dim=None):
    if dim is None:
        dim = len(box)
    points = hammersley_sequence(dim, n_points)
    a = np.array([b[0] for b in box], dtype=float)
    b_arr = np.array([b[1] for b in box], dtype=float)
    scale = b_arr - a
    volume = np.prod(scale)


    phys_points = points * scale + a
    vals = np.array([f(phys_points[i]) for i in range(n_points)])
    estimate = volume * np.mean(vals)
    return estimate


def verify_fem_with_qmc(y_fem, nodes, elements, f_integrand, a, b, n_qmc=2000):

    fem_val = 0.0
    for e in elements:
        p = nodes[e]
        area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                         - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))

        xc = np.mean(p[:, 0])
        yc = np.mean(p[:, 1])
        yc_fem = np.mean(y_fem[e])
        fem_val += area * f_integrand(xc, yc, yc_fem)


    def f_wrapper(x, y):

        dists = (nodes[:, 0] - x) ** 2 + (nodes[:, 1] - y) ** 2
        idx = np.argmin(dists)
        return f_integrand(x, y, y_fem[idx])

    qmc_val = qmc_integrate_ellipse(f_wrapper, a, b, n_qmc)
    return qmc_val, fem_val
