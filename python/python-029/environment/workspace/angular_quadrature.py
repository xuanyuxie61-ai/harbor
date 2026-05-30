
import numpy as np





_LEBEDEV_RULES = {
    6: {
        'npts': 6,
        'precision': 3,
        'seeds': [
            (0.0, 0.0, 0.1666666666666667, 1),
        ]
    },
    14: {
        'npts': 14,
        'precision': 5,
        'seeds': [
            (0.0, 0.0, 0.0666666666666667, 1),
            (0.0, 0.0, 0.0750000000000000, 3),
        ]
    },
    26: {
        'npts': 26,
        'precision': 7,
        'seeds': [
            (0.0, 0.0, 0.0476190476190476, 1),
            (0.0, 0.0, 0.0380952380952381, 2),
            (0.0, 0.0, 0.0321428571428571, 3),
        ]
    },
    38: {
        'npts': 38,
        'precision': 9,
        'seeds': [
            (0.0, 0.0, 0.0095238095238095, 1),
            (0.0, 0.0, 0.0321428571428571, 3),
            (0.0, 0.0, 0.0285714285714286, 4),
        ]
    },
    50: {
        'npts': 50,
        'precision': 11,
        'seeds': [
            (0.0, 0.0, 0.0214285714285714, 1),
            (0.0, 0.0, 0.0206349206349206, 2),
            (0.0, 0.0, 0.0214285714285714, 3),
            (0.0, 0.0, 0.0238095238095238, 4),
        ]
    },
}


def _gen_oh_symmetry(code, a, b, v):
    x, y, z, w = [], [], [], []

    if code == 1:
        n = 6
        pts = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    elif code == 2:
        n = 12
        s = 1.0 / np.sqrt(2.0)
        a = s
        pts = [(0, a, a), (0, a, -a), (0, -a, a), (0, -a, -a),
               (a, 0, a), (a, 0, -a), (-a, 0, a), (-a, 0, -a),
               (a, a, 0), (a, -a, 0), (-a, a, 0), (-a, -a, 0)]
    elif code == 3:
        n = 8
        s = 1.0 / np.sqrt(3.0)
        a = s
        pts = [(a, a, a), (a, a, -a), (a, -a, a), (a, -a, -a),
               (-a, a, a), (-a, a, -a), (-a, -a, a), (-a, -a, -a)]
    elif code == 4:
        n = 24


        b = 0.5
        a = np.sqrt((1.0 - b * b) / 2.0)
        pts = []
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    pts.extend([
                        (sx * a, sy * a, sz * b),
                        (sx * a, sy * b, sz * a),
                        (sx * b, sy * a, sz * a),
                    ])
    elif code == 5:
        n = 24
        b = np.sqrt(2.0 / 3.0)
        a = np.sqrt(1.0 / 3.0)
        pts = []
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    pts.extend([
                        (sx * a, sy * b, 0),
                        (sx * b, sy * a, 0),
                        (sx * a, 0, sy * b),
                        (sx * b, 0, sy * a),
                        (0, sx * a, sy * b),
                        (0, sx * b, sy * a),
                    ])
    elif code == 6:
        n = 48
        c = np.sqrt(3.0) / 3.0
        b = np.sqrt(3.0) / 3.0
        a = np.sqrt(3.0) / 3.0
        pts = []
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    pts.extend([
                        (sx * a, sy * b, sz * c),
                        (sx * a, sy * c, sz * b),
                        (sx * b, sy * a, sz * c),
                        (sx * b, sy * c, sz * a),
                        (sx * c, sy * a, sz * b),
                        (sx * c, sy * b, sz * a),
                    ])
    else:
        raise ValueError(f"未知的对称性 code: {code}")

    w_per_pt = v / n
    for px, py, pz in pts:
        x.append(px)
        y.append(py)
        z.append(pz)
        w.append(w_per_pt)

    return np.array(x), np.array(y), np.array(z), np.array(w)


def lebedev_rule(order):
    if order not in _LEBEDEV_RULES:
        available = sorted(_LEBEDEV_RULES.keys())

        order = min(available, key=lambda o: abs(o - order))

    rule = _LEBEDEV_RULES[order]
    x_all, y_all, z_all, w_all = [], [], [], []

    for seed in rule['seeds']:
        a, b, v, code = seed
        x, y, z, w = _gen_oh_symmetry(code, a, b, v)
        x_all.append(x)
        y_all.append(y)
        z_all.append(z)
        w_all.append(w)

    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    z = np.concatenate(z_all)
    w = np.concatenate(w_all)


    w = w / np.sum(w)
    return x, y, z, w


def integrate_on_sphere(func, order=26):
    x, y, z, w = lebedev_rule(order)
    f_vals = func(x, y, z)
    f_vals = np.asarray(f_vals, dtype=float)
    return 4.0 * np.pi * np.sum(w * f_vals)


def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.clip(z / np.where(r > 0, r, 1), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def integrate_differential_cross_section(dsigma_func, order=38):
    x, y, z, w = lebedev_rule(order)
    theta, phi = cartesian_to_spherical(x, y, z)
    dsigma = dsigma_func(theta, phi)
    dsigma = np.asarray(dsigma, dtype=float)
    return 4.0 * np.pi * np.sum(w * dsigma)


def compute_angular_momentum_transfer_integral(l1, l2, order=26):
    from scipy.special import sph_harm

    x, y, z, w = lebedev_rule(order)
    theta, phi = cartesian_to_spherical(x, y, z)


    Y1 = sph_harm(0, l1, phi, theta)
    Y2 = sph_harm(0, l2, phi, theta)
    integral = 4.0 * np.pi * np.sum(w * np.conj(Y1) * Y2)
    return integral.real


if __name__ == "__main__":

    for order in [6, 14, 26, 38, 50]:
        x, y, z, w = lebedev_rule(order)
        print(f"Order {order}: npts={len(w)}, sum(w)={np.sum(w):.15f}, max|xi^2+yi^2+zi^2-1|={np.max(np.abs(x**2+y**2+z**2-1)):.2e}")


    val = integrate_on_sphere(lambda x, y, z: np.ones_like(x), order=26)
    print(f"∫ 1 dΩ = {val:.10f} (期望 {4*np.pi:.10f})")


    ortho = compute_angular_momentum_transfer_integral(2, 2, order=38)
    print(f"<Y_2^0|Y_2^0> = {ortho:.10f} (期望 1.0)")
