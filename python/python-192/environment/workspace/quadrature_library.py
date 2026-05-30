
import numpy as np
from utils_numerical import safe_divide


def hexagon_lyness_rule(rule_id: int) -> tuple:
    rules = {
        1: {
            'n': 6,
            'r': np.sqrt(5.0 / 12.0),
            'w': 1.0 / 6.0,
            'strength': 3
        },
        2: {
            'n': 6,
            'r': 0.6507114129304177,
            'w': 1.0 / 6.0,
            'strength': 5
        },
        3: {
            'n': 12,
            'r1': 0.4620981203732968,
            'r2': 0.799216485305405,
            'w1': 0.1882035356199803,
            'w2': 0.1451297977133530,
            'strength': 7
        }
    }


    if rule_id == 1:
        n = 6
        r = np.sqrt(5.0 / 12.0)
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.ones(n) / 6.0
        for i in range(n):
            angle = 2.0 * np.pi * i / 6.0
            x[i] = r * np.cos(angle)
            y[i] = r * np.sin(angle)
        strength = 3

    elif rule_id == 2:
        n = 6
        r = 0.6507114129304177
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.ones(n) / 6.0
        for i in range(n):
            angle = 2.0 * np.pi * i / 6.0
            x[i] = r * np.cos(angle)
            y[i] = r * np.sin(angle)
        strength = 5

    elif rule_id == 3:

        n = 12
        r1 = 0.4620981203732968
        r2 = 0.799216485305405
        w1 = 0.1882035356199803
        w2 = 0.1451297977133530
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.zeros(n)
        for i in range(6):
            angle = 2.0 * np.pi * i / 6.0
            x[i] = r1 * np.cos(angle)
            y[i] = r1 * np.sin(angle)
            w[i] = w1
            x[i + 6] = r2 * np.cos(angle)
            y[i + 6] = r2 * np.sin(angle)
            w[i + 6] = w2

        w /= np.sum(w)
        strength = 7

    else:

        n = 18
        x = np.zeros(n)
        y = np.zeros(n)
        w = np.ones(n) / n
        radii = [0.3, 0.6, 0.85]
        for ring in range(3):
            for i in range(6):
                idx = ring * 6 + i
                angle = 2.0 * np.pi * i / 6.0 + ring * np.pi / 6.0
                x[idx] = radii[ring] * np.cos(angle)
                y[idx] = radii[ring] * np.sin(angle)
        w /= np.sum(w)
        strength = 5

    return n, x, y, w, strength


def wandzura_triangle_rule(rule_id: int = 1) -> tuple:


    if rule_id == 1:
        degree = 5
        suborders = [
            {'type': 1, 'xi': [1.0/3.0], 'eta': [1.0/3.0], 'w': [0.2250000000000000]},
            {'type': 3, 'xi': [0.0597158717897708, 0.4701420641051151, 0.4701420641051151],
             'eta': [0.4701420641051151, 0.0597158717897708, 0.4701420641051151],
             'w': [0.1323941527885060] * 3},
            {'type': 3, 'xi': [0.7974269853530870, 0.1012865073234564, 0.1012865073234564],
             'eta': [0.1012865073234564, 0.7974269853530870, 0.1012865073234564],
             'w': [0.1259391805448270] * 3}
        ]

    elif rule_id == 2:

        degree = 10
        suborders = [
            {'type': 1, 'xi': [1.0/3.0], 'eta': [1.0/3.0], 'w': [0.0908179903827543]},
            {'type': 3, 'xi': [0.0288447332326857, 0.4855776333836571, 0.4855776333836571],
             'eta': [0.4855776333836571, 0.0288447332326857, 0.4855776333836571],
             'w': [0.0367259577564673] * 3},
            {'type': 3, 'xi': [0.7810368490299922, 0.1094815754850039, 0.1094815754850039],
             'eta': [0.1094815754850039, 0.7810368490299922, 0.1094815754850039],
             'w': [0.0453210594355287] * 3},
            {'type': 6, 'xi': [0.14170721931088, 0.30793983882147, 0.55035294186764,
                               0.55035294186764, 0.30793983882147, 0.14170721931088],
             'eta': [0.30793983882147, 0.14170721931088, 0.30793983882147,
                     0.14170721931088, 0.55035294186764, 0.55035294186764],
             'w': [0.0727579168455165] * 6}
        ]

    else:

        degree = 7
        suborders = [
            {'type': 1, 'xi': [1.0/3.0], 'eta': [1.0/3.0], 'w': [0.2651155203943937]},
            {'type': 3, 'xi': [0.0597158717897708, 0.4701420641051151, 0.4701420641051151],
             'eta': [0.4701420641051151, 0.0597158717897708, 0.4701420641051151],
             'w': [0.1550713368142661] * 3},
            {'type': 3, 'xi': [0.7974269853530870, 0.1012865073234564, 0.1012865073234564],
             'eta': [0.1012865073234564, 0.7974269853530870, 0.1012865073234564],
             'w': [0.1479592946419152] * 3},
            {'type': 3, 'xi': [0.25, 0.25, 0.5],
             'eta': [0.25, 0.5, 0.25],
             'w': [0.0319513189684825] * 3}
        ]


    xi_list = []
    eta_list = []
    w_list = []
    for so in suborders:
        xi_list.extend(so['xi'])
        eta_list.extend(so['eta'])
        w_list.extend(so['w'])

    xy = np.array([xi_list, eta_list])
    w = np.array(w_list)
    return xy, w, degree


def reference_to_physical_t3(xy_ref: np.ndarray, t3_nodes: np.ndarray) -> tuple:
    x1, y1 = t3_nodes[:, 0]
    x2, y2 = t3_nodes[:, 1]
    x3, y3 = t3_nodes[:, 2]

    J = np.array([
        [x2 - x1, x3 - x1],
        [y2 - y1, y3 - y1]
    ])

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]


    if abs(detJ) < 1e-14:
        detJ = 1e-14 if detJ >= 0 else -1e-14

    n_pts = xy_ref.shape[1]
    xy_phys = np.zeros((2, n_pts))
    for i in range(n_pts):
        xi, eta = xy_ref[:, i]
        xy_phys[:, i] = t3_nodes[:, 0] + J @ np.array([xi, eta])

    return xy_phys, detJ, J


def integrate_scalar_on_triangle(f_func, t3_nodes: np.ndarray, rule_id: int = 1) -> float:
    xy_ref, w, _ = wandzura_triangle_rule(rule_id)
    xy_phys, detJ, _ = reference_to_physical_t3(xy_ref, t3_nodes)

    n_pts = xy_phys.shape[1]
    vals = np.array([f_func(xy_phys[0, i], xy_phys[1, i]) for i in range(n_pts)])

    integral = 0.5 * abs(detJ) * np.sum(w * vals)
    return float(integral)


def integrate_scalar_on_hexagon(f_func, center: tuple = (0.0, 0.0), R: float = 1.0, rule_id: int = 2) -> float:
    n, x, y, w, _ = hexagon_lyness_rule(rule_id)
    area = (3.0 * np.sqrt(3.0) / 2.0) * R * R

    cx, cy = center
    vals = np.array([f_func(cx + R * x[i], cy + R * y[i]) for i in range(n)])

    integral = area * np.sum(w * vals)
    return float(integral)
