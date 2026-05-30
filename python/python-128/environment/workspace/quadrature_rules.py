
import numpy as np





def prism_rule_order(p: int):
    p = int(p)
    if p < 0 or p > 5:
        raise ValueError("prism_rule_order: 当前仅支持 p ∈ [0,5]")


    if p == 0:

        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        z = np.array([0.5])
        w = np.array([0.5])
    elif p == 1:

        x = np.array([1.0 / 3.0, 1.0 / 3.0])
        y = np.array([1.0 / 3.0, 1.0 / 3.0])
        z = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        w = np.array([0.25, 0.25])
    elif p == 2:

        tri_x = np.array([0.5, 0.5, 0.0])
        tri_y = np.array([0.5, 0.0, 0.5])
        tri_w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        z_nodes = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        z_w = np.array([0.5, 0.5])
        x, y, w = [], [], []
        for i in range(3):
            for j in range(2):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 3)
        w = np.array(w)
    elif p == 3:


        a1, a2 = 0.6590276222, 0.23193336855
        b1, b2 = 0.6590276222, 0.23193336855
        tri_x = np.array([a1, a2, a2, b1, b2, b2])
        tri_y = np.array([a2, a1, a2, b2, b1, b2])
        tri_w = np.array([0.1099517437, 0.1099517437, 0.1099517437,
                          0.1099517437, 0.1099517437, 0.1099517437])
        z_nodes = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        z_w = np.array([0.5, 0.5])
        x, y, w = [], [], []
        for i in range(6):
            for j in range(2):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 6)
        w = np.array(w)
    elif p == 4:

        a1, a2 = 0.6590276222, 0.23193336855
        tri_x = np.array([a1, a2, a2, a1, a2, a2])
        tri_y = np.array([a2, a1, a2, a2, a1, a2])
        tri_w = np.full(6, 1.0 / 12.0)
        z_nodes = np.array([0.5 - 0.3872983346, 0.5, 0.5 + 0.3872983346])
        z_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y, w = [], [], []
        for i in range(6):
            for j in range(3):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 6)
        w = np.array(w)
    else:


        tri_x = np.array([0.3333333333, 0.7974269853, 0.1012865073, 0.1012865073,
                          0.4701420641, 0.4701420641, 0.0597158718, 0.0597158718])
        tri_y = np.array([0.3333333333, 0.1012865073, 0.7974269853, 0.1012865073,
                          0.0597158718, 0.4701420641, 0.4701420641, 0.0597158718])
        tri_w = np.array([0.2250000000, 0.1259391805, 0.1259391805, 0.1259391805,
                          0.1323941527, 0.1323941527, 0.1323941527, 0.1323941527])
        z_nodes = np.array([0.5 - 0.3872983346, 0.5, 0.5 + 0.3872983346])
        z_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y, w = [], [], []
        for i in range(8):
            for j in range(3):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 8)
        w = np.array(w)

    return x, y, z, w


def integrate_over_prism(f, p: int = 4):
    x, y, z, w = prism_rule_order(p)
    s = 0.0
    for i in range(x.size):
        s += w[i] * f(x[i], y[i], z[i])
    return float(s)





def cell_ecm_contact_integral(cell_position, cell_shape, ecm_density_func,
                              contact_stiffness: float = 1.0, p: int = 4):
    a, b, c = cell_shape
    x0, y0, z0 = cell_position





    def local_f(xp, yp, zp):
        xl = x0 + a * xp
        yl = y0 + b * yp
        zl = z0 - c + c * zp
        rho = ecm_density_func(np.array([xl, yl, zl]))

        penetration = max(0.0, 1.0 - zp)
        return contact_stiffness * rho * penetration

    return integrate_over_prism(local_f, p)





def average_concentration_in_cell(cell_position, cell_shape, concentration_func,
                                   n_prisms: int = 6, p: int = 3):
    a, b, c = cell_shape
    x0, y0, z0 = cell_position


    total = 0.0
    vol_total = 0.0
    for layer in range(n_prisms):
        z_bot = -c + 2.0 * c * layer / n_prisms
        z_top = -c + 2.0 * c * (layer + 1) / n_prisms


        z_mid = 0.5 * (z_bot + z_top)
        scale = max(0.0, 1.0 - (z_mid / c) ** 2)
        if scale < 1e-12:
            continue
        a_sec = a * np.sqrt(scale)
        b_sec = b * np.sqrt(scale)

        vol_layer = 0.5 * a_sec * b_sec * (z_top - z_bot)

        x, y, z, w = prism_rule_order(p)

        for i in range(x.size):
            xl = x0 + a_sec * x[i]
            yl = y0 + b_sec * y[i]
            zl = z0 + z_bot + (z_top - z_bot) * z[i]
            total += w[i] * concentration_func(np.array([xl, yl, zl])) * vol_layer
        vol_total += vol_layer * np.sum(w)

    if vol_total < 1e-15:
        return 0.0
    return float(total / vol_total)
