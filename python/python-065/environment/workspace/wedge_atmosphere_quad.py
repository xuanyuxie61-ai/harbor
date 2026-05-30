
import numpy as np


def line_gauss_legendre(order):
    if order == 1:
        return np.array([2.0]), np.array([0.0])
    elif order == 2:
        return np.array([1.0, 1.0]), np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    elif order == 3:
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        x = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        return w, x
    elif order == 4:
        x_vals = np.array([
            -np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            -np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
            np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0)),
        ])
        w_vals = np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0,
        ])
        return w_vals, x_vals
    elif order == 5:
        x_vals = np.array([
            -np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            -np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            0.0,
            np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
            np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0,
        ])
        w_vals = np.array([
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0 * np.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * np.sqrt(70.0)) / 900.0,
        ])
        return w_vals, x_vals
    else:
        raise ValueError("不支持的线段规则阶数")


def triangle_rule(order):
    if order == 1:
        return np.array([1.0]), np.array([[1.0 / 3.0, 1.0 / 3.0]]).T
    elif order == 3:
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        xy = np.array([[0.5, 0.5, 0.0],
                       [0.5, 0.0, 0.5]])
        return w, xy
    elif order == 7:
        a1 = 0.059715871789770
        b1 = 0.797426985353087
        c1 = 0.142
        w = np.array([
            0.225,
            0.132394152788506, 0.132394152788506,
            0.125939180544827, 0.125939180544827,
            0.125939180544827, 0.125939180544827,
        ])
        xy = np.array([
            [1.0 / 3.0, a1, b1, c1, b1, a1, c1],
            [1.0 / 3.0, a1, a1, b1, c1, b1, c1],
        ])
        return w, xy
    elif order == 12:

        return triangle_rule(7)
    else:
        raise ValueError("不支持的三角形规则阶数")


def wedge_rule(line_order, triangle_order):
    line_w, line_x = line_gauss_legendre(line_order)
    tri_w, tri_xy = triangle_rule(triangle_order)

    order = line_order * tri_w.shape[0]
    w = np.zeros(order)
    xyz = np.zeros((3, order))

    k = 0
    for i in range(line_order):
        for j in range(tri_w.shape[0]):
            w[k] = line_w[i] * tri_w[j]
            xyz[0, k] = tri_xy[0, j]
            xyz[1, k] = tri_xy[1, j]
            xyz[2, k] = line_x[i]
            k += 1
    return w, xyz


def integrate_wedge_region(fun, line_order=3, triangle_order=7):
    w, xyz = wedge_rule(line_order, triangle_order)
    vals = np.array([fun(xyz[:, i]) for i in range(w.shape[0])])
    return float(np.dot(w, vals))


def map_wedge_to_atmospheric_column(xyz_ref, tri_vertices, z_bottom, z_top):
    xi = xyz_ref[0, :]
    eta = xyz_ref[1, :]
    zeta = xyz_ref[2, :]


    x1, x2, x3 = tri_vertices[0, :]
    y1, y2, y3 = tri_vertices[1, :]
    x_phys = x1 + (x2 - x1) * xi + (x3 - x1) * eta
    y_phys = y1 + (y2 - y1) * xi + (y3 - y1) * eta


    z_phys = 0.5 * (z_top - z_bottom) * zeta + 0.5 * (z_top + z_bottom)

    return np.vstack([x_phys, y_phys, z_phys])


def integrate_over_atmospheric_column(fun, tri_vertices, z_bottom, z_top,
                                       line_order=3, triangle_order=7):
    w, xyz_ref = wedge_rule(line_order, triangle_order)
    xyz_phys = map_wedge_to_atmospheric_column(xyz_ref, tri_vertices, z_bottom, z_top)


    area = 0.5 * abs(
        tri_vertices[0, 0] * (tri_vertices[1, 1] - tri_vertices[1, 2])
        + tri_vertices[0, 1] * (tri_vertices[1, 2] - tri_vertices[1, 0])
        + tri_vertices[0, 2] * (tri_vertices[1, 0] - tri_vertices[1, 1])
    )
    jacobian = area * 0.5 * (z_top - z_bottom)

    vals = np.array([fun(xyz_phys[:, i]) for i in range(w.shape[0])])
    return float(jacobian * np.dot(w, vals))


def test_wedge():

    def f(xyz):
        return 1.0
    val = integrate_wedge_region(f, line_order=3, triangle_order=7)
    assert abs(val - 1.0) < 1e-10
    print("wedge_atmosphere_quad 自测试通过")


if __name__ == "__main__":
    test_wedge()
