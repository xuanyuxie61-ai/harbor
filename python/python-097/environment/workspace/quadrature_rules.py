
import numpy as np




WANDZURA_TRIANGLE_7 = {
    'order': 6,
    'degree': 7,
    'points': np.array([
        [0.501426509658179, 0.249286745170910],
        [0.249286745170910, 0.501426509658179],
        [0.249286745170910, 0.249286745170910],
        [0.873821971016996, 0.063089014491502],
        [0.063089014491502, 0.873821971016996],
        [0.063089014491502, 0.063089014491502],
    ]),
    'weights': np.array([
        0.116786275726379,
        0.116786275726379,
        0.116786275726379,
        0.050844906370207,
        0.050844906370207,
        0.050844906370207,
    ])
}


WANDZURA_TRIANGLE_13 = {
    'order': 12,
    'degree': 13,
    'points': np.array([
        [0.501426509658179, 0.249286745170910],
        [0.249286745170910, 0.501426509658179],
        [0.249286745170910, 0.249286745170910],
        [0.873821971016996, 0.063089014491502],
        [0.063089014491502, 0.873821971016996],
        [0.063089014491502, 0.063089014491502],
        [0.053145049844817, 0.310352451033784],
        [0.310352451033784, 0.053145049844817],
        [0.636502499121399, 0.053145049844817],
        [0.636502499121399, 0.310352451033784],
        [0.053145049844817, 0.636502499121399],
        [0.310352451033784, 0.636502499121399],
    ]),
    'weights': np.array([
        0.082851075618374,
        0.082851075618374,
        0.082851075618374,
        0.026673617804419,
        0.026673617804419,
        0.026673617804419,
        0.043692544538037,
        0.043692544538037,
        0.043692544538037,
        0.043692544538037,
        0.043692544538037,
        0.043692544538037,
    ])
}


def integrate_triangle_wandzura(f, vertices, rule_degree=7):
    vertices = np.asarray(vertices)
    if vertices.shape != (3, 2):
        raise ValueError("vertices必须是(3,2)数组")

    if rule_degree == 7:
        rule = WANDZURA_TRIANGLE_7
    elif rule_degree == 13:
        rule = WANDZURA_TRIANGLE_13
    else:
        raise ValueError("仅支持7阶和13阶规则")


    area = 0.5 * abs(
        vertices[0, 0] * (vertices[1, 1] - vertices[2, 1]) +
        vertices[1, 0] * (vertices[2, 1] - vertices[0, 1]) +
        vertices[2, 0] * (vertices[0, 1] - vertices[1, 1])
    )




    points_ref = rule['points']
    weights = rule['weights']

    result = 0.0
    for i in range(rule['order']):
        xi, eta = points_ref[i]
        x = vertices[0, 0] + (vertices[1, 0] - vertices[0, 0]) * xi + (vertices[2, 0] - vertices[0, 0]) * eta
        y = vertices[0, 1] + (vertices[1, 1] - vertices[0, 1]) * xi + (vertices[2, 1] - vertices[0, 1]) * eta
        result += weights[i] * f(x, y)

    return area * 2.0 * result


def gauss_legendre_1d(n):
    if n < 1 or n > 10:
        raise ValueError("节点数必须在1到10之间")


    tables = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502691896258, 0.5773502691896258], [1.0, 1.0]),
        3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
            [0.5555555555555556, 0.8888888888888889, 0.5555555555555556]),
        4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
        5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
            [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
    }

    if n in tables:
        nodes, weights = tables[n]
        return np.array(nodes), np.array(weights)


    from numpy.polynomial.legendre import leggauss
    return leggauss(n)


def integrate_3d_pyramid_gauss(f, base_vertices, apex, height, n_r=4, n_z=4):
    base_vertices = np.asarray(base_vertices)
    if base_vertices.shape[0] < 3:
        raise ValueError("底面至少需要3个顶点")

    nodes_r, weights_r = gauss_legendre_1d(n_r)
    nodes_z, weights_z = gauss_legendre_1d(n_z)

    result = 0.0
    for iz in range(n_z):

        z = 0.5 * height * (nodes_z[iz] + 1.0)
        wz = 0.5 * height * weights_z[iz]


        scale = 1.0 - z / height if height > 1e-15 else 0.0
        cx, cy = apex


        for tri_idx in range(base_vertices.shape[0] - 2):
            tri = np.array([
                base_vertices[0],
                base_vertices[tri_idx + 1],
                base_vertices[tri_idx + 2]
            ])

            tri_scaled = np.zeros_like(tri)
            for i in range(3):
                tri_scaled[i, 0] = cx + scale * (tri[i, 0] - cx)
                tri_scaled[i, 1] = cy + scale * (tri[i, 1] - cy)

            area_tri = 0.5 * abs(
                tri_scaled[0, 0] * (tri_scaled[1, 1] - tri_scaled[2, 1]) +
                tri_scaled[1, 0] * (tri_scaled[2, 1] - tri_scaled[0, 1]) +
                tri_scaled[2, 0] * (tri_scaled[0, 1] - tri_scaled[1, 1])
            )



            centroid_x = np.mean(tri_scaled[:, 0])
            centroid_y = np.mean(tri_scaled[:, 1])
            result += wz * area_tri * f(centroid_x, centroid_y, z)

    return result


def integrate_field_energy_quadrature(E, H, epsilon, mu, dx, dy, dz, order=3):
    from physics_constants import electromagnetic_energy_density
    w = electromagnetic_energy_density(E, H, epsilon, mu)


    nodes, weights = gauss_legendre_1d(order)
    total = 0.0



    for ix in range(order):
        for iy in range(order):
            for iz in range(order):


                wx = 0.5 * dx * weights[ix]
                wy = 0.5 * dy * weights[iy]
                wz = 0.5 * dz * weights[iz]

                total += wx * wy * wz * np.sum(w)



    return np.sum(w) * dx * dy * dz
