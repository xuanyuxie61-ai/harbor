
import numpy as np




_TETRAHEDRON_RULES = {
    1: {
        "points": np.array([[0.25, 0.25, 0.25]]),
        "weights": np.array([1.0]) / 6.0,
    },
    2: {
        "points": np.array([
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660],
        ]),
        "weights": np.array([0.25, 0.25, 0.25, 0.25]) / 6.0,
    },
    3: {
        "points": np.array([
            [0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
            [0.5000000000000000, 0.1666666666666667, 0.1666666666666667],
            [0.1666666666666667, 0.5000000000000000, 0.1666666666666667],
            [0.1666666666666667, 0.1666666666666667, 0.5000000000000000],
            [0.1666666666666667, 0.1666666666666667, 0.1666666666666667],
        ]),
        "weights": np.array([-0.8, 0.45, 0.45, 0.45, 0.45]) / 6.0,
    },
    4: {

        "points": np.array([
            [0.25, 0.25, 0.25],
            [0.785714285714286, 0.071428571428571, 0.071428571428571],
            [0.071428571428571, 0.785714285714286, 0.071428571428571],
            [0.071428571428571, 0.071428571428571, 0.785714285714286],
            [0.071428571428571, 0.071428571428571, 0.071428571428571],
            [0.399403576166799, 0.399403576166799, 0.100596423833201],
            [0.399403576166799, 0.100596423833201, 0.399403576166799],
            [0.399403576166799, 0.100596423833201, 0.100596423833201],
            [0.100596423833201, 0.399403576166799, 0.399403576166799],
            [0.100596423833201, 0.399403576166799, 0.100596423833201],
            [0.100596423833201, 0.100596423833201, 0.399403576166799],
        ]),
        "weights": np.array([
            -0.013155555555556, 0.007622222222222, 0.007622222222222,
            0.007622222222222, 0.007622222222222, 0.024888888888889,
            0.024888888888889, 0.024888888888889, 0.024888888888889,
            0.024888888888889, 0.024888888888889
        ]) / 6.0,
    },
    5: {

        "points": np.array([
            [0.25, 0.25, 0.25],
            [0.5, 0.1666666667, 0.1666666667],
            [0.1666666667, 0.5, 0.1666666667],
            [0.1666666667, 0.1666666667, 0.5],
            [0.1666666667, 0.1666666667, 0.1666666667],
            [0.8464398480, 0.0511866873, 0.0511866873],
            [0.0511866873, 0.8464398480, 0.0511866873],
            [0.0511866873, 0.0511866873, 0.8464398480],
            [0.0511866873, 0.0511866873, 0.0511866873],
            [0.4042339137, 0.4042339137, 0.0957660863],
            [0.4042339137, 0.0957660863, 0.4042339137],
            [0.4042339137, 0.0957660863, 0.0957660863],
            [0.0957660863, 0.4042339137, 0.4042339137],
            [0.0957660863, 0.4042339137, 0.0957660863],
            [0.0957660863, 0.0957660863, 0.4042339137],
        ]),
        "weights": np.array([
            0.0197530864, 0.0116450600, 0.0116450600, 0.0116450600, 0.0116450600,
            0.0019090913, 0.0019090913, 0.0019090913, 0.0019090913,
            0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893, 0.0305310893
        ]) / 6.0,
    },
}


def tetrahedron_arbq(degree):
    if degree not in _TETRAHEDRON_RULES:
        raise ValueError(f"Degree {degree} not supported. Use 1~5.")
    rule = _TETRAHEDRON_RULES[degree]
    return rule["points"].copy(), rule["weights"].copy()


def integrate_tetrahedron(f, n_per_dim=8):
    nodes, weights = np.polynomial.legendre.leggauss(n_per_dim)

    u_nodes = 0.5 * (nodes + 1.0)
    u_weights = 0.5 * weights
    v_nodes = 0.5 * (nodes + 1.0)
    v_weights = 0.5 * weights
    w_nodes = 0.5 * (nodes + 1.0)
    w_weights = 0.5 * weights

    total = 0.0
    for i in range(n_per_dim):
        u = u_nodes[i]
        w_u = u_weights[i]
        for j in range(n_per_dim):
            v = v_nodes[j]
            w_v = v_weights[j]
            jac = (1.0 - u) ** 2 * (1.0 - v)
            for k in range(n_per_dim):
                w = w_nodes[k]
                w_w = w_weights[k]
                x = u
                y = v * (1.0 - u)
                z = w * (1.0 - u) * (1.0 - v)
                total += w_u * w_v * w_w * jac * f(x, y, z)
    return total


def wedge_exactness_monomial_integral(e1, e2, e3):
    if e1 < 0 or e2 < 0 or e3 < 0:
        raise ValueError("指数必须非负")

    if e3 % 2 == 1:
        z_integral = 0.0
    else:
        z_integral = 2.0 / (e3 + 1)

    from math import factorial
    xy_integral = factorial(e1) * factorial(e2) / factorial(e1 + e2 + 2)
    return z_integral * xy_integral


def integrate_wedge_gauss(f, n_xy=8, n_z=8):

    z_nodes, z_weights = np.polynomial.legendre.leggauss(n_z)

    z_nodes = z_nodes
    z_weights = z_weights



    from numpy.polynomial.legendre import leggauss
    u_nodes, u_weights = leggauss(n_xy)
    v_nodes, v_weights = leggauss(n_xy)

    total = 0.0
    for i in range(n_xy):
        for j in range(n_xy):


            u = u_nodes[i]
            v = v_nodes[j]
            x = 0.25 * (1 + u) * (1 - v)
            y = 0.25 * (1 + u) * (1 + v)
            jac = 0.125 * (1 + u)
            w_xy = u_weights[i] * v_weights[j] * jac
            for k in range(n_z):
                z = z_nodes[k]
                w_z = z_weights[k]
                total += w_xy * w_z * f(x, y, z)
    return total


def test_quadrature_rules():

    f1 = lambda x, y, z: x
    val1 = integrate_tetrahedron(f1, degree=4)
    exact1 = 1.0 / 24.0
    print(f"[quadrature_rules] Tetrahedron ∫x dV = {val1:.6e}, exact = {exact1:.6e}, err = {abs(val1-exact1):.3e}")
    assert abs(val1 - exact1) < 1e-10


    exact2 = wedge_exactness_monomial_integral(1, 1, 0)

    f2 = lambda x, y, z: x * y
    val2 = integrate_wedge_gauss(f2, n_xy=8, n_z=4)
    print(f"[quadrature_rules] Wedge ∫xy dV = {val2:.6e}, exact = {exact2:.6e}, err = {abs(val2-exact2):.3e}")


if __name__ == "__main__":
    test_quadrature_rules()
