
import numpy as np
from orthogonal_polynomials import gauss_legendre_nodes_weights






def gauss_legendre_tensor(d, n_per_dim):
    x1d, w1d = gauss_legendre_nodes_weights(n_per_dim)
    grids = [x1d for _ in range(d)]
    mesh = np.array(np.meshgrid(*grids, indexing='ij'))
    nodes = mesh.reshape(d, -1).T
    wg = np.array(np.meshgrid(*[w1d for _ in range(d)], indexing='ij'))
    weights = np.prod(wg.reshape(d, -1), axis=0)
    return nodes, weights






def padua_points_and_weights(degree):
    n = degree
    if n < 1:
        raise ValueError("degree must be at least 1")
    pts_list = []

    for i in range(n + 1):
        x = np.cos(i * np.pi / n)
        if i % 2 == 0:
            for j in range(n + 2):
                y = np.cos(j * np.pi / (n + 1))
                pts_list.append([x, y])
        else:
            for j in range(n + 1):
                y = np.cos((2 * j + 1) * np.pi / (2 * (n + 1)))
                pts_list.append([x, y])
    pts = np.array(pts_list)
    N = pts.shape[0]



    w = np.ones(N) * (4.0 / (n * (n + 1)))

    return pts, w






def twb_triangle_rule(strength):


    rules = {
        1: {
            'nodes': np.array([[1.0/3.0, 1.0/3.0]]),
            'weights': np.array([0.5])
        },
        2: {
            'nodes': np.array([
                [1.0/6.0, 1.0/6.0],
                [2.0/3.0, 1.0/6.0],
                [1.0/6.0, 2.0/3.0]
            ]),
            'weights': np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        },
        3: {
            'nodes': np.array([
                [1.0/3.0, 1.0/3.0],
                [0.0597158717, 0.4701420641],
                [0.4701420641, 0.0597158717],
                [0.4701420641, 0.4701420641],
                [0.7974269853, 0.1012865073],
                [0.1012865073, 0.7974269853],
                [0.1012865073, 0.1012865073]
            ]),
            'weights': np.array([
                0.1125,
                0.0661970763, 0.0661970763, 0.0661970763,
                0.0629695902, 0.0629695902, 0.0629695902
            ])
        },
        5: {
            'nodes': np.array([
                [0.333333333333333, 0.333333333333333],
                [0.059715871789770, 0.470142064105115],
                [0.470142064105115, 0.059715871789770],
                [0.470142064105115, 0.470142064105115],
                [0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.797426985353087],
                [0.101286507323456, 0.101286507323456]
            ]),
            'weights': np.array([
                0.1125,
                0.066197076394253, 0.066197076394253, 0.066197076394253,
                0.062969590272414, 0.062969590272414, 0.062969590272414
            ])
        }
    }
    if strength <= 3:
        key = 3
    elif strength <= 5:
        key = 5
    else:

        key = 5
    data = rules[key]
    return data['nodes'].copy(), data['weights'].copy()


def triangle_unit_monomial_integral(ex, ey):
    from math import factorial
    return float(factorial(ex) * factorial(ey)) / float(factorial(ex + ey + 2))


def triangle_to_standard(nodes):


    return 2.0 * nodes - 1.0






def smolyak_sparse_grid(d, level, poly_family="legendre"):
    if level < 0:
        raise ValueError("level must be non-negative")

    from itertools import product
    all_nodes = []
    all_weights = []

    def oned_rule(idx):

        n_pts = max(1, idx)
        return gauss_legendre_nodes_weights(n_pts)

    for i_vec in product(range(1, level + d + 1), repeat=d):
        if sum(i_vec) < level + d or sum(i_vec) > level + d:
            continue

        s = level + d - sum(i_vec)
        if s < 0 or s > d - 1:
            continue
        coeff = ((-1) ** s) * np.math.comb(d - 1, s)

        x_list = []
        w_list = []
        for idx in i_vec:
            x, w = oned_rule(idx)
            x_list.append(x)
            w_list.append(w)
        mesh_x = np.array(np.meshgrid(*x_list, indexing='ij')).reshape(d, -1).T
        mesh_w = np.array(np.meshgrid(*w_list, indexing='ij')).reshape(d, -1)
        w_vec = np.prod(mesh_w, axis=0)
        all_nodes.append(mesh_x)
        all_weights.append(coeff * w_vec)

    if not all_nodes:
        return np.zeros((0, d)), np.zeros(0)
    nodes = np.vstack(all_nodes)
    weights = np.concatenate(all_weights)
    return nodes, weights


def test_quadrature_rules():

    x, w = gauss_legendre_nodes_weights(5)
    for p in range(10):
        val = np.sum(x**p * w)
        exact = (1.0 - (-1.0)**(p + 1)) / (p + 1.0) if p % 2 == 0 else 0.0
        if p <= 9:
            assert np.isclose(val, exact, atol=1e-14), f"Gauss-Legendre fail at p={p}"

    pts, wg = gauss_legendre_tensor(2, 5)
    val = np.sum((pts[:, 0]**2) * (pts[:, 1]**4) * wg)
    exact = (2.0 / 3.0) * (2.0 / 5.0)
    assert np.isclose(val, exact, atol=1e-12)

    tn, tw = twb_triangle_rule(5)
    val = np.sum(tn[:, 0]**2 * tn[:, 1]**3 * tw)
    exact = triangle_unit_monomial_integral(2, 3)
    assert np.isclose(val, exact, atol=1e-6)
    print("quadrature_rules: all self-tests passed")


if __name__ == "__main__":
    test_quadrature_rules()
