
import numpy as np
from utils import legendre_monomial_integral, chebyshev1_monomial_integral


def gauss_legendre_nodes_weights(n):
    tabulated = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
        3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
            [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
        5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
            [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
    }
    if n in tabulated:
        return np.array(tabulated[n][0]), np.array(tabulated[n][1])
    try:
        x, w = np.polynomial.legendre.leggauss(n)
        return x, w
    except Exception:
        raise ValueError(f"Cannot compute Gauss-Legendre quadrature for n={n}")


def triangle_quadrature_3():
    pts = np.array([[1.0 / 6.0, 1.0 / 6.0],
                    [2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0]])
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return pts, w


def triangle_quadrature_7():
    pts = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.470142064105115],
        [0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.101286507323456],
    ])
    w = np.array([
        0.225000000000000,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ]) * 0.5
    return pts, w


def assemble_fem_matrices(nodes, elements, diffusion_func=None, reaction_func=None,
                          source_func=None, quad_order=7):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    if diffusion_func is None:
        diffusion_func = lambda x, y: 1.0
    if reaction_func is None:
        reaction_func = lambda x, y: 0.0
    if source_func is None:
        source_func = lambda x, y: 1.0

    if quad_order == 3:
        qpts, qw = triangle_quadrature_3()
    else:
        qpts, qw = triangle_quadrature_7()
    nq = len(qw)

    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)










    raise NotImplementedError("HOLE_1: Implement FEM element stiffness assembly.")


def apply_dirichlet_bc(K, F, bc_nodes, bc_values):
    K = np.array(K, dtype=float, copy=True)
    F = np.array(F, dtype=float, copy=True)
    n = K.shape[0]
    bc_nodes = np.asarray(bc_nodes, dtype=int)
    bc_values = np.asarray(bc_values, dtype=float)

    for idx in range(len(bc_nodes)):
        i = bc_nodes[idx]
        if i < 0 or i >= n:
            continue
        g = bc_values[idx]

        for j in range(n):
            if j != i:
                F[j] -= K[j, i] * g
                K[j, i] = 0.0

        K[i, :] = 0.0
        K[i, i] = 1.0
        F[i] = g
    return K, F


def exactness_test_fem_quadrature(max_degree=5):
    pts3, w3 = triangle_quadrature_3()
    pts7, w7 = triangle_quadrature_7()







    def exact_integral_ref(p, q):
        import math
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)


    all_pass = True
    for p in range(max_degree + 1):
        for q in range(max_degree + 1 - p):
            exact = exact_integral_ref(p, q)
            approx3 = sum(w * (pt[0] ** p) * (pt[1] ** q) for pt, w in zip(pts3, w3))
            approx7 = sum(w * (pt[0] ** p) * (pt[1] ** q) for pt, w in zip(pts7, w7))
            if p + q <= 2:
                if abs(approx3 - exact) > 1e-14:
                    all_pass = False
            if p + q <= 5:
                if abs(approx7 - exact) > 1e-14:
                    all_pass = False
    return all_pass
