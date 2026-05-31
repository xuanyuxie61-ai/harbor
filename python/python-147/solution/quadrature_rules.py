"""
quadrature_rules.py
===================
High-precision numerical quadrature rules for computing integrals in the
physics-informed loss functional.

The PINN loss functional involves domain integrals of the form:

    L_phys = \int_\Omega |r(t,x)|^2  d\Omega

where r(t,x) is the PDE residual.  To evaluate this accurately, we employ:

  1. Gauss-Legendre quadrature in 1D and tensor-product 2D
  2. Gauss-Kronrod rules for adaptive error estimation
  3. Padua points for non-tensorial interpolation-based quadrature on the square

These rules are adapted from seed projects 1143_square_exactness and
629_kronrod_rule.
"""

import numpy as np


# Precomputed Gauss-Legendre nodes and weights on [-1, 1] for orders 1..10
_LEGENDRE_NODES = {
    1: np.array([0.0]),
    2: np.array([-0.5773502691896258, 0.5773502691896258]),
    3: np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
    4: np.array([-0.8611363115940526, -0.3399810435848563,
                 0.3399810435848563, 0.8611363115940526]),
    5: np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                 0.5384693101056831, 0.9061798459386640]),
    6: np.array([-0.9324695142031521, -0.6612093864662645,
                 -0.2386191860831969, 0.2386191860831969,
                 0.6612093864662645, 0.9324695142031521]),
    7: np.array([-0.9491079123427585, -0.7415311855993945,
                 -0.4058451513773972, 0.0,
                 0.4058451513773972, 0.7415311855993945,
                 0.9491079123427585]),
    8: np.array([-0.9602898564975363, -0.7966664774136267,
                 -0.5255324099163290, -0.1834346424956498,
                 0.1834346424956498, 0.5255324099163290,
                 0.7966664774136267, 0.9602898564975363]),
    9: np.array([-0.9681602395076261, -0.8360311073266358,
                 -0.6133714327005904, -0.3242534234038089, 0.0,
                 0.3242534234038089, 0.6133714327005904,
                 0.8360311073266358, 0.9681602395076261]),
    10: np.array([-0.9739065285171717, -0.8650633666889845,
                  -0.6794095682990244, -0.4333953941292472,
                  -0.1488743389816312, 0.1488743389816312,
                  0.4333953941292472, 0.6794095682990244,
                  0.8650633666889845, 0.9739065285171717]),
}

_LEGENDRE_WEIGHTS = {
    1: np.array([2.0]),
    2: np.array([1.0, 1.0]),
    3: np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556]),
    4: np.array([0.3478548451374538, 0.6521451548625461,
                 0.6521451548625461, 0.3478548451374538]),
    5: np.array([0.2369268850561891, 0.4786286704993665,
                 0.5688888888888889, 0.4786286704993665,
                 0.2369268850561891]),
    6: np.array([0.1713244923791704, 0.3607615730481386,
                 0.4679139345726910, 0.4679139345726910,
                 0.3607615730481386, 0.1713244923791704]),
    7: np.array([0.1294849661688697, 0.2797053914892766,
                 0.3818300505051189, 0.4179591836734694,
                 0.3818300505051189, 0.2797053914892766,
                 0.1294849661688697]),
    8: np.array([0.1012285362903763, 0.2223810344533745,
                 0.3137066458778873, 0.3626837833783620,
                 0.3626837833783620, 0.3137066458778873,
                 0.2223810344533745, 0.1012285362903763]),
    9: np.array([0.0812743883615744, 0.1806481606948574,
                 0.2606106964029354, 0.3123470770400029,
                 0.3302393550012598, 0.3123470770400029,
                 0.2606106964029354, 0.1806481606948574,
                 0.0812743883615744]),
    10: np.array([0.0666713443086881, 0.1494513491505806,
                  0.2190863625159820, 0.2692667193099963,
                  0.2955242247147529, 0.2955242247147529,
                  0.2692667193099963, 0.2190863625159820,
                  0.1494513491505806, 0.0666713443086881]),
}


def gauss_legendre_1d(n, a=-1.0, b=1.0):
    """
    Return Gauss-Legendre quadrature nodes x and weights w on [a, b].

    The integral is approximated as:
        \int_a^b f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    Parameters
    ----------
    n : int
        Quadrature order (1 <= n <= 10).
    a, b : float
        Integration interval.

    Returns
    -------
    x, w : ndarray
        Nodes and weights.
    """
    if n not in _LEGENDRE_NODES:
        raise ValueError(f"Order n={n} not supported. Use 1 <= n <= 10.")
    x_ref = _LEGENDRE_NODES[n]
    w_ref = _LEGENDRE_WEIGHTS[n]
    # Affine map from [-1, 1] to [a, b]:
    #   x = (b-a)/2 * x_ref + (a+b)/2
    #   w = (b-a)/2 * w_ref
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return scale * x_ref + shift, scale * w_ref


def gauss_legendre_2d(nx, ny, ax, bx, ay, by):
    """
    Tensor-product 2D Gauss-Legendre quadrature on the rectangle
    [ax, bx] x [ay, by].

    The 2D integral is:
        \int_{ay}^{by} \int_{ax}^{bx} f(x,y) dx dy
        \approx \sum_{i=1}^{nx} \sum_{j=1}^{ny} w_{ij} f(x_i, y_j)

    where w_{ij} = w^x_i * w^y_j.

    Adapted from seed project 1143_square_exactness (legendre_2d_set).
    """
    x_1d, wx = gauss_legendre_1d(nx, ax, bx)
    y_1d, wy = gauss_legendre_1d(ny, ay, by)
    n = nx * ny
    x = np.zeros(n)
    y = np.zeros(n)
    w = np.zeros(n)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            x[idx] = x_1d[i]
            y[idx] = y_1d[j]
            w[idx] = wx[i] * wy[j]
            idx += 1
    return x, y, w


def kronrod_nodes_weights(n):
    """
    Return precomputed Gauss-Kronrod (7,15) rule nodes and weights.

    The (2n+1)-point Kronrod rule reuses the n Gauss points and adds
    n+1 optimally placed Kronrod points.  For adaptive quadrature,
    the difference between the Gauss and Kronrod estimates provides
    an error bound.

    Here we provide the classic (7, 15) pair from seed project 629.

    Parameters
    ----------
    n : int
        Order of embedded Gauss rule (must be 7 for precomputed tables).

    Returns
    -------
    x : ndarray, shape (15,)
        Nodes on [-1, 1] in ascending order.
    w_kronrod : ndarray, shape (15,)
        Weights for the 15-point Kronrod rule.
    w_gauss : ndarray, shape (15,)
        Weights for the embedded 7-point Gauss rule (non-Gauss entries are 0).
    """
    if n != 7:
        raise ValueError("Only n=7 precomputed Kronrod table is available.")

    # Nonnegative nodes in descending order: a1 > a2 > ... > a7 > 0
    x_pos = np.array([
        0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
        0.7415311855993945, 0.5860872354676911, 0.4058451513773972,
        0.2077849550078985, 0.0
    ])
    a = x_pos[:-1]  # [a1, a2, ..., a7]
    # Ascending full nodes: [-a1, -a2, ..., -a7, 0, a7, a6, ..., a1]
    x = np.concatenate([-a, [0.0], a[::-1]])

    # Kronrod weights symmetric about center
    w_pos = np.array([
        0.02293532201052922, 0.06309209262997856, 0.1047900103222502,
        0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
        0.2044329400752989
    ])
    w0 = 0.2094821410847278
    w_kronrod = np.concatenate([w_pos, [w0], w_pos[::-1]])

    # Embedded Gauss weights (at a2, a4, a6, 0, a6, a4, a2)
    wg_inner = np.array([0.1294849661688697, 0.2797053914892766,
                         0.3818300505051189])
    wg0 = 0.4179591836734694
    w_gauss = np.array([
        0.0, wg_inner[0], 0.0, wg_inner[1], 0.0, wg_inner[2], 0.0,
        wg0,
        0.0, wg_inner[2], 0.0, wg_inner[1], 0.0, wg_inner[0], 0.0
    ])
    return x, w_kronrod, w_gauss


def padua_point_set(level):
    """
    Return the first-kind Padua points of level L on the square [-1,1]^2.

    Padua points are the first optimal unisolvent set for bivariate
    polynomial interpolation in the square.  They are also nearly optimal
    for cubature.  The number of points is N = (L+1)(L+2)/2.

    For level L, the points are defined as:
        x_i = cos( (i * pi) / L ),   i = 0, ..., L
        and y generated from Chebyshev-like distribution.

    Here we use the explicit point sets for levels 0..5 from seed project
    1143_square_exactness for reproducibility.

    Parameters
    ----------
    level : int
        Level L (0 <= L <= 5 for precomputed tables).

    Returns
    -------
    x, y : ndarray
        Padua points.
    """
    if level == 0:
        return np.array([0.0]), np.array([0.0])
    elif level == 1:
        x = np.array([-1.0, -1.0, 1.0])
        y = np.array([-1.0, 1.0, 0.0])
        return x, y
    elif level == 2:
        x = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
        y = np.array([-1.0, 0.5, -0.5, 1.0, -1.0, 0.5])
        return x, y
    elif level == 3:
        x = np.array([
            -1.0, -1.0, -1.0, -0.5, -0.5,
            0.5, 0.5, 0.5, 1.0, 1.0
        ])
        y = np.array([
            -1.0, 0.0, 1.0, -0.7071067811865475, 0.7071067811865476,
            -1.0, 0.0, 1.0, -0.7071067811865475, 0.7071067811865476
        ])
        return x, y
    elif level == 4:
        x = np.array([
            -1.0, -1.0, -1.0, -0.7071067811865475, -0.7071067811865475,
            -0.7071067811865475, 0.0, 0.0, 0.0, 0.7071067811865476,
            0.7071067811865476, 0.7071067811865476, 1.0, 1.0, 1.0
        ])
        y = np.array([
            -1.0, -0.3090169943749473, 0.8090169943749475,
            -0.8090169943749473, 0.3090169943749475, 1.0,
            -1.0, -0.3090169943749473, 0.8090169943749475,
            -0.8090169943749473, 0.3090169943749475, 1.0,
            -1.0, -0.3090169943749473, 0.8090169943749475
        ])
        return x, y
    elif level == 5:
        x = np.array([
            -1.0, -1.0, -1.0, -1.0, -0.8090169943749473,
            -0.8090169943749473, -0.8090169943749473, -0.3090169943749473,
            -0.3090169943749473, -0.3090169943749473, -0.3090169943749473,
            0.3090169943749475, 0.3090169943749475, 0.3090169943749475,
            0.3090169943749475, 0.8090169943749475, 0.8090169943749475,
            0.8090169943749475, 1.0, 1.0, 1.0, 1.0
        ])
        y = np.array([
            -1.0, -0.5, 0.5, 1.0, -0.8660254037844387, 0.0,
            0.8660254037844387, -1.0, -0.5, 0.5, 1.0,
            -0.8660254037844387, -0.5, 0.5, 1.0,
            -1.0, -0.5, 0.5, -0.8660254037844387, 0.0,
            0.8660254037844387
        ])
        return x, y
    else:
        raise ValueError(f"Padua level {level} not precomputed. Use 0 <= L <= 5.")


def integrate_2d_gauss_legendre(f, nx, ny, ax, bx, ay, by):
    """
    Integrate f(x,y) over [ax,bx]x[ay,by] using tensor-product Gauss-Legendre.

    Parameters
    ----------
    f : callable
        Function f(x, y) where x, y are 1D arrays of the same length.
    nx, ny : int
        Quadrature orders.
    ax, bx, ay, by : float
        Integration bounds.

    Returns
    -------
    integral : float
        Approximate integral value.
    error_estimate : float
        Naive error estimate (difference between nx,ny and nx-1,ny-1 rules).
    """
    x, y, w = gauss_legendre_2d(nx, ny, ax, bx, ay, by)
    fx = f(x, y)
    integral = np.sum(w * fx)

    # Error estimate via lower-order rule
    if nx > 1 and ny > 1:
        x2, y2, w2 = gauss_legendre_2d(nx - 1, ny - 1, ax, bx, ay, by)
        fx2 = f(x2, y2)
        integral2 = np.sum(w2 * fx2)
        error_estimate = abs(integral - integral2)
    else:
        error_estimate = np.nan
    return integral, error_estimate
