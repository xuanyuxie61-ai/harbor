
import numpy as np


class ChebyshevEvaluator:

    def __init__(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.n = len(self.coeffs)
        if self.n < 1:
            raise ValueError("Chebyshev 级数项数必须 >= 1")
        if self.n > 1000:
            raise ValueError("Chebyshev 级数项数必须 <= 1000")

    def evaluate(self, x):
        x = float(x)
        if x < -1.1 or x > 1.1:
            raise ValueError(f"x = {x} 超出 Chebyshev 级数定义域 [-1.1, 1.1]")
        b1 = 0.0
        b0 = 0.0
        for i in range(self.n - 1, -1, -1):
            b2 = b1
            b1 = b0
            b0 = 2.0 * x * b1 - b2 + self.coeffs[i]
        return 0.5 * (b0 - b2)


def clausen_function(x):

    xa = -0.5 * np.pi
    xc = 1.5 * np.pi
    x2 = x
    two_pi = 2.0 * np.pi
    while x2 < xa:
        x2 += two_pi
    while x2 > xc:
        x2 -= two_pi


    if abs(x2) < np.finfo(float).eps:
        return 0.0


    c1 = np.array([
        0.05590566394715132269,
        0.0,
        0.00017630887438981157,
        0.0,
        0.00000126627414611565,
        0.0,
        0.00000001171718181344,
        0.0,
        0.00000000012300641288,
        0.0,
        0.00000000000139527290,
        0.0,
        0.00000000000001669078,
        0.0,
        0.00000000000000020761,
        0.0,
        0.00000000000000000266,
        0.0,
        0.00000000000000000003
    ], dtype=float)


    c2 = np.array([
        0.0,
        -0.96070972149008358753,
        0.0,
        0.04393661151911392781,
        0.0,
        0.00078014905905217505,
        0.0,
        0.00002621984893260601,
        0.0,
        0.00000109292497472610,
        0.0,
        0.00000005122618343931,
        0.0,
        0.00000000258863512670,
        0.0,
        0.00000000013787545462,
        0.0,
        0.00000000000763448721,
        0.0,
        0.00000000000043556938,
        0.0,
        0.00000000000002544696,
        0.0,
        0.00000000000000151561,
        0.0,
        0.00000000000000009172,
        0.0,
        0.00000000000000000563,
        0.0,
        0.00000000000000000035,
        0.0,
        0.00000000000000000002
    ], dtype=float)

    xb = 0.5 * np.pi
    if x2 < xb:

        x3 = 2.0 * x2 / np.pi
        cheb = ChebyshevEvaluator(c1)
        value = x2 - x2 * np.log(abs(x2)) + 0.5 * x2 ** 3 * cheb.evaluate(x3)
    else:

        x3 = 2.0 * x2 / np.pi - 2.0
        cheb = ChebyshevEvaluator(c2)
        value = cheb.evaluate(x3)

    return value


def periodic_torsion_potential(phi, n_terms=3):

    V_n = np.array([2.0, 1.5, 0.8, 0.3, 0.1, 0.05])[:n_terms]
    gamma_n = np.array([0.0, np.pi / 3, np.pi / 2, np.pi, 4 * np.pi / 3, 0.0])[:n_terms]

    energy = 0.0
    for i in range(n_terms):
        n = i + 1
        energy += (V_n[i] / 2.0) * (1.0 + np.cos(n * phi - gamma_n[i]))



    energy -= 0.1 * clausen_function(2.0 * phi)

    return energy


def angular_partition_function(theta_range, temperature=300.0):
    kB = 0.0019872041
    beta = 1.0 / (kB * temperature)

    n_points = 200
    phi_vals = np.linspace(-np.pi, np.pi, n_points)
    dphi = 2.0 * np.pi / (n_points - 1)

    integrand = np.exp(-beta * np.array([periodic_torsion_potential(p) for p in phi_vals]))

    q_ang = np.trapezoid(integrand, phi_vals)

    return q_ang
