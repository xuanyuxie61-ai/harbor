
import numpy as np


class GaussQuadrature:

    @staticmethod
    def gauss_legendre_3point():
        nodes = np.array([
            -0.7745966692414834,
             0.0,
             0.7745966692414834
        ])
        weights = np.array([
            0.5555555555555556,
            0.8888888888888889,
            0.5555555555555556
        ])
        return nodes, weights

    @staticmethod
    def gauss_legendre_5point():
        nodes = np.array([
            -0.9061798459386640,
            -0.5384693101056831,
             0.0,
             0.5384693101056831,
             0.9061798459386640
        ])
        weights = np.array([
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891
        ])
        return nodes, weights

    @staticmethod
    def gauss_hermite_5point():
        nodes = np.array([
            -2.0201828704560856,
            -0.9585724646138195,
             0.0,
             0.9585724646138195,
             2.0201828704560856
        ])
        weights = np.array([
            0.0199532420590459,
            0.3936193231522412,
            0.9453087204829419,
            0.3936193231522412,
            0.0199532420590459
        ])
        return nodes, weights

    @staticmethod
    def generalized_hermite_integral(expon, alpha):
        if alpha <= -1.0:
            raise ValueError("alpha 必须大于 -1")

        if expon % 2 == 1:
            return 0.0

        a = alpha + expon
        if a <= -1.0:
            return -np.inf

        from scipy.special import gamma
        return gamma((a + 1.0) / 2.0)


class HypercubeIntegrals:

    @staticmethod
    def monomial_integral(m, exponents):
        exponents = np.asarray(exponents)
        if len(exponents) != m:
            raise ValueError("exponents 长度必须等于维数 m")
        if np.any(exponents < 0):
            raise ValueError("所有指数必须为非负")

        integral = 1.0
        for e in exponents:
            integral /= (e + 1.0)
        return integral

    @staticmethod
    def sample_hypercube(m, n):
        return np.random.rand(m, n)

    @staticmethod
    def monte_carlo_integral(func, m, n_samples, domain=(0.0, 1.0)):
        a, b = domain
        samples = a + (b - a) * np.random.rand(m, n_samples)

        values = np.array([func(samples[:, i]) for i in range(n_samples)])

        volume = (b - a) ** m
        estimate = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)

        return estimate, std_error


class CompositeQuadrature:

    @staticmethod
    def composite_simpson(f, a, b, n):
        if n % 2 != 0:
            n += 1

        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])

        integral = y[0] + y[-1]
        integral += 4.0 * np.sum(y[1:-1:2])
        integral += 2.0 * np.sum(y[2:-1:2])
        integral *= h / 3.0

        return integral

    @staticmethod
    def composite_trapezoid(f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])

        integral = 0.5 * (y[0] + y[-1]) + np.sum(y[1:-1])
        integral *= h

        return integral

    @staticmethod
    def gauss_legendre_composite(f, a, b, n_elements, order=3):
        if order == 3:
            nodes, weights = GaussQuadrature.gauss_legendre_3point()
        elif order == 5:
            nodes, weights = GaussQuadrature.gauss_legendre_5point()
        else:
            raise ValueError("order 必须为 3 或 5")

        h = (b - a) / n_elements
        integral = 0.0

        for e in range(n_elements):
            x_left = a + e * h
            x_right = x_left + h

            for i in range(len(nodes)):
                x = 0.5 * (x_left + x_right) + 0.5 * h * nodes[i]
                integral += 0.5 * h * weights[i] * f(x)

        return integral


class QuadratureExactnessTest:

    def __init__(self, quad_nodes, quad_weights):
        self.nodes = np.asarray(quad_nodes)
        self.weights = np.asarray(quad_weights)

    def test_monomial_exactness(self, max_degree, integral_func):
        errors = {}
        for degree in range(max_degree + 1):

            quad_value = np.sum(self.weights * (self.nodes ** degree))


            exact_value = integral_func(degree)

            if exact_value == 0.0:
                rel_error = abs(quad_value)
            else:
                rel_error = abs((quad_value - exact_value) / exact_value)

            errors[degree] = rel_error

        return errors


def compute_phase_field_energy_integral(phi, epsilon, a_func, quadrature_order=5):

    nx, ny = phi.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)


    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
    grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)
    grad_sq = grad_x ** 2 + grad_y ** 2


    W = 0.25 * (phi ** 2 - 1.0) ** 2


    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')


    energy_density = 0.5 * epsilon ** 2 * grad_sq + W + a_func(X, Y) * phi ** 2


    energy = np.sum(energy_density) * dx * dy

    return energy
