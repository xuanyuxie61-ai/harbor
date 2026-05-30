
import numpy as np
from typing import Tuple


def biharmonic_w1(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    term_x = a * np.cosh(g * X) + b * np.sinh(g * X) + c * X * np.cosh(g * X) + d * X * np.sinh(g * X)
    term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
    W = term_x * term_y
    return W


def biharmonic_r1(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)


    term_x_4 = (a * g ** 4 * np.cosh(g * X)
                + b * g ** 4 * np.sinh(g * X)
                + c * (4.0 * g ** 3 * np.sinh(g * X) + g ** 4 * X * np.cosh(g * X))
                + d * (4.0 * g ** 3 * np.cosh(g * X) + g ** 4 * X * np.sinh(g * X)))
    term_y = e * np.cos(g * Y) + f * np.sin(g * Y)
    w_xxxx = term_x_4 * term_y


    term_x = a * np.cosh(g * X) + b * np.sinh(g * X) + c * X * np.cosh(g * X) + d * X * np.sinh(g * X)
    term_y_4 = g ** 4 * (e * np.cos(g * Y) + f * np.sin(g * Y))
    w_yyyy = term_x * term_y_4


    w_yy = -g ** 2 * term_x * term_y
    term_x_2 = (a * g ** 2 * np.cosh(g * X)
                + b * g ** 2 * np.sinh(g * X)
                + c * (2.0 * g * np.sinh(g * X) + g ** 2 * X * np.cosh(g * X))
                + d * (2.0 * g * np.cosh(g * X) + g ** 2 * X * np.sinh(g * X)))
    w_xxyy = -g ** 2 * term_x_2 * term_y

    R = w_xxxx + 2.0 * w_xxyy + w_yyyy
    return R


def biharmonic_w2(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    term_x = a * np.cos(g * X) + b * np.sin(g * X) + c * X * np.cos(g * X) + d * X * np.sin(g * X)
    term_y = e * np.cosh(g * Y) + f * np.sinh(g * Y)
    W = term_x * term_y
    return W


def biharmonic_r2(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 0.0,
                  d: float = 0.0, e: float = 1.0, f: float = 0.0,
                  g: float = 1.0) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)


    term_x = a * np.cos(g * X) + b * np.sin(g * X) + c * X * np.cos(g * X) + d * X * np.sin(g * X)
    term_y = e * np.cosh(g * Y) + f * np.sinh(g * Y)


    term_x_4 = (a * g ** 4 * np.cos(g * X)
                + b * g ** 4 * np.sin(g * X)
                + c * (-4.0 * g ** 3 * np.sin(g * X) + g ** 4 * X * np.cos(g * X))
                + d * (4.0 * g ** 3 * np.cos(g * X) + g ** 4 * X * np.sin(g * X)))
    w_xxxx = term_x_4 * term_y


    term_y_4 = g ** 4 * term_y
    w_yyyy = term_x * term_y_4


    term_x_2 = (-a * g ** 2 * np.cos(g * X)
                - b * g ** 2 * np.sin(g * X)
                + c * (-2.0 * g * np.sin(g * X) - g ** 2 * X * np.cos(g * X))
                + d * (2.0 * g * np.cos(g * X) - g ** 2 * X * np.sin(g * X)))
    w_xxyy = g ** 2 * term_x_2 * term_y

    R = w_xxxx + 2.0 * w_xxyy + w_yyyy
    return R


def biharmonic_w3(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 1.0,
                  d: float = 1.0, e: float = 0.5, f: float = 0.5) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    dx = X - e
    dy = Y - f
    R = np.sqrt(dx ** 2 + dy ** 2)


    R = np.where(R < 1e-10, 1e-10, R)

    W = a * R ** 2 * np.log(R) + b * R ** 2 + c * np.log(R) + d
    return W


def biharmonic_r3(X: np.ndarray, Y: np.ndarray,
                  a: float = 1.0, b: float = 1.0, c: float = 1.0,
                  d: float = 1.0, e: float = 0.5, f: float = 0.5) -> np.ndarray:
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    dx = X - e
    dy = Y - f
    R = np.sqrt(dx ** 2 + dy ** 2)
    R = np.where(R < 1e-10, 1e-10, R)

    R2 = R ** 2
    R4 = R ** 4

    residual = 8.0 * a / R2 + 16.0 * b - 8.0 * c / R4
    return residual


def verify_biharmonic_discretization(Nx: int = 32, Ny: int = 32) -> dict:
    x = np.linspace(-1.0, 1.0, Nx)
    y = np.linspace(-1.0, 1.0, Ny)
    X, Y = np.meshgrid(x, y)


    W_exact = biharmonic_w1(X, Y)
    R_exact = biharmonic_r1(X, Y)


    dx = x[1] - x[0]
    dy = y[1] - y[0]


    W_xxxx = np.zeros_like(W_exact)
    W_xxxx[2:-2, 2:-2] = (W_exact[2:-2, :-4] - 4.0 * W_exact[2:-2, 1:-3]
                           + 6.0 * W_exact[2:-2, 2:-2]
                           - 4.0 * W_exact[2:-2, 3:-1]
                           + W_exact[2:-2, 4:]) / dx ** 4


    W_yyyy = np.zeros_like(W_exact)
    W_yyyy[2:-2, 2:-2] = (W_exact[:-4, 2:-2] - 4.0 * W_exact[1:-3, 2:-2]
                           + 6.0 * W_exact[2:-2, 2:-2]
                           - 4.0 * W_exact[3:-1, 2:-2]
                           + W_exact[4:, 2:-2]) / dy ** 4


    W_xxyy = np.zeros_like(W_exact)
    W_xxyy[1:-1, 1:-1] = ((W_exact[2:, 2:] - 2.0 * W_exact[2:, 1:-1] + W_exact[2:, :-2])
                           - 2.0 * (W_exact[1:-1, 2:] - 2.0 * W_exact[1:-1, 1:-1] + W_exact[1:-1, :-2])
                           + (W_exact[:-2, 2:] - 2.0 * W_exact[:-2, 1:-1] + W_exact[:-2, :-2])) / (dx ** 2 * dy ** 2)

    R_numerical = W_xxxx + 2.0 * W_xxyy + W_yyyy


    interior = slice(2, -2)
    diff = np.abs(R_numerical[interior, interior] - R_exact[interior, interior])
    max_error = np.max(diff)
    l2_error = np.sqrt(np.mean(diff ** 2))

    return {
        'max_error': max_error,
        'l2_error': l2_error,
        'dx': dx,
        'dy': dy
    }


def plate_bending_energy(W: np.ndarray, D: float, dx: float, dy: float) -> float:

    W_xx = np.zeros_like(W)
    W_yy = np.zeros_like(W)
    W_xy = np.zeros_like(W)

    W_xx[1:-1, 1:-1] = (W[1:-1, 2:] - 2.0 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dx ** 2
    W_yy[1:-1, 1:-1] = (W[2:, 1:-1] - 2.0 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dy ** 2
    W_xy[1:-1, 1:-1] = (W[2:, 2:] - W[2:, :-2] - W[:-2, 2:] + W[:-2, :-2]) / (4.0 * dx * dy)


    energy_density = (W_xx + W_yy) ** 2 - 2.0 * (1.0 - 0.3) * (W_xx * W_yy - W_xy ** 2)


    U = 0.5 * D * np.sum(energy_density[1:-1, 1:-1]) * dx * dy
    return U
