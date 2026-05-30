
import numpy as np
from typing import Tuple


class ChebyshevGraphConv:

    def __init__(self, in_channels: int, out_channels: int, K: int = 4):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K


        limit = np.sqrt(6.0 / (in_channels + out_channels))
        self.theta = np.random.uniform(-limit, limit, (K, in_channels, out_channels))
        self.bias = np.zeros(out_channels, dtype=np.float64)

    def __call__(self, x: np.ndarray, laplacian_mul) -> np.ndarray:
        n_nodes = x.shape[0]

        cheb_x = np.zeros((self.K, n_nodes, self.in_channels), dtype=np.float64)

        cheb_x[0] = x
        if self.K > 1:

            cheb_x[1] = laplacian_mul(x)

        for k in range(2, self.K):
            cheb_x[k] = 2.0 * laplacian_mul(cheb_x[k - 1]) - cheb_x[k - 2]


        y = np.zeros((n_nodes, self.out_channels), dtype=np.float64)
        for k in range(self.K):
            y += cheb_x[k] @ self.theta[k]
        y += self.bias
        return y

    def parameters(self) -> list:
        return [self.theta, self.bias]


def chebyshev_coefficients_1d(nd: int, xd: np.ndarray, yd: np.ndarray) -> Tuple[np.ndarray, float, float]:
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    xmin, xmax = xd.min(), xd.max()
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0

    t = 2.0 * (xd - xmin) / (xmax - xmin) - 1.0

    t = np.clip(t, -1.0, 1.0)
    theta = np.arccos(t)

    A = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        A[i, :] = np.cos(i * theta)

    c, _, _, _ = np.linalg.lstsq(A.T, yd, rcond=None)
    return c, xmin, xmax


def chebyshev_value_1d(c: np.ndarray, xmin: float, xmax: float, xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    t = 2.0 * (xi - xmin) / (xmax - xmin) - 1.0
    t = np.clip(t, -1.0, 1.0)
    nd = len(c)

    b0 = np.zeros_like(t)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for i in range(nd - 1, 0, -1):
        b0 = 2.0 * t * b1 - b2 + c[i]
        b2 = b1
        b1 = b0
    return b1 * t - b2 + c[0]
