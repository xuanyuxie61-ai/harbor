
import numpy as np
from scipy import integrate
from typing import Callable, Tuple, Optional





def csevl(x: float, cs: np.ndarray, n: Optional[int] = None) -> float:
    if n is None:
        n = len(cs)
    if n < 1:
        raise ValueError("csevl: Number of terms must be >= 1.")
    if n > 1000:
        raise ValueError("csevl: Number of terms must be <= 1000.")
    if x < -1.1 or x > 1.1:
        raise ValueError(f"csevl: X outside valid range [-1.1, 1.1], got {x}")

    b0 = 0.0
    b1 = 0.0
    b2 = 0.0

    for i in range(n - 1, -1, -1):
        b2 = b1
        b1 = b0
        b0 = 2.0 * x * b1 - b2 + cs[i]

    value = 0.5 * (b0 - b2)
    return value


def inits(cs: np.ndarray, eta: float = 1e-12) -> int:
    n = len(cs)
    for i in range(n - 1, -1, -1):
        if abs(cs[i]) > eta:
            return i + 1
    return 1





def dot_l2(f: Callable[[float], float], g: Callable[[float], float],
           a: float, b: float, epsabs: float = 1e-14,
           epsrel: float = 1e-12) -> float:
    if a >= b:
        raise ValueError("Interval [a,b] must satisfy a < b.")

    def integrand(x: float) -> float:
        return f(x) * g(x)

    val, err = integrate.quad(integrand, a, b, limit=100,
                              epsabs=epsabs, epsrel=epsrel)
    if err > 1e-8:
        raise RuntimeError(f"L2 inner product integration failed with error {err}")
    return val





class BoneDensityField:

    def __init__(self, cheb_coeffs: Optional[np.ndarray] = None,
                 nx_cheb: int = 8, ny_cheb: int = 8):
        self.nx_cheb = nx_cheb
        self.ny_cheb = ny_cheb

        if cheb_coeffs is not None:
            if cheb_coeffs.shape != (nx_cheb, ny_cheb):
                raise ValueError("cheb_coeffs shape mismatch")
            self.coeffs = cheb_coeffs.copy()
        else:

            self.coeffs = self._generate_default_coeffs()

    def _generate_default_coeffs(self) -> np.ndarray:
        nx, ny = self.nx_cheb, self.ny_cheb
        coeffs = np.zeros((nx, ny))


        coeffs[0, 0] = 1.0

        if nx > 2:
            coeffs[2, 0] = -0.15
        if ny > 2:
            coeffs[0, 2] = -0.15

        if nx > 1 and ny > 1:
            coeffs[1, 1] = 0.05

        if nx > 4 and ny > 4:
            coeffs[4, 0] = 0.02
            coeffs[0, 4] = 0.02

        return coeffs

    def evaluate(self, xi: float, eta: float) -> float:
        if not (-1.0 <= xi <= 1.0 and -1.0 <= eta <= 1.0):

            xi = max(-1.0, min(1.0, xi))
            eta = max(-1.0, min(1.0, eta))

        nx = inits(self.coeffs[:, 0], eta=1e-12)
        ny = inits(self.coeffs[0, :], eta=1e-12)


        row_vals = np.zeros(ny)
        for j in range(ny):
            row_vals[j] = csevl(xi, self.coeffs[:, j], nx)


        val = csevl(eta, row_vals, ny)
        return val

    def evaluate_physical(self, x: float, y: float,
                          xlim: Tuple[float, float] = (0.0, 20.0),
                          ylim: Tuple[float, float] = (0.0, 30.0)) -> float:
        x_min, x_max = xlim
        y_min, y_max = ylim

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid physical domain bounds.")

        xi = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        eta = 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        return self.evaluate(xi, eta)

    def evaluate_batch(self, xy: np.ndarray,
                       xlim: Tuple[float, float] = (0.0, 20.0),
                       ylim: Tuple[float, float] = (0.0, 30.0)) -> np.ndarray:
        if xy.shape[0] != 2:
            raise ValueError("xy must have shape (2, N)")
        N = xy.shape[1]
        vals = np.zeros(N)
        for i in range(N):
            vals[i] = self.evaluate_physical(xy[0, i], xy[1, i], xlim, ylim)
        return vals

    def elastic_modulus_from_density(self, rho: float,
                                     E0: float = 17.0e3,
                                     power: float = 2.0) -> float:
        rho_clip = max(0.0, min(1.0, rho))
        return E0 * (rho_clip ** power)

    def set_coefficients(self, coeffs: np.ndarray):
        if coeffs.shape != (self.nx_cheb, self.ny_cheb):
            raise ValueError(f"Expected shape ({self.nx_cheb}, {self.ny_cheb}), got {coeffs.shape}")
        self.coeffs = coeffs.copy()

    def compute_l2_norm(self, xlim: Tuple[float, float] = (0.0, 20.0),
                        ylim: Tuple[float, float] = (0.0, 30.0)) -> float:
        x_min, x_max = xlim
        y_min, y_max = ylim

        def integrand(y: float, x: float) -> float:
            val = self.evaluate_physical(x, y, xlim, ylim)
            return val * val

        val, err = integrate.dblquad(integrand, x_min, x_max,
                                     lambda x: y_min, lambda x: y_max,
                                     epsabs=1e-10, epsrel=1e-10)
        if err > 1e-6:
            raise RuntimeError(f"L2 norm integration failed with error {err}")
        return np.sqrt(val)
