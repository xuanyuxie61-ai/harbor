
import numpy as np
from typing import Callable, Tuple, Optional


class GammaLogCalculator:

    @staticmethod
    def alogam(x: float) -> Tuple[float, int]:
        if x <= 0.0:
            return 0.0, 1

        ifault = 0
        y = float(x)

        if x < 7.0:
            f = 1.0
            z = y
            while z < 7.0:
                f *= z
                z += 1.0
            y = z
            f = -np.log(f)
        else:
            f = 0.0

        z = 1.0 / (y * y)
        value = (f + (y - 0.5) * np.log(y) - y + 0.918938533204673
                 + (((-0.000595238095238 * z
                      + 0.000793650793651) * z
                     - 0.002777777777778) * z
                    + 0.083333333333333) / y)

        return value, ifault

    @staticmethod
    def gamma(x: float) -> float:
        log_g, err = GammaLogCalculator.alogam(x)
        if err != 0:
            return 0.0
        return np.exp(log_g)

    @staticmethod
    def factorial(n: int) -> float:
        if n < 0:
            return 0.0
        if n <= 1:
            return 1.0
        return GammaLogCalculator.gamma(n + 1.0)


class FermiIntegralCalculator:

    def __init__(self):
        self.gamma_calc = GammaLogCalculator()

    def fermi_integral(self, n: int, eta: float,
                        num_points: int = 10000) -> float:
        if n < 0:
            raise ValueError("阶数n必须非负")


        if eta > 0:
            x_max = max(eta + 20.0, 50.0)
        else:
            x_max = max(20.0, -eta + 20.0)

        x = np.linspace(0.0, x_max, num_points)
        dx = x[1] - x[0]


        denom = np.exp(x - eta) + 1.0

        denom = np.where(denom > 1e300, 1e300, denom)
        denom = np.where(denom < 1e-300, 1e-300, denom)

        integrand = x ** n / denom
        integral = np.trapezoid(integrand, x)

        return float(integral)

    def asymptotic_classical(self, n: int, eta: float) -> float:
        gamma_val = self.gamma_calc.gamma(n + 1.0)
        return gamma_val * np.exp(eta)

    def asymptotic_degenerate(self, n: int, eta: float) -> float:
        return eta ** (n + 1.0) / (n + 1.0)

    def sommerfeld_expansion(self, n: int, eta: float) -> float:
        main = eta ** (n + 1.0) / (n + 1.0)
        correction = 1.0 + (n * (n + 1.0) * np.pi ** 2) / (6.0 * eta ** 2)
        return main * correction


class HookeJeevesOptimizer:

    def __init__(self, rho: float = 0.5, eps: float = 1e-6,
                 itermax: int = 5000):
        if not (0.0 < rho < 1.0):
            raise ValueError("rho必须在(0,1)之间")
        self.rho = rho
        self.eps = eps
        self.itermax = itermax

    def _best_nearby(self, delta: np.ndarray, point: np.ndarray,
                     prev_best: float, nvars: int,
                     f: Callable, funevals: int) -> Tuple[float,
                                                           np.ndarray, int]:
        z = np.copy(point)
        best_f = prev_best

        for i in range(nvars):
            z[i] = point[i] + delta[i]
            ftmp = f(z)
            funevals += 1

            if ftmp < best_f:
                best_f = ftmp
            else:
                delta[i] = -delta[i]
                z[i] = point[i] + delta[i]
                ftmp = f(z)
                funevals += 1
                if ftmp < best_f:
                    best_f = ftmp
                else:
                    z[i] = point[i]

        return best_f, z, funevals

    def minimize(self, f: Callable, startpt: np.ndarray) -> Tuple[np.ndarray,
                                                                    float, int]:
        nvars = len(startpt)
        newx = np.copy(startpt)
        xbefore = np.copy(startpt)

        delta = np.zeros(nvars)
        for i in range(nvars):
            if abs(startpt[i]) < 1e-15:
                delta[i] = self.rho
            else:
                delta[i] = self.rho * abs(startpt[i])

        funevals = 0
        steplength = self.rho
        iters = 0
        fbefore = f(newx)
        funevals += 1
        newf = fbefore

        while iters < self.itermax and self.eps < steplength:
            iters += 1
            newx = np.copy(xbefore)
            newf, newx, funevals = self._best_nearby(
                delta, newx, fbefore, nvars, f, funevals)

            keep = True
            while newf < fbefore and keep:
                for i in range(nvars):
                    if newx[i] <= xbefore[i]:
                        delta[i] = -abs(delta[i])
                    else:
                        delta[i] = abs(delta[i])
                    tmp = xbefore[i]
                    xbefore[i] = newx[i]
                    newx[i] = newx[i] + newx[i] - tmp

                fbefore = newf
                newf, newx, funevals = self._best_nearby(
                    delta, newx, fbefore, nvars, f, funevals)

                if fbefore <= newf:
                    break

                keep = False
                for i in range(nvars):
                    if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                        keep = True
                        break

            if self.eps <= steplength and fbefore <= newf:
                steplength *= self.rho
                delta *= self.rho

        endpt = np.copy(xbefore)
        return endpt, fbefore, iters


class BTKFittingModel:

    def __init__(self, voltages: np.ndarray,
                 conductance_data: np.ndarray):
        self.V = np.asarray(voltages, dtype=np.float64)
        self.G_data = np.asarray(conductance_data, dtype=np.float64)

    def btk_conductance(self, V: np.ndarray,
                         delta: float,
                         barrier_strength: float,
                         gamma: float) -> np.ndarray:
        z2 = barrier_strength ** 2
        e = np.abs(V)


        denom = np.sqrt(np.maximum(e ** 2 - delta ** 2, 0.0) + 1e-15)
        u2 = 0.5 * (1.0 + np.sqrt(np.maximum(e ** 2 - delta ** 2, 0.0))
                    / (np.abs(e) + 1e-15))
        u2 = np.clip(u2, 0.0, 1.0)
        v2 = 1.0 - u2

        a = u2
        b = v2
        g_btk = (1.0 + z2) / ((a - b * z2) ** 2 + 1e-15)
        g_btk = np.clip(g_btk, 0.0, 10.0)


        g_mzm = gamma ** 2 / (V ** 2 + gamma ** 2 + 1e-15)


        g_total = g_btk + 0.5 * g_mzm
        return g_total

    def objective(self, params: np.ndarray) -> float:
        delta, z, gamma = params[0], params[1], params[2]


        if delta < 0.01 or delta > 5.0:
            return 1e10
        if z < 0.0 or z > 10.0:
            return 1e10
        if gamma < 0.001 or gamma > 1.0:
            return 1e10

        g_model = self.btk_conductance(self.V, delta, z, gamma)
        residual = self.G_data - g_model
        return float(np.sum(residual ** 2))

    def fit(self, initial_guess: np.ndarray) -> Tuple[np.ndarray, float]:
        optimizer = HookeJeevesOptimizer(rho=0.85, eps=1e-8,
                                          itermax=3000)
        best_params, fmin, iters = optimizer.minimize(
            self.objective, initial_guess)

        rmse = np.sqrt(fmin / len(self.V))
        return best_params, rmse


def demo():

    gl = GammaLogCalculator()
    print("ln Γ(5.5) =", gl.alogam(5.5)[0])
    print("Γ(5) =", gl.factorial(4))


    fi = FermiIntegralCalculator()
    for eta in [-5.0, 0.0, 5.0]:
        f_val = fi.fermi_integral(n=1, eta=eta)
        print(f"F_1({eta}) = {f_val:.4f}")


    V = np.linspace(-2.0, 2.0, 81)

    true_delta, true_z, true_gamma = 0.8, 0.5, 0.05
    model = BTKFittingModel(V, np.zeros_like(V))
    g_true = model.btk_conductance(V, true_delta, true_z, true_gamma)

    g_noisy = g_true + 0.02 * np.random.randn(len(V))

    model.G_data = g_noisy
    params, rmse = model.fit(initial_guess=np.array([0.6, 0.3, 0.03]))
    print(f"Fitted: delta={params[0]:.4f}, Z={params[1]:.4f}, gamma={params[2]:.4f}")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    demo()
