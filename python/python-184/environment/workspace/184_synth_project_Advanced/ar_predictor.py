
import numpy as np
from typing import Tuple


class ARPredictor:

    def __init__(self, order: int = 5, max_iter: int = 200, tol: float = 1e-12):
        if order < 1:
            raise ValueError("AR order must be >= 1.")
        self.p = order
        self.max_iter = max_iter
        self.tol = tol
        self.ar_coefs: np.ndarray | None = None
        self.sigma_e: float = 1.0
        self.roots: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "ARPredictor":
        if len(series) < 2 * self.p:
            raise ValueError(f"Series length {len(series)} too short for order {self.p}.")

        n = len(series)
        mean_val = np.mean(series)
        x = series - mean_val


        autocorr = np.zeros(self.p + 1)
        autocorr[0] = np.dot(x, x) / n
        for lag in range(1, self.p + 1):
            autocorr[lag] = np.dot(x[:-lag], x[lag:]) / n

        if autocorr[0] <= 1e-15:
            raise ValueError("Series variance is zero or negative.")


        rho = autocorr / autocorr[0]


        from toeplitz_solver import ToeplitzSolver
        solver = ToeplitzSolver()
        a, k = solver.solve_yule_walker(autocorr)
        self.ar_coefs = a.copy()


        self.sigma_e = np.sqrt(max(autocorr[0] * (1.0 - np.dot(rho[1:], a)), 1e-15))



        poly = np.concatenate(([1.0], a))
        self.roots = self._newton_maehly_roots(poly)

        return self

    def predict(self, history: np.ndarray, steps: int = 1) -> np.ndarray:
        if self.ar_coefs is None:
            raise RuntimeError("Model not fitted yet.")
        if len(history) < self.p:
            raise ValueError(f"History length {len(history)} < AR order {self.p}.")

        pred = np.zeros(steps)
        buf = np.concatenate([history[-self.p:].copy(), pred])
        for h in range(steps):
            idx = self.p + h
            pred[h] = -np.dot(self.ar_coefs, buf[idx - self.p:idx][::-1])
            buf[idx] = pred[h]
        return pred

    def _poly_and_derivative(self, coefs: np.ndarray, z: np.complex128) -> Tuple[np.complex128, np.complex128]:
        p = coefs[0]
        d = 0.0 + 0.0j
        for c in coefs[1:]:
            d = z * d + p
            p = z * p + c
        return p, d

    def _newton_maehly_roots(self, coefs: np.ndarray) -> np.ndarray:
        p = len(coefs) - 1
        if p == 0:
            return np.array([])
        if p == 1:
            return np.array([-coefs[1] / coefs[0]])


        cauchy_bound = 1.0 + np.max(np.abs(coefs[1:])) / abs(coefs[0])


        roots = cauchy_bound * np.exp(2j * np.pi * np.arange(p) / p + 1j * np.pi / (2 * p))

        for _ in range(self.max_iter):
            max_diff = 0.0
            for i in range(p):
                z = roots[i]
                pz, dpz = self._poly_and_derivative(coefs, z)

                if abs(pz) < self.tol:
                    continue


                deflate = 0.0 + 0.0j
                for j in range(p):
                    if i == j:
                        continue
                    diff = z - roots[j]
                    if abs(diff) < self.tol:
                        continue
                    deflate += 1.0 / diff

                denom = dpz - pz * deflate
                if abs(denom) < 1e-15:
                    continue

                dz = pz / denom
                roots[i] -= dz
                max_diff = max(max_diff, abs(dz))

            if max_diff < self.tol:
                break

        return roots

    def stability_analysis(self) -> dict:
        if self.roots is None or len(self.roots) == 0:
            return {"stable": True, "modes": []}

        modes = []
        stable = True
        for r in self.roots:
            mod = abs(r)
            if mod > 1.0 + 1e-8:
                stable = False

            if mod > 0 and mod < 1.0:
                tau = -1.0 / np.log(mod) if mod > 0 else np.inf
            else:
                tau = np.inf

            angle = np.angle(r)
            freq = abs(angle) / (2.0 * np.pi)
            modes.append({
                "root": r,
                "modulus": mod,
                "time_constant": tau,
                "frequency": freq,
                "damping_ratio": -np.log(mod) / np.sqrt(np.log(mod) ** 2 + angle ** 2) if mod > 0 else 1.0
            })
        return {"stable": stable, "modes": modes}

    def forecast_interval(self, history: np.ndarray, steps: int = 1, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = self.predict(history, steps)
        psi = np.zeros(steps)
        psi[0] = 1.0
        for h in range(1, steps):
            s = 0.0
            for i in range(1, min(h, self.p) + 1):
                s -= self.ar_coefs[i - 1] * psi[h - i]
            psi[h] = s

        var = np.cumsum(psi ** 2) * self.sigma_e ** 2
        std = np.sqrt(var)
        z = 1.96 if confidence >= 0.95 else 1.645
        lower = pred - z * std
        upper = pred + z * std
        return pred, lower, upper
