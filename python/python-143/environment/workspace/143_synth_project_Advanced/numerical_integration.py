
import numpy as np
from typing import Tuple, Optional


class ChebyshevBasis:

    @staticmethod
    def evaluate(m: int, x: np.ndarray) -> np.ndarray:
        n = len(x)
        V = np.zeros((n, m))
        V[:, 0] = 1.0
        if m > 1:
            V[:, 1] = x
        for j in range(2, m):
            V[:, j] = 2.0 * x * V[:, j - 1] - V[:, j - 2]
        return V

    @staticmethod
    def moments(m: int, a: float, b: float) -> np.ndarray:
        mu = np.zeros(m)
        scale = (b - a) / 2.0

        mu[0] = scale * 2.0

        if m > 1:
            mu[1] = 0.0


        for j in range(2, m):
            val_at_1 = 1.0 / (j + 1) - 1.0 / (j - 1)
            val_at_m1 = ((-1.0) ** (j + 1)) / (j + 1) - ((-1.0) ** (j - 1)) / (j - 1)
            mu[j] = scale * 0.5 * (val_at_1 - val_at_m1)
        return mu


class FeketeQuadrature:

    def __init__(self, a: float = -1.0, b: float = 1.0):
        self.a = a
        self.b = b

    def compute_fekete_points(self, m: int, n_samples: int = 200) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if n_samples < m:
            n_samples = m



        k = np.arange(1, n_samples + 1)
        x_candidates = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_samples))

        x_candidates = 0.5 * (self.b - self.a) * x_candidates + 0.5 * (self.a + self.b)



        t_candidates = (2.0 * x_candidates - self.a - self.b) / (self.b - self.a)
        V_tall = ChebyshevBasis.evaluate(m, t_candidates)
        V_wide = V_tall.T


        mom = ChebyshevBasis.moments(m, self.a, self.b)



        w, _, _, _ = np.linalg.lstsq(V_wide, mom, rcond=None)


        tol = 1e-10 * np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else 1e-10
        ind = np.where(np.abs(w) > tol)[0]
        nf = len(ind)

        if nf == 0:

            ind = np.argsort(np.abs(w))[-m:]
            nf = len(ind)

        xf = x_candidates[ind]
        wf = w[ind]
        vf = V_tall[ind, :]


        sum_w = np.sum(wf)
        if abs(sum_w) > 1e-12:
            wf = wf * (self.b - self.a) / sum_w

        return nf, xf, wf, vf

    def integrate(self, f: callable, m: int = 10, n_samples: int = 200) -> float:
        nf, xf, wf, _ = self.compute_fekete_points(m, n_samples)
        return float(np.sum(wf * f(xf)))


class FinancialExpectation:

    @staticmethod
    def expected_payoff_fekete(payoff_func: callable,
                                m: int = 15,
                                a: float = -5.0,
                                b: float = 5.0) -> float:
        fq = FeketeQuadrature(a, b)
        nf, xf, wf, _ = fq.compute_fekete_points(m)


        phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xf ** 2)


        composite_weights = wf * phi
        return float(np.sum(composite_weights * payoff_func(xf)))

    @staticmethod
    def expected_shortfall_integral(returns: np.ndarray,
                                     confidence: float = 0.95,
                                     m: int = 20) -> float:
        if len(returns) < 10:
            return 0.0


        from scipy.stats import gaussian_kde
        kde = gaussian_kde(returns)
        var = -np.percentile(returns, (1.0 - confidence) * 100.0)


        a = np.min(returns) - 3.0 * np.std(returns)
        b = -var

        if b <= a:
            return 0.0

        fq = FeketeQuadrature(a, b)
        nf, xf, wf, _ = fq.compute_fekete_points(m)

        integrand = (-xf) * kde(xf)
        integral = np.sum(wf * integrand)
        return float(integral / (1.0 - confidence))


class MultidimensionalQuadrature:

    @staticmethod
    def tensor_product_2d(f: callable,
                          m1: int = 8, m2: int = 8,
                          a1: float = -1.0, b1: float = 1.0,
                          a2: float = -1.0, b2: float = 1.0) -> float:
        fq1 = FeketeQuadrature(a1, b1)
        nf1, x1, w1, _ = fq1.compute_fekete_points(m1)

        fq2 = FeketeQuadrature(a2, b2)
        nf2, x2, w2, _ = fq2.compute_fekete_points(m2)

        total = 0.0
        for i in range(nf1):
            for j in range(nf2):
                total += w1[i] * w2[j] * f(x1[i], x2[j])

        return float(total)
