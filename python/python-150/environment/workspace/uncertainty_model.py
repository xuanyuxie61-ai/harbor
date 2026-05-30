
import numpy as np
from typing import Tuple






def error_function(x: float) -> float:
    x = float(x)
    if x < 0:
        return -error_function(-x)


    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return y


def incomplete_gamma(a: float, x: float) -> float:
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    gln = gammaln(a)
    ap = a
    sum_ = 1.0 / a
    del_ = sum_
    n = 1
    while n <= 10000:
        ap += 1.0
        del_ *= x / ap
        sum_ += del_
        if abs(del_) < abs(sum_) * 1e-10:
            break
        n += 1
    return sum_ * np.exp(-x + a * np.log(x) - gln)


def gammaln(x: float) -> float:
    if x <= 0:
        return np.inf
    p = np.array([
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ], dtype=np.float64)
    x = float(x)
    if x < 0.5:
        return np.log(np.pi) - np.log(np.sin(np.pi * x)) - gammaln(1.0 - x)
    x -= 1.0
    z = x + 0.5 + 7.0
    A = 0.99999999999980993
    for i, pi in enumerate(p):
        A += pi / (x + i + 1.0)

    return 0.5 * np.log(2.0 * np.pi) + np.log(A) + (x + 0.5) * np.log(z) - z


def digamma(x: float) -> float:
    if x <= 0:
        return np.nan
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += np.log(x) - 0.5 * inv_x - inv_x2 * (
        1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0))
    )
    return result






class EvidentialRegressor:

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)


        limit2 = np.sqrt(6.0 / (hidden_dim + 4))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, 4))
        self.b2 = np.zeros(4)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self._forward(x)

        gamma = raw[:, 0]
        nu = np.log(1.0 + np.exp(raw[:, 1])) + 1e-6
        alpha = np.log(1.0 + np.exp(raw[:, 2])) + 1.01
        beta = np.log(1.0 + np.exp(raw[:, 3])) + 1e-6
        return gamma, nu, alpha, beta

    def nig_nll(self, gamma: np.ndarray, nu: np.ndarray,
                alpha: np.ndarray, beta: np.ndarray, y: np.ndarray) -> float:



        raise NotImplementedError("nig_nll is not implemented")

    def uncertainty(self, alpha: np.ndarray, beta: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        safe_alpha = np.where(alpha > 1.0, alpha, 1.01)
        aleatoric = beta / (safe_alpha - 1.0)
        epistemic = beta / (nu * (safe_alpha - 1.0))
        return aleatoric, epistemic

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]






def erf_correlation_kernel(distances: np.ndarray, r_cut: float, lengthscale: float = 0.5) -> np.ndarray:
    arg = (r_cut - distances) / (np.sqrt(2.0) * lengthscale)
    k = 0.5 * (1.0 + np.array([error_function(float(a)) for a in arg]))
    k = np.where(distances > r_cut, 0.0, k)
    return k






def multivariate_normal_sample(mean: np.ndarray, cov: np.ndarray, n_samples: int) -> np.ndarray:
    d = len(mean)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
    eps = np.random.randn(n_samples, d)
    return mean.reshape(1, d) + eps @ L.T
