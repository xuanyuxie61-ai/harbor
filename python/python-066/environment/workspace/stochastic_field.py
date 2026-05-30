
import numpy as np
from math import exp, sqrt, pi, ceil, log2







_IRREDUCIBLE_POLYS = [
    3,
    7,
    11,
    19,
    37,
    67,
    131,
    285,
    529,
    1033,
    2053,
    4179,
    8219,
    16427,
]


def _degree_poly(p: int) -> int:
    if p == 0:
        return -1
    return p.bit_length() - 1


def _plymul2(p: int, q: int) -> int:
    result = 0
    while q:
        if q & 1:
            result ^= p
        p <<= 1
        q >>= 1
    return result


def _plymod2(p: int, m: int) -> int:
    deg_m = _degree_poly(m)
    if deg_m < 0:
        raise ZeroDivisionError
    while _degree_poly(p) >= deg_m:
        shift = _degree_poly(p) - deg_m
        p ^= m << shift
    return p


def _calcv2(poly: int, degree: int) -> list[int]:
    v = [0] * (degree + 1)

    for j in range(1, degree + 1):
        v[j] = _plymod2(1 << (j - 1), poly)
    return v


def _calcc2(dim: int, maxbits: int = 32) -> np.ndarray:
    C = np.zeros((dim, maxbits, maxbits), dtype=np.uint32)
    for d in range(dim):
        poly = _IRREDUCIBLE_POLYS[d % len(_IRREDUCIBLE_POLYS)]
        deg = _degree_poly(poly)
        v = _calcv2(poly, deg)



        for j in range(maxbits):
            if j < deg:

                m = (1 << (j + 1)) | (v[j + 1] << 1)
            else:


                m = C[d, j - deg, 0] if j >= deg else 1
                for k in range(1, deg):
                    if (poly >> k) & 1:
                        m ^= C[d, j - k, 0] if j >= k else 0
                m ^= C[d, j - deg, 0] if j >= deg else 1

            for r in range(maxbits):
                if (m >> r) & 1:
                    C[d, j, r] = 1
    return C


class NiederreiterGenerator:

    def __init__(self, dim: int, seed: int = 0):
        if dim < 1 or dim > len(_IRREDUCIBLE_POLYS):
            raise ValueError(f"维度必须在 1 到 {len(_IRREDUCIBLE_POLYS)} 之间")
        self.dim = dim
        self.seed = seed
        self._n = seed
        self._maxbits = 32

        self._directions = self._init_directions()

    def _init_directions(self) -> np.ndarray:
        directions = np.zeros((self.dim, self._maxbits), dtype=np.uint32)
        for d in range(self.dim):
            poly = _IRREDUCIBLE_POLYS[d]
            deg = _degree_poly(poly)

            for j in range(deg):
                directions[d, j] = 1 << (self._maxbits - 1 - j)

            for j in range(deg, self._maxbits):

                val = directions[d, j - deg]
                for k in range(1, deg):
                    if (poly >> k) & 1:
                        val ^= directions[d, j - k]
                directions[d, j] = val >> 1
        return directions

    def next_point(self) -> np.ndarray:
        x = np.zeros(self.dim)
        g = self._n ^ (self._n >> 1)
        for d in range(self.dim):
            val = np.uint32(0)
            for j in range(self._maxbits):
                if (g >> j) & 1:
                    val ^= self._directions[d, j]
            x[d] = val / (1 << self._maxbits)
        self._n += 1
        return x

    def generate(self, N: int) -> np.ndarray:
        points = np.zeros((N, self.dim))
        for i in range(N):
            points[i, :] = self.next_point()
        return points






def rejection_sample_1d(pdf: callable, pdf_max: float, a: float, b: float,
                        N: int, seed: int = 42) -> np.ndarray:
    if pdf_max <= 0:
        raise ValueError("pdf_max 必须为正")
    if a >= b:
        raise ValueError("必须满足 a < b")
    if N < 1:
        raise ValueError("N 必须为正整数")

    rng = np.random.default_rng(seed)
    samples = np.zeros(N)
    accepted = 0
    max_iter = N * ceil(2.0 * pdf_max * (b - a)) + 1000
    total_tried = 0

    while accepted < N and total_tried < max_iter:
        x_cand = rng.uniform(a, b, size=N)
        y_cand = rng.uniform(0.0, pdf_max, size=N)
        for i in range(N):
            if accepted >= N:
                break
            total_tried += 1
            if y_cand[i] <= pdf(x_cand[i]):
                samples[accepted] = x_cand[i]
                accepted += 1

    if accepted < N:

        samples[accepted:] = rng.uniform(a, b, size=N - accepted)
    return samples


def lognormal_k_field(x: np.ndarray, mu: float, sigma: float,
                      correlation_length: float, seed: int = 42) -> np.ndarray:
    if correlation_length <= 0:
        raise ValueError("相关长度必须为正")
    if sigma < 0:
        raise ValueError("标准差必须非负")
    n = len(x)
    if n == 0:
        raise ValueError("坐标数组不能为空")

    rng = np.random.default_rng(seed)

    dx = np.subtract.outer(x, x)
    C = sigma ** 2 * np.exp(-np.abs(dx) / correlation_length)

    C += np.eye(n) * 1e-12

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    Z = rng.standard_normal(n)
    Y = mu + L @ Z
    K = np.exp(Y)
    return K


def quasirandom_k_parameters(N: int, dim: int = 3,
                             mu_bounds: tuple = (0.0, 1.0),
                             sigma_bounds: tuple = (0.1, 2.0),
                             lambda_bounds: tuple = (0.5, 5.0)) -> np.ndarray:
    if N < 1:
        raise ValueError("N 必须为正整数")
    gen = NiederreiterGenerator(dim=dim, seed=0)
    u = gen.generate(N)

    mu_vals = mu_bounds[0] + u[:, 0] * (mu_bounds[1] - mu_bounds[0])
    sigma_vals = sigma_bounds[0] + u[:, 1] * (sigma_bounds[1] - sigma_bounds[0])
    lambda_vals = lambda_bounds[0] + u[:, 2] * (lambda_bounds[1] - lambda_bounds[0])

    return np.column_stack([mu_vals, sigma_vals, lambda_vals])


if __name__ == "__main__":

    gen = NiederreiterGenerator(dim=3)
    pts = gen.generate(100)
    assert pts.shape == (100, 3)
    assert np.all((pts >= 0) & (pts <= 1))

    x_grid = np.linspace(0, 10, 50)
    K = lognormal_k_field(x_grid, mu=-2.0, sigma=1.0, correlation_length=2.0)
    assert K.shape == x_grid.shape
    assert np.all(K > 0)


    def chebyshev2_pdf(x):
        return (2.0 / np.pi) * np.sqrt(np.maximum(0.0, 1.0 - x ** 2))

    samples = rejection_sample_1d(chebyshev2_pdf, 2.0 / np.pi, -1.0, 1.0, 200)
    assert len(samples) == 200
    print("stochastic_field: 自测试通过")
