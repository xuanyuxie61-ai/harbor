
import numpy as np
from typing import Tuple, Optional






def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)

    if n == 1:
        return x, np.array([1.0, 1.0])

    w = np.zeros(n + 1)
    interior = np.arange(1, n)
    v = np.ones(n - 1)

    if n % 2 == 0:
        w[0] = 1.0 / (n ** 2 - 1)
        w[n] = 1.0 / (n ** 2 - 1)
        for k in range(1, n // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k ** 2 - 1)
        v = v - np.cos(n * theta[interior]) / (n ** 2 - 1)
    else:
        w[0] = 1.0 / (n ** 2)
        w[n] = 1.0 / (n ** 2)
        for k in range(1, (n + 1) // 2):
            v = v - 2.0 * np.cos(2 * k * theta[interior]) / (4 * k ** 2 - 1)

    w[interior] = 2.0 * v / n
    return x, w


def fejer1_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    j = np.arange(1, n + 1)
    x = np.cos((2.0 * j - 1.0) * np.pi / (2.0 * n))


    N = n
    k = np.arange(1, N)


    alpha = np.ones(N)
    alpha[1:] = 2.0 / (1.0 - 4.0 * k ** 2)

    theta = (2.0 * j - 1.0) * np.pi / (2.0 * N)
    w = np.zeros(N)
    for idx in range(N):
        w[idx] = np.sum(alpha * np.cos(np.arange(N) * theta[idx]))
    w *= (2.0 / N)
    return x, w


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])


    i = np.arange(1.0, n)
    beta = i / np.sqrt(4.0 * i ** 2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)


    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals

    w = 2.0 * (eigvecs[0, :] ** 2)
    return x, w


def composite_quad_rule(f, a: float, b: float, n_sub: int = 8, n_point: int = 8,
                        rule: str = "cc") -> float:
    if a >= b:
        if a == b:
            return 0.0
        a, b = b, a
    if n_sub < 1 or n_point < 1:
        raise ValueError("n_sub and n_point must be >= 1")

    if rule == "cc":
        x_local, w_local = clenshaw_curtis_nodes_weights(n_point)
    elif rule == "fejer1":
        x_local, w_local = fejer1_nodes_weights(n_point)
    elif rule == "gl":
        x_local, w_local = gauss_legendre_nodes_weights(n_point)
    else:
        raise ValueError(f"Unknown rule: {rule}")


    h = (b - a) / n_sub
    total = 0.0
    for s in range(n_sub):
        a_s = a + s * h
        b_s = a_s + h

        t = 0.5 * (b_s - a_s) * x_local + 0.5 * (b_s + a_s)
        jac = 0.5 * (b_s - a_s)
        ft = np.array([f(ti) for ti in t])
        total += np.sum(w_local * ft) * jac
    return float(total)






def hermite_polynomial_prob(n: int, x: np.ndarray) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    x = np.asarray(x)
    if n == 0:
        return np.ones((1, x.size))
    if n == 1:
        return np.vstack([np.ones(x.shape), x])

    H = np.zeros((n + 1, x.size))
    H[0, :] = 1.0
    H[1, :] = x
    for k in range(2, n + 1):
        H[k, :] = x * H[k - 1, :] - (k - 1) * H[k - 2, :]
    return H


def hermite_polynomial_derivative(n: int, x: np.ndarray) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.zeros_like(x)
    H = hermite_polynomial_prob(n - 1, x)
    return n * H[n - 1, :]


def gaussian_hermite_expand(coeffs: np.ndarray, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    N = len(coeffs)
    z = x / sigma
    H = hermite_polynomial_prob(N - 1, z)
    result = np.zeros_like(x, dtype=float)
    for k in range(N):
        result += coeffs[k] * H[k, :] * np.exp(-0.5 * z ** 2)
    return result






def cholesky_decompose(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    U = np.zeros((n, n))
    for i in range(n):
        sum_sq = np.dot(U[:i, i], U[:i, i])
        diag = A[i, i] - sum_sq
        if diag <= tol:
            raise ValueError(f"Matrix is not positive definite (diag[{i}]={diag:.3e})")
        U[i, i] = np.sqrt(diag)
        for j in range(i + 1, n):
            sum_prod = np.dot(U[:i, i], U[:i, j])
            U[i, j] = (A[i, j] - sum_prod) / U[i, i]
    return U


def spd_inverse(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    U = cholesky_decompose(A, tol)
    n = U.shape[0]
    Uinv = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        Uinv[i, i] = 1.0 / U[i, i]
        for j in range(i + 1, n):
            s = np.dot(U[i, i + 1:j + 1], Uinv[i + 1:j + 1, j])
            Uinv[i, j] = -s / U[i, i]

    return Uinv @ Uinv.T






def prime_factors(n: int) -> list:
    if n < 2:
        return []
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append(temp)
    return factors


def next_fftfriendly_size(n: int) -> int:
    if n < 1:
        return 1
    candidate = n
    while True:
        facs = prime_factors(candidate)

        if all(p in (2, 3, 5) for p in facs):
            return candidate
        candidate += 1






def safe_sqrt(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    x = np.asarray(x)
    return np.sqrt(np.maximum(x, eps))


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    result = np.zeros_like(a, dtype=float)
    mask = np.abs(b) > eps
    result[mask] = a[mask] / b[mask]
    return result


def givens_rotation(a: float, b: float) -> Tuple[float, float]:
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau ** 2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau ** 2)
        s = c * tau
    return c, s





if __name__ == "__main__":

    f_test = lambda x: np.exp(x)
    exact = np.exp(1) - np.exp(-1)
    for rule_name in ["cc", "fejer1", "gl"]:
        val = composite_quad_rule(f_test, -1.0, 1.0, n_sub=2, n_point=8, rule=rule_name)
        print(f"{rule_name}: {val:.12f}, err={abs(val-exact):.2e}")


    x = np.linspace(-3, 3, 7)
    H = hermite_polynomial_prob(4, x)
    print("He_4(0)=", H[4, 3])


    A = np.array([[4.0, 2.0, 1.0],
                  [2.0, 5.0, 3.0],
                  [1.0, 3.0, 6.0]])
    U = cholesky_decompose(A)
    print("Cholesky residual:", np.max(np.abs(A - U.T @ U)))


    print("prime_factors(360)=", prime_factors(360))
    print("next_fftfriendly(127)=", next_fftfriendly_size(127))
