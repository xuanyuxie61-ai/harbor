
import numpy as np
from typing import List, Tuple
from math import comb






def mono_upto_enum(m: int, n: int) -> int:
    return comb(m + n, n)


def mono_next_grlex(m: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int32).copy()
    j = m - 1
    while j >= 0 and x[j] == 0:
        j -= 1
    if j < 0:
        x[m - 1] = 1
        return x
    if j == 0:
        t = x[0]
        x[0] = 0
        x[1] = t + 1
    elif j > 0:
        x[j - 1] += 1
        t = x[j] - 1
        x[j] = 0
        x[m - 1] = t
    return x


def generate_monomials(m: int, degree: int) -> np.ndarray:
    monos = []
    for d in range(degree + 1):
        def backtrack(remaining, current, pos):
            if pos == m - 1:
                monos.append(current + [remaining])
                return
            for v in range(remaining + 1):
                backtrack(remaining - v, current + [v], pos + 1)
        backtrack(d, [], 0)
    return np.array(monos, dtype=np.int32)






def evaluate_monomial(exponents: np.ndarray, point: np.ndarray) -> float:
    val = 1.0
    for i, e in enumerate(exponents):
        if e > 0:
            pi = float(point[i])

            if abs(pi) < 1e-15 and e == 0:
                continue
            val *= pi ** e
    return val


def evaluate_polynomial(coeffs: np.ndarray, monomials: np.ndarray,
                        point: np.ndarray) -> float:
    total = 0.0
    for c, alpha in zip(coeffs, monomials):
        total += c * evaluate_monomial(alpha, point)
    return total


def polynomial_gradient(coeffs: np.ndarray, monomials: np.ndarray,
                        point: np.ndarray) -> np.ndarray:
    m = monomials.shape[1]
    grad = np.zeros(m, dtype=np.float64)
    for c, alpha in zip(coeffs, monomials):
        for j in range(m):
            if alpha[j] > 0:
                beta = alpha.copy()
                beta[j] -= 1
                grad[j] += c * (alpha[j]) * evaluate_monomial(beta, point)
    return grad






def compute_polynomial_descriptors(atoms: np.ndarray, degree: int = 3) -> np.ndarray:
    n = atoms.shape[0]
    if n < 2:
        return np.zeros(mono_upto_enum(3, degree), dtype=np.float64)


    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(atoms[i] - atoms[j])
            r = max(r, 0.5)
            dists.append([r, 1.0 / r, r ** 2])
    dists = np.array(dists, dtype=np.float64)

    m = 3
    monos = generate_monomials(m, degree)
    n_mono = monos.shape[0]
    desc = np.zeros(n_mono, dtype=np.float64)
    coeffs = np.ones(n_mono, dtype=np.float64)

    for q in dists:
        vals = np.array([evaluate_monomial(monos[k], q) for k in range(n_mono)], dtype=np.float64)
        desc += vals


    norm = np.linalg.norm(desc)
    if norm > 1e-12:
        desc = desc / norm
    return desc


def polynomial_axpy(a: float, p_coeffs: np.ndarray, q_coeffs: np.ndarray) -> np.ndarray:
    return a * p_coeffs + q_coeffs


def falling_factorial(x: float, n: int) -> float:
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    val = 1.0
    for k in range(n):
        val *= (x - k)
    return val
