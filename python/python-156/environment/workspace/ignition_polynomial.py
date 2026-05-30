
import numpy as np


def poly_eval(c, x):
    d = len(c) - 1
    if np.isscalar(x):
        value = c[0]
        xi = 1.0
        for i in range(1, d + 1):
            xi *= x
            value += c[i] * xi
    else:
        x = np.asarray(x)
        value = c[0] * np.ones_like(x, dtype=complex)
        xi = np.ones_like(x, dtype=complex)
        for i in range(1, d + 1):
            xi *= x
            value += c[i] * xi
    return value


def wdk_roots(c, tol=1.0e-12, max_iter=1000):
    c = np.asarray(c, dtype=complex)
    d = len(c) - 1

    if d < 1:
        raise ValueError("多项式次数必须 >= 1")
    if abs(c[d]) < 1.0e-30:
        raise ValueError("首项系数不能为零")


    R = 1.0 + np.max(np.abs(c[:-1] / c[d]))


    theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
    roots = R * np.exp(1.0j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            zi = roots_old[i]
            denom = 1.0 + 0.0j
            for j in range(d):
                if i != j:
                    diff = zi - roots[j]
                    if abs(diff) < 1.0e-30:
                        diff = 1.0e-30 * (1.0 + 0.0j)
                    denom *= diff

            if abs(denom) < 1.0e-30:
                denom = 1.0e-30 * (1.0 + 0.0j)

            roots[i] = zi - poly_eval(c, zi) / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            return roots, True

    return roots, False


def critical_damkohler_polynomial(Ze, beta, sigma, order=4):
    if Ze <= 0.0 or beta <= 0.0 or sigma <= 0.0:
        raise ValueError("Ze, beta, sigma 必须为正数")

    c = np.zeros(order + 1, dtype=complex)


    c[order] = 1.0 + 0.0j
    c[order - 1] = -3.0 / Ze
    c[order - 2] = 2.0 / (Ze ** 2) - beta * (1.0 - sigma) / (2.0 * Ze)
    c[order - 3] = -0.5 / (Ze ** 3)
    c[0] = -np.e / (Ze ** 2)


    for k in range(1, order - 3):
        c[k] = c[0] * ((-1.0) ** k) / np.math.factorial(k) * (1.0 / Ze) ** k

    return c


def analyze_ignition_extinction(Ze=8.0, beta=10.0, sigma=0.135):
    c = critical_damkohler_polynomial(Ze, beta, sigma, order=4)
    roots, converged = wdk_roots(c, tol=1.0e-14, max_iter=500)


    real_positive = []
    for r in roots:
        if np.imag(r) < 1.0e-6 and np.real(r) > 0:
            real_positive.append(float(np.real(r)))

    Da_cr = min(real_positive) if real_positive else float(np.real(roots[0]))
    if Da_cr <= 0:
        Da_cr = np.e / (Ze ** 2)

    results = {
        'Zel_dovich_number': Ze,
        'activation_energy_beta': beta,
        'temperature_ratio_sigma': sigma,
        'polynomial_coefficients': c.tolist(),
        'roots': roots.tolist(),
        'converged': converged,
        'critical_Damkohler': Da_cr,
        'ignition_limit': Da_cr * 0.8,
        'extinction_limit': Da_cr * 1.2,
    }

    return results
