
import numpy as np
from typing import Tuple


def jacobi_polynomial(
    m: int, n: int, alpha: float, beta: float, x: np.ndarray
) -> np.ndarray:
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha 和 beta 必须 > -1")
    if n < 0:
        raise ValueError("jacobi_polynomial: n 必须 >= 0")

    x = np.asarray(x, dtype=float).ravel()
    m_actual = x.shape[0]
    if m_actual != m:

        pass

    v = np.zeros((m_actual, n + 1))
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * \
             (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)

        if abs(c1) < 1e-15:
            c1 = 1e-15

        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v


def imtqlx(n: int, d: np.ndarray, e: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = d.copy()
    e = e.copy()
    z = z.copy()
    itn = 30
    prec = np.finfo(float).eps

    if n == 1:
        return d, z

    e[n - 1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = l
            while m < n - 1:
                if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                    break
                m += 1

            p = d[l]
            if m == l:
                break

            if j == itn:
                raise RuntimeError("imtqlx: 迭代次数超限")
            j += 1

            g = (d[l + 1] - p) / (2.0 * e[l])
            r = np.sqrt(g * g + 1.0)
            t = g - r if g < 0.0 else g + r

            g = d[m] - p + e[l] / (g + t)
            s, c, p_val = 1.0, 1.0, 0.0
            mml = m - l

            for ii in range(1, mml + 1):
                i = m - ii
                f = s * e[i]
                b = c * e[i]

                if abs(g) <= abs(f):
                    c_val = g / f
                    r_val = np.sqrt(c_val * c_val + 1.0)
                    e[i + 1] = f * r_val
                    s_val = 1.0 / r_val
                    c_val *= s_val
                else:
                    s_val = f / g
                    r_val = np.sqrt(s_val * s_val + 1.0)
                    e[i + 1] = g * r_val
                    c_val = 1.0 / r_val
                    s_val *= c_val

                g = d[i + 1] - p_val
                r_val = (d[i] - g) * s_val + 2.0 * c_val * b
                p_val = s_val * r_val
                d[i + 1] = g + p_val
                g = c_val * r_val - b
                f_val = z[i + 1]
                z[i + 1] = s_val * z[i] + c_val * f_val
                z[i] = c_val * z[i] - s_val * f_val
                s, c = s_val, c_val

            d[l] -= p_val
            e[l] = g
            e[m] = 0.0


    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            p = z[i]
            z[i] = z[k]
            z[k] = p

    return d, z


def gauss_legendre_quadrature(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("gauss_legendre_quadrature: n >= 1")

    bj = np.zeros(n)
    for i in range(1, n + 1):
        bj[i - 1] = (i * i) / (4.0 * i * i - 1.0)
    bj = np.sqrt(bj)

    d = np.zeros(n)
    z = np.zeros(n)
    z[0] = np.sqrt(2.0)

    d, z = imtqlx(n, d, bj, z)
    w = z ** 2


    x = 0.5 * ((1.0 - d) * a + (d + 1.0) * b)
    w = w * (b - a) / 2.0

    return x, w


def spectral_galerkin_rhs(
    f_func: callable, n_modes: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:

    n_quad = n_modes + 5
    x_gl, w_gl = gauss_legendre_quadrature(n_quad)



    fx = f_func(x_gl)
    fx = np.asarray(fx, dtype=float).ravel()


    jac_vals = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl)

    F = np.zeros(n_modes)
    for i in range(n_modes):


        weight_factor = ((1.0 - x_gl) ** alpha) * ((1.0 + x_gl) ** beta)
        F[i] = np.sum(w_gl * fx * jac_vals[:, i] * weight_factor)

    return F


def build_spectral_stiffness_matrix(
    n_modes: int, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    n_quad = n_modes + 5
    x_gl, w_gl = gauss_legendre_quadrature(n_quad)

    jac_vals = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl)


    dx = 1e-6
    jac_plus = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl + dx)
    jac_minus = jacobi_polynomial(n_quad, n_modes - 1, alpha, beta, x_gl - dx)
    d_jac = (jac_plus - jac_minus) / (2.0 * dx)

    K = np.zeros((n_modes, n_modes))
    weight_factor = ((1.0 - x_gl) ** alpha) * ((1.0 + x_gl) ** beta)

    for i in range(n_modes):
        for j in range(i, n_modes):
            val = np.sum(w_gl * d_jac[:, i] * d_jac[:, j] * weight_factor)
            K[i, j] = val
            K[j, i] = val

    return K


def solve_spectral_diffusion(
    f_func: callable, n_modes: int, diffusion_coeff: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    K = build_spectral_stiffness_matrix(n_modes)
    F = spectral_galerkin_rhs(f_func, n_modes)



    A = diffusion_coeff * K + 1e-10 * np.eye(n_modes)
    coeffs = np.linalg.solve(A, F)

    x_plot = np.linspace(-1.0, 1.0, 201)
    jac_plot = jacobi_polynomial(201, n_modes - 1, 0.0, 0.0, x_plot)
    u_approx = jac_plot @ coeffs

    return x_plot, u_approx
