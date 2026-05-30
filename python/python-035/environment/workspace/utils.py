import numpy as np
from constants import EPS, TINY, MAX_ITER




def lu_factor_scaled(a):
    n = a.shape[0]
    lu = a.astype(float).copy()
    pivot = np.arange(n, dtype=int)
    scale = np.zeros(n)
    iflag = 0


    for i in range(n):
        row_max = np.max(np.abs(lu[i, :]))
        if row_max < TINY:
            iflag = 1
            row_max = TINY
        scale[i] = row_max

    for k in range(n - 1):

        max_ratio = 0.0
        pivot_row = k
        for i in range(k, n):
            ratio = abs(lu[i, k]) / scale[i]
            if ratio > max_ratio:
                max_ratio = ratio
                pivot_row = i

        if max_ratio < TINY:
            iflag = 1
            break


        if pivot_row != k:
            lu[[k, pivot_row], :] = lu[[pivot_row, k], :]
            scale[[k, pivot_row]] = scale[[pivot_row, k]]
            pivot[[k, pivot_row]] = pivot[[pivot_row, k]]


        if abs(lu[k, k]) > TINY:
            for i in range(k + 1, n):
                lu[i, k] /= lu[k, k]
                for j in range(k + 1, n):
                    lu[i, j] -= lu[i, k] * lu[k, j]

    if abs(lu[n - 1, n - 1]) < TINY:
        iflag = 1

    return lu, pivot, iflag


def lu_solve(lu, pivot, b):
    n = lu.shape[0]
    y = b.astype(float).copy()


    y = y[pivot]


    for i in range(1, n):
        for j in range(i):
            y[i] -= lu[i, j] * y[j]


    x = y.copy()
    for i in range(n - 1, -1, -1):
        if abs(lu[i, i]) < TINY:
            x[i] = 0.0
        else:
            for j in range(i + 1, n):
                x[i] -= lu[i, j] * x[j]
            x[i] /= lu[i, i]
    return x





def inverse_iteration(a, shift, max_iter=100, tol=1.0e-12):
    n = a.shape[0]
    a_shifted = a - shift * np.eye(n)
    lu, pivot, iflag = lu_factor_scaled(a_shifted)
    if iflag != 0:
        return shift, np.zeros(n), False

    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    eigval = shift

    for _ in range(max_iter):
        v_new = lu_solve(lu, pivot, v)
        norm_v = np.linalg.norm(v_new)
        if norm_v < TINY:
            break
        v_new /= norm_v
        eigval_new = float(v_new.T @ (a @ v_new))
        if abs(eigval_new - eigval) < tol * max(1.0, abs(eigval_new)):
            return eigval_new, v_new, True
        eigval = eigval_new
        v = v_new
    return eigval, v, False





def horner_eval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    p = np.zeros_like(x)
    for c in reversed(coeffs):
        p = c + x * p
    return p





def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def cooley_tukey_fft(x):
    x = np.asarray(x, dtype=complex)
    N = x.size
    if N <= 1:
        return x.copy()
    
    factors = prime_factors(N)
    if len(factors) == 0 or (len(factors) == 1 and factors[0] == N):

        return np.fft.fft(x)
    

    if N % 2 == 0:
        even = cooley_tukey_fft(x[0::2])
        odd = cooley_tukey_fft(x[1::2])
        twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        return np.concatenate([even + twiddle * odd, even - twiddle * odd])
    else:

        return np.fft.fft(x)





def bisection(f, a, b, tol=1.0e-12, max_iter=100):
    fa = float(f(a))
    fb = float(f(b))
    if fa * fb > 0:
        return None, {"status": "same_sign", "iter": 0}
    
    for k in range(max_iter):
        c = (a + b) / 2.0
        fc = float(f(c))
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, {"status": "converged", "iter": k + 1, "residual": abs(fc)}
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0, {"status": "max_iter", "iter": max_iter}


def muller_method(f, x0, x1, x2, tol=1.0e-12, max_iter=100):
    f0, f1, f2 = float(f(x0)), float(f(x1)), float(f(x2))
    for k in range(max_iter):
        h0, h1 = x0 - x2, x1 - x2
        if abs(h0) < TINY or abs(h1) < TINY:
            break
        d0 = (f0 - f2) / h0
        d1 = (f1 - f2) / h1
        a = (d0 - d1) / (h0 - h1)
        w = d0 - a * h0
        disc = w * w - 4.0 * a * f2
        if disc < 0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        if abs(w + sqrt_disc) > abs(w - sqrt_disc):
            den = w + sqrt_disc
        else:
            den = w - sqrt_disc
        if abs(den) < TINY:
            break
        dx = -2.0 * f2 / den
        x3 = x2 + dx
        f3 = float(f(x3))
        if abs(f3) < tol or abs(dx) < tol * max(1.0, abs(x3)):
            return x3, {"status": "converged", "iter": k + 1, "residual": abs(f3)}
        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3
    return x2, {"status": "max_iter", "iter": max_iter}





def rk2_step(f, t, y, h):
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h, y + h * k1), dtype=float)
    return y + 0.5 * h * (k1 + k2)


def rk2_integrate(f, t_span, y0, n_steps):
    t0, t1 = t_span
    h = (t1 - t0) / n_steps
    y = np.asarray(y0, dtype=float)
    t = t0
    t_array = [t]
    y_array = [y.copy()]
    for _ in range(n_steps):
        y = rk2_step(f, t, y, h)
        t += h
        t_array.append(t)
        y_array.append(y.copy())
    return np.array(t_array), np.array(y_array)





def safe_divide(a, b, default=0.0):
    b = np.asarray(b, dtype=float)
    result = np.where(np.abs(b) > TINY, np.asarray(a, dtype=float) / b, default)
    return result


def safe_sqrt(x, default=0.0):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.0, np.sqrt(x), default)
