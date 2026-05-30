
import numpy as np







def cosine_integral(x: float) -> float:
    x = float(x)
    if x == 0.0:
        return -np.inf

    ax = abs(x)
    gamma_e = 0.5772156649015328606

    if ax <= 16.0:

        x2 = x * x
        term = -x2 / 4.0
        s = term
        k = 2
        while abs(term) > 1e-16:
            term *= -x2 / ((2 * k) * (2 * k - 1))
            term /= (2.0 * k)
            s += term
            k += 1
            if k > 100:
                break
        return gamma_e + np.log(ax) + s
    elif ax <= 32.0:




        nseg = 200
        dt = ax / nseg
        integral = 0.0
        for i in range(nseg):
            t0 = i * dt
            t1 = (i + 1) * dt

            f0 = (np.cos(t0) - 1.0) / t0 if t0 > 1e-12 else 0.0
            fm = (np.cos(0.5 * (t0 + t1)) - 1.0) / (0.5 * (t0 + t1)) if (t0 + t1) > 2e-12 else 0.0
            f1 = (np.cos(t1) - 1.0) / t1
            integral += dt / 6.0 * (f0 + 4.0 * fm + f1)
        return gamma_e + np.log(ax) + integral
    else:


        inv_x = 1.0 / ax
        sinx = np.sin(ax)
        cosx = np.cos(ax)

        f = 1.0 - 2.0 * inv_x * inv_x + 24.0 * inv_x**4
        g = inv_x - 6.0 * inv_x**3 + 120.0 * inv_x**5
        return sinx / ax * f - cosx / ax * g






def sine_integral(x: float) -> float:
    x = float(x)
    ax = abs(x)
    if ax < 1e-10:
        return x
    if ax <= 16.0:
        x2 = x * x
        term = x
        s = x
        k = 1
        while abs(term) > 1e-16:
            term *= -x2 / ((2 * k + 1) * (2 * k))
            s += term
            k += 1
            if k > 100:
                break
        return s
    else:

        inv_x = 1.0 / ax
        sinx = np.sin(ax)
        cosx = np.cos(ax)
        f = 1.0 - 2.0 * inv_x * inv_x
        g = inv_x - 6.0 * inv_x**3
        val = 0.5 * np.pi - cosx / ax * f - sinx / ax * g
        return val if x > 0 else -val







def spherical_bessel_j(l: int, x: float) -> float:
    x = float(x)
    if x == 0.0:
        return 1.0 if l == 0 else 0.0
    if l == 0:
        return np.sin(x) / x
    if l == 1:
        return np.sin(x) / (x * x) - np.cos(x) / x

    jlm2 = np.sin(x) / x
    jlm1 = np.sin(x) / (x * x) - np.cos(x) / x
    for ll in range(2, l + 1):
        jl = (2.0 * ll - 1.0) / x * jlm1 - jlm2
        jlm2, jlm1 = jlm1, jl
    return jlm1


def spherical_bessel_y(l: int, x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return -np.inf
    if l == 0:
        return -np.cos(x) / x
    if l == 1:
        return -np.cos(x) / (x * x) - np.sin(x) / x
    ylm2 = -np.cos(x) / x
    ylm1 = -np.cos(x) / (x * x) - np.sin(x) / x
    for ll in range(2, l + 1):
        yl = (2.0 * ll - 1.0) / x * ylm1 - ylm2
        ylm2, ylm1 = ylm1, yl
    return ylm1






def modified_spherical_bessel_i(l: int, x: float) -> float:
    x = float(x)
    if x == 0.0:
        return 1.0 if l == 0 else 0.0

    if l == 0:
        return np.sinh(x) / x
    if l == 1:
        return np.cosh(x) / x - np.sinh(x) / (x * x)
    ilm2 = np.sinh(x) / x
    ilm1 = np.cosh(x) / x - np.sinh(x) / (x * x)
    for ll in range(2, l + 1):
        il = -(2.0 * ll - 1.0) / x * ilm1 + ilm2
        ilm2, ilm1 = ilm1, il
    return ilm1


def modified_spherical_bessel_k(l: int, x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return np.inf
    if l == 0:
        return np.pi * 0.5 * np.exp(-x) / x
    if l == 1:
        return np.pi * 0.5 * np.exp(-x) * (1.0 / x + 1.0 / (x * x))
    klm2 = np.pi * 0.5 * np.exp(-x) / x
    klm1 = np.pi * 0.5 * np.exp(-x) * (1.0 / x + 1.0 / (x * x))
    for ll in range(2, l + 1):
        kl = (2.0 * ll - 1.0) / x * klm1 + klm2
        klm2, klm1 = klm1, kl
    return klm1







def magnetic_diffusion_kernel(r: float, rp: float, t: float, eta: float, l: int) -> float:
    if t <= 0.0 or eta <= 0.0 or r <= 0.0 or rp <= 0.0:
        return 0.0
    diff = 4.0 * eta * t
    gaussian = np.exp(-(r - rp) ** 2 / diff) / np.sqrt(np.pi * diff)
    bessel_part = spherical_bessel_j(l, r * rp / (2.0 * eta * t))
    decay = np.exp(-l * (l + 1.0) * eta * t / (r * r))
    return gaussian * bessel_part * decay





def safe_log(x: float) -> float:
    if x <= 0.0:
        return -700.0
    return np.log(x)


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < 1e-30:
        return fallback
    return a / b





def _self_test():
    import math
    eps = 1e-6
    assert abs(cosine_integral(1.0) - 0.3374039229009681347) < eps
    assert abs(sine_integral(1.0) - 0.9460830703671830149) < eps
    assert abs(spherical_bessel_j(0, 1.0) - math.sin(1.0) / 1.0) < eps
    assert abs(spherical_bessel_j(1, 1.0) - (math.sin(1.0) / 1.0 - math.cos(1.0)) / 1.0) < eps
    assert abs(modified_spherical_bessel_i(0, 1.0) - math.sinh(1.0) / 1.0) < eps
    print("special_functions: self-test passed.")


if __name__ == "__main__":
    _self_test()
