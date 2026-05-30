
import numpy as np


def r8vec_bracket3(n, x, xval, left):
    if n < 2 or xval < x[0] or xval > x[-1]:
        return max(0, min(left, n - 2))

    lo = 0
    hi = n - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid - 1
        elif xval >= x[mid + 1]:
            lo = mid + 1
        else:
            return mid
    return max(0, min(lo, n - 2))


def pwl_product_integral(a, b, f_x, f_v, g_x, g_v):
    f_x = np.asarray(f_x, dtype=float)
    f_v = np.asarray(f_v, dtype=float)
    g_x = np.asarray(g_x, dtype=float)
    g_v = np.asarray(g_v, dtype=float)

    if f_x.size < 2 or g_x.size < 2:
        return 0.0
    if a >= b:
        return 0.0
    if f_x[-1] <= a or g_x[-1] <= a:
        return 0.0


    xr_max = min(b, f_x[-1], g_x[-1])
    xr = a

    f_left = r8vec_bracket3(f_x.size, f_x, xr, 0)

    if f_x[f_left + 1] == f_x[f_left]:
        fr = f_v[f_left]
    else:
        fr = f_v[f_left] + (xr - f_x[f_left]) * (f_v[f_left + 1] - f_v[f_left]) / (f_x[f_left + 1] - f_x[f_left])

    g_left = r8vec_bracket3(g_x.size, g_x, xr, 0)
    if g_x[g_left + 1] == g_x[g_left]:
        gr = g_v[g_left]
    else:
        gr = g_v[g_left] + (xr - g_x[g_left]) * (g_v[g_left + 1] - g_v[g_left]) / (g_x[g_left + 1] - g_x[g_left])

    integral = 0.0
    max_iter = (f_x.size + g_x.size) * 2
    it = 0

    while xr < xr_max - 1e-15 and it < max_iter:
        it += 1
        xl = xr
        fl = fr
        gl = gr

        xr_new = xr_max

        for i in range(1, 3):
            if f_left + i < f_x.size:
                if xl < f_x[f_left + i] < xr_new:
                    xr_new = f_x[f_left + i]
                    break
        for i in range(1, 3):
            if g_left + i < g_x.size:
                if xl < g_x[g_left + i] < xr_new:
                    xr_new = g_x[g_left + i]
                    break
        xr = xr_new


        f_left = r8vec_bracket3(f_x.size, f_x, xr, f_left)
        if f_x[f_left + 1] == f_x[f_left]:
            fr = f_v[f_left]
        else:
            fr = f_v[f_left] + (xr - f_x[f_left]) * (f_v[f_left + 1] - f_v[f_left]) / (f_x[f_left + 1] - f_x[f_left])

        g_left = r8vec_bracket3(g_x.size, g_x, xr, g_left)
        if g_x[g_left + 1] == g_x[g_left]:
            gr = g_v[g_left]
        else:
            gr = g_v[g_left] + (xr - g_x[g_left]) * (g_v[g_left + 1] - g_v[g_left]) / (g_x[g_left + 1] - g_x[g_left])

        h = xr - xl
        if h > 1e-15:

            fm = 0.5 * (fl + fr)
            gm = 0.5 * (gl + gr)
            bit = h / 6.0 * (fl * gl + 4.0 * fm * gm + fr * gr)
            integral += bit

    return integral


def pulse_nonlinear_overlap(t, A1, A2):
    if t.size < 2 or A1.size != t.size or A2.size != t.size:
        return 0.0

    I1 = np.abs(A1) ** 2
    I2 = np.abs(A2) ** 2

    return pwl_product_integral(t[0], t[-1], t, I1, t, I2)


def pulse_inner_product(t, A1, A2):
    if t.size < 2 or A1.size != t.size or A2.size != t.size:
        return 0.0 + 0.0j


    re1 = np.real(A1)
    im1 = np.imag(A1)
    re2 = np.real(A2)
    im2 = np.imag(A2)

    re_re = pwl_product_integral(t[0], t[-1], t, re1, t, re2)
    im_im = pwl_product_integral(t[0], t[-1], t, im1, t, im2)
    re_im = pwl_product_integral(t[0], t[-1], t, re1, t, im2)
    im_re = pwl_product_integral(t[0], t[-1], t, im1, t, re2)

    return (re_re + im_im) + 1j * (re_im - im_re)


def raman_response_convolution(t, A, h_R):
    if t.size < 2 or A.size != t.size or h_R.size != t.size:
        return np.zeros_like(A)

    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("raman_response_convolution: time grid must be strictly increasing")




    raise NotImplementedError("Hole 2: raman_response_convolution 待实现")
