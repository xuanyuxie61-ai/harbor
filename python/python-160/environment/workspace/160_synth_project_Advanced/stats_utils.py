
import math
import numpy as np


def alnorm(x, upper=False):
    if math.isnan(x):
        return math.nan
    ax = abs(x)
    if ax <= 1.0e-15:
        return 0.5 if upper else 0.5


    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034

    if ax <= 1.28:
        y = 0.5 * ax * ax
        al = 0.5 - ax * (c1 + y * (c2 + y * (c3 + y * (c4 + y * (c5 + y * c6)))))
    else:
        y = 0.5 * ax * ax
        al = math.exp(-y) / ax * (c5 + y * (c6 + y * (a1 + y * (a2 + y * a3))))
        al = al / (d1 + y * (d2 + y * (d3 + y * (d4 + y * d5))))
        al = al / (1.0 + y * (b1 + y * b2) / (1.0 + y * (a1 + y * (a2 + y * a3))))

    if x < 0.0:
        al = 1.0 - al
    if upper:
        al = 1.0 - al
    return al


def gammad(x, p):
    elimit = -88.0
    oflo = 1.0e37
    plimit = 1000.0
    tol = 1.0e-14
    xbig = 1.0e8

    value = 0.0
    if x < 0.0:
        return 0.0, 1
    if p <= 0.0:
        return 0.0, 1

    if x == 0.0:
        return 0.0, 0


    if p > plimit:
        pn1 = 3.0 * math.sqrt(p) * ((x / p) ** (1.0 / 3.0) + 1.0 / (9.0 * p) - 1.0)
        return alnorm(pn1, upper=False), 0


    if x > xbig:
        return 1.0, 0

    if x <= 1.0 or x < p:

        arg = p * math.log(x) - x - math.lgamma(p + 1.0)
        c = 1.0
        value = 1.0
        a = p
        while True:
            a = a + 1.0
            c = c * x / a
            value = value + c
            if c <= tol:
                break
        arg = arg + math.log(value)
        if arg >= elimit:
            value = math.exp(arg)
        else:
            value = 0.0
    else:

        arg = p * math.log(x) - x - math.lgamma(p)
        a = 1.0 - p
        b = a + x + 1.0
        c = 0.0
        pn1 = 1.0
        pn2 = x
        pn3 = x + 1.0
        pn4 = x * b
        value = pn3 / pn4
        while True:
            a = a + 1.0
            b = b + 2.0
            c = c + 1.0
            an = a * c
            pn5 = b * pn3 - an * pn1
            pn6 = b * pn4 - an * pn2
            if pn6 != 0.0:
                rn = pn5 / pn6
                if abs(value - rn) <= min(tol, tol * rn):
                    break
                value = rn
            pn1 = pn3
            pn2 = pn4
            pn3 = pn5
            pn4 = pn6
            if abs(pn5) >= oflo:
                pn1 = pn1 / oflo
                pn2 = pn2 / oflo
                pn3 = pn3 / oflo
                pn4 = pn4 / oflo
        arg = arg + math.log(value)
        if arg >= elimit:
            value = 1.0 - math.exp(arg)
        else:
            value = 1.0
    return value, 0


def ppnd(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0, 1
    try:
        from scipy.special import ndtri
        x = float(ndtri(p))
        return x, 0
    except Exception:
        return 0.0, 1


def ppchi2(p, v, g):
    aa = 0.6931471806
    c1 = 0.01
    c2 = 0.222222
    c3 = 0.32
    c4 = 0.4
    c5 = 1.24
    c6 = 2.2
    c7 = 4.67
    c8 = 6.66
    c9 = 6.73
    c10 = 13.32
    c11 = 60.0
    c12 = 70.0
    c13 = 84.0
    c14 = 105.0
    c15 = 120.0
    c16 = 127.0
    c17 = 140.0
    c18 = 175.0
    c19 = 210.0
    c20 = 252.0
    c21 = 264.0
    c22 = 294.0
    c23 = 346.0
    c24 = 420.0
    c25 = 462.0
    c26 = 606.0
    c27 = 672.0
    c28 = 707.0
    c29 = 735.0
    c30 = 889.0
    c31 = 932.0
    c32 = 966.0
    c33 = 1141.0
    c34 = 1182.0
    c35 = 1278.0
    c36 = 1740.0
    c37 = 2520.0
    c38 = 5040.0
    e = 0.5e-06
    maxit = 20
    pmax = 0.999998
    pmin = 0.000002

    value = -1.0
    if p < pmin or pmax < p:
        return value, 1
    if v <= 0.0:
        return value, 2

    ifault = 0
    xx = 0.5 * v
    c = xx - 1.0

    if v < -c5 * math.log(p):
        ch = (p * xx * math.exp(g + xx * aa)) ** (1.0 / xx)
        if ch < e:
            return ch, 0
    elif v <= c3:
        ch = c4
        a = math.log(1.0 - p)
        while True:
            q = ch
            p1 = 1.0 + ch * (c7 + ch)
            p2 = ch * (c9 + ch * (c8 + ch))
            t = -0.5 + (c7 + 2.0 * ch) / p1 - (c9 + ch * (c10 + 3.0 * ch)) / p2
            ch = ch - (1.0 - math.exp(a + g + 0.5 * ch + c * aa) * p2 / p1) / t
            if abs(q / ch - 1.0) <= c1:
                break
    else:
        x, _ = ppnd(p)
        p1 = c2 / v
        ch = v * (x * math.sqrt(p1) + 1.0 - p1) ** 3
        if c6 * v + 6.0 < ch:
            ch = -2.0 * (math.log(1.0 - p) - c * math.log(0.5 * ch) + g)

    for i in range(1, maxit + 1):
        q = ch
        p1 = 0.5 * ch
        temp, if1 = gammad(p1, xx)
        p2 = p - temp
        if if1 != 0:
            ifault = 3
            return ch, ifault
        t = p2 * math.exp(xx * aa + g + p1 - c * math.log(ch))
        b = t / ch
        a = 0.5 * t - b * c
        s1 = (c19 + a * (c17 + a * (c14 + a * (c13 + a * (c12 + c11 * a))))) / c24
        s2 = (c24 + a * (c29 + a * (c32 + a * (c33 + c35 * a)))) / c37
        s3 = (c19 + a * (c25 + a * (c28 + c31 * a))) / c37
        s4 = (c20 + a * (c27 + c34 * a) + c * (c22 + a * (c30 + c36 * a))) / c38
        s5 = (c13 + c21 * a + c * (c18 + c26 * a)) / c37
        s6 = (c15 + c * (c23 + c16 * c)) / c38
        ch = ch + t * (1.0 + 0.5 * t * s1 - b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))))
        if abs(q / ch - 1.0) <= e:
            value = ch
            return value, 0

    ifault = 4
    value = ch
    return value, ifault


def betain(x, a, b, beta_log):
    if x <= 0.0:
        return 0.0, 0
    if x >= 1.0:
        return 1.0, 0
    if a <= 0.0 or b <= 0.0:
        return 0.0, 1


    maxit = 200
    eps = 3.0e-7
    am = 1.0
    bm = 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap

    for m in range(1, maxit + 1):
        m2 = 2 * m
        d = m * (b - m) * x / ((qam + m2) * (a + m2))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - aold) < eps * abs(az):
            break

    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - beta_log)
    return front * az / a, 0


def betanc(x, a, b, lam):
    if x <= 0.0:
        return 0.0, 0
    if x >= 1.0:
        return 1.0, 0
    if a <= 0.0 or b <= 0.0:
        return 0.0, 1
    if lam <= 0.0:
        beta_log = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return betain(x, a, b, beta_log)[0], 0

    c = 0.5 * lam
    xj = 0.0
    m = int(math.floor(c + 0.5))
    mr = m
    iterlo = m - int(math.floor(5.0 * math.sqrt(mr)))
    iterhi = m + int(math.floor(5.0 * math.sqrt(mr)))
    t = -c + mr * math.log(c) - math.lgamma(mr + 1.0)
    q = math.exp(t)
    r = q
    psum = q

    beta = math.lgamma(a + mr) + math.lgamma(b) - math.lgamma(a + mr + b)
    s1 = (a + mr) * math.log(x) + b * math.log(1.0 - x) - math.log(a + mr) - beta
    gx = math.exp(s1)
    fx = gx
    temp, _ = betain(x, a + mr, b, beta)
    ftemp = temp
    xj = xj + 1.0
    sm = q * temp
    iter1 = m

    while iter1 >= iterlo and q > 1.0e-14:
        q = q * iter1 / c
        xj = xj + 1.0
        gx = (a + iter1) / (x * (a + b + iter1 - 1.0)) * gx
        iter1 = iter1 - 1
        temp = temp + gx
        psum = psum + q
        sm = sm + q * temp

    t0 = math.lgamma(a + b) - math.lgamma(a + 1.0) - math.lgamma(b)
    s0 = a * math.log(x) + b * math.log(1.0 - x)
    s = 0.0
    for i in range(1, iter1 + 1):
        j = i - 1
        s = s + math.exp(t0 + s0 + j * math.log(x))
        t1 = math.log(a + b + j) - math.log(a + 1.0 + j) + t0
        t0 = t1

    errbd = (1.0 - gammad(c, iter1)[0]) * (temp + s)
    q = r
    temp = ftemp
    gx = fx
    iter2 = m

    while True:
        ebd = errbd + (1.0 - psum) * temp
        if ebd < 1.0e-14 or iterhi <= iter2:
            break
        iter2 = iter2 + 1
        xj = xj + 1.0
        q = q * c / iter2
        psum = psum + q
        temp = temp - gx
        gx = x * (a + b + iter2 - 1.0) / (a + iter2) * gx
        sm = sm + q * temp

    return sm, 0


def ncbeta(a, b, lam, x, errmax=1.0e-14):
    ifault = 0
    value = x
    if lam <= 0.0:
        ifault = 3
        return value, ifault
    if a <= 0.0:
        ifault = 3
        return value, ifault
    if b <= 0.0:
        ifault = 3
        return value, ifault
    if x <= 0.0:
        return 0.0, 0
    if x >= 1.0:
        return 1.0, 0

    c = 0.5 * lam
    m = int(math.floor(c + 0.5))
    mr = m
    iterlo = m - int(math.floor(5.0 * math.sqrt(mr)))
    iterhi = m + int(math.floor(5.0 * math.sqrt(mr)))
    t = -c + mr * math.log(c) - math.lgamma(mr + 1.0)
    q = math.exp(t)
    r = q
    psum = q

    beta = math.lgamma(a + mr) + math.lgamma(b) - math.lgamma(a + mr + b)
    s1 = (a + mr) * math.log(x) + b * math.log(1.0 - x) - math.log(a + mr) - beta
    gx = math.exp(s1)
    fx = gx
    temp, _ = betain(x, a + mr, b, beta)
    ftemp = temp
    xj = 0.0
    sm = q * temp
    iter1 = m

    while iter1 >= iterlo and q > errmax:
        q = q * iter1 / c
        xj = xj + 1.0
        gx = (a + iter1) / (x * (a + b + iter1 - 1.0)) * gx
        iter1 = iter1 - 1
        temp = temp + gx
        psum = psum + q
        sm = sm + q * temp

    t0 = math.lgamma(a + b) - math.lgamma(a + 1.0) - math.lgamma(b)
    s0 = a * math.log(x) + b * math.log(1.0 - x)
    s = 0.0
    for i in range(1, iter1 + 1):
        j = i - 1
        s = s + math.exp(t0 + s0 + j * math.log(x))
        t1 = math.log(a + b + j) - math.log(a + 1.0 + j) + t0
        t0 = t1

    errbd = (1.0 - gammad(c, iter1)[0]) * (temp + s)
    q = r
    temp = ftemp
    gx = fx
    iter2 = m

    while True:
        ebd = errbd + (1.0 - psum) * temp
        if ebd < errmax or iterhi <= iter2:
            break
        iter2 = iter2 + 1
        xj = xj + 1.0
        q = q * c / iter2
        psum = psum + q
        temp = temp - gx
        gx = x * (a + b + iter2 - 1.0) / (a + iter2) * gx
        sm = sm + q * temp

    return sm, 0


def setup_discrete_histogram(s, s_min, s_max):
    s = np.asarray(s, dtype=float)
    if s_max < s_min:
        s_min, s_max = s_max, s_min
    s = np.clip(s, s_min, s_max)
    s = np.sort(s)

    x_list = [s_min]
    c_list = [0]
    for val in s:
        if abs(val - x_list[-1]) > 1.0e-14:
            x_list.append(val)
            c_list.append(1)
        else:
            c_list[-1] += 1

    if abs(x_list[-1] - s_max) > 1.0e-14:
        x_list.append(s_max)
        c_list.append(0)

    x = np.array(x_list, dtype=float)
    c = np.array(c_list, dtype=float)
    x_num = len(x)
    y = np.zeros(x_num, dtype=float)

    if x_num >= 2:
        dx1 = x[1] - x[0]
        if abs(dx1) > 1.0e-15:
            y[0] = c[0] / dx1
        for i in range(1, x_num - 1):
            dx = x[i + 1] - x[i - 1]
            if abs(dx) > 1.0e-15:
                y[i] = c[i] / dx
        dxn = x[x_num - 1] - x[x_num - 2]
        if abs(dxn) > 1.0e-15:
            y[x_num - 1] = c[x_num - 1] / dxn


    y_int = 0.0
    for i in range(x_num - 1):
        y_int += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) * 0.5

    if abs(y_int) > 1.0e-15:
        y = y / y_int
    return x, y
