
import math
import numpy as np


def betain(x, p, q, beta_log):
    acu = 1.0e-14
    ifault = 0

    if p <= 0.0 or q <= 0.0:
        return x, 1
    if x < 0.0 or x > 1.0:
        return x, 2
    if x == 0.0 or x == 1.0:
        return x, 0

    psq = p + q
    cx = 1.0 - x


    if p < psq * x:
        xx = cx
        cx = x
        pp = q
        qq = p
        indx = 1
    else:
        xx = x
        pp = p
        qq = q
        indx = 0

    term = 1.0
    ai = 1.0
    value = 1.0
    ns = math.floor(qq + cx * psq)

    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    while True:
        term = term * temp * rx / (pp + ai)
        value = value + term
        temp = abs(term)

        if temp <= acu and temp <= acu * value:
            value = value * math.exp(pp * math.log(xx) + (qq - 1.0) * math.log(cx) - beta_log) / pp
            if indx:
                value = 1.0 - value
            break

        ai = ai + 1.0
        ns = ns - 1
        if 0 <= ns:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = psq
            psq = psq + 1.0

    return value, ifault


def digamma(x):
    c = 8.5
    d1 = -0.5772156649
    s = 0.00001
    s3 = 0.08333333333
    s4 = 0.0083333333333
    s5 = 0.003968253968

    if x <= 0.0:
        return 0.0, 1

    ifault = 0
    y = x
    value = 0.0

    if y <= s:
        value = d1 - 1.0 / y
        return value, ifault

    while y < c:
        value = value - 1.0 / y
        y = y + 1.0

    r = 1.0 / y
    value = value + math.log(y) - 0.5 * r
    r = r * r
    value = value - r * (s3 - r * (s4 - r * s5))
    return value, ifault


def trigamma(x):
    a = 0.0001
    b = 5.0
    b2 = 0.1666666667
    b4 = -0.03333333333
    b6 = 0.02380952381
    b8 = -0.03333333333

    if x <= 0.0:
        return 0.0, 1

    ifault = 0
    z = x

    if x <= a:
        return 1.0 / (x * x), ifault

    value = 0.0
    while z < b:
        value = value + 1.0 / (z * z)
        z = z + 1.0

    y = 1.0 / (z * z)
    value = value + 0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z
    return value, ifault


def cos_power_int(a, b, n):
    if n < 0:
        raise ValueError("cos_power_int: n must be non-negative")

    sa = math.sin(a)
    sb = math.sin(b)
    ca = math.cos(a)
    cb = math.cos(b)

    if n % 2 == 0:
        value = b - a
        mlo = 2
    else:
        value = sb - sa
        mlo = 3

    for m in range(mlo, n + 1, 2):
        value = ((m - 1) * value - (ca ** (m - 1)) * sa + (cb ** (m - 1)) * sb) / m

    return value


def log_beta(p, q):
    from math import lgamma
    return lgamma(p) + lgamma(q) - lgamma(p + q)


def incomplete_beta_cdf(x, p, q):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = log_beta(p, q)
    val, ierr = betain(x, p, q, lb)
    if ierr != 0:

        return 0.0 if x < 0.5 else 1.0
    return val
