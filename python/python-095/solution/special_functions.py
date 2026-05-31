"""
special_functions.py
声学主动噪声控制所需的特殊数学函数库

融合原始项目:
  - 031_asa063 (不完全Beta函数 betain)
  - 053_asa266 (digamma, trigamma)
  - 300_disk01_integrands (cos_power_int)

科学背景:
  不完全Beta函数用于计算多通道误差的置信区间与统计显著性检验;
  digamma/trigamma 用于 Dirichlet 分布参数估计中的牛顿迭代;
  cos_power_int 用于圆形活塞辐射器的指向性函数积分.
"""

import math
import numpy as np


def betain(x, p, q, beta_log):
    """
    计算不完全Beta函数比值 I_x(p,q).

    数学定义:
        I_x(p,q) = (1/B(p,q)) * \int_0^x t^{p-1} (1-t)^{q-1} dt

    其中 B(p,q) = Gamma(p)Gamma(q)/Gamma(p+q), beta_log = ln(B(p,q)).

    参数:
        x: 积分上限, 0 <= x <= 1
        p, q: Beta分布参数, p>0, q>0
        beta_log: ln(B(p,q))

    返回:
        value: I_x(p,q)
        ifault: 错误标志 (0=无错误)
    """
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

    # 改变尾部以确保收敛
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
    """
    计算 digamma 函数: psi(x) = d/dx [ln(Gamma(x))]

    数学定义:
        psi(x) = Gamma'(x) / Gamma(x)

    用于 Dirichlet 分布的最大似然估计.
    """
    c = 8.5
    d1 = -0.5772156649  # Euler-Mascheroni 常数
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
    """
    计算 trigamma 函数: psi'(x) = d^2/dx^2 [ln(Gamma(x))]

    数学定义:
        psi'(x) = \sum_{n=0}^{\infty} 1/(x+n)^2

    用于 Dirichlet 估计中的 Fisher 信息矩阵.
    """
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
    """
    计算余弦幂积分:
        \int_a^b cos^n(t) dt

    使用递推公式:
        \int cos^n(t) dt = -(1/n)[cos^{n-1}(t) sin(t) + (n-1)\int cos^{n-2}(t) dt]

    声学应用:
        计算圆形活塞辐射器的远场指向性函数:
        D(\theta) = 2 J_1(k a sin\theta) / (k a sin\theta)
        其功率积分与 cos^n 项相关.
    """
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
    """
    计算 ln(B(p,q)) = ln(Gamma(p)) + ln(Gamma(q)) - ln(Gamma(p+q))
    """
    from math import lgamma
    return lgamma(p) + lgamma(q) - lgamma(p + q)


def incomplete_beta_cdf(x, p, q):
    """
    包装函数: 计算正则化不完全Beta函数,并处理边界.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = log_beta(p, q)
    val, ierr = betain(x, p, q, lb)
    if ierr != 0:
        # 回退到 scipy 风格近似或返回边界值
        return 0.0 if x < 0.5 else 1.0
    return val
