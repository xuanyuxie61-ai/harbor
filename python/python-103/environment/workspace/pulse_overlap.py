"""
pulse_overlap.py
分段线性脉冲重叠积分模块（对应种子项目 929_pwl_product_integral）

在光纤非线性光学中，交叉相位调制（XPM）和四波混频（FWM）等效应
涉及多个脉冲之间的时域重叠积分。当脉冲以分段线性（PWL）形式表示时，
需要精确计算其乘积积分。

核心物理公式：
  非线性相互作用强度:
    I_{NL} = ∫_{t_1}^{t_2} |A_1(t)|² |A_2(t)|² dt

  对于分段线性函数 F(t) 和 G(t)，在子区间 [t_l, t_r] 上：
    F(t) = f_l + (f_r - f_l)(t - t_l)/(t_r - t_l)
    G(t) = g_l + (g_r - g_l)(t - t_l)/(t_r - t_l)

  乘积 H(t) = F(t)G(t) 为二次多项式，可解析积分：
    ∫_{t_l}^{t_r} H(t) dt = (t_r - t_l)/6 * [f_l g_l + 4 f_m g_m + f_r g_r]
    其中 f_m = (f_l + f_r)/2, g_m = (g_l + g_r)/2 （Simpson法则）

本模块同时提供复值脉冲（包络）的内积计算，用于模式耦合分析。
"""

import numpy as np


def r8vec_bracket3(n, x, xval, left):
    """
    在有序数组x中定位xval所在的区间[left, left+1]，返回left索引。
    采用二分搜索保证O(log n)效率。
    """
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
    """
    计算分段线性函数 F(t) 和 G(t) 在区间 [a, b] 上的乘积积分。

    参数:
        a, b: float, 积分上下限
        f_x: ndarray, F的节点坐标（升序）
        f_v: ndarray, F的节点值
        g_x: ndarray, G的节点坐标（升序）
        g_v: ndarray, G的节点值

    返回:
        integral: float, ∫_a^b F(t)G(t) dt
    """
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

    # 确定有效右边界
    xr_max = min(b, f_x[-1], g_x[-1])
    xr = a

    f_left = r8vec_bracket3(f_x.size, f_x, xr, 0)
    # 左端点线性插值
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
        # 找到下一个节点位置
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

        # 更新插值
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
            # Simpson积分: h/6 * (f_l*g_l + 4*f_m*g_m + f_r*g_r)
            fm = 0.5 * (fl + fr)
            gm = 0.5 * (gl + gr)
            bit = h / 6.0 * (fl * gl + 4.0 * fm * gm + fr * gr)
            integral += bit

    return integral


def pulse_nonlinear_overlap(t, A1, A2):
    """
    计算两个复值脉冲包络的非线性重叠积分:
      I = ∫ |A1(t)|² |A2(t)|² dt

    物理意义：交叉相位调制（XPM）的有效作用强度。
    """
    if t.size < 2 or A1.size != t.size or A2.size != t.size:
        return 0.0

    I1 = np.abs(A1) ** 2
    I2 = np.abs(A2) ** 2

    return pwl_product_integral(t[0], t[-1], t, I1, t, I2)


def pulse_inner_product(t, A1, A2):
    """
    计算两个复值脉冲的内积:
      <A1|A2> = ∫ A1*(t) A2(t) dt

    采用分段线性（实部和虚部分开）后解析积分。
    """
    if t.size < 2 or A1.size != t.size or A2.size != t.size:
        return 0.0 + 0.0j

    # 分离实部虚部并分别积分
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
    """
    计算Raman响应函数的卷积积分:
      S(t) = ∫_0^∞ h_R(τ) |A(t-τ)|² dτ

    其中h_R(τ)为归一化的Raman响应函数。
    在分步傅里叶法中，非线性项包含:
      iγ(1 + i/ω_0 ∂/∂T) [A(z,T) S(z,T)]

    参数:
        t: ndarray, 时间网格（均匀）
        A: ndarray, 脉冲包络
        h_R: ndarray, Raman响应函数在相同网格上的采样
    """
    if t.size < 2 or A.size != t.size or h_R.size != t.size:
        return np.zeros_like(A)

    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("raman_response_convolution: time grid must be strictly increasing")

    # TODO: Hole 2 — 实现Raman响应函数的因果卷积
    # S(t) = ∫_0^∞ h_R(τ) |A(t-τ)|² dτ
    # 使用FFT加速，注意零填充处理循环卷积与线性卷积的差异
    raise NotImplementedError("Hole 2: raman_response_convolution 待实现")
