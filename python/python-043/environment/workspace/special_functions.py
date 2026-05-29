"""
特殊函数库 (special_functions.py)
=====================================
基于种子项目 221_cosine_integral 的核心算法，扩展至地球物理地核发电机模拟所需的
特殊函数族，包括：
  - 余弦积分 Ci(x) 与正弦积分 Si(x)
  - 球 Bessel 函数 j_l(x), n_l(x)
  - 修正球 Bessel 函数 i_l(x), k_l(x)
  - 地核磁扩散核函数 D(x, tau)

这些函数在地核磁场径向扩散、球谐展开系数计算、以及核幔边界热流分析中
均有直接应用。
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. 余弦积分 Ci(x)
#    Ci(x) = gamma + ln|x| + integral_0^x (cos(t)-1)/t dt
#    对于 x -> 0, Ci(x) ~ gamma + ln|x| - x^2/4 + O(x^4)
#    对于 x -> inf, 渐近展开
# ---------------------------------------------------------------------------
def cosine_integral(x: float) -> float:
    """
    计算余弦积分 Ci(x)。
    采用分段策略：
      |x| <= 16 : 幂级数展开
      16 < |x| <= 32 : Bessel 函数递推展开
      |x| > 32 : 渐近展开
    """
    x = float(x)
    if x == 0.0:
        return -np.inf  # 对数奇点

    ax = abs(x)
    gamma_e = 0.5772156649015328606  # Euler-Mascheroni 常数

    if ax <= 16.0:
        # 小参数幂级数：Ci(x) = gamma + ln(x) + sum_{k=1}^{inf} (-1)^k x^{2k} / (2k * (2k)!)
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
        # 中等参数：使用球 Bessel 函数辅助展开
        # Ci(x) = gamma + ln(x) - C(x) cos(x) - S(x) sin(x)
        # 其中 C(x), S(x) 为辅助 Fresnel 型积分
        # 这里采用直接数值积分加速
        nseg = 200
        dt = ax / nseg
        integral = 0.0
        for i in range(nseg):
            t0 = i * dt
            t1 = (i + 1) * dt
            # Simpson 规则
            f0 = (np.cos(t0) - 1.0) / t0 if t0 > 1e-12 else 0.0
            fm = (np.cos(0.5 * (t0 + t1)) - 1.0) / (0.5 * (t0 + t1)) if (t0 + t1) > 2e-12 else 0.0
            f1 = (np.cos(t1) - 1.0) / t1
            integral += dt / 6.0 * (f0 + 4.0 * fm + f1)
        return gamma_e + np.log(ax) + integral
    else:
        # 大参数渐近展开
        # Ci(x) ~ sin(x)/x * (1 - 2!/x^2 + 4!/x^4 - ...) - cos(x)/x * (1/x - 3!/x^3 + ...)
        inv_x = 1.0 / ax
        sinx = np.sin(ax)
        cosx = np.cos(ax)
        # 取前几项
        f = 1.0 - 2.0 * inv_x * inv_x + 24.0 * inv_x**4
        g = inv_x - 6.0 * inv_x**3 + 120.0 * inv_x**5
        return sinx / ax * f - cosx / ax * g


# ---------------------------------------------------------------------------
# 2. 正弦积分 Si(x)
#    Si(x) = integral_0^x sin(t)/t dt
# ---------------------------------------------------------------------------
def sine_integral(x: float) -> float:
    """计算正弦积分 Si(x)。"""
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
        # 渐近：Si(x) ~ pi/2 - cos(x)/x * (1 - 2!/x^2 + ...) - sin(x)/x * (1/x - 3!/x^3 + ...)
        inv_x = 1.0 / ax
        sinx = np.sin(ax)
        cosx = np.cos(ax)
        f = 1.0 - 2.0 * inv_x * inv_x
        g = inv_x - 6.0 * inv_x**3
        val = 0.5 * np.pi - cosx / ax * f - sinx / ax * g
        return val if x > 0 else -val


# ---------------------------------------------------------------------------
# 3. 球 Bessel 函数 j_l(x)
#    满足递推：j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
#    在地核发电机谱方法中，径向方程经常涉及球 Bessel 函数
# ---------------------------------------------------------------------------
def spherical_bessel_j(l: int, x: float) -> float:
    """计算第一类球 Bessel 函数 j_l(x)。"""
    x = float(x)
    if x == 0.0:
        return 1.0 if l == 0 else 0.0
    if l == 0:
        return np.sin(x) / x
    if l == 1:
        return np.sin(x) / (x * x) - np.cos(x) / x
    # 递推
    jlm2 = np.sin(x) / x
    jlm1 = np.sin(x) / (x * x) - np.cos(x) / x
    for ll in range(2, l + 1):
        jl = (2.0 * ll - 1.0) / x * jlm1 - jlm2
        jlm2, jlm1 = jlm1, jl
    return jlm1


def spherical_bessel_y(l: int, x: float) -> float:
    """计算第二类球 Bessel 函数 y_l(x)（球 Neumann 函数）。"""
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


# ---------------------------------------------------------------------------
# 4. 修正球 Bessel 函数
#    在地核热扩散与磁扩散的格林函数中经常出现
# ---------------------------------------------------------------------------
def modified_spherical_bessel_i(l: int, x: float) -> float:
    """计算修正第一类球 Bessel 函数 i_l(x) = sqrt(pi/(2x)) I_{l+1/2}(x)。"""
    x = float(x)
    if x == 0.0:
        return 1.0 if l == 0 else 0.0
    # 递推：i_{l+1} = -(2l+1)/x i_l + i_{l-1}
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
    """计算修正第二类球 Bessel 函数 k_l(x) = sqrt(pi/(2x)) K_{l+1/2}(x)。"""
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


# ---------------------------------------------------------------------------
# 5. 地核磁扩散核函数
#    D(r, r', t) 描述地核内部磁场径向分量的扩散行为
#    基于球 Bessel 函数与指数衰减核的乘积构造
# ---------------------------------------------------------------------------
def magnetic_diffusion_kernel(r: float, rp: float, t: float, eta: float, l: int) -> float:
    """
    计算球壳内磁扩散核函数。

    参数:
      r, rp : 径向坐标 (单位: m)
      t     : 时间 (单位: s)
      eta   : 磁扩散系数 (单位: m^2/s)
      l     : 球谐阶数

    公式:
      D_l(r, r', t) = (1 / sqrt(4*pi*eta*t)) * exp(-(r-r')^2/(4*eta*t))
                      * j_l(r*r'/(2*eta*t)) * exp(-l*(l+1)*eta*t/r^2)
    """
    if t <= 0.0 or eta <= 0.0 or r <= 0.0 or rp <= 0.0:
        return 0.0
    diff = 4.0 * eta * t
    gaussian = np.exp(-(r - rp) ** 2 / diff) / np.sqrt(np.pi * diff)
    bessel_part = spherical_bessel_j(l, r * rp / (2.0 * eta * t))
    decay = np.exp(-l * (l + 1.0) * eta * t / (r * r))
    return gaussian * bessel_part * decay


# ---------------------------------------------------------------------------
# 6. 数值鲁棒性工具
# ---------------------------------------------------------------------------
def safe_log(x: float) -> float:
    """安全对数，避免 x<=0 时崩溃。"""
    if x <= 0.0:
        return -700.0  # 接近 log(min_float)
    return np.log(x)


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    """安全除法，避免除以零。"""
    if abs(b) < 1e-30:
        return fallback
    return a / b


# ---------------------------------------------------------------------------
# 单元测试/边界检查
# ---------------------------------------------------------------------------
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
