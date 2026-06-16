"""
special_functions.py — 等离子体碰撞理论中的特殊函数
===================================================

核心特殊函数:
  1. Chandrasekhar 函数  G(x) = [erf(x) - 2x/√π exp(-x²)] / (2x²)
  2. 误差函数 erf(x) 及其渐近展开
  3. 余弦积分 Ci(x)  (源自电磁屏蔽积分)
  4. 不完全 Gamma 函数
  5. 等离子体色散函数 Z(ζ) (Fried-Conte 函数)

参考文献:
  Chandrasekhar, S., ApJ 97, 255 (1943).
  Zhang & Jin, "Computation of Special Functions", Wiley 1996.
  Fried & Conte, Phys. Fluids 4, 154 (1961).
"""

import math
import numpy as np
from scipy import special as sp
from scipy import integrate as spi

from physical_constants import PI, SQRT_PI, TWO_PI, FOUR_PI


# ===========================================================================
#  §1  Chandrasekhar 函数  (动力学摩擦核心)
# ===========================================================================
def chandrasekhar_G(x):
    """Chandrasekhar 函数:

        G(x) = [erf(x) - (2x/√π) exp(-x²)] / (2 x²)

    物理含义: 各向同性 Maxwellian 场粒子产生的动力学摩擦因子.
    渐近行为:
      x → 0:  G(x) → 2/(3√π) · x  (线性)
      x → ∞:  G(x) → 1/(2x²)       (缓慢衰减)

    实现细节:
      x < 1e-8:  Taylor 展开至 O(x^5)
      otherwise: 直接公式 (使用 scipy.special.erf)
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    small = np.abs(x) < 1.0e-8
    large = ~small

    # 小 x: Taylor 展开 G(x) = 2x/(3√π) - 2x³/(15√π) + x⁵/(35√π) + ...
    if np.any(small):
        xs = x[small]
        result[small] = (2.0 * xs / (3.0 * SQRT_PI)
                         - 2.0 * xs**3 / (15.0 * SQRT_PI)
                         + xs**5 / (35.0 * SQRT_PI))

    # 正常范围
    if np.any(large):
        xl = x[large]
        erf_xl = sp.erf(xl)
        exp_xl = np.exp(-xl**2)
        result[large] = (erf_xl - 2.0 * xl / SQRT_PI * exp_xl) / (2.0 * xl**2)

    return float(result) if result.ndim == 0 else result


def chandrasekhar_G_derivative(x):
    """G'(x) = dG/dx.

    解析求导:
      G'(x) = { [2/√π · (2x²+1) · e^{-x²} - erf(x)/x] / (2x²)
               - [erf(x) - 2x/√π · e^{-x²}] / x³ }

    简化后:
      G'(x) = -2G(x)/x + [2/√π · (1+2x²) · e^{-x²} - erf(x)] / (2x³)
              ... 但更实用的形式:

    直接对 G 的分子求导:
      d/dx [erf(x) - 2x/√π · e^{-x²}] = 2/√π · e^{-x²} (1 + 2x²) - 2/√π · e^{-x²}
                                        = 4x²/√π · e^{-x²}
      再用商法则.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    small = np.abs(x) < 1.0e-6
    large = ~small

    if np.any(small):
        xs = x[small]
        # Taylor: G'(x) ≈ 2/(3√π) - 2x²/(5√π) + x⁴/(14√π)
        result[small] = (2.0 / (3.0 * SQRT_PI)
                         - 2.0 * xs**2 / (5.0 * SQRT_PI)
                         + xs**4 / (14.0 * SQRT_PI))

    if np.any(large):
        xl = x[large]
        xl2 = xl**2
        xl3 = xl**3
        erf_x = sp.erf(xl)
        exp_x = np.exp(-xl2)
        # 分子: N(x) = erf(x) - 2x/√π · e^{-x²}
        N = erf_x - 2.0 * xl / SQRT_PI * exp_x
        # 分子导数: N'(x) = 4x²/√π · e^{-x²}
        Np = 4.0 * xl2 / SQRT_PI * exp_x
        # G'(x) = (N' · 2x² - N · 4x) / (4x⁴)
        #        = (N' x - 2N) / (2x³)
        result[large] = (Np * xl - 2.0 * N) / (2.0 * xl3)

    return float(result) if result.ndim == 0 else result


def friction_coefficient(x, f_v, v_grid):
    """动力学摩擦系数 A(x):

    A(x) = 2 G(x) M[f](x) / x²

    where M[f](x) = 4π ∫_0^x f(v') v'² dv'  是累积质量分布函数.

    Parameters
    ----------
    x : ndarray  速度网格
    f_v : ndarray  分布函数 f(x)
    v_grid : ndarray  速度网格 (同 x)
    """
    dv = np.diff(v_grid)
    # 被积函数: 4π v² f(v)
    integrand = FOUR_PI * v_grid**2 * f_v
    # 累积积分 (梯形法则)
    M = np.zeros_like(x)
    for i in range(1, len(x)):
        M[i] = M[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * dv[i-1]

    G = chandrasekhar_G(x)
    A = np.zeros_like(x)
    mask = x > 1.0e-10
    A[mask] = 2.0 * G[mask] * M[mask] / x[mask]**2
    # x→0 极限: A(0) = 0 (G(0)·M(0)/0² 的不定式, 但 M(0)=0)
    return A


def diffusion_coefficient(x, f_v, v_grid):
    """速度扩散系数 D(x):

    D(x) = G(x) M[f](x)/x + [1 - G(x)/(2x²)] · N[f](x)/3 + G(x)/(2x) · ...

    简化实现:  D(x) = G(x) M(x)/x + (1/(2x²))[erf(x) - ...]

    对于各向同性分布, Rosenbluth 势 H(v) 的径向导数给出扩散:
    D(v) = Γ/(2v) · [M(v)/(2v) · (erf(x)-...) + ...]

    本实现采用简化形式: D(x) ≈ G(x) · (M(x)/x + C(x)/x)
    其中 C(x) 是与热速度弥散相关的修正.
    """
    dv = np.diff(v_grid)
    integrand_M = FOUR_PI * v_grid**2 * f_v
    M = np.zeros_like(x)
    for i in range(1, len(x)):
        M[i] = M[i-1] + 0.5 * (integrand_M[i] + integrand_M[i-1]) * dv[i-1]

    G = chandrasekhar_G(x)
    D = np.zeros_like(x)
    mask = x > 1.0e-10

    # 主导项: D ≈ G(x) M(x) / x
    D[mask] = G[mask] * M[mask] / x[mask]

    # 高阶修正项 (来自 Rosenbluth 势 H 的二阶导数)
    # 包含来自快速粒子 (v'>v) 的贡献
    # N(x) = 4π ∫_x^∞ f(v') v' dv'
    integrand_N = FOUR_PI * v_grid * f_v
    N_total = np.sum(0.5 * (integrand_N[1:] + integrand_N[:-1]) * dv)
    N_cumul = np.zeros_like(x)
    for i in range(1, len(x)):
        N_cumul[i] = N_cumul[i-1] + 0.5 * (integrand_N[i] + integrand_N[i-1]) * dv[i-1]
    N_tail = N_total - N_cumul  # ∫_x^∞

    # 修正: 来自外部速度壳层的贡献
    erf_x = sp.erf(x)
    correction = np.zeros_like(x)
    correction[mask] = (erf_x[mask] / (2.0 * x[mask]**2)
                        - G[mask]) * N_tail[mask] / (3.0 * x[mask])
    D[mask] += correction[mask]

    return D


# ===========================================================================
#  §2  余弦积分  Ci(x) = γ + ln(x) + ∫_0^x (cos t - 1)/t dt
# ===========================================================================
def cosine_integral(x):
    """余弦积分 Ci(x).

    分段实现 (Zhang & Jin, 1996):
      x ≤ 16:  Taylor 级数
      16 < x ≤ 32: Bessel 函数后向递推
      x > 32:  渐近展开

    物理应用: 电磁屏蔽势的积分表达, 以及库仑对数计算中出现的
    振荡积分 ∫ cos(kr)/r dk.
    """
    x = float(x)
    xabs = abs(x)
    el = 0.5772156649015329  # Euler-Mascheroni 常数 γ
    eps = 1.0e-15

    if xabs == 0.0:
        return -float('inf')

    elif xabs <= 16.0:
        # Taylor 级数: Ci(x) = γ + ln|x| + Σ_{k=1}^∞ (-1)^k x^{2k} / (2k·(2k)!)
        xr = -0.25 * xabs**2
        value = el + math.log(xabs) + xr
        for k in range(2, 50):
            xr *= (-0.5 * (k - 1) / (k * k * (2 * k - 1))) * xabs**2
            value += xr
            if abs(xr) < abs(value) * eps:
                return value
        return value

    elif xabs <= 32.0:
        # Bessel 函数后向递推 (Miller 算法)
        m = int(math.floor(47.2 + 0.82 * xabs))
        bj = np.zeros(m + 1)
        xa1 = 0.0
        xa0 = 1.0e-100
        for k in range(m, 0, -1):
            xa = 4.0 * k * xa0 / xabs - xa1
            if k <= m:
                bj[k] = xa
            xa1 = xa0
            xa0 = xa
        xs = bj[1]
        for k in range(3, m + 1, 2):
            xs += 2.0 * bj[k]
        for k in range(1, m + 1):
            bj[k] /= xs
        xr = 1.0
        xg1 = bj[1]
        for k in range(2, m + 1):
            xr *= 0.25 * (2.0 * k - 3.0)**2 / ((k - 1.0) * (2.0 * k - 1.0)**2) * xabs
            xg1 += bj[k] * xr
        xr = 1.0
        xg2 = bj[1]
        for k in range(2, m + 1):
            xr *= 0.25 * (2.0 * k - 5.0)**2 / ((k - 1.0) * (2.0 * k - 3.0)**2) * xabs
            xg2 += bj[k] * xr
        xcs = math.cos(xabs / 2.0)
        xss = math.sin(xabs / 2.0)
        value = (el + math.log(xabs) - xabs * xss * xg1
                 + 2.0 * xcs * xg2 - 2.0 * xcs**2)
        return value
    else:
        # 渐近展开: Ci(x) ~ sin(x)/x · f(x) - cos(x)/x · g(x)
        xr = 1.0
        xf = 1.0
        for k in range(1, 10):
            xr *= -2.0 * k * (2 * k - 1) / xabs**2
            xf += xr
        xr = 1.0 / xabs
        xg = xr
        for k in range(1, 9):
            xr *= -2.0 * (2 * k + 1) * k / xabs**2
            xg += xr
        value = xf * math.sin(xabs) / xabs - xg * math.cos(xabs) / xabs
        return value


# ===========================================================================
#  §3  等离子体色散函数  Z(ζ) (Fried-Conte)
# ===========================================================================
def plasma_dispersion_function(zeta):
    """等离子体色散函数 (Fried-Conte function):

    Z(ζ) = π^{-1/2} ∫_{-∞}^{∞} exp(-t²)/(t - ζ) dt
           (Im(ζ) > 0, 解析延拓到整个复平面)

    对于实数 ζ:
    Z(ζ) = i√π exp(-ζ²) + 2 D(ζ)
    其中 D(ζ) = exp(-ζ²) ∫_0^ζ exp(t²) dt 是 Dawson 函数.

    应用: Landau 阻尼色散关系, 离子声波稳定性分析.
    """
    if isinstance(zeta, complex) or isinstance(zeta, np.complexfloating):
        # 复数情形: 使用 Fadeeva 函数 w(z)
        # Z(ζ) = i√π · w(ζ)
        from scipy.special import wofz
        return 1j * SQRT_PI * wofz(zeta)
    else:
        # 实数情形
        dawson = sp.dawsn(float(zeta))
        return 1j * SQRT_PI * np.exp(-float(zeta)**2) + 2.0 * dawson


def plasma_dispersion_derivative(zeta):
    """Z'(ζ) = -2(1 + ζ Z(ζ)).

    这是 Z 函数的递推关系, 可避免数值微分.
    """
    Z = plasma_dispersion_function(zeta)
    return -2.0 * (1.0 + zeta * Z)


# ===========================================================================
#  §4  不完全 Gamma 函数  γ(a, x) / Γ(a)
# ===========================================================================
def regularized_gamma_lower(a, x):
    """正则化下不完全 Gamma 函数 P(a,x) = γ(a,x)/Γ(a).

    用于碰撞频率的速度依赖计算.
    P(a,x) = γ(a,x)/Γ(a),  γ(a,x) = ∫_0^x t^{a-1} e^{-t} dt
    """
    return float(sp.gammainc(a, x))


# ===========================================================================
#  §5  库仑对数的速度依赖修正
# ===========================================================================
def coulomb_logarithm_velocity_dependent(x, ln_lambda_0, v_th):
    """速度依赖的库仑对数.

    对于高速粒子 (v >> v_th), 屏蔽距离由 Debye 长度决定:
      ln Λ(v) = ln(1 + (λ_D / b_min(v))²) / 2

    对于低速粒子, 需要考虑动态屏蔽效应:
      ln Λ(v) ≈ ln Λ_0 · [1 - exp(-x²)]

    Parameters
    ----------
    x : float or ndarray  无量纲速度 v/v_th
    ln_lambda_0 : float  热库仑对数
    v_th : float  热速度 (仅用于量纲)
    """
    x = np.asarray(x, dtype=np.float64)
    # 速度依赖修正因子: 低速粒子的碰撞频率降低 (动态屏蔽)
    # 源自 BPS 理论 (Baalrud, Daligault, Simakov)
    correction = 1.0 - np.exp(-x**2)
    return ln_lambda_0 * np.maximum(correction, 0.01)


# ===========================================================================
#  §6  测试入口
# ===========================================================================
if __name__ == "__main__":
    print("=== Chandrasekhar 函数测试 ===")
    for xv in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        G_val = chandrasekhar_G(xv)
        print(f"  G({xv:6.2f}) = {G_val:.10e}")

    print("\n=== 余弦积分测试 ===")
    for xv in [0.5, 1.0, 5.0, 10.0, 20.0, 40.0]:
        ci_val = cosine_integral(xv)
        print(f"  Ci({xv:6.1f}) = {ci_val:.10e}")

    print("\n=== 等离子体色散函数测试 ===")
    for zv in [0.0, 0.5, 1.0, 2.0, 3.0]:
        Z_val = plasma_dispersion_function(zv)
        print(f"  Z({zv:4.1f}) = {Z_val}")
