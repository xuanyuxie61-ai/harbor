"""
special_functions.py
====================
金融数学特殊函数与数值计算工具

本模块基于以下种子项目融合:
- 221_cosine_integral: 余弦积分 Ci(x) 的级数计算 → 波动率核密度估计中的特殊函数

核心数学模型:
--------------
1.  余弦积分 Ci(x):
        Ci(x) = -∫_x^∞ cos(t)/t dt = γ + ln|x| + ∫_0^x (cos(t)-1)/t dt
    其中 γ ≈ 0.5772156649 为 Euler-Mascheroni 常数.

    在金融数学中, Ci(x) 出现在某些路径依赖期权的定价公式中,
    以及波动率曲面模型的积分核中.

    分段计算策略:
    a) |x| ≤ 16: 幂级数展开
        Ci(x) = γ + ln(x) - x²/4 + x⁴/(96) - ...
              = γ + ln(x) + Σ_{k=2}^∞ (-1)^k x^{2k-2} / [(2k-2)(2k-2)!]
    b) 16 < |x| ≤ 32: Bessel 函数展开
    c) |x| > 32: 渐近展开
        Ci(x) ≈ sin(x)/x * P(x) - cos(x)/x * Q(x)
        其中 P, Q 为渐近多项式.

2.  正弦积分 Si(x):
        Si(x) = ∫_0^x sin(t)/t dt
    与 Ci(x) 共同构成复指数积分:
        Ei(ix) = Ci(x) + i [Si(x) - π/2]

3.  对数正态分布累积函数的高精度计算:
    在 Black-Scholes 框架下, 期权 Delta:
        Δ_call = N(d1)
        d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    其中 N(·) 为标准正态 CDF.
    对极端值 (深度实值/虚值), 需要误差函数的高精度实现:
        N(x) = 0.5 [1 + erf(x/√2)]

4.  波动率核密度估计中的特殊函数核:
    采用余弦积分型核函数:
        K_h(x) = Ci(|x|/h) / (π h)
    该核函数具有快速衰减特性, 适合高频收益分布估计.

5.  数值稳定性边界处理:
    - x → 0 时, Ci(x) → -∞, 采用截断处理
    - 大参数时, 渐近级数截断点选择
    - 避免中间结果下溢/上溢
"""

import numpy as np
from typing import Union


class SpecialFunctions:
    """
    金融数学特殊函数库.
    """

    # Euler-Mascheroni 常数
    EULER_GAMMA = 0.5772156649015329
    PI_HALF = 1.570796326794897
    EPS = 1.0e-15

    @staticmethod
    def ci(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        余弦积分 Ci(x).
        基于 221_cosine_integral 的算法, 做了向量化与数值鲁棒性增强.
        """
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x)

        # 处理 x = 0
        zero_mask = np.abs(x) < 1e-18
        result[zero_mask] = -np.inf

        # 小参数 |x| ≤ 16: 幂级数
        small_mask = (~zero_mask) & (np.abs(x) <= 16.0)
        if np.any(small_mask):
            xs = x[small_mask]
            x2 = xs ** 2
            xr = -0.25 * x2
            val = SpecialFunctions.EULER_GAMMA + np.log(np.abs(xs)) + xr
            for k in range(2, 60):
                xr = -0.5 * xr * (k - 1.0) / (k * k * (2.0 * k - 1.0)) * x2
                val += xr
                if np.all(np.abs(xr) < np.abs(val) * SpecialFunctions.EPS):
                    break
            result[small_mask] = val

        # 中参数 16 < |x| ≤ 32: Bessel 展开
        medium_mask = (~zero_mask) & (np.abs(x) > 16.0) & (np.abs(x) <= 32.0)
        if np.any(medium_mask):
            xm = x[medium_mask]
            xabs = np.abs(xm)
            m = np.floor(47.2 + 0.82 * xabs).astype(int)
            m = np.clip(m, 10, 500)

            vals = np.zeros(len(xm))
            for idx in range(len(xm)):
                ma = m[idx]
                xa1 = 0.0
                xa0 = 1.0e-100
                bj = np.zeros(ma)
                for k in range(ma, 0, -1):
                    xa = 4.0 * k * xa0 / xabs[idx] - xa1
                    bj[k - 1] = xa
                    xa1 = xa0
                    xa0 = xa
                xs_sum = bj[0]
                for k in range(2, ma, 2):
                    xs_sum += 2.0 * bj[k]
                if abs(xs_sum) < 1e-18:
                    xs_sum = 1.0
                bj = bj / xs_sum

                xr = 1.0
                xg1 = bj[0]
                for k in range(1, ma):
                    xr = (0.25 * xr * (2.0 * k - 1.0) ** 2
                          / ((k) * (2.0 * k - 1.0) ** 2) * xabs[idx])
                    xg1 += bj[k] * xr

                xr = 1.0
                xg2 = bj[0]
                for k in range(1, ma):
                    xr = (0.25 * xr * (2.0 * k - 5.0) ** 2
                          / ((k) * (2.0 * k - 3.0) ** 2) * xabs[idx])
                    xg2 += bj[k] * xr

                xcs = np.cos(xabs[idx] / 2.0)
                xss = np.sin(xabs[idx] / 2.0)
                vals[idx] = (SpecialFunctions.EULER_GAMMA + np.log(xabs[idx])
                             - xabs[idx] * xss * xg1
                             + 2.0 * xcs * xg2
                             - 2.0 * xcs * xcs)
            result[medium_mask] = vals

        # 大参数 |x| > 32: 渐近展开
        large_mask = (~zero_mask) & (np.abs(x) > 32.0)
        if np.any(large_mask):
            xl = x[large_mask]
            x2 = xl ** 2
            xabs = np.abs(xl)

            xr = 1.0
            xf = 1.0
            for k in range(1, 10):
                xr = -2.0 * xr * k * (2 * k - 1) / x2
                xf += xr

            xr = 1.0 / xabs
            xg = xr.copy()
            for k in range(1, 9):
                xr = -2.0 * xr * (2 * k + 1) * k / x2
                xg += xr

            result[large_mask] = (xf * np.sin(xabs) / xabs
                                  - xg * np.cos(xabs) / xabs)

        if scalar_input:
            return float(result[0])
        return result

    @staticmethod
    def si(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        正弦积分 Si(x) = ∫_0^x sin(t)/t dt.
        采用幂级数展开:
            Si(x) = Σ_{k=0}^∞ (-1)^k x^{2k+1} / [(2k+1)(2k+1)!]
        """
        scalar_input = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))

        val = x.copy()
        term = x.copy()
        x2 = x ** 2

        for k in range(1, 80):
            term *= -x2 / ((2.0 * k) * (2.0 * k + 1.0) ** 2 / (2.0 * k - 1.0))
            # 更稳定的递推
            term = -term * x2 / ((2.0 * k) * (2.0 * k + 1.0))
            # 重新计算避免累积误差
            term = ((-1.0) ** k) * (x ** (2 * k + 1)) / ((2 * k + 1) * np.math.factorial(2 * k + 1))
            # 直接计算会导致溢出, 改用比值递推
            # 简化: 直接用泰勒级数逐项比
            # 实际上对 |x| 不太大时直接求和即可
            break

        # 更稳定的实现
        result = np.zeros_like(x)
        for i in range(len(x)):
            xv = x[i]
            if abs(xv) < 1e-12:
                result[i] = 0.0
                continue
            s = xv
            term = xv
            for k in range(1, 100):
                term *= -xv * xv / ((2.0 * k) * (2.0 * k + 1.0))
                s += term
                if abs(term) < abs(s) * SpecialFunctions.EPS:
                    break
            result[i] = s

        if scalar_input:
            return float(result[0])
        return result

    @staticmethod
    def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        标准正态累积分布函数.
        采用互补误差函数:
            N(x) = 0.5 * [1 + erf(x/√2)]
        对极端值采用 Abramowitz & Stegun 近似.
        """
        from math import erf
        if np.isscalar(x):
            if x < -8.0:
                return 0.0
            if x > 8.0:
                return 1.0
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

        x_arr = np.asarray(x, dtype=float)
        result = np.empty_like(x_arr)

        # 极端值截断
        result[x_arr <= -8.0] = 0.0
        result[x_arr >= 8.0] = 1.0

        mid_mask = (x_arr > -8.0) & (x_arr < 8.0)
        if np.any(mid_mask):
            xm = x_arr[mid_mask]
            # 向量化的 erf 不总是可用, 手动近似
            # Hart 近似
            result[mid_mask] = SpecialFunctions._hart_cdf(xm)

        return result

    @staticmethod
    def _hart_cdf(x: np.ndarray) -> np.ndarray:
        """Hart 近似公式计算 N(x), 适合向量化."""
        # 基于误差函数的近似
        # 使用 scipy.special 如果可用, 否则用近似
        try:
            from scipy.special import erf
            return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        except ImportError:
            # 简单的有理近似 (Abramowitz & Stegun 26.2.17)
            abs_x = np.abs(x)
            t = 1.0 / (1.0 + 0.2316419 * abs_x)
            poly = (0.319381530 * t
                    - 0.356563782 * t ** 2
                    + 1.781477937 * t ** 3
                    - 1.821255978 * t ** 4
                    + 1.330274429 * t ** 5)
            pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * abs_x ** 2)
            cdf = 1.0 - pdf * poly
            result = np.where(x < 0, 1.0 - cdf, cdf)
            return result

    @staticmethod
    def black_scholes_delta(S: float, K: float, T: float,
                            r: float, sigma: float,
                            option_type: str = "call") -> float:
        """
        Black-Scholes Delta.
            d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
            Δ_call = N(d1),  Δ_put = N(d1) - 1
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = SpecialFunctions.normal_cdf(d1)
        if option_type == "put":
            delta -= 1.0
        return float(delta)

    @staticmethod
    def kernel_ci(x: np.ndarray, h: float) -> np.ndarray:
        """
        基于余弦积分的核密度估计核函数.
            K_h(x) = Ci(|x|/h) / (π h)
        注意: Ci 在 0 附近为负, 需要截断保证非负性.
        实际使用修正版本:
            K_h(x) = max(0, Ci(|x|/h) + γ + ln(h)) / C
        其中 C 为归一化常数.
        """
        if h <= 0:
            raise ValueError("带宽 h 必须为正.")
        u = np.abs(x) / h
        # 修正核: 利用 Ci(u) + γ + ln(u) 在 u→0 时趋于 0
        raw = SpecialFunctions.ci(u) + SpecialFunctions.EULER_GAMMA + np.log(u + 1e-18)
        raw = np.maximum(raw, 0.0)
        # 数值归一化
        C = np.trapz(raw, x) if len(x) > 1 else 1.0
        if C > 0:
            raw /= C
        return raw
