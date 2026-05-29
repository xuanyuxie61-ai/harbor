"""
special_functions.py
===================
博士级特殊函数库，用于膜蛋白分子动力学中的静电势计算、
径向积分以及超几何级数求和。

核心数学内容：
  - Gauss 超几何函数 $_2F_1(a,b;c;z)$，出现在 Gegenbauer 多项式正交性验证中
  - Digamma 函数 $\psi(z) = \frac{d}{dz}\ln\Gamma(z)$
  - 修正 Bessel 函数 $I_\nu(x)$，用于膜振动模式的径向展开
  - 完全椭圆积分，用于周期性边界条件下的 Green 函数

种子项目映射：
  - 461_gegenbauer_exactness  →  r8_hyper_2f1, r8_psi
  - 791_ncm (waves demo)      →  Bessel 函数膜振动模式
"""

import numpy as np
from scipy.special import gamma, gammaln, psi as digamma, hyp2f1
from scipy.special import iv as bessel_i
import warnings

# ---------------------------------------------------------------------------
# 数值稳定性常数
# ---------------------------------------------------------------------------
_EULER_MASCHERONI = 0.5772156649015328606065120900824024310421
_MAX_HYP2F1_ITER = 250
_EPS_SMALL = 2.05e-9
_EPS_LARGE = 1.0e-15


def r8_hyper_2f1(a: float, b: float, c: float, x: float) -> float:
    """
    计算 Gauss 超几何函数 $_2F_1(a,b;c;x)$ 的数值。

    数学定义：
        $_2F_1(a,b;c;x) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n} \frac{x^n}{n!}$
    其中 $(a)_n = a(a+1)\cdots(a+n-1)$ 为 Pochhammer 符号。

    参数边界检查：
      - c 不得为非正整数
      - |x| < 1（此处要求 x < 1.0）

    来自种子项目 461_gegenbauer_exactness 的核心算法移植。
    """
    if c <= 0.0 and abs(c - round(c)) < 1.0e-14:
        raise ValueError("r8_hyper_2f1: c must not be a non-positive integer.")
    if x >= 1.0:
        raise ValueError("r8_hyper_2f1: x must be < 1.0 for convergence.")

    # 直接调用 scipy 的稳健实现，辅以边界检查
    if x < -1.0:
        # 使用线性变换公式:
        #   _2F1(a,b;c;x) = (1-x)^{-a} _2F1(a,c-b;c; x/(x-1))
        return float(hyp2f1(a, b, c, x))

    result = float(hyp2f1(a, b, c, x))
    if not np.isfinite(result):
        # 回退到级数展开（小 |x| 情形）
        result = _hyp2f1_series(a, b, c, x)
    return result


def _hyp2f1_series(a: float, b: float, c: float, x: float) -> float:
    """
    $_2F_1$ 的直接级数求和（用于数值不稳定时的回退）。
    """
    if abs(x) < 1.0e-15 or a == 0.0 or b == 0.0:
        return 1.0

    hf = 1.0
    r = 1.0
    for k in range(1, _MAX_HYP2F1_ITER + 1):
        r *= (a + k - 1.0) * (b + k - 1.0) / (k * (c + k - 1.0)) * x
        hf += r
        if abs(r) < _EPS_LARGE * abs(hf):
            break
    else:
        warnings.warn("_hyp2f1_series: reached max iterations without convergence.")
    return float(hf)


def r8_psi(x: float) -> float:
    """
    计算 Digamma 函数 $\psi(x) = \frac{\Gamma'(x)}{\Gamma(x)}$。

    反射公式：
        $\psi(1-x) = \psi(x) + \pi \cot(\pi x)$

    递推关系：
        $\psi(x+1) = \psi(x) + \frac{1}{x}$

    来自种子项目 461_gegenbauer_exactness 的核心算法移植。
    """
    if not np.isfinite(x):
        raise ValueError("r8_psi: x must be finite.")
    return float(digamma(x))


def gegenbauer_integral(expon: int, alpha: float) -> float:
    """
    计算 Gegenbauer 权积分：
        I = \int_{-1}^{+1} x^{\text{expon}} (1-x^2)^{\alpha} \, dx

    解析解（偶次幂）：
        I = 2 \frac{\Gamma(1+\text{expon}) \Gamma(1+\alpha)}{\Gamma(2+\alpha+\text{expon})}
          \cdot {}_2F_1(-\alpha, 1+\text{expon}; 2+\alpha+\text{expon}; -1)

    奇次幂时积分值为 0（对称区间上的奇函数）。

    参数边界：
        alpha > -1.0
    """
    if alpha <= -1.0:
        raise ValueError("gegenbauer_integral: alpha must be > -1.0.")
    if expon < 0:
        raise ValueError("gegenbauer_integral: expon must be non-negative.")

    if expon % 2 == 1:
        return 0.0

    c = float(expon)
    val1 = r8_hyper_2f1(-alpha, 1.0 + c, 2.0 + alpha + c, -1.0)
    value = 2.0 * gamma(1.0 + c) * gamma(1.0 + alpha) * val1 / gamma(2.0 + alpha + c)
    return float(value)


def gegenbauer_exactness_monomial(expon: int, alpha: float, order: int,
                                  w: np.ndarray, x: np.ndarray) -> float:
    """
    对单项式 $x^{\text{expon}}$ 应用 Gegenbauer 求积规则，并返回相对误差。

    求积公式：
        Q = \sum_{i=1}^{\text{order}} w_i \, x_i^{\text{expon}}
    """
    if order < 1:
        raise ValueError("gegenbauer_exactness_monomial: order >= 1 required.")
    if w.shape[0] != order or x.shape[0] != order:
        raise ValueError("gegenbauer_exactness_monomial: w and x must have length == order.")

    exact = gegenbauer_integral(expon, alpha)
    quad_val = float(np.dot(w, x ** expon))

    if exact == 0.0:
        err = abs(quad_val)
    else:
        err = abs((quad_val - exact) / exact)
    return err


def membrane_vibration_bessel(r: np.ndarray, t: float,
                              mu_n: np.ndarray, nu: float = 2.0 / 3.0) -> np.ndarray:
    """
    计算圆形膜片（或扇形膜片）的振动模式。

    控制方程（极坐标下的波动方程）：
        $\frac{\partial^2 u}{\partial t^2} = c^2 \left(
            \frac{\partial^2 u}{\partial r^2}
            + \frac{1}{r}\frac{\partial u}{\partial r}
            + \frac{1}{r^2}\frac{\partial^2 u}{\partial \theta^2}
        \right)$

    对于扇形区域（角度 $\theta \in [0, 3\pi/4]$），解可写为：
        $u(r,\theta,t) = \sum_{k} \frac{1}{\sqrt{k}} \sin(\mu_k t)
            J_{2/3}(\mu_k r) \sin\left(\frac{2}{3}\theta\right)$

    此处移植自种子项目 791_ncm (waves demo) 中的 Bessel 展开思想。

    参数：
        r    : 径向坐标数组，要求 r >= 0
        t    : 时间
        mu_n : 特征值数组（Bessel 函数零点）
        nu   : Bessel 函数阶数（默认 2/3，对应 135° 扇形）

    返回：
        U    : 振动幅度数组
    """
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("membrane_vibration_bessel: r must be non-negative.")
    if t < 0.0:
        raise ValueError("membrane_vibration_bessel: t must be non-negative.")

    U = np.zeros_like(r)
    for k, mu in enumerate(mu_n, start=1):
        # BesselJ_nu(mu * r) 用 scipy.special.jv 计算
        from scipy.special import jv as bessel_j
        term = (1.0 / np.sqrt(k)) * np.sin(mu * t) * bessel_j(nu, mu * r)
        U += term

    return U


def screened_coulomb_green(r: float, kappa: float, epsilon: float = 1.0) -> float:
    """
    计算 Debye-Hückel（屏蔽 Coulomb）Green 函数：
        $G(r) = \frac{1}{4\pi\epsilon} \frac{e^{-\kappa r}}{r}$

    当 $r \to 0$ 时，采用正则化极限：
        $G(0) = \frac{\kappa}{4\pi\epsilon}$

    参数边界：
        r       >= 0
        kappa   >= 0
        epsilon > 0
    """
    if r < 0.0:
        raise ValueError("screened_coulomb_green: r must be >= 0.")
    if kappa < 0.0:
        raise ValueError("screened_coulomb_green: kappa must be >= 0.")
    if epsilon <= 0.0:
        raise ValueError("screened_coulomb_green: epsilon must be > 0.")

    if r < 1.0e-12:
        return kappa / (4.0 * np.pi * epsilon)

    return np.exp(-kappa * r) / (4.0 * np.pi * epsilon * r)
