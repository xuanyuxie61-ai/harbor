"""
special_functions.py
====================
催化剂孔扩散与表面反应模拟所需的特殊数学函数。

基于种子项目 1268_toms243 与 461_gegenbauer_exactness 重构：
- toms243 提供复数自然对数的稳定数值算法；
- gegenbauer_exactness 包含超几何函数 2F1、Psi (Digamma) 函数、
  Gegenbauer 积分与 Gamma 函数组合。

在本系统中，这些特殊函数用于：
1. 球坐标下扩散方程的解析解（涉及复对数与超几何函数）；
2. 孔径分布的矩计算（涉及 Gamma 函数与不完全 Gamma 函数）；
3. 催化剂颗粒温度场的谱方法展开（Gegenbauer 多项式）。
"""

import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy.special import digamma as scipy_psi
from scipy.special import hyp2f1 as scipy_hyp2f1


class SpecialFunctionError(Exception):
    """特殊函数计算异常。"""
    pass


def complex_log_stable(z):
    """
    计算复数的自然对数，采用 toms243 的稳定算法思想。

    对于复数 z = a + bi，有
        ln(z) = ln|z| + i * arg(z)
    其中 |z| = sqrt(a^2 + b^2)，arg(z) = atan2(b, a)。

    toms243 的核心在于通过缩放避免大数溢出：
    若 |a/2| < 0.5 且 |b/2| < 0.5，使用小值路径；
    否则使用大值路径。

    Parameters
    ----------
    z : complex 或 ndarray of complex

    Returns
    -------
    ln_z : complex 或 ndarray
        ln(z) 的稳定数值结果。
    """
    z = np.asarray(z, dtype=complex)
    a = z.real
    b = z.imag

    # 零值处理
    is_zero = (a == 0.0) & (b == 0.0)

    e = a / 2.0
    f = b / 2.0
    small_mask = (np.abs(e) < 0.5) & (np.abs(f) < 0.5)

    c = np.empty_like(a)
    d = np.empty_like(a)

    # 小值路径
    if np.any(small_mask):
        ca = np.abs(2.0 * a[small_mask]) + np.abs(2.0 * b[small_mask])
        # 避免除以零
        ca = np.where(ca == 0, np.finfo(float).tiny, ca)
        da = 8.0 * (a[small_mask] / ca) * a[small_mask] \
             + 8.0 * (b[small_mask] / ca) * b[small_mask]
        c[small_mask] = 0.5 * (np.log(ca) + np.log(da)) - np.log(np.sqrt(8.0))

    # 大值路径
    large_mask = ~small_mask
    if np.any(large_mask):
        cb = np.abs(e[large_mask] / 2.0) + np.abs(f[large_mask] / 2.0)
        db = 0.5 * (e[large_mask] / cb) * e[large_mask] \
             + 0.5 * (f[large_mask] / cb) * f[large_mask]
        c[large_mask] = 0.5 * (np.log(cb) + np.log(db)) + np.log(np.sqrt(8.0))

    # 幅角计算（atan2 天然稳定）
    d = np.arctan2(b, a)

    ln_z = c + 1j * d
    if np.any(is_zero):
        if np.isscalar(ln_z):
            ln_z = np.nan + 1j * np.nan
        else:
            ln_z = np.asarray(ln_z, dtype=complex)
            ln_z[is_zero] = np.nan + 1j * np.nan
    return ln_z


def gegenbauer_integral(expon, alpha):
    r"""
    计算带 Gegenbauer 权重的单项式积分：

        I = \int_{-1}^{+1} x^{expon} (1 - x^2)^{\alpha} dx

    解析解涉及超几何函数 2F1 与 Gamma 函数：
        I = 2 \frac{\Gamma(1+expon) \Gamma(1+\alpha)}
                  {\Gamma(2+\alpha+expon)}
            {}_2F_1(-\alpha, 1+expon; 2+\alpha+expon; -1)

    若 expon 为奇数，被积函数为奇函数，积分值为 0。

    Parameters
    ----------
    expon : int
        单项式指数，非负整数。
    alpha : float
        Gegenbauer 参数，alpha > -1。

    Returns
    -------
    value : float
    """
    if expon < 0:
        raise SpecialFunctionError("expon 必须为非负整数")
    if alpha <= -1.0:
        raise SpecialFunctionError("alpha 必须大于 -1")

    if expon % 2 == 1:
        return 0.0

    # 使用 scipy 的特殊函数计算
    arg1 = -alpha
    arg2 = 1.0 + expon
    arg3 = 2.0 + alpha + expon
    arg4 = -1.0

    value1 = scipy_hyp2f1(arg1, arg2, arg3, arg4)
    value = (2.0 * scipy_gamma(1.0 + expon) * scipy_gamma(1.0 + alpha)
             * value1 / scipy_gamma(2.0 + alpha + expon))
    return float(value)


def thiele_modulus_efficiency_factor(phi, shape_factor=3):
    r"""
    计算催化剂颗粒的内部效率因子 η。

    对于球形颗粒（shape_factor = 3）：
        η = \frac{3}{\phi^2} (\phi \coth\phi - 1)

    对于无限长圆柱（shape_factor = 2）：
        η = \frac{2}{\phi} \frac{I_1(\phi)}{I_0(\phi)}

    对于无限大平板（shape_factor = 1）：
        η = \frac{\tanh\phi}{\phi}

    其中 φ 为 Thiele 模数，定义如下：
        φ = L \sqrt{\frac{k_{obs}}{D_e}}
    L 为特征长度（球半径、圆柱半径或平板半厚度）。

    Parameters
    ----------
    phi : float or ndarray
        Thiele 模数，必须为非负数。
    shape_factor : int, default 3
        颗粒形状因子（1=平板, 2=圆柱, 3=球）。

    Returns
    -------
    eta : float or ndarray
        内部效率因子，范围 (0, 1]。
    """
    phi = np.asarray(phi, dtype=float)
    if np.any(phi < 0):
        raise SpecialFunctionError("Thiele 模数 phi 必须非负")

    eps = np.finfo(float).eps
    phi_safe = np.where(phi < eps, eps, phi)

    if shape_factor == 3:
        # 球形
        eta = 3.0 / (phi_safe ** 2) * (phi_safe / np.tanh(phi_safe) - 1.0)
        # phi -> 0 极限为 1
        eta = np.where(phi < eps, 1.0, eta)
    elif shape_factor == 2:
        from scipy.special import i0, i1
        eta = 2.0 / phi_safe * i1(phi_safe) / i0(phi_safe)
        eta = np.where(phi < eps, 1.0, eta)
    elif shape_factor == 1:
        eta = np.tanh(phi_safe) / phi_safe
        eta = np.where(phi < eps, 1.0, eta)
    else:
        raise SpecialFunctionError("shape_factor 必须为 1、2 或 3")

    # 数值截断
    eta = np.clip(eta, 0.0, 1.0)
    return eta


def knudsen_diffusivity(pore_diameter, temperature, molecular_weight):
    r"""
    计算 Knudsen 扩散系数 D_Kn。

    公式：
        D_{Kn} = \frac{d_p}{3} \sqrt{\frac{8 R T}{\pi M}}

    其中：
        d_p : 孔径 [m]
        R   : 通用气体常数 8.314 J/(mol·K)
        T   : 温度 [K]
        M   : 分子量 [kg/mol]

    Parameters
    ----------
    pore_diameter : float
        孔径 [m]，必须为正。
    temperature : float
        温度 [K]，必须为正。
    molecular_weight : float
        分子量 [kg/mol]，必须为正。

    Returns
    -------
    D_kn : float
        Knudsen 扩散系数 [m²/s]。
    """
    R = 8.314462618  # J/(mol·K)
    if pore_diameter <= 0 or temperature <= 0 or molecular_weight <= 0:
        raise SpecialFunctionError("孔径、温度、分子量必须为正")

    D_kn = (pore_diameter / 3.0) * np.sqrt((8.0 * R * temperature)
                                           / (np.pi * molecular_weight))
    return D_kn


def effective_diffusivity(pore_diameter, temperature, molecular_weight,
                          bulk_diffusivity, tortuosity, porosity):
    r"""
    计算催化剂有效扩散系数 D_e。

    综合 Bulk 扩散与 Knudsen 扩散的并联阻力模型：
        \frac{1}{D_{eff, pore}} = \frac{1}{D_{bulk}} + \frac{1}{D_{Kn}}

    再考虑孔隙率 ε 与曲折因子 τ：
        D_e = \frac{\varepsilon}{\tau} D_{eff, pore}

    Parameters
    ----------
    pore_diameter : float
    temperature : float
    molecular_weight : float
    bulk_diffusivity : float
        体相扩散系数 [m²/s]。
    tortuosity : float
        曲折因子 τ ≥ 1。
    porosity : float
        孔隙率 ε ∈ (0, 1)。

    Returns
    -------
    D_e : float
    """
    if bulk_diffusivity <= 0:
        raise SpecialFunctionError("bulk_diffusivity 必须为正")
    if tortuosity < 1.0:
        raise SpecialFunctionError("tortuosity 必须 ≥ 1")
    if not (0.0 < porosity < 1.0):
        raise SpecialFunctionError("porosity 必须在 (0, 1) 之间")

    D_kn = knudsen_diffusivity(pore_diameter, temperature, molecular_weight)
    D_eff_pore = 1.0 / (1.0 / bulk_diffusivity + 1.0 / D_kn)
    D_e = (porosity / tortuosity) * D_eff_pore
    return D_e


def arrhenius_rate(pre_exp, activation_energy, temperature):
    r"""
    Arrhenius 速率常数：
        k = A \exp\left(-\frac{E_a}{R T}\right)

    Parameters
    ----------
    pre_exp : float
        指前因子 A。
    activation_energy : float
        活化能 E_a [J/mol]。
    temperature : float
        温度 [K]。

    Returns
    -------
    k : float
    """
    R = 8.314462618
    if temperature <= 0:
        raise SpecialFunctionError("温度必须为正")
    k = pre_exp * np.exp(-activation_energy / (R * temperature))
    return k
