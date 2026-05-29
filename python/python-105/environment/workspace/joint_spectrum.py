r"""
joint_spectrum.py
=================
联合光谱振幅 (Joint Spectral Amplitude, JSA) 与纠缠光谱分析 ——
融合原项目 1082_sinc (归一化 sinc 函数)。

在 Type-II SPDC 中，信号与闲置光子的联合态在频域可写为

.. math::
    |\Psi\rangle = \int d\omega_s \int d\omega_i \,
    f(\omega_s, \omega_i) \, a_s^\dagger(\omega_s) a_i^\dagger(\omega_i) |0\rangle

联合光谱振幅 (JSA) 分解为泵浦包络 :math:`\alpha(\omega_s+\omega_i)`
与相位匹配函数 :math:`\Phi(\omega_s, \omega_i)` 的乘积：

.. math::
    f(\omega_s, \omega_i) = \alpha(\omega_s + \omega_i) \,
    \Phi(\omega_s, \omega_i)

**相位匹配函数**（有限晶体长度 :math:`L`）

.. math::
    \Phi(\omega_s, \omega_i) = \text{sinc}\!
    \left(\frac{\Delta k(\omega_s, \omega_i) L}{2}\right)
    \exp\!\left(i \frac{\Delta k(\omega_s, \omega_i) L}{2}\right)

其中 sinc 采用归一化定义

.. math::
    \text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}, \quad \text{sinc}(0)=1

相位失配

.. math::
    \Delta k = k_p(\omega_s+\omega_i) - k_s(\omega_s) - k_i(\omega_i) - \frac{2\pi}{\Lambda}

:math:`\Lambda` 为准相位匹配周期。

**Schmidt 分解**

对 JSA 进行奇异值分解：

.. math::
    f(\omega_s, \omega_i) = \sum_{n} \sqrt{\lambda_n} \,
    u_n(\omega_s) v_n(\omega_i)

其中 :math:`\lambda_n` 为 Schmidt 系数，满足 :math:`\sum_n \lambda_n = 1`。
有效 Schmidt 数 :math:`K = 1 / \sum_n \lambda_n^2` 表征光谱纠缠维度。
态纯度 :math:`\mathcal{P} = \sum_n \lambda_n^2 = 1/K`。
"""

import numpy as np
from scipy.linalg import svd


def normalized_sinc(x: np.ndarray) -> np.ndarray:
    r"""
    归一化 sinc 函数 :math:`\text{sinc}(x) = \sin(\pi x)/(\pi x)`。

    参数
    ----
    x : np.ndarray

    返回
    ----
    s : np.ndarray
        在 x=0 处取值为 1。
    """
    x = np.atleast_1d(x)
    s = np.ones_like(x, dtype=np.float64)
    nz = np.abs(x) > 1e-16
    x_nz = x[nz]
    s[nz] = np.sin(np.pi * x_nz) / (np.pi * x_nz)
    return s


def phase_mismatch(omega_s: np.ndarray, omega_i: np.ndarray,
                   omega_p0: float, Lambda: float,
                   sellmeier_p: callable, sellmeier_s: callable,
                   sellmeier_i: callable) -> np.ndarray:
    r"""
    计算相位失配 :math:`\Delta k`。

    参数
    ----
    omega_s, omega_i : np.ndarray
        信号、闲置角频率网格（广播兼容）。
    omega_p0 : float
        泵浦中心角频率。
    Lambda : float
        准相位匹配周期。
    sellmeier_p, sellmeier_s, sellmeier_i : callable(omega) -> float
        色散关系，返回折射率 n(omega)。

    返回
    ----
    dk : np.ndarray
        相位失配量，rad/m。
    """
    c = 2.99792458e8  # m/s
    omega_p = omega_s + omega_i
    # wave numbers k = n(omega) * omega / c
    kp = sellmeier_p(omega_p) * omega_p / c
    ks = sellmeier_s(omega_s) * omega_s / c
    ki = sellmeier_i(omega_i) * omega_i / c
    dk = kp - ks - ki - 2.0 * np.pi / Lambda
    return dk


def pump_envelope_gaussian(omega_sum: np.ndarray,
                           omega_p0: float,
                           sigma_p: float) -> np.ndarray:
    r"""
    高斯型泵浦光谱包络。

    .. math::
        \alpha(\omega) = \exp\!\left(-\frac{(\omega - \omega_{p0})^2}
        {2 \sigma_p^2}\right)

    参数
    ----
    omega_sum : np.ndarray
        和频 :math:`\omega_s + \omega_i`。
    omega_p0 : float
        泵浦中心频率。
    sigma_p : float
        泵浦谱宽，> 0。

    返回
    ----
    alpha : np.ndarray
    """
    if sigma_p <= 0.0:
        raise ValueError("sigma_p 必须为正。")
    return np.exp(-((omega_sum - omega_p0) ** 2) / (2.0 * sigma_p ** 2))


def phase_matching_function(dk: np.ndarray, L: float) -> np.ndarray:
    r"""
    计算相位匹配函数 :math:`\Phi = \text{sinc}(\Delta k L / 2)`。

    参数
    ----
    dk : np.ndarray
        相位失配。
    L : float
        晶体长度，> 0。

    返回
    ----
    Phi : np.ndarray
    """
    if L <= 0.0:
        raise ValueError("晶体长度 L 必须为正。")
    arg = dk * L / (2.0 * np.pi)
    return normalized_sinc(arg)


def compute_jsa(omega_s: np.ndarray, omega_i: np.ndarray,
                omega_p0: float, sigma_p: float, L: float, Lambda: float,
                sellmeier_p: callable, sellmeier_s: callable,
                sellmeier_i: callable) -> np.ndarray:
    """
    计算联合光谱振幅 JSA 矩阵。

    参数
    ----
    omega_s, omega_i : np.ndarray, shape (n_s,) 和 (n_i,)
    omega_p0, sigma_p, L, Lambda : float
    sellmeier_* : callable(omega)->float

    返回
    ----
    jsa : np.ndarray, shape (n_s, n_i)
        复联合光谱振幅。
    """
    Os, Oi = np.meshgrid(omega_s, omega_i, indexing='ij')
    alpha = pump_envelope_gaussian(Os + Oi, omega_p0, sigma_p)
    dk = phase_mismatch(Os, Oi, omega_p0, Lambda,
                        sellmeier_p, sellmeier_s, sellmeier_i)
    phi = phase_matching_function(dk, L)
    jsa = alpha * phi
    # 归一化
    norm = np.sqrt(np.sum(np.abs(jsa) ** 2))
    if norm > 1e-20:
        jsa /= norm
    return jsa


def schmidt_decomposition_jsa(jsa: np.ndarray) -> tuple:
    r"""
    对 JSA 进行 Schmidt 分解，返回 Schmidt 系数与模式函数。

    .. math::
        f(\omega_s, \omega_i) = \sum_n \sqrt{\lambda_n} \, u_n(\omega_s) v_n(\omega_i)

    参数
    ----
    jsa : np.ndarray, shape (n_s, n_i)

    返回
    ----
    lambdas : np.ndarray
        Schmidt 系数 :math:`\lambda_n`。
    u_modes : np.ndarray, shape (n_s, r)
        信号模式函数。
    v_modes : np.ndarray, shape (n_i, r)
        闲置模式函数。
    K : float
        Schmidt 数 :math:`K = 1 / \sum \lambda_n^2`。
    purity : float
        态纯度 :math:`\mathcal{P} = 1/K`。
    """
    # TODO (Hole 1): 实现 Schmidt 分解
    # 需要使用 SVD 对 JSA 矩阵进行分解，计算 Schmidt 系数、Schmidt 数 K 与纯度 P
    raise NotImplementedError("Hole 1: 请实现 Schmidt 分解")
