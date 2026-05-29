"""
Sigmoid 平滑开关函数与高阶导数计算模块
基于 sigmoid 核心算法：Logistic 函数及其任意阶导数的幂级数展开。

在蛋白质折叠中的应用：
- 势能截断的平滑过渡（避免 MD 模拟中的能量漂移）
- 神经网络势能函数 (ANI, SchNet) 的激活函数
- 隐式溶剂模型中介电常数的连续过渡
- 反应坐标/自由能面的软开关函数

数学基础:
    Sigmoid 函数:
        σ(x) = 1 / (1 + e^{-x})
    
    高阶导数展开定理 (McKenna 2018):
        σ^{(n)}(x) = Σ_{j=1}^{n+1} c_{n,j} * σ(x)^j
    
    系数:
        c_{n,k} = Σ_{j=0}^{k} (-1)^j * (j+1)^n * C(k, j)
    
    平滑截断函数 (switching function):
        S(r) = σ( (r_c - r) / w ) = 1 / (1 + exp(-(r_c - r)/w))
    
    当 r << r_c 时 S ≈ 1；当 r >> r_c 时 S ≈ 0；
    过渡区宽度由 w 控制。
"""

import numpy as np
from typing import Tuple
from math import comb


def sigmoid_coef(n: int) -> np.ndarray:
    """
    计算 sigmoid 函数第 n 阶导数的展开系数。
    
    系数公式:
        c_{n,k} = Σ_{j=0}^{k} (-1)^j * (j+1)^n * C(k, j)
    
    Parameters
    ----------
    n : int
        导数阶数，要求 n >= 0。
    
    Returns
    -------
    coeffs : np.ndarray, shape (n+1,)
        系数 [c_{n,1}, ..., c_{n,n+1}]。
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    coeffs = np.zeros(n + 1)
    for k in range(1, n + 2):
        c_nk = 0.0
        for j in range(k + 1):
            c_nk += ((-1) ** j) * ((j + 1) ** n) * comb(k, j)
        coeffs[k - 1] = c_nk
    return coeffs


def sigmoid_value(x: np.ndarray, n_derivative: int = 0) -> np.ndarray:
    """
    计算 sigmoid 函数或其 n 阶导数在 x 处的值。
    
    使用展开定理避免直接高阶求导的数值不稳定性。
    
    Parameters
    ----------
    x : np.ndarray
        输入值。
    n_derivative : int
        导数阶数，0 表示函数本身。
    
    Returns
    -------
    values : np.ndarray
        函数值或导数值。
    """
    if n_derivative < 0:
        raise ValueError("n_derivative must be non-negative")
    
    sigma = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    if n_derivative == 0:
        return sigma
    
    coeffs = sigmoid_coef(n_derivative)
    values = np.zeros_like(x)
    for j, c in enumerate(coeffs):
        values += c * (sigma ** (j + 1))
    return values


def smooth_cutoff_function(r: np.ndarray, r_cut: float, width: float = 0.5) -> np.ndarray:
    """
    平滑截断开关函数。
    
    定义:
        S(r) = σ( (r_cut - r) / width )
             = 1 / (1 + exp(-(r_cut - r)/width))
    
    性质:
        r << r_cut  → S ≈ 1
        r = r_cut   → S = 0.5
        r >> r_cut  → S ≈ 0
    
    导数:
        dS/dr = -S'(x) / width,  其中 x = (r_cut - r) / width
    
    Parameters
    ----------
    r : np.ndarray
        距离值，要求 r >= 0。
    r_cut : float
        截断半径。
    width : float
        过渡区宽度。
    
    Returns
    -------
    S : np.ndarray
        开关函数值，范围 (0, 1)。
    """
    if width <= 0:
        raise ValueError("width must be positive")
    x = (r_cut - r) / width
    return sigmoid_value(x, 0)


def smooth_cutoff_derivative(r: np.ndarray, r_cut: float, width: float = 0.5,
                              order: int = 1) -> np.ndarray:
    """
    计算平滑截断函数的高阶导数。
    
    链式法则:
        d^n S/dr^n = (-1/width)^n * σ^{(n)}( (r_cut - r)/width )
    
    Parameters
    ----------
    r : np.ndarray
        距离值。
    r_cut : float
        截断半径。
    width : float
        过渡宽度。
    order : int
        导数阶数。
    
    Returns
    -------
    deriv : np.ndarray
        第 order 阶导数值。
    """
    if width <= 0:
        raise ValueError("width must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    
    x = (r_cut - r) / width
    sig_n = sigmoid_value(x, order)
    deriv = ((-1.0 / width) ** order) * sig_n
    return deriv


def dielectric_switch_function(r: np.ndarray, r_in: float, r_out: float,
                                eps_in: float = 4.0, eps_out: float = 80.0) -> np.ndarray:
    """
    介电常数从蛋白质内部到外部的连续过渡函数。
    
    在 Generalized Born / 隐式溶剂模型中，介电常数 ε(r) 从内部值 ε_in
    连续过渡到外部值 ε_out：
        ε(r) = ε_in + (ε_out - ε_in) * S(r)
    
    其中 S(r) 为 sigmoid 型开关函数，过渡区在 [r_in, r_out]。
    
    Parameters
    ----------
    r : np.ndarray
        到蛋白质中心的距离。
    r_in : float
        内部边界（介电常数开始过渡）。
    r_out : float
        外部边界（过渡完成）。
    eps_in : float
        内部介电常数（蛋白质内部约 2-4）。
    eps_out : float
        外部介电常数（水约 80）。
    
    Returns
    -------
    eps : np.ndarray
        空间依赖的介电常数。
    """
    if r_out <= r_in:
        raise ValueError("r_out must be greater than r_in")
    width = 0.5 * (r_out - r_in)
    r_mid = 0.5 * (r_in + r_out)
    S = smooth_cutoff_function(r, r_mid, width)
    eps = eps_in + (eps_out - eps_in) * S
    return eps


def force_switching(r: np.ndarray, r_on: float, r_off: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    分子动力学力场的平滑开关函数（能量和力的连续截断）。
    
    定义 (CHARMM 风格):
        S(r) = 1                                    (r <= r_on)
        S(r) = (r_off^2 - r^2)^2 * (r_off^2 + 2r^2 - 3r_on^2) / (r_off^2 - r_on^2)^3   (r_on < r < r_off)
        S(r) = 0                                    (r >= r_off)
    
    本函数用 sigmoid 函数近似实现等价的平滑过渡。
    
    Parameters
    ----------
    r : np.ndarray
        原子间距离。
    r_on : float
        开关开启半径。
    r_off : float
        完全截断半径。
    
    Returns
    -------
    S : np.ndarray
        能量开关函数。
    dS_dr : np.ndarray
        力开关函数 (dS/dr)。
    """
    if r_off <= r_on:
        raise ValueError("r_off must be greater than r_on")
    
    width = 0.5 * (r_off - r_on)
    r_mid = 0.5 * (r_on + r_off)
    
    S = smooth_cutoff_function(r, r_mid, width)
    dS = smooth_cutoff_derivative(r, r_mid, width, order=1)
    return S, dS
