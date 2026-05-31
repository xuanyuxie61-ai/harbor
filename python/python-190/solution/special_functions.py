"""
special_functions.py
====================
基于种子项目 187_clausen 的特殊函数模块。
提供 Clausen 函数 Cl₂(x)、Chebyshev 级数求值以及基于特殊函数的
激活函数变体，用于物理信息 GAN 的生成器/判别器网络层。

核心数学：
  1. Clausen 函数定义：
       Cl₂(x) = -∫₀ˣ ln|2 sin(t/2)| dt = Σ_{k=1}^∞ sin(kx) / k²
     性质：
       · 2π-周期函数
       · 奇函数：Cl₂(-x) = -Cl₂(x)
       · Cl₂(0) = Cl₂(2π) = 0
       · 在 x = 0 处可去奇点，导数趋向 +∞

  2. Chebyshev 级数（Clenshaw 递推）：
       给定系数 a_1, ..., a_n，求 S(x) = Σ_{k=1}^n a_k · T_{k-1}(x')
       其中 x' ∈ [-1,1]。
       递推公式：
         b_{n+2} = b_{n+1} = 0
         b_k = 2x'·b_{k+1} - b_{k+2} + a_k   (k = n, n-1, ..., 1)
         S(x) = 0.5·(b_1 - b_3)

  3. 基于 Cl₂(x) 的周期性激活函数（用于保持 PDE 的周期边界条件）：
       σ_clausen(x) = Cl₂(π·x) / (π²/6)
       归一化因子 ζ(2) = π²/6 使得 |σ(x)| ≤ 1。
"""

import numpy as np


def r8_csevl(x: float, a: np.ndarray) -> float:
    """
    使用 Clenshaw 递推计算 Chebyshev 级数在 x 处的值。

    Parameters
    ----------
    x : float
        求值点，必须在 [-1, 1] 区间内。
    a : np.ndarray
        Chebyshev 系数数组。

    Returns
    -------
    value : float
        级数值。
    """
    if x < -1.0 or x > 1.0:
        raise ValueError("Chebyshev 求值点 x 必须在 [-1, 1] 内。")
    n = len(a)
    if n == 0:
        return 0.0
    if n == 1:
        return float(a[0])

    b0 = float(a[-1])
    b1 = 0.0
    b2 = 0.0
    for i in range(n - 2, -1, -1):
        b2 = b1
        b1 = b0
        b0 = 2.0 * x * b1 - b2 + float(a[i])
    return 0.5 * (b0 - b2)


def clausen(x: float) -> float:
    """
    计算 Clausen 函数 Cl₂(x)。

    使用定义级数求和：
      Cl₂(x) = Σ_{k=1}^∞ sin(k·x) / k²
    该级数绝对收敛，取前 20000 项可达到双精度精度。

    Parameters
    ----------
    x : float
        自变量（弧度）。

    Returns
    -------
    value : float
        Cl₂(x) 的近似值。
    """
    # 约化到 [-π, π] 区间（利用 2π 周期性）
    twopi = 2.0 * np.pi
    x_red = x
    while x_red < -np.pi:
        x_red += twopi
    while x_red > np.pi:
        x_red -= twopi

    eps = np.finfo(float).eps
    if abs(x_red) < eps:
        return 0.0

    # 级数求和（向量化，20000 项）
    k = np.arange(1, 20001)
    value = float(np.sum(np.sin(k * x_red) / (k ** 2)))
    return value


def clausen_array(x: np.ndarray) -> np.ndarray:
    """向量化 Clausen 函数。"""
    x = np.asarray(x, dtype=float)
    return np.vectorize(clausen)(x)


def clausen_activation(x: np.ndarray) -> np.ndarray:
    """
    基于 Clausen 函数的周期性激活函数。
    归一化到 [-1, 1] 区间，适合用于具有周期性边界条件的物理场生成。

    σ_clausen(x) = Cl₂(π·x) / (π²/6)

    Parameters
    ----------
    x : np.ndarray
        输入。

    Returns
    -------
    out : np.ndarray
        激活后输出，值域约为 [-1, 1]。
    """
    x = np.asarray(x, dtype=float)
    scale = 6.0 / (np.pi ** 2)
    return scale * clausen_array(np.pi * x)


def clausen_activation_derivative(x: np.ndarray) -> np.ndarray:
    """
    Clausen 激活函数的导数。

    d/dx Cl₂(π·x) = -π·ln|2·sin(π·x/2)|
    σ'(x) = -6/π · ln|2·sin(π·x/2)|

    边界处理：当 sin(π·x/2) 接近 0 时，ln 趋向 -∞，需截断。
    """
    x = np.asarray(x, dtype=float)
    s = np.sin(np.pi * x * 0.5)
    s = np.where(np.abs(s) < 1e-12, 1e-12, s)
    # 导数 = -ln|2·s|
    deriv = -np.log(2.0 * np.abs(s))
    # 截断防止爆炸
    deriv = np.clip(deriv, -50.0, 50.0)
    scale = 6.0 / np.pi
    return scale * deriv


def special_function_spectral_basis(x: np.ndarray, n_modes: int = 8) -> np.ndarray:
    """
    构建基于 Clausen 函数的谱基函数，用于生成器的特征展开。

    φ_k(x) = Cl₂(k·π·x / n_modes) / (π²/6)

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        输入坐标。
    n_modes : int
        模态数。

    Returns
    -------
    basis : np.ndarray, shape (N, n_modes)
        谱基函数矩阵。
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    basis = np.zeros((N, n_modes))
    for k in range(1, n_modes + 1):
        basis[:, k - 1] = clausen_activation(k * x / n_modes)
    return basis
