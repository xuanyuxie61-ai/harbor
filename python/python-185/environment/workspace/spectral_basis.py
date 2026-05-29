"""
spectral_basis.py
=================
基于切比雪夫多项式与拉格朗日插值的谱稀疏表示模块

科学背景：
---------
在压缩感知图像重建中，信号的稀疏表示基 \Psi 决定了重建性能。
本模块结合两种经典谱方法：

1. 切比雪夫多项式基（Chebyshev polynomials of the first kind）：
   T_n(x) = \cos(n \arccos x),  x \in [-1, 1]
   切比雪夫多项式在 L^\infty 意义下具有最优逼近性质，满足：
   \|f - P_n^*\|_\infty \leq \frac{2}{\pi} \log(n+1) \cdot E_n(f)
   其中 E_n(f) 为 n 次最佳一致逼近误差。

2. 拉格朗日插值基（Lagrange basis）：
   给定节点 x_0, x_1, ..., x_n，拉格朗日基函数为
   L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}
   插值多项式 P(x) = \sum_{i=0}^{n} f(x_i) L_i(x)
   满足 P(x_i) = f(x_i)。

在图像重建中，我们将一维谱基通过张量积扩展到二维，构造：
   \Psi(x, y) = \Psi_x(x) \otimes \Psi_y(y)
从而将图像表示为稀疏的谱系数向量。
"""

import numpy as np
from typing import Callable, Tuple


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable) -> np.ndarray:
    """
    计算函数 f 在区间 [a, b] 上的 n 阶切比雪夫插值系数。

    数学推导：
        令 x_j = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{(2j-1)\pi}{2n}\right),
        j = 1, ..., n 为切比雪夫节点。
        则系数 c_k 满足：
            c_k = \frac{2}{n} \sum_{j=1}^{n} f(x_j) \cos\left(\frac{(k-1)(2j-1)\pi}{2n}\right)

    参数:
        a, b: 定义域端点
        n: 插值阶数（n >= 1）
        f: 可调用函数，接受数组输入
    返回:
        切比雪夫系数向量 c，形状为 (n,)
    """
    if n <= 0:
        raise ValueError("插值阶数 n 必须为正整数")
    if b <= a:
        raise ValueError("区间右端点 b 必须大于左端点 a")

    # 切比雪夫节点：x_j = cos((2j-1)*pi/(2n)), j=1,...,n
    angles = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)
    x_nodes = np.cos(angles)
    # 线性映射到 [a, b]
    x_physical = 0.5 * (a + b) + 0.5 * (b - a) * x_nodes
    fx = f(x_physical)

    c = np.zeros(n, dtype=float)
    for k in range(n):
        c[k] = np.sum(fx * np.cos((k) * angles))

    c *= 2.0 / n
    return c


def chebyshev_interpolant(a: float, b: float, n: int, c: np.ndarray,
                          x_eval: np.ndarray) -> np.ndarray:
    """
    利用切比雪夫系数 c 在指定点 x_eval 上求值。

    使用 Clenshaw 递推算法高效计算：
        y = (2x - a - b) / (b - a)  将 x 映射到 [-1, 1]
        d_{n+1} = d_{n+2} = 0
        d_k = 2 y d_{k+1} - d_{k+2} + c_{k+1},  k = n-1, ..., 0
        P(y) = y d_1 - d_2 + 0.5 c_1

    参数:
        a, b: 定义域端点
        n: 插值阶数
        c: 切比雪夫系数，形状为 (n,)
        x_eval: 求值点数组
    返回:
        插值结果，形状与 x_eval 相同
    """
    if len(c) != n:
        raise ValueError("系数向量长度必须与插值阶数一致")

    x_eval = np.asarray(x_eval, dtype=float).ravel()
    if b == a:
        return np.full_like(x_eval, c[0] * 0.5 if n > 0 else 0.0)

    y = (2.0 * x_eval - a - b) / (b - a)
    m = len(x_eval)

    d1 = np.zeros(m, dtype=float)
    d2 = np.zeros(m, dtype=float)

    # Clenshaw 递推
    for i in range(n - 1, 0, -1):
        d0 = 2.0 * y * d1 - d2 + c[i]
        d2 = d1
        d1 = d0

    value = y * d1 - d2 + 0.5 * c[0]
    return value


def lagrange_basis_1d(xd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    计算一维拉格朗日基函数在求值点上的取值矩阵。

    数学定义：
        给定节点 x_0, ..., x_{n-1}，第 j 个基函数为
        L_j(x) = \prod_{k \neq j} \frac{x - x_k}{x_j - x_k}

    参数:
        xd: 插值节点数组，形状为 (nd,)
        xi: 求值点数组，形状为 (ni,)
    返回:
        基函数矩阵 lb，形状为 (ni, nd)，lb[i,j] = L_j(xi[i])
    """
    xd = np.asarray(xd, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()
    nd = len(xd)
    ni = len(xi)

    if nd == 0:
        raise ValueError("插值节点不能为空")

    lb = np.ones((ni, nd), dtype=float)
    for j in range(nd):
        for k in range(nd):
            if k != j:
                denom = xd[j] - xd[k]
                if abs(denom) < 1e-14:
                    raise ValueError(f"插值节点重复或过于接近：xd[{j}]={xd[j]}, xd[{k}]={xd[k]}")
                lb[:, j] *= (xi - xd[k]) / denom
    return lb


def lagrange_value_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    利用拉格朗日插值在求值点 xi 上计算插值结果。

    数学公式：
        P(x) = \sum_{j=0}^{n-1} y_j L_j(x)

    参数:
        xd: 插值节点
        yd: 节点上的函数值
        xi: 求值点
    返回:
        插值结果数组
    """
    yd = np.asarray(yd, dtype=float).ravel()
    lb = lagrange_basis_1d(xd, xi)
    return lb @ yd


def build_2d_chebyshev_basis(image_shape: Tuple[int, int], order: int) -> np.ndarray:
    """
    构造二维切比雪夫张量积基矩阵。

    对于 H \times W 的图像，将其展平为 N = H \cdot W 维向量。
    二维切比雪夫基函数为：
        \phi_{k,l}(x, y) = T_k(x') \cdot T_l(y')
    其中 x', y' 为 [-1, 1] 上的归一化坐标。

    参数:
        image_shape: 图像尺寸 (H, W)
        order: 每维的切比雪夫阶数
    返回:
        基矩阵 Psi，形状为 (H*W, order*order)
    """
    # TODO [Hole_1]: 实现二维切比雪夫张量积基的构造
    # 科学知识点：
    #   切比雪夫多项式 T_n(x) = cos(n * arccos(x)), x \in [-1, 1]
    #   二维基通过张量积构造：\phi_{k,l}(x, y) = T_k(x') \cdot T_l(y')
    #   其中 x', y' 为归一化到 [-1, 1] 的图像坐标
    #   最终需对基矩阵 Psi 进行列归一化以保证数值稳定性
    # 提示： Psi 的形状应为 (H*W, order*order)
    # ========================================
    # 请在下方实现完整的基构造逻辑
    # ========================================
    raise NotImplementedError("Hole_1: 二维切比雪夫张量积基构造待实现")


def image_to_chebyshev_coefficients(image: np.ndarray, order: int) -> np.ndarray:
    """
    将二维图像投影到切比雪夫张量积基上，得到稀疏系数。

    数学模型：
        I(x, y) = \sum_{k=0}^{K-1} \sum_{l=0}^{K-1} c_{k,l} \phi_{k,l}(x, y)
        系数通过最小二乘求解：c = (\Psi^T \Psi)^{-1} \Psi^T \cdot \text{vec}(I)

    参数:
        image: 输入图像，形状为 (H, W)
        order: 切比雪夫阶数
    返回:
        系数向量 c，形状为 (order*order,)
    """
    if image.ndim != 2:
        raise ValueError("输入必须是二维图像")

    H, W = image.shape
    Psi = build_2d_chebyshev_basis((H, W), order)
    vec = image.ravel()

    # 正规方程求解：Psi^T Psi c = Psi^T vec
    A = Psi.T @ Psi
    b = Psi.T @ vec

    # 添加正则化以提高数值稳定性
    reg = 1e-10 * np.eye(A.shape[0])
    c = np.linalg.solve(A + reg, b)
    return c


def chebyshev_coefficients_to_image(coeffs: np.ndarray, image_shape: Tuple[int, int],
                                    order: int) -> np.ndarray:
    """
    由切比雪夫系数重建图像。

    参数:
        coeffs: 系数向量，形状为 (order*order,)
        image_shape: 目标图像尺寸 (H, W)
        order: 切比雪夫阶数
    返回:
        重建图像，形状为 (H, W)
    """
    Psi = build_2d_chebyshev_basis(image_shape, order)
    vec = Psi @ coeffs
    return vec.reshape(image_shape)
