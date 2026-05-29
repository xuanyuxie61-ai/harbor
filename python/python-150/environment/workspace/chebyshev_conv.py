"""
chebyshev_conv.py
=================
Chebyshev 谱图卷积层

融合种子项目:
  - 160_chebyshev_interp_1d : Chebyshev 多项式插值与 Vandermonde 系统

科学背景:
  在谱图理论中，图卷积由谱域滤波器定义:
      y = U g_θ(Λ) U^T x
  其中 L = UΛU^T 为图拉普拉斯特征分解。为规避 O(N^3) 分解，
  Defferrard et al. (2016) 提出用 Chebyshev 多项式逼近 g_θ:

      g_θ(Λ) ≈ Σ_{k=0}^{K-1} θ_k T_k(Ḻ)

  其中 Ḻ = 2L/λ_max - I 将特征值缩放至 [-1, 1]，T_k 满足递推:
      T_0(x) = 1
      T_1(x) = x
      T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)

  该层严格遵循上述数学结构，仅依赖矩阵-向量乘法，计算复杂度 O(K|E|)。
"""

import numpy as np
from typing import Tuple


class ChebyshevGraphConv:
    """
    Chebyshev 谱图卷积层。
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 4):
        """
        Parameters
        ----------
        in_channels : int
            输入特征维度。
        out_channels : int
            输出特征维度。
        K : int
            Chebyshev 多项式阶数 (截断阶数 K-1)。
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        # 可学习参数 θ: shape (K, in_channels, out_channels)
        # 使用 Xavier 风格初始化
        limit = np.sqrt(6.0 / (in_channels + out_channels))
        self.theta = np.random.uniform(-limit, limit, (K, in_channels, out_channels))
        self.bias = np.zeros(out_channels, dtype=np.float64)

    def __call__(self, x: np.ndarray, laplacian_mul) -> np.ndarray:
        """
        前向传播。

        Parameters
        ----------
        x : np.ndarray, shape (n_nodes, in_channels)
            节点特征。
        laplacian_mul : callable
            接受 x 并返回 Ḻ @ x 的函数。

        Returns
        -------
        y : np.ndarray, shape (n_nodes, out_channels)
            输出特征。
        """
        n_nodes = x.shape[0]
        # 收集 T_k(Ḻ) x
        cheb_x = np.zeros((self.K, n_nodes, self.in_channels), dtype=np.float64)
        # T_0 = I
        cheb_x[0] = x
        if self.K > 1:
            # T_1 = Ḻ x
            cheb_x[1] = laplacian_mul(x)
        # 递推 T_k = 2 Ḻ T_{k-1} - T_{k-2}
        for k in range(2, self.K):
            cheb_x[k] = 2.0 * laplacian_mul(cheb_x[k - 1]) - cheb_x[k - 2]

        # 线性组合: y = Σ_k T_k(x) @ θ_k + b
        y = np.zeros((n_nodes, self.out_channels), dtype=np.float64)
        for k in range(self.K):
            y += cheb_x[k] @ self.theta[k]
        y += self.bias
        return y

    def parameters(self) -> list:
        return [self.theta, self.bias]


def chebyshev_coefficients_1d(nd: int, xd: np.ndarray, yd: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    计算一维 Chebyshev 插值系数（源自 chebyshev_interp_1d）。

    将数据点 xd 仿射映射到 [-1, 1]，求解线性系统:
        A_{ij} = cos(i * arccos(x_j))
    得到系数 c，使得 y(t) ≈ Σ_{i=0}^{nd-1} c_i T_i(t)。

    Returns
    -------
    c : np.ndarray
        Chebyshev 系数。
    xmin, xmax : float
        原始数据域边界。
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    xmin, xmax = xd.min(), xd.max()
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0
    # 仿射映射到 [-1, 1]
    t = 2.0 * (xd - xmin) / (xmax - xmin) - 1.0
    # 避免 arccos 定义域越界
    t = np.clip(t, -1.0, 1.0)
    theta = np.arccos(t)
    # Vandermonde 矩阵
    A = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        A[i, :] = np.cos(i * theta)
    # 最小二乘求解（允许 nd 与数据点数不同）
    c, _, _, _ = np.linalg.lstsq(A.T, yd, rcond=None)
    return c, xmin, xmax


def chebyshev_value_1d(c: np.ndarray, xmin: float, xmax: float, xi: np.ndarray) -> np.ndarray:
    """
    用 Clenshaw 递推在任意点 xi 处求值 Chebyshev 展开。
    """
    xi = np.asarray(xi, dtype=np.float64)
    t = 2.0 * (xi - xmin) / (xmax - xmin) - 1.0
    t = np.clip(t, -1.0, 1.0)
    nd = len(c)
    # Clenshaw
    b0 = np.zeros_like(t)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for i in range(nd - 1, 0, -1):
        b0 = 2.0 * t * b1 - b2 + c[i]
        b2 = b1
        b1 = b0
    return b1 * t - b2 + c[0]
