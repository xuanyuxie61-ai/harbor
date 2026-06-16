"""
uncertain_field.py
------------------
不确定性场构造模块 —— 映射自种子项目 592_interp_equal
核心思想：利用等距节点的 Newton 插值 (均差表) 构造
不确定性参数的空间相关场，用于鲁棒优化中的随机场建模。

科学背景：
    在 SMB 色谱中，柱内扩散系数 D(x)、流速 v(x) 等参数
    沿柱长存在空间变化，可建模为随机场：
        D(x) = D_0(x) + sigma(x) * xi(x)
    其中 xi(x) 为高斯随机场，其协方差由 Toeplitz 结构近似。

    本模块利用均差插值构造随机场的空间基函数：
        p(x) = f[x_0] + f[x_0,x_1](x-x_0) + ... + f[x_0,...,x_n] prod(x-x_i)
    其中 f[x_0,...,x_k] 为 k 阶均差。

核心公式 (Atkinson-Han)：
    均差表：
        f[x_i] = y_i
        f[x_i,...,x_{i+k}] = (f[x_{i+1},...,x_{i+k}] - f[x_i,...,x_{i+k-1}]) / (x_{i+k} - x_i)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Callable


def divdif(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    均差表 (divided differences)：
        f[x_i] = y_i
        f[x_i,...,x_{i+k}] = (f[x_{i+1},...,x_{i+k}] - f[x_i,...,x_{i+k-1}]) / (x_{i+k} - x_i)
    返回长度为 n 的均差向量 (最高阶均差在末尾)。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if y.size != n:
        raise ValueError("divdif: x 和 y 长度不匹配")
    if n == 0:
        return np.zeros(0)
    # 检查节点是否重合
    for i in range(n - 1):
        if abs(x[i + 1] - x[i]) < 1e-14:
            raise ValueError(f"divdif: 节点 x[{i}] = x[{i+1}] = {x[i]} 重合")
    yd = y.copy()
    for k in range(1, n):
        for j in range(n - 1, k - 1, -1):
            yd[j] = (yd[j] - yd[j - 1]) / (x[j] - x[j - k])
    return yd


def eval_newton(xd: np.ndarray, yd: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """
    均差形式的 Newton 插值：
        p(x) = yd[n-1]
        for i = n-2..0:
            p = yd[i] + (x - xd[i]) * p
    """
    xd = np.asarray(xd, dtype=float).ravel()
    yd = np.asarray(yd, dtype=float).ravel()
    xp = np.asarray(xp, dtype=float)
    n = xd.size
    if yd.size != n:
        raise ValueError("eval_newton: 维度不匹配")
    if n == 0:
        return np.zeros_like(xp)
    yp = np.full_like(xp, yd[n - 1], dtype=float)
    for i in range(n - 2, -1, -1):
        yp = yd[i] + (xp - xd[i]) * yp
    return yp


class UncertainField:
    """
    一维随机场构造器。
    基函数：Newton 插值多项式。
    随机系数：高斯随机变量 (Toeplitz 相关)。
    """

    def __init__(
        self,
        n_nodes: int,
        domain: Tuple[float, float] = (0.0, 1.0),
        mean_fn: Optional[Callable] = None,
        sigma: float = 0.1,
        correlation_length: float = 0.2,
        toeplitz_rho: float = 0.5,
        seed: int = 0,
    ):
        self.n_nodes = n_nodes
        self.domain = domain
        self.sigma = sigma
        self.correlation_length = correlation_length
        self.toeplitz_rho = toeplitz_rho
        self.rng = np.random.default_rng(seed)

        # 等距节点
        self.x_nodes = np.linspace(domain[0], domain[1], n_nodes)
        # 均值函数
        if mean_fn is None:
            self.mean_fn = lambda x: np.ones_like(x)
        else:
            self.mean_fn = mean_fn

        # 预计算均值在节点的值
        self.mean_vals = self.mean_fn(self.x_nodes)
        # 构造 Toeplitz 协方差的 Cholesky 因子
        self.L = self._build_cholesky()

    def _build_cholesky(self) -> np.ndarray:
        """AR(1) 协方差的 Cholesky 因子."""
        n = self.n_nodes
        r = self.toeplitz_rho
        L = np.zeros((n, n))
        sq = np.sqrt(1.0 - r * r)
        for i in range(n):
            L[i, i] = 1.0
            for j in range(i):
                L[i, j] = (r ** (i - j)) * sq
        return L

    def sample_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样一个随机场实现：
            xi = L @ z,   z ~ N(0, I)
            D(x) = D_0(x) + sigma * xi
        返回 (x_nodes, D_values).
        """
        n = self.n_nodes
        z = self.rng.standard_normal(n)
        xi = self.L @ z
        D_vals = self.mean_vals + self.sigma * xi
        # 保证正值
        D_vals = np.maximum(D_vals, 1e-10)
        return self.x_nodes, D_vals

    def interpolate_field(
        self,
        x_nodes: np.ndarray,
        D_values: np.ndarray,
        x_query: np.ndarray,
    ) -> np.ndarray:
        """使用 Newton 插值在查询点评估场."""
        yd = divdif(x_nodes, D_values)
        return eval_newton(x_nodes, yd, x_query)

    def sample_and_interpolate(
        self,
        x_query: np.ndarray,
    ) -> np.ndarray:
        """采样并插值到查询点."""
        x_nodes, D_vals = self.sample_field()
        return self.interpolate_field(x_nodes, D_vals, x_query)


class KarhunenLoeveExpansion:
    """
    Karhunen-Loève 展开：
        xi(x) = sum_{k=1}^K sqrt(lambda_k) phi_k(x) z_k
    其中 (lambda_k, phi_k) 为协方差算子的特征对。
    """

    def __init__(
        self,
        n_terms: int,
        domain: Tuple[float, float] = (0.0, 1.0),
        correlation_length: float = 0.2,
        n_quad: int = 50,
        seed: int = 0,
    ):
        self.n_terms = n_terms
        self.domain = domain
        self.correlation_length = correlation_length
        self.rng = np.random.default_rng(seed)

        # 离散化协方差算子
        x_quad = np.linspace(domain[0], domain[1], n_quad)
        dx = (domain[1] - domain[0]) / (n_quad - 1)
        C = np.zeros((n_quad, n_quad))
        for i in range(n_quad):
            for j in range(n_quad):
                r = abs(x_quad[i] - x_quad[j])
                C[i, j] = np.exp(-r / correlation_length)
        C *= dx  # 积分权重
        # 特征分解
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        self.eigenvalues = vals[idx][:n_terms]
        self.eigenvectors = vecs[:, idx][:, :n_terms]
        self.x_quad = x_quad

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """采样 n_samples 个 KL 展开实现."""
        z = self.rng.standard_normal((n_samples, self.n_terms))
        sqrt_lam = np.sqrt(np.maximum(self.eigenvalues, 0.0))
        return z * sqrt_lam[None, :]

    def evaluate(self, coeffs: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """在查询点评估 KL 展开."""
        x_query = np.atleast_1d(x_query)
        n_q = x_query.size
        result = np.zeros(n_q)
        dx = (self.domain[1] - self.domain[0]) / (self.x_quad.size - 1)
        for k in range(self.n_terms):
            # 插值特征函数
            yd = divdif(self.x_quad, self.eigenvectors[:, k])
            phi_k = eval_newton(self.x_quad, yd, x_query)
            result += coeffs[k] * phi_k
        return result


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 测试均差插值
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.sin(x)
    yd = divdif(x, y)
    xp = np.linspace(0, 3, 20)
    yp = eval_newton(x, yd, xp)
    print(f"Interpolation error: {np.max(np.abs(yp - np.sin(xp))):.4e}")

    # 随机场
    uf = UncertainField(n_nodes=10, sigma=0.1, seed=42)
    x_n, D_n = uf.sample_field()
    print(f"Random field: mean={D_n.mean():.4f}, std={D_n.std():.4f}")

    # KL 展开
    kl = KarhunenLoeveExpansion(n_terms=5, correlation_length=0.3)
    coeffs = kl.sample(1)[0]
    x_q = np.linspace(0, 1, 50)
    xi = kl.evaluate(coeffs, x_q)
    print(f"KL field: min={xi.min():.4f}, max={xi.max():.4f}")
