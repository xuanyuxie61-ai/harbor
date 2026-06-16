"""
nonlocal_operator.py
====================
非局部算子 (Nonlocal Operator) 的实现, 灵感来自 Majorana 非局域关联。

数学背景
--------
在拓扑超导/量子线的物理背景下, Majorana 费米子的非局域关联函数
表现为长程相互作用. 我们将其推广为 VI 中的非局部算子:

    (Φ_NL(x))_i = Σ_j K(i,j; λ) · g(x_j)

其中 K(i,j; λ) 为非局部核函数, λ 为关联长度参数.

核函数类型:
    1. 指数衰减:   K(i,j) = exp(-|i-j|/λ)
    2. Yukawa 型:  K(i,j) = exp(-|i-j|/λ) / |i-j|^{d-2}  (d ≥ 3)
    3. Majorana 型: K(i,j) = exp(-|i-j|/λ) · cos(κ_F · |i-j|)
       其中 κ_F 为 Fermi 波矢, 反映振荡衰减特征
    4. RKKY 型:    K(i,j) = cos(2κ_F r)/r^d · exp(-r/λ)

在 VI 的框架下, 非局部算子 Φ_NL 使得问题从局部互补变为
耦合互补: 每个分量的互补条件受全局状态影响.

关键公式
--------
离散非局部积分:
    (Φ_NL x)_i = h^d Σ_{j} K(x_i, x_j; λ) · σ(x_j)

其中 h 为网格间距, σ 为激活函数 (如 ReLU 或 tanh).

关联长度 λ 的物理意义:
    λ → 0: 局部极限 (退化为对角算子)
    λ → ∞: 全局平均 (完全耦合)

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from scipy.special import kv as besselk  # 修正 Bessel 函数 K_ν
from scipy.special import gamma as gamma_fn


class NonlocalKernel:
    """
    非局部核函数的抽象基类与具体实现.

    物理类比:
        - 指数核: Debye-Hückel 屏蔽势
        - Yukawa 核: 核力的介子交换模型
        - Majorana 核: 拓扑超导中 Majorana 零模的波函数重叠
        - RKKY 核: 金属中磁性杂质的间接交换作用
    """

    def __init__(self, lambda_corr: float = 1.0, kappa_F: float = 2.0,
                 dimension: int = 1):
        """
        Parameters
        ----------
        lambda_corr : float
            关联长度 λ > 0
        kappa_F : float
            Fermi 波矢 κ_F > 0 (仅 Majorana/RKKY 核使用)
        dimension : int
            空间维度 d
        """
        if lambda_corr <= 0:
            raise ValueError(f"关联长度必须为正, 得到 λ={lambda_corr}")
        self.lambda_corr = lambda_corr
        self.kappa_F = kappa_F
        self.dimension = dimension

    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        """计算标量核函数 K(r)."""
        raise NotImplementedError

    def evaluate_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        计算核矩阵 K[i,j] = K(|x_i - x_j|).

        Parameters
        ----------
        positions : ndarray (N, d)
            N 个空间点的位置

        Returns
        -------
        K : ndarray (N, N)
            核矩阵
        """
        N = positions.shape[0]
        # 计算两两距离矩阵
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff**2, axis=-1) + 1e-30)
        return self.evaluate_scalar(r)


class ExponentialKernel(NonlocalKernel):
    """
    指数衰减核 (Debye-Hückel 型):
        K(r) = exp(-r / λ)
    """
    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-r / self.lambda_corr)


class YukawaKernel(NonlocalKernel):
    """
    Yukawa 核 (修正 Coulomb 势):
        K(r) = exp(-r/λ) / max(r, ε)^{d-2}     当 d ≥ 3
        K(r) = exp(-r/λ) · log(1/r)              当 d = 2
        K(r) = exp(-r/λ)                          当 d = 1
    """
    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, 1e-14)
        exp_part = np.exp(-r / self.lambda_corr)
        if self.dimension == 1:
            return exp_part
        elif self.dimension == 2:
            return exp_part * np.log(1.0 / r_safe + 1.0)
        else:
            return exp_part / (r_safe ** (self.dimension - 2))


class MajoranaKernel(NonlocalKernel):
    """
    Majorana 非局域关联核:
        K(r) = exp(-r/λ) · cos(κ_F · r) / max(r, ε)^{(d-1)/2}

    物理来源: 一维拓扑超导链中, Majorana 零模的波函数为
        ψ_M(x) ~ exp(-x/ξ) · cos(k_F x)
    两个 Majorana 模式的重叠积分给出上述核函数.

    当 κ_F · λ >> 1 时, 核函数呈现振荡衰减, 反映拓扑保护的
    非局域量子关联.
    """
    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, 1e-14)
        exp_decay = np.exp(-r / self.lambda_corr)
        oscillation = np.cos(self.kappa_F * r)
        power_law = 1.0 / (r_safe ** ((self.dimension - 1) / 2.0))
        # r = 0 时核值为 1 (自关联)
        result = exp_decay * oscillation * power_law
        result = np.where(r < 1e-14, 1.0, result)
        return result


class RKKYKernel(NonlocalKernel):
    """
    RKKY (Ruderman-Kittel-Kasuya-Yosida) 核:
        K(r) = [2κ_F r cos(2κ_F r) - sin(2κ_F r)] / (2κ_F r)^4 · exp(-r/λ)

    这是金属中两个磁性杂质间通过传导电子产生的间接交换作用的核.
    远距离表现为 cos(2κ_F r) / r^d 的振荡.
    """
    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, 1e-14)
        x = 2.0 * self.kappa_F * r_safe
        num = x * np.cos(x) - np.sin(x)
        den = x**4
        rkkY_osc = num / den
        exp_decay = np.exp(-r / self.lambda_corr)
        result = rkkY_osc * exp_decay
        # r → 0 极限
        result = np.where(r < 1e-14, 1.0 / 3.0, result)
        return result


class MaternCorrelationKernel(NonlocalKernel):
    """
    Matérn 关联核 (来自 Gaussian 过程理论):
        K(r) = σ² · c(r)
        c(r) = (2√ν · r/ρ)^ν · K_ν(2√ν · r/ρ) / (Γ(ν) · 2^{ν-1})

    其中 ρ = 关联长度, ν = 光滑性参数, K_ν = 修正 Bessel 函数.

    特殊值:
        ν = 0.5: 指数核 (OU 过程)
        ν = 1.5: 一次可微
        ν = 2.5: 二次可微
        ν → ∞: 高斯核 exp(-(r/ρ)²)
    """
    def __init__(self, lambda_corr: float = 1.0, nu: float = 2.5,
                 sigma_sq: float = 1.0, dimension: int = 1,
                 kappa_F: float = 2.0):
        super().__init__(lambda_corr, kappa_F, dimension)
        self.nu = nu
        self.sigma_sq = sigma_sq

    def evaluate_scalar(self, r: np.ndarray) -> np.ndarray:
        rho = self.lambda_corr
        nu = self.nu
        # 缩放距离
        scaled = 2.0 * np.sqrt(nu) * r / rho
        scaled_safe = np.maximum(scaled, 1e-14)
        # Matérn 公式
        bess_part = besselk(nu, scaled_safe)
        norm_const = gamma_fn(nu) * (2.0 ** (nu - 1.0))
        c_r = (scaled_safe ** nu) * bess_part / norm_const
        # r = 0 时 c(0) = 1
        c_r = np.where(r < 1e-14, 1.0, c_r)
        return self.sigma_sq * c_r


class NonlocalOperator:
    """
    非局部算子 Φ_NL: R^n → R^n 的离散实现.

    (Φ_NL(x))_i = h^d Σ_j K(x_i, x_j; λ) · σ(x_j)

    其中 σ 为非线性激活函数, 用于模拟物理系统中的非线性响应.

    激活函数选项:
        'identity': σ(x) = x
        'relu':     σ(x) = max(0, x)
        'tanh':     σ(x) = tanh(x)
        'softplus': σ(x) = log(1 + exp(x))
    """

    ACTIVATION_FUNCS = {
        'identity': lambda x: x,
        'relu': lambda x: np.maximum(0.0, x),
        'tanh': np.tanh,
        'softplus': lambda x: np.log1p(np.exp(np.clip(x, -500, 500))),
    }

    def __init__(
        self,
        kernel: NonlocalKernel,
        positions: np.ndarray,
        activation: str = 'relu',
        quadrature_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        kernel : NonlocalKernel
            非局部核函数实例
        positions : ndarray (N, d)
            空间离散点的位置
        activation : str
            激活函数名称
        quadrature_weight : float
            积分权重 h^d (网格间距的 d 次方)
        """
        if activation not in self.ACTIVATION_FUNCS:
            raise ValueError(f"未知激活函数: {activation}")
        self.kernel = kernel
        self.positions = positions
        self.activation = activation
        self.activation_fn = self.ACTIVATION_FUNCS[activation]
        self.quadrature_weight = quadrature_weight
        self.N = positions.shape[0]

        # 预计算核矩阵
        self.K_matrix = kernel.evaluate_matrix(positions)
        # 对称化 (保证正定性)
        self.K_matrix = 0.5 * (self.K_matrix + self.K_matrix.T)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        计算非局部算子: Φ_NL(x) = h^d · K · σ(x).

        其中 K 为预计算的核矩阵.
        """
        if len(x) != self.N:
            raise ValueError(f"输入维度 {len(x)} 与网格点 {self.N} 不匹配")
        sigma_x = self.activation_fn(x)
        return self.quadrature_weight * (self.K_matrix @ sigma_x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        非局部算子的 Jacobian:
            ∂(Φ_NL)_i / ∂x_j = h^d · K_{ij} · σ'(x_j)

        其中 σ' 为激活函数的导数.
        """
        sigma_prime = self._activation_derivative(x)
        return self.quadrature_weight * self.K_matrix * sigma_prime[np.newaxis, :]

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """激活函数的导数."""
        if self.activation == 'identity':
            return np.ones_like(x)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1.0 - np.tanh(x)**2
        elif self.activation == 'softplus':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return np.ones_like(x)

    def spectral_radius(self) -> float:
        """核矩阵的谱半径 (最大特征值的绝对值)."""
        eigvals = np.linalg.eigvalsh(self.K_matrix)
        return float(np.max(np.abs(eigvals)))

    def effective_range(self) -> float:
        """有效作用范围: 核函数衰减到 1/e 的距离."""
        return self.kernel.lambda_corr
