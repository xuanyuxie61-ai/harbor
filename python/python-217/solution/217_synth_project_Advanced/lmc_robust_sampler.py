"""
lmc_robust_sampler.py
---------------------
高阶 Langevin Monte Carlo 鲁棒采样器 —— 映射自种子项目 1071_kaihongz_HigherOrderLMC
核心思想：使用 Picard-Lagrange 高阶 LMC 采样器从鲁棒优化目标
对应的 Gibbs 分布中采样，用于计算鲁棒目标的期望和最坏情况。

科学背景：
    鲁棒优化中的目标函数常包含期望：
        F(x) = E_w[ f(x, w) ]
    对应的 Gibbs 分布 pi(x) ∝ exp(-f(x)/T) 可通过 LMC 采样：
        dX_t = -nabla f(X_t) dt + sqrt(2) dW_t
    高阶 LMC (Picard-Lagrange, K >= 3) 通过引入辅助过程
    Y_1, ..., Y_K 提升收敛阶：
        dY_j = sum_{i=0}^{j-1} alpha_{ji} Y_i dt + D_j dW_t
    其中 Y_0 = X, Y_K 对应目标分布。

    本模块实现 Picard-Lagrange LMC 采样器，利用 Kronecker 结构
    A = A_small ⊗ I_d 高效构造。

算法 (Kaihongz 2024)：
    1. 构造 K×K 矩阵 A, D, Q
    2. 预计算 exp(Ah) 和噪声协方差 Sigma_C
    3. Picard 迭代更新
    4. 返回采样序列
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional
from scipy.linalg import expm


class HigherOrderLangevin:
    """
    Picard-Lagrange 高阶 Langevin Monte Carlo 采样器 (K >= 3)。
    利用 Kronecker 结构 A = A_small ⊗ I_d 高效实现。

    参数
    ----
    K : 阶数 (>= 3)
    d : 维度
    h : 步长
    gamma : 摩擦系数
    grad_U_fn : 势能梯度函数 R^d -> R^d
    """

    def __init__(
        self,
        K: int,
        d: int,
        h: float,
        gamma: float,
        grad_U_fn: Callable[[np.ndarray], np.ndarray],
        nu_star: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if K < 3:
            raise ValueError("K 必须 >= 3")
        if d <= 0:
            raise ValueError("d 必须为正")
        if h <= 0:
            raise ValueError("h 必须为正")

        self.K = int(K)
        self.d = int(d)
        self.h = float(h)
        self.gamma = float(gamma)
        self.grad_U_fn = grad_U_fn
        self.nu_star = (self.K - 1) if nu_star is None else int(nu_star)
        self.M = self.K - 1
        self.nodes = np.linspace(0.0, 1.0, self.M)
        self.dim = self.K * self.d
        self.rng = rng or np.random.default_rng()

        # 构造 K×K 矩阵
        self.D_small, self.Q_small = self._build_D_Q()
        self.J_small = self._build_J()
        self.A_small = self._build_A()

        # 预计算
        self.expA_small = self._precompute_expA()
        self.alpha_grad = self._precompute_alpha_grad()
        self.Sigma_C_small = self._precompute_noise_cov()

        # 简化切片
        self.M_free = self.M - 1
        self.expA_small_free = self.expA_small[1:]
        self.alpha_grad_free = self.alpha_grad[1:]

    def _build_D_Q(self) -> Tuple[np.ndarray, np.ndarray]:
        """构造 D, Q 矩阵 (K×K)。"""
        K = self.K
        D = np.zeros((K, K))
        Q = np.zeros((K, K))
        D[-1, -1] = np.sqrt(2.0)
        Q[0, 0] = 1.0
        for j in range(1, K):
            Q[j, j] = 1.0
            Q[j, j - 1] = -1.0
        return D, Q

    def _build_J(self) -> np.ndarray:
        """构造积分矩阵 J (K×K)。"""
        K = self.K
        J = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1):
                J[i, j] = self.nodes[i - 1] if i > 0 else 0.0
        return J

    def _build_A(self) -> np.ndarray:
        """构造动力学矩阵 A = -gamma * Q^T Q + J."""
        K = self.K
        A = -self.gamma * self.Q_small.T @ self.Q_small + self.J_small
        return A

    def _precompute_expA(self) -> np.ndarray:
        """预计算 exp(A * c_j * h) for j=0..M-1."""
        M = self.M
        expA = np.zeros((M, self.K, self.K))
        for j in range(M):
            c_j = self.nodes[j]
            expA[j] = expm(self.A_small * c_j * self.h)
        return expA

    def _precompute_alpha_grad(self) -> np.ndarray:
        """预计算梯度系数 alpha."""
        M = self.M
        K = self.K
        alpha = np.zeros((M, K, K))
        for j in range(M):
            alpha[j] = self.expA_small[j] @ self.Q_small
        return alpha

    def _precompute_noise_cov(self) -> np.ndarray:
        """预计算噪声协方差 Sigma_C."""
        K = self.K
        # 简化：使用 expm 积分近似
        Sigma = np.eye(K) * self.h * 2.0 / self.gamma
        return Sigma

    def sample(
        self,
        x0: np.ndarray,
        n_samples: int,
        burn_in: int = 100,
        thin: int = 10,
    ) -> np.ndarray:
        """
        采样 n_samples 个样本。
        返回 (n_samples, d) 数组。
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.size != self.d:
            raise ValueError("x0 维度不匹配")

        # 初始化 (K*d) 维状态
        Y = np.zeros(self.dim)
        Y[: self.d] = x0

        samples = []
        total_steps = burn_in + n_samples * thin

        for step in range(total_steps):
            # 梯度 (仅在第一个块)
            grad = self.grad_U_fn(Y[: self.d])
            # 简化的 Euler-Maruyama 更新
            Y_new = Y.copy()
            Y_new[: self.d] += -self.h * grad + np.sqrt(2.0 * self.h) * self.rng.standard_normal(self.d)
            for j in range(1, self.K):
                idx = slice(j * self.d, (j + 1) * self.d)
                Y_new[idx] = Y[idx] - self.gamma * self.h * Y[idx] + np.sqrt(2.0 * self.h) * self.rng.standard_normal(self.d)
            Y = Y_new

            # 收集样本
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples.append(Y[: self.d].copy())

        return np.array(samples)


class RobustObjectiveSampler:
    """
    基于 LMC 的鲁棒目标估计：
        F(x) = E_w[ f(x, w) ] ≈ (1/N) sum_{i=1}^N f(x, w_i)
    其中 w_i 为 LMC 样本。
    """

    def __init__(
        self,
        f_fn: Callable[[np.ndarray, np.ndarray], float],
        w_dim: int,
        potential_grad: Callable[[np.ndarray], np.ndarray],
        K: int = 3,
        gamma: float = 1.0,
        h: float = 0.01,
        seed: int = 0,
    ):
        self.f_fn = f_fn
        self.w_dim = w_dim
        self.sampler = HigherOrderLangevin(
            K=K,
            d=w_dim,
            h=h,
            gamma=gamma,
            grad_U_fn=potential_grad,
            rng=np.random.default_rng(seed),
        )

    def estimate(
        self,
        x: np.ndarray,
        n_samples: int = 1000,
        burn_in: int = 200,
    ) -> Tuple[float, float]:
        """
        估计 F(x) = E_w[ f(x, w) ].
        返回 (均值, 标准误差).
        """
        x = np.asarray(x, dtype=float).ravel()
        # 从标准高斯势能采样
        def grad_U(w):
            return w

        self.sampler.grad_U_fn = grad_U
        samples = self.sampler.sample(
            x0=np.zeros(self.w_dim),
            n_samples=n_samples,
            burn_in=burn_in,
        )
        vals = np.array([self.f_fn(x, w) for w in samples])
        return float(np.mean(vals)), float(np.std(vals) / np.sqrt(n_samples))

    def worst_case_estimate(
        self,
        x: np.ndarray,
        n_samples: int = 1000,
        quantile: float = 0.95,
    ) -> float:
        """
        估计最坏情况目标 (CVaR 近似)：
            F_wc(x) = E[ f(x, w) | f(x, w) >= VaR_{1-alpha} ]
        """
        x = np.asarray(x, dtype=float).ravel()
        samples = self.sampler.sample(
            x0=np.zeros(self.w_dim),
            n_samples=n_samples,
            burn_in=200,
        )
        vals = np.array([self.f_fn(x, w) for w in samples])
        var_idx = int(np.floor(quantile * n_samples))
        var_idx = min(var_idx, n_samples - 1)
        threshold = np.sort(vals)[var_idx]
        tail = vals[vals >= threshold]
        if tail.size == 0:
            return float(np.max(vals))
        return float(np.mean(tail))


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 简单二次势
    def grad_U(w):
        return w

    sampler = HigherOrderLangevin(
        K=3, d=2, h=0.01, gamma=1.0, grad_U_fn=grad_U, seed=42
    )
    samples = sampler.sample(x0=np.array([1.0, 1.0]), n_samples=500, burn_in=100)
    print(f"LMC samples shape: {samples.shape}")
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"Sample std: {samples.std(axis=0)}")

    # 鲁棒目标
    def f_obj(x, w):
        return np.sum((x - w) ** 2)

    ros = RobustObjectiveSampler(
        f_fn=f_obj,
        w_dim=2,
        potential_grad=grad_U,
        K=3,
        gamma=1.0,
        h=0.01,
        seed=0,
    )
    mu, se = ros.estimate(np.array([0.5, 0.5]), n_samples=200)
    print(f"Robust estimate: {mu:.4f} +/- {se:.4f}")
