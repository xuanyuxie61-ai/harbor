"""
wiener_process.py
时空维纳过程构造与高质量伪随机数生成

融合种子项目:
  - 1040_rnglib: L'Ecuyer 高质量伪随机数生成器思想
  - 533_high_card_parfor: 蒙特卡洛采样与最优停止的分层思想

科学背景:
  对于随机偏微分方程 (SPDE)
      dU = [A U + F(U)] dt + G(U) dW(t),
  其中 W(t) 是柱维纳过程 (cylindrical Wiener process)。
  在 Galerkin 投影后，dW(t) 被截断为有限维 Q-维纳过程:
      W_N(t) = sum_{k=1}^{N} sqrt(q_k) e_k beta_k(t),
  其中 beta_k(t) 是独立标准布朗运动，q_k 是协方差算子 Q 的特征值，
  e_k 为对应的特征函数。截断误差满足:
      E[ || W(t) - W_N(t) ||^2 ] = t * sum_{k>N} q_k.
  对于核 Q 具有指数衰减特征值 q_k ~ O(k^{-2p}) 的情形，
  要达到精度 epsilon，需 N ~ O(epsilon^{-1/(2p-1)})。

核心公式:
  1. 协方差核 (Matérn 类):
       q_k = sigma^2 * (lambda^2 / (lambda^2 + k^2))^nu
  2. 时间离散增量 (Euler-Maruyama):
       Delta W_n = W(t_{n+1}) - W(t_n) ~ N(0, Delta_t * Q)
  3. 高斯随机变量生成 (Box-Muller 变换):
       Z = sqrt(-2 ln U_1) * cos(2 pi U_2),  U_1,U_2 ~ U(0,1)
  4. 反变量法方差缩减:
       Z' = -Z, 使得 Cov(Z, Z') = -Var(Z)
"""

import numpy as np
from typing import Tuple, Optional


class LEcuyerRNG:
    """
    基于 L'Ecuyer 组合多重递归生成器 (CMRG) 思想的高质量伪随机数生成器。
    参考: L'Ecuyer & Cote, ACM TOMS 17(1), 1991.

    递推关系:
        x_{1,n} = (a1 * x_{1,n-1}) mod m1
        x_{2,n} = (a2 * x_{2,n-1}) mod m2
        y_n = (x_{1,n} - x_{2,n}) mod m1
        u_n = y_n / m1   (若 y_n <= 0, 则 u_n = (y_n + m1) / m1)

    其中:
        m1 = 2147483647, a1 = 40014
        m2 = 2145483479, a2 = 40692
    """

    def __init__(self, seed1: int = 12345, seed2: int = 67890):
        if not isinstance(seed1, int) or not isinstance(seed2, int):
            raise TypeError("Seeds must be integers")
        self.m1 = 2147483647
        self.a1 = 40014
        self.m2 = 2145483479
        self.a2 = 40692
        # 边界处理：确保种子在合法范围内
        self.x1 = max(1, abs(seed1) % self.m1)
        self.x2 = max(1, abs(seed2) % self.m2)

    def _advance(self) -> float:
        self.x1 = (self.a1 * self.x1) % self.m1
        self.x2 = (self.a2 * self.x2) % self.m2
        y = (self.x1 - self.x2) % self.m1
        if y <= 0:
            y += self.m1
        return y / self.m1

    def uniform(self, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if size is None:
            return np.array(self._advance(), dtype=np.float64)
        arr = np.empty(size, dtype=np.float64)
        for idx in np.ndindex(size):
            arr[idx] = self._advance()
        return arr

    def gaussian(self, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        Box-Muller 变换生成标准正态分布。
        若启用 antithetic (反变量法)，则同时返回 -Z 以减少方差。
        """
        if size is None:
            size = (1,)
            squeeze = True
        else:
            squeeze = False

        n = np.prod(size)
        u1 = self.uniform(size=(n,))
        u2 = self.uniform(size=(n,))
        # 避免 log(0)
        eps = np.finfo(np.float64).eps
        u1 = np.clip(u1, eps, 1.0 - eps)
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        result = z.reshape(size)
        if squeeze:
            result = result.item()
        return result


class QWienerProcess:
    """
    有限维截断 Q-维纳过程。

    数学模型:
        W_N(t, x) = sum_{k=1}^{N_modes} sqrt(q_k) * e_k(x) * beta_k(t)

    其中对于 1D 空间区域 [0, L]，采用傅里叶基:
        e_k(x) = sqrt(2/L) * sin(k * pi * x / L)
        q_k = sigma^2 * k^{-2 * alpha}   (alpha > 0.5 保证迹类)

    时间增量:
        Delta beta_k ~ N(0, Delta_t)
    """

    def __init__(self,
                 spatial_grid: np.ndarray,
                 n_modes: int,
                 alpha: float = 1.0,
                 sigma: float = 1.0,
                 rng: Optional[LEcuyerRNG] = None,
                 use_antithetic: bool = False):
        if spatial_grid.ndim != 1:
            raise ValueError("spatial_grid must be 1D")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if alpha <= 0.5:
            raise ValueError("alpha must be > 0.5 for trace-class covariance")

        self.x = spatial_grid.copy()
        self.L = float(spatial_grid[-1] - spatial_grid[0])
        if self.L <= 0:
            raise ValueError("Domain length must be positive")
        self.n_modes = n_modes
        self.alpha = alpha
        self.sigma = sigma
        self.rng = rng if rng is not None else LEcuyerRNG()
        self.use_antithetic = use_antithetic

        # 特征值 q_k = sigma^2 * k^{-2*alpha}
        k = np.arange(1, n_modes + 1, dtype=np.float64)
        self.qk = (sigma ** 2) * np.power(k, -2.0 * alpha)
        # 保证数值稳定性：截断极小特征值
        q_min = np.finfo(np.float64).eps * 10.0
        self.qk = np.where(self.qk < q_min, q_min, self.qk)

        # 预计算特征函数在网格上的值 (n_modes, nx)
        nx = len(spatial_grid)
        self.eigenfuncs = np.zeros((n_modes, nx), dtype=np.float64)
        for idx, kk in enumerate(k):
            self.eigenfuncs[idx, :] = np.sqrt(2.0 / self.L) * np.sin(kk * np.pi * spatial_grid / self.L)

        # 缓存上一次的正态变量（用于反变量法）
        self._cached_normal: Optional[np.ndarray] = None
        self._cache_valid = False

    def increment(self, dt: float) -> np.ndarray:
        """
        生成时空维纳过程增量 dW ~ N(0, dt * Q)。

        返回: array shape (nx,)
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self.use_antithetic and self._cache_valid:
            dbeta = -self._cached_normal
            self._cache_valid = False
        else:
            dbeta = np.sqrt(dt) * self.rng.gaussian(size=(self.n_modes,))
            if self.use_antithetic:
                self._cached_normal = dbeta.copy()
                self._cache_valid = True

        # W_increment(x) = sum_k sqrt(q_k) * e_k(x) * dbeta_k
        coeffs = np.sqrt(self.qk) * dbeta
        dW = self.eigenfuncs.T @ coeffs
        return dW

    def strong_error_estimate(self, dt: float, p: int = 2) -> float:
        """
        估计时间离散强误差上界。
        对于 Euler-Maruyama 格式，强误差阶为 O(dt^{1/2})。
        对于 p-阶矩，误差常数涉及 Gamma 函数。
        """
        from math import gamma
        if p < 1:
            raise ValueError("p must be >= 1")
        Cp = (gamma(p + 1) / np.power(2.0, p / 2.0) / gamma(p / 2.0 + 1.0)) ** (1.0 / p)
        return Cp * np.sqrt(dt)

    def spectral_truncation_error(self, t: float) -> float:
        """
        截断误差: E[||W - W_N||^2] = t * sum_{k>N} q_k。
        对于 q_k ~ k^{-2alpha}，尾部可用积分近似:
            sum_{k>N} k^{-2alpha} ~ integral_N^infty x^{-2alpha} dx = N^{1-2alpha} / (2alpha - 1)
        """
        if self.alpha <= 0.5:
            return np.inf
        tail = np.power(self.n_modes, 1.0 - 2.0 * self.alpha) / (2.0 * self.alpha - 1.0)
        return t * (self.sigma ** 2) * tail
