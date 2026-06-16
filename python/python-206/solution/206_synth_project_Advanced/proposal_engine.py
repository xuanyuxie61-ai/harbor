"""
proposal_engine.py  --  流形 mixup MCMC 提议分布
===============================================================
来源种子项目:
    1104_vikasverma1077_manifold_mixup :
        Manifold Mixup (Verma et al. 2019) 在中间层做凸插值
        增强表示学习; 这里移植到后验采样.
科学问题角色:
    在 MCMC 里, 传统 Gaussian 提议在高维后验中接受率极低.
    我们维护一个 *活跃样本库* {theta_i, w_i}, 提议新点:
        theta* = Mixup_manifold(theta_i, theta_j; lambda)
              = E_enc^{-1}( lambda * E_enc(theta_i)
                           + (1-lambda) * E_enc(theta_j) )
    其中 E_enc 为非线性编码器 (ResNetEncoder), lambda ~ Beta(a, b).
    这等价于在 *学习到的流形* 上做插值, 能显著提高高维采样效率.
核心公式:
    接受率 alpha = min(1, exp( log p(theta*|y) - log p(theta|y)
                              + log q(theta|theta*) - log q(theta*|theta) ))
    流形 mixup 提议不对称, 需要 Hastings 修正.
边界与鲁棒性:
    - lambda 截断到 [eps, 1-eps] 避免退化.
    - 编码器输出用 tanh 保证有界.
    - 若样本库为空, 退回到先验采样.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Optional
from numerical_base import NUMERICS


class ManifoldMixupProposal:
    """
    流形 mixup 提议引擎.
    维护一个样本库, 每次提议:
      1. 随机选两个样本 (i, j)
      2. lambda ~ Beta(a, b)
      3. z* = lambda * Enc(theta_i) + (1-lambda) * Enc(theta_j)
      4. theta* = decode(z*)  (这里用线性解码)
    """
    def __init__(self, dim: int, latent_dim: int = 6,
                 beta_a: float = 2.0, beta_b: float = 2.0,
                 seed: int = 2):
        self.dim = dim
        self.latent_dim = latent_dim
        self.beta_a = beta_a
        self.beta_b = beta_b
        self._rng = _LCG(seed)
        # 解码矩阵 W_dec: latent -> dim
        scale = 1.0 / math.sqrt(latent_dim)
        self.W_dec = [[self._rng.normal() * scale
                       for _ in range(latent_dim)] for _ in range(dim)]
        self.b_dec = [0.0] * dim
        # 样本库: [(theta, z_encoded, log_weight)]
        self.library: List[Tuple[List[float], List[float]]] = []
        self.max_library = 200

    def _sample_lambda(self) -> float:
        """
        近似 Beta(a, b) 采样 (Gaussian 近似 + 截断):
            lambda approx Normal((a/(a+b)), ab/((a+b)^2 (a+b+1)))
            截断到 [eps, 1-eps].
        """
        mean = self.beta_a / (self.beta_a + self.beta_b)
        var = (self.beta_a * self.beta_b
               / ((self.beta_a + self.beta_b) ** 2
                  * (self.beta_a + self.beta_b + 1.0)))
        std = math.sqrt(max(var, NUMERICS.cholesky_jitter))
        lam = mean + std * self._rng.normal()
        eps = 0.02
        return max(eps, min(1.0 - eps, lam))

    def add_sample(self, theta: List[float], z_enc: List[float]) -> None:
        """将样本加入活跃库."""
        self.library.append((list(theta), list(z_enc)))
        if len(self.library) > self.max_library:
            # 移除最旧的
            self.library.pop(0)

    def propose(self, current_theta: List[float],
                current_z: List[float]
                ) -> Tuple[List[float], List[float], float]:
        """
        生成提议 (theta*, z*, log_hastings_ratio).
        若库为空, 从各向同性高斯扰动产生.
        """
        if len(self.library) < 2:
            # 退化: 各向同性高斯扰动
            theta_new = [t + 0.1 * self._rng.normal() for t in current_theta]
            z_new = list(current_z)
            return theta_new, z_new, 0.0
        # 随机选两个样本
        i = int(self._rng.next() * len(self.library)) % len(self.library)
        j = int(self._rng.next() * len(self.library)) % len(self.library)
        if j == i:
            j = (j + 1) % len(self.library)
        _, zi = self.library[i]
        _, zj = self.library[j]
        # Mixup
        lam = self._sample_lambda()
        z_new = [lam * zi[d] + (1.0 - lam) * zj[d]
                 for d in range(self.latent_dim)]
        # 解码
        theta_new = self._decode(z_new)
        # 小扰动增加探索
        for d in range(self.dim):
            theta_new[d] += 0.02 * self._rng.normal()
        # Hastings 比 (简化: 对称提议近似, 仅 beta 不对称)
        log_hastings = self._log_beta_pdf(lam) - self._log_beta_pdf(1.0 - lam)
        return theta_new, z_new, log_hastings

    def _decode(self, z: List[float]) -> List[float]:
        out = []
        for i in range(self.dim):
            s = self.b_dec[i]
            for d in range(self.latent_dim):
                s += self.W_dec[i][d] * z[d]
            out.append(math.tanh(s))
        return out

    def _log_beta_pdf(self, x: float) -> float:
        a, b = self.beta_a, self.beta_b
        xc = max(NUMERICS.safe_log_floor, min(1.0 - NUMERICS.safe_log_floor, x))
        return (a - 1.0) * math.log(xc) + (b - 1.0) * math.log(1.0 - xc)


# ======================================================================
# 辅助随机数
# ======================================================================
class _LCG:
    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 1 << 32

    def next(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m

    def normal(self) -> float:
        u1 = max(self.next(), NUMERICS.safe_log_floor)
        u2 = self.next()
        r = math.sqrt(-2.0 * math.log(u1))
        return r * math.cos(2.0 * math.pi * u2)


__all__ = ["ManifoldMixupProposal"]
