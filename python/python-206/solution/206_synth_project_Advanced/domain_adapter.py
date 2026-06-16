"""
domain_adapter.py  --  Sim-to-Real 域自适应校准
===============================================================
来源种子项目:
    1079_jinlong17_S2R-ViT : 多智能体 sim-to-real 迁移,
                              ViT 风格特征对齐.
科学问题角色:
    贝叶斯校准常面临 *模型-现实差距* (model-form uncertainty):
    正向算子 G_sim(theta) 与真实观测 y_real 存在系统偏差
        y_real = G_sim(theta) + eta + delta
    其中 delta 为模型误差 (discrepancy).
    我们用 *域自适应* 估计 delta:
        1. 在 sim 域训练编码器 E_sim
        2. 在 real 域训练编码器 E_real
        3. 对齐两者的潜在分布: min || mu_sim - mu_real ||^2
                                      + || Sigma_sim - Sigma_real ||_F^2
    对齐后, 校准在 real 域进行, 消除模型偏差.
核心公式:
    MMD^2(P, Q) = || mu_P - mu_Q ||^2_Hk
    这里用线性近似: MMD^2 approx || mean(z_sim) - mean(z_real) ||^2
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict
from numerical_base import NUMERICS


class DomainAdapter:
    """
    Sim-to-Real 域自适应.
    维护 sim 域和 real 域的潜在表示, 通过 MMD 最小化对齐.
    """
    def __init__(self, latent_dim: int = 6, learning_rate: float = 0.05,
                 seed: int = 3):
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self._rng = _LCG(seed)
        # 对齐变换: z_real = W * z_sim + b
        scale = 1.0 / math.sqrt(latent_dim)
        self.W = [[self._rng.normal() * scale
                   for _ in range(latent_dim)] for _ in range(latent_dim)]
        self.b = [0.0] * latent_dim
        # 统计量
        self.sim_samples: List[List[float]] = []
        self.real_samples: List[List[float]] = []
        self.alignment_loss_history: List[float] = []

    def add_sim_sample(self, z: List[float]) -> None:
        self.sim_samples.append(list(z))

    def add_real_sample(self, z: List[float]) -> None:
        self.real_samples.append(list(z))

    def compute_mmd(self) -> float:
        """
        线性 MMD^2 = || mean(z_sim) - mean(z_real) ||^2.
        """
        if not self.sim_samples or not self.real_samples:
            return 0.0
        mu_sim = self._mean(self.sim_samples)
        mu_real = self._mean(self.real_samples)
        return sum((mu_sim[d] - mu_real[d]) ** 2
                   for d in range(self.latent_dim))

    def align(self, n_iters: int = 50) -> List[float]:
        """
        通过对齐变换 W, b 最小化 MMD.
        简化: 梯度下降.
        """
        for it in range(n_iters):
            if not self.sim_samples or not self.real_samples:
                break
            # 计算当前 MMD 梯度 (数值差分)
            loss = self.compute_mmd()
            self.alignment_loss_history.append(loss)
            # 对 W, b 做数值梯度
            eps = 1e-4
            for i in range(self.latent_dim):
                for j in range(self.latent_dim):
                    self.W[i][j] += eps
                    loss_p = self.compute_mmd()
                    self.W[i][j] -= 2 * eps
                    loss_m = self.compute_mmd()
                    self.W[i][j] += eps
                    grad = (loss_p - loss_m) / (2 * eps)
                    self.W[i][j] -= self.lr * grad
                # b
                self.b[i] += eps
                loss_p = self.compute_mmd()
                self.b[i] -= 2 * eps
                loss_m = self.compute_mmd()
                self.b[i] += eps
                grad = (loss_p - loss_m) / (2 * eps)
                self.b[i] -= self.lr * grad
        return self.alignment_loss_history

    def transform_sim_to_real(self, z_sim: List[float]) -> List[float]:
        """z_real = W * z_sim + b."""
        out = []
        for i in range(self.latent_dim):
            s = self.b[i]
            for j in range(self.latent_dim):
                s += self.W[i][j] * z_sim[j]
            out.append(s)
        return out

    def _mean(self, samples: List[List[float]]) -> List[float]:
        n = len(samples)
        if n == 0:
            return [0.0] * self.latent_dim
        mu = [0.0] * self.latent_dim
        for z in samples:
            for d in range(self.latent_dim):
                mu[d] += z[d]
        return [m / n for m in mu]

    def summary(self) -> Dict[str, float]:
        return {
            "mmd_before": self.alignment_loss_history[0] if self.alignment_loss_history else 0.0,
            "mmd_after": self.alignment_loss_history[-1] if self.alignment_loss_history else 0.0,
            "n_sim": len(self.sim_samples),
            "n_real": len(self.real_samples),
        }


# ======================================================================
# Sim-to-Real 校准流程
# ======================================================================
class SimToRealCalibration:
    """
    完整 sim-to-real 校准流程:
      1. 在 sim 域生成训练数据 { (theta_i, G_sim(theta_i)) }
      2. 收集 real 域少量观测 y_real
      3. 对齐 sim/real 潜在空间
      4. 在 real 域做贝叶斯校准
    """
    def __init__(self, latent_dim: int = 6, seed: int = 3):
        self.adapter = DomainAdapter(latent_dim, seed=seed)
        self.latent_dim = latent_dim

    def run_alignment(self, sim_z: List[List[float]],
                      real_z: List[List[float]],
                      n_iters: int = 30) -> Dict[str, float]:
        for z in sim_z:
            self.adapter.add_sim_sample(z)
        for z in real_z:
            self.adapter.add_real_sample(z)
        self.adapter.align(n_iters)
        return self.adapter.summary()


# ======================================================================
# 辅助
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


__all__ = ["DomainAdapter", "SimToRealCalibration"]
