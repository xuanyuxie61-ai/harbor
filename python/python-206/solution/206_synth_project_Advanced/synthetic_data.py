"""
synthetic_data.py  --  合成观测数据生成器
===============================================================
科学问题角色:
    为贝叶斯校准提供 "ground truth" 观测.
    1. 给定真实参数 theta_true.
    2. 运行正向算子 G(theta_true) 得到无噪输出.
    3. 加高斯观测噪声: y_obs = G(theta_true) + epsilon,
       epsilon ~ N(0, Sigma_obs).
    4. 同时生成 sim 域和 real 域数据 (用于域自适应).
"""
from __future__ import annotations
import math
from typing import List, Dict, Tuple
from numerical_base import NUMERICS
from forward_model import GrayScottForward


class SyntheticDataGenerator:
    """
    合成观测数据生成器.
    """
    def __init__(self, theta_true: Dict[str, float],
                 nx: int = 11, ny: int = 11,
                 obs_noise_std: float = 0.02,
                 model_discrepancy: float = 0.01,
                 seed: int = 42):
        self.theta_true = theta_true
        self.forward = GrayScottForward(nx=nx, ny=ny, seed=seed)
        self.obs_noise_std = obs_noise_std
        self.model_discrepancy = model_discrepancy
        self._rng = _LCG(seed)
        self.n_obs = self.forward.n_obs

    def generate_sim_data(self) -> Tuple[List[float], List[float]]:
        """
        生成 sim 域数据: y_sim = G(theta_true) + noise.
        返回 (y_sim, theta_true_vec).
        """
        y_clean = self.forward.evaluate(self.theta_true)
        y_sim = [y + self.obs_noise_std * self._rng.normal()
                 for y in y_clean]
        return y_sim, self._theta_to_vec(self.theta_true)

    def generate_real_data(self) -> Tuple[List[float], List[float]]:
        """
        生成 real 域数据: y_real = G(theta_true) + noise + discrepancy.
        discrepancy 模拟模型误差.
        """
        y_clean = self.forward.evaluate(self.theta_true)
        y_real = [y + self.obs_noise_std * self._rng.normal()
                  + self.model_discrepancy * self._rng.normal()
                  for y in y_clean]
        return y_real, self._theta_to_vec(self.theta_true)

    def generate_training_set(self, n_samples: int = 20
                               ) -> Tuple[List[List[float]],
                                          List[List[float]]]:
        """
        生成代理训练集: 在参数空间采样 {theta_i}, 计算 {G(theta_i)}.
        """
        thetas = []
        ys = []
        for _ in range(n_samples):
            # 在 theta_true 附近扰动
            theta = {
                "Du": self.theta_true["Du"] * (1.0 + 0.3 * self._rng.normal()),
                "Dv": self.theta_true["Dv"] * (1.0 + 0.3 * self._rng.normal()),
                "f": max(0.01, min(0.99, self.theta_true["f"] + 0.1 * self._rng.normal())),
                "k": max(0.01, min(0.99, self.theta_true["k"] + 0.1 * self._rng.normal())),
            }
            y = self.forward.evaluate(theta)
            thetas.append(self._theta_to_vec(theta))
            ys.append(y)
        return thetas, ys

    def _theta_to_vec(self, theta: Dict[str, float]) -> List[float]:
        """theta dict -> [log(Du), log(Dv), f, k]."""
        return [
            math.log(max(NUMERICS.safe_log_floor, theta["Du"])),
            math.log(max(NUMERICS.safe_log_floor, theta["Dv"])),
            theta["f"],
            theta["k"],
        ]


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


__all__ = ["SyntheticDataGenerator"]
