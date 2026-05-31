"""
random_generator.py
高级伪随机数生成器与事件采样

基于种子项目:
- 763_middle_square: 中平方法伪随机数生成

物理应用:
1. 蒙特卡洛事件生成中的随机数需求。
2. 部分子级联中的随机抽样。
3. 事件-by-事件涨落采样。

改进方案:
将middle-square与线性同余生成器(LCG)混合，
形成更鲁棒的混合伪随机数生成器。
"""

import numpy as np
from typing import Tuple, List, Optional


class MiddleSquareHybrid:
    """
    混合中平方法-线性同余伪随机数生成器。

    算法:
    1. 线性同余: s_{n+1} = (a * s_n + c) mod m
    2. 中平方扰动: r = (s² // 10^{d}) mod 10^{2d}
    3. 混合输出: u = frac( (s + r) / m )
    """

    def __init__(self, seed: int = 123456789, d: int = 4):
        """
        初始化随机数生成器。

        Parameters
        ----------
        seed : int
            初始种子
        d : int
            中平方位数的一半
        """
        self.d = d
        self.modulus = 10 ** (2 * d)
        self.state = seed % self.modulus
        # LCG参数 (Numerical Recipes推荐)
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32

    def _middle_square_step(self, s: int) -> int:
        """
        单步中平方运算。

        s_{next} = floor( (s² mod 10^{4d}) / 10^{d} )
        """
        sq = s * s
        # 取中间2d位
        sq_mod = sq % (10 ** (4 * self.d))
        mid = sq_mod // (10 ** self.d)
        return mid % self.modulus

    def _lcg_step(self, s: int) -> int:
        """
        线性同余步骤。
        """
        return (self.a * s + self.c) % self.m

    def next_int(self) -> int:
        """
        生成下一个整数随机数。
        """
        self.state = self._lcg_step(self.state)
        ms = self._middle_square_step(self.state % self.modulus)
        mixed = (self.state + ms) % self.m
        return mixed

    def random(self) -> float:
        """
        生成 [0, 1) 均匀分布随机数。
        """
        return self.next_int() / self.m

    def random_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        生成均匀分布随机数组。
        """
        return np.array([self.random() for _ in range(int(np.prod(size)))])

    def cycle_length(self, max_steps: int = 100000) -> Tuple[int, int]:
        """
        分析序列的周期长度。

        基于middle_square_cycle_length思想。

        Parameters
        ----------
        max_steps : int
            最大步数

        Returns
        -------
        cycle_length : int
            周期长度
        steps_to_cycle : int
            进入周期的步数
        """
        seen = {}
        value = self.state
        steps = 0
        while steps < max_steps:
            if value in seen:
                cycle_len = steps - seen[value]
                return cycle_len, seen[value]
            seen[value] = steps
            self.state = value
            value = self.next_int()
            steps += 1
        return -1, max_steps


class QGPEventSampler:
    """
    QGP蒙特卡洛事件采样器。
    """

    def __init__(self, rng: Optional[MiddleSquareHybrid] = None):
        """
        初始化事件采样器。

        Parameters
        ----------
        rng : MiddleSquareHybrid
            随机数生成器实例
        """
        self.rng = rng if rng is not None else MiddleSquareHybrid()

    def sample_impact_parameter(self, b_max: float = 15.0,
                                n_samples: int = 1000) -> np.ndarray:
        """
        按碰撞参数分布采样: dN/db ∝ b。

        Parameters
        ----------
        b_max : float
            最大碰撞参数 [fm]
        n_samples : int
            采样数

        Returns
        -------
        np.ndarray
            碰撞参数样本
        """
        samples = []
        while len(samples) < n_samples:
            b = b_max * np.sqrt(self.rng.random())
            samples.append(b)
        return np.array(samples)

    def sample_thermal_momentum(self, T: float, m: float,
                                n_samples: int = 1000) -> np.ndarray:
        """
        从相对论性Maxwell-Boltzmann分布采样动量大小。

        f(p) ∝ p² exp(-√(p² + m²)/T)

        使用拒绝采样法。

        Parameters
        ----------
        T : float
            温度 [GeV]
        m : float
            粒子质量 [GeV]
        n_samples : int
            采样数

        Returns
        -------
        np.ndarray
            动量样本 [GeV]
        """
        samples = []
        # 建议分布: 指数分布 p ~ exp(-p/T_eff)
        T_eff = max(T, 1e-6)
        p_max = 10.0 * T_eff + 3.0 * m
        while len(samples) < n_samples:
            p = -T_eff * np.log(self.rng.random() + 1e-20)
            if p > p_max:
                continue
            # 目标/建议比例
            E = np.sqrt(p ** 2 + m ** 2)
            f_target = p ** 2 * np.exp(-E / T_eff)
            f_proposal = np.exp(-p / T_eff) / T_eff
            ratio = f_target / (f_proposal + 1e-20)
            if self.rng.random() < ratio / (p_max ** 2 * T_eff + 1e-20):
                samples.append(p)
        return np.array(samples)

    def sample_azimuthal_angle(self, v2: float, n_samples: int = 1000) -> np.ndarray:
        """
        采样包含集体流各向异性的方位角分布。

        dN/dφ ∝ 1 + 2 v₂ cos(2(φ - Ψ₂))

        Parameters
        ----------
        v2 : float
            椭圆流系数
        n_samples : int
            采样数

        Returns
        -------
        np.ndarray
            方位角样本 [rad]
        """
        samples = []
        v2_clip = np.clip(v2, -0.5, 0.5)
        while len(samples) < n_samples:
            phi = 2.0 * np.pi * self.rng.random()
            pdf = 1.0 + 2.0 * v2_clip * np.cos(2.0 * phi)
            if self.rng.random() < pdf / (1.0 + 2.0 * abs(v2_clip)):
                samples.append(phi)
        return np.array(samples)

    def sample_fluctuation(self, mean: float, std: float,
                           n_samples: int = 1000) -> np.ndarray:
        """
        使用Box-Muller变换采样高斯涨落。

        Parameters
        ----------
        mean : float
            均值
        std : float
            标准差
        n_samples : int
            采样数

        Returns
        -------
        np.ndarray
            高斯样本
        """
        samples = []
        while len(samples) < n_samples:
            u1 = self.rng.random()
            u2 = self.rng.random()
            if u1 < 1e-20:
                continue
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            samples.append(mean + std * z0)
        return np.array(samples)
