# -*- coding: utf-8 -*-
"""
random_generator.py
裂隙介质渗流与示踪试验伪随机数生成模块

基于种子项目 763_middle_square 的中平方伪随机数生成算法，
用于裂隙网络参数随机化和蒙特卡洛不确定性采样。

核心算法：
    中平方法 (Middle-Square Method):
        r_{k+1} = floor( r_k^2 / 10^d ) mod 10^{2d}

物理应用：
    - 裂隙开度的随机扰动
    - 渗透率场的蒙特卡洛采样
    - 示踪剂注入位置的随机化
"""

import numpy as np


class MiddleSquareGenerator:
    """
    中平方伪随机数生成器

    基于 Von Neumann 中平方法，用于可复现的科学计算随机序列。
    在裂隙介质研究中，该方法确保不同研究团队使用相同种子时
    获得完全一致的裂隙几何实现。
    """

    def __init__(self, seed: int = 1234, d: int = 4):
        """
        初始化中平方随机数生成器

        Parameters
        ----------
        seed : int
            初始种子，不超过 2*d 位十进制数字
        d : int
            半位数，典型值 2, 3, 4
        """
        if d < 1 or d > 9:
            raise ValueError("d 必须在 [1, 9] 范围内")
        max_seed = 10 ** (2 * d)
        if not (0 <= seed < max_seed):
            raise ValueError(f"seed 必须在 [0, {max_seed}) 范围内")
        self._s = int(seed)
        self._d = int(d)
        self._mod = 10 ** (2 * d)
        self._div = 10 ** d

    def _next(self) -> int:
        """计算下一个中平方值"""
        r = self._s * self._s
        r = r // self._div
        r = r % self._mod
        self._s = r
        return r

    def cycle_length(self, max_iter: int = 100000) -> int:
        """
        计算当前种子下的循环长度

        Parameters
        ----------
        max_iter : int
            最大迭代次数

        Returns
        -------
        int
            循环长度，若未找到循环返回 -1
        """
        seen = {}
        s0 = self._s
        for i in range(max_iter):
            if self._s in seen:
                length = i - seen[self._s]
                self._s = s0
                return length
            seen[self._s] = i
            self._next()
        self._s = s0
        return -1

    def random(self) -> float:
        """生成 [0, 1) 区间均匀分布随机数"""
        return self._next() / self._mod

    def randn(self) -> float:
        """
        使用 Box-Muller 变换生成标准正态分布随机数

        变换公式：
            Z_1 = sqrt(-2 ln U_1) * cos(2π U_2)
            Z_2 = sqrt(-2 ln U_2) * sin(2π U_1)
        """
        u1 = max(self.random(), 1e-10)
        u2 = self.random()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """生成 [low, high) 区间均匀分布随机数"""
        if low >= high:
            raise ValueError("low 必须小于 high")
        return low + self.random() * (high - low)

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        生成对数正态分布随机数

        对数正态分布 PDF:
            f(x) = 1/(x σ sqrt(2π)) * exp(-(ln x - μ)^2 / (2σ^2))

        在裂隙水文学中，裂隙开度常服从对数正态分布。
        """
        if sigma <= 0:
            raise ValueError("sigma 必须为正")
        z = self.randn()
        return np.exp(mu + sigma * z)

    def exponential(self, scale: float = 1.0) -> float:
        """
        生成指数分布随机数

        逆变换采样：X = -λ^{-1} ln(1-U)
        """
        if scale <= 0:
            raise ValueError("scale 必须为正")
        u = max(self.random(), 1e-10)
        return -scale * np.log(1.0 - u)

    def generate_array(self, n: int, dist: str = "uniform", **kwargs) -> np.ndarray:
        """生成随机数组"""
        if n < 0:
            raise ValueError("n 必须为非负整数")
        arr = np.zeros(n)
        for i in range(n):
            if dist == "uniform":
                arr[i] = self.uniform(kwargs.get("low", 0.0), kwargs.get("high", 1.0))
            elif dist == "normal":
                arr[i] = kwargs.get("mu", 0.0) + kwargs.get("sigma", 1.0) * self.randn()
            elif dist == "lognormal":
                arr[i] = self.lognormal(kwargs.get("mu", 0.0), kwargs.get("sigma", 1.0))
            elif dist == "exponential":
                arr[i] = self.exponential(kwargs.get("scale", 1.0))
            else:
                raise ValueError(f"不支持的分布类型: {dist}")
        return arr
