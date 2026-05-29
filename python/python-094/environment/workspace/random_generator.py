"""
random_generator.py
===================
随机数生成器，用于 Monte Carlo 采样与不确定性量化。

融合种子项目：
  - 1039_rng_cliff : Cliff 随机数生成器

科学应用：
  在非线性声学中，随机采样用于：
  - Monte Carlo 积分计算声能量
  - 介质参数不确定性传播（随机声速、密度）
  - 传感器测量噪声模拟

  Cliff 随机数生成器：
  .. math::
      x_{n+1} = -100 \ln(x_n) \pmod{1}

  该生成器具有混沌特性，适合高维空间填充采样。
"""

import numpy as np


def cliff_next(x):
    """
    Cliff 随机数生成器的下一步。

    原始算法来自 1039_rng_cliff/rng_cliff_next.m。

    Parameters
    ----------
    x : float
        当前值，必须在 (0, 1) 内。

    Returns
    -------
    float
        下一个值。若输入越界返回 NaN。
    """
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return (-100.0 * np.log(x)) % 1.0


class CliffGenerator:
    """
    Cliff 伪随机数生成器封装。
    """

    def __init__(self, seed=0.314159265):
        """
        Parameters
        ----------
        seed : float
            初始种子，必须在 (0, 1) 内。
        """
        if seed <= 0.0 or seed >= 1.0:
            raise ValueError("seed must be in (0, 1).")
        self.state = float(seed)
        self._validate_state()

    def _validate_state(self):
        if not (0.0 < self.state < 1.0) or not np.isfinite(self.state):
            raise RuntimeError("Cliff generator state became invalid.")

    def next(self):
        """
        生成下一个随机数。

        Returns
        -------
        float
            [0, 1) 内的随机数。
        """
        self.state = cliff_next(self.state)
        if np.isnan(self.state):
            # 恢复状态
            self.state = 0.5
        self._validate_state()
        return self.state

    def rand(self, size=None):
        """
        生成指定形状的随机数组。

        Parameters
        ----------
        size : int or tuple or None

        Returns
        -------
        float or np.ndarray
        """
        if size is None:
            return self.next()
        size = tuple(np.atleast_1d(size))
        arr = np.zeros(size, dtype=float)
        for idx in np.ndindex(size):
            arr[idx] = self.next()
        return arr

    def randn(self, size=None):
        """
        使用 Box-Muller 变换生成正态分布随机数。

        .. math::
            Z = \sqrt{-2 \ln U_1} \cos(2 \pi U_2)

        Returns
        -------
        float or np.ndarray
        """
        if size is None:
            u1 = self.next()
            u2 = self.next()
            while u1 <= 1e-10:
                u1 = self.next()
            return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

        size = tuple(np.atleast_1d(size))
        arr = np.zeros(size, dtype=float)
        flat = arr.ravel()
        for i in range(0, len(flat), 2):
            u1 = self.next()
            u2 = self.next()
            while u1 <= 1e-10:
                u1 = self.next()
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
            flat[i] = z0
            if i + 1 < len(flat):
                flat[i + 1] = z1
        return arr


class StratifiedSampler:
    """
    分层采样器，用于高维 Monte Carlo 积分。
    """

    def __init__(self, dim, n_strata):
        """
        Parameters
        ----------
        dim : int
            维度。
        n_strata : int
            每维分层数。
        """
        self.dim = int(dim)
        self.n_strata = int(n_strata)

    def sample(self, a, b, rng=None):
        """
        在 [a, b]^dim 内生成分层采样点。

        Parameters
        ----------
        a, b : float or np.ndarray
            边界。
        rng : np.random.Generator or None
            随机数生成器。

        Returns
        -------
        np.ndarray, shape (n_points, dim)
            采样点。
        """
        if rng is None:
            rng = np.random.default_rng()
        a = np.full(self.dim, a, dtype=float) if np.isscalar(a) else np.asarray(a, dtype=float)
        b = np.full(self.dim, b, dtype=float) if np.isscalar(b) else np.asarray(b, dtype=float)

        n_points = self.n_strata ** self.dim
        samples = np.zeros((n_points, self.dim), dtype=float)

        idx = 0
        # 使用递归或网格生成
        # 简化为均匀网格 + 抖动
        grids = [np.linspace(a[d], b[d], self.n_strata + 1) for d in range(self.dim)]
        # 使用 itertools.product
        from itertools import product
        for cell in product(range(self.n_strata), repeat=self.dim):
            point = np.zeros(self.dim, dtype=float)
            for d in range(self.dim):
                low = grids[d][cell[d]]
                high = grids[d][cell[d] + 1]
                point[d] = low + rng.random() * (high - low)
            samples[idx, :] = point
            idx += 1

        return samples


def latin_hypercube_sampling(n_samples, dim, a=0.0, b=1.0, rng=None):
    """
    Latin Hypercube 采样。

    .. math::
        x_{ij} = \frac{\pi_j(i) + u_{ij}}{n_{samples}} (b_j - a_j) + a_j

    Parameters
    ----------
    n_samples : int
        采样数。
    dim : int
        维度。
    a, b : float
        边界。
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray, shape (n_samples, dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = np.zeros((n_samples, dim), dtype=float)
    for d in range(dim):
        perm = rng.permutation(n_samples)
        u = rng.random(n_samples)
        samples[:, d] = (perm + u) / n_samples
    return a + samples * (b - a)
