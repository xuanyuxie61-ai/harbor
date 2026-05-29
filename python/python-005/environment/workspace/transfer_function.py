# -*- coding: utf-8 -*-
"""
transfer_function.py
CMB 转移函数 T_l(k) 的计算与 Chebyshev 谱插值

核心物理：
    转移函数描述初始曲率扰动 ζ(k) 如何映射到今天的 CMB 温度各向异性：
        Θ_l(k, η_0) = T_l(k) ζ(k)
    其中线-of-sight 积分为
        T_l(k) = ∫_0^{η_0} dη S(k,η) j_l[k(η_0-η)]
    S(k,η) 为源函数，j_l 为球 Bessel 函数。

    本模块采用简化但物理精确的参数化形式：
        T_l(k) ≈ j_l[k(η_0 - η_rec)] · D(k) · A_peak(l)
    其中 D(k) = exp[-(k/k_D)^2] 为 Silk 阻尼因子，
    A_peak(l) 为声学峰调制幅值，
    j_l 为球 Bessel 函数，通过 utils 模块高效计算。

    对每个 l，在 k 区间上建立 Chebyshev 谱插值，
    使得功率谱积分可在 O(1) 内求值。

融合种子项目 159_chebyshev（Chebyshev 插值）与
boltzmann_solver 的源函数输出。
"""

import numpy as np
from typing import Callable
from utils import spherical_bessel_j, clip_to_unit


class ChebyshevInterpolator:
    """
    Chebyshev 谱插值器。
    在区间 [a, b] 上，利用第一类 Chebyshev 节点：
        x_i = 0.5(a+b) + 0.5(b-a) cos[(2i-1)π/(2n)]
    计算离散 Chebyshev 变换系数：
        c_j = (2/n) Σ_{i=1}^n f(x_i) T_{j-1}(\tilde{x}_i)
    其中 \tilde{x} = (2x-a-b)/(b-a) 为归一化坐标。
    使用 Clenshaw 递推进行稳定求值：
        b_{N+2} = b_{N+1} = 0
        b_k = 2\tilde{x} b_{k+1} - b_{k+2} + c_k
        f(x) = 0.5(b_0 - b_2)
    """

    def __init__(self, a: float, b: float, n: int, f: Callable[[float], float]):
        """
        Parameters
        ----------
        a, b : float
            插值区间 [a, b]。
        n : int
            Chebyshev 节点数（≥2）。
        f : callable
            目标函数 f(x)。
        """
        if b <= a:
            raise ValueError("插值区间必须满足 b > a")
        if n < 2:
            raise ValueError("Chebyshev 节点数 n 必须 ≥ 2")
        self.a = a
        self.b = b
        self.n = n
        self.coeffs = self._compute_coefficients(f)

    def _xt(self, x: float) -> float:
        """将 x ∈ [a,b] 映射到 \tilde{x} ∈ [-1,1]。"""
        return (2.0 * x - self.a - self.b) / (self.b - self.a)

    def _compute_coefficients(self, f: Callable[[float], float]) -> np.ndarray:
        """通过第一类 Chebyshev 节点采样并计算离散余弦变换系数。"""
        i = np.arange(1, self.n + 1)
        x_tilde = np.cos((2.0 * i - 1.0) * np.pi / (2.0 * self.n))
        x_nodes = 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * x_tilde
        f_vals = np.array([f(x) for x in x_nodes])
        # 离散 Chebyshev 变换
        c = np.zeros(self.n)
        for j in range(self.n):
            Tj = np.cos(j * np.arccos(clip_to_unit(x_tilde)))
            c[j] = (2.0 / self.n) * np.sum(f_vals * Tj)
        return c

    def evaluate(self, x: float) -> float:
        """Clenshaw 递推求值。"""
        xt = clip_to_unit(self._xt(x))
        # 递推
        b2 = 0.0
        b1 = 0.0
        for j in range(self.n - 1, 0, -1):
            b0 = 2.0 * xt * b1 - b2 + self.coeffs[j]
            b2 = b1
            b1 = b0
        return 0.5 * (self.coeffs[0] + b1 * xt - b2)

    def evaluate_array(self, x_arr: np.ndarray) -> np.ndarray:
        """对数组求值。"""
        return np.array([self.evaluate(x) for x in x_arr])


class TransferFunctionComputer:
    """
    计算 CMB 转移函数 T_l(k) 并在 k-空间上进行 Chebyshev 插值。
    采用物理简化的参数化模型以确保数值稳定性：
        T_l(k) = j_l[k(η_0 - η_rec)] · exp[-(k/k_D)^2] · [1 + 0.3·sin(k/k_p)]
    """

    def __init__(self, lmax: int = 100, k_min: float = 1e-4,
                 k_max: float = 1.0, n_cheb: int = 32):
        """
        Parameters
        ----------
        lmax : int
            最大多极矩 l。
        k_min, k_max : float
            波数范围 [Mpc^{-1}]。
        n_cheb : int
            每个 l 的 Chebyshev 插值节点数。
        """
        self.lmax = lmax
        self.k_min = k_min
        self.k_max = k_max
        self.n_cheb = n_cheb
        # 宇宙学常数
        self.eta0 = 14000.0       # 今天共形时间 [Mpc]
        self.eta_rec = 280.0      # 复合时期共形时间 [Mpc]
        self.k_D = 0.14           # Silk 阻尼尺度 [Mpc^{-1}]
        self.k_p = 0.05           # 峰值调制波数
        # 存储每个 l 的插值器
        self.interpolators = {}

    def _transfer_analytic(self, l: int, k: float) -> float:
        """
        参数化转移函数解析表达式。
        包含 Sachs-Wolfe（球 Bessel 函数）、Silk 阻尼和声学峰调制。
        """
        # TODO: 请补全参数化转移函数的核心公式
        # 提示：需要结合球 Bessel 函数 j_l、Silk 阻尼因子和声学峰调制
        raise NotImplementedError("Hole_2: 请补全 _transfer_analytic 的实现")

    def build_interpolator(self, l: int) -> ChebyshevInterpolator:
        """
        为给定的 l 建立 T_l(k) 的 Chebyshev 插值器。
        """
        def Tl_of_k(k: float) -> float:
            return self._transfer_analytic(l, k)
        return ChebyshevInterpolator(self.k_min, self.k_max, self.n_cheb, Tl_of_k)

    def precompute_all(self):
        """为 l = 2...lmax 预计算所有插值器。"""
        for l in range(2, self.lmax + 1):
            self.interpolators[l] = self.build_interpolator(l)

    def get_transfer(self, l: int, k: float) -> float:
        """通过插值器获取 T_l(k)。"""
        if l < 2:
            return 1.0
        if l not in self.interpolators:
            # 动态构建
            self.interpolators[l] = self.build_interpolator(l)
        k_clipped = max(self.k_min, min(self.k_max, k))
        return self.interpolators[l].evaluate(k_clipped)
