# -*- coding: utf-8 -*-
"""
eigenvalue_solver.py
本征值问题求解与谐振模式分析

核心公式与物理背景
------------------
1. 谐振腔特征方程
   对于微环谐振腔，谐振条件为：
       2π·R_eff·n_eff = m·λ_m
   其中 R_eff 为有效半径，n_eff 为有效折射率，m 为方位角模式数。
   对应的谐振频率：
       ω_m = m·c / (R_eff·n_eff)

2. 色散关系与本征值问题
   Helmholtz 方程的离散形式可写为广义本征值问题：
       L · E = -k² · D(n) · E
   其中 L 为离散拉普拉斯矩阵，D(n) = diag(n²)。
   本征值 λ = -k² 对应谐振波数。

3. 二分法求根
   对非线性特征方程 f(ω) = 0，在区间 [a,b] 上若 f(a)·f(b) < 0，
   则反复对半缩小区间，直到 |b-a| < tol。
   收敛速度：线性，每步误差减半。

4. Chebyshev 加速（多项式迭代）
   对迭代格式 x_{k+1} = G·x_k + c，引入 Chebyshev 多项式 T_m(z) 加速：
       x^{(m)} = T_m(G) · x^{(0)} + (I - T_m(G)) · x*
   这里用 Collatz-like 多项式映射作为预处理算子，提升幂迭代的收敛速度。

融合来源
--------
- 094_bisection          : 二分法求根（求谐振频率）
- 198_collatz_polynomial : GF(2) 上多项式迭代，用作预处理算子设计思想
"""

import numpy as np
from typing import Callable, Tuple, Optional


class BisectionSolver:
    """
    二分法求解单变量方程 f(x)=0 的稳健根查找器。
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-12):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, f: Callable[[float], float], a: float, b: float) -> Tuple[float, int]:
        """
        在 [a,b] 上求解 f(x)=0。

        前置条件
        --------
        f(a)·f(b) < 0（符号相反）。

        返回
        ----
        root : float
            根的近似值
        iters : int
            实际迭代次数
        """
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError(f"区间 [{a}, {b}] 两端同号，无法保证根存在")
        if np.isnan(fa) or np.isnan(fb):
            raise ValueError("边界函数值为 NaN")

        for it in range(1, self.max_iter + 1):
            c = 0.5 * (a + b)
            fc = f(c)
            if abs(fc) < 1e-30 or abs(b - a) < self.tol:
                return c, it
            if fa * fc <= 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        c = 0.5 * (a + b)
        return c, self.max_iter


class CollatzPolynomial:
    """
    GF(2) 上的 Collatz-like 多项式动力系统。
    用于构造离散预处理算子的代数原型：
        若 P(x) 的常数项为 0，则 P' = P / x
        否则 P' = P·(x+1) + 1   (mod 2)
    本类同时提供在实数域上的平滑类比，用于特征值迭代分析。
    """

    def __init__(self, coeffs: np.ndarray):
        """
        coeffs : np.ndarray of 0/1
            多项式系数，从低次到高次。
        """
        self.coeffs = np.array(coeffs, dtype=int) % 2
        self.coeffs = self._trim(self.coeffs)

    @staticmethod
    def _trim(c: np.ndarray) -> np.ndarray:
        """去掉高次零系数"""
        if len(c) == 0:
            return np.array([0], dtype=int)
        idx = len(c) - 1
        while idx > 0 and c[idx] == 0:
            idx -= 1
        return c[:idx + 1]

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def next_poly(self) -> "CollatzPolynomial":
        """
        执行一次 Collatz-like 映射：
            若常数项为 0：P → P / x
            否则：P → P·(x+1) + 1  (mod 2)
        """
        if self.coeffs[0] == 0:
            # 除以 x：右移
            new_coeffs = self.coeffs[1:] if len(self.coeffs) > 1 else np.array([0], dtype=int)
        else:
            # 乘以 (x+1)：卷积 + 移位
            conv = np.convolve(self.coeffs, [1, 1]) % 2
            new_coeffs = (conv + np.array([1] + [0] * (len(conv) - 1))) % 2
        return CollatzPolynomial(new_coeffs)

    def sequence(self, max_steps: int = 100) -> list:
        """生成序列直到达到常数或步数上限"""
        seq = [self.coeffs.copy()]
        current = self
        for _ in range(max_steps):
            if current.degree() == 0:
                break
            current = current.next_poly()
            seq.append(current.coeffs.copy())
        return seq

    @staticmethod
    def smooth_analog(x: float, max_iter: int = 50, threshold: float = 1e-12) -> float:
        """
        实数域上的平滑类比映射，用于分析不动点与收敛域：
            若 |x| < 1：x → x / 2
            否则：x → (x² + x + 1) / 3
        返回不动点近似值。
        """
        for _ in range(max_iter):
            if abs(x) < threshold:
                return x
            if abs(x) < 1.0:
                x = x / 2.0
            else:
                x = (x * x + x + 1.0) / 3.0
        return x


class ResonanceEigenSolver:
    """
    微腔谐振模式本征值分析器。
    结合二分法与幂迭代，求解有效折射率与谐振波长。
    """

    def __init__(self, R_major: float, n_nominal: float):
        self.R_major = R_major
        self.n_nominal = n_nominal
        self.c = 2.99792458e8  # 光速 [m/s]

    def resonance_condition(self, m: int, lambda_nm: float, n_eff: float) -> float:
        """
        谐振条件残差：
            f(λ) = 2π·R·n_eff(λ) - m·λ
        返回残差值（单位：m）。
        """
        lambda_m = lambda_nm * 1e-9
        return 2.0 * np.pi * self.R_major * n_eff - m * lambda_m

    def find_resonance_wavelength(self, m: int,
                                   n_eff_func: Callable[[float], float],
                                   lambda_min_nm: float = 1500.0,
                                   lambda_max_nm: float = 1600.0) -> Tuple[float, int]:
        """
        用二分法寻找满足谐振条件的波长 λ [nm]。

        参数
        ----
        m : int
            方位角模式数
        n_eff_func : Callable
            输入波长 [nm]，返回有效折射率
        lambda_min_nm, lambda_max_nm : float
            搜索区间

        返回
        ----
        lambda_res_nm : float
            谐振波长 [nm]
        iters : int
            二分迭代次数
        """
        def f(lam):
            return self.resonance_condition(m, lam, n_eff_func(lam))

        bisect = BisectionSolver(max_iter=300, tol=1e-6)
        root, iters = bisect.solve(f, lambda_min_nm, lambda_max_nm)
        return root, iters

    def compute_mode_spacing(self, m: int, lambda_nm: float, n_eff: float,
                             ng: float) -> float:
        """
        计算相邻纵模的自由光谱范围（FSR）：
            FSR = λ² / (2π·R·n_g)
        其中 n_g = n_eff - λ·dn_eff/dλ 为群折射率。
        """
        lambda_m = lambda_nm * 1e-9
        fsr = lambda_m ** 2 / (2.0 * np.pi * self.R_major * ng)
        return fsr

    def power_iteration_eigenvalue(self, A: np.ndarray, max_iter: int = 500,
                                    tol: float = 1e-10) -> Tuple[float, np.ndarray, int]:
        """
        幂迭代法求矩阵 A 的绝对值最大本征值及其本征向量。

        算法
        ----
        1. 随机初始化 x₀
        2. y = A·x_k
        3. λ_k = y^H · x_k / (x_k^H · x_k)   (Rayleigh 商)
        4. x_{k+1} = y / ‖y‖
        5. 当 |λ_k - λ_{k-1}| < tol 时停止
        """
        n = A.shape[0]
        x = np.random.default_rng(42).random(n)
        x = x / np.linalg.norm(x)
        lam = 0.0
        for it in range(1, max_iter + 1):
            y = A @ x
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-30:
                raise ValueError("迭代向量收敛到零")
            y = y / norm_y
            lam_new = float(np.dot(y.conj(), A @ y))
            if abs(lam_new - lam) < tol:
                return lam_new, y, it
            lam = lam_new
            x = y
        return lam, x, max_iter

    def sensitivity_dlambda_dn(self, m: int, lambda_nm: float, n_eff: float) -> float:
        """
        灵敏度分析：dλ/dn_eff。
        由谐振条件 2πRn_eff = mλ 微分得：
            dλ/dn = 2πR / m = λ / n_eff
        """
        return lambda_nm / n_eff
