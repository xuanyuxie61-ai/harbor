# -*- coding: utf-8 -*-
"""
stochastic_uq.py
================
复合材料力学性能随机不确定性的谱量化分析模块。

源自种子项目 519_hermite_exactness（Gauss-Hermite 求积精确性测试），
扩展为随机多项式混沌展开（Polynomial Chaos Expansion, PCE）
与不确定性传播计算。

科学背景：
---------
复合材料力学性能具有显著的随机性（纤维体积分数波动、制造缺陷、
孔隙率分布等）。传统蒙特卡洛方法计算成本高，而谱方法（PCE）
通过正交多项式基展开随机过程，实现高效的统计矩计算。

核心公式：
---------
设随机输入 ξ ~ N(0,1)（标准正态），输出量 Y = f(ξ)。
PCE 展开：
  Y(ξ) ≈ Σ_{k=0}^{P} y_k * H_k(ξ)
其中 H_k(ξ) 为概率学家归一化的 Hermite 多项式（ physicist's: H_k^{(p)}(ξ) = (-1)^k e^{ξ²/2} d^k/dξ^k e^{-ξ²/2} ）。

正交性：
  ∫_{-∞}^{∞} H_m(ξ) H_n(ξ) w(ξ) dξ = δ_{mn} * n!
  其中 w(ξ) = (1/√(2π)) exp(-ξ²/2) 为标准正态 PDF。

展开系数（投影公式）：
  y_k = (1/(k!)) * E[Y(ξ) * H_k(ξ)]
      ≈ (1/(k!)) * Σ_{j=1}^{Q} w_j * f(ξ_j) * H_k(ξ_j)
  其中 {ξ_j, w_j} 为 Gauss-Hermite 求积节点和权重。

统计矩：
  均值：μ_Y = y_0
  方差：σ²_Y = Σ_{k=1}^{P} (y_k)² * k!
  三阶中心矩（偏度）：μ_3 = Σ_{k=1}^{P} (y_k)³ * E[H_k³]
  （对高斯输入，三阶矩主要由 k=1,3 贡献）

可靠性分析：
  失效概率 P_f = P(Y > Y_limit) ≈ PCE 的尾部积分。

复合材料随机参数典型分布：
  - 纤维体积分数 V_f ~ N(μ_Vf, σ_Vf)
  - 纵向模量 E1 ~ Lognormal(μ_E1, σ_E1)
  - 界面强度 τ_int ~ Weibull(λ, k)

本模块实现：
  1. Gauss-Hermite 求积节点与权重生成；
  2. 概率学/物理学 Hermite 多项式；
  3. PCE 系数计算与统计矩提取；
  4. 随机输入下复合材料弹性模量的不确定性传播。
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.hermite import hermgauss
from typing import Callable, Tuple, Optional


class HermiteQuadrature:
    """
    Gauss-Hermite 求积规则。
    支持概率学 Hermite (He) 和物理学 Hermite (H) 两种归一化。
    """

    @staticmethod
    def probabilist_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        概率学归一化 Hermite 多项式 He_n(x) 的 Gauss 求积节点和权重。
        权重函数：w(x) = exp(-x²/2) / √(2π)。
        精确性：对 2n-1 次多项式精确。

        使用 numpy.polynomial.hermite_e 的 hermegauss。
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        nodes, weights = hermegauss(n)
        # hermegauss 返回的权重对 w(x)=exp(-x²/2) 精确，需归一化到概率密度
        weights = weights / np.sqrt(2.0 * np.pi)
        return nodes, weights

    @staticmethod
    def physicist_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        物理学归一化 Hermite 多项式 H_n(x) 的 Gauss 求积节点和权重。
        权重函数：w(x) = exp(-x²)。
        """
        if n < 1:
            raise ValueError("n must be >= 1.")
        nodes, weights = hermgauss(n)
        return nodes, weights

    @staticmethod
    def exactness_test(n: int, max_degree: Optional[int] = None) -> bool:
        """
        验证 Gauss-Hermite (概率学) 对奇数次和偶数次单项式的精确性。
        对标准正态分布，矩：
          E[x^k] = 0           (k 为奇数)
          E[x^k] = (k-1)!!     (k 为偶数)
        """
        if max_degree is None:
            max_degree = 2 * n - 1
        nodes, weights = HermiteQuadrature.probabilist_nodes_weights(n)
        for k in range(max_degree + 1):
            exact_moment = 0.0 if k % 2 == 1 else np.prod(np.arange(1, k, 2), dtype=float)
            approx_moment = np.sum(weights * (nodes ** k))
            if not np.isclose(approx_moment, exact_moment, atol=1e-12):
                print(f"Exactness failed for degree {k}: exact={exact_moment}, approx={approx_moment}")
                return False
        return True


class HermitePolynomials:
    """
    Hermite 多项式计算。
    """

    @staticmethod
    def probabilist_hermite(n: int, x: np.ndarray) -> np.ndarray:
        """
        概率学 Hermite 多项式 He_n(x)。
        递推：
          He_0(x) = 1
          He_1(x) = x
          He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
        """
        x = np.asarray(x)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return x.copy()
        H_prev2 = np.ones_like(x)
        H_prev1 = x.copy()
        H_curr = np.zeros_like(x)
        for k in range(1, n):
            H_curr = x * H_prev1 - k * H_prev2
            H_prev2, H_prev1 = H_prev1, H_curr
        return H_curr

    @staticmethod
    def physicist_hermite(n: int, x: np.ndarray) -> np.ndarray:
        """
        物理学 Hermite 多项式 H_n(x)。
        递推：
          H_0(x) = 1
          H_1(x) = 2x
          H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
        """
        x = np.asarray(x)
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return 2.0 * x
        H_prev2 = np.ones_like(x)
        H_prev1 = 2.0 * x
        H_curr = np.zeros_like(x)
        for k in range(1, n):
            H_curr = 2.0 * x * H_prev1 - 2.0 * k * H_prev2
            H_prev2, H_prev1 = H_prev1, H_curr
        return H_curr


class PolynomialChaosExpansion:
    """
    一维多项式混沌展开（PCE）用于不确定性传播。
    """

    def __init__(self, max_order: int, num_quad_points: Optional[int] = None):
        """
        Parameters
        ----------
        max_order : int
            PCE 截断阶数 P。
        num_quad_points : int or None
            Gauss-Hermite 求积点数 Q；None 时取 P+2。
        """
        self.P = max_order
        self.Q = num_quad_points if num_quad_points is not None else max_order + 2
        if self.Q < (self.P + 1):
            self.Q = self.P + 2
        self.nodes, self.weights = HermiteQuadrature.probabilist_nodes_weights(self.Q)

    def compute_coefficients(self, f: Callable) -> np.ndarray:
        """
        计算 PCE 系数 y_k = E[f(ξ) * He_k(ξ)] / k! 。

        Parameters
        ----------
        f : callable
            输入标量 ξ，返回标量 f(ξ)。

        Returns
        -------
        coeffs : np.ndarray, shape (P+1,)
        """
        coeffs = np.zeros(self.P + 1)
        f_vals = np.array([f(xi) for xi in self.nodes])
        for k in range(self.P + 1):
            He_k = HermitePolynomials.probabilist_hermite(k, self.nodes)
            factorial_k = np.math.factorial(k)
            coeffs[k] = np.sum(self.weights * f_vals * He_k) / factorial_k
        return coeffs

    def evaluate(self, coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        在给定点 ξ 处求值 PCE 近似。
        """
        xi = np.asarray(xi)
        result = np.zeros_like(xi)
        for k, c in enumerate(coeffs):
            result += c * HermitePolynomials.probabilist_hermite(k, xi)
        return result

    def mean(self, coeffs: np.ndarray) -> float:
        """均值 μ = y_0。"""
        return coeffs[0]

    def variance(self, coeffs: np.ndarray) -> float:
        """
        方差 σ² = Σ_{k=1}^{P} y_k² * k!。
        """
        var = 0.0
        for k in range(1, len(coeffs)):
            var += coeffs[k] ** 2 * np.math.factorial(k)
        return var

    def standard_deviation(self, coeffs: np.ndarray) -> float:
        return np.sqrt(self.variance(coeffs))

    def skewness(self, coeffs: np.ndarray) -> float:
        """
        偏度 γ_1 = μ_3 / σ³。
        对高斯输入，三阶矩需利用 Hermite 多项式的乘积公式计算。
        简化：仅考虑低阶贡献。
        """
        sigma = self.standard_deviation(coeffs)
        if sigma < 1e-30:
            return 0.0
        mu3 = 0.0
        # He_k * He_m * He_n 的期望非零仅当 k+m+n 为偶数且满足三角不等式
        for k in range(len(coeffs)):
            for m in range(len(coeffs)):
                for n in range(len(coeffs)):
                    if (k + m + n) % 2 == 0 and self._triangle_inequality(k, m, n):
                        E_triple = self._expectation_triple_product(k, m, n)
                        mu3 += coeffs[k] * coeffs[m] * coeffs[n] * E_triple
        return mu3 / (sigma ** 3 + 1e-30)

    @staticmethod
    def _triangle_inequality(a: int, b: int, c: int) -> bool:
        """三角不等式：|a-b| ≤ c ≤ a+b。"""
        return abs(a - b) <= c <= a + b

    @staticmethod
    def _expectation_triple_product(k: int, m: int, n: int) -> float:
        """
        E[He_k * He_m * He_n] 对标准正态分布。
        仅当 k+m+n 为偶数且满足三角不等式时非零。
        公式（线性化系数）：
          E[He_k He_m He_n] = (k! m! n!) / (s! (s-k)! (s-m)! (s-n)!)
        其中 s = (k+m+n)/2，且要求 s 为整数，s ≥ max(k,m,n)。
        """
        total = k + m + n
        if total % 2 != 0:
            return 0.0
        s = total // 2
        if s < max(k, m, n):
            return 0.0
        # 使用对数避免溢出
        log_num = (np.math.lgamma(k + 1) + np.math.lgamma(m + 1) + np.math.lgamma(n + 1))
        log_den = (np.math.lgamma(s + 1) + np.math.lgamma(s - k + 1)
                   + np.math.lgamma(s - m + 1) + np.math.lgamma(s - n + 1))
        return np.exp(log_num - log_den)


class CompositeRandomProperties:
    """
    复合材料随机力学性能的不确定性传播。
    """

    @staticmethod
    def fiber_volume_fraction_to_modulus(Vf: float,
                                          Ef: float = 230e9,
                                          Em: float = 3.5e9) -> float:
        """
        混合律（Rule of Mixtures）：
          E1 = Vf * Ef + (1 - Vf) * Em
        """
        Vf_clip = np.clip(Vf, 0.0, 1.0)
        return Vf_clip * Ef + (1.0 - Vf_clip) * Em

    @staticmethod
    def random_Vf_pce_analysis(mu_Vf: float = 0.60, sigma_Vf: float = 0.03,
                                max_order: int = 4) -> dict:
        """
        分析纤维体积分数随机波动对纵向模量 E1 的影响。

        假设 Vf ~ N(μ_Vf, σ_Vf²)，则标准化变量 ξ = (Vf - μ_Vf) / σ_Vf ~ N(0,1)。
        E1(ξ) = (μ_Vf + σ_Vf * ξ) * Ef + (1 - μ_Vf - σ_Vf * ξ) * Em
              = (μ_Vf * Ef + (1-μ_Vf)*Em) + σ_Vf * (Ef - Em) * ξ
        这是 ξ 的线性函数，PCE 精确到一阶。
        """
        Ef = 230e9
        Em = 3.5e9
        pce = PolynomialChaosExpansion(max_order=max_order)

        def E1_func(xi):
            Vf = mu_Vf + sigma_Vf * xi
            return CompositeRandomProperties.fiber_volume_fraction_to_modulus(Vf, Ef, Em)

        coeffs = pce.compute_coefficients(E1_func)
        mean_E1 = pce.mean(coeffs)
        std_E1 = pce.standard_deviation(coeffs)
        skew_E1 = pce.skewness(coeffs)

        return {
            "E1_mean": mean_E1,
            "E1_std": std_E1,
            "E1_cov": std_E1 / (mean_E1 + 1e-30),
            "E1_skewness": skew_E1,
            "pce_coefficients": coeffs,
        }

    @staticmethod
    def lognormal_property_pce_analysis(mu_ln: float, sigma_ln: float,
                                        max_order: int = 5) -> dict:
        """
        对数正态分布材料属性的 PCE 分析。
        若 X ~ LogNormal(μ_ln, σ_ln²)，则 ln X = μ_ln + σ_ln * ξ，ξ~N(0,1)。
        X(ξ) = exp(μ_ln + σ_ln * ξ)。
        """
        pce = PolynomialChaosExpansion(max_order=max_order)

        def X_func(xi):
            return np.exp(mu_ln + sigma_ln * xi)

        coeffs = pce.compute_coefficients(X_func)
        mean_X = pce.mean(coeffs)
        std_X = pce.standard_deviation(coeffs)

        # 解析均值和方差对比
        exact_mean = np.exp(mu_ln + 0.5 * sigma_ln ** 2)
        exact_var = (np.exp(sigma_ln ** 2) - 1.0) * np.exp(2 * mu_ln + sigma_ln ** 2)

        return {
            "mean_pce": mean_X,
            "mean_exact": exact_mean,
            "std_pce": std_X,
            "std_exact": np.sqrt(exact_var),
            "relative_error_mean": abs(mean_X - exact_mean) / (exact_mean + 1e-30),
        }


if __name__ == "__main__":
    # 自测试 1：Hermite 精确性
    assert HermiteQuadrature.exactness_test(n=5, max_degree=9)
    print("Hermite exactness test PASSED.")

    # 自测试 2：PCE 对线性函数应精确
    pce = PolynomialChaosExpansion(max_order=3)
    coeffs = pce.compute_coefficients(lambda xi: 2.0 + 3.0 * xi)
    print("PCE coeffs for linear func:", coeffs)
    assert np.isclose(coeffs[0], 2.0, atol=1e-12)
    assert np.isclose(coeffs[1], 3.0, atol=1e-12)
    assert np.isclose(pce.variance(coeffs), 9.0, atol=1e-12)

    # 自测试 3：复合材料随机属性
    result = CompositeRandomProperties.random_Vf_pce_analysis()
    print("Composite UQ result:", result)
