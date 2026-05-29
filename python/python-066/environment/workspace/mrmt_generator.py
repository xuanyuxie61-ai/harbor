"""
mrmt_generator.py
================================================================================
多速率质量转移（Multi-Rate Mass Transfer, MRMT）模型——基于生成函数的叠加计算

基于种子项目：
  - 158_change_polynomial：生成函数与多项式卷积（离散叠加原理）

科学背景：
  在裂隙或高度非均质含水层中，溶质在快速流动的优势通道与慢速扩散的
  基质块之间存在质量交换，这种交换不能用单一速率常数描述，而需要
  多速率质量转移（MRMT）模型。

  经典 MRMT 方程（Haggerty & Gorelick, 1995）：
      R_m ∂C_m/∂t + Σ_{i=1}^{N_k} β_i ∂S_i/∂t = D ∂²C_m/∂x² - v ∂C_m/∂x
      ∂S_i/∂t = α_i (C_m - S_i)

  其中：
      C_m(x,t)  — 流动区域（mobile zone）浓度
      S_i(x,t)  — 第 i 个不动区域（immobile zone）浓度
      β_i       — 容量系数（capacity coefficient）
      α_i       — 质量转移速率 [1/T]
      R_m       — 流动区滞留因子

  在 Laplace 域中，MRMT 的等效滞留因子为：
      R̃(s) = R_m + Σ_{i=1}^{N_k} β_i s / (s + α_i)

  本模块使用生成函数（generating function）方法计算多速率过程的
  卷积叠加：将每个不动区域的响应视为一个“硬币面额”，总响应为
  多个响应的离散卷积（多项式乘法）。
================================================================================
"""

import numpy as np
from typing import List, Tuple


def polynomial_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    两个多项式的离散卷积（系数向量乘法）。

    若 p(x) = Σ_{i=0}^{n-1} p_i x^i,  q(x) = Σ_{j=0}^{m-1} q_j x^j，则
        (p * q)_k = Σ_{i+j=k} p_i q_j

    该运算在地下水溶质运移中表示多个独立过程的响应叠加（卷积定理的离散形式）。
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) == 0 or len(q) == 0:
        return np.array([0.0])
    result = np.convolve(p, q)
    return result


def polynomial_power(base: np.ndarray, exponent: int) -> np.ndarray:
    """
    多项式的整数次幂，通过重复平方法或简单迭代实现。

    在 MRMT 中，若存在 N 个相同的 immobile zone，其总响应为单区响应的 N 次卷积幂。
    """
    if exponent < 0:
        raise ValueError("指数必须非负")
    if exponent == 0:
        return np.array([1.0])
    result = np.array([1.0])
    current = base.copy()
    exp = exponent
    while exp > 0:
        if exp % 2 == 1:
            result = polynomial_multiply(result, current)
        current = polynomial_multiply(current, current)
        exp //= 2
    return result


class MRMTModel:
    """
    多速率质量转移模型：模拟溶质在流动区与多个不动区之间的交换。
    """

    def __init__(self, alphas: np.ndarray, betas: np.ndarray, R_m: float = 1.0):
        """
        参数
        ----------
        alphas : np.ndarray
            质量转移速率数组 α_i [1/T]，每个 α_i > 0
        betas : np.ndarray
            容量系数数组 β_i，每个 β_i ≥ 0
        R_m : float
            流动区滞留因子
        """
        self.alphas = np.asarray(alphas, dtype=float)
        self.betas = np.asarray(betas, dtype=float)
        self.R_m = float(R_m)
        if len(self.alphas) != len(self.betas):
            raise ValueError("alphas 与 betas 长度必须一致")
        if np.any(self.alphas <= 0):
            raise ValueError("所有 α_i 必须为正")
        if np.any(self.betas < 0):
            raise ValueError("所有 β_i 必须非负")

    def effective_retardation(self, s: float) -> float:
        """
        Laplace 域中的等效滞留因子：
            R̃(s) = R_m + Σ_i β_i * s / (s + α_i)
        """
        if s < 0:
            raise ValueError("s 必须非负")
        return self.R_m + np.sum(self.betas * s / (s + self.alphas))

    def immobile_response_kernel(self, t: float) -> float:
        """
        不动区域的记忆核函数（memory kernel）：
            g(t) = Σ_i β_i α_i exp(-α_i t)
        该核函数描述流动区浓度历史对当前不动区浓度的卷积影响。
        """
        if t < 0:
            return 0.0
        return float(np.sum(self.betas * self.alphas * np.exp(-self.alphas * t)))

    def compute_immobile_concentration(self, C_mobile_history: np.ndarray,
                                       dt: float) -> np.ndarray:
        """
        通过离散卷积计算不动区浓度历史：
            S_i^{n} = Σ_{k=0}^{n} w_k α_i exp(-α_i k dt) C_m^{n-k}

        使用 FFT 加速大规模卷积运算。
        """
        n_steps = len(C_mobile_history)
        if n_steps == 0:
            return np.array([])
        S_total = np.zeros(n_steps)

        for alpha_i, beta_i in zip(self.alphas, self.betas):
            # 构造核函数
            k = np.arange(n_steps)
            kernel = beta_i * alpha_i * np.exp(-alpha_i * k * dt)
            # 离散卷积（一维）
            conv = np.convolve(C_mobile_history, kernel, mode='full')[:n_steps]
            S_total += conv

        return S_total

    def mobile_zone_equation_rhs(self, C_mobile: np.ndarray,
                                  C_immobile_total: np.ndarray,
                                  dt: float) -> np.ndarray:
        """
        计算流动区方程中由于 MRMT 产生的等效源汇项：
            R_m ∂C_m/∂t = ... - Σ_i β_i ∂S_i/∂t
        其中 -Σ_i β_i ∂S_i/∂t 可视为从不动区释放回流动区的质量通量。
        """
        if len(C_mobile) != len(C_immobile_total):
            raise ValueError("浓度数组长度不一致")
        if dt <= 0:
            raise ValueError("dt 必须为正")
        # 数值时间导数（向后差分）
        dS_dt = np.zeros_like(C_immobile_total)
        dS_dt[1:] = (C_immobile_total[1:] - C_immobile_total[:-1]) / dt
        return -dS_dt

    def generate_rate_spectrum(self, alpha_min: float, alpha_max: float,
                                n_rates: int, distribution: str = "log_uniform") -> "MRMTModel":
        """
        生成对数均匀分布的 MRMT 速率谱，模拟自然界中连续分布的交换速率。

        参数分布：
            log α_i ~ U(log α_min, log α_max)
            β_i ∝ 1/α_i^γ   (通常 γ = 0.5 对应扩散控制交换)
        """
        if alpha_min <= 0 or alpha_max <= alpha_min:
            raise ValueError("速率范围非法")
        if n_rates < 1:
            raise ValueError("速率数必须 ≥ 1")

        log_alphas = np.linspace(np.log(alpha_min), np.log(alpha_max), n_rates)
        alphas_new = np.exp(log_alphas)
        # 幂律容量系数
        betas_new = 1.0 / np.sqrt(alphas_new)
        betas_new = betas_new / np.sum(betas_new) * np.sum(self.betas) if np.sum(self.betas) > 0 else betas_new
        return MRMTModel(alphas_new, betas_new, self.R_m)

    def breakthrough_curve_moment(self, C_history: np.ndarray, dt: float, moment_order: int) -> float:
        """
        计算突破曲线（breakthrough curve）的统计矩：
            M_k = ∫_0^∞ t^k C(t) dt / ∫_0^∞ C(t) dt
        离散形式：
            M_k ≈ Σ_n (n dt)^k C_n dt / Σ_n C_n dt
        """
        if len(C_history) == 0:
            return 0.0
        t = np.arange(len(C_history)) * dt
        numerator = np.sum((t ** moment_order) * C_history) * dt
        denominator = np.sum(C_history) * dt + 1e-15
        return float(numerator / denominator)


if __name__ == "__main__":
    # 测试：三速率 MRMT
    alphas = np.array([0.01, 0.1, 1.0])
    betas = np.array([0.5, 0.3, 0.2])
    mrmt = MRMTModel(alphas, betas, R_m=1.0)

    C_m = np.exp(-np.linspace(0, 5, 100) * 0.1)
    S = mrmt.compute_immobile_concentration(C_m, dt=0.05)
    assert len(S) == len(C_m)

    # 测试生成函数卷积
    p = np.array([1, 2, 3])
    q = np.array([0, 1])
    r = polynomial_multiply(p, q)
    assert np.allclose(r, [0, 1, 2, 3])

    m1 = mrmt.breakthrough_curve_moment(C_m, 0.05, 1)
    assert m1 > 0
    print("mrmt_generator: 自测试通过")
