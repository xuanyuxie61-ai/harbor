"""
随机微物理参数化模块 (Stochastic Microphysics Parameterization)

集成种子项目:
- 642_laguerre_product: Laguerre 多项式内积与广义 Polynomial Chaos
- 1269_toms291: log-Gamma 函数, 用于 Gamma 分布雨滴谱

科学背景:
  中尺度对流系统中的微物理过程具有高度不确定性.
  使用广义 Polynomial Chaos (gPC) 展开来量化不确定性:
    q(ξ) = Σ_{i=0}^{P} q_i * L_i(ξ)
  其中 ξ 为服从指数/Gamma 分布的随机变量, L_i 为 Laguerre 多项式.

核心公式:
  Khrgian-Mazin 雨滴谱 (Gamma 分布):
    N(D) = N0 * D^μ * exp(-ΛD)
    log N0 ~ Γ(k, θ)  或经变换后使用 Laguerre 展开

  凝结率:
    C = (qv - qvs) / τ_cond   (若 qv > qvs)
  蒸发率:
    E = (qvs - qv) / τ_evap * f(a_L)   (若 qv < qvs)

  这里使用 Laguerre 混沌将微物理参数 (如 N0, τ_cond) 视为随机变量,
  通过 Galerkin 投影得到各阶混沌系数.
"""

import numpy as np
from typing import Tuple, List


def laguerre_polynomials(order: int, x: np.ndarray) -> np.ndarray:
    """
    计算 Laguerre 多项式 L_0(x) ... L_order(x) (基于 642_laguerre_product).

    三项递推关系:
      L_0(x) = 1
      L_1(x) = 1 - x
      (n+1) L_{n+1}(x) = (2n+1 - x) L_n(x) - n L_{n-1}(x)

    权函数: w(x) = exp(-x) on [0, ∞).
    正交性: ∫_0^∞ L_i(x) L_j(x) e^{-x} dx = δ_{ij}
    """
    x = np.asarray(x)
    shape = x.shape
    x = x.flatten()
    npts = len(x)
    L = np.zeros((order + 1, npts))
    L[0, :] = 1.0
    if order >= 1:
        L[1, :] = 1.0 - x
    for n in range(1, order):
        L[n+1, :] = ((2.0 * n + 1.0 - x) * L[n, :] - n * L[n-1, :]) / (n + 1.0)
    return L.reshape((order + 1,) + shape)


def gauss_laguerre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 n 点 Gauss-Laguerre 求积节点与权重.
    节点为 L_n(x) 的根, 通过 Newton 迭代精化.
    """
    if n < 1:
        raise ValueError("n >= 1 required")
    if n == 1:
        return np.array([1.0]), np.array([1.0])

    # 初始猜测 (渐进公式)
    k = np.arange(1, n + 1)
    x_init = np.pi**2 * (k - 0.25)**2 / (4.0 * n)
    x = x_init.copy()

    # Newton 迭代精化根
    for _ in range(50):
        L = laguerre_polynomials(n, x)
        # 导数递推: L_n'(x) = -Σ_{k=0}^{n-1} L_k(x)
        dL = -np.sum(L[:-1, :], axis=0)
        dx = L[n, :] / (dL + 1e-30)
        x -= dx
        if np.max(np.abs(dx)) < 1e-14:
            break

    # 权重
    Ln_1 = laguerre_polynomials(n - 1, x)[n - 1, :]
    w = 1.0 / (n * Ln_1**2 + 1e-30)
    # 边界保护
    x = np.where(x > 0, x, 1e-10)
    w = np.where(w > 0, w, 0.0)
    return x, w


def laguerre_exponential_product(order: int, b: float, n_quad: int = 32) -> np.ndarray:
    """
    计算 Laguerre 指数内积矩阵 (基于 642_laguerre_product).

    T_{ij} = ∫_0^∞ exp(b*x) * L_i(x) * L_j(x) * exp(-x) dx
           = ∫_0^∞ L_i(x) * L_j(x) * exp(-(1-b)*x) dx

    用于随机微物理中参数扰动的 Galerkin 投影.
    """
    x, w = gauss_laguerre_nodes_weights(n_quad)
    L = laguerre_polynomials(order, x)
    T = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(i, order + 1):
            val = np.sum(w * np.exp(b * x) * L[i, :] * L[j, :])
            T[i, j] = val
            T[j, i] = val
    return T


def log_gamma_for_microphysics(x: float) -> float:
    """
    基于 toms291 的 log-Gamma, 用于 Gamma 分布雨滴谱参数.
    """
    if x <= 0.0:
        return -np.inf
    if x < 7.0:
        f = 1.0
        y = x
        while y < 7.0:
            f *= y
            y += 1.0
        return log_gamma_for_microphysics(y) - np.log(f)
    z = 1.0 / x**2
    s = (1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0)))) / x
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi) + s


def gamma_distribution_pdf(D: np.ndarray, N0: float, mu: float, lam: float) -> np.ndarray:
    """
    Gamma 分布雨滴谱密度 N(D) (m^{-3} mm^{-1}).

    N(D) = N0 * D^μ * exp(-ΛD)

    其中:
      N0: 截距参数
      μ: 形状参数
      Λ: 斜率参数
    """
    D = np.asarray(D)
    D = np.where(D > 0, D, 1e-10)
    return N0 * (D**mu) * np.exp(-lam * D)


def gamma_moment(k: int, N0: float, mu: float, lam: float) -> float:
    """
    Gamma 分布的第 k 阶矩:
      M_k = ∫ D^k N(D) dD = N0 * Γ(k+μ+1) / Λ^{k+μ+1}
    """
    if lam <= 0.0:
        return 0.0
    log_moment = np.log(N0 + 1e-30) + log_gamma_for_microphysics(k + mu + 1.0) - (k + mu + 1.0) * np.log(lam)
    return np.exp(log_moment)


class StochasticMicrophysics:
    """
    基于 Laguerre Polynomial Chaos 的随机微物理参数化.
    """

    def __init__(self, chaos_order: int = 4, n_quad: int = 16):
        self.order = chaos_order
        self.n_quad = n_quad
        # 预计算 Gauss-Laguerre 求积点
        self.xi_nodes, self.xi_weights = gauss_laguerre_nodes_weights(n_quad)
        # Laguerre 多项式在求积点上的值
        self.L_poly = laguerre_polynomials(chaos_order, self.xi_nodes)

    def project_to_chaos(self, f_values: np.ndarray) -> np.ndarray:
        """
        将函数在随机变量上的取值投影到 Laguerre 混沌系数.

        f_i = Σ_k w_k * f(ξ_k) * L_i(ξ_k)
        """
        coeffs = np.zeros(self.order + 1)
        for i in range(self.order + 1):
            coeffs[i] = np.sum(self.xi_weights * f_values * self.L_poly[i, :])
        return coeffs

    def evaluate_from_chaos(self, coeffs: np.ndarray, xi: float) -> float:
        """
        由混沌系数在特定 ξ 处重构函数值.
        """
        L = laguerre_polynomials(self.order, np.array([xi]))[:, 0]
        return float(np.dot(coeffs, L))

    def condensate_rate_ensemble(self, qv: float, qvs: float,
                                  tau_mean: float = 300.0,
                                  tau_std: float = 60.0) -> np.ndarray:
        """
        返回凝结率的 Laguerre 混沌系数.
        假设凝结时间尺度 τ 服从对数正态/Gamma 扰动.
        """
        # 将 τ 的扰动映射到 Laguerre 变量
        # τ(ξ) = τ_mean * exp(-ξ * tau_std / tau_mean)
        tau_vals = tau_mean * np.exp(-self.xi_nodes * tau_std / tau_mean)
        tau_vals = np.where(tau_vals > 1.0, tau_vals, 1.0)

        if qv > qvs:
            rates = (qv - qvs) / tau_vals
        else:
            rates = np.zeros_like(tau_vals)

        return self.project_to_chaos(rates)

    def precipitation_rate(self, ql: float, N0_mean: float = 8e6,
                           mu: float = 2.0, lam: float = 3.0) -> float:
        """
        基于 Gamma 分布谱的降水率估算 (mm/hr 量级, 归一化输出).

        使用 Marshall-Palmer 类谱的 3.67 阶矩近似:
          Z ∝ M_{3.67} ∝ N0 * Γ(3.67+μ+1) / Λ^{3.67+μ+1}
          R ∝ Z^{1/2} (经验 Z-R 关系)
        """
        # === HOLE 2 START ===
        # 任务: 基于 Gamma 分布雨滴谱矩计算降水率 (Z-R 关系)
        # 科学背景:
        #   1. 雷达反射率 Z 与雨滴谱 k 阶矩 M_k 成正比
        #   2. Marshall-Palmer 类谱中 Z ∝ M_{3.67}
        #   3. 经验 Z-R 关系: R = a * Z^b (通常 a≈200, b≈1.6  for mm/hr)
        #   4. 本代码使用简化归一化公式: R = a_coef * (M3 / M3_ref)^b_coef [mm/s]
        #   5. N0 = N0_mean * ql * 1e3 (液态水含量与截距参数关联)
        # TODO: implement precipitation rate from Gamma moment
        raise NotImplementedError("HOLE 2: precipitation_rate 的 Z-R 关系尚未实现")
        # === HOLE 2 END ===

    def stochastic_precipitation(self, ql_field: np.ndarray,
                                  xi_sample: float = 0.0) -> np.ndarray:
        """
        对整场的液态水含量计算随机降水率.
        """
        result = np.zeros_like(ql_field)
        for idx in np.ndindex(ql_field.shape):
            # 使用混沌展开加入随机扰动
            pert = 1.0 + 0.1 * xi_sample * np.sin(np.pi * idx[0] / max(1, ql_field.shape[0] - 1))
            result[idx] = self.precipitation_rate(ql_field[idx] * pert)
        return result
