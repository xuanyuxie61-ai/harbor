"""
uncertainty_model.py
====================
概率不确定性量化模块

融合种子项目:
  - 918_prob : 概率密度函数、累积分布函数、特殊函数
               (误差函数 ERF、不完全 Gamma、不完全 Beta、Bessel 函数等)

科学背景:
  深度学习势函数常存在外推误差。为量化预测不确定性，
  本模块实现基于变分贝叶斯与证据学习的不确定性估计。

  关键概率模型:
    1. 高斯似然: p(y|x) = N(μ(x), σ²(x))
    2. Gamma 先验: p(σ²) = Ga(α, β)
    3. 预测分布为 Student-t (共轭高斯-Gamma):
        p(y|x) = St(μ(x), β/α, 2α)
    4. 使用不完全 Gamma 函数计算分位数与置信区间。

  此外，利用误差函数 ERF 构造基于距离的原子相关性核:
      k(r) = erf( (r_cut - r) / (sqrt(2) * l) )
"""

import numpy as np
from typing import Tuple


# ------------------------------------------------------------------
# 1. 特殊函数 (源自 prob)
# ------------------------------------------------------------------

def error_function(x: float) -> float:
    """
    误差函数 ERF(x) = (2/√π) ∫_0^x exp(-t²) dt。
    使用 Taylor 展开与渐近展开混合计算。
    """
    x = float(x)
    if x < 0:
        return -error_function(-x)
    # 使用 scipy 近似或手动实现
    # Abramowitz & Stegun 公式 7.1.26
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return y


def incomplete_gamma(a: float, x: float) -> float:
    """
    正则化下不完全 Gamma 函数 P(a, x) = γ(a, x) / Γ(a)。
    使用级数展开 (Pearson 级数, AS239)。
    """
    if x < 0.0 or a <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0
    # 级数展开
    gln = gammaln(a)
    ap = a
    sum_ = 1.0 / a
    del_ = sum_
    n = 1
    while n <= 10000:
        ap += 1.0
        del_ *= x / ap
        sum_ += del_
        if abs(del_) < abs(sum_) * 1e-10:
            break
        n += 1
    return sum_ * np.exp(-x + a * np.log(x) - gln)


def gammaln(x: float) -> float:
    """Log Gamma 函数，使用 Lanczos 近似。
    基于 Godfrey 实现的 Numerical Recipes 公式。
    """
    if x <= 0:
        return np.inf
    p = np.array([
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ], dtype=np.float64)
    x = float(x)
    if x < 0.5:
        return np.log(np.pi) - np.log(np.sin(np.pi * x)) - gammaln(1.0 - x)
    x -= 1.0
    z = x + 0.5 + 7.0  # = x + 7.5
    A = 0.99999999999980993
    for i, pi in enumerate(p):
        A += pi / (x + i + 1.0)
    # Lanczos 公式: ln Γ(x+1) = 0.5 ln(2π) + ln A + (x+0.5) ln z - z
    return 0.5 * np.log(2.0 * np.pi) + np.log(A) + (x + 0.5) * np.log(z) - z


def digamma(x: float) -> float:
    """
    Digamma 函数 ψ(x) = d/dx ln Γ(x)。
    使用 Bernardo 算法 (AS103)。
    """
    if x <= 0:
        return np.nan
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += np.log(x) - 0.5 * inv_x - inv_x2 * (
        1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0))
    )
    return result


# ------------------------------------------------------------------
# 2. 不确定性模型
# ------------------------------------------------------------------

class EvidentialRegressor:
    """
    证据回归器：预测 μ(x), ν(x), α(x), β(x) 四个参数，
    输出正态逆 Gamma (NIG) 分布，从而得到预测均值与不确定性。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        """
        构建一个简单的两层 MLP 参数化 NIG 分布。
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 第一层
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # 第二层 -> 4 个 NIG 参数
        limit2 = np.sqrt(6.0 / (hidden_dim + 4))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, 4))
        self.b2 = np.zeros(4)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播到 NIG 原始参数。"""
        h = np.maximum(0.0, x @ self.W1 + self.b1)  # ReLU
        out = h @ self.W2 + self.b2
        return out

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        返回 NIG 参数 (gamma, nu, alpha, beta)，均通过 softplus 保证正。

        gamma : 预测均值 μ
        nu    : 精度参数 (伪观测数)
        alpha : Gamma 形状参数
        beta  : Gamma 率参数
        """
        raw = self._forward(x)
        # softplus: log(1 + exp(x))，保证正且数值稳定
        gamma = raw[:, 0]
        nu = np.log(1.0 + np.exp(raw[:, 1])) + 1e-6
        alpha = np.log(1.0 + np.exp(raw[:, 2])) + 1.01  # > 1 保证有限方差
        beta = np.log(1.0 + np.exp(raw[:, 3])) + 1e-6
        return gamma, nu, alpha, beta

    def nig_nll(self, gamma: np.ndarray, nu: np.ndarray,
                alpha: np.ndarray, beta: np.ndarray, y: np.ndarray) -> float:
        """
        NIG 负对数似然 (用于回归):
            L = 0.5 log(π/ν) - α log(2β) + (α+0.5) log(ν(y-γ)² + 2β)
                + log Γ(α) - log Γ(α+0.5)
        """
        # TODO: implement NIG negative log-likelihood computation
        # Hint: use the formula involving omega, pi, nu, alpha, beta, gamma, y
        # and the gammaln helper defined above.
        raise NotImplementedError("nig_nll is not implemented")

    def uncertainty(self, alpha: np.ndarray, beta: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回预测方差（偶然不确定性 aleatoric）与
        参数方差（认知不确定性 epistemic）。

        aleatoric = β / (α - 1)   (若 α > 1)
        epistemic = β / (ν (α - 1))
        """
        safe_alpha = np.where(alpha > 1.0, alpha, 1.01)
        aleatoric = beta / (safe_alpha - 1.0)
        epistemic = beta / (nu * (safe_alpha - 1.0))
        return aleatoric, epistemic

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]


# ------------------------------------------------------------------
# 3. 基于误差函数的原子相关性核
# ------------------------------------------------------------------

def erf_correlation_kernel(distances: np.ndarray, r_cut: float, lengthscale: float = 0.5) -> np.ndarray:
    """
    误差函数截断核:
        k(r) = 0.5 * (1 + erf((r_cut - r) / (√2 * l)))
    在 r → 0 时趋近 1，在 r → r_cut 时平滑趋近 0。
    """
    arg = (r_cut - distances) / (np.sqrt(2.0) * lengthscale)
    k = 0.5 * (1.0 + np.array([error_function(float(a)) for a in arg]))
    k = np.where(distances > r_cut, 0.0, k)
    return k


# ------------------------------------------------------------------
# 4. 多维正态采样 (源自 prob 的 multivariate_normal_sample)
# ------------------------------------------------------------------

def multivariate_normal_sample(mean: np.ndarray, cov: np.ndarray, n_samples: int) -> np.ndarray:
    """
    从多维正态分布 N(mean, cov) 采样。
    使用 Cholesky 分解: cov = L L^T,  z = mean + L ε,  ε ~ N(0, I)。
    """
    d = len(mean)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
    eps = np.random.randn(n_samples, d)
    return mean.reshape(1, d) + eps @ L.T
