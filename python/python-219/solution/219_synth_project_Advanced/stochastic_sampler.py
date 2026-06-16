"""
stochastic_sampler.py
=====================

随机采样与不确定性量化模块。

本模块融合两个种子项目:
  - 047_asa183          : Wichmann-Hill 三种子伪随机数发生器 (周期 6.95e12)
  - 699_log_normal_truncated_ab : 截断对数正态分布采样 / CDF / PDF / 方差

在最优控制问题中, 参数不确定性通过随机抽样建模:
  - 比冲 I_sp 的扰动服从截断对数正态分布  (物理上必须为正且有上界)
  - 大气密度的随机偏差
  - 推力的乘性噪声

核心公式:
---------
1. Wichmann-Hill PRNG 递推:
     s1 <- (171 * s1) mod 30269
     s2 <- (172 * s2) mod 30307
     s3 <- (170 * s3) mod 30323
     x   <- s1/30269 + s2/30307 + s3/30323  (mod 1)

2. 对数正态 PDF (参数 mu, sigma):
     f(x) = (1 / (x sigma sqrt(2 pi))) exp(-(ln x - mu)^2 / (2 sigma^2))

3. 截断对数正态 CDF (截断区间 [a, b]):
     F_{[a,b]}(x) = (F(x) - F(a)) / (F(b) - F(a)),  x in [a, b]

4. Inverse-CDF 采样:
     X = F^{-1}(F(a) + U (F(b) - F(a))),  U ~ Uniform(0, 1)

5. 截断对数正态方差:
     Var = E[X^2] - E[X]^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ===========================================================================
# 1. Wichmann-Hill PRNG (来自 047_asa183 / r8_random)
# ===========================================================================
class WichmannHillPRNG:
    """Wichmann-Hill 三种子伪随机数发生器 (AS 183).

    循环长度约为 6.95 x 10^12, 输出在 (0, 1) 上近似均匀分布.
    三个种子 s1, s2, s3 必须满足:
        1 <= s1 <= 30268,  1 <= s2 <= 30306,  1 <= s3 <= 30322

    Reference
    ---------
    B. Wichmann, D. Hill, "Algorithm AS 183: An Efficient and Portable
    Pseudo-Random Number Generator", Applied Statistics 31 (1982), 188-190.
    """

    MOD1: int = 30269
    MOD2: int = 30307
    MOD3: int = 30323
    MUL1: int = 171
    MUL2: int = 172
    MUL3: int = 170

    def __init__(self, s1: int = 12345, s2: int = 23456, s3: int = 34567):
        self._s1 = self._validate(s1, 1, self.MOD1 - 1, "s1")
        self._s2 = self._validate(s2, 1, self.MOD2 - 1, "s2")
        self._s3 = self._validate(s3, 1, self.MOD3 - 1, "s3")

    @staticmethod
    def _validate(v: int, lo: int, hi: int, name: str) -> int:
        if not (lo <= v <= hi):
            raise ValueError(f"{name} = {v} 不在合法范围 [{lo}, {hi}]")
        return int(v)

    def next_float(self) -> float:
        """返回一个在 (0, 1) 内的伪随机数."""
        self._s1 = (self.MUL1 * self._s1) % self.MOD1
        self._s2 = (self.MUL2 * self._s2) % self.MOD2
        self._s3 = (self.MUL3 * self._s3) % self.MOD3
        x = self._s1 / self.MOD1 + self._s2 / self.MOD2 + self._s3 / self.MOD3
        return x - math.floor(x)

    def sample_uniform(self, a: float, b: float) -> float:
        """Uniform(a, b) 采样."""
        if b <= a:
            raise ValueError(f"区间参数非法: a={a}, b={b}")
        return a + (b - a) * self.next_float()

    def sample_vector(self, n: int) -> List[float]:
        """生成 n 个 Uniform(0, 1) 样本."""
        if n <= 0:
            return []
        return [self.next_float() for _ in range(n)]


# ===========================================================================
# 2. 标准正态 CDF 及其逆 (erf 实现)
# ===========================================================================
def normal_01_cdf(x: float) -> float:
    """标准正态分布 CDF: Phi(x) = (1 + erf(x / sqrt(2))) / 2."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_01_cdf_inv(p: float) -> float:
    """标准正态 CDF 的逆, 采用 Abramowitz & Stegun 有理逼近 + Halley 精化.

    适用范围 0 < p < 1. 边界 p -> 0 或 p -> 1 使用对称性.

    算法:
    -----
    1. 初始近似: Beasley-Springer-Moro 有理逼近
    2. Halley 精化 2 次: x_{k+1} = x_k - f/f' * (1 / (1 - 0.5 f f'' / f'^2))
       其中 f(x) = Phi(x) - p, f'(x) = phi(x), f''(x) = -x phi(x)
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p = {p} 不在 (0, 1) 内")

    # 对称性处理
    if p > 0.5:
        return -normal_01_cdf_inv(1.0 - p)

    # Beasley-Springer-Moro 有理逼近 (对 p <= 0.5)
    t = math.sqrt(math.log(1.0 / (p * p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    x = -x  # 因 p < 0.5 故 x 应为负

    # Halley 精化
    for _ in range(3):
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        f = normal_01_cdf(x) - p
        x = x - f / phi  # Newton 步
    return x


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """正态分布 N(mu, sigma^2) 的 CDF."""
    if sigma <= 0.0:
        raise ValueError(f"sigma = {sigma} 必须为正")
    return normal_01_cdf((x - mu) / sigma)


def normal_cdf_inv(p: float, mu: float, sigma: float) -> float:
    """正态分布 CDF 的逆."""
    if sigma <= 0.0:
        raise ValueError(f"sigma = {sigma} 必须为正")
    return mu + sigma * normal_01_cdf_inv(p)


# ===========================================================================
# 3. 对数正态分布及其截断版本 (来自 699_log_normal_truncated_ab)
# ===========================================================================
def log_normal_pdf(x: float, mu: float, sigma: float) -> float:
    """对数正态 PDF:
        f(x) = (1 / (x sigma sqrt(2 pi))) exp(-(ln x - mu)^2 / (2 sigma^2))
    要求 x > 0, sigma > 0.
    """
    if x <= 0.0:
        return 0.0
    if sigma <= 0.0:
        raise ValueError(f"sigma = {sigma} 必须为正")
    z = (math.log(x) - mu) / sigma
    return math.exp(-0.5 * z * z) / (x * sigma * math.sqrt(2.0 * math.pi))


def log_normal_cdf(x: float, mu: float, sigma: float) -> float:
    """对数正态 CDF:  F(x) = Phi((ln x - mu) / sigma)."""
    if x <= 0.0:
        return 0.0
    if sigma <= 0.0:
        raise ValueError(f"sigma = {sigma} 必须为正")
    return normal_01_cdf((math.log(x) - mu) / sigma)


def log_normal_cdf_inv(p: float, mu: float, sigma: float) -> float:
    """对数正态 CDF 的逆:  F^{-1}(p) = exp(mu + sigma * Phi^{-1}(p))."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p = {p} 不在 [0, 1] 内")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return math.inf
    return math.exp(mu + sigma * normal_01_cdf_inv(p))


def log_normal_truncated_ab_check(
    mu: float, sigma: float, a: float, b: float
) -> bool:
    """检查截断对数正态参数合法性:
        sigma > 0,  0 <= a < b.
    """
    if sigma <= 0.0:
        return False
    if not (0.0 <= a < b):
        return False
    return True


def log_normal_truncated_ab_pdf(
    x: float, mu: float, sigma: float, a: float, b: float
) -> float:
    """截断对数正态 PDF:
        f_{[a,b]}(x) = f(x) / (F(b) - F(a)),  x in [a, b]
                       0,                        otherwise
    """
    if not log_normal_truncated_ab_check(mu, sigma, a, b):
        raise ValueError(
            f"参数非法: mu={mu}, sigma={sigma}, a={a}, b={b}"
        )
    if not (a <= x <= b):
        return 0.0
    denom = log_normal_cdf(b, mu, sigma) - log_normal_cdf(a, mu, sigma)
    if denom <= 0.0:
        raise ValueError("截断区间的 CDF 差分为零, 请扩大 [a, b] 或减小 sigma")
    return log_normal_pdf(x, mu, sigma) / denom


def log_normal_truncated_ab_cdf(
    x: float, mu: float, sigma: float, a: float, b: float
) -> float:
    """截断对数正态 CDF:
        F_{[a,b]}(x) = (F(x) - F(a)) / (F(b) - F(a)),  x in [a, b]
                       0,                                x < a
                       1,                                x > b
    """
    if not log_normal_truncated_ab_check(mu, sigma, a, b):
        raise ValueError("参数非法")
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    fa = log_normal_cdf(a, mu, sigma)
    fb = log_normal_cdf(b, mu, sigma)
    fx = log_normal_cdf(x, mu, sigma)
    denom = fb - fa
    if denom <= 0.0:
        raise ValueError("截断区间的 CDF 差分为零")
    return (fx - fa) / denom


def log_normal_truncated_ab_cdf_inv(
    p: float, mu: float, sigma: float, a: float, b: float
) -> float:
    """截断对数正态 CDF 逆 (Inverse-CDF 采样):
        X = F^{-1}(F(a) + p * (F(b) - F(a)))
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p = {p} 不在 [0, 1] 内")
    if p == 0.0:
        return a
    if p == 1.0:
        return b
    fa = log_normal_cdf(a, mu, sigma)
    fb = log_normal_cdf(b, mu, sigma)
    target = fa + p * (fb - fa)
    return log_normal_cdf_inv(target, mu, sigma)


def log_normal_truncated_ab_sample(
    mu: float, sigma: float, a: float, b: float, rng: WichmannHillPRNG
) -> float:
    """采样截断对数正态分布 (Inverse-CDF 法).

    算法:  令 U ~ Uniform(0, 1),
           X = F^{-1}(F(a) + U * (F(b) - F(a))).

    Parameters
    ----------
    mu, sigma : float
        底层对数正态参数 (即 ln X 的均值和标准差).
    a, b : float
        截断区间 [a, b], 0 <= a < b.
    rng : WichmannHillPRNG
        随机数发生器.
    """
    if not log_normal_truncated_ab_check(mu, sigma, a, b):
        raise ValueError("参数非法")
    u = rng.next_float()
    return log_normal_truncated_ab_cdf_inv(u, mu, sigma, a, b)


def log_normal_truncated_ab_variance(
    mu: float, sigma: float, a: float, b: float, n_quad: int = 200
) -> float:
    """截断对数正态方差 (数值积分):
        Var[X] = E[X^2] - E[X]^2
               = integral_a^b x^2 f_{[a,b]}(x) dx
                 - (integral_a^b x f_{[a,b]}(x) dx)^2

    采用复合 Simpson 法, n_quad 个子区间 (偶数).
    """
    if not log_normal_truncated_ab_check(mu, sigma, a, b):
        raise ValueError("参数非法")
    if n_quad % 2 != 0:
        n_quad += 1
    h = (b - a) / n_quad
    denom = log_normal_cdf(b, mu, sigma) - log_normal_cdf(a, mu, sigma)
    if denom <= 0.0:
        raise ValueError("截断区间概率为零")

    def integrand(x: float, power: int) -> float:
        return (x ** power) * log_normal_pdf(x, mu, sigma) / denom

    # Simpson 复合求积
    def simpson(power: int) -> float:
        s = integrand(a, power) + integrand(b, power)
        for i in range(1, n_quad):
            x = a + i * h
            coeff = 4 if (i % 2 == 1) else 2
            s += coeff * integrand(x, power)
        return s * h / 3.0

    ex = simpson(1)
    ex2 = simpson(2)
    var = ex2 - ex * ex
    return max(var, 0.0)  # 数值误差处理


# ===========================================================================
# 4. 多维场景采样器
# ===========================================================================
@dataclass
class UncertaintyScenario:
    """单个不确定性场景."""
    isp_perturbation: float     # 比冲乘性扰动 (如 1.02 表示 +2%)
    thrust_perturbation: float  # 推力乘性扰动
    density_perturbation: float # 大气密度乘性扰动
    thermal_perturbation: float # 热扩散系数乘性扰动
    weight: float               # 场景权重 (用于求积)

    def as_dict(self) -> dict:
        return {
            "isp": self.isp_perturbation,
            "thrust": self.thrust_perturbation,
            "density": self.density_perturbation,
            "thermal": self.thermal_perturbation,
            "weight": self.weight,
        }


def generate_scenarios(
    n_scenarios: int,
    seed_tuple: Tuple[int, int, int] = (12345, 23456, 34567),
    isp_params: Tuple[float, float, float, float] = (0.0, 0.03, 0.90, 1.10),
    thrust_params: Tuple[float, float, float, float] = (0.0, 0.02, 0.95, 1.05),
    density_params: Tuple[float, float, float, float] = (0.0, 0.10, 0.70, 1.30),
    thermal_params: Tuple[float, float, float, float] = (0.0, 0.05, 0.85, 1.15),
) -> List[UncertaintyScenario]:
    """生成 n_scenarios 个不确定性场景.

    每个参数的扰动建模为截断对数正态分布, 以 1.0 为基准.
    对于参数 theta, 扰动因子 xi = 1 + delta, 其中 delta ~ LogNormal(mu, sigma^2)
    截断于 [a_rel, b_rel] (相对扰动).

    Parameters
    ----------
    n_scenarios : int
        场景数量.
    seed_tuple : (s1, s2, s3)
        Wichmann-Hill PRNG 的种子.
    *_params : (mu, sigma, a_rel, b_rel)
        各参数的截断对数正态设置.
    """
    if n_scenarios <= 0:
        raise ValueError("场景数必须为正")
    rng = WichmannHillPRNG(*seed_tuple)

    def _sample_one(mu_, sigma_, a_rel, b_rel) -> float:
        # delta ~ TruncLogNormal; xi = 1 + delta
        a_abs = max(0.0, a_rel - 1.0) if a_rel > 0 else a_rel
        b_abs = b_rel - 1.0 if b_rel > 1.0 else b_rel
        if a_abs >= b_abs:
            a_abs, b_abs = 0.0, 0.0
        delta = log_normal_truncated_ab_sample(mu_, sigma_, a_abs, b_abs, rng)
        return 1.0 + delta

    scenarios: List[UncertaintyScenario] = []
    for _ in range(n_scenarios):
        xi_isp = _sample_one(*isp_params)
        xi_thr = _sample_one(*thrust_params)
        xi_den = _sample_one(*density_params)
        xi_thm = _sample_one(*thermal_params)
        scenarios.append(
            UncertaintyScenario(
                isp_perturbation=xi_isp,
                thrust_perturbation=xi_thr,
                density_perturbation=xi_den,
                thermal_perturbation=xi_thm,
                weight=1.0 / n_scenarios,
            )
        )
    return scenarios


# ===========================================================================
# 5. 自检
# ===========================================================================
def self_check() -> None:
    """对采样器进行内部一致性自检."""
    rng = WichmannHillPRNG(999, 1000, 1001)
    samples = rng.sample_vector(10000)
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    print(f"[WH-PRNG] mean = {mean:.5f}  (期望 0.5)")
    print(f"[WH-PRNG] var  = {var:.5f}  (期望 ~1/12)")

    # 截断对数正态采样
    mu_, sigma_, a_, b_ = 0.0, 0.3, 0.5, 2.0
    samples2 = [
        log_normal_truncated_ab_sample(mu_, sigma_, a_, b_, rng)
        for _ in range(2000)
    ]
    mean2 = sum(samples2) / len(samples2)
    var2 = log_normal_truncated_ab_variance(mu_, sigma_, a_, b_)
    print(
        f"[TruncLogNormal] mean={mean2:.4f}, analytic_var={var2:.5f}"
    )


if __name__ == "__main__":
    self_check()
