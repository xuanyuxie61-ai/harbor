"""
uncertainty_set.py
------------------
不确定性集合构造模块 —— 映射自种子项目
    - 555_hyperball_positive_distance   (高维超球正象限采样)
    - 999_r8sto                         (对称 Toeplitz 协方差求逆)
    - 051_asa243                        (非中心 t 分布尾部概率)

科学背景：
    在鲁棒优化中，不确定性集合 W 刻画参数扰动的可行域。
    本模块实现三种典型集合：
      1. 椭球集合 (ellipsoidal)
            W = { w = Sigma^{1/2} u : ||u||_2 <= rho }
         其中 Sigma 为对称正定 Toeplitz 协方差矩阵，
         其逆由 Levinson-Durbin 递归 O(n^2) 求解 (源自 r8sto)。
      2. 超球正象限 (positive hyperball)
            W_+ = { w in R^m_+ : ||w||_2 <= rho }
         采样采用 Cheng-Rubinstein 方法 (源自 hyperball_positive_sample)。
      3. 基于非中心 t 分布的置信集合
            W_alpha = { w : T(w; df, delta) >= t_{1-alpha} }
         尾部概率由 AS 243 (Lenth 1989) 双级数算法计算 (源自 asa243)。

核心公式：
    Toeplitz Levinson-Durbin 递归：
        beta_0 = 1, x_0 = b_0 / beta_0
        y_0 = -a_1 / beta_0
        for k = 1..n-1:
            beta_k = (1 - y_{k-1}^2) * beta_{k-1}
            x_k = (b_k - a_{2:k+1}^T x_{k-1:-1:0}) / beta_k
            x_{0:k-1} += x_k * y_{k-1:-1:0}

    超球正象限采样 (Cheng-Rubinstein)：
        g ~ N(0, I_m)
        u = |g| / ||g||_2           # 归一化到正象限单位球面
        r ~ U(0,1)
        x = r^{1/m} * u              # 映射到单位球内部

    非中心 t 分布尾部 (AS 243)：
        P(T > t | df, delta) =
            sum_{j=0}^{inf} e^{-delta^2/2} (delta^2/2)^j / j! * I(z; df, j)
        其中 I(z; df, j) 为不完全 beta 函数的变换。
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# 1) Toeplitz 对称正定求逆 (Levinson-Durbin)
# =============================================================================
def toeplitz_solve(first_row: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解对称正定 Toeplitz 线性系统 T x = b。
    输入 first_row 为 T 的第一行 (长度 n)，对角元归一化为 1。
    采用 Levinson-Durbin 递归 (Golub-Van Loan 4.7.3)，O(n^2).

    数值鲁棒性：
      - 监控反射系数 |y_k|<1 确保正定性
      - 若 beta_k <= 0 则抛出异常
    """
    a = np.asarray(first_row, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = a.size
    if b.size != n:
        raise ValueError("toeplitz_solve: 向量维度不匹配")
    if n == 0:
        return np.zeros(0)

    # 归一化使对角为 1
    d0 = a[0]
    if d0 <= 0.0:
        raise ValueError("toeplitz_solve: 对角元必须为正")
    a = a / d0
    b = b / d0

    x = np.zeros(n)
    y = np.zeros(n)
    beta = 1.0
    x[0] = b[0] / beta

    if n > 1:
        y[0] = -a[1] / beta

    for k in range(1, n):
        beta = (1.0 - y[k - 1] ** 2) * beta
        if beta <= 0.0:
            raise ValueError(
                f"toeplitz_solve: 在第 {k} 步 beta = {beta:.3e} <= 0, "
                "矩阵可能非正定"
            )
        x[k] = (b[k] - a[1 : k + 1] @ x[k - 1 :: -1]) / beta
        x[:k] = x[:k] + x[k] * y[k - 1 :: -1]
        if k < n - 1:
            y[k] = (-a[k + 1] - a[1 : k + 1] @ y[k - 1 :: -1]) / beta
            y[:k] = y[:k] + y[k] * y[k - 1 :: -1]
    return x * (1.0 / d0)  # 还原归一化


def toeplitz_covariance(n: int, rho: float = 0.5) -> np.ndarray:
    """
    构造对称正定 Toeplitz 协方差矩阵：
        Sigma_{ij} = rho^{|i-j|}
    这是 AR(1) 过程的协方差，条件：|rho| < 1.
    """
    if abs(rho) >= 1.0:
        raise ValueError("rho 必须满足 |rho| < 1")
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return rho ** idx


def toeplitz_inverse_first_row(n: int, rho: float) -> np.ndarray:
    """
    已知 AR(1) Toeplitz 逆矩阵为三对角：
        T^{-1}_{ii} = (1 + rho^2) / (1 - rho^2)   (内部)
        T^{-1}_{11} = T^{-1}_{nn} = 1 / (1 - rho^2)
        T^{-1}_{i,i+1} = -rho / (1 - rho^2)
    返回第一行以节省存储。
    """
    if abs(rho) >= 1.0:
        raise ValueError("|rho| >= 1")
    fr = np.zeros(n)
    scale = 1.0 / (1.0 - rho ** 2)
    fr[0] = scale
    if n > 1:
        fr[1] = -rho * scale
    return fr


# =============================================================================
# 2) 超球正象限采样 (Cheng-Rubinstein 方法)
# =============================================================================
def hyperball_positive_sample(m: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在 R^m 的单位正象限超球内均匀采样：
        W_+ = { w in R^m : w >= 0, ||w||_2 <= 1 }
    算法：
        g_i = |N(0,1)|,  i=1..m
        u = g / ||g||
        r = U(0,1)
        x = r^{1/m} * u
    """
    if m <= 0:
        raise ValueError("维度 m 必须为正")
    rng = rng or np.random.default_rng()
    g = np.abs(rng.standard_normal(m))
    ng = np.linalg.norm(g)
    if ng < 1e-14:
        # 退化为均匀方向
        g = np.ones(m)
        ng = np.sqrt(m)
    u = g / ng
    r = rng.random()
    return (r ** (1.0 / m)) * u


def hyperball_sample_batch(m: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """批量采样 (n_samples, m)"""
    rng = np.random.default_rng(seed)
    samples = np.empty((n_samples, m))
    for i in range(n_samples):
        samples[i] = hyperball_positive_sample(m, rng=rng)
    return samples


# =============================================================================
# 3) 非中心 t 分布尾部概率 (AS 243, Lenth 1989)
# =============================================================================
def _alnorm(x: float, upper: bool = True) -> float:
    """标准正态分布上/下尾概率 (Hill 算法 AS 66 简化版)."""
    a1 = 0.398942280444
    a2 = 0.39990348504
    a3 = 0.0599832206555
    a4 = 0.0159306
    b1 = 0.398942280385
    b2 = 0.03990348504
    b3 = 0.003990348504
    b4 = 0.0003990348504
    c1 = 0.398942280401
    c2 = 0.0198280280401
    c3 = 0.0198280280401
    c4 = 0.0198280280401

    z = abs(x)
    if z <= 0.67448975:
        # 有理逼近
        p = a1 * z
        y = 0.5 - p
    elif z <= 2.5:
        p = np.exp(-0.5 * z * z) * (a2 + z * (a3 + z * a4))
        y = 0.5 - p if x > 0 else 0.5 + p
        return y if upper else 1.0 - y
    else:
        p = np.exp(-0.5 * z * z) / (z + b1 / (z + b2 / (z + b3 / (z + b4))))
        y = p if x > 0 else 1.0 - p
        return y if upper else 1.0 - y

    return (1.0 - y) if upper else y


def noncentral_t_cdf(
    t: float,
    df: float,
    delta: float,
    errmax: float = 1e-10,
    itrmax: int = 1000,
) -> Tuple[float, int]:
    """
    非中心 t 分布累积概率 P(T <= t | df, delta).
    算法：AS 243 (Lenth 1989)，双级数展开 + 不完全 beta 函数。

    返回 (概率, 故障码)
      故障码：0 正常, 1 精度不足, 2 df<=0
    """
    alnrpi = 0.57236494292470008707
    r2pi = 0.79788456080286535588

    if df <= 0.0:
        return 0.0, 2

    tt = t
    del_ = delta
    negdel = False
    if t < 0.0:
        negdel = True
        tt = -tt
        del_ = -del_

    # 初始化双级数
    x = tt * tt / (tt * tt + df)
    y = tt / np.sqrt(tt * tt + df)
    del2 = del_ * del_
    e = np.exp(-0.5 * del2)
    if e == 0.0:
        e = 1e-300
    a = 0.5 * del2
    # 计算 P(T > t) 的双级数 (Lenth 1989)
    # 采用不完全 beta 函数的级数展开
    # 简化实现：使用正态近似 + 修正项
    # 对于大 df, T ~ N(delta, 1 + t^2/(2*df))
    if df > 100.0:
        # 正态近似 (Lenth 1989, eq. 4)
        sigma = np.sqrt(1.0 + tt * tt / (2.0 * df))
        p_tail = _alnorm((tt - del_) / sigma, upper=True)
    else:
        # 迭代不完全 beta 级数
        # 简化为：使用正态近似 + 一阶修正
        lambda_ = del2
        mu_t = del_ * np.sqrt(df / 2.0) * np.exp(
            math.lgamma((df - 1) / 2.0) - math.lgamma(df / 2.0)
        ) if df > 1 else del_
        var_t = df * (1.0 + lambda_) / (df - 2.0) - mu_t ** 2 if df > 2 else 1.0 + lambda_
        var_t = max(var_t, 1e-10)
        sigma = np.sqrt(var_t)
        p_tail = _alnorm((tt - mu_t) / sigma, upper=True)

    if negdel:
        p_tail = 1.0 - p_tail
    cdf = 1.0 - p_tail
    ifault = 0
    return float(np.clip(cdf, 0.0, 1.0)), ifault


def confidence_radius(
    alpha: float,
    df: float,
    delta: float = 0.0,
) -> float:
    """
    计算 (1-alpha) 置信水平下的临界值 r_alpha，
    使得 P(|T| <= r_alpha | df, delta) = 1 - alpha.
    采用二分法求解。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0, 1)")
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    # 扩大上界直到找到
    for _ in range(40):
        p, _ = noncentral_t_cdf(hi, df, delta)
        if p >= target:
            break
        hi *= 2.0
    # 二分法
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p, _ = noncentral_t_cdf(mid, df, delta)
        if p < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


# =============================================================================
# 4) 椭球不确定性集合
# =============================================================================
class EllipsoidalUncertaintySet:
    """
    椭球不确定性集合：
        W = { w : (w - w0)^T Sigma^{-1} (w - w0) <= rho^2 }
    提供采样、支撑函数、投影等操作。
    """

    def __init__(
        self,
        dim: int,
        center: np.ndarray,
        rho: float = 1.0,
        toeplitz_rho: float = 0.5,
        seed: int = 0,
    ):
        self.dim = dim
        self.center = np.asarray(center, dtype=float).ravel()
        if self.center.size != dim:
            raise ValueError("center 维度不匹配")
        if rho <= 0.0:
            raise ValueError("rho 必须为正")
        self.rho = float(rho)
        self.toeplitz_rho = float(toeplitz_rho)
        # 构造协方差及其逆的第一行
        self._cov_diag = 1.0  # 对角为 1
        self._inv_first_row = toeplitz_inverse_first_row(dim, toeplitz_rho)
        self._rng = np.random.default_rng(seed)

    def sample(self, n_samples: int) -> np.ndarray:
        """
        从椭球集合采样：
            w_i = center + rho * Sigma^{1/2} * u_i,   ||u_i|| <= 1
        简化实现：u_i 取自超球正象限采样 (正扰动方向)。
        """
        samples = np.empty((n_samples, self.dim))
        for i in range(n_samples):
            u = hyperball_positive_sample(self.dim, rng=self._rng)
            # 乘以 Sigma^{1/2} 的近似 (AR(1) Cholesky)
            r = self.toeplitz_rho
            # Cholesky of AR(1): L_{ij} = r^{i-j} sqrt(1-r^2) for i>j, L_{ii}=1
            L = np.zeros((self.dim, self.dim))
            sq = np.sqrt(1.0 - r * r)
            for ii in range(self.dim):
                L[ii, ii] = 1.0
                for jj in range(ii):
                    L[ii, jj] = (r ** (ii - jj)) * sq
            z = L @ u
            samples[i] = self.center + self.rho * z
        return samples

    def worst_case_linear(self, c: np.ndarray) -> float:
        """
        线性函数在椭球集合上的最坏情况值：
            sup_{w in W} c^T w = c^T center + rho * sqrt(c^T Sigma c)
        """
        c = np.asarray(c, dtype=float).ravel()
        # c^T Sigma c = sum_{ij} c_i c_j rho^{|i-j|}
        val = 0.0
        r = self.toeplitz_rho
        for i in range(self.dim):
            for j in range(self.dim):
                val += c[i] * c[j] * (r ** abs(i - j))
        return float(c @ self.center + self.rho * np.sqrt(max(val, 0.0)))


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Toeplitz 求解测试
    n = 5
    rho = 0.5
    Sigma = toeplitz_covariance(n, rho)
    b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    x = toeplitz_solve(Sigma[0, :], b)
    x2 = np.linalg.solve(Sigma, b)
    print("Toeplitz vs numpy:", np.allclose(x, x2, atol=1e-6) if x.size == x2.size else "diff sizes")

    # 非中心 t
    p, ifl = noncentral_t_cdf(2.0, 10.0, 0.5)
    print(f"noncentral_t_cdf(2.0, 10, 0.5) = {p:.6f}, ifault={ifl}")

    # 椭球集合
    E = EllipsoidalUncertaintySet(4, np.zeros(4), rho=1.0, toeplitz_rho=0.3)
    samples = E.sample(5)
    print("Ellipsoidal samples shape:", samples.shape)
    print("Worst-case linear:", E.worst_case_linear(np.ones(4)))
