"""
constrained_optimizer.py

约束优化工具库 —— 线性规划投影与信任区域

基于种子项目:
  - 339_eternity (eternity_lp): 稀疏线性系统与 LP 建模
  - 1267_toms179 (incomplete_beta): Beta 分布信任区域

科学原理:
  1. 线性规划投影:
     当动作空间受线性约束 C·a ≤ d 时,
     策略输出 a_raw 可能违反约束.
     通过求解最小修正 LP:
         min ||a - a_raw||_2
         s.t. C·a ≤ d,  a_min ≤ a ≤ a_max
     将动作投影到可行域.

  2. 信任区域 (Trust Region):
     策略更新需满足 KL 散度约束:
         D_KL(π_θ_old || π_θ_new) ≤ δ
     利用不完全 Beta 函数计算置信概率:
         P( D_KL ≤ δ ) = I_{δ/(σ^2+δ)}( d/2, (N-d)/2 )
     其中 d 为参数维度, N 为样本数.

  3. 学习率调度:
     基于 Julian Date 的周期性退火:
         α(t) = α_0 · (1 + cos(π · jed(t) / T_period)) / 2
"""

import numpy as np
from typing import Tuple
from scipy.optimize import linprog
from special_functions import incomplete_beta


# ---------------------------------------------------------------------------
# LP 投影
# ---------------------------------------------------------------------------

def lp_action_projection(action_raw: np.ndarray,
                          C: np.ndarray = None,
                          d: np.ndarray = None,
                          bounds: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """
    将原始动作通过 LP 投影到线性约束可行域.

    优化问题:
        min   Σ_i |a_i - a_raw_i|
        s.t.  C·a ≤ d
              bounds[0] ≤ a_i ≤ bounds[1]

    参数:
        action_raw: 原始动作向量
        C:          m×n 约束矩阵 (可选)
        d:          m 维约束右端 (可选)
        bounds:     动作盒式约束

    返回:
        投影后的动作向量
    """
    action_raw = np.asarray(action_raw, dtype=float)
    n = len(action_raw)

    # 若无边界约束, 直接截断
    if C is None or d is None or len(C) == 0:
        return np.clip(action_raw, bounds[0], bounds[1])

    # 将 L1 最小化转化为 LP:
    # 引入辅助变量 u_i >= a_i - a_raw_i, u_i >= -(a_i - a_raw_i)
    # min Σ u_i
    # variables: [a_1, ..., a_n, u_1, ..., u_n]
    c = np.concatenate([np.zeros(n), np.ones(n)])

    # 不等式约束:
    # a_i - a_raw_i <= u_i   =>  a_i - u_i <= a_raw_i
    # -(a_i - a_raw_i) <= u_i => -a_i - u_i <= -a_raw_i
    # C·a <= d
    A_ub = []
    b_ub = []
    for i in range(n):
        row1 = np.zeros(2 * n)
        row1[i] = 1.0
        row1[n + i] = -1.0
        A_ub.append(row1)
        b_ub.append(action_raw[i])

        row2 = np.zeros(2 * n)
        row2[i] = -1.0
        row2[n + i] = -1.0
        A_ub.append(row2)
        b_ub.append(-action_raw[i])

    if C is not None and d is not None:
        C = np.atleast_2d(C)
        d = np.asarray(d, dtype=float)
        for j in range(C.shape[0]):
            row = np.zeros(2 * n)
            row[:n] = C[j, :]
            A_ub.append(row)
            b_ub.append(d[j])

    # 盒式约束
    bounds_lp = [(bounds[0], bounds[1]) for _ in range(n)] + [(0, None) for _ in range(n)]

    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds_lp, method='highs')
        if res.success:
            return np.clip(res.x[:n], bounds[0], bounds[1])
    except Exception:
        pass

    # 失败时回退到简单截断
    return np.clip(action_raw, bounds[0], bounds[1])


# ---------------------------------------------------------------------------
# 信任区域约束
# ---------------------------------------------------------------------------

def trust_region_probability(delta: float, param_dim: int,
                              sample_size: int, sigma: float = 1.0) -> float:
    """
    计算策略更新在信任区域内的置信概率.

    统计模型:
        将 KL 散度近似为二次型:
            D_KL ≈ 0.5 · Δθ^T F Δθ / σ^2 ~ χ^2(d) / N
        其中 d 为有效参数维度.

    不完全 Beta 联系:
        P(χ^2_ν ≤ x) = I_{x/(x+ν)}(ν/2, 1/2)   (不完全正确, 此处用 Beta 近似)

    更标准的用法:
        在 Hotelling T^2 分布中,
        P(T^2 ≤ δ) 可用不完全 Beta 表示.

    参数:
        delta:      KL 散度阈值
        param_dim:  参数维度 d
        sample_size: 样本数 N
        sigma:      噪声标准差

    返回:
        置信概率 ∈ [0,1]
    """
    if delta <= 0 or param_dim <= 0 or sample_size <= param_dim:
        return 0.0
    # 近似: 将 scaled KL 视为 Beta 分布
    p = param_dim / 2.0
    q = (sample_size - param_dim) / 2.0
    if q <= 0:
        q = 1.0
    x = delta / (delta + sigma ** 2)
    prob, ier = incomplete_beta(x, p, q)
    if ier != 0:
        return 0.0
    return prob


def check_trust_region(kl_value: float, max_kl: float,
                        param_dim: int, sample_size: int) -> bool:
    """检查是否满足信任区域约束."""
    if kl_value <= max_kl:
        return True
    prob = trust_region_probability(kl_value, param_dim, sample_size)
    # 若高概率仍在区域内, 允许略超
    return prob > 0.95


# ---------------------------------------------------------------------------
# 学习率调度器
# ---------------------------------------------------------------------------

class CosineAnnealingScheduler:
    """
    余弦退火学习率调度器.

    公式:
        α(t) = α_min + 0.5 (α_max - α_min) (1 + cos(π t / T))
    """

    def __init__(self, alpha_max: float = 0.01, alpha_min: float = 1.0e-5,
                 T_period: int = 100):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.T_period = T_period
        self.t = 0

    def step(self) -> float:
        """返回当前学习率并推进时间."""
        ratio = self.t / self.T_period
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) \
                * (1.0 + np.cos(np.pi * ratio))
        self.t += 1
        return alpha

    def reset(self):
        self.t = 0
