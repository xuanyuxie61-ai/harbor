"""
后激波-中微子信号因果诊断 (from 1222_Bagged_TimeSeries_Causality).

原项目：Bagged Time Series Causality (Eyring Climate Group)
使用 bootstrap 时间序列进行 PCMCI 风格因果检验.

类比到超新星：
  我们要检验 "激波脉动 → 中微子光度变化" 因果链是否成立.
  时间序列 X(t) = 激波半径脉动, Y(t) = 中微子光度脉动.
  零假设 H_0: X 不对 Y 有预测性 (X ↛ Y).

PCMCI 简化版本 (Runge 2019):
  1. 条件选择 (PC 算法)：对每对 (X, Y), 找出其"父母集" P
  2. 矩条件独立 (MCI) 检验：X_t ⊥ Y_{t+τ} | P
     若 p-value < α, 拒绝独立性, 认为有因果.

这里简化为**Granger 因果检验** (Granger 1969):
  X → Y 成立 ⟺ VAR(p) 模型中 X 的滞后项显著预测 Y
  F 统计量：
    F = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1))
  其中 RSS_r = 受限模型残差平方和, RSS_u = 不受限.

Bootstrap 不确定性 (Bagging):
  对时间序列做 moving block bootstrap (MBB):
    块长 L, 随机抽取 N/L 块, 拼接为新序列.
  重复 B 次得 F 的 bootstrap 分布, 计算 95% CI.
"""
from __future__ import annotations
import math
import numpy as np


def vector_autoregression(X: np.ndarray, p: int):
    """VAR(p) 模型拟合.

  X: (T, d) 矩阵, d 为变量数.
  X_t = Σ_{k=1}^p A_k X_{t-k} + ε_t
  OLS 拟合：β = (Z^T Z)^{-1} Z^T Y
  Z = (X_{p-1}, ..., X_0; X_p, ..., X_1; ...)
  Y = (X_p, X_{p+1}, ...)
  """
    T, d = X.shape
    if T <= p + 1:
        raise ValueError("时间序列太短, 无法拟合 VAR(p)")
    # 构造设计矩阵 Z (T-p, p*d) 和响应 Y (T-p, d)
    n = T - p
    Z = np.zeros((n, p * d), dtype=np.float64)
    Y = X[p:]
    for k in range(p):
        Z[:, k * d:(k + 1) * d] = X[p - 1 - k:T - 1 - k]
    # OLS: β = (Z^T Z)^{-1} Z^T Y
    ZtZ = Z.T @ Z
    reg = 1.0e-8 * np.eye(p * d)  # 正则化
    beta = np.linalg.solve(ZtZ + reg, Z.T @ Y)
    Y_hat = Z @ beta
    resid = Y - Y_hat
    rss = float(np.sum(resid ** 2))
    return {'beta': beta, 'residuals': resid, 'rss': rss, 'n': n}


def granger_causality(X: np.ndarray, Y: np.ndarray, p: int = 3) -> dict:
    """Granger 因果检验 X → Y.

    不受限模型：Y_t = Σ a_k Y_{t-k} + Σ b_k X_{t-k} + ε_t
    受限模型：Y_t = Σ a_k Y_{t-k} + ε_t  (b_k = 0)
    F 统计量：((RSS_r - RSS_u)/p) / (RSS_u/(n-2p-1))
    """
    if len(X) != len(Y):
        raise ValueError("X 和 Y 长度必须一致")
    T = len(X)
    # 受限模型
    Y_lag = np.column_stack([Y[p - 1 - k:T - 1 - k] for k in range(p)])
    Zr = np.column_stack([np.ones(T - p), Y_lag])
    Yr = Y[p:]
    beta_r = np.linalg.lstsq(Zr, Yr, rcond=None)[0]
    RSS_r = float(np.sum((Yr - Zr @ beta_r) ** 2))
    # 不受限模型
    X_lag = np.column_stack([X[p - 1 - k:T - 1 - k] for k in range(p)])
    Zu = np.column_stack([np.ones(T - p), Y_lag, X_lag])
    beta_u = np.linalg.lstsq(Zu, Yr, rcond=None)[0]
    RSS_u = float(np.sum((Yr - Zu @ beta_u) ** 2))
    n = T - p
    if RSS_u <= 0:
        RSS_u = 1.0e-30
    F_stat = ((RSS_r - RSS_u) / p) / (RSS_u / max(n - 2 * p - 1, 1))
    # 近似 p-value (F 分布)
    from math import erf
    # F(p, n-2p-1) 分布近似；简化为正态近似 (大样本)
    df1, df2 = p, max(n - 2 * p - 1, 2)
    # Wilson-Hilferty 变换
    z = ((F_stat / df1) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df2))) / \
        math.sqrt(2.0 / (9.0 * df2)) if df2 > 2 else 0.0
    p_value = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return {'F_statistic': F_stat, 'p_value': p_value,
            'RSS_restricted': RSS_r, 'RSS_unrestricted': RSS_u,
            'n': n, 'df1': df1, 'df2': df2}


def moving_block_bootstrap(x: np.ndarray, block_len: int,
                           rng: np.random.Generator,
                           n_bootstrap: int = 50) -> np.ndarray:
    """Moving block bootstrap (Politis-Romano 1994).

  从 x 中抽取 n_bootstrap 个长度为 block_len 的连续块,
  拼接成长度与原序列相当的 bootstrap 样本.
  """
    n = len(x)
    n_blocks = (n + block_len - 1) // block_len
    boots = np.zeros((n_bootstrap, n), dtype=np.float64)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        parts = [x[s:s + block_len] for s in starts]
        concat = np.concatenate(parts)[:n]
        boots[b] = concat
    return boots


def bagged_granger_causality(X: np.ndarray, Y: np.ndarray,
                              p: int = 3, block_len: int = 10,
                              n_bootstrap: int = 20,
                              seed: int = 42) -> dict:
    """Bootstrap 加权的 Granger 因果检验.

    返回 F 统计量的 bootstrap 中位数和 95% CI.
    """
    rng = np.random.default_rng(seed)
    # 原始
    res_orig = granger_causality(X, Y, p)
    # Bootstrap
    F_boots = []
    for _ in range(n_bootstrap):
        Xb = moving_block_bootstrap(X, block_len, rng, 1)[0]
        Yb = moving_block_bootstrap(Y, block_len, rng, 1)[0]
        try:
            res = granger_causality(Xb, Yb, p)
            F_boots.append(res['F_statistic'])
        except Exception:
            continue
    F_boots = np.array(F_boots)
    if len(F_boots) == 0:
        F_boots = np.array([res_orig['F_statistic']])
    return {'F_median': float(np.median(F_boots)),
            'F_2.5': float(np.percentile(F_boots, 2.5)),
            'F_97.5': float(np.percentile(F_boots, 97.5)),
            'original': res_orig}
