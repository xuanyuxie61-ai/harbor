r"""
time_series_utils.py
================================================================================
时间序列对齐、差分与因果时滞分析工具

原项目映射: 135_calpak — 日期时间差计算与时间序列索引对齐

科学背景
--------
在纵向因果推断（Longitudinal Causal Inference）中，不同变量往往以不同
采样频率观测，或存在时间戳偏差。为了正确估计因果效应，必须先将时间序列
对齐到统一的离散时间网格上。

此外，因果效应往往存在时滞（temporal delay）。Granger 因果检验的核心思想
即为：若 $X$ 的过去值能显著提升对 $Y$ 的预测能力，则称 $X$ Granger-导致 $Y$。

核心公式
--------
1. 时间网格对齐（最近邻插值）：
   给定非均匀观测 $\{(t_k, x_k)\}$，对齐到网格 $T = \{\tau_j\}$：
   $$ \tilde{x}(\tau_j) = x_{k^*}, \quad k^* = \arg\min_k |t_k - \tau_j| $$

2. 向前差分（离散速度）：
   $$ \Delta x_j = x_{j+1} - x_j $$

3. Granger 因果检验统计量（简化版，基于自回归）：
   受限模型：$y_t = \sum_{i=1}^{p}\alpha_i y_{t-i} + \epsilon_t$
   完整模型：$y_t = \sum_{i=1}^{p}\alpha_i y_{t-i} + \sum_{j=1}^{q}\beta_j x_{t-j} + \eta_t$
   F 统计量：
   $$ F = \frac{(RSS_r - RSS_f)/q}{RSS_f / (n - p - q - 1)} \sim F_{q, n-p-q-1} $$

4. 互相关函数（Cross-Correlation Function, CCF）：
   $$ \rho_{xy}(h) = \frac{\sum_t (x_t - \bar{x})(y_{t+h} - \bar{y})}{\sqrt{\sum_t (x_t - \bar{x})^2 \sum_t (y_t - \bar{y})^2}} $$
   峰值位置对应潜在因果时滞。
r"""

import numpy as np
from typing import Tuple, List, Optional


def align_time_series(times: np.ndarray,
                      values: np.ndarray,
                      grid: np.ndarray) -> np.ndarray:
    r"""
    将非均匀时间序列对齐到规则网格（最近邻插值）。

    Parameters
    ----------
    times : ndarray, shape (n,)
        原始时间戳（已排序）。
    values : ndarray, shape (n,)
        原始观测值。
    grid : ndarray, shape (m,)
        目标时间网格。

    Returns
    -------
    aligned : ndarray, shape (m,)
        对齐后的序列。
    r"""
    if len(times) != len(values):
        raise ValueError("times 和 values 长度必须相同。")
    if not np.all(np.diff(times) >= 0):
        # 排序
        idx = np.argsort(times)
        times = times[idx]
        values = values[idx]

    m = len(grid)
    aligned = np.zeros(m)
    for j in range(m):
        tau = grid[j]
        # 二分查找最近邻
        idx = np.searchsorted(times, tau)
        if idx == 0:
            aligned[j] = values[0]
        elif idx >= len(times):
            aligned[j] = values[-1]
        else:
            if abs(times[idx] - tau) < abs(times[idx - 1] - tau):
                aligned[j] = values[idx]
            else:
                aligned[j] = values[idx - 1]
    return aligned


def forward_difference(x: np.ndarray) -> np.ndarray:
    r"""
    计算向前差分 $\Delta x_j = x_{j+1} - x_j$。
    r"""
    if len(x) < 2:
        raise ValueError("序列长度至少为 2。")
    return np.diff(x)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    计算互相关函数 $\rho_{xy}(h)$，$h = -\text{max_lag},\dots,\text{max_lag}$。

    Returns
    -------
    lags : ndarray
    ccf : ndarray
    r"""
    n = len(x)
    if len(y) != n:
        raise ValueError("x 和 y 长度必须相同。")
    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag 超出范围。")

    xc = x - np.mean(x)
    yc = y - np.mean(y)
    norm = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    if norm < 1e-14:
        lags = np.arange(-max_lag, max_lag + 1)
        return lags, np.zeros(len(lags))

    ccf = np.zeros(2 * max_lag + 1)
    for h in range(-max_lag, max_lag + 1):
        if h >= 0:
            s = np.sum(xc[:n - h] * yc[h:])
        else:
            s = np.sum(xc[-h:] * yc[:n + h])
        ccf[h + max_lag] = s / norm
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, ccf


def find_peak_lag(lags: np.ndarray, ccf: np.ndarray) -> Tuple[int, float]:
    r"""
    找到互相关函数的峰值及对应滞后。

    Returns
    -------
    lag : int
        峰值滞后。
    peak_val : float
        峰值大小。
    r"""
    idx = np.argmax(np.abs(ccf))
    return int(lags[idx]), float(ccf[idx])


def granger_causality_f_stat(x: np.ndarray, y: np.ndarray,
                              max_lag: int = 3) -> Tuple[float, float]:
    r"""
    简化版 Granger 因果检验：检验 x 是否 Granger-导致 y。

    返回 F 统计量与近似 p 值（基于 F 分布）。
    r"""
    n = len(y)
    if len(x) != n:
        raise ValueError("x 和 y 长度必须相同。")
    if n <= 2 * max_lag + 2:
        raise ValueError("样本量不足以进行 Granger 检验。")

    p = max_lag
    # 构造滞后矩阵
    Y = y[max_lag:]
    X_lag = np.zeros((len(Y), p))
    Y_lag = np.zeros((len(Y), p))
    for i in range(p):
        X_lag[:, i] = x[max_lag - 1 - i:n - 1 - i]
        Y_lag[:, i] = y[max_lag - 1 - i:n - 1 - i]

    # 受限模型：仅 y 的滞后
    Xr = np.hstack([np.ones((len(Y), 1)), Y_lag])
    beta_r = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    resid_r = Y - Xr @ beta_r
    rss_r = np.sum(resid_r ** 2)

    # 完整模型：y 的滞后 + x 的滞后
    Xf = np.hstack([np.ones((len(Y), 1)), Y_lag, X_lag])
    beta_f = np.linalg.lstsq(Xf, Y, rcond=None)[0]
    resid_f = Y - Xf @ beta_f
    rss_f = np.sum(resid_f ** 2)

    q = p
    df1 = q
    df2 = n - 2 * p - 1
    if df2 <= 0 or rss_f < 1e-14:
        return 0.0, 1.0

    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    # 近似 p 值（简化计算）
    from math import erf
    pval = max(0.0, 1.0 - erf(np.sqrt(df1 * F / 2.0)))
    return float(F), float(pval)


def demo():
    r"""模块自测试。"""
    np.random.seed(17)
    n = 200
    t = np.linspace(0, 10, n)
    # x 领先 y 5 步
    x = np.sin(t) + 0.2 * np.random.randn(n)
    y = np.sin(t - 0.25) + 0.3 * x[np.clip(np.arange(n) - 5, 0, n - 1)] + 0.2 * np.random.randn(n)

    grid = np.linspace(0, 10, 50)
    x_aligned = align_time_series(t, x, grid)
    y_aligned = align_time_series(t, y, grid)

    lags, ccf = cross_correlation(x_aligned, y_aligned, max_lag=10)
    peak_lag, peak_val = find_peak_lag(lags, ccf)
    print(f"[time_series_utils] 互相关峰值滞后: {peak_lag}, 峰值: {peak_val:.4f}")

    F, pval = granger_causality_f_stat(x, y, max_lag=3)
    print(f"[time_series_utils] Granger F={F:.3f}, p={pval:.4f}")
    return peak_lag, F


if __name__ == "__main__":
    demo()
