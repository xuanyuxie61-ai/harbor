
import numpy as np
from typing import Tuple, List, Optional


def align_time_series(times: np.ndarray,
                      values: np.ndarray,
                      grid: np.ndarray) -> np.ndarray:
    if len(times) != len(values):
        raise ValueError("times 和 values 长度必须相同。")
    if not np.all(np.diff(times) >= 0):

        idx = np.argsort(times)
        times = times[idx]
        values = values[idx]

    m = len(grid)
    aligned = np.zeros(m)
    for j in range(m):
        tau = grid[j]

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
    if len(x) < 2:
        raise ValueError("序列长度至少为 2。")
    return np.diff(x)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
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
    idx = np.argmax(np.abs(ccf))
    return int(lags[idx]), float(ccf[idx])


def granger_causality_f_stat(x: np.ndarray, y: np.ndarray,
                              max_lag: int = 3) -> Tuple[float, float]:
    n = len(y)
    if len(x) != n:
        raise ValueError("x 和 y 长度必须相同。")
    if n <= 2 * max_lag + 2:
        raise ValueError("样本量不足以进行 Granger 检验。")

    p = max_lag

    Y = y[max_lag:]
    X_lag = np.zeros((len(Y), p))
    Y_lag = np.zeros((len(Y), p))
    for i in range(p):
        X_lag[:, i] = x[max_lag - 1 - i:n - 1 - i]
        Y_lag[:, i] = y[max_lag - 1 - i:n - 1 - i]


    Xr = np.hstack([np.ones((len(Y), 1)), Y_lag])
    beta_r = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    resid_r = Y - Xr @ beta_r
    rss_r = np.sum(resid_r ** 2)


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

    from math import erf
    pval = max(0.0, 1.0 - erf(np.sqrt(df1 * F / 2.0)))
    return float(F), float(pval)


def demo():
    np.random.seed(17)
    n = 200
    t = np.linspace(0, 10, n)

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
