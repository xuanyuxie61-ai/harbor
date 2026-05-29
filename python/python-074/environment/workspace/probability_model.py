r"""
probability_model.py
====================
涡脱落统计特性与不完全 Beta 分布概率模型模块。

科学背景
--------
在圆柱涡激振动中，涡脱落呈现准周期性行为，但受湍流、结构振动与流动
不稳定性的影响，其相位角 \theta(t) 存在随机涨落。我们将涡脱落相位
建模为在 [0, 2\pi] 上的随机变量，其概率密度函数可用 Beta 分布的
周期延拓描述：

    f(\theta) = \frac{1}{B(p,q)} \left(\frac{\theta}{2\pi}\right)^{p-1}
                \left(1 - \frac{\theta}{2\pi}\right)^{q-1}
                \frac{1}{2\pi}, \quad 0 \le \theta \le 2\pi

其中 B(p,q) = \Gamma(p)\Gamma(q) / \Gamma(p+q) 为 Beta 函数，
p, q > 0 为形状参数。

不完全 Beta 函数定义为：

    I_x(p,q) = \frac{1}{B(p,q)} \int_0^x t^{p-1} (1-t)^{q-1} dt,
    \quad 0 \le x \le 1

它在相位累积分布函数（CDF）计算中起核心作用：

    F(\theta) = I_{\theta/(2\pi)}(p, q)

当 p = q 时分布对称，对应锁定（lock-in）状态；
当 p \ne q 时分布偏斜，对应非同步或过渡状态。

本模块同时提供：
- 涡脱落频率的统计估计（自相关法、谱峰法）
- Strouhal 数置信区间
- 升力系数的幅值分布（对数正态近似）

对应原种子项目：
- 1267_toms179（不完全 Beta 函数 mdbeta + alogam 算法）
r"""

import numpy as np
from scipy.special import gammaln, beta as beta_func


def alogam(x):
    r"""
    计算 \ln \Gamma(x) 的近似值（基于 Stirling 展开与递推）。

    对应原种子项目 1267_toms179 中的 alogam 算法。
    对 x <= 0 返回错误标志。
    """
    if x <= 0.0:
        return 0.0, 1

    y = x
    if x < 7.0:
        f = 1.0
        z = y
        while z < 7.0:
            f *= z
            z += 1.0
        y = z
        f = -np.log(f)
    else:
        f = 0.0

    z = 1.0 / (y * y)
    value = (
        f
        + (y - 0.5) * np.log(y)
        - y
        + 0.918938533204673
        + (((-0.000595238095238 * z + 0.000793650793651) * z
            - 0.002777777777778) * z + 0.083333333333333) / y
    )
    return value, 0


def log_beta(p, q):
    r"""
    计算 \ln B(p,q) = \ln \Gamma(p) + \ln \Gamma(q) - \ln \Gamma(p+q)。
    利用 alogam 或 scipy gammaln。
    """
    if p <= 0.0 or q <= 0.0:
        raise ValueError("log_beta: p 和 q 必须为正。")
    lp, _ = alogam(p)
    lq, _ = alogam(q)
    lpq, _ = alogam(p + q)
    return lp + lq - lpq


def incomplete_beta_series(x, p, q, max_iter=1000, tol=1e-14):
    r"""
    不完全 Beta 函数的级数展开计算。

    对 x <= 0.5 直接计算；对 x > 0.5 利用对称性：
    I_x(p,q) = 1 - I_{1-x}(q,p)。

    算法核心（来自原 mdbeta）：
    1. 若 x <= 0.5，设 y=x, a=p, b=q
    2. 计算级数展开 S = \sum_{i=0}^{\infty} c_i
       c_0 = (p+q)/(p) * y^p
       c_{i+1} = c_i * (a+i)/(a+b+i) * y
    3. 结果 = c_0 / B(p,q) * (1 + S)

    此处采用更稳定的对数域计算。
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("incomplete_beta_series: x 必须在 [0,1] 内。")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # 对称性处理
    if x > 0.5:
        return 1.0 - incomplete_beta_series(1.0 - x, q, p, max_iter, tol)

    # 使用超几何函数级数：
    # I_x(p,q) = x^p / (p B(p,q)) * 2F1(p, 1-q; p+1; x)
    # 2F1 级数：c_n = c_{n-1} * (p+n-1)(1-q+n-1) / ((p+n) * n) * x
    log_b = log_beta(p, q)
    prefactor = np.exp(p * np.log(x) - log_b - np.log(p))

    coeff = 1.0
    sum_series = 1.0
    for n in range(1, max_iter):
        coeff *= (p + n - 1.0) * (1.0 - q + n - 1.0) / ((p + n) * n) * x
        sum_series += coeff
        if abs(coeff) < tol * abs(sum_series):
            break

    return prefactor * sum_series


def incomplete_beta(x, p, q):
    r"""
    不完全 Beta 函数 I_x(p,q) 的鲁棒计算接口。
    对小参数用级数，对大参数用 scipy 后备。
    """
    try:
        return incomplete_beta_series(x, p, q)
    except Exception:
        from scipy.special import betainc
        return betainc(p, q, x)


def phase_cdf(theta, p, q, period=2.0 * np.pi):
    r"""
    涡脱落相位累积分布函数：
    F(\theta) = I_{\theta/T}(p, q)，其中 T = 2\pi。
    """
    if theta < 0.0:
        return 0.0
    if theta > period:
        return 1.0
    x = theta / period
    return incomplete_beta(x, p, q)


def phase_pdf(theta, p, q, period=2.0 * np.pi):
    r"""
    涡脱落相位概率密度函数：
    f(\theta) = \frac{1}{B(p,q)} \left(\frac{\theta}{T}\right)^{p-1}
                \left(1-\frac{\theta}{T}\right)^{q-1} \frac{1}{T}
    """
    if theta < 0.0 or theta > period:
        return 0.0
    x = theta / period
    log_b = log_beta(p, q)
    log_f = (p - 1.0) * np.log(x) + (q - 1.0) * np.log(1.0 - x) - log_b - np.log(period)
    return np.exp(log_f)


def estimate_vortex_shedding_frequency(lift_history, dt):
    r"""
    由升力系数时间序列估计涡脱落频率。

    方法：
    1. 计算自相关函数 R(\tau)。
    2. 寻找第一峰值位置估计周期 T。
    3. 频率 f = 1/T。

    参数
    ----
    lift_history : ndarray
        升力系数时间序列。
    dt : float
        采样间隔。

    返回
    ----
    f_est : float
        估计频率（Hz）。
    strouhal : float
        Strouhal 数 St = f D / U（此处 D=U=1 归一化，返回 f_est）。
    """
    n = len(lift_history)
    if n < 10:
        return 0.0, 0.0

    # 去均值
    signal = lift_history - np.mean(lift_history)

    # 自相关（利用 FFT 加速）
    f_signal = np.fft.fft(signal, n=2 * n)
    autocorr = np.fft.ifft(f_signal * np.conj(f_signal)).real
    autocorr = autocorr[:n]
    if abs(autocorr[0]) > 1e-15:
        autocorr /= autocorr[0]
    else:
        autocorr[:] = 0.0

    # 寻找第一个正峰值（排除 \tau=0）
    peak_idx = None
    for i in range(2, n // 2):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            if autocorr[i] > 0.1:  # 阈值避免噪声峰值
                peak_idx = i
                break

    if peak_idx is None or peak_idx == 0:
        # 备选：FFT 峰值法
        freqs = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.abs(np.fft.rfft(signal))
        # 排除直流
        fft_vals[0] = 0.0
        peak_f_idx = np.argmax(fft_vals)
        f_est = freqs[peak_f_idx]
    else:
        f_est = 1.0 / (peak_idx * dt)

    return f_est, f_est  # St 在归一化单位下等于 f


def fit_beta_parameters(phase_samples):
    r"""
    由相位样本估计 Beta 分布参数 p, q（矩估计法）。

    对 x_i = \theta_i / (2\pi)，有
    \bar{x} = p/(p+q), \quad s^2 = \frac{pq}{(p+q)^2(p+q+1)}

    解得：
    p = \bar{x} (\bar{x}(1-\bar{x})/s^2 - 1)
    q = (1-\bar{x}) (\bar{x}(1-\bar{x})/s^2 - 1)
    """
    x = np.asarray(phase_samples) / (2.0 * np.pi)
    x = np.clip(x, 1e-8, 1.0 - 1e-8)

    mean_x = np.mean(x)
    var_x = np.var(x)

    if var_x < 1e-12:
        # 方差过小，退化为确定性
        p = q = 1000.0
        return p, q

    factor = mean_x * (1.0 - mean_x) / var_x - 1.0
    p = mean_x * factor
    q = (1.0 - mean_x) * factor

    p = max(p, 0.1)
    q = max(q, 0.1)
    return p, q


def log_normal_cl_cdf(cl_amp, mu_ln, sigma_ln):
    r"""
    升力幅值的对数正态分布累积函数：

    F(A) = \Phi\left( \frac{\ln A - \mu}{\sigma} \right)

    其中 \Phi 为标准正态 CDF。
    """
    from scipy.stats import norm
    if cl_amp <= 0:
        return 0.0
    z = (np.log(cl_amp) - mu_ln) / sigma_ln
    return norm.cdf(z)
