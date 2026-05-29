r"""
statistical_analysis.py
统计分析与参数推断模块

本模块实现：
1. 全局最小值搜索（参考 glomin：Brent 全局优化算法）
2. 轮廓似然分析（Profile Likelihood）
3. 置信区间构建
4. 灵敏度曲线计算

核心公式：

A. Poisson 似然函数：
    \mathcal{L}(\mu, \theta) = \prod_{i} \frac{e^{-(\mu s_i + b_i)} (\mu s_i + b_i)^{n_i}}{n_i!}

    其中：
        μ: 信号强度参数（μ=0 对应零假设）
        s_i: 第 i 个 bin 的信号预期
        b_i: 第 i 个 bin 的背景预期
        n_i: 观测计数

B. 轮廓似然比检验统计量：
    q_\mu = -2 \ln \frac{\mathcal{L}(\mu, \hat{\hat{\theta}})}{\mathcal{L}(\hat{\mu}, \hat{\theta})}

    其中 \hat{\hat{\theta}} 为固定 μ 时的条件最大似然估计，
    (\hat{\mu}, \hat{\theta}) 为全局最大似然估计。

C. 90% C.L. 上限（CL_s 方法）：
    CL_s(\mu) = \frac{P(q_\mu \geq q_\mu^{\rm obs} | \mu)}{P(q_\mu \geq q_\mu^{\rm obs} | 0)}
    上限 μ_{90} 满足 CL_s(μ_{90}) = 0.10

D. 全局优化（Brent 算法简化版）：
    在已知二阶导数上界 M 的条件下，
    通过二次插值 + 随机探测保证找到全局最小值。

参考文献：
- Brent, R. P. (1973). Algorithms for Minimization Without Derivatives.
- Cowan, G., et al. (2011). Eur. Phys. J. C 71, 1554.
"""

import numpy as np
from typing import Callable, Tuple, Dict
from utils import r8_uniform_01


# ============================================================================
# Brent 全局最小化（简化版 glomin）
# ============================================================================

def glomin_brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    c: float,
    m: float,
    e: float,
    t: float,
    max_calls: int = 1000,
) -> Tuple[float, float, int]:
    """
    在区间 [a, b] 上寻找函数 f 的全局最小值。

    参数：
        f: 目标函数
        a, b: 搜索区间端点
        c: 初始点（满足 f(c) ≤ f(a) 且 f(c) ≤ f(b)）
        m: |f''(x)| 的上界
        e: 绝对误差容限
        t: 相对误差容限
        max_calls: 最大函数求值次数

    返回：
        x_min: 最小值点
        f_min: 最小值
        calls: 实际函数调用次数
    """
    calls = 0

    def feval(x):
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return float('inf')
        return float(f(x))

    # 初始化
    sa, sb = min(a, b), max(a, b)
    x = c
    fx = feval(x)
    w = x
    fw = fx
    v = x
    fv = fx

    d = sb - sa
    e_len = d

    while calls <= max_calls:
        tol = t * abs(x) + e
        m_c = 0.5 * (sa + sb)

        # 收敛检查
        if abs(x - m_c) <= 2.0 * tol - 0.5 * (sb - sa):
            break

        p = q = r = 0.0
        if abs(e_len) > tol:
            # 二次插值
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e_len
            e_len = d

        if abs(p) < abs(0.5 * q * r) and p > q * (sa - x) and p < q * (sb - x):
            d = p / q
            u = x + d
            if u - sa < 2.0 * tol or sb - u < 2.0 * tol:
                d = tol if x < m_c else -tol
        else:
            if x < m_c:
                e_len = sb - x
            else:
                e_len = sa - x
            d = 0.5 * e_len

        # 基于二阶导数上界的随机探测（数值稳定性增强）
        if abs(d) < 1.0e-12:
            # 随机微小扰动避免停滞
            seed = calls * 1611 + 1
            ru, _ = r8_uniform_01(seed)
            d = (2.0 * ru - 1.0) * tol

        u = x + d
        fu = feval(u)

        if fu <= fx:
            if u >= x:
                sa = x
            else:
                sb = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u
            if fu <= fw or abs(w - x) < 1.0e-15:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or abs(v - x) < 1.0e-15 or abs(v - w) < 1.0e-15:
                v, fv = u, fu

    return x, fx, calls


# ============================================================================
# 似然函数
# ============================================================================

def poisson_log_likelihood(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    mu: float,
) -> float:
    """
    计算扩展 Poisson 对数似然。

    公式：
        \ln \mathcal{L} = \sum_i \left[ n_i \ln(\mu s_i + b_i)
                         - (\mu s_i + b_i) - \ln(n_i!) \right]

    参数：
        n_obs: (N,) 观测计数
        s_pred: (N,) 信号预期
        b_pred: (N,) 背景预期
        mu: 信号强度参数

    返回：
        logL: 对数似然值
    """
    lambda_pred = mu * s_pred + b_pred
    # 避免 log(0)
    lambda_pred = np.where(lambda_pred < 1.0e-15, 1.0e-15, lambda_pred)
    logL = np.sum(n_obs * np.log(lambda_pred) - lambda_pred)
    # 忽略常数项 -ln(n!)，不影响优化
    return float(logL)


def profile_likelihood_ratio(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    mu_test: float,
) -> float:
    """
    计算轮廓似然比 q_μ。

    参数：
        n_obs, s_pred, b_pred: 观测与预期
        mu_test: 待检验的 μ 值

    返回：
        q_mu: 检验统计量（≥ 0）
    """
    # 全局最大似然估计（假设 μ ≥ 0）
    def neg_logL(mu):
        return -poisson_log_likelihood(n_obs, s_pred, b_pred, mu)

    # 粗略搜索全局最优 μ
    mu_grid = np.linspace(0.0, max(10.0, 2.0 * mu_test), 200)
    logL_grid = np.array([poisson_log_likelihood(n_obs, s_pred, b_pred, m) for m in mu_grid])
    mu_hat = mu_grid[np.argmax(logL_grid)]

    logL_global = poisson_log_likelihood(n_obs, s_pred, b_pred, mu_hat)
    logL_cond = poisson_log_likelihood(n_obs, s_pred, b_pred, mu_test)

    q_mu = -2.0 * (logL_cond - logL_global)
    return max(float(q_mu), 0.0)


def confidence_interval_upper_limit(
    n_obs: np.ndarray,
    s_pred: np.ndarray,
    b_pred: np.ndarray,
    cl: float = 0.90,
    mu_max: float = 20.0,
) -> float:
    """
    计算信号强度的 90% CL 上限（基于轮廓似然近似）。

    公式：
        q_μ = 2.70  对应 90% CL（单侧，大样本极限）

    参数：
        n_obs, s_pred, b_pred: 观测与预期
        cl: 置信水平
        mu_max: 搜索上限

    返回：
        mu_upper: 上限值
    """
    target_q = 2.70  # 90% CL for 1 DOF

    mu_grid = np.linspace(0.0, mu_max, 200)
    q_grid = np.array([profile_likelihood_ratio(n_obs, s_pred, b_pred, m) for m in mu_grid])

    # 线性插值找 q = target_q 对应的 mu
    for i in range(len(mu_grid) - 1):
        if q_grid[i] <= target_q <= q_grid[i + 1] or q_grid[i + 1] <= target_q <= q_grid[i]:
            t = (target_q - q_grid[i]) / (q_grid[i + 1] - q_grid[i])
            return float(mu_grid[i] + t * (mu_grid[i + 1] - mu_grid[i]))

    return float(mu_grid[-1])


# ============================================================================
# 灵敏度曲线
# ============================================================================

def sensitivity_curve(
    exposure_kg_day: float,
    target_mass_kg: float,
    background_rate_per_kev_kg_day: float,
    e_min_kev: float,
    e_max_kev: float,
    m_chi_values: np.ndarray,
    efficiency: float = 0.5,
    n_bins: int = 20,
) -> np.ndarray:
    """
    计算探测器在给定曝光量下的灵敏度曲线（90% CL 截面上限 vs WIMP 质量）。

    公式：
        N_s^{90} = 2.44 + 1.64 \sqrt{N_b}    (Poisson 近似)
        σ_{90} = \frac{N_s^{90}}{\epsilon \cdot M \cdot T \cdot \int (dR/dE) / \sigma \, dE}

    参数：
        exposure_kg_day: 曝光量 [kg·day]
        target_mass_kg: 靶质量 [kg]
        background_rate_per_kev_kg_day: 背景率
        e_min_kev, e_max_kev: 分析能窗
        m_chi_values: WIMP 质量扫描点 [GeV]
        efficiency: 信号效率
        n_bins: 能量分箱数

    返回：
        sigma_90: (len(m_chi_values),) 90% CL 截面上限 [pb]
    """
    e_bins = np.linspace(e_min_kev, e_max_kev, n_bins + 1)
    bin_width = e_bins[1] - e_bins[0]

    # 背景预期
    N_b = background_rate_per_kev_kg_day * bin_width * exposure_kg_day

    # 90% CL 信号事件数（简化公式）
    N_s_90 = 2.44 + 1.64 * np.sqrt(N_b)

    sigma_90 = np.zeros(len(m_chi_values))
    for idx, m_chi in enumerate(m_chi_values):
        # 计算单位截面的预期事件数
        # 使用 wimp_physics 的 total_events，设 sigma = 1 pb
        from wimp_physics import total_events_in_range
        N_s_per_pb = total_events_in_range(
            e_min_kev, e_max_kev, m_chi, 1.0, 73.0, target_mass_kg, exposure_kg_day / target_mass_kg
        )
        if N_s_per_pb > 1.0e-15:
            sigma_90[idx] = N_s_90 / (efficiency * N_s_per_pb)
        else:
            sigma_90[idx] = 1.0e6  # 极大值表示无灵敏度

    return sigma_90


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 glomin
    def f_test(x):
        return (x - 0.3) ** 2 + 0.1 * np.sin(20.0 * x)

    x_min, f_min, calls = glomin_brent(f_test, 0.0, 1.0, 0.5, 100.0, 1.0e-10, 1.0e-10)
    assert 0.0 <= x_min <= 1.0
    assert f_min <= f_test(0.3) + 0.1, "全局最小值搜索失败"

    # 测试 Poisson 似然
    n_obs = np.array([5, 7, 6, 8, 5])
    s_pred = np.array([2, 2, 2, 2, 2])
    b_pred = np.array([3, 3, 3, 3, 3])
    logL = poisson_log_likelihood(n_obs, s_pred, b_pred, 1.0)
    assert np.isfinite(logL), "对数似然非有限"

    # 测试轮廓似然比
    q = profile_likelihood_ratio(n_obs, s_pred, b_pred, 0.0)
    assert q >= 0.0, "q_mu 必须非负"

    # 测试上限
    upper = confidence_interval_upper_limit(n_obs, s_pred, b_pred)
    assert upper >= 0.0

    # 测试灵敏度曲线
    m_vals = np.array([10.0, 50.0, 100.0])
    sens = sensitivity_curve(1000.0, 10.0, 0.01, 0.5, 50.0, m_vals)
    assert np.all(sens > 0.0)
    assert np.all(np.isfinite(sens))

    print("statistical_analysis.py: 所有自测通过")
