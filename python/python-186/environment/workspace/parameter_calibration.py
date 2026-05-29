"""
parameter_calibration.py
参数反演与校准模块

基于种子项目:
- 097_bisection_rc: 二分法求根
- 867_persistence: 流式统计 (Welford算法)
"""

import numpy as np
from typing import Tuple, List, Callable, Optional


class StreamingStatistics:
    """
    Welford在线算法计算流式统计量。

    单次遍历计算均值、方差和标准差:
        n_{k} = n_{k-1} + 1
        mean_k = mean_{k-1} + (x_k - mean_{k-1}) / n_k
        M2_k = M2_{k-1} + (x_k - mean_{k-1}) * (x_k - mean_k)
        var_k = M2_k / (n_k - 1)  (样本方差)
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = -float('inf')

    def update(self, x: float):
        """更新统计量"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    def variance(self) -> float:
        """样本方差"""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    def std(self) -> float:
        """样本标准差"""
        return np.sqrt(self.variance())

    def get_stats(self) -> dict:
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': self.variance(),
            'std': self.std(),
            'min': self.min_val,
            'max': self.max_val
        }


def bisection_root_finder(f: Callable[[float], float],
                          a: float, b: float,
                          tol: float = 1e-8,
                          max_iter: int = 100) -> Tuple[float, int, bool]:
    """
    二分法求根。

    要求 f(a) * f(b) < 0。

    算法:
        c = (a + b) / 2
        if f(c) == 0: 返回 c
        elif f(a)*f(c) < 0: b = c
        else: a = c
        重复直到 |b-a| < tol

    收敛速度: 线性，每步误差减半。
    迭代次数估计: n >= log2((b-a)/tol)
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        return (a + b) / 2.0, 0, False

    iterations = 0
    c = a

    while iterations < max_iter and abs(b - a) > tol:
        c = (a + b) / 2.0
        fc = f(c)

        if abs(fc) < tol:
            return c, iterations, True

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iterations += 1

    return c, iterations, True


def calibrate_beta_target(target_r0: float,
                          params_template: dict,
                          ode_solver: Callable,
                          tol: float = 1e-4) -> float:
    """
    通过二分法校准接触率 beta，使得基本再生数 R_0 达到目标值。

    问题: 给定 R_0^{target}，求 beta 使得 R_0(beta) = R_0^{target}。

    R_0 关于 beta 是单调递增的:
        R_0(beta) = beta * (p_sym/(gamma_I+alpha_H) + eta_A*(1-p_sym)/gamma_A)

    因此可以建立标量方程:
        g(beta) = R_0(beta) - target_r0 = 0
    """
    # TODO Hole_3: 实现基于R_0目标的接触率beta校准
    # 需要构建标量方程 g(beta) = R_0(beta) - target_r0 = 0
    # 其中R_0(beta)的公式必须与epidemic_dynamics.py中的compute_reproduction_number一致
    # 使用二分法在合适的搜索区间内求根
    raise NotImplementedError("Hole_3: calibrate_beta_target 尚未实现")


def maximum_likelihood_estimation(observed_data: np.ndarray,
                                  model_func: Callable,
                                  param_bounds: List[Tuple[float, float]],
                                  n_grid: int = 50) -> Tuple[np.ndarray, float]:
    """
    网格搜索最大似然估计。

    假设观测误差服从正态分布:
        p(y_obs | theta) ~ N(model(theta), sigma^2)

    对数似然:
        log L(theta) = -0.5 * sum_i (y_obs_i - model_i(theta))^2 / sigma^2 + const

    最大化对数似然等价于最小化残差平方和。
    """
    n_params = len(param_bounds)

    # 生成参数网格
    grids = [np.linspace(b[0], b[1], n_grid) for b in param_bounds]

    best_params = None
    best_ll = -float('inf')

    if n_params == 1:
        for p0 in grids[0]:
            try:
                pred = model_func(p0)
                residuals = observed_data - pred
                ll = -0.5 * np.sum(residuals**2)
                if ll > best_ll:
                    best_ll = ll
                    best_params = np.array([p0])
            except Exception:
                continue
    elif n_params == 2:
        for p0 in grids[0]:
            for p1 in grids[1]:
                try:
                    pred = model_func(p0, p1)
                    residuals = observed_data - pred
                    ll = -0.5 * np.sum(residuals**2)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = np.array([p0, p1])
                except Exception:
                    continue

    return best_params, best_ll


def akaike_information_criterion(log_likelihood: float, k: int, n: int) -> float:
    """
    AIC信息准则。

    AIC = 2k - 2 * ln(L)

    其中 k 为参数个数，n 为样本量。
    AICc (小样本修正):
        AICc = AIC + 2k(k+1)/(n-k-1)
    """
    aic = 2.0 * k - 2.0 * log_likelihood
    if n > k + 1:
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        return aicc
    return aic
