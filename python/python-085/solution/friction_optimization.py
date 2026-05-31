"""
friction_optimization.py
摩擦参数反演与优化模块
融合种子项目：
  - 1266_toms178（Hooke-Jeeves 直接搜索优化）
  - 856_peaks_movie（峰值测试函数，提取其数学形式用于验证）
"""
import numpy as np
from typing import Callable, Tuple, Optional


def peaks_function(x: float, y: float) -> float:
    r"""
    MATLAB peaks 测试函数（提取自 856_peaks_movie）：

    f(x,y) = 3(1-x)^2 \exp(-x^2 - (y+1)^2)
             - 10(\frac{x}{5} - x^3 - y^5) \exp(-x^2 - y^2)
             - \frac{1}{3} \exp(-(x+1)^2 - y^2)

    用于构造接触本构的非线性测试曲面。
    """
    term1 = 3.0 * (1.0 - x) ** 2 * np.exp(-(x ** 2) - (y + 1.0) ** 2)
    term2 = -10.0 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-(x ** 2) - y ** 2)
    term3 = -(1.0 / 3.0) * np.exp(-((x + 1.0) ** 2) - y ** 2)
    return term1 + term2 + term3


def peaks_gradient(x: float, y: float) -> Tuple[float, float]:
    r"""
    peaks 函数的数值梯度（有限差分）。
    \nabla f = [\partial f / \partial x, \partial f / \partial y]^T
    """
    h = 1e-6
    fx = (peaks_function(x + h, y) - peaks_function(x - h, y)) / (2.0 * h)
    fy = (peaks_function(x, y + h) - peaks_function(x, y - h)) / (2.0 * h)
    return fx, fy


class HookeJeevesOptimizer:
    r"""
    Hooke-Jeeves 直接搜索优化算法（融合 1266_toms178）。

    算法步骤：
    1. 探索搜索（exploratory move）：沿各坐标轴试探
    2. 模式移动（pattern move）：沿改善方向外推
    3. 步长缩减（step reduction）：rho \in (0,1)

    用于无梯度优化摩擦系数 \mu，使得模拟结果与实验数据误差最小。
    """

    def __init__(self, rho: float = 0.5, eps: float = 1e-6, itermax: int = 500):
        self.rho = rho
        self.eps = eps
        self.itermax = itermax

    def optimize(self, f: Callable[[np.ndarray], float],
                 x0: np.ndarray) -> Tuple[np.ndarray, int, dict]:
        nvars = len(x0)
        xbefore = np.array(x0, dtype=float)
        newx = xbefore.copy()
        delta = np.array([self.rho if xi == 0.0 else self.rho * abs(xi) for xi in xbefore])
        steplength = self.rho
        iters = 0
        fbefore = f(xbefore)
        funevals = 1
        history = {"fvals": [fbefore], "xvals": [xbefore.copy()]}

        while iters < self.itermax and self.eps < steplength:
            iters += 1
            newx = xbefore.copy()
            newf, newx, funevals = self._best_nearby(delta, newx, fbefore, nvars, f, funevals)
            keep = True
            while newf < fbefore and keep:
                for i in range(nvars):
                    if newx[i] <= xbefore[i]:
                        delta[i] = -abs(delta[i])
                    else:
                        delta[i] = abs(delta[i])
                    tmp = xbefore[i]
                    xbefore[i] = newx[i]
                    newx[i] = newx[i] + newx[i] - tmp
                fbefore = newf
                newf, newx, funevals = self._best_nearby(delta, newx, fbefore, nvars, f, funevals)
                if fbefore <= newf:
                    break
                keep = False
                for i in range(nvars):
                    if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                        keep = True
                        break
            if self.eps <= steplength and fbefore <= newf:
                steplength *= self.rho
                delta *= self.rho
            history["fvals"].append(fbefore)
            history["xvals"].append(xbefore.copy())

        return xbefore, iters, history

    def _best_nearby(self, delta: np.ndarray, x: np.ndarray, fbefore: float,
                     nvars: int, f: Callable, funevals: int) -> Tuple[float, np.ndarray, int]:
        z = x.copy()
        fnow = fbefore
        for i in range(nvars):
            z[i] = x[i] + delta[i]
            ftmp = f(z)
            funevals += 1
            if ftmp < fnow:
                fnow = ftmp
            else:
                z[i] = x[i] - delta[i]
                ftmp = f(z)
                funevals += 1
                if ftmp < fnow:
                    fnow = ftmp
                else:
                    z[i] = x[i]
        return fnow, z, funevals


def friction_coefficient_calibration(
    simulated_func: Callable[[float], float],
    target_value: float,
    mu_bounds: Tuple[float, float] = (0.05, 1.0)
) -> Tuple[float, dict]:
    r"""
    使用 Hooke-Jeeves 优化校准摩擦系数 \mu。

    目标函数：
    J(\mu) = (Q_{sim}(\mu) - Q_{target})^2

    其中 Q 为观测到的接触力学量（如最大切向位移、摩擦耗散等）。
    """
    def objective(mu_vec: np.ndarray) -> float:
        mu = float(np.clip(mu_vec[0], mu_bounds[0], mu_bounds[1]))
        try:
            q = simulated_func(mu)
        except Exception:
            q = 1e10
        return (q - target_value) ** 2

    optimizer = HookeJeevesOptimizer(rho=0.5, eps=1e-7, itermax=200)
    mu0 = np.array([0.3])
    mu_opt, iters, hist = optimizer.optimize(objective, mu0)
    mu_opt_clipped = float(np.clip(mu_opt[0], mu_bounds[0], mu_bounds[1]))
    info = {
        "iterations": iters,
        "final_objective": hist["fvals"][-1],
        "history": hist
    }
    return mu_opt_clipped, info


def peaks_surface_contact_potential(x: np.ndarray, y: np.ndarray,
                                     amplitude: float = 1e8) -> float:
    r"""
    将 peaks 函数作为接触势能：
    \Phi(x,y) = amplitude \cdot peaks(x, y)
    用于测试非线性接触本构。
    """
    return amplitude * peaks_function(x, y)
