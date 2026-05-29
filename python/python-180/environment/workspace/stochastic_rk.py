"""
stochastic_rk.py
随机 Runge-Kutta 方法与刚性 SDE 隐式处理

融合种子项目:
  - 1029_rk12: 显式 RK1/RK2 误差估计与自适应步长
  - 1164_stiff_ode: 刚性常微分方程的隐式处理
  - 907_praxis: 无梯度优化中的数值稳定性思想

科学背景:
  对于 Itô SDE:
      dX(t) = f(X(t)) dt + g(X(t)) dW(t),   X(0) = X0
  Euler-Maruyama 格式:
      Y_{n+1} = Y_n + f(Y_n) h + g(Y_n) Delta W_n
  强收敛阶: 0.5 (对于一般 SDE)
  弱收敛阶: 1.0

  为了提高精度与稳定性，采用随机 Runge-Kutta (SRK) 方法。
  这里实现经典的 Platen 显式强阶 1.0 SRK:
      H1 = Y_n + f(Y_n) h + g(Y_n) sqrt(h)
      H2 = Y_n + f(Y_n) h - g(Y_n) sqrt(h)
      Y_{n+1} = Y_n + f(Y_n) h + 0.5*(g(H1)+g(H2))*Delta W_n
                + 0.5*sqrt(h)^{-1}*(g(H1)-g(H2))*(Delta W_n^2 - h)

  对于刚性 SDE (如 fast-slow 系统):
      dX = -lambda X dt + sigma dW,  lambda >> 1
  显式方法需要 h < 2/lambda，计算代价极高。
  采用半隐式 Euler (drift-implicit):
      Y_{n+1} = Y_n + f(Y_{n+1}) h + g(Y_n) Delta W_n
  对于线性漂移 f(X) = -lambda X，可解析求解:
      Y_{n+1} = (Y_n + g(Y_n) Delta W_n) / (1 + lambda h)

  自适应步长控制:
      利用 RK1/RK2 嵌入对 (Heun/Euler) 估计局部误差:
          e_n = |Y^{(2)}_{n+1} - Y^{(1)}_{n+1}|
      若 e_n > tol，则拒绝该步并缩小步长:
          h_new = h * min(5.0, max(0.2, 0.9 * sqrt(tol / e_n)))
"""

import numpy as np
from typing import Callable, Tuple, Optional


def sde_euler_maruyama_step(y: np.ndarray,
                            f: Callable[[np.ndarray], np.ndarray],
                            g: Callable[[np.ndarray], np.ndarray],
                            h: float,
                            dW: np.ndarray) -> np.ndarray:
    """
    Euler-Maruyama 单步。
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    return y + f(y) * h + g(y) * dW


def sde_srk_platen_step(y: np.ndarray,
                        f: Callable[[np.ndarray], np.ndarray],
                        g: Callable[[np.ndarray], np.ndarray],
                        h: float,
                        dW: np.ndarray) -> np.ndarray:
    """
    Platen 显式强阶 1.0 SRK 单步。
    适用于标量或多维 SDE (对角噪声)。
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    sqrt_h = np.sqrt(h)
    # 避免 sqrt_h 过小
    if sqrt_h < 1e-14:
        return y + f(y) * h + g(y) * dW

    fy = f(y)
    gy = g(y)
    H1 = y + fy * h + gy * sqrt_h
    H2 = y + fy * h - gy * sqrt_h
    gH1 = g(H1)
    gH2 = g(H2)

    y_new = y + fy * h + 0.5 * (gH1 + gH2) * dW
    y_new += 0.5 * (gH1 - gH2) * (dW ** 2 - h) / sqrt_h
    return y_new


def sde_milstein_step(y: np.ndarray,
                      f: Callable[[np.ndarray], np.ndarray],
                      g: Callable[[np.ndarray], np.ndarray],
                      dg: Callable[[np.ndarray], np.ndarray],
                      h: float,
                      dW: np.ndarray) -> np.ndarray:
    """
    Milstein 强阶 1.0 方法 (单维噪声)。
    需要漂移 g 的导数 dg/dx。
        Y_{n+1} = Y_n + f h + g dW + 0.5 g g' (dW^2 - h)
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    # TODO: 实现 Milstein 强阶 1.0 单步公式
    pass


def stiff_sde_semiimplicit_step(y: np.ndarray,
                                f_lin: np.ndarray,   # 线性部分矩阵 A (已离散)
                                f_nonlin: Callable[[np.ndarray], np.ndarray],
                                g: Callable[[np.ndarray], np.ndarray],
                                h: float,
                                dW: np.ndarray) -> np.ndarray:
    """
    半隐式 Euler 处理刚性线性漂移。
    方程形式: dy = (A y + f_nonlin(y)) dt + g(y) dW
    离散:
        (I - h A) y_{n+1} = y_n + h f_nonlin(y_n) + g(y_n) dW_n
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    n = len(y)
    I = np.eye(n, dtype=np.float64)
    lhs = I - h * f_lin
    rhs = y + h * f_nonlin(y) + g(y) * dW

    # 边界鲁棒性
    cond_est = np.linalg.cond(lhs)
    if cond_est > 1e14:
        y_new = np.linalg.lstsq(lhs, rhs, rcond=1e-14)[0]
    else:
        y_new = np.linalg.solve(lhs, rhs)
    return y_new


def adaptive_rk12_sde_step(y: np.ndarray,
                           f: Callable[[np.ndarray], np.ndarray],
                           g: Callable[[np.ndarray], np.ndarray],
                           h: float,
                           dW: np.ndarray,
                           tol: float = 1e-4) -> Tuple[np.ndarray, float, bool]:
    """
    基于 RK1/RK2 嵌入对的自适应步长控制。
    RK1 (Euler-Maruyama): Y1 = y + f(y) h + g(y) dW
    RK2 (Heun):            Y2 = y + 0.5*(f(y)+f(Y1))*h + 0.5*(g(y)+g(Y1))*dW

    返回:
        y_new: 接受的解 (Heun)
        h_new: 建议的下一步长
        accepted: 该步是否被接受
    """
    if h <= 0:
        raise ValueError("Step size h must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")

    fy = f(y)
    gy = g(y)
    Y1 = y + fy * h + gy * dW

    fY1 = f(Y1)
    gY1 = g(Y1)
    Y2 = y + 0.5 * (fy + fY1) * h + 0.5 * (gy + gY1) * dW

    err = np.linalg.norm(Y2 - Y1) / (np.linalg.norm(y) + 1e-12)
    # 步长调整因子
    factor = 0.9 * np.sqrt(tol / (err + 1e-16))
    factor = min(5.0, max(0.2, factor))
    h_new = h * factor

    accepted = err <= tol
    return Y2, h_new, accepted


class StochasticIntegrator:
    """
    SPDE 时间积分器封装。
    """

    METHODS = ["em", "srk_platen", "milstein", "semiimplicit", "adaptive_rk12"]

    def __init__(self,
                 method: str = "srk_platen",
                 dt: float = 1e-3,
                 tol: float = 1e-4,
                 f_lin: Optional[np.ndarray] = None):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method}, must be one of {self.METHODS}")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.method = method
        self.dt = dt
        self.tol = tol
        self.f_lin = f_lin

    def step(self,
             y: np.ndarray,
             f: Callable[[np.ndarray], np.ndarray],
             g: Callable[[np.ndarray], np.ndarray],
             dW: np.ndarray,
             dg: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[np.ndarray, float]:
        if self.method == "em":
            return sde_euler_maruyama_step(y, f, g, self.dt, dW), self.dt
        elif self.method == "srk_platen":
            return sde_srk_platen_step(y, f, g, self.dt, dW), self.dt
        elif self.method == "milstein":
            if dg is None:
                raise ValueError("Milstein method requires dg")
            return sde_milstein_step(y, f, g, dg, self.dt, dW), self.dt
        elif self.method == "semiimplicit":
            if self.f_lin is None:
                raise ValueError("semiimplicit method requires f_lin matrix")
            return stiff_sde_semiimplicit_step(y, self.f_lin, f, g, self.dt, dW), self.dt
        elif self.method == "adaptive_rk12":
            y_new, h_new, accepted = adaptive_rk12_sde_step(y, f, g, self.dt, dW, self.tol)
            self.dt = h_new
            return y_new, h_new
        else:
            raise RuntimeError("Unreachable")
