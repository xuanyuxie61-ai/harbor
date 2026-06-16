"""
ode_optimizers.py
=================
ODE 梯度流优化器 —— 将优化问题转化为动力系统

融合种子项目:
  - 764_midpoint: 隐式中点法 (二阶辛几何积分器)
  - 1434_zombie_ode: 含守恒量的 ODE 系统

核心公式:
  1. 梯度流: dx/dt = -∇f(x)  (连续极限, 保证 f 单调递减)
  2. 隐式中点法: x_{n+1} = x_n + h·F((x_n + x_{n+1})/2)
     (二阶精度, 保持辛结构, 能量几乎不增)
  3. Heavy-Ball 动力学: ẍ + γẋ + ∇f(x) = 0 (Polyak, 1964)
  4. Nesterov 加速: ẍ + (r/t)ẋ + ∇f(x) = 0 (Su-Boyd-Candes, 2014)
  5. 守恒量: H(x,v) = f(x) + ½||v||² (Hamilton 系统)
  6. Newton 流: dx/dt = -H^{-1}(x) ∇f(x)  (连续 Newton 法)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
import math


# ---------------------------------------------------------------------------
# 1. 梯度流 ODE (连续优化)
# ---------------------------------------------------------------------------

def gradient_flow_rhs(t: float, x: np.ndarray,
                      grad_f: Callable) -> np.ndarray:
    """梯度流右端: dx/dt = -∇f(x).
    连续极限下, f(x(t)) 单调递减:
      d/dt f(x(t)) = ∇f(x)^T ẋ = -||∇f(x)||² ≤ 0.
    """
    return -grad_f(x)


def newton_flow_rhs(t: float, x: np.ndarray,
                    grad_f: Callable, hess_f: Callable) -> np.ndarray:
    """连续 Newton 流: dx/dt = -H^{-1}(x) ∇f(x).
    局部等价于 Newton 法, 具有二次收敛速率.
    需要 Hessian 正定; 否则修正.
    """
    g = grad_f(x)
    H = hess_f(x)
    # 正则化
    H += 1e-8 * np.eye(len(x))
    try:
        d = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        d = -g
    return d


# ---------------------------------------------------------------------------
# 2. 隐式中点法 (源自 764_midpoint)
# ---------------------------------------------------------------------------

def implicit_midpoint(f_rhs: Callable, tspan: Tuple[float, float],
                      y0: np.ndarray, n_steps: int,
                      tol: float = 1e-10, max_fsolve_iter: int = 20
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """隐式中点法求解 ODE: y' = f(t, y).
    源自 764_midpoint/midpoint.

    离散格式:
      y_{n+1} = y_n + h · f(t_n + h/2, (y_n + y_{n+1})/2)

    这是一个隐式方程, 需要 Newton 迭代求解 y_{n+1}.
    性质:
      - 二阶精度 O(h²)
      - 辛几何 (保持 Hamilton 系统的辛结构)
      - A-稳定 (对刚性方程稳定)

    在优化中用于梯度流积分, 保证能量几乎不增.

    返回 (t_array, y_array), t_array 形状 (n+1,), y_array 形状 (n+1, m).
    """
    m = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    dt = (tspan[1] - tspan[0]) / n_steps

    t[0] = tspan[0]
    y[0] = y0.copy()

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        yn = y[i]

        # 显式 Euler 作为初始猜测
        f_n = f_rhs(t[i], yn)
        y_mid_guess = yn + 0.5 * dt * f_n
        y_next = yn + dt * f_n  # 初始猜测

        # Newton 迭代求解隐式方程
        # G(y_{n+1}) = y_{n+1} - y_n - h*f(t+h/2, (y_n+y_{n+1})/2) = 0
        for newton_iter in range(max_fsolve_iter):
            y_mid = 0.5 * (yn + y_next)
            t_mid = t[i] + 0.5 * dt
            f_mid = f_rhs(t_mid, y_mid)

            G = y_next - yn - dt * f_mid
            if np.linalg.norm(G) < tol:
                break

            # 近似 Jacobian: I - (h/2) * ∂f/∂y ≈ I (简化)
            # 使用简化 Newton: y ← y - G
            # 更精确需要 ∂f/∂y, 此处用固定点迭代
            y_next = yn + dt * f_mid

        y[i + 1] = y_next

    return t, y


def explicit_midpoint(f_rhs: Callable, tspan: Tuple[float, float],
                      y0: np.ndarray, n_steps: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """显式中点法 (改进 Euler / RK2).
    k₁ = f(tₙ, yₙ)
    k₂ = f(tₙ + h/2, yₙ + h/2 · k₁)
    y_{n+1} = yₙ + h · k₂

    二阶显式 Runge-Kutta, 无需解隐式方程.
    """
    m = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    dt = (tspan[1] - tspan[0]) / n_steps

    t[0] = tspan[0]
    y[0] = y0.copy()

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        k1 = f_rhs(t[i], y[i])
        k2 = f_rhs(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1)
        y[i + 1] = y[i] + dt * k2

    return t, y


# ---------------------------------------------------------------------------
# 3. RK4 经典四阶 Runge-Kutta
# ---------------------------------------------------------------------------

def rk4(f_rhs: Callable, tspan: Tuple[float, float],
        y0: np.ndarray, n_steps: int
        ) -> Tuple[np.ndarray, np.ndarray]:
    """经典四阶 Runge-Kutta 方法.
    k₁ = f(tₙ, yₙ)
    k₂ = f(tₙ + h/2, yₙ + h/2 · k₁)
    k₃ = f(tₙ + h/2, yₙ + h/2 · k₂)
    k₄ = f(tₙ + h, yₙ + h · k₃)
    y_{n+1} = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

    四阶精度 O(h⁴), 最广泛使用的显式方法.
    """
    m = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))

    dt = (tspan[1] - tspan[0]) / n_steps

    t[0] = tspan[0]
    y[0] = y0.copy()

    for i in range(n_steps):
        t[i + 1] = t[i] + dt
        yi = y[i]
        ti = t[i]
        k1 = f_rhs(ti, yi)
        k2 = f_rhs(ti + 0.5 * dt, yi + 0.5 * dt * k1)
        k3 = f_rhs(ti + 0.5 * dt, yi + 0.5 * dt * k2)
        k4 = f_rhs(ti + dt, yi + dt * k3)
        y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


# ---------------------------------------------------------------------------
# 4. 守恒量监控 (源自 1434_zombie_ode/zombie_conserved)
# ---------------------------------------------------------------------------

def compute_conserved_quantity(y: np.ndarray, conserved_func: Callable) -> np.ndarray:
    """计算 ODE 解的守恒量.
    源自 1434_zombie_ode/zombie_conserved.

    对梯度流 dx/dt = -∇f(x), Hamilton 量 H = f(x) 单调递减.
    对 Hamilton 系统 ẋ = J∇H, H 严格守恒.

    返回各时间步的守恒量值.
    """
    n_steps = len(y)
    H = np.zeros(n_steps)
    for i in range(n_steps):
        H[i] = conserved_func(y[i])
    return H


def conservation_error(H: np.ndarray) -> Tuple[float, float]:
    """守恒量误差分析.
    返回 (最大绝对误差, 最大相对误差).
    """
    H0 = H[0]
    abs_err = np.max(np.abs(H - H0))
    rel_err = abs_err / max(abs(H0), 1e-300)
    return float(abs_err), float(rel_err)


# ---------------------------------------------------------------------------
# 5. Heavy-Ball 优化 (Polyak, 1964)
# ---------------------------------------------------------------------------

def heavy_ball_optimization(f: Callable, grad_f: Callable,
                            x0: np.ndarray,
                            lr: float = 0.01, momentum: float = 0.9,
                            tol: float = 1e-10, max_iter: int = 2000
                            ) -> Dict:
    """Heavy-Ball (重球) 优化.
    等价于离散化: ẍ + γẋ + ∇f(x) = 0

    更新:
      v_{k+1} = μ v_k - α ∇f(x_k)
      x_{k+1} = x_k + v_{k+1}

    参数 μ ∈ (0,1): 动量系数 (摩擦)
    参数 α > 0: 学习率

    对强凸函数, 最优参数:
      α = 4/(√L + √μ_f)²,  μ = ((√L - √μ_f)/(√L + √μ_f))²
    其中 L, μ_f 为 Lipschitz 和强凸常数.
    """
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    fk = f(x)
    gk = grad_f(x)
    history = []

    for k in range(max_iter):
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        v = momentum * v - lr * gk
        x = x + v
        fk = f(x)
        gk = grad_f(x)

    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


# ---------------------------------------------------------------------------
# 6. Nesterov 加速梯度法
# ---------------------------------------------------------------------------

def nesterov_accelerated(f: Callable, grad_f: Callable,
                         x0: np.ndarray,
                         L: float = 1.0,
                         tol: float = 1e-10, max_iter: int = 2000
                         ) -> Dict:
    """Nesterov 加速梯度法 (FISTA 型).

    连续极限: ẍ + (3/t)ẋ + ∇f(x) = 0 (Su-Boyd-Candès, 2014).

    离散:
      y_k = x_k + (k-1)/(k+2) · (x_k - x_{k-1})
      x_{k+1} = y_k - (1/L) ∇f(y_k)

    收敛率: O(1/k²) 对比梯度下降的 O(1/k).
    最优 (对光滑凸函数, Nesterov 1983).
    """
    x = x0.copy().astype(float)
    x_prev = x.copy()
    fk = f(x)
    gk = grad_f(x)
    history = []

    for k in range(max_iter):
        gnorm = np.linalg.norm(gk)
        history.append({"iter": k, "f": fk, "gnorm": gnorm})

        if gnorm < tol:
            return {"x": x, "f_val": fk, "grad_norm": gnorm,
                    "iterations": k, "converged": True, "history": history}

        # Nesterov 外推
        beta = k / (k + 3.0)
        y = x + beta * (x - x_prev)

        # 梯度步
        x_prev = x.copy()
        x = y - (1.0 / L) * grad_f(y)
        fk = f(x)
        gk = grad_f(x)

    return {"x": x, "f_val": fk, "grad_norm": np.linalg.norm(gk),
            "iterations": max_iter, "converged": False, "history": history}


# ---------------------------------------------------------------------------
# 7. ODE 优化器 (梯度流积分)
# ---------------------------------------------------------------------------

def ode_optimizer(f: Callable, grad_f: Callable,
                  x0: np.ndarray,
                  t_final: float = 10.0,
                  n_steps: int = 1000,
                  method: str = "rk4") -> Dict:
    """基于 ODE 积分的优化器.

    将 min f(x) 转化为梯度流:
      dx/dt = -∇f(x),  x(0) = x0

    积分到 t_final, 返回终态作为优化解.
    利用辛积分器 (隐式中点) 保持能量耗散结构.

    返回 dict: {x, f_val, grad_norm, t_array, y_array}.
    """
    def rhs(t, x):
        return -grad_f(x)

    if method == "midpoint":
        t_arr, y_arr = implicit_midpoint(rhs, (0, t_final), x0, n_steps)
    elif method == "rk4":
        t_arr, y_arr = rk4(rhs, (0, t_final), x0, n_steps)
    else:
        t_arr, y_arr = explicit_midpoint(rhs, (0, t_final), x0, n_steps)

    x_opt = y_arr[-1]
    fk = f(x_opt)
    gk = grad_f(x_opt)

    # 能量历史
    energy = np.array([f(y_arr[i]) for i in range(len(t_arr))])

    return {
        "x": x_opt,
        "f_val": fk,
        "grad_norm": np.linalg.norm(gk),
        "t_array": t_arr,
        "y_array": y_arr,
        "energy": energy,
        "converged": np.linalg.norm(gk) < 1e-8
    }


# ---------------------------------------------------------------------------
# 8.  zombie 型动力学 (源自 1434_zombie_ode)
# ---------------------------------------------------------------------------

class PopulationDynamics:
    """种群动力学系统 (源自 1434_zombie_ode).

    重构为优化测试问题: 通过调整参数使系统守恒量最小化.

    原始 SZR 模型:
      dS/dt = -βSZ - δS
      dZ/dt = βSZ - αSZ + γR
      dR/dt = αSZ - γR + δS
    守恒量: S + Z + R = const.

    优化目标: 找参数 (α,β,γ,δ) 使终态 Zombies(Z) 最小化.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.01,
                 gamma: float = 0.1, delta: float = 0.01):
        self.alpha = alpha  # zombie 销毁率
        self.beta = beta    # 新 zombie 率
        self.gamma = gamma  # zombie 复活率
        self.delta = delta  # 背景死亡率

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """SZR 模型右端 (源自 1434_zombie_ode/zombie_deriv)."""
        S, Z, R = y[0], y[1], y[2]
        dSdt = -self.beta * S * Z - self.delta * S
        dZdt = self.beta * S * Z - self.alpha * S * Z + self.gamma * R
        dRdt = self.alpha * S * Z - self.gamma * R + self.delta * S
        return np.array([dSdt, dZdt, dRdt])

    def conserved(self, y: np.ndarray) -> float:
        """守恒量: S + Z + R (源自 1434_zombie_ode/zombie_conserved)."""
        return np.sum(y)

    def simulate(self, y0: np.ndarray, t_final: float = 50.0,
                 n_steps: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """模拟 SZR 动力学 (使用足够小的步长保证稳定)."""
        return rk4(self.rhs, (0, t_final), y0, n_steps)

    def objective(self, params: np.ndarray) -> float:
        """优化目标: 终态 zombie 数量.
        params = [alpha, beta, gamma, delta].
        """
        # 参数边界处理
        params = np.clip(params, 1e-4, 5.0)
        self.alpha, self.beta, self.gamma, self.delta = params
        y0 = np.array([500.0, 10.0, 0.0])  # 初始 S, Z, R
        t_arr, y_arr = self.simulate(y0, t_final=50.0, n_steps=5000)
        # 处理 NaN
        if np.any(np.isnan(y_arr)):
            return 1e10
        # 目标: 最小化终态 zombie, 同时惩罚总人口损失
        Z_final = y_arr[-1, 1]
        total_conserved = self.conserved(y_arr[-1])
        total_init = self.conserved(y0)
        conservation_violation = abs(total_conserved - total_init)
        return Z_final + 100.0 * conservation_violation
